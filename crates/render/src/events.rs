//! Event camera simulation (ESIM-style).
//!
//! An event camera reports, per pixel and asynchronously, when the log intensity has changed by a
//! contrast threshold C since the pixel's last event: e = (x, y, t, p). We simulate it from the
//! renderer:
//!
//! 1. Sensor steps: the log intensity is evaluated at times chosen so that image motion between
//!    consecutive steps stays below `max_px_per_step` (from the exact geometry: the 3D points of
//!    a render projected with the step poses), and lamp flicker is resolved
//!    (`flicker_steps_per_period`). Vibration therefore produces dense steps only when it
//!    actually moves the image.
//! 2. Images at the steps (`mode`):
//!    * `render`: every motion step is rendered; flicker-only sub-steps evaluate the rendered
//!      flicker split (`FrameOut::radiance_at`) and interpolate between renders per pixel.
//!    * `warp`: keyframes are rendered every `key_px` of image motion; the steps in between
//!      reproject both neighbouring keyframes with the exact pose and per-pixel geometry and
//!      blend them (occlusion-checked). Exact for a static Lambertian scene except at
//!      occlusion edges and view-dependent shading (water glint, haze path); ~5-10x faster.
//! 3. Per pixel, log intensity is linearly interpolated between steps; every crossing of the
//!    reference level ± C emits an event at the interpolated time (several per step if the
//!    change spans several thresholds), subject to a refractory period.
//! 4. Sensor non-idealities: photoreceptor bandwidth, high-pass, per-pixel threshold mismatch,
//!    leak and shot-noise background events, hot pixels, timestamp jitter, rate controller.
//!
//! Output: `<camera.path>/events/{x, y, t, p, ms_index}` in the sequence file. `t` is i64 µs on
//! the sequence clock (shared with frames, IMU and poses), `p` is 1 for ON and 0 for OFF, and
//! `ms_index[m]` is the index of the first event at or after m ms (plus one closing entry).
//! The event camera's pose at any time is the body pose (`/pose`) ∘ `calib/T_body_cam`.

use crate::cache::TileCache;
use crate::camera::CameraModel;
use crate::raster::{FrameOut, RenderSettings, Renderer};
use crate::scenario::{CameraSpec, Scenario};
use crate::trajectory::{self, CamPose, Pose};
use anyhow::{bail, Result};
use geodesy::Ellipsoid;
use glam::{DVec2, DVec3};
use h5::Attrs;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EventMode {
    /// Render every motion step (reference quality).
    Render,
    /// Render keyframes every `key_px` of motion, reproject them for the steps in between.
    #[default]
    Warp,
}

/// Event camera knobs. Names in parentheses are the corresponding Prophesee biases.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EventConfig {
    // ---- pixel front-end
    /// Log-intensity contrast thresholds ON / OFF (bias_diff_on / bias_diff_off).
    pub contrast_pos: f64,
    pub contrast_neg: f64,
    /// Per-pixel threshold mismatch σ (fixed pattern).
    pub contrast_sigma: f64,
    /// Refractory period after an event (bias_refr), µs.
    pub refractory_us: f64,
    /// Photoreceptor bandwidth (bias_fo): 3 dB cutoff grows with the photocurrent and saturates:
    /// fc = cutoff_hz · I / (I + cutoff_half_lum), floored at `cutoff_min_hz` (I = luminance,
    /// 1 = sunlit white ≈ 1e5 lux, so 1e-5 ≈ 1 lux). Low light → slow pixels → smeared,
    /// delayed events.
    pub cutoff_hz: f64,
    pub cutoff_half_lum: f64,
    pub cutoff_min_hz: f64,
    /// High-pass (bias_hpf): the reference level relaxes towards the signal with this corner
    /// frequency (Hz), suppressing slow changes. 0 = off.
    pub hpf_hz: f64,
    /// Offset added before the log: L = ln(gain * lum + eps) (dark current).
    pub log_eps: f64,
    pub gain: f64,

    // ---- noise
    /// Leak (background) events: mean rate per pixel (Hz, mostly ON) and log-normal spread.
    pub leak_hz: f64,
    pub leak_sigma: f64,
    /// Shot-noise background activity (Hz per pixel) at luminance `noise_ref_lum`; grows as
    /// sqrt(ref / lum) in the dark, up to `shot_noise_dark_gain` times.
    pub shot_noise_hz: f64,
    pub noise_ref_lum: f64,
    pub shot_noise_dark_gain: f64,
    /// Fraction of hot pixels and their event rate (Hz).
    pub hot_pixel_fraction: f64,
    pub hot_pixel_hz: f64,
    /// Timestamp jitter σ (µs).
    pub timestamp_jitter_us: f64,
    /// Event rate controller: max events per second (Mev/s); excess events are dropped at
    /// random within each 1 ms window. 0 = unlimited.
    pub max_rate_mev_s: f64,

    // ---- simulation
    /// How the images between renders are obtained (see the module docs).
    pub mode: EventMode,
    /// `warp` mode: max image motion between rendered keyframes (px).
    pub key_px: f64,
    /// Max image motion between sensor steps (px).
    pub max_px_per_step: f64,
    /// Sensor step-rate bounds (Hz).
    pub min_rate_hz: f64,
    pub max_rate_hz: f64,
    /// Supersampling of the internal renders (anti-aliasing reduces spurious events).
    pub supersample: u32,
    /// When artificial lights are on and flicker is enabled, step at least this many times per
    /// flicker period (resolves the 100/120 Hz lamp modulation).
    pub flicker_steps_per_period: f64,
    pub seed: u64,
}

impl Default for EventConfig {
    fn default() -> Self {
        EventConfig {
            contrast_pos: 0.25,
            contrast_neg: 0.25,
            contrast_sigma: 0.03,
            refractory_us: 100.0,
            cutoff_hz: 3000.0,
            cutoff_half_lum: 2e-5,
            cutoff_min_hz: 0.5,
            hpf_hz: 0.0,
            log_eps: 1e-3,
            gain: 1.0,
            leak_hz: 0.1,
            leak_sigma: 0.5,
            shot_noise_hz: 0.3,
            noise_ref_lum: 0.2,
            shot_noise_dark_gain: 20.0,
            hot_pixel_fraction: 2e-5,
            hot_pixel_hz: 200.0,
            timestamp_jitter_us: 2.0,
            max_rate_mev_s: 50.0,
            mode: EventMode::Warp,
            key_px: 4.0,
            max_px_per_step: 0.5,
            min_rate_hz: 100.0,
            max_rate_hz: 5000.0,
            supersample: 2,
            flicker_steps_per_period: 12.0,
            seed: 11,
        }
    }
}

/// splitmix64 RNG
#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub t_us: i64,
    pub x: u16,
    pub y: u16,
    pub p: i8,
}

/// State of one pixel.
#[derive(Clone, Copy, Debug)]
struct Pixel {
    /// low-pass filtered log intensity (photoreceptor output)
    lp: f32,
    /// reference level of the change detector
    l_ref: f32,
    last_t: f64,
    c_pos: f32,
    c_neg: f32,
    /// leak rate (Hz)
    leak: f32,
}

/// Per-pixel event generator.
pub struct EventSensor {
    w: usize,
    cfg: EventConfig,
    px: Vec<Pixel>,
    hot: Vec<(usize, f64, f64)>,
    t_prev: Option<f64>,
    n_steps: u64,
    rng: Rng,
}

impl EventSensor {
    pub fn new(cfg: EventConfig, w: usize, h: usize) -> Self {
        let mut rng = Rng(cfg.seed ^ 0xE7E7);
        let n = w * h;
        let px = (0..n)
            .map(|_| Pixel {
                lp: 0.0,
                l_ref: 0.0,
                last_t: f64::MIN,
                c_pos: (cfg.contrast_pos + cfg.contrast_sigma * rng.gauss()).max(0.01) as f32,
                c_neg: (cfg.contrast_neg + cfg.contrast_sigma * rng.gauss()).max(0.01) as f32,
                leak: (cfg.leak_hz * (cfg.leak_sigma * rng.gauss() - 0.5 * cfg.leak_sigma * cfg.leak_sigma).exp()) as f32,
            })
            .collect();
        // hot pixels: random positions, rate spread, polarity bias
        let nh = (cfg.hot_pixel_fraction * n as f64).round() as usize;
        let hot = (0..nh)
            .map(|_| ((rng.next() % n as u64) as usize, cfg.hot_pixel_hz * (0.3 + 1.4 * rng.uniform()), 0.2 + 0.6 * rng.uniform()))
            .collect();
        EventSensor { w, px, hot, t_prev: None, n_steps: 0, rng, cfg }
    }

    /// Log intensity (photoreceptor input) from linear RGB radiance.
    pub fn log_image(&self, radiance: &[f32]) -> Vec<f32> {
        let g = self.cfg.gain as f32;
        let eps = self.cfg.log_eps as f32;
        radiance.par_chunks_exact(3).map(|c| (g * (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) + eps).ln()).collect()
    }

    fn poisson(&mut self, mean: f64) -> u32 {
        if mean <= 0.0 {
            return 0;
        }
        if mean > 30.0 {
            return (mean + mean.sqrt() * self.rng.gauss()).round().max(0.0) as u32;
        }
        let l = (-mean).exp();
        let mut k = 0;
        let mut p = 1.0;
        loop {
            p *= self.rng.uniform();
            if p <= l {
                return k;
            }
            k += 1;
        }
    }

    /// Feed the log image at time `t` (s); returns the events in (t_prev, t], sorted.
    /// Pixels are processed in parallel by rows (deterministic: per-row RNG streams).
    pub fn step(&mut self, t: f64, l: &[f32]) -> Vec<Event> {
        let Some(t0) = self.t_prev else {
            for (p, &v) in self.px.iter_mut().zip(l) {
                p.lp = v;
                p.l_ref = v;
            }
            self.t_prev = Some(t);
            return vec![];
        };
        let dt = t - t0;
        if dt <= 0.0 {
            return vec![];
        }
        self.n_steps += 1;
        let c = &self.cfg;
        let w = self.w;
        let refr = c.refractory_us * 1e-6;
        let eps = c.log_eps as f32;
        let a_hpf = if c.hpf_hz > 0.0 { (1.0 - (-std::f64::consts::TAU * c.hpf_hz * dt).exp()) as f32 } else { 0.0 };
        let step_seed = c.seed ^ self.n_steps.wrapping_mul(0xA24B_AED4_963E_E407);
        let rows: Vec<Vec<Event>> = self
            .px
            .par_chunks_mut(w)
            .zip(l.par_chunks(w))
            .enumerate()
            .map(|(y, (row, lrow))| {
                let mut rng = Rng(step_seed ^ (y as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25));
                let mut ev = Vec::new();
                for (x, (p, &lin)) in row.iter_mut().zip(lrow).enumerate() {
                    // photoreceptor low-pass: cutoff grows with the photocurrent (luminance)
                    let lum = ((lin.exp() - eps) / c.gain as f32).max(0.0) as f64;
                    let fc = (c.cutoff_hz * lum / (lum + c.cutoff_half_lum)).max(c.cutoff_min_hz);
                    let alpha = (1.0 - (-std::f64::consts::TAU * fc * dt).exp()) as f32;
                    let lp0 = p.lp;
                    let lp1 = lp0 + alpha * (lin - lp0);
                    p.lp = lp1;
                    // high-pass: reference relaxes towards the signal; leak: reference drifts down
                    if a_hpf > 0.0 {
                        p.l_ref += a_hpf * (lp0 - p.l_ref);
                    }
                    p.l_ref -= (p.leak as f64 * dt) as f32 * p.c_pos;
                    let dl = lp1 - lp0;
                    let (xu, yu) = (x as u16, y as u16);
                    loop {
                        let up = lp1 - p.l_ref >= p.c_pos;
                        let down = p.l_ref - lp1 >= p.c_neg;
                        if !up && !down {
                            break;
                        }
                        let target = if up { p.l_ref + p.c_pos } else { p.l_ref - p.c_neg };
                        // crossing time by linear interpolation over the step (leak crossings land
                        // uniformly in the step)
                        let a = if dl.abs() > 1e-9 { ((target - lp0) / dl).clamp(0.0, 1.0) as f64 } else { rng.uniform() };
                        let te = t0 + a * dt;
                        p.l_ref = target;
                        if te - p.last_t >= refr {
                            p.last_t = te;
                            ev.push(Event { t_us: (te * 1e6).round() as i64, x: xu, y: yu, p: if up { 1 } else { 0 } });
                        }
                    }
                    // shot-noise background activity, stronger in the dark
                    if c.shot_noise_hz > 0.0 {
                        let dark = (c.noise_ref_lum / (lum + 1e-6)).sqrt().clamp(1.0 / c.shot_noise_dark_gain.max(1.0), c.shot_noise_dark_gain);
                        if rng.uniform() < c.shot_noise_hz * dark * dt {
                            let te = t0 + rng.uniform() * dt;
                            if te - p.last_t >= refr {
                                p.last_t = te;
                                ev.push(Event { t_us: (te * 1e6).round() as i64, x: xu, y: yu, p: (rng.next() & 1) as i8 });
                            }
                        }
                    }
                }
                ev
            })
            .collect();
        let mut ev: Vec<Event> = rows.concat();
        // hot pixels
        for i in 0..self.hot.len() {
            let (k, rate, pol) = self.hot[i];
            for _ in 0..self.poisson(rate * dt) {
                let te = t0 + self.rng.uniform() * dt;
                let p = if self.rng.uniform() < pol { 1 } else { 0 };
                ev.push(Event { t_us: (te * 1e6).round() as i64, x: (k % w) as u16, y: (k / w) as u16, p });
            }
        }
        let c = &self.cfg;
        // timestamp jitter
        if c.timestamp_jitter_us > 0.0 {
            let (lo, hi) = ((t0 * 1e6).ceil() as i64, (t * 1e6).floor() as i64);
            for e in ev.iter_mut() {
                e.t_us = (e.t_us + (c.timestamp_jitter_us * self.rng.gauss()).round() as i64).clamp(lo, hi.max(lo));
            }
        }
        ev.par_sort_by_key(|e| e.t_us);
        // event rate controller: cap events per 1 ms window, drop the excess at random
        if c.max_rate_mev_s > 0.0 && !ev.is_empty() {
            let cap = c.max_rate_mev_s * 1e3; // events per ms
            let mut out = Vec::with_capacity(ev.len());
            let mut i = 0;
            while i < ev.len() {
                let ms = ev[i].t_us.div_euclid(1000);
                let mut j = i;
                while j < ev.len() && ev[j].t_us.div_euclid(1000) == ms {
                    j += 1;
                }
                // the window may be only partially covered by this step
                let span_us = (((t * 1e6) as i64).min((ms + 1) * 1000) - ((t0 * 1e6) as i64).max(ms * 1000)).clamp(1, 1000) as f64;
                let keep = (cap * span_us / 1000.0 / (j - i) as f64).min(1.0);
                for e in &ev[i..j] {
                    if keep >= 1.0 || self.rng.uniform() < keep {
                        out.push(*e);
                    }
                }
                i = j;
            }
            ev = out;
        }
        self.t_prev = Some(t);
        ev
    }
}

/// Streaming writer of `<camera>/events`.
pub struct EventWriter {
    group: h5::Group,
    x: h5::Dataset,
    y: h5::Dataset,
    t: h5::Dataset,
    p: h5::Dataset,
    n: usize,
    /// index of the first event at or after each millisecond
    ms_index: Vec<u64>,
    t0_us: i64,
}

impl EventWriter {
    /// Replaces `events` in the camera group `g`. `t0` = trajectory time of the sequence start.
    pub fn new(g: &h5::Group, t0: f64, level: u8) -> Result<Self> {
        if g.exists("events") {
            g.delete("events")?;
        }
        let g = g.ensure_group("events")?;
        let chunk = 1 << 16;
        let lvl = level.min(9);
        let x = g.new_dataset::<u16>().shape(&[0]).max_shape(&[None]).chunk(&[chunk]).shuffle(true).deflate(lvl).create("x")?;
        let y = g.new_dataset::<u16>().shape(&[0]).max_shape(&[None]).chunk(&[chunk]).shuffle(true).deflate(lvl).create("y")?;
        let t = g.new_dataset::<i64>().shape(&[0]).max_shape(&[None]).chunk(&[chunk]).shuffle(true).deflate(lvl).create("t")?;
        let p = g.new_dataset::<i8>().shape(&[0]).max_shape(&[None]).chunk(&[chunk]).deflate(lvl).create("p")?;
        Ok(EventWriter { group: g, x, y, t, p, n: 0, ms_index: vec![], t0_us: (t0 * 1e6).round() as i64 })
    }

    /// Append events (absolute µs, sorted).
    pub fn write(&mut self, ev: &[Event]) -> Result<()> {
        if ev.is_empty() {
            return Ok(());
        }
        let ts: Vec<i64> = ev.iter().map(|e| e.t_us - self.t0_us).collect();
        for (i, t) in ts.iter().enumerate() {
            let ms = (t.max(&0) / 1000) as usize;
            while self.ms_index.len() <= ms {
                self.ms_index.push((self.n + i) as u64);
            }
        }
        let n1 = self.n + ev.len();
        for ds in [&self.x, &self.y, &self.t, &self.p] {
            ds.resize(&[n1])?;
        }
        let xs: Vec<u16> = ev.iter().map(|e| e.x).collect();
        let ys: Vec<u16> = ev.iter().map(|e| e.y).collect();
        let ps: Vec<i8> = ev.iter().map(|e| e.p).collect();
        self.x.write_slice(&xs, &[self.n], &[ev.len()])?;
        self.y.write_slice(&ys, &[self.n], &[ev.len()])?;
        self.t.write_slice(&ts, &[self.n], &[ev.len()])?;
        self.p.write_slice(&ps, &[self.n], &[ev.len()])?;
        self.n = n1;
        Ok(())
    }

    pub fn finish(mut self, cfg: &EventConfig) -> Result<usize> {
        // closing entry one past the end, so events of ms m are [ms_index[m], ms_index[m+1])
        self.ms_index.push(self.n as u64);
        self.group.new_dataset::<u64>().shape(&[self.ms_index.len()]).create("ms_index")?.write_all(&self.ms_index)?;
        self.group.set_attr_str("events_yaml", &serde_yaml::to_string(cfg)?)?;
        Ok(self.n)
    }
}

/// A rendered keyframe.
struct Key {
    t: f64,
    cam: CamPose,
    frame: FrameOut,
}

impl Key {
    /// World point seen by pixel k; sky pixels get a far point along their ray, so that they
    /// move with the camera rotation only.
    fn point(&self, k: usize, model: &dyn CameraModel) -> DVec3 {
        let w = model.width() as usize;
        let o = self.frame.sample_offset;
        self.frame.points[k].unwrap_or_else(|| self.cam.cam_to_world(model.unproject(DVec2::new((k % w) as f64 + o, (k / w) as f64 + o)).unwrap_or(DVec3::Z) * 1e7))
    }

    fn points(&self, model: &dyn CameraModel) -> Vec<DVec3> {
        (0..self.frame.points.len()).into_par_iter().map(|k| self.point(k, model)).collect()
    }
}

/// Largest image displacement of a sparse grid of the key's points between poses `a` and `b`.
fn max_motion(key: &Key, a: &CamPose, b: &CamPose, model: &dyn CameraModel) -> f64 {
    let (w, h) = (model.width() as usize, model.height() as usize);
    let step = (w.max(h) / 24).max(1);
    let mut maxd: f64 = 0.0;
    for y in (step / 2..h).step_by(step) {
        for x in (step / 2..w).step_by(step) {
            let p = key.point(y * w + x, model);
            if let (Some(u), Some(v)) = (model.project(a.world_to_cam(p)), model.project(b.world_to_cam(p))) {
                maxd = maxd.max((u - v).length());
            }
        }
    }
    maxd
}

/// Catmull-Rom weights for fractional offset f.
#[inline]
fn cubic_w(f: f32) -> [f32; 4] {
    let (f2, f3) = (f * f, f * f * f);
    [-0.5 * f3 + f2 - 0.5 * f, 1.5 * f3 - 2.5 * f2 + 1.0, -1.5 * f3 + 2.0 * f2 + 0.5 * f, 0.5 * f3 - 0.5 * f2]
}

/// Reproject keyframe radiance `rad` (rendered at the pose of `pts`, the world points of its
/// pixels sampled at pixel + `offset`) into the camera `cam`.
/// Backward warp: for each target pixel x find the source pixel s with s + F(s) = x, where F is
/// the exact forward flow of the key's points, by fixed-point iteration. Returns the warped
/// radiance and a validity mask (false where the iteration does not converge: occlusions,
/// image borders).
#[allow(clippy::too_many_arguments)]
pub fn warp(rad: &[f32], pts: &[DVec3], offset: f64, cam: &CamPose, w: usize, h: usize, model: &dyn CameraModel) -> (Vec<f32>, Vec<bool>) {
    let flow: Vec<[f32; 2]> = pts
        .par_iter()
        .enumerate()
        .map(|(k, p)| match model.project(cam.world_to_cam(*p)) {
            Some(px) => [(px.x - (k % w) as f64 - offset) as f32, (px.y - (k / w) as f64 - offset) as f32],
            None => [f32::NAN; 2],
        })
        .collect();
    let bil_flow = |x: f32, y: f32| -> Option<[f32; 2]> {
        if !(x >= 0.0 && y >= 0.0 && x <= (w - 1) as f32 && y <= (h - 1) as f32) {
            return None;
        }
        let (x0, y0) = ((x as usize).min(w - 2), (y as usize).min(h - 2));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let f = |i: usize, j: usize| flow[j * w + i];
        let (a, b, c, d) = (f(x0, y0), f(x0 + 1, y0), f(x0, y0 + 1), f(x0 + 1, y0 + 1));
        let mut out = [0f32; 2];
        for k in 0..2 {
            out[k] = (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy;
        }
        out[0].is_finite().then_some(out)
    };
    let mut out = vec![0f32; w * h * 3];
    let mut valid = vec![false; w * h];
    out.par_chunks_mut(w * 3).zip(valid.par_chunks_mut(w)).enumerate().for_each(|(y, (orow, vrow))| {
        for x in 0..w {
            let (tx, ty) = (x as f32, y as f32);
            let mut s = match flow[y * w + x] {
                f if f[0].is_finite() => [tx - f[0], ty - f[1]],
                _ => [tx, ty],
            };
            let mut ok = false;
            for _ in 0..4 {
                let Some(f) = bil_flow(s[0], s[1]) else { break };
                let n = [tx - f[0], ty - f[1]];
                let r = (n[0] - s[0]).abs().max((n[1] - s[1]).abs());
                s = n;
                if r < 0.02 {
                    ok = true;
                    break;
                }
            }
            if !ok || !(s[0] >= 0.0 && s[1] >= 0.0 && s[0] <= (w - 1) as f32 && s[1] <= (h - 1) as f32) {
                continue;
            }
            // bicubic (Catmull-Rom) resampling keeps the pixel-scale texture contrast that
            // bilinear interpolation would wash out (fewer, wrong events)
            let (xi, yi) = (s[0].floor(), s[1].floor());
            let (wx, wy) = (cubic_w(s[0] - xi), cubic_w(s[1] - yi));
            let mut acc = [0f32; 3];
            for (j, wyj) in wy.iter().enumerate() {
                let yy = (yi as isize + j as isize - 1).clamp(0, h as isize - 1) as usize;
                for (i, wxi) in wx.iter().enumerate() {
                    let xx = (xi as isize + i as isize - 1).clamp(0, w as isize - 1) as usize;
                    let k = (yy * w + xx) * 3;
                    let wgt = wyj * wxi;
                    for ch in 0..3 {
                        acc[ch] += wgt * rad[k + ch];
                    }
                }
            }
            for ch in 0..3 {
                orow[3 * x + ch] = acc[ch].max(0.0);
            }
            vrow[x] = true;
        }
    });
    (out, valid)
}

/// Box-filter an RGB image of (w·s) x (h·s) down to w x h.
fn box_down(img: &[f32], w: usize, h: usize, s: usize) -> Vec<f32> {
    let ws = w * s;
    let norm = 1.0 / (s * s) as f32;
    let mut out = vec![0f32; w * h * 3];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            for j in 0..s {
                let base = ((y * s + j) * ws + x * s) * 3;
                for i in 0..s {
                    for ch in 0..3 {
                        row[3 * x + ch] += img[base + 3 * i + ch];
                    }
                }
            }
            for v in &mut row[3 * x..3 * x + 3] {
                *v *= norm;
            }
        }
    });
    out
}

/// Radiance at `t` between keyframes `k0`, `k1` (fraction `a` from k0 to k1).
#[allow(clippy::too_many_arguments)]
fn between(k0: &Key, k1: &Key, p0: &[DVec3], p1: &[DVec3], cam: &CamPose, t: f64, a: f64, omega: f64, mode: EventMode, model: &dyn CameraModel) -> Vec<f32> {
    let (r0, r1) = (k0.frame.radiance_at(t, omega), k1.frame.radiance_at(t, omega));
    let a = a as f32;
    match mode {
        // motion below a step: per-pixel interpolation (as the sensor does in log space)
        EventMode::Render => r0.par_iter().zip(&r1).map(|(u, v)| u + a * (v - u)).collect(),
        EventMode::Warp => {
            let (w, h) = (model.width() as usize, model.height() as usize);
            let ((w0, v0), (w1, v1)) = rayon::join(|| warp(&r0, p0, k0.frame.sample_offset, cam, w, h, model), || warp(&r1, p1, k1.frame.sample_offset, cam, w, h, model));
            let mut out = vec![0f32; w * h * 3];
            out.par_chunks_mut(3).enumerate().for_each(|(k, o)| {
                let (wa, wb) = match (v0[k], v1[k]) {
                    (true, true) => (1.0 - a, a),
                    (true, false) => (1.0, 0.0),
                    (false, true) => (0.0, 1.0),
                    // neither keyframe sees it: the nearer keyframe unwarped
                    (false, false) => {
                        let src = if a < 0.5 { &r0 } else { &r1 };
                        o.copy_from_slice(&src[3 * k..3 * k + 3]);
                        return;
                    }
                };
                for ch in 0..3 {
                    o[ch] = wa * w0[3 * k + ch] + wb * w1[3 * k + ch];
                }
            });
            out
        }
    }
}

/// Simulation statistics of one camera.
#[derive(Debug, Default, Clone, Copy)]
pub struct EventStats {
    pub events: usize,
    pub renders: usize,
    pub steps: usize,
    pub render_s: f64,
    pub step_s: f64,
}

/// Simulate the events of one camera over [t_start, t_end] (trajectory time; t_start is the
/// sequence start) into `<spec.path>/events` of the open sequence file.
/// `progress(t_done, t_total)`.
#[allow(clippy::too_many_arguments)]
pub fn simulate(scn: &Scenario, spec: &CameraSpec, poses: &[Pose], cache: Arc<TileCache>, ell: Ellipsoid, file: &h5::File, (t_start, t_end): (f64, f64), progress: &dyn Fn(f64, f64)) -> Result<EventStats> {
    let Some(ec) = &spec.events else { bail!("camera {} has no events modality", spec.path) };
    if poses.len() < 2 {
        bail!("event simulation needs a trajectory");
    }
    let ext = &spec.extrinsics;
    let model: Arc<dyn CameraModel> = spec.intrinsics.build()?;
    let (w, h) = (model.width() as usize, model.height() as usize);
    // Keyframes: in warp mode rendered at the supersampled resolution without supersampling
    // (same cost), warped there and box-filtered to sensor pixels afterwards. A sub-pixel shift
    // of the scene changes a pixel's box average in ways that cannot be interpolated from the
    // downsampled image (it lost ~1/3 of the events against the render reference).
    let ss = ec.supersample.max(1);
    let (km, kss): (Arc<dyn CameraModel>, u32) = match ec.mode {
        EventMode::Warp => (model.scaled(ss), ss),
        EventMode::Render => (model.clone(), 1),
    };
    let mut rs: RenderSettings = scn.render.clone();
    rs.supersample = ss / kss;
    // same level of detail as a supersampled render: LOD targets are in key-image pixels
    rs.texel_px *= kss as f64;
    rs.mesh_px *= kss as f64;
    rs.min_zoom = scn.tiles.min_zoom;
    rs.max_zoom = rs.max_zoom.min(scn.tiles.max_zoom);
    let mut renderer = Renderer::new(km.clone(), rs, ell, cache);
    let km = km.as_ref();
    // key image → sensor pixels (box filter)
    let down = |img: Vec<f32>| -> Vec<f32> { if kss == 1 { img } else { box_down(&img, w, h, kss as usize) } };
    // image motion between two poses in sensor pixels
    let motion = |key: &Key, a: &CamPose, b: &CamPose| max_motion(key, a, b, km) / kss as f64;
    renderer.split_flicker = true;
    let mut sensor = EventSensor::new(ec.clone(), w, h);
    let g = file.ensure_group(crate::scenario::h5path(&spec.path))?;
    if !g.exists("calib") {
        crate::output::write_camera_calib(&g, &spec.intrinsics, crate::output::transform_4x4(ext.r_body_cam(), ext.t_body_cam()))?;
    }
    let mut writer = EventWriter::new(&g, t_start, scn.output.compression.level)?;
    let mut st = EventStats::default();

    let tr0 = poses[0].t;
    let lighting = &scn.render.lighting;
    let omega = lighting.flicker.omega();
    let cam_at = |t: f64| trajectory::interpolate(poses, t).camera(ext, &ell);
    let render_key = |t: f64, st: &mut EventStats| -> Key {
        let pose = trajectory::interpolate(poses, t);
        let cam = pose.camera(ext, &ell);
        let mut sun = lighting.sun_at(t - tr0, pose.geo.lat, pose.geo.lon);
        sun.exposure = 0.0; // events see instantaneous light
        let c0 = std::time::Instant::now();
        let frame = renderer.render(&cam, &sun);
        st.render_s += c0.elapsed().as_secs_f64();
        st.renders += 1;
        Key { t, cam, frame }
    };
    let mut step = |t: f64, rad: &[f32], st: &mut EventStats, writer: &mut EventWriter| -> Result<()> {
        let c0 = std::time::Instant::now();
        let ev = sensor.step(t, &sensor.log_image(rad));
        st.step_s += c0.elapsed().as_secs_f64();
        st.steps += 1;
        st.events += ev.len();
        writer.write(&ev)
    };
    // flicker needs sensor steps even without motion
    let flicker_dt = |t: f64| -> f64 {
        let pose = trajectory::interpolate(poses, t);
        let sun = lighting.sun_at(t - tr0, pose.geo.lat, pose.geo.lon);
        if sun.lights > 0.01 && sun.flicker.enabled && ec.flicker_steps_per_period > 0.0 {
            1.0 / (2.0 * sun.flicker.mains_hz * ec.flicker_steps_per_period)
        } else {
            f64::INFINITY
        }
    };
    let key_px = match ec.mode {
        EventMode::Render => ec.max_px_per_step,
        EventMode::Warp => ec.key_px.max(ec.max_px_per_step),
    };
    let (dt_min, dt_max) = (1.0 / ec.max_rate_hz.max(1.0), 1.0 / ec.min_rate_hz.max(1e-3));
    // keyframes may be further apart than a step when the image hardly moves
    let key_dt_max = dt_max * (key_px / ec.max_px_per_step).max(1.0);

    progress(0.0, t_end - t_start);
    let mut k0 = render_key(t_start, &mut st);
    step(t_start, &down(k0.frame.radiance_at(t_start, omega)), &mut st, &mut writer)?;
    let mut p0 = if ec.mode == EventMode::Warp { k0.points(km) } else { vec![] };
    let mut dt = dt_min * 4.0;
    while k0.t < t_end {
        // Walk the sensor steps along the path (image motion of k0's points ≤ max_px_per_step
        // per step, so vibration inside a keyframe interval is sampled), until the image has
        // moved key_px from k0: that is the next keyframe.
        let dt_cap = dt_max.min(flicker_dt(k0.t)).max(dt_min);
        let mut times = vec![];
        let (mut t, mut cam) = (k0.t, k0.cam);
        loop {
            let tn = (t + dt.min(dt_cap)).min(t_end);
            let cam_n = cam_at(tn);
            let d = motion(&k0, &cam, &cam_n);
            if d > ec.max_px_per_step && tn - t > dt_min * 1.001 {
                dt = ((tn - t) * 0.9 * ec.max_px_per_step / d).max(dt_min);
                continue;
            }
            if d < 0.5 * ec.max_px_per_step {
                dt = (dt * 1.5).min(dt_cap);
            }
            times.push(tn);
            (t, cam) = (tn, cam_n);
            // render mode: every step that moves the image is rendered (only flicker / rate
            // sub-steps are interpolated)
            let new_key = match ec.mode {
                EventMode::Render => d >= 0.25 * ec.max_px_per_step || motion(&k0, &k0.cam, &cam_n) >= 0.5 * ec.max_px_per_step,
                EventMode::Warp => motion(&k0, &k0.cam, &cam_n) >= key_px,
            };
            if new_key || tn >= t_end || tn - k0.t >= key_dt_max * 0.999 {
                break;
            }
        }
        let t1 = *times.last().unwrap();
        let k1 = render_key(t1, &mut st);
        let p1 = if ec.mode == EventMode::Warp { k1.points(km) } else { vec![] };
        let span = t1 - k0.t;
        for &ts in &times[..times.len() - 1] {
            let rad = between(&k0, &k1, &p0, &p1, &cam_at(ts), ts, (ts - k0.t) / span, omega, ec.mode, km);
            step(ts, &down(rad), &mut st, &mut writer)?;
        }
        step(t1, &down(k1.frame.radiance_at(t1, omega)), &mut st, &mut writer)?;
        k0 = k1;
        p0 = p1;
        progress(k0.t - t_start, t_end - t_start);
    }
    writer.finish(ec)?;
    Ok(st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_edge_emits_expected_events() {
        // a step edge moving 10 px right over 1 s: each crossed pixel brightens by ln(4)/C events
        let (w, h) = (32, 4);
        let cfg = EventConfig {
            contrast_sigma: 0.0,
            shot_noise_hz: 0.0,
            leak_hz: 0.0,
            refractory_us: 0.0,
            hot_pixel_fraction: 0.0,
            timestamp_jitter_us: 0.0,
            cutoff_hz: 1e9,
            ..Default::default()
        };
        let mut s = EventSensor::new(cfg, w, h);
        let mut total = 0;
        for k in 0..=100 {
            let edge = 5.0 + 10.0 * k as f64 / 100.0;
            let rad: Vec<f32> = (0..w * h)
                .flat_map(|i| {
                    let x = (i % w) as f64 + 0.5;
                    let v = if x < edge { 1.0 } else { 0.25 };
                    [v as f32; 3]
                })
                .collect();
            let l = s.log_image(&rad);
            let ev = s.step(k as f64 * 0.01, &l);
            assert!(ev.iter().all(|e| e.p == 1));
            assert!(ev.windows(2).all(|p| p[0].t_us <= p[1].t_us));
            total += ev.len();
        }
        // 10 columns x 4 rows, each crossing ln(4)=1.386 → 5 events at C=0.25
        assert_eq!(total, 10 * 4 * 5);
    }
}

#[cfg(test)]
mod noise_tests {
    use super::*;

    #[test]
    fn background_activity_rates() {
        // static, uniform scene: only noise events; dark scenes are noisier than bright ones
        let (w, h) = (200, 100);
        let base = EventConfig { hot_pixel_fraction: 0.0, max_rate_mev_s: 0.0, ..Default::default() };
        let mut rates = vec![];
        for lum in [0.5f32, 0.002] {
            let mut s = EventSensor::new(base.clone(), w, h);
            let rad = vec![lum; w * h * 3];
            let l = s.log_image(&rad);
            let mut n = 0;
            for k in 0..=200 {
                n += s.step(k as f64 * 0.01, &l).len();
            }
            rates.push(n as f64 / (w * h) as f64 / 2.0);
        }
        // bright: ~shot (0.3*sqrt(0.2/0.5)=0.19) + leak (~0.1) Hz/px
        assert!(rates[0] > 0.1 && rates[0] < 0.6, "{rates:?}");
        assert!(rates[1] > 3.0 * rates[0], "{rates:?}");
    }

    #[test]
    fn low_light_bandwidth_delays_events() {
        // a sudden brightening: the bright pixel reacts within ~1 ms, the dark one much later
        let cfg = EventConfig { contrast_sigma: 0.0, shot_noise_hz: 0.0, leak_hz: 0.0, hot_pixel_fraction: 0.0, timestamp_jitter_us: 0.0, ..Default::default() };
        let first = |lum0: f32| -> f64 {
            let mut s = EventSensor::new(cfg.clone(), 1, 1);
            s.step(0.0, &s.log_image(&[lum0; 3]));
            for k in 1..=4000 {
                let ev = s.step(k as f64 * 1e-4, &s.log_image(&[lum0 * 4.0; 3]));
                if let Some(e) = ev.first() {
                    return e.t_us as f64 * 1e-6;
                }
            }
            f64::MAX
        };
        let (tb, td) = (first(0.5), first(2e-7));
        assert!(tb < 2e-3 && td > 10.0 * tb, "bright {tb} dark {td}");
    }
}
