//! Event camera simulation (ESIM-style).
//!
//! An event camera reports, per pixel and asynchronously, when the log intensity has changed by a
//! contrast threshold C since the pixel's last event: e = (x, y, t, p). We simulate it from the
//! renderer:
//!
//! 1. Adaptive sampling: the scene is rendered (log-luminance, no sensor post-processing) at
//!    times chosen so that image motion between consecutive renders stays below
//!    `max_px_per_step` (estimated from the exact geometry: the 3D points of the current render
//!    projected with the next pose). Trajectory vibration therefore produces dense sampling only
//!    when it actually moves the image.
//! 2. Per pixel, log intensity is linearly interpolated between renders; every crossing of the
//!    reference level ± C emits an event at the interpolated time (several per step if the
//!    change spans several thresholds), subject to a refractory period.
//! 3. Sensor non-idealities: per-pixel threshold mismatch (Gaussian σ), shot-noise events
//!    (Poisson, random polarity) and leak events (slow drift of the reference level → positive
//!    events).
//!
//! Output: `<camera.path>/events/{x, y, t, p, ms_index}` in the sequence file. `t` is i64 µs on
//! the sequence clock (shared with frames, IMU and poses), `p` is 1 for ON and 0 for OFF, and
//! `ms_index[m]` is the index of the first event at or after m ms (plus one closing entry).
//! The event camera's pose at any time is the body pose (`/pose`) ∘ `calib/T_body_cam`.

use crate::cache::TileCache;
use crate::camera::CameraModel;
use crate::lighting::SunState;
use crate::raster::{RenderSettings, Renderer};
use crate::scenario::{CameraSpec, Scenario};
use crate::trajectory::{self, Pose};
use anyhow::{bail, Result};
use geodesy::Ellipsoid;
use h5::Attrs;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    /// Max image motion between internal renders (px).
    pub max_px_per_step: f64,
    /// Internal render-rate bounds (Hz).
    pub min_rate_hz: f64,
    pub max_rate_hz: f64,
    /// Supersampling of the internal renders (anti-aliasing reduces spurious events).
    pub supersample: u32,
    /// When artificial lights are on and flicker is enabled, render at least this many times per
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

/// Per-pixel event generator state.
pub struct EventSensor {
    w: usize,
    h: usize,
    cfg: EventConfig,
    /// low-pass filtered log intensity (photoreceptor output) and the reference level
    lp: Vec<f32>,
    l_ref: Vec<f32>,
    last_t: Vec<f64>,
    c_pos: Vec<f32>,
    c_neg: Vec<f32>,
    leak: Vec<f32>,
    hot: Vec<(usize, f64, f64)>,
    t_prev: Option<f64>,
    rng: Rng,
}

impl EventSensor {
    pub fn new(cfg: EventConfig, w: usize, h: usize) -> Self {
        let mut rng = Rng(cfg.seed ^ 0xE7E7);
        let n = w * h;
        let mut c_pos = Vec::with_capacity(n);
        let mut c_neg = Vec::with_capacity(n);
        let mut leak = Vec::with_capacity(n);
        for _ in 0..n {
            c_pos.push((cfg.contrast_pos + cfg.contrast_sigma * rng.gauss()).max(0.01) as f32);
            c_neg.push((cfg.contrast_neg + cfg.contrast_sigma * rng.gauss()).max(0.01) as f32);
            leak.push((cfg.leak_hz * (cfg.leak_sigma * rng.gauss() - 0.5 * cfg.leak_sigma * cfg.leak_sigma).exp()) as f32);
        }
        // hot pixels: random positions, rate spread, polarity bias
        let nh = (cfg.hot_pixel_fraction * n as f64).round() as usize;
        let hot = (0..nh)
            .map(|_| ((rng.next() % n as u64) as usize, cfg.hot_pixel_hz * (0.3 + 1.4 * rng.uniform()), 0.2 + 0.6 * rng.uniform()))
            .collect();
        EventSensor { w, h, lp: vec![0.0; n], l_ref: vec![0.0; n], last_t: vec![f64::MIN; n], c_pos, c_neg, leak, hot, t_prev: None, rng, cfg }
    }

    /// Log intensity (photoreceptor input) from linear RGB radiance.
    pub fn log_image(&self, radiance: &[f32]) -> Vec<f32> {
        let g = self.cfg.gain as f32;
        let eps = self.cfg.log_eps as f32;
        radiance.chunks_exact(3).map(|c| (g * (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) + eps).ln()).collect()
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

    /// Feed the log image rendered at time `t` (s); returns the events in (t_prev, t], sorted.
    pub fn step(&mut self, t: f64, l: &[f32]) -> Vec<Event> {
        let Some(t0) = self.t_prev else {
            self.lp.copy_from_slice(l);
            self.l_ref.copy_from_slice(l);
            self.t_prev = Some(t);
            return vec![];
        };
        let dt = t - t0;
        if dt <= 0.0 {
            return vec![];
        }
        let c = self.cfg.clone();
        let refr = c.refractory_us * 1e-6;
        let eps = c.log_eps as f32;
        let a_hpf = if c.hpf_hz > 0.0 { (1.0 - (-std::f64::consts::TAU * c.hpf_hz * dt).exp()) as f32 } else { 0.0 };
        let mut ev = Vec::new();
        for k in 0..self.w * self.h {
            // photoreceptor low-pass: cutoff ∝ photocurrent (luminance)
            let lum = ((l[k].exp() - eps) / c.gain as f32).max(0.0) as f64;
            let fc = (c.cutoff_hz * lum / (lum + c.cutoff_half_lum)).max(c.cutoff_min_hz);
            let alpha = (1.0 - (-std::f64::consts::TAU * fc * dt).exp()) as f32;
            let lp0 = self.lp[k];
            let lp1 = lp0 + alpha * (l[k] - lp0);
            self.lp[k] = lp1;
            // high-pass: reference relaxes towards the signal; leak: reference drifts down
            if a_hpf > 0.0 {
                self.l_ref[k] += a_hpf * (lp0 - self.l_ref[k]);
            }
            self.l_ref[k] -= (self.leak[k] as f64 * dt) as f32 * self.c_pos[k];
            let dl = lp1 - lp0;
            let (x, y) = ((k % self.w) as u16, (k / self.w) as u16);
            loop {
                let up = lp1 - self.l_ref[k] >= self.c_pos[k];
                let down = self.l_ref[k] - lp1 >= self.c_neg[k];
                if !up && !down {
                    break;
                }
                let target = if up { self.l_ref[k] + self.c_pos[k] } else { self.l_ref[k] - self.c_neg[k] };
                // crossing time by linear interpolation over the step (leak crossings land
                // uniformly in the step)
                let a = if dl.abs() > 1e-9 { ((target - lp0) / dl).clamp(0.0, 1.0) as f64 } else { self.rng.uniform() };
                let te = t0 + a * dt;
                self.l_ref[k] = target;
                if te - self.last_t[k] >= refr {
                    self.last_t[k] = te;
                    ev.push(Event { t_us: (te * 1e6).round() as i64, x, y, p: if up { 1 } else { 0 } });
                }
            }
            // shot-noise background activity, stronger in the dark
            if c.shot_noise_hz > 0.0 {
                let dark = (c.noise_ref_lum / (lum + 1e-6)).sqrt().clamp(1.0 / c.shot_noise_dark_gain.max(1.0), c.shot_noise_dark_gain);
                let mean = c.shot_noise_hz * dark * dt;
                if self.rng.uniform() < mean {
                    let te = t0 + self.rng.uniform() * dt;
                    if te - self.last_t[k] >= refr {
                        self.last_t[k] = te;
                        ev.push(Event { t_us: (te * 1e6).round() as i64, x, y, p: (self.rng.next() & 1) as i8 });
                    }
                }
            }
        }
        // hot pixels
        for i in 0..self.hot.len() {
            let (k, rate, pol) = self.hot[i];
            let n = self.poisson(rate * dt);
            for _ in 0..n {
                let te = t0 + self.rng.uniform() * dt;
                let p = if self.rng.uniform() < pol { 1 } else { 0 };
                ev.push(Event { t_us: (te * 1e6).round() as i64, x: (k % self.w) as u16, y: (k / self.w) as u16, p });
            }
        }
        // timestamp jitter
        if c.timestamp_jitter_us > 0.0 {
            let (lo, hi) = ((t0 * 1e6).ceil() as i64, (t * 1e6).floor() as i64);
            for e in ev.iter_mut() {
                e.t_us = (e.t_us + (c.timestamp_jitter_us * self.rng.gauss()).round() as i64).clamp(lo, hi.max(lo));
            }
        }
        ev.sort_unstable_by_key(|e| e.t_us);
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
                let allowed = cap * span_us / 1000.0;
                let keep = (allowed / (j - i) as f64).min(1.0);
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

/// Simulate the events of one camera over [t_start, t_end] (trajectory time; t_start is the
/// sequence start) into `<spec.path>/events` of the open sequence file.
/// `progress(t_done, t_total)`.
#[allow(clippy::too_many_arguments)]
pub fn simulate(scn: &Scenario, spec: &CameraSpec, poses: &[Pose], cache: Arc<TileCache>, ell: Ellipsoid, file: &h5::File, (t_start, t_end): (f64, f64), progress: &dyn Fn(f64, f64)) -> Result<usize> {
    let Some(ec) = &spec.events else { bail!("camera {} has no events modality", spec.path) };
    if poses.len() < 2 {
        bail!("event simulation needs a trajectory");
    }
    let ext = &spec.extrinsics;
    let model: Arc<dyn CameraModel> = spec.intrinsics.build()?;
    let (w, h) = (model.width() as usize, model.height() as usize);
    let mut rs: RenderSettings = scn.render.clone();
    rs.supersample = ec.supersample.max(1);
    rs.min_zoom = scn.tiles.min_zoom;
    rs.max_zoom = rs.max_zoom.min(scn.tiles.max_zoom);
    let renderer = Renderer::new(model.clone(), rs, ell, cache);
    let mut sensor = EventSensor::new(ec.clone(), w, h);
    let g = file.ensure_group(crate::scenario::h5path(&spec.path))?;
    if !g.exists("calib") {
        crate::output::write_camera_calib(&g, &spec.intrinsics, crate::output::transform_4x4(ext.r_body_cam(), ext.t_body_cam()))?;
    }
    let mut writer = EventWriter::new(&g, t_start, scn.output.compression.level)?;

    let tr0 = poses[0].t;
    let sun_at = |t: f64, p: &Pose| -> SunState { scn.render.lighting.sun_at(t - tr0, p.geo.lat, p.geo.lon) };
    let mut t = t_start;
    let mut dt = 1.0 / ec.max_rate_hz.max(1.0);
    let mut n_events = 0;
    progress(0.0, t_end - t_start);
    while t <= t_end + 1e-12 {
        let pose = trajectory::interpolate(poses, t);
        let cam = pose.camera(ext, &ell);
        let sun = sun_at(t, &pose);
        let frame = renderer.render(&cam, &sun);
        let l = sensor.log_image(&frame.radiance);
        let ev = sensor.step(t, &l);
        n_events += ev.len();
        writer.write(&ev)?;
        // next step: limit the image motion of the rendered points to max_px_per_step
        let probe = (t + dt).min(t_end.max(t + 1e-6));
        let cam_n = trajectory::interpolate(poses, probe).camera(ext, &ell);
        let mut maxd: f64 = 0.0;
        let step = (w.max(h) / 24).max(1);
        for y in (step / 2..h).step_by(step) {
            for x in (step / 2..w).step_by(step) {
                if let Some(p) = frame.points[y * w + x] {
                    if let (Some(a), Some(b)) = (model.project(cam.world_to_cam(p)), model.project(cam_n.world_to_cam(p))) {
                        maxd = maxd.max((a - b).length());
                    }
                }
            }
        }
        let rate = if maxd > 1e-9 { maxd / (probe - t).max(1e-9) } else { 0.0 }; // px/s
        dt = if rate > 0.0 { ec.max_px_per_step / rate } else { 1.0 / ec.min_rate_hz };
        dt = dt.clamp(1.0 / ec.max_rate_hz, 1.0 / ec.min_rate_hz);
        if sun.lights > 0.01 && sun.flicker.enabled && ec.flicker_steps_per_period > 0.0 {
            dt = dt.min(1.0 / (2.0 * sun.flicker.mains_hz * ec.flicker_steps_per_period)).max(1.0 / ec.max_rate_hz);
        }
        if t >= t_end {
            break;
        }
        t = (t + dt).min(t_end);
        progress(t - t_start, t_end - t_start);
    }
    writer.finish(ec)?;
    Ok(n_events)
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
