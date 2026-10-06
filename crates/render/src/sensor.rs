//! Camera sensor model: motion blur, optics, auto exposure (1st-order ODE on EV), noise, tone curve.
//!
//! Input is the renderer's scene-linear radiance (sunlit white Lambertian ≈ 1.0). Processing order:
//! motion blur → chromatic aberration → defocus → bloom → exposure (time × gain) → vignetting →
//! noise (shot + read, scaled by gain; fixed-pattern PRNU) → white balance / saturation → tone
//! curve → 8-bit quantization. Ground truth (depth / flow / poses) refers to the mid-exposure pose.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExposureMode {
    Auto,
    Manual,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExposureConfig {
    pub mode: ExposureMode,
    /// Manual: exposure multiplier in stops relative to `base_time`. Auto: compensation (stops).
    pub ev: f64,
    /// Auto: target mean linear luminance of the exposed image.
    pub target: f64,
    /// Auto: adaptation time constant (s) of the EV ODE  ev' = (ev* - ev) / tau.
    pub tau: f64,
    /// Exposure time (s) at EV 0.
    pub base_time: f64,
    pub min_time: f64,
    pub max_time: f64,
    /// Max analog gain (linear). Exposure beyond max_time is reached with gain.
    pub max_gain: f64,
    /// Fraction of the metering weight on the image centre (0 = average metering).
    pub center_weight: f64,
}

impl Default for ExposureConfig {
    fn default() -> Self {
        ExposureConfig {
            mode: ExposureMode::Auto,
            ev: 0.0,
            target: 0.16,
            tau: 0.6,
            base_time: 1.0 / 1000.0,
            min_time: 1.0 / 20000.0,
            max_time: 1.0 / 60.0,
            max_gain: 16.0,
            center_weight: 0.3,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MotionBlurConfig {
    pub enabled: bool,
    /// Max sub-frame samples along the exposure window.
    pub max_samples: u32,
    /// Fraction of the exposure time the (global) shutter is open (1 = full).
    pub shutter: f64,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        MotionBlurConfig { enabled: true, max_samples: 16, shutter: 1.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpticsConfig {
    /// Gaussian defocus / lens softness (sigma, px).
    pub defocus_px: f64,
    /// Lateral chromatic aberration: R/B channel displacement at the image corner (px).
    pub chromatic_aberration_px: f64,
    /// Vignetting: brightness factor 1 - v * (r / r_corner)^2.
    pub vignette: f64,
    /// Bloom strength and threshold (exposed linear value) — glare around lights / sun glints.
    pub bloom: f64,
    pub bloom_threshold: f64,
    pub bloom_sigma_px: f64,
    /// Diffraction spikes (aperture blades) on very bright points: strength, threshold
    /// (exposed linear value), streak length (px), number of rays (4 or 8).
    pub starburst: f64,
    pub starburst_threshold: f64,
    pub starburst_len_px: f64,
    pub starburst_rays: u32,
}

impl Default for OpticsConfig {
    fn default() -> Self {
        OpticsConfig {
            defocus_px: 0.35,
            chromatic_aberration_px: 0.35,
            vignette: 0.25,
            bloom: 0.05,
            bloom_threshold: 1.0,
            bloom_sigma_px: 6.0,
            starburst: 0.15,
            starburst_threshold: 2.0,
            starburst_len_px: 18.0,
            starburst_rays: 8,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NoiseConfig {
    pub enabled: bool,
    /// Read noise (8-bit DN at gain 1, grows with gain).
    pub read: f64,
    /// Shot noise scale: sigma_DN = shot * sqrt(gain * signal_DN).
    pub shot: f64,
    /// Photo-response non-uniformity (fixed pattern, relative).
    pub prnu: f64,
    pub seed: u64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        NoiseConfig { enabled: true, read: 0.12, shot: 0.15, prnu: 0.004, seed: 7 }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToneCurve {
    Srgb,
    Gamma,
    /// Filmic shoulder (ACES-like fit) — softer highlights.
    Filmic,
    /// No curve (linear 8-bit).
    Linear,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToneConfig {
    pub curve: ToneCurve,
    pub gamma: f64,
    pub white_balance: [f64; 3],
    pub saturation: f64,
}

impl Default for ToneConfig {
    fn default() -> Self {
        ToneConfig { curve: ToneCurve::Srgb, gamma: 2.2, white_balance: [1.0, 1.0, 1.0], saturation: 1.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct SensorSettings {
    pub exposure: ExposureConfig,
    pub motion_blur: MotionBlurConfig,
    pub optics: OpticsConfig,
    pub noise: NoiseConfig,
    pub tone: ToneConfig,
}

/// Exposure chosen for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exposure {
    pub ev: f64,
    pub time: f64,
    pub gain: f64,
    /// overall linear multiplier applied to radiance
    pub scale: f64,
}

/// Sensor state carried between frames (AE ODE state, fixed pattern noise).
pub struct Sensor {
    pub cfg: SensorSettings,
    w: usize,
    h: usize,
    ev: Option<f64>,
    last_t: Option<f64>,
    metered: Option<f64>,
    prnu: Vec<f32>,
}

/// splitmix-based Gaussian generator
struct Rng(u64);
impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    #[inline]
    fn gauss(&mut self) -> f64 {
        // sum of 4 uniforms: cheap, adequate for sensor noise
        let mut s = 0.0;
        for _ in 0..4 {
            s += (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        }
        (s - 2.0) * (3.0f64).sqrt()
    }
}

#[inline]
fn lum(c: [f32; 3]) -> f64 {
    0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64
}

fn gaussian_kernel(sigma: f64) -> Vec<f32> {
    let r = (3.0 * sigma).ceil().max(1.0) as i32;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Separable Gaussian blur of an RGB f32 image (in place).
pub fn blur_rgb(img: &mut [f32], w: usize, h: usize, sigma: f64) {
    use rayon::prelude::*;
    if sigma < 0.05 {
        return;
    }
    let k = gaussian_kernel(sigma);
    let r = (k.len() / 2) as isize;
    let src = img.to_vec();
    let mut tmp = vec![0f32; img.len()];
    tmp.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = [0f32; 3];
            for (i, kv) in k.iter().enumerate() {
                let xx = (x as isize + i as isize - r).clamp(0, w as isize - 1) as usize;
                let p = &src[(y * w + xx) * 3..(y * w + xx) * 3 + 3];
                acc[0] += kv * p[0];
                acc[1] += kv * p[1];
                acc[2] += kv * p[2];
            }
            row[x * 3..x * 3 + 3].copy_from_slice(&acc);
        }
    });
    img.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = [0f32; 3];
            for (i, kv) in k.iter().enumerate() {
                let yy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
                let p = &tmp[(yy * w + x) * 3..(yy * w + x) * 3 + 3];
                acc[0] += kv * p[0];
                acc[1] += kv * p[1];
                acc[2] += kv * p[2];
            }
            row[x * 3..x * 3 + 3].copy_from_slice(&acc);
        }
    });
}

/// Exponentially decaying streaks along 2 (rays=4) or 4 (rays=8) line directions through every
/// bright pixel (recursive filters forward and backward). Energy per ray ≈ source / rays.
fn starburst(src: &[f32], w: usize, h: usize, len: f64, rays: u32) -> Vec<f32> {
    let a = (-1.0 / len.max(1.0)).exp() as f32;
    let dirs: &[(isize, isize)] = if rays >= 8 { &[(1, 0), (0, 1), (1, 1), (1, -1)] } else { &[(1, 0), (0, 1)] };
    let norm = (1.0 - a) / (2 * dirs.len()) as f32;
    let mut out = vec![0f32; src.len()];
    let idx = |x: isize, y: isize| -> Option<usize> {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            Some((y as usize * w + x as usize) * 3)
        } else {
            None
        }
    };
    let mut buf = vec![0f32; src.len()];
    for &(dx, dy) in dirs {
        for sign in [1isize, -1] {
            let (dx, dy) = (dx * sign, dy * sign);
            // visit pixels so that the predecessor (x-dx, y-dy) is processed first
            let ys: Vec<isize> = if dy >= 0 { (0..h as isize).collect() } else { (0..h as isize).rev().collect() };
            let xs: Vec<isize> = if dx >= 0 { (0..w as isize).collect() } else { (0..w as isize).rev().collect() };
            for &y in &ys {
                for &x in &xs {
                    let k = idx(x, y).unwrap();
                    let prev = idx(x - dx, y - dy);
                    for c in 0..3 {
                        let p = prev.map(|pk| buf[pk + c]).unwrap_or(0.0);
                        buf[k + c] = src[k + c] + a * p;
                    }
                }
            }
            for (o, (b, s0)) in out.iter_mut().zip(buf.iter().zip(src.iter())) {
                *o += (b - s0) * norm;
            }
        }
    }
    out
}

#[inline]
fn bilinear(img: &[f32], w: usize, h: usize, x: f64, y: f64, c: usize) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let p = |xx: usize, yy: usize| img[(yy * w + xx) * 3 + c];
    let a = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * fx;
    let b = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * fx;
    a + (b - a) * fy
}

impl Sensor {
    pub fn new(cfg: SensorSettings, w: usize, h: usize) -> Self {
        let mut rng = Rng(cfg.noise.seed ^ 0xF1AE_D9A7);
        let prnu = (0..w * h * 3).map(|_| (1.0 + cfg.noise.prnu * rng.gauss()) as f32).collect();
        Sensor { cfg, w, h, ev: None, last_t: None, metered: None, prnu }
    }

    /// Exposure for the frame at time `t` (advances the AE ODE using the previous metering).
    pub fn exposure_for(&mut self, t: f64) -> Exposure {
        let c = &self.cfg.exposure;
        let ev = match c.mode {
            ExposureMode::Manual => c.ev,
            ExposureMode::Auto => {
                let target = self.metered.map(|m| (c.target / m.max(1e-6)).log2() + c.ev);
                match (self.ev, target, self.last_t) {
                    (Some(ev), Some(tg), Some(t0)) => {
                        let dt = (t - t0).max(0.0);
                        ev + (tg - ev) * (1.0 - (-dt / c.tau.max(1e-3)).exp())
                    }
                    (_, Some(tg), _) => tg,
                    (Some(ev), None, _) => ev,
                    (None, None, _) => 0.0,
                }
            }
        };
        self.ev = Some(ev);
        self.last_t = Some(t);
        let e = 2f64.powf(ev);
        let time = (c.base_time * e).clamp(c.min_time, c.max_time);
        let gain = (e * c.base_time / time).clamp(1.0, c.max_gain);
        Exposure { ev, time, gain, scale: time * gain / c.base_time }
    }

    /// Meter a radiance image (sets the AE input for the next frame). Before the first frame is
    /// processed, call this once on the first rendered frame to start converged.
    pub fn meter(&mut self, radiance: &[f32]) {
        let (w, h) = (self.w, self.h);
        let cw = self.cfg.exposure.center_weight;
        let mut sum = 0.0;
        let mut wsum = 0.0;
        for y in (0..h).step_by(2) {
            for x in (0..w).step_by(2) {
                let k = (y * w + x) * 3;
                let dx = (x as f64 / w as f64 - 0.5) * 2.0;
                let dy = (y as f64 / h as f64 - 0.5) * 2.0;
                let wt = 1.0 - cw + cw * (1.0 - (dx * dx + dy * dy) * 0.5).max(0.0) * 2.0;
                // clip highlights so the sun / glints do not dominate the metering
                sum += lum([radiance[k], radiance[k + 1], radiance[k + 2]]).min(4.0) * wt;
                wsum += wt;
            }
        }
        self.metered = Some(sum / wsum.max(1e-9));
    }

    pub fn has_metering(&self) -> bool {
        self.metered.is_some()
    }

    /// Motion blur by gathering along per-pixel image trajectories. `disp` holds, for every pixel,
    /// `k` sub-frame displacements (px, x/y interleaved) of the scene point seen at that pixel
    /// relative to the mid-exposure image: blurred(x) = mean_i radiance(x - d_i).
    pub fn motion_blur(&self, radiance: &[f32], disp: &[f32], k: usize) -> Vec<f32> {
        use rayon::prelude::*;
        let (w, h) = (self.w, self.h);
        if k <= 1 {
            return radiance.to_vec();
        }
        let mut out = vec![0f32; radiance.len()];
        out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let d = &disp[(y * w + x) * 2 * k..(y * w + x + 1) * 2 * k];
                let mut acc = [0f32; 3];
                for i in 0..k {
                    let sx = x as f64 - d[2 * i] as f64;
                    let sy = y as f64 - d[2 * i + 1] as f64;
                    for c in 0..3 {
                        acc[c] += bilinear(radiance, w, h, sx, sy, c);
                    }
                }
                for c in 0..3 {
                    row[x * 3 + c] = acc[c] / k as f32;
                }
            }
        });
        out
    }

    /// Full sensor pipeline from (already motion-blurred) radiance to 8-bit sRGB.
    pub fn develop(&self, radiance: &[f32], ex: &Exposure, frame_idx: u64) -> Vec<u8> {
        use rayon::prelude::*;
        let (w, h) = (self.w, self.h);
        let o = &self.cfg.optics;
        let mut img = radiance.to_vec();
        // chromatic aberration (radial scaling of R and B about the centre)
        if o.chromatic_aberration_px.abs() > 1e-3 {
            let cx = (w as f64 - 1.0) * 0.5;
            let cy = (h as f64 - 1.0) * 0.5;
            let rc = (cx * cx + cy * cy).sqrt();
            let k = o.chromatic_aberration_px / rc;
            let src = img.clone();
            img.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
                for x in 0..w {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    row[x * 3] = bilinear(&src, w, h, cx + dx * (1.0 + k), cy + dy * (1.0 + k), 0);
                    row[x * 3 + 2] = bilinear(&src, w, h, cx + dx * (1.0 - k), cy + dy * (1.0 - k), 2);
                }
            });
        }
        blur_rgb(&mut img, w, h, o.defocus_px);
        // exposure
        let scale = ex.scale as f32;
        img.par_iter_mut().for_each(|v| *v *= scale);
        // bloom from over-exposed regions
        if o.bloom > 0.0 {
            let thr = o.bloom_threshold as f32;
            let mut b: Vec<f32> = img.iter().map(|v| (v - thr).max(0.0)).collect();
            if b.iter().any(|v| *v > 0.0) {
                blur_rgb(&mut b, w, h, o.bloom_sigma_px);
                let s = o.bloom as f32 * 4.0;
                img.par_iter_mut().zip(b.par_iter()).for_each(|(v, bb)| *v += s * bb);
            }
        }
        if o.starburst > 0.0 {
            let thr = o.starburst_threshold as f32;
            let src: Vec<f32> = img.iter().map(|v| (v - thr).max(0.0)).collect();
            if src.iter().any(|v| *v > 0.0) {
                let streaks = starburst(&src, w, h, o.starburst_len_px, o.starburst_rays);
                let s = o.starburst as f32;
                img.par_iter_mut().zip(streaks.par_iter()).for_each(|(v, b)| *v += s * b);
            }
        }
        let t = &self.cfg.tone;
        let n = &self.cfg.noise;
        let cx = (w as f64 - 1.0) * 0.5;
        let cy = (h as f64 - 1.0) * 0.5;
        let r2c = cx * cx + cy * cy;
        let mut out = vec![0u8; w * h * 3];
        out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            let mut rng = Rng(n.seed ^ frame_idx.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ (y as u64).wrapping_mul(0x9E37_79B9));
            for x in 0..w {
                let k = (y * w + x) * 3;
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let vig = 1.0 - o.vignette * (dx * dx + dy * dy) / r2c;
                let mut c = [0f64; 3];
                for ch in 0..3 {
                    let mut s = img[k + ch] as f64 * vig * t.white_balance[ch];
                    if n.enabled {
                        s *= self.prnu[k + ch] as f64;
                        let dn = (s * 255.0).max(0.0);
                        let sigma = ((n.read * ex.gain).powi(2) + n.shot * n.shot * ex.gain * dn).sqrt();
                        s += sigma * rng.gauss() / 255.0;
                    }
                    c[ch] = s;
                }
                if t.saturation != 1.0 {
                    let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                    for v in &mut c {
                        *v = l + (*v - l) * t.saturation;
                    }
                }
                for ch in 0..3 {
                    let v = c[ch].max(0.0);
                    let e = match t.curve {
                        ToneCurve::Srgb => {
                            let v = v.min(1.0);
                            if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
                        }
                        ToneCurve::Gamma => v.min(1.0).powf(1.0 / t.gamma),
                        ToneCurve::Filmic => {
                            let a = v * 0.8;
                            let f = (a * (2.51 * a + 0.03)) / (a * (2.43 * a + 0.59) + 0.14);
                            f.clamp(0.0, 1.0).powf(1.0 / 2.2)
                        }
                        ToneCurve::Linear => v.min(1.0),
                    };
                    row[x * 3 + ch] = (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_exposure_converges_with_lag() {
        let mut s = Sensor::new(SensorSettings::default(), 8, 8);
        let bright = vec![1.0f32; 8 * 8 * 3];
        let dark = vec![0.002f32; 8 * 8 * 3];
        s.meter(&bright);
        let e0 = s.exposure_for(0.0);
        assert!((e0.ev - (0.16f64).log2()).abs() < 1e-9);
        s.meter(&dark);
        let e1 = s.exposure_for(0.1);
        let e_inf = (0.16f64 / 0.002).log2();
        // after 0.1 s with tau 0.6 only ~15% of the way
        let frac = (e1.ev - e0.ev) / (e_inf - e0.ev);
        assert!(frac > 0.1 && frac < 0.2, "{frac}");
        let mut e = e1;
        for k in 2..100 {
            s.meter(&dark);
            e = s.exposure_for(k as f64 * 0.1);
        }
        assert!((e.ev - e_inf).abs() < 0.01);
        // dark scene needs more than max_time → gain kicks in
        assert!(e.time <= 1.0 / 60.0 + 1e-12 && e.gain > 1.0);
    }
}
