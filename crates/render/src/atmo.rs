//! Simple analytic sky and aerial perspective (exponential Rayleigh + Mie layers).

use crate::lighting::SunState;
use glam::DVec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AtmoParams {
    pub enabled: bool,
    /// Meteorological visibility at sea level (km) — sets the haze (Mie) density.
    pub visibility_km: f64,
    #[serde(rename = "rayleigh_scale_height_m")]
    pub rayleigh_scale_height: f64,
    #[serde(rename = "mie_scale_height_m")]
    pub mie_scale_height: f64,
    /// Brightness of the in-scattered light relative to a sunlit white surface.
    pub inscatter: f64,
}

impl Default for AtmoParams {
    fn default() -> Self {
        AtmoParams { enabled: true, visibility_km: 60.0, rayleigh_scale_height: 8000.0, mie_scale_height: 1600.0, inscatter: 0.85 }
    }
}

const BETA_R: [f64; 3] = [5.8e-6, 13.5e-6, 33.1e-6];

pub struct Atmosphere {
    pub(crate) p: AtmoParams,
    pub(crate) beta_m: f64,
    /// Linear-light colours
    pub(crate) rayleigh_col: DVec3,
    pub(crate) mie_col: DVec3,
    pub(crate) zenith: DVec3,
    pub(crate) horizon: DVec3,
    pub(crate) sun_col: DVec3,
    pub sun_dir: DVec3,
    pub moon_dir: DVec3,
    pub(crate) moon_col: DVec3,
    pub(crate) moon_disc: f64,
}

impl Atmosphere {
    /// `sun_dir`: unit vector towards the sun (ECEF); `sun`: current sun / sky state.
    pub fn new(p: AtmoParams, sun_dir: DVec3, moon_dir: DVec3, sun: &SunState) -> Self {
        let beta_m = 3.912 / (p.visibility_km.max(0.1) * 1000.0);
        let el = sun.elevation;
        let low = 1.0 - (el.max(0.0) / 0.45).min(1.0); // 1 at sunrise/sunset, 0 at high sun
        let sun_col = DVec3::new(1.0, 0.95 - 0.30 * low * low, 0.90 - 0.60 * low * low) * sun.direct;
        let sky = sun.sky;
        let dusk = DVec3::new(1.0, 0.62 - 0.1 * low, 0.40);
        let horizon_day = DVec3::new(0.62, 0.72, 0.85);
        let horizon = (horizon_day * (1.0 - 0.6 * low) + dusk * (0.6 * low)) * sky;
        let night = DVec3::new(0.4, 0.6, 1.2) * 1e-7;
        let moon_col = DVec3::new(0.80, 0.86, 1.0) * sun.moon_direct;
        Atmosphere {
            beta_m,
            rayleigh_col: DVec3::new(0.30, 0.48, 0.85) * (0.7 * sky) + sun_col * 0.15 * DVec3::new(0.3, 0.45, 0.8) + night,
            mie_col: DVec3::new(0.80, 0.82, 0.86) * (0.5 * sky) + sun_col * 0.35 + moon_col * 0.35 + night,
            zenith: DVec3::new(0.10, 0.22, 0.55) * sky * (1.0 - 0.5 * low) + night,
            horizon: horizon + night * 1.5,
            sun_col,
            sun_dir,
            moon_dir,
            moon_col,
            // lunar disc radiance relative to a sunlit white surface (~2500 cd/m² at full moon)
            moon_disc: 0.03 * sun.moon_phase * if sun.moon_elevation > -0.01 { 1.0 } else { 0.0 },
            p,
        }
    }

    pub fn sun_color(&self) -> DVec3 {
        self.sun_col
    }
    pub fn moon_color(&self) -> DVec3 {
        self.moon_col
    }

    /// Optical depth per channel between heights `h0` and `h1` (m above the ellipsoid) over
    /// slant distance `d` (straight segment, linear height change).
    fn optical_depth(&self, h0: f64, h1: f64, d: f64) -> (DVec3, f64) {
        let integ = |hs: f64| -> f64 {
            let a = (-h0.max(-500.0) / hs).exp();
            let b = (-h1.max(-500.0) / hs).exp();
            let dh = h1 - h0;
            if dh.abs() < 1.0 {
                0.5 * (a + b)
            } else {
                (a - b) * hs / dh
            }
        };
        let fr = integ(self.p.rayleigh_scale_height) * d;
        let fm = integ(self.p.mie_scale_height) * d;
        (DVec3::new(BETA_R[0], BETA_R[1], BETA_R[2]) * fr, self.beta_m * fm)
    }

    /// Transmittance and in-scattered radiance along a view segment of length `d` between heights
    /// `h_cam` and `h_pt` (aerial perspective: `colour * T + inscatter`).
    pub fn transmittance(&self, h_cam: f64, h_pt: f64, d: f64, view: DVec3) -> (DVec3, DVec3) {
        if !self.p.enabled {
            return (DVec3::ONE, DVec3::ZERO);
        }
        let (tr, tm) = self.optical_depth(h_cam, h_pt, d);
        let tau = tr + DVec3::splat(tm);
        let t = DVec3::new((-tau.x).exp(), (-tau.y).exp(), (-tau.z).exp());
        // forward scattering towards the sun brightens the haze
        let cos_s = view.dot(self.sun_dir);
        let mie_phase = 0.7 + 0.8 * cos_s.max(0.0).powi(4);
        let wr = tr.x + tr.y + tr.z;
        let wm = 3.0 * tm;
        let mix = (self.rayleigh_col * wr + self.mie_col * mie_phase * wm) / (wr + wm).max(1e-12);
        (t, mix * (DVec3::ONE - t) * self.p.inscatter)
    }

    /// Apply aerial perspective to a surface colour seen over distance `d`.
    pub fn apply(&self, col: DVec3, h_cam: f64, h_pt: f64, d: f64, view: DVec3) -> DVec3 {
        let (t, ins) = self.transmittance(h_cam, h_pt, d, view);
        col * t + ins
    }

    /// Sky radiance for a view direction (`up`: local up at the camera).
    pub fn sky(&self, dir: DVec3, up: DVec3) -> DVec3 {
        let e = dir.dot(up);
        let t = (1.0 - e.max(0.0)).powf(3.0);
        let mut c = self.zenith * (1.0 - t) + self.horizon * t;
        if e < 0.0 {
            // below the geometric horizon without terrain: hazy ground glow
            c = self.horizon * (0.9 + 0.1 * e);
        }
        let cs = dir.dot(self.sun_dir);
        c += self.sun_col * (0.25 * cs.max(0.0).powi(8) + 0.4 * cs.max(0.0).powi(64));
        if cs > 0.99996 && e > -0.01 {
            c += self.sun_col * 20.0;
        }
        if self.moon_disc > 0.0 {
            let cm = dir.dot(self.moon_dir);
            if cm > 0.99999 {
                c += DVec3::new(0.95, 0.95, 1.0) * self.moon_disc;
            }
            c += DVec3::new(0.8, 0.85, 1.0) * (self.moon_disc * 2e-3 * cm.max(0.0).powi(512));
        }
        c
    }
}
