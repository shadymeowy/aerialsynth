//! Stars from a real catalogue at their apparent positions for the scenario's date, time and
//! the camera's position (see docs/stars.md).
//!
//! Per render: catalogue (ICRS, J2000) → proper motion → annual parallax → light deflection by
//! the Sun → aberration (orbital + diurnal velocity) → GCRS → ITRS (precession, nutation, sidereal time, polar motion) →
//! refraction at the observer (standard atmosphere at its altitude) → camera frame → camera
//! model. Brightness is radiometric: V magnitude → irradiance relative to the Sun's
//! (V = −26.74), dimmed by the renderer's atmosphere along the line of sight, coloured by the
//! star's blackbody temperature (from B−V) relative to the Sun's white. Each star is a point
//! source integrated over the pixels with a Gaussian PSF, added where the pixel shows sky.
//! Ground truth: every catalogue star in the image with its sub-pixel position.

pub mod astro;
pub mod catalog;
pub mod ephem;
pub mod planets;

use crate::atmo::AtmoParams;
use crate::camera::CameraModel;
use crate::raster::FrameOut;
use crate::trajectory::CamPose;
use anyhow::Result;
use astro::Sky;
use catalog::Catalog;
use geodesy::Ellipsoid;
use glam::{DMat3, DVec2, DVec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StarsConfig {
    /// Catalogue file (`scripts/build_stars.py`); default: built-in Hipparcos + Tycho-2, V ≤ 9.
    pub catalog: Option<String>,
    /// Faintest V magnitude rendered.
    pub mag_limit: f64,
    /// Gaussian PSF σ (px) of a star image, integrated over the pixels (keeps sub-pixel
    /// positions in the image); the camera's optics blur (`sensor.optics.defocus_px`) adds to it.
    pub psf_sigma_px: f64,
    /// Brightness scale (1 = physical).
    pub brightness: f64,
    pub refraction: bool,
    pub extinction: bool,
    pub aberration: bool,
    /// UT1 − UTC (s, IERS Bulletin A); |dut1| < 0.9 s, ≈ 15″ of sky rotation per second.
    pub dut1_s: f64,
    /// Polar motion x_p, y_p (arcsec, IERS).
    pub polar_motion_arcsec: [f64; 2],
    /// Planets (Mercury … Neptune, DE440, 1990–2060) as stars or discs; with the Moon in the
    /// ground truth.
    pub planets: bool,
}

impl Default for StarsConfig {
    fn default() -> Self {
        StarsConfig {
            catalog: None,
            mag_limit: 99.0,
            psf_sigma_px: 0.5,
            brightness: 1.0,
            refraction: true,
            extinction: true,
            aberration: true,
            dut1_s: 0.0,
            polar_motion_arcsec: [0.0, 0.0],
            planets: true,
        }
    }
}

/// A catalogue star in a rendered image.
#[derive(Clone, Copy, Debug)]
pub struct StarObs {
    pub id: u32,
    /// sub-pixel position at the frame time (pixel centres at integer coordinates)
    pub x: f32,
    pub y: f32,
    /// position averaged over the exposure (the centroid of the star's trail; = x, y without
    /// motion)
    pub xm: f32,
    pub ym: f32,
    /// catalogue V magnitude
    pub v: f32,
    /// V-band irradiance at the camera relative to the Sun's outside the atmosphere
    /// (after atmospheric extinction)
    pub irradiance: f32,
    /// the star's pixel shows sky (not hidden by terrain)
    pub visible: bool,
}

/// Apparent sun magnitude (V)
const V_SUN: f64 = -26.74;
/// Renderer radiance of a white Lambertian surface under the Sun outside the atmosphere
/// (direct sunlight is 1 at one air mass, transmittance 0.9; see lighting.rs).
const SUN_TOA: f64 = 1.0 / 0.9;

/// A star's ground truth (if in the image at the frame time) and its (pixel, rgb) additions.
type StarSplat = (Option<StarObs>, Vec<(usize, DVec3)>);

pub struct StarField {
    cat: Arc<Catalog>,
    cfg: StarsConfig,
    /// number of catalogue stars within the magnitude limit (the catalogue is sorted by V)
    n: usize,
    /// linear RGB per 0.01 of B−V from −0.5 (luminance 1, the Sun = white)
    colours: Vec<DVec3>,
}

impl StarField {
    pub fn new(cfg: &StarsConfig) -> Result<StarField> {
        let cat = Catalog::load(cfg.catalog.as_deref())?;
        let n = cat.stars.partition_point(|s| (s.v as f64) <= cfg.mag_limit);
        let sun = blackbody_rgb(bv_temperature(0.65));
        let colours = (0..=350)
            .map(|i| {
                let c = blackbody_rgb(bv_temperature(-0.5 + i as f64 * 0.01)) / sun;
                c / (0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z)
            })
            .collect();
        Ok(StarField { cat, cfg: cfg.clone(), n, colours })
    }

    /// Apparent directions (ITRS / ECEF unit vectors) of the catalogue stars within the
    /// magnitude limit for an observer at ECEF `pos` at Unix time `unix` (UTC), with the local
    /// up vector and the refraction parameters used.
    pub fn apparent(&self, unix: f64, pos: DVec3, ell: &Ellipsoid) -> Vec<DVec3> {
        let c = &self.cfg;
        let as2r = std::f64::consts::PI / 180.0 / 3600.0;
        let sky = Sky::new(unix, c.dut1_s, c.polar_motion_arcsec[0] * as2r, c.polar_motion_arcsec[1] * as2r);
        let vobs = sky.observer_vel_c(pos);
        let geo = geodesy::frames::ecef2geodetic(pos, ell);
        let up = DVec3::new(geo.lat.cos() * geo.lon.cos(), geo.lat.cos() * geo.lon.sin(), geo.lat.sin());
        let (p_hpa, t_k) = if c.refraction { astro::standard_atmosphere(geo.h) } else { (0.0, 288.0) };
        let dt = sky.years - (self.cat.epoch - 2000.0);
        let m = sky.gcrs_to_itrs;
        self.cat.stars[..self.n]
            .par_iter()
            .map(|s| {
                let mut u = (s.dir + s.pm * dt).normalize();
                if s.plx > 0.0 {
                    u = (u - sky.earth_pos * s.plx).normalize();
                }
                if c.aberration {
                    u = astro::deflect(u, sky.sun_to_earth, sky.sun_dist);
                    u = astro::aberrate(u, vobs, sky.sun_dist);
                }
                refract(m * u, up, p_hpa, t_k)
            })
            .collect()
    }

    /// The planets (drawn) and the Moon (ground truth only; the sky draws it) at Unix time
    /// `unix` for an observer at ECEF `pos`: apparent ITRS directions after refraction. Empty
    /// outside 1990–2060 or with `planets: false`.
    pub fn bodies(&self, unix: f64, pos: DVec3, ell: &Ellipsoid) -> Vec<Source> {
        let c = &self.cfg;
        if !c.planets {
            return vec![];
        }
        let as2r = std::f64::consts::PI / 180.0 / 3600.0;
        let sky = Sky::new(unix, c.dut1_s, c.polar_motion_arcsec[0] * as2r, c.polar_motion_arcsec[1] * as2r);
        let geo = geodesy::frames::ecef2geodetic(pos, ell);
        let up = DVec3::new(geo.lat.cos() * geo.lon.cos(), geo.lat.cos() * geo.lon.sin(), geo.lat.sin());
        let (p_hpa, t_k) = if c.refraction { astro::standard_atmosphere(geo.h) } else { (0.0, 288.0) };
        let mut list = planets::PLANETS.to_vec();
        list.push(ephem::Body::Moon);
        let Some(app) = planets::apparent(&sky, pos, &list) else { return vec![] };
        app.iter()
            .filter(|a| a.v <= c.mag_limit)
            .map(|a| Source {
                id: planets::body_id(a.body),
                dir: refract(sky.gcrs_to_itrs * a.dir, up, p_hpa, t_k),
                v: a.v as f32,
                bv: planets::colour_bv(a.body) as f32,
                radius: a.radius,
                draw: a.body != ephem::Body::Moon,
            })
            .collect()
    }

    /// Add the stars to `frame` (an instantaneous render with camera `cam`; `model` at output
    /// resolution) and record them in `frame.stars`.
    pub fn render(&self, frame: &mut FrameOut, model: &dyn CameraModel, cam: &CamPose, unix: f64, ell: &Ellipsoid, atmo: &AtmoParams) {
        let (w, h) = (frame.width as usize, frame.height as usize);
        frame.stars = self.render_track(&mut frame.radiance, &frame.points, (w, h), model, std::slice::from_ref(cam), cam, unix, ell, atmo);
    }

    /// Stars over an exposure: `track` holds the camera poses at equal time steps across the
    /// open shutter (one pose: instantaneous). Each star is splatted along its image track in
    /// steps of at most 0.25 px with equal energy per unit time, so camera rotation draws
    /// trails of the right length and brightness. Stars are added to the sky pixels of
    /// `radiance` (empty: ground truth only). Returns the stars in the image at the pose `mid`
    /// (the frame time).
    #[allow(clippy::too_many_arguments)]
    pub fn render_track(&self, radiance: &mut [f32], points: &[Option<DVec3>], (w, h): (usize, usize), model: &dyn CameraModel, track: &[CamPose], mid: &CamPose, unix: f64, ell: &Ellipsoid, atmo: &AtmoParams) -> Vec<StarObs> {
        let dirs = self.apparent(unix, mid.pos, ell);
        let mut sources: Vec<Source> = dirs.iter().zip(&self.cat.stars[..self.n]).map(|(d, s)| Source { id: s.id, dir: *d, v: s.v, bv: s.bv, radius: 0.0, draw: true }).collect();
        sources.extend(self.bodies(unix, mid.pos, ell));
        let rt = mid.r_ecef_cam.transpose();
        let geo = geodesy::frames::ecef2geodetic(mid.pos, ell);
        let up = DVec3::new(geo.lat.cos() * geo.lon.cos(), geo.lat.cos() * geo.lon.sin(), geo.lat.sin());
        // vertical optical depth above the camera per channel (the renderer's atmosphere)
        let tau = if self.cfg.extinction && atmo.enabled {
            let fr = (-geo.h.max(0.0) / atmo.rayleigh_scale_height).exp() * atmo.rayleigh_scale_height;
            let fm = (-geo.h.max(0.0) / atmo.mie_scale_height).exp() * atmo.mie_scale_height * 3.912 / (atmo.visibility_km.max(0.1) * 1000.0);
            DVec3::new(5.8e-6, 13.5e-6, 33.1e-6) * fr + DVec3::splat(fm)
        } else {
            DVec3::ZERO
        };
        let inside = |p: DVec2| p.x >= -0.5 && p.y >= -0.5 && p.x <= w as f64 - 0.5 && p.y <= h as f64 - 0.5;
        let tracks: Vec<DMat3> = track.iter().map(|c| c.r_ecef_cam.transpose()).collect();
        let sigma = self.cfg.psf_sigma_px.max(0.0);
        let r = if sigma < 0.05 { 0 } else { (3.0 * sigma + 0.5).ceil() as i64 };
        let sky = |x: i64, y: i64| points.get(y as usize * w + x as usize).is_some_and(|q| q.is_none());
        // per star: ground truth at `mid`, and its splat (pixel index, rgb) contributions
        let found: Vec<StarSplat> = sources
            .par_iter()
            .filter_map(|s| {
                let d = &s.dir;
                let pm = model.project(rt * *d);
                // positions along the exposure
                let pts: Vec<Option<DVec2>> = tracks.iter().map(|m| model.project(*m * *d)).collect();
                let gt_in = pm.is_some_and(inside);
                if !gt_in && !pts.iter().flatten().any(|p| inside(*p)) {
                    return None;
                }
                let el = d.dot(up).asin().to_degrees().max(-1.0);
                let airmass = 1.0 / (el.to_radians().sin().max(0.0) + 0.50572 * (el + 6.07995).powf(-1.6364));
                let t = (-tau * airmass).exp();
                let e = 10f64.powf(-0.4 * (s.v as f64 - V_SUN));
                let obs = pm.filter(|p| inside(*p)).map(|p| {
                    let (xi, yi) = (p.x.round() as i64, p.y.round() as i64);
                    // exposure-averaged position: mean of the track at equal time steps
                    let seg: Vec<DVec2> = pts.windows(2).filter_map(|q| Some((q[0]? + q[1]?) * 0.5)).collect();
                    let pmean = if seg.is_empty() { p } else { seg.iter().fold(DVec2::ZERO, |a, b| a + *b) / seg.len() as f64 };
                    StarObs { id: s.id, x: p.x as f32, y: p.y as f32, xm: pmean.x as f32, ym: pmean.y as f32, v: s.v, irradiance: (e * t.y) as f32, visible: sky(xi, yi) }
                });
                if radiance.is_empty() || !s.draw {
                    return obs.map(|o| (Some(o), vec![]));
                }
                let col = self.colours[(((s.bv as f64 + 0.5) / 0.01).round() as usize).min(350)];
                let rgb = col * t * e;
                // solid angle of a pixel at the star (camera model Jacobian)
                let c0 = pm.or_else(|| pts.iter().flatten().next().copied())?;
                let un = |dx: f64, dy: f64| model.unproject(DVec2::new(c0.x + dx, c0.y + dy)).map(|v| v.normalize());
                let omega = match (un(-0.5, 0.0), un(0.5, 0.0), un(0.0, -0.5), un(0.0, 0.5)) {
                    (Some(a), Some(b), Some(c), Some(d)) => (b - a).cross(d - c).length(),
                    _ => return obs.map(|o| (Some(o), vec![])),
                };
                if omega <= 0.0 {
                    return obs.map(|o| (Some(o), vec![]));
                }
                let scale = std::f64::consts::PI * SUN_TOA * self.cfg.brightness / omega;
                // sample positions at equal time steps along the track, each weighted by its
                // share of the exposure
                let mut samples: Vec<(DVec2, f64)> = vec![];
                if pts.len() == 1 {
                    samples.extend(pts[0].map(|p| (p, 1.0)));
                } else {
                    let nseg = (pts.len() - 1) as f64;
                    for seg in pts.windows(2) {
                        if let (Some(a), Some(b)) = (seg[0], seg[1]) {
                            let m = (((b - a).length() / 0.25).ceil() as usize).clamp(1, 4096);
                            samples.extend((0..m).map(|i| (a + (b - a) * ((i as f64 + 0.5) / m as f64), 1.0 / (nseg * m as f64))));
                        }
                    }
                }
                // resolved discs (planets): a uniform disc of the apparent radius, sampled on a
                // grid (≤ ~400 points) around every track sample
                let rpx = s.radius / omega.sqrt();
                if rpx > 0.3 {
                    let step = 0.25f64.max(rpx * (std::f64::consts::PI / 400.0).sqrt());
                    let k = (rpx / step).ceil() as i64;
                    let disc: Vec<DVec2> = (-k..=k)
                        .flat_map(|j| (-k..=k).map(move |i| DVec2::new(i as f64 * step, j as f64 * step)))
                        .filter(|o| o.length() <= rpx)
                        .collect();
                    let nd = disc.len().max(1) as f64;
                    samples = samples.iter().flat_map(|(p, w)| disc.iter().map(move |o| (*p + *o, *w / nd))).collect();
                }
                let mut adds: Vec<(usize, DVec3)> = vec![];
                for (p, wt) in &samples {
                    let (cx, cy) = (p.x.round() as i64, p.y.round() as i64);
                    let wts = |c: f64, i0: i64| -> Vec<f64> {
                        (i0 - r..=i0 + r)
                            .map(|i| {
                                if r == 0 {
                                    return 1.0;
                                }
                                let s = sigma * std::f64::consts::SQRT_2;
                                0.5 * (erf((i as f64 + 0.5 - c) / s) - erf((i as f64 - 0.5 - c) / s))
                            })
                            .collect()
                    };
                    let (wx, wy) = (wts(p.x, cx), wts(p.y, cy));
                    for (jy, y) in (cy - r..=cy + r).enumerate() {
                        if y < 0 || y >= h as i64 {
                            continue;
                        }
                        for (jx, x) in (cx - r..=cx + r).enumerate() {
                            if x < 0 || x >= w as i64 || !sky(x, y) {
                                continue;
                            }
                            adds.push((y as usize * w + x as usize, rgb * (wx[jx] * wy[jy] * *wt * scale)));
                        }
                    }
                }
                Some((obs, adds))
            })
            .collect();
        let mut out = Vec::with_capacity(found.len());
        for (o, adds) in found {
            for (k, v) in adds {
                radiance[3 * k] += v.x as f32;
                radiance[3 * k + 1] += v.y as f32;
                radiance[3 * k + 2] += v.z as f32;
            }
            out.extend(o);
        }
        out
    }
}

/// Topocentric azimuth (from north, clockwise), elevation (no refraction) and illuminated
/// fraction of the Moon from DE440 for an observer on the ellipsoid at `lat`, `lon` (rad).
/// None outside 1990–2060.
pub fn moon_topocentric(unix: f64, lat: f64, lon: f64) -> Option<(f64, f64, f64)> {
    let sky = Sky::new(unix, 0.0, 0.0, 0.0);
    let pos = geodesy::frames::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &Ellipsoid::WGS84);
    let a = planets::apparent(&sky, pos, &[ephem::Body::Moon])?[0];
    let e = geodesy::frames::ecef2enuv(sky.gcrs_to_itrs * a.dir, lat, lon);
    let el = e.z.clamp(-1.0, 1.0).asin();
    let az = e.x.atan2(e.y).rem_euclid(std::f64::consts::TAU);
    Some((az, el, 0.5 * (1.0 + a.phase.to_radians().cos())))
}

/// A point or disc source in the sky: a catalogue star or a solar-system body.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    /// HIP number, 1<<31 | Tycho-2, or 1<<30 | NAIF id (planets, Moon)
    pub id: u32,
    /// apparent direction (ITRS / ECEF, refracted)
    pub dir: DVec3,
    pub v: f32,
    pub bv: f32,
    /// apparent radius (rad; 0 for stars)
    pub radius: f64,
    /// drawn by the star renderer (the Moon is drawn by the sky)
    pub draw: bool,
}

/// Rotate `d` towards the zenith `up` by the refraction at its elevation.
fn refract(d: DVec3, up: DVec3, p_hpa: f64, t_k: f64) -> DVec3 {
    if p_hpa <= 0.0 {
        return d;
    }
    let el = d.dot(up).clamp(-1.0, 1.0).asin();
    let r = astro::refraction(el, p_hpa, t_k);
    let tang = up - d * d.dot(up);
    if tang.length_squared() > 1e-24 {
        (d * r.cos() + tang.normalize() * r.sin()).normalize()
    } else {
        d
    }
}

/// Effective temperature (K) from B−V (Ballesteros 2012).
pub fn bv_temperature(bv: f64) -> f64 {
    4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
}

/// Linear sRGB (D65) of a blackbody at `t` K (CIE 1931 2° observer, multi-lobe fit of Wyman,
/// Sloan & Shirley 2013), arbitrary scale.
pub fn blackbody_rgb(t: f64) -> DVec3 {
    let g = |x: f64, mu: f64, s1: f64, s2: f64| {
        let s = if x < mu { s1 } else { s2 };
        (-0.5 * ((x - mu) / s).powi(2)).exp()
    };
    let mut xyz = DVec3::ZERO;
    let mut l = 380.0;
    while l <= 780.0 {
        let xb = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2);
        let yb = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1);
        let zb = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8);
        let lm = l * 1e-9;
        let b = 1.0 / (lm.powi(5) * ((1.438_776_9e-2 / (lm * t)).exp() - 1.0));
        xyz += DVec3::new(xb, yb, zb) * b;
        l += 5.0;
    }
    DVec3::new(
        3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
        0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z,
    )
    .max(DVec3::splat(1e-30 * xyz.y))
}

/// Error function (Abramowitz & Stegun 7.1.26, |ε| < 1.5e-7).
fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use geodesy::frames::{ecef2enuv, geodetic2ecef, Geodetic};

    /// Apparent topocentric positions (no refraction) against Skyfield 1.53 with the JPL DE421
    /// ephemeris, IAU 2000A nutation and its own UT1 (passed as dut1), for the same catalogue
    /// records: bright, high proper motion (α Cen A, 61 Cyg A) and high parallax stars.
    #[test]
    fn positions_match_skyfield() {
        type Case = (f64, f64, f64, f64, f64, &'static [(u32, f64, f64)]);
        let cases: [Case; 2] = [
            // 2026-03-19 21:30 UTC, lat 41.5705 lon 32.97 h 1500.0 m (Skyfield, DE421)
            (1773955800.0, 41.5705, 32.97, 1500.0, 0.051546, &[
                (32349, 245.206470669, 2.178336034),
                (91262, 50.488495659, 14.463243025),
                (11767, 359.338462323, 41.196513944),
                (27989, 271.451858665, 9.567755203),
                (30438, 220.635154324, -25.102898281),
                (71683, 157.689034797, -19.529870960),
                (104214, 27.204426347, -3.174344156),
            ]),
            // 2031-07-01 12:00 UTC, lat -64.1 lon -21.9 h 10000.0 m (Skyfield, DE421)
            (1940673600.0, -64.1, -21.9, 10000.0, 0.080771, &[
                (32349, 30.728381601, 39.865951680),
                (91262, 217.907885707, -61.557347676),
                (11767, 359.335117981, -63.562452007),
                (27989, 12.271747986, 17.942157577),
                (30438, 49.041462602, 75.094274174),
                (71683, 158.212201720, 38.113771770),
                (104214, 263.246281809, -47.458546730),
            ]),
        ];
        let ell = geodesy::Ellipsoid::WGS84;
        let f = StarField::new(&StarsConfig { refraction: false, ..Default::default() }).unwrap();
        let mut worst: f64 = 0.0;
        for (unix, lat, lon, h, dut1, stars) in cases {
            let (la, lo) = (lat.to_radians(), lon.to_radians());
            let pos = geodetic2ecef(Geodetic::new(la, lo, h), &ell);
            let f = StarField { cfg: StarsConfig { dut1_s: dut1, ..f.cfg.clone() }, cat: f.cat.clone(), n: f.n, colours: vec![] };
            let dirs = f.apparent(unix, pos, &ell);
            for &(id, az, el) in stars {
                let k = f.cat.stars.iter().position(|s| s.id == id).unwrap();
                let (sa, ca) = az.to_radians().sin_cos();
                let (se, ce) = el.to_radians().sin_cos();
                let want = DVec3::new(ce * sa, ce * ca, se);
                let got = ecef2enuv(dirs[k], la, lo);
                let sep = got.angle_between(want).to_degrees() * 3600.0;
                worst = worst.max(sep);
                assert!(sep < 0.005, "{}: {sep:.4}\"", catalog::designation(id));
            }
        }
        eprintln!("worst separation from Skyfield: {worst:.4}\"");
    }

    /// Refraction: Green's model with ERFA refco's constants (pyerfa: A = 57.1545″,
    /// B = −0.0654″ at 1013.25 hPa, 15 °C, dry, 0.574 µm), true → apparent.
    #[test]
    fn refraction_matches_erfa() {
        for (el, want) in [(20.0, 155.317), (30.0, 98.547), (45.0, 57.058), (60.0, 32.973), (80.0, 10.075)] {
            let r = astro::refraction(f64::to_radians(el), 1013.25, 288.15).to_degrees() * 3600.0;
            assert!((r - want).abs() < 0.02, "el {el}: {r:.3}\" vs {want}\"");
        }
        // horizon (Sæmundsson): ≈ 29′ at 0° for these conditions; monotonic down to the horizon
        let r0 = astro::refraction(0.0, 1013.25, 288.15).to_degrees() * 60.0;
        assert!((r0 - 28.6).abs() < 1.0, "{r0}");
        let mut prev = f64::MAX;
        for k in 0..=90 {
            let r = astro::refraction(f64::to_radians(k as f64), 1013.25, 288.15);
            assert!(r <= prev + 1e-12, "not monotonic at {k}°");
            prev = r;
        }
    }

    /// Planets and the Moon (system barycentres, no refraction) against Skyfield 1.53 with JPL
    /// DE440s; magnitudes against skyfield.magnitudelib (Mallama & Hilton 2018).
    #[test]
    fn planets_match_skyfield() {
        // 2026-03-19 21:30 UTC, lat 41.5705 lon 32.97 h 1500 m: (NAIF, az, el, V)
        let want = [
            (199, 22.526078691, -53.730670832, 1.1842),
            (299, 330.545216800, -38.305505284, -3.9177),
            (499, 13.256217174, -55.207123713, 1.1662),
            (599, 273.149770103, 32.332391173, -2.2934),
            (699, 342.643150145, -47.454123280, 0.9292),
            (799, 302.390759908, -5.481492348, 5.7799),
            (899, 346.132847664, -48.099017103, 7.8251),
            (301, 339.866486399, -40.577419047, f64::NAN),
        ];
        let (la, lo) = (41.5705f64.to_radians(), 32.97f64.to_radians());
        let ell = geodesy::Ellipsoid::WGS84;
        let pos = geodetic2ecef(Geodetic::new(la, lo, 1500.0), &ell);
        let f = StarField::new(&StarsConfig { refraction: false, dut1_s: 0.051546, ..Default::default() }).unwrap();
        let bodies = f.bodies(1773955800.0, pos, &ell);
        assert_eq!(bodies.len(), want.len());
        for (naif, az, el, v) in want {
            let b = bodies.iter().find(|b| b.id == (1 << 30) | naif).unwrap();
            let (sa, ca) = f64::to_radians(az).sin_cos();
            let (se, ce) = f64::to_radians(el).sin_cos();
            let sep = ecef2enuv(b.dir, la, lo).angle_between(DVec3::new(ce * sa, ce * ca, se)).to_degrees() * 3600.0;
            assert!(sep < 0.02, "{naif}: {sep:.4}\"");
            if v.is_finite() {
                assert!((b.v as f64 - v).abs() < 0.05, "{naif}: V {} vs {v}", b.v);
            }
            assert_eq!(b.draw, naif != 301);
        }
        // the lighting Moon uses the same ephemeris (no refraction, observer on the ellipsoid)
        let (az, el, phase) = crate::lighting::moon_position(1773955800.0, la, lo);
        assert!((az.to_degrees() - 339.866).abs() < 0.01 && (el.to_degrees() + 40.577).abs() < 0.03, "{az} {el}");
        assert!(phase < 0.03, "new moon on 2026-03-19: {phase}");
    }

    #[test]
    fn colours_and_flux() {
        // the Sun (B−V 0.65) is white; hot stars blue, cool stars red; luminance 1
        let f = StarField::new(&StarsConfig::default()).unwrap();
        let c = |bv: f64| f.colours[((bv + 0.5) / 0.01).round() as usize];
        let sun = c(0.65);
        assert!((sun - DVec3::ONE).abs().max_element() < 0.02, "{sun}");
        assert!(c(-0.3).z > c(-0.3).x && c(1.8).x > c(1.8).z);
        for bv in [-0.4, 0.0, 0.65, 1.5, 2.5] {
            let v = c(bv);
            assert!((0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z - 1.0).abs() < 1e-9);
        }
        // catalogue: sorted by V, Sirius first, ~130k stars to V = 9
        assert_eq!(f.cat.stars[0].id, 32349);
        assert!(f.cat.stars.windows(2).all(|w| w[0].v <= w[1].v));
        assert!(f.cat.stars.len() > 120_000);
    }
}
