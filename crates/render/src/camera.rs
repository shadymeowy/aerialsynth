//! Camera models and the camera rig (intrinsics + body→camera extrinsics).
//!
//! Conventions:
//! * Camera frame = OpenCV: x right, y down, z forward (optical axis).
//! * Pixel coordinates = OpenCV: the centre of the top-left pixel is (0, 0); the image spans
//!   [-0.5, w-0.5] x [-0.5, h-0.5].
//! * Body frame = FRD (x forward, y right, z down).
//!
//! The models and their YAML description follow camodocal's `calib/camera.{h,cpp}` exactly
//! (formulas after camodocal): `pinhole`, `pinhole_full`, `kannala_brandt`, `mei`,
//! `scaramuzza`, with the same keys (`model, width, height, intrinsics, distortion, xi,
//! max_fov_deg, inv_poly, affine, center`). The forward maps are copied verbatim; the inverse
//! maps (needed for rendering) are solved analytically / by Newton iterations.

use anyhow::{bail, Result};
use glam::{DMat3, DQuat, DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::sync::Arc;

/// A generic central camera model.
pub trait CameraModel: Send + Sync + std::fmt::Debug {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    /// Project a point in the camera frame to pixel coordinates. `None` if not imageable
    /// (outside the model's valid field of view).
    fn project(&self, p: DVec3) -> Option<DVec2>;
    /// Back-project a pixel to a unit ray in the camera frame (`None` outside the image circle).
    fn unproject(&self, px: DVec2) -> Option<DVec3>;
    /// Half-angle (rad) of a cone around +z that contains every pixel's ray (may exceed π/2).
    fn max_half_angle(&self) -> f64;
    /// Effective focal length in pixels at the image centre (LOD / footprint estimates).
    fn focal_px(&self) -> f64;
    /// The same camera rendered at `s` times the resolution (sub-sample centres line up so that
    /// for odd `s` the central sub-sample coincides with the original pixel centre).
    fn scaled(&self, s: u32) -> Arc<dyn CameraModel>;
    /// Serializable description (camodocal YAML schema).
    fn config(&self) -> CameraConfig;
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// Camera intrinsics in camodocal's YAML schema:
/// ```yaml
/// model: pinhole            # pinhole | pinhole_full | kannala_brandt | mei | scaramuzza
/// width: 752
/// height: 480
/// intrinsics: [fx, fy, cx, cy]     # model's own naming (mu mv u0 v0 / gamma1 gamma2 u0 v0)
/// distortion: [k1, k2, p1, p2]     # pinhole_full: [k1 k2 p1 p2 k3 k4 k5 k6]; kannala_brandt: [k2 k3 k4 k5]
/// xi: 0.0                          # mei
/// max_fov_deg: 0.0                 # kannala_brandt (0 = unlimited)
/// inv_poly: [...]                  # scaramuzza (world2cam polynomial in theta)
/// affine: [C, D, E]                # scaramuzza
/// center: [cx, cy]                 # scaramuzza
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CameraConfig {
    pub model: String,
    pub width: u32,
    pub height: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub intrinsics: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub distortion: Vec<f64>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub xi: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub max_fov_deg: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inv_poly: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub affine: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub center: Vec<f64>,
}

impl CameraConfig {
    fn base(model: &str, width: u32, height: u32) -> Self {
        CameraConfig {
            model: model.into(),
            width,
            height,
            intrinsics: vec![],
            distortion: vec![],
            xi: 0.0,
            max_fov_deg: 0.0,
            inv_poly: vec![],
            affine: vec![],
            center: vec![],
        }
    }

    /// Distortion-free pinhole with a given horizontal field of view (deg), centred.
    pub fn pinhole_hfov(width: u32, height: u32, hfov_deg: f64) -> Self {
        let fx = 0.5 * width as f64 / (0.5 * hfov_deg.to_radians()).tan();
        let mut c = Self::base("pinhole", width, height);
        c.intrinsics = vec![fx, fx, (width as f64 - 1.0) * 0.5, (height as f64 - 1.0) * 0.5];
        c.distortion = vec![0.0; 4];
        c
    }

    pub fn build(&self) -> Result<Arc<dyn CameraModel>> {
        let need = |v: &Vec<f64>, n: usize, what: &str| -> Result<()> {
            if v.len() != n {
                bail!("camera: {} {} needs {} values, got {}", self.model, what, n, v.len());
            }
            Ok(())
        };
        if self.width == 0 || self.height == 0 {
            bail!("camera: width/height must be > 0");
        }
        let (w, h) = (self.width, self.height);
        let i = &self.intrinsics;
        let d = &self.distortion;
        Ok(match self.model.as_str() {
            "pinhole" => {
                need(i, 4, "intrinsics")?;
                need(d, 4, "distortion")?;
                Arc::new(Cam::new(Pinhole { fx: i[0], fy: i[1], cx: i[2], cy: i[3], k1: d[0], k2: d[1], p1: d[2], p2: d[3] }, w, h))
            }
            "pinhole_full" => {
                need(i, 4, "intrinsics")?;
                need(d, 8, "distortion")?;
                Arc::new(Cam::new(
                    PinholeFull { fx: i[0], fy: i[1], cx: i[2], cy: i[3], k1: d[0], k2: d[1], p1: d[2], p2: d[3], k3: d[4], k4: d[5], k5: d[6], k6: d[7] },
                    w,
                    h,
                ))
            }
            "kannala_brandt" => {
                need(i, 4, "intrinsics")?;
                need(d, 4, "distortion")?;
                let max_theta = if self.max_fov_deg > 0.0 { self.max_fov_deg * PI / 360.0 } else { PI };
                Arc::new(Cam::new(
                    KannalaBrandt { mu: i[0], mv: i[1], u0: i[2], v0: i[3], k2: d[0], k3: d[1], k4: d[2], k5: d[3], max_theta, max_fov_deg: self.max_fov_deg },
                    w,
                    h,
                ))
            }
            "mei" => {
                need(i, 4, "intrinsics")?;
                need(d, 4, "distortion")?;
                Arc::new(Cam::new(Mei { gamma1: i[0], gamma2: i[1], u0: i[2], v0: i[3], xi: self.xi, k1: d[0], k2: d[1], p1: d[2], p2: d[3] }, w, h))
            }
            "scaramuzza" => {
                need(&self.affine, 3, "affine")?;
                need(&self.center, 2, "center")?;
                if self.inv_poly.is_empty() {
                    bail!("camera: scaramuzza needs inv_poly");
                }
                Arc::new(Cam::new(
                    Scaramuzza { inv_poly: self.inv_poly.clone(), c: self.affine[0], d: self.affine[1], e: self.affine[2], center_x: self.center[0], center_y: self.center[1] },
                    w,
                    h,
                ))
            }
            m => bail!("camera: unknown model '{m}' (pinhole | pinhole_full | kannala_brandt | mei | scaramuzza)"),
        })
    }
}

/// Model-specific maps used by the generic `Cam` wrapper.
trait Proj: Clone + Send + Sync + std::fmt::Debug + 'static {
    fn proj(&self, p: DVec3) -> Option<DVec2>;
    fn unproj(&self, px: DVec2) -> Option<DVec3>;
    /// Pixel → s·pixel + (s-1)/2.
    fn scale(&self, s: f64) -> Self;
    fn cfg(&self, w: u32, h: u32) -> CameraConfig;
    /// Principal point (where the optical axis lands), for domain / focal estimates.
    fn principal(&self) -> DVec2;
}

/// Gauss–Newton inversion of a 2D map with a numeric Jacobian.
fn invert2(f: impl Fn(DVec2) -> DVec2, target: DVec2, init: DVec2) -> Option<DVec2> {
    let mut x = init;
    for _ in 0..40 {
        let r = f(x) - target;
        if r.length_squared() < 1e-26 {
            return Some(x);
        }
        let e = 1e-7 * (1.0 + x.length());
        let jx = (f(x + DVec2::new(e, 0.0)) - f(x - DVec2::new(e, 0.0))) / (2.0 * e);
        let jy = (f(x + DVec2::new(0.0, e)) - f(x - DVec2::new(0.0, e))) / (2.0 * e);
        let det = jx.x * jy.y - jy.x * jx.y;
        if det.abs() < 1e-18 {
            return None;
        }
        let dx = DVec2::new(jy.y * r.x - jy.x * r.y, -jx.y * r.x + jx.x * r.y) / det;
        x -= dx;
        if !x.is_finite() {
            return None;
        }
    }
    let ok = (f(x) - target).length() < 1e-9;
    ok.then_some(x)
}

/// Radial-tangential offset shared by `pinhole` and `mei` (camodocal's distortion()).
#[inline]
fn radtan4(p: DVec2, k1: f64, k2: f64, p1: f64, p2: f64) -> DVec2 {
    let (x, y) = (p.x, p.y);
    let (x2, y2, xy) = (x * x, y * y, x * y);
    let r2 = x2 + y2;
    let rad = k1 * r2 + k2 * r2 * r2;
    DVec2::new(x * rad + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2), y * rad + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2))
}

#[derive(Clone, Debug)]
struct Pinhole {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
}

impl Pinhole {
    fn distort(&self, u: DVec2) -> DVec2 {
        u + radtan4(u, self.k1, self.k2, self.p1, self.p2)
    }
}

impl Proj for Pinhole {
    fn proj(&self, p: DVec3) -> Option<DVec2> {
        if p.z <= 1e-9 {
            return None;
        }
        let u = self.distort(DVec2::new(p.x / p.z, p.y / p.z));
        Some(DVec2::new(self.fx * u.x + self.cx, self.fy * u.y + self.cy))
    }
    fn unproj(&self, px: DVec2) -> Option<DVec3> {
        let xd = DVec2::new((px.x - self.cx) / self.fx, (px.y - self.cy) / self.fy);
        let u = invert2(|u| self.distort(u), xd, xd)?;
        Some(DVec3::new(u.x, u.y, 1.0).normalize())
    }
    fn scale(&self, s: f64) -> Self {
        let o = (s - 1.0) * 0.5;
        Pinhole { fx: self.fx * s, fy: self.fy * s, cx: self.cx * s + o, cy: self.cy * s + o, ..*self }
    }
    fn cfg(&self, w: u32, h: u32) -> CameraConfig {
        let mut c = CameraConfig::base("pinhole", w, h);
        c.intrinsics = vec![self.fx, self.fy, self.cx, self.cy];
        c.distortion = vec![self.k1, self.k2, self.p1, self.p2];
        c
    }
    fn principal(&self) -> DVec2 {
        DVec2::new(self.cx, self.cy)
    }
}

/// OpenCV's rational model: k1..k6 radial, p1/p2 tangential.
#[derive(Clone, Debug)]
struct PinholeFull {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    k6: f64,
}

impl PinholeFull {
    /// normalized → distorted normalized
    fn distort(&self, u: DVec2) -> DVec2 {
        let (x, y) = (u.x, u.y);
        let r2 = x * x + y * y;
        let (r4, r6) = (r2 * r2, r2 * r2 * r2);
        let (a1, a2, a3) = (2.0 * x * y, r2 + 2.0 * x * x, r2 + 2.0 * y * y);
        let cdist = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let icdist2 = 1.0 / (1.0 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6);
        DVec2::new(x * cdist * icdist2 + self.p1 * a1 + self.p2 * a2, y * cdist * icdist2 + self.p1 * a3 + self.p2 * a1)
    }
}

impl Proj for PinholeFull {
    fn proj(&self, p: DVec3) -> Option<DVec2> {
        if p.z <= 1e-9 {
            return None;
        }
        let u = self.distort(DVec2::new(p.x / p.z, p.y / p.z));
        Some(DVec2::new(self.fx * u.x + self.cx, self.fy * u.y + self.cy))
    }
    fn unproj(&self, px: DVec2) -> Option<DVec3> {
        let xd = DVec2::new((px.x - self.cx) / self.fx, (px.y - self.cy) / self.fy);
        let u = invert2(|u| self.distort(u), xd, xd)?;
        Some(DVec3::new(u.x, u.y, 1.0).normalize())
    }
    fn scale(&self, s: f64) -> Self {
        let o = (s - 1.0) * 0.5;
        PinholeFull { fx: self.fx * s, fy: self.fy * s, cx: self.cx * s + o, cy: self.cy * s + o, ..*self }
    }
    fn cfg(&self, w: u32, h: u32) -> CameraConfig {
        let mut c = CameraConfig::base("pinhole_full", w, h);
        c.intrinsics = vec![self.fx, self.fy, self.cx, self.cy];
        c.distortion = vec![self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6];
        c
    }
    fn principal(&self) -> DVec2 {
        DVec2::new(self.cx, self.cy)
    }
}

/// Equidistant fisheye; the leading coefficient is fixed at 1 (camodocal k2..k5 = OpenCV k1..k4).
#[derive(Clone, Debug)]
struct KannalaBrandt {
    mu: f64,
    mv: f64,
    u0: f64,
    v0: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    max_theta: f64,
    max_fov_deg: f64,
}

impl KannalaBrandt {
    fn r(&self, th: f64) -> f64 {
        let t2 = th * th;
        th * (1.0 + t2 * (self.k2 + t2 * (self.k3 + t2 * (self.k4 + t2 * self.k5))))
    }
}

impl Proj for KannalaBrandt {
    fn proj(&self, p: DVec3) -> Option<DVec2> {
        let n = p.length();
        if n < 1e-12 {
            return None;
        }
        let theta = (p.z / n).clamp(-1.0, 1.0).acos();
        if theta > self.max_theta {
            return None;
        }
        let phi = p.y.atan2(p.x);
        let r = self.r(theta);
        Some(DVec2::new(self.mu * r * phi.cos() + self.u0, self.mv * r * phi.sin() + self.v0))
    }
    fn unproj(&self, px: DVec2) -> Option<DVec3> {
        let mx = (px.x - self.u0) / self.mu;
        let my = (px.y - self.v0) / self.mv;
        let rd = (mx * mx + my * my).sqrt();
        if rd < 1e-12 {
            return Some(DVec3::Z);
        }
        // Newton on θ (r(θ) increasing on the valid domain)
        let mut th = rd.min(self.max_theta);
        for _ in 0..50 {
            let f = self.r(th) - rd;
            let e = 1e-7;
            let df = (self.r(th + e) - self.r(th - e)) / (2.0 * e);
            if df <= 1e-12 {
                return None;
            }
            let step = f / df;
            th -= step;
            if step.abs() < 1e-14 {
                break;
            }
        }
        if !(0.0..=self.max_theta).contains(&th) || (self.r(th) - rd).abs() > 1e-9 {
            return None;
        }
        let s = th.sin();
        Some(DVec3::new(mx / rd * s, my / rd * s, th.cos()))
    }
    fn scale(&self, s: f64) -> Self {
        let o = (s - 1.0) * 0.5;
        KannalaBrandt { mu: self.mu * s, mv: self.mv * s, u0: self.u0 * s + o, v0: self.v0 * s + o, ..*self }
    }
    fn cfg(&self, w: u32, h: u32) -> CameraConfig {
        let mut c = CameraConfig::base("kannala_brandt", w, h);
        c.intrinsics = vec![self.mu, self.mv, self.u0, self.v0];
        c.distortion = vec![self.k2, self.k3, self.k4, self.k5];
        c.max_fov_deg = self.max_fov_deg;
        c
    }
    fn principal(&self) -> DVec2 {
        DVec2::new(self.u0, self.v0)
    }
}

/// Unified (Mei) omnidirectional model: lift onto the sphere, offset by xi, then radtan.
#[derive(Clone, Debug)]
struct Mei {
    gamma1: f64,
    gamma2: f64,
    u0: f64,
    v0: f64,
    xi: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
}

impl Proj for Mei {
    fn proj(&self, p: DVec3) -> Option<DVec2> {
        let z = p.z + self.xi * p.length();
        if z <= 1e-9 {
            return None;
        }
        let mut u = DVec2::new(p.x / z, p.y / z);
        u += radtan4(u, self.k1, self.k2, self.p1, self.p2);
        Some(DVec2::new(self.gamma1 * u.x + self.u0, self.gamma2 * u.y + self.v0))
    }
    fn unproj(&self, px: DVec2) -> Option<DVec3> {
        let md = DVec2::new((px.x - self.u0) / self.gamma1, (px.y - self.v0) / self.gamma2);
        let m = invert2(|u| u + radtan4(u, self.k1, self.k2, self.p1, self.p2), md, md)?;
        // lift the undistorted point onto the unit sphere
        let r2 = m.length_squared();
        let disc = 1.0 + (1.0 - self.xi * self.xi) * r2;
        if disc < 0.0 {
            return None;
        }
        let lam = (self.xi + disc.sqrt()) / (1.0 + r2);
        let p = DVec3::new(lam * m.x, lam * m.y, lam - self.xi);
        (p.z + self.xi > 1e-9).then(|| p.normalize())
    }
    fn scale(&self, s: f64) -> Self {
        let o = (s - 1.0) * 0.5;
        Mei { gamma1: self.gamma1 * s, gamma2: self.gamma2 * s, u0: self.u0 * s + o, v0: self.v0 * s + o, ..*self }
    }
    fn cfg(&self, w: u32, h: u32) -> CameraConfig {
        let mut c = CameraConfig::base("mei", w, h);
        c.intrinsics = vec![self.gamma1, self.gamma2, self.u0, self.v0];
        c.distortion = vec![self.k1, self.k2, self.p1, self.p2];
        c.xi = self.xi;
        c
    }
    fn principal(&self) -> DVec2 {
        DVec2::new(self.u0, self.v0)
    }
}

/// Scaramuzza / OCam. theta is measured from the image plane toward -Z, so the model covers
/// past 180°; the affine term is [[C, D], [E, 1]].
#[derive(Clone, Debug)]
struct Scaramuzza {
    inv_poly: Vec<f64>,
    c: f64,
    d: f64,
    e: f64,
    center_x: f64,
    center_y: f64,
}

impl Scaramuzza {
    fn rho(&self, theta: f64) -> f64 {
        let mut rho = 0.0;
        let mut t = 1.0;
        for c in &self.inv_poly {
            rho += t * c;
            t *= theta;
        }
        rho
    }
}

impl Proj for Scaramuzza {
    fn proj(&self, p: DVec3) -> Option<DVec2> {
        let n = p.x.hypot(p.y);
        if n < 1e-12 {
            return None;
        }
        let theta = (-p.z).atan2(n);
        let rho = self.rho(theta);
        let xn = DVec2::new(p.x / n * rho, p.y / n * rho);
        Some(DVec2::new(xn.x * self.c + xn.y * self.d + self.center_x, xn.x * self.e + xn.y + self.center_y))
    }
    fn unproj(&self, px: DVec2) -> Option<DVec3> {
        // invert the affine [[C, D], [E, 1]]
        let (u, v) = (px.x - self.center_x, px.y - self.center_y);
        let det = self.c - self.d * self.e;
        if det.abs() < 1e-15 {
            return None;
        }
        let xn = DVec2::new((u - self.d * v) / det, (-self.e * u + self.c * v) / det);
        let rho = xn.length();
        if rho < 1e-12 {
            return Some(DVec3::Z);
        }
        // rho(theta) increases from the optical axis (theta = -pi/2) outwards: bisection
        let (mut lo, mut hi) = (-PI / 2.0 + 1e-9, PI / 2.0 - 1e-9);
        let (flo, fhi) = (self.rho(lo) - rho, self.rho(hi) - rho);
        if flo * fhi > 0.0 {
            return None;
        }
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            let fm = self.rho(mid) - rho;
            if (fm > 0.0) == (fhi > 0.0) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let th = 0.5 * (lo + hi);
        let (s, c) = th.sin_cos();
        Some(DVec3::new(c * xn.x / rho, c * xn.y / rho, -s).normalize())
    }
    fn scale(&self, s: f64) -> Self {
        let o = (s - 1.0) * 0.5;
        Scaramuzza { inv_poly: self.inv_poly.iter().map(|c| c * s).collect(), center_x: self.center_x * s + o, center_y: self.center_y * s + o, ..self.clone() }
    }
    fn cfg(&self, w: u32, h: u32) -> CameraConfig {
        let mut c = CameraConfig::base("scaramuzza", w, h);
        c.inv_poly = self.inv_poly.clone();
        c.affine = vec![self.c, self.d, self.e];
        c.center = vec![self.center_x, self.center_y];
        c
    }
    fn principal(&self) -> DVec2 {
        DVec2::new(self.center_x, self.center_y)
    }
}

/// Generic wrapper: derives the model's valid angular domain (where the projection stays
/// monotonic), the image's half-angle and the effective focal length.
#[derive(Clone, Debug)]
struct Cam<M: Proj> {
    m: M,
    w: u32,
    h: u32,
    half_angle: f64,
    /// rays beyond this angle from +z are rejected (prevents wrap-around of distortion terms)
    angle_limit: f64,
    focal: f64,
}

impl<M: Proj> Cam<M> {
    fn new(m: M, w: u32, h: u32) -> Self {
        // valid domain: radial image distance must keep increasing with the angle from the axis
        let pp = m.principal();
        let mut domain = PI - 0.02;
        for az in [0.0, 0.5 * PI, PI, 1.5 * PI, 0.25 * PI, 0.75 * PI] {
            let mut prev = 0.0;
            let mut a: f64 = 0.0;
            while a < PI - 0.02 {
                a += 0.002;
                let dir = DVec3::new(a.sin() * f64::cos(az), a.sin() * f64::sin(az), a.cos());
                match m.proj(dir) {
                    Some(px) => {
                        let r = (px - pp).length();
                        if r <= prev || !r.is_finite() {
                            domain = domain.min(a - 0.002);
                            break;
                        }
                        prev = r;
                    }
                    None => {
                        domain = domain.min(a - 0.002);
                        break;
                    }
                }
            }
        }
        // half angle covering the image border (sampled)
        let mut ha: f64 = 0.0;
        let (wf, hf) = (w as f64, h as f64);
        for k in 0..=32 {
            let t = k as f64 / 32.0;
            for (u, v) in [(t * wf - 0.5, -0.5), (t * wf - 0.5, hf - 0.5), (-0.5, t * hf - 0.5), (wf - 0.5, t * hf - 0.5)] {
                match m.unproj(DVec2::new(u, v)) {
                    Some(r) if r.z.acos() <= domain => ha = ha.max(r.z.clamp(-1.0, 1.0).acos()),
                    _ => ha = domain,
                }
            }
        }
        let half_angle = ha.min(domain);
        let angle_limit = (half_angle * 1.3 + 0.15).min(domain);
        // focal: angular size of one pixel at the principal point
        let focal = match (m.unproj(pp), m.unproj(pp + DVec2::new(1.0, 0.0)), m.unproj(pp + DVec2::new(0.0, 1.0))) {
            (Some(a), Some(b), Some(c)) => {
                let ang = 0.5 * (a.angle_between(b) + a.angle_between(c));
                if ang > 0.0 { 1.0 / ang } else { 500.0 }
            }
            _ => 500.0,
        };
        Cam { m, w, h, half_angle, angle_limit, focal }
    }
}

impl<M: Proj> CameraModel for Cam<M> {
    fn width(&self) -> u32 {
        self.w
    }
    fn height(&self) -> u32 {
        self.h
    }
    #[inline]
    fn project(&self, p: DVec3) -> Option<DVec2> {
        let n = p.length();
        if n < 1e-12 || (p.z / n) < self.angle_limit.cos() {
            return None;
        }
        self.m.proj(p)
    }
    fn unproject(&self, px: DVec2) -> Option<DVec3> {
        let r = self.m.unproj(px)?;
        (r.z.clamp(-1.0, 1.0).acos() <= self.angle_limit).then_some(r)
    }
    fn max_half_angle(&self) -> f64 {
        self.half_angle
    }
    fn focal_px(&self) -> f64 {
        self.focal
    }
    fn scaled(&self, s: u32) -> Arc<dyn CameraModel> {
        let sf = s as f64;
        Arc::new(Cam {
            m: self.m.scale(sf),
            w: self.w * s,
            h: self.h * s,
            half_angle: self.half_angle,
            angle_limit: self.angle_limit,
            focal: self.focal * sf,
        })
    }
    fn config(&self) -> CameraConfig {
        self.m.cfg(self.w, self.h)
    }
}

/// How the camera is mounted on the body.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Extrinsics {
    /// Base mounting: `nadir` (optical axis = body down, image top = forward) or
    /// `forward` (optical axis = body forward, image top = up).
    pub mount: Mount,
    /// Additional rotation of the camera about the body axes applied after the mount
    /// (degrees). Pitch > 0 tilts the optical axis towards body-forward for a nadir mount and
    /// upwards for a forward mount.
    pub roll_deg: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
    /// Camera centre in the body frame (m).
    pub translation: [f64; 3],
    /// Explicit rotation body→camera as quaternion [w, x, y, z] (camera axes expressed in body);
    /// overrides mount/roll/pitch/yaw when given.
    pub q_body_cam: Option<[f64; 4]>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Mount {
    #[default]
    Nadir,
    Forward,
}

impl Default for Extrinsics {
    fn default() -> Self {
        Extrinsics { mount: Mount::Nadir, roll_deg: 0.0, pitch_deg: 0.0, yaw_deg: 0.0, translation: [0.0; 3], q_body_cam: None }
    }
}

impl Extrinsics {
    /// Rotation mapping camera-frame vectors into the body frame (columns = camera axes in body).
    pub fn r_body_cam(&self) -> DMat3 {
        if let Some([w, x, y, z]) = self.q_body_cam {
            return DMat3::from_quat(DQuat::from_xyzw(x, y, z, w).normalize());
        }
        let base = match self.mount {
            // x_cam = body right, y_cam = body backward, z_cam = body down
            Mount::Nadir => DMat3::from_cols(DVec3::Y, -DVec3::X, DVec3::Z),
            // x_cam = body right, y_cam = body down, z_cam = body forward
            Mount::Forward => DMat3::from_cols(DVec3::Y, DVec3::Z, DVec3::X),
        };
        // extra rotation about body axes: yaw (z), pitch (y), roll (x)
        let (r, p, y) = (self.roll_deg.to_radians(), self.pitch_deg.to_radians(), self.yaw_deg.to_radians());
        // Ry(+p) maps body-down towards body-forward and body-forward towards body-up, so the
        // same sign convention works for both mounts.
        let extra = DMat3::from_rotation_z(y) * DMat3::from_rotation_y(p) * DMat3::from_rotation_x(r);
        extra * base
    }
    pub fn t_body_cam(&self) -> DVec3 {
        DVec3::from_array(self.translation)
    }
}

/// Camera rig description (YAML).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RigConfig {
    pub camera: CameraConfig,
    #[serde(default)]
    pub extrinsics: Extrinsics,
}

impl Default for RigConfig {
    fn default() -> Self {
        RigConfig { camera: CameraConfig::pinhole_hfov(640, 512, 70.0), extrinsics: Extrinsics::default() }
    }
}

impl RigConfig {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        Ok(serde_yaml::from_str(&std::fs::read_to_string(path)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(c: &CameraConfig, pts: &[(f64, f64)]) {
        let m = c.build().unwrap();
        for &(u, v) in pts {
            let r = m.unproject(DVec2::new(u, v)).unwrap_or_else(|| panic!("{} unproject {u},{v}", c.model));
            assert!((r.length() - 1.0).abs() < 1e-9);
            let px = m.project(r * 7.0).unwrap_or_else(|| panic!("{} project {u},{v}", c.model));
            assert!((px - DVec2::new(u, v)).length() < 1e-6, "{}: {u},{v} -> {px:?}", c.model);
        }
    }

    fn cfg(yaml: &str) -> CameraConfig {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn all_models_roundtrip() {
        let pts = [(0.0, 0.0), (751.0, 479.0), (376.0, 240.0), (100.0, 400.0), (700.0, 30.0)];
        // EuRoC cam0 (camodocal config/euroc.yaml)
        roundtrip(&cfg("{model: pinhole, width: 752, height: 480, intrinsics: [458.654, 457.296, 367.215, 248.375], distortion: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]}"), &pts);
        roundtrip(&cfg("{model: pinhole_full, width: 752, height: 480, intrinsics: [460, 458, 370, 250], distortion: [0.2, -0.05, 0.0005, -0.0003, 0.01, 0.48, 0.02, 0.005]}"), &pts);
        roundtrip(&cfg("{model: kannala_brandt, width: 752, height: 480, intrinsics: [190, 190, 376, 240], distortion: [0.0034, 0.0007, -0.0024, 0.0003]}"), &pts);
        roundtrip(&cfg("{model: mei, width: 752, height: 480, intrinsics: [700, 700, 376, 240], distortion: [-0.1, 0.05, 0.0002, -0.0001], xi: 1.4}"), &pts);
        roundtrip(&cfg("{model: scaramuzza, width: 752, height: 480, inv_poly: [283.65, 171.15, -6.0], affine: [1.0, 0.0, 0.0], center: [376, 240]}"), &[(376.5, 240.5), (300.0, 200.0), (600.0, 400.0), (10.0, 20.0)]);
    }

    #[test]
    fn wide_models_exceed_90_degrees() {
        let kb = cfg("{model: kannala_brandt, width: 800, height: 800, intrinsics: [180, 180, 399.5, 399.5], distortion: [0, 0, 0, 0]}").build().unwrap();
        assert!(kb.max_half_angle() > 1.9, "{}", kb.max_half_angle());
        let r = kb.unproject(DVec2::new(0.0, 399.5)).unwrap();
        assert!(r.z < 0.0);
        let mei = cfg("{model: mei, width: 800, height: 800, intrinsics: [400, 400, 399.5, 399.5], distortion: [0, 0, 0, 0], xi: 1.2}").build().unwrap();
        assert!(mei.max_half_angle() > PI / 2.0, "{}", mei.max_half_angle());
        let kbf = cfg("{model: kannala_brandt, width: 800, height: 800, intrinsics: [180, 180, 399.5, 399.5], distortion: [0, 0, 0, 0], max_fov_deg: 160}").build().unwrap();
        assert!(kbf.unproject(DVec2::new(0.0, 399.5)).is_none());
    }

    #[test]
    fn scaled_centre_subsample() {
        for c in [
            CameraConfig::pinhole_hfov(64, 48, 70.0),
            cfg("{model: mei, width: 64, height: 48, intrinsics: [60, 60, 31.5, 23.5], distortion: [0.01, 0, 0, 0], xi: 0.9}"),
            cfg("{model: scaramuzza, width: 64, height: 48, inv_poly: [62.83, 40.0, 1.5], affine: [1.0, 0.01, 0.0], center: [31.5, 23.5]}"),
        ] {
            let m = c.build().unwrap();
            let s = m.scaled(3);
            let p = DVec3::new(0.3, -0.2, 1.0);
            let a = m.project(p).unwrap();
            let b = s.project(p).unwrap();
            assert!((b - (a * 3.0 + DVec2::splat(1.0))).length() < 1e-9, "{}", c.model);
        }
    }

    #[test]
    fn yaml_roundtrip() {
        let c = cfg("{model: mei, width: 640, height: 480, intrinsics: [1, 2, 3, 4], distortion: [0.1, 0.2, 0.3, 0.4], xi: 0.8}");
        let s = serde_yaml::to_string(&c).unwrap();
        assert_eq!(cfg(&s), c);
    }

    #[test]
    fn nadir_mount_axes() {
        let e = Extrinsics::default();
        let r = e.r_body_cam();
        // optical axis points body-down; image top (−y_cam) points body-forward
        assert!((r * DVec3::Z - DVec3::Z).length() < 1e-12);
        assert!((r * -DVec3::Y - DVec3::X).length() < 1e-12);
        let f = Extrinsics { mount: Mount::Forward, pitch_deg: -30.0, ..Default::default() };
        let z = f.r_body_cam() * DVec3::Z;
        assert!(z.z > 0.49 && z.x > 0.86, "{z:?}");
        let n = Extrinsics { pitch_deg: 20.0, ..Default::default() };
        let z = n.r_body_cam() * DVec3::Z;
        assert!(z.x > 0.34 && z.z > 0.93, "{z:?}");
    }
}
