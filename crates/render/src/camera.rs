//! Camera models and the camera rig (intrinsics + body→camera extrinsics).
//!
//! Conventions:
//! * Camera frame = OpenCV: x right, y down, z forward (optical axis).
//! * Pixel coordinates = OpenCV: the centre of the top-left pixel is (0, 0); the image spans
//!   [-0.5, w-0.5] x [-0.5, h-0.5].
//! * Body frame = FRD (x forward, y right, z down).

use anyhow::{bail, Result};
use glam::{DMat3, DQuat, DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A generic central camera model.
pub trait CameraModel: Send + Sync + std::fmt::Debug {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    /// Project a point in the camera frame to pixel coordinates. `None` if not imageable
    /// (behind the camera / outside the model's valid domain).
    fn project(&self, p: DVec3) -> Option<DVec2>;
    /// Back-project a pixel to a unit ray in the camera frame.
    fn unproject(&self, px: DVec2) -> Option<DVec3>;
    /// Half-angle (rad) of a cone around +z that contains every pixel's ray.
    fn max_half_angle(&self) -> f64;
    /// Focal length in pixels (used for LOD / footprint estimates).
    fn focal_px(&self) -> f64;
    /// The same camera rendered at `s` times the resolution (sub-sample centres line up so that
    /// for odd `s` the central sub-sample coincides with the original pixel centre).
    fn scaled(&self, s: u32) -> Arc<dyn CameraModel>;
    /// Serializable description.
    fn config(&self) -> CameraConfig;
}

/// Serializable camera intrinsics (tagged by `model`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "model", rename_all = "snake_case", deny_unknown_fields)]
pub enum CameraConfig {
    /// Pinhole with OpenCV radial-tangential distortion (k1, k2, p1, p2, k3).
    PinholeRadtan {
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        #[serde(default)]
        k1: f64,
        #[serde(default)]
        k2: f64,
        #[serde(default)]
        p1: f64,
        #[serde(default)]
        p2: f64,
        #[serde(default)]
        k3: f64,
    },
}

impl CameraConfig {
    /// Pinhole camera with a given horizontal field of view (deg) and principal point at the centre.
    pub fn pinhole_hfov(width: u32, height: u32, hfov_deg: f64) -> Self {
        let fx = 0.5 * width as f64 / (0.5 * hfov_deg.to_radians()).tan();
        CameraConfig::PinholeRadtan {
            width,
            height,
            fx,
            fy: fx,
            cx: (width as f64 - 1.0) * 0.5,
            cy: (height as f64 - 1.0) * 0.5,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        }
    }

    pub fn build(&self) -> Result<Arc<dyn CameraModel>> {
        match *self {
            CameraConfig::PinholeRadtan { width, height, fx, fy, cx, cy, k1, k2, p1, p2, k3 } => {
                if width == 0 || height == 0 || fx <= 0.0 || fy <= 0.0 {
                    bail!("invalid pinhole intrinsics");
                }
                Ok(Arc::new(PinholeRadtan::new(width, height, fx, fy, cx, cy, [k1, k2, p1, p2, k3])))
            }
        }
    }
}

/// Pinhole + OpenCV radtan distortion.
#[derive(Clone, Debug)]
pub struct PinholeRadtan {
    pub w: u32,
    pub h: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// k1, k2, p1, p2, k3
    pub d: [f64; 5],
    /// Max normalized radius² for which the distortion is monotonic & the model valid.
    r2_max: f64,
    half_angle: f64,
}

impl PinholeRadtan {
    pub fn new(w: u32, h: u32, fx: f64, fy: f64, cx: f64, cy: f64, d: [f64; 5]) -> Self {
        let mut c = PinholeRadtan { w, h, fx, fy, cx, cy, d, r2_max: f64::MAX, half_angle: 0.0 };
        // largest undistorted radius covering the image corners (+ margin)
        let mut r_img: f64 = 0.0;
        for (u, v) in [(-0.5, -0.5), (w as f64 - 0.5, -0.5), (-0.5, h as f64 - 0.5), (w as f64 - 0.5, h as f64 - 0.5)] {
            let xn = DVec2::new((u - cx) / fx, (v - cy) / fy);
            let und = c.undistort(xn);
            r_img = r_img.max(und.length());
        }
        // validity limit: where radial distortion stops being monotonic, or 3x image radius
        let mut r2_lim = (3.0 * r_img).powi(2);
        let mut r = 0.0;
        let mut prev = 0.0;
        while r < 3.0 * r_img {
            r += r_img * 0.01;
            let r2 = r * r;
            let rd = r * (1.0 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2);
            if rd <= prev {
                r2_lim = (r - r_img * 0.01).powi(2);
                break;
            }
            prev = rd;
        }
        c.r2_max = r2_lim;
        c.half_angle = r_img.atan();
        c
    }

    #[inline]
    fn distort(&self, x: DVec2) -> DVec2 {
        let [k1, k2, p1, p2, k3] = self.d;
        let r2 = x.length_squared();
        let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        let xy = x.x * x.y;
        DVec2::new(
            x.x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x.x * x.x),
            x.y * radial + p1 * (r2 + 2.0 * x.y * x.y) + 2.0 * p2 * xy,
        )
    }

    /// Invert the distortion (Gauss-Newton with numeric Jacobian; converges in a few iterations).
    fn undistort(&self, xd: DVec2) -> DVec2 {
        if self.d.iter().all(|v| *v == 0.0) {
            return xd;
        }
        let mut x = xd;
        for _ in 0..20 {
            let f = self.distort(x) - xd;
            if f.length_squared() < 1e-24 {
                break;
            }
            let e = 1e-7;
            let jx = (self.distort(x + DVec2::new(e, 0.0)) - self.distort(x)) / e;
            let jy = (self.distort(x + DVec2::new(0.0, e)) - self.distort(x)) / e;
            let det = jx.x * jy.y - jy.x * jx.y;
            if det.abs() < 1e-15 {
                break;
            }
            let dx = DVec2::new(jy.y * f.x - jy.x * f.y, -jx.y * f.x + jx.x * f.y) / det;
            x -= dx;
        }
        x
    }
}

impl CameraModel for PinholeRadtan {
    fn width(&self) -> u32 {
        self.w
    }
    fn height(&self) -> u32 {
        self.h
    }
    #[inline]
    fn project(&self, p: DVec3) -> Option<DVec2> {
        if p.z <= 1e-9 {
            return None;
        }
        let x = DVec2::new(p.x / p.z, p.y / p.z);
        if x.length_squared() > self.r2_max {
            return None;
        }
        let xd = self.distort(x);
        Some(DVec2::new(self.fx * xd.x + self.cx, self.fy * xd.y + self.cy))
    }
    fn unproject(&self, px: DVec2) -> Option<DVec3> {
        let xd = DVec2::new((px.x - self.cx) / self.fx, (px.y - self.cy) / self.fy);
        let x = self.undistort(xd);
        Some(DVec3::new(x.x, x.y, 1.0).normalize())
    }
    fn max_half_angle(&self) -> f64 {
        self.half_angle
    }
    fn focal_px(&self) -> f64 {
        self.fx.max(self.fy)
    }
    fn scaled(&self, s: u32) -> Arc<dyn CameraModel> {
        let sf = s as f64;
        let off = (sf - 1.0) * 0.5;
        let mut c = self.clone();
        c.w = self.w * s;
        c.h = self.h * s;
        c.fx *= sf;
        c.fy *= sf;
        c.cx = self.cx * sf + off;
        c.cy = self.cy * sf + off;
        Arc::new(c)
    }
    fn config(&self) -> CameraConfig {
        let [k1, k2, p1, p2, k3] = self.d;
        CameraConfig::PinholeRadtan { width: self.w, height: self.h, fx: self.fx, fy: self.fy, cx: self.cx, cy: self.cy, k1, k2, p1, p2, k3 }
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

    #[test]
    fn radtan_roundtrip() {
        let c = PinholeRadtan::new(640, 480, 400.0, 405.0, 320.0, 240.0, [-0.28, 0.07, 0.001, -0.0005, 0.0]);
        for (u, v) in [(0.0, 0.0), (639.0, 479.0), (320.0, 240.0), (100.0, 400.0)] {
            let r = c.unproject(DVec2::new(u, v)).unwrap();
            let px = c.project(r * 7.0).unwrap();
            assert!((px - DVec2::new(u, v)).length() < 1e-6, "{u},{v} -> {px:?}");
        }
    }

    #[test]
    fn scaled_centre_subsample() {
        let c = PinholeRadtan::new(64, 48, 50.0, 50.0, 31.5, 23.5, [0.0; 5]);
        let s = c.scaled(3);
        let p = DVec3::new(0.3, -0.2, 1.0);
        let a = c.project(p).unwrap();
        let b = s.project(p).unwrap();
        assert!((b - (a * 3.0 + DVec2::splat(1.0))).length() < 1e-9);
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
        assert!(z.z > 0.49 && z.x > 0.86, "{z:?}"); // tilted down by 30°
        let n = Extrinsics { pitch_deg: 20.0, ..Default::default() };
        let z = n.r_body_cam() * DVec3::Z;
        assert!(z.x > 0.34 && z.z > 0.93, "{z:?}"); // nadir tilted forward by 20°
    }
}
