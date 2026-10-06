//! Trajectories: poses with timestamps, CSV IO in several frames, and synthetic flights.
//!
//! Pose convention: body frame FRD, attitude `q_ned_body` maps body vectors into the local NED
//! frame *at the body's own position* (v_ned = q * v_body). Positions are geodetic (WGS84 by
//! default, heights above the ellipsoid).
//!
//! CSV formats (header required, `#` comments allowed, columns matched by name):
//! * `t,lat,lon,h,qw,qx,qy,qz`      — degrees / meters, q = body→NED
//! * `t,lat,lon,h,roll,pitch,yaw`   — degrees (aerospace ZYX)
//! * `t,x,y,z,qw,qx,qy,qz`          — ECEF meters, q = body→ECEF
//! * `t,n,e,d,qw,qx,qy,qz`          — local NED meters w.r.t. `# origin: lat,lon,h` (deg, deg, m),
//!   q = body→NED(origin)  (pymap3d `ned2geodetic` semantics)

use crate::camera::Extrinsics;
use anyhow::{anyhow, bail, Context, Result};
use geodesy::{Ellipsoid, Geodetic};
use glam::{DMat3, DQuat, DVec3};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub t: f64,
    pub geo: Geodetic,
    pub q_ned_body: DQuat,
}

/// Camera pose in ECEF (f64).
#[derive(Clone, Copy, Debug)]
pub struct CamPose {
    pub t: f64,
    pub pos: DVec3,
    /// Columns = camera axes in ECEF (maps camera-frame vectors to ECEF).
    pub r_ecef_cam: DMat3,
}

impl CamPose {
    #[inline]
    pub fn world_to_cam(&self, p: DVec3) -> DVec3 {
        self.r_ecef_cam.transpose() * (p - self.pos)
    }
    #[inline]
    pub fn cam_to_world(&self, p: DVec3) -> DVec3 {
        self.r_ecef_cam * p + self.pos
    }
    pub fn q_ecef_cam(&self) -> DQuat {
        DQuat::from_mat3(&self.r_ecef_cam).normalize()
    }
}

impl Pose {
    pub fn ecef(&self, ell: &Ellipsoid) -> DVec3 {
        geodesy::geodetic2ecef(self.geo, ell)
    }
    pub fn r_ecef_body(&self) -> DMat3 {
        geodesy::rot_ecef2ned(self.geo.lat, self.geo.lon).transpose() * DMat3::from_quat(self.q_ned_body)
    }
    pub fn camera(&self, ext: &Extrinsics, ell: &Ellipsoid) -> CamPose {
        let rb = self.r_ecef_body();
        CamPose { t: self.t, pos: self.ecef(ell) + rb * ext.t_body_cam(), r_ecef_cam: rb * ext.r_body_cam() }
    }
}

fn parse_csv(text: &str) -> Result<(Vec<String>, Vec<Vec<f64>>, Option<Geodetic>)> {
    let mut header: Option<Vec<String>> = None;
    let mut rows = Vec::new();
    let mut origin = None;
    for (ln, line) in text.lines().enumerate() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if let Some(c) = l.strip_prefix('#') {
            let c = c.trim();
            if let Some(o) = c.strip_prefix("origin:") {
                let v: Vec<f64> = o.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
                if v.len() != 3 {
                    bail!("origin needs lat,lon,h");
                }
                origin = Some(Geodetic::from_deg(v[0], v[1], v[2]));
            }
            continue;
        }
        if header.is_none() {
            header = Some(l.split(',').map(|s| s.trim().to_lowercase()).collect());
            continue;
        }
        let v: Vec<f64> = l
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .with_context(|| format!("line {}", ln + 1))?;
        rows.push(v);
    }
    Ok((header.ok_or_else(|| anyhow!("empty trajectory"))?, rows, origin))
}

pub fn load(path: &Path, ell: &Ellipsoid) -> Result<Vec<Pose>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text, ell)
}

pub fn parse(text: &str, ell: &Ellipsoid) -> Result<Vec<Pose>> {
    let (h, rows, origin) = parse_csv(text)?;
    let col = |n: &str| h.iter().position(|c| c == n);
    let need = |n: &str| col(n).ok_or_else(|| anyhow!("trajectory column '{n}' missing (have {h:?})"));
    let t = need("t").or_else(|_| need("time")).or_else(|_| need("timestamp"))?;
    let quat = (col("qw"), col("qx"), col("qy"), col("qz"));
    let get_q = |r: &[f64]| -> Option<DQuat> {
        match quat {
            (Some(w), Some(x), Some(y), Some(z)) => Some(DQuat::from_xyzw(r[x], r[y], r[z], r[w]).normalize()),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(rows.len());
    if col("lat").is_some() {
        let (la, lo, hh) = (need("lat")?, need("lon")?, need("h").or_else(|_| need("alt"))?);
        for r in &rows {
            let geo = Geodetic::from_deg(r[la], r[lo], r[hh]);
            let q = match get_q(r) {
                Some(q) => q,
                None => {
                    let (ro, pi, ya) = (need("roll")?, need("pitch")?, need("yaw")?);
                    geodesy::euler_zyx_to_quat(r[ya].to_radians(), r[pi].to_radians(), r[ro].to_radians())
                }
            };
            out.push(Pose { t: r[t], geo, q_ned_body: q });
        }
    } else if col("x").is_some() {
        let (x, y, z) = (need("x")?, need("y")?, need("z")?);
        for r in &rows {
            let p = DVec3::new(r[x], r[y], r[z]);
            let geo = geodesy::ecef2geodetic(p, ell);
            let q_ecef_body = get_q(r).ok_or_else(|| anyhow!("ECEF trajectory needs qw,qx,qy,qz"))?;
            let q = geodesy::body2ecef_to_body2ned(q_ecef_body, geo.lat, geo.lon);
            out.push(Pose { t: r[t], geo, q_ned_body: q });
        }
    } else if col("n").is_some() {
        let o = origin.ok_or_else(|| anyhow!("NED trajectory needs a '# origin: lat,lon,h' line"))?;
        let (n, e, d) = (need("n")?, need("e")?, need("d")?);
        let r_ecef_ned0 = geodesy::rot_ecef2ned(o.lat, o.lon).transpose();
        for r in &rows {
            let geo = geodesy::ned2geodetic(DVec3::new(r[n], r[e], r[d]), o, ell);
            let q0 = get_q(r).ok_or_else(|| anyhow!("NED trajectory needs qw,qx,qy,qz"))?;
            let q_ecef_body = DQuat::from_mat3(&r_ecef_ned0) * q0;
            let q = geodesy::body2ecef_to_body2ned(q_ecef_body, geo.lat, geo.lon);
            out.push(Pose { t: r[t], geo, q_ned_body: q });
        }
    } else {
        bail!("unrecognized trajectory columns {h:?}");
    }
    Ok(out)
}

/// Interpolate a (time-sorted) trajectory at time `t` (clamped to its span): geodetic position
/// linearly, attitude by slerp.
pub fn interpolate(poses: &[Pose], t: f64) -> Pose {
    assert!(!poses.is_empty());
    if t <= poses[0].t {
        return Pose { t, ..poses[0] };
    }
    let last = poses[poses.len() - 1];
    if t >= last.t {
        return Pose { t, ..last };
    }
    let i = poses.partition_point(|p| p.t <= t).max(1);
    let (a, b) = (&poses[i - 1], &poses[i]);
    let f = ((t - a.t) / (b.t - a.t).max(1e-12)).clamp(0.0, 1.0);
    let mut dlon = b.geo.lon - a.geo.lon;
    if dlon > std::f64::consts::PI {
        dlon -= std::f64::consts::TAU;
    } else if dlon < -std::f64::consts::PI {
        dlon += std::f64::consts::TAU;
    }
    let qb = if a.q_ned_body.dot(b.q_ned_body) < 0.0 { -b.q_ned_body } else { b.q_ned_body };
    Pose {
        t,
        geo: Geodetic::new(a.geo.lat + (b.geo.lat - a.geo.lat) * f, a.geo.lon + dlon * f, a.geo.h + (b.geo.h - a.geo.h) * f),
        q_ned_body: a.q_ned_body.slerp(qb, f).normalize(),
    }
}

/// True IMU samples carried by a trajectory CSV (columns `f_x..f_z, w_x..w_z`, body FRD), as
/// written by `terrain traj`.
#[derive(Clone, Copy, Debug)]
pub struct ImuTruth {
    pub t: f64,
    pub f: DVec3,
    pub w: DVec3,
}

/// Load IMU truth columns if the trajectory has them.
pub fn load_imu_truth(path: &Path) -> Result<Option<Vec<ImuTruth>>> {
    let text = std::fs::read_to_string(path)?;
    let (h, rows, _) = parse_csv(&text)?;
    let col = |n: &str| h.iter().position(|c| c == n);
    let (Some(t), Some(fx), Some(fy), Some(fz), Some(wx), Some(wy), Some(wz)) =
        (col("t"), col("f_x"), col("f_y"), col("f_z"), col("w_x"), col("w_y"), col("w_z"))
    else {
        return Ok(None);
    };
    Ok(Some(
        rows.iter()
            .map(|r| ImuTruth { t: r[t], f: DVec3::new(r[fx], r[fy], r[fz]), w: DVec3::new(r[wx], r[wy], r[wz]) })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_roundtrip_and_ned() {
        let ell = Ellipsoid::WGS84;
        let sc = crate::dynamics::SynthConfig { kind: crate::dynamics::PathKind::Circle, duration: 30.0, rate: 2.0, ..Default::default() };
        let recs = crate::dynamics::simulate(&sc, (39.9, 32.8), &ell, None);
        let poses: Vec<Pose> = recs.iter().map(|r| r.pose).collect();
        let dir = std::env::temp_dir().join(format!("traj-{}.csv", std::process::id()));
        crate::dynamics::save_records(&dir, &recs, "test").unwrap();
        let back = load(&dir, &ell).unwrap();
        assert_eq!(back.len(), poses.len());
        for (a, b) in poses.iter().zip(&back) {
            assert!((a.ecef(&ell) - b.ecef(&ell)).length() < 1e-3);
            assert!(a.q_ned_body.dot(b.q_ned_body).abs() > 1.0 - 1e-12);
        }
        // NED format with origin
        let o = poses[0].geo;
        let mut s = format!("# origin: {},{},{}\nt,n,e,d,qw,qx,qy,qz\n", o.lat.to_degrees(), o.lon.to_degrees(), o.h);
        let r0 = geodesy::rot_ecef2ned(o.lat, o.lon);
        for p in &poses {
            let ned = geodesy::geodetic2ned(p.geo, o, &ell);
            let q0 = DQuat::from_mat3(&(r0 * p.r_ecef_body()));
            s += &format!("{},{},{},{},{},{},{},{}\n", p.t, ned.x, ned.y, ned.z, q0.w, q0.x, q0.y, q0.z);
        }
        let back = parse(&s, &ell).unwrap();
        for (a, b) in poses.iter().zip(&back) {
            assert!((a.ecef(&ell) - b.ecef(&ell)).length() < 1e-6);
            assert!(a.r_ecef_body().abs_diff_eq(b.r_ecef_body(), 1e-9));
        }
        std::fs::remove_file(dir).ok();
    }
}
