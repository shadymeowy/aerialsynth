//! Trajectories: poses with timestamps, CSV IO in several frames, and synthetic flights.
//!
//! Pose convention: body frame FRD, attitude `q_ned_body` maps body vectors into the local NED
//! frame *at the body's own position* (v_ned = q * v_body). Positions are geodetic (WGS84 by
//! default, heights above the ellipsoid).
//!
//! CSV formats (header required, `#` comments allowed, columns matched by name, other columns
//! ignored; `t` in seconds, strictly increasing):
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

/// A trajectory CSV: the header (lower case), the rows' fields with their line numbers, and the
/// `# origin:` line. Fields are parsed as numbers only where a column is used.
struct Table {
    header: Vec<String>,
    rows: Vec<(usize, Vec<String>)>,
    origin: Option<Geodetic>,
}

impl Table {
    fn col(&self, n: &str) -> Option<usize> {
        self.header.iter().position(|c| c == n)
    }

    fn need(&self, n: &str) -> Result<usize> {
        self.col(n).ok_or_else(|| anyhow!("trajectory column '{n}' missing (have {:?})", self.header))
    }

    /// The time column (`t`, `time` or `timestamp`).
    fn time_col(&self) -> Option<usize> {
        self.col("t").or(self.col("time")).or(self.col("timestamp"))
    }

    /// Column `c` of every row, as finite numbers.
    fn column(&self, c: usize) -> Result<Vec<f64>> {
        self.rows
            .iter()
            .map(|(ln, r)| {
                let v: f64 = r[c].parse().map_err(|_| anyhow!("line {ln}, column '{}': {:?} is not a number", self.header[c], r[c]))?;
                if !v.is_finite() {
                    bail!("line {ln}, column '{}': {v} is not finite", self.header[c]);
                }
                Ok(v)
            })
            .collect()
    }

    /// The times of column `c`, which must increase strictly.
    fn times(&self, c: usize) -> Result<Vec<f64>> {
        let t = self.column(c)?;
        for i in 1..t.len() {
            if t[i] <= t[i - 1] {
                bail!("line {}, column '{}': {} does not increase (the row before has {})", self.rows[i].0, self.header[c], t[i], t[i - 1]);
            }
        }
        Ok(t)
    }

    /// The attitude quaternions (`qw, qx, qy, qz`, normalised), if the table has them.
    fn quats(&self) -> Result<Option<Vec<DQuat>>> {
        let (Some(w), Some(x), Some(y), Some(z)) = (self.col("qw"), self.col("qx"), self.col("qy"), self.col("qz")) else {
            return Ok(None);
        };
        let (w, x, y, z) = (self.column(w)?, self.column(x)?, self.column(y)?, self.column(z)?);
        let mut out = Vec::with_capacity(w.len());
        for i in 0..w.len() {
            let q = DQuat::from_xyzw(x[i], y[i], z[i], w[i]);
            if q.length() < 1e-9 {
                bail!("line {}: the quaternion (qw, qx, qy, qz) is zero", self.rows[i].0);
            }
            out.push(q.normalize());
        }
        Ok(Some(out))
    }
}

fn parse_csv(text: &str) -> Result<Table> {
    let mut header: Option<Vec<String>> = None;
    let mut rows = Vec::new();
    let mut origin = None;
    for (i, line) in text.lines().enumerate() {
        let (ln, l) = (i + 1, line.trim());
        if l.is_empty() {
            continue;
        }
        if let Some(c) = l.strip_prefix('#') {
            let c = c.trim();
            if let Some(o) = c.strip_prefix("origin:") {
                let v: Vec<f64> = o.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>().with_context(|| format!("line {ln}: origin"))?;
                if v.len() != 3 || v.iter().any(|x| !x.is_finite()) {
                    bail!("line {ln}: origin needs lat,lon,h");
                }
                origin = Some(Geodetic::from_deg(v[0], v[1], v[2]));
            }
            continue;
        }
        let fields: Vec<String> = l.split(',').map(|s| s.trim().to_string()).collect();
        match &header {
            None => header = Some(fields.into_iter().map(|f| f.to_lowercase()).collect()),
            Some(h) => {
                if fields.len() != h.len() {
                    bail!("line {ln}: {} fields, the header has {}", fields.len(), h.len());
                }
                rows.push((ln, fields));
            }
        }
    }
    Ok(Table { header: header.ok_or_else(|| anyhow!("empty trajectory"))?, rows, origin })
}

pub fn load(path: &Path, ell: &Ellipsoid) -> Result<Vec<Pose>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text, ell).with_context(|| format!("trajectory {}", path.display()))
}

pub fn parse(text: &str, ell: &Ellipsoid) -> Result<Vec<Pose>> {
    let tb = parse_csv(text)?;
    let t = tb.times(tb.time_col().ok_or_else(|| anyhow!("trajectory column 't' missing (have {:?})", tb.header))?)?;
    let quats = tb.quats()?;
    let mut out = Vec::with_capacity(t.len());
    if tb.col("lat").is_some() {
        let (la, lo) = (tb.column(tb.need("lat")?)?, tb.column(tb.need("lon")?)?);
        let hh = tb.column(tb.need("h").or_else(|_| tb.need("alt"))?)?;
        let q = match quats {
            Some(q) => q,
            None => {
                let (ro, pi, ya) = (tb.column(tb.need("roll")?)?, tb.column(tb.need("pitch")?)?, tb.column(tb.need("yaw")?)?);
                (0..t.len()).map(|i| geodesy::euler_zyx_to_quat(ya[i].to_radians(), pi[i].to_radians(), ro[i].to_radians())).collect()
            }
        };
        for i in 0..t.len() {
            out.push(Pose { t: t[i], geo: Geodetic::from_deg(la[i], lo[i], hh[i]), q_ned_body: q[i] });
        }
    } else if tb.col("x").is_some() {
        let (x, y, z) = (tb.column(tb.need("x")?)?, tb.column(tb.need("y")?)?, tb.column(tb.need("z")?)?);
        let q = quats.ok_or_else(|| anyhow!("ECEF trajectory needs qw,qx,qy,qz"))?;
        for i in 0..t.len() {
            let geo = geodesy::ecef2geodetic(DVec3::new(x[i], y[i], z[i]), ell);
            out.push(Pose { t: t[i], geo, q_ned_body: geodesy::body2ecef_to_body2ned(q[i], geo.lat, geo.lon) });
        }
    } else if tb.col("n").is_some() {
        let o = tb.origin.ok_or_else(|| anyhow!("NED trajectory needs a '# origin: lat,lon,h' line"))?;
        let (n, e, d) = (tb.column(tb.need("n")?)?, tb.column(tb.need("e")?)?, tb.column(tb.need("d")?)?);
        let q = quats.ok_or_else(|| anyhow!("NED trajectory needs qw,qx,qy,qz"))?;
        let r_ecef_ned0 = geodesy::rot_ecef2ned(o.lat, o.lon).transpose();
        for i in 0..t.len() {
            let geo = geodesy::ned2geodetic(DVec3::new(n[i], e[i], d[i]), o, ell);
            let q_ecef_body = DQuat::from_mat3(&r_ecef_ned0) * q[i];
            out.push(Pose { t: t[i], geo, q_ned_body: geodesy::body2ecef_to_body2ned(q_ecef_body, geo.lat, geo.lon) });
        }
    } else {
        bail!("unrecognized trajectory columns {:?}", tb.header);
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
/// written by `terrain run --step traj`.
#[derive(Clone, Copy, Debug)]
pub struct ImuTruth {
    pub t: f64,
    pub f: DVec3,
    pub w: DVec3,
}

/// Load IMU truth columns if the trajectory has them. Each record is the mean specific force /
/// inertial body rate (body frame) over the interval since the previous record.
pub fn load_imu_truth(path: &Path) -> Result<Option<Vec<ImuTruth>>> {
    let text = std::fs::read_to_string(path)?;
    let tb = parse_csv(&text).with_context(|| format!("trajectory {}", path.display()))?;
    let (Some(t), Some(fx), Some(fy), Some(fz), Some(wx), Some(wy), Some(wz)) =
        (tb.time_col(), tb.col("f_x"), tb.col("f_y"), tb.col("f_z"), tb.col("w_x"), tb.col("w_y"), tb.col("w_z"))
    else {
        return Ok(None);
    };
    let t = tb.times(t)?;
    let (fx, fy, fz) = (tb.column(fx)?, tb.column(fy)?, tb.column(fz)?);
    let (wx, wy, wz) = (tb.column(wx)?, tb.column(wy)?, tb.column(wz)?);
    Ok(Some((0..t.len()).map(|i| ImuTruth { t: t[i], f: DVec3::new(fx[i], fy[i], fz[i]), w: DVec3::new(wx[i], wy[i], wz[i]) }).collect()))
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

    #[test]
    fn csv_checks_rows_and_ignores_other_columns() {
        let ell = Ellipsoid::WGS84;
        let err = |s: &str| format!("{:#}", parse(s, &ell).unwrap_err());
        // the viewer's frames.csv: extra columns, a text column
        let rec = "frame,t,lat,lon,h,roll,pitch,yaw,km,mode,max_zoom\n0,0.0,7.0,-102.0,6000.0,0,-30,89,9.7,surface,16\n1,0.04,7.0,-101.99,5900.0,0,-29.9,89.1,9.5,surface,16\n";
        let p = parse(rec, &ell).unwrap();
        assert_eq!(p.len(), 2);
        assert!((p[1].geo.h - 5900.0).abs() < 1e-9 && (p[1].t - 0.04).abs() < 1e-12);
        let head = "t,lat,lon,h,qw,qx,qy,qz\n";
        let short = format!("{head}0,7,-102,100,1,0,0,0\n1,7,-102,100,1,0,0\n");
        assert!(err(&short).contains("line 3: 7 fields, the header has 8"), "{}", err(&short));
        let nan = format!("{head}0,7,-102,nan,1,0,0,0\n");
        assert!(err(&nan).contains("line 2, column 'h'"), "{}", err(&nan));
        let text = format!("{head}0,7,-102,x,1,0,0,0\n");
        assert!(err(&text).contains("\"x\" is not a number"), "{}", err(&text));
        let zero = format!("{head}0,7,-102,100,0,0,0,0\n");
        assert!(err(&zero).contains("line 2: the quaternion"), "{}", err(&zero));
        let unsorted = format!("{head}0,7,-102,100,1,0,0,0\n2,7,-102,100,1,0,0,0\n1,7,-102,100,1,0,0,0\n");
        assert!(err(&unsorted).contains("line 4, column 't': 1 does not increase"), "{}", err(&unsorted));
    }
}
