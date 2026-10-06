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
use std::io::Write;
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

pub fn save(path: &Path, poses: &[Pose]) -> Result<()> {
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "# body FRD; q = body->NED (Hamilton); lat/lon deg; h m above WGS84 ellipsoid")?;
    writeln!(f, "t,lat,lon,h,qw,qx,qy,qz,roll,pitch,yaw")?;
    for p in poses {
        let q = p.q_ned_body;
        let (y, pi, r) = geodesy::quat_to_euler_zyx(q);
        writeln!(
            f,
            "{:.6},{:.10},{:.10},{:.4},{:.9},{:.9},{:.9},{:.9},{:.4},{:.4},{:.4}",
            p.t,
            p.geo.lat.to_degrees(),
            p.geo.lon.to_degrees(),
            p.geo.h,
            q.w,
            q.x,
            q.y,
            q.z,
            r.to_degrees(),
            pi.to_degrees(),
            y.to_degrees()
        )?;
    }
    Ok(())
}

/// Parameters of a synthetic flight.
#[derive(Clone, Debug)]
pub struct SynthParams {
    pub kind: SynthKind,
    pub lat: f64,
    pub lon: f64,
    /// Altitude (m): above the ellipsoid, or above the smoothed ground when `ground` is given.
    pub alt: f64,
    pub speed: f64,
    pub rate: f64,
    pub duration: f64,
    pub heading_deg: f64,
    /// Turn radius for circle / figure8 / lawnmower turns (m).
    pub radius: f64,
    /// Leg length for lawnmower (m).
    pub leg: f64,
    pub seed: u64,
    /// Attitude jitter amplitude (deg), smooth.
    pub wobble_deg: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthKind {
    Line,
    Circle,
    Figure8,
    Lawnmower,
    Random,
}

impl std::str::FromStr for SynthKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "line" => SynthKind::Line,
            "circle" => SynthKind::Circle,
            "figure8" => SynthKind::Figure8,
            "lawnmower" => SynthKind::Lawnmower,
            "random" => SynthKind::Random,
            _ => bail!("unknown trajectory kind {s} (line|circle|figure8|lawnmower|random)"),
        })
    }
}

fn hrand(seed: u64, k: u64) -> f64 {
    let mut h = seed ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Synthesize a flight. `ground(lat, lon)` (radians) gives the terrain height to follow (AGL
/// mode); it is low-pass filtered along the path.
pub fn synthesize(p: &SynthParams, ell: &Ellipsoid, ground: Option<&dyn Fn(f64, f64) -> f64>) -> Vec<Pose> {
    let g = 9.80665;
    let dt_int = 0.02; // integration step
    let n_out = (p.duration * p.rate).floor() as usize + 1;
    let v = p.speed;
    // turn rate as a function of time
    let turn_rate = |t: f64| -> f64 {
        match p.kind {
            SynthKind::Line => 0.0,
            SynthKind::Circle => v / p.radius,
            SynthKind::Figure8 => {
                let period = std::f64::consts::TAU * p.radius / v;
                if (t / period).floor() as i64 % 2 == 0 { v / p.radius } else { -v / p.radius }
            }
            SynthKind::Lawnmower => {
                let t_leg = p.leg / v;
                let t_turn = std::f64::consts::PI * p.radius / v;
                let cyc = t_leg + t_turn;
                let k = (t / cyc).floor() as i64;
                let f = t - k as f64 * cyc;
                if f < t_leg {
                    0.0
                } else if k % 2 == 0 {
                    v / p.radius
                } else {
                    -v / p.radius
                }
            }
            SynthKind::Random => {
                let mut w = 0.0;
                for k in 0..5 {
                    let f = 0.004 + 0.03 * hrand(p.seed, 10 + k);
                    let a = (0.6 + hrand(p.seed, 20 + k)) * v / p.radius / 3.0;
                    w += a * (std::f64::consts::TAU * f * t + 6.28 * hrand(p.seed, 30 + k)).sin();
                }
                w.clamp(-v / p.radius, v / p.radius)
            }
        }
    };
    // integrate heading and position on the ellipsoid
    let mut lat = p.lat.to_radians();
    let mut lon = p.lon.to_radians();
    let mut heading = p.heading_deg.to_radians();
    let mut samples = Vec::new(); // (t, lat, lon, heading, turn_rate)
    let mut t = 0.0;
    let t_end = p.duration + 1e-9;
    let mut next_out = 0usize;
    while next_out < n_out {
        let t_out = next_out as f64 / p.rate;
        while t + 1e-12 < t_out {
            let h = (t_out - t).min(dt_int);
            let w = turn_rate(t + 0.5 * h);
            let hd = heading + 0.5 * h * w;
            let m = ell.meridian_radius(lat) + p.alt;
            let nr = ell.prime_vertical_radius(lat) + p.alt;
            lat += v * hd.cos() * h / m;
            lon += v * hd.sin() * h / (nr * lat.cos());
            heading += w * h;
            t += h;
        }
        samples.push((t_out, lat, lon, heading, turn_rate(t_out)));
        next_out += 1;
        if t_out > t_end {
            break;
        }
    }
    // altitude profile
    let mut alts: Vec<f64> = match ground {
        None => vec![p.alt; samples.len()],
        Some(gf) => {
            let raw: Vec<f64> = samples.iter().map(|s| gf(s.1, s.2)).collect();
            // running max over ±1.5 km then smoothing over ±3 km (keeps clearance, smooth climb)
            let win = ((1500.0 / v) * p.rate).ceil() as isize;
            let n = raw.len() as isize;
            let mx: Vec<f64> = (0..n)
                .map(|i| (i - win..=i + win).filter(|j| *j >= 0 && *j < n).map(|j| raw[j as usize]).fold(f64::MIN, f64::max))
                .collect();
            let win2 = 2 * win;
            (0..n)
                .map(|i| {
                    let r: Vec<f64> = (i - win2..=i + win2).filter(|j| *j >= 0 && *j < n).map(|j| mx[j as usize]).collect();
                    r.iter().sum::<f64>() / r.len() as f64 + p.alt
                })
                .collect()
        }
    };
    if alts.is_empty() {
        alts.push(p.alt);
    }
    let mut out = Vec::with_capacity(samples.len());
    for (k, &(t, lat, lon, heading, w)) in samples.iter().enumerate() {
        let h = alts[k];
        let climb = if k + 1 < samples.len() { (alts[k + 1] - h) * p.rate } else if k > 0 { (h - alts[k - 1]) * p.rate } else { 0.0 };
        let bank = (v * w / g).atan();
        let pitch = (climb / v).atan() + 2f64.to_radians(); // small angle of attack
        let (mut r, mut pi, mut ya) = (bank, pitch, heading);
        if p.wobble_deg > 0.0 {
            let a = p.wobble_deg.to_radians();
            for (j, x) in [&mut r, &mut pi, &mut ya].into_iter().enumerate() {
                let mut s = 0.0;
                for k2 in 0..3u64 {
                    let f = 0.1 + 0.5 * hrand(p.seed, 100 + 10 * j as u64 + k2);
                    s += (std::f64::consts::TAU * f * t + 6.28 * hrand(p.seed, 200 + 10 * j as u64 + k2)).sin() / 3.0;
                }
                *x += a * s;
            }
        }
        let q = geodesy::euler_zyx_to_quat(ya, pi, r);
        out.push(Pose { t, geo: Geodetic::new(lat, lon, h), q_ned_body: q });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_roundtrip_and_ned() {
        let ell = Ellipsoid::WGS84;
        let p = SynthParams {
            kind: SynthKind::Circle,
            lat: 39.9,
            lon: 32.8,
            alt: 1000.0,
            speed: 50.0,
            rate: 2.0,
            duration: 30.0,
            heading_deg: 10.0,
            radius: 800.0,
            leg: 1000.0,
            seed: 1,
            wobble_deg: 0.0,
        };
        let poses = synthesize(&p, &ell, None);
        assert_eq!(poses.len(), 61);
        let dir = std::env::temp_dir().join(format!("traj-{}.csv", std::process::id()));
        save(&dir, &poses).unwrap();
        let back = load(&dir, &ell).unwrap();
        for (a, b) in poses.iter().zip(&back) {
            assert!((a.ecef(&ell) - b.ecef(&ell)).length() < 1e-3);
            assert!(a.q_ned_body.dot(b.q_ned_body).abs() > 1.0 - 1e-12);
        }
        // speed check
        let d = (poses[1].ecef(&ell) - poses[0].ecef(&ell)).length();
        assert!((d - 25.0).abs() < 0.2, "{d}");
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
