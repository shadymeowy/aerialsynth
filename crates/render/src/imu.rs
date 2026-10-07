//! Synthetic IMU from the trajectory.
//!
//! Truth: `terrain traj` records the exact specific force and inertial angular rate of the body
//! (computed inside the flight simulator from the analytic kinematics + Coriolis / transport
//! rate − WGS84 normal gravity, integrated at 1 kHz). For trajectories without those columns
//! the truth is derived numerically from the poses (lower fidelity).
//!
//! Output: `<imu.path>/{t, accel, gyro, gt_accel, gt_gyro, gt_bias_accel, gt_bias_gyro}` and
//! `calib/T_body_imu`, on the sequence clock (i64 µs since the sequence start).
//!
//! Sensor model per axis (gyro and accelerometer alike):
//!     y = sat( (I + M) (1 + s) x_true + b + n ),   b_{k+1} = b_k + σ_bw √Δt ξ,   n ~ σ_n √f_s ξ
//! with misalignment M (small random skew), scale factor error s, turn-on bias b_0, bias random
//! walk σ_bw and white noise density σ_n (Kalibr naming), and saturation.
//! The IMU frame is given by `extrinsics` relative to the body (FRD); a lever arm adds
//! ω̇ × r + ω × (ω × r) to the specific force.

use crate::trajectory::{self, ImuTruth, Pose};
use anyhow::{bail, Result};
use geodesy::Ellipsoid;
use glam::{DMat3, DQuat, DVec3};
use h5::Attrs;
use serde::{Deserialize, Serialize};

/// Noise / error parameters of one sensor triad. When given in YAML, all fields are required
/// (gyro and accel have different defaults); omit the block to keep the defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImuNoise {
    /// White noise density (gyro rad/s/√Hz, accel m/s²/√Hz).
    pub noise_density: f64,
    /// Bias random walk (gyro rad/s²/√Hz, accel m/s³/√Hz).
    pub random_walk: f64,
    /// Turn-on bias σ (rad/s or m/s²), drawn once per run.
    pub bias_init: f64,
    /// Scale factor error σ (relative) and axis misalignment σ (rad).
    pub scale_sigma: f64,
    pub misalignment_sigma: f64,
    /// Measurement range (rad/s or m/s²); 0 = unlimited.
    pub saturation: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImuExtrinsics {
    /// Rotation IMU → body as quaternion [w, x, y, z] (IMU axes expressed in the body FRD frame).
    pub q_body_imu: [f64; 4],
    /// IMU position in the body frame (m).
    pub translation: [f64; 3],
}

impl Default for ImuExtrinsics {
    fn default() -> Self {
        ImuExtrinsics { q_body_imu: [1.0, 0.0, 0.0, 0.0], translation: [0.0; 3] }
    }
}

impl ImuExtrinsics {
    pub fn r_body_imu(&self) -> DMat3 {
        let [w, x, y, z] = self.q_body_imu;
        DMat3::from_quat(DQuat::from_xyzw(x, y, z, w).normalize())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImuConfig {
    /// HDF5 group of the IMU.
    pub path: String,
    pub rate_hz: f64,
    pub extrinsics: ImuExtrinsics,
    pub gyro: ImuNoise,
    pub accel: ImuNoise,
    pub seed: u64,
}

impl Default for ImuConfig {
    fn default() -> Self {
        ImuConfig {
            path: "/imu".into(),
            rate_hz: 200.0,
            extrinsics: ImuExtrinsics::default(),
            // typical tactical/consumer MEMS (EuRoC ADIS16448-like)
            gyro: ImuNoise { noise_density: 1.7e-4, random_walk: 1.9e-5, bias_init: 2e-3, scale_sigma: 1e-3, misalignment_sigma: 5e-4, saturation: 8.7 },
            accel: ImuNoise { noise_density: 2.0e-3, random_walk: 3.0e-3, bias_init: 0.03, scale_sigma: 1e-3, misalignment_sigma: 5e-4, saturation: 160.0 },
            seed: 5,
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = ((self.next() >> 11) as f64 / (1u64 << 53) as f64).max(1e-300);
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn gvec(&mut self) -> DVec3 {
        DVec3::new(self.gauss(), self.gauss(), self.gauss())
    }
}

/// One synthesized IMU stream (IMU frame).
pub struct ImuData {
    pub t: Vec<f64>,
    pub accel: Vec<DVec3>,
    pub omega: Vec<DVec3>,
    pub gt_accel: Vec<DVec3>,
    pub gt_omega: Vec<DVec3>,
    pub bias_accel: Vec<DVec3>,
    pub bias_gyro: Vec<DVec3>,
    pub from_truth_columns: bool,
}

/// Numerical truth from poses at time t (fallback when the trajectory has no IMU columns):
/// central differences with step h, the stencil kept inside the trajectory.
fn numeric_truth(poses: &[Pose], ell: &Ellipsoid, t: f64, h: f64) -> (DVec3, DVec3) {
    let (lo, hi) = (poses[0].t, poses[poses.len() - 1].t);
    let h = h.min(0.5 * (hi - lo)).max(1e-6);
    let t = t.clamp(lo + h, hi - h);
    let p = |tt: f64| trajectory::interpolate(poses, tt);
    let (a, b, c) = (p(t - h), p(t), p(t + h));
    let (pa, pb, pc) = (a.ecef(ell), b.ecef(ell), c.ecef(ell));
    let v = (pc - pa) / (2.0 * h);
    let acc = (pc - 2.0 * pb + pa) / (h * h);
    let w_ie_e = DVec3::new(0.0, 0.0, geodesy::EARTH_RATE);
    // specific force in ECEF: a + 2 Ω×v − g_normal (normal gravity includes the centrifugal term)
    let r_ne = geodesy::rot_ecef2ned(b.geo.lat, b.geo.lon);
    let g_e = r_ne.transpose() * DVec3::new(0.0, 0.0, geodesy::normal_gravity(b.geo.lat, b.geo.h));
    let f_e = acc + 2.0 * w_ie_e.cross(v) - g_e;
    let r_eb = b.r_ecef_body();
    // body rate from the neighbouring attitudes (ECEF-referenced → inertial apart from Ω)
    let (ra, rc) = (DQuat::from_mat3(&a.r_ecef_body()), DQuat::from_mat3(&c.r_ecef_body()));
    let dq = (ra.inverse() * rc).normalize();
    let dq = if dq.w < 0.0 { -dq } else { dq };
    (r_eb.transpose() * f_e, dq.to_scaled_axis() / (2.0 * h) + r_eb.transpose() * w_ie_e)
}

/// Truth source: simulator records (piecewise constant over each record interval, integrated
/// exactly over a window) or numeric differentiation of the poses.
enum Truth<'a> {
    Records { tr: &'a [ImuTruth], cum_f: Vec<DVec3>, cum_w: Vec<DVec3>, centre: Vec<f64> },
    Numeric { poses: &'a [Pose], ell: &'a Ellipsoid },
}

impl Truth<'_> {
    fn new<'a>(truth: Option<&'a [ImuTruth]>, poses: &'a [Pose], ell: &'a Ellipsoid) -> Truth<'a> {
        match truth {
            Some(tr) if tr.len() >= 2 => {
                // record i covers (t[i-1], t[i]]; the first one an interval as long as the next
                let start = |i: usize| if i == 0 { 2.0 * tr[0].t - tr[1].t } else { tr[i - 1].t };
                let (mut cf, mut cw) = (vec![DVec3::ZERO], vec![DVec3::ZERO]);
                for (i, r) in tr.iter().enumerate() {
                    let d = r.t - start(i);
                    cf.push(cf[i] + r.f * d);
                    cw.push(cw[i] + r.w * d);
                }
                let centre = (0..tr.len()).map(|i| 0.5 * (start(i) + tr[i].t)).collect();
                Truth::Records { tr, cum_f: cf, cum_w: cw, centre }
            }
            _ => Truth::Numeric { poses, ell },
        }
    }

    /// ∫ (f, ω) dt from the start of the records to `t` (clamped to the record span).
    fn integral(tr: &[ImuTruth], cum_f: &[DVec3], cum_w: &[DVec3], t: f64) -> (DVec3, DVec3) {
        let t_first = 2.0 * tr[0].t - tr[1].t;
        let t = t.clamp(t_first, tr[tr.len() - 1].t);
        let i = tr.partition_point(|r| r.t < t).min(tr.len() - 1); // record covering t
        let s = if i == 0 { t_first } else { tr[i - 1].t };
        (cum_f[i] + tr[i].f * (t - s), cum_w[i] + tr[i].w * (t - s))
    }

    /// Mean (f, ω) over [a, b] and the instantaneous ω at a and b (for ω̇).
    fn window(&self, a: f64, b: f64) -> (DVec3, DVec3, DVec3, DVec3) {
        match self {
            Truth::Records { tr, cum_f, cum_w, centre } => {
                // windows reaching past the records are averaged over their covered part
                let (t_first, t_last) = (2.0 * tr[0].t - tr[1].t, tr[tr.len() - 1].t);
                let (ca, cb) = (a.clamp(t_first, t_last), b.clamp(t_first, t_last));
                let (fa, wa) = Self::integral(tr, cum_f, cum_w, ca);
                let (fb, wb) = Self::integral(tr, cum_f, cum_w, cb);
                // instantaneous rate: Catmull-Rom through the records at their interval centres
                // (linear interpolation cost ~3.5% of the lever term at engine vibration rates)
                let w_at = |t: f64| {
                    let n = tr.len();
                    let j = centre.partition_point(|&c| c < t).clamp(1, n - 1);
                    let u = ((t - centre[j - 1]) / (centre[j] - centre[j - 1]).max(1e-12)).clamp(0.0, 1.0);
                    let (p0, p1, p2, p3) = (tr[j.saturating_sub(2)].w, tr[j - 1].w, tr[j].w, tr[(j + 1).min(n - 1)].w);
                    let (u2, u3) = (u * u, u * u * u);
                    0.5 * (2.0 * p1 + (p2 - p0) * u + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * u3)
                };
                if cb - ca < 1e-9 {
                    let i = tr.partition_point(|r| r.t < ca).min(tr.len() - 1);
                    return (tr[i].f, tr[i].w, w_at(a), w_at(b));
                }
                ((fb - fa) / (cb - ca), (wb - wa) / (cb - ca), w_at(a), w_at(b))
            }
            Truth::Numeric { poses, ell } => {
                let h = (b - a).max(0.01);
                let (f, w) = numeric_truth(poses, ell, 0.5 * (a + b), h);
                (f, w, numeric_truth(poses, ell, a, h).1, numeric_truth(poses, ell, b, h).1)
            }
        }
    }
}

/// Synthesize the IMU over [t0, t1] (trajectory time). Sample k at t is the mean over the
/// centred window [t - dt/2, t + dt/2] (delta-velocity / delta-angle semantics, no delay).
pub fn synthesize(cfg: &ImuConfig, poses: &[Pose], truth: Option<&[ImuTruth]>, ell: &Ellipsoid, t0: f64, t1: f64) -> Result<ImuData> {
    if cfg.rate_hz <= 0.0 {
        bail!("imu.rate_hz must be > 0");
    }
    if poses.len() < 2 || poses[poses.len() - 1].t - poses[0].t < 1e-3 {
        bail!("imu: the trajectory must span at least 1 ms (2 poses)");
    }
    let dt = 1.0 / cfg.rate_hz;
    let n = ((t1 - t0) / dt).floor() as usize + 1;
    let r_bi = cfg.extrinsics.r_body_imu();
    let lever = DVec3::from_array(cfg.extrinsics.translation);
    let mut rng = Rng(cfg.seed ^ 0x1A0);
    let skew = |v: DVec3| DMat3::from_cols(DVec3::new(0.0, v.z, -v.y), DVec3::new(-v.z, 0.0, v.x), DVec3::new(v.y, -v.x, 0.0));
    let mis_g = DMat3::IDENTITY + skew(rng.gvec() * cfg.gyro.misalignment_sigma);
    let mis_a = DMat3::IDENTITY + skew(rng.gvec() * cfg.accel.misalignment_sigma);
    let scale_g = DVec3::ONE + rng.gvec() * cfg.gyro.scale_sigma;
    let scale_a = DVec3::ONE + rng.gvec() * cfg.accel.scale_sigma;
    let mut bg = rng.gvec() * cfg.gyro.bias_init;
    let mut ba = rng.gvec() * cfg.accel.bias_init;
    let src = Truth::new(truth, poses, ell);

    let mut d = ImuData {
        t: Vec::with_capacity(n),
        accel: Vec::with_capacity(n),
        omega: Vec::with_capacity(n),
        gt_accel: Vec::with_capacity(n),
        gt_omega: Vec::with_capacity(n),
        bias_accel: Vec::with_capacity(n),
        bias_gyro: Vec::with_capacity(n),
        from_truth_columns: matches!(src, Truth::Records { .. }),
    };
    let sat = |v: DVec3, s: f64| if s > 0.0 { v.clamp(DVec3::splat(-s), DVec3::splat(s)) } else { v };
    for k in 0..n {
        let t = t0 + k as f64 * dt;
        let (f_b, w_b, w_a, w_e) = src.window(t - 0.5 * dt, t + 0.5 * dt);
        // lever arm (body frame): f_imu = f + ω̇ × r + ω × (ω × r), ω̇ averaged over the window
        let wd = (w_e - w_a) / dt;
        let f_l = f_b + wd.cross(lever) + w_b.cross(w_b.cross(lever));
        // into the IMU frame
        let f_i = r_bi.transpose() * f_l;
        let w_i = r_bi.transpose() * w_b;
        // sensor errors
        bg += rng.gvec() * (cfg.gyro.random_walk * dt.sqrt());
        ba += rng.gvec() * (cfg.accel.random_walk * dt.sqrt());
        let ng = rng.gvec() * (cfg.gyro.noise_density / dt.sqrt());
        let na = rng.gvec() * (cfg.accel.noise_density / dt.sqrt());
        let wm = sat(mis_g * (w_i * scale_g) + bg + ng, cfg.gyro.saturation);
        let am = sat(mis_a * (f_i * scale_a) + ba + na, cfg.accel.saturation);
        d.t.push(t);
        d.accel.push(am);
        d.omega.push(wm);
        d.gt_accel.push(f_i);
        d.gt_omega.push(w_i);
        d.bias_accel.push(ba);
        d.bias_gyro.push(bg);
    }
    Ok(d)
}

/// Write the IMU group into the sequence file. `t0` = trajectory time of the sequence start.
pub fn write_h5(f: &h5::File, cfg: &ImuConfig, d: &ImuData, t0: f64, level: u8) -> Result<()> {
    use crate::output::{fresh_group, to_us, transform_4x4};
    let g = fresh_group(f, &cfg.path)?;
    let n = d.t.len();
    let ts: Vec<i64> = d.t.iter().map(|t| to_us(*t, t0)).collect();
    g.new_dataset::<i64>().shape(&[n]).create("t")?.write_all(&ts)?;
    let put = |name: &str, v: &[DVec3]| -> Result<()> {
        let flat: Vec<f64> = v.iter().flat_map(|x| x.to_array()).collect();
        g.new_dataset::<f64>().shape(&[n, 3]).chunk(&[n.clamp(1, 4096), 3]).deflate(level.min(9)).create(name)?.write_all(&flat)?;
        Ok(())
    };
    put("accel", &d.accel)?;
    put("gyro", &d.omega)?;
    put("gt_accel", &d.gt_accel)?;
    put("gt_gyro", &d.gt_omega)?;
    put("gt_bias_accel", &d.bias_accel)?;
    put("gt_bias_gyro", &d.bias_gyro)?;
    let c = g.ensure_group("calib")?;
    let t = transform_4x4(cfg.extrinsics.r_body_imu(), DVec3::from_array(cfg.extrinsics.translation));
    c.new_dataset::<f64>().shape(&[4, 4]).create("T_body_imu")?.write_all(&t)?;
    c.set_attr("rate_hz", cfg.rate_hz)?;
    c.set_attr("gyroscope_noise_density", cfg.gyro.noise_density)?;
    c.set_attr("gyroscope_random_walk", cfg.gyro.random_walk)?;
    c.set_attr("accelerometer_noise_density", cfg.accel.noise_density)?;
    c.set_attr("accelerometer_random_walk", cfg.accel.random_walk)?;
    c.set_attr_str("imu_yaml", &serde_yaml::to_string(cfg)?)?;
    g.set_attr_str(
        "conventions",
        "t: i64 µs since the sequence start; accel: specific force (m/s²), gyro: angular rate w.r.t. inertial space (rad/s), both in the IMU frame; \
         gt_* = error-free values and the true biases; calib/T_body_imu = row-major 4x4 IMU → body (FRD)",
    )?;
    g.set_attr("truth_from_simulator", d.from_truth_columns as i32)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{simulate, AltitudeRef, SynthConfig};

    #[test]
    fn imu_noise_statistics_and_level_flight() {
        let ell = Ellipsoid::WGS84;
        let sc = SynthConfig { kind: crate::dynamics::PathKind::Line, duration: 60.0, altitude_ref: AltitudeRef::Ellipsoid, altitude: 1000.0, ..Default::default() };
        let mut sc = sc;
        sc.wind.turbulence = 0.0;
        sc.wind.gust_rate_per_min = 0.0;
        sc.vibration.harmonic_deg = 0.0;
        sc.vibration.broadband_deg = 0.0;
        let recs = simulate(&sc, (39.9, 32.8), &ell, None);
        let poses: Vec<Pose> = recs.iter().map(|r| r.pose).collect();
        let truth: Vec<ImuTruth> = recs.iter().map(|r| ImuTruth { t: r.pose.t, f: r.imu_f, w: r.imu_w }).collect();
        let cfg = ImuConfig::default();
        let d = synthesize(&cfg, &poses, Some(&truth), &ell, 1.0, 59.0).unwrap();
        // level unaccelerated flight: |f| ≈ g, gyro ≈ Earth rate
        let g0 = geodesy::normal_gravity(39.9f64.to_radians(), 1000.0);
        let mf = d.gt_accel[100..].iter().fold(DVec3::ZERO, |a, v| a + *v) / (d.t.len() - 100) as f64;
        assert!((mf.length() - g0).abs() < 0.01, "{mf:?}");
        let mw = d.gt_omega[100..].iter().fold(DVec3::ZERO, |a, v| a + *v) / (d.t.len() - 100) as f64;
        assert!((mw.length() - geodesy::EARTH_RATE).abs() < 3e-5, "{mw:?}");
        // white-noise level: std of (measured - truth - bias) ≈ density * sqrt(rate)
        let res: Vec<f64> = (0..d.t.len()).map(|k| (d.omega[k] - d.bias_gyro[k] - d.gt_omega[k]).x).collect();
        let sd = (res.iter().map(|x| x * x).sum::<f64>() / res.len() as f64).sqrt();
        let want = cfg.gyro.noise_density * cfg.rate_hz.sqrt();
        assert!((sd / want - 1.0).abs() < 0.2, "{sd} vs {want}");
        // the numeric fallback agrees with the simulator truth on smooth motion
        let dn = synthesize(&cfg, &poses, None, &ell, 5.0, 10.0).unwrap();
        let k = dn.t.len() / 2;
        assert!((dn.gt_accel[k] - d.gt_accel[(4.0 * cfg.rate_hz) as usize + k]).length() < 0.2);
    }

    #[test]
    fn lever_arm_and_timing() {
        // vibrating flight recorded at 1 kHz; IMU at 200 Hz with a 10 cm lever arm
        let ell = Ellipsoid::WGS84;
        let sc = SynthConfig { kind: crate::dynamics::PathKind::Circle, duration: 6.0, rate: 1000.0, ..Default::default() };
        let recs = simulate(&sc, (39.9, 32.8), &ell, None);
        let poses: Vec<Pose> = recs.iter().map(|r| r.pose).collect();
        let truth: Vec<ImuTruth> = recs.iter().map(|r| ImuTruth { t: r.pose.t, f: r.imu_f, w: r.imu_w }).collect();
        let lever = DVec3::new(0.1, 0.0, 0.05);
        let cfg = ImuConfig { extrinsics: ImuExtrinsics { translation: lever.to_array(), ..Default::default() }, ..Default::default() };
        let d = synthesize(&cfg, &poses, Some(&truth), &ell, 1.0, 5.0).unwrap();
        // reference: lever-corrected force per record (central-difference ω̇), box-averaged
        let n = truth.len();
        let fl: Vec<DVec3> = (0..n)
            .map(|i| {
                let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
                let wd = (truth[b].w - truth[a].w) / (truth[b].t - truth[a].t);
                truth[i].f + wd.cross(lever) + truth[i].w.cross(truth[i].w.cross(lever))
            })
            .collect();
        let (mut e2, mut l2, mut m) = (0.0, 0.0, 0);
        for (k, &t) in d.t.iter().enumerate() {
            // records (each covering the ms before its stamp) weighted by their overlap with
            // the centred window [t - dt/2, t + dt/2]
            let (a, b) = (t - 0.0025, t + 0.0025);
            let (mut r, mut f0, mut ws) = (DVec3::ZERO, DVec3::ZERO, 0.0);
            for i in 1..n {
                let wgt = (truth[i].t.min(b) - truth[i - 1].t.max(a)).max(0.0);
                r += fl[i] * wgt;
                f0 += truth[i].f * wgt;
                ws += wgt;
            }
            let (r, f0) = (r / ws, f0 / ws);
            e2 += (d.gt_accel[k] - r).length_squared();
            l2 += (r - f0).length_squared();
            m += 1;
        }
        let (e, l) = ((e2 / m as f64).sqrt(), (l2 / m as f64).sqrt());
        eprintln!("lever-arm error {e:.3} vs lever term {l:.3} m/s² RMS");
        assert!(e < 0.1 * l, "lever-arm error {e} vs lever term {l}");
        // timing: gyro truth matches the attitude rate at the sample time, not half a sample late
        let rate_at = |t: f64| {
            let (a, b) = (trajectory::interpolate(&poses, t - 0.001), trajectory::interpolate(&poses, t + 0.001));
            let dq = (DQuat::from_mat3(&a.r_ecef_body()).inverse() * DQuat::from_mat3(&b.r_ecef_body())).normalize();
            (if dq.w < 0.0 { -dq } else { dq }).to_scaled_axis() / 0.002
        };
        let err = |lag: f64| -> f64 { (0..d.t.len()).map(|k| (d.gt_omega[k] - rate_at(d.t[k] + lag)).length_squared()).sum::<f64>().sqrt() };
        assert!(err(0.0) < err(0.0025) && err(0.0) < err(-0.0025), "{} {} {}", err(-0.0025), err(0.0), err(0.0025));
    }
}