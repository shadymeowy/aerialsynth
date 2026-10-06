//! Synthetic flight recorder.
//!
//! The nominal flight is a smooth, predictable spline: control points from the path kind (or user
//! waypoints) → centripetal Catmull–Rom spline in a local tangent plane → arc-length
//! parameterization at constant ground speed. Nominal attitude follows from the path kinematics
//! (heading = track + crab angle into the mean crosswind, bank from curvature, pitch from climb).
//!
//! Small local effects are added by tiny ODE models integrated at a high rate (default 1 kHz):
//! * turbulence (Gauss–Markov, Dryden-like) and discrete 1-cosine gusts,
//! * bounded position deviations from the spline (2nd-order tracking loop driven by gusts),
//! * attitude deviations (2nd-order roll / pitch / yaw responses to turbulence),
//! * camera vibration: engine harmonics + broadband airframe resonance,
//! * optional stabilized gimbal (camera isolated from body roll/pitch with a lag).
//! The record is written at `rate` Hz so the renderer can reconstruct sub-frame motion.

use crate::trajectory::Pose;
use geodesy::{Ellipsoid, Geodetic};
use glam::DQuat;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PathKind {
    Line,
    Circle,
    Figure8,
    Lawnmower,
    /// Smooth random meandering through random control points.
    Random,
    /// Spline through `waypoints` in order.
    Waypoints,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AltitudeRef {
    /// Height above the WGS84 ellipsoid.
    Ellipsoid,
    /// Height above the (smoothed) generated terrain.
    Agl,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WindConfig {
    /// Mean wind speed (m/s).
    pub speed: f64,
    /// Direction the wind blows FROM (deg, meteorological).
    pub direction_deg: f64,
    /// Turbulence intensity σ (m/s): ~0.5 light, 1.5 moderate, 3 severe.
    pub turbulence: f64,
    /// Turbulence length scale (m).
    pub length_scale: f64,
    /// Discrete gusts per minute.
    pub gust_rate_per_min: f64,
    /// Gust amplitude (m/s).
    pub gust_amplitude: f64,
    /// Gust length (m).
    pub gust_length: f64,
}

impl Default for WindConfig {
    fn default() -> Self {
        WindConfig {
            speed: 5.0,
            direction_deg: 270.0,
            turbulence: 0.8,
            length_scale: 250.0,
            gust_rate_per_min: 0.6,
            gust_amplitude: 3.0,
            gust_length: 120.0,
        }
    }
}

/// How the vehicle responds to disturbances (all perturbations are small deviations around the
/// nominal spline flight).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResponseConfig {
    pub max_bank_deg: f64,
    /// Nose-up offset of the body relative to the flight path (deg).
    pub trim_aoa_deg: f64,
    /// Crab into the mean crosswind (heading ≠ track), like a real aircraft.
    pub crab: bool,
    /// Position tracking loop (natural frequency Hz, damping) pulling deviations back to the path.
    pub track_hz: f64,
    pub track_damping: f64,
    /// Fraction of the gust velocity that pushes the vehicle off the path (0..1).
    pub gust_follow: f64,
    /// Attitude responses to turbulence: natural frequency (Hz), damping, gain (deg per m/s).
    pub roll_hz: f64,
    pub roll_damping: f64,
    pub roll_gain: f64,
    pub pitch_hz: f64,
    pub pitch_damping: f64,
    pub pitch_gain: f64,
    pub yaw_hz: f64,
    pub yaw_damping: f64,
    pub yaw_gain: f64,
}

impl Default for ResponseConfig {
    fn default() -> Self {
        ResponseConfig {
            max_bank_deg: 35.0,
            trim_aoa_deg: 2.0,
            crab: true,
            track_hz: 0.12,
            track_damping: 0.8,
            gust_follow: 0.6,
            roll_hz: 1.2,
            roll_damping: 0.45,
            roll_gain: 1.5,
            pitch_hz: 1.5,
            pitch_damping: 0.55,
            pitch_gain: 0.8,
            yaw_hz: 0.8,
            yaw_damping: 0.7,
            yaw_gain: 0.5,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VibrationConfig {
    /// RMS angular amplitude of the engine harmonic (deg).
    pub harmonic_deg: f64,
    /// Engine/propeller fundamental (Hz); a 2nd harmonic at half amplitude is added.
    pub harmonic_hz: f64,
    /// Slow random variation of the engine frequency (fraction).
    pub harmonic_jitter: f64,
    /// RMS broadband (airframe) vibration (deg) and its resonance (Hz) / damping.
    pub broadband_deg: f64,
    pub broadband_hz: f64,
    pub broadband_damping: f64,
}

impl Default for VibrationConfig {
    fn default() -> Self {
        VibrationConfig {
            harmonic_deg: 0.04,
            harmonic_hz: 38.0,
            harmonic_jitter: 0.05,
            broadband_deg: 0.03,
            broadband_hz: 12.0,
            broadband_damping: 0.15,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GimbalConfig {
    /// Stabilize the camera against body roll/pitch (camera stays level; heading follows body).
    pub stabilized: bool,
    /// Gimbal response time constant (s).
    pub tau: f64,
    /// Fraction of vibration that passes through the gimbal / mount isolators.
    pub vibration_transmission: f64,
}

impl Default for GimbalConfig {
    fn default() -> Self {
        GimbalConfig { stabilized: false, tau: 0.15, vibration_transmission: 1.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SynthConfig {
    pub kind: PathKind,
    /// Start position (deg). None = the world's home.
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub heading_deg: f64,
    pub altitude: f64,
    pub altitude_ref: AltitudeRef,
    /// True airspeed (m/s).
    pub speed: f64,
    pub duration: f64,
    /// Recording rate (Hz). Keep ≥ ~4x the vibration frequency for faithful blur.
    pub rate: f64,
    /// Integration step (s).
    pub dt: f64,
    /// Turn radius for circle/figure8/lawnmower and the random path's typical turn (m).
    pub radius: f64,
    /// Lawnmower leg length (m).
    pub leg: f64,
    /// [lat, lon] (deg) for `waypoints` (the spline passes through them).
    pub waypoints: Vec<[f64; 2]>,
    pub seed: u64,
    pub wind: WindConfig,
    pub response: ResponseConfig,
    pub vibration: VibrationConfig,
    pub gimbal: GimbalConfig,
}

impl Default for SynthConfig {
    fn default() -> Self {
        SynthConfig {
            kind: PathKind::Random,
            lat: None,
            lon: None,
            heading_deg: 30.0,
            altitude: 800.0,
            altitude_ref: AltitudeRef::Agl,
            speed: 45.0,
            duration: 60.0,
            rate: 200.0,
            dt: 0.001,
            radius: 1200.0,
            leg: 3000.0,
            waypoints: vec![],
            seed: 1,
            wind: WindConfig::default(),
            response: ResponseConfig::default(),
            vibration: VibrationConfig::default(),
            gimbal: GimbalConfig::default(),
        }
    }
}

/// Deterministic PRNG (splitmix64) with Gaussian draws.
struct Rng(u64, Option<f64>);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x5DEE_CE66_D1CE_4E5B, None)
    }
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        if let Some(v) = self.1.take() {
            return v;
        }
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        self.1 = Some(r * s);
        r * c
    }
}

/// First-order Gauss–Markov process x' = -x/τ + σ sqrt(2/τ) η (exact discretization).
#[derive(Default, Clone, Copy)]
struct GaussMarkov {
    x: f64,
}
impl GaussMarkov {
    fn step(&mut self, rng: &mut Rng, tau: f64, sigma: f64, dt: f64) -> f64 {
        let a = (-dt / tau.max(1e-6)).exp();
        self.x = a * self.x + sigma * (1.0 - a * a).sqrt() * rng.gauss();
        self.x
    }
}

/// Damped oscillator driven by white noise, scaled to a target RMS.
#[derive(Default, Clone, Copy)]
struct Oscillator {
    x: f64,
    v: f64,
}
impl Oscillator {
    fn step(&mut self, rng: &mut Rng, f: f64, zeta: f64, rms: f64, dt: f64) -> f64 {
        let w = std::f64::consts::TAU * f;
        // white-noise force intensity giving stationary RMS of x: σx² = q / (4 ζ ω³)
        let q = rms * rms * 4.0 * zeta * w * w * w;
        let force = (q / dt).sqrt() * rng.gauss();
        // semi-implicit Euler (stable for ω dt << 1)
        self.v += (-2.0 * zeta * w * self.v - w * w * self.x + force) * dt;
        self.x += self.v * dt;
        self.x
    }
}

/// A recorded trajectory sample with extra diagnostics.
#[derive(Clone, Copy, Debug)]
pub struct Record {
    pub pose: Pose,
    /// ground velocity NED (m/s)
    pub vel_ned: [f64; 3],
    /// total wind NED (m/s)
    pub wind_ned: [f64; 3],
}

/// Arc-length parameterized planar spline (local tangent plane, meters).
pub struct PathSpline {
    pts: Vec<(f64, f64)>,
    s: Vec<f64>,
}

impl PathSpline {
    /// Centripetal Catmull–Rom through `ctrl`, densely sampled (~`step` m) and arc-length indexed.
    pub fn new(ctrl: &[(f64, f64)], step: f64) -> Self {
        let mut c: Vec<(f64, f64)> = Vec::with_capacity(ctrl.len() + 2);
        // phantom end points continue the end segments
        let (a, b) = (ctrl[0], ctrl[1]);
        c.push((2.0 * a.0 - b.0, 2.0 * a.1 - b.1));
        c.extend_from_slice(ctrl);
        let (y, z) = (ctrl[ctrl.len() - 2], ctrl[ctrl.len() - 1]);
        c.push((2.0 * z.0 - y.0, 2.0 * z.1 - y.1));
        let mut pts = Vec::new();
        for i in 1..c.len() - 2 {
            let (p0, p1, p2, p3) = (c[i - 1], c[i], c[i + 1], c[i + 2]);
            let d = |a: (f64, f64), b: (f64, f64)| ((b.0 - a.0).hypot(b.1 - a.1)).sqrt().max(1e-6);
            let t0 = 0.0;
            let t1 = t0 + d(p0, p1);
            let t2 = t1 + d(p1, p2);
            let t3 = t2 + d(p2, p3);
            let seg = (p2.0 - p1.0).hypot(p2.1 - p1.1);
            let n = ((seg / step).ceil() as usize).max(1);
            for k in 0..n {
                let t = t1 + (t2 - t1) * k as f64 / n as f64;
                let l = |pa: (f64, f64), pb: (f64, f64), ta: f64, tb: f64| {
                    let w = (t - ta) / (tb - ta);
                    (pa.0 + (pb.0 - pa.0) * w, pa.1 + (pb.1 - pa.1) * w)
                };
                let a1 = l(p0, p1, t0, t1);
                let a2 = l(p1, p2, t1, t2);
                let a3 = l(p2, p3, t2, t3);
                let b1 = l(a1, a2, t0, t2);
                let b2 = l(a2, a3, t1, t3);
                pts.push(l(b1, b2, t1, t2));
            }
        }
        pts.push(*ctrl.last().unwrap());
        let mut s = Vec::with_capacity(pts.len());
        let mut acc = 0.0;
        for i in 0..pts.len() {
            if i > 0 {
                acc += (pts[i].0 - pts[i - 1].0).hypot(pts[i].1 - pts[i - 1].1);
            }
            s.push(acc);
        }
        PathSpline { pts, s }
    }

    pub fn length(&self) -> f64 {
        *self.s.last().unwrap()
    }

    /// Position (east, north) at arc length `s` (extrapolates linearly beyond both ends).
    pub fn at(&self, s: f64) -> (f64, f64) {
        let n = self.pts.len();
        if s <= 0.0 {
            let (a, b) = (self.pts[0], self.pts[1]);
            let l = self.s[1].max(1e-9);
            let f = s / l;
            return (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f);
        }
        if s >= self.s[n - 1] {
            let (a, b) = (self.pts[n - 2], self.pts[n - 1]);
            let l = (self.s[n - 1] - self.s[n - 2]).max(1e-9);
            let f = (s - self.s[n - 2]) / l;
            return (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f);
        }
        let i = self.s.partition_point(|v| *v <= s).clamp(1, n - 1);
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let l = (self.s[i] - self.s[i - 1]).max(1e-9);
        let f = ((s - self.s[i - 1]) / l).clamp(0.0, 1.0);
        (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f)
    }

    /// Track angle (rad, clockwise from north) and signed curvature (1/m, + = right turn).
    pub fn track(&self, s: f64, ds: f64) -> (f64, f64) {
        let h = |s: f64| {
            let a = self.at(s - ds);
            let b = self.at(s + ds);
            (b.0 - a.0).atan2(b.1 - a.1)
        };
        let psi = h(s);
        let mut d = h(s + ds) - h(s - ds);
        while d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        }
        while d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        (psi, d / (2.0 * ds))
    }
}

/// Control points of the nominal path in the local plane (east, north), meters.
fn control_points(cfg: &SynthConfig, length: f64, ell: &Ellipsoid, origin: Geodetic, rng: &mut Rng) -> Vec<(f64, f64)> {
    let hd = cfg.heading_deg.to_radians();
    let dir = (hd.sin(), hd.cos());
    let right = (hd.cos(), -hd.sin());
    let r = cfg.radius.max(50.0);
    let mut c = Vec::new();
    match cfg.kind {
        PathKind::Line => {
            let n = ((length / 1000.0).ceil() as usize).max(2);
            for k in 0..=n {
                let d = length * 1.05 * k as f64 / n as f64;
                c.push((dir.0 * d, dir.1 * d));
            }
        }
        PathKind::Circle => {
            let center = (right.0 * r, right.1 * r);
            let turns = length / (std::f64::consts::TAU * r) + 0.2;
            let n = (turns * 24.0).ceil() as usize + 2;
            let a0 = (-right.0).atan2(-right.1); // angle of the start point seen from the centre
            for k in 0..=n {
                let a = a0 + std::f64::consts::TAU * k as f64 / 24.0;
                c.push((center.0 + r * a.sin(), center.1 + r * a.cos()));
            }
        }
        PathKind::Figure8 => {
            // lemniscate of Gerono scaled to ~4 r long, start at the crossing point heading `hd`
            let n_per = 48;
            let per = 8.6 * r;
            let loops = (length / per).ceil() as usize + 1;
            for k in 0..=(loops * n_per) {
                let t = std::f64::consts::TAU * k as f64 / n_per as f64;
                let x = 2.0 * r * t.sin(); // along heading
                let y = 2.0 * r * t.sin() * t.cos(); // across
                c.push((dir.0 * x + right.0 * y, dir.1 * x + right.1 * y));
            }
        }
        PathKind::Lawnmower => {
            let leg = cfg.leg.max(2.0 * r);
            let mut pos = (0.0, 0.0);
            let mut sign = 1.0;
            let mut total = 0.0;
            while total < length * 1.1 {
                // straight leg
                for k in 0..=4 {
                    let d = leg * k as f64 / 4.0;
                    c.push((pos.0 + dir.0 * sign * d, pos.1 + dir.1 * sign * d));
                }
                pos = (pos.0 + dir.0 * sign * leg, pos.1 + dir.1 * sign * leg);
                // semicircle turn to the right (alternating legs stack rightwards)
                let center = (pos.0 + right.0 * r, pos.1 + right.1 * r);
                for k in 1..12 {
                    let a = std::f64::consts::PI * k as f64 / 12.0;
                    let (ca, sa) = (a.cos(), a.sin());
                    // rotate from -right towards +right through the forward direction
                    let v = (-right.0 * ca + dir.0 * sign * sa, -right.1 * ca + dir.1 * sign * sa);
                    c.push((center.0 + r * v.0, center.1 + r * v.1));
                }
                pos = (pos.0 + right.0 * 2.0 * r, pos.1 + right.1 * 2.0 * r);
                sign = -sign;
                total += leg + std::f64::consts::PI * r;
            }
        }
        PathKind::Random => {
            let seg = r * 0.7;
            let mut p = (0.0, 0.0);
            let mut h = hd;
            let mut turn = 0.0;
            c.push(p);
            let mut total = 0.0;
            while total < length * 1.1 {
                // smooth random turning: turn rate is itself a random walk (bounded)
                turn = (0.85 * turn + 0.5 * rng.gauss() * seg / r).clamp(-1.2 * seg / r, 1.2 * seg / r);
                h += turn;
                p = (p.0 + seg * h.sin(), p.1 + seg * h.cos());
                c.push(p);
                total += seg;
            }
        }
        PathKind::Waypoints => {
            c.push((0.0, 0.0));
            for w in &cfg.waypoints {
                let enu = geodesy::geodetic2enu(Geodetic::from_deg(w[0], w[1], 0.0), origin, ell);
                c.push((enu.x, enu.y));
            }
            if c.len() < 2 {
                c.push((dir.0 * length, dir.1 * length));
            }
        }
    }
    // drop consecutive duplicates
    c.dedup_by(|a, b| (a.0 - b.0).hypot(a.1 - b.1) < 1e-3);
    if c.len() < 2 {
        c.push((dir.0 * length, dir.1 * length));
    }
    c
}

/// Run the flight recorder. `ground(lat, lon)` (radians) gives terrain height for AGL mode.
pub fn simulate(cfg: &SynthConfig, home: (f64, f64), ell: &Ellipsoid, ground: Option<&(dyn Fn(f64, f64) -> f64 + Sync)>) -> Vec<Record> {
    let g = 9.80665;
    let dt = cfg.dt.clamp(1e-4, 0.01);
    let v = cfg.speed.max(1.0); // ground speed along the path
    let rc = &cfg.response;
    let mut rng = Rng::new(cfg.seed);
    let origin = Geodetic::from_deg(cfg.lat.unwrap_or(home.0), cfg.lon.unwrap_or(home.1), 0.0);
    let length = v * cfg.duration + 10.0;
    let ctrl = control_points(cfg, length, ell, origin, &mut rng);
    let path = PathSpline::new(&ctrl, 2.0);
    let to_geo = |e: f64, n: f64| geodesy::enu2geodetic(glam::DVec3::new(e, n, 0.0), origin, ell);

    // ---- altitude profile along the arc length
    let ds_h = 50.0;
    let nh = (length / ds_h).ceil() as usize + 2;
    let h_prof: Vec<f64> = match (cfg.altitude_ref, ground) {
        (AltitudeRef::Agl, Some(gf)) => {
            let raw: Vec<f64> = (0..nh)
                .map(|k| {
                    let (e, n) = path.at(k as f64 * ds_h);
                    let geo = to_geo(e, n);
                    gf(geo.lat, geo.lon)
                })
                .collect();
            // clearance envelope (running max over ±1 km) then smoothing (±2 km)
            let w1 = (1000.0 / ds_h) as isize;
            let w2 = (2000.0 / ds_h) as isize;
            let n = raw.len() as isize;
            let env: Vec<f64> = (0..n).map(|i| ((i - w1).max(0)..=(i + w1).min(n - 1)).map(|j| raw[j as usize]).fold(f64::MIN, f64::max)).collect();
            (0..n)
                .map(|i| {
                    let r = (i - w2).max(0)..=(i + w2).min(n - 1);
                    let len = r.clone().count() as f64;
                    r.map(|j| env[j as usize]).sum::<f64>() / len + cfg.altitude
                })
                .collect()
        }
        _ => vec![cfg.altitude; nh],
    };
    let alt_at = |s: f64| -> (f64, f64) {
        let x = (s / ds_h).clamp(0.0, (nh - 1) as f64);
        let i = (x.floor() as usize).min(nh - 2);
        let f = x - i as f64;
        let h = h_prof[i] + (h_prof[i + 1] - h_prof[i]) * f;
        (h, (h_prof[i + 1] - h_prof[i]) / ds_h)
    };

    // ---- disturbance states
    let wdir = cfg.wind.direction_deg.to_radians();
    let wmean = (-cfg.wind.speed * wdir.sin(), -cfg.wind.speed * wdir.cos()); // (east, north) blowing TO
    let mut turb = [GaussMarkov::default(); 3];
    let mut turb_att = [GaussMarkov::default(); 3];
    let mut gust: Option<(f64, f64, [f64; 3])> = None;
    // position deviation (cross, vertical, along) and rates
    let mut dev = [0.0f64; 3];
    let mut dev_d = [0.0f64; 3];
    // attitude (roll, pitch) following commands + turbulence, yaw deviation
    let max_bank = rc.max_bank_deg.to_radians();
    let mut att = [0.0f64; 3];
    let mut att_d = [0.0f64; 3];
    let (psi0, kappa0) = path.track(0.0, 15.0);
    att[0] = (v * v * kappa0 / g).atan().clamp(-max_bank, max_bank);
    att[1] = rc.trim_aoa_deg.to_radians();
    let _ = psi0;

    let vib = &cfg.vibration;
    let mut harm_phase = [rng.uniform() * 6.28, rng.uniform() * 6.28, rng.uniform() * 6.28];
    let harm_amp = [rng.uniform() * 0.5 + 0.75, rng.uniform() * 0.5 + 0.75, rng.uniform() * 0.5 + 0.25];
    let mut harm_f_jit = GaussMarkov::default();
    let mut bb = [Oscillator::default(); 3];
    let mut gimbal_rp = [att[0], att[1]];

    let mut out = Vec::new();
    let n_steps = (cfg.duration / dt).round() as usize;
    let rec_every = (1.0 / (cfg.rate * dt)).round().max(1.0) as usize;
    let tau_t = cfg.wind.length_scale / v;
    let sig = cfg.wind.turbulence;
    for step in 0..=n_steps {
        let t = step as f64 * dt;
        let s = v * t;

        // ---------------- disturbances
        let tu = turb[0].step(&mut rng, tau_t, sig, dt);
        let tv = turb[1].step(&mut rng, tau_t, sig, dt);
        let tw = turb[2].step(&mut rng, tau_t * 0.5, sig * 0.8, dt);
        let ta = [
            turb_att[0].step(&mut rng, tau_t * 0.25, sig, dt),
            turb_att[1].step(&mut rng, tau_t * 0.3, sig * 0.8, dt),
            turb_att[2].step(&mut rng, tau_t * 0.4, sig * 0.6, dt),
        ];
        // gust components in path axes (along, cross, up)
        let mut gv = [tu, tv, tw];
        if gust.is_none() && rng.uniform() < cfg.wind.gust_rate_per_min / 60.0 * dt {
            let a = rng.uniform() * std::f64::consts::TAU;
            let amp = cfg.wind.gust_amplitude * (0.5 + rng.uniform());
            gust = Some((s, cfg.wind.gust_length * (0.5 + rng.uniform()), [amp * a.cos(), amp * a.sin(), 0.5 * amp * rng.gauss()]));
        }
        if let Some((s0, len, a)) = gust {
            let x = (s - s0) / len;
            if x > 1.0 {
                gust = None;
            } else {
                let f = 0.5 * (1.0 - (std::f64::consts::TAU * x).cos());
                for k in 0..3 {
                    gv[k] += a[k] * f;
                }
            }
        }

        // ---------------- position deviation: δ'' = -ω² δ - 2ζω (δ' - k·gust)
        let w = std::f64::consts::TAU * rc.track_hz;
        let gin = [gv[1], gv[2], gv[0]]; // cross, vertical, along
        for k in 0..3 {
            let acc = -w * w * dev[k] - 2.0 * rc.track_damping * w * (dev_d[k] - rc.gust_follow * gin[k]);
            dev_d[k] += acc * dt;
            dev[k] += dev_d[k] * dt;
        }

        // ---------------- nominal kinematics at the (deviated) arc length
        let s_eff = s + dev[2];
        let (track, kappa) = path.track(s_eff, 15.0);
        let (h_nom, dhds) = alt_at(s_eff);
        let bank_cmd = (v * v * kappa / g).atan().clamp(-max_bank, max_bank);
        let gamma = (dhds + dev_d[1] / v).atan();
        let pitch_cmd = gamma + rc.trim_aoa_deg.to_radians();
        // heading: crab into the crosswind (air velocity = ground velocity - wind)
        let (st, ct) = track.sin_cos();
        let heading = if rc.crab { (v * st - wmean.0 - gv[1] * ct).atan2(v * ct - wmean.1 + gv[1] * st) } else { track };

        // ---------------- attitude responses (2nd order towards command + turbulence)
        let cmds = [bank_cmd, pitch_cmd, 0.0];
        let hz = [rc.roll_hz, rc.pitch_hz, rc.yaw_hz];
        let zeta = [rc.roll_damping, rc.pitch_damping, rc.yaw_damping];
        let gain = [rc.roll_gain, rc.pitch_gain, rc.yaw_gain];
        for k in 0..3 {
            let wn = std::f64::consts::TAU * hz[k];
            let tgt = cmds[k] + gain[k].to_radians() * ta[k];
            let acc = wn * wn * (tgt - att[k]) - 2.0 * zeta[k] * wn * att_d[k];
            att_d[k] += acc * dt;
            att[k] += att_d[k] * dt;
        }

        // ---------------- vibration
        let fj = 1.0 + vib.harmonic_jitter * harm_f_jit.step(&mut rng, 2.0, 1.0, dt);
        let mut vib_a = [0.0f64; 3];
        for k in 0..3 {
            harm_phase[k] += std::f64::consts::TAU * vib.harmonic_hz * fj * dt;
            let a = vib.harmonic_deg.to_radians() * std::f64::consts::SQRT_2 * harm_amp[k];
            vib_a[k] = a * (harm_phase[k].sin() + 0.5 * (2.0 * harm_phase[k] + 1.0).sin()) / 1.118;
            vib_a[k] += bb[k].step(&mut rng, vib.broadband_hz * (0.8 + 0.2 * k as f64), vib.broadband_damping, vib.broadband_deg.to_radians(), dt);
        }

        if step % rec_every != 0 {
            // gimbal state still has to advance
            if cfg.gimbal.stabilized {
                let a = (dt / cfg.gimbal.tau.max(1e-3)).min(1.0);
                gimbal_rp[0] += (att[0] - gimbal_rp[0]) * a;
                gimbal_rp[1] += (att[1] - gimbal_rp[1]) * a;
            }
            continue;
        }
        let (mut r_out, mut p_out) = (att[0], att[1]);
        if cfg.gimbal.stabilized {
            let a = (dt / cfg.gimbal.tau.max(1e-3)).min(1.0);
            gimbal_rp[0] += (att[0] - gimbal_rp[0]) * a;
            gimbal_rp[1] += (att[1] - gimbal_rp[1]) * a;
            r_out = att[0] - gimbal_rp[0];
            p_out = att[1] - gimbal_rp[1];
        }
        let tr = cfg.gimbal.vibration_transmission;
        let q = geodesy::euler_zyx_to_quat(heading + att[2] + tr * vib_a[2], p_out + tr * vib_a[1], r_out + tr * vib_a[0]);

        // ---------------- position: path + cross / vertical deviation
        let (e, n) = path.at(s_eff);
        let (ce, cn) = (ct, -st); // right-hand normal of the track (east, north)
        let geo = to_geo(e + ce * dev[0], n + cn * dev[0]);
        let h = h_nom + dev[1];
        let vel = [v * ct + dev_d[0] * cn, v * st + dev_d[0] * ce, -(v * dhds + dev_d[1])];
        out.push(Record {
            pose: Pose { t, geo: Geodetic::new(geo.lat, geo.lon, h), q_ned_body: q },
            vel_ned: vel,
            wind_ned: [wmean.1 + gv[0] * ct - gv[1] * st, wmean.0 + gv[0] * st + gv[1] * ct, -gv[2]],
        });
    }
    out
}

/// Save records as a trajectory CSV (standard columns + diagnostics).
pub fn save_records(path: &std::path::Path, recs: &[Record], header_note: &str) -> anyhow::Result<()> {
    use std::io::Write;
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "# synthetic flight record: body FRD; q = body->NED (Hamilton); lat/lon deg; h m above WGS84 ellipsoid")?;
    for l in header_note.lines() {
        writeln!(f, "# {l}")?;
    }
    writeln!(f, "t,lat,lon,h,qw,qx,qy,qz,roll,pitch,yaw,vn,ve,vd,wind_n,wind_e,wind_d")?;
    for r in recs {
        let p = &r.pose;
        let q: DQuat = p.q_ned_body;
        let (y, pi, ro) = geodesy::quat_to_euler_zyx(q);
        writeln!(
            f,
            "{:.5},{:.10},{:.10},{:.4},{:.9},{:.9},{:.9},{:.9},{:.4},{:.4},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            p.t,
            p.geo.lat.to_degrees(),
            p.geo.lon.to_degrees(),
            p.geo.h,
            q.w,
            q.x,
            q.y,
            q.z,
            ro.to_degrees(),
            pi.to_degrees(),
            y.to_degrees(),
            r.vel_ned[0],
            r.vel_ned[1],
            r.vel_ned[2],
            r.wind_ned[0],
            r.wind_ned[1],
            r.wind_ned[2]
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flight_is_smooth_and_holds_altitude() {
        let cfg = SynthConfig { duration: 40.0, altitude_ref: AltitudeRef::Ellipsoid, altitude: 1000.0, ..Default::default() };
        for kind in [PathKind::Line, PathKind::Circle, PathKind::Figure8, PathKind::Lawnmower] {
            let c = SynthConfig { kind, ..cfg.clone() };
            let r = simulate(&c, (39.9, 32.8), &Ellipsoid::WGS84, None);
            assert_eq!(r.len(), 40 * 200 + 1);
        }
        let ell = Ellipsoid::WGS84;
        let recs = simulate(&cfg, (39.9, 32.8), &ell, None);
        assert_eq!(recs.len(), 40 * 200 + 1);
        let hs: Vec<f64> = recs.iter().map(|r| r.pose.geo.h).collect();
        let max_dev = hs.iter().map(|h| (h - 1000.0).abs()).fold(0.0, f64::max);
        assert!(max_dev < 15.0, "altitude deviation {max_dev}");
        // speed over ground roughly airspeed ± wind
        let d = (recs[200].pose.ecef(&ell) - recs[0].pose.ecef(&ell)).length();
        assert!(d > 40.0 && d < 50.0, "{d}");
        // vibration exists but is small: consecutive-sample attitude change bounded
        let mut maxd: f64 = 0.0;
        for w in recs.windows(2) {
            maxd = maxd.max(w[0].pose.q_ned_body.angle_between(w[1].pose.q_ned_body));
        }
        assert!(maxd > 1e-5 && maxd < 0.01, "{maxd}");
    }
}
