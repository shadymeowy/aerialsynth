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
//!
//! The record is written at `rate` Hz so the renderer can reconstruct sub-frame motion.

use crate::trajectory::Pose;
use geodesy::{Ellipsoid, Geodetic};
use glam::{DMat3, DQuat, DVec3};
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
    #[serde(rename = "speed_mps")]
    pub speed: f64,
    /// Direction the wind blows FROM (deg, meteorological).
    pub direction_deg: f64,
    /// Turbulence intensity σ (m/s): ~0.5 light, 1.5 moderate, 3 severe.
    pub turbulence: f64,
    /// Turbulence length scale (m).
    #[serde(rename = "length_scale_m")]
    pub length_scale: f64,
    /// Discrete gusts per minute.
    pub gust_rate_per_min: f64,
    /// Gust amplitude (m/s).
    #[serde(rename = "gust_amplitude_mps")]
    pub gust_amplitude: f64,
    /// Gust length (m).
    #[serde(rename = "gust_length_m")]
    pub gust_length: f64,
}

impl Default for WindConfig {
    fn default() -> Self {
        WindConfig { speed: 5.0, direction_deg: 270.0, turbulence: 0.8, length_scale: 250.0, gust_rate_per_min: 0.6, gust_amplitude: 3.0, gust_length: 120.0 }
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
        VibrationConfig { harmonic_deg: 0.04, harmonic_hz: 38.0, harmonic_jitter: 0.05, broadband_deg: 0.03, broadband_hz: 12.0, broadband_damping: 0.15 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GimbalConfig {
    /// Stabilize against body roll / pitch (heading follows the body). The recorded pose is
    /// then that of the stabilized platform: every sensor (cameras and IMU) is assumed to sit
    /// on the gimbal, and /pose describes the gimbal frame, not the airframe.
    pub stabilized: bool,
    /// Gimbal response time constant (s).
    #[serde(rename = "tau_s")]
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
    #[serde(rename = "altitude_m")]
    pub altitude: f64,
    pub altitude_ref: AltitudeRef,
    /// Climb (m/s, negative: descent): the altitude target rises along the path at this rate
    /// (pitch follows from the climb).
    #[serde(rename = "climb_rate_mps")]
    pub climb_rate: f64,
    /// True airspeed (m/s).
    #[serde(rename = "speed_mps")]
    pub speed: f64,
    #[serde(rename = "duration_s")]
    pub duration: f64,
    /// Recording rate (Hz). Keep ≥ ~4x the vibration frequency for faithful blur. Records are
    /// written every round(1 / (rate·dt)) integration steps, so the effective rate is
    /// 1 / (k·dt) (e.g. 300 Hz with dt = 1 ms becomes 333 Hz).
    #[serde(rename = "rate_hz")]
    pub rate: f64,
    /// Integration step (s).
    #[serde(rename = "dt_s")]
    pub dt: f64,
    /// Turn radius for circle/figure8/lawnmower and the random path's typical turn (m); a
    /// negative radius makes the circle turn left.
    #[serde(rename = "radius_m")]
    pub radius: f64,
    /// Lawnmower leg length (m).
    #[serde(rename = "leg_m")]
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
            climb_rate: 0.0,
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

/// Damped oscillator driven by band-limited noise (Gauss-Markov force with a corner at 8x the
/// resonance), scaled to a target RMS. A white force would make the angular acceleration white
/// at the integration rate, which no physical airframe does (and which would make the IMU
/// lever-arm term meaningless).
#[derive(Default, Clone, Copy)]
struct Oscillator {
    x: f64,
    v: f64,
    force: f64,
}
impl Oscillator {
    fn step(&mut self, rng: &mut Rng, f: f64, zeta: f64, rms: f64, dt: f64) -> f64 {
        let w = std::f64::consts::TAU * f;
        // white-force intensity giving stationary RMS of x: σx² = q / (4 ζ ω³); the Gauss-Markov
        // force with variance q / (2τ) has the same spectral level below its corner
        let q = rms * rms * 4.0 * zeta * w * w * w;
        let tau = 1.0 / (8.0 * w);
        let a = (-dt / tau).exp();
        self.force = a * self.force + (q / (2.0 * tau) * (1.0 - a * a)).sqrt() * rng.gauss();
        // semi-implicit Euler (stable for ω dt << 1)
        self.v += (-2.0 * zeta * w * self.v - w * w * self.x + self.force) * dt;
        self.x += self.v * dt;
        self.x
    }
}

/// A recorded trajectory sample with extra diagnostics.
#[derive(Clone, Copy, Debug)]
pub struct Record {
    pub pose: Pose,
    /// True IMU output in the body frame (FRD), averaged over the record interval:
    /// specific force (m/s²) and angular rate w.r.t. inertial space (rad/s).
    pub imu_f: DVec3,
    pub imu_w: DVec3,
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
        // cubic Hermite between the dense samples (tangents from the neighbours): C1, so the
        // turn is spread along the path instead of concentrated at polyline vertices (which
        // made the positions disagree with the IMU's v²κ by spikes of up to 15 m/s²)
        let i = self.s.partition_point(|v| *v <= s).clamp(1, n - 1);
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let l = (self.s[i] - self.s[i - 1]).max(1e-9);
        let u = ((s - self.s[i - 1]) / l).clamp(0.0, 1.0);
        let tangent = |k: usize| {
            let (k0, k1) = (k.saturating_sub(1), (k + 1).min(n - 1));
            let d = (self.s[k1] - self.s[k0]).max(1e-9);
            ((self.pts[k1].0 - self.pts[k0].0) / d, (self.pts[k1].1 - self.pts[k0].1) / d)
        };
        let (ta, tb) = (tangent(i - 1), tangent(i));
        let (u2, u3) = (u * u, u * u * u);
        let (h00, h10, h01, h11) = (2.0 * u3 - 3.0 * u2 + 1.0, u3 - 2.0 * u2 + u, -2.0 * u3 + 3.0 * u2, u3 - u2);
        (h00 * a.0 + h10 * l * ta.0 + h01 * b.0 + h11 * l * tb.0, h00 * a.1 + h10 * l * ta.1 + h01 * b.1 + h11 * l * tb.1)
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
            // a negative radius turns left (centre to the left of the heading)
            let side = if cfg.radius < 0.0 { -1.0 } else { 1.0 };
            let r = cfg.radius.abs().max(50.0);
            let center = (right.0 * r * side, right.1 * r * side);
            let turns = length / (std::f64::consts::TAU * r) + 0.2;
            let n = (turns * 24.0).ceil() as usize + 2;
            let a0 = (-right.0 * side).atan2(-right.1 * side); // angle of the start point seen from the centre
            for k in 0..=n {
                let a = a0 + side * std::f64::consts::TAU * k as f64 / 24.0;
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

/// Run the flight recorder. `ground(points)` gives the terrain height at (lat, lon) points
/// (radians) for AGL mode.
#[allow(clippy::type_complexity)]
pub fn simulate(cfg: &SynthConfig, home: (f64, f64), ell: &Ellipsoid, ground: Option<&(dyn Fn(&[(f64, f64)]) -> Vec<f64> + Sync)>) -> Vec<Record> {
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
            let pts: Vec<(f64, f64)> = (0..nh)
                .map(|k| {
                    let (e, n) = path.at(k as f64 * ds_h);
                    let geo = to_geo(e, n);
                    (geo.lat, geo.lon)
                })
                .collect();
            let raw = gf(&pts);
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
    let climb = cfg.climb_rate / v.max(1e-3); // m per m of path
    let h_prof: Vec<f64> = h_prof.iter().enumerate().map(|(k, h)| h + climb * k as f64 * ds_h).collect();
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
    let mut harm_phase = [rng.uniform() * std::f64::consts::TAU, rng.uniform() * std::f64::consts::TAU, rng.uniform() * std::f64::consts::TAU];
    let harm_amp = [rng.uniform() * 0.5 + 0.75, rng.uniform() * 0.5 + 0.75, rng.uniform() * 0.5 + 0.25];
    let mut harm_f_jit = GaussMarkov::default();
    let mut bb = [Oscillator::default(); 3];
    let mut gimbal_rp = [att[0], att[1]];

    let mut out = Vec::new();
    let mut steps: Vec<(DVec3, f64, f64, f64, DQuat)> = Vec::new();
    let mut rec_steps: Vec<usize> = Vec::new();
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
        let (track, kappa) = path.track(s_eff, 3.0);
        let (h_nom, dhds) = alt_at(s_eff);
        let bank_cmd = (v * v * kappa / g).atan().clamp(-max_bank, max_bank);
        let gamma = (dhds + dev_d[1] / v).atan();
        let pitch_cmd = gamma + rc.trim_aoa_deg.to_radians();
        // The path lives in the tangent plane at the origin; the local north at the aircraft is
        // turned against the plane's north (meridian convergence, ~0.4° after 50 km east-west at
        // 40°N). Track, heading, velocity and acceleration are expressed in the local NED frame:
        // measure the plane direction of travel in local NED through the plane → geodetic map.
        let (e, n) = path.at(s_eff);
        let (stp, ctp) = track.sin_cos();
        let conv = {
            let (g0, g1) = (to_geo(e, n), to_geo(e + stp, n + ctp));
            let d = geodesy::geodetic2ned(Geodetic::new(g1.lat, g1.lon, 0.0), Geodetic::new(g0.lat, g0.lon, 0.0), ell);
            (d.y.atan2(d.x) - track + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
        };
        let track = track + conv;
        // heading: crab into the crosswind (air velocity = ground velocity - wind)
        let (st, ct) = track.sin_cos();
        // the crab angle is the yaw command (the airframe weathervanes through its yaw dynamics;
        // feeding it straight into the heading would make the yaw rate white noise)
        let crab = if rc.crab {
            let hd = (v * st - wmean.0 - gv[1] * ct).atan2(v * ct - wmean.1 + gv[1] * st);
            (hd - track + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
        } else {
            0.0
        };
        let heading = track;

        // ---------------- attitude responses (2nd order towards command + turbulence)
        let cmds = [bank_cmd, pitch_cmd, crab];
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

        // ---------------- position: path + cross / vertical deviation (in the plane)
        let geo = to_geo(e + ctp * dev[0], n - stp * dev[0]);
        let (ce, cn) = (ct, -st); // right-hand normal of the track (east, north), local
        let h = h_nom + dev[1];
        // along-track speed includes the along deviation rate (s_eff = s + dev[2])
        let va = v + dev_d[2];
        let vel = [va * ct + dev_d[0] * cn, va * st + dev_d[0] * ce, -(va * dhds + dev_d[1])];

        // per-step state for the IMU truth (computed from the actual positions after the loop)
        steps.push((geodesy::geodetic2ecef(Geodetic::new(geo.lat, geo.lon, h), ell), geo.lat, geo.lon, h, q));

        if step % rec_every != 0 {
            continue;
        }
        rec_steps.push(steps.len() - 1);
        out.push(Record {
            pose: Pose { t, geo: Geodetic::new(geo.lat, geo.lon, h), q_ned_body: q },
            imu_f: DVec3::ZERO,
            imu_w: DVec3::ZERO,
            vel_ned: vel,
            wind_ned: [wmean.1 + gv[0] * ct - gv[1] * st, wmean.0 + gv[0] * st + gv[1] * ct, -gv[2]],
        });
    }
    imu_truth(&steps, &rec_steps, dt, &mut out);
    out
}

/// IMU truth from the simulated 1 kHz states, consistent with the recorded poses by
/// construction: specific force from the second difference of the ECEF position + Coriolis −
/// WGS84 normal gravity, angular rate from the ECEF attitude difference + Earth rate; both as
/// means over each integration step, averaged over the steps of each record interval.
fn imu_truth(steps: &[(DVec3, f64, f64, f64, DQuat)], rec_steps: &[usize], dt: f64, out: &mut [Record]) {
    let n = steps.len();
    if n < 3 {
        return;
    }
    let om = DVec3::new(0.0, 0.0, geodesy::EARTH_RATE);
    let r_eb = |i: usize| {
        let (_, lat, lon, _, q) = steps[i];
        geodesy::rot_ecef2ned(lat, lon).transpose() * DMat3::from_quat(q)
    };
    // instantaneous specific force (body frame) at the interior steps
    let mut f_inst = vec![DVec3::ZERO; n];
    for i in 1..n - 1 {
        let (p0, p1, p2) = (steps[i - 1].0, steps[i].0, steps[i + 1].0);
        let a = (p2 - 2.0 * p1 + p0) / (dt * dt);
        let v = (p2 - p0) / (2.0 * dt);
        let (_, lat, lon, h, _) = steps[i];
        let g = geodesy::up_vector(lat, lon) * -geodesy::normal_gravity(lat, h);
        f_inst[i] = r_eb(i).transpose() * (a + 2.0 * om.cross(v) - g);
    }
    f_inst[0] = f_inst[1];
    f_inst[n - 1] = f_inst[n - 2];
    // step-interval means (i-1, i]
    let mut f_int = vec![DVec3::ZERO; n];
    let mut w_int = vec![DVec3::ZERO; n];
    let mut r_prev = r_eb(0);
    for i in 1..n {
        let r = r_eb(i);
        f_int[i] = 0.5 * (f_inst[i - 1] + f_inst[i]);
        let dq = DQuat::from_mat3(&(r_prev.transpose() * r)).normalize();
        let dq = if dq.w < 0.0 { -dq } else { dq };
        w_int[i] = dq.to_scaled_axis() / dt + r.transpose() * om;
        r_prev = r;
    }
    f_int[0] = f_int[1];
    w_int[0] = w_int[1];
    // records: mean over the steps since the previous record (the first record: its own step)
    let mut prev = 0usize;
    for (rec, &j) in out.iter_mut().zip(rec_steps) {
        let lo = if j == 0 { 0 } else { prev + 1 };
        let k = (j + 1 - lo) as f64;
        rec.imu_f = f_int[lo..=j].iter().fold(DVec3::ZERO, |a, v| a + *v) / k;
        rec.imu_w = w_int[lo..=j].iter().fold(DVec3::ZERO, |a, v| a + *v) / k;
        prev = j;
    }
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
    writeln!(f, "# f_*: true specific force, w_*: true angular rate (inertial), body FRD axes, averaged over the sample interval")?;
    writeln!(f, "t,lat,lon,h,qw,qx,qy,qz,roll,pitch,yaw,vn,ve,vd,wind_n,wind_e,wind_d,f_x,f_y,f_z,w_x,w_y,w_z")?;
    for r in recs {
        let p = &r.pose;
        let q: DQuat = p.q_ned_body;
        let (y, pi, ro) = geodesy::quat_to_euler_zyx(q);
        writeln!(
            f,
            "{:.6},{:.14},{:.14},{:.9},{:.12},{:.12},{:.12},{:.12},{:.4},{:.4},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.7},{:.7},{:.7},{:.9},{:.9},{:.9}",
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
            r.wind_ned[2],
            r.imu_f.x,
            r.imu_f.y,
            r.imu_f.z,
            r.imu_w.x,
            r.imu_w.y,
            r.imu_w.z
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heading and velocity are in the local NED frame along a long east-west line (the path
    /// lives in the origin's tangent plane; local north turns by the meridian convergence).
    #[test]
    fn local_frame_heading_far_from_origin() {
        let ell = Ellipsoid::WGS84;
        let mut cfg = SynthConfig {
            kind: PathKind::Line,
            heading_deg: 90.0,
            speed: 100.0,
            duration: 500.0,
            rate: 10.0,
            altitude_ref: AltitudeRef::Ellipsoid,
            altitude: 1000.0,
            ..Default::default()
        };
        cfg.wind.speed = 0.0;
        cfg.wind.turbulence = 0.0;
        cfg.wind.gust_rate_per_min = 0.0;
        cfg.vibration.harmonic_deg = 0.0;
        cfg.vibration.broadband_deg = 0.0;
        let recs = simulate(&cfg, (39.9, 32.8), &ell, None);
        let (mut max_yaw, mut max_vel, mut conv): (f64, f64, f64) = (0.0, 0.0, 0.0);
        for w in recs[10..].windows(3).step_by(50) {
            // track from positions, in the local NED frame of the middle record
            let d = geodesy::geodetic2ned(w[2].pose.geo, w[0].pose.geo, &ell);
            let track = d.y.atan2(d.x);
            let (yaw, _, _) = geodesy::quat_to_euler_zyx(w[1].pose.q_ned_body);
            let v = w[1].vel_ned;
            max_yaw = max_yaw.max((yaw - track).abs());
            max_vel = max_vel.max((v[1].atan2(v[0]) - track).abs());
            conv = conv.max((track - std::f64::consts::FRAC_PI_2).abs());
        }
        // the local track turns by ~0.4° over 50 km; heading and velocity follow it
        assert!(conv.to_degrees() > 0.2, "{}", conv.to_degrees());
        assert!(max_yaw.to_degrees() < 0.01 && max_vel.to_degrees() < 0.01, "yaw {} vel {} (deg)", max_yaw.to_degrees(), max_vel.to_degrees());
    }

    /// Strapdown check: integrating the IMU truth with the navigation equation reproduces the
    /// recorded velocity changes; a coordinated turn shows the expected load factor.
    #[test]
    fn imu_truth_is_consistent() {
        let ell = Ellipsoid::WGS84;
        let cfg = SynthConfig {
            kind: PathKind::Circle,
            duration: 30.0,
            altitude_ref: AltitudeRef::Ellipsoid,
            altitude: 1000.0,
            radius: 600.0,
            rate: 200.0,
            ..Default::default()
        };
        let recs = simulate(&cfg, (39.9, 32.8), &ell, None);
        let dtr = 1.0 / cfg.rate;
        let mut max_err: f64 = 0.0;
        for win in recs[400..].windows(201).step_by(200) {
            let mut v = DVec3::from_array(win[0].vel_ned);
            for k in 0..200 {
                let (a, b) = (&win[k], &win[k + 1]);
                let q = a.pose.q_ned_body.slerp(b.pose.q_ned_body, 0.5);
                let geo = a.pose.geo;
                let (sl, cl) = geo.lat.sin_cos();
                let w_ie = DVec3::new(geodesy::EARTH_RATE * cl, 0.0, -geodesy::EARTH_RATE * sl);
                let rm = ell.meridian_radius(geo.lat) + geo.h;
                let rn = ell.prime_vertical_radius(geo.lat) + geo.h;
                let rho = DVec3::new(v.y / rn, -v.x / rm, -v.y * sl / cl / rn);
                let g = DVec3::new(0.0, 0.0, geodesy::normal_gravity(geo.lat, geo.h));
                let vdot = DMat3::from_quat(q) * b.imu_f - (2.0 * w_ie + rho).cross(v) + g;
                v += vdot * dtr;
            }
            let err = (v - DVec3::from_array(win[200].vel_ned)).length();
            max_err = max_err.max(err);
        }
        assert!(max_err < 0.3, "velocity drift over 1 s: {max_err} m/s");
        // steady coordinated turn: |f| ≈ sqrt(g² + (v²/r)²), mostly along body −z
        let g0 = geodesy::normal_gravity(0.7, 1000.0);
        let ac = cfg.speed * cfg.speed / cfg.radius;
        let mean_f = recs[2000..4000].iter().fold(DVec3::ZERO, |a, r| a + r.imu_f) / 2000.0;
        assert!((mean_f.length() - (g0 * g0 + ac * ac).sqrt()).abs() < 0.15, "{mean_f:?}");
        assert!(mean_f.z < -9.0 && mean_f.y.abs() < 0.5, "{mean_f:?}");
        // turn rate: |ω| matches the rate of change of the ground track
        let mean_w = recs[2000..4000].iter().fold(DVec3::ZERO, |a, r| a + r.imu_w) / 2000.0;
        // heading (not track: the crab angle into the wind changes around the circle)
        let trk = |r: &Record| geodesy::quat_to_euler_zyx(r.pose.q_ned_body).0;
        let mut dtrk = trk(&recs[4000]) - trk(&recs[2000]);
        while dtrk < 0.0 {
            dtrk += std::f64::consts::TAU;
        }
        let track_rate = dtrk / (2000.0 * dtr);
        assert!((mean_w.length() - track_rate).abs() < 0.002, "{mean_w:?} vs heading rate {track_rate}");
    }

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
