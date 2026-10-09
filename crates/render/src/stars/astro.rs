//! Astrometry for star directions: time scales, IAU 2006 precession + IAU 2000B nutation
//! (≈1 mas), Greenwich apparent sidereal time, polar motion, the Earth's barycentric position
//! and velocity (Keplerian elements with the Sun's reflex motion; aberration to ≈0.01″),
//! stellar aberration (including the observer's diurnal velocity), annual parallax and
//! atmospheric refraction. Algorithms and constants follow ERFA (BSD-3, derived from IAU SOFA)
//! and the IERS Conventions 2003; validated against Skyfield (see tests and docs/stars.md).

use glam::{DMat3, DVec3};

const DAS2R: f64 = std::f64::consts::PI / 180.0 / 3600.0;
const TURNAS: f64 = 1296000.0;
/// Unix time of J2000.0 (2000-01-01 12:00:00; used with UT1 and TT alike)
const UNIX_J2000: f64 = 946_728_000.0;
/// astronomical unit (m), speed of light (m/s), AU per day → v/c
const AU: f64 = 149_597_870_700.0;
const C: f64 = 299_792_458.0;
const C_AU_DAY: f64 = C * 86400.0 / AU;
/// Schwarzschild radius of the Sun (AU)
const SRS: f64 = 1.974_125_743_36e-8;
/// Earth rotation rate (rad/s)
const OMEGA_E: f64 = 7.292_115_146_7e-5;

/// TAI − UTC (leap seconds) from the given Unix time (UTC) on.
const LEAP: [(f64, f64); 28] = [
    (63_072_000.0, 10.0),
    (78_796_800.0, 11.0),
    (94_694_400.0, 12.0),
    (126_230_400.0, 13.0),
    (157_766_400.0, 14.0),
    (189_302_400.0, 15.0),
    (220_924_800.0, 16.0),
    (252_460_800.0, 17.0),
    (283_996_800.0, 18.0),
    (315_532_800.0, 19.0),
    (362_793_600.0, 20.0),
    (394_329_600.0, 21.0),
    (425_865_600.0, 22.0),
    (489_024_000.0, 23.0),
    (567_993_600.0, 24.0),
    (631_152_000.0, 25.0),
    (662_688_000.0, 26.0),
    (709_948_800.0, 27.0),
    (741_484_800.0, 28.0),
    (773_020_800.0, 29.0),
    (820_454_400.0, 30.0),
    (867_715_200.0, 31.0),
    (915_148_800.0, 32.0),
    (1_136_073_600.0, 33.0),
    (1_230_768_000.0, 34.0),
    (1_341_100_800.0, 35.0),
    (1_435_708_800.0, 36.0),
    (1_483_228_800.0, 37.0),
];

/// TAI − UTC (s) at Unix time `unix` (UTC). Before 1972 the 1972 value; future leap seconds
/// are unknown (none announced as of 2026).
pub fn tai_utc(unix: f64) -> f64 {
    LEAP.iter().rev().find(|(t, _)| unix >= *t).map_or(10.0, |l| l.1)
}

/// Rotation matrices in ERFA's (passive) convention.
fn rows(r0: [f64; 3], r1: [f64; 3], r2: [f64; 3]) -> DMat3 {
    DMat3::from_cols(DVec3::from(r0), DVec3::from(r1), DVec3::from(r2)).transpose()
}
pub fn rx(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    rows([1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c])
}
pub fn ry(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    rows([c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c])
}
pub fn rz(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    rows([c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0])
}

/// IAU 2000B luni-solar nutation: multipliers of (l, l', F, D, Ω) and (ψ sin, ψ t·sin, ψ cos,
/// ε cos, ε t·cos, ε sin) in 0.1 µas. From ERFA nut00b.c (BSD-3, derived from IAU SOFA).
const NUT00B: [([i8; 5], [f64; 6]); 77] = [
    ([0, 0, 0, 0, 1], [-172064161.0, -174666.0, 33386.0, 92052331.0, 9086.0, 15377.0]),
    ([0, 0, 2, -2, 2], [-13170906.0, -1675.0, -13696.0, 5730336.0, -3015.0, -4587.0]),
    ([0, 0, 2, 0, 2], [-2276413.0, -234.0, 2796.0, 978459.0, -485.0, 1374.0]),
    ([0, 0, 0, 0, 2], [2074554.0, 207.0, -698.0, -897492.0, 470.0, -291.0]),
    ([0, 1, 0, 0, 0], [1475877.0, -3633.0, 11817.0, 73871.0, -184.0, -1924.0]),
    ([0, 1, 2, -2, 2], [-516821.0, 1226.0, -524.0, 224386.0, -677.0, -174.0]),
    ([1, 0, 0, 0, 0], [711159.0, 73.0, -872.0, -6750.0, 0.0, 358.0]),
    ([0, 0, 2, 0, 1], [-387298.0, -367.0, 380.0, 200728.0, 18.0, 318.0]),
    ([1, 0, 2, 0, 2], [-301461.0, -36.0, 816.0, 129025.0, -63.0, 367.0]),
    ([0, -1, 2, -2, 2], [215829.0, -494.0, 111.0, -95929.0, 299.0, 132.0]),
    ([0, 0, 2, -2, 1], [128227.0, 137.0, 181.0, -68982.0, -9.0, 39.0]),
    ([-1, 0, 2, 0, 2], [123457.0, 11.0, 19.0, -53311.0, 32.0, -4.0]),
    ([-1, 0, 0, 2, 0], [156994.0, 10.0, -168.0, -1235.0, 0.0, 82.0]),
    ([1, 0, 0, 0, 1], [63110.0, 63.0, 27.0, -33228.0, 0.0, -9.0]),
    ([-1, 0, 0, 0, 1], [-57976.0, -63.0, -189.0, 31429.0, 0.0, -75.0]),
    ([-1, 0, 2, 2, 2], [-59641.0, -11.0, 149.0, 25543.0, -11.0, 66.0]),
    ([1, 0, 2, 0, 1], [-51613.0, -42.0, 129.0, 26366.0, 0.0, 78.0]),
    ([-2, 0, 2, 0, 1], [45893.0, 50.0, 31.0, -24236.0, -10.0, 20.0]),
    ([0, 0, 0, 2, 0], [63384.0, 11.0, -150.0, -1220.0, 0.0, 29.0]),
    ([0, 0, 2, 2, 2], [-38571.0, -1.0, 158.0, 16452.0, -11.0, 68.0]),
    ([0, -2, 2, -2, 2], [32481.0, 0.0, 0.0, -13870.0, 0.0, 0.0]),
    ([-2, 0, 0, 2, 0], [-47722.0, 0.0, -18.0, 477.0, 0.0, -25.0]),
    ([2, 0, 2, 0, 2], [-31046.0, -1.0, 131.0, 13238.0, -11.0, 59.0]),
    ([1, 0, 2, -2, 2], [28593.0, 0.0, -1.0, -12338.0, 10.0, -3.0]),
    ([-1, 0, 2, 0, 1], [20441.0, 21.0, 10.0, -10758.0, 0.0, -3.0]),
    ([2, 0, 0, 0, 0], [29243.0, 0.0, -74.0, -609.0, 0.0, 13.0]),
    ([0, 0, 2, 0, 0], [25887.0, 0.0, -66.0, -550.0, 0.0, 11.0]),
    ([0, 1, 0, 0, 1], [-14053.0, -25.0, 79.0, 8551.0, -2.0, -45.0]),
    ([-1, 0, 0, 2, 1], [15164.0, 10.0, 11.0, -8001.0, 0.0, -1.0]),
    ([0, 2, 2, -2, 2], [-15794.0, 72.0, -16.0, 6850.0, -42.0, -5.0]),
    ([0, 0, -2, 2, 0], [21783.0, 0.0, 13.0, -167.0, 0.0, 13.0]),
    ([1, 0, 0, -2, 1], [-12873.0, -10.0, -37.0, 6953.0, 0.0, -14.0]),
    ([0, -1, 0, 0, 1], [-12654.0, 11.0, 63.0, 6415.0, 0.0, 26.0]),
    ([-1, 0, 2, 2, 1], [-10204.0, 0.0, 25.0, 5222.0, 0.0, 15.0]),
    ([0, 2, 0, 0, 0], [16707.0, -85.0, -10.0, 168.0, -1.0, 10.0]),
    ([1, 0, 2, 2, 2], [-7691.0, 0.0, 44.0, 3268.0, 0.0, 19.0]),
    ([-2, 0, 2, 0, 0], [-11024.0, 0.0, -14.0, 104.0, 0.0, 2.0]),
    ([0, 1, 2, 0, 2], [7566.0, -21.0, -11.0, -3250.0, 0.0, -5.0]),
    ([0, 0, 2, 2, 1], [-6637.0, -11.0, 25.0, 3353.0, 0.0, 14.0]),
    ([0, -1, 2, 0, 2], [-7141.0, 21.0, 8.0, 3070.0, 0.0, 4.0]),
    ([0, 0, 0, 2, 1], [-6302.0, -11.0, 2.0, 3272.0, 0.0, 4.0]),
    ([1, 0, 2, -2, 1], [5800.0, 10.0, 2.0, -3045.0, 0.0, -1.0]),
    ([2, 0, 2, -2, 2], [6443.0, 0.0, -7.0, -2768.0, 0.0, -4.0]),
    ([-2, 0, 0, 2, 1], [-5774.0, -11.0, -15.0, 3041.0, 0.0, -5.0]),
    ([2, 0, 2, 0, 1], [-5350.0, 0.0, 21.0, 2695.0, 0.0, 12.0]),
    ([0, -1, 2, -2, 1], [-4752.0, -11.0, -3.0, 2719.0, 0.0, -3.0]),
    ([0, 0, 0, -2, 1], [-4940.0, -11.0, -21.0, 2720.0, 0.0, -9.0]),
    ([-1, -1, 0, 2, 0], [7350.0, 0.0, -8.0, -51.0, 0.0, 4.0]),
    ([2, 0, 0, -2, 1], [4065.0, 0.0, 6.0, -2206.0, 0.0, 1.0]),
    ([1, 0, 0, 2, 0], [6579.0, 0.0, -24.0, -199.0, 0.0, 2.0]),
    ([0, 1, 2, -2, 1], [3579.0, 0.0, 5.0, -1900.0, 0.0, 1.0]),
    ([1, -1, 0, 0, 0], [4725.0, 0.0, -6.0, -41.0, 0.0, 3.0]),
    ([-2, 0, 2, 0, 2], [-3075.0, 0.0, -2.0, 1313.0, 0.0, -1.0]),
    ([3, 0, 2, 0, 2], [-2904.0, 0.0, 15.0, 1233.0, 0.0, 7.0]),
    ([0, -1, 0, 2, 0], [4348.0, 0.0, -10.0, -81.0, 0.0, 2.0]),
    ([1, -1, 2, 0, 2], [-2878.0, 0.0, 8.0, 1232.0, 0.0, 4.0]),
    ([0, 0, 0, 1, 0], [-4230.0, 0.0, 5.0, -20.0, 0.0, -2.0]),
    ([-1, -1, 2, 2, 2], [-2819.0, 0.0, 7.0, 1207.0, 0.0, 3.0]),
    ([-1, 0, 2, 0, 0], [-4056.0, 0.0, 5.0, 40.0, 0.0, -2.0]),
    ([0, -1, 2, 2, 2], [-2647.0, 0.0, 11.0, 1129.0, 0.0, 5.0]),
    ([-2, 0, 0, 0, 1], [-2294.0, 0.0, -10.0, 1266.0, 0.0, -4.0]),
    ([1, 1, 2, 0, 2], [2481.0, 0.0, -7.0, -1062.0, 0.0, -3.0]),
    ([2, 0, 0, 0, 1], [2179.0, 0.0, -2.0, -1129.0, 0.0, -2.0]),
    ([-1, 1, 0, 1, 0], [3276.0, 0.0, 1.0, -9.0, 0.0, 0.0]),
    ([1, 1, 0, 0, 0], [-3389.0, 0.0, 5.0, 35.0, 0.0, -2.0]),
    ([1, 0, 2, 0, 0], [3339.0, 0.0, -13.0, -107.0, 0.0, 1.0]),
    ([-1, 0, 2, -2, 1], [-1987.0, 0.0, -6.0, 1073.0, 0.0, -2.0]),
    ([1, 0, 0, 0, 2], [-1981.0, 0.0, 0.0, 854.0, 0.0, 0.0]),
    ([-1, 0, 0, 1, 0], [4026.0, 0.0, -353.0, -553.0, 0.0, -139.0]),
    ([0, 0, 2, 1, 2], [1660.0, 0.0, -5.0, -710.0, 0.0, -2.0]),
    ([-1, 0, 2, 4, 2], [-1521.0, 0.0, 9.0, 647.0, 0.0, 4.0]),
    ([-1, 1, 0, 1, 1], [1314.0, 0.0, 0.0, -700.0, 0.0, 0.0]),
    ([0, -2, 2, -2, 1], [-1283.0, 0.0, 0.0, 672.0, 0.0, 0.0]),
    ([1, 0, 2, 2, 1], [-1331.0, 0.0, 8.0, 663.0, 0.0, 4.0]),
    ([-2, 0, 2, 2, 2], [1383.0, 0.0, -2.0, -594.0, 0.0, -2.0]),
    ([-1, 0, 0, 0, 2], [1405.0, 0.0, 4.0, -610.0, 0.0, 2.0]),
    ([1, 1, 2, -2, 2], [1290.0, 0.0, 0.0, -556.0, 0.0, 0.0]),
];

/// Nutation in longitude and obliquity (rad), IAU 2000B, with the IAU 2006 adjustments.
/// `t`: TT Julian centuries since J2000.
pub fn nutation(t: f64) -> (f64, f64) {
    let el = (485868.249036 + 1717915923.2178 * t).rem_euclid(TURNAS) * DAS2R;
    let elp = (1287104.79305 + 129596581.0481 * t).rem_euclid(TURNAS) * DAS2R;
    let f = (335779.526232 + 1739527262.8478 * t).rem_euclid(TURNAS) * DAS2R;
    let d = (1072260.70369 + 1602961601.2090 * t).rem_euclid(TURNAS) * DAS2R;
    let om = (450160.398036 - 6962890.5431 * t).rem_euclid(TURNAS) * DAS2R;
    let (mut dp, mut de) = (0.0, 0.0);
    for (n, c) in NUT00B.iter().rev() {
        let arg = (n[0] as f64 * el + n[1] as f64 * elp + n[2] as f64 * f + n[3] as f64 * d + n[4] as f64 * om).rem_euclid(std::f64::consts::TAU);
        let (s, co) = arg.sin_cos();
        dp += (c[0] + c[1] * t) * s + c[2] * co;
        de += (c[3] + c[4] * t) * co + c[5] * s;
    }
    let u2r = DAS2R / 1e7;
    // fixed offsets for the omitted planetary terms
    let dpsi = dp * u2r - 0.135e-3 * DAS2R;
    let deps = de * u2r + 0.388e-3 * DAS2R;
    // IAU 2006 adjustments (J2 rate, ERFA nut06a)
    let fj2 = -2.7774e-6 * t;
    (dpsi * (1.0 + 0.4697e-6 + fj2), deps * (1.0 + fj2))
}

/// Mean obliquity of the ecliptic, IAU 2006 (rad).
pub fn obliquity(t: f64) -> f64 {
    (84381.406 + (-46.836769 + (-0.0001831 + (0.00200340 + (-0.000000576 - 0.0000000434 * t) * t) * t) * t) * t) * DAS2R
}

/// Bias-precession-nutation matrix (GCRS → true equator and equinox of date), IAU 2006/2000B,
/// and the equation of the equinoxes (rad).
pub fn npb(t: f64) -> (DMat3, f64) {
    let gamb = (-0.052928 + (10.556378 + (0.4932044 + (-0.00031238 + (-0.000002788 + 0.0000000260 * t) * t) * t) * t) * t) * DAS2R;
    let phib = (84381.412819 + (-46.811016 + (0.0511268 + (0.00053289 + (-0.000000440 - 0.0000000176 * t) * t) * t) * t) * t) * DAS2R;
    let psib = (-0.041775 + (5038.481484 + (1.5584175 + (-0.00018522 + (-0.000026452 - 0.0000000148 * t) * t) * t) * t) * t) * DAS2R;
    let epsa = obliquity(t);
    let (dpsi, deps) = nutation(t);
    let m = rx(-(epsa + deps)) * rz(-(psib + dpsi)) * rx(phib) * rz(gamb);
    // equation of the equinoxes: dpsi cos(eps) + complementary terms (ERFA eect00, > 1 µas)
    let om = (450160.398036 + (-6962890.5431 + (7.4722 + (0.007702 - 0.00005939 * t) * t) * t) * t).rem_euclid(TURNAS) * DAS2R;
    let f = (335779.526232 + (1739527262.8478 + (-12.7512 + (-0.001037 + 0.00000417 * t) * t) * t) * t).rem_euclid(TURNAS) * DAS2R;
    let d = (1072260.703692 + (1602961601.2090 + (-6.3706 + (0.006593 - 0.00003169 * t) * t) * t) * t).rem_euclid(TURNAS) * DAS2R;
    let lp = (1287104.793048 + (129596581.0481 + (-0.5532 + (0.000136 - 0.00001149 * t) * t) * t) * t).rem_euclid(TURNAS) * DAS2R;
    let l = (485868.249036 + (1717915923.2178 + (31.8792 + (0.051635 - 0.00024470 * t) * t) * t) * t).rem_euclid(TURNAS) * DAS2R;
    let fd = 2.0 * f - 2.0 * d;
    let ect = 2640.96e-6 * om.sin() - 0.39e-6 * om.cos() + 63.52e-6 * (2.0 * om).sin() - 0.02e-6 * (2.0 * om).cos()
        + 11.75e-6 * (fd + 3.0 * om).sin()
        + 11.21e-6 * (fd + om).sin()
        - 4.55e-6 * (fd + 2.0 * om).sin()
        + 2.02e-6 * (2.0 * f + 3.0 * om).sin()
        + 1.98e-6 * (2.0 * f + om).sin()
        - 1.72e-6 * (3.0 * om).sin()
        - 1.41e-6 * (lp + om).sin()
        - 1.26e-6 * (lp - om).sin()
        - 0.63e-6 * (l - om).sin()
        - 0.63e-6 * (l + om).sin()
        - 0.87e-6 * t * om.sin();
    (m, dpsi * epsa.cos() + ect * DAS2R)
}

/// Earth rotation angle (rad) from UT1 days since J2000.
pub fn era(du: f64) -> f64 {
    let f = du.rem_euclid(1.0);
    (std::f64::consts::TAU * (f + 0.7790572732640 + 0.00273781191135448 * du)).rem_euclid(std::f64::consts::TAU)
}

/// Greenwich mean sidereal time, IAU 2006 (rad).
pub fn gmst(du: f64, t: f64) -> f64 {
    (era(du) + (0.014506 + (4612.156534 + (1.3915817 + (-0.00000044 + (-0.000029956 - 0.0000000368 * t) * t) * t) * t) * t) * DAS2R)
        .rem_euclid(std::f64::consts::TAU)
}

/// Heliocentric position (AU) and velocity (AU/day), ecliptic and equinox J2000, from JPL's
/// approximate Keplerian elements (Standish; 1800–2050): [a, e, I, L, ϖ, Ω] and rates per
/// Julian century.
fn kepler(el: &[[f64; 6]; 2], t: f64) -> (DVec3, DVec3) {
    let deg = std::f64::consts::PI / 180.0;
    let a = el[0][0] + el[1][0] * t;
    let e = el[0][1] + el[1][1] * t;
    let i = (el[0][2] + el[1][2] * t) * deg;
    let l = (el[0][3] + el[1][3] * t) * deg;
    let w = (el[0][4] + el[1][4] * t) * deg;
    let o = (el[0][5] + el[1][5] * t) * deg;
    let m = (l - w).rem_euclid(std::f64::consts::TAU);
    let n = el[1][3] * deg / 36525.0; // mean motion (rad/day)
    let mut ea = m + e * m.sin();
    for _ in 0..8 {
        ea -= (ea - e * ea.sin() - m) / (1.0 - e * ea.cos());
    }
    let (se, ce) = ea.sin_cos();
    let b = (1.0 - e * e).sqrt();
    let (x, y) = (a * (ce - e), a * b * se);
    let k = a * n / (1.0 - e * ce);
    let (vx, vy) = (-k * se, k * b * ce);
    let om = w - o; // argument of perihelion
    let r = rz(-o) * rx(-i) * rz(-om);
    (r * DVec3::new(x, y, 0.0), r * DVec3::new(vx, vy, 0.0))
}

const EMB: [[f64; 6]; 2] =
    [[1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0], [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0]];
const JUPITER: [[f64; 6]; 2] = [
    [5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
    [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106],
];
const SATURN: [[f64; 6]; 2] = [
    [9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
    [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794],
];

/// Barycentric position (AU) and velocity (AU/day) of the Earth in ICRS axes, and its
/// heliocentric position (AU). `t`: TT (≈ TDB) Julian centuries since J2000.
pub fn earth_barycentric(t: f64) -> (DVec3, DVec3, DVec3) {
    let (pe, ve) = kepler(&EMB, t);
    let (pj, vj) = kepler(&JUPITER, t);
    let (ps, vs) = kepler(&SATURN, t);
    // the Sun's reflex motion about the barycentre (Jupiter, Saturn)
    let (mj, ms) = (1.0 / 1047.3486, 1.0 / 3497.898);
    let sun_p = -(pj * mj + ps * ms) / (1.0 + mj + ms);
    let sun_v = -(vj * mj + vs * ms) / (1.0 + mj + ms);
    let ecl = rx(-84381.448 * DAS2R); // ecliptic J2000 → equatorial
    (ecl * (pe + sun_p), ecl * (ve + sun_v), ecl * pe)
}

/// Light deflection by the Sun (ERFA ldsun): star direction `p`, Sun → observer unit vector
/// `e` at `em` AU.
pub fn deflect(p: DVec3, e: DVec3, em: f64) -> DVec3 {
    let dlim = 1e-6 / (em * em).max(1.0);
    let w = SRS / em / p.dot(p + e).max(dlim);
    p + p.cross(e.cross(p)) * w
}

/// Stellar aberration (ERFA ab): natural direction `p` → proper direction for an observer with
/// barycentric velocity `v` (units of c), at `s` AU from the Sun.
pub fn aberrate(p: DVec3, v: DVec3, s: f64) -> DVec3 {
    let bm1 = (1.0 - v.length_squared()).sqrt();
    let pdv = p.dot(v);
    let w1 = 1.0 + pdv / (1.0 + bm1);
    let w2 = SRS / s;
    (p * bm1 + v * w1 + (v - p * pdv) * w2).normalize()
}

/// Everything that maps catalogue (ICRS) directions to apparent directions in the Earth-fixed
/// frame at one instant.
#[derive(Clone, Copy, Debug)]
pub struct Sky {
    /// GCRS → ITRS (polar motion · sidereal time · bias-precession-nutation)
    pub gcrs_to_itrs: DMat3,
    /// Earth's barycentric position (AU) and velocity (units of c), Sun → Earth unit vector
    /// and distance (AU)
    pub earth_pos: DVec3,
    pub earth_vel_c: DVec3,
    pub sun_to_earth: DVec3,
    pub sun_dist: f64,
    /// Julian years since J2000 (TT), for proper motion
    pub years: f64,
    /// Julian date (TT ≈ TDB)
    pub jd_tt: f64,
}

impl Sky {
    /// `unix`: UTC (s); `dut1`: UT1 − UTC (s); `xp`, `yp`: polar motion (rad).
    pub fn new(unix: f64, dut1: f64, xp: f64, yp: f64) -> Sky {
        let tt = unix + tai_utc(unix) + 32.184;
        let t = (tt - UNIX_J2000) / 86400.0 / 36525.0;
        let du = (unix + dut1 - UNIX_J2000) / 86400.0;
        let (m, ee) = npb(t);
        let gast = gmst(du, t) + ee;
        let w = rx(-yp) * ry(-xp);
        let jd_tt = tt / 86400.0 + 2440587.5;
        // Earth from DE440 when covered (1990–2060), else Keplerian elements
        let eph = super::ephem::Ephemeris::builtin();
        let (pos, vel, helio) = match (eph.barycentric(super::ephem::Body::Earth, jd_tt), eph.barycentric(super::ephem::Body::Sun, jd_tt)) {
            (Some((pe, ve)), Some((ps, _))) => {
                let au = AU / 1000.0;
                (pe / au, ve / au, (pe - ps) / au)
            }
            _ => earth_barycentric(t),
        };
        Sky {
            gcrs_to_itrs: w * rz(gast) * m,
            earth_pos: pos,
            earth_vel_c: vel / C_AU_DAY,
            sun_to_earth: helio.normalize(),
            sun_dist: helio.length(),
            years: t * 100.0,
            jd_tt,
        }
    }

    /// Observer velocity (units of c, GCRS axes) of an Earth-fixed point at ECEF `pos` (m):
    /// orbital plus diurnal.
    pub fn observer_vel_c(&self, pos: DVec3) -> DVec3 {
        let v_itrs = DVec3::new(-OMEGA_E * pos.y, OMEGA_E * pos.x, 0.0);
        self.earth_vel_c + self.gcrs_to_itrs.transpose() * v_itrs / C
    }
}

/// Atmospheric refraction (rad) to add to the geometric elevation `el` (rad) of a star, for
/// pressure `p_hpa` and temperature `t_k` at the observer: Green's A tan z + B tan³ z model
/// (ERFA refco, dry air, λ = 0.574 µm) above 20° elevation, Sæmundsson's formula below 10°, blended between.
pub fn refraction(el: f64, p_hpa: f64, t_k: f64) -> f64 {
    if p_hpa <= 0.0 {
        return 0.0;
    }
    // optical refractivity at λ = 0.574 µm (ERFA refco, dry air)
    let wl2 = 0.574f64 * 0.574;
    let gamma = (77.53484e-6 + (4.39108e-7 + 3.666e-9 / wl2) / wl2) * p_hpa / t_k;
    let beta = 4.4474e-6 * t_k;
    let (ra, rb) = (gamma * (1.0 - beta), -gamma * (beta - gamma / 2.0));
    let green = |e: f64| {
        // iterate on the apparent zenith distance
        let mut r = 0.0;
        for _ in 0..3 {
            let tz = (std::f64::consts::FRAC_PI_2 - (e + r)).tan();
            r = (ra + rb * tz * tz) * tz;
        }
        r
    };
    let saem = |e: f64| {
        let h = e.to_degrees().max(-1.0);
        (1.02 / (h + 10.3 / (h + 5.11)).to_radians().tan()).to_radians() / 60.0 * (p_hpa / 1010.0) * (283.0 / t_k)
    };
    let ed = el.to_degrees();
    if ed >= 20.0 {
        green(el)
    } else if ed <= 10.0 {
        saem(el)
    } else {
        let a = (ed - 10.0) / 10.0;
        saem(el) * (1.0 - a) + green(el) * a
    }
}

/// Refraction (rad) for an observer at height `h` (m) in the standard atmosphere, for a
/// geometric elevation `el` (rad), also below the horizontal: seen from altitude, a ray dips
/// to its tangent height and climbs out again, bending about twice the horizontal refraction
/// there, less the part above the observer.
pub fn refraction_h(el: f64, h: f64) -> f64 {
    let (p, t) = standard_atmosphere(h);
    if p <= 0.0 {
        return 0.0;
    }
    if el >= 0.0 {
        return refraction(el, p, t);
    }
    let re = 6_371_000.0;
    let ht = ((re + h) * el.cos() - re).max(0.0);
    let hor = |x: f64| {
        let (p, t) = standard_atmosphere(x);
        refraction(0.0, p, t)
    };
    // below the Earth's limb (dip) the body is hidden: fade out over 2°
    let dip = (re / (re + h.max(0.0))).acos();
    let fade = (1.0 - (-el - dip) / 2f64.to_radians()).clamp(0.0, 1.0);
    (2.0 * hor(ht) - hor(h)) * fade
}

/// U.S. Standard Atmosphere 1976 (to 47 km): pressure (hPa) and temperature (K) at
/// geometric height `h` (m); zero pressure above 80 km.
pub fn standard_atmosphere(h: f64) -> (f64, f64) {
    if h > 80_000.0 {
        return (0.0, 200.0);
    }
    let h = h.max(-500.0);
    if h <= 11_000.0 {
        let t = 288.15 - 0.0065 * h;
        (1013.25 * (t / 288.15).powf(5.255_877), t)
    } else if h <= 20_000.0 {
        (226.32 * (-(h - 11_000.0) / 6341.62).exp(), 216.65)
    } else {
        let t = 216.65 + 0.001 * (h - 20_000.0);
        (54.749 * (t / 216.65).powf(-34.163_19), t)
    }
}
