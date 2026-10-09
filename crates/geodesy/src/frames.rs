//! Geodetic / ECEF / local-tangent-plane (ENU, NED) / AER conversions.
//!
//! Function names and semantics mirror [pymap3d](https://github.com/geospace-code/pymap3d),
//! but angles are always **radians** and points are passed as [`DVec3`] / [`Geodetic`].
//!
//! * `ecef2enu(p, o, ell)`: position of ECEF point `p` in the ENU frame anchored at the
//!   geodetic origin `o` (the origin itself maps to zero).
//! * `*v` functions (`ecef2enuv`, ...) rotate *vectors* (velocities, directions,
//!   displacements); they involve no translation and only need the origin's lat/lon.
//! * The local "up" axis is the **geodetic** surface normal ([`up_vector`]).

use std::f64::consts::TAU;

use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};

use crate::ellipsoid::Ellipsoid;

/// Geodetic coordinates: latitude and longitude in **radians**, height `h` in meters above
/// the reference ellipsoid.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Geodetic {
    /// Geodetic latitude (rad), positive north.
    pub lat: f64,
    /// Longitude (rad), positive east.
    pub lon: f64,
    /// Height above the ellipsoid (m).
    pub h: f64,
}

impl Geodetic {
    /// From radians and meters.
    #[inline]
    pub fn new(lat: f64, lon: f64, h: f64) -> Self {
        Self { lat, lon, h }
    }

    /// From latitude/longitude in **degrees** and height in meters.
    #[inline]
    pub fn from_deg(lat_deg: f64, lon_deg: f64, h: f64) -> Self {
        Self::new(lat_deg.to_radians(), lon_deg.to_radians(), h)
    }

    /// Latitude in degrees.
    #[inline]
    pub fn lat_deg(&self) -> f64 {
        self.lat.to_degrees()
    }

    /// Longitude in degrees.
    #[inline]
    pub fn lon_deg(&self) -> f64 {
        self.lon.to_degrees()
    }
}

// ---------------------------------------------------------------------------------------
// Geodetic <-> ECEF
// ---------------------------------------------------------------------------------------

/// Geodetic → ECEF (m).
pub fn geodetic2ecef(g: Geodetic, ell: &Ellipsoid) -> DVec3 {
    let n = ell.prime_vertical_radius(g.lat);
    let (sl, cl) = g.lat.sin_cos();
    let (so, co) = g.lon.sin_cos();
    let b2a2 = (ell.b / ell.a) * (ell.b / ell.a);
    DVec3::new((n + g.h) * cl * co, (n + g.h) * cl * so, (n * b2a2 + g.h) * sl)
}

/// ECEF (m) → geodetic.
///
/// Uses the closed-form solution of H. Vermeille (2011), *"An analytical method to
/// transform geocentric into geodetic coordinates"*, J. Geodesy 85:105–117, which is valid
/// everywhere including the poles, the equator, very high altitudes and the interior of the
/// ellipsoid (where it returns the **nearest** surface point, i.e. negative `h` of smallest
/// magnitude). The height is then recomputed from the latitude as the projection of the
/// offset from the foot point onto the normal,
/// `h = (w − N cos φ) cos φ + (z − N (1 − e²) sin φ) sin φ`, which is well conditioned.
///
/// Accuracy (vs. a 50-digit reference, WGS84): latitude within 1 ulp; height within
/// ~2 ulp of the coordinate magnitude (≈ 2e-9 m near the surface, ≈ 4e-9 m at 1e7 m
/// altitude) — i.e. at the resolution of the `f64` ECEF input itself.
///
/// On the polar axis the longitude is `atan2(y, x)` (i.e. `0` for `x = y = 0`).
pub fn ecef2geodetic(p: DVec3, ell: &Ellipsoid) -> Geodetic {
    let a = ell.a;
    let e2 = ell.e2();
    let e4 = e2 * e2;
    let w2 = p.x * p.x + p.y * p.y;
    let w = w2.sqrt();
    let z = p.z;

    let pp = w2 / (a * a);
    let q = (1.0 - e2) / (a * a) * z * z;
    let r = (pp + q - e4) / 6.0;
    let evolute = 8.0 * r * r * r + e4 * pp * q;

    let lat = if evolute > 0.0 || q != 0.0 {
        let u = if evolute > 0.0 {
            // Outside the evolute: general case.
            let rad1 = evolute.sqrt();
            let rad2 = (e4 * pp * q).sqrt();
            let rad3 = ((rad1 + rad2) * (rad1 + rad2)).cbrt();
            // `rad3 > 0` here because `evolute > 0`.
            r + 0.5 * rad3 + 2.0 * r * r / rad3
        } else {
            // Inside the evolute (deep interior, within ~43 km of the centre), off the
            // equatorial plane.
            let rad1 = (-evolute).sqrt();
            let rad2 = (-8.0 * r * r * r).sqrt();
            let rad3 = (e4 * pp * q).sqrt();
            let t = 2.0 * rad3.atan2(rad1 + rad2) / 3.0;
            -4.0 * r * t.sin() * (std::f64::consts::FRAC_PI_6 + t).cos()
        };
        let v = (u * u + e4 * q).sqrt();
        let ww = e2 * (u + v - q) / (2.0 * v);
        let k = (u + v) / ((ww * ww + u + v).sqrt() + ww);
        let d = k * w / (k + e2);
        let s = (d * d + z * z).sqrt();
        2.0 * z.atan2(s + d)
    } else {
        // Singular disc: z == 0 and w <= a e² (inside the evolute on the equatorial plane).
        // The nearest surface points are at ±φ with cos²φ = p(1-e²) / (e²(e²-p)).
        let lat = (e4 - pp).max(0.0).sqrt().atan2((pp * (1.0 - e2)).sqrt());
        lat.copysign(z)
    };

    // Height along the normal from the foot point S = (N cos φ, N (1 - e²) sin φ); the
    // residual form keeps cancellation errors at the ~1 ulp level of the input.
    let (sl, cl) = lat.sin_cos();
    let n = a / (1.0 - e2 * sl * sl).sqrt();
    let h = (w - n * cl) * cl + (z - n * (1.0 - e2) * sl) * sl;
    Geodetic { lat, lon: p.y.atan2(p.x), h }
}

// ---------------------------------------------------------------------------------------
// Rotations
// ---------------------------------------------------------------------------------------

/// Geodetic surface normal ("up") at `lat`, `lon`, as an ECEF unit vector.
#[inline]
pub fn up_vector(lat: f64, lon: f64) -> DVec3 {
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    DVec3::new(cl * co, cl * so, sl)
}

/// Local east, north and up unit vectors (in ECEF) at `lat`, `lon`.
#[inline]
fn enu_axes(lat: f64, lon: f64) -> (DVec3, DVec3, DVec3) {
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    let e = DVec3::new(-so, co, 0.0);
    let n = DVec3::new(-sl * co, -sl * so, cl);
    let u = DVec3::new(cl * co, cl * so, sl);
    (e, n, u)
}

/// Rotation `R` with `v_enu = R * v_ecef` at geodetic `lat`, `lon`. Its rows are the
/// east, north and up unit vectors expressed in ECEF; `Rᵀ` maps ENU → ECEF.
pub fn rot_ecef2enu(lat: f64, lon: f64) -> DMat3 {
    let (e, n, u) = enu_axes(lat, lon);
    DMat3::from_cols(e, n, u).transpose()
}

/// Rotation `R` with `v_ned = R * v_ecef` at geodetic `lat`, `lon`. Rows: north, east, down.
pub fn rot_ecef2ned(lat: f64, lon: f64) -> DMat3 {
    let (e, n, u) = enu_axes(lat, lon);
    DMat3::from_cols(n, e, -u).transpose()
}

#[inline]
fn enu_to_ned(v: DVec3) -> DVec3 {
    DVec3::new(v.y, v.x, -v.z)
}

#[inline]
fn ned_to_enu(v: DVec3) -> DVec3 {
    DVec3::new(v.y, v.x, -v.z)
}

/// Rotate an ECEF vector into ENU at `lat`, `lon` (no translation).
#[inline]
pub fn ecef2enuv(v: DVec3, lat: f64, lon: f64) -> DVec3 {
    rot_ecef2enu(lat, lon) * v
}

/// Rotate an ENU vector into ECEF at `lat`, `lon` (no translation).
#[inline]
pub fn enu2ecefv(v: DVec3, lat: f64, lon: f64) -> DVec3 {
    rot_ecef2enu(lat, lon).transpose() * v
}

/// Rotate an ECEF vector into NED at `lat`, `lon` (no translation).
#[inline]
pub fn ecef2nedv(v: DVec3, lat: f64, lon: f64) -> DVec3 {
    rot_ecef2ned(lat, lon) * v
}

/// Rotate a NED vector into ECEF at `lat`, `lon` (no translation).
#[inline]
pub fn ned2ecefv(v: DVec3, lat: f64, lon: f64) -> DVec3 {
    rot_ecef2ned(lat, lon).transpose() * v
}

// ---------------------------------------------------------------------------------------
// Point conversions (pymap3d style)
// ---------------------------------------------------------------------------------------

/// ECEF point → ENU coordinates relative to geodetic origin `o`.
pub fn ecef2enu(p: DVec3, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    ecef2enuv(p - geodetic2ecef(o, ell), o.lat, o.lon)
}

/// ENU coordinates relative to `o` → ECEF point.
pub fn enu2ecef(enu: DVec3, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    geodetic2ecef(o, ell) + enu2ecefv(enu, o.lat, o.lon)
}

/// ECEF point → NED coordinates relative to geodetic origin `o`.
pub fn ecef2ned(p: DVec3, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    ecef2nedv(p - geodetic2ecef(o, ell), o.lat, o.lon)
}

/// NED coordinates relative to `o` → ECEF point.
pub fn ned2ecef(ned: DVec3, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    geodetic2ecef(o, ell) + ned2ecefv(ned, o.lat, o.lon)
}

/// Geodetic point → ENU relative to `o`.
pub fn geodetic2enu(g: Geodetic, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    ecef2enu(geodetic2ecef(g, ell), o, ell)
}

/// ENU relative to `o` → geodetic point.
pub fn enu2geodetic(enu: DVec3, o: Geodetic, ell: &Ellipsoid) -> Geodetic {
    ecef2geodetic(enu2ecef(enu, o, ell), ell)
}

/// Geodetic point → NED relative to `o`.
pub fn geodetic2ned(g: Geodetic, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    ecef2ned(geodetic2ecef(g, ell), o, ell)
}

/// NED relative to `o` → geodetic point.
pub fn ned2geodetic(ned: DVec3, o: Geodetic, ell: &Ellipsoid) -> Geodetic {
    ecef2geodetic(ned2ecef(ned, o, ell), ell)
}

// ---------------------------------------------------------------------------------------
// AER
// ---------------------------------------------------------------------------------------

/// ENU → `(azimuth, elevation, slant_range)`. Azimuth is clockwise from north in
/// `[0, 2π)`, elevation in `[-π/2, π/2]`, both radians.
///
/// Unlike pymap3d, components below 1 mm are *not* snapped to zero.
pub fn enu2aer(enu: DVec3) -> DVec3 {
    let r = enu.x.hypot(enu.y);
    let srange = r.hypot(enu.z);
    let el = enu.z.atan2(r);
    let mut az = enu.x.atan2(enu.y).rem_euclid(TAU);
    if az >= TAU {
        az = 0.0;
    }
    DVec3::new(az, el, srange)
}

/// `(azimuth, elevation, slant_range)` → ENU.
pub fn aer2enu(aer: DVec3) -> DVec3 {
    let (az, el, srange) = (aer.x, aer.y, aer.z);
    let (se, ce) = el.sin_cos();
    let (sa, ca) = az.sin_cos();
    let r = srange * ce;
    DVec3::new(r * sa, r * ca, srange * se)
}

/// NED → AER (see [`enu2aer`]).
#[inline]
pub fn ned2aer(ned: DVec3) -> DVec3 {
    enu2aer(ned_to_enu(ned))
}

/// AER → NED (see [`aer2enu`]).
#[inline]
pub fn aer2ned(aer: DVec3) -> DVec3 {
    enu_to_ned(aer2enu(aer))
}

/// ECEF point → AER as seen from geodetic origin `o`.
pub fn ecef2aer(p: DVec3, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    enu2aer(ecef2enu(p, o, ell))
}

/// AER seen from `o` → ECEF point.
pub fn aer2ecef(aer: DVec3, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    enu2ecef(aer2enu(aer), o, ell)
}

/// Geodetic point → AER as seen from `o`.
pub fn geodetic2aer(g: Geodetic, o: Geodetic, ell: &Ellipsoid) -> DVec3 {
    enu2aer(geodetic2enu(g, o, ell))
}

/// AER seen from `o` → geodetic point.
pub fn aer2geodetic(aer: DVec3, o: Geodetic, ell: &Ellipsoid) -> Geodetic {
    enu2geodetic(aer2enu(aer), o, ell)
}

// ---------------------------------------------------------------------------------------
// Local tangent frame
// ---------------------------------------------------------------------------------------

/// Axis convention of a [`LocalFrame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalConvention {
    /// x east, y north, z up.
    Enu,
    /// x north, y east, z down.
    Ned,
}

/// A local tangent-plane frame anchored at a geodetic origin, with the rotation and origin
/// ECEF position precomputed.
///
/// `local = r_ecef2local * (ecef - origin_ecef)`.
#[derive(Clone, Copy, Debug)]
pub struct LocalFrame {
    /// Geodetic origin.
    pub origin: Geodetic,
    /// ECEF position of `origin` (m).
    pub origin_ecef: DVec3,
    /// Axis convention.
    pub convention: LocalConvention,
    /// Rotation with `v_local = r_ecef2local * v_ecef`.
    pub r_ecef2local: DMat3,
    /// Ellipsoid used for all conversions.
    pub ell: Ellipsoid,
}

impl LocalFrame {
    /// Build a frame at `origin`.
    pub fn new(origin: Geodetic, convention: LocalConvention, ell: Ellipsoid) -> Self {
        let r_ecef2local = match convention {
            LocalConvention::Enu => rot_ecef2enu(origin.lat, origin.lon),
            LocalConvention::Ned => rot_ecef2ned(origin.lat, origin.lon),
        };
        Self { origin, origin_ecef: geodetic2ecef(origin, &ell), convention, r_ecef2local, ell }
    }

    /// Rotation with `v_ecef = R * v_local`.
    #[inline]
    pub fn rot_local2ecef(&self) -> DMat3 {
        self.r_ecef2local.transpose()
    }

    /// ECEF point → local coordinates.
    #[inline]
    pub fn ecef_to_local(&self, p: DVec3) -> DVec3 {
        self.r_ecef2local * (p - self.origin_ecef)
    }

    /// Local coordinates → ECEF point.
    #[inline]
    pub fn local_to_ecef(&self, l: DVec3) -> DVec3 {
        self.origin_ecef + self.rot_local2ecef() * l
    }

    /// Rotate an ECEF vector into the local frame.
    #[inline]
    pub fn vec_ecef_to_local(&self, v: DVec3) -> DVec3 {
        self.r_ecef2local * v
    }

    /// Rotate a local vector into ECEF.
    #[inline]
    pub fn vec_local_to_ecef(&self, v: DVec3) -> DVec3 {
        self.rot_local2ecef() * v
    }

    /// Geodetic point → local coordinates.
    #[inline]
    pub fn geodetic_to_local(&self, g: Geodetic) -> DVec3 {
        self.ecef_to_local(geodetic2ecef(g, &self.ell))
    }

    /// Local coordinates → geodetic point.
    #[inline]
    pub fn local_to_geodetic(&self, l: DVec3) -> Geodetic {
        ecef2geodetic(self.local_to_ecef(l), &self.ell)
    }
}

// ---------------------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------------------

/// Great-circle distance (m) between `a` and `b` on a sphere of `radius` (haversine
/// formula; heights are ignored).
pub fn haversine_distance(a: Geodetic, b: Geodetic, radius: f64) -> f64 {
    let dlat = b.lat - a.lat;
    let dlon = b.lon - a.lon;
    let s1 = (0.5 * dlat).sin();
    let s2 = (0.5 * dlon).sin();
    let hav = (s1 * s1 + a.lat.cos() * b.lat.cos() * s2 * s2).clamp(0.0, 1.0);
    2.0 * radius * hav.sqrt().asin()
}

/// Intersect the ray `o + t d` (ECEF) with the ellipsoid inflated by `h`, i.e. with
/// semi-axes `(a + h, a + h, b + h)` — an approximation of the constant-height surface
/// (exact for a sphere). Returns the smallest `t > 0`, or `None` if the ray misses or the
/// surface lies entirely behind the origin.
///
/// `t` is measured in units of `|d|` (the distance in meters when `d` is a unit vector).
/// If `o` is inside the surface, the exit point is returned.
pub fn ray_ellipsoid(o: DVec3, d: DVec3, ell: &Ellipsoid, h: f64) -> Option<f64> {
    let ra = ell.a + h;
    let rb = ell.b + h;
    if ra <= 0.0 || rb <= 0.0 {
        return None;
    }
    let s = DVec3::new(1.0 / ra, 1.0 / ra, 1.0 / rb);
    let os = o * s;
    let ds = d * s;
    // |os + t ds|² = 1  ->  A t² + 2 B t + C = 0
    let qa = ds.length_squared();
    if qa == 0.0 {
        return None;
    }
    let qb = os.dot(ds);
    let qc = os.length_squared() - 1.0;
    let disc = qb * qb - qa * qc;
    if disc < 0.0 {
        return None;
    }
    // Numerically stable roots.
    let qq = -(qb + disc.sqrt().copysign(qb));
    let (t1, t2) = if qq == 0.0 {
        (0.0, 0.0)
    } else {
        let r1 = qq / qa;
        let r2 = qc / qq;
        (r1.min(r2), r1.max(r2))
    };
    if t1 > 0.0 {
        Some(t1)
    } else if t2 > 0.0 {
        Some(t2)
    } else {
        None
    }
}

/// Distance (m) from altitude `h` to the geometric horizon over a sphere of radius `r`:
/// `sqrt(2 r h + h²)` (straight-line distance to the tangent point).
#[inline]
pub fn horizon_distance(h: f64, r: f64) -> f64 {
    (2.0 * r * h + h * h).max(0.0).sqrt()
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::approx_constant)] // reference values happen to equal π/2, π/4
pub(crate) mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const WGS84: Ellipsoid = Ellipsoid::WGS84;

    /// Tiny deterministic PRNG (SplitMix64) for randomized tests.
    pub(crate) struct Rng(u64);
    impl Rng {
        pub(crate) fn new(seed: u64) -> Self {
            Self(seed)
        }
        pub(crate) fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in [lo, hi).
        pub(crate) fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * u
        }
    }

    fn assert_vec(a: DVec3, b: DVec3, tol: f64, what: &str) {
        assert!((a - b).abs().max_element() <= tol, "{what}: got {a:?}, want {b:?}, diff {:?}", a - b);
    }

    fn assert_geo(g: Geodetic, lat: f64, lon: f64, h: f64, tol_rad: f64, tol_m: f64, what: &str) {
        assert!(
            (g.lat - lat).abs() <= tol_rad && (g.lon - lon).abs() <= tol_rad && (g.h - h).abs() <= tol_m,
            "{what}: got {g:?}, want ({lat}, {lon}, {h}); dlat={:e} dlon={:e} dh={:e}",
            g.lat - lat,
            g.lon - lon,
            g.h - h
        );
    }

    #[test]
    fn geodetic_struct() {
        let g = Geodetic::from_deg(39.92, 32.85, 1200.0);
        assert!((g.lat_deg() - 39.92).abs() < 1e-12);
        assert!((g.lon_deg() - 32.85).abs() < 1e-12);
        assert_eq!(g.h, 1200.0);
        assert_eq!(Geodetic::new(1.0, 2.0, 3.0), Geodetic { lat: 1.0, lon: 2.0, h: 3.0 });
    }

    /// Reference: pymap3d 3.2 `geodetic2ecef(lat_deg, lon_deg, h)`.
    #[test]
    fn geodetic2ecef_vs_pymap3d() {
        let cases = [
            ((39.92, 32.85, 1200.0), (4115897.96464049, 2657601.65201406, 4071947.0991998757)),
            ((-33.8688, 151.2093, 5000.0), (-4649689.653406782, 2555205.7903041113, -3537158.8531592987)),
            ((89.999, -120.0, 10000.0), (-55.934256239316795, -96.88097369007322, 6366752.313268944)),
            ((0.0, -179.5, -3000.0), (-6374894.254317552, -55632.85933749726, 0.0)),
            ((60.0, 10.0, 10000000.0), (8072572.149454638, 1423412.2736645795, 14160731.171783026)),
            ((-90.0, 0.0, 0.0), (3.918620924814471e-10, 0.0, -6356752.31424518)),
            ((45.0, 45.0, -5000.0), (3191919.145060574, 3191919.1450605737, 4483812.874959988)),
        ];
        for ((la, lo, h), (x, y, z)) in cases {
            let p = geodetic2ecef(Geodetic::from_deg(la, lo, h), &WGS84);
            assert_vec(p, DVec3::new(x, y, z), 1e-6, "geodetic2ecef");
        }
    }

    /// Reference: 50-digit Decimal Newton iteration on the exact normal condition (nearest
    /// surface point). pymap3d's one-step You (2000) formula deviates from these deep
    /// inside the earth (up to ~1e-5 rad), so the high-precision values are used.
    #[test]
    fn ecef2geodetic_vs_reference() {
        let cases = [
            ((4198944.0, 2710080.0, 4078442.0), (0.6877236917837433, 0.573148386239883, 80968.39366691113)),
            ((-4646678.0, 2553000.0, -3534000.0), (-0.5910339715038466, 2.639191183829611, 166.05464675071596)),
            ((1000.0, 2000.0, 6356000.0), (1.5704468779214924, 1.1071487177940904, -751.9235494578057)),
            ((16378137.0, 0.0, 100000.0), (0.006121582897332895, 0.0, 10000306.080108143)),
            ((3000000.0, 3000000.0, 3000000.0), (0.6193682525067219, 0.7853981633974483, -1174825.1460048922)),
            ((20000.0, 0.0, 10.0), (1.0848348103836167, 0.0, -6352073.365657186)),
            ((0.0, 0.0, -6357752.314245179), (-1.5707963267948966, 0.0, 999.999999999798)),
            ((-1000000.0, 1000000.0, -2000000.0), (-0.9635037114545407, 2.356194490192345, -3914316.202318265)),
            ((6378137.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((40000.0, 0.0, 1000.0), (0.4725960762208234, 0.0, -6337641.066987242)),
            ((30000.0, 10000.0, -50.0), (-0.7403801857075114, 0.3217505543966422, -6345036.909391149)),
            ((0.0, 0.0, 16356752.31424518), (1.5707963267948966, 0.0, 10000000.0)),
        ];
        for ((x, y, z), (la, lo, h)) in cases {
            let g = ecef2geodetic(DVec3::new(x, y, z), &WGS84);
            // Points deep inside (|h| > 1000 km) are ill-conditioned; allow slightly more.
            let (tr, tm) = if h < -1.0e6 { (1e-11, 1e-7) } else { (1e-12, 1e-8) };
            assert_geo(g, la, lo, h, tr, tm, "ecef2geodetic");
        }
    }

    #[test]
    fn ecef2geodetic_special_points() {
        let b = WGS84.b;
        // Poles and exactly on the z-axis.
        for (z, lat) in [(b, FRAC_PI_2), (-b, -FRAC_PI_2), (b + 5000.0, FRAC_PI_2)] {
            let g = ecef2geodetic(DVec3::new(0.0, 0.0, z), &WGS84);
            assert_eq!(g.lat, lat);
            assert_eq!(g.lon, 0.0);
            assert!((g.h - (z.abs() - b)).abs() < 1e-9, "{g:?}");
        }
        // Equator.
        let g = ecef2geodetic(DVec3::new(0.0, -WGS84.a - 123.0, 0.0), &WGS84);
        assert_geo(g, 0.0, -FRAC_PI_2, 123.0, 0.0, 1e-9, "equator");
        // Centre of the earth: nearest surface points are the poles.
        let g = ecef2geodetic(DVec3::ZERO, &WGS84);
        assert!((g.lat.abs() - FRAC_PI_2).abs() < 1e-15 && (g.h + b).abs() < 1e-9, "{g:?}");
        // Equatorial plane inside the evolute (w < a e²): cos²φ = p(1-e²)/(e²(e²-p)).
        let w = 20000.0;
        let g = ecef2geodetic(DVec3::new(w, 0.0, 0.0), &WGS84);
        let e2 = WGS84.e2();
        let pp = (w / WGS84.a).powi(2);
        let lat = (pp * (1.0 - e2) / (e2 * (e2 - pp))).sqrt().acos();
        let hh = -WGS84.prime_vertical_radius(lat) * (1.0 - e2);
        assert_geo(g, lat, 0.0, hh, 1e-12, 1e-7, "singular disc");
        // Sphere.
        let s = Ellipsoid::sphere(6371000.0);
        let g = ecef2geodetic(DVec3::new(1.0e6, 2.0e6, -3.0e6), &s);
        let r = DVec3::new(1.0e6, 2.0e6, -3.0e6).length();
        assert_geo(g, (-3.0e6 / r).asin(), 2f64.atan(), r - 6371000.0, 1e-14, 1e-8, "sphere");
    }

    #[test]
    fn geodetic_ecef_round_trip_random() {
        let mut rng = Rng::new(42);
        let mut max_dh: f64 = 0.0;
        let mut max_dpos: f64 = 0.0;
        for i in 0..200_000 {
            let lat = match i % 50 {
                0 => FRAC_PI_2,
                1 => -FRAC_PI_2,
                2 => 0.0,
                _ => rng.uniform(-FRAC_PI_2, FRAC_PI_2),
            };
            let lon = rng.uniform(-PI, PI);
            let h = if i % 3 == 0 { rng.uniform(-5000.0, 20000.0) } else { rng.uniform(-5000.0, 1.0e7) };
            let g = Geodetic::new(lat, lon, h);
            let p = geodetic2ecef(g, &WGS84);
            let g2 = ecef2geodetic(p, &WGS84);
            let p2 = geodetic2ecef(g2, &WGS84);
            max_dh = max_dh.max((g2.h - h).abs());
            max_dpos = max_dpos.max((p2 - p).length());
            assert!((g2.lat - lat).abs() < 1e-14, "lat {g:?} -> {g2:?}");
            if lat.abs() < FRAC_PI_2 - 1e-9 {
                let dlon = (g2.lon - lon + PI).rem_euclid(2.0 * PI) - PI;
                assert!(dlon.abs() < 1e-14, "lon {g:?} -> {g2:?}");
            }
        }
        assert!(max_dh < 1e-8, "max |dh| = {max_dh:e}");
        assert!(max_dpos < 1e-8, "max |dpos| = {max_dpos:e}");
    }

    #[test]
    fn ecef2geodetic_random_interior_consistency() {
        // For arbitrary points (incl. deep interior), the result must reproduce the point.
        let mut rng = Rng::new(7);
        for _ in 0..100_000 {
            let r = rng.uniform(1.0e5, 2.0e7);
            let dir = DVec3::new(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)).normalize();
            let p = dir * r;
            let g = ecef2geodetic(p, &WGS84);
            let p2 = geodetic2ecef(g, &WGS84);
            assert!((p2 - p).length() < 1e-8 * r.max(WGS84.a) / WGS84.a * 10.0, "{p:?} -> {g:?} -> {p2:?}");
        }
    }

    /// Reference: pymap3d `geodetic2enu`, `geodetic2ned`, `geodetic2aer`, origin
    /// (39.9°, 32.8°, 900 m).
    #[test]
    fn geodetic2local_vs_pymap3d() {
        let o = Geodetic::from_deg(39.9, 32.8, 900.0);
        let cases = [
            (
                (39.95, 32.9, 3000.0),
                [8549.62037822379, 5559.062696769496, 2091.8527026138577],
                [5559.062696769496, 8549.62037822379, -2091.8527026138577],
                [0.9942723395133616, 0.20231743541061012, 10410.333059407203],
            ),
            (
                (39.0, 33.5, 100.0),
                [60637.92368060029, -99681.53402173566, -1868.932707654305],
                [-99681.53402173566, 60637.92368060029, 1868.932707654305],
                [2.595080496526937, -0.016016726673812695, 116691.29754478903],
            ),
            (
                (41.0, 30.0, 10000.0),
                [-235853.43528285058, 126029.5001058293, 3503.580195164359],
                [126029.5001058293, -235853.43528285058, -3503.580195164359],
                [5.203141508732269, 0.013100955827357255, 267437.00736029586],
            ),
        ];
        for ((la, lo, h), enu, ned, aer) in cases {
            let g = Geodetic::from_deg(la, lo, h);
            assert_vec(geodetic2enu(g, o, &WGS84), DVec3::from_array(enu), 1e-6, "geodetic2enu");
            assert_vec(geodetic2ned(g, o, &WGS84), DVec3::from_array(ned), 1e-6, "geodetic2ned");
            let a = geodetic2aer(g, o, &WGS84);
            let want = DVec3::from_array(aer);
            assert!((a.x - want.x).abs() < 1e-12 && (a.y - want.y).abs() < 1e-12, "aer {a:?} {want:?}");
            assert!((a.z - want.z).abs() < 1e-6, "aer {a:?} {want:?}");
            // LocalFrame must agree.
            let fe = LocalFrame::new(o, LocalConvention::Enu, WGS84);
            let fn_ = LocalFrame::new(o, LocalConvention::Ned, WGS84);
            assert_vec(fe.geodetic_to_local(g), DVec3::from_array(enu), 1e-6, "LocalFrame ENU");
            assert_vec(fn_.geodetic_to_local(g), DVec3::from_array(ned), 1e-6, "LocalFrame NED");
        }
    }

    /// Reference: pymap3d `enu2geodetic`, `ned2geodetic` (degrees out), origin
    /// (39.9°, 32.8°, 900 m).
    #[test]
    fn local2geodetic_vs_pymap3d() {
        let o = Geodetic::from_deg(39.9, 32.8, 900.0);
        let cases: [([f64; 3], [f64; 3], [f64; 3]); 3] = [
            ([1000.0, 2000.0, 300.0], [39.91800869432484, 32.811694269070756, 1200.3925914338054], [39.909003140027444, 32.82338767244412, 600.3916972637954]),
            (
                [-5000.0, 12000.0, -800.0],
                [40.00805862719269, 32.74144192272254, 113.27453314391312],
                [39.854895379067386, 32.9401914959106, 1713.2343829950223],
            ),
            (
                [25000.0, -40000.0, 9000.0],
                [39.53993646128127, 33.09036205847329, 10074.410524008921],
                [40.124487228633775, 32.33014066089345, -7925.4053253476195],
            ),
        ];
        let tol_rad = 1e-12;
        for (v, genu, gned) in cases {
            let v = DVec3::from_array(v);
            let g = enu2geodetic(v, o, &WGS84);
            assert_geo(g, genu[0].to_radians(), genu[1].to_radians(), genu[2], tol_rad, 1e-6, "enu2geodetic");
            let g = ned2geodetic(v, o, &WGS84);
            assert_geo(g, gned[0].to_radians(), gned[1].to_radians(), gned[2], tol_rad, 1e-6, "ned2geodetic");
            let fe = LocalFrame::new(o, LocalConvention::Enu, WGS84);
            let g = fe.local_to_geodetic(v);
            assert_geo(g, genu[0].to_radians(), genu[1].to_radians(), genu[2], tol_rad, 1e-6, "LocalFrame");
        }
    }

    /// Reference: pymap3d `aer2geodetic`, `aer2ecef` (radians), origin (39.9°, 32.8°, 900 m).
    #[test]
    fn aer2geodetic_vs_pymap3d() {
        let o = Geodetic::from_deg(39.9, 32.8, 900.0);
        let cases = [
            (
                [0.5235987755982988, -1.0471975511965976, 5000.0],
                [0.6967268659653847, 0.5727233146478786, -3429.636019739638],
                [4114583.1549833515, 2653154.988160305, 4068934.3830635413],
            ),
            (
                [4.363323129985824, 0.17453292519943295, 80000.0],
                [0.6921048411534693, 0.5574475591850269, 15276.859066005767],
                [4182811.3546358403, 2607562.467367871, 4058289.9286759216],
            ),
        ];
        for (aer, g, p) in cases {
            let aer = DVec3::from_array(aer);
            let got = aer2geodetic(aer, o, &WGS84);
            assert_geo(got, g[0], g[1], g[2], 1e-12, 1e-6, "aer2geodetic");
            let pe = aer2ecef(aer, o, &WGS84);
            assert_vec(pe, DVec3::from_array(p), 1e-6, "aer2ecef");
            let back = ecef2aer(pe, o, &WGS84);
            assert_vec(back, aer, 1e-9, "ecef2aer round trip");
        }
    }

    /// Reference: pymap3d `ecef2enu`, `ecef2ned`, `ecef2aer`, origin (39.9°, 32.8°, 900 m).
    #[test]
    fn ecef2local_vs_pymap3d() {
        let o = Geodetic::from_deg(39.9, 32.8, 900.0);
        let cases = [
            (
                [4198944.0, 2710080.0, 4078442.0],
                [3400.301484135751, -55807.02007172584, 79825.78574700335],
                [-55807.02007172584, 3400.301484135751, -79825.78574700335],
                [3.080738247738389, 0.9597974740939281, 97458.40964019221],
            ),
            (
                [4200000.0, 2700000.0, 4090000.0],
                [-5644.653749159319, -44006.91692731023, 83731.58279867425],
                [-44006.91692731023, -5644.653749159319, -83731.58279867425],
                [3.2691634969914762, 1.083533659636073, 94759.9536268688],
            ),
        ];
        for (p, enu, ned, aer) in cases {
            let p = DVec3::from_array(p);
            let (enu, ned, aer) = (DVec3::from_array(enu), DVec3::from_array(ned), DVec3::from_array(aer));
            assert_vec(ecef2enu(p, o, &WGS84), enu, 1e-6, "ecef2enu");
            assert_vec(ecef2ned(p, o, &WGS84), ned, 1e-6, "ecef2ned");
            let a = ecef2aer(p, o, &WGS84);
            assert!((a.x - aer.x).abs() < 1e-12 && (a.y - aer.y).abs() < 1e-12 && (a.z - aer.z).abs() < 1e-6);
            assert_vec(enu2ecef(enu, o, &WGS84), p, 1e-6, "enu2ecef");
            assert_vec(ned2ecef(ned, o, &WGS84), p, 1e-6, "ned2ecef");
            let f = LocalFrame::new(o, LocalConvention::Ned, WGS84);
            assert_vec(f.ecef_to_local(p), ned, 1e-6, "LocalFrame::ecef_to_local");
            assert_vec(f.local_to_ecef(ned), p, 1e-6, "LocalFrame::local_to_ecef");
        }
    }

    /// Reference: pymap3d `enu2aer` (radians) and `ecef2enuv` / `ecef2nedv`.
    #[test]
    fn aer_and_vectors_vs_pymap3d() {
        let cases = [
            ([100.0, 200.0, 30.0], [0.4636476090008061, 0.13336767777472905, 225.61028345356954]),
            ([-300.0, -50.0, -10.0], [4.547240302970063, -0.032867956565042586, 304.3024810940588]),
        ];
        for (enu, aer) in cases {
            let (enu, aer) = (DVec3::from_array(enu), DVec3::from_array(aer));
            assert_vec(enu2aer(enu), aer, 1e-12, "enu2aer");
            assert_vec(aer2enu(aer), enu, 1e-10, "aer2enu");
            let ned = DVec3::new(enu.y, enu.x, -enu.z);
            assert_vec(ned2aer(ned), aer, 1e-12, "ned2aer");
            assert_vec(aer2ned(aer), ned, 1e-10, "aer2ned");
        }
        // Due north / due west.
        assert_vec(enu2aer(DVec3::new(0.0, 10.0, 0.0)), DVec3::new(0.0, 0.0, 10.0), 0.0, "north");
        let w = enu2aer(DVec3::new(-10.0, 0.0, 0.0));
        assert!((w.x - 1.5 * PI).abs() < 1e-15);

        let (lat, lon) = (39.9f64.to_radians(), 32.8f64.to_radians());
        let v = DVec3::new(100.0, -200.0, 300.0);
        let enu = DVec3::new(-222.28414172741083, 245.72713811319565, 173.80429779268064);
        let ned = DVec3::new(245.72713811319565, -222.28414172741083, -173.80429779268064);
        assert_vec(ecef2enuv(v, lat, lon), enu, 1e-10, "ecef2enuv");
        assert_vec(ecef2nedv(v, lat, lon), ned, 1e-10, "ecef2nedv");
        assert_vec(enu2ecefv(enu, lat, lon), v, 1e-10, "enu2ecefv");
        assert_vec(ned2ecefv(ned, lat, lon), v, 1e-10, "ned2ecefv");
    }

    #[test]
    fn rotations_are_orthonormal_and_consistent() {
        let mut rng = Rng::new(3);
        for _ in 0..1000 {
            let lat = rng.uniform(-FRAC_PI_2, FRAC_PI_2);
            let lon = rng.uniform(-PI, PI);
            for r in [rot_ecef2enu(lat, lon), rot_ecef2ned(lat, lon)] {
                let i = r * r.transpose();
                assert!(i.abs_diff_eq(DMat3::IDENTITY, 1e-15));
                assert!((r.determinant() - 1.0).abs() < 1e-14);
            }
            let up = up_vector(lat, lon);
            assert_vec(rot_ecef2enu(lat, lon) * up, DVec3::Z, 1e-15, "up in ENU");
            assert_vec(rot_ecef2ned(lat, lon) * up, -DVec3::Z, 1e-15, "up in NED");
            // Up vector is the ellipsoid normal: parallel to gradient (x/a², y/a², z/b²).
            let p = geodetic2ecef(Geodetic::new(lat, lon, 0.0), &WGS84);
            let grad = DVec3::new(p.x / WGS84.a.powi(2), p.y / WGS84.a.powi(2), p.z / WGS84.b.powi(2)).normalize();
            assert_vec(grad, up, 1e-14, "normal");
        }
    }

    #[test]
    fn local_frame_round_trips() {
        let mut rng = Rng::new(11);
        for conv in [LocalConvention::Enu, LocalConvention::Ned] {
            for _ in 0..1000 {
                let o = Geodetic::new(rng.uniform(-1.5, 1.5), rng.uniform(-PI, PI), rng.uniform(-100.0, 3000.0));
                let f = LocalFrame::new(o, conv, WGS84);
                let l = DVec3::new(rng.uniform(-5e4, 5e4), rng.uniform(-5e4, 5e4), rng.uniform(-1e4, 1e4));
                assert_vec(f.ecef_to_local(f.local_to_ecef(l)), l, 1e-8, "local round trip");
                let g = f.local_to_geodetic(l);
                assert_vec(f.geodetic_to_local(g), l, 1e-8, "geodetic round trip");
                assert_vec(f.vec_ecef_to_local(f.vec_local_to_ecef(l)), l, 1e-9, "vec round trip");
                assert_vec(f.local_to_ecef(DVec3::ZERO), f.origin_ecef, 0.0, "origin");
                let g0 = f.local_to_geodetic(DVec3::ZERO);
                assert_geo(g0, o.lat, o.lon, o.h, 1e-14, 1e-8, "origin geodetic");
            }
        }
    }

    #[test]
    fn haversine() {
        let r = 6371000.0;
        let a = Geodetic::from_deg(0.0, 0.0, 0.0);
        let b = Geodetic::from_deg(0.0, 90.0, 0.0);
        assert!((haversine_distance(a, b, r) - r * FRAC_PI_2).abs() < 1e-6);
        let n = Geodetic::from_deg(90.0, 0.0, 0.0);
        let s = Geodetic::from_deg(-90.0, 0.0, 0.0);
        assert!((haversine_distance(n, s, r) - r * PI).abs() < 1e-6);
        // Ankara -> Istanbul, reference via Python haversine with r = 6371 km.
        let ank = Geodetic::from_deg(39.92, 32.85, 0.0);
        let ist = Geodetic::from_deg(41.0, 29.0, 0.0);
        assert!((haversine_distance(ank, ist, r) - 347_118.363_751_589_8).abs() < 1e-6);
    }

    #[test]
    fn ray_ellipsoid_hits() {
        let e = WGS84;
        // Straight down from above the north pole.
        let o = DVec3::new(0.0, 0.0, e.b + 10_000.0);
        assert!((ray_ellipsoid(o, -DVec3::Z, &e, 0.0).unwrap() - 10_000.0).abs() < 1e-8);
        assert!((ray_ellipsoid(o, -DVec3::Z, &e, 2_000.0).unwrap() - 8_000.0).abs() < 1e-8);
        // Looking up: miss.
        assert!(ray_ellipsoid(o, DVec3::Z, &e, 0.0).is_none());
        // Horizontal at the pole: misses the surface below.
        assert!(ray_ellipsoid(o, DVec3::X, &e, 0.0).is_none());
        // From inside: exit point.
        let t = ray_ellipsoid(DVec3::ZERO, DVec3::X, &e, 0.0).unwrap();
        assert!((t - e.a).abs() < 1e-8);
        // Nadir from 5 km above the equator.
        let o = DVec3::new(e.a + 5000.0, 0.0, 0.0);
        assert!((ray_ellipsoid(o, -DVec3::X, &e, 0.0).unwrap() - 5000.0).abs() < 1e-8);
        // Oblique ray from 3 km at 45°N: intersection lies on the surface.
        let mut rng = Rng::new(5);
        for _ in 0..1000 {
            let g = Geodetic::new(rng.uniform(-1.4, 1.4), rng.uniform(-PI, PI), rng.uniform(500.0, 10_000.0));
            let p = geodetic2ecef(g, &e);
            let az = rng.uniform(0.0, 2.0 * PI);
            let el = rng.uniform(-FRAC_PI_2, -0.3);
            let d = enu2ecefv(aer2enu(DVec3::new(az, el, 1.0)), g.lat, g.lon);
            let t = ray_ellipsoid(p, d, &e, 0.0).expect("downward ray must hit");
            let hit = ecef2geodetic(p + t * d, &e);
            assert!(hit.h.abs() < 1e-6, "{hit:?}");
            assert!(t >= g.h * 0.999 && t < g.h / 0.29, "t={t} h={}", g.h);
        }
    }

    #[test]
    fn horizon() {
        assert_eq!(horizon_distance(0.0, 6371000.0), 0.0);
        let d = horizon_distance(10_000.0, 6371000.0);
        assert!((d - (2.0f64 * 6371000.0 * 10_000.0 + 1e8).sqrt()).abs() < 1e-9);
        assert!((d - 357_099.425_930_650_3).abs() < 1e-6);
    }
}

/// Earth rotation rate (rad/s, WGS84).
pub const EARTH_RATE: f64 = 7.292115e-5;

/// WGS84 normal gravity magnitude (m/s²) at geodetic latitude `lat` (rad) and height `h` (m):
/// Somigliana's closed formula on the ellipsoid plus the second-order free-air correction.
/// Points along the ellipsoid normal (down); includes the centrifugal term.
pub fn normal_gravity(lat: f64, h: f64) -> f64 {
    let s2 = lat.sin().powi(2);
    let g0 = 9.780_325_335_9 * (1.0 + 0.001_931_852_652_41 * s2) / (1.0 - 0.006_694_379_990_13 * s2).sqrt();
    let a = 6_378_137.0;
    let f = 1.0 / 298.257_223_563;
    let m = 0.003_449_786_003_08;
    g0 * (1.0 - 2.0 / a * (1.0 + f + m - 2.0 * f * s2) * h + 3.0 * h * h / (a * a))
}

#[cfg(test)]
mod gravity_tests {
    #[test]
    fn normal_gravity_values() {
        // equator 9.7803, pole 9.8322, 45° ≈ 9.8062; −3.086e-6 /m free air
        assert!((super::normal_gravity(0.0, 0.0) - 9.780_325).abs() < 1e-5);
        assert!((super::normal_gravity(std::f64::consts::FRAC_PI_2, 0.0) - 9.832_185).abs() < 1e-5);
        let d = super::normal_gravity(0.7, 1000.0) - super::normal_gravity(0.7, 0.0);
        assert!((d + 3.086e-3).abs() < 2e-5, "{d}");
    }
}
