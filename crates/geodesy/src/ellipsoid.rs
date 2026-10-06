//! Reference ellipsoids of revolution (oblate spheroids).

use serde::{Deserialize, Serialize};

/// An ellipsoid of revolution defined by its semi-major axis `a` (equatorial radius) and
/// semi-minor axis `b` (polar radius), both in meters. A sphere has `a == b`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ellipsoid {
    /// Semi-major (equatorial) axis (m).
    pub a: f64,
    /// Semi-minor (polar) axis (m).
    pub b: f64,
}

impl Ellipsoid {
    /// WGS84: `a = 6378137 m`, `1/f = 298.257223563`.
    pub const WGS84: Ellipsoid = Ellipsoid {
        a: 6_378_137.0,
        b: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
    };

    /// Build from semi-major axis and inverse flattening. `inv_f` of `0` or `±inf`
    /// yields a sphere of radius `a`.
    pub fn from_a_invf(a: f64, inv_f: f64) -> Self {
        if inv_f == 0.0 || inv_f.is_infinite() {
            Self::sphere(a)
        } else {
            Self {
                a,
                b: a * (1.0 - 1.0 / inv_f),
            }
        }
    }

    /// Sphere of radius `r`.
    pub fn sphere(r: f64) -> Self {
        Self { a: r, b: r }
    }

    /// Flattening `f = (a - b) / a`.
    #[inline]
    pub fn f(&self) -> f64 {
        (self.a - self.b) / self.a
    }

    /// First eccentricity squared `e² = 1 - b²/a² = f (2 - f)`.
    #[inline]
    pub fn e2(&self) -> f64 {
        let f = self.f();
        f * (2.0 - f)
    }

    /// Second eccentricity squared `e'² = a²/b² - 1`.
    #[inline]
    pub fn ep2(&self) -> f64 {
        let e2 = self.e2();
        e2 / (1.0 - e2)
    }

    /// Prime-vertical radius of curvature `N(φ) = a / sqrt(1 - e² sin²φ)` (m).
    ///
    /// `N cos φ` is the radius of the parallel at geodetic latitude `φ`.
    #[inline]
    pub fn prime_vertical_radius(&self, lat: f64) -> f64 {
        let s = lat.sin();
        self.a / (1.0 - self.e2() * s * s).sqrt()
    }

    /// Meridian radius of curvature `M(φ) = a (1 - e²) / (1 - e² sin²φ)^{3/2}` (m).
    #[inline]
    pub fn meridian_radius(&self, lat: f64) -> f64 {
        let e2 = self.e2();
        let s = lat.sin();
        let w2 = 1.0 - e2 * s * s;
        self.a * (1.0 - e2) / (w2 * w2.sqrt())
    }

    /// IUGG mean radius `(2a + b) / 3` (m).
    #[inline]
    pub fn mean_radius(&self) -> f64 {
        (2.0 * self.a + self.b) / 3.0
    }
}

impl Default for Ellipsoid {
    fn default() -> Self {
        Self::WGS84
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgs84_constants() {
        let e = Ellipsoid::WGS84;
        assert_eq!(e.a, 6378137.0);
        assert!((e.b - 6356752.31424518).abs() < 1e-8);
        assert!((1.0 / e.f() - 298.257223563).abs() < 1e-7);
        assert!((e.e2() - 0.00669437999014).abs() < 1e-14);
        assert!((e.ep2() - 0.00673949674228).abs() < 1e-14);
        assert!((e.mean_radius() - 6371008.7714).abs() < 1e-3);
        assert_eq!(Ellipsoid::default(), Ellipsoid::WGS84);
    }

    #[test]
    fn from_a_invf_and_sphere() {
        let e = Ellipsoid::from_a_invf(6378137.0, 298.257223563);
        assert!((e.b - Ellipsoid::WGS84.b).abs() < 1e-9);
        for inv in [0.0, f64::INFINITY] {
            let s = Ellipsoid::from_a_invf(6371000.0, inv);
            assert_eq!(s, Ellipsoid::sphere(6371000.0));
            assert_eq!(s.e2(), 0.0);
            assert_eq!(s.prime_vertical_radius(0.7), 6371000.0);
            assert_eq!(s.meridian_radius(0.7), 6371000.0);
        }
    }

    #[test]
    fn radii_of_curvature() {
        let e = Ellipsoid::WGS84;
        let lat = 45f64.to_radians();
        // Reference: closed-form evaluated in Python (double precision).
        assert!((e.prime_vertical_radius(lat) - 6388838.290121148).abs() < 1e-6);
        assert!((e.meridian_radius(lat) - 6367381.815619548).abs() < 1e-6);
        // Equator: N = a, M = b²/a. Pole: N = M = a²/b.
        assert!((e.prime_vertical_radius(0.0) - e.a).abs() < 1e-9);
        assert!((e.meridian_radius(0.0) - e.b * e.b / e.a).abs() < 1e-6);
        let pole = std::f64::consts::FRAC_PI_2;
        assert!((e.prime_vertical_radius(pole) - e.a * e.a / e.b).abs() < 1e-6);
        assert!((e.meridian_radius(pole) - e.a * e.a / e.b).abs() < 1e-6);
    }
}
