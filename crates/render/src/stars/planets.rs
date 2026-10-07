//! Planets and the Moon: apparent topocentric directions from the DE440 ephemeris (light time,
//! light deflection by the Sun, aberration with the observer's orbital + diurnal velocity), V
//! magnitudes (Mallama & Hilton 2018, as in the Astronomical Almanac and Skyfield), phase angle and
//! apparent radius. Positions are those of the planet system barycentres (Jupiter's centre is
//! within ~200 km of it, 0.07″).

use super::astro::{aberrate, Sky};
use super::ephem::{Body, Ephemeris};
use glam::DVec3;

const AU_KM: f64 = 149_597_870.7;
const C_KM_DAY: f64 = 299_792.458 * 86400.0;
const SRS: f64 = 1.974_125_743_36e-8;

/// The bodies drawn as point sources / discs (the Moon is drawn by the sky shader).
pub const PLANETS: [Body; 7] = [Body::Mercury, Body::Venus, Body::Mars, Body::Jupiter, Body::Saturn, Body::Uranus, Body::Neptune];

/// Ground-truth id of a solar-system body: 1<<30 | NAIF id (stars: HIP number or 1<<31 | Tycho).
pub fn body_id(b: Body) -> u32 {
    (1 << 30) | b.naif()
}

/// Equatorial radius (km).
fn radius_km(b: Body) -> f64 {
    match b {
        Body::Sun => 695_700.0,
        Body::Mercury => 2_439.7,
        Body::Venus => 6_051.8,
        Body::Earth => 6_378.1,
        Body::Moon => 1_737.4,
        Body::Mars => 3_396.2,
        Body::Jupiter => 71_492.0,
        Body::Saturn => 60_268.0,
        Body::Uranus => 25_559.0,
        Body::Neptune => 24_764.0,
    }
}

/// Mean B−V (reflected sunlight) for the colour.
pub fn colour_bv(b: Body) -> f64 {
    match b {
        Body::Mercury => 0.93,
        Body::Venus => 0.82,
        Body::Mars => 1.36,
        Body::Jupiter => 0.83,
        Body::Saturn => 1.04,
        Body::Uranus => 0.56,
        Body::Neptune => 0.41,
        Body::Moon => 0.92,
        _ => 0.65,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Apparent {
    pub body: Body,
    /// apparent direction, GCRS axes (before rotation to ITRS and refraction)
    pub dir: DVec3,
    /// distance from the observer (AU, light-time corrected) and from the Sun (AU)
    pub delta: f64,
    pub r: f64,
    /// phase angle (deg): Sun–body–observer
    pub phase: f64,
    /// apparent equatorial radius (rad)
    pub radius: f64,
    pub v: f64,
}

/// ERFA ld: deflection of the direction `p` to a body at unit vector `q` from the deflector
/// (seen from the deflector), observer at unit vector `e` and distance `em` (AU) from it.
fn deflect_body(p: DVec3, q: DVec3, e: DVec3, em: f64) -> DVec3 {
    let dlim = 1e-6 / (em * em).max(1.0);
    let w = SRS / em / q.dot(q + e).max(dlim);
    p + p.cross(e.cross(q)) * w
}

/// Apparent places of `bodies` for an observer at ECEF `obs` (m) at the instant of `sky`.
/// None outside the ephemeris' dates (1990–2060).
pub fn apparent(sky: &Sky, obs: DVec3, bodies: &[Body]) -> Option<Vec<Apparent>> {
    let eph = Ephemeris::builtin();
    let jd = sky.jd_tt;
    let (pe, ve) = eph.barycentric(Body::Earth, jd)?;
    let (ps, _) = eph.barycentric(Body::Sun, jd)?;
    // observer: geocentre + its GCRS offset (km), velocity orbital + diurnal (units of c)
    let m_t = sky.gcrs_to_itrs.transpose();
    let po = pe + m_t * obs / 1000.0;
    let vobs = ve / C_KM_DAY + (sky.observer_vel_c(obs) - sky.earth_vel_c);
    let e_sun = po - ps;
    let em = e_sun.length() / AU_KM;
    let mut out = Vec::with_capacity(bodies.len());
    for &b in bodies {
        // light time
        let mut tau = 0.0;
        let mut pb = DVec3::ZERO;
        for _ in 0..3 {
            pb = eph.barycentric(b, jd - tau)?.0;
            tau = (pb - po).length() / C_KM_DAY;
        }
        let ps_ret = eph.barycentric(Body::Sun, jd - tau)?.0;
        let rel = pb - po;
        let delta = rel.length() / AU_KM;
        let sun_body = pb - ps_ret;
        let r = sun_body.length() / AU_KM;
        let mut u = rel.normalize();
        if b != Body::Sun {
            u = deflect_body(u, sun_body.normalize(), e_sun.normalize(), em);
        }
        let u = aberrate(u, vobs, em);
        let phase = (-sun_body).angle_between(-rel).to_degrees();
        let radius = (radius_km(b) / rel.length()).asin();
        let v = magnitude(b, r, delta, phase, -rel.normalize(), -sun_body.normalize());
        out.push(Apparent { body: b, dir: u, delta, r, phase, radius, v });
    }
    Some(out)
}

/// Unit vector of an IAU pole (right ascension, declination in degrees).
fn pole(ra: f64, de: f64) -> DVec3 {
    let (ra, de) = (ra.to_radians(), de.to_radians());
    DVec3::new(de.cos() * ra.cos(), de.cos() * ra.sin(), de.sin())
}

/// V magnitude (Mallama & Hilton 2018). `to_obs`, `to_sun`: unit vectors from the body to the
/// observer and the Sun (Saturn's ring tilt, Uranus' sub-observer / sub-solar latitudes).
pub fn magnitude(b: Body, r: f64, delta: f64, a: f64, to_obs: DVec3, to_sun: DVec3) -> f64 {
    let d = 5.0 * (r * delta).log10();
    let poly = |c: &[f64]| c.iter().rev().fold(0.0, |s, k| s * a + k);
    d + match b {
        Body::Mercury => poly(&[-0.613, 6.3280e-02, -1.6336e-03, 3.3644e-05, -3.4265e-07, 1.6893e-09, -3.0334e-12]),
        Body::Venus => {
            if a <= 163.7 {
                poly(&[-4.384, -1.044e-03, 3.687e-04, -2.814e-06, 8.938e-09])
            } else {
                poly(&[236.05828, -2.81914, 8.39034e-03])
            }
        }
        Body::Mars => {
            if a <= 50.0 {
                poly(&[-1.601, 2.267e-02, -1.302e-04])
            } else {
                poly(&[-0.367, -0.02573, 3.445e-04])
            }
        }
        Body::Jupiter => {
            if a <= 12.0 {
                poly(&[-9.395, -3.7e-04, 6.16e-04])
            } else {
                let x = a / 180.0;
                -9.428 - 2.5 * (1.0 - 1.507 * x - 0.363 * x * x - 0.062 * x.powi(3) + 2.809 * x.powi(4) - 1.876 * x.powi(5)).log10()
            }
        }
        Body::Saturn => {
            // ring plane tilt towards the observer (IAU pole: α 40.589°, δ 83.537°)
            let sb = pole(40.589, 83.537).dot(to_obs).abs();
            -8.914 - 1.825 * sb + 0.026 * a - 0.378 * sb * (-2.25 * a).exp()
        }
        Body::Uranus => {
            // mean of the planetographic sub-observer and sub-solar latitudes (deg; flattening 0.0229)
            let p = pole(257.311, -15.175);
            let graphic = |u: DVec3| (p.dot(u).clamp(-1.0, 1.0).asin().tan() / (1.0f64 - 0.0229).powi(2)).atan().to_degrees().abs();
            poly(&[-7.110, 6.587e-03, 1.045e-04]) - 8.4e-04 * 0.5 * (graphic(to_obs) + graphic(to_sun))
        }
        Body::Neptune => -7.00,
        // Allen (2000), at the mean distance, scaled to the actual one
        Body::Moon => -d - 12.73 + 0.026 * a + 4e-9 * a.powi(4) + 5.0 * (delta * AU_KM / 384_400.0).log10(),
        Body::Sun => -26.74 - d + 5.0 * delta.log10(),
        Body::Earth => 0.0,
    }
}
