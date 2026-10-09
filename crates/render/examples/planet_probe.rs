//! Apparent topocentric azimuth / elevation (no refraction), V magnitude, radius and phase of the
//! planets and the Moon, one line per body (for validation against Skyfield).
//!
//!     cargo run --release -p render --example planet_probe -- UNIX LAT_DEG LON_DEG H_M DUT1
use geodesy::frames::{ecef2enuv, geodetic2ecef, Geodetic};
use geodesy::Ellipsoid;
use render::stars::astro::Sky;
use render::stars::ephem::Body;
use render::stars::planets;

fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (unix, lat, lon, h, dut1) = (a[0], a[1].to_radians(), a[2].to_radians(), a[3], a[4]);
    let pos = geodetic2ecef(Geodetic::new(lat, lon, h), &Ellipsoid::WGS84);
    let sky = Sky::new(unix, dut1, 0.0, 0.0);
    let mut bodies = planets::PLANETS.to_vec();
    bodies.push(Body::Moon);
    for p in planets::apparent(&sky, pos, &bodies).expect("date outside the ephemeris") {
        let e = ecef2enuv(sky.gcrs_to_itrs * p.dir, lat, lon);
        let el = e.z.clamp(-1.0, 1.0).asin().to_degrees();
        let az = e.x.atan2(e.y).to_degrees().rem_euclid(360.0);
        println!("{} {az:.9} {el:.9} {:.4} {:.6} {:.4}", p.body.naif(), p.v, p.radius.to_degrees() * 3600.0, p.phase);
    }
}
