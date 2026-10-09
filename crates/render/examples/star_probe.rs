//! Apparent topocentric azimuth / elevation of the first catalogue stars, one line per star (for
//! validation against Skyfield).
//!
//!     cargo run --release -p render --example star_probe -- UNIX LAT_DEG LON_DEG H_M DUT1 [REFRACTION 0|1] [NSTARS]
use geodesy::frames::{ecef2enuv, geodetic2ecef, Geodetic};
use geodesy::Ellipsoid;
use render::stars::{StarField, StarsConfig};

fn main() -> anyhow::Result<()> {
    let a: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (unix, lat, lon, h, dut1) = (a[0], a[1].to_radians(), a[2].to_radians(), a[3], a[4]);
    let refraction = a.get(5).copied().unwrap_or(0.0) > 0.5;
    let n = a.get(6).copied().unwrap_or(400.0) as usize;
    let ell = Ellipsoid::from_a_invf(6378137.0, 298.257223563);
    let pos = geodetic2ecef(Geodetic::new(lat, lon, h), &ell);
    let cfg = StarsConfig { dut1_s: dut1, refraction, mag_limit: None, ..Default::default() };
    let f = StarField::new(&cfg)?;
    let dirs = f.apparent(unix, pos, &ell);
    for (i, d) in dirs.iter().enumerate().take(n) {
        let e = ecef2enuv(*d, lat, lon);
        let el = e.z.clamp(-1.0, 1.0).asin().to_degrees();
        let az = e.x.atan2(e.y).to_degrees().rem_euclid(360.0);
        println!("{i} {az:.9} {el:.9}");
    }
    Ok(())
}
