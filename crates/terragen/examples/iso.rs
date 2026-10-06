fn main() {
    let g = terragen::Generator::new(terragen::Config::default());
    let ell = g.world.ell;
    let n = 256;
    let mut img = vec![0u8; n * n];
    let (lat0, lon0) = ((-21.0f64).to_radians(), (-147.6f64).to_radians());
    let m = g.world.macro_at(terragen::world::Ctx::new(lat0, lon0, 2.4, &ell).p, 2.4);
    println!("style {:?}", m.style);
    for j in 0..n {
        for i in 0..n {
            let lat = lat0 - (j as f64 * 2.4) / 6.37e6;
            let lon = lon0 + (i as f64 * 2.4) / (6.37e6 * lat0.cos());
            let ctx = terragen::world::Ctx::new(lat, lon, 2.4, &ell);
            let d = g.world.dunes(&ctx, &m, 2.4);
            img[j * n + i] = (d * 200.0).clamp(0.0, 255.0) as u8;
        }
    }
    image::save_buffer("out/iso.png", &img, n as u32, n as u32, image::ExtendedColorType::L8).unwrap();
}
