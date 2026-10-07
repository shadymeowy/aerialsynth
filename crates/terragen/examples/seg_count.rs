// Dev check: number of river pieces passed to a tile's river queries: seg_count Z X Y
fn main() {
    let a: Vec<u32> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let g = terragen::Generator::new(terragen::Config::default());
    let id = geodesy::tiles::TileId::new(a[0] as u8, a[1], a[2]);
    let b = id.bounds();
    let (lat, lon) = ((b.lat_min + b.lat_max) * 0.5, (b.lon_min + b.lon_max) * 0.5);
    let ctx = terragen::world::Ctx::new(lat, lon, 100.0, &g.world.ell);
    let size = (b.lon_max - b.lon_min).abs() * g.world.ell.a * lat.cos();
    let gsd = size / 256.0;
    println!("z{} pieces {}", a[0], g.world.river_segments(ctx.p, size, gsd).len());
}
