// Dev check: albedo | emission | height above local ground (0..25 m) of one tile as a PNG:
// emission_tile Z X Y OUT.png
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (z, x, y): (u8, u32, u32) = (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap());
    let g = terragen::Generator::new(terragen::Config::default());
    let t = g.tile(geodesy::tiles::TileId::new(z, x, y));
    let n = 256i32;
    let e = |i: i32, j: i32| t.elevation[(j.clamp(0, n - 1) * n + i.clamp(0, n - 1)) as usize];
    let mut img = image::RgbImage::new(768, 256);
    for j in 0..n {
        for i in 0..n {
            let k = (j * n + i) as usize;
            img.put_pixel(i as u32, j as u32, image::Rgb([t.albedo[3 * k], t.albedo[3 * k + 1], t.albedo[3 * k + 2]]));
            img.put_pixel(256 + i as u32, j as u32, image::Rgb([t.emission[3 * k], t.emission[3 * k + 1], t.emission[3 * k + 2]]));
            let mut lo = f32::MAX;
            for dj in -12..=12 {
                for di in -12..=12 {
                    lo = lo.min(e(i + di, j + dj));
                }
            }
            let v = ((e(i, j) - lo) / 25.0).clamp(0.0, 1.0);
            img.put_pixel(512 + i as u32, j as u32, image::Rgb([(255.0 * v) as u8, (255.0 * v.powf(2.0)) as u8, (255.0 * (1.0 - v) * v * 2.0) as u8]));
        }
    }
    img.save(&a[4]).unwrap();
}
