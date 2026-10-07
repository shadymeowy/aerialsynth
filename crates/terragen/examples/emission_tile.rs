// Dev check: albedo | emission | height above local ground (0..25 m) | hillshaded albedo of one
// tile as a PNG:
// emission_tile Z X Y OUT.png
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (z, x, y): (u8, u32, u32) = (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap());
    let g = terragen::Generator::new(terragen::Config::default());
    let t = g.tile(geodesy::tiles::TileId::new(z, x, y));
    let n = 256i32;
    let e = |i: i32, j: i32| t.elevation[(j.clamp(0, n - 1) * n + i.clamp(0, n - 1)) as usize];
    let mut img = image::RgbImage::new(1024, 256);
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
            // hillshade from the tile normals (sun from the north-west, 45°)
            let nv = [t.normal[3 * k] as f32 / 127.0, t.normal[3 * k + 1] as f32 / 127.0, t.normal[3 * k + 2] as f32 / 127.0];
            let l = [-0.5f32, 0.5, 0.7071];
            let shade = (0.25 + 0.95 * (nv[0] * l[0] + nv[1] * l[1] + nv[2] * l[2]).max(0.0)).min(1.4);
            let px = [0, 1, 2].map(|c| (t.albedo[3 * k + c] as f32 * shade).min(255.0) as u8);
            img.put_pixel(768 + i as u32, j as u32, image::Rgb(px));
        }
    }
    img.save(&a[4]).unwrap();
}
