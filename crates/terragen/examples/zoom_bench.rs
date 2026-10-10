//! Per-zoom benchmark and stills of fixed places (the performance budget of
//! `docs/design/terrain-next.md` §9, and a visual before / after check).
//!
//!     cargo run --release -p terragen --example zoom_bench -- [cpu|gpu] [OPTIONS]
//!
//! * `--zooms 4,8,12,14,16`: the zoom levels timed;
//! * `--per-place N`: tiles per place and zoom (an N x N block around the place, default 2);
//! * `--places a,b,..`: only these places (names from `PLACES`);
//! * `--png DIR`: also write each place's block as `DIR/<place>_z<zoom>.png` (rgb);
//! * `--find N`: instead, sample N random land points and print a few per landscape kind
//!   (to choose places).
//!
//! Tiles are generated in batches of 16 (the GPU batch); the first batch (pipeline set-up, caches)
//! is timed separately.
use geodesy::tiles::{tile_for_latlon, TileId};
use std::time::Instant;
use terragen::{Generator, TileData, TILE_SIZE};

/// Places of the default world (seed 1): name, lat, lon (deg).
const PLACES: &[(&str, f64, f64)] = &[
    ("home_town", 39.894, 32.912),
    ("north_town", 41.562, 32.955),
    ("coast", 40.951, 33.889),
    ("farmland", 43.584, 116.529),
    ("desert_sand", 21.389, 47.868),
    ("desert_bare", -22.936, 23.655),
    ("rainforest", 6.950, -71.649),
    ("savanna", -11.866, -14.744),
    ("steppe", 28.195, 46.717),
    ("boreal", 68.543, -161.880),
    ("tundra", -72.030, -97.544),
    ("mountains", -17.737, 80.319),
    ("mesa", -19.593, -145.629),
];

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let backend = match args.first().map(|s| s.as_str()) {
        Some("cpu") => terragen::Backend::Cpu,
        Some("gpu") => terragen::Backend::Gpu,
        _ => terragen::Backend::Auto,
    };
    let cfg = terragen::Config::default();
    if let Some(n) = arg(&args, "--find") {
        return find(cfg, n.parse()?);
    }
    let zooms: Vec<u8> = arg(&args, "--zooms").unwrap_or("4,8,12,14,16").split(',').map(|z| z.trim().parse()).collect::<Result<_, _>>()?;
    let per: u32 = arg(&args, "--per-place").map(|s| s.parse()).transpose()?.unwrap_or(2);
    let only: Option<Vec<&str>> = arg(&args, "--places").map(|s| s.split(',').collect());
    let png = arg(&args, "--png").map(std::path::PathBuf::from);
    let places: Vec<(&str, f64, f64)> = PLACES.iter().copied().filter(|p| only.as_ref().is_none_or(|o| o.contains(&p.0))).collect();
    let t0 = Instant::now();
    let gen = Generator::try_with_backend(cfg, backend)?;
    gen.tiles(&[TileId::new(3, 4, 3)])?;
    eprintln!("generator on {} ready in {:.2} s (incl. one tile)", gen.backend_name(), t0.elapsed().as_secs_f64());
    if let Some(d) = &png {
        std::fs::create_dir_all(d)?;
    }
    println!("| zoom | tiles | seconds | tiles/s | ms/tile |");
    println!("|---|---|---|---|---|");
    for &z in &zooms {
        let mut ids = Vec::new();
        let mut blocks = Vec::new();
        for &(name, lat, lon) in &places {
            let c = tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
            let n = 1i64 << z;
            let h = per as i64 / 2;
            let first = ids.len();
            for dy in 0..per as i64 {
                for dx in 0..per as i64 {
                    let x = (c.x as i64 + dx - h).rem_euclid(n) as u32;
                    let y = (c.y as i64 + dy - h).clamp(0, n - 1) as u32;
                    ids.push(TileId::new(z, x, y));
                }
            }
            blocks.push((name, first));
        }
        let t = Instant::now();
        let tiles = gen.tiles(&ids)?;
        let dt = t.elapsed().as_secs_f64();
        println!("| z{z} | {} | {dt:.2} | {:.2} | {:.1} |", ids.len(), ids.len() as f64 / dt, 1000.0 * dt / ids.len() as f64);
        if let Some(d) = &png {
            for (name, first) in blocks {
                let block = &tiles[first..first + (per * per) as usize];
                mosaic(block, per as usize).save(d.join(format!("{name}_z{z}.png")))?;
            }
        }
    }
    Ok(())
}

fn mosaic(tiles: &[TileData], per: usize) -> image::RgbImage {
    let n = TILE_SIZE;
    let mut img = image::RgbImage::new((per * n) as u32, (per * n) as u32);
    for (k, t) in tiles.iter().enumerate() {
        let (tx, ty) = (k % per, k / per);
        for j in 0..n {
            for i in 0..n {
                let o = 3 * (j * n + i);
                img.put_pixel((tx * n + i) as u32, (ty * n + j) as u32, image::Rgb([t.rgb[o], t.rgb[o + 1], t.rgb[o + 2]]));
            }
        }
    }
    img
}

/// Random land points by landscape kind (pass A at 100 m pixels).
fn find(cfg: terragen::Config, n: usize) -> anyhow::Result<()> {
    use rayon::prelude::*;
    let w = terragen::world::World::new(cfg);
    let kinds: Vec<(usize, f64, f64, terragen::world::Terrain)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let h = terragen::noise::mix64(0xF1ED ^ i as u64);
            let lat = (2.0 * terragen::noise::u01k(h, 1) - 1.0).asin().to_degrees();
            let lon = 360.0 * terragen::noise::u01k(h, 2) - 180.0;
            let t = w.terrain(&terragen::world::Ctx::new(lat.to_radians(), lon.to_radians(), 100.0, &w.ell));
            (i, lat, lon, t)
        })
        .collect();
    type IsClass = fn(&terragen::world::Terrain) -> bool;
    let classes: [(&str, IsClass); 10] = [
        ("desert_sand", |t| t.sand > 0.5),
        ("desert_bare", |t| t.moist < 0.15 && t.sand < 0.2 && t.temp > 10.0),
        ("rainforest", |t| t.temp > 22.0 && t.moist > 0.7),
        ("savanna", |t| t.temp > 20.0 && (0.25..0.5).contains(&t.moist)),
        ("steppe", |t| (5.0..18.0).contains(&t.temp) && (0.2..0.35).contains(&t.moist)),
        ("farmland", |t| t.agri > 0.5),
        ("boreal", |t| (-4.0..4.0).contains(&t.temp) && t.moist > 0.5),
        ("tundra", |t| t.temp < -4.0 && t.mountain < 0.2),
        ("mountains", |t| t.mountain > 0.6),
        ("mesa", |t| t.mesa > 0.4),
    ];
    for (name, f) in classes {
        let hits: Vec<String> = kinds
            .iter()
            .filter(|(_, _, _, t)| t.water_kind == 0 && t.ground > 5.0 && f(t))
            .take(4)
            .map(|(_, la, lo, t)| format!("({la:.3}, {lo:.3}) T {:.0} M {:.2} h {:.0}", t.temp, t.moist, t.ground))
            .collect();
        println!("{name}: {}", hits.join("; "));
    }
    Ok(())
}
