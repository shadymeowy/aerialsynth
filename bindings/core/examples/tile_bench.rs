//! Timing of tile access through `aerialsynth-core` (the numbers of `bindings/README.md`).
//!
//!     cargo run --release -p aerialsynth-core --example tile_bench -- DIR [N]
//!
//! Uses the default world at zoom 12 (N tiles per measurement, default 32) in fresh tile stores
//! under DIR (removed first). CPU only: `VK_DRIVER_FILES=/nonexistent.json
//! VK_ICD_FILENAMES=/nonexistent.json`.

use aerialsynth_core::{Layer, World};
use std::path::Path;
use std::time::Instant;

const Z: u32 = 12;

/// `n` tiles of a block of 8 columns at zoom 12 starting at tile x0, y0 (Northern Italy).
fn block(x0: u32, y0: u32, n: usize) -> Vec<[u32; 3]> {
    (0..n as u32).map(|i| [Z, x0 + i % 8, y0 + i / 8]).collect()
}

fn ms_per_tile(t: Instant, n: usize) -> f64 {
    t.elapsed().as_secs_f64() * 1e3 / n as f64
}

fn per_tile(w: &World, ids: &[[u32; 3]], layer: Layer) -> f64 {
    let t = Instant::now();
    for &[z, x, y] in ids {
        std::hint::black_box(w.tile(z, x, y, layer).unwrap());
    }
    ms_per_tile(t, ids.len())
}

fn batch(w: &World, ids: &[[u32; 3]], layer: Layer) -> f64 {
    let mut out = vec![0u8; ids.len() * layer.tile_bytes()];
    let t = Instant::now();
    w.tiles_into(ids, layer, &mut out).unwrap();
    std::hint::black_box(&out);
    ms_per_tile(t, ids.len())
}

fn open(path: &Path, cache_mb: usize) -> World {
    let w = World::open(path, None, None).unwrap();
    w.set_cache_mb(cache_mb);
    w
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("usage: tile_bench DIR [N]"));
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(32);
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let (a, b, stored) = (dir.join("a.h5"), dir.join("b.h5"), dir.join("stored.h5"));
    let warm_up = |w: &World| w.tile(Z, 100, 100, Layer::Landcover).unwrap(); // GPU / generator start-up

    // generation of n missing tiles: one call per tile vs one batch (fresh stores, same tiles)
    let gen_ids = block(2160, 1470, n);
    let w = open(&a, 0); // (as before the cache existed)
    warm_up(&w);
    let one = per_tile(&w, &gen_ids, Layer::Rgb);
    drop(w);
    let w = open(&b, 256);
    warm_up(&w);
    let many = batch(&w, &gen_ids, Layer::Rgb);
    drop(w);
    println!("generate {n} missing z{Z} tiles (rgb): per tile {one:.1} ms/tile, batch {many:.1} ms/tile");

    // stored tiles: read right after opening (cold: decompressed), then again (warm: cached)
    let w = open(&stored, 256);
    let ids = block(2160, 1470, n);
    w.tiles_into(&ids, Layer::Rgb, &mut vec![0u8; n * Layer::Rgb.tile_bytes()]).unwrap();
    drop(w);
    for layer in [Layer::Rgb, Layer::Elevation, Layer::Landcover] {
        for cache_mb in [256, 0] {
            let w = open(&stored, cache_mb);
            let cold = per_tile(&w, &ids, layer);
            let warm = per_tile(&w, &ids, layer);
            drop(w);
            let w = open(&stored, cache_mb);
            let cold_batch = batch(&w, &ids, layer);
            let warm_batch = batch(&w, &ids, layer);
            drop(w);
            println!(
                "read {n} stored {:9} cache {cache_mb:3} MB: per tile cold {cold:.3} warm {warm:.3} ms/tile; batch cold {cold_batch:.3} warm {warm_batch:.3} ms/tile",
                layer.name()
            );
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}
