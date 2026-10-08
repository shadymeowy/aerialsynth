//! Tiles from the GPU and the CPU generator side by side: `cargo run --release -p terragen
//! --example gpu_vs_cpu -- OUT.png z/x/y ...` writes one row per tile: GPU rgb, CPU rgb,
//! |difference| × 10, GPU emission (night lights).
use geodesy::tiles::TileId;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().cloned().unwrap_or_else(|| "gpu_vs_cpu.png".into());
    let ids: Vec<TileId> = args[1..]
        .iter()
        .map(|a| {
            let v: Vec<u32> = a.split('/').map(|s| s.parse().unwrap()).collect();
            TileId::new(v[0] as u8, v[1], v[2])
        })
        .collect();
    let cfg = terragen::Config::default();
    let gpu = terragen::gpu::GpuGenerator::new(cfg.clone())?;
    let cpu = terragen::Generator::with_backend(cfg, terragen::Backend::Cpu);
    let n = 256u32;
    let mut img = image::RgbImage::new(4 * n, n * ids.len() as u32);
    for (r, &id) in ids.iter().enumerate() {
        let g = gpu.tiles(&[id])?.remove(0);
        let c = cpu.tile_cpu(id);
        for k in 0..(n * n) as usize {
            let (x, y) = (k as u32 % n, k as u32 / n + r as u32 * n);
            let px = |v: &[u8]| image::Rgb([v[3 * k], v[3 * k + 1], v[3 * k + 2]]);
            img.put_pixel(x, y, px(&g.rgb));
            img.put_pixel(n + x, y, px(&c.rgb));
            let d = |ch: usize| ((g.rgb[3 * k + ch] as i32 - c.rgb[3 * k + ch] as i32).unsigned_abs() * 10).min(255) as u8;
            img.put_pixel(2 * n + x, y, image::Rgb([d(0), d(1), d(2)]));
            img.put_pixel(3 * n + x, y, px(&g.emission));
        }
        eprintln!("{id}");
    }
    img.save(&out)?;
    Ok(())
}
