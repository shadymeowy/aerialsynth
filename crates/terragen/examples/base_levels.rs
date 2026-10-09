//! Time the viewer's base levels: every tile of z0..=Z (default 4) with a fresh generator, level
//! by level in batches, as `terrain view` generates them on an empty store.
//!
//!     cargo run --release -p terragen --example base_levels -- [Z] [BATCH] [cpu|gpu|auto]
use geodesy::tiles::TileId;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let zmax: u8 = args.first().map(|s| s.parse()).transpose()?.unwrap_or(4);
    let batch: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(16);
    let backend = match args.get(2).map(|s| s.as_str()) {
        Some("cpu") => terragen::Backend::Cpu,
        Some("gpu") => terragen::Backend::Gpu,
        _ => terragen::Backend::Auto,
    };
    let t0 = Instant::now();
    let gen = terragen::Generator::try_with_backend(terragen::Config::default(), backend)?;
    eprintln!("generator on {} ready in {:.2} s", gen.backend_name(), t0.elapsed().as_secs_f64());
    let t_all = Instant::now();
    for z in 0..=zmax {
        let ids: Vec<TileId> = (0..1u32 << z).flat_map(|y| (0..1u32 << z).map(move |x| TileId::new(z, x, y))).collect();
        let t = Instant::now();
        let mut worst = 0f64;
        for chunk in ids.chunks(batch) {
            let tb = Instant::now();
            gen.tiles(chunk)?;
            worst = worst.max(tb.elapsed().as_secs_f64());
        }
        let dt = t.elapsed().as_secs_f64();
        eprintln!(
            "z{z}: {:4} tiles in {:6.2} s ({:6.2} tiles/s, slowest batch {:.2} s), total {:.2} s",
            ids.len(),
            dt,
            ids.len() as f64 / dt,
            worst,
            t_all.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
