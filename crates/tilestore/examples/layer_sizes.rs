//! Compressed size of each layer of a tile store, over all its levels (what the optional layers
//! cost): `cargo run --release -p tilestore --example layer_sizes -- STORE.h5`.

use tilestore::{Layer, TileStore};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: layer_sizes STORE.h5"))?;
    let s = TileStore::open(&path)?;
    let levels = s.file().group("levels")?;
    let mut bytes = [0u64; Layer::ALL.len()];
    let mut tiles = 0usize;
    for z in s.zooms() {
        let g = levels.group(&z.to_string())?;
        tiles += s.tiles_at(z).len();
        for (i, l) in Layer::ALL.iter().enumerate() {
            if !s.meta().has(*l) || !g.exists(l.name()) {
                continue;
            }
            let ds = g.dataset(l.name())?;
            for row in 0..ds.shape()?[0] {
                let off = if l.channels() > 1 { vec![row, 0, 0, 0] } else { vec![row, 0, 0] };
                // (rows of an interrupted write have no chunk)
                if let Ok((_, b)) = ds.read_chunk_raw(&off) {
                    bytes[i] += b.len() as u64;
                }
            }
        }
    }
    let sum: u64 = bytes.iter().sum();
    println!("{path}: {tiles} tiles, {:.1} MB of tile layers", sum as f64 / 1e6);
    for (l, b) in Layer::ALL.iter().zip(bytes) {
        if b > 0 {
            println!("  {:10} {:8.1} MB {:5.1} %  {:6.0} KB / tile", l.name(), b as f64 / 1e6, 100.0 * b as f64 / sum as f64, b as f64 / 1e3 / tiles.max(1) as f64);
        }
    }
    Ok(())
}
