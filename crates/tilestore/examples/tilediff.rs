//! Compare the tiles two stores have in common, layer by layer (regression checks of the
//! generator: outputs must stay bit-identical): `cargo run --release -p tilestore --example
//! tilediff -- A.h5 B.h5`. Prints the differing tiles and layers; exits 1 if any differ.

use tilestore::{Layer, TileStore};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [a, b] = &args[..] else { anyhow::bail!("usage: tilediff A.h5 B.h5") };
    let (sa, sb) = (TileStore::open(a)?, TileStore::open(b)?);
    let (mut same, mut diff, mut only) = (0usize, 0usize, 0usize);
    for id in sa.tiles() {
        if !sb.contains(id) {
            only += 1;
            continue;
        }
        let (ta, tb) = (sa.read_tile(id, &Layer::ALL)?.unwrap(), sb.read_tile(id, &Layer::ALL)?.unwrap());
        let mut bad = Vec::new();
        for l in Layer::ALL {
            let (x, y) = (ta.layer_bytes(l), tb.layer_bytes(l));
            if x != y {
                let n = x.iter().zip(y.iter()).filter(|(p, q)| p != q).count();
                bad.push(format!("{} ({n} bytes)", l.name()));
            }
        }
        if bad.is_empty() {
            same += 1;
        } else {
            diff += 1;
            println!("{}/{}/{}: {}", id.z, id.x, id.y, bad.join(", "));
        }
    }
    println!("{same} identical, {diff} different, {only} only in {a}, {} only in {b}", sb.tiles().iter().filter(|t| !sa.contains(**t)).count());
    if diff > 0 {
        std::process::exit(1);
    }
    Ok(())
}
