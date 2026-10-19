//! Terrain heights at points: `lat lon [gsd_m]` per line on stdin (degrees; default 30 m),
//! `surface ground water_kind` per line on stdout (m above the ellipsoid); with `--cover`, the
//! land-cover class name and the height of what stands on the ground (trees, buildings; m) at
//! the given gsd as fourth and fifth columns (CPU, slower). An optional positional argument sets
//! the world seed; `--generator classic|next` the generator (default: the default config's,
//! classic, the one `terrain run` uses unless `world.generator` says otherwise). Used by
//! `showcase/route.py` for the airliner's airports and terrain clearance.
//!
//!     echo "47.0 8.0" | cargo run --release -p terragen --example ground -- [SEED] [--cover] [--generator classic|next]
use std::io::{BufRead, Write};

fn main() -> anyhow::Result<()> {
    let mut cfg = terragen::Config::default();
    let mut cover = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--cover" => cover = true,
            "--generator" => cfg.generator = args.next().ok_or_else(|| anyhow::anyhow!("--generator needs classic or next"))?.parse()?,
            _ => cfg.seed = a.parse()?,
        }
    }
    let g = terragen::Generator::new(cfg);
    let mut pts = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let v: Vec<f64> = line?.split_whitespace().map(|s| s.parse()).collect::<Result<_, _>>()?;
        if v.len() >= 2 {
            pts.push((v[0].to_radians(), v[1].to_radians(), v.get(2).copied().unwrap_or(30.0)));
        }
    }
    let out = g.terrain_points(&pts)?;
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    let covers: Vec<(u8, f64)> = if cover {
        use rayon::prelude::*;
        pts.par_iter()
            .map(|p| {
                let (t, h, c) = g.probe(p.0, p.1, p.2);
                (c, h - t.ground)
            })
            .collect()
    } else {
        Vec::new()
    };
    for (i, t) in out.iter().enumerate() {
        write!(w, "{:.2} {:.2} {}", t.surface(), t.ground, t.water_kind)?;
        if let Some((c, above)) = covers.get(i) {
            write!(w, " {} {:.2}", terragen::landcover::name(*c), above)?;
        }
        writeln!(w)?;
    }
    Ok(())
}
