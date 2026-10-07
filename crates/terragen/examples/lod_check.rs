// Dev check: surface height (DSM) at fine vs coarse pixel sizes at random points around the given
// places; prints the worst mismatches (coarse − fine). lod_check LAT LON [LAT LON ...]
fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let g = terragen::Generator::new(terragen::Config::default());
    let mut rows = vec![];
    let mut h = 0x1234_5678u64;
    let mut rnd = || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        (h >> 11) as f64 / (1u64 << 53) as f64
    };
    for c in a.chunks(2) {
        for _ in 0..120 {
            let (lat, lon) = (c[0] + 0.08 * (rnd() - 0.5), c[1] + 0.1 * (rnd() - 0.5));
            let (_, fine, _) = g.probe(lat.to_radians(), lon.to_radians(), 2.0);
            for gsd in [32.0, 128.0, 512.0] {
                let (_, coarse, _) = g.probe(lat.to_radians(), lon.to_radians(), gsd);
                rows.push((coarse - fine, lat, lon, gsd));
            }
        }
    }
    rows.sort_by(|x, y| y.0.total_cmp(&x.0)); // coarse above fine first (walls)
    for r in rows.iter().take(15) {
        println!("{:+8.1} m at {:.5},{:.5} gsd {}", r.0, r.1, r.2, r.3);
    }
    let big = rows.iter().filter(|r| r.0 > 25.0).count();
    println!("{} of {} samples: coarse more than 25 m above fine", big, rows.len());
}
