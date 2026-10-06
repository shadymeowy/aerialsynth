// Find towns: scan a grid and print locations classified as building.
fn main() {
    let g = terragen::Generator::new(terragen::Config::default());
    let mut found = 0;
    for j in 0..120 {
        for i in 0..120 {
            let lat = 39.6 + j as f64 * 0.006;
            let lon = 32.5 + i as f64 * 0.006;
            let (_, _, c) = g.probe(lat.to_radians(), lon.to_radians(), 1.0);
            if c == terragen::landcover::BUILDING || c == terragen::landcover::URBAN {
                if found % 15 == 0 { println!("{lat:.4},{lon:.4}"); }
                found += 1;
            }
        }
    }
    eprintln!("{found} urban samples");
}
