// Find example locations of each land-cover class on a coarse global grid.
fn main() {
    let g = terragen::Generator::new(terragen::Config::default());
    let mut found: std::collections::BTreeMap<u8, Vec<(f64, f64)>> = Default::default();
    for j in 0..140 {
        for i in 0..360 {
            let lat = -65.0 + j as f64;
            let lon = -180.0 + i as f64 + 0.37;
            let (_, _, c) = g.probe(lat.to_radians(), lon.to_radians(), 5.0);
            let v = found.entry(c).or_default();
            if v.len() < 400 { v.push((lat, lon)); }
        }
    }
    for (c, v) in &found {
        let pick: Vec<String> = v.iter().step_by((v.len() / 4).max(1)).take(4).map(|(a, b)| format!("{a:.1},{b:.1}")).collect();
        println!("{:10} n={:4} e.g. {}", terragen::landcover::NAMES[*c as usize], v.len(), pick.join("  "));
    }
}
