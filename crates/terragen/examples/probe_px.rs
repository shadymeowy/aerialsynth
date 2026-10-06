// Dev check: pass-A terrain and surface at a point: probe_px LAT LON [GSD]
fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let g = terragen::Generator::new(terragen::Config::default());
    let (t, h, c) = g.probe(a[0].to_radians(), a[1].to_radians(), a.get(2).copied().unwrap_or(5.0));
    println!(
        "temp {:.1} moist {:.2} agri {:.2} ground {:8.2} water {:8.2} kind {} cont {:.4} mountain {:.2} mesa {:.2} river_hw {:.1} river_d {:.0} dsm {h:.2} class {c}",
        t.temp, t.moist, t.agri, t.ground, t.water, t.water_kind, t.cont, t.mountain, t.mesa, t.river_hw, t.river_d
    );
}
