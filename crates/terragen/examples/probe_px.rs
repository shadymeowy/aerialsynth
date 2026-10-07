// Dev check: terrain/climate values at a point: probe_px LAT LON [GSD]
// or list the rivers ending in closed basins within ~R km: probe_px sinks LAT LON R
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let g = terragen::Generator::new(terragen::Config::default());
    if args[0] == "sinks" {
        let a: Vec<f64> = args[1..].iter().map(|s| s.parse().unwrap()).collect();
        let ctx = terragen::world::Ctx::new(a[0].to_radians(), a[1].to_radians(), 50.0, &g.world.ell);
        for s in g.world.river_segments(ctx.p, a[2] * 1000.0, 50.0).iter().filter(|s| s.sink) {
            let ge = geodesy::ecef2geodetic(s.b, &g.world.ell);
            println!("sink {:.5} {:.5} level {} hw {:.1} h {:.0}", ge.lat.to_degrees(), ge.lon.to_degrees(), s.level, s.hw, s.hb);
        }
        return;
    }
    if args[0] == "scan" {
        // scan LAT0 LAT1 LON0 LON1 STEP: one line of features per grid point (coarse, 200 m)
        let a: Vec<f64> = args[1..].iter().map(|s| s.parse().unwrap()).collect();
        let mut lat = a[0];
        while lat <= a[1] {
            let mut lon = a[2];
            while lon <= a[3] {
                let (t, _, c) = g.probe(lat.to_radians(), lon.to_radians(), 200.0);
                println!(
                    "{lat:.3} {lon:.3} temp {:.1} moist {:.2} ground {:.0} mountain {:.2} mesa {:.2} sand {:.2} kind {} river_hw {:.0} agri {:.2} habit {:.2} class {c}",
                    t.temp, t.moist, t.ground, t.mountain, t.mesa, t.sand, t.water_kind, t.river_hw, t.agri, t.habit
                );
                lon += a[4];
            }
            lat += a[4];
        }
        return;
    }
    let a: Vec<f64> = args.iter().map(|s| s.parse().unwrap()).collect();
    let (t, h, c) = g.probe(a[0].to_radians(), a[1].to_radians(), a.get(2).copied().unwrap_or(5.0));
    println!(
        "temp {:.1} moist {:.2} agri {:.2} ground {:8.2} water {:8.2} kind {} cont {:.4} mountain {:.2} mesa {:.2} river_hw {:.1} river_d {:.0} dsm {h:.2} class {c}",
        t.temp, t.moist, t.agri, t.ground, t.water, t.water_kind, t.cont, t.mountain, t.mesa, t.river_hw, t.river_d
    );
}
