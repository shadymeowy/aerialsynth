//! Terrain / climate values of the default world at points:
//!
//!     cargo run --release -p terragen --example probe_px -- LAT LON [GSD_M]
//!     cargo run --release -p terragen --example probe_px -- scan LAT0 LAT1 LON0 LON1 STEP_DEG
//!     cargo run --release -p terragen --example probe_px -- sinks LAT LON R_KM
//!     cargo run --release -p terragen --example probe_px -- info LAT LON [LAT LON ...]
//!     cargo run --release -p terragen --example probe_px -- bands
//!
//! The first prints the world fields at one point (default 5 m pixels); `scan` prints one line of
//! features per grid point (200 m pixels) to look for places; `sinks` lists the rivers ending in
//! closed basins within about `R_KM` km.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let g = terragen::NextGenerator::new(terragen::Config::default());
    if args[0] == "info" {
        // info LAT LON [LAT LON ...]: ecoregion, biome and land use
        let a: Vec<f64> = args[1..].iter().map(|s| s.parse().unwrap()).collect();
        for ll in a.chunks(2) {
            println!("{:.4} {:.4} {}", ll[0], ll[1], g.describe(ll[0].to_radians(), ll[1].to_radians(), 5.0));
        }
        return;
    }
    if args[0] == "bands" {
        // bands: per 10° latitude band, the land's Köppen classes and the biomes an ecoregion
        // would pick at sites on a 1° grid
        let reg = &g.surface.registry;
        let w = &g.world;
        let mut lat0 = -60;
        while lat0 < 80 {
            let mut kc = std::collections::BTreeMap::<String, usize>::new();
            let mut bc = std::collections::BTreeMap::<String, usize>::new();
            let mut n = 0usize;
            for lat in (lat0..lat0 + 10).map(|l| l as f64 + 0.5) {
                for lon in (-180..180).map(|l| l as f64 + 0.5) {
                    let site = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &w.ell);
                    if w.atlas().sample(site.normalize()).coast_km <= 0.0 {
                        continue;
                    }
                    n += 1;
                    let id = terragen::noise::mix64((((lat * 10.0) as i64 as u64) << 20) ^ (lon * 10.0) as i64 as u64);
                    let e = terragen::eco::compute(w, reg, id, site);
                    *kc.entry(format!("{:?}", terragen::atlas::Koppen::ALL[e.climate.koppen as usize])).or_default() += 1;
                    *bc.entry(reg.biomes[e.biome as usize].name.clone()).or_default() += 1;
                }
            }
            let fmt = |m: &std::collections::BTreeMap<String, usize>| {
                let mut v: Vec<_> = m.iter().collect();
                v.sort_by(|a, b| b.1.cmp(a.1));
                v.iter().take(10).map(|(k, c)| format!("{k} {:.0}", 100.0 * **c as f64 / n.max(1) as f64)).collect::<Vec<_>>().join(", ")
            };
            println!("{lat0:+}..{:+} land {n}\n  koppen: {}\n  biomes: {}", lat0 + 10, fmt(&kc), fmt(&bc));
            lat0 += 10;
        }
        return;
    }
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
