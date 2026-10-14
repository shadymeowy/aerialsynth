//! Terrain / climate values of the default world at points:
//!
//!     cargo run --release -p terragen --example probe_px -- LAT LON [GSD_M]
//!     cargo run --release -p terragen --example probe_px -- scan LAT0 LAT1 LON0 LON1 STEP_DEG
//!     cargo run --release -p terragen --example probe_px -- sinks LAT LON R_KM
//!     cargo run --release -p terragen --example probe_px -- info LAT LON [LAT LON ...]
//!     cargo run --release -p terragen --example probe_px -- bands
//!     cargo run --release -p terragen --example probe_px -- towns LAT LON R_KM [MIN_LEVEL]
//!     cargo run --release -p terragen --example probe_px -- audit N SEED
//!     cargo run --release -p terragen --example probe_px -- slopes LAT LON KM STEP_M [classic]
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
    if args[0] == "towns" {
        // towns LAT LON R_KM [MIN_LEVEL]: the settlements within R_KM km (level ≥ MIN_LEVEL,
        // default 2: towns), with the terrain at their centre (to find A/B places)
        let a: Vec<f64> = args[1..].iter().map(|s| s.parse().unwrap()).collect();
        let ctx = terragen::world::Ctx::new(a[0].to_radians(), a[1].to_radians(), 50.0, &g.world.ell);
        let min = a.get(3).copied().unwrap_or(2.0) as u8;
        for s in terragen::kits::settlements::places::settlements_near_min_level(&g.world, ctx.p, a[2] * 1000.0, min) {
            let (t, _, c) = g.probe(s.lat_deg.to_radians(), s.lon_deg.to_radians(), 200.0);
            println!(
                "{:.4} {:.4} level {} pop {:.0} R {:.0} arch {} dev {:.2} coastal {} temp {:.1} moist {:.2} agri {:.2} ground {:.0} class {c}",
                s.lat_deg, s.lon_deg, s.level, s.population, s.radius_m, s.archetype, s.development, s.coastal, t.temp, t.moist, t.agri, t.ground
            );
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
    if args[0] == "slopes" {
        // slopes LAT LON KM STEP_M [classic]: the slope distribution of the bare ground over a
        // KM x KM box (central differences of pass A's ground STEP_M apart)
        let a: Vec<f64> = args[1..5].iter().map(|s| s.parse().unwrap()).collect();
        let classic = args.get(5).is_some_and(|s| s == "classic");
        let cg = classic.then(|| {
            let mut cfg = terragen::Config::default();
            cfg.generator = terragen::GeneratorKind::Classic;
            terragen::classic::Generator::with_backend(cfg, terragen::Backend::Cpu)
        });
        let n = (a[2] * 1000.0 / a[3]) as i64;
        let (dlat, dlon) = (a[3] / 111_200.0, a[3] / (111_200.0 * a[0].to_radians().cos()));
        let h = |i: i64, j: i64| {
            let (la, lo) = ((a[0] + (i - n / 2) as f64 * dlat).to_radians(), (a[1] + (j - n / 2) as f64 * dlon).to_radians());
            match &cg {
                Some(c) => c.probe(la, lo, a[3]).0.ground,
                None => g.probe(la, lo, a[3]).0.ground,
            }
        };
        let grid: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| h(i, j)).collect()).collect();
        let mut sl = vec![];
        for i in 1..n as usize - 1 {
            for j in 1..n as usize - 1 {
                let gx = (grid[i][j + 1] - grid[i][j - 1]) / (2.0 * a[3]);
                let gy = (grid[i + 1][j] - grid[i - 1][j]) / (2.0 * a[3]);
                sl.push(gx.hypot(gy).atan().to_degrees());
            }
        }
        sl.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let q = |f: f64| sl[((sl.len() - 1) as f64 * f) as usize];
        let mean = sl.iter().sum::<f64>() / sl.len() as f64;
        let share = |d: f64| sl.iter().filter(|&&x| x > d).count() as f64 / sl.len() as f64;
        let (mn, mx) = grid.iter().flatten().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        println!(
            "slope mean {mean:.1} p50 {:.1} p90 {:.1} p99 {:.1}; >30 {:.2} >40 {:.2} >50 {:.2}; ground {mn:.0}..{mx:.0} m",
            q(0.5),
            q(0.9),
            q(0.99),
            share(30.0),
            share(40.0),
            share(50.0)
        );
        return;
    }
    if args[0] == "audit" {
        // audit N SEED: the A/B audit's random land places (CSV)
        audit(&g, args[1].parse().unwrap(), args[2].parse().unwrap());
        return;
    }
    let a: Vec<f64> = args.iter().map(|s| s.parse().unwrap()).collect();
    let (t, h, c) = g.probe(a[0].to_radians(), a[1].to_radians(), a.get(2).copied().unwrap_or(5.0));
    println!(
        "temp {:.1} moist {:.2} agri {:.2} ground {:8.2} water {:8.2} kind {} cont {:.4} mountain {:.2} mesa {:.2} river_hw {:.1} river_d {:.0} dsm {h:.2} class {c}",
        t.temp, t.moist, t.agri, t.ground, t.water, t.water_kind, t.cont, t.mountain, t.mesa, t.river_hw, t.river_d
    );
}

/// `audit N SEED`: `N` deterministic pseudo-random land places of the next world (the A/B
/// audit, `scripts/ab_v020/audit_places.csv`) as CSV. Candidates are area-uniform within
/// |lat| <= 70°; a candidate is kept when the next world has land there at least 2 km from the
/// sea (no ocean, lake or river at the point), its Köppen group still has room (quotas A 6, B 8,
/// C 7, D 6, E 3: every climate, roughly in proportion to the land), and it is at least 400 km
/// from the places kept before it. The classic (v0.2.0) world has other continents: its place
/// is the same point when that is land there, else the nearest classic land point (rings of
/// 16 directions, 10 km apart).
fn audit(g: &terragen::NextGenerator, n: usize, seed: u64) {
    use terragen::noise::mix64;
    let mut cfg = terragen::Config::default();
    cfg.generator = terragen::GeneratorKind::Classic;
    let cg = terragen::classic::Generator::with_backend(cfg, terragen::Backend::Cpu);
    let classic_land = |lat: f64, lon: f64| {
        let (t, _, _) = cg.probe(lat.to_radians(), lon.to_radians(), 30.0);
        t.water_kind == terragen::classic::world::water::NONE && t.cont > 0.0
    };
    let quota = |grp: char| match grp {
        'A' => 6,
        'B' => 8,
        'C' => 7,
        'D' => 6,
        _ => 3,
    };
    let unit = |k: u64| (mix64(seed ^ mix64(k)) >> 11) as f64 / (1u64 << 53) as f64;
    let dist_km = |a: (f64, f64), b: (f64, f64)| {
        let (p1, p2) = (a.0.to_radians(), b.0.to_radians());
        let c = p1.sin() * p2.sin() + p1.cos() * p2.cos() * (a.1 - b.1).to_radians().cos();
        6371.0 * c.clamp(-1.0, 1.0).acos()
    };
    let mut kept: Vec<(f64, f64)> = vec![];
    let mut count = std::collections::HashMap::<char, usize>::new();
    println!("id,lat,lon,koppen,biome,ground_m,classic_lat,classic_lon,classic_shift_km");
    let mut k = 0u64;
    while kept.len() < n && k < 200_000 {
        let (u, v) = (unit(2 * k), unit(2 * k + 1));
        k += 1;
        let lat = (2.0 * u - 1.0).asin().to_degrees();
        let lon = 360.0 * v - 180.0;
        if lat.abs() > 70.0 || kept.iter().any(|&p| dist_km(p, (lat, lon)) < 400.0) {
            continue;
        }
        let (t, _, _) = g.probe(lat.to_radians(), lon.to_radians(), 30.0);
        if t.coast_km < 2.0 || t.water_kind != terragen::world::water::NONE {
            continue;
        }
        let e = terragen::eco::compute(&g.world, &g.surface.registry, t.eco.id, t.eco.center);
        let kop = terragen::atlas::Koppen::ALL[e.climate.koppen as usize].code();
        let grp = kop.chars().next().unwrap();
        if count.get(&grp).copied().unwrap_or(0) >= quota(grp) {
            continue;
        }
        // the classic world's place: the same point or the nearest land point
        let mut cp = None;
        'rings: for ring in 0..80 {
            let r = 10.0 * ring as f64;
            for d in 0..if ring == 0 { 1 } else { 16 } {
                let az = (d as f64 * 22.5).to_radians();
                let clat = lat + r / 111.2 * az.cos();
                let clon = lon + r / (111.2 * lat.to_radians().cos().max(0.2)) * az.sin();
                if classic_land(clat, clon) {
                    cp = Some((clat, clon, r));
                    break 'rings;
                }
            }
        }
        let Some((clat, clon, r)) = cp else { continue };
        *count.entry(grp).or_default() += 1;
        kept.push((lat, lon));
        println!(
            "a{:02},{lat:.4},{lon:.4},{kop},{},{:.0},{clat:.4},{clon:.4},{r:.0}",
            kept.len(),
            g.surface.registry.biomes[e.biome as usize].name,
            t.ground
        );
    }
}
