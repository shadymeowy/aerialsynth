//! Tests of the transport kit.

use super::build::{self, DS};
use super::net::{self, network as nw};
use super::*;
use crate::features::CpuPoints;
use crate::noise::*;
use crate::world::World;

fn world() -> World {
    World::new(crate::Config::default())
}

/// Places of the default world where the kit draws: (name, lat, lon), for the tests and stills
/// (`find_transport_places`).
pub(crate) const PLACES: &[(&str, f64, f64)] = &[("interchange", 28.6579, 178.6129), ("city", -6.1773, -114.2395), ("town", -12.198, -21.4536)];

fn tile_at(name: &str, z: u8) -> geodesy::tiles::TileId {
    let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == name).unwrap();
    geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)
}

fn ecef(w: &World, lat: f64, lon: f64) -> DVec3 {
    geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &w.ell)
}

/// Populated land points of the default world.
fn land_points(w: &World, n: usize, pop: f64, seed: u64) -> Vec<DVec3> {
    let mut v = Vec::new();
    for i in 0..(n as u64) * 600 {
        let h = mix64(seed ^ i);
        let lat = (2.0 * u01k(h, 1) - 1.0).asin();
        let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
        let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
        let a = w.atlas().sample(p.normalize());
        if a.coast_km > 20.0 && a.population > pop {
            v.push(p);
            if v.len() == n {
                break;
            }
        }
    }
    v
}

#[test]
fn legacy_roads_are_off() {
    assert!(!crate::kits::legacy_roads());
}

#[test]
fn registry_compiles_with_the_transport_kit() {
    crate::registry::Registry::builtin().unwrap();
}

/// The box filters integrate exactly: a band of half width hw covers 2 hw, a dash pattern its
/// duty cycle, at any footprint.
#[test]
fn box_filters_are_exact() {
    for &fw in &[0.05, 0.3, 1.0, 7.0, 40.0] {
        let n = 4000;
        let span = 60.0 + 4.0 * fw;
        let mut acc = 0.0;
        let mut dsh = 0.0;
        for i in 0..n {
            let d = -span + 2.0 * span * (i as f64 + 0.5) / n as f64;
            acc += px::boxcov(d, 3.5, fw) * 2.0 * span / n as f64;
            dsh += px::dash(d * 7.0 + 3.0, 12.0, 4.0, fw) / n as f64;
        }
        assert!((acc - 7.0).abs() < 0.02 * 7.0 + 1e-3, "fw {fw}: {acc}");
        assert!((dsh - 4.0 / 12.0).abs() < 0.02, "fw {fw}: {dsh}");
    }
}

/// The networks are relative-neighbourhood graphs: symmetric, no node closer to both ends of an
/// edge than they are to each other, the same after the caches are cleared.
#[test]
fn graph_is_a_relative_neighbourhood_graph() {
    let w = world();
    for p in land_points(&w, 4, 0.2, 0x7A0) {
        for n in [nw::ARTERIAL, nw::REGIONAL, nw::LOCAL] {
            let nodes = net::nodes(&w, n, p, 60_000.0);
            for a in nodes.iter().take(6) {
                let nb = net::neighbours(&w, n, a);
                for b in nb.iter() {
                    let back = net::neighbours(&w, n, b);
                    assert!(back.iter().any(|x| x.id == a.id), "net {n}: {} → {} not symmetric", a.id, b.id);
                    let d = (a.p - b.p).length();
                    for c in net::nodes(&w, n, (a.p + b.p) * 0.5, d) {
                        if c.id != a.id && c.id != b.id {
                            assert!((c.p - a.p).length() >= d || (c.p - b.p).length() >= d, "net {n}: blocked edge");
                        }
                    }
                }
            }
        }
    }
    let p = land_points(&w, 1, 0.2, 0x7A1)[0];
    let a = net::edges_near(&w, nw::REGIONAL, p, 30_000.0);
    net::clear_cache();
    let b = net::edges_near(&w, nw::REGIONAL, p, 30_000.0);
    assert_eq!(a, b);
}

/// Edges near overlapping discs agree: an edge seen from both is the same edge, and an edge
/// whose chord passes near both centres is seen from both.
#[test]
fn edges_are_tile_independent() {
    let w = world();
    for p in land_points(&w, 3, 0.2, 0x7A2) {
        let (_, _, east, _) = crate::kits::settlements::places::frame_at(&w, p);
        let q = route::surface(&w, p + east * 8_000.0);
        for n in [nw::REGIONAL, nw::LOCAL, nw::HAMLET] {
            let a = net::edges_near(&w, n, p, 6_000.0);
            let b = net::edges_near(&w, n, q, 6_000.0);
            for e in &a {
                if let Some(o) = b.iter().find(|x| x.key == e.key) {
                    assert_eq!(e, o);
                }
                let mid = (p + q) * 0.5;
                if net::chord_dist(e.a.p, e.b.p, mid) < 500.0 && net::chord_dist(e.a.p, e.b.p, q) < 2_000.0 {
                    assert!(b.iter().any(|x| x.key == e.key), "net {n}: edge {} not seen from the neighbour", e.key);
                }
            }
        }
    }
}

/// Routes are pure functions of the edge: the same polyline after the caches are cleared, ends
/// at the nodes.
#[test]
fn routes_are_deterministic() {
    let w = world();
    let p = land_points(&w, 1, 0.2, 0x7A3)[0];
    let edges = net::edges_near(&w, nw::LOCAL, p, 10_000.0);
    assert!(!edges.is_empty());
    let a: Vec<_> = edges.iter().take(4).map(|e| build::routed(&w, e)).collect();
    build::clear_cache();
    for (e, ra) in edges.iter().take(4).zip(&a) {
        let rb = build::routed(&w, e);
        match (ra, &rb) {
            (Some(x), Some(y)) => {
                assert_eq!(x.pts, y.pts);
                assert!((x.pts[0] - e.a.p).length() < 1.0 && (x.pts.last().unwrap() - e.b.p).length() < 1.0);
            }
            (None, None) => {}
            _ => panic!("route existence differs"),
        }
    }
}

/// Profiles are local operators: a stretch computed alone has the same decks as within a longer
/// one, and the deck climbs no steeper than the grade limit (plus the sample rounding).
#[test]
fn profiles_are_local_and_grade_limited() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let mut tested = 0;
    for p in land_points(&w, 12, 0.2, 0x7A4) {
        let edges = net::edges_near(&w, nw::REGIONAL, p, 5_000.0);
        let Some(e) = edges.iter().find(|e| e.class == cls::TRUNK) else { continue };
        let Some(line) = build::routed(&w, e) else { continue };
        let s_mid = 0.5 * line.len();
        let a = build::profile(&mut ctx, e, &line, s_mid - 600.0, s_mid + 600.0).unwrap();
        let b = build::profile(&mut ctx, e, &line, s_mid - 3_000.0, s_mid + 3_000.0).unwrap();
        for j in a.j0..a.j0 + a.deck.len() as i64 {
            let da = a.deck[(j - a.j0) as usize];
            let db = b.deck[(j - b.j0) as usize];
            assert!((da - db).abs() < 1e-9, "deck at sample {j}: {da} vs {db}");
        }
        // (the grade limit holds where the terrain falls less than the limit allows over the
        // radius: most of the way; steeper anywhere is a tunnel or a bridge)
        let mut g: Vec<f64> = (1..b.deck.len()).map(|k| (b.deck[k] - b.deck[k - 1]).abs() / DS).collect();
        g.sort_by(|x, y| x.total_cmp(y));
        let p90 = g[g.len() * 9 / 10];
        assert!(p90 < 0.065 + 0.01, "90th percentile grade {p90}");
        eprintln!("trunk deck grades: median {:.3}, 90 % {p90:.3}, steepest {:.3}", g[g.len() / 2], g[g.len() - 1]);
        assert!(g[g.len() - 1] < 0.5, "steepest {}", g[g.len() - 1]);
        tested += 1;
        if tested >= 2 {
            break;
        }
    }
    assert!(tested > 0);
}

/// The features of overlapping areas agree where both have them (the same segments bit for
/// bit), and twice the same area (cleared caches) gives the same features.
#[test]
fn features_are_tile_independent() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let mut tested = 0;
    for p in land_points(&w, 6, 0.25, 0x7A5) {
        let (_, _, east, _) = crate::kits::settlements::places::frame_at(&w, p);
        let q = route::surface(&w, p + east * 1_500.0);
        let fa = build::features(&mut ctx, &Area { center: p, radius: 2_000.0, gsd: 10.0 });
        let fb = build::features(&mut ctx, &Area { center: q, radius: 2_000.0, gsd: 10.0 });
        if fa.segs.is_empty() {
            continue;
        }
        let mut shared = 0;
        for s in &fa.segs {
            let mid = (s.a + s.b) * 0.5;
            if (mid - q).length() < 1_000.0 {
                let o = fb.segs.iter().find(|x| (x.a - s.a).length() < 1e-6 && (x.b - s.b).length() < 1e-6 && x.kind == s.kind);
                let o = o.unwrap_or_else(|| panic!("segment of kind {} at {:?} missing from the neighbour", s.kind, mid));
                assert_eq!((o.flags, o.s0.to_bits(), o.v.map(f32::to_bits)), (s.flags, s.s0.to_bits(), s.v.map(f32::to_bits)));
                assert!(o.deck_a.to_bits() == s.deck_a.to_bits() && o.deck_b.to_bits() == s.deck_b.to_bits());
                shared += 1;
            }
        }
        net::clear_cache();
        build::clear_cache();
        let fa2 = build::features(&mut ctx, &Area { center: p, radius: 2_000.0, gsd: 10.0 });
        assert_eq!(fa.segs.len(), fa2.segs.len());
        assert_eq!(fa.stamps.len(), fa2.stamps.len());
        if shared > 0 {
            tested += 1;
        }
        if tested >= 2 {
            break;
        }
    }
    assert!(tested > 0, "no overlapping features found");
}

/// Debug: where the kit draws what in the default world: `find_transport_places -- --ignored
/// --nocapture`.
#[test]
#[ignore]
fn find_transport_places() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let mut found: std::collections::BTreeMap<String, Vec<(f64, f64)>> = Default::default();
    let note = |found: &mut std::collections::BTreeMap<String, Vec<(f64, f64)>>, what: &str, p: DVec3| {
        let g = geodesy::ecef2geodetic(p, &w.ell);
        let c = (g.lat.to_degrees(), g.lon.to_degrees());
        let v = found.entry(what.to_string()).or_default();
        if v.len() < 6 && !v.iter().any(|x| (x.0 - c.0).abs() + (x.1 - c.1).abs() < 0.05) {
            v.push(c);
        }
    };
    let mut cities: Vec<net::Node> = Vec::new();
    for p in land_points(&w, 30, 0.2, 0x7A6) {
        for c in net::nodes(&w, nw::ARTERIAL, p, 150_000.0) {
            if !cities.iter().any(|x| x.id == c.id) {
                cities.push(c);
            }
        }
        if cities.len() > 24 {
            break;
        }
    }
    for c in &cities {
        let t0 = std::time::Instant::now();
        let f = build::features(&mut ctx, &Area { center: c.p, radius: 25_000.0, gsd: 30.0 });
        eprintln!("city {} level {} dev {:.2}: {} segs {} stamps in {:.1} s", c.id, c.level, c.dev, f.segs.len(), f.stamps.len(), t0.elapsed().as_secs_f64());
        for s in &f.segs {
            let k = (s.kind - cls::KIND0) as usize;
            let mid = (s.a + s.b) * 0.5;
            if k == cls::RAMP {
                note(&mut found, "ramp", mid);
            }
            if s.flags & flag::TUNNEL != 0 {
                note(&mut found, &format!("tunnel_{k}"), mid);
            }
            if s.flags & flag::BRIDGE != 0 && s.flags & flag::OVER == 0 {
                note(&mut found, &format!("bridge_{k}"), mid);
            }
            if k == cls::POWER {
                let a = w.atlas().sample(mid.normalize());
                if a.precip_mm > 900.0 && a.temp_c > 4.0 {
                    note(&mut found, "power_wet", mid);
                }
            }
            if k == cls::MOTORWAY && s.hb > 0.0 {
                note(&mut found, "lit_motorway", mid);
            }
            if k == cls::TRACK {
                note(&mut found, "track", mid);
            }
        }
        for st in &f.stamps {
            note(&mut found, &format!("stamp_{}", st.template), st.center);
        }
    }
    for (k, v) in &found {
        let l: Vec<String> = v.iter().map(|(a, b)| format!("({a:.4}, {b:.4})")).collect();
        eprintln!("{k}: {}", l.join(" "));
    }
}

/// Debug: render tiles to PNG (`TR_OUT` dir, `TR_SPEC` "name:lat,lon,z;…", 2 × 2 tiles each,
/// `TR_GPU=1` on the GPU, `TR_NIGHT=1` the emission): `render_tiles -- --ignored --nocapture`.
#[test]
#[ignore]
fn render_tiles() {
    let out = std::path::PathBuf::from(std::env::var("TR_OUT").unwrap_or("/tmp/transport".into()));
    std::fs::create_dir_all(&out).unwrap();
    let spec = std::env::var("TR_SPEC").unwrap_or_default();
    let backend = if std::env::var("TR_GPU").is_ok() { crate::Backend::Gpu } else { crate::Backend::Cpu };
    let night = std::env::var("TR_NIGHT").is_ok();
    let gen = crate::Generator::try_with_backend(crate::Config::default(), backend).unwrap();
    for item in spec.split(';').filter(|s| !s.is_empty()) {
        let (name, rest) = item.split_once(':').unwrap();
        let v: Vec<f64> = rest.split(',').map(|x| x.trim().parse().unwrap()).collect();
        let z = v[2] as u8;
        let px = geodesy::tiles::latlon_to_pixel(v[0].to_radians(), v[1].to_radians(), z, crate::TILE_SIZE as u32);
        let (x0, y0) = ((px.x / crate::TILE_SIZE as f64 - 0.5).floor() as u32, (px.y / crate::TILE_SIZE as f64 - 0.5).floor() as u32);
        let ids: Vec<_> = (0..4).map(|k| geodesy::tiles::TileId::new(z, x0 + (k % 2), y0 + (k / 2))).collect();
        let t0 = std::time::Instant::now();
        let tiles = gen.tiles(&ids).unwrap();
        let n = crate::TILE_SIZE;
        let mut img = image::RgbImage::new(2 * n as u32, 2 * n as u32);
        for (k, t) in tiles.iter().enumerate() {
            for j in 0..n {
                for i in 0..n {
                    let o = 3 * (j * n + i);
                    let px = if night {
                        let e = |c: usize| (255.0 * (t.emission[o + c] as f64 / 255.0).sqrt()) as u8;
                        [e(0), e(1), e(2)]
                    } else {
                        [t.rgb[o], t.rgb[o + 1], t.rgb[o + 2]]
                    };
                    img.put_pixel(((k % 2) * n + i) as u32, ((k / 2) * n + j) as u32, image::Rgb(px));
                }
            }
        }
        img.save(out.join(format!("{name}_z{z}{}.png", if night { "_night" } else { "" }))).unwrap();
        eprintln!("{name} z{z}: {:.1} s", t0.elapsed().as_secs_f64());
    }
}

/// Parent ≈ mean of children where the kit draws (elevation, albedo).
#[test]
fn lod_consistent_at_places() {
    let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
    let n = crate::TILE_SIZE;
    for &(name, z) in &[("interchange", 13u8), ("city", 14), ("town", 15)] {
        let parent = tile_at(name, z);
        let p = g.tile(parent);
        let kids = parent.children().map(|c| g.tile(c));
        let (mut eh, mut ea, mut cnt) = (0.0, 0.0, 0.0);
        for (k, kid) in kids.iter().enumerate() {
            let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
            for j in (0..n).step_by(2) {
                for i in (0..n).step_by(2) {
                    let ix = [j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1];
                    let m: f32 = ix.iter().map(|&q| kid.elevation[q]).sum::<f32>() / 4.0;
                    let pi = (oy + j / 2) * n + ox + i / 2;
                    eh += (m - p.elevation[pi]).abs() as f64;
                    for c in 0..3 {
                        let ma: f64 = ix.iter().map(|&q| kid.albedo[3 * q + c] as f64).sum::<f64>() / 4.0;
                        ea += (ma - p.albedo[3 * pi + c] as f64).abs();
                    }
                    cnt += 1.0;
                }
            }
        }
        let (eh, ea) = (eh / cnt, ea / (3.0 * cnt));
        eprintln!("{name} z{z}: |parent − mean(children)| elevation {eh:.2} m, albedo {ea:.2} DN");
        assert!(eh < 6.0, "{name}: elevation {eh}");
        assert!(ea < 14.0, "{name}: albedo {ea}");
    }
}

/// No seam between neighbouring tiles where the kit draws (E-W and N-S).
#[test]
fn seamless_at_places() {
    let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
    let n = crate::TILE_SIZE;
    for &(name, z) in &[("interchange", 15u8), ("city", 14), ("town", 16)] {
        let id = tile_at(name, z);
        let a = g.tile(id);
        let e = g.tile(geodesy::tiles::TileId::new(z, id.x + 1, id.y));
        let s = g.tile(geodesy::tiles::TileId::new(z, id.x, id.y + 1));
        let (mut ea, mut ei, mut na, mut ni) = (0.0, 0.0, 0.0, 0.0);
        for j in 0..n {
            for c in 0..3 {
                ea += (a.albedo[3 * (j * n + n - 1) + c] as f64 - e.albedo[3 * (j * n) + c] as f64).abs();
                ei += (a.albedo[3 * (j * n + n - 2) + c] as f64 - a.albedo[3 * (j * n + n - 1) + c] as f64).abs();
                na += (a.albedo[3 * ((n - 1) * n + j) + c] as f64 - s.albedo[3 * j + c] as f64).abs();
                ni += (a.albedo[3 * ((n - 2) * n + j) + c] as f64 - a.albedo[3 * ((n - 1) * n + j) + c] as f64).abs();
                ei += (e.albedo[3 * (j * n) + c] as f64 - e.albedo[3 * (j * n + 1) + c] as f64).abs();
                ni += (s.albedo[3 * j + c] as f64 - s.albedo[3 * (n + j) + c] as f64).abs();
            }
        }
        eprintln!("{name} z{z}: albedo E-W across {ea:.0} inside {ei:.0}; N-S across {na:.0} inside {ni:.0}");
        assert!(ea < 2.0 * ei + 2.0 * n as f64, "{name}: E-W across {ea} inside {ei}");
        assert!(na < 2.0 * ni + 2.0 * n as f64, "{name}: N-S across {na} inside {ni}");
    }
}

/// Tiles are a pure function of the world: the same tile twice (fresh generators, cleared
/// caches) is the same bits.
#[test]
fn deterministic_at_places() {
    let cfg = crate::Config { tile_supersample: 1, ..crate::Config::default() };
    let a = crate::Generator::new(cfg.clone());
    let list = [("interchange", 15u8), ("town", 16)];
    let ta: Vec<_> = list.iter().map(|&(nm, z)| a.tile(tile_at(nm, z))).collect();
    net::clear_cache();
    build::clear_cache();
    let b = crate::Generator::new(cfg);
    for (k, &(nm, z)) in list.iter().enumerate() {
        let y = b.tile(tile_at(nm, z));
        let x = &ta[k];
        assert!(x.elevation == y.elevation && x.albedo == y.albedo && x.landcover == y.landcover && x.emission == y.emission, "{nm}");
    }
}

/// Share of samples differing by more than `tol`.
#[cfg(feature = "gpu")]
fn off<T: Copy>(a: &[T], b: &[T], tol: f64, f: impl Fn(T) -> f64) -> f64 {
    a.iter().zip(b).filter(|(x, y)| (f(**x) - f(**y)).abs() > tol).count() as f64 / a.len() as f64
}

/// The GPU's tiles are the CPU's where the kit draws.
#[cfg(feature = "gpu")]
#[test]
fn tiles_match_the_cpu() {
    let cfg = crate::Config::default();
    let gen = match crate::gpu::GpuGenerator::new(cfg.clone()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU generator ({e}): skipped");
            return;
        }
    };
    let cpu = crate::Generator::new(cfg);
    let mut worst: f64 = 0.0;
    for &(name, z) in &[("interchange", 12u8), ("interchange", 15), ("interchange", 17), ("city", 13), ("city", 16), ("town", 15), ("town", 18)] {
        let id = tile_at(name, z);
        let g = gen.tiles(&[id]).unwrap().remove(0);
        let c = cpu.tile_cpu(id);
        let rgb = off(&g.rgb, &c.rgb, 2.0, |v| v as f64);
        let el = off(&g.elevation, &c.elevation, 0.05, |v| v as f64);
        let lc = off(&g.landcover, &c.landcover, 0.5, |v| v as f64);
        let em = off(&g.emission, &c.emission, 2.0, |v| v as f64);
        eprintln!("{name} z{z}: rgb off {:.3}% elevation off {:.3}% landcover off {:.3}% emission off {:.3}%", rgb * 100.0, el * 100.0, lc * 100.0, em * 100.0);
        worst = worst.max(rgb).max(el).max(lc).max(em);
    }
    assert!(worst < 0.005, "worst {:.3}%", worst * 100.0);
}

/// Every place generates on the GPU at all zooms (no CPU fallback).
#[cfg(feature = "gpu")]
#[test]
fn places_stay_on_the_gpu() {
    let Ok(g) = crate::gpu::shared() else { return };
    if g.check_generator().is_err() {
        return;
    }
    let gen = crate::Generator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
    for &(name, lat, lon) in PLACES {
        let ids: Vec<_> = [6u8, 8, 10, 12, 14, 16, 18].iter().map(|&z| geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)).collect();
        if let Err(e) = gen.tiles(&ids) {
            panic!("{name}: GPU generation failed: {e:#}");
        }
    }
    assert!(gen.backend_name().starts_with("GPU"), "{}", gen.backend_name());
}

#[allow(dead_code)]
fn unused(w: &World, lat: f64, lon: f64) -> DVec3 {
    ecef(w, lat, lon)
}

/// Interchange ramps connect: every end of a ramp polyline lies on a road (the major or the
/// minor route, or another ramp).
#[test]
fn ramps_connect() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == "interchange").unwrap();
    let f = build::features(&mut ctx, &Area { center: ecef(&w, lat, lon), radius: 1_500.0, gsd: 2.0 });
    let ramps: Vec<&FSeg> = f.segs.iter().filter(|s| s.kind == cls::KIND0 + cls::RAMP as u32).collect();
    let roads: Vec<&FSeg> = f.segs.iter().filter(|s| s.kind < cls::KIND0 + cls::RAIL as u32).collect();
    assert!(!ramps.is_empty(), "no interchange at the place");
    let dist = |p: DVec3, s: &FSeg| net::chord_dist(s.a, s.b, p);
    let mut bad = 0;
    for r in &ramps {
        for e in [r.a, r.b] {
            let shared = ramps.iter().filter(|o| (o.a - e).length() < 0.01 || (o.b - e).length() < 0.01).count();
            if shared >= 2 {
                continue;
            }
            let d = roads.iter().filter(|s| s.kind != r.kind || (s.a - r.a).length() > 1.0).map(|s| dist(e, s)).fold(f64::MAX, f64::min);
            if d > 25.0 {
                let g = geodesy::ecef2geodetic(e, &w.ell);
                eprintln!("ramp end at ({:.5}, {:.5}) is {d:.0} m from any road", g.lat.to_degrees(), g.lon.to_degrees());
                bad += 1;
            }
        }
    }
    assert!(bad == 0, "{bad} loose ramp ends");
}

/// Debug: GPU − CPU differences of one tile (`TR_SPEC` "name,z" of `PLACES`, `TR_OUT` dir):
/// a PNG of the differing emission / rgb pixels and the first values.
#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn debug_parity() {
    let spec = std::env::var("TR_SPEC").unwrap_or("town,18".into());
    let v: Vec<&str> = spec.split(',').collect();
    let (name, z) = (v[0], v[v.len() - 1].parse::<u8>().unwrap());
    let out = std::path::PathBuf::from(std::env::var("TR_OUT").unwrap_or("/tmp/transport".into()));
    std::fs::create_dir_all(&out).unwrap();
    let cfg = crate::Config::default();
    let gen = crate::gpu::GpuGenerator::new(cfg.clone()).unwrap();
    let cpu = crate::Generator::new(cfg);
    let id = if v.len() == 3 {
        tile_at(name, z)
    } else {
        geodesy::tiles::tile_for_latlon(v[1].parse::<f64>().unwrap().to_radians(), v[2].parse::<f64>().unwrap().to_radians(), z)
    };
    let g = gen.tiles(&[id]).unwrap().remove(0);
    let c = cpu.tile_cpu(id);
    let n = crate::TILE_SIZE;
    let mut img = image::RgbImage::new(n as u32, n as u32);
    let mut shown = 0;
    for j in 0..n {
        for i in 0..n {
            let o = 3 * (j * n + i);
            let de = (0..3).map(|k| (g.emission[o + k] as i32 - c.emission[o + k] as i32).abs()).max().unwrap();
            let dr = (0..3).map(|k| (g.rgb[o + k] as i32 - c.rgb[o + k] as i32).abs()).max().unwrap();
            img.put_pixel(i as u32, j as u32, image::Rgb([(de * 40).min(255) as u8, (dr * 40).min(255) as u8, c.rgb[o] / 3]));
            if (de > 2 || dr > 2) && shown < 20 {
                eprintln!(
                    "({i}, {j}): emission gpu {:?} cpu {:?}; rgb gpu {:?} cpu {:?}",
                    &g.emission[o..o + 3],
                    &c.emission[o..o + 3],
                    &g.rgb[o..o + 3],
                    &c.rgb[o..o + 3]
                );
                shown += 1;
            }
        }
    }
    img.save(out.join(format!("parity_{name}_z{z}.png"))).unwrap();
    for (tag, t) in [("gpu", &g), ("cpu", &c)] {
        let mut im = image::RgbImage::new(n as u32, n as u32);
        for j in 0..n {
            for i in 0..n {
                let o = 3 * (j * n + i);
                let e = |k: usize| (255.0 * (t.emission[o + k] as f64 / 255.0).sqrt()) as u8;
                im.put_pixel(i as u32, j as u32, image::Rgb([e(0), e(1), e(2)]));
            }
        }
        im.save(out.join(format!("parity_{name}_z{z}_{tag}.png"))).unwrap();
        let mut im = image::RgbImage::new(n as u32, n as u32);
        for j in 0..n {
            for i in 0..n {
                let o = 3 * (j * n + i);
                im.put_pixel(i as u32, j as u32, image::Rgb([t.rgb[o], t.rgb[o + 1], t.rgb[o + 2]]));
            }
        }
        im.save(out.join(format!("parity_{name}_z{z}_{tag}_rgb.png"))).unwrap();
    }
}

/// Debug: switchbacks, interchanges by size, river bridges, power lines away from towns:
/// `find_transport_places2 -- --ignored --nocapture`.
#[test]
#[ignore]
fn find_transport_places2() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let ll = |p: DVec3| {
        let g = geodesy::ecef2geodetic(p, &w.ell);
        (g.lat.to_degrees(), g.lon.to_degrees())
    };
    // switchbacks: minor routes in the mountains winding far more than their chord
    let mut n_sw = 0;
    for p in land_points(&w, 400, 0.04, 0x7A8) {
        if w.smooth_relief(p, 3000.0).1 < 500.0 {
            continue;
        }
        for n in [nw::LOCAL, nw::HAMLET] {
            for e in net::edges_near(&w, n, p, 6_000.0) {
                let Some(l) = build::routed(&w, &e) else { continue };
                let k = l.pts.len();
                let mut best = (0.0, 0usize);
                for i in 12..k.saturating_sub(12) {
                    let arc = l.s[i + 12] - l.s[i - 12];
                    let ch = (l.pts[i + 12] - l.pts[i - 12]).length().max(1.0);
                    if arc / ch > best.0 {
                        best = (arc / ch, i);
                    }
                }
                if best.0 > 2.5 && n_sw < 8 {
                    eprintln!("switchbacks (class {}): {:?} winding {:.1}", e.class, ll(l.pts[best.1]), best.0);
                    n_sw += 1;
                }
            }
        }
        if n_sw >= 8 {
            break;
        }
    }
    // interchanges, river bridges, power lines, tracks around cities
    let mut cities: Vec<net::Node> = Vec::new();
    for p in land_points(&w, 40, 0.2, 0x7A9) {
        for c in net::nodes(&w, nw::ARTERIAL, p, 200_000.0) {
            if c.dev > 0.45 && !cities.iter().any(|x| x.id == c.id) {
                cities.push(c);
            }
        }
        if cities.len() > 16 {
            break;
        }
    }
    let mut shown = std::collections::BTreeMap::<String, usize>::new();
    let mut show = |what: String, p: (f64, f64)| {
        let c = shown.entry(what.clone()).or_insert(0);
        if *c < 5 {
            eprintln!("{what}: ({:.4}, {:.4})", p.0, p.1);
            *c += 1;
        }
    };
    for c in &cities {
        let f = build::features(&mut ctx, &Area { center: c.p, radius: 30_000.0, gsd: 100.0 });
        let f2 = build::features(&mut ctx, &Area { center: c.p, radius: 12_000.0, gsd: 20.0 });
        let ramps: Vec<DVec3> = f2.segs.iter().filter(|s| s.kind == cls::KIND0 + cls::RAMP as u32).map(|s| s.a).collect();
        let mut done: Vec<DVec3> = Vec::new();
        for r in &ramps {
            if done.iter().any(|d| (*d - *r).length() < 1_500.0) {
                continue;
            }
            let n = ramps.iter().filter(|q| (**q - *r).length() < 900.0).count();
            done.push(*r);
            show(
                format!(
                    "interchange_{}",
                    if n > 150 {
                        "cloverleaf"
                    } else if n > 110 {
                        "parclo"
                    } else if n > 69 {
                        "diamond"
                    } else {
                        "trumpet"
                    }
                ),
                ll(*r),
            );
        }
        for s in &f2.segs {
            let k = (s.kind - cls::KIND0) as usize;
            if s.flags & flag::BRIDGE != 0 && s.flags & flag::OVER == 0 && k <= cls::TRUNK {
                show(format!("bridge_{k}"), ll(s.a));
            }
            if s.flags & flag::TUNNEL != 0 {
                show(format!("tunnel_{k}"), ll(s.a));
            }
        }
        for s in &f.segs {
            if s.kind == cls::KIND0 + cls::POWER as u32 {
                let a = w.atlas().sample(s.a.normalize());
                if a.precip_mm > 1000.0 && (s.a - c.p).length() > 1.5 * c.radius + 4_000.0 {
                    show("power_wet".into(), ll(s.a));
                }
            }
        }
        for st in &f2.stamps {
            show(format!("stamp_{}", st.template), ll(st.center));
        }
    }
}
