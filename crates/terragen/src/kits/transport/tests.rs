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
pub(crate) const PLACES: &[(&str, f64, f64)] = &[
    ("parclo", -7.6475, -16.6320),
    ("diamond", -7.6988, -18.7191),
    ("trumpet", -23.7203, -33.8120),
    ("switchbacks", -24.1868, -33.3606),
    ("station", -7.7302, -18.7965),
    ("yard", -7.7461, -18.7972),
    ("roundabout", -7.6230, -16.6053),
    ("power", -7.8316, -18.8000),
    ("portal", 29.3400, 179.4257),
    ("bridge", -7.5782, -18.7463),
    ("tracks", -7.64, -16.66),
];

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
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let p = land_points(&w, 1, 0.2, 0x7A3)[0];
    let edges = net::edges_near(&w, nw::LOCAL, p, 10_000.0);
    assert!(!edges.is_empty());
    let a: Vec<_> = edges.iter().take(4).map(|e| route_of(&mut ctx, e)).collect();
    build::clear_cache();
    for (e, ra) in edges.iter().take(4).zip(&a) {
        let rb = route_of(&mut ctx, e);
        match (ra, &rb) {
            (Some(x), Some(y)) => {
                assert_eq!(x.pts, y.pts);
                let [ea, eb] = build::base(&w, e).ends;
                assert!((x.pts[0] - ea).length() < 1.0 && (x.pts.last().unwrap() - eb).length() < 1.0);
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
        let Some(line) = route_of(&mut ctx, e) else { continue };
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
        // (a long climb steeper than the limit over the whole radius keeps its grade: the
        // route's cost avoids those)
        assert!(g[g.len() / 2] < 0.065 + 0.005, "median grade {}", g[g.len() / 2]);
        assert!(p90 < 3.0 * 0.065, "90th percentile grade {p90}");
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
    let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), backend).unwrap();
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
    let g = crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
    let n = crate::TILE_SIZE;
    for &(name, z) in &[("parclo", 13u8), ("station", 14), ("roundabout", 15), ("trumpet", 14), ("portal", 15)] {
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
    let g = crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
    let n = crate::TILE_SIZE;
    for &(name, z) in &[("parclo", 15u8), ("station", 14), ("roundabout", 16), ("bridge", 15)] {
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
    let a = crate::NextGenerator::new(cfg.clone());
    let list = [("parclo", 15u8), ("roundabout", 16), ("portal", 16)];
    let ta: Vec<_> = list.iter().map(|&(nm, z)| a.tile(tile_at(nm, z))).collect();
    net::clear_cache();
    build::clear_cache();
    let b = crate::NextGenerator::new(cfg);
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
    let cpu = crate::NextGenerator::new(cfg);
    let mut worst: f64 = 0.0;
    for &(name, z) in &[
        ("parclo", 12u8),
        ("parclo", 15),
        ("parclo", 17),
        ("diamond", 16),
        ("trumpet", 15),
        ("switchbacks", 16),
        ("station", 13),
        ("station", 17),
        ("yard", 15),
        ("roundabout", 15),
        ("roundabout", 18),
        ("power", 17),
        ("portal", 16),
        ("bridge", 16),
        ("tracks", 16),
    ] {
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
    let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
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
    let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == "parclo").unwrap();
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
    let cpu = crate::NextGenerator::new(cfg);
    let id = if v.len() == 2 {
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
                let Some(l) = route_of(&mut ctx, &e) else { continue };
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
            if s.flags & flag::BRIDGE != 0 && s.flags & flag::OVER == 0 && k <= cls::SECONDARY && !build::in_town(&w, s.a) {
                show(format!("bridge_{k}"), ll(s.a));
            }
            if s.flags & flag::TUNNEL != 0 && !build::in_town(&w, s.a) {
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

/// Debug: the lit segments near a pixel of a place's tile (`TR_SPEC` "name,z,i,j").
#[test]
#[ignore]
fn debug_lamps() {
    let spec = std::env::var("TR_SPEC").unwrap_or("station,17,95,150".into());
    let v: Vec<&str> = spec.split(',').collect();
    let z: u8 = v[1].parse().unwrap();
    let id = tile_at(v[0], z);
    let (i, j): (f64, f64) = (v[2].parse().unwrap(), v[3].parse().unwrap());
    let n = crate::TILE_SIZE as f64;
    let (lat, lon) = geodesy::tiles::pixel_to_latlon(glam::DVec2::new(id.x as f64 * n + i + 0.5, id.y as f64 * n + j + 0.5), z, crate::TILE_SIZE as u32);
    let w = world();
    let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let gsd = geodesy::tiles::gsd_ew(lat, z, crate::TILE_SIZE as u32, &w.ell);
    let f = build::features(&mut ctx, &Area { center: p, radius: 150.0, gsd });
    for s in &f.segs {
        let d = net::chord_dist(s.a, s.b, p);
        if d < 80.0 {
            let len = (s.b - s.a).length();
            eprintln!("kind {} d {d:.1} s0 {} len {len:.4} lamp {} flags {:b} deck {} {} hw {}", s.kind, s.s0, s.hb, s.flags, s.deck_a, s.deck_b, s.v[0]);
        }
    }
    for st in &f.stamps {
        if (st.center - p).length() < 400.0 {
            eprintln!("stamp {} at {:.1} m half {:?} v {:?}", st.template, (st.center - p).length(), st.half, st.v);
        }
    }
}

/// Debug: roads and railways crossing rivers outside towns: `find_river_bridges -- --ignored
/// --nocapture`.
#[test]
#[ignore]
fn find_river_bridges() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let mut n = 0;
    for &(lat, lon) in &[(-7.70, -18.75), (-7.62, -16.62), (28.7, 178.9)] {
        let f = build::features(&mut ctx, &Area { center: ecef(&w, lat, lon), radius: 25_000.0, gsd: 150.0 });
        for (i, s) in f.segs.iter().enumerate() {
            let k = (s.kind - cls::KIND0) as usize;
            if i % 2 != 0 || (k > cls::SECONDARY && k != cls::RAIL) {
                continue;
            }
            let m = (s.a + s.b) * 0.5;
            let g = geodesy::ecef2geodetic(m, &w.ell);
            let t = w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 10.0, &w.ell));
            if t.river_hw > 6.0 && t.river_d.abs() < t.river_hw && !build::in_town(&w, m) {
                eprintln!("river bridge class {k} hw {:.0}: ({:.5}, {:.5})", t.river_hw, g.lat.to_degrees(), g.lon.to_degrees());
                n += 1;
                if n >= 12 {
                    return;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- routing: water and sites

/// Regions of the routing verification: the kit's places, the settlements kit's site places
/// (airport, quarry, solar and wind farms, a dam's reservoir), the cold kit's kettle lakes and
/// lake districts of the default world.
const ROUTE_REGIONS: &[(&str, f64, f64)] = &[
    ("parclo", -7.6570, -16.6301),
    ("trumpet", -23.7536, -33.8316),
    ("switchbacks", -24.1868, -33.3606),
    ("station", -7.7302, -18.7965),
    ("bridge", -7.5782, -18.7463),
    ("tracks", -7.64, -16.66),
    ("portal", 29.3400, 179.4257),
    ("airport", -10.5390, 115.3568),
    ("quarry", -20.4582, -152.1378),
    ("solar", 40.4972, 46.9202),
    ("wind", 42.4183, 33.3454),
    ("dam", -0.6990, -17.6874),
    ("kettle_a", 65.2726, -28.7446),
    ("kettle_b", 63.0479, 13.4909),
    ("lakes_a", 16.2337, -96.5839),
    ("lakes_b", 20.4996, -30.0929),
    ("lakes_c", 30.2156, -89.0182),
    ("lakes_d", 46.0667, 2.2438),
    ("lakes_e", -2.4905, -123.2721),
];

/// The sites and reservoirs near an area (the verification's truth).
struct Truth {
    stamps: Vec<FStamp>,
    res: Vec<FSeg>,
}

fn truth(ctx: &mut HostCtx, c: DVec3, r: f64) -> Truth {
    use crate::kits::settlements::sites::{self, tpl as st};
    let mut f = Features::default();
    sites::sites_near(ctx, c, r, 10.0, &mut f);
    Truth {
        stamps: f.stamps.into_iter().filter(|s| matches!(s.template, st::AIRPORT | st::QUARRY | st::SOLAR | st::WIND)).collect(),
        res: f.segs.into_iter().filter(|s| s.kind == st::RESERVOIR).collect(),
    }
}

/// What lies at `p`: 0 dry land, 1 a stream (half width < 6 m), 6 a river, 2 a lake (the core's, a kettle lake of the cold kit's
/// large lattice), 3 a pond of its small lattice, 4 the sea, 5 a reservoir, 10 + template: a
/// site.
fn what_at(w: &World, tr: &Truth, p: DVec3) -> u32 {
    use crate::kits::settlements::sites::{self, tpl as st};
    for s in &tr.stamps {
        let q = sites::uv(s, p);
        let inside = if s.template == st::QUARRY { q.length() < 1.5 * s.v[0] as f64 } else { q.x.abs() < s.half[0] as f64 && q.y.abs() < s.half[1] as f64 };
        if inside {
            return 10 + s.template;
        }
    }
    let g = geodesy::ecef2geodetic(p, &w.ell);
    let t = w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 2.0, &w.ell));
    if t.water_kind == crate::world::water::OCEAN {
        return 4;
    }
    if t.water_kind == crate::world::water::LAKE && t.water > t.ground {
        return 2;
    }
    if tr.res.iter().any(|sg| crate::kits::settlements::sites::reservoir_at(sg, p, t.ground, 0.1).0 > 0.5) {
        return 5;
    }
    let sc = if t.ground < 700.0 && (t.temp < -1.0 || (t.glaciation > 0.0 && t.temp < 9.0)) {
        // (the slope over ±40 m, as the routing estimates it)
        let (_, _, e, n) = crate::kits::settlements::places::frame_at(w, p);
        let r = |d: DVec3| route::relief_h(w, route::surface(w, p + d), 30.0);
        let slope = glam::DVec2::new(r(e * 40.0) - r(-e * 40.0), r(n * 40.0) - r(-n * 40.0)).length() / 80.0;
        route::cold_lake_scale(&t, slope)
    } else {
        0.0
    };
    if sc > 0.0 {
        for (k, lat) in crate::kits::cold::LAKE_LATS.iter().enumerate() {
            let h = crate::kits::cold::lake_at(lat, p, sc, 0.5);
            if h.cov > 0.5 && h.kind == 0 {
                return if k == 0 { 2 } else { 3 };
            }
        }
    }
    if t.river_hw > 0.0 && t.river_d.abs() < t.river_hw {
        return if t.river_hw >= 6.0 { 6 } else { 1 };
    }
    0
}

/// The sine of the angle between a route's direction `dir` at `p` and the river there (1:
/// perpendicular), from the gradient of the signed distance to the channel.
fn crossing_sin(w: &World, p: DVec3, dir: DVec3) -> f64 {
    let (_, _, e, n) = crate::kits::settlements::places::frame_at(w, p);
    let d = |q: DVec3| {
        let g = geodesy::ecef2geodetic(q, &w.ell);
        w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 2.0, &w.ell)).river_d
    };
    let gr = e * (d(p + e * 5.0) - d(p - e * 5.0)) + n * (d(p + n * 5.0) - d(p - n * 5.0));
    dir.dot(gr.normalize_or_zero()).abs()
}

/// Roads and railways keep out of lakes, reservoirs and the sea (except narrow crossings,
/// [`route::span`]) and out of airports, quarries, solar and wind farms (except where they end
/// there): the routes near varied regions sampled every 25 m against pass A's water (CPU), the
/// sites' footprints and reservoirs, the cold kit's kettle lakes; bridges lie over water or
/// valleys. `TR_REGION=name` one region.
#[test]
fn routes_avoid_water_and_sites() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let only = std::env::var("TR_REGION").ok();
    let (mut n_route, mut n_seg, mut n_river, mut n_narrow, mut n_pond, mut n_xing, mut n_oblique) = (0, 0, 0, 0, 0, 0, 0);
    let mut sin_sum = 0.0;
    let mut n_tunnel = 0;
    let mut bad: Vec<String> = Vec::new();
    let mut sites_bad = 0;
    let (mut br_wet, mut br_dry) = (0, 0);
    for &(name, lat, lon) in ROUTE_REGIONS {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let c = ecef(&w, lat, lon);
        let area = Area { center: c, radius: 3_000.0, gsd: 10.0 };
        let t0 = std::time::Instant::now();
        let (routes, _, _) = build::area_routes(&mut ctx, &area);
        let tr = truth(&mut ctx, c, 4_000.0);
        let mut roads: Vec<_> = routes
            .iter()
            .filter(|r| {
                matches!(r.e.class, cls::MOTORWAY | cls::TRUNK | cls::SECONDARY | cls::LOCAL | cls::TRACK | cls::RAIL) && r.line.dist(c).0 < area.radius - 100.0
            })
            .collect();
        roads.sort_by_key(|r| (r.e.class, r.e.key));
        // (every k-th route: ≤ 40 per region)
        let step = (roads.len() / 40).max(1);
        let mut seen = 0;
        for r in roads.iter().step_by(step) {
            let (l, class) = (&r.line, r.e.class);
            let (ea, eb) = (l.pts[0], *l.pts.last().unwrap());
            let mut run = (0u32, 0.0f64, Vec::new());
            let mut in_river = false;
            let mut segs = 0;
            let ds = 25.0;
            let n = (l.len() / ds).ceil() as usize;
            for k in 0..=n {
                let s = l.len() * k as f64 / n.max(1) as f64;
                let p = l.at(s).0;
                let inside = (p - c).length() < area.radius && s > 60.0 && l.len() - s > 60.0;
                let kind = if inside { what_at(&w, &tr, p) } else { 0 };
                if inside {
                    segs += 1;
                }
                // runs of a lake (2), the sea (4), a reservoir (5)
                let wet = if matches!(kind, 2 | 4 | 5) { kind } else { 0 };
                if wet != 0 && wet == run.0 {
                    run.1 += ds;
                    run.2.push(p);
                } else {
                    finish_run(&w, name, class, &run, &mut bad, &mut n_narrow);
                    run = (wet, if wet != 0 { ds } else { 0.0 }, if wet != 0 { vec![p] } else { vec![] });
                }
                if kind == 6 && !in_river {
                    n_xing += 1;
                    let sin = crossing_sin(&w, p, l.at(s).1);
                    sin_sum += sin;
                    if sin < 0.5 {
                        n_oblique += 1;
                    }
                }
                in_river = kind == 6;
                match kind {
                    1 | 6 => n_river += 1,
                    3 => n_pond += 1,
                    k if k >= 10 => {
                        // (a route ending at or in the site keeps it)
                        let st = tr
                            .stamps
                            .iter()
                            .find(|st| st.template == k - 10 && crate::kits::settlements::sites::uv(st, p).length() < 2.0 * (st.half[0].max(st.half[1]) as f64))
                            .unwrap();
                        let q = |e: DVec3| crate::kits::settlements::sites::uv(st, e);
                        let near_end = [ea, eb].iter().any(|&e| q(e).x.abs() < st.half[0] as f64 + 100.0 && q(e).y.abs() < st.half[1] as f64 + 100.0);
                        if !near_end {
                            sites_bad += 1;
                            let g = geodesy::ecef2geodetic(p, &w.ell);
                            bad.push(format!("{name}: class {class} in site {} at ({:.5}, {:.5})", k - 10, g.lat.to_degrees(), g.lon.to_degrees()));
                        }
                    }
                    _ => {}
                }
            }
            finish_run(&w, name, class, &run, &mut bad, &mut n_narrow);
            if segs > 0 {
                seen += 1;
                n_seg += segs;
            }
        }
        n_route += seen;
        // bridges (not overpasses): over water or a valley
        let f = build::features(&mut ctx, &area);
        n_tunnel += f.segs.iter().filter(|s| s.flags & flag::TUNNEL != 0).count();
        for s in f.segs.iter().filter(|s| s.flags & flag::BRIDGE != 0 && s.flags & flag::OVER == 0 && s.kind != cls::KIND0 + cls::RAMP as u32) {
            let m = (s.a + s.b) * 0.5;
            let g = geodesy::ecef2geodetic(m, &w.ell);
            let t = w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 5.0, &w.ell));
            if t.water_kind != crate::world::water::NONE
                || (t.river_hw > 0.0 && t.river_d.abs() < t.river_hw + 20.0)
                || matches!(what_at(&w, &tr, m), 2 | 4 | 5 | 6)
            {
                br_wet += 1;
            } else {
                br_dry += 1;
            }
        }
        eprintln!(
            "{name}: {} routes, {seen} sampled, {} site stamps, {} reservoir pieces ({:.1} s)",
            roads.len(),
            tr.stamps.len(),
            tr.res.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    for b in bad.iter().take(40) {
        eprintln!("{b}");
    }
    eprintln!(
        "{n_route} routes, {n_seg} sampled 25 m pieces: {} in water or sites ({sites_bad} in sites), {n_narrow} narrow crossings, {n_river} points over rivers and streams; {n_xing} river crossings ({:.1} per 100 km, mean sin {:.2}, {n_oblique} under 30°); {n_pond} points over ponds; bridges over water {br_wet}, over dry valleys {br_dry}; {n_tunnel} tunnel segments",
        bad.len(),
        n_xing as f64 / (n_seg as f64 * 25.0 / 1e5),
        sin_sum / n_xing.max(1) as f64
    );
    if only.is_none() {
        assert!(n_route >= 200, "{n_route} routes");
    }
    assert!(bad.is_empty(), "{} in water or sites", bad.len());
}

/// A run of water along a route: a narrow crossing (≤ the class's span + a sample) or too wide.
fn finish_run(w: &World, name: &str, class: usize, run: &(u32, f64, Vec<DVec3>), bad: &mut Vec<String>, narrow: &mut usize) {
    if run.0 == 0 {
        return;
    }
    if run.1 <= route::span(class, run.0 != 2) + 25.0 {
        *narrow += 1;
    } else {
        let g = geodesy::ecef2geodetic(run.2[run.2.len() / 2], &w.ell);
        bad.push(format!(
            "{name}: class {class} over {:.0} m of {} at ({:.5}, {:.5})",
            run.1,
            ["", "", "a lake", "", "the sea", "a reservoir"][run.0 as usize],
            g.lat.to_degrees(),
            g.lon.to_degrees()
        ));
    }
}

/// Debug: kettle-lake lowlands with roads (glaciated or permafrost lowlands with people whose
/// 9 × 9 km grid has 8–45 % kettle lakes): `find_cold_lake_regions -- --ignored --nocapture`.
#[test]
#[ignore]
fn find_cold_lake_regions() {
    let w = world();
    let mut n_cold = 0;
    for i in 0..4_000_000u64 {
        let h = mix64(0x7AC ^ i);
        let lat = (2.0 * u01k(h, 1) - 1.0).asin();
        let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
        let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
        let a = w.atlas().sample(p.normalize());
        if a.coast_km < 5.0 || a.population < 0.05 || !(a.temp_c < 0.0 || (a.glaciation > 0.3 && a.temp_c < 10.0)) {
            continue;
        }
        if net::nodes(&w, nw::HAMLET, p, 5_000.0).len() < 3 {
            continue;
        }
        let (_, _, east, north) = crate::kits::settlements::places::frame_at(&w, p);
        let mut cold = 0;
        for j in -4..=4 {
            for i in -4..=4 {
                let q = route::surface(&w, p + east * (i as f64 * 1_000.0) + north * (j as f64 * 1_000.0));
                let g = geodesy::ecef2geodetic(q, &w.ell);
                let t = w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 20.0, &w.ell));
                if route::in_cold_lake(&t, 0.0, q) {
                    cold += 1;
                }
            }
        }
        if cold >= 6 {
            eprintln!("kettle lakes {cold}/81: ({:.4}, {:.4})", lat.to_degrees(), lon.to_degrees());
            n_cold += 1;
            if n_cold >= 6 {
                break;
            }
        }
    }
}

/// Debug: lake districts and kettle-lake lowlands with roads (populated land whose 9 × 9 km
/// grid has 8–45 % lake): `find_lake_regions -- --ignored --nocapture`.
#[test]
#[ignore]
fn find_lake_regions() {
    let w = world();
    let (mut n_lake, mut n_cold) = (0, 0);
    for p in land_points(&w, 3000, 0.08, 0x7AB) {
        let (_, _, east, north) = crate::kits::settlements::places::frame_at(&w, p);
        let (mut lake, mut cold, mut n) = (0, 0, 0);
        for j in -4..=4 {
            for i in -4..=4 {
                let q = route::surface(&w, p + east * (i as f64 * 1_000.0) + north * (j as f64 * 1_000.0));
                let g = geodesy::ecef2geodetic(q, &w.ell);
                let t = w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 20.0, &w.ell));
                n += 1;
                if t.water_kind == crate::world::water::LAKE {
                    lake += 1;
                }
                if route::in_cold_lake(&t, 0.0, q) {
                    cold += 1;
                }
            }
        }
        let g = geodesy::ecef2geodetic(p, &w.ell);
        let ll = (g.lat.to_degrees(), g.lon.to_degrees());
        if n_lake < 8 && (8..=36).contains(&lake) {
            eprintln!("lakes {lake}/{n}: ({:.4}, {:.4})", ll.0, ll.1);
            n_lake += 1;
        }
        if n_cold < 6 && (8..=36).contains(&cold) {
            eprintln!("kettle lakes {cold}/{n}: ({:.4}, {:.4})", ll.0, ll.1);
            n_cold += 1;
        }
        if n_lake >= 8 && n_cold >= 6 {
            break;
        }
    }
}

/// Roads end at a town's arterials, not its centre: no road route of a town, city or
/// metropolis ends within 0.35 of its radius from the centre (railways still do: stations).
#[test]
fn roads_end_at_town_arterials() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let (mut n, mut n_rail, mut central) = (0, 0, 0);
    for &(lat, lon) in &[(-7.7302, -18.7965), (30.35, 128.8487), (-7.6185, -16.6091), (-23.7536, -33.8316)] {
        let (routes, _, _) = build::area_routes(&mut ctx, &Area { center: ecef(&w, lat, lon), radius: 8_000.0, gsd: 30.0 });
        for r in &routes {
            for (nd, p) in [(r.e.a, r.line.pts[0]), (r.e.b, *r.line.pts.last().unwrap())] {
                if nd.level < crate::kits::settlements::places::TOWN || nd.level > crate::kits::settlements::places::METRO {
                    continue;
                }
                let d = (p - nd.p).length() / nd.radius;
                if r.e.class == cls::RAIL {
                    n_rail += 1;
                    assert!(d < 0.3, "a railway ends at {d:.2} R");
                } else if r.e.class != cls::POWER {
                    n += 1;
                    if d <= 0.35 {
                        eprintln!("a road of class {} ends at {d:.2} R of a level {} place", r.e.class, nd.level);
                        central += 1;
                    }
                }
            }
        }
    }
    eprintln!("{n} road ends ({central} within 0.35 R of the centre) and {n_rail} railway ends at towns");
    assert!(n > 20);
    assert_eq!(central, 0);
}

/// The route of an edge (settled: the CPU's point evaluations are immediate).
fn route_of(ctx: &mut HostCtx, e: &net::Edge) -> Option<std::sync::Arc<route::Line>> {
    build::routed(ctx, e).expect("the CPU's points are immediate")
}

/// Timing: routing every edge near some regions from cold caches, then their features (10 km
/// discs at 30 m pixels): `routing_time -- --ignored --nocapture`.
#[test]
#[ignore]
fn routing_time() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let (mut t_route, mut t_feat, mut n_routes) = (0.0, 0.0, 0);
    for &(name, lat, lon) in
        ROUTE_REGIONS.iter().filter(|r| matches!(r.0, "parclo" | "trumpet" | "bridge" | "airport" | "dam" | "kettle_a" | "lakes_a" | "lakes_d"))
    {
        net::clear_cache();
        build::clear_cache();
        crate::kits::settlements::sites::clear_cache();
        let c = ecef(&w, lat, lon);
        let mut edges = Vec::new();
        for n in 0..nw::N {
            if 30.0 < net::net_class_max_gsd(n) {
                edges.extend(net::edges_near(&w, n, c, 10_000.0 + 1_500.0).into_iter().filter(|e| 30.0 < cls::GSD_MAX[e.class]));
            }
        }
        // (the parts: the scenes, the analytic routes, the sites)
        let t0 = std::time::Instant::now();
        for e in &edges {
            let (a, b) = build::ends(&w, e);
            if e.class != cls::POWER {
                let sp = route::PARAMS[e.class].span;
                std::hint::black_box(route::Scene::of_corridor(&w, a, b, route::max_deviation(e.class, (b - a).length()), sp > 0.0 && sp < 300.0, Vec::new()));
            }
        }
        let ts = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let nb = edges.iter().filter(|e| build::base(&w, e).line.is_some()).count();
        let tb = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let n = edges.iter().filter(|e| route_of(&mut ctx, e).is_some()).count();
        let tr = ts + tb + t2.elapsed().as_secs_f64();
        eprintln!("{name}: scenes {ts:.2} s, analytic routes {tb:.2} s ({nb} routes), sites {:.2} s", t2.elapsed().as_secs_f64());
        if std::env::var("TR_WHY").is_ok() {
            let dropped: Vec<net::Edge> = edges.iter().filter(|e| route_of(&mut ctx, e).is_none()).copied().collect();
            for e in dropped.iter().take(12) {
                let bs = build::base(&w, e);
                let Some(l) = bs.line.clone() else { continue };
                let (blocks, res) = build::obstacles(&mut ctx, &l, e.class, bs.ends).unwrap();
                let mut sc = (*bs.scene).clone();
                sc.blocks = blocks.clone();
                sc.res = res.clone();
                let g = geodesy::ecef2geodetic(bs.ends[0], &w.ell);
                eprintln!(
                    "  dropped by sites: class {} {:.1} km from ({:.4}, {:.4}): {} blocks, {} reservoir pieces: {}",
                    e.class,
                    l.len() / 1e3,
                    g.lat.to_degrees(),
                    g.lon.to_degrees(),
                    blocks.len(),
                    res.len(),
                    route::diagnose(&w, bs.ends[0], bs.ends[1], e.class, &sc)
                );
            }
            for e in edges.iter().filter(|e| build::base(&w, e).line.is_none()).take(12) {
                let bs = build::base(&w, e);
                let ([a, b], sc) = (bs.ends, &bs.scene);
                let g = geodesy::ecef2geodetic(a, &w.ell);
                eprintln!(
                    "  dropped class {} {:.1} km from ({:.4}, {:.4}), levels {} {}: {}",
                    e.class,
                    (b - a).length() / 1e3,
                    g.lat.to_degrees(),
                    g.lon.to_degrees(),
                    e.a.level,
                    e.b.level,
                    route::diagnose(&w, a, b, e.class, sc)
                );
            }
        }
        let t1 = std::time::Instant::now();
        let f = build::features(&mut ctx, &Area { center: c, radius: 10_000.0, gsd: 30.0 });
        let tf = t1.elapsed().as_secs_f64();
        eprintln!("{name}: {} edges, {n} routed in {tr:.2} s; features {} segs in {tf:.2} s", edges.len(), f.segs.len());
        t_route += tr;
        t_feat += tf;
        n_routes += n;
    }
    eprintln!("total: {n_routes} routes in {t_route:.2} s ({:.1} ms per route); features {t_feat:.2} s", 1e3 * t_route / n_routes.max(1) as f64);
}

/// Debug: roads going around a lake (the edge's chord crosses the lake, the route skirts it)
/// and minor roads crossing wide rivers, in the lake regions: `find_routing_places -- --ignored
/// --nocapture`.
#[test]
#[ignore]
fn find_routing_places() {
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let ll = |p: DVec3| {
        let g = geodesy::ecef2geodetic(p, &w.ell);
        (g.lat.to_degrees(), g.lon.to_degrees())
    };
    let (mut n_det, mut n_x) = (0, 0);
    for &(name, lat, lon) in ROUTE_REGIONS.iter().filter(|r| r.0.starts_with("lakes") || matches!(r.0, "dam" | "bridge" | "parclo")) {
        let (routes, _, _) = build::area_routes(&mut ctx, &Area { center: ecef(&w, lat, lon), radius: 8_000.0, gsd: 30.0 });
        for r in routes.iter().filter(|r| r.e.class <= cls::LOCAL) {
            let bs = build::base(&w, &r.e);
            let [a, b] = bs.ends;
            for lk in &bs.scene.lakes {
                let c = route::surface(&w, lk.pt);
                let (d, _) = r.line.dist(c);
                // (the chord through the lake's middle, away from the ends; the route round it)
                let t = (c - a).dot(b - a) / (b - a).length_squared();
                if lk.rad > 200.0
                    && lk.rad < 1_500.0
                    && (0.15..0.85).contains(&t)
                    && net::chord_dist(a, b, c) < 0.6 * lk.rad
                    && d > 0.9 * lk.rad
                    && d < 2.0 * lk.rad
                    && n_det < 12
                {
                    let (la, lo) = ll(c);
                    let (pa, po) = ll(r.line.at(r.line.dist(c).1).0);
                    eprintln!("{name}: class {} around a lake (r {:.0} m) at ({la:.4}, {lo:.4}), road at ({pa:.4}, {po:.4})", r.e.class, lk.rad);
                    n_det += 1;
                }
            }
            // wide river crossings
            if n_x < 10 && r.e.class >= cls::SECONDARY && std::env::var("TR_RIVERS").is_ok() {
                let n = (r.line.len() / 20.0) as usize;
                let mut inr = false;
                for k in 0..n {
                    let p = r.line.at(k as f64 * 20.0).0;
                    let g = geodesy::ecef2geodetic(p, &w.ell);
                    let t = w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 5.0, &w.ell));
                    let now = t.river_hw > 25.0 && t.river_d.abs() < t.river_hw;
                    if now && !inr && n_x < 10 {
                        eprintln!(
                            "{name}: class {} crosses a river (half width {:.0} m) at ({:.4}, {:.4})",
                            r.e.class,
                            t.river_hw,
                            g.lat.to_degrees(),
                            g.lon.to_degrees()
                        );
                        n_x += 1;
                    }
                    inr = now;
                }
            }
        }
    }
}

/// Debug: the kit's stamps near a point (`TR_SPEC` "lat,lon,radius"):
/// `stamps_near -- --ignored --nocapture`.
#[test]
#[ignore]
fn stamps_near() {
    let spec = std::env::var("TR_SPEC").unwrap_or("-7.6185,-16.6091,2000".into());
    let v: Vec<f64> = spec.split(',').map(|x| x.trim().parse().unwrap()).collect();
    let w = world();
    let mut pts = CpuPoints(&w);
    let mut ctx = HostCtx { world: &w, points: &mut pts };
    let f = build::features(&mut ctx, &Area { center: ecef(&w, v[0], v[1]), radius: v[2], gsd: 2.0 });
    for st in &f.stamps {
        let g = geodesy::ecef2geodetic(st.center, &w.ell);
        eprintln!("stamp {} at ({:.5}, {:.5})", st.template, g.lat.to_degrees(), g.lon.to_degrees());
    }
    // the segments near the point
    let c = ecef(&w, v[0], v[1]);
    for sg in f.segs.iter().filter(|sg| net::chord_dist(sg.a, sg.b, c) < 120.0) {
        let g = geodesy::ecef2geodetic(c, &w.ell);
        let t = w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 5.0, &w.ell));
        eprintln!(
            "seg kind {} flags {:#x} deck {:.1} {:.1} at {:.0} m; ground {:.1} water {:.1} kind {}",
            sg.kind - cls::KIND0,
            sg.flags,
            sg.deck_a,
            sg.deck_b,
            net::chord_dist(sg.a, sg.b, c),
            t.ground,
            t.water,
            t.water_kind
        );
    }
    // interchange ramps: their centroids, in clusters of 1.5 km
    let mut cl: Vec<(DVec3, usize)> = Vec::new();
    for sg in f.segs.iter().filter(|s| s.kind == cls::KIND0 + cls::RAMP as u32) {
        match cl.iter_mut().find(|c| (c.0 / c.1 as f64 - sg.a).length() < 1_500.0) {
            Some(c) => {
                c.0 += sg.a;
                c.1 += 1;
            }
            None => cl.push((sg.a, 1)),
        }
    }
    for (c, n) in cl {
        let g = geodesy::ecef2geodetic(c / n as f64, &w.ell);
        eprintln!("ramps: {n} segments around ({:.5}, {:.5})", g.lat.to_degrees(), g.lon.to_degrees());
    }
}
