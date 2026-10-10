//! Atlas tests: f16 packing, determinism across thread counts, continuity across cube faces,
//! the disk cache, climate sanity (belts, rain shadows), Köppen classes, build time, and the
//! WGSL sampler against the CPU's.

use super::geom::{dir_at, Grid};
use super::*;
use crate::noise::{mix64, u01};
use std::sync::OnceLock;

fn cfg(res: u32, seed: u64) -> Config {
    let mut c = Config { seed, ..Config::default() };
    c.atlas.resolution = res;
    c
}

/// A small atlas shared by the tests (resolution 128, default world).
fn small() -> &'static Atlas {
    static A: OnceLock<Atlas> = OnceLock::new();
    A.get_or_init(|| Atlas::build(&World::new(cfg(128, 1))))
}

fn continuous(s: &AtlasSample) -> [f64; 16] {
    [
        s.elevation_m,
        s.coast_km,
        s.wind_e,
        s.wind_n,
        s.precip_mm,
        s.temp_c,
        s.temp_range_c,
        s.regime,
        s.plate_dist_km,
        s.convergence_mm_yr,
        s.uplift,
        s.volcanism,
        s.glaciation,
        s.development,
        s.population,
        0.0,
    ]
}

const NAMES: [&str; 15] = [
    "elevation",
    "coast",
    "wind_e",
    "wind_n",
    "precip",
    "temp",
    "temp_range",
    "regime",
    "plate_dist",
    "convergence",
    "uplift",
    "volcanism",
    "glaciation",
    "development",
    "population",
];

#[test]
fn f16_round_trip() {
    for h in 0..=u16::MAX {
        let e = (h >> 10) & 0x1f;
        if e == 0 || e == 31 {
            continue; // subnormals (flushed), inf / nan
        }
        assert_eq!(f16_bits(f16_value(h)), h, "{h:04x}");
    }
    assert_eq!(f16_value(f16_bits(1.0)), 1.0);
    assert_eq!(f16_value(f16_bits(-2.5)), -2.5);
    assert_eq!(f16_value(f16_bits(1e9)), 65504.0);
    assert_eq!(f16_bits(1e-6), 0);
    assert_eq!(f16_bits(f32::NAN), 0);
    // round to nearest: relative error ≤ 2⁻¹¹
    let mut h = 7u64;
    for _ in 0..10000 {
        h = mix64(h);
        let x = ((u01(h) - 0.5) * 2e4) as f32;
        if x.abs() > 1e-4 {
            let y = f16_value(f16_bits(x));
            assert!(((y - x) / x).abs() <= 1.0 / 2048.0 + 1e-7, "{x} -> {y}");
        }
    }
}

#[test]
fn same_bytes_for_any_thread_count() {
    let w = World::new(cfg(64, 3));
    let run = |threads: usize| rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| Atlas::build(&w));
    let a = run(1);
    let b = run(8);
    let c = run(3);
    assert!(a.words() == b.words() && a.words() == c.words(), "atlas bytes depend on the thread count");
    assert_eq!(a.cultures(), b.cultures());
    assert_eq!(a.cache_key(), Atlas::key(&w.cfg));
}

#[test]
fn key_covers_the_inputs() {
    let base = Config::default();
    let k = Atlas::key(&base);
    let mut c = base.clone();
    c.satellite.haze = 0.5;
    c.landuse.towns = 0.0;
    assert_eq!(Atlas::key(&c), k, "settings the atlas does not read must not change its key");
    for f in [
        (|c: &mut Config| c.seed = 2) as fn(&mut Config),
        |c| c.atlas.plates = 12,
        |c| c.atlas.resolution = 256,
        |c| c.continents.threshold = 0.1,
        |c| c.relief.mountain_height_m = 1000.0,
        |c| c.climate.moisture_bias = 0.1,
        |c| c.home = None,
    ] {
        let mut c = base.clone();
        f(&mut c);
        assert_ne!(Atlas::key(&c), k);
    }
}

/// Along every face edge and corner: the samples just on either side of the edge agree (the
/// aprons continue each face into its neighbours).
#[test]
fn no_seams_across_faces() {
    let a = small();
    let r = a.resolution();
    let mut worst = [0.0f64; 15];
    let mut scale = [0.0f64; 15];
    let mut n = 0;
    for f in 0..6 {
        for side in 0..4 {
            for k in 0..200 {
                let t = -1.0 + 2.0 * (k as f64 + 0.5) / 200.0;
                let (su, sv) = match side {
                    0 => (1.0, t),
                    1 => (-1.0, t),
                    2 => (t, 1.0),
                    _ => (t, -1.0),
                };
                // a step of 1e-6 rad across the edge, and one texel inward (the local variation)
                let (inu, inv) = (
                    if side == 0 {
                        -1.0
                    } else if side == 1 {
                        1.0
                    } else {
                        0.0
                    },
                    if side == 2 {
                        -1.0
                    } else if side == 3 {
                        1.0
                    } else {
                        0.0
                    },
                );
                let eps = 1e-6;
                let p_in = dir_at(f, su + inu * eps, sv + inv * eps);
                let p_out = dir_at(f, su - inu * eps, sv - inv * eps);
                let p_far = dir_at(f, su + inu * 2.0 / r as f64, sv + inv * 2.0 / r as f64);
                assert_eq!(geom::face_of(p_in), f);
                assert_ne!(geom::face_of(p_out), f);
                let (vi, vo, vf) = (continuous(&a.sample(p_in)), continuous(&a.sample(p_out)), continuous(&a.sample(p_far)));
                for c in 0..15 {
                    worst[c] = worst[c].max((vi[c] - vo[c]).abs());
                    scale[c] += (vi[c] - vf[c]).abs();
                }
                n += 1;
            }
        }
    }
    for c in 0..15 {
        let mean_step = scale[c] / n as f64;
        eprintln!("{:12} max jump across faces {:10.4}, mean change over one texel {:10.4}", NAMES[c], worst[c], mean_step);
        // (the samples are 2e-6 rad apart: ~1e-4 of a texel)
        assert!(worst[c] <= 0.01 * mean_step + 1e-6 * vi_scale(c), "{}: jump {} vs one-texel change {}", NAMES[c], worst[c], mean_step);
    }
}

/// Typical magnitude of a channel (for the f16 rounding allowance).
fn vi_scale(c: usize) -> f64 {
    [5000.0, 3000.0, 10.0, 10.0, 3000.0, 30.0, 40.0, 1.0, 3000.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0][c]
}

#[test]
fn cache_round_trip() {
    let a = small();
    let dir = std::env::temp_dir().join(format!("terragen-atlas-test-{}", std::process::id()));
    let path = dir.join("atlas-test.bin");
    a.save(&path).unwrap();
    let b = Atlas::load(&path, a.cache_key()).unwrap();
    assert!(a.words() == b.words());
    assert_eq!(a.cultures(), b.cultures());
    assert!(Atlas::load(&path, a.cache_key() ^ 1).is_err(), "another world's key must be refused");
    // a damaged file is refused
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[1000] ^= 0x40;
    std::fs::write(&path, &bytes).unwrap();
    assert!(Atlas::load(&path, a.cache_key()).unwrap_err().to_string().contains("checksum"));
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&path, &bytes).unwrap();
    assert!(Atlas::load(&path, a.cache_key()).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Equirectangular sample grid over the globe: (lat°, lon°, sample).
fn globe(a: &Atlas, w: &World, step: f64) -> Vec<(f64, f64, AtlasSample)> {
    let mut out = Vec::new();
    let mut lat = -89.0;
    while lat < 90.0 {
        let mut lon = -180.0;
        while lon < 180.0 {
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::from_deg(lat, lon, 0.0), &w.ell);
            out.push((lat, lon, a.sample(p)));
            lon += step;
        }
        lat += step;
    }
    out
}

#[test]
fn climate_is_plausible() {
    let a = small();
    let w = World::new(cfg(128, 1));
    let g = globe(a, &w, 2.0);
    type Pt = (f64, f64, AtlasSample);
    let mean = |f: &dyn Fn(&Pt) -> Option<f64>| {
        let v: Vec<f64> = g.iter().filter_map(f).collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    let t_eq = mean(&|x| (x.0.abs() < 10.0).then_some(x.2.temp_c));
    let t_pole = mean(&|x| (x.0.abs() > 70.0).then_some(x.2.temp_c));
    assert!(t_eq > 20.0 && t_pole < 0.0, "temperature belts: equator {t_eq}, poles {t_pole}");
    // wet tropics, dry subtropics, wetter mid-latitudes (over the oceans: zonal means)
    let p_band = |lo: f64, hi: f64| mean(&|x| (x.0.abs() >= lo && x.0.abs() < hi && x.2.coast_km < -300.0).then_some(x.2.precip_mm));
    let (p_itcz, p_sub, p_mid) = (p_band(0.0, 8.0), p_band(20.0, 32.0), p_band(42.0, 60.0));
    eprintln!("ocean precipitation: ITCZ {p_itcz:.0}, subtropics {p_sub:.0}, mid-latitudes {p_mid:.0} mm/yr");
    assert!(p_itcz > 1.5 * p_sub && p_mid > 1.3 * p_sub, "precipitation belts");
    // seasonality grows inland and poleward; regime within −1..1
    let r_coast = mean(&|x| (x.0.abs() > 40.0 && x.0.abs() < 60.0 && x.2.coast_km < -200.0).then_some(x.2.temp_range_c));
    let r_inland = mean(&|x| (x.0.abs() > 40.0 && x.0.abs() < 60.0 && x.2.coast_km > 1000.0).then_some(x.2.temp_range_c));
    eprintln!("temperature range at 40–60°: ocean {r_coast:.1}, deep inland {r_inland:.1} °C");
    assert!(r_inland.is_nan() || r_inland > r_coast + 8.0);
    for (_, _, s) in &g {
        assert!(s.precip_mm >= 0.0 && s.regime.abs() <= 1.0 + 1e-3 && (-1e-6..=1.0 + 1e-6).contains(&s.glaciation), "{s:?}");
        assert!((s.plate as usize) < w.cfg.atlas.plates as usize);
        assert!((s.culture as usize) < a.cultures().len());
    }
    // every Köppen group occurs on land
    let mut groups = std::collections::HashSet::new();
    for (_, _, s) in g.iter().filter(|x| x.2.coast_km > 0.0) {
        groups.insert(format!("{:?}", koppen(s, s.elevation_m).group()));
    }
    assert!(groups.len() >= 4, "Köppen groups on land: {groups:?}");
}

/// Rain shadows: land that the wind climbs gets more rain than land it descends.
#[test]
fn rain_shadows() {
    let a = small();
    let w = World::new(cfg(128, 1));
    let g = globe(a, &w, 1.0);
    let (mut up, mut down) = (Vec::new(), Vec::new());
    for (lat, lon, s) in &g {
        if s.coast_km < 200.0 || lat.abs() > 60.0 {
            continue;
        }
        let ctx = crate::world::Ctx::new(lat.to_radians(), lon.to_radians(), 1.0, &w.ell);
        let wind = DVec3::new(s.wind_e, s.wind_n, 0.0);
        if wind.length() < 2.0 {
            continue;
        }
        let back = (ctx.p - (ctx.east * s.wind_e + ctx.north * s.wind_n) / wind.length() * 250e3).normalize();
        let rise = s.elevation_m.max(0.0) - a.sample(back).elevation_m.max(0.0);
        if rise > 300.0 {
            up.push(s.precip_mm);
        } else if rise < -300.0 {
            down.push(s.precip_mm);
        }
    }
    let m = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    eprintln!("windward {} pts {:.0} mm, leeward {} pts {:.0} mm", up.len(), m(&up), down.len(), m(&down));
    assert!(up.len() > 20 && down.len() > 20);
    assert!(m(&up) > 1.3 * m(&down), "windward slopes must be wetter than leeward ones");
}

#[test]
fn koppen_classes() {
    let k = |t: f64, range: f64, p: f64, regime: f64| {
        koppen(&AtlasSample { temp_c: t, temp_range_c: range, precip_mm: p, regime, ..Default::default() }, 0.0).code()
    };
    assert_eq!(k(27.5, 1.5, 2300.0, 0.0), "Af"); // Singapore
    assert_eq!(k(26.0, 4.0, 900.0, 0.8), "Aw"); // savanna
    assert_eq!(k(22.0, 14.0, 25.0, 0.0), "BWh"); // Cairo
    assert_eq!(k(11.0, 13.0, 600.0, 0.0), "Cfb"); // London
    assert_eq!(k(15.5, 17.0, 800.0, -0.6), "Csa"); // Rome
    assert_eq!(k(5.8, 26.0, 700.0, 0.2), "Dfb"); // Moscow
    assert_eq!(k(-3.0, 30.0, 450.0, 0.3), "Dfc"); // taiga
    assert_eq!(k(-8.0, 24.0, 250.0, 0.0), "ET"); // tundra
    assert_eq!(k(-30.0, 25.0, 100.0, 0.0), "EF");
    // the lapse rate: a tropical plateau at 2.5 km is temperate
    let s = AtlasSample { temp_c: 26.0, temp_range_c: 3.0, precip_mm: 1200.0, regime: 0.6, ..Default::default() };
    assert_eq!(koppen(&s, 2500.0).group(), KoppenGroup::C);
}

#[test]
fn build_time() {
    if cfg!(debug_assertions) {
        eprintln!("build_time: skipped in debug builds");
        return;
    }
    let w = World::new(Config::default());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let t = std::time::Instant::now();
    let a = pool.install(|| Atlas::build(&w));
    let s = t.elapsed().as_secs_f64();
    eprintln!("atlas {}² × 6 (default config) on 8 threads: {s:.2} s, {} MB, {} cultures", a.resolution(), (a.words().len() * 4) >> 20, a.cultures().len());
    assert!(s < 30.0, "atlas build took {s:.1} s");
    assert!(a.words().len() * 4 <= 64 << 20, "the default atlas must stay under 64 MB");
}

#[cfg(feature = "gpu")]
#[test]
fn wgsl_constants_match() {
    let src = crate::gpu::ATLAS_WGSL;
    for (name, v) in [("ATLAS_APRON", APRON), ("ATLAS_WORDS", WORDS), ("ATLAS_HEADER", HEADER)] {
        assert!(src.contains(&format!("const {name}: u32 = {v}u;")), "{name} differs between atlas.wgsl and atlas/mod.rs");
    }
}

#[test]
fn sample_reads_texel_values() {
    // at a texel centre far from the edges the B-spline gives (1·4·1)/6-weighted neighbours:
    // the value stays within the range of the 3 × 3 neighbourhood
    let a = small();
    let r = a.resolution();
    let g = Grid::new(r);
    for k in (0..g.n()).step_by(997) {
        let (f, i, j) = g.fij(k);
        let s = a.sample(g.dir(k));
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for dj in 0..3 {
            for di in 0..3 {
                let v = f16_value(a.raw(f, i + 1 + di, j + 1 + dj, slot::TEMP)) as f64;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        assert!(s.temp_c >= lo - 1e-3 && s.temp_c <= hi + 1e-3, "{k}: {} not in [{lo}, {hi}]", s.temp_c);
        let raw = |sl| a.raw(f, i + APRON, j + APRON, sl) as u32;
        assert_eq!(s.plate, raw(slot::PLATE) & 63);
        assert_eq!(s.culture, raw(slot::CULTURE) & 0xFFF);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_sampler_matches_the_cpu() {
    use crate::gpu::{output, read_back, shared, storage};
    let Ok(g) = shared() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let a = small();
    const TEST: &str = r#"
@group(1) @binding(0) var<storage, read> dirs: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;

const NO: u32 = 19u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&dirs)) {
        return;
    }
    let s = atlas_sample(dirs[i].xyz);
    let o = i * NO;
    out[o + 0u] = s.elevation;
    out[o + 1u] = s.coast_km;
    out[o + 2u] = s.wind.x;
    out[o + 3u] = s.wind.y;
    out[o + 4u] = s.precip_mm;
    out[o + 5u] = s.temp_c;
    out[o + 6u] = s.temp_range_c;
    out[o + 7u] = s.regime;
    out[o + 8u] = s.plate_dist_km;
    out[o + 9u] = s.convergence;
    out[o + 10u] = s.uplift;
    out[o + 11u] = s.volcanism;
    out[o + 12u] = s.glaciation;
    out[o + 13u] = s.development;
    out[o + 14u] = s.population;
    out[o + 15u] = f32(atlas_plate_id(s));
    out[o + 16u] = f32(atlas_boundary(s));
    out[o + 17u] = f32(atlas_litho(s) + 8u * atlas_litho2(s));
    out[o + 18u] = f32(atlas_culture_id(s) + 4096u * atlas_archetype(s));
}
"#;
    // random directions, plus points on and near the face edges and corners
    let mut dirs: Vec<[f32; 4]> = Vec::new();
    let mut h = 99u64;
    let mut rnd = || {
        h = mix64(h.wrapping_add(1));
        u01(h)
    };
    for k in 0..20000 {
        let d = if k % 4 == 0 {
            let f = (rnd() * 6.0) as usize % 6;
            let e = if rnd() < 0.5 { 1.0 - 1e-4 * rnd() } else { -1.0 + 1e-4 * rnd() };
            let t = 2.0 * rnd() - 1.0;
            if rnd() < 0.5 {
                dir_at(f, e, t)
            } else {
                dir_at(f, t, e)
            }
        } else {
            let z = 2.0 * rnd() - 1.0;
            let ph = std::f64::consts::TAU * rnd();
            let s = (1.0 - z * z).sqrt();
            DVec3::new(s * ph.cos(), s * ph.sin(), z) * (0.5 + rnd() * 1e7)
        };
        let d32 = d.as_vec3();
        dirs.push([d32.x, d32.y, d32.z, 0.0]);
    }
    let src = format!("{}{}", crate::gpu::ATLAS_WGSL, TEST);
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("atlas-test"), source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pipe = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let b_atlas = storage(&g.device, "atlas", a.words());
    let b_dirs = storage(&g.device, "dirs", &dirs);
    let no = 19usize;
    let b_out = output(&g.device, "out", (dirs.len() * no * 4) as u64);
    let bg0 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 5, resource: b_atlas.as_entire_binding() }],
    });
    let bg1 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b_dirs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_out.as_entire_binding() },
        ],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg0, &[]);
        cp.set_bind_group(1, &bg1, &[]);
        cp.dispatch_workgroups((dirs.len() as u32).div_ceil(64), 1, 1);
    }
    g.queue.submit([enc.finish()]);
    let out: Vec<f32> = read_back(&g, &b_out, dirs.len() * no).unwrap();
    let mut worst = [0.0f64; 15];
    let mut discrete_diff = 0;
    let mut near_edge = 0;
    for (i, d) in dirs.iter().enumerate() {
        // the CPU samples the same f32 direction
        let dir = DVec3::new(d[0] as f64, d[1] as f64, d[2] as f64);
        let s = a.sample(dir);
        let cpu = continuous(&s);
        for c in 0..15 {
            let e = (out[i * no + c] as f64 - cpu[c]).abs() / (1.0 + vi_scale(c));
            worst[c] = worst[c].max(e);
        }
        let disc = [s.plate as f32, s.boundary as u32 as f32, (s.litho as u32 + 8 * s.litho2 as u32) as f32, (s.culture + 4096 * s.archetype as u32) as f32];
        if (0..4).any(|k| out[i * no + 15 + k] != disc[k]) {
            if a.texel_edge_distance(dir) < 1e-3 {
                near_edge += 1;
            } else {
                discrete_diff += 1;
            }
        }
    }
    eprintln!("GPU vs CPU atlas sampling, worst errors relative to the channel scale: {worst:?}; discrete differences {discrete_diff} (+{near_edge} within 1e-3 texel of a texel border)");
    for (c, &e) in worst.iter().enumerate() {
        assert!(e < 1e-4, "{}: error {e}", NAMES[c]);
    }
    assert_eq!(discrete_diff, 0);
}
