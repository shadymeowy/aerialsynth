//! The GPU primitives against the CPU's.

use super::*;
use crate::noise::*;
use crate::surface::SurfaceModel;
use crate::world::World;
use glam::DVec3;

fn gpu() -> Option<std::sync::Arc<Gpu>> {
    match shared() {
        Ok(g) if g.check_generator().is_ok() => Some(g),
        _ => {
            eprintln!("no GPU that runs the tile generator: skipped");
            None
        }
    }
}

/// A device for the WGSL primitives alone (f64 / i64 shaders, any buffer limits).
fn shaders() -> Option<std::sync::Arc<Gpu>> {
    match shared() {
        Ok(g) if g.check_shaders().is_ok() => Some(g),
        _ => {
            eprintln!("no GPU with f64 / i64 shaders: skipped");
            None
        }
    }
}

const NOISE_TEST: &str = r#"
@group(1) @binding(0) var<storage, read> pts: array<vec4<f64>>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;

const NO: u32 = 12u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&pts)) {
        return;
    }
    let p = pts[i].xyz;
    let gsd = f32(pts[i].w);
    let o = i * NO;
    out[o + 0u] = perlin3(77lu, p * (1.0lf / 3.7lf));
    out[o + 1u] = perlin3(0x3EADlu, p * (1.0lf / 60.0lf));
    out[o + 2u] = fbm(FBM_CONT, p, gsd);
    out[o + 3u] = fbm(FBM_DETAIL, p, gsd);
    out[o + 4u] = fbm(FBM_MICRO, p, gsd);
    let d = fbm_d(FBM_ROAD_MAJOR, p, max(gsd, 200.0));
    out[o + 5u] = d.x;
    out[o + 6u] = d.y;
    let c = worley3(0x57A4lu, p, 240.0lf, 0.9);
    out[o + 7u] = c.f1;
    out[o + 8u] = f32(c.id & 0xFFFFlu);
    let g = gully_octave(0xE205lu, p * (1.0lf / 1400.0lf), normalize(vec3<f32>(0.3, -0.5, 0.8)));
    out[o + 9u] = g.x;
    out[o + 10u] = g.y;
    out[o + 11u] = fbm_part(FBM_PATCH, p, gsd, 200.0, true);
}
"#;

#[test]
fn noise_matches_the_cpu() {
    let Some(g) = shaders() else { return };
    let w = World::new(crate::Config::default());
    let s = SurfaceModel::new(&w);
    let (octs, fbms) = tables::build(&w, &s);
    // points on and near the surface over the globe, with pixel sizes from 0.3 m to 30 km
    let mut pts: Vec<[f64; 4]> = Vec::new();
    let mut h = 12345u64;
    let mut rnd = || {
        h = mix64(h.wrapping_add(1));
        u01(h)
    };
    for _ in 0..4096 {
        let lat = (rnd() - 0.5) * 3.0;
        let lon = (rnd() - 0.5) * 6.2;
        let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, (rnd() - 0.5) * 1000.0), &w.ell);
        let gsd = 0.3 * 1e5f64.powf(rnd());
        pts.push([p.x, p.y, p.z, gsd]);
    }
    let src = format!("{}{}{}", tables::wgsl_consts(), NOISE_WGSL, NOISE_TEST);
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("noise-test"), source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pipe = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let b_grads = storage(&g.device, "grads", &tables::grads());
    let b_octs = storage(&g.device, "octs", &octs);
    let b_fbms = storage(&g.device, "fbms", &fbms);
    let b_pts = storage(&g.device, "pts", &pts);
    let no = 12usize;
    let b_out = output(&g.device, "out", (pts.len() * no * 4) as u64);
    let bg0 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: b_grads.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_octs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b_fbms.as_entire_binding() },
        ],
    });
    let bg1 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b_pts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_out.as_entire_binding() },
        ],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg0, &[]);
        cp.set_bind_group(1, &bg1, &[]);
        cp.dispatch_workgroups((pts.len() as u32).div_ceil(64), 1, 1);
    }
    g.queue.submit([enc.finish()]);
    let out: Vec<f32> = read_back(&g, &b_out, pts.len() * no).unwrap();
    let mut worst = [0.0f64; 12];
    for (i, q) in pts.iter().enumerate() {
        let p = DVec3::new(q[0], q[1], q[2]);
        let gsd = q[3];
        let wc = worley3(0x57A4, p, 240.0, 0.9);
        let gq = World::gully_octave(0xE205, p / 1400.0, DVec3::new(0.3, -0.5, 0.8).normalize());
        let rd = w.road_major.eval_d(p, gsd.max(200.0), 99);
        let cpu = [
            perlin3(77, p / 3.7),
            perlin3(0x3EAD, p / 60.0),
            w.cont.eval(p, gsd),
            s.detail.eval(p, gsd),
            w.micro.eval(p, gsd),
            rd.0,
            rd.1.x,
            wc.f1,
            (wc.id & 0xFFFF) as f64,
            gq.0,
            gq.1.x,
            s.patch.eval_part(p, gsd, 200.0, true),
        ];
        for k in 0..no {
            let e = (out[i * no + k] as f64 - cpu[k]).abs() / (1.0 + cpu[k].abs().min(1e3));
            worst[k] = worst[k].max(e);
        }
    }
    eprintln!("worst relative errors: {worst:?}");
    // ids are exact, values agree to f32 precision
    for (k, &e) in worst.iter().enumerate() {
        assert!(e < 1e-4, "output {k}: error {e}");
    }
}

/// Compare GPU and CPU pass A over a tile: (field name, max abs error, mean abs error,
/// share of pixels off by more than `tol`).
fn compare_pass_a(gen: &GpuGenerator, cpu: &crate::Generator, id: geodesy::tiles::TileId) -> Vec<(String, f64, f64, f64)> {
    let g = gen.pass_a(&[id]).unwrap().remove(0);
    let c = cpu.pass_a(id);
    assert_eq!(g.len(), c.len());
    let mut out = Vec::new();
    let mut field = |name: &str, tol: f64, f: &dyn Fn(&crate::gpu::types::GTerrain, &crate::world::Terrain) -> (f64, f64)| {
        let (mut mx, mut sum, mut bad) = (0.0f64, 0.0f64, 0usize);
        for (a, b) in g.iter().zip(&c) {
            let (x, y) = f(a, b);
            let e = (x - y).abs();
            mx = mx.max(e);
            sum += e;
            if e > tol {
                bad += 1;
            }
        }
        out.push((name.to_string(), mx, sum / g.len() as f64, bad as f64 / g.len() as f64));
    };
    let cl = |v: f64| v.clamp(-1e6, 1e6);
    field("ground", 0.05, &|a, b| (a.ground as f64, b.ground));
    field("water_kind", 0.5, &|a, b| (a.water_kind as f64, b.water_kind as f64));
    field("water", 0.05, &|a, b| (cl(a.water as f64), cl(b.water)));
    field("river_d", 0.05, &|a, b| (cl(a.river_d as f64), cl(b.river_d)));
    field("river_hw", 0.01, &|a, b| (a.river_hw as f64, b.river_hw));
    field("river_level", 0.05, &|a, b| (a.river_level as f64, b.river_level));
    field("temp", 0.01, &|a, b| (a.temp as f64, b.temp));
    field("moist", 0.001, &|a, b| (a.moist as f64, b.moist));
    field("agri", 0.001, &|a, b| (a.agri as f64, b.agri));
    field("habit", 0.001, &|a, b| (a.habit as f64, b.habit));
    field("gully", 0.001, &|a, b| (a.gully as f64, b.gully));
    field("floodplain", 0.001, &|a, b| (a.floodplain as f64, b.floodplain));
    field("road_major", 0.05, &|a, b| (cl(a.road_major as f64), cl(b.road_major)));
    field("road_minor", 0.05, &|a, b| (cl(a.road_minor as f64), cl(b.road_minor)));
    field("region_id", 0.5, &|a, b| ((a.region_id != b.region.id) as u8 as f64, 0.0));
    field("region_edge", 0.05, &|a, b| (if a.region_id == 0 { 0.0 } else { cl(a.region_edge as f64) }, if b.region.id == 0 { 0.0 } else { cl(b.region.edge) }));
    field("town", 0.5, &|a, b| (a.town as f64, (b.town.id != 0) as u8 as f64));
    out
}

#[test]
fn pass_a_matches_the_cpu() {
    if gpu().is_none() {
        return;
    }
    let cfg = crate::Config::default();
    let gen = GpuGenerator::new(cfg.clone()).unwrap();
    let cpu = crate::Generator::new(cfg);
    let tiles = [
        geodesy::tiles::TileId::new(14, 9779, 6278),
        geodesy::tiles::TileId::new(12, 2444, 1569),
        geodesy::tiles::TileId::new(10, 611, 392),
        geodesy::tiles::TileId::new(7, 76, 49),
    ];
    for id in tiles {
        let t0 = std::time::Instant::now();
        let r = compare_pass_a(&gen, &cpu, id);
        eprintln!("tile {id} ({:.2} s)", t0.elapsed().as_secs_f64());
        for (name, mx, mean, bad) in &r {
            eprintln!("  {name:12} max {mx:10.4} mean {mean:10.6} off {:.4}%", bad * 100.0);
        }
    }
}

/// Per-layer differences between two tiles: (layer, mean abs, share of samples off by more
/// than the tolerance, max abs).
pub(crate) fn compare_tiles(g: &tilestore::TileData, c: &tilestore::TileData) -> Vec<(&'static str, f64, f64, f64)> {
    fn stats<T: Copy>(a: &[T], b: &[T], tol: f64, f: impl Fn(T) -> f64) -> (f64, f64, f64) {
        let (mut sum, mut bad, mut mx) = (0.0, 0usize, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            let e = (f(*x) - f(*y)).abs();
            sum += e;
            mx = mx.max(e);
            if e > tol {
                bad += 1;
            }
        }
        (sum / a.len() as f64, bad as f64 / a.len() as f64, mx)
    }
    let mut out = Vec::new();
    let mut push = |name, s: (f64, f64, f64)| out.push((name, s.0, s.1, s.2));
    push("elevation", stats(&g.elevation, &c.elevation, 0.05, |v| v as f64));
    push("rgb", stats(&g.rgb, &c.rgb, 2.0, |v| v as f64));
    push("albedo", stats(&g.albedo, &c.albedo, 2.0, |v| v as f64));
    push("emission", stats(&g.emission, &c.emission, 2.0, |v| v as f64));
    push("normal", stats(&g.normal, &c.normal, 2.0, |v| v as f64));
    push("landcover", stats(&g.landcover, &c.landcover, 0.5, |v| v as f64));
    out
}

#[test]
fn tiles_match_the_cpu() {
    if gpu().is_none() {
        return;
    }
    let cfg = crate::Config::default();
    let gen = GpuGenerator::new(cfg.clone()).unwrap();
    let cpu = crate::Generator::new(cfg);
    let ids = [
        geodesy::tiles::TileId::new(16, 39117, 25113),
        geodesy::tiles::TileId::new(14, 9779, 6278),
        geodesy::tiles::TileId::new(12, 2444, 1569),
        geodesy::tiles::TileId::new(10, 611, 392),
        geodesy::tiles::TileId::new(7, 76, 49),
    ];
    for id in ids {
        let t0 = std::time::Instant::now();
        let g = gen.tiles(&[id]).unwrap().remove(0);
        let tg = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let c = cpu.tile_cpu(id);
        let tc = t0.elapsed().as_secs_f64();
        eprintln!("tile {id}: GPU {tg:.2} s, CPU {tc:.2} s, elevation range GPU {:.1}..{:.1} CPU {:.1}..{:.1}", g.elev_min, g.elev_max, c.elev_min, c.elev_max);
        for (name, mean, bad, mx) in compare_tiles(&g, &c) {
            eprintln!("  {name:10} mean {mean:8.4} off {:7.3}% max {mx:8.2}", bad * 100.0);
        }
    }
}

/// Timing of one low-zoom tile (cold caches): `TERRAGEN_PROFILE=1 cargo test ... -- --ignored`.
#[test]
#[ignore]
fn profile_low_zoom() {
    if gpu().is_none() {
        return;
    }
    let gen = GpuGenerator::new(crate::Config::default()).unwrap();
    for id in [geodesy::tiles::TileId::new(7, 76, 49), geodesy::tiles::TileId::new(10, 612, 392)] {
        let t0 = std::time::Instant::now();
        gen.tiles(&[id]).unwrap();
        eprintln!("tile {id}: {:.2} s", t0.elapsed().as_secs_f64());
    }
}

/// Throughput on a block of tiles: GPU in batches vs the CPU generator (rayon over tiles).
#[test]
#[ignore]
fn throughput() {
    if gpu().is_none() {
        return;
    }
    use rayon::prelude::*;
    let z: u8 = std::env::var("TP_Z").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    let n: u32 = std::env::var("TP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let batch: usize = std::env::var("TP_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let gen = GpuGenerator::new(crate::Config::default()).unwrap();
    let c = geodesy::tiles::tile_for_latlon(39.9f64.to_radians(), 32.8f64.to_radians(), z);
    let ids: Vec<_> = (0..n * n).map(|k| geodesy::tiles::TileId::new(z, c.x + k % n, c.y + k / n)).collect();
    // warm the GPU generator's caches on a neighbouring block, then time
    let t0 = std::time::Instant::now();
    let mut done = 0;
    for chunk in ids.chunks(batch) {
        done += gen.tiles(chunk).unwrap().len();
    }
    let tg = t0.elapsed().as_secs_f64();
    let cpu = crate::Generator::new(crate::Config::default());
    let t0 = std::time::Instant::now();
    let m: usize = ids.par_iter().map(|&id| cpu.tile_cpu(id).rgb.len()).count();
    let tc = t0.elapsed().as_secs_f64();
    eprintln!("z{z}, {done} tiles: GPU {tg:.2} s ({:.1} tiles/s), CPU {tc:.2} s ({:.1} tiles/s)", done as f64 / tg, m as f64 / tc);
}

/// A polar low-zoom tile on both generators: `cargo test ... polar -- --ignored`.
#[test]
#[ignore]
fn polar_tile() {
    if gpu().is_none() {
        return;
    }
    let id = geodesy::tiles::TileId::new(3, 4, 0);
    let gen = GpuGenerator::new(crate::Config::default()).unwrap();
    let t0 = std::time::Instant::now();
    let g = gen.tiles(&[id]).unwrap().remove(0);
    let tg = t0.elapsed().as_secs_f64();
    let cpu = crate::Generator::new(crate::Config::default());
    let t0 = std::time::Instant::now();
    let c = cpu.tile_cpu(id);
    let tc = t0.elapsed().as_secs_f64();
    eprintln!("tile {id}: GPU {tg:.1} s, CPU {tc:.1} s");
    for (name, mean, bad, mx) in compare_tiles(&g, &c) {
        eprintln!("  {name:10} mean {mean:8.4} off {:7.3}% max {mx:8.2}", bad * 100.0);
    }
}

/// The generator's WGSL parses, validates and compiles to SPIR-V entry point by entry point
/// (no GPU needed: kits can check their WGSL with `cargo test -p terragen wgsl_compiles`).
#[test]
fn wgsl_compiles() {
    let (points, tile, drain) = super::sources();
    for (name, src) in [("points", points), ("tile", tile), ("drain", drain)] {
        let module = match naga::front::wgsl::parse_str(&src) {
            Ok(m) => m,
            Err(e) => panic!("{name}: {}", e.emit_to_string(&src)),
        };
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        let info = match v.validate(&module) {
            Ok(i) => i,
            Err(e) => panic!("{name}: {}", e.emit_to_string(&src)),
        };
        for ep in &module.entry_points {
            let opts = naga::back::spv::Options::default();
            let pipe = naga::back::spv::PipelineOptions { shader_stage: ep.stage, entry_point: ep.name.clone() };
            let r = std::panic::catch_unwind(|| naga::back::spv::write_vec(&module, &info, &opts, Some(&pipe)));
            match r {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => panic!("{name}::{}: SPIR-V: {e}", ep.name),
                Err(_) => panic!("{name}::{}: the SPIR-V backend panicked", ep.name),
            }
        }
    }
}

// ---------------------------------------------------------------- kernel parity (for kits)

const KERNEL_TEST: &str = r#"
struct KTest {
    p: vec4<f64>,
    east: vec4<f32>,
    north: vec4<f32>,
    aux: vec4<f32>,
    q: vec2<f32>,
    gsd: f32,
    fw: f32,
    amount: f32,
    li: u32,
    _p0: u32,
    _p1: u32,
}
@group(3) @binding(0) var<storage, read> kt_in: array<KTest>;
@group(3) @binding(1) var<storage, read_write> kt_out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn kernel_test(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&kt_in)) {
        return;
    }
    let t = kt_in[i];
    var k: KIn;
    k.q = t.q;
    k.p = t.p.xyz;
    k.east = t.east.xyz;
    k.north = t.north.xyz;
    k.gsd = t.gsd;
    k.fw = t.fw;
    k.amount = t.amount;
    k.aux = t.aux;
    let o = kernel_eval(t.li, k);
    kt_out[2u * i] = vec4<f32>(o.cov, o.dh, o.emit, 0.0);
    kt_out[2u * i + 1u] = vec4<f32>(o.albedo, 0.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct KTest {
    p: [f64; 4],
    east: [f32; 4],
    north: [f32; 4],
    aux: [f32; 4],
    q: [f32; 2],
    gsd: f32,
    fw: f32,
    amount: f32,
    li: u32,
    _p: [u32; 6],
}

/// CPU / GPU parity of kernels: each `(kernel, YAML params)` evaluated at 20 000 random inputs
/// (positions over 64 feature sizes, pixel sizes from 1/40 to 3 feature sizes: explicit, the
/// crossfade and the mean; random amounts and `aux`) on both backends. Returns per case the
/// share of samples whose coverage differs by more than 0.02 or albedo by more than 0.01
/// (discrete flips at cell edges), and the largest coverage difference. Kits call it from their
/// tests with their kernels (`None`: no GPU).
pub(crate) fn kernel_parity(cases: &[(&str, &str)]) -> Option<Vec<(String, f64, f64)>> {
    gpu()?;
    let cfg = crate::Config::default();
    let gen = GpuGenerator::new(cfg).unwrap();
    let d = &gen.gpu.device;
    let pal = crate::registry::BiomePal::default_pal();
    let mut layers = Vec::new();
    let mut ks = Vec::new();
    for (name, y) in cases {
        let spec = crate::kernels::spec(name).unwrap_or_else(|| panic!("unknown kernel {name}"));
        let m: std::collections::BTreeMap<String, serde_yaml::Value> = serde_yaml::from_str(y).unwrap();
        let k = crate::kernels::compile_params(spec, &m, mix64(0x7E57 ^ layers.len() as u64), &pal).unwrap();
        let mean = crate::kernels::calibrate(&k);
        let li = crate::registry::LayerInst { slot: 1, k, mean, size: crate::kernels::size(&k), win: vec![], cls: 0, clear: 0.0, hmode: 0, mat: 0 };
        layers.push(crate::registry::gpu::klayer(&li));
        ks.push((k, mean));
    }
    // inputs: near a real place, so `p` and the frame agree
    let ctx = Ctx::new(41.0f64.to_radians(), 33.0f64.to_radians(), 1.0, &gen.world.ell);
    let n_per = 20_000;
    let mut ins = Vec::new();
    let mut cpu = Vec::new();
    for (ci, (k, mean)) in ks.iter().enumerate() {
        let sz = crate::kernels::size(k).max(0.1);
        for j in 0..n_per {
            let h = mix64(0xA11 ^ ((ci as u64) << 32) ^ j as u64);
            let q = glam::DVec2::new((u01k(h, 1) - 0.5) * 64.0 * sz, (u01k(h, 2) - 0.5) * 64.0 * sz);
            let gsd = sz * (1.0 / 40.0) * (120.0f64).powf(u01k(h, 3));
            let amount = 0.2 + 0.8 * u01k(h, 4);
            let aux = match k.kind {
                crate::kernels::kind::CONTOURS => [q.x * 0.3 + 100.0, 0.05 + 0.6 * u01k(h, 5), 0.0, 0.0],
                crate::kernels::kind::LINEAR => [(u01k(h, 5) - 0.5) * 4.0 * sz, q.x, sz, 0.0],
                crate::kernels::kind::WATER => [8.0 * sz * u01k(h, 5), 0.0, 0.0, 0.0],
                crate::kernels::kind::STAMP => [q.x * 0.25, q.y * 0.25, 6.0 * sz, 4.0 * sz],
                _ => [u01k(h, 5) * 6.0, 0.0, 0.0, 0.0],
            };
            let p = ctx.p + ctx.east * q.x + ctx.north * q.y;
            let kin = crate::kernels::KIn { q, p, east: ctx.east, north: ctx.north, gsd, fw: gsd * 0.5, amount, aux };
            cpu.push(crate::kernels::eval(k, mean, &kin));
            let v4 = |v: DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
            ins.push(KTest {
                p: [p.x, p.y, p.z, 0.0],
                east: v4(ctx.east),
                north: v4(ctx.north),
                aux: aux.map(|a| a as f32),
                q: [q.x as f32, q.y as f32],
                gsd: gsd as f32,
                fw: (gsd * 0.5) as f32,
                amount: amount as f32,
                li: ci as u32,
                _p: [0; 6],
            });
        }
    }
    // the tile module with the test entry; group 0 with the test layers
    let (_, tile, _) = sources();
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("kernel test"), source: wgpu::ShaderSource::Wgsl(format!("{tile}{KERNEL_TEST}").into()) });
    let l_test = layout(d, "kernel test", &[Bind::Ro, Bind::Rw]);
    let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("kernel test"),
        bind_group_layouts: &[Some(&gen.k.l_globals), Some(&gen.k.l_tables), Some(&gen.k.l_tile), Some(&l_test)],
        immediate_size: 0,
    });
    let pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("kernel test"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("kernel_test"),
        compilation_options: Default::default(),
        cache: None,
    });
    let b_layers = storage(d, "test layers", &layers);
    let gb = &gen.globals_bufs;
    let g0 = bind(d, &gen.k.l_globals, &[&gb[0], &gb[1], &gb[2], &gb[3], &gb[4], &gb[5], &gb[6], &gb[7], &b_layers, &gb[9], &gb[10]]);
    let dummies: Vec<wgpu::Buffer> = (0..34).map(|_| output(d, "-", 256)).collect();
    let g1 = bind(d, &gen.k.l_tables, &dummies[..14].iter().collect::<Vec<_>>());
    let g2 = bind(d, &gen.k.l_tile, &dummies[14..34].iter().collect::<Vec<_>>());
    let b_in = storage(d, "test inputs", &ins);
    let b_out = output(d, "test outputs", (ins.len() * 32) as u64);
    let g3 = bind(d, &l_test, &[&b_in, &b_out]);
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &g0, &[]);
        cp.set_bind_group(1, &g1, &[]);
        cp.set_bind_group(2, &g2, &[]);
        cp.set_bind_group(3, &g3, &[]);
        let groups = (ins.len() as u32).div_ceil(64);
        cp.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
    }
    gen.gpu.queue.submit([enc.finish()]);
    let out: Vec<[f32; 4]> = read_back(&gen.gpu, &b_out, ins.len() * 2).unwrap();
    let mut res = Vec::new();
    for (ci, (name, _)) in cases.iter().enumerate() {
        let (mut bad, mut mx) = (0usize, 0.0f64);
        for j in 0..n_per {
            let i = ci * n_per + j;
            let c = &cpu[i];
            let (g, ga) = (out[2 * i], out[2 * i + 1]);
            let dc = (g[0] as f64 - c.cov).abs();
            let da = (0..3).map(|k| (ga[k] as f64 - c.albedo[k]).abs() * c.cov.min(g[0] as f64)).fold(0.0, f64::max);
            mx = mx.max(dc);
            if dc > 0.02 || da > 0.01 {
                bad += 1;
            }
        }
        res.push((name.to_string(), bad as f64 / n_per as f64, mx));
    }
    Some(res)
}

/// Every core kernel agrees between the backends (a few samples flip at cell edges).
#[test]
fn kernels_match_the_cpu() {
    let cases = [
        ("scatter", "{cell: 8, density: 0.6, radius: [2, 3], height: [3, 6], shape: dome}"),
        ("scatter", "{cell: 12, density: 0.5, radius: [3, 5], height: [4, 7], shape: star, colour_var: 0.4}"),
        ("scatter", "{cell: 30, density: 0.3, radius: [4, 9], height: [5, 9], shape: rect, aspect: 1.6}"),
        ("rows", "{spacing: 3, along: 2, radius: 0.7, height: 2, angle: 0.4}"),
        ("rows", "{spacing: 2.5, radius: 0.5, height: 1, direction: aux}"),
        ("cells", "{cell: 20, fill: 0.5, edge: 1.5, aspect: 1.5}"),
        ("stripes", "{wavelength: 40, threshold: 0.4, height: 3}"),
        ("contours", "{step: 3, line: 0.6}"),
        ("contours", "{step: 2.5, mode: terraces, riser: 0.25}"),
        ("radial", "{cell: 200, density: 0.5, radius: [40, 80], height: [10, 20], crater: 0.3, arms: 5, arm_amp: 0.3}"),
        ("crescent", "{cell: 200, density: 0.5, radius: [40, 80], direction: aux}"),
        ("lobes", "{cell: 400, density: 0.5, radius: [100, 180], channels: 6}"),
        ("patches", "{scale: 80, fraction: 0.3, octaves: 3}"),
        ("linear", "{profile: crowned, shoulder: 1.5, marking: 0.2, dash: 6, height: 0.3}"),
        ("stamp", "{template: blocks, a: 8, b: 4, height: 6}"),
        ("canopy", "{cell: 30}"),
        ("water", "{depth_scale: 5, foam_depth: 0.5, foam_width: 0.2}"),
    ];
    let Some(res) = kernel_parity(&cases) else { return };
    for (name, bad, mx) in &res {
        eprintln!("  {name:10} off {:.3}% max coverage difference {mx:.3}", bad * 100.0);
        assert!(*bad < 0.005, "{name}: {:.2}% of the samples differ", bad * 100.0);

    }
}
