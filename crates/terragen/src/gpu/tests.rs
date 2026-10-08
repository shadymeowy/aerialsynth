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
    let Some(g) = gpu() else { return };
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
    let pipe = pipeline(&g.device, "noise-test", &src, "main");
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
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: b_pts.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: b_out.as_entire_binding() }],
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
