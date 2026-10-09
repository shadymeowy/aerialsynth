//! Headless wgpu backend of the renderer (see docs/gpu.md).
//!
//! The CPU selects the LOD units and, once per unit and vertex stride, builds the unit's mesh
//! geometry (f64, the CPU renderer's code); meshes stay on the GPU relative to a per-unit origin.
//! Per frame the GPU projects them through the camera model (vertex shader), rasterizes a
//! supersampled G-buffer and shades it in a compute pass that ports the CPU shading. Tiles live in
//! a GPU slot pool. Over a flight each tile and each mesh crosses the bus once; per frame only
//! small per-unit transforms and tables are uploaded. One device for the whole process.

pub mod device;
pub mod events;
pub mod tiles;

use crate::lighting::SunState;
use crate::lod::Unit;
use crate::raster::{FrameOut, Renderer, Shading, TileView};
use crate::trajectory::CamPose;
use bytemuck::{Pod, Zeroable};
use device::Gpu;
use geodesy::tiles::TileId;
use glam::DVec3;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tiles::TilePool;
use tilestore::TileData;
use wgpu::util::DeviceExt;

/// Tile slots of the GPU pool (≈1.1 MB of VRAM each): as many as a texture array can hold, at
/// most 2048 (≈2.2 GB). A frame never needs more than a few hundred to ~1500 tiles.
pub const POOL_SLOTS: u32 = 2048;

/// Budget of GPU memory for cached meshes (bytes).
pub const MESH_BUDGET: u64 = 1536 << 20;

/// Can the GPU backend render `model` at supersample `ss`: a GPU, the projection within the
/// shader's parameter block, the supersampled G-buffer within the texture size limit and the ray
/// table / output buffers within the buffer size limits?
pub fn supports(model: &dyn crate::camera::CameraModel, ss: u32) -> Result<(), String> {
    model.gpu_fits()?;
    let gpu = device::shared().map_err(|e| format!("no usable GPU ({e:#})"))?;
    let l = gpu.device.limits();
    let (w, h) = (model.width() as u64 * ss.max(1) as u64, model.height() as u64 * ss.max(1) as u64);
    if w.max(h) > l.max_texture_dimension_2d as u64 {
        return Err(format!("{w}×{h} px (supersample {ss}) exceeds the GPU's texture size limit of {} px", l.max_texture_dimension_2d));
    }
    let buf = l.max_storage_buffer_binding_size.min(l.max_buffer_size);
    let rays = w * h * 16;
    if rays > buf {
        return Err(format!("the {w}×{h} ray table ({} MB) exceeds the GPU's buffer size limit ({} MB)", rays >> 20, buf >> 20));
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    /// relative to the unit's origin (ECEF axes)
    pos: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshUniforms {
    rt0: [f32; 4],
    rt1: [f32; 4],
    rt2: [f32; 4],
    p: [[f32; 4]; 4],
    lim: [f32; 4],
    kind: [u32; 4],
}

/// A unit's mesh is fixed by the unit, its vertex stride and which neighbour tiles exist (corner
/// heights average neighbouring pixels, falling back to the tile itself where one is missing).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MeshKey {
    id: TileId,
    data: TileId,
    rect: [u32; 4],
    stride: u32,
    neighbours: u16,
}

struct CachedMesh {
    vb: wgpu::Buffer,
    origin: DVec3,
    nx: usize,
    ny: usize,
    bytes: u64,
    last: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    dims: [u32; 4],
    cfg: [u32; 4],
    cam_pos: [f32; 4],
    r0: [f32; 4],
    r1: [f32; 4],
    r2: [f32; 4],
    cam_up: [f32; 4],
    sun_dir: [f32; 4],
    moon_dir: [f32; 4],
    sun_col: [f32; 4],
    moon_col: [f32; 4],
    ray_col: [f32; 4],
    mie_col: [f32; 4],
    zenith: [f32; 4],
    horizon: [f32; 4],
    sun: [f32; 4],
    sun2: [f32; 4],
    flick: [f32; 4],
    flick2: [f32; 4],
    ell: [f32; 4],
}

const F_RELIT: u32 = 1;
const F_GLINT: u32 = 2;
const F_SHADOWS: u32 = 4;
const F_SPLIT: u32 = 8;
const F_GEOM: u32 = 16;
const F_ATMO: u32 = 32;
const F_FLICKER: u32 = 64;
const F_LIGHTS: u32 = 128;
const F_POLLUTION: u32 = 256;

/// Render targets and read-back buffers of one resolution.
struct Targets {
    w: u32,
    h: u32,
    ow: u32,
    oh: u32,
    gbuf: wgpu::Texture,
    depth: wgpu::Texture,
    out: [wgpu::Buffer; 4],
    read: [wgpu::Buffer; 4],
}

struct Ctx {
    gpu: Arc<Gpu>,
    meshes: HashMap<MeshKey, CachedMesh>,
    mesh_bytes: u64,
    /// grid + skirt index buffers by mesh size (nx, ny): (buffer, index count)
    indices: HashMap<(usize, usize), (wgpu::Buffer, u32)>,
    frame: u64,
    gbuf_pipe: wgpu::RenderPipeline,
    shade_pipe: wgpu::ComputePipeline,
    pools: HashMap<Shading, TilePool>,
    targets: Option<Targets>,
    /// unit rays of the renderers' supersampled grids, by renderer id (most recent few)
    rays: Vec<(u64, wgpu::Buffer)>,
}

static CTX: OnceLock<Result<Mutex<Ctx>, String>> = OnceLock::new();

fn ctx() -> anyhow::Result<&'static Mutex<Ctx>> {
    match CTX.get_or_init(|| Ctx::new().map(Mutex::new).map_err(|e| format!("{e:#}"))) {
        Ok(c) => Ok(c),
        Err(e) => anyhow::bail!("render.backend gpu: no usable GPU ({e})"),
    }
}

impl Ctx {
    fn new() -> anyhow::Result<Ctx> {
        let gpu = device::shared()?;
        let d = &gpu.device;
        let gmod =
            d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mesh"), source: wgpu::ShaderSource::Wgsl(include_str!("mesh.wgsl").into()) });
        let gbuf_pipe = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gbuf"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &gmod,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &gmod,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba32Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let smod =
            d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("shade"), source: wgpu::ShaderSource::Wgsl(include_str!("shade.wgsl").into()) });
        let shade_pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shade"),
            layout: None,
            module: &smod,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Ctx {
            gpu,
            meshes: HashMap::new(),
            mesh_bytes: 0,
            indices: HashMap::new(),
            frame: 0,
            gbuf_pipe,
            shade_pipe,
            pools: HashMap::new(),
            targets: None,
            rays: Vec::new(),
        })
    }

    fn ensure_targets(&mut self, w: u32, h: u32, ow: u32, oh: u32) {
        if matches!(&self.targets, Some(t) if (t.w, t.h, t.ow, t.oh) == (w, h, ow, oh)) {
            return;
        }
        let d = &self.gpu.device;
        let tex = |label, format, usage| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let gbuf = tex("gbuf", wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let depth = tex("depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let size = (ow * oh * 16) as u64;
        let buf = |label, usage| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let su = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let ru = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        self.targets = Some(Targets {
            w,
            h,
            ow,
            oh,
            gbuf,
            depth,
            out: [buf("rad", su), buf("geo", su), buf("fc", su), buf("fs", su)],
            read: [buf("rad r", ru), buf("geo r", ru), buf("fc r", ru), buf("fs r", ru)],
        });
    }
}

fn v4(v: DVec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}

/// Index buffer of a grid mesh of nx × ny vertices followed by its skirt ring (the vertex layout
/// of `Renderer::unit_geometry`): two triangles per quad, two per skirt segment.
fn grid_indices(nx: usize, ny: usize) -> Vec<u32> {
    let mut idx = Vec::with_capacity(6 * (nx - 1) * (ny - 1) + 12 * (nx + ny));
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let (a, b, c, d) = (j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1);
            idx.extend([a, b, d, a, d, c].map(|k| k as u32));
        }
    }
    let mut ring = Vec::new();
    for i in 0..nx {
        ring.push(i);
    }
    for j in 1..ny {
        ring.push(j * nx + nx - 1);
    }
    for i in (0..nx - 1).rev() {
        ring.push((ny - 1) * nx + i);
    }
    for j in (1..ny - 1).rev() {
        ring.push(j * nx);
    }
    let n = ring.len();
    let sb = nx * ny;
    for s in 0..n {
        let (ka, kb) = (ring[s], ring[(s + 1) % n]);
        let (sa, sbk) = (sb + s, sb + (s + 1) % n);
        idx.extend([ka, kb, sbk, ka, sbk, sa].map(|k| k as u32));
    }
    idx
}

/// Bits of the 3x3 tile neighbourhood of `t` present in the view with elevation.
fn neighbour_mask(t: TileId, view: &TileView) -> u16 {
    let mut m = 0u16;
    let mut b = 0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if let Some(n) = t.neighbor(dx, dy) {
                if view.get(n).is_some_and(|d| !d.elevation.is_empty()) {
                    m |= 1 << b;
                }
            }
            b += 1;
        }
    }
    m
}

/// Render one frame on the GPU (same output as the CPU path).
pub fn render(r: &Renderer, cam: &CamPose, sun_state: &SunState) -> anyhow::Result<FrameOut> {
    let prof = std::env::var_os("RENDER_PROFILE").is_some();
    let t0 = std::time::Instant::now();
    let ss = r.settings.supersample.max(1);
    let ms = &r.model_ss;
    let (w, h) = (ms.width(), ms.height());
    let (ow, oh) = (r.model.width(), r.model.height());
    // ---------------- CPU: LOD units, tiles, meshes (shared with the CPU renderer)
    let units = r.select_units(cam);
    let need = r.gather_ids(&units);
    r.cache.prefetch(&need);
    let tiles: Vec<(TileId, Arc<TileData>)> = need.iter().filter_map(|id| r.cache.get(*id).map(|t| (*id, t))).collect();
    let view = TileView::new(tiles.iter().cloned().collect(), false);
    // the units' mesh keys (stride from the camera distance)
    let items: Vec<(usize, MeshKey, f64)> = units
        .iter()
        .enumerate()
        .filter_map(|(ui, u)| {
            let data = view.get(u.data)?;
            if data.elevation.is_empty() {
                return None;
            }
            let (stride, texel) = r.unit_stride(cam, u, data);
            Some((ui, MeshKey { id: u.id, data: u.data, rect: u.rect, stride, neighbours: neighbour_mask(u.data, &view) }, texel))
        })
        .collect();

    let mut guard = ctx()?.lock();
    let c = &mut *guard;
    c.gpu.check()?;
    c.frame += 1;
    let frame = c.frame;
    c.ensure_targets(w, h, ow, oh);
    // ---------------- meshes: build (CPU, parallel) and upload the ones not cached on the GPU
    let missing: Vec<&(usize, MeshKey, f64)> = items.iter().filter(|(_, k, _)| !c.meshes.contains_key(k)).collect();
    let built: Vec<(MeshKey, Vec<Vertex>, DVec3, usize, usize)> = missing
        .par_iter()
        .map(|(ui, key, texel)| {
            let u: &Unit = &units[*ui];
            let data = view.get(u.data).unwrap();
            let g = r.unit_geometry(u, data, key.stride, *texel, &view);
            let o = g.ecef[0];
            let rel = |p: DVec3| [(p.x - o.x) as f32, (p.y - o.y) as f32, (p.z - o.z) as f32];
            let mut v: Vec<Vertex> = g.ecef.iter().zip(&g.uv).map(|(p, uv)| Vertex { pos: rel(*p), uv: *uv }).collect();
            v.extend(g.ring.iter().zip(&g.skirt).map(|(&k, p)| Vertex { pos: rel(*p), uv: g.uv[k] }));
            (*key, v, o, g.nx, g.ny)
        })
        .collect();
    let mut mesh_uploads = 0;
    for (key, v, origin, nx, ny) in built {
        let bytes = (v.len() * std::mem::size_of::<Vertex>()) as u64;
        let vb = c.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh"),
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::VERTEX,
        });
        c.mesh_bytes += bytes;
        c.meshes.insert(key, CachedMesh { vb, origin, nx, ny, bytes, last: frame });
        mesh_uploads += 1;
        if !c.indices.contains_key(&(nx, ny)) {
            let ix = grid_indices(nx, ny);
            let ib = c.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("grid idx"),
                contents: bytemuck::cast_slice(&ix),
                usage: wgpu::BufferUsages::INDEX,
            });
            c.indices.insert((nx, ny), (ib, ix.len() as u32));
        }
    }
    for (_, k, _) in &items {
        c.meshes.get_mut(k).unwrap().last = frame;
    }
    // LRU eviction beyond the budget (never this frame's meshes)
    if c.mesh_bytes > MESH_BUDGET {
        let mut old: Vec<(u64, MeshKey)> = c.meshes.iter().filter(|(_, m)| m.last != frame).map(|(k, m)| (m.last, *k)).collect();
        old.sort_unstable_by_key(|x| x.0);
        for (_, k) in old {
            if c.mesh_bytes <= MESH_BUDGET * 3 / 4 {
                break;
            }
            let m = c.meshes.remove(&k).unwrap();
            c.mesh_bytes -= m.bytes;
        }
    }
    let t_cpu = t0.elapsed().as_secs_f64();
    // ---------------- tiles: resident slots and the frame's lookup table
    let shading = r.settings.shading;
    if !c.pools.contains_key(&shading) {
        let slots = POOL_SLOTS.min(c.gpu.device.limits().max_texture_array_layers);
        let p = TilePool::new(&c.gpu.device, slots, shading);
        c.pools.insert(shading, p);
    }
    let pool = c.pools.get_mut(&shading).unwrap();
    let up0 = pool.uploads;
    let mut refs: Vec<(TileId, &TileData)> = tiles.iter().map(|(id, t)| (*id, t.as_ref())).collect();
    if refs.len() > pool.slots as usize {
        // more tiles than slots: keep the units' own data (finest first), then neighbours /
        // ancestors by zoom; the rest fall back to coarser data in the shader
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "render[gpu]: a frame needs {} tiles, the GPU pool holds {}; the excess is drawn from coarser data (raise gpu::POOL_SLOTS)",
                refs.len(),
                pool.slots
            );
        }
        let own: std::collections::HashSet<TileId> = units.iter().map(|u| u.data).collect();
        refs.sort_by_key(|(id, _)| (!own.contains(id), std::cmp::Reverse(id.z), *id));
    }
    let resident = pool.ensure(&c.gpu.queue, &refs);
    let uploaded = pool.uploads - up0;
    let (table, mask) = tiles::lookup_table(&resident);
    if !c.rays.iter().any(|(id, _)| *id == r.id) {
        let rv: Vec<[f32; 4]> = r.rays.iter().map(|q| [q[0], q[1], q[2], 0.0]).collect();
        let b = c.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rays"),
            contents: bytemuck::cast_slice(&rv),
            usage: wgpu::BufferUsages::STORAGE,
        });
        if c.rays.len() >= 8 {
            c.rays.remove(0);
        }
        c.rays.push((r.id, b));
    }
    let rays_buf = &c.rays.iter().find(|(id, _)| *id == r.id).unwrap().1;
    // ---------------- uniforms
    let cam_geo = geodesy::ecef2geodetic(cam.pos, &r.ell);
    let enu2ecef = geodesy::rot_ecef2enu(cam_geo.lat, cam_geo.lon).transpose();
    let dir_of = |az: f64, el: f64| enu2ecef * DVec3::new(az.sin() * el.cos(), az.cos() * el.cos(), el.sin());
    let sun = dir_of(sun_state.azimuth, sun_state.elevation);
    let moon = dir_of(sun_state.moon_azimuth, sun_state.moon_elevation);
    let atmo = crate::atmo::Atmosphere::new(r.settings.atmosphere.clone(), sun, moon, sun_state);
    let cam_up = geodesy::up_vector(cam_geo.lat, cam_geo.lon);
    let relit = shading == Shading::Relit;
    let lights = sun_state.lights > 1e-3;
    let split = r.split_flicker && !r.geometry_only && lights && sun_state.flicker.enabled;
    let mut flags = 0;
    for (on, f) in [
        (r.geometry_only, F_GEOM),
        (relit, F_RELIT),
        (r.settings.water_glint, F_GLINT),
        (relit && r.settings.lighting.shadows && sun_state.direct > 1e-4, F_SHADOWS),
        (lights, F_LIGHTS),
        (lights && sun_state.light_pollution > 0.0, F_POLLUTION),
        (lights && sun_state.flicker.enabled, F_FLICKER),
        (split, F_SPLIT),
        (atmo.p.enabled, F_ATMO),
    ] {
        if on {
            flags |= f;
        }
    }
    let fl = &sun_state.flicker;
    let om = fl.omega();
    let x = 0.5 * om * sun_state.exposure;
    let sinc = if x.abs() < 1e-9 { 1.0 } else { x.sin() / x };
    let rm = cam.r_ecef_cam;
    let un = Uniforms {
        dims: [w, h, ow, oh],
        cfg: [ss, r.settings.max_aniso, flags, mask],
        cam_pos: v4(cam.pos, cam_geo.h),
        r0: [rm.x_axis.x as f32, rm.y_axis.x as f32, rm.z_axis.x as f32, 0.0],
        r1: [rm.x_axis.y as f32, rm.y_axis.y as f32, rm.z_axis.y as f32, 0.0],
        r2: [rm.x_axis.z as f32, rm.y_axis.z as f32, rm.z_axis.z as f32, 0.0],
        cam_up: v4(cam_up, 1.0 / ms.focal_px()),
        sun_dir: v4(atmo.sun_dir, sun_state.elevation.max(0.005).tan()),
        moon_dir: v4(atmo.moon_dir, atmo.moon_disc),
        sun_col: v4(atmo.sun_col, 0.0),
        moon_col: v4(atmo.moon_col, atmo.beta_m),
        ray_col: v4(atmo.rayleigh_col, atmo.p.rayleigh_scale_height),
        mie_col: v4(atmo.mie_col, atmo.p.mie_scale_height),
        zenith: v4(atmo.zenith, atmo.p.inscatter),
        horizon: v4(atmo.horizon, view.max_elev),
        sun: [sun_state.direct as f32, sun_state.sky as f32, sun_state.lights as f32, sun_state.light_pollution as f32],
        sun2: [sun_state.azimuth as f32, sun_state.elevation as f32, 0.0, 0.0],
        flick: [(om * sun_state.time).cos() as f32, (om * sun_state.time).sin() as f32, sinc as f32, fl.led_fraction as f32],
        flick2: [fl.led_depth as f32, fl.depth as f32, 0.0, 0.0],
        ell: [r.ell.a as f32, r.ell.e2() as f32, 0.0, 0.0],
    };
    let mut unit_data: Vec<[u32; 4]> = units.iter().map(|u| [u.data.z as u32, u.data.x, u.data.y, 0]).collect();
    if unit_data.is_empty() {
        unit_data.push([0; 4]);
    }

    let d = &c.gpu.device;
    let queue = &c.gpu.queue;
    let tg = c.targets.as_ref().unwrap();
    let mk = |label: &str, contents: &[u8], usage| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
    let rt = cam.r_ecef_cam.transpose();
    let gc = ms.gpu();
    let half_lim = (ms.max_half_angle() * 1.35 + 0.05).min(std::f64::consts::PI - 0.03);
    let mu = MeshUniforms {
        rt0: [rt.x_axis.x as f32, rt.y_axis.x as f32, rt.z_axis.x as f32, 0.0],
        rt1: [rt.x_axis.y as f32, rt.y_axis.y as f32, rt.z_axis.y as f32, 0.0],
        rt2: [rt.x_axis.z as f32, rt.y_axis.z as f32, rt.z_axis.z as f32, 0.0],
        p: [
            [gc.p[0], gc.p[1], gc.p[2], gc.p[3]],
            [gc.p[4], gc.p[5], gc.p[6], gc.p[7]],
            [gc.p[8], gc.p[9], gc.p[10], gc.p[11]],
            [gc.p[12], gc.p[13], gc.p[14], gc.p[15]],
        ],
        lim: [gc.angle_limit.cos() as f32, half_lim.cos() as f32, w as f32, h as f32],
        kind: [gc.kind, 0, 0, 0],
    };
    let gu = mk("mesh u", bytemuck::bytes_of(&mu), wgpu::BufferUsages::UNIFORM);
    let mut draw_data: Vec<[f32; 4]> = items
        .iter()
        .map(|(ui, k, _)| {
            let o = c.meshes[k].origin - cam.pos;
            [o.x as f32, o.y as f32, o.z as f32, f32::from_bits(*ui as u32)]
        })
        .collect();
    if draw_data.is_empty() {
        draw_data.push([0.0; 4]);
    }
    let draws_b = mk("draws", bytemuck::cast_slice(&draw_data), wgpu::BufferUsages::STORAGE);
    let ub = mk("shade u", bytemuck::bytes_of(&un), wgpu::BufferUsages::UNIFORM);
    let units_b = mk("units", bytemuck::cast_slice(&unit_data), wgpu::BufferUsages::STORAGE);
    let table_b = mk("table", bytemuck::cast_slice(&table), wgpu::BufferUsages::STORAGE);
    let mut enc = d.create_command_encoder(&Default::default());
    // ---------------- G-buffer
    {
        let gview = tg.gbuf.create_view(&Default::default());
        let dview = tg.depth.create_view(&Default::default());
        let clear = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuf"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &gview,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &dview,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if !items.is_empty() {
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &c.gbuf_pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gu.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: draws_b.as_entire_binding() },
                ],
            });
            rp.set_pipeline(&c.gbuf_pipe);
            rp.set_bind_group(0, &bg, &[]);
            for (i, (_, k, _)) in items.iter().enumerate() {
                let m = &c.meshes[k];
                let (ib, n) = &c.indices[&(m.nx, m.ny)];
                rp.set_vertex_buffer(0, m.vb.slice(..));
                rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..*n, 0, i as u32..i as u32 + 1);
            }
        }
    }
    // ---------------- shading
    {
        let pool = &c.pools[&shading];
        let tv = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let (vc, vn, ve, vz, vl, vb) = (tv(&pool.color), tv(&pool.normal), tv(&pool.emission), tv(&pool.elev), tv(&pool.lc), tv(&pool.blockmax));
        let gview = tg.gbuf.create_view(&Default::default());
        let tvr = |v| wgpu::BindingResource::TextureView(v);
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.shade_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tvr(&gview) },
                wgpu::BindGroupEntry { binding: 2, resource: rays_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: units_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: table_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: tvr(&vc) },
                wgpu::BindGroupEntry { binding: 6, resource: tvr(&vn) },
                wgpu::BindGroupEntry { binding: 7, resource: tvr(&ve) },
                wgpu::BindGroupEntry { binding: 8, resource: tvr(&vz) },
                wgpu::BindGroupEntry { binding: 9, resource: tvr(&vl) },
                wgpu::BindGroupEntry { binding: 10, resource: tvr(&vb) },
                wgpu::BindGroupEntry { binding: 11, resource: tg.out[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: tg.out[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: tg.out[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: tg.out[3].as_entire_binding() },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&c.shade_pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(ow.div_ceil(8), oh.div_ceil(8), 1);
    }
    // geometry (depth, points, land cover) unless only the radiance is wanted (the stars mask
    // the sky with it)
    let need_geo = !r.radiance_only || r.geometry_only || (sun_state.stars && r.stars_in_render);
    let bufs: Vec<usize> = (0..if split { 4 } else { 2 }).filter(|&k| k != 1 || need_geo).collect();
    for &k in &bufs {
        enc.copy_buffer_to_buffer(&tg.out[k], 0, &tg.read[k], 0, (ow * oh * 16) as u64);
    }
    queue.submit([enc.finish()]);
    let slices: Vec<wgpu::BufferSlice<'_>> = bufs.iter().map(|&k| tg.read[k].slice(..)).collect();
    if let Err(e) = c.gpu.map_read(&slices) {
        for &k in &bufs {
            tg.read[k].unmap(); // (a buffer mapped before the failure must not stay mapped)
        }
        return Err(e.context("render.backend gpu"));
    }
    let t_gpu = t0.elapsed().as_secs_f64();
    // ---------------- read back into a FrameOut
    let n = (ow * oh) as usize;
    let get = |k: usize| -> anyhow::Result<Vec<[f32; 4]>> {
        let v = bytemuck::cast_slice::<u8, [f32; 4]>(&tg.read[k].slice(..).get_mapped_range()?).to_vec();
        tg.read[k].unmap();
        Ok(v)
    };
    let rad = get(0)?;
    let geo = if need_geo { get(1)? } else { Vec::new() };
    let cs = (ss / 2) as usize;
    let mut out = FrameOut {
        width: ow,
        height: oh,
        radiance: Vec::with_capacity(n * 3),
        depth: Vec::with_capacity(n),
        points: Vec::with_capacity(n),
        landcover: Vec::with_capacity(n),
        sample_offset: (cs as f64 + 0.5) / ss as f64 - 0.5,
        stars: Vec::new(),
        flicker_cos: Vec::new(),
        flicker_sin: Vec::new(),
        units,
    };
    let wss = w as usize;
    for k in 0..n {
        out.radiance.extend_from_slice(&rad[k][..3]);
        if !need_geo {
            continue;
        }
        let g = geo[k];
        let range = g[1];
        if range > 0.0 {
            out.depth.push(g[0]);
            let (oxk, oyk) = (k % ow as usize, k / ow as usize);
            let ray = r.rays[(oyk * ss as usize + cs) * wss + oxk * ss as usize + cs];
            let ray = DVec3::new(ray[0] as f64, ray[1] as f64, ray[2] as f64);
            out.points.push(Some(cam.pos + cam.r_ecef_cam * ray * range as f64));
            out.landcover.push(g[2] as u8);
        } else {
            out.depth.push(f32::INFINITY);
            out.points.push(None);
            out.landcover.push(255);
        }
    }
    if split {
        let (fc, fs) = (get(2)?, get(3)?);
        out.flicker_cos = fc.iter().flat_map(|v| [v[0], v[1], v[2]]).collect();
        out.flicker_sin = fs.iter().flat_map(|v| [v[0], v[1], v[2]]).collect();
    }
    if prof {
        eprintln!(
            "render[gpu]: {} units, {} tiles ({} uploaded), {} meshes ({} built, {} MB cached) | cpu {:.3}s gpu {:.3}s readback {:.3}s",
            out.units.len(),
            tiles.len(),
            uploaded,
            items.len(),
            mesh_uploads,
            c.mesh_bytes >> 20,
            t_cpu,
            t_gpu - t_cpu,
            t0.elapsed().as_secs_f64() - t_gpu
        );
    }
    Ok(out)
}
