//! Globe renderer: XYZ tiles drawn as displaced grid patches on the ellipsoid.
//!
//! * Level of detail: a quadtree walk from z0 refines a tile while its texels look larger than
//!   `lod_bias` pixels and a finer tile can be had (resident, in the store, or generated). A tile
//!   whose data is not here yet is drawn with the region of its nearest resident ancestor, so
//!   there are no holes while tiles stream in.
//! * Precision: every patch is a mesh relative to its own centre (f64 on the CPU), drawn with
//!   the centre minus the eye, so positions are camera-relative.
//! * Residency: colour (albedo + class), elevation and slope of each tile live in a layer of three
//!   texture arrays; least recently used tiles are evicted.

use crate::tiles::{Payload, Service, N};
use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu, wgpu};
use geodesy::tiles::{uv_to_latlon, TileId, MAX_MERCATOR_LAT_RAD};
use geodesy::{ecef2geodetic, geodetic2ecef, Ellipsoid, Geodetic};
use glam::{DMat4, DVec2, DVec3, DVec4};
use std::collections::HashMap;

/// Grid vertices per tile side (+ a skirt ring).
const GRID: usize = 33;
const VSIDE: usize = GRID + 2;
const SAMPLES: u32 = 4;
const DRAW_STRIDE: u64 = 256;
const MAX_DRAWS: usize = 6000;
const CAP_LAYER: u32 = u32::MAX;
/// CPU copy of each resident tile's elevation (for the camera), per side
const ELEV_LO: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
    skirt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam: [f32; 4],
    sun: [f32; 4],
    params: [f32; 4],
    ell: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawU {
    rel_center: [f32; 3],
    layer: u32,
    abs_center: [f32; 3],
    skirt: f32,
    uv_off: [f32; 2],
    uv_scale: f32,
    zoom: f32,
    cap_color: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Albedo,
    Elevation,
    Landcover,
    Relief,
}

/// What the user controls.
#[derive(Clone, Debug)]
pub struct Settings {
    pub exaggeration: f32,
    pub mode: Mode,
    pub borders: bool,
    /// refine while a texel is larger than this many pixels
    pub lod_bias: f64,
    /// generate missing tiles of the view (else only stored ones are shown)
    pub dynamic: bool,
    pub gen_max_zoom: u8,
    pub view_max_zoom: u8,
    pub base_zoom: u8,
    /// sun from the upper left of the view (hillshade), else from a fixed subsolar point
    pub sun_follows_view: bool,
    pub sun_lat: f64,
    pub sun_lon: f64,
}

/// Orbit camera around a target point on the surface.
#[derive(Clone, Debug)]
pub struct Camera {
    pub lat: f64,
    pub lon: f64,
    /// distance from the target (m)
    pub dist: f64,
    /// view azimuth, clockwise from north (rad)
    pub heading: f64,
    /// 0 = looking straight down (rad)
    pub tilt: f64,
    pub fov_y: f64,
    /// height of the target (terrain, exaggerated), m above the ellipsoid
    pub target_h: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct CamFrame {
    pub eye: DVec3,
    pub view_proj: DMat4,
    /// view direction and the camera's up (ECEF)
    pub dir: DVec3,
    pub cam_up: DVec3,
    /// vertical field of view (rad)
    pub fov_y: f64,
}

impl Camera {
    fn enu(&self) -> (DVec3, DVec3, DVec3) {
        let (sl, cl) = self.lat.sin_cos();
        let (so, co) = self.lon.sin_cos();
        (DVec3::new(-so, co, 0.0), DVec3::new(-sl * co, -sl * so, cl), DVec3::new(cl * co, cl * so, sl))
    }

    pub fn frame(&self, ell: &Ellipsoid, aspect: f64) -> CamFrame {
        let (e, n, u) = self.enu();
        let target = geodetic2ecef(Geodetic { lat: self.lat, lon: self.lon, h: self.target_h }, ell);
        let (sh, ch) = self.heading.sin_cos();
        let (st, ct) = self.tilt.sin_cos();
        let fwd_h = e * sh + n * ch;
        let dir = fwd_h * st - u * ct;
        let cam_up = fwd_h * ct + u * st;
        let eye = target - dir * self.dist;
        let alt = (ecef2geodetic(eye, ell).h - self.target_h.min(0.0)).max(1.0);
        let near = (alt * 0.05).clamp(0.05, 50_000.0).min(self.dist * 0.2);
        let view = DMat4::look_to_rh(DVec3::ZERO, dir, cam_up);
        let proj = DMat4::perspective_infinite_reverse_rh(self.fov_y, aspect, near);
        CamFrame { eye, view_proj: proj * view, dir, cam_up, fov_y: self.fov_y }
    }

    /// Move the target by a screen drag (pixels) at `m_per_px`.
    pub fn pan(&mut self, dx: f64, dy: f64, m_per_px: f64, ell: &Ellipsoid) {
        let (sh, ch) = self.heading.sin_cos();
        let (right, fwd) = (-dx * m_per_px, dy * m_per_px);
        let de = right * ch + fwd * sh;
        let dn = -right * sh + fwd * ch;
        self.lat = (self.lat + dn / ell.a).clamp(-1.55, 1.55);
        self.lon += de / (ell.a * self.lat.cos().max(0.02));
        self.lon = (self.lon + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    }
}

struct Slot {
    layer: u32,
    last: u64,
    elev_max: f32,
    elev_lo: Vec<f32>,
}

struct Geo {
    center: DVec3,
    radius: f64,
    width_m: f64,
}

struct Mesh {
    buf: wgpu::Buffer,
    center: DVec3,
    last: u64,
}

struct Target {
    w: u32,
    h: u32,
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolve: wgpu::TextureView,
    image: wgpu::Texture,
}

#[derive(Default, Clone)]
pub struct FrameStats {
    pub drawn: usize,
    pub resident: usize,
    pub capacity: usize,
    pub uploads: usize,
    pub pending_uploads: usize,
    pub max_zoom_drawn: u8,
    pub want_load: usize,
    pub want_gen: usize,
}

struct DrawItem {
    tile: TileId,
    src: TileId,
    layer: u32,
}

pub struct Globe {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ell: Ellipsoid,
    pipe_terrain: wgpu::RenderPipeline,
    pipe_sky: wgpu::RenderPipeline,
    bg0: wgpu::BindGroup,
    bg1: wgpu::BindGroup,
    globals: wgpu::Buffer,
    draws: wgpu::Buffer,
    color: wgpu::Texture,
    elev: wgpu::Texture,
    grad: wgpu::Texture,
    capacity: u32,
    index: wgpu::Buffer,
    index_count: u32,
    caps: [(wgpu::Buffer, wgpu::Buffer, u32, DVec3); 2],
    cap_colors: [[f32; 4]; 2],
    cap_heights: [f32; 2],
    target: Option<Target>,
    pub texture_id: Option<egui::TextureId>,
    resident: HashMap<TileId, Slot>,
    free: Vec<u32>,
    meshes: HashMap<TileId, Mesh>,
    geo: HashMap<TileId, Geo>,
    requested: HashMap<TileId, u64>,
    pending: Vec<Payload>,
    frame: u64,
    pub stats: FrameStats,
}

fn tex_array(device: &wgpu::Device, label: &str, layers: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: N as u32, height: N as u32, depth_or_array_layers: layers },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn array_view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
}

impl Globe {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, ell: Ellipsoid, want_capacity: u32) -> Globe {
        let capacity = want_capacity.min(device.limits().max_texture_array_layers).max(64);
        let color = tex_array(device, "tile color", capacity, wgpu::TextureFormat::Rgba8UnormSrgb);
        let elev = tex_array(device, "tile elevation", capacity, wgpu::TextureFormat::R32Float);
        let grad = tex_array(device, "tile slope", capacity, wgpu::TextureFormat::Rg16Float);
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draws = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draws"),
            size: DRAW_STRIDE * MAX_DRAWS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tile sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let vis = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let tex_entry = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globe bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vis,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex_entry(1, true),
                tex_entry(2, false),
                tex_entry(3, true),
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: vis, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globe bgl1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vis,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<DrawU>() as u64),
                },
                count: None,
            }],
        });
        let (cv, ev, gv) = (array_view(&color), array_view(&elev), array_view(&grad));
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe bg0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&cv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&ev) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&gv) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe bg1"),
            layout: &bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &draws,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<DrawU>() as u64),
                }),
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("globe"), source: wgpu::ShaderSource::Wgsl(include_str!("globe.wgsl").into()) });
        let layout_t = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("terrain"), bind_group_layouts: &[Some(&bgl0), Some(&bgl1)], immediate_size: 0 });
        let layout_s = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("sky"), bind_group_layouts: &[Some(&bgl0)], immediate_size: 0 });
        let target = |blend: Option<wgpu::BlendState>| [Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend, write_mask: wgpu::ColorWrites::ALL })];
        let ms = wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: false };
        let pipe_terrain = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain"),
            layout: Some(&layout_t),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32],
                })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: ms,
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &target(None) }),
            multiview_mask: None,
            cache: None,
        });
        let pipe_sky = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky"),
            layout: Some(&layout_s),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_sky"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: ms,
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs_sky"), compilation_options: Default::default(), targets: &target(None) }),
            multiview_mask: None,
            cache: None,
        });
        // shared grid indices
        let mut idx: Vec<u32> = Vec::new();
        for j in 0..VSIDE - 1 {
            for i in 0..VSIDE - 1 {
                let a = (j * VSIDE + i) as u32;
                let (b, c, d) = (a + 1, a + VSIDE as u32, a + VSIDE as u32 + 1);
                idx.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        let index = buffer(device, "grid index", bytemuck::cast_slice(&idx), wgpu::BufferUsages::INDEX);
        let caps = [cap_mesh(device, &ell, 1.0), cap_mesh(device, &ell, -1.0)];
        Globe {
            device: device.clone(),
            queue: queue.clone(),
            ell,
            pipe_terrain,
            pipe_sky,
            bg0,
            bg1,
            globals,
            draws,
            color,
            elev,
            grad,
            capacity,
            index,
            index_count: idx.len() as u32,
            caps,
            cap_colors: [[0.02, 0.05, 0.12, 1.0], [0.8, 0.82, 0.85, 0.0]],
            cap_heights: [0.0, 0.0],
            target: None,
            texture_id: None,
            resident: HashMap::new(),
            free: (0..capacity).rev().collect(),
            meshes: HashMap::new(),
            geo: HashMap::new(),
            requested: HashMap::new(),
            pending: Vec::new(),
            frame: 0,
            stats: FrameStats::default(),
        }
    }

    pub fn ellipsoid(&self) -> Ellipsoid {
        self.ell
    }

    /// Terrain height (m, not exaggerated) under (lat, lon) from the finest resident tile.
    pub fn height_at(&self, lat: f64, lon: f64, max_z: u8) -> Option<f64> {
        for z in (0..=max_z.min(22)).rev() {
            let t = geodesy::tiles::tile_for_latlon(lat, lon, z);
            if let Some(s) = self.resident.get(&t) {
                let px = geodesy::tiles::latlon_to_pixel(lat, lon, z, N as u32) - DVec2::new(t.x as f64, t.y as f64) * N as f64;
                let f = (px / N as f64 * ELEV_LO as f64 - 0.5).clamp(DVec2::ZERO, DVec2::splat(ELEV_LO as f64 - 1.0));
                let (i0, j0) = (f.x.floor() as usize, f.y.floor() as usize);
                let (i1, j1) = ((i0 + 1).min(ELEV_LO - 1), (j0 + 1).min(ELEV_LO - 1));
                let (fx, fy) = (f.x - i0 as f64, f.y - j0 as f64);
                let e = |i: usize, j: usize| s.elev_lo[j * ELEV_LO + i] as f64;
                let a = e(i0, j0) + (e(i1, j0) - e(i0, j0)) * fx;
                let b = e(i0, j1) + (e(i1, j1) - e(i0, j1)) * fx;
                return Some(a + (b - a) * fy);
            }
        }
        None
    }

    fn geo(&mut self, t: TileId) -> &Geo {
        let ell = self.ell;
        self.geo.entry(t).or_insert_with(|| {
            let n = (1u64 << t.z) as f64;
            let at = |u: f64, v: f64| {
                let (lat, lon) = uv_to_latlon(DVec2::new((t.x as f64 + u) / n, (t.y as f64 + v) / n));
                geodetic2ecef(Geodetic { lat, lon, h: 0.0 }, &ell)
            };
            let center = at(0.5, 0.5);
            let mut radius: f64 = 0.0;
            for v in [0.0, 0.5, 1.0] {
                for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    radius = radius.max((at(u, v) - center).length());
                }
            }
            let (lat_c, _) = t.center();
            Geo { center, radius, width_m: std::f64::consts::TAU * ell.a * lat_c.cos() / n }
        })
    }

    fn mesh(&mut self, t: TileId) -> (DVec3, &wgpu::Buffer) {
        let frame = self.frame;
        if !self.meshes.contains_key(&t) {
            let ell = self.ell;
            let n = (1u64 << t.z) as f64;
            let (lat_c, lon_c) = uv_to_latlon(DVec2::new((t.x as f64 + 0.5) / n, (t.y as f64 + 0.5) / n));
            let center = geodetic2ecef(Geodetic { lat: lat_c, lon: lon_c, h: 0.0 }, &ell);
            let mut v = Vec::with_capacity(VSIDE * VSIDE);
            for j in 0..VSIDE {
                for i in 0..VSIDE {
                    let (gi, gj) = (i.clamp(1, GRID) - 1, j.clamp(1, GRID) - 1);
                    let skirt = (i == 0 || j == 0 || i == VSIDE - 1 || j == VSIDE - 1) as u8 as f32;
                    let (u, w) = (gi as f64 / (GRID - 1) as f64, gj as f64 / (GRID - 1) as f64);
                    let (lat, lon) = uv_to_latlon(DVec2::new((t.x as f64 + u) / n, (t.y as f64 + w) / n));
                    let p = geodetic2ecef(Geodetic { lat, lon, h: 0.0 }, &ell) - center;
                    v.push(Vertex { pos: p.as_vec3().to_array(), uv: [u as f32, w as f32], skirt });
                }
            }
            let buf = buffer(&self.device, "tile mesh", bytemuck::cast_slice(&v), wgpu::BufferUsages::VERTEX);
            self.meshes.insert(t, Mesh { buf, center, last: frame });
        }
        let m = self.meshes.get_mut(&t).unwrap();
        m.last = frame;
        (m.center, &m.buf)
    }

    fn upload(&mut self, p: Payload) {
        if let Some(s) = self.resident.get(&p.id) {
            // regenerated: overwrite in place
            let layer = s.layer;
            self.write_layer(layer, &p);
            return;
        }
        let layer = match self.free.pop() {
            Some(l) => l,
            None => {
                // the least recently used tile not drawn this frame (keep the coarsest levels)
                let victim = self
                    .resident
                    .iter()
                    .filter(|(t, s)| s.last < self.frame && t.z > 2)
                    .min_by_key(|(_, s)| s.last)
                    .map(|(t, _)| *t);
                let Some(v) = victim else { return };
                self.resident.remove(&v).unwrap().layer
            }
        };
        self.write_layer(layer, &p);
        let mut lo = vec![0f32; ELEV_LO * ELEV_LO];
        let k = N / ELEV_LO;
        for j in 0..ELEV_LO {
            for i in 0..ELEV_LO {
                lo[j * ELEV_LO + i] = p.elev[(j * k + k / 2) * N + i * k + k / 2];
            }
        }
        if p.id.z == 0 {
            // polar caps: the mean colour / height of the map's edge rows
            for (c, row) in [(0usize, 0usize), (1, N - 1)] {
                let mut acc = [0.0f64; 3];
                let mut water = 0usize;
                let mut h = 0.0f64;
                for i in 0..N {
                    let k = row * N + i;
                    for (ch, a) in acc.iter_mut().enumerate() {
                        *a += srgb_to_linear(p.color[4 * k + ch] as f64 / 255.0);
                    }
                    water += matches!(p.color[4 * k + 3], 1..=3) as usize;
                    h += p.elev[k].max(0.0) as f64;
                }
                let m = N as f64;
                self.cap_colors[c] = [(acc[0] / m) as f32, (acc[1] / m) as f32, (acc[2] / m) as f32, (water * 2 > N) as u8 as f32];
                self.cap_heights[c] = (h / m) as f32;
            }
        }
        self.resident.insert(p.id, Slot { layer, last: self.frame, elev_max: p.elev_max.max(p.elev_min), elev_lo: lo });
        self.stats.uploads += 1;
    }

    fn write_layer(&self, layer: u32, p: &Payload) {
        let w = |t: &wgpu::Texture, data: &[u8], bpp: u32| {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: t, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: layer }, aspect: wgpu::TextureAspect::All },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(N as u32 * bpp), rows_per_image: Some(N as u32) },
                wgpu::Extent3d { width: N as u32, height: N as u32, depth_or_array_layers: 1 },
            );
        };
        w(&self.color, &p.color, 4);
        w(&self.elev, bytemuck::cast_slice(&p.elev), 4);
        w(&self.grad, bytemuck::cast_slice(&p.grad), 4);
    }

    fn ensure_target(&mut self, w: u32, h: u32, renderer: Option<&mut egui_wgpu::Renderer>) {
        if matches!(&self.target, Some(t) if t.w == w && t.h == h) {
            return;
        }
        let mk = |label: &str, format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let msaa = mk("globe msaa", wgpu::TextureFormat::Rgba8Unorm, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = mk("globe depth", wgpu::TextureFormat::Depth32Float, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let image = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("globe image"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let resolve = image.create_view(&Default::default());
        if let Some(renderer) = renderer {
            match self.texture_id {
                Some(id) => renderer.update_egui_texture_from_wgpu_texture(&self.device, &resolve, wgpu::FilterMode::Linear, id),
                None => self.texture_id = Some(renderer.register_native_texture(&self.device, &resolve, wgpu::FilterMode::Linear)),
            }
        }
        self.target = Some(Target { w, h, msaa, depth, resolve, image });
    }

    /// Draw a frame of `w` x `h` pixels into the globe image and ask the service for tiles.
    pub fn render(&mut self, cf: &CamFrame, s: &Settings, svc: &Service, w: u32, h: u32, renderer: Option<&mut egui_wgpu::Renderer>) {
        self.frame += 1;
        self.stats.uploads = 0;
        let (w, h) = (w.max(16), h.max(16));
        self.ensure_target(w, h, renderer);

        // ---- uploads: what was asked for recently (or the coarsest levels), a budget per frame
        let frame = self.frame;
        let mut incoming = std::mem::take(&mut self.pending);
        incoming.extend(svc.take_results());
        incoming.retain(|p| p.id.z <= 2 || self.requested.get(&p.id).is_some_and(|&f| frame - f < 600));
        incoming.sort_by_key(|p| (p.id.z, std::cmp::Reverse(self.requested.get(&p.id).copied().unwrap_or(0))));
        let budget = 48;
        for (k, p) in incoming.into_iter().enumerate() {
            if k < budget {
                self.upload(p);
            } else {
                self.pending.push(p);
            }
        }

        // ---- level of detail
        let eye = cf.eye;
        let planes = frustum_planes(&cf.view_proj);
        let pix = cf.fov_y / h as f64;
        let exag = s.exaggeration as f64;
        let mut walk = Walk { eye, planes, pix, exag, s, svc, draws: Vec::new(), want_load: Vec::new(), want_gen: Vec::new() };
        self.visit(TileId::new(0, 0, 0), None, &mut walk);
        let Walk { draws, mut want_load, mut want_gen, .. } = walk;
        want_load.sort_by(|a, b| a.1.total_cmp(&b.1));
        want_gen.sort_by(|a, b| a.1.total_cmp(&b.1));
        self.stats.want_load = want_load.len();
        self.stats.want_gen = want_gen.len();
        svc.want(want_load.into_iter().take(64).map(|x| x.0).collect(), want_gen.into_iter().take(32).map(|x| x.0).collect());

        // ---- uniforms
        let sun = if s.sun_follows_view {
            // from behind the camera, upper left (reads as relief; lights the whole visible disc)
            (-cf.dir * 0.75 + cf.cam_up * 0.55 - cf.dir.cross(cf.cam_up) * 0.55).normalize()
        } else {
            let (lat, lon) = (s.sun_lat.to_radians(), s.sun_lon.to_radians());
            DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
        };
        let vp = cf.view_proj.as_mat4();
        let g = Globals {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            cam: [eye.x as f32, eye.y as f32, eye.z as f32, 0.0],
            sun: [sun.x as f32, sun.y as f32, sun.z as f32, 0.05],
            params: [
                s.exaggeration,
                match s.mode {
                    Mode::Albedo => 0.0,
                    Mode::Elevation => 1.0,
                    Mode::Landcover => 2.0,
                    Mode::Relief => 3.0,
                },
                s.borders as u8 as f32,
                350_000.0,
            ],
            ell: [(1.0 / (self.ell.a * self.ell.a)) as f32, (1.0 / (self.ell.a * self.ell.a)) as f32, (1.0 / (self.ell.b * self.ell.b)) as f32, 6_371_000.0],
        };
        self.queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&g));

        let mut recs: Vec<(DrawU, Option<TileId>, usize)> = Vec::new();
        let mut max_z = 0u8;
        for d in draws.iter().take(MAX_DRAWS - 2) {
            let (center, _) = self.mesh(d.tile);
            let geo_w = self.geo(d.tile).width_m;
            let dz = d.tile.z - d.src.z;
            let scale = 1.0 / (1u64 << dz) as f64;
            let off = DVec2::new((d.tile.x - (d.src.x << dz)) as f64 * scale, (d.tile.y - (d.src.y << dz)) as f64 * scale);
            let rel = center - eye;
            recs.push((
                DrawU {
                    rel_center: rel.as_vec3().to_array(),
                    layer: d.layer,
                    abs_center: center.as_vec3().to_array(),
                    skirt: (geo_w * 0.02 + 40.0 * exag) as f32,
                    uv_off: [off.x as f32, off.y as f32],
                    uv_scale: scale as f32,
                    zoom: d.tile.z as f32,
                    cap_color: [0.0; 4],
                },
                Some(d.tile),
                0,
            ));
            max_z = max_z.max(d.tile.z);
        }
        if self.resident.contains_key(&TileId::new(0, 0, 0)) {
            for c in 0..2 {
                let pole = self.caps[c].3;
                recs.push((
                    DrawU {
                        rel_center: (pole - eye).as_vec3().to_array(),
                        layer: CAP_LAYER,
                        abs_center: pole.as_vec3().to_array(),
                        skirt: 0.0,
                        uv_off: [0.0; 2],
                        uv_scale: self.cap_heights[c],
                        zoom: 0.0,
                        cap_color: self.cap_colors[c],
                    },
                    None,
                    c,
                ));
            }
        }
        let mut bytes = vec![0u8; recs.len() * DRAW_STRIDE as usize];
        for (k, r) in recs.iter().enumerate() {
            let b = bytemuck::bytes_of(&r.0);
            bytes[k * DRAW_STRIDE as usize..k * DRAW_STRIDE as usize + b.len()].copy_from_slice(b);
        }
        if !bytes.is_empty() {
            self.queue.write_buffer(&self.draws, 0, &bytes);
        }

        // ---- passes
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("globe") });
        {
            let t = self.target.as_ref().unwrap();
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("globe"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.msaa,
                    depth_slice: None,
                    resolve_target: Some(&t.resolve),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, &self.bg0, &[]);
            rp.set_pipeline(&self.pipe_sky);
            rp.draw(0..3, 0..1);
            rp.set_pipeline(&self.pipe_terrain);
            rp.set_index_buffer(self.index.slice(..), wgpu::IndexFormat::Uint32);
            for (k, (_, tile, cap)) in recs.iter().enumerate() {
                rp.set_bind_group(1, &self.bg1, &[(k as u64 * DRAW_STRIDE) as u32]);
                match tile {
                    Some(t) => {
                        let m = &self.meshes[t];
                        rp.set_vertex_buffer(0, m.buf.slice(..));
                        rp.draw_indexed(0..self.index_count, 0, 0..1);
                    }
                    None => {
                        let (vb, ib, n, _) = &self.caps[*cap];
                        rp.set_vertex_buffer(0, vb.slice(..));
                        rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        rp.draw_indexed(0..*n, 0, 0..1);
                        rp.set_index_buffer(self.index.slice(..), wgpu::IndexFormat::Uint32);
                    }
                }
            }
        }
        self.queue.submit([enc.finish()]);

        // ---- housekeeping
        if self.meshes.len() > 3000 {
            let mut v: Vec<(TileId, u64)> = self.meshes.iter().map(|(t, m)| (*t, m.last)).collect();
            v.sort_by_key(|x| x.1);
            for (t, _) in v.into_iter().take(self.meshes.len() - 2500) {
                self.meshes.remove(&t);
            }
        }
        if self.geo.len() > 200_000 {
            self.geo.clear();
        }
        if self.frame.is_multiple_of(600) {
            self.requested.retain(|_, f| frame - *f < 600);
        }
        self.stats.drawn = recs.len();
        self.stats.resident = self.resident.len();
        self.stats.capacity = self.capacity as usize;
        self.stats.pending_uploads = self.pending.len();
        self.stats.max_zoom_drawn = max_z;
    }

    fn obtainable(&self, t: TileId, w: &Walk) -> bool {
        self.resident.contains_key(&t) || t.z <= w.s.base_zoom || w.svc.store.contains(t) || (w.s.dynamic && t.z <= w.s.gen_max_zoom)
    }

    fn visit(&mut self, t: TileId, fallback: Option<(TileId, u32, f32)>, w: &mut Walk) {
        let frame = self.frame;
        let own = self.resident.get_mut(&t).map(|s| {
            s.last = frame;
            (s.layer, s.elev_max)
        });
        let hmax = own.map(|o| o.1).or(fallback.map(|f| f.2)).unwrap_or(9000.0).max(0.0) as f64;
        let (center, radius, width_m) = {
            let g = self.geo(t);
            (g.center, g.radius, g.width_m)
        };
        let rel = center - w.eye;
        let ext = radius + hmax * w.exag + 1.0;
        // horizon: behind the plane of the visible cap of a sphere of the polar radius
        let el = w.eye.length();
        let r = self.ell.b;
        if el > r + 1000.0 && center.dot(w.eye) / el + ext < r * r / el {
            return;
        }
        for p in &w.planes {
            if p.truncate().dot(rel) + p.w < -ext * p.truncate().length() {
                return;
            }
        }
        let dist = (rel.length() - ext).max(1.0);
        let src = own.map(|o| (t, o.0)).or(fallback.map(|f| (f.0, f.1)));
        if own.is_none() {
            self.requested.insert(t, frame);
            // most wanted first: the blurriest stand-in on screen (the texel size of the ancestor
            // drawn instead, seen from here; nothing to draw at all comes first)
            let blur = match fallback {
                Some((a, _, _)) => width_m * (1u64 << (t.z - a.z)) as f64 / N as f64 / dist,
                None => f64::MAX,
            };
            if w.svc.store.contains(t) {
                w.want_load.push((t, -blur));
            } else if w.s.dynamic && t.z <= w.s.gen_max_zoom {
                w.want_gen.push((t, -blur));
            }
        }
        let Some((src_t, layer)) = src else { return };
        let refine = t.z < w.s.view_max_zoom && width_m / N as f64 / dist > w.s.lod_bias * w.pix;
        if refine && t.children().iter().any(|c| self.obtainable(*c, w)) {
            let fb = Some((src_t, layer, hmax as f32));
            for c in t.children() {
                self.visit(c, fb, w);
            }
        } else {
            w.draws.push(DrawItem { tile: t, src: src_t, layer });
        }
    }

    /// The last frame as RGBA8 (sRGB), for snapshots.
    pub fn read_image(&self) -> Option<(u32, u32, Vec<u8>)> {
        let t = self.target.as_ref()?;
        let row = (t.w * 4).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("snapshot"),
            size: (row * t.h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &t.image, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(t.h) } },
            wgpu::Extent3d { width: t.w, height: t.h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("snapshot read-back"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let data = buf.slice(..).get_mapped_range().ok()?;
        let mut out = Vec::with_capacity((t.w * t.h * 4) as usize);
        for j in 0..t.h as usize {
            out.extend_from_slice(&data[j * row as usize..j * row as usize + t.w as usize * 4]);
        }
        Some((t.w, t.h, out))
    }

    /// True when nothing the view wants is loading, generating or waiting for upload.
    pub fn settled(&self, svc: &Service) -> bool {
        self.pending.is_empty() && svc.in_flight() == 0 && self.stats.want_load == 0 && (self.stats.want_gen == 0) && svc.base_pending() == 0
    }

    /// Ray from the eye through normalized device coordinates hit with the ellipsoid (lat, lon).
    pub fn pick(&self, cf: &CamFrame, ndc: DVec2) -> Option<(f64, f64)> {
        let inv = cf.view_proj.inverse();
        let a = inv * DVec4::new(ndc.x, ndc.y, 1.0, 1.0);
        let b = inv * DVec4::new(ndc.x, ndc.y, 0.5, 1.0);
        let dir = (b.truncate() / b.w - a.truncate() / a.w).normalize();
        // scaled to a unit sphere
        let s = DVec3::new(1.0 / self.ell.a, 1.0 / self.ell.a, 1.0 / self.ell.b);
        let (o, d) = (cf.eye * s, dir * s);
        let (qa, qb, qc) = (d.dot(d), 2.0 * o.dot(d), o.dot(o) - 1.0);
        let disc = qb * qb - 4.0 * qa * qc;
        if disc < 0.0 {
            return None;
        }
        let t = (-qb - disc.sqrt()) / (2.0 * qa);
        if t < 0.0 {
            return None;
        }
        let g = ecef2geodetic(cf.eye + dir * t, &self.ell);
        Some((g.lat, g.lon))
    }
}

struct Walk<'a> {
    eye: DVec3,
    planes: [DVec4; 4],
    pix: f64,
    exag: f64,
    s: &'a Settings,
    svc: &'a Service,
    draws: Vec<DrawItem>,
    want_load: Vec<(TileId, f64)>,
    want_gen: Vec<(TileId, f64)>,
}

/// Left, right, bottom, top planes (a·x + d >= 0 inside) of a camera-relative view-projection.
fn frustum_planes(m: &DMat4) -> [DVec4; 4] {
    let (r0, r1, r3) = (m.row(0), m.row(1), m.row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1]
}

fn buffer(device: &wgpu::Device, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let b = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: data.len() as u64, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: true });
    b.slice(..).get_mapped_range_mut().expect("mapped at creation").copy_from_slice(data);
    b.unmap();
    b
}

/// A polar cap beyond the Mercator limit: a fan around the pole, relative to the pole.
fn cap_mesh(device: &wgpu::Device, ell: &Ellipsoid, sign: f64) -> (wgpu::Buffer, wgpu::Buffer, u32, DVec3) {
    let pole = geodetic2ecef(Geodetic { lat: sign * std::f64::consts::FRAC_PI_2, lon: 0.0, h: 0.0 }, ell);
    let segs = 256usize;
    let rings = 6usize;
    let mut v = vec![Vertex { pos: [0.0; 3], uv: [0.0; 2], skirt: 0.0 }];
    for r in 1..=rings {
        // a little beyond the Mercator edge, under the edge tiles' skirts
        let lat = sign * (std::f64::consts::FRAC_PI_2 - (std::f64::consts::FRAC_PI_2 - MAX_MERCATOR_LAT_RAD + 0.002) * r as f64 / rings as f64);
        for k in 0..segs {
            let lon = std::f64::consts::TAU * k as f64 / segs as f64;
            let p = geodetic2ecef(Geodetic { lat, lon, h: 0.0 }, ell) - pole;
            v.push(Vertex { pos: p.as_vec3().to_array(), uv: [0.0; 2], skirt: 0.0 });
        }
    }
    let mut idx: Vec<u32> = Vec::new();
    for k in 0..segs as u32 {
        let k1 = (k + 1) % segs as u32;
        idx.extend_from_slice(&[0, 1 + k, 1 + k1]);
    }
    for r in 0..rings as u32 - 1 {
        let (a0, b0) = (1 + r * segs as u32, 1 + (r + 1) * segs as u32);
        for k in 0..segs as u32 {
            let k1 = (k + 1) % segs as u32;
            idx.extend_from_slice(&[a0 + k, b0 + k, a0 + k1, a0 + k1, b0 + k, b0 + k1]);
        }
    }
    let vb = buffer(device, "cap vertices", bytemuck::cast_slice(&v), wgpu::BufferUsages::VERTEX);
    let ib = buffer(device, "cap indices", bytemuck::cast_slice(&idx), wgpu::BufferUsages::INDEX);
    (vb, ib, idx.len() as u32, pole)
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
