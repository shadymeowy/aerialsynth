//! Tile generation on the GPU (wgpu compute): the generator of `tile.rs` / `world.rs` /
//! `surface.rs` in WGSL, with the same hashes and noise frames, so it builds the same world.
//!
//! The per-pixel and per-point terrain runs on the GPU; the host (`host.rs`) keeps the parts
//! that are graphs or site lists (drainage network, lake levels, regions, towns), computed from
//! GPU point evaluations and cached across batches.

pub mod device;
pub(crate) mod host;
pub(crate) mod tables;
pub mod types;

pub use device::{shared, Gpu};

use crate::surface::SurfaceModel;
use crate::world::{Ctx, World};
use crate::Config;
use anyhow::{bail, Result};
use geodesy::tiles::{gsd_ew, gsd_ns, pixel_to_latlon, TileId};
use glam::{DVec2, DVec3};
use host::{gsink, point_in, Cache, Prep, PointKey, PointReq};
use std::sync::{Arc, Mutex};
use types::*;
use wgpu::util::DeviceExt;

/// WGSL sources, in dependency order.
pub(crate) const NOISE_WGSL: &str = include_str!("wgsl/noise.wgsl");
const WORLD_WGSL: &str = include_str!("wgsl/world.wgsl");
const POINTS_WGSL: &str = include_str!("wgsl/points.wgsl");
const TILE_A_WGSL: &str = include_str!("wgsl/tile_a.wgsl");
const SURFACE_WGSL: &str = include_str!("wgsl/surface.wgsl");

const N: usize = 256;
/// pass-A grid side (tile + 2-pixel apron)
const NA2: usize = N + 4;
const NBIN: usize = 17;
const NODE_F: usize = 72;
const NODE_IDS: usize = 8;
const PIX_F: usize = 32;
/// grid node spacing (pixels)
const G: f64 = 16.0;
const NG: usize = N / 16 + 5;

/// A storage buffer holding `data`.
pub(crate) fn storage<T: bytemuck::Pod>(d: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    if bytes.is_empty() {
        return d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: 256, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    }
    d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST })
}

/// An output storage buffer of `size` bytes (copyable to a read-back buffer).
pub(crate) fn output(d: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(256),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Copy `buf` back to the host (waits for the GPU).
pub(crate) fn read_back<T: bytemuck::Pod>(g: &Gpu, buf: &wgpu::Buffer, n: usize) -> Result<Vec<T>> {
    let bytes = (n * std::mem::size_of::<T>()) as u64;
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let rb = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("read-back"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &rb, 0, bytes);
    g.queue.submit([enc.finish()]);
    rb.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("GPU read-back"));
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let out = bytemuck::cast_slice::<u8, T>(&rb.slice(..).get_mapped_range()?).to_vec();
    rb.unmap();
    Ok(out)
}

/// A compute pipeline of `src` with its layout derived from the shader.
pub(crate) fn pipeline(d: &wgpu::Device, label: &str, src: &str, entry: &str) -> wgpu::ComputePipeline {
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
    d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: None, module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
}

#[derive(Clone, Copy)]
enum Bind {
    Uniform,
    Ro,
    Rw,
}

fn layout(d: &wgpu::Device, label: &str, kinds: &[Bind]) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = kinds
        .iter()
        .enumerate()
        .map(|(i, k)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match k {
                    Bind::Uniform => wgpu::BufferBindingType::Uniform,
                    Bind::Ro => wgpu::BufferBindingType::Storage { read_only: true },
                    Bind::Rw => wgpu::BufferBindingType::Storage { read_only: false },
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries })
}

fn bind(d: &wgpu::Device, l: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: l, entries: &entries })
}

struct Kernels {
    l_drain: wgpu::BindGroupLayout,
    l_points: wgpu::BindGroupLayout,
    l_tile: wgpu::BindGroupLayout,
    points: wgpu::ComputePipeline,
    nodes: wgpu::ComputePipeline,
    a1: wgpu::ComputePipeline,
    bins: wgpu::ComputePipeline,
    a2: wgpu::ComputePipeline,
}

/// The GPU tile generator of one world.
pub struct GpuGenerator {
    pub world: World,
    pub surface: SurfaceModel,
    gpu: Arc<Gpu>,
    k: Kernels,
    globals: wgpu::BindGroup,
    cache: Mutex<Cache>,
}

/// The WGSL of the point kernels and of the tile kernels.
fn sources() -> (String, String) {
    let consts = tables::wgsl_consts();
    let points = format!("{consts}{NOISE_WGSL}{WORLD_WGSL}{POINTS_WGSL}");
    let tile = format!("{consts}{NOISE_WGSL}{WORLD_WGSL}{TILE_A_WGSL}{SURFACE_WGSL}");
    (points, tile)
}

impl GpuGenerator {
    pub fn new(cfg: Config) -> Result<Self> {
        let gpu = shared()?;
        gpu.check_generator()?;
        if cfg.hydro.levels.len() > 4 {
            bail!("the GPU generator supports up to 4 drainage levels (world.hydro.levels has {})", cfg.hydro.levels.len());
        }
        let world = World::new(cfg);
        let surface = SurfaceModel::new(&world);
        let d = &gpu.device;
        let (octs, fbms) = tables::build(&world, &surface);
        let g_cfg = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("cfg"), contents: bytemuck::bytes_of(&gpu_cfg(&world, &surface)), usage: wgpu::BufferUsages::UNIFORM });
        let g_grads = storage(d, "grads", &tables::grads());
        let g_octs = storage(d, "octs", &octs);
        let g_fbms = storage(d, "fbms", &fbms);
        use Bind::*;
        let l_globals = layout(d, "globals", &[Uniform, Ro, Ro, Ro]);
        let l_drain = layout(d, "drain", &[Ro, Ro, Ro, Ro, Ro]);
        let l_points = layout(d, "points", &[Ro, Rw]);
        let l_tile = layout(d, "tile", &[Ro, Ro, Ro, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw]);
        let globals = bind(d, &l_globals, &[&g_cfg, &g_grads, &g_octs, &g_fbms]);
        let (src_points, src_tile) = sources();
        let pipe = |label: &str, src: &str, l2: &wgpu::BindGroupLayout, entry: &str| {
            let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
            let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(&l_globals), Some(&l_drain), Some(l2)], immediate_size: 0 });
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let k = Kernels {
            points: pipe("points", &src_points, &l_points, "eval_points"),
            nodes: pipe("grid nodes", &src_tile, &l_tile, "grid_nodes"),
            a1: pipe("pass a1", &src_tile, &l_tile, "pass_a1"),
            bins: pipe("bins", &src_tile, &l_tile, "bin_segments"),
            a2: pipe("pass a2", &src_tile, &l_tile, "pass_a2"),
            l_drain,
            l_points,
            l_tile,
        };
        Ok(GpuGenerator { world, surface, gpu, k, globals, cache: Mutex::new(Cache::default()) })
    }

    /// Evaluate points on the GPU (pass A with exact macro fields, see `points.wgsl`).
    fn eval_points(&self, reqs: &[PointReq], lakes: &FxLakes) -> Result<Vec<GTerrain>> {
        let d = &self.gpu.device;
        let mut segs = Vec::new();
        let mut sinks = Vec::new();
        let mut pts = Vec::with_capacity(reqs.len());
        for r in reqs {
            let dr = GDrain { seg0: segs.len() as u32, nseg: r.segs.len() as u32, sink0: sinks.len() as u32, nsink: r.sinks.len() as u32 };
            segs.extend(r.segs.iter().map(GSeg::from));
            sinks.extend_from_slice(&r.sinks);
            pts.push(point_in(&r.ctx, r.mode, dr));
        }
        let list: Vec<u32> = (0..segs.len() as u32).collect();
        let (keys, vals) = lake_table(lakes);
        let b_segs = storage(d, "segs", &segs);
        let b_list = storage(d, "seg list", &list);
        let b_sinks = storage(d, "sinks", &sinks);
        let b_keys = storage(d, "lake keys", &keys);
        let b_vals = storage(d, "lake levels", &vals);
        let b_pts = storage(d, "points", &pts);
        let b_out = output(d, "point results", (pts.len() * std::mem::size_of::<GTerrain>()) as u64);
        let g1 = bind(d, &self.k.l_drain, &[&b_segs, &b_list, &b_sinks, &b_keys, &b_vals]);
        let g2 = bind(d, &self.k.l_points, &[&b_pts, &b_out]);
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.k.points);
            cp.set_bind_group(0, &self.globals, &[]);
            cp.set_bind_group(1, &g1, &[]);
            cp.set_bind_group(2, &g2, &[]);
            cp.dispatch_workgroups((pts.len() as u32).div_ceil(64), 1, 1);
        }
        self.gpu.queue.submit([enc.finish()]);
        read_back(&self.gpu, &b_out, pts.len())
    }

    /// Run `f` on the host caches until it needs no more GPU point evaluations.
    fn settle<T>(&self, cache: &mut Cache, mut f: impl FnMut(&mut Prep) -> T) -> Result<T> {
        for _ in 0..64 {
            let (out, need) = {
                let mut prep = Prep::new(&self.world, cache);
                let out = f(&mut prep);
                if prep.missing == 0 {
                    return Ok(out);
                }
                (out, std::mem::take(&mut prep.need))
            };
            drop(out);
            if need.is_empty() {
                bail!("GPU generator: host preparation is stuck");
            }
            let reqs: Vec<PointReq> = need.into_values().collect();
            let res = self.eval_points(&reqs, &cache.lattice_lakes)?;
            for (r, t) in reqs.iter().zip(res) {
                cache.points.insert(PointKey::new(r.mode, &r.ctx), t);
            }
        }
        bail!("GPU generator: host preparation did not settle")
    }

    /// Pass A at one point (exact macro fields), like `World::terrain`.
    pub fn terrain(&self, lat: f64, lon: f64, gsd: f64) -> Result<GTerrain> {
        let ctx = Ctx::new(lat, lon, gsd, &self.world.ell);
        let mut cache = self.cache.lock().unwrap();
        let t = self.settle(&mut cache, |p| p.point(MODE_FULL, &ctx))?;
        Ok(t.expect("settled"))
    }

    /// Pass A of tiles: the terrain per pass-A pixel centre (tile + 2-pixel apron, 260 x 260).
    pub fn pass_a(&self, ids: &[TileId]) -> Result<Vec<Vec<GTerrain>>> {
        let d = &self.gpu.device;
        let ell = self.world.ell;
        let nt = ids.len();
        // ---- drainage pieces and sink lakes per tile
        struct TileDrain {
            segs: Vec<crate::world::Seg>,
            sinks: Vec<GSink>,
        }
        let mut cache = self.cache.lock().unwrap();
        cache.trim();
        let drains: Vec<TileDrain> = self.settle(&mut cache, |prep| {
            ids.iter()
                .map(|id| {
                    let (ox, oy) = (id.x as f64 * N as f64, id.y as f64 * N as f64);
                    let (lat_c, lon_c) = pixel_to_latlon(DVec2::new(ox + 128.0, oy + 128.0), id.z, N as u32);
                    let c = Ctx::new(lat_c, lon_c, 1.0, &ell);
                    let radius = [(ox - 2.0, oy - 2.0), (ox + 258.0, oy - 2.0), (ox - 2.0, oy + 258.0), (ox + 258.0, oy + 258.0)]
                        .iter()
                        .map(|&(px, py)| {
                            let (la, lo) = pixel_to_latlon(DVec2::new(px, py), id.z, N as u32);
                            (Ctx::new(la, lo, 1.0, &ell).p - c.p).length()
                        })
                        .fold(0.0, f64::max);
                    let gsd_c = gsd_ew(lat_c, id.z, N as u32, &ell);
                    let segs = prep.river_segments(c.p, radius, gsd_c);
                    let mut sinks = Vec::new();
                    for (sid, sc, rad) in World::sink_lakes(&segs) {
                        if (sc - c.p).length() < radius + 1.6 * rad + 10.0 {
                            let level = prep.lake_level_forced(sid, sc, rad);
                            sinks.push(gsink(sid, sc, rad, level));
                        }
                    }
                    TileDrain { segs, sinks }
                })
                .collect()
        })?;
        // ---- tile tables
        let mut rows: Vec<GRow> = Vec::new();
        let mut cols: Vec<GCol> = Vec::new();
        let mut infos: Vec<GTileInfo> = Vec::new();
        let mut segs: Vec<GSeg> = Vec::new();
        let mut sinks: Vec<GSink> = Vec::new();
        for (t, id) in ids.iter().enumerate() {
            let z = id.z;
            let (ox, oy) = (id.x as f64 * N as f64, id.y as f64 * N as f64);
            let row = |py: f64| -> GRow {
                let (lat, _) = pixel_to_latlon(DVec2::new(ox, py), z, N as u32);
                let n = ell.prime_vertical_radius(lat);
                let (sl, cl) = lat.sin_cos();
                let b2a2 = (ell.b / ell.a) * (ell.b / ell.a);
                GRow { ncl: n * cl, z: (n * b2a2) * sl, sl: sl as f32, cl: cl as f32, lat: lat as f32, gsd: gsd_ew(lat, z, N as u32, &ell) as f32, gsd_ns: gsd_ns(lat, z, N as u32, &ell) as f32, _p: 0.0 }
            };
            let col = |px: f64| -> GCol {
                let (_, lon) = pixel_to_latlon(DVec2::new(px, oy), z, N as u32);
                let (so, co) = lon.sin_cos();
                GCol { co, so }
            };
            const GRID_REF_LAT: f64 = 40.0;
            let g_m = G * gsd_ew(GRID_REF_LAT.to_radians(), z, N as u32, &ell);
            let use_grid = g_m <= 2000.0;
            let mut flags = 0;
            if use_grid {
                flags |= TF_GRID;
                if g_m <= 1000.0 {
                    flags |= TF_WARP_GRID;
                }
                if g_m <= 100.0 {
                    flags |= TF_GULLY_GRID;
                }
                if g_m <= 400.0 {
                    flags |= TF_ROADS_GRID;
                }
            }
            let pf_cut = 8.0 * G * gsd_ew(0.0, z, N as u32, &ell);
            let gk0x = (ox / G) as i64 - 2;
            let gk0y = (oy / G) as i64 - 2;
            let mut info = GTileInfo {
                z: z as u32,
                flags,
                ng: NG as u32,
                pre_flags: 0,
                pf_cut: pf_cut as f32,
                relief_cut_r: (2.0 * pf_cut) as f32,
                relief_cut_h: pf_cut as f32,
                gu0: ((ox - 2.0 + 0.5) / G - gk0x as f64) as f32,
                row_a: rows.len() as u32,
                col_a: cols.len() as u32,
                node0: (t * NG * NG) as u32,
                pix0: (t * NA2 * NA2) as u32,
                bin0: (t * NBIN * NBIN) as u32,
                seg0: segs.len() as u32,
                nseg: drains[t].segs.len() as u32,
                sink0: sinks.len() as u32,
                nsink: drains[t].sinks.len() as u32,
                ss: self.world.cfg.tile_supersample.max(1),
                ..Default::default()
            };
            for j in 0..NA2 {
                rows.push(row(oy + j as f64 - 2.0 + 0.5));
            }
            for i in 0..NA2 {
                cols.push(col(ox + i as f64 - 2.0 + 0.5));
            }
            info.row_n = rows.len() as u32;
            info.col_n = cols.len() as u32;
            for k in 0..NG {
                rows.push(row((gk0y + k as i64) as f64 * G));
                cols.push(col((gk0x + k as i64) as f64 * G));
            }
            segs.extend(drains[t].segs.iter().map(GSeg::from));
            sinks.extend_from_slice(&drains[t].sinks);
            infos.push(info);
        }
        // ---- GPU: grid nodes, relief (lake requests), bins, rest of pass A
        let b_tiles = storage(d, "tiles", &infos);
        let b_rows = storage(d, "rows", &rows);
        let b_cols = storage(d, "cols", &cols);
        let b_node_f = output(d, "node fields", (nt * NG * NG * NODE_F * 4) as u64);
        let b_node_ids = output(d, "node ids", (nt * NG * NG * NODE_IDS * 8) as u64);
        let b_node_pts = output(d, "node points", (nt * NG * NG * NODE_IDS * 32) as u64);
        let b_pix = output(d, "pass a1", (nt * NA2 * NA2 * PIX_F * 4) as u64);
        let b_terr = output(d, "pass a", (nt * NA2 * NA2 * std::mem::size_of::<GTerrain>()) as u64);
        let b_bins = output(d, "bins", (nt * NBIN * NBIN * 16) as u64);
        let b_segs = storage(d, "segs", &segs);
        let b_sinks = storage(d, "sinks", &sinks);
        let empty_list = storage::<u32>(d, "empty", &[0]);
        let lake_cap = 4096usize;
        let b_lake_req = output(d, "lake requests", (lake_cap * 48) as u64);
        let b_counters = output(d, "counters", 16);
        let wg = |n: usize| n.div_ceil(16) as u32;
        // relief first: it reports the lattice lakes whose levels are still unknown
        let mut bin_cap = (segs.len() * NBIN * NBIN / 4).clamp(1 << 16, 1 << 24);
        loop {
            let (keys, vals) = lake_table(&cache.lattice_lakes);
            let b_keys = storage(d, "lake keys", &keys);
            let b_vals = storage(d, "lake levels", &vals);
            let b_list = output(d, "bin list", (bin_cap * 4) as u64);
            let g1 = bind(d, &self.k.l_drain, &[&b_segs, &empty_list, &b_sinks, &b_keys, &b_vals]);
            let g2 = bind(d, &self.k.l_tile, &[&b_tiles, &b_rows, &b_cols, &b_node_f, &b_node_ids, &b_node_pts, &b_pix, &b_terr, &b_bins, &b_list, &b_counters, &b_lake_req]);
            let mut enc = d.create_command_encoder(&Default::default());
            enc.clear_buffer(&b_counters, 0, None);
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_bind_group(0, &self.globals, &[]);
                cp.set_bind_group(1, &g1, &[]);
                cp.set_bind_group(2, &g2, &[]);
                cp.set_pipeline(&self.k.nodes);
                cp.dispatch_workgroups(((NG * NG) as u32).div_ceil(64), nt as u32, 1);
                cp.set_pipeline(&self.k.a1);
                cp.dispatch_workgroups(wg(NA2), wg(NA2), nt as u32);
                cp.set_pipeline(&self.k.bins);
                cp.dispatch_workgroups((NBIN * NBIN) as u32, nt as u32, 1);
            }
            self.gpu.queue.submit([enc.finish()]);
            let counters: Vec<u32> = read_back(&self.gpu, &b_counters, 4)?;
            if counters[1] != 0 {
                bin_cap = counters[0] as usize + 1024;
                continue;
            }
            let nreq = counters[2] as usize;
            if nreq > 0 {
                #[repr(C)]
                #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
                struct LakeReq {
                    pt: [f64; 4],
                    id: u64,
                    _p: u64,
                }
                let reqs: Vec<LakeReq> = read_back(&self.gpu, &b_lake_req, nreq.min(lake_cap))?;
                let reqs: Vec<(u64, DVec3)> = reqs.iter().map(|r| (r.id, DVec3::new(r.pt[0], r.pt[1], r.pt[2]))).collect();
                self.settle(&mut cache, |prep| {
                    for &(id, pt) in &reqs {
                        prep.lattice_lake_level(id, pt);
                    }
                })?;
                if nreq > lake_cap {
                    // (more than the request buffer holds: the rest come next round)
                }
                continue;
            }
            // the rest of pass A, with the bins' pieces as the piece list
            let g1 = bind(d, &self.k.l_drain, &[&b_segs, &b_list, &b_sinks, &b_keys, &b_vals]);
            let dummy = output(d, "dummy", 32);
            let g2 = bind(d, &self.k.l_tile, &[&b_tiles, &b_rows, &b_cols, &b_node_f, &b_node_ids, &b_node_pts, &b_pix, &b_terr, &b_bins, &dummy, &b_counters, &b_lake_req]);
            let mut enc = d.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_bind_group(0, &self.globals, &[]);
                cp.set_bind_group(1, &g1, &[]);
                cp.set_bind_group(2, &g2, &[]);
                cp.set_pipeline(&self.k.a2);
                cp.dispatch_workgroups(wg(NA2), wg(NA2), nt as u32);
            }
            self.gpu.queue.submit([enc.finish()]);
            break;
        }
        let all: Vec<GTerrain> = read_back(&self.gpu, &b_terr, nt * NA2 * NA2)?;
        Ok(all.chunks(NA2 * NA2).map(|c| c.to_vec()).collect())
    }
}

type FxLakes = crate::noise::FxHashMap<u64, Option<f64>>;

/// The lattice lake levels as an open-addressing table (key 0 = empty; NONE_F = no lake).
fn lake_table(lakes: &FxLakes) -> (Vec<u64>, Vec<f32>) {
    let n = (lakes.len() * 2).next_power_of_two().max(16);
    let mut keys = vec![0u64; n];
    let mut vals = vec![NONE_F; n];
    for (&id, &level) in lakes {
        let mut k = (crate::noise::mix64(id) % n as u64) as usize;
        while keys[k] != 0 {
            k = (k + 1) % n;
        }
        keys[k] = id;
        vals[k] = level.map_or(NONE_F, |l| l as f32);
    }
    (keys, vals)
}

/// The world config for the GPU.
fn gpu_cfg(w: &World, s: &SurfaceModel) -> GCfg {
    let c = &w.cfg;
    let mut g = GCfg::default();
    if let Some((hp, r, st)) = w.home {
        g.home_p = [hp.x, hp.y, hp.z, 1.0];
        g.home_r = r as f32;
        g.home_st = st as f32;
    }
    for (li, lc) in c.hydro.levels.iter().enumerate().take(4) {
        g.lvl_inv_mlam[li] = 1.0 / w.meander_wavelength(li);
        g.lvl_inv_fplam[li] = 1.0 / w.floodplain_wavelength(li);
        g.lvl_a[li] = [(lc.cell_km * 1000.0) as f32, lc.width_m[0] as f32, lc.width_m[1] as f32, lc.valley_m as f32];
        g.lvl_b[li] = [lc.wet_moisture as f32, lc.meander as f32, lc.max_depth_m as f32, w.meander_wavelength(li) as f32];
    }
    g.inv_gully_lam = 1.0 / c.relief.gully_wavelength_m;
    g.lake_cell = c.hydro.lake_cell_km * 1000.0;
    g.region_cell = c.landuse.region_km * 1000.0;
    g.town_cell = c.landuse.town_cell_km * 1000.0;
    g.seed = w.seed();
    g.nlevels = c.hydro.levels.len() as u32;
    let mut flags = 0;
    if c.hydro.rivers {
        flags |= CF_RIVERS;
    }
    if c.vegetation.trees_in_dsm {
        flags |= CF_TREES_DSM;
    }
    if c.landuse.buildings_in_dsm {
        flags |= CF_BUILDINGS_DSM;
    }
    if c.satellite.shadows {
        flags |= CF_SHADOWS;
    }
    if c.tile_supersample == 2 && c.tile_supersample_adaptive {
        flags |= CF_ADAPTIVE;
    }
    g.flags = flags;
    g.cont_warp = c.continents.warp as f32;
    g.cont_threshold = c.continents.threshold as f32;
    g.mtn_height = c.relief.mountain_height_m as f32;
    g.hill_height = c.relief.hill_height_m as f32;
    g.micro_height = c.relief.micro_height_m as f32;
    g.mesas = c.relief.mesas as f32;
    g.dune_height = c.relief.dune_height_m as f32;
    g.erosion = c.relief.erosion as f32;
    g.gully_lam = c.relief.gully_wavelength_m as f32;
    g.lake_density = c.hydro.lake_density as f32;
    g.eq_temp = c.climate.equator_temp_c as f32;
    g.pole_drop = c.climate.pole_drop_c as f32;
    g.lapse = c.climate.lapse_rate_c_per_km as f32;
    g.moist_bias = c.climate.moisture_bias as f32;
    g.tree_density = c.vegetation.tree_density as f32;
    g.agriculture = c.landuse.agriculture as f32;
    g.towns = c.landuse.towns as f32;
    g.roads = c.landuse.roads as f32;
    g.sun_hx = s.sun_h.x as f32;
    g.sun_hy = s.sun_h.y as f32;
    g.sun_tan = s.sun_tan as f32;
    let look = &c.satellite;
    g.ambient = look.ambient as f32;
    g.direct = look.direct as f32;
    g.exposure = look.exposure as f32;
    g.haze = look.haze as f32;
    let az = look.sun_azimuth_deg.to_radians();
    let el = look.sun_elevation_deg.to_radians();
    g.l0 = (look.ambient + look.direct * el.sin()) as f32;
    g.sun_e = (az.sin() * el.cos()) as f32;
    g.sun_n = (az.cos() * el.cos()) as f32;
    g.sun_u = el.sin() as f32;
    g.saturation = c.albedo.saturation as f32;
    g.brightness = c.albedo.brightness as f32;
    g
}

#[cfg(test)]
mod tests;
