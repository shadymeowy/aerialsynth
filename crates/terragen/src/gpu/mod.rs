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
use anyhow::{bail, Context, Result};
use geodesy::tiles::{gsd_ew, gsd_ns, pixel_to_latlon, TileId};
use glam::{DVec2, DVec3};
use host::{gsink, point_in, Cache, PointKey, PointReq, Prep};
use std::sync::{Arc, Mutex};
use tilestore::TileData;
use types::*;
use wgpu::util::DeviceExt;

/// WGSL sources, in dependency order.
pub(crate) const NOISE_WGSL: &str = include_str!("wgsl/noise.wgsl");
const WORLD_WGSL: &str = include_str!("wgsl/world.wgsl");
const POINTS_WGSL: &str = include_str!("wgsl/points.wgsl");
const TILE_A_WGSL: &str = include_str!("wgsl/tile_a.wgsl");
const SURFACE_WGSL: &str = include_str!("wgsl/surface.wgsl");
const TILE_B_WGSL: &str = include_str!("wgsl/tile_b.wgsl");
const DRAIN_WGSL: &str = include_str!("wgsl/drain.wgsl");

const N: usize = 256;
/// pass-B grid side (tile + 1-pixel apron)
const NA: usize = N + 2;
/// pass-A grid side (tile + 2-pixel apron)
const NA2: usize = N + 4;
const PB_F: usize = 12;
const OUT_U: usize = 6;
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
    d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
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
    let rb = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-back"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &rb, 0, bytes);
    g.queue.submit([enc.finish()]);
    g.map_read(&[rb.slice(..)])?;
    let out = bytemuck::cast_slice::<u8, T>(&rb.slice(..).get_mapped_range()?).to_vec();
    rb.unmap();
    Ok(out)
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
    let entries: Vec<wgpu::BindGroupEntry> =
        bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: l, entries: &entries })
}

struct Kernels {
    l_drain: wgpu::BindGroupLayout,
    l_tables: wgpu::BindGroupLayout,
    l_points: wgpu::BindGroupLayout,
    l_tile: wgpu::BindGroupLayout,
    points: wgpu::ComputePipeline,
    nodes: wgpu::ComputePipeline,
    a1: wgpu::ComputePipeline,
    bins: wgpu::ComputePipeline,
    a2: wgpu::ComputePipeline,
    region_req: wgpu::ComputePipeline,
    town_req: wgpu::ComputePipeline,
    pass_b: wgpu::ComputePipeline,
    open_min_x: wgpu::ComputePipeline,
    open_min_y: wgpu::ComputePipeline,
    open_max_x: wgpu::ComputePipeline,
    open_max_y: wgpu::ComputePipeline,
    open_apply: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    l_lat: wgpu::BindGroupLayout,
    dk: DrainKernels,
}

struct DrainKernels {
    enumerate: wgpu::ComputePipeline,
    heights: wgpu::ComputePipeline,
    mark_targets: wgpu::ComputePipeline,
    targets: wgpu::ComputePipeline,
    mark_sources: wgpu::ComputePipeline,
    sources: wgpu::ComputePipeline,
    count: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    write: wgpu::ComputePipeline,
}

/// The drainage lattice on the GPU: a hash table of lattice points (key: level and cell) with
/// their surface point, height, flags and flow target; kept across batches.
struct Lattice {
    keys: wgpu::Buffer,
    /// the keys after the last enumeration (read-only lookups)
    keys_ro: wgpu::Buffer,
    s: wgpu::Buffer,
    h: wgpu::Buffer,
    flags: wgpu::Buffer,
    tgt: wgpu::Buffer,
    /// per slot: queued for a target / source in this batch; the work lists
    mark: wgpu::Buffer,
    work_t: wgpu::Buffer,
    work_s: wgpu::Buffer,
    cap: usize,
    used: usize,
}

/// Lattice table slots (~50 bytes each).
const LAT_CAP: usize = 1 << 23;
/// Lattice cells per enumeration / gather chunk.
const CHUNK: usize = 4096;

/// A drainage query: the channel pieces whose valley could reach within `radius` of `center`
/// at pixel size `gsd` (`World::river_segments`).
pub(crate) struct DQuery {
    pub center: DVec3,
    pub radius: f64,
    pub gsd: f64,
}

/// The drainage pieces of queries: on the GPU, with each query's range and its sink pieces
/// (end point, half width, in order).
pub(crate) struct DrainOut {
    pub segs: wgpu::Buffer,
    pub ranges: Vec<[u32; 2]>,
    pub sinks: Vec<SinkPieces>,
}

/// The GPU tile generator of one world.
pub struct GpuGenerator {
    pub world: World,
    pub surface: SurfaceModel,
    gpu: Arc<Gpu>,
    k: Kernels,
    globals: wgpu::BindGroup,
    cache: Mutex<Cache>,
    lat: Mutex<Lattice>,
}

/// The WGSL of the point kernels and of the tile kernels.
fn sources() -> (String, String, String) {
    let consts = tables::wgsl_consts();
    let w = World::new(Config::default());
    let (_, pal) = tables::palette(&SurfaceModel::new(&w).pal);
    let points = format!("{consts}{NOISE_WGSL}{WORLD_WGSL}{POINTS_WGSL}");
    let tile = format!("{consts}{pal}{NOISE_WGSL}{WORLD_WGSL}{TILE_A_WGSL}{SURFACE_WGSL}{TILE_B_WGSL}");
    let drain = format!("{consts}{NOISE_WGSL}{WORLD_WGSL}{DRAIN_WGSL}");
    (points, tile, drain)
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
        let g_cfg = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cfg"),
            contents: bytemuck::bytes_of(&gpu_cfg(&world, &surface)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let g_grads = storage(d, "grads", &tables::grads());
        let g_octs = storage(d, "octs", &octs);
        let g_fbms = storage(d, "fbms", &fbms);
        use Bind::*;
        let (pal, _) = tables::palette(&surface.pal);
        let g_pal = storage(d, "palette", &pal);
        let l_globals = layout(d, "globals", &[Uniform, Ro, Ro, Ro, Ro]);
        let l_drain = layout(d, "drain", &[Ro, Ro, Ro, Ro, Ro]);
        let l_tables = layout(d, "tables", &[Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro]);
        let l_points = layout(d, "points", &[Ro, Rw, Rw, Rw]);
        let l_lat = layout(d, "drainage", &[Rw, Rw, Rw, Rw, Rw, Ro, Ro, Rw, Rw, Rw, Rw, Rw, Rw, Ro, Ro, Rw, Rw, Rw]);
        let l_tile = layout(d, "tile", &[Ro, Ro, Ro, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw]);
        let globals = bind(d, &l_globals, &[&g_cfg, &g_grads, &g_octs, &g_fbms, &g_pal]);
        let (src_points, src_tile, src_drain) = sources();
        let t_compile = std::time::Instant::now();
        let cache = PipelineCache::open(&gpu);
        let m_points = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("points"), source: wgpu::ShaderSource::Wgsl(src_points.into()) });
        let m_tile = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("tile"), source: wgpu::ShaderSource::Wgsl(src_tile.into()) });
        let m_drain = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("drainage"), source: wgpu::ShaderSource::Wgsl(src_drain.into()) });
        let pl_drain = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("drainage"),
            bind_group_layouts: &[Some(&l_globals), Some(&l_drain), Some(&l_lat)],
            immediate_size: 0,
        });
        let pl_points = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points"),
            bind_group_layouts: &[Some(&l_globals), Some(&l_drain), Some(&l_points)],
            immediate_size: 0,
        });
        let pl_tile = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile"),
            bind_group_layouts: &[Some(&l_globals), Some(&l_tables), Some(&l_tile)],
            immediate_size: 0,
        });
        let pipe = |module: &wgpu::ShaderModule, pl: &wgpu::PipelineLayout, entry: &str| {
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: cache.as_ref().map(|c| &c.cache),
            })
        };
        // (the driver compiles each pipeline on its own: in parallel)
        let names = [
            "grid_nodes",
            "pass_a1",
            "bin_segments",
            "pass_a2",
            "region_requests",
            "town_requests",
            "pass_b",
            "open_min_x",
            "open_min_y",
            "open_max_x",
            "open_max_y",
            "open_apply",
            "finish",
        ];
        let dnames = ["lat_enum", "lat_heights", "mark_targets", "lat_targets", "mark_sources", "lat_sources", "gather_count", "gather_scan", "gather_write"];
        let (points, mut tile, mut drain): (wgpu::ComputePipeline, Vec<wgpu::ComputePipeline>, Vec<wgpu::ComputePipeline>) = std::thread::scope(|s| {
            let hs: Vec<_> = names.iter().map(|n| s.spawn(|| pipe(&m_tile, &pl_tile, n))).collect();
            let hd: Vec<_> = dnames.iter().map(|n| s.spawn(|| pipe(&m_drain, &pl_drain, n))).collect();
            let points = pipe(&m_points, &pl_points, "eval_points");
            (points, hs.into_iter().map(|h| h.join().expect("pipeline")).collect(), hd.into_iter().map(|h| h.join().expect("pipeline")).collect())
        });
        let mut dk = || drain.remove(0);
        let dk = DrainKernels {
            enumerate: dk(),
            heights: dk(),
            mark_targets: dk(),
            targets: dk(),
            mark_sources: dk(),
            sources: dk(),
            count: dk(),
            scan: dk(),
            write: dk(),
        };
        if let Some(c) = &cache {
            c.save();
        }
        if std::env::var_os("TERRAGEN_PROFILE").is_some() {
            eprintln!("GPU generator: pipelines ready in {:.1} s", t_compile.elapsed().as_secs_f64());
        }
        let mut t = || tile.remove(0);
        let k = Kernels {
            points,
            nodes: t(),
            a1: t(),
            bins: t(),
            a2: t(),
            region_req: t(),
            town_req: t(),
            pass_b: t(),
            open_min_x: t(),
            open_min_y: t(),
            open_max_x: t(),
            open_max_y: t(),
            open_apply: t(),
            finish: t(),
            l_drain,
            l_tables,
            l_points,
            l_tile,
            l_lat,
            dk,
        };
        let lat = Lattice {
            keys: output(d, "lattice keys", (LAT_CAP * 8) as u64),
            keys_ro: output(d, "lattice keys (copy)", (LAT_CAP * 8) as u64),
            s: output(d, "lattice points", (LAT_CAP * 32) as u64),
            h: output(d, "lattice heights", (LAT_CAP * 4) as u64),
            flags: output(d, "lattice flags", (LAT_CAP * 4) as u64),
            tgt: output(d, "lattice targets", (LAT_CAP * 4) as u64),
            mark: output(d, "lattice marks", (LAT_CAP * 4) as u64),
            work_t: output(d, "target work", (LAT_CAP * 4) as u64),
            work_s: output(d, "source work", (LAT_CAP * 4) as u64),
            cap: LAT_CAP,
            used: 0,
        };
        Ok(GpuGenerator { world, surface, gpu, k, globals, cache: Mutex::new(Cache::default()), lat: Mutex::new(lat) })
    }

    /// Evaluate points on the GPU (pass A with exact macro fields, see `points.wgsl`): their
    /// results, and for `MODE_REPORT` requests the sink pieces they keep. Points are grouped by
    /// area: one drainage query per group, filtered per point (`DR_KEEP`).
    fn eval_points(&self, reqs: &[PointReq], lakes: &FxLakes) -> Result<(Vec<GTerrain>, Vec<SinkPieces>)> {
        const GROUP: f64 = 30_000.0;
        const GROUPS_PER_PASS: usize = 2048;
        let d = &self.gpu.device;
        // key: (gsd bits, GROUP-sized cell of the point)
        let mut groups: crate::noise::FxHashMap<(u64, [i64; 3]), Vec<usize>> = Default::default();
        for (i, r) in reqs.iter().enumerate() {
            let g = (r.ctx.p / GROUP).floor();
            groups.entry((r.ctx.gsd.to_bits(), [g.x as i64, g.y as i64, g.z as i64])).or_default().push(i);
        }
        let mut groups: Vec<_> = groups.into_iter().collect();
        groups.sort_unstable_by_key(|g| g.0);
        let (keys, vals) = lake_table(lakes);
        let b_keys = storage(d, "lake keys", &keys);
        let b_vals = storage(d, "lake levels", &vals);
        let empty = output(d, "empty", 256);
        let mut results = vec![GTerrain::default(); reqs.len()];
        let mut reports: Vec<SinkPieces> = vec![Vec::new(); reqs.len()];
        for pass in groups.chunks(GROUPS_PER_PASS) {
            let queries: Vec<DQuery> = pass
                .iter()
                .map(|(_, idx)| {
                    let c = idx.iter().fold(DVec3::ZERO, |a, &i| a + reqs[i].ctx.p) / idx.len() as f64;
                    let r = idx.iter().map(|&i| (reqs[i].ctx.p - c).length()).fold(0.0, f64::max);
                    DQuery { center: c, radius: r, gsd: reqs[idx[0]].ctx.gsd }
                })
                .collect();
            let dr_out = self.drainage(&queries)?;
            let mut sinks = Vec::new();
            let mut pts = Vec::new();
            let mut order = Vec::new();
            for (g, (_, idx)) in pass.iter().enumerate() {
                for &i in idx {
                    let r = &reqs[i];
                    let dr = GDrain {
                        seg0: dr_out.ranges[g][0],
                        nseg: dr_out.ranges[g][1],
                        sink0: sinks.len() as u32,
                        nsink: r.sinks.len() as u32,
                        flags: DR_DIRECT | DR_KEEP,
                        _p: [0; 3],
                    };
                    sinks.extend_from_slice(&r.sinks);
                    pts.push(point_in(&r.ctx, r.mode, dr));
                    order.push(i);
                }
            }
            let rep_cap = 1usize << 16;
            let b_sinks = storage(d, "sinks", &sinks);
            let b_pts = storage(d, "points", &pts);
            let b_out = output(d, "point results", (pts.len() * std::mem::size_of::<GTerrain>()) as u64);
            let b_reps = output(d, "sink reports", (rep_cap * std::mem::size_of::<GSinkPiece>()) as u64);
            let b_rep_n = output(d, "sink report count", 16);
            let g1 = bind(d, &self.k.l_drain, &[&dr_out.segs, &empty, &b_sinks, &b_keys, &b_vals]);
            let g2 = bind(d, &self.k.l_points, &[&b_pts, &b_out, &b_reps, &b_rep_n]);
            let mut enc = d.create_command_encoder(&Default::default());
            enc.clear_buffer(&b_rep_n, 0, None);
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&self.k.points);
                cp.set_bind_group(0, &self.globals, &[]);
                cp.set_bind_group(1, &g1, &[]);
                cp.set_bind_group(2, &g2, &[]);
                let groups = (pts.len() as u32).div_ceil(64);
                let gx = groups.min(65535);
                cp.dispatch_workgroups(gx, groups.div_ceil(gx), 1);
            }
            self.gpu.queue.submit([enc.finish()]);
            let out: Vec<GTerrain> = read_back(&self.gpu, &b_out, pts.len())?;
            for (k, &i) in order.iter().enumerate() {
                results[i] = out[k];
            }
            let nrep = read_back::<u32>(&self.gpu, &b_rep_n, 4)?[0] as usize;
            if nrep > rep_cap {
                bail!("GPU generator: {nrep} sink pieces reported in one pass (at most {rep_cap})");
            }
            let mut reps: Vec<GSinkPiece> = read_back(&self.gpu, &b_reps, nrep)?;
            reps.sort_unstable_by_key(|r| (r.owner, r.order));
            for r in reps {
                reports[order[r.owner as usize]].push((DVec3::new(r.b[0], r.b[1], r.b[2]), r.hw as f64));
            }
        }
        Ok((results, reports))
    }

    /// The drainage pieces of `queries` (`World::river_segments`), from the lattice on the GPU:
    /// the lattice points the queries need are added (with their heights), flow targets and
    /// sources computed, and each query's pieces gathered in the CPU's order.
    fn drainage(&self, queries: &[DQuery]) -> Result<DrainOut> {
        let d = &self.gpu.device;
        let w = &self.world;
        let ell = w.ell;
        let mut boxes_e: Vec<GBox> = Vec::new();
        let mut boxes_t: Vec<GBox> = Vec::new();
        let mut boxes_g: Vec<GBox> = Vec::new();
        let mut q_boxes: Vec<[u32; 2]> = Vec::new();
        for (qi, q) in queries.iter().enumerate() {
            let first = boxes_g.len();
            if w.cfg.hydro.rivers {
                for (lvl, lc) in w.cfg.hydro.levels.iter().enumerate() {
                    let cell = lc.cell_km * 1000.0;
                    if lc.valley_m < 0.2 * q.gsd && lc.width_m[1] < 0.15 * q.gsd {
                        continue;
                    }
                    let reach = q.radius + 4.6 * cell + lc.valley_m;
                    let lo = ((q.center - DVec3::splat(reach)) / cell).floor();
                    let hi = ((q.center + DVec3::splat(reach)) / cell).floor();
                    boxes_g.push(dbox(&ell, q.center, lo, hi, cell, reach, lvl, qi, q.radius));
                    boxes_e.push(dbox(&ell, q.center, lo - DVec3::splat(4.0), hi + DVec3::splat(4.0), cell, reach, lvl, qi, q.radius));
                    boxes_t.push(dbox(&ell, q.center, lo - DVec3::splat(2.0), hi + DVec3::splat(2.0), cell, reach, lvl, qi, q.radius));
                }
            }
            q_boxes.push([first as u32, (boxes_g.len() - first) as u32]);
        }
        let chunks = |boxes: &mut [GBox]| -> Vec<u32> {
            let mut out = Vec::new();
            for (b, bx) in boxes.iter_mut().enumerate() {
                bx.chunk0 = out.len() as u32;
                let n = (bx.dims[0] as usize * bx.dims[1] as usize * bx.dims[2] as usize).div_ceil(CHUNK);
                out.extend(std::iter::repeat_n(b as u32, n));
            }
            out
        };
        let chunks_e = chunks(&mut boxes_e);
        let chunks_t = chunks(&mut boxes_t);
        let chunks_g = chunks(&mut boxes_g);
        let t_drain = std::time::Instant::now();
        let mut lat = self.lat.lock().unwrap();
        let mut seg_cap = (1usize << 18).max(queries.len() * 64);
        let mut sink_cap = 1usize << 16;
        let b_boxes_e = storage(d, "boxes", &boxes_e);
        let b_chunks_e = storage(d, "chunks", &chunks_e);
        let b_boxes_t = storage(d, "boxes", &boxes_t);
        let b_chunks_t = storage(d, "chunks", &chunks_t);
        let b_boxes_g = storage(d, "boxes", &boxes_g);
        let b_chunks_g = storage(d, "chunks", &chunks_g);
        let b_qboxes = storage(d, "query boxes", &q_boxes);
        let b_count = output(d, "drainage counters", 32);
        let b_chunk_n = output(d, "chunk counts", (chunks_g.len().max(1) * 4) as u64);
        let b_range = output(d, "query ranges", (queries.len().max(1) * 8) as u64);
        let wg2 = |n: usize| {
            let g = n.max(1) as u32;
            let gx = g.min(65535);
            [gx, g.div_ceil(gx), 1]
        };
        let run = |lat: &Lattice,
                   keys_ro: &wgpu::Buffer,
                   enc: &mut wgpu::CommandEncoder,
                   new_list: &wgpu::Buffer,
                   segs: &wgpu::Buffer,
                   b_sinks: &wgpu::Buffer,
                   boxes: &wgpu::Buffer,
                   chunks: &wgpu::Buffer,
                   passes: &[(&wgpu::ComputePipeline, [u32; 3])]| {
            let g1 = bind(d, &self.k.l_drain, &[&b_dummy(d), &b_dummy(d), &b_dummy(d), &b_dummy(d), &b_dummy(d)]);
            let g2 = bind(
                d,
                &self.k.l_lat,
                &[
                    &lat.keys,
                    &lat.s,
                    &lat.h,
                    &lat.flags,
                    &lat.tgt,
                    boxes,
                    chunks,
                    new_list,
                    &b_count,
                    segs,
                    &b_chunk_n,
                    &b_range,
                    b_sinks,
                    &b_qboxes,
                    keys_ro,
                    &lat.mark,
                    &lat.work_t,
                    &lat.work_s,
                ],
            );
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_bind_group(0, &self.globals, &[]);
            cp.set_bind_group(1, &g1, &[]);
            cp.set_bind_group(2, &g2, &[]);
            for (p, n) in passes {
                cp.set_pipeline(p);
                cp.dispatch_workgroups(n[0], n[1], n[2]);
            }
        };
        // ---- the lattice points the queries need (a full table is cleared and filled again)
        let n_cells: usize = boxes_e.iter().map(|b| b.dims[0] as usize * b.dims[1] as usize * b.dims[2] as usize).sum();
        let new_cap = n_cells.min(lat.cap).max(1);
        let b_new = output(d, "new lattice points", (new_cap * 4) as u64);
        let dummy_segs = output(d, "-", 256);
        let dummy_ro = output(d, "-", 256);
        let dummy_sinks = output(d, "-", 256);
        let mut n_new = 0usize;
        for attempt in 0..2 {
            if lat.used > lat.cap * 3 / 5 || attempt == 1 {
                let mut enc = d.create_command_encoder(&Default::default());
                enc.clear_buffer(&lat.keys, 0, None);
                enc.clear_buffer(&lat.flags, 0, None);
                self.gpu.queue.submit([enc.finish()]);
                lat.used = 0;
            }
            let mut enc = d.create_command_encoder(&Default::default());
            enc.clear_buffer(&b_count, 0, None);
            if !chunks_e.is_empty() {
                run(&lat, &dummy_ro, &mut enc, &b_new, &dummy_segs, &dummy_sinks, &b_boxes_e, &b_chunks_e, &[(&self.k.dk.enumerate, wg2(chunks_e.len()))]);
            }
            enc.copy_buffer_to_buffer(&lat.keys, 0, &lat.keys_ro, 0, (lat.cap * 8) as u64);
            self.gpu.queue.submit([enc.finish()]);
            let c: Vec<u32> = read_back(&self.gpu, &b_count, 8)?;
            n_new = c[0] as usize;
            if c[1] == 0 && n_new <= new_cap {
                break;
            }
            if attempt == 1 {
                bail!("GPU generator: the drainage lattice table is too small ({} points for one batch)", n_new);
            }
        }
        lat.used += n_new;
        let prof = std::env::var_os("TERRAGEN_PROFILE").is_some();
        let t_enum = t_drain.elapsed().as_secs_f64();
        // ---- heights, targets, sources, the queries' pieces
        loop {
            let b_segs = output(d, "drainage pieces", (seg_cap * std::mem::size_of::<GSeg>()) as u64);
            let b_sinks = output(d, "sink pieces", (sink_cap * std::mem::size_of::<GSinkPiece>()) as u64);
            let mut enc = d.create_command_encoder(&Default::default());
            enc.clear_buffer(&b_count, 12, None);
            enc.clear_buffer(&lat.mark, 0, None);
            // (targets: the gather boxes and 2 cells around, for the sources and the next edge)
            let passes: Vec<(&str, &wgpu::ComputePipeline, [u32; 3], &wgpu::Buffer, &wgpu::Buffer)> = vec![
                ("heights", &self.k.dk.heights, wg2(n_new.div_ceil(64)), &b_boxes_g, &b_chunks_g),
                ("mark_t", &self.k.dk.mark_targets, wg2(chunks_t.len()), &b_boxes_t, &b_chunks_t),
                ("targets", &self.k.dk.targets, wg2(lat.cap.min(n_cells).div_ceil(64)), &b_boxes_t, &b_chunks_t),
                ("mark_s", &self.k.dk.mark_sources, wg2(chunks_g.len()), &b_boxes_g, &b_chunks_g),
                ("sources", &self.k.dk.sources, wg2(lat.cap.min(n_cells).div_ceil(64)), &b_boxes_g, &b_chunks_g),
                ("count", &self.k.dk.count, wg2(chunks_g.len()), &b_boxes_g, &b_chunks_g),
                ("scan", &self.k.dk.scan, [(queries.len() as u32).div_ceil(64).max(1), 1, 1], &b_boxes_g, &b_chunks_g),
                ("write", &self.k.dk.write, wg2(chunks_g.len()), &b_boxes_g, &b_chunks_g),
            ];
            let mut times = String::new();
            for (name, p, n, bx, ch) in &passes {
                let t = std::time::Instant::now();
                run(&lat, &lat.keys_ro, &mut enc, &b_new, &b_segs, &b_sinks, bx, ch, &[(p, *n)]);
                if prof {
                    // each pass on its own (timing)
                    self.gpu.queue.submit([std::mem::replace(&mut enc, d.create_command_encoder(&Default::default())).finish()]);
                    d.poll(wgpu::PollType::wait_indefinitely())?;
                    times += &format!(" {name} {:.3}", t.elapsed().as_secs_f64());
                }
            }
            self.gpu.queue.submit([enc.finish()]);
            if prof {
                eprintln!("    drainage passes:{times}");
            }
            n_new = 0;
            let c: Vec<u32> = read_back(&self.gpu, &b_count, 8)?;
            if c[2] != 0 {
                bail!("GPU generator: drainage data missing for {} lattice points", c[2]);
            }
            let (total, nsink) = (c[3] as usize, c[4] as usize);
            if total > seg_cap || nsink > sink_cap {
                // too small for this batch: larger buffers, the same passes again
                seg_cap = seg_cap.max(total + 1024);
                sink_cap = sink_cap.max(nsink + 1024);
                continue;
            }
            if prof {
                eprintln!(
                    "    drainage: {} queries, {} cells, {} new points, {} pieces: enumerate {:.3} s, all {:.3} s",
                    queries.len(),
                    n_cells,
                    lat.used,
                    total,
                    t_enum,
                    t_drain.elapsed().as_secs_f64()
                );
            }
            let ranges: Vec<[u32; 2]> = if queries.is_empty() { vec![] } else { read_back(&self.gpu, &b_range, queries.len())? };
            let mut sp: Vec<GSinkPiece> = read_back(&self.gpu, &b_sinks, nsink)?;
            sp.sort_unstable_by_key(|p| p.order);
            let mut sinks = vec![Vec::new(); queries.len()];
            for p in sp {
                sinks[p.owner as usize].push((DVec3::new(p.b[0], p.b[1], p.b[2]), p.hw as f64));
            }
            return Ok(DrainOut { segs: b_segs, ranges, sinks });
        }
    }

    /// The GPU's name.
    pub fn adapter(&self) -> &str {
        &self.gpu.info.name
    }

    /// Run `f` on the host caches until it needs no more GPU point evaluations.
    fn settle<T>(&self, cache: &mut Cache, mut f: impl FnMut(&mut Prep) -> T) -> Result<T> {
        let prof = std::env::var_os("TERRAGEN_PROFILE").is_some();
        for round in 0..64 {
            let t0 = std::time::Instant::now();
            let (out, mut need) = {
                let mut prep = Prep::new(&self.world, cache);
                let out = f(&mut prep);
                prep.prepare();
                if prof {
                    eprintln!("  settle round {round}: host {:.3} s, {} missing, {} requests", t0.elapsed().as_secs_f64(), prep.missing, prep.need.len());
                }
                if prep.missing == 0 {
                    return Ok(out);
                }
                (out, std::mem::take(&mut prep.need))
            };
            drop(out);
            if need.is_empty() {
                bail!("GPU generator: host preparation is stuck");
            }
            let mut reqs: Vec<PointReq> = need.drain().map(|(_, r)| r).collect();
            reqs.sort_unstable_by(|a, b| {
                (a.mode, a.ctx.p.x, a.ctx.p.y, a.ctx.p.z).partial_cmp(&(b.mode, b.ctx.p.x, b.ctx.p.y, b.ctx.p.z)).unwrap_or(std::cmp::Ordering::Equal)
            });
            let (res, reps) = self.eval_points(&reqs, &cache.lattice_lakes)?;
            for ((r, t), rep) in reqs.iter().zip(res).zip(reps) {
                if r.mode == MODE_REPORT {
                    cache.point_sinks.insert(PointKey::new(MODE_FULL, &r.ctx), rep);
                } else {
                    cache.points.insert(PointKey::new(r.mode, &r.ctx), t);
                }
            }
        }
        bail!("GPU generator: host preparation did not settle")
    }

    /// Pass A at points (lat, lon in radians, pixel size in metres; exact macro fields), like
    /// `World::terrain`.
    pub fn terrain_points(&self, pts: &[(f64, f64, f64)]) -> Result<Vec<GTerrain>> {
        let ell = self.world.ell;
        let ctxs: Vec<Ctx> = pts.iter().map(|&(lat, lon, gsd)| Ctx::new(lat, lon, gsd, &ell)).collect();
        let mut cache = self.cache.lock().unwrap();
        cache.trim();
        let out = self.gpu.scoped(|| self.settle(&mut cache, |p| ctxs.iter().map(|c| p.point(MODE_FULL, c)).collect::<Vec<_>>()))?;
        out.into_iter().map(|t| t.context("GPU generator: a point did not settle")).collect()
    }

    /// Pass A of tiles: the terrain per pass-A pixel centre (tile + 2-pixel apron, 260 x 260).
    pub fn pass_a(&self, ids: &[TileId]) -> Result<Vec<Vec<GTerrain>>> {
        Ok(self.gpu.scoped(|| self.run(ids, false))?.0)
    }

    /// Generate tiles (one batch on the GPU; halved while its land-use sites overflow the
    /// request buffers).
    pub fn tiles(&self, ids: &[TileId]) -> Result<Vec<TileData>> {
        self.gpu.scoped(|| self.tiles_split(ids))
    }

    fn tiles_split(&self, ids: &[TileId]) -> Result<Vec<TileData>> {
        match self.run(ids, true) {
            Ok(r) => Ok(r.1),
            Err(e) if e.is::<Overflow>() && ids.len() > 1 => {
                let (a, b) = ids.split_at(ids.len() / 2);
                let mut out = self.tiles_split(a)?;
                out.extend(self.tiles_split(b)?);
                Ok(out)
            }
            Err(e) => Err(e),
        }
    }

    /// The tile pipeline: pass A (and with `full` pass B and the output layers).
    fn run(&self, ids: &[TileId], full: bool) -> Result<(Vec<Vec<GTerrain>>, Vec<TileData>)> {
        let d = &self.gpu.device;
        let ell = self.world.ell;
        let nt = ids.len();
        if nt == 0 {
            return Ok((vec![], vec![]));
        }
        let ss = self.world.cfg.tile_supersample.max(1) as usize;
        let prof = std::env::var_os("TERRAGEN_PROFILE").is_some();
        let t_run = std::time::Instant::now();
        let stamp = |what: &str| {
            if prof {
                eprintln!(
                    "  batch of {nt} (z{}..{}): {what} at {:.3} s",
                    ids.iter().map(|i| i.z).min().unwrap_or(0),
                    ids.iter().map(|i| i.z).max().unwrap_or(0),
                    t_run.elapsed().as_secs_f64()
                );
            }
        };
        // ---- drainage pieces (on the GPU) and sink lakes per tile
        let mut cache = self.cache.lock().unwrap();
        cache.trim();
        let mut tq = Vec::with_capacity(nt);
        for id in ids {
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
            tq.push(DQuery { center: c.p, radius, gsd: gsd_ew(lat_c, id.z, N as u32, &ell) });
        }
        let drain_out = self.drainage(&tq)?;
        let tile_sinks: Vec<Vec<GSink>> = self.settle(&mut cache, |prep| {
            tq.iter()
                .zip(&drain_out.sinks)
                .map(|(q, pieces)| {
                    let segs: Vec<crate::world::Seg> = pieces
                        .iter()
                        .map(|&(b, hw)| crate::world::Seg { a: b, b, ha: 0.0, hb: 0.0, level: 0, hw, valley: 0.0, hw_b: hw, sink: true })
                        .collect();
                    let mut sinks = Vec::new();
                    for (sid, sc, rad) in World::sink_lakes(&segs) {
                        if (sc - q.center).length() < q.radius + 1.6 * rad + 10.0 {
                            let level = prep.lake_level_forced(sid, sc, rad);
                            sinks.push(gsink(sid, sc, rad, level));
                        }
                    }
                    sinks
                })
                .collect()
        })?;
        stamp("drainage");
        // ---- tile tables
        let mut rows: Vec<GRow> = Vec::new();
        let mut cols: Vec<GCol> = Vec::new();
        let mut infos: Vec<GTileInfo> = Vec::new();
        let mut sinks: Vec<GSink> = Vec::new();
        for (t, id) in ids.iter().enumerate() {
            let z = id.z;
            let (ox, oy) = (id.x as f64 * N as f64, id.y as f64 * N as f64);
            let row = |py: f64| -> GRow {
                let (lat, _) = pixel_to_latlon(DVec2::new(ox, py), z, N as u32);
                let n = ell.prime_vertical_radius(lat);
                let (sl, cl) = lat.sin_cos();
                let b2a2 = (ell.b / ell.a) * (ell.b / ell.a);
                GRow {
                    ncl: n * cl,
                    z: (n * b2a2) * sl,
                    sl: sl as f32,
                    cl: cl as f32,
                    lat: lat as f32,
                    gsd: gsd_ew(lat, z, N as u32, &ell) as f32,
                    gsd_ns: gsd_ns(lat, z, N as u32, &ell) as f32,
                    _p: 0.0,
                }
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
                seg0: drain_out.ranges[t][0],
                nseg: drain_out.ranges[t][1],
                sink0: sinks.len() as u32,
                nsink: tile_sinks[t].len() as u32,
                ss: ss as u32,
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
            // pass-B sub-samples
            info.row_b = rows.len() as u32;
            info.col_b = cols.len() as u32;
            let sub = |s: usize| (s as f64 + 0.5) / ss as f64 - 0.5;
            for j in 0..NA {
                for sy in 0..ss {
                    rows.push(row(oy + j as f64 - 1.0 + 0.5 + sub(sy)));
                }
            }
            for i in 0..NA {
                for sx in 0..ss {
                    cols.push(col(ox + i as f64 - 1.0 + 0.5 + sub(sx)));
                }
            }
            sinks.extend_from_slice(&tile_sinks[t]);
            infos.push(info);
        }
        // ---- GPU buffers
        let b_tiles = storage(d, "tiles", &infos);
        let b_rows = storage(d, "rows", &rows);
        let b_cols = storage(d, "cols", &cols);
        let b_node_f = output(d, "node fields", (nt * NG * NG * NODE_F * 4) as u64);
        let b_node_ids = output(d, "node ids", (nt * NG * NG * NODE_IDS * 8) as u64);
        let b_node_pts = output(d, "node points", (nt * NG * NG * NODE_IDS * 32) as u64);
        let b_pix = output(d, "pass a1", (nt * NA2 * NA2 * PIX_F * 4) as u64);
        let b_terr = output(d, "pass a", (nt * NA2 * NA2 * std::mem::size_of::<GTerrain>()) as u64);
        let b_bins = output(d, "bins", (nt * NBIN * NBIN * 16) as u64);
        let b_segs = &drain_out.segs;
        let b_sinks = storage(d, "sinks", &sinks);
        // stand-ins for unused bindings (read-only in group 1, writable in group 2)
        let empty = output(d, "empty", 256);
        let unused = output(d, "unused", 256);
        let lake_cap = 1usize << 16;
        let site_cap = 1 << 18;
        let b_lake_req = output(d, "lake requests", (lake_cap * 64) as u64);
        let b_region_req = output(d, "region requests", (site_cap * 64) as u64);
        let b_town_req = output(d, "town requests", (site_cap * 16) as u64);
        let b_counters = output(d, "counters", 32);
        let (b_pixb, b_scr_a, b_scr_b, b_out, b_ranges) = if full {
            (
                output(d, "pass b", (nt * NA2 * NA2 * PB_F * 4) as u64),
                output(d, "scratch a", (nt * NA2 * NA2 * 4) as u64),
                output(d, "scratch b", (nt * NA2 * NA2 * 4) as u64),
                output(d, "tile out", (nt * N * N * OUT_U * 4) as u64),
                output(d, "ranges", (nt * 8) as u64),
            )
        } else {
            (output(d, "-", 256), output(d, "-", 256), output(d, "-", 256), output(d, "-", 256), output(d, "-", 256))
        };
        let wg = |n: usize| n.div_ceil(16) as u32;
        let group2 = |list: &wgpu::Buffer| {
            bind(
                d,
                &self.k.l_tile,
                &[
                    &b_tiles,
                    &b_rows,
                    &b_cols,
                    &b_node_f,
                    &b_node_ids,
                    &b_node_pts,
                    &b_pix,
                    &b_terr,
                    &b_bins,
                    list,
                    &b_counters,
                    &b_lake_req,
                    &b_pixb,
                    &b_scr_a,
                    &b_scr_b,
                    &b_out,
                    &b_ranges,
                    &b_region_req,
                    &b_town_req,
                ],
            )
        };
        let run_passes = |g1: &wgpu::BindGroup, g2: &wgpu::BindGroup, passes: &[(&wgpu::ComputePipeline, [u32; 3])], clear: bool| {
            let mut enc = d.create_command_encoder(&Default::default());
            if clear {
                enc.clear_buffer(&b_counters, 0, None);
            }
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_bind_group(0, &self.globals, &[]);
                cp.set_bind_group(1, g1, &[]);
                cp.set_bind_group(2, g2, &[]);
                for (p, n) in passes {
                    cp.set_pipeline(p);
                    cp.dispatch_workgroups(n[0], n[1], n[2]);
                }
            }
            self.gpu.queue.submit([enc.finish()]);
        };
        // ---- grid nodes and relief (reporting the lattice lakes whose levels are unknown), bins
        let n_segs: usize = drain_out.ranges.iter().map(|r| r[1] as usize).sum();
        let mut bin_cap = (n_segs * NBIN * NBIN / 4).clamp(1 << 16, 1 << 24);
        let (b_list, b_keys, b_vals) = loop {
            let (keys, vals) = lake_table(&cache.lattice_lakes);
            let b_keys = storage(d, "lake keys", &keys);
            let b_vals = storage(d, "lake levels", &vals);
            let b_list = output(d, "bin list", (bin_cap * 4) as u64);
            let g1 = bind(d, &self.k.l_tables, &[b_segs, &empty, &b_sinks, &b_keys, &b_vals, &empty, &empty, &empty, &empty, &empty, &empty, &empty]);
            let g2 = group2(&b_list);
            run_passes(
                &g1,
                &g2,
                &[
                    (&self.k.nodes, [((NG * NG) as u32).div_ceil(64), nt as u32, 1]),
                    (&self.k.a1, [wg(NA2), wg(NA2), nt as u32]),
                    (&self.k.bins, [(NBIN * NBIN) as u32, nt as u32, 1]),
                ],
                true,
            );
            let counters: Vec<u32> = read_back(&self.gpu, &b_counters, 8)?;
            if counters[1] != 0 {
                bin_cap = counters[0] as usize + 1024;
                continue;
            }
            let nreq = counters[2] as usize;
            if nreq == 0 {
                break (b_list, b_keys, b_vals);
            }
            let reqs: Vec<GSiteReq> = read_back(&self.gpu, &b_lake_req, nreq.min(lake_cap))?;
            self.settle(&mut cache, |prep| {
                for r in &reqs {
                    prep.lattice_lake_level(r.id, DVec3::new(r.pt[0], r.pt[1], r.pt[2]));
                }
            })?;
        };
        stamp("relief, lakes, bins");
        // ---- the rest of pass A (the bins' pieces as the piece list), the sites pass B needs
        let g1 = bind(d, &self.k.l_tables, &[b_segs, &b_list, &b_sinks, &b_keys, &b_vals, &empty, &empty, &empty, &empty, &empty, &empty, &empty]);
        let g2 = group2(&unused);
        let mut passes = vec![(&self.k.a2, [wg(NA2), wg(NA2), nt as u32])];
        if full {
            passes.push((&self.k.region_req, [wg(NA), wg(NA), nt as u32]));
            passes.push((&self.k.town_req, [wg(NA), wg(NA), nt as u32]));
        }
        run_passes(&g1, &g2, &passes, true);
        if !full {
            let all: Vec<GTerrain> = read_back(&self.gpu, &b_terr, nt * NA2 * NA2)?;
            return Ok((all.chunks(NA2 * NA2).map(|c| c.to_vec()).collect(), vec![]));
        }
        let counters: Vec<u32> = read_back(&self.gpu, &b_counters, 8)?;
        stamp("pass a");
        let (nreg, ntown) = (counters[3] as usize, counters[4] as usize);
        if nreg > site_cap || ntown > site_cap {
            return Err(Overflow(format!("{nreg} regions, {ntown} town cells")).into());
        }
        let reg_reqs: Vec<GSiteReq> = read_back(&self.gpu, &b_region_req, nreg)?;
        let town_reqs: Vec<[i32; 4]> = read_back(&self.gpu, &b_town_req, ntown)?;
        let mut reg_ids: Vec<(u64, DVec3)> = reg_reqs.iter().map(|r| (r.id, DVec3::new(r.pt[0], r.pt[1], r.pt[2]))).collect();
        reg_ids.sort_by_key(|r| r.0);
        reg_ids.dedup_by_key(|r| r.0);
        let mut cells: Vec<host::Cell> = town_reqs.iter().map(|c| (c[0] as i64, c[1] as i64, c[2] as i64)).collect();
        cells.sort_unstable();
        cells.dedup();
        let (regions, town_cells) = self.settle(&mut cache, |prep| {
            let regions: Vec<(u64, GRegion)> = reg_ids.iter().filter_map(|&(id, pt)| prep.region(id, pt).map(|r| (id, r))).collect();
            let towns: Vec<(host::Cell, Vec<(u64, GTown)>)> = cells.iter().filter_map(|&c| prep.town_candidates(c).map(|v| (c, v))).collect();
            (regions, towns)
        })?;
        stamp(&format!("{} regions, {} town cells", regions.len(), town_cells.len()));
        let (rk, ri, rv) = region_table(&regions);
        let (tk, tc, tl, tv) = town_table(&town_cells);
        let b_rk = storage(d, "region keys", &rk);
        let b_ri = storage(d, "region index", &ri);
        let b_rv = storage(d, "regions", &rv);
        let b_tk = storage(d, "town keys", &tk);
        let b_tc = storage(d, "town cells", &tc);
        let b_tl = storage(d, "town list", &tl);
        let b_tv = storage(d, "towns", &tv);
        // ---- pass B, canopy opening, outputs
        let g1 = bind(d, &self.k.l_tables, &[b_segs, &b_list, &b_sinks, &b_keys, &b_vals, &b_rk, &b_ri, &b_rv, &b_tk, &b_tc, &b_tl, &b_tv]);
        let g2 = group2(&unused);
        let grid = [wg(NA), wg(NA), nt as u32];
        run_passes(
            &g1,
            &g2,
            &[
                (&self.k.pass_b, grid),
                (&self.k.open_min_x, grid),
                (&self.k.open_min_y, grid),
                (&self.k.open_max_x, grid),
                (&self.k.open_max_y, grid),
                (&self.k.open_apply, grid),
                (&self.k.finish, [wg(N), wg(N), nt as u32]),
            ],
            true,
        );
        let counters: Vec<u32> = read_back(&self.gpu, &b_counters, 8)?;
        if counters[5] != 0 {
            bail!("GPU generator: pass B lacked land-use data (flags {:#x})", counters[5]);
        }
        stamp("pass b");
        let out: Vec<u32> = read_back(&self.gpu, &b_out, nt * N * N * OUT_U)?;
        let ranges: Vec<u32> = read_back(&self.gpu, &b_ranges, nt * 2)?;
        let tiles = ids
            .iter()
            .enumerate()
            .map(|(t, &id)| {
                let o = &out[t * N * N * OUT_U..(t + 1) * N * N * OUT_U];
                let n = N * N;
                let mut td = TileData {
                    id,
                    rgb: vec![0; n * 3],
                    albedo: vec![0; n * 3],
                    elevation: vec![0.0; n],
                    normal: vec![0; n * 3],
                    landcover: vec![0; n],
                    emission: vec![0; n * 3],
                    elev_min: from_orderable(!ranges[2 * t]),
                    elev_max: from_orderable(ranges[2 * t + 1]),
                };
                for k in 0..n {
                    let w = &o[k * OUT_U..(k + 1) * OUT_U];
                    for ch in 0..3 {
                        td.rgb[3 * k + ch] = (w[0] >> (8 * ch)) as u8;
                        td.albedo[3 * k + ch] = (w[1] >> (8 * ch)) as u8;
                        td.emission[3 * k + ch] = (w[2] >> (8 * ch)) as u8;
                        td.normal[3 * k + ch] = (w[3] >> (8 * ch)) as u8 as i8;
                    }
                    td.elevation[k] = f32::from_bits(w[4]);
                    td.landcover[k] = w[5] as u8;
                }
                td
            })
            .collect();
        Ok((vec![], tiles))
    }
}

/// More land-use sites in a batch than the request buffers hold.
#[derive(Debug)]
struct Overflow(String);

impl std::fmt::Display for Overflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU generator: too many land-use sites in one batch ({})", self.0)
    }
}

impl std::error::Error for Overflow {}

fn from_orderable(u: u32) -> f32 {
    if u & 0x8000_0000 != 0 {
        f32::from_bits(u & 0x7fff_ffff)
    } else {
        f32::from_bits(!u)
    }
}

/// Land-use regions as an open-addressing table: keys, index into the region list, regions.
fn region_table(regions: &[(u64, GRegion)]) -> (Vec<u64>, Vec<u32>, Vec<GRegion>) {
    let n = (regions.len() * 2).next_power_of_two().max(16);
    let mut keys = vec![0u64; n];
    let mut idx = vec![0u32; n];
    for (i, (id, _)) in regions.iter().enumerate() {
        let mut k = (crate::noise::mix64(*id) % n as u64) as usize;
        while keys[k] != 0 {
            k = (k + 1) % n;
        }
        keys[k] = *id;
        idx[k] = i as u32;
    }
    (keys, idx, regions.iter().map(|r| r.1).collect())
}

/// Town lattice cells as an open-addressing table: keys, (first, count) into the candidate
/// list, the candidate list (indices into the towns), the towns.
fn town_table(cells: &[(host::Cell, Vec<(u64, GTown)>)]) -> (Vec<u64>, Vec<[u32; 2]>, Vec<u32>, Vec<GTown>) {
    let n = (cells.len() * 2).next_power_of_two().max(16);
    let mut keys = vec![0u64; n];
    let mut vals = vec![[0u32; 2]; n];
    let mut list = Vec::new();
    let mut towns = Vec::new();
    let mut index: crate::noise::FxHashMap<u64, u32> = Default::default();
    for (c, cands) in cells {
        let key = crate::noise::hash3(0x7C311, c.0, c.1, c.2) | 1;
        let first = list.len() as u32;
        for (id, t) in cands {
            let ti = *index.entry(*id).or_insert_with(|| {
                towns.push(*t);
                (towns.len() - 1) as u32
            });
            list.push(ti);
        }
        let mut k = (crate::noise::mix64(key) % n as u64) as usize;
        while keys[k] != 0 {
            k = (k + 1) % n;
        }
        keys[k] = key;
        vals[k] = [first, cands.len() as u32];
    }
    (keys, vals, list, towns)
}

/// A stand-in buffer for an unused binding.
fn b_dummy(d: &wgpu::Device) -> wgpu::Buffer {
    output(d, "-", 256)
}

/// A box of drainage lattice cells for a query, with the shell of possibly active cells
/// (`World::shell_cells`).
#[allow(clippy::too_many_arguments)]
fn dbox(ell: &geodesy::Ellipsoid, center: DVec3, lo: DVec3, hi: DVec3, cell: f64, reach: f64, lvl: usize, query: usize, radius: f64) -> GBox {
    let m = 0.5 * cell + 200.0;
    let half = ((hi - lo) + DVec3::ONE).max_element() * cell * 0.5 * 3f64.sqrt();
    let r = center.length();
    let sz = center.z / r.max(1.0);
    let (a, b) = (ell.a, ell.b);
    let r_gc = a * b / ((b * b * (1.0 - sz * sz)) + (a * a * sz * sz)).sqrt();
    let dr = 2.0 * (a - b) * (half / b) + 100.0;
    let (r_lo, r_hi) = (r_gc - dr - m, r_gc + dr + m);
    let c = center.to_array();
    let ax = (0..3).max_by(|&i, &j| c[i].abs().total_cmp(&c[j].abs())).unwrap();
    GBox {
        center: [center.x, center.y, center.z, 0.0],
        r_lo2: r_lo * r_lo,
        r_hi2: r_hi * r_hi,
        cell,
        reach,
        lo: [lo.x as i32, lo.y as i32, lo.z as i32, 0],
        dims: [(hi.x - lo.x) as u32 + 1, (hi.y - lo.y) as u32 + 1, (hi.z - lo.z) as u32 + 1, 0],
        level: lvl as u32,
        ax: ax as u32,
        sign: if c[ax].signum() >= 0.0 { 1 } else { -1 },
        query: query as u32,
        chunk0: 0,
        radius: radius as f32,
        _p: [0; 2],
    }
}

/// The compiled pipelines kept on disk (`~/.cache/terrain/pipelines-<adapter>.bin`): the
/// driver's own shader cache is per executable, and compiling the generator takes minutes.
struct PipelineCache {
    cache: wgpu::PipelineCache,
    path: std::path::PathBuf,
}

impl PipelineCache {
    fn open(gpu: &Gpu) -> Option<PipelineCache> {
        if !gpu.device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return None;
        }
        let key = wgpu::util::pipeline_cache_key(&gpu.info)?;
        let dir = std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".cache")))?
            .join("terrain");
        let path = dir.join(format!("pipelines-{key}.bin"));
        let data = std::fs::read(&path).ok();
        // SAFETY: the data is what `get_data` returned for this adapter key (or nothing); wgpu and
        // the driver validate it and fall back to an empty cache
        let cache =
            unsafe { gpu.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("terrain"), data: data.as_deref(), fallback: true }) };
        Some(PipelineCache { cache, path })
    }

    fn save(&self) {
        let Some(data) = self.cache.get_data() else { return };
        if std::fs::read(&self.path).is_ok_and(|old| old == data) {
            return;
        }
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, &data).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }
}

type FxLakes = crate::noise::FxHashMap<u64, Option<f64>>;

/// Sink pieces of a drainage query or `MODE_REPORT` point: (end point, half width), in order.
type SinkPieces = Vec<(DVec3, f64)>;

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
    g.ell_a = w.ell.a;
    g.ell_b = w.ell.b;
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
