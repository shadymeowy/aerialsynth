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
use tilestore::TileData;
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
const TILE_B_WGSL: &str = include_str!("wgsl/tile_b.wgsl");

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
    let w = World::new(Config::default());
    let (_, pal) = tables::palette(&SurfaceModel::new(&w).pal);
    let points = format!("{consts}{NOISE_WGSL}{WORLD_WGSL}{POINTS_WGSL}");
    let tile = format!("{consts}{pal}{NOISE_WGSL}{WORLD_WGSL}{TILE_A_WGSL}{SURFACE_WGSL}{TILE_B_WGSL}");
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
        let (pal, _) = tables::palette(&surface.pal);
        let g_pal = storage(d, "palette", &pal);
        let l_globals = layout(d, "globals", &[Uniform, Ro, Ro, Ro, Ro]);
        let l_drain = layout(d, "drain", &[Ro, Ro, Ro, Ro, Ro]);
        let l_tables = layout(d, "tables", &[Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro, Ro]);
        let l_points = layout(d, "points", &[Ro, Rw]);
        let l_tile = layout(d, "tile", &[Ro, Ro, Ro, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw, Rw]);
        let globals = bind(d, &l_globals, &[&g_cfg, &g_grads, &g_octs, &g_fbms, &g_pal]);
        let (src_points, src_tile) = sources();
        let t_compile = std::time::Instant::now();
        let cache = PipelineCache::open(&gpu);
        let m_points = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("points"), source: wgpu::ShaderSource::Wgsl(src_points.into()) });
        let m_tile = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("tile"), source: wgpu::ShaderSource::Wgsl(src_tile.into()) });
        let pl_points = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("points"), bind_group_layouts: &[Some(&l_globals), Some(&l_drain), Some(&l_points)], immediate_size: 0 });
        let pl_tile = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("tile"), bind_group_layouts: &[Some(&l_globals), Some(&l_tables), Some(&l_tile)], immediate_size: 0 });
        let pipe = |module: &wgpu::ShaderModule, pl: &wgpu::PipelineLayout, entry: &str| {
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(pl), module, entry_point: Some(entry), compilation_options: Default::default(), cache: cache.as_ref().map(|c| &c.cache) })
        };
        // (the driver compiles each pipeline on its own: in parallel)
        let names = ["grid_nodes", "pass_a1", "bin_segments", "pass_a2", "region_requests", "town_requests", "pass_b", "open_min_x", "open_min_y", "open_max_x", "open_max_y", "open_apply", "finish"];
        let (points, mut tile): (wgpu::ComputePipeline, Vec<wgpu::ComputePipeline>) = std::thread::scope(|s| {
            let hs: Vec<_> = names.iter().map(|n| s.spawn(|| pipe(&m_tile, &pl_tile, n))).collect();
            let points = pipe(&m_points, &pl_points, "eval_points");
            (points, hs.into_iter().map(|h| h.join().expect("pipeline")).collect())
        });
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
            let groups = (pts.len() as u32).div_ceil(64);
            let gx = groups.min(65535);
            cp.dispatch_workgroups(gx, groups.div_ceil(gx), 1);
        }
        self.gpu.queue.submit([enc.finish()]);
        read_back(&self.gpu, &b_out, pts.len())
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
            let (out, mut need, heights) = {
                let mut prep = Prep::new(&self.world, cache);
                let out = f(&mut prep);
                prep.prepare();
                if prof {
                    eprintln!("  settle round {round}: host {:.3} s, {} missing, {} requests, {} heights", t0.elapsed().as_secs_f64(), prep.missing, prep.need.len(), prep.heights_need.len());
                }
                if prep.missing == 0 {
                    return Ok(out);
                }
                (out, std::mem::take(&mut prep.need), std::mem::take(&mut prep.heights_need))
            };
            drop(out);
            if need.is_empty() && heights.is_empty() {
                bail!("GPU generator: host preparation is stuck");
            }
            let heights: Vec<((usize, host::Cell), Ctx)> = heights.into_iter().collect();
            let mut reqs: Vec<PointReq> = need.drain().map(|(_, r)| r).collect();
            let n_points = reqs.len();
            reqs.extend(heights.iter().map(|(_, ctx)| PointReq { ctx: *ctx, mode: MODE_RELIEF, segs: vec![], sinks: vec![] }));
            let res = self.eval_points(&reqs, &cache.lattice_lakes)?;
            for (r, t) in reqs[..n_points].iter().zip(&res) {
                cache.points.insert(PointKey::new(r.mode, &r.ctx), *t);
            }
            for (((lvl, c), ctx), t) in heights.iter().zip(&res[n_points..]) {
                cache.set_height(*lvl, *c, ctx, t.ground as f64);
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
        let out = self.settle(&mut cache, |p| ctxs.iter().map(|c| p.point(MODE_FULL, c)).collect::<Vec<_>>())?;
        Ok(out.into_iter().map(|t| t.expect("settled")).collect())
    }

    /// Pass A of tiles: the terrain per pass-A pixel centre (tile + 2-pixel apron, 260 x 260).
    pub fn pass_a(&self, ids: &[TileId]) -> Result<Vec<Vec<GTerrain>>> {
        Ok(self.run(ids, false)?.0)
    }

    /// Generate tiles (one batch on the GPU; halved while its land-use sites overflow the
    /// request buffers).
    pub fn tiles(&self, ids: &[TileId]) -> Result<Vec<TileData>> {
        match self.run(ids, true) {
            Ok(r) => Ok(r.1),
            Err(e) if e.is::<Overflow>() && ids.len() > 1 => {
                let (a, b) = ids.split_at(ids.len() / 2);
                let mut out = self.tiles(a)?;
                out.extend(self.tiles(b)?);
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
                eprintln!("  batch of {nt} (z{}..{}): {what} at {:.3} s", ids.iter().map(|i| i.z).min().unwrap_or(0), ids.iter().map(|i| i.z).max().unwrap_or(0), t_run.elapsed().as_secs_f64());
            }
        };
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
        stamp("drainage");
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
            segs.extend(drains[t].segs.iter().map(GSeg::from));
            sinks.extend_from_slice(&drains[t].sinks);
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
        let b_segs = storage(d, "segs", &segs);
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
                &[&b_tiles, &b_rows, &b_cols, &b_node_f, &b_node_ids, &b_node_pts, &b_pix, &b_terr, &b_bins, list, &b_counters, &b_lake_req, &b_pixb, &b_scr_a, &b_scr_b, &b_out, &b_ranges, &b_region_req, &b_town_req],
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
        let mut bin_cap = (segs.len() * NBIN * NBIN / 4).clamp(1 << 16, 1 << 24);
        let (b_list, b_keys, b_vals) = loop {
            let (keys, vals) = lake_table(&cache.lattice_lakes);
            let b_keys = storage(d, "lake keys", &keys);
            let b_vals = storage(d, "lake levels", &vals);
            let b_list = output(d, "bin list", (bin_cap * 4) as u64);
            let g1 = bind(d, &self.k.l_tables, &[&b_segs, &empty, &b_sinks, &b_keys, &b_vals, &empty, &empty, &empty, &empty, &empty, &empty, &empty]);
            let g2 = group2(&b_list);
            run_passes(
                &g1,
                &g2,
                &[(&self.k.nodes, [((NG * NG) as u32).div_ceil(64), nt as u32, 1]), (&self.k.a1, [wg(NA2), wg(NA2), nt as u32]), (&self.k.bins, [(NBIN * NBIN) as u32, nt as u32, 1])],
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
        let g1 = bind(d, &self.k.l_tables, &[&b_segs, &b_list, &b_sinks, &b_keys, &b_vals, &empty, &empty, &empty, &empty, &empty, &empty, &empty]);
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
        let g1 = bind(d, &self.k.l_tables, &[&b_segs, &b_list, &b_sinks, &b_keys, &b_vals, &b_rk, &b_ri, &b_rv, &b_tk, &b_tc, &b_tl, &b_tv]);
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
        let dir = std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".cache")))?.join("terrain");
        let path = dir.join(format!("pipelines-{key}.bin"));
        let data = std::fs::read(&path).ok();
        // SAFETY: the data is what `get_data` returned for this adapter key (or nothing); wgpu and
        // the driver validate it and fall back to an empty cache
        let cache = unsafe { gpu.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("terrain"), data: data.as_deref(), fallback: true }) };
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
