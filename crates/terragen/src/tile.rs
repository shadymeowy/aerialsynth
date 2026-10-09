//! Tile generation: combines pass A (per pixel centre, with a 2-pixel apron) and pass B
//! (supersampled detail, 1-pixel apron), then derives normals and the baked "satellite" rgb.
//! Pass B interpolates pass A between pixel centres, so its apron pixels need pass-A values one
//! pixel further out; the normals of the edge pixels use pass B's apron.

use crate::config::Config;
use crate::surface::{l2s, Caches, Local, PixFields, Surface, SurfaceModel};
use crate::world::{water, Ctx, Macro, NearSegs, Pre, Terrain, World};
use geodesy::tiles::{gsd_ew, gsd_ns, pixel_to_latlon, TileId};
use glam::{DVec2, DVec3};
use rayon::prelude::*;

pub use tilestore::TILE_SIZE;

pub use tilestore::TileData;

pub struct Generator {
    pub world: World,
    pub surface: SurfaceModel,
    backend: Backend,
    /// the GPU generator, compiled on first use (Err: unavailable)
    #[cfg(feature = "gpu")]
    gpu: std::sync::OnceLock<Result<crate::gpu::GpuGenerator, String>>,
}

/// Where tiles are generated. The GPU generator (`terragen::gpu`) builds the same world as the
/// CPU one: tiles agree to f32 precision (a few pixels in 10,000 differ, by a few DN or cm).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// the GPU when it has 64-bit float and integer shaders, else the CPU
    #[default]
    Auto,
    Gpu,
    Cpu,
}

/// Pass A at a point ([`Generator::terrain_points`]).
#[derive(Clone, Copy, Debug)]
pub struct PointTerrain {
    /// bare ground (m above the ellipsoid; the bed under water)
    pub ground: f64,
    /// standing water level, or -inf
    pub water: f64,
    /// `world::water` kind
    pub water_kind: u8,
}

impl PointTerrain {
    /// The surface: ground or water, whichever is higher.
    pub fn surface(&self) -> f64 {
        if self.water_kind != 0 {
            self.water.max(self.ground)
        } else {
            self.ground
        }
    }
}

/// Tiles per GPU batch (~25 MB of GPU memory each).
#[cfg(feature = "gpu")]
const GPU_BATCH: usize = 16;

/// A node of the tile's coarse grid.
struct Node {
    m: Macro,
    pre: Pre,
    /// long octaves of the pixel fields
    pf_low: [f64; PixFields::N],
    /// `pre` (with the mountain warp) flattened for interpolation
    flat: [f64; NPRE],
    /// forest stand
    stand: u64,
}

/// Grid-interpolated inputs of pass A, flattened: mountain warp, gully gradient, roads, relief
/// octaves, meander warps, region warp, gully octaves, floodplain-edge noise.
const NPRE: usize = 34;

fn pack_pre(m: &Macro, p: &Pre) -> [f64; NPRE] {
    let mut f = [0.0; NPRE];
    f[0..2].copy_from_slice(&m.mtn_warp);
    f[2..4].copy_from_slice(&p.gully.unwrap_or_default());
    f[4..7].copy_from_slice(&p.road_major.unwrap_or_default());
    f[7..10].copy_from_slice(&p.road_minor.unwrap_or_default());
    f[10..15].copy_from_slice(&p.relief.unwrap_or_default());
    for li in 0..4 {
        f[15 + 2 * li..17 + 2 * li].copy_from_slice(&p.river_warp[li].unwrap_or_default());
    }
    f[23..26].copy_from_slice(&p.region_warp.unwrap_or_default());
    f[26..30].copy_from_slice(&p.gully_oct.unwrap_or_default());
    for li in 0..4 {
        f[30 + li] = p.floodplain[li].unwrap_or_default();
    }
    f
}

/// The interpolated `f` with the fields present in `like` (every node has the same).
fn unpack_pre(f: &[f64; NPRE], like: &Pre) -> ([f64; 2], Pre) {
    let arr = |a: usize| -> [f64; 3] { [f[a], f[a + 1], f[a + 2]] };
    let mut p = Pre { relief_cut: like.relief_cut, ..Pre::default() };
    p.gully = like.gully.map(|_| [f[2], f[3]]);
    p.road_major = like.road_major.map(|_| arr(4));
    p.road_minor = like.road_minor.map(|_| arr(7));
    p.relief = like.relief.map(|_| [f[10], f[11], f[12], f[13], f[14]]);
    for li in 0..4 {
        p.river_warp[li] = like.river_warp[li].map(|_| [f[15 + 2 * li], f[16 + 2 * li]]);
    }
    p.region_warp = like.region_warp.map(|_| arr(23));
    p.gully_oct = like.gully_oct.map(|_| [f[26], f[27], f[28], f[29]]);
    for li in 0..4 {
        p.floodplain[li] = like.floodplain[li].map(|_| f[30 + li]);
    }
    ([f[0], f[1]], p)
}

/// Catmull-Rom weights of four equally spaced samples at t in [0, 1] between the middle two.
#[inline]
fn catmull_rom_weights(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2)]
}

#[inline]
fn bilerp(v: [f64; 4], fx: f64, fy: f64) -> f64 {
    let a = v[0] + (v[1] - v[0]) * fx;
    let b = v[2] + (v[3] - v[2]) * fx;
    a + (b - a) * fy
}

/// Pass A of a tile and the coarse grid (for pass B).
struct PassA {
    grid: Vec<Terrain>,
    nodes: Vec<Node>,
    use_grid: bool,
    gk0x: i64,
    gk0y: i64,
    ng: usize,
    pf_cut: f64,
}

impl Generator {
    /// A generator on the GPU when there is one (`Backend::Auto`). The config is not checked:
    /// use [`Generator::try_new`] for configs from users.
    pub fn new(cfg: Config) -> Self {
        Self::with_backend(cfg, Backend::Auto)
    }

    /// A generator on `backend` (the GPU generator is compiled on first use).
    pub fn with_backend(cfg: Config, backend: Backend) -> Self {
        let world = World::new(cfg);
        let surface = SurfaceModel::new(&world);
        Generator {
            world,
            surface,
            backend,
            #[cfg(feature = "gpu")]
            gpu: std::sync::OnceLock::new(),
        }
    }

    /// A generator on the GPU when there is one, for a validated config ([`Config::validate`]).
    pub fn try_new(cfg: Config) -> anyhow::Result<Self> {
        Self::try_with_backend(cfg, Backend::Auto)
    }

    /// A generator on `backend` for a validated config ([`Config::validate`]); `Backend::Gpu`
    /// fails without a suitable GPU.
    pub fn try_with_backend(cfg: Config, backend: Backend) -> anyhow::Result<Self> {
        cfg.validate()?;
        if backend == Backend::Gpu {
            #[cfg(feature = "gpu")]
            crate::gpu::shared()?.check_generator()?;
            #[cfg(not(feature = "gpu"))]
            anyhow::bail!("this build has no GPU tile generator (feature `gpu`)");
        }
        Ok(Self::with_backend(cfg, backend))
    }

    /// The GPU generator, if tiles are generated on the GPU.
    #[cfg(feature = "gpu")]
    pub fn gpu(&self) -> Option<&crate::gpu::GpuGenerator> {
        if self.backend == Backend::Cpu {
            return None;
        }
        if self.backend == Backend::Auto && !crate::gpu::shared().is_ok_and(|g| g.check_generator().is_ok()) {
            return None;
        }
        match self.gpu.get_or_init(|| crate::gpu::GpuGenerator::new(self.world.cfg.clone()).map_err(|e| format!("{e:#}"))) {
            Ok(g) => Some(g),
            Err(e) => {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| eprintln!("warning: GPU tile generator unavailable ({e}); generating on the CPU"));
                None
            }
        }
    }

    /// `"GPU (<adapter>)"` or `"CPU"`: where tiles are generated.
    pub fn backend_name(&self) -> String {
        #[cfg(feature = "gpu")]
        if let Some(g) = self.gpu() {
            return format!("GPU ({})", g.adapter());
        }
        "CPU".into()
    }

    /// Generate tiles: on the GPU in batches, or on the CPU in parallel.
    pub fn tiles(&self, ids: &[TileId]) -> anyhow::Result<Vec<TileData>> {
        #[cfg(feature = "gpu")]
        if let Some(g) = self.gpu() {
            let mut out = Vec::with_capacity(ids.len());
            for chunk in ids.chunks(GPU_BATCH) {
                out.extend(g.tiles(chunk)?);
            }
            return Ok(out);
        }
        Ok(ids.par_iter().map(|&id| self.tile_cpu(id)).collect())
    }

    /// Pass A at points (lat, lon in radians, pixel size in metres): the bare ground and the
    /// standing water (on the GPU when tiles are generated there).
    pub fn terrain_points(&self, pts: &[(f64, f64, f64)]) -> anyhow::Result<Vec<PointTerrain>> {
        #[cfg(feature = "gpu")]
        if let Some(g) = self.gpu() {
            return Ok(g
                .terrain_points(pts)?
                .iter()
                .map(|t| PointTerrain {
                    ground: t.ground as f64,
                    water: if t.water_kind != 0 { t.water as f64 } else { f64::NEG_INFINITY },
                    water_kind: t.water_kind as u8,
                })
                .collect());
        }
        let ell = self.world.ell;
        Ok(pts
            .par_iter()
            .map(|&(lat, lon, gsd)| {
                let t = self.world.terrain(&Ctx::new(lat, lon, gsd, &ell));
                PointTerrain { ground: t.ground, water: t.water, water_kind: t.water_kind }
            })
            .collect())
    }

    /// Generate one tile (see [`Generator::tiles`]; panics if the GPU fails).
    pub fn tile(&self, id: TileId) -> TileData {
        self.tiles(&[id]).expect("tile generation").remove(0)
    }

    pub fn config(&self) -> &Config {
        &self.world.cfg
    }

    /// Evaluate pass A + B at a single geodetic point (radians) for a given pixel size.
    /// Useful for planning/inspection. Returns (pass A terrain, DSM height, landcover).
    pub fn probe(&self, lat: f64, lon: f64, gsd: f64) -> (Terrain, f64, u8) {
        let ell = self.world.ell;
        let ctx = Ctx::new(lat, lon, gsd, &ell);
        let t = self.world.terrain(&ctx);
        let local = Local {
            t: &t,
            ground: t.ground,
            water: t.water,
            water_kind: t.water_kind,
            river_d: t.river_d.min(1e7),
            river_hw: t.river_hw,
            river_level: t.river_level,
            road_major: t.road_major.min(1e7),
            road_minor: t.road_minor.min(1e7),
            slope: 0.0,
            fw: gsd,
        };
        let mut caches = Caches::default();
        let pf = self.surface.pixel_fields(ctx.p, gsd);
        let s = self.surface.eval(&self.world, &mut caches, &ctx, &local, &pf);
        (t, s.height, s.class)
    }

    /// Pass A of a tile: the terrain at the pixel centres of the tile and a 2-pixel apron
    /// (260 x 260, row-major).
    pub fn pass_a(&self, id: TileId) -> Vec<Terrain> {
        self.pass_a_impl(id).grid
    }

    fn pass_a_impl(&self, id: TileId) -> PassA {
        let n = TILE_SIZE;
        let na2 = n + 4; // pass A, 2-px apron
        let z = id.z;
        let ell = self.world.ell;
        let ox = id.x as f64 * n as f64;
        let oy = id.y as f64 * n as f64;
        // ---------------- macro fields: on a coarse grid aligned to global multiples of 16 px
        // (shared by neighbouring tiles → seamless), or exactly at low zooms where the grid would
        // be too coarse for the macro wavelengths.
        const G: f64 = 16.0;
        // The grid / exact choices below depend on the zoom only: the node spacing at
        // GRID_REF_LAT. (Taken at each tile's centre latitude, the north-south neighbours at the
        // switch latitudes took different paths for adjacent pixels: seams of up to 17 m.) At
        // 40° the choices equal the old per-tile ones for tile centres between 35.2° and 49.1°;
        // towards the equator the grid spacing is up to 1.3× the limits below.
        const GRID_REF_LAT: f64 = 40.0;
        let g_m = G * gsd_ew(GRID_REF_LAT.to_radians(), z, n as u32, &ell);
        let use_grid = g_m <= 2000.0;
        // two nodes beyond the 2-px apron on each side: bicubic needs a 4x4 neighbourhood
        let gk0x = (ox / G) as i64 - 2;
        let gk0y = (oy / G) as i64 - 2;
        let ng = (n as f64 / G) as usize + 5;
        // Smooth per-pixel inputs taken from the grid (bicubic) where its spacing is a small
        // fraction of their shortest wavelength, else evaluated exactly per pixel: the mountain
        // domain warp (>= 7.5 km), the low-passed relief gradient for the gullies (>= 700 m) and
        // the road networks (band-limited at >= 200 m).
        let warp_on_grid = use_grid && g_m <= 1000.0;
        let gully_on_grid = use_grid && g_m <= 100.0;
        let roads_on_grid = use_grid && g_m <= 400.0;
        // the octaves of the pixel fields (surface) with wavelengths >= 8 node spacings come from
        // the grid too, the rest per pixel; the cut depends on the zoom only (node spacing at the
        // equator), so neighbouring tiles agree
        let pf_cut = 8.0 * G * gsd_ew(0.0, z, n as u32, &ell);
        // the long octaves of the relief likewise (ridged: 16 spacings, its creases need more)
        let relief_cut = if use_grid { Some([2.0 * pf_cut, pf_cut]) } else { None };
        let nodes: Vec<Node> = if use_grid {
            (0..ng * ng)
                .into_par_iter()
                .map(|k| {
                    let gx = (gk0x + (k % ng) as i64) as f64 * G;
                    let gy = (gk0y + (k / ng) as i64) as f64 * G;
                    let (lat, lon) = pixel_to_latlon(DVec2::new(gx, gy), z, n as u32);
                    let gsd = gsd_ew(lat, z, n as u32, &ell);
                    let ctx = Ctx::new(lat, lon, gsd, &ell);
                    let m = self.world.macro_at(ctx.p, gsd);
                    let pre = self.world.pre_at(&ctx, &m, gully_on_grid, roads_on_grid, relief_cut);
                    let pf_low = self.surface.pixel_fields_part(ctx.p, gsd, Some((pf_cut, true)), true);
                    let flat = pack_pre(&m, &pre);
                    let stand = self.surface.stand_id(ctx.p, None);
                    Node { m, pre, pf_low, flat, stand }
                })
                .collect()
        } else {
            vec![]
        };
        let macro_at = |px: f64, py: f64, ctx: &Ctx| -> Macro {
            if !use_grid {
                return self.world.macro_at(ctx.p, ctx.gsd);
            }
            let u = px / G - gk0x as f64;
            let v = py / G - gk0y as f64;
            let (i0, j0) = ((u.floor() as usize).clamp(1, ng - 3), (v.floor() as usize).clamp(1, ng - 3));
            let (fx, fy) = (u - i0 as f64, v - j0 as f64);
            let g = |i: usize, j: usize| &nodes[j * ng + i];
            let mut m = Macro::bilerp(&g(i0, j0).m, &g(i0 + 1, j0).m, &g(i0, j0 + 1).m, &g(i0 + 1, j0 + 1).m, fx, fy);
            // the smooth inputs: Catmull-Rom over the 4x4 nodes around the pixel
            let (wx, wy) = (catmull_rom_weights(fx), catmull_rom_weights(fy));
            let mut f = [0.0; NPRE];
            for (b, wyb) in wy.iter().enumerate() {
                for (a, wxa) in wx.iter().enumerate() {
                    let w = wxa * wyb;
                    for (fk, nk) in f.iter_mut().zip(&g(i0 + a - 1, j0 + b - 1).flat) {
                        *fk += w * nk;
                    }
                }
            }
            let (mtn_warp, mut pre) = unpack_pre(&f, &g(i0, j0).pre);
            // lattice sites: when the four nodes around the pixel have the same two nearest
            // sites, so has the pixel (the region of points with a given pair is convex)
            for k in 0..3 {
                let s0 = g(i0, j0).pre.sites[k];
                let same = |n: &Node| match (n.pre.sites[k], s0) {
                    (Some([a, b]), Some([c, d])) => (a.0 == c.0 && b.0 == d.0) || (a.0 == d.0 && b.0 == c.0),
                    _ => false,
                };
                if same(g(i0 + 1, j0)) && same(g(i0, j0 + 1)) && same(g(i0 + 1, j0 + 1)) {
                    pre.sites[k] = s0;
                }
            }
            m.mtn_warp = if warp_on_grid { mtn_warp } else { self.world.mtn_warp_at(ctx.p, ctx.gsd) };
            m.pre = Some(pre);
            m
        };

        // ---------------- drainage segments that can affect this tile
        let segs = {
            let (lat_c, lon_c) = pixel_to_latlon(DVec2::new(ox + 128.0, oy + 128.0), z, n as u32);
            let c = Ctx::new(lat_c, lon_c, 1.0, &ell);
            // the farthest apron corner (Mercator tiles are wider on their equator side)
            let radius = [(ox - 2.0, oy - 2.0), (ox + 258.0, oy - 2.0), (ox - 2.0, oy + 258.0), (ox + 258.0, oy + 258.0)]
                .iter()
                .map(|&(px, py)| {
                    let (la, lo) = pixel_to_latlon(DVec2::new(px, py), z, n as u32);
                    (Ctx::new(la, lo, 1.0, &ell).p - c.p).length()
                })
                .fold(0.0, f64::max);
            let gsd_c = gsd_ew(lat_c, z, n as u32, &ell);
            self.world.river_segments(c.p, radius, gsd_c)
        };
        let sinks = World::sink_lakes(&segs);

        // ---------------- pass A on pixel centres incl. 2px apron
        let rows_a: Vec<Vec<Terrain>> = (0..na2)
            .into_par_iter()
            .map(|j| {
                let mut row = Vec::with_capacity(na2);
                let py = oy + j as f64 - 2.0 + 0.5;
                let (lat, _) = pixel_to_latlon(DVec2::new(ox, py), z, n as u32);
                let gsd = gsd_ew(lat, z, n as u32, &ell);
                // channel pieces prefiltered per chunk of the row (coarse tiles hold tens of
                // thousands of pieces, most of them far from any given pixel); exact, see
                // `World::local_segments`
                // (smaller at low zooms, where a chunk spans many valleys)
                let chunk: usize = if z <= 12 { 16 } else { 32 };
                let pt = |i: f64| {
                    let px = ox + i - 2.0 + 0.5;
                    let (lat, lon) = pixel_to_latlon(DVec2::new(px, py), z, n as u32);
                    let ctx = Ctx::new(lat, lon, gsd, &ell);
                    let m = macro_at(px, py, &ctx);
                    (ctx, m)
                };
                // per chunk: centre, radius, bound of the ground before carving
                let chunks: Vec<(usize, usize, DVec3, f64, f64)> = (0..na2)
                    .step_by(chunk)
                    .map(|i0| {
                        let i1 = (i0 + chunk).min(na2);
                        let probes = [pt(i0 as f64), pt(0.5 * (i0 + i1 - 1) as f64), pt((i1 - 1) as f64)];
                        let center = probes[1].0.p;
                        let radius = (probes[0].0.p - center).length().max((probes[2].0.p - center).length()) + gsd;
                        let h_max = probes.iter().map(|(c, m)| self.world.relief_height(c, m)).fold(f64::MIN, f64::max) + 150.0 + 0.1 * 2.0 * radius;
                        (i0, i1, center, radius, h_max)
                    })
                    .collect();
                // the row's pieces first (a superset of every chunk's), then each chunk's
                let row_c = pt(0.5 * (na2 - 1) as f64).0.p;
                let row_r = chunks.iter().map(|c| (c.2 - row_c).length() + c.3).fold(0.0, f64::max);
                let row_h = chunks.iter().map(|c| c.4).fold(f64::MIN, f64::max);
                let row_segs = self.world.local_segments(&segs, row_c, row_r, row_h);
                for &(i0, i1, center, radius, h_max) in &chunks {
                    let local = self.world.local_segments(&row_segs, center, radius, h_max);
                    for i in i0..i1 {
                        let (ctx, m) = pt(i as f64);
                        let near = NearSegs { local: &local, h_max, sinks: &sinks };
                        row.push(self.world.terrain_with_near(&ctx, &m, &segs, &near));
                    }
                }
                row
            })
            .collect();
        let grid: Vec<Terrain> = rows_a.into_iter().flatten().collect();
        PassA { grid, nodes, use_grid, gk0x, gk0y, ng, pf_cut }
    }

    /// Generate one tile on the CPU (parallel over rows internally).
    pub fn tile_cpu(&self, id: TileId) -> TileData {
        let n = TILE_SIZE;
        let na = n + 2; // pass B, 1-px apron
        let na2 = n + 4; // pass A, 2-px apron
        let z = id.z;
        let ell = self.world.ell;
        let ss = self.world.cfg.tile_supersample.max(1) as usize;
        let adaptive = ss == 2 && self.world.cfg.tile_supersample_adaptive;
        let ox = id.x as f64 * n as f64;
        let oy = id.y as f64 * n as f64;
        const G: f64 = 16.0;

        let prof = std::env::var_os("TERRAGEN_PROFILE").is_some();
        let t_start = std::time::Instant::now();
        let PassA { grid, nodes, use_grid, gk0x, gk0y, ng, pf_cut } = self.pass_a_impl(id);
        // pass-A value at pass-B grid coordinates (i, j) ∈ [-1, na]
        let at = |i: isize, j: isize| -> &Terrain {
            let i = (i + 1).clamp(0, na2 as isize - 1) as usize;
            let j = (j + 1).clamp(0, na2 as isize - 1) as usize;
            &grid[j * na2 + i]
        };

        // per-row pixel size, east-west and north-south (mercator scale varies with latitude;
        // on the ellipsoid the N-S size is smaller by M/N)
        let row_lat: Vec<f64> = (0..na)
            .map(|j| {
                let py = oy + j as f64 - 1.0 + 0.5;
                pixel_to_latlon(DVec2::new(ox, py), z, n as u32).0
            })
            .collect();
        let row_gsd: Vec<f64> = row_lat.iter().map(|&lat| gsd_ew(lat, z, n as u32, &ell)).collect();
        let row_gsd_ns: Vec<f64> = row_lat.iter().map(|&lat| gsd_ns(lat, z, n as u32, &ell)).collect();

        // ---------------- slope of the bare ground at pixel scale
        let slope: Vec<f64> = (0..na * na)
            .map(|k| {
                let (i, j) = ((k % na) as isize, (k / na) as isize);
                let dx = (at(i + 1, j).ground - at(i - 1, j).ground) / (2.0 * row_gsd[j as usize]);
                let dy = (at(i, j + 1).ground - at(i, j - 1).ground) / (2.0 * row_gsd_ns[j as usize]);
                (dx * dx + dy * dy).sqrt()
            })
            .collect();

        let t_a = t_start.elapsed().as_secs_f64();
        // ---------------- pass B, supersampled, on the apron grid
        struct PixB {
            emission: DVec3,
            albedo: DVec3,
            height: f64,
            /// bare ground under the pixel (for the canopy clean-up below)
            ground: f64,
            lit: f64,
            class: u8,
        }
        let rows_b: Vec<Vec<PixB>> = (0..na)
            .into_par_iter()
            .map(|j| {
                // one cache per worker thread, kept across rows and tiles (towns and regions are
                // expensive to set up: a cache per row made town-rich tiles several times slower)
                thread_local! {
                    static CACHES: std::cell::RefCell<(u64, Caches)> = std::cell::RefCell::new((0, Caches::default()));
                }
                CACHES.with(|cc| {
                    let mut cc = cc.borrow_mut();
                    if cc.0 != self.world.cache_key {
                        *cc = (self.world.cache_key, Caches::default());
                    }
                    cc.1.trim();
                    let caches = &mut cc.1;
                    let mut row = Vec::with_capacity(na);
                    let gsd = row_gsd[j];
                    let fw = gsd / ss as f64;
                    for i in 0..na {
                        let pf = {
                            let px = ox + i as f64 - 1.0 + 0.5;
                            let py = oy + j as f64 - 1.0 + 0.5;
                            let (lat, lon) = pixel_to_latlon(DVec2::new(px, py), z, n as u32);
                            let p = Ctx::new(lat, lon, gsd, &ell).p;
                            if use_grid {
                                let mut f = self.surface.pixel_fields_part(p, gsd, Some((pf_cut, false)), false);
                                let u = px / G - gk0x as f64;
                                let v = py / G - gk0y as f64;
                                let (i0, j0) = ((u.floor() as usize).clamp(1, ng - 3), (v.floor() as usize).clamp(1, ng - 3));
                                let (wx, wy) = (catmull_rom_weights(u - i0 as f64), catmull_rom_weights(v - j0 as f64));
                                for (b, wyb) in wy.iter().enumerate() {
                                    for (a, wxa) in wx.iter().enumerate() {
                                        let w = wxa * wyb;
                                        let node = &nodes[(j0 + b - 1) * ng + i0 + a - 1].pf_low;
                                        for (fk, nk) in f.iter_mut().zip(node) {
                                            *fk += w * nk;
                                        }
                                    }
                                }
                                let mut pf = PixFields::from_parts(f, p, gsd, Some(pf_cut));
                                // the forest stand: known when the four nodes around agree
                                let st = nodes[j0 * ng + i0].stand;
                                if nodes[j0 * ng + i0 + 1].stand == st && nodes[(j0 + 1) * ng + i0].stand == st && nodes[(j0 + 1) * ng + i0 + 1].stand == st {
                                    pf.stand_id = Some(st);
                                }
                                pf
                            } else {
                                self.surface.pixel_fields(p, gsd)
                            }
                        };
                        // one sample at sub-pixel (sx, sy): the surface and the bare ground
                        let sample = |sx: usize, sy: usize, caches: &mut Caches| -> (Surface, f64) {
                            let fxo = (sx as f64 + 0.5) / ss as f64 - 0.5;
                            let fyo = (sy as f64 + 0.5) / ss as f64 - 0.5;
                            // neighbours for bilinear interpolation
                            let (ii, jj) = (i as isize, j as isize);
                            let i0 = if fxo < 0.0 { ii - 1 } else { ii };
                            let j0 = if fyo < 0.0 { jj - 1 } else { jj };
                            let fx = if fxo < 0.0 { 1.0 + fxo } else { fxo };
                            let fy = if fyo < 0.0 { 1.0 + fyo } else { fyo };
                            let nb = [at(i0, j0), at(i0 + 1, j0), at(i0, j0 + 1), at(i0 + 1, j0 + 1)];
                            let t = at(ii, jj);
                            let ground = bilerp(nb.map(|t| t.ground), fx, fy);
                            let mut wl = f64::NEG_INFINITY;
                            let mut wk = water::NONE;
                            for t in nb {
                                if t.water_kind != water::NONE && t.water > wl {
                                    wl = t.water;
                                    wk = t.water_kind;
                                }
                            }
                            let rd = bilerp(nb.map(|t| t.river_d.clamp(-1e6, 1e6)), fx, fy);
                            let rhw = nb.iter().map(|t| t.river_hw).fold(0.0, f64::max);
                            let rl = bilerp(nb.map(|t| if t.river_hw > 0.0 { t.river_level } else { ground }), fx, fy);
                            let road_major = bilerp(nb.map(|t| t.road_major.clamp(-1e6, 1e6)), fx, fy);
                            let road_minor = bilerp(nb.map(|t| t.road_minor.clamp(-1e6, 1e6)), fx, fy);
                            let mut tt = *t;
                            tt.region.edge = bilerp(nb.map(|t| t.region.edge.min(1e6)), fx, fy);
                            let local = Local {
                                t: &tt,
                                ground,
                                water: wl,
                                water_kind: wk,
                                river_d: rd,
                                river_hw: rhw,
                                river_level: rl,
                                road_major,
                                road_minor,
                                slope: slope[j * na + i],
                                fw,
                            };
                            let px = ox + i as f64 - 1.0 + 0.5 + fxo;
                            let py = oy + j as f64 - 1.0 + 0.5 + fyo;
                            let (lat, lon) = pixel_to_latlon(DVec2::new(px, py), z, n as u32);
                            let ctx = Ctx::new(lat, lon, gsd, &ell);
                            (self.surface.eval(&self.world, caches, &ctx, &local, &pf), ground)
                        };
                        let mut acc_a = DVec3::ZERO;
                        let mut acc_e = DVec3::ZERO;
                        let mut acc_h = 0.0;
                        let mut acc_l = 0.0;
                        let mut acc_g = 0.0;
                        let mut counts = [0u16; 32];
                        let mut add = |(s, ground): (Surface, f64)| {
                            acc_a += s.albedo;
                            acc_e += s.emission;
                            acc_h += s.height;
                            acc_l += s.lit;
                            acc_g += ground;
                            counts[(s.class as usize).min(31)] += 1;
                        };
                        let mut taken = ss * ss;
                        if adaptive {
                            // the diagonal pair first; the other two only where it disagrees
                            let s0 = sample(0, 0, caches);
                            let s1 = sample(1, 1, caches);
                            let similar = s0.0.class == s1.0.class
                                && (s0.0.albedo - s1.0.albedo).abs().max_element() < 0.012
                                && (s0.0.emission - s1.0.emission).abs().max_element() < 0.02
                                && (s0.0.height - s1.0.height).abs() < 0.15
                                && (s0.0.lit - s1.0.lit).abs() < 0.05;
                            add(s0);
                            add(s1);
                            if similar {
                                taken = 2;
                            } else {
                                add(sample(1, 0, caches));
                                add(sample(0, 1, caches));
                            }
                        } else {
                            for sy in 0..ss {
                                for sx in 0..ss {
                                    add(sample(sx, sy, caches));
                                }
                            }
                        }
                        let inv = 1.0 / taken as f64;
                        let class = counts.iter().enumerate().max_by_key(|(_, c)| **c).map(|(k, _)| k as u8).unwrap_or(0);
                        row.push(PixB { emission: acc_e * inv, albedo: acc_a * inv, height: acc_h * inv, ground: acc_g * inv, lit: acc_l * inv, class });
                    }
                    row
                })
            })
            .collect();
        let mut pb: Vec<PixB> = rows_b.into_iter().flatten().collect();

        // ---------------- canopy clean-up: morphological opening of the height above ground with
        // a ~1 m radius. Tree / land-use densities are evaluated per pixel, so wherever an input
        // mask changes sharply (slope stripes on gully walls, field edges) a crown can be clipped
        // into a sliver: a needle in the DSM. Whole crowns, hedges and buildings are wider than
        // the structuring element and survive.
        let r_open = (0.9 / row_gsd[na / 2]).floor() as usize;
        if r_open >= 1 {
            let r = r_open.min(4);
            let canopy: Vec<f64> = pb.iter().map(|p| (p.height - p.ground).max(0.0)).collect();
            let filt = |src: &[f64], max: bool| -> Vec<f64> {
                // separable square min / max filter over (2r+1)^2, clamped at the grid border
                let pick = |a: f64, b: f64| if max { a.max(b) } else { a.min(b) };
                let mut tmp = vec![0.0; na * na];
                for j in 0..na {
                    for i in 0..na {
                        let (lo, hi) = (i.saturating_sub(r), (i + r).min(na - 1));
                        tmp[j * na + i] = (lo..=hi).map(|k| src[j * na + k]).fold(if max { f64::MIN } else { f64::MAX }, pick);
                    }
                }
                let mut out = vec![0.0; na * na];
                for j in 0..na {
                    for i in 0..na {
                        let (lo, hi) = (j.saturating_sub(r), (j + r).min(na - 1));
                        out[j * na + i] = (lo..=hi).map(|k| tmp[k * na + i]).fold(if max { f64::MIN } else { f64::MAX }, pick);
                    }
                }
                out
            };
            let opened = filt(&filt(&canopy, false), true);
            for (k, p) in pb.iter_mut().enumerate() {
                if opened[k] < canopy[k] - 0.5 {
                    p.height = p.ground + opened[k];
                }
            }
        }

        if prof {
            eprintln!("tile {id}: pass A {:.3}s, pass B {:.3}s", t_a, t_start.elapsed().as_secs_f64() - t_a);
        }
        // ---------------- outputs
        let look = &self.world.cfg.satellite;
        let albedo_look = &self.world.cfg.albedo;
        let az = look.sun_azimuth_deg.to_radians();
        let el = look.sun_elevation_deg.to_radians();
        let sun = DVec3::new(az.sin() * el.cos(), az.cos() * el.cos(), el.sin());
        let l0 = look.ambient + look.direct * el.sin();
        let haze_col = DVec3::new(0.50, 0.58, 0.70);

        let mut rgb = vec![0u8; n * n * 3];
        let mut albedo = vec![0u8; n * n * 3];
        let mut elevation = vec![0f32; n * n];
        let mut normal = vec![0i8; n * n * 3];
        let mut landcover = vec![0u8; n * n];
        let mut emission = vec![0u8; n * n * 3];
        let mut emin = f32::MAX;
        let mut emax = f32::MIN;
        let hb = |i: usize, j: usize| pb[j * na + i].height;
        for j in 0..n {
            let (g, gn) = (row_gsd[j + 1], row_gsd_ns[j + 1]);
            for i in 0..n {
                let (ia, ja) = (i + 1, j + 1);
                let px = &pb[ja * na + ia];
                let dhdx = (hb(ia + 1, ja) - hb(ia - 1, ja)) / (2.0 * g);
                let dhdn = -(hb(ia, ja + 1) - hb(ia, ja - 1)) / (2.0 * gn);
                let nrm = DVec3::new(-dhdx, -dhdn, 1.0).normalize();
                let k = j * n + i;
                let h = px.height as f32;
                elevation[k] = h;
                emin = emin.min(h);
                emax = emax.max(h);
                landcover[k] = px.class;
                normal[3 * k] = (nrm.x * 127.0).round() as i8;
                normal[3 * k + 1] = (nrm.y * 127.0).round() as i8;
                normal[3 * k + 2] = (nrm.z * 127.0).round() as i8;
                let light = look.ambient * (0.55 + 0.45 * nrm.z) + look.direct * nrm.dot(sun).max(0.0) * px.lit;
                let alb = {
                    let a = px.albedo;
                    let lum = 0.2126 * a.x + 0.7152 * a.y + 0.0722 * a.z;
                    ((DVec3::splat(lum) + (a - DVec3::splat(lum)) * albedo_look.saturation) * albedo_look.brightness).max(DVec3::ZERO)
                };
                let mut c = alb * (light / l0) * look.exposure;
                c = c * (1.0 - look.haze) + haze_col * look.haze;
                for ch in 0..3 {
                    albedo[3 * k + ch] = (l2s(alb[ch]) * 255.0).round() as u8;
                    rgb[3 * k + ch] = (l2s(c[ch]) * 255.0).round() as u8;
                    emission[3 * k + ch] = ((px.emission[ch] / 16.0).clamp(0.0, 1.0).powf(1.0 / 3.0) * 255.0).round() as u8;
                }
            }
        }
        TileData { id, rgb, albedo, elevation, normal, landcover, emission, elev_min: emin, elev_max: emax }
    }
}
