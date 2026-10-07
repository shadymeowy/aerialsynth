//! CPU reference renderer: tile meshes generated on the fly, band-parallel rasterization into a
//! supersampled G-buffer, deferred shading with pyramid texture filtering, aerial perspective.
//!
//! Geometry: each render unit (a tile, or a quadrant of an ancestor tile when finer data is
//! missing) becomes a grid mesh whose vertices sit on pixel *corners*; a corner's height is the
//! mean of the four surrounding pixel centres (shared by neighbouring tiles → watertight between
//! equal zooms). Skirts hide cracks between different zooms. All transforms are f64; the camera
//! model is applied per vertex (incl. distortion) and LOD keeps triangles ~pixel sized, so
//! straight-edge rasterization is accurate to a small fraction of a pixel.

use crate::atmo::{AtmoParams, Atmosphere};
use crate::lighting::{LightingConfig, SunState};
use crate::cache::TileCache;
use crate::camera::CameraModel;
use crate::lod::{LodParams, Selector, Unit};
use crate::trajectory::CamPose;
use geodesy::tiles::{gsd_ew, TileId};
use geodesy::Ellipsoid;
use glam::{DVec2, DVec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use tilestore::{TileData, TILE_SIZE};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Shading {
    /// Drape the satellite-look `rgb` layer (baked lighting) — closest to the map imagery.
    Satellite,
    /// Re-light the `albedo` layer with the renderer's sun (different time of day than the map).
    Relit,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderSettings {
    /// Supersampling per axis (odd values make the GT sample coincide with the pixel centre).
    pub supersample: u32,
    /// LOD: refine until a texel covers at most this many output pixels.
    pub texel_px: f64,
    /// Target mesh edge length in output pixels (vertex stride is chosen per tile).
    pub mesh_px: f64,
    pub min_zoom: u8,
    pub max_zoom: u8,
    pub shading: Shading,
    /// Sun / day cycle / artificial lights.
    pub lighting: LightingConfig,
    pub atmosphere: AtmoParams,
    /// Max anisotropic texture taps.
    pub max_aniso: u32,
    /// Specular sun glint on water (relit mode).
    pub water_glint: bool,
    /// cpu (reference) or gpu (wgpu, headless; needs the `gpu` feature)
    pub backend: Backend,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Cpu,
    Gpu,
}

impl Default for RenderSettings {
    fn default() -> Self {
        RenderSettings {
            supersample: 3,
            texel_px: 1.0,
            mesh_px: 2.0,
            min_zoom: 2,
            max_zoom: 19,
            shading: Shading::Relit,
            lighting: LightingConfig::default(),
            atmosphere: AtmoParams::default(),
            max_aniso: 8,
            water_glint: true,
            backend: Backend::Cpu,
        }
    }
}

/// One rendered frame (output resolution).
pub struct FrameOut {
    pub width: u32,
    pub height: u32,
    /// Scene-linear radiance, f32 x3 (sunlit white Lambertian ≈ 1); exposure/tone in `sensor`.
    pub radiance: Vec<f32>,
    /// z-depth along the optical axis (m); +inf where no terrain (sky)
    pub depth: Vec<f32>,
    /// ECEF point of each pixel (None = sky), taken at the central sub-sample; used for flow.
    /// With an even supersample there is no sub-sample at the pixel centre: the points (and
    /// depth, land cover) belong to pixel + `sample_offset` in both axes.
    pub points: Vec<Option<DVec3>>,
    /// Offset (px) of the geometry sample from the pixel centre: 0 for odd supersampling,
    /// 1/4 for 2x, 1/8 for 4x.
    pub sample_offset: f64,
    /// land-cover class at the pixel centre (255 = sky)
    pub landcover: Vec<u8>,
    /// Lamp flicker split (only with `Renderer::split_flicker` while lights flicker; else empty):
    /// `radiance` then holds the flicker-free mean, and the radiance at time t is
    /// `radiance + flicker_cos·cos(ωt) + flicker_sin·sin(ωt)` with ω = 2π · 2 · mains_hz
    /// (instantaneous, no exposure averaging). f32 x3 like `radiance`.
    pub flicker_cos: Vec<f32>,
    pub flicker_sin: Vec<f32>,
    /// render units used (for statistics / planning feedback)
    pub units: Vec<Unit>,
}

impl FrameOut {
    /// Instantaneous radiance at time `t` (trajectory time) given the flicker split.
    pub fn radiance_at(&self, t: f64, omega: f64) -> Vec<f32> {
        if self.flicker_cos.is_empty() {
            return self.radiance.clone();
        }
        let (c, s) = ((omega * t).cos() as f32, (omega * t).sin() as f32);
        self.radiance.iter().zip(&self.flicker_cos).zip(&self.flicker_sin).map(|((r, a), b)| r + a * c + b * s).collect()
    }
}

/// One shaded output row.
struct RowOut {
    rad: Vec<f32>,
    dep: Vec<f32>,
    pts: Vec<Option<DVec3>>,
    lcs: Vec<u8>,
    fc: Vec<f32>,
    fs: Vec<f32>,
}

pub(crate) const NO_UNIT: u32 = u32::MAX;

#[derive(Clone, Copy, Default)]
pub(crate) struct Vert {
    pub sx: f64,
    pub sy: f64,
    pub z: f64,
    pub u: f32,
    pub v: f32,
    pub ok: bool,
}

pub(crate) struct Mesh {
    pub unit_idx: u32,
    pub nx: usize,
    pub ny: usize,
    pub verts: Vec<Vert>,
    /// skirt vertices: same layout as the border ring of `verts`, pushed down
    pub skirt: Vec<(usize, Vert)>,
    pub bbox: [f64; 4],
    pub row_y: Vec<(f64, f64)>,
}

#[derive(Clone, Copy)]
struct GSample {
    z: f32,
    unit: u32,
    u: f32,
    v: f32,
}

/// Small fast hasher for tile ids (FxHash-style multiply/rotate).
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);
impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0.rotate_left(5) ^ *b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}
type FxBuild = std::hash::BuildHasherDefault<FxHasher>;

/// Frame-local view of the tiles needed for shading.
pub(crate) struct TileView {
    pub tiles: HashMap<TileId, Arc<TileData>, FxBuild>,
    /// max elevation over 16x16-pixel blocks of each tile (shadow-ray empty-space skipping)
    pub blockmax: HashMap<TileId, Box<[f32; 256]>, FxBuild>,
    /// highest DSM point of all tiles in view (shadow rays above it can stop)
    pub max_elev: f64,
}

impl TileView {
    pub(crate) fn new(tiles: HashMap<TileId, Arc<TileData>, FxBuild>, with_blockmax: bool) -> Self {
        let blockmax = if with_blockmax {
            tiles
                .par_iter()
                .filter(|(_, t)| !t.elevation.is_empty())
                .map(|(id, t)| {
                    let mut b = Box::new([f32::MIN; 256]);
                    for j in 0..256 {
                        for i in 0..256 {
                            let k = (j / 16) * 16 + i / 16;
                            b[k] = b[k].max(t.elevation[j * 256 + i]);
                        }
                    }
                    (*id, b)
                })
                .collect()
        } else {
            HashMap::default()
        };
        let max_elev = tiles.values().filter(|t| !t.elevation.is_empty()).map(|t| t.elev_max as f64).fold(f64::MIN, f64::max);
        TileView { tiles, blockmax, max_elev }
    }

    /// Max elevation of the 16x16 block containing global pixel (gx, gy) at zoom z, if known.
    #[inline]
    fn block_max(&self, z: u8, gx: f64, gy: f64) -> Option<f32> {
        let n = 1i64 << z;
        let (px, py) = (gx.floor() as i64, gy.floor() as i64);
        let ty = py.div_euclid(256);
        if ty < 0 || ty >= n {
            return None;
        }
        let tx = px.div_euclid(256).rem_euclid(n);
        let b = self.blockmax.get(&TileId::new(z, tx as u32, ty as u32))?;
        Some(b[((py.rem_euclid(256) / 16) * 16 + px.rem_euclid(256) / 16) as usize])
    }

    fn get(&self, id: TileId) -> Option<&Arc<TileData>> {
        self.tiles.get(&id)
    }

    /// Elevation at integer global pixel (gx, gy) of zoom z, falling back to the given tile when the
    /// neighbour is missing (clamped).
    fn elev_px(&self, z: u8, gx: i64, gy: i64, fallback: &TileData) -> f32 {
        let n = 1i64 << z;
        let tx = gx.div_euclid(256).rem_euclid(n);
        let ty = gy.div_euclid(256);
        if ty >= 0 && ty < n {
            let id = TileId::new(z, tx as u32, ty as u32);
            // most texels are in the tile being meshed itself (no lookup)
            if fallback.id == id && !fallback.elevation.is_empty() {
                return fallback.elevation[(gy.rem_euclid(256) * 256 + gx.rem_euclid(256)) as usize];
            }
            if let Some(t) = self.tiles.get(&id) {
                if !t.elevation.is_empty() {
                    return t.elevation[(gy.rem_euclid(256) * 256 + gx.rem_euclid(256)) as usize];
                }
            }
        }
        let lx = gx - fallback.id.x as i64 * 256;
        let ly = gy - fallback.id.y as i64 * 256;
        fallback.elev(lx as isize, ly as isize)
    }

    /// Bilinear sample of a u8x3 layer (sRGB → linear) or the normal layer at continuous global
    /// pixel coords (pixel centres at +0.5) at zoom z; falls back to ancestors when missing.
    fn sample3(&self, z: u8, gx: f64, gy: f64, which: Which) -> Option<DVec3> {
        let mut z = z;
        let mut gx = gx;
        let mut gy = gy;
        loop {
            let x = gx - 0.5;
            let y = gy - 0.5;
            let x0 = x.floor();
            let y0 = y.floor();
            let fx = x - x0;
            let fy = y - y0;
            let (x0, y0) = (x0 as i64, y0 as i64);
            let mut acc = DVec3::ZERO;
            let mut wsum = 0.0;
            let mut found_any = false;
            let taps = [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)];
            // fast path: all four texels in one tile (one tile lookup instead of four; same sums)
            let (lx, ly) = (x0.rem_euclid(256), y0.rem_euclid(256));
            let n = 1i64 << z;
            let ty = y0.div_euclid(256);
            let tile = if lx < 255 && ly < 255 && ty >= 0 && ty < n {
                self.tiles.get(&TileId::new(z, x0.div_euclid(256).rem_euclid(n) as u32, ty as u32))
            } else {
                None
            };
            if let Some(t) = tile {
                for (dx, dy, w) in taps {
                    if let Some(c) = Self::texel_of(t, ((ly + dy) * 256 + lx + dx) as usize, which) {
                        acc += c * w;
                        wsum += w;
                        found_any = true;
                    }
                }
            } else {
                for (dx, dy, w) in taps {
                    if let Some(c) = self.texel(z, x0 + dx, y0 + dy, which) {
                        acc += c * w;
                        wsum += w;
                        found_any = true;
                    }
                }
            }
            if found_any && wsum > 1e-6 {
                return Some(acc / wsum);
            }
            if z == 0 {
                return None;
            }
            z -= 1;
            gx *= 0.5;
            gy *= 0.5;
        }
    }

    #[inline]
    fn texel(&self, z: u8, gx: i64, gy: i64, which: Which) -> Option<DVec3> {
        let n = 1i64 << z;
        let ty = gy.div_euclid(256);
        if ty < 0 || ty >= n {
            return None;
        }
        let tx = gx.div_euclid(256).rem_euclid(n);
        let t = self.tiles.get(&TileId::new(z, tx as u32, ty as u32))?;
        let k = (gy.rem_euclid(256) * 256 + gx.rem_euclid(256)) as usize;
        Self::texel_of(t, k, which)
    }

    /// Texel `k` (row-major index within the tile) of layer `which`.
    #[inline]
    fn texel_of(t: &TileData, k: usize, which: Which) -> Option<DVec3> {
        match which {
            Which::Rgb if !t.rgb.is_empty() => Some(srgb3(&t.rgb[3 * k..3 * k + 3])),
            Which::Albedo if !t.albedo.is_empty() => Some(srgb3(&t.albedo[3 * k..3 * k + 3])),
            Which::Emission if !t.emission.is_empty() => {
                let lut = EMIS_LUT.get_or_init(|| {
                    let mut l = [0.0; 256];
                    for (i, v) in l.iter_mut().enumerate() {
                        *v = 16.0 * (i as f64 / 255.0).powi(3);
                    }
                    l
                });
                let e = &t.emission[3 * k..3 * k + 3];
                Some(DVec3::new(lut[e[0] as usize], lut[e[1] as usize], lut[e[2] as usize]))
            }
            Which::Normal if !t.normal.is_empty() => {
                Some(DVec3::new(t.normal[3 * k] as f64, t.normal[3 * k + 1] as f64, t.normal[3 * k + 2] as f64) / 127.0)
            }
            _ => None,
        }
    }

    /// Bilinear DSM height at continuous global pixel coords of zoom z (ancestor fallback).
    fn elev(&self, z: u8, gx: f64, gy: f64) -> Option<f64> {
        let (mut z, mut gx, mut gy) = (z, gx, gy);
        loop {
            let n = 1i64 << z;
            let x = gx - 0.5;
            let y = gy - 0.5;
            let (x0, y0) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            let mut acc = 0.0;
            let mut ws = 0.0;
            let taps = [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)];
            // fast path: all four texels in one tile (one lookup; same sums)
            let (lx, ly) = (x0.rem_euclid(256), y0.rem_euclid(256));
            let ty0 = y0.div_euclid(256);
            if lx < 255 && ly < 255 && ty0 >= 0 && ty0 < n {
                if let Some(t) = self.tiles.get(&TileId::new(z, x0.div_euclid(256).rem_euclid(n) as u32, ty0 as u32)) {
                    if !t.elevation.is_empty() {
                        for (dx, dy, w) in taps {
                            acc += w * t.elevation[((ly + dy) * 256 + lx + dx) as usize] as f64;
                            ws += w;
                        }
                    }
                    if ws > 1e-6 {
                        return Some(acc / ws);
                    }
                    if z == 0 {
                        return None;
                    }
                    z -= 1;
                    gx *= 0.5;
                    gy *= 0.5;
                    continue;
                }
            }
            for (dx, dy, w) in taps {
                let (px, py) = (x0 + dx, y0 + dy);
                let ty = py.div_euclid(256);
                if ty < 0 || ty >= n {
                    continue;
                }
                let tx = px.div_euclid(256).rem_euclid(n);
                if let Some(t) = self.tiles.get(&TileId::new(z, tx as u32, ty as u32)) {
                    if !t.elevation.is_empty() {
                        acc += w * t.elevation[(py.rem_euclid(256) * 256 + px.rem_euclid(256)) as usize] as f64;
                        ws += w;
                    }
                }
            }
            if ws > 1e-6 {
                return Some(acc / ws);
            }
            if z == 0 {
                return None;
            }
            z -= 1;
            gx *= 0.5;
            gy *= 0.5;
        }
    }

    fn landcover(&self, z: u8, gx: f64, gy: f64) -> u8 {
        let n = 1i64 << z;
        let (gx, gy) = (gx.floor() as i64, gy.floor() as i64);
        let ty = gy.div_euclid(256);
        if ty < 0 || ty >= n {
            return 0;
        }
        let tx = gx.div_euclid(256).rem_euclid(n);
        match self.tiles.get(&TileId::new(z, tx as u32, ty as u32)) {
            Some(t) if !t.landcover.is_empty() => t.landcover[(gy.rem_euclid(256) * 256 + gx.rem_euclid(256)) as usize],
            _ => 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Which {
    Rgb,
    Albedo,
    Normal,
    Emission,
}

static SRGB_LUT: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
static EMIS_LUT: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
#[inline]
fn srgb3(c: &[u8]) -> DVec3 {
    let lut = SRGB_LUT.get_or_init(|| {
        let mut l = [0.0; 256];
        for (i, v) in l.iter_mut().enumerate() {
            let c = i as f64 / 255.0;
            *v = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        }
        l
    });
    DVec3::new(lut[c[0] as usize], lut[c[1] as usize], lut[c[2] as usize])
}
/// Latitude (rad) of a global mercator pixel row coordinate.
#[inline]
fn lat_of(gy: f64, z: u8) -> f64 {
    let n = (TILE_SIZE as f64) * (1u64 << z) as f64;
    (PI * (1.0 - 2.0 * gy / n)).sinh().atan()
}
#[inline]
fn lon_of(gx: f64, z: u8) -> f64 {
    let n = (TILE_SIZE as f64) * (1u64 << z) as f64;
    gx / n * 2.0 * PI - PI
}

pub struct Renderer {
    pub model: Arc<dyn CameraModel>,
    pub(crate) model_ss: Arc<dyn CameraModel>,
    pub settings: RenderSettings,
    pub ell: Ellipsoid,
    pub cache: Arc<TileCache>,
    /// Skip shading: only depth, 3D points and land cover (radiance stays zero). Used for
    /// cameras that produce geometry ground truth but no images.
    pub geometry_only: bool,
    /// Return lamp flicker as separate cos / sin images instead of applying it (see `FrameOut`).
    pub split_flicker: bool,
    /// unit rays of the supersampled grid (camera frame)
    pub(crate) rays: Vec<[f32; 3]>,
    /// unique id of this renderer (GPU-side caches of its ray table)
    pub(crate) id: u64,
}

impl Renderer {
    pub fn new(model: Arc<dyn CameraModel>, settings: RenderSettings, ell: Ellipsoid, cache: Arc<TileCache>) -> Self {
        let ss = settings.supersample.max(1);
        let model_ss = model.scaled(ss);
        let (w, h) = (model_ss.width() as usize, model_ss.height() as usize);
        let rays: Vec<[f32; 3]> = (0..w * h)
            .into_par_iter()
            .map(|k| {
                // pixels outside the camera model's domain get a zero ray (rendered black)
                let r = model_ss.unproject(DVec2::new((k % w) as f64, (k / w) as f64)).unwrap_or(DVec3::ZERO);
                [r.x as f32, r.y as f32, r.z as f32]
            })
            .collect();
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Renderer { model, model_ss, settings, ell, cache, geometry_only: false, split_flicker: false, rays, id }
    }

    pub fn select_units(&self, cam: &CamPose) -> Vec<Unit> {
        let params = LodParams {
            min_zoom: self.settings.min_zoom,
            max_zoom: self.settings.max_zoom,
            texel_px: self.settings.texel_px,
            ..Default::default()
        };
        let sel = Selector::new(cam, self.model.as_ref(), self.ell, &params, self.cache.as_ref());
        sel.select()
    }

    /// Tiles of the selected units with their neighbours and ancestors (the texture pyramid).
    pub(crate) fn gather_ids(&self, units: &[Unit]) -> Vec<TileId> {
        let mut need: Vec<TileId> = Vec::new();
        for u in units {
            need.push(u.data);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(n) = u.data.neighbor(dx, dy) {
                        if self.cache.available(n) {
                            need.push(n);
                        }
                    }
                }
            }
            let mut a = u.data;
            for _ in 0..4 {
                match a.parent() {
                    Some(p) if p.z >= self.settings.min_zoom.saturating_sub(2) => {
                        if self.cache.available(p) {
                            need.push(p);
                        }
                        a = p;
                    }
                    _ => break,
                }
            }
        }
        need.sort_unstable();
        need.dedup();
        need
    }

    /// The tiles of `gather_ids`, fetched (and generated if lazy).
    pub(crate) fn gather(&self, units: &[Unit], sun_state: &SunState) -> TileView {
        let need = self.gather_ids(units);
        self.cache.prefetch(&need);
        let shadows_needed = !self.geometry_only && self.settings.shading == Shading::Relit && self.settings.lighting.shadows && sun_state.direct > 1e-4;
        TileView::new(need.iter().filter_map(|id| self.cache.get(*id).map(|t| (*id, t))).collect(), shadows_needed)

    }

    /// Grid meshes of the units (camera-model projected vertices, skirts), f64 throughout.
    pub(crate) fn build_meshes(&self, cam: &CamPose, units: &[Unit], view: &TileView) -> Vec<Mesh> {
        let ss = self.settings.supersample.max(1) as usize;
        let ms = &self.model_ss;
        let (w, h) = (ms.width() as usize, ms.height() as usize);
        let rt = cam.r_ecef_cam.transpose();
        let half_lim = (ms.max_half_angle() * 1.35 + 0.05).min(PI - 0.03);
        let cos_lim = half_lim.cos();
        let focal = ms.focal_px();
        let e2 = self.ell.e2();
        units
            .par_iter()
            .enumerate()
            .filter_map(|(ui, u)| {
                let data = view.get(u.data)?.clone();
                if data.elevation.is_empty() {
                    return None;
                }
                let z = u.data.z;
                let [x0, y0, x1, y1] = u.rect;
                // vertex stride from the projected texel size at the unit's nearest point
                let (c, r) = crate::lod::tile_sphere(u.id, (data.elev_min, data.elev_max), &self.ell);
                let dmin = ((c - cam.pos).length() - r).max(1.0);
                let lat_c = u.id.center().0;
                let texel = gsd_ew(lat_c, z, 256, &self.ell);
                let proj = texel / dmin * focal / ss as f64; // output px per texel
                let mut stride = 1u32;
                while stride < 32 && (stride * 2) as f64 * proj <= self.settings.mesh_px && (x1 - x0) / (stride * 2) >= 2 {
                    stride *= 2;
                }
                let nx = ((x1 - x0) / stride) as usize + 1;
                let ny = ((y1 - y0) / stride) as usize + 1;
                let gx0 = u.data.x as i64 * 256 + x0 as i64;
                let gy0 = u.data.y as i64 * 256 + y0 as i64;
                // separable trig
                let cols: Vec<(f64, f64)> = (0..nx).map(|i| lon_of((gx0 + (i as i64) * stride as i64) as f64, z).sin_cos()).collect();
                let rows: Vec<(f64, f64, f64)> = (0..ny)
                    .map(|j| {
                        let lat = lat_of((gy0 + (j as i64) * stride as i64) as f64, z);
                        let (s, c) = lat.sin_cos();
                        let nrad = self.ell.a / (1.0 - e2 * s * s).sqrt();
                        (s, c, nrad)
                    })
                    .collect();
                let to_vert = |i: usize, j: usize, hgt: f64| -> Vert {
                    let (so, co) = cols[i];
                    let (s, c, nrad) = rows[j];
                    let p = DVec3::new((nrad + hgt) * c * co, (nrad + hgt) * c * so, (nrad * (1.0 - e2) + hgt) * s);
                    let pc = rt * (p - cam.pos);
                    let mut v = Vert {
                        // range (not z): valid for wide-angle models beyond 90°
                        z: pc.length(),
                        u: (x0 + i as u32 * stride) as f32,
                        v: (y0 + j as u32 * stride) as f32,
                        ..Default::default()
                    };
                    let len = pc.length();
                    if len > 0.0 && pc.z / len > cos_lim {
                        if let Some(px) = ms.project(pc) {
                            v.sx = px.x;
                            v.sy = px.y;
                            v.ok = true;
                        }
                    }
                    v
                };
                let corner_h = |i: usize, j: usize| -> f64 {
                    let gx = gx0 + (i as i64) * stride as i64;
                    let gy = gy0 + (j as i64) * stride as i64;
                    let a = view.elev_px(z, gx - 1, gy - 1, &data);
                    let b = view.elev_px(z, gx, gy - 1, &data);
                    let c = view.elev_px(z, gx - 1, gy, &data);
                    let d = view.elev_px(z, gx, gy, &data);
                    0.25 * (a + b + c + d) as f64
                };
                let mut verts = Vec::with_capacity(nx * ny);
                let mut heights = Vec::with_capacity(nx * ny);
                for j in 0..ny {
                    for i in 0..nx {
                        let hh = corner_h(i, j);
                        heights.push(hh);
                        verts.push(to_vert(i, j, hh));
                    }
                }
                // skirts along the border ring
                let skirt_depth = (4.0 * texel * stride as f64 + 2.0).min(800.0);
                let mut ring = Vec::new();
                for i in 0..nx {
                    ring.push((i, 0));
                }
                for j in 1..ny {
                    ring.push((nx - 1, j));
                }
                for i in (0..nx - 1).rev() {
                    ring.push((i, ny - 1));
                }
                for j in (1..ny - 1).rev() {
                    ring.push((0, j));
                }
                let skirt: Vec<(usize, Vert)> = ring
                    .iter()
                    .map(|&(i, j)| {
                        let k = j * nx + i;
                        (k, to_vert(i, j, heights[k] - skirt_depth))
                    })
                    .collect();
                let mut bbox = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
                let mut any = false;
                let mut row_y = Vec::with_capacity(ny.saturating_sub(1));
                for j in 0..ny {
                    let mut lo = f64::MAX;
                    let mut hi = f64::MIN;
                    for i in 0..nx {
                        let v = &verts[j * nx + i];
                        if v.ok {
                            any = true;
                            bbox[0] = bbox[0].min(v.sx);
                            bbox[1] = bbox[1].min(v.sy);
                            bbox[2] = bbox[2].max(v.sx);
                            bbox[3] = bbox[3].max(v.sy);
                            lo = lo.min(v.sy);
                            hi = hi.max(v.sy);
                        }
                    }
                    row_y.push((lo, hi));
                }
                for (_, v) in &skirt {
                    if v.ok {
                        bbox[0] = bbox[0].min(v.sx);
                        bbox[1] = bbox[1].min(v.sy);
                        bbox[2] = bbox[2].max(v.sx);
                        bbox[3] = bbox[3].max(v.sy);
                    }
                }
                if !any || bbox[2] < -1.0 || bbox[0] > w as f64 || bbox[3] < -1.0 || bbox[1] > h as f64 {
                    return None;
                }
                // per quad row y range = union of its two vertex rows
                let row_y = (0..ny - 1)
                    .map(|j| (row_y[j].0.min(row_y[j + 1].0), row_y[j].1.max(row_y[j + 1].1)))
                    .collect();
                Some(Mesh { unit_idx: ui as u32, nx, ny, verts, skirt, bbox, row_y })
            })
            .collect()
    }

    /// Render one frame for camera pose `cam` under the given sun / light state.
    pub fn render(&self, cam: &CamPose, sun_state: &SunState) -> FrameOut {
        #[cfg(feature = "gpu")]
        if self.settings.backend == Backend::Gpu {
            return crate::gpu::render(self, cam, sun_state);
        }
        #[cfg(not(feature = "gpu"))]
        if self.settings.backend == Backend::Gpu {
            panic!("render.backend: gpu needs the `gpu` feature of the render crate");
        }
        let ss = self.settings.supersample.max(1) as usize;
        let ms = &self.model_ss;
        let (w, h) = (ms.width() as usize, ms.height() as usize);
        let prof = std::env::var_os("RENDER_PROFILE").is_some();
        let t0 = std::time::Instant::now();
        let units = self.select_units(cam);
        let t_sel = t0.elapsed().as_secs_f64();
        let view = self.gather(&units, sun_state);
        let t_fetch = t0.elapsed().as_secs_f64();
        let meshes = self.build_meshes(cam, &units, &view);

        let t_mesh = t0.elapsed().as_secs_f64();
        // ---------------- rasterize (parallel over horizontal bands)
        let band_h = 16usize;
        let mut gbuf = vec![GSample { z: f32::INFINITY, unit: NO_UNIT, u: 0.0, v: 0.0 }; w * h];
        gbuf.par_chunks_mut(band_h * w).enumerate().for_each(|(b, buf)| {
            let y0 = (b * band_h) as f64;
            let y1 = (y0 as usize + buf.len() / w) as f64;
            let rows_in_band = buf.len() / w;
            for m in &meshes {
                if m.bbox[3] < y0 - 1.0 || m.bbox[1] > y1 + 1.0 {
                    continue;
                }
                let nx = m.nx;
                for j in 0..m.ny - 1 {
                    let (lo, hi) = m.row_y[j];
                    if hi < y0 - 1.0 || lo > y1 + 1.0 {
                        continue;
                    }
                    for i in 0..nx - 1 {
                        let a = &m.verts[j * nx + i];
                        let bq = &m.verts[j * nx + i + 1];
                        let c = &m.verts[(j + 1) * nx + i];
                        let d = &m.verts[(j + 1) * nx + i + 1];
                        raster_tri(a, bq, d, m.unit_idx, buf, w, y0 as usize, rows_in_band);
                        raster_tri(a, d, c, m.unit_idx, buf, w, y0 as usize, rows_in_band);
                    }
                }
                // skirt quads
                let n = m.skirt.len();
                for s in 0..n {
                    let (ka, sa) = &m.skirt[s];
                    let (kb, sb) = &m.skirt[(s + 1) % n];
                    let a = &m.verts[*ka];
                    let bq = &m.verts[*kb];
                    raster_tri(a, bq, sb, m.unit_idx, buf, w, y0 as usize, rows_in_band);
                    raster_tri(a, sb, sa, m.unit_idx, buf, w, y0 as usize, rows_in_band);
                }
            }
        });

        let t_raster = t0.elapsed().as_secs_f64();
        // ---------------- shading
        let ow = self.model.width() as usize;
        let oh = self.model.height() as usize;
        let cam_geo0 = geodesy::ecef2geodetic(cam.pos, &self.ell);
        let sun = {
            let (az, el) = (sun_state.azimuth, sun_state.elevation);
            geodesy::rot_ecef2enu(cam_geo0.lat, cam_geo0.lon).transpose() * DVec3::new(az.sin() * el.cos(), az.cos() * el.cos(), el.sin())
        };
        let moon = {
            let (az, el) = (sun_state.moon_azimuth, sun_state.moon_elevation);
            geodesy::rot_ecef2enu(cam_geo0.lat, cam_geo0.lon).transpose() * DVec3::new(az.sin() * el.cos(), az.cos() * el.cos(), el.sin())
        };
        let atmo = Atmosphere::new(self.settings.atmosphere.clone(), sun, moon, sun_state);
        let cam_geo = geodesy::ecef2geodetic(cam.pos, &self.ell);
        let cam_up = geodesy::up_vector(cam_geo.lat, cam_geo.lon);
        let cs = ss / 2; // central sub-sample
        let alpha = 1.0 / ms.focal_px(); // sub-sample angular size
        let do_shadow = self.settings.shading == Shading::Relit && self.settings.lighting.shadows && sun_state.direct > 1e-4;
        let split = self.split_flicker && !self.geometry_only && sun_state.lights > 1e-3 && sun_state.flicker.enabled;
        let rows_out: Vec<RowOut> = (0..oh)
            .into_par_iter()
            .map(|oy| {
                let mut rad = vec![0f32; ow * 3];
                let mut dep = vec![f32::INFINITY; ow];
                let mut pts = vec![None; ow];
                let mut lcs = vec![255u8; ow];
                let (mut fc, mut fs) = if split { (vec![0f32; ow * 3], vec![0f32; ow * 3]) } else { (vec![], vec![]) };
                for ox in 0..ow {
                    if self.geometry_only {
                        let k = (oy * ss + cs) * w + ox * ss + cs;
                        let (g, ray) = (gbuf[k], self.rays[k]);
                        if g.unit != NO_UNIT && ray != [0.0; 3] {
                            let ray = DVec3::new(ray[0] as f64, ray[1] as f64, ray[2] as f64);
                            let u = &units[g.unit as usize];
                            let range = g.z as f64;
                            dep[ox] = (range * ray.z) as f32;
                            pts[ox] = Some(cam.pos + cam.r_ecef_cam * ray * range);
                            lcs[ox] = view.landcover(u.data.z, u.data.x as f64 * 256.0 + g.u as f64, u.data.y as f64 * 256.0 + g.v as f64);
                        }
                        continue;
                    }
                    // Per-pixel shading context from the central sub-sample (or the first terrain
                    // sub-sample): footprint / LOD, lighting incl. cast shadow, atmosphere. The
                    // other sub-samples only fetch texture, unless they lie at a clearly different
                    // depth (silhouettes), in which case they get their own context.
                    let mut order: [(usize, usize); 25] = [(0, 0); 25];
                    let mut no = 0;
                    order[no] = (cs, cs);
                    no += 1;
                    for sy in 0..ss {
                        for sx in 0..ss {
                            if (sx, sy) != (cs, cs) && no < 25 {
                                order[no] = (sx, sy);
                                no += 1;
                            }
                        }
                    }
                    let mut ctx: Option<PixShade> = None;
                    // radiance, flicker cos / sin parts
                    let mut acc = [DVec3::ZERO; 3];
                    // surface sample: lit colour + lamp emission (flickering, or split into cos/sin)
                    let add = |acc: &mut [DVec3; 3], pc: &PixShade, (base, emis): (DVec3, DVec3)| {
                        if split {
                            let (d, phase) = pc.flicker_mod;
                            acc[0] += base + emis;
                            acc[1] += emis * (d * phase.cos());
                            acc[2] -= emis * (d * phase.sin());
                        } else {
                            acc[0] += base + emis * pc.flicker;
                        }
                    };
                    for &(sx, sy) in &order[..no] {
                        let x = ox * ss + sx;
                        let y = oy * ss + sy;
                        let g = gbuf[y * w + x];
                        let ray = self.rays[y * w + x];
                        let ray = DVec3::new(ray[0] as f64, ray[1] as f64, ray[2] as f64);
                        if ray == DVec3::ZERO {
                            continue; // outside the lens image circle
                        }
                        let dir_w = cam.r_ecef_cam * ray;
                        if g.unit == NO_UNIT {
                            acc[0] += atmo.sky(dir_w, cam_up);
                            continue;
                        }
                        let u = &units[g.unit as usize];
                        let range = g.z as f64;
                        let z = u.data.z;
                        let gx = u.data.x as f64 * 256.0 + g.u as f64;
                        let gy = u.data.y as f64 * 256.0 + g.v as f64;
                        if sx == cs && sy == cs {
                            dep[ox] = (range * ray.z) as f32; // z-depth along the optical axis
                            pts[ox] = Some(cam.pos + dir_w * range);
                            lcs[ox] = view.landcover(z, gx, gy);
                        }
                        let reuse = matches!(&ctx, Some(c) if (range - c.range).abs() < 0.03 * c.range);
                        if !reuse {
                            let shadow = if do_shadow { self.sun_visibility(&view, z, gx, gy, sun_state) } else { 1.0 };
                            let pc = self.pixel_shade(&view, &atmo, sun_state, z, gx, gy, range, dir_w, alpha, cam.pos, cam_geo.h, shadow);
                            if ctx.is_none() {
                                ctx = Some(pc);
                            } else {
                                add(&mut acc, &pc, self.texture(&view, &pc, z, gx, gy));
                                continue;
                            }
                        }
                        let pc = ctx.as_ref().unwrap();
                        add(&mut acc, pc, self.texture(&view, pc, z, gx, gy));
                    }
                    let n = (ss * ss) as f64;
                    for ch in 0..3 {
                        rad[3 * ox + ch] = (acc[0][ch] / n) as f32;
                        if split {
                            fc[3 * ox + ch] = (acc[1][ch] / n) as f32;
                            fs[3 * ox + ch] = (acc[2][ch] / n) as f32;
                        }
                    }
                }
                RowOut { rad, dep, pts, lcs, fc, fs }
            })
            .collect();
        if prof {
            let ntri: usize = meshes.iter().map(|m| 2 * (m.nx - 1) * (m.ny - 1)).sum();
            eprintln!(
                "render: {} units, {} tiles, {} tris | select {:.3}s fetch {:.3}s mesh {:.3}s raster {:.3}s shade {:.3}s",
                units.len(),
                view.tiles.len(),
                ntri,
                t_sel,
                t_fetch - t_sel,
                t_mesh - t_fetch,
                t_raster - t_mesh,
                t0.elapsed().as_secs_f64() - t_raster
            );
        }
        let mut out = FrameOut {
            width: ow as u32,
            height: oh as u32,
            radiance: Vec::with_capacity(ow * oh * 3),
            depth: Vec::with_capacity(ow * oh),
            points: Vec::with_capacity(ow * oh),
            landcover: Vec::with_capacity(ow * oh),
            sample_offset: (cs as f64 + 0.5) / ss as f64 - 0.5,
            flicker_cos: Vec::with_capacity(if split { ow * oh * 3 } else { 0 }),
            flicker_sin: Vec::with_capacity(if split { ow * oh * 3 } else { 0 }),
            units,
        };
        for r in rows_out {
            out.radiance.extend(r.rad);
            out.depth.extend(r.dep);
            out.points.extend(r.pts);
            out.landcover.extend(r.lcs);
            out.flicker_cos.extend(r.fc);
            out.flicker_sin.extend(r.fs);
        }
        out
    }

    /// Fraction of direct sunlight reaching a surface point (1 = lit): march the DSM towards the
    /// sun through finer-then-coarser pyramid levels, including Earth curvature.
    fn sun_visibility(&self, view: &TileView, z: u8, gx: f64, gy: f64, sun: &SunState) -> f64 {
        let Some(h0) = view.elev(z, gx, gy) else { return 1.0 };
        let lat = lat_of(gy, z);
        let tan_e = sun.elevation.max(0.005).tan();
        // horizontal direction towards the sun in texel space (x east, y south)
        let dir = DVec2::new(sun.azimuth.sin(), -sun.azimuth.cos());
        let r_earth = self.ell.a;
        let mut zl = z;
        let mut px = DVec2::new(gx, gy);
        let mut texel = gsd_ew(lat, zl, 256, &self.ell);
        let mut dist = 0.0;
        let mut step = 1.0; // texels of the current level
        let bias = 0.4 * texel + 0.3;
        let mut i = 0;
        while i < 160 {
            i += 1;
            // empty-space skipping: if the ray is above the current block's max, jump to the
            // block exit (blocks are 16x16 texels of the current level)
            let ray_here = h0 + bias + dist * tan_e - dist * dist / (2.0 * r_earth);
            if ray_here > view.max_elev + 1.0 {
                break; // above everything that could occlude
            }
            if let Some(bm) = view.block_max(zl, px.x, px.y) {
                if ray_here > bm as f64 + 0.01 {
                    // (x − 16⌊x/16⌋ is exact for a power-of-two divisor and avoids libm's fmod,
                    // ~9% of the render time)
                    let fx = px.x - 16.0 * (px.x * 0.0625).floor();
                    let fy = px.y - 16.0 * (px.y * 0.0625).floor();
                    let tx = if dir.x > 1e-9 { (16.0 - fx) / dir.x } else if dir.x < -1e-9 { fx / -dir.x } else { f64::MAX };
                    let ty = if dir.y > 1e-9 { (16.0 - fy) / dir.y } else if dir.y < -1e-9 { fy / -dir.y } else { f64::MAX };
                    let adv = tx.min(ty) + 0.05;
                    px += dir * adv;
                    dist += adv * texel;
                    if dist > 40_000.0 || ray_here > 9000.0 {
                        break;
                    }
                    // far from the start, continue on a coarser level
                    if dist > 64.0 * texel && zl > 0 {
                        zl -= 1;
                        px *= 0.5;
                        texel *= 2.0;
                    }
                    continue;
                }
            }
            px += dir * step;
            dist += step * texel;
            let ray_h = h0 + bias + dist * tan_e - dist * dist / (2.0 * r_earth);
            if ray_h > 9000.0 {
                break;
            }
            if let Some(hh) = view.elev(zl, px.x, px.y) {
                if hh > ray_h {
                    return 0.0;
                }
            }
            // every 12 fine steps move one level coarser (double the step length)
            if i % 12 == 11 && zl > 0 {
                zl -= 1;
                px *= 0.5;
                texel *= 2.0;
            } else if i % 12 == 5 {
                step = (step * 1.5).min(2.0);
            }
            if dist > 40_000.0 {
                break;
            }
        }
        1.0
    }

    /// Shading context of one pixel: texture footprint and the affine colour transform
    /// `radiance = tex * mul + add + emission * emis` (lighting and atmosphere folded in).
    #[allow(clippy::too_many_arguments)]
    fn pixel_shade(&self, view: &TileView, atmo: &Atmosphere, sun_state: &SunState, z: u8, gx: f64, gy: f64, range: f64, dir_w: DVec3, alpha: f64, cam_pos: DVec3, h_cam: f64, shadow: f64) -> PixShade {
        let lat = lat_of(gy, z);
        let lon = lon_of(gx, z);
        let (sl, cl) = lat.sin_cos();
        let (so, co) = lon.sin_cos();
        let up = DVec3::new(cl * co, cl * so, sl);
        let east = DVec3::new(-so, co, 0.0);
        let north = DVec3::new(-sl * co, -sl * so, cl);
        let v = -dir_w;
        let v_enu = DVec3::new(v.dot(east), v.dot(north), v.dot(up));
        // footprint: minor axis = range * pixel angle, major stretched by the grazing angle
        let cos_inc = v_enu.z.max(0.03);
        let e2 = self.ell.e2();
        let nrad = self.ell.a / (1.0 - e2 * sl * sl).sqrt();
        let texel = 2.0 * PI * nrad * cl / (256.0 * (1u64 << z) as f64);
        let minor = range * alpha;
        let major = minor / cos_inc;
        let lam = (minor / texel).log2();
        let aniso = (major / minor).clamp(1.0, self.settings.max_aniso as f64);
        let hdir = DVec2::new(v_enu.x, -v_enu.y);
        let hdir = if hdir.length_squared() > 1e-12 { hdir.normalize() } else { DVec2::X };
        let sky_amb = DVec3::new(0.80, 0.90, 1.10) * 0.32 * sun_state.sky;
        let (mut mul, mut add) = match self.settings.shading {
            Shading::Satellite => {
                // baked lighting; dimmed by the current daylight
                let day = (0.75 * sun_state.direct + 0.25 * sun_state.sky).min(1.0);
                (DVec3::splat(day), DVec3::ZERO)
            }
            Shading::Relit => {
                let n_enu = view.sample3(z, gx, gy, Which::Normal).unwrap_or(DVec3::Z);
                let n = (east * n_enu.x + north * n_enu.y + up * n_enu.z).normalize();
                let ndl = n.dot(atmo.sun_dir).max(0.0);
                let ndm = n.dot(atmo.moon_dir).max(0.0);
                (
                    atmo.sun_color() * (ndl * 1.25 * shadow) + atmo.moon_color() * (ndm * 1.25) + sky_amb * (0.6 + 0.4 * n.dot(up)),
                    DVec3::ZERO,
                )
            }
        };
        // light of the nearby lamps falling on everything around them (walls, roofs, yards, trees):
        // the emission averaged over ~25 m as irradiance; without it the night town was black
        // between the lamp pools, buildings only silhouettes
        if self.settings.shading == Shading::Relit && sun_state.lights > 1e-3 {
            let k = ((25.0 / texel).log2().ceil().max(0.0) as u8).min(z);
            let s = 0.5f64.powi(k as i32);
            if let Some(e) = view.sample3(z - k, gx * s, gy * s, Which::Emission) {
                mul += e * (0.2 * sun_state.lights);
            }
        }
        if self.settings.shading == Shading::Relit && self.settings.water_glint && terragen::landcover::is_water(view.landcover(z, gx, gy)) {
            let sun = atmo.sun_dir;
            let hv = (v + sun).normalize();
            let nh = up.dot(hv).max(0.0);
            let fres = 0.02 + 0.98 * (1.0 - v.dot(up).max(0.0)).powi(5);
            let sky_c = atmo.sky((dir_w - up * 2.0 * dir_w.dot(up)).normalize(), up);
            mul *= 1.0 - fres;
            add += sky_c * fres + atmo.sun_color() * (shadow * 1.5 * nh.powf(300.0));
        }
        // height of the point above the ellipsoid (closed form given its geodetic latitude)
        let p_w = cam_pos - v * range;
        let h_pt = if cl.abs() > 0.1 { (p_w.x * p_w.x + p_w.y * p_w.y).sqrt() / cl - nrad } else { p_w.z.abs() / sl.abs() - nrad * (1.0 - e2) };
        let (t, mut ins) = atmo.transmittance(h_cam, h_pt, range, dir_w);
        // light pollution: artificial light scattered by the haze. Averaged over ~1.5 km so it is
        // a broad glow dome, not a halo painted in the shape of each lit area (which a ~0.5 km
        // average and a strong gain produced: glowing ellipses on the ground around towns)
        if sun_state.lights > 1e-3 && sun_state.light_pollution > 0.0 {
            let k = ((1500.0 / texel).log2().ceil().max(0.0) as u8).min(z);
            let s = 0.5f64.powi(k as i32);
            if let Some(e) = view.sample3(z - k, gx * s, gy * s, Which::Emission) {
                ins += e * (DVec3::ONE - t) * (sun_state.lights * sun_state.light_pollution * 1.5);
            }
        }
        // lamp flicker: one supply phase per ~40 m cell (global zoom-17 grid of 32 px)
        let (flicker, flicker_mod) = if sun_state.lights > 1e-3 && sun_state.flicker.enabled {
            let s17 = 2f64.powi(17 - z as i32);
            let cell = ((gx * s17 / 32.0).floor() as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((gy * s17 / 32.0).floor() as i64 as u64);
            (sun_state.flicker.factor(cell, sun_state.time, sun_state.exposure), sun_state.flicker.modulation(cell))
        } else {
            (1.0, (0.0, 0.0))
        };
        PixShade {
            flicker,
            flicker_mod,
            z,
            range,
            lam,
            aniso,
            hdir,
            major_texels: major / texel,
            mul: mul * t,
            add: add * t + ins,
            emis: t * sun_state.lights,
        }
    }

    /// Filtered texture fetch (trilinear across pyramid levels, anisotropic taps) for a sample
    /// of zoom `z` at global pixel coords (gx, gy), with the pixel's shading context.
    /// Returns (lit surface colour, lamp emission before flicker).
    fn texture(&self, view: &TileView, ps: &PixShade, z: u8, gx: f64, gy: f64) -> (DVec3, DVec3) {
        let which = match self.settings.shading {
            Shading::Satellite => Which::Rgb,
            Shading::Relit => Which::Albedo,
        };
        // footprint relative to this sample's data zoom
        let dz = z as f64 - ps.z as f64;
        let lam = (ps.lam + dz).max(0.0);
        let major_texels = ps.major_texels * 2f64.powf(dz);
        let taps = ps.aniso.ceil() as usize;
        let lights_on = ps.emis.max_element() > 1e-9;
        let l0 = lam.floor();
        let t = lam - l0;
        let mut col = DVec3::ZERO;
        let mut emis = DVec3::ZERO;
        let mut wsum = 0.0;
        for (lvl, wl) in [(l0, 1.0 - t), (l0 + 1.0, t)] {
            if wl <= 1e-4 {
                continue;
            }
            let k = (lvl as u8).min(z);
            let zl = z - k;
            let s = 0.5f64.powi(k as i32);
            let ext = (major_texels * s).min(64.0);
            for i in 0..taps {
                let o = if taps > 1 { (i as f64 + 0.5) / taps as f64 - 0.5 } else { 0.0 };
                let off = ps.hdir * (o * ext);
                let (sx, sy) = (gx * s + off.x, gy * s + off.y);
                if let Some(c) = view.sample3(zl, sx, sy, which) {
                    col += c * wl;
                    wsum += wl;
                }
            }
        }
        // lamps: sampled 1.5 levels sharper than the footprint and without the anisotropic taps,
        // so a lamp stays a point (filtered with the footprint, every lamp was a soft blob of 2–3
        // pixels); the sub-samples, motion blur and bloom average the residual aliasing
        let mut ewsum = 0.0;
        if lights_on {
            let lam_e = (lam - 1.5).max(0.0);
            let l0e = lam_e.floor();
            let te = lam_e - l0e;
            for (lvl, wl) in [(l0e, 1.0 - te), (l0e + 1.0, te)] {
                if wl <= 1e-4 {
                    continue;
                }
                let k = (lvl as u8).min(z);
                let s = 0.5f64.powi(k as i32);
                if let Some(e) = view.sample3(z - k, gx * s, gy * s, Which::Emission) {
                    emis += e * wl;
                    ewsum += wl;
                }
            }
        }
        let e0 = if ewsum > 0.0 { emis / ewsum } else { DVec3::ZERO };
        let c0 = if wsum > 0.0 { col / wsum } else { DVec3::splat(0.2) };
        (c0 * ps.mul + ps.add, e0 * ps.emis)
    }
}

/// Per-pixel shading context (see `Renderer::pixel_shade`).
struct PixShade {
    z: u8,
    range: f64,
    lam: f64,
    aniso: f64,
    hdir: DVec2,
    major_texels: f64,
    mul: DVec3,
    add: DVec3,
    emis: DVec3,
    /// lamp flicker factor of the pixel's lamp cell (at the render time, exposure-averaged)
    flicker: f64,
    /// (depth, phase) of that cell's modulation, for the cos / sin split
    flicker_mod: (f64, f64),
}

/// Rasterize one triangle into a band of the G-buffer (rows [y0, y0+rows)).
#[inline]
#[allow(clippy::too_many_arguments)]
fn raster_tri(a: &Vert, b: &Vert, c: &Vert, unit: u32, buf: &mut [GSample], w: usize, y0: usize, rows: usize) {
    if !(a.ok && b.ok && c.ok) {
        return;
    }
    let minx = a.sx.min(b.sx).min(c.sx);
    let maxx = a.sx.max(b.sx).max(c.sx);
    let miny = a.sy.min(b.sy).min(c.sy);
    let maxy = a.sy.max(b.sy).max(c.sy);
    let ylo = (y0 as f64).max(miny.ceil());
    let yhi = ((y0 + rows - 1) as f64).min(maxy.floor());
    if ylo > yhi {
        return;
    }
    let xlo = minx.ceil().max(0.0);
    let xhi = maxx.floor().min((w - 1) as f64);
    if xlo > xhi {
        return;
    }
    let area = (b.sx - a.sx) * (c.sy - a.sy) - (b.sy - a.sy) * (c.sx - a.sx);
    if area.abs() < 1e-12 {
        return;
    }
    // reject huge triangles (vertices straddling the validity limit of the camera model)
    if (maxx - minx) * (maxy - miny) > (w * w) as f64 * 4.0 {
        return;
    }
    let inv_area = 1.0 / area;
    let (iza, izb, izc) = (1.0 / a.z, 1.0 / b.z, 1.0 / c.z);
    for py in ylo as usize..=yhi as usize {
        let y = py as f64;
        let row = &mut buf[(py - y0) * w..(py - y0 + 1) * w];
        for px in xlo as usize..=xhi as usize {
            let x = px as f64;
            let w0 = ((b.sx - x) * (c.sy - y) - (b.sy - y) * (c.sx - x)) * inv_area;
            let w1 = ((c.sx - x) * (a.sy - y) - (c.sy - y) * (a.sx - x)) * inv_area;
            let w2 = 1.0 - w0 - w1;
            if w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9 {
                continue;
            }
            let iz = w0 * iza + w1 * izb + w2 * izc;
            let zz = 1.0 / iz;
            let s = &mut row[px];
            if (zz as f32) < s.z && zz > 0.0 {
                let p0 = w0 * iza * zz;
                let p1 = w1 * izb * zz;
                let p2 = w2 * izc * zz;
                s.z = zz as f32;
                s.unit = unit;
                s.u = (p0 * a.u as f64 + p1 * b.u as f64 + p2 * c.u as f64) as f32;
                s.v = (p0 * a.v as f64 + p1 * b.v as f64 + p2 * c.v as f64) as f32;
            }
        }
    }
}
