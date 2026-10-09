//! Quadtree LOD selection of XYZ tiles for a camera view (shared by planning and rendering).
//!
//! A tile is refined while one of its texels, seen from the closest point of its bounding sphere,
//! covers more than `texel_px` image pixels. Culling uses the camera's bounding cone and a
//! horizon test against a sphere inscribed in the ellipsoid.

use crate::camera::CameraModel;
use crate::trajectory::CamPose;
use geodesy::tiles::{gsd_ew, TileId};
use geodesy::{Ellipsoid, Geodetic};
use glam::DVec3;

#[derive(Clone, Debug)]
pub struct LodParams {
    /// Coarsest level at which the quadtree starts (whole world at this zoom).
    pub min_zoom: u8,
    pub max_zoom: u8,
    /// Refine until a texel projects to at most this many pixels.
    pub texel_px: f64,
    /// Elevation range assumed for tiles whose range is unknown.
    pub default_range: (f32, f32),
    /// Extra angular margin on the view cone (radians).
    pub cone_margin: f64,
}

impl Default for LodParams {
    fn default() -> Self {
        LodParams { min_zoom: 2, max_zoom: 19, texel_px: 1.0, default_range: (-100.0, 6000.0), cone_margin: 0.03 }
    }
}

/// What the quadtree may know about tiles.
pub trait TileOracle {
    /// Elevation range of a tile if known.
    fn range(&self, id: TileId) -> Option<(f32, f32)>;
    /// Whether tile data exists (planning: always true).
    fn exists(&self, id: TileId) -> bool;
    /// Refine tiles whose elevation range is only inherited from an ancestor (or the default).
    /// The pre-render dry run says no: it descends one level per pass and learns the real
    /// ranges on the way, instead of refining conservative (huge) volumes to the finest zoom.
    fn refine_unknown(&self) -> bool {
        true
    }
}

/// Planning oracle: every tile "exists"; elevation ranges come from an estimator (e.g. the
/// generator at coarse resolution) or a fixed range.
pub struct PlanOracle<'a> {
    pub estimate: Option<&'a (dyn Fn(TileId) -> (f32, f32) + Sync)>,
    pub fixed: (f32, f32),
}
impl<'a> PlanOracle<'a> {
    pub fn fixed(range: (f32, f32)) -> Self {
        PlanOracle { estimate: None, fixed: range }
    }
}
impl TileOracle for PlanOracle<'_> {
    fn range(&self, id: TileId) -> Option<(f32, f32)> {
        Some(match self.estimate {
            Some(f) => f(id),
            None => self.fixed,
        })
    }
    fn exists(&self, _id: TileId) -> bool {
        true
    }
}

/// A selected render unit: the logical tile `id` drawn from `data` (which is `id` itself or an
/// ancestor when finer data is missing). `rect` is the sub-rectangle of `data` in its pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Unit {
    pub id: TileId,
    pub data: TileId,
    /// x0, y0, x1, y1 in data-tile pixels (exclusive end)
    pub rect: [u32; 4],
}

impl Unit {
    pub fn new(id: TileId, data: TileId) -> Self {
        let k = id.z - data.z;
        let size = 256u32 >> k;
        let x0 = (id.x - (data.x << k)) * size;
        let y0 = (id.y - (data.y << k)) * size;
        Unit { id, data, rect: [x0, y0, x0 + size, y0 + size] }
    }
}

/// Sample points of a tile's surface (5x5 mercator-uniform grid) at heights `hs`.
pub fn tile_points(id: TileId, hs: &[f64], ell: &Ellipsoid) -> Vec<DVec3> {
    let b = id.bounds();
    let n = 5;
    let mut pts = Vec::with_capacity(n * n * hs.len());
    for j in 0..n {
        let v = j as f64 / (n - 1) as f64;
        let y = id.y as f64 + v;
        let lat = ((std::f64::consts::PI * (1.0 - 2.0 * y / (1u64 << id.z) as f64)).sinh()).atan();
        for i in 0..n {
            let lon = b.lon_min + (b.lon_max - b.lon_min) * i as f64 / (n - 1) as f64;
            for &h in hs {
                pts.push(geodesy::geodetic2ecef(Geodetic::new(lat, lon, h), ell));
            }
        }
    }
    pts
}

/// Bounding sphere of a tile's terrain volume.
pub fn tile_sphere(id: TileId, range: (f32, f32), ell: &Ellipsoid) -> (DVec3, f64) {
    let pts = tile_points(id, &[range.0 as f64, range.1 as f64], ell);
    let c = pts.iter().fold(DVec3::ZERO, |a, p| a + *p) / pts.len() as f64;
    let mut r = pts.iter().map(|p| (*p - c).length()).fold(0.0, f64::max);
    // curvature sag between samples
    let b = id.bounds();
    let extent = (b.lon_max - b.lon_min).abs() * ell.a;
    let seg = extent / 4.0;
    r += seg * seg / (8.0 * ell.b) + 1.0;
    (c, r)
}

/// Horizon test: is point `p` hidden by the sphere of radius `r0` (centred at the origin) when
/// seen from `cam`?
#[inline]
pub fn occluded_by_sphere(cam: DVec3, p: DVec3, r0: f64) -> bool {
    let vh2 = cam.length_squared() - r0 * r0;
    if vh2 <= 0.0 {
        return false;
    }
    let vt = p - cam;
    let vt_dot_vc = -vt.dot(cam);
    vt_dot_vc > vh2 && vt_dot_vc * vt_dot_vc / vt.length_squared() > vh2
}

pub struct Selector<'a> {
    pub cam: &'a CamPose,
    pub model: &'a dyn CameraModel,
    pub ell: Ellipsoid,
    pub params: &'a LodParams,
    pub oracle: &'a dyn TileOracle,
    axis: DVec3,
    cos_half: f64,
    half: f64,
    /// camera height above the ellipsoid, local up, and the lowest elevation angle in the view cone
    cam_h: f64,
    up: DVec3,
    cone_low: f64,
}

impl<'a> Selector<'a> {
    pub fn new(cam: &'a CamPose, model: &'a dyn CameraModel, ell: Ellipsoid, params: &'a LodParams, oracle: &'a dyn TileOracle) -> Self {
        let axis = cam.r_ecef_cam * DVec3::Z;
        let half = (model.max_half_angle() + params.cone_margin).min(std::f64::consts::PI);
        let geo = geodesy::ecef2geodetic(cam.pos, &ell);
        let up = DVec3::new(geo.lat.cos() * geo.lon.cos(), geo.lat.cos() * geo.lon.sin(), geo.lat.sin());
        let cone_low = axis.dot(up).clamp(-1.0, 1.0).asin() - half;
        Selector { cam, model, ell, params, oracle, axis, cos_half: half.cos(), half, cam_h: geo.h, up, cone_low }
    }

    pub fn visible(&self, id: TileId, range: (f32, f32)) -> Option<(f64, f64)> {
        self.visible_known(id, range, true)
    }

    /// `known`: the range is the tile's own (else inherited from an ancestor ± 50 m, which may
    /// miss tall features: the below-view cull then keeps 300 m more headroom).
    fn visible_known(&self, id: TileId, range: (f32, f32), known: bool) -> Option<(f64, f64)> {
        let (c, r) = tile_sphere(id, range, &self.ell);
        let d = c - self.cam.pos;
        let dist = d.length();
        // below the view: with the camera above the tile's top, the tile volume lies below the
        // highest elevation angle of its top surface (attained on its boundary); cull when that
        // is under the lowest elevation of the view cone. Matters when the camera is inside the
        // bounding sphere (low flight, coarse tiles), where the cone test does not apply.
        if self.cone_low > -std::f64::consts::FRAC_PI_2 && self.cam_h > range.1 as f64 + 1.0 {
            let b = id.bounds();
            let seg = (b.lon_max - b.lon_min).abs() * self.ell.a / 4.0;
            let top = range.1 as f64 + seg * seg / (8.0 * self.ell.b) + if known { 10.0 } else { 300.0 };
            if top < self.cam_h {
                let emax = tile_points(id, &[top], &self.ell).iter().map(|p| (*p - self.cam.pos).normalize().dot(self.up).asin()).fold(f64::MIN, f64::max);
                if emax < self.cone_low - 0.01 {
                    return None;
                }
            }
        }
        if dist > r {
            // view cone test
            let cosang = d.dot(self.axis) / dist;
            let ang = cosang.clamp(-1.0, 1.0).acos();
            let spread = (r / dist).clamp(-1.0, 1.0).asin();
            if ang - spread > self.half {
                return None;
            }
            // horizon: cull when every surface sample (at the max height, lifted by the sag
            // between samples) is hidden behind a sphere inscribed in the ellipsoid
            let b = id.bounds();
            let seg = (b.lon_max - b.lon_min).abs() * self.ell.a / 4.0;
            let lift = range.1 as f64 + seg * seg / (8.0 * self.ell.b) + 10.0;
            let r0 = self.ell.b - 50.0;
            if tile_points(id, &[lift], &self.ell).iter().all(|p| occluded_by_sphere(self.cam.pos, *p, r0)) {
                return None;
            }
        }
        let _ = self.cos_half;
        Some((dist, r))
    }

    fn wants_refine(&self, id: TileId, dist: f64, r: f64) -> bool {
        if id.z >= self.params.max_zoom {
            return false;
        }
        let b = id.bounds();
        let lat = if b.lat_min <= 0.0 && b.lat_max >= 0.0 { 0.0 } else { b.lat_min.abs().min(b.lat_max.abs()) };
        let texel = gsd_ew(lat, id.z, 256, &self.ell);
        let dmin = (dist - r).max(1.0);
        texel / dmin * self.model.focal_px() > self.params.texel_px
    }

    /// Select render units for the view.
    pub fn select(&self) -> Vec<Unit> {
        let mut out = Vec::new();
        let z0 = self.params.min_zoom;
        let n = 1u32 << z0;
        for y in 0..n {
            for x in 0..n {
                let id = TileId::new(z0, x, y);
                let data = if self.oracle.exists(id) { Some(id) } else { None };
                self.recurse(id, data, &mut out);
            }
        }
        out
    }

    fn recurse(&self, id: TileId, data: Option<TileId>, out: &mut Vec<Unit>) {
        // elevation range: the tile's own if known, else the nearest ancestor's with a margin
        // (a tile that exists but is not generated yet, or is drawn from ancestor data), else the
        // default. The default (-100..6000 m) makes far tiles look close and over-refines them.
        let own = data.is_some().then(|| self.oracle.range(id)).flatten();
        let range = own
            .or_else(|| {
                let mut a = id;
                while let Some(p) = a.parent() {
                    if let Some((lo, hi)) = self.oracle.range(p) {
                        return Some((lo - 50.0, hi + 50.0));
                    }
                    a = p;
                }
                None
            })
            .unwrap_or(self.params.default_range);
        let Some((dist, r)) = self.visible_known(id, range, own.is_some()) else { return };
        if self.wants_refine(id, dist, r) && (own.is_some() || self.oracle.refine_unknown()) && id.children().iter().any(|c| self.oracle.exists(*c)) {
            for c in id.children() {
                // children without data are drawn from the nearest ancestor that has data
                let cd = if self.oracle.exists(c) { Some(c) } else { data };
                if cd.is_some() {
                    self.recurse(c, cd, out);
                }
            }
            return;
        }
        if let Some(d) = data {
            out.push(Unit::new(id, d));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraConfig, Extrinsics};
    use crate::trajectory::Pose;

    #[test]
    fn nadir_selection_is_local_and_fine() {
        let ell = Ellipsoid::WGS84;
        let cam_model = CameraConfig::pinhole_hfov(640, 512, 70.0).build().unwrap();
        let pose = Pose { t: 0.0, geo: Geodetic::from_deg(39.9, 32.8, 1500.0), q_ned_body: glam::DQuat::IDENTITY };
        let cam = pose.camera(&Extrinsics::default(), &ell);
        let params = LodParams { max_zoom: 18, ..Default::default() };
        let oracle = PlanOracle::fixed((0.0, 500.0));
        let sel = Selector::new(&cam, cam_model.as_ref(), ell, &params, &oracle);
        let units = sel.select();
        assert!(!units.is_empty());
        let zmax = units.iter().map(|u| u.id.z).max().unwrap();
        // ~1500 m altitude, f ≈ 457 px -> ground pixel ≈ 3.3 m -> zoom 15/16
        assert!((15..=17).contains(&zmax), "zmax {zmax}");
        // everything selected must be near the camera (nadir view)
        for u in &units {
            let (lat, lon) = u.id.center();
            let d = geodesy::haversine_distance(pose.geo, Geodetic::new(lat, lon, 0.0), ell.a);
            assert!(d < 40_000.0, "{:?} at {d}", u.id);
        }
    }

    #[test]
    fn horizon_occlusion() {
        let r0 = 6.3e6;
        let cam = DVec3::new(r0 + 10_000.0, 0.0, 0.0);
        assert!(!occluded_by_sphere(cam, DVec3::new(r0, 100_000.0, 0.0), r0 - 1.0));
        assert!(occluded_by_sphere(cam, DVec3::new(-r0, 0.0, 0.0), r0 - 1.0));
    }
}
