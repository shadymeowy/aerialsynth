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
    /// the highest elevation angle in the view cone, relative to the geocentric direction of the
    /// camera (for [`Selector::above_view`])
    cone_top_gc: f64,
}

impl<'a> Selector<'a> {
    pub fn new(cam: &'a CamPose, model: &'a dyn CameraModel, ell: Ellipsoid, params: &'a LodParams, oracle: &'a dyn TileOracle) -> Self {
        let axis = cam.r_ecef_cam * DVec3::Z;
        let half = (model.max_half_angle() + params.cone_margin).min(std::f64::consts::PI);
        let geo = geodesy::ecef2geodetic(cam.pos, &ell);
        let up = DVec3::new(geo.lat.cos() * geo.lon.cos(), geo.lat.cos() * geo.lon.sin(), geo.lat.sin());
        let cone_low = axis.dot(up).clamp(-1.0, 1.0).asin() - half;
        let cone_top_gc = axis.dot(cam.pos.normalize()).clamp(-1.0, 1.0).asin() + half;
        Selector { cam, model, ell, params, oracle, axis, cos_half: half.cos(), half, cam_h: geo.h, up, cone_low, cone_top_gc }
    }

    pub fn visible(&self, id: TileId, range: (f32, f32)) -> Option<(f64, f64)> {
        self.visible_known(id, range, true)
    }

    /// `known`: the range is the tile's own (else inherited from an ancestor ± 50 m, which may
    /// miss tall features: the below-view cull then keeps 300 m more headroom).
    fn visible_known(&self, id: TileId, range: (f32, f32), known: bool) -> Option<(f64, f64)> {
        let headroom = if known { 10.0 } else { 300.0 };
        if self.above_view(id, range.0 as f64 - headroom, range.1 as f64 + headroom) {
            return None;
        }
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

    /// Above the view: is every point of the tile's volume (heights `lo..=hi`) either seen
    /// higher than the top of the view cone or hidden behind the planet? For a camera looking
    /// down (the cone's top below the horizon) the ground it sees is near: tiles beyond its reach
    /// are culled although the camera is inside their bounding sphere (coarse tiles under a low
    /// flight, where the cone test does not apply).
    ///
    /// A bound, not a sample: the elevation angle of a point at geocentric angle γ from the
    /// camera and radius r is atan2(r cos γ − R, r sin γ) (R: the camera's radius); it grows with
    /// r, and in γ it rises to the horizon and falls beyond (one maximum), so over the tile it is
    /// at least its value at the tile's smallest radius and at the ends of the tile's range of γ.
    /// That range is bounded with the haversine formula over the tile's geocentric latitude and
    /// longitude intervals, and cut at the horizon of the sphere inscribed in the ellipsoid
    /// (beyond it a point of radius <= the tile's largest is hidden, as in the horizon test).
    fn above_view(&self, id: TileId, lo: f64, hi: f64) -> bool {
        use std::f64::consts::{PI, TAU};
        let top = self.cone_top_gc;
        if top >= 0.0 {
            return false;
        }
        let rc = self.cam.pos.length();
        let b = id.bounds();
        let (a, e2) = (self.ell.a, self.ell.e2());
        // the smallest radius of a point at height >= lo: its projection on the radial
        // direction of its foot on the ellipsoid is >= the foot's radius + lo (the cosine of the
        // angle between normal and radius is > 0.99999); the ellipsoid's radius shrinks with
        // |latitude|
        let foot = {
            let (s, c) = b.lat_min.abs().max(b.lat_max.abs()).min(std::f64::consts::FRAC_PI_2).sin_cos();
            let n = a / (1.0 - e2 * s * s).sqrt();
            ((n * c).powi(2) + (n * (1.0 - e2) * s).powi(2)).sqrt()
        };
        let r_min = foot + lo - lo.abs() * 1e-5 - 1.0;
        if r_min >= rc - 1.0 {
            return false;
        }
        // geocentric latitude of geodetic latitude `phi` at height `h` (increasing in both)
        let gc = |phi: f64, h: f64| {
            let (s, c) = phi.sin_cos();
            let n = a / (1.0 - e2 * s * s).sqrt();
            ((n * (1.0 - e2) + h) * s).atan2((n + h) * c)
        };
        let (psi_lo, psi_hi) = (gc(b.lat_min, lo).min(gc(b.lat_min, hi)), gc(b.lat_max, lo).max(gc(b.lat_max, hi)));
        let cdir = self.cam.pos / rc;
        let psi_c = cdir.z.clamp(-1.0, 1.0).asin();
        let lam_c = cdir.y.atan2(cdir.x);
        let dpsi_min = if psi_c < psi_lo {
            psi_lo - psi_c
        } else if psi_c > psi_hi {
            psi_c - psi_hi
        } else {
            0.0
        };
        let dpsi_max = (psi_c - psi_lo).abs().max((psi_c - psi_hi).abs());
        // longitude differences to the interval [lon_min, lon_max] (in -π..π)
        let wrap = |x: f64| (x + PI).rem_euclid(TAU) - PI;
        let width = b.lon_max - b.lon_min;
        let (d1, d2) = (wrap(b.lon_min - lam_c).abs(), wrap(b.lon_max - lam_c).abs());
        let inside = |l: f64| (l - b.lon_min).rem_euclid(TAU) <= width;
        let dl_min = if inside(lam_c) { 0.0 } else { d1.min(d2) };
        let dl_max = if inside(lam_c + PI) { PI } else { d1.max(d2) };
        let hav = |x: f64| (0.5 * x).sin().powi(2);
        let cos_min = psi_lo.abs().max(psi_hi.abs()).min(PI / 2.0).cos();
        let h_lb = hav(dpsi_min) + psi_c.cos() * cos_min * hav(dl_min);
        let h_ub = hav(dpsi_max) + psi_c.cos() * hav(dl_max);
        let g_lb = 2.0 * h_lb.clamp(0.0, 1.0).sqrt().asin();
        let g_ub = 2.0 * h_ub.clamp(0.0, 1.0).sqrt().asin();
        // the horizon: a point of radius <= r_max farther than this is behind the sphere
        let (r0, r_max) = (self.ell.b - 50.0, a + hi.max(0.0) + 1.0);
        let g_hor = (r0 / rc).clamp(-1.0, 1.0).acos() + (r0 / r_max).clamp(-1.0, 1.0).acos();
        let g_ub = g_ub.min(g_hor);
        if g_lb > g_ub {
            return true;
        }
        let elev = |g: f64| (r_min * g.cos() - rc).atan2(r_min * g.sin());
        elev(g_lb).min(elev(g_ub)) > top + 1e-3
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
        self.selection().units
    }

    /// Select render units for the view, with the tiles the selection refined and those whose
    /// elevation range it needs but does not know (see [`Selection`]).
    pub fn selection(&self) -> Selection {
        let mut sel = Selection::default();
        let z0 = self.params.min_zoom;
        let n = 1u32 << z0;
        for y in 0..n {
            for x in 0..n {
                let id = TileId::new(z0, x, y);
                let data = if self.oracle.exists(id) { Some(id) } else { None };
                self.recurse(id, data, &mut sel);
            }
        }
        sel
    }

    fn recurse(&self, id: TileId, data: Option<TileId>, sel: &mut Selection) {
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
        // a tile that exists without a known range: needed when it is visible (selected or
        // refined below), or when it is culled only for want of its range (with a range
        // UNKNOWN_SLACK wider it would not be: its own range could make it visible)
        let unknown = own.is_none() && data == Some(id);
        let Some((dist, r)) = self.visible_known(id, range, own.is_some()) else {
            if unknown && self.visible_known(id, (range.0 - UNKNOWN_SLACK, range.1 + UNKNOWN_SLACK), false).is_some() {
                sel.unknown.push(id);
            }
            return;
        };
        if unknown {
            sel.unknown.push(id);
        }
        if self.wants_refine(id, dist, r) && (own.is_some() || self.oracle.refine_unknown()) && id.children().iter().any(|c| self.oracle.exists(*c)) {
            sel.refined.push(id);
            for c in id.children() {
                // children without data are drawn from the nearest ancestor that has data
                let cd = if self.oracle.exists(c) { Some(c) } else { data };
                if cd.is_some() {
                    self.recurse(c, cd, sel);
                }
            }
            return;
        }
        if let Some(d) = data {
            sel.units.push(Unit::new(id, d));
        }
    }
}

/// How much wider (m, both ways) than the range a culled tile of unknown range is tested: if it
/// is still culled, its own range (within its parent's ± this) would not make it visible.
pub const UNKNOWN_SLACK: f32 = 1000.0;

/// The outcome of a LOD selection ([`Selector::selection`]).
#[derive(Clone, Debug, Default)]
pub struct Selection {
    /// What to draw.
    pub units: Vec<Unit>,
    /// The visible tiles that were refined into their children.
    pub refined: Vec<TileId>,
    /// Tiles that exist (`TileOracle::exists`) but whose elevation range is unknown
    /// (`TileOracle::range`) although the selection depends on it: visible ones (selected, or
    /// refined on an inherited range), and culled ones that a wider range would not cull.
    ///
    /// Once these are known (generated) and the view is selected again, until none are left, the
    /// selection is the one a store holding every tile gives: it no longer depends on which
    /// tiles were there before (lazy generation does this, `Renderer::select_units`). A culled
    /// tile that even its parent's range ± [`UNKNOWN_SLACK`] culls is culled with its own range
    /// too, known or not.
    pub unknown: Vec<TileId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraConfig, Extrinsics};
    use crate::trajectory::Pose;
    use std::sync::Arc;

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

    /// A made-up world: rolling hills (deterministic per tile; finer tiles see a little more
    /// relief, as real ones do). `known`: the tiles whose range a lazy store has learnt (None:
    /// every tile is known, a full store).
    struct Hills {
        known: Option<std::cell::RefCell<std::collections::HashSet<TileId>>>,
    }
    impl Hills {
        fn truth(id: TileId) -> (f32, f32) {
            let h = ((id.x.wrapping_mul(2654435761) ^ id.y.wrapping_mul(40503)) % 97) as f32;
            (100.0 + h, 300.0 + 2.0 * h + 4.0 * id.z as f32)
        }
    }
    impl TileOracle for Hills {
        fn range(&self, id: TileId) -> Option<(f32, f32)> {
            match &self.known {
                Some(k) => k.borrow().contains(&id).then(|| Self::truth(id)),
                None => Some(Self::truth(id)),
            }
        }
        fn exists(&self, id: TileId) -> bool {
            id.z <= 18
        }
        fn refine_unknown(&self) -> bool {
            self.known.is_none()
        }
    }

    /// The camera of the C example (160 x 120, 70°, 30° down, 1500 m above the hills).
    fn small_camera() -> (CamPose, Arc<dyn CameraModel>) {
        let ell = Ellipsoid::WGS84;
        let model = CameraConfig::pinhole_hfov(160, 120, 70.0).build().unwrap();
        let q = geodesy::euler_zyx_to_quat(45f64.to_radians(), -30f64.to_radians(), 0.0);
        let pose = Pose { t: 0.0, geo: Geodetic::from_deg(45.0, 10.0, 1800.0), q_ned_body: q };
        let ext = Extrinsics { mount: crate::camera::Mount::Forward, ..Default::default() };
        (pose.camera(&ext, &ell), model)
    }

    /// Lazy selection: learn the unknown tiles a pass needs, select again, until none are left.
    fn converge(oracle: &Hills, cam: &CamPose, model: &dyn CameraModel, params: &LodParams) -> (Vec<Unit>, usize, usize) {
        let mut learnt = 0;
        for pass in 1..30 {
            let s = Selector::new(cam, model, Ellipsoid::WGS84, params, oracle).selection();
            if s.unknown.is_empty() {
                return (s.units, pass, learnt);
            }
            learnt += s.unknown.len();
            oracle.known.as_ref().unwrap().borrow_mut().extend(s.unknown);
        }
        panic!("no convergence");
    }

    /// A small camera far from the ground selects coarse tiles, also when the elevation ranges
    /// are learnt lazily (the first render into an empty store), and the lazily converged
    /// selection is the one of a full store, whatever the store held before.
    #[test]
    fn small_camera_selection_is_coarse_and_lazy_selection_converges_to_the_full_one() {
        let (cam, model) = small_camera();
        let params = LodParams { max_zoom: 18, ..Default::default() };
        let full = Selector::new(&cam, model.as_ref(), Ellipsoid::WGS84, &params, &Hills { known: None }).select();
        // nearest ground ~1.8 km, focal 114 px: a pixel is ~15 m on the ground, a z14 texel
        // ~7 m (zoom 13/14; texel_px 1 per output pixel, supersampling does not refine)
        let zmax = full.iter().map(|u| u.id.z).max().unwrap();
        assert!(zmax <= 15, "zmax {zmax}");

        let empty = Hills { known: Some(Default::default()) };
        let (lazy, passes, learnt) = converge(&empty, &cam, model.as_ref(), &params);
        assert_eq!(lazy, full);
        assert!(passes <= 18 && learnt < 400, "{passes} passes, {learnt} tiles");
        // a store that already holds other tiles (neighbours of every tile learnt above, and a
        // finer patch) converges to the same selection
        let more: std::collections::HashSet<TileId> = empty
            .known
            .as_ref()
            .unwrap()
            .borrow()
            .iter()
            .flat_map(|id| (-1..=1).flat_map(move |dy| (-1..=1).filter_map(move |dx| id.neighbor(dx, dy))))
            .collect();
        let other = Hills { known: Some(std::cell::RefCell::new(more)) };
        other.known.as_ref().unwrap().borrow_mut().extend(lazy.iter().flat_map(|u| u.id.children()));
        assert_eq!(converge(&other, &cam, model.as_ref(), &params).0, full);
    }

    /// The above-view cull is a bound: no tile it culls has a point inside the view cone
    /// (random downward views and tiles near them, points sampled densely in the tile volume).
    #[test]
    fn above_view_never_culls_a_visible_point() {
        let ell = Ellipsoid::WGS84;
        let model = CameraConfig::pinhole_hfov(320, 240, 60.0).build().unwrap();
        let params = LodParams::default();
        let oracle = PlanOracle::fixed((0.0, 500.0));
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut culled, mut checked) = (0, 0);
        for _ in 0..60 {
            let (lat, lon) = (rnd() * 160.0 - 80.0, rnd() * 360.0 - 180.0);
            let h = 50.0 + rnd() * 20_000.0;
            let pitch = -30.0 - rnd() * 60.0;
            let q = geodesy::euler_zyx_to_quat((rnd() * 360f64).to_radians(), pitch.to_radians(), 0.0);
            let pose = Pose { t: 0.0, geo: Geodetic::from_deg(lat, lon, h), q_ned_body: q };
            let cam = pose.camera(&Extrinsics { mount: crate::camera::Mount::Forward, ..Default::default() }, &ell);
            let sel = Selector::new(&cam, model.as_ref(), ell, &params, &oracle);
            for z in [2u8, 5, 8, 11, 14, 17] {
                let c = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let Some(id) = c.neighbor(dx, dy) else { continue };
                        let lo = rnd() * 400.0 - 100.0;
                        checked += 1;
                        if !sel.above_view(id, lo, lo + 3000.0) {
                            continue;
                        }
                        culled += 1;
                        let b = id.bounds();
                        for j in 0..=24 {
                            for i in 0..=24 {
                                let la = b.lat_min + (b.lat_max - b.lat_min) * j as f64 / 24.0;
                                let lo_ = b.lon_min + (b.lon_max - b.lon_min) * i as f64 / 24.0;
                                for hh in [lo, lo + 300.0, lo + 3000.0] {
                                    let p = geodesy::geodetic2ecef(Geodetic::new(la, lo_, hh), &ell);
                                    let ang = (p - cam.pos).normalize().dot(sel.axis).clamp(-1.0, 1.0).acos();
                                    let hidden = occluded_by_sphere(cam.pos, p, ell.b - 50.0);
                                    assert!(ang > sel.half || hidden, "{id} culled above the view, but {la} {lo_} {hh} is in it");
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(culled > checked / 10, "{culled} of {checked} culled");
    }

    /// A nadir camera low over the ground: the coarse tiles beside its own are above the view
    /// (no longer refined because the camera is inside their bounding spheres): only the tiles
    /// under its footprint (~1.2 km across; here near a corner of tiles of zooms 8 to 12) are.
    #[test]
    fn low_nadir_camera_refines_only_the_tiles_below_it() {
        let ell = Ellipsoid::WGS84;
        let model = CameraConfig::pinhole_hfov(320, 256, 70.0).build().unwrap();
        let pose = Pose { t: 0.0, geo: Geodetic::from_deg(39.9, 32.8, 1630.0), q_ned_body: glam::DQuat::IDENTITY };
        let cam = pose.camera(&Extrinsics::default(), &ell);
        let params = LodParams { max_zoom: 15, ..Default::default() };
        let oracle = PlanOracle::fixed((600.0, 800.0));
        let sel = Selector::new(&cam, model.as_ref(), ell, &params, &oracle).selection();
        let (lat, lon) = (39.9f64.to_radians(), 32.8f64.to_radians());
        let near = 0.02f64.to_radians(); // ~2 km
        for t in &sel.refined {
            let b = t.bounds();
            let inside = b.lat_min - near <= lat && lat <= b.lat_max + near && b.lon_min - near <= lon && lon <= b.lon_max + near;
            assert!(inside, "{t} refined, away from the camera: {:?}", sel.refined);
        }
        assert_eq!(sel.refined.iter().filter(|t| t.z == 2).count(), 1);
    }

    #[test]
    fn horizon_occlusion() {
        let r0 = 6.3e6;
        let cam = DVec3::new(r0 + 10_000.0, 0.0, 0.0);
        assert!(!occluded_by_sphere(cam, DVec3::new(r0, 100_000.0, 0.0), r0 - 1.0));
        assert!(occluded_by_sphere(cam, DVec3::new(-r0, 0.0, 0.0), r0 - 1.0));
    }
}
