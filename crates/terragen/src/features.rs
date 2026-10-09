//! Linear features and stamps (`docs/design/terrain-next.md` §3.5): graphs and sites that need
//! terrain samples (roads, rail, power lines, dams, airports, ports, deltas) are built by host
//! Rust code shared by both backends ([`HostFn`] of a kit) and reach pass B as binned lists.

use crate::world::{Ctx, World};
use glam::DVec3;

/// A segment of a linear feature.
#[derive(Clone, Copy, Debug, Default)]
pub struct FSeg {
    /// ECEF ends (surface points) and their heights (m)
    pub a: DVec3,
    pub b: DVec3,
    pub ha: f32,
    pub hb: f32,
    /// kit-defined kind (its profile), land-cover class
    pub kind: u32,
    pub class: u8,
    /// half width (m)
    pub hw: f32,
    /// arc length at `a` (m): along-distance patterns stay continuous across segments
    pub s0: f32,
    /// engineered heights at the ends (m; NaN: on the ground)
    pub deck_a: f32,
    pub deck_b: f32,
    pub flags: u32,
    pub v: [f32; 4],
}

/// An oriented box with a template.
#[derive(Clone, Copy, Debug, Default)]
pub struct FStamp {
    pub center: DVec3,
    /// unit axes in the tangent plane
    pub ex: DVec3,
    pub ey: DVec3,
    pub half: [f32; 2],
    /// kit-defined template, its height (m) and parameters
    pub template: u32,
    pub h: f32,
    pub v: [f32; 8],
}

/// The features of an area.
#[derive(Clone, Debug, Default)]
pub struct Features {
    pub segs: Vec<FSeg>,
    pub stamps: Vec<FStamp>,
}

/// The area a host function builds features for: a disc on the surface and its pixel size.
#[derive(Clone, Copy, Debug)]
pub struct Area {
    pub center: DVec3,
    pub radius: f64,
    pub gsd: f64,
}

/// Terrain at points for host code: the CPU evaluates `World::terrain`; the GPU generator
/// answers `None` until its batch of point evaluations ran (the host function is then run
/// again).
pub trait PointSource {
    fn terrain(&mut self, ctx: &Ctx) -> Option<crate::tile::PointTerrain>;
}

/// What a host function sees.
pub struct HostCtx<'a> {
    pub world: &'a World,
    pub points: &'a mut dyn PointSource,
}

impl HostCtx<'_> {
    /// The terrain at (lat, lon) (radians) at pixel size `gsd`; None: pending.
    pub fn terrain(&mut self, lat: f64, lon: f64, gsd: f64) -> Option<crate::tile::PointTerrain> {
        let ctx = Ctx::new(lat, lon, gsd, &self.world.ell);
        self.points.terrain(&ctx)
    }
}

/// A kit's host function.
pub type HostFn = fn(&mut HostCtx, &Area) -> Features;

/// Signed distance (m, + left of a→b), along-distance from `a` (m, + `s0`) and the fraction
/// 0..1 along the segment of the point `p`.
pub fn seg_frame(s: &FSeg, p: DVec3) -> (f64, f64, f64) {
    let ab = s.b - s.a;
    let l2 = ab.length_squared().max(1e-9);
    let t = ((p - s.a).dot(ab) / l2).clamp(0.0, 1.0);
    let foot = s.a + ab * t;
    let up = foot.normalize_or_zero();
    let left = up.cross(ab).normalize_or_zero();
    let d = (p - foot).dot(left);
    let side = if d < 0.0 { -1.0 } else { 1.0 };
    (side * (p - foot).length(), s.s0 as f64 + t * l2.sqrt(), t)
}
