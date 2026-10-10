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

/// A tile's features binned like its pass-A pixels (17 × 17 bins of 16 pixels; the same bins
/// as the instance lists and the river pieces): per bin the indices of the segments and stamps
/// that can reach it.
#[derive(Clone, Debug, Default)]
pub struct Binned {
    pub f: Features,
    pub segs: Vec<Vec<u32>>,
    pub stamps: Vec<Vec<u32>>,
}

/// The features of all kits for an area (empty without kits that build any).
pub fn build(w: &World, points: &mut dyn PointSource, area: &Area) -> Features {
    let mut out = Features::default();
    for k in crate::kits::KITS {
        if let Some(f) = k.host {
            let mut ctx = HostCtx { world: w, points: &mut *points };
            let r = f(&mut ctx, area);
            out.segs.extend(r.segs);
            out.stamps.extend(r.stamps);
        }
    }
    out
}

/// Do any kits build features?
pub fn any() -> bool {
    crate::kits::KITS.iter().any(|k| k.host.is_some())
}

/// Bin `f` into the discs `bins` (centre, radius): a segment within its half width (+ 1 %
/// and 2 m) of a disc, a stamp whose box's circle reaches it.
pub fn bin(f: Features, bins: &[(DVec3, f64)]) -> Binned {
    let mut b = Binned { segs: vec![Vec::new(); bins.len()], stamps: vec![Vec::new(); bins.len()], f };
    for (k, s) in b.f.segs.iter().enumerate() {
        let ab = s.b - s.a;
        for (i, &(c, r)) in bins.iter().enumerate() {
            let t = ((c - s.a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
            if (c - (s.a + ab * t)).length() <= r + 1.01 * s.hw as f64 + 2.0 {
                b.segs[i].push(k as u32);
            }
        }
    }
    for (k, s) in b.f.stamps.iter().enumerate() {
        let reach = (s.half[0] as f64).hypot(s.half[1] as f64) + 2.0;
        for (i, &(c, r)) in bins.iter().enumerate() {
            if (c - s.center).length() <= r + reach {
                b.stamps[i].push(k as u32);
            }
        }
    }
    b
}

/// The pass-A bins of a tile (centre and radius, as `bin_disc` of tile_a.wgsl).
pub fn tile_bins(w: &World, id: geodesy::tiles::TileId) -> Vec<(DVec3, f64)> {
    use geodesy::tiles::{gsd_ew, pixel_to_latlon};
    const NBIN: usize = 17;
    let n = crate::TILE_SIZE;
    let na2 = n + 4;
    let (ox, oy) = (id.x as f64 * n as f64, id.y as f64 * n as f64);
    let at_px = |i: usize, j: usize| {
        let (lat, lon) = pixel_to_latlon(glam::DVec2::new(ox + i as f64 - 2.0 + 0.5, oy + j as f64 - 2.0 + 0.5), id.z, n as u32);
        (Ctx::new(lat, lon, 1.0, &w.ell).p, gsd_ew(lat, id.z, n as u32, &w.ell))
    };
    (0..NBIN * NBIN)
        .map(|bin| {
            let (bx, by) = (bin % NBIN, bin / NBIN);
            let (i0, j0) = (bx * 16, by * 16);
            let (i1, j1) = ((i0 + 15).min(na2 - 1), (j0 + 15).min(na2 - 1));
            let (c, _) = at_px((i0 + i1) / 2, (j0 + j1) / 2);
            let mut r: f64 = 0.0;
            for (i, j) in [(i0, j0), (i1, j0), (i0, j1), (i1, j1)] {
                r = r.max((at_px(i, j).0 - c).length());
            }
            (c, r + at_px(i0, j0).1.max(at_px(i0, j1).1))
        })
        .collect()
}

/// The CPU's point source: `World::terrain`.
pub struct CpuPoints<'a>(pub &'a World);

impl PointSource for CpuPoints<'_> {
    fn terrain(&mut self, ctx: &Ctx) -> Option<crate::tile::PointTerrain> {
        let t = self.0.terrain(ctx);
        Some(crate::tile::PointTerrain { ground: t.ground, water: t.water, water_kind: t.water_kind })
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    /// `FSeg` of features.wgsl.
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    pub struct GFSeg {
        pub a: [f64; 4],
        pub b: [f64; 4],
        pub ha: f32,
        pub hb: f32,
        pub hw: f32,
        pub s0: f32,
        pub deck_a: f32,
        pub deck_b: f32,
        pub kind: u32,
        pub cls: u32,
        pub flags: u32,
        pub _p: [u32; 3],
        pub v: [f32; 4],
    }

    /// `FStamp` of features.wgsl.
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    pub struct GFStamp {
        pub center: [f64; 4],
        pub ex: [f32; 4],
        pub ey: [f32; 4],
        pub half: [f32; 2],
        pub template: u32,
        pub h: f32,
        pub v: [f32; 8],
        pub _p: [f32; 4],
    }

    pub fn seg(s: &FSeg) -> GFSeg {
        GFSeg {
            a: [s.a.x, s.a.y, s.a.z, 0.0],
            b: [s.b.x, s.b.y, s.b.z, 0.0],
            ha: s.ha,
            hb: s.hb,
            hw: s.hw,
            s0: s.s0,
            deck_a: s.deck_a,
            deck_b: s.deck_b,
            kind: s.kind,
            cls: s.class as u32,
            flags: s.flags,
            _p: [0; 3],
            v: s.v,
        }
    }

    pub fn stamp(s: &FStamp) -> GFStamp {
        let v = |d: DVec3| [d.x as f32, d.y as f32, d.z as f32, 0.0];
        GFStamp { center: [s.center.x, s.center.y, s.center.z, 0.0], ex: v(s.ex), ey: v(s.ey), half: s.half, template: s.template, h: s.h, v: s.v, _p: [0.0; 4] }
    }

    #[test]
    fn sizes() {
        assert_eq!(std::mem::size_of::<GFSeg>(), 128);
        assert_eq!(std::mem::size_of::<GFStamp>(), 128);
    }
}
