//! The core layers of the composite stack ([`crate::stack`]): the generator's own look, slot by
//! slot. Kits add theirs next to these ([`crate::kits`]); the WGSL twins are in
//! `gpu/wgsl/layers.wgsl`.
//!
//! | slot | core layers |
//! |---|---|
//! | 1 zonal | [`ground::zonal`]: soil / grass / tundra / marsh ground of the biome, textures, drainage lines |
//! | 3 azonal | [`azonal::layer`]: rock, sand seas, beaches; then [`azonal::finish_ground`] (micro-relief, masks, region frame, riparian) |
//! | 5 agriculture | [`agriculture::layer`]: fields of the land-use regions |
//! | canopy | [`canopy::layer`]: trees and shrubs of the biome's crown layers |
//! | 6 linear | [`linear::layer`]: noise-network roads, farm tracks |
//! | 7 built | [`built::select_town`] (before the canopy), [`built::layer`]: farmsteads, towns, lights |
//! | 8 water | [`water::standing`] (standing water), [`water::rivers`] |
//! | 9 seasonal | [`snow::layer`]: snow cover (its mask: [`snow::prepare`], first) |

pub mod agriculture;
pub mod azonal;
pub mod built;
pub mod canopy;
pub mod ground;
pub mod linear;
pub mod snow;
pub mod water;

use crate::kernels::{self, KIn};
use crate::registry::pal;
use crate::stack::{Layer, Stack};
use glam::{DVec2, DVec3};

impl Stack<'_> {
    /// A colour of the sample's natural palette: the biome's (blended where two are), with the
    /// ecoregion's style (lithology soil and rock, grass and crown tints).
    pub fn pal(&self, i: usize) -> DVec3 {
        let c = self.bio.pal(&self.sm.registry, i);
        let st = &self.bio.style;
        // (the lithology's share varies within the ecoregion: no uniform polygons)
        match i {
            pal::SOIL..=3 => c + (st.soil - c) * (st.soil_w * (0.25 + 1.3 * self.t.style[2]).clamp(0.0, 1.2)).min(1.0),
            pal::ROCK..=11 => c + (st.rock - c) * (st.rock_w * (0.4 + 1.2 * self.t.style[1]).clamp(0.0, 1.2)).min(1.0),
            pal::GRASS_WET | pal::GRASS_DRY | pal::GRASS_COLD | pal::TUNDRA | pal::MARSH => c * st.grass,
            pal::CROWN_CONIFER..=pal::SHRUB => c * st.crown,
            _ => c,
        }
    }

    /// A vegetation parameter of the sample's biome(s).
    pub fn veg(&self, f: impl Fn(&crate::registry::Veg) -> f64) -> f64 {
        self.bio.veg(&self.sm.registry, f)
    }

    /// The sample's agricultural intensity: pass A's, by the biome's and the culture's
    /// agriculture.
    pub fn agri(&self) -> f64 {
        let reg = &self.sm.registry;
        let b = reg.biomes[self.bio.a as usize].agriculture;
        let b = if self.bio.w > 0.0 { b + (reg.biomes[self.bio.b as usize].agriculture - b) * self.bio.w } else { b };
        (self.t.agri * b * self.bio.style.agri).clamp(0.0, 1.0)
    }
}

/// The local frame of the sample: the land-use region's (east / north at its centre, and the
/// region's rotated field frame), else the tangent plane at the sample.
pub fn frame(s: &mut Stack) {
    let p = s.ctx.p;
    if s.t.region.id != 0 {
        let r = s.sm.region_info(s.world, s.cache, s.t);
        let d = p - r.center;
        s.q_loc = DVec2::new(d.dot(r.east), d.dot(r.north));
        s.q_rot = DVec2::new(d.dot(r.ex), d.dot(r.ey));
        s.region = Some(r);
    } else {
        s.q_loc = DVec2::new(p.dot(s.ctx.east), p.dot(s.ctx.north));
    }
}

/// The axes of the sample's local frame (ECEF).
pub fn frame_axes(s: &Stack) -> (DVec3, DVec3) {
    match &s.region {
        Some(r) => (r.east, r.north),
        None => (s.ctx.east, s.ctx.north),
    }
}

/// Kernel inputs at the sample (local frame, amount; `aux` by kernel: contours get the ground
/// and slope, water the depth).
pub fn kin(s: &Stack, kind: u32, amount: f64) -> KIn {
    let (east, north) = frame_axes(s);
    let aux = match kind {
        kernels::kind::CONTOURS => [s.l.ground, s.l.slope, 0.0, 0.0],
        kernels::kind::WATER => [(s.l.water - s.l.ground).max(0.0), 0.0, 0.0, 0.0],
        _ => [0.0; 4],
    };
    KIn { q: s.q_loc, p: s.ctx.p, east, north, gsd: s.ctx.gsd, fw: s.l.fw, amount, aux }
}

/// The kernel layers of the sample's biome(s) at `slot` (registry `layers:`): each masked,
/// evaluated (explicit or its calibrated mean by the pixel size) and composited. Where two
/// biomes blend (coarse pixels at ecotones), each contributes by its weight.
pub fn biome_layers(s: &mut Stack, slot: usize) {
    let reg = &s.sm.registry;
    let band = crate::registry::band_of(s.ctx.gsd);
    let pairs = [(s.bio.a, 1.0 - s.bio.w), (s.bio.b, s.bio.w)];
    for (k, &(bi, wt)) in pairs.iter().enumerate() {
        if wt <= 0.0 || (k == 1 && bi == s.bio.a) {
            continue;
        }
        let b = &reg.biomes[bi as usize];
        // the same biome on both sides: one pass at full weight
        let wt = if k == 0 && s.bio.b == bi { 1.0 } else { wt };
        for &li in &b.bands[band] {
            let l = &b.layers[li as usize];
            if l.slot != slot || s.done {
                continue;
            }
            let m = l.mask(s);
            if m <= 0.0 {
                continue;
            }
            let o = kernels::eval(&l.k, &l.mean, &kin(s, l.k.kind, m * wt));
            if o.cov <= 0.0 && o.emit <= 0.0 {
                continue;
            }
            s.composite(Layer {
                cov: o.cov,
                albedo: o.albedo,
                dh: o.dh,
                hmode: l.hmode,
                cls: l.cls,
                emit: l.k.col[2] * o.emit,
                clear: l.clear,
                mat: l.mat,
                ..Default::default()
            });
        }
    }
}
