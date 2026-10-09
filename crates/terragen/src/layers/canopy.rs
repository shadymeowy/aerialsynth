//! The canopy: natural vegetation (crown layers of the biome), masked by everything that
//! claims the ground.

use crate::landcover as lc;
use crate::noise::*;
use crate::surface::*;
use glam::{DVec2, DVec3};

/// One tree layer (crowns on a jittered grid).
#[derive(Clone, Copy)]
pub(crate) struct TreeLayer {
    pub cell: f64,
    pub seed: u64,
    pub density: f64,
    /// density of the whole stand (all layers): crowns grow to close the canopy
    pub closure: f64,
    pub height: f64,
    pub color: DVec3,
    /// crown shape (`kernels::shape`)
    pub shape: u32,
    /// crown size of the stand (age): ~0.7 young … ~1.25 old
    pub scale: f64,
    /// colour of the stand (species, age, health)
    pub tone: DVec3,
}

impl SurfaceModel {
    /// Tree crowns of several layers at local position `q`. Returns (colour, canopy height, coverage).
    pub(crate) fn trees(&self, layers: &[TreeLayer], q: DVec2, gsd: f64, fw: f64, p: DVec3) -> (DVec3, f64, f64) {
        let mut best_h = 0.0f64;
        let mut best_col = DVec3::ZERO;
        let mut cov_total = 0.0f64;
        let mut mean_col = DVec3::ZERO;
        let mut mean_w = 0.0;
        let mut mean_h = 0.0;
        for layer in layers {
            if layer.density <= 1e-4 {
                continue;
            }
            let explicit = smoothstep(1.2 * gsd, 3.0 * gsd, layer.cell);
            // expected coverage when unresolved
            let mean_r = layer.cell * 0.48;
            let cov_mean = (layer.density * std::f64::consts::PI * mean_r * mean_r / (layer.cell * layer.cell)).min(1.0);
            if explicit < 1.0 {
                let w = cov_mean * (1.0 - explicit);
                // stochastic canopy texture (gaps, crown clusters) at scales just above the pixel
                let tex = perlin3(layer.seed, p / (layer.cell * 3.0)) * band(layer.cell * 3.0, gsd)
                    + 0.6 * perlin3(layer.seed ^ 5, p / (layer.cell * 9.0)) * band(layer.cell * 9.0, gsd);
                mean_col += layer.color * layer.tone * (0.85 + 0.3 * tex) * w;
                mean_w += w;
                mean_h += layer.height * 0.6 * cov_mean * (1.0 - explicit);
            }
            if explicit <= 0.0 {
                continue;
            }
            let cq = q / layer.cell;
            let cf = cq.floor();
            let (ix, iy) = (cf.x as i64, cf.y as i64);
            let fq = cq - cf;
            // crowns of cells farther than the largest crown radius (+ the filter width) cannot
            // reach the sample: skipped before hashing
            let reach = (0.6 * 1.55 * layer.scale.max(0.0) + fw / layer.cell) * (1.0 + 1e-9) + 1e-9;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (gx, gy) = (crate::noise::nb_gap(fq.x, dx, 0.4), crate::noise::nb_gap(fq.y, dy, 0.4));
                    if gx * gx + gy * gy > reach * reach {
                        continue;
                    }
                    let h = hash2(layer.seed, ix + dx, iy + dy);
                    let u = u01k(h, 3);
                    if u >= layer.density {
                        continue;
                    }
                    // density is evaluated per pixel: where it falls (stand / field edges) a crown
                    // would be clipped into a wall or a spike; fade it out with the margin instead
                    let fade = smoothstep(0.0, 0.3, (layer.density - u) / layer.density.max(1e-6));
                    let c = DVec2::new((ix + dx) as f64 + 0.5 + 0.8 * (u01k(h, 1) - 0.5), (iy + dy) as f64 + 0.5 + 0.8 * (u01k(h, 2) - 0.5)) * layer.cell;
                    // crowns grow with the stand density: dense forest closes its canopy
                    let mut r = layer.cell * (0.36 + 0.24 * u01k(h, 4)) * (1.0 + 0.55 * smoothstep(0.35, 0.9, layer.closure)) * layer.scale;
                    if layer.shape == crate::kernels::shape::STAR {
                        // palm fronds: a star-shaped crown
                        let a = (q.y - c.y).atan2(q.x - c.x);
                        r *= 0.82 + 0.18 * (8.0 * a + u01k(h, 10) * 6.3).cos();
                    }
                    let d = (q - c).length();
                    if d > r + fw {
                        continue;
                    }
                    let cov = ((r - d) / fw + 0.5).clamp(0.0, 1.0) * explicit * fade;
                    let x = (d / r).min(1.0);
                    let hh = layer.height * (0.55 + 0.9 * u01k(h, 5)) * (0.45 + 0.55 * layer.scale);
                    // crown surfaces taper towards the ground (cone / dome): a tall vertical wall at
                    // the crown rim made every tree a column, seen as a spike from low angles
                    let prof = crate::kernels::profile(layer.shape, x);
                    // foliage clumps: a bumpy crown surface gives the shading texture (a smooth cone or
                    // dome shades as a glossy blob)
                    let clump = perlin3(h ^ 0xC1, p / 1.6) * band(1.6, gsd) + 0.5 * perlin3(h ^ 0xC2, p / 0.8) * band(0.8, gsd);
                    let th = hh * prof * (1.0 + 0.10 * clump * (0.3 + 0.7 * (1.0 - x))) * fade;
                    if th > best_h {
                        best_h = th;
                        // neighbouring crowns of a stand look alike (strong per-tree tints made
                        // the canopy a camouflage pattern); the stand tone carries the variety
                        let tint = 0.86 + 0.24 * u01k(h, 6);
                        let hv = u01k(h, 7) - 0.5;
                        let hue = DVec3::new(1.0 + 0.16 * hv, 1.0 + 0.04 * (u01k(h, 9) - 0.5), 1.0 - 0.12 * hv);
                        // leafy mottling inside the crown (clumps of foliage, ~1 m)
                        let leaf = 0.82 + 0.36 * (0.5 + 0.5 * perlin3(h, p / 1.1)) * band(1.1, gsd);
                        // crowns are a bit darker at the rim (self-shading inside the crown)
                        best_col = layer.color * layer.tone * tint * hue * leaf * (0.75 + 0.35 * (1.0 - x));
                    }
                    cov_total = cov_total.max(cov);
                }
            }
        }
        // combine explicit and mean contributions
        let mean_cov = mean_w.min(1.0);
        let mc = if mean_w > 0.0 { mean_col / mean_w } else { DVec3::ZERO };
        let cov = (cov_total + mean_cov * (1.0 - cov_total)).min(1.0);
        if cov <= 0.0 {
            return (DVec3::ZERO, 0.0, 0.0);
        }
        let col = if cov_total > 0.0 { mixc(mc, best_col, cov_total / cov) } else { mc };
        let h = best_h.max(mean_h);
        (col, h, cov)
    }
}

/// The canopy: the trees and shrubs of the biome's crown layers, by the canopy model (forest
/// pattern × climate × the biome's and ecoregion's tree density; farmland cleared except its
/// woodlots; riparian belts; savanna trees and groves; shrubs), masked by everything below
/// that claims the ground (`Masks::natural_ok`, fields) and above it (`Masks::veg`: towns,
/// kits' layers that clear it).
pub fn layer(s: &mut crate::stack::Stack) {
    use crate::registry::{pal, share};
    use crate::stack::{hmode, Layer};
    let world = s.world;
    let veg = &world.cfg.vegetation;
    let (t, pf, l) = (s.t, s.pf, s.l);
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, l.fw);
    let m = s.m;
    if !(veg.tree_density > 0.0 && m.natural_ok * m.veg > 0.05) {
        return;
    }
    let (wet, temp, slope) = (t.moist, t.temp, l.slope);
    let st = &t.style;
    let style = s.bio.style;
    let trees_mult = s.veg(|v| v.trees) * style.trees;
    let base_cover = smoothstep(0.3, 0.68, wet) * smoothstep(-6.0, 2.0, temp) * veg.tree_density * trees_mult;
    let fpu = 0.5 + 0.5 * pf.forest; // ~[0, 1]
    // forests where the patch field is below the cover fraction (crisp but noisy edges)
    let forest = smoothstep(-0.03, 0.03, base_cover - fpu);
    // savanna / steppe scattered trees, groves
    let savanna = smoothstep(0.18, 0.35, wet) * (1.0 - smoothstep(0.55, 0.7, wet)) * smoothstep(12.0, 20.0, temp) * 0.12 * s.veg(|v| v.savanna);
    let groves = 0.04 * smoothstep(0.15, 0.3, wet) * s.veg(|v| v.groves);
    // farmland is cleared except on steep ground (and its woodlots): what cannot be a field
    // there is meadow, not forest
    let clear = 1.0 - 0.85 * smoothstep(0.05, 0.4, s.agri()) * (1.0 - smoothstep(0.45, 0.7, slope)) * (1.0 - m.woodlot);
    let base = m.natural_ok * (1.0 - m.field_cov) * veg.tree_density;
    let mut dens = (forest * 0.9 * clear + savanna + groves) * base;
    let mut dens_sav = savanna * base;
    dens = dens.max(0.8 * m.riparian * veg.tree_density);
    // drainage lines carry scrub (crowns clipped into slivers there made spikes)
    let gully_scrub = 0.55 * smoothstep(0.2, 0.9, -t.gully) * smoothstep(0.2, 0.5, wet) * m.natural_ok * (1.0 - m.field_cov);
    // steep slopes, the tree line, the snow line, what claims the ground above
    let limit = (1.0 - smoothstep(0.9, 1.4, slope))
        * (1.0 - smoothstep(0.0, 0.6, t.mountain * smoothstep(-2.0, -6.0, temp)))
        * (1.0 - smoothstep(-1.2, -2.4, temp + 1.0 * pf.snow))
        * m.veg;
    dens *= limit;
    dens_sav *= limit;
    // shrubs / bushes in steppe, maquis and rocky slopes
    let shrub_clim = smoothstep(0.15, 0.3, wet) * (1.0 - smoothstep(0.6, 0.8, wet)) * smoothstep(2.0, 10.0, temp);
    let shrub_patch = smoothstep(-0.2, 0.5, pf.patch + 0.4 * pf.land);
    let shrub = ((0.45 * shrub_clim * shrub_patch * (1.0 - forest) * m.natural_ok.max(0.4 * m.rock) * (1.0 - m.field_cov) * s.veg(|v| v.shrubs) * style.shrubs + gully_scrub)
        * veg.tree_density
        * m.veg)
        .clamp(0.0, 0.7);
    // forest stands (~240 m, irregular borders): each of its own age (crown size, height), tone
    // and conifer / broadleaf mix, with small canopy gaps
    let stand_id = pf.stand_id.unwrap_or_else(|| s.sm.stand_id(p, Some(pf.stand_warp)));
    let age = u01k(stand_id, 1);
    let tone_u = u01k(stand_id, 2);
    let stand_tone = mixc(DVec3::new(0.86, 0.93, 0.92), DVec3::new(1.12, 1.08, 0.88), tone_u) * (0.92 + 0.12 * age);
    let gap = smoothstep(0.3, 0.6, perlin3(0x6A9, p / 30.0) + 0.5 * perlin3(0x6AA, p / 11.0)) * (0.2 + 0.8 * u01k(stand_id, 4));
    dens *= 1.0 - 0.9 * gap * smoothstep(0.3, 0.7, dens);
    if !(dens > 0.0 || shrub > 0.01) {
        return;
    }
    // conifers in proper stands; lone and scattered trees in open land are broadleaf
    let stand_d = smoothstep(0.3, 0.75, dens);
    let conifer = (1.0 - smoothstep(4.0, 13.0, temp)) * stand_d.max(1.0 - smoothstep(-5.0, 1.0, temp));
    // mixed forests: stands lean conifer or broadleaf; the biome and ecoregion shift the mix
    let conifer = (conifer + 0.9 * (u01k(stand_id, 3) - 0.5) * (1.0 - (2.0 * conifer - 1.0).abs()) + s.veg(|v| v.conifer) + style.conifer).clamp(0.0, 1.0);
    let scale = (0.7 + 0.55 * age) * s.veg(|v| v.crown_scale);
    let tropic = (smoothstep(19.0, 25.0, temp) * smoothstep(0.55, 0.75, wet) + s.veg(|v| v.tropic)).clamp(0.0, 1.0);
    let dry = 1.0 - smoothstep(0.3, 0.5, wet);
    let tall = 0.5 + 0.5 * st[3] + s.veg(|v| v.tall);
    let reg = &s.sm.registry;
    let crowns = &reg.biomes[s.bio.a as usize].crowns;
    let mut layers = [TreeLayer { cell: 1.0, seed: 0, density: 0.0, closure: 0.0, height: 0.0, color: DVec3::ZERO, shape: 0, scale: 1.0, tone: DVec3::ONE }; crate::registry::MAX_CROWNS];
    let n = crowns.len().min(layers.len());
    for (k, c) in crowns.iter().enumerate().take(n) {
        let share = match c.share {
            share::CONIFER => dens * conifer,
            share::BROADLEAF => dens * (1.0 - conifer) * (1.0 - tropic),
            share::TROPICAL => dens * tropic,
            share::SHRUB => shrub,
            share::SAVANNA => dens_sav,
            _ => dens,
        };
        let colour = match c.colour_dry {
            Some(d) => mixc(c.colour, d, dry),
            None => c.colour,
        };
        layers[k] = TreeLayer {
            cell: c.cell,
            seed: c.seed,
            density: share * c.density,
            closure: if c.closure { dens } else { 0.0 },
            height: (c.h0 + c.h_tall * tall) * (c.open_height + (1.0 - c.open_height) * stand_d),
            color: colour * style.crown,
            shape: c.shape,
            scale: if c.stand { scale } else { 1.0 },
            tone: if c.stand { stand_tone } else { DVec3::ONE },
        };
    }
    let layers = &layers[..n];
    // forest floor: shaded litter and understory, not sunlit grass, between the crowns
    let floor = smoothstep(0.25, 0.8, dens);
    s.composite(Layer::paint(floor * 0.85, s.pal(pal::FLOOR) * 0.7));
    let (mut tc, th, tcov) = s.sm.trees(layers, s.q_loc, gsd, fw, p);
    if tcov > 0.0 {
        // forest stands of different age / species composition
        let stand = s.sm.pf_lazy(pf, crate::surface::PF_STAND);
        tc *= DVec3::new(1.0 + 0.10 * stand, 1.0 + 0.14 * stand, 1.0 + 0.05 * stand);
        let cls = if dens < 0.05 && shrub > dens {
            lc::SHRUB
        } else if tropic > 0.5 && wet > 0.6 {
            lc::TROPICAL_RAINFOREST
        } else if conifer > 0.7 {
            lc::NEEDLELEAF_FOREST
        } else if conifer < 0.3 {
            lc::BROADLEAF_FOREST
        } else {
            lc::MIXED_FOREST
        };
        let hm = if veg.trees_in_dsm { hmode::MAX } else { hmode::NONE };
        s.composite(Layer { cov: tcov, albedo: tc, dh: th, hmode: hm, cls, ..Default::default() });
    }
    // cast shadows from neighbouring trees onto the ground / other crowns
    if world.cfg.satellite.shadows && tcov < 0.99 {
        let mut shadow: f64 = 0.0;
        for layer in layers {
            if layer.density <= 0.0 || layer.cell < 2.0 * gsd {
                continue;
            }
            let off = s.sm.sun_h * (0.6 * layer.height / s.sm.sun_tan);
            let (_, _, sc) = s.sm.trees(std::slice::from_ref(layer), s.q_loc + off, gsd, fw, p);
            shadow = shadow.max(sc);
        }
        s.lit = s.lit.min(1.0 - 0.9 * shadow * (1.0 - tcov));
    }
    // dark understory under sparse prefiltered forest
    if tcov < 0.01 && dens > 0.0 {
        s.tint(DVec3::splat(1.0 - 0.15 * dens));
    }
}
