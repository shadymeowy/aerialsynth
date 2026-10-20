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
    /// size factor of the crowns' resolution (1: drawn one by one from a cell of ~1.2–3
    /// pixels; shrubs 0.4: only where their cell spans ~3–8 pixels)
    pub resolve: f64,
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
            // (round 11: crowns of 1-3 pixels drawn one by one, each with its tint and dark rim, were
            // the scenes' main high-frequency clutter, audit a26 a20 a25: their tints and rims fade in
            // up to 6 pixels; drawing them only from 1.6 pixels on moved the crossfade of the canopy's
            // heights and classes between zooms beyond the LOD tests' limits)
            let rc = layer.cell * layer.resolve;
            let explicit = smoothstep(1.2 * gsd, 3.0 * gsd, rc);
            let contrast = smoothstep(2.5 * gsd, 6.0 * gsd, rc);
            // expected coverage when unresolved
            // (round 15: the crowns' area grown with the stand's closure and age, as the explicit
            // crowns; it was the ungrown disc of 0.48 cells: a closed stand's mean showed a third
            // of pale floor, the far forest a pale smear above the dark resolved one, z14 / z15)
            let cov_mean = mean_cover(layer);
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
                        // (faded to their mean, 0.98 × 0.867: tint and rim over the crown's area)
                        best_col = layer.color * layer.tone * leaf * mixc(DVec3::splat(0.85), tint * hue * (0.75 + 0.35 * (1.0 - x)), contrast);
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

/// The expected coverage of a crown layer's crowns: discs of the crowns' mean area (radius
/// 0.36–0.6 cells, grown with the stand's closure and age as the explicit crowns), linear in the
/// density (a parent pixel's mean of its children's, LOD), a closed canopy at a density of 0.9.
pub(crate) fn mean_cover(ly: &TreeLayer) -> f64 {
    let grow = (1.0 + 0.55 * smoothstep(0.35, 0.9, ly.closure)) * ly.scale.max(0.0);
    (ly.density * (0.739 * grow * grow).min(1.11)).min(1.0)
}

/// The shading a crown layer's relief adds where its crowns are not drawn one by one (their
/// mean): the darkening of the pixel by the crowns' shaded sides and their shadows, as the
/// renderer shows the resolved crowns, at a sun- and view-independent mean. Calibrated (round
/// 15, `round15/tools/lodp.py`, `band.py`) with renders without haze: nadir at one zoom each
/// (pixel = texel), where a flat broadleaf canopy rendered 2.3× as bright as its resolved crowns
/// (conifers 1.7×, the tropics' closed canopy 1.3×), and oblique views from 1.5 km (pitch -22°),
/// where they appear less dark (crowns hide the shadows between them): a compromise between the
/// two. The mean's share fades out as the crowns span 1.5–4.5 pixels (their shading shows only
/// then; the tropics' big closed crowns from 1.2–3), in proportion to the layer's cover. Shrubs
/// are left out: they are their mean down to ~1 m pixels, where their low relief hardly
/// shades.
pub(crate) fn crown_relief(share: u32, shape: u32, ly: &TreeLayer, gsd: f64) -> f64 {
    use crate::registry::share as sh;
    if ly.density <= 1e-4 || share == sh::SHRUB {
        return 0.0;
    }
    let x = ly.cell * ly.resolve / gsd;
    let cone = shape == crate::kernels::shape::CONE;
    // (the crowns' shading shows only once they span 3-5 pixels: their mean's shading stays
    // until then; the tropics' big closed crowns shade from 1.5-3)
    let mean =
        if share == sh::TROPICAL { 0.85 * (1.0 - smoothstep(1.2, 3.0, x)) + 0.15 * (1.0 - smoothstep(2.0, 8.0, x)) } else { 1.0 - smoothstep(1.5, 4.5, x) };
    if mean <= 0.0 {
        return 0.0;
    }
    let d = if share == sh::TROPICAL {
        0.2
    } else if cone {
        0.6
    } else {
        0.33
    };
    // in proportion to the layer's cover (linear: LOD)
    d * mean_cover(ly) * mean
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
    // (cold climates need less rain for a closed forest: the boreal forest of dry continental
    // interiors, not the high mountains; the moisture window shifts down below 8 C, as far as
    // v0.2.0's closed taiga at a third of the moisture; and the temperature window by 2-3 C:
    // the taiga of a -1 C lowland is closed, not half open)
    let cold = (1.0 - smoothstep(0.0, 8.0, temp)) * (1.0 - smoothstep(0.35, 0.7, t.mountain));
    let base_cover =
        smoothstep(0.3 - 0.2 * cold, 0.68 - 0.36 * cold, wet) * smoothstep(-6.0 - 3.0 * cold, 2.0 - 2.0 * cold, temp) * veg.tree_density * trees_mult;
    // ~[0, 1]
    let fpu = 0.5 + 0.5 * pf.forest;
    // forests where the patch field is below the cover fraction (noisy edges, feathered over
    // ~70 m: the trees thin out into the open land in clumps; a crisp edge drew dark forest blobs
    // cut out of the grass; round 13: a 100 m feather of lone crowns dotted it), and the
    // farmland's woodlots whatever the climate's cover (they were bare meadow with a few trees
    // where the forest share is low)
    let forest = smoothstep(-0.06, 0.06, base_cover - fpu).max(m.woodlot * smoothstep(0.05, 0.4, s.agri()));
    // savanna / steppe scattered trees, groves
    let savanna = smoothstep(0.18, 0.35, wet) * (1.0 - smoothstep(0.55, 0.7, wet)) * smoothstep(12.0, 20.0, temp) * 0.12 * s.veg(|v| v.savanna);
    let groves = 0.04 * smoothstep(0.15, 0.3, wet) * s.veg(|v| v.groves);
    let base = m.natural_ok * (1.0 - m.field_cov) * veg.tree_density;
    // farmland is cleared except on steep ground (and its woodlots): what cannot be a field
    // there is meadow, not forest
    let clear = 1.0 - 0.85 * smoothstep(0.05, 0.4, s.agri()) * (1.0 - smoothstep(0.45, 0.7, slope)) * (1.0 - m.woodlot);
    let mut dens = (forest * 0.9 * clear + savanna + groves) * base;
    let mut dens_sav = savanna * base;
    dens = dens.max(m.riparian * veg.tree_density);
    // drainage lines carry scrub (crowns clipped into slivers there made spikes)
    let gully_scrub = 0.55 * smoothstep(0.2, 0.9, -t.gully) * smoothstep(0.2, 0.5, wet) * m.natural_ok * (1.0 - m.field_cov);
    // steep slopes, the tree line, the snow line, what claims the ground above
    let limit = (1.0 - smoothstep(0.9, 1.4, slope))
        * (1.0 - smoothstep(0.0, 0.6, t.mountain * smoothstep(-2.0, -6.0, temp)))
        * (1.0 - smoothstep(-1.2, -2.4, temp + 1.0 * pf.snow))
        * m.veg;
    dens *= limit;
    dens_sav *= limit;
    // shrubs / bushes in steppe, maquis and rocky slopes, sparse down to the semi-desert (none
    // below a moisture of 0.15 left semi-arid shrubland a bare brown plain)
    let shrub_clim = smoothstep(0.04, 0.26, wet) * (1.0 - smoothstep(0.6, 0.8, wet)) * smoothstep(2.0, 10.0, temp);
    // (in dry land a sparse scatter between the patches too)
    let shrub_patch = smoothstep(-0.2, 0.5, pf.patch + 0.4 * pf.land).max(0.6 * (1.0 - smoothstep(0.2, 0.35, wet)));
    let shrub = ((0.45 * shrub_clim * shrub_patch * (1.0 - forest) * m.natural_ok.max(0.4 * m.rock) * (1.0 - m.field_cov) * s.veg(|v| v.shrubs) * style.shrubs
            + gully_scrub)
            * veg.tree_density
            * m.veg
            // (round 12: in clumps over tens to hundreds of metres in dry land: the far semi-arid
            // plain read as bare, a16)
            * (1.0 + 0.9 * (1.0 - smoothstep(0.3, 0.45, wet)) * crate::layers::ground::semi_clumps(p, gsd)))
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
    // (round 13) groves, clumps and closed stands with open ground between them instead of a
    // uniform scatter of lone crowns; the savanna's own trees stay mostly lone, shrubs mottle
    // the ground in patches
    // (lone trees only in the savanna biomes: the warm climate's savanna share dotted steppe and
    // desert with lone trees, audit a14 a21)
    let sav_biome = smoothstep(1.05, 1.5, s.veg(|v| v.savanna));
    let sav_frac = if dens > 0.0 { (dens_sav / dens).min(1.0) * sav_biome } else { 0.0 };
    // (dense forest keeps its canopy: clearings cut by the clumps read as moth-eaten; and the
    // heights of clumped closed stands, resolved or a mean, broke the LOD at the airport)
    let cf = clump_field(p, gsd);
    let dens_c = clumped_at(dens, 0.9 * (1.0 - 0.7 * sav_frac) * (1.0 - smoothstep(0.3, 0.75, dens)), cf);
    let dens_sav_c = clumped_at(dens_sav, 0.9 - 0.6 * sav_biome, cf);
    let shrub_c = clumped_at(shrub, 0.6, cf);
    // (the species of a mixed stand in patches, not tree by tree: dark spruces dotted among pale
    // broadleaves, audit a25; conifers where the clump field is high, on the stands' sparser
    // ground: one field, a second one grew the pass's compile time)
    let conifer_c = clumped_at(conifer, 1.0, cf.map(|(z, u2)| (-z, u2)));
    let reg = &s.sm.registry;
    let crowns = &reg.biomes[s.bio.a as usize].crowns;
    let mut layers =
        [TreeLayer { cell: 1.0, seed: 0, density: 0.0, closure: 0.0, height: 0.0, color: DVec3::ZERO, shape: 0, scale: 1.0, tone: DVec3::ONE, resolve: 1.0 };
            crate::registry::MAX_CROWNS];
    let n = crowns.len().min(layers.len());
    for (k, c) in crowns.iter().enumerate().take(n) {
        let share = match c.share {
            share::CONIFER => dens_c * conifer_c,
            share::BROADLEAF => dens_c * (1.0 - conifer_c) * (1.0 - tropic),
            share::TROPICAL => dens_c * tropic,
            share::SHRUB => shrub_c,
            share::SAVANNA => dens_sav_c,
            _ => dens_c,
        };
        let colour = match c.colour_dry {
            Some(d) => mixc(c.colour, d, dry),
            None => c.colour,
        };
        // (shrubs a third of the way to the ground's colour: dark dots of 1-2 pixels over bright
        // grass read as noise rather than texture)
        let colour = if c.share == share::SHRUB { mixc(colour, s.col, 0.35) } else { colour };
        layers[k] = TreeLayer {
            cell: c.cell,
            seed: c.seed,
            density: share * c.density,
            closure: if c.closure { dens_c } else { 0.0 },
            height: (c.h0 + c.h_tall * tall) * (c.open_height + (1.0 - c.open_height) * stand_d),
            color: colour * style.crown,
            shape: c.shape,
            scale: if c.stand { scale } else { 1.0 },
            tone: if c.stand { stand_tone } else { DVec3::ONE },
            // (shrubs, dwarf shrubs: in the ground's texture, not one by one: dark specks of
            // 1-3 pixels with their shading and shadows read as noise, round 13)
            resolve: if c.share == share::SHRUB { 0.4 } else { 1.0 },
        };
    }
    let layers = &layers[..n];
    // (round 15) the relief of crowns not yet drawn one by one: resolved, they are domes and
    // cones the renderer shades (sunlit and shaded sides, shadows on their neighbours and the
    // ground); as their mean, a flat canopy, up to twice as bright: forests stepped from light
    // to dark where the view passed from one zoom to the next (z14 → z15). Their mean carries
    // the expected shading instead (`crown_relief`).
    let relief = layers.iter().zip(crowns).map(|(ly, c)| crown_relief(c.share, c.shape, ly, gsd)).sum::<f64>().min(0.65);
    // riparian belts: lush grass and scrub under their trees (the dry land's bare soil between
    // them read as brown streaks along the streams)
    if m.riparian > 0.0 {
        s.composite(Layer::paint(0.85 * m.riparian * limit, mixc(s.pal(pal::GRASS_WET), s.pal(pal::FLOOR), 0.4)));
    }
    // forest floor: shaded litter and understory, not sunlit grass, between the crowns
    let floor = smoothstep(0.25, 0.8, dens_c);
    s.composite(Layer::paint(floor * 0.85, s.pal(pal::FLOOR) * 0.7));
    // (round 13) closed stands: the gaps between the crowns show shaded lower crowns and
    // understorey, not the pale floor (the crowns read as dark specks on the lichen or the grass,
    // audit a26 a22); colour only, in clusters (a smooth sheet read as plastic); round 15: also
    // under the mean of unresolved crowns (the same gaps)
    {
        let (mut fill_w, mut fill_col, mut fill_n) = (0.0f64, DVec3::ZERO, 0.0);
        for layer in layers {
            if layer.density <= 1e-4 {
                continue;
            }
            // (round 15: at every zoom: the mean's gaps showed the pale floor, a far boreal forest
            // a pale smear over the lichen above its dark resolved crowns)
            let fe = smoothstep(0.55, 0.9, layer.closure);
            if fe > 0.0 {
                fill_w = fill_w.max(fe);
                fill_col += layer.color * layer.tone * (layer.density * fe);
                fill_n += layer.density * fe;
            }
        }
        if fill_w > 0.0 {
            s.composite(Layer::paint(fill_w, fill_col / fill_n * (0.6 * (1.0 + 0.5 * s.pf.detail))));
        }
    }
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
        // (the forest stand is the instance: not for shrubs)
        let inst = if cls == lc::SHRUB { 0 } else { crate::stack::instance_id(stand_id) };
        s.composite(Layer { cov: tcov, albedo: tc, dh: th, hmode: hm, ..Default::default() });
        // (round 15: the class by the canopy's cover, dithered on a 1.5 m lattice, not where it
        // covers half the pixel: the mean of unresolved crowns (cover ~0.6-1) made a stand all
        // forest, its resolved crowns only their own share; LOD of the class shares)
        let u = u01k(hash2(0xC1A55, (s.q_loc.x / 1.5).floor() as i64, (s.q_loc.y / 1.5).floor() as i64), 1);
        if u < tcov {
            s.class = cls;
            s.instance = inst;
        }
    }
    if relief > 0.0 {
        s.tint(DVec3::splat(1.0 - relief));
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
