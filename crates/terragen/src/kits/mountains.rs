//! The mountains kit (`docs/design/terrain-next.md` §5.3, §5.4, `docs/kits/mountains.md`): karst,
//! folded and fault-block ranges, canyons, badlands, glacial troughs with cirques and tarns,
//! fjords and alluvial fans. The WGSL twins are `gpu/wgsl/kits/mountains_relief.wgsl` (zones,
//! relief, carve hook, fans) and `mountains.wgsl` (the layer): the same functions in the same
//! order of operations.
//!
//! **Provinces.** Where a landform lives is a *zone* (0..1, [`zones`]) from smooth fields only:
//! the atlas' lithology at the pixel's nearest texel (looked up at a position warped by two
//! noises of 14 and 11 km, so the 20 km texel borders are ragged), climate (moisture,
//! temperature at the smooth elevation), the mountain mask, the tectonic uplift, the continent
//! value and the coast distance. A zone is crisp where it matters (a tower is there or not); only
//! its extent is smooth.
//!
//! **Relief (pass A, [`relief`]).**
//! * *Tower and cockpit karst* on carbonate (humid and warm: towers; humid and cooler: cockpits):
//!   one 3D Worley lattice of 420 m cells. Towers: the floor is flattened towards the 5 km smooth
//!   elevation, 70 % of the cells carry a tower `H·(1 − F1/R)^0.6` (H 90–260 m, R 0.4–0.7 cell, an
//!   oval ragged outline, clipped at the bisector so it never meets a neighbour's cut); cockpits:
//!   a cone hill in every cell (H 60–160 m) with the star-shaped depressions between them.
//!   Unresolved cells show the calibrated mean. Near the coast the plain is drowned (down to
//!   −11 m): the towers are islands.
//! * *Folded ridges* (old orogens and foreland belts: the mountain mask with little tectonic
//!   uplift, or a modest positive uplift beside it) and *fault blocks* (extensional provinces:
//!   negative uplift): parallel ridges `dot(p − corner, N)/λ` on an 8-corner lattice of 140 km
//!   cells (each corner its own strike, wavelength and phase; the weights sharpened), so the
//!   stripes keep their phase at any zoom (R-wave); a 38 km noise bends them, a 26 km noise
//!   makes the ridges plunge. Folds: triangle crests, λ 6–12 km, 520 m; blocks: tilted
//!   sawtooth, λ 11–21 km, 900 m, a scarp on a fifth of the period.
//! * *Badlands*: three octaves of creased noise (300, 130 and 60 m) on soft sediment × arid.
//! * *Alluvial fans*: family `mountains_fans` at mountain fronts in dry climates (the atlas'
//!   uplift falling at least 0.03 per 24 km: the fan points down the gradient): a
//!   conical apron `H(1 − r/R)^1.3` over a 70 % blend to the smooth elevation in a ±60° sector.
//!
//! **Carve hook ([`carve`], `kits::Carve`).** The core carves every channel's valley; the kit
//! changes its profile where the ground says so, never the water rules: *glacial troughs* (U
//! profile, deeper by 2.6 on rivers, 8 on tributaries, wider floors, over-deepened), *cirques*
//! (a round floor at the head of a glaciated tributary, a *tarn* where the head is), *hanging
//! tributaries* (the shallower side valleys end above the deeper trough), *fjords* (the floor
//! of a glaciated coastal river goes to −70 … −530 m for 40–70 km inland and the ocean floods it;
//! it fades with the land mask into the open sea), *stepped canyons* (arid sedimentary plateaus:
//! floors lowered by up to 1.2 km along rivers, steep walls with benches of 40–70 m).
//!
//! **Pass B.** Slot 6 (it is the only stage with compile headroom; the layer is composited after
//! the canopy, which suits rock and sediment): banded badland and mesa strata, karst cliffs,
//! glacial rock walls, the pale radial fans.

use crate::instances::{Family, InstSite, Instance};
use crate::kits::{Carve, CarveIn, Kit, ReliefIn};
use crate::noise::*;
use crate::stack::{slot, Layer, Stack};
use crate::surface::{mixc, srgb};
use crate::world::{Ctx, Macro, World};
use glam::DVec3;
use std::f64::consts::TAU;

pub static KIT: Kit = Kit {
    name: "mountains",
    biomes_yaml: "",
    wgsl: include_str!("../gpu/wgsl/kits/mountains.wgsl"),
    layers: &[(slot::LINEAR, linear)],
    relief: Some(relief),
    relief_wgsl: include_str!("../gpu/wgsl/kits/mountains_relief.wgsl"),
    carve: Some(carve),
    families: &[Family { name: "mountains_fans", cell: 18_000.0, jitter: 0.8, reach: 7_200.0, relief: true, exists: fans_exist }],
    ..Kit::EMPTY
};

/// Land-cover v2 class ids (`registry::classes`; checked by a test).
mod cls {
    pub const ROCK: u8 = 6;
    pub const GRAVEL: u8 = 33;
    pub const BADLANDS: u8 = 34;
    pub const CLIFF: u8 = 37;
}

const SEED: u64 = 0x4D54_5000;

/// Lithology codes of the atlas (`atlas::Lithology`).
const SEDIMENTARY: u32 = 0;
const CARBONATE: u32 = 1;
const CRYSTALLINE: u32 = 2;
const VOLCANIC: u32 = 3;
const UNCONSOLIDATED: u32 = 4;

// ============================================================================= fields and zones

/// Wavelengths (m) of the two noises that warp the lithology lookup, and the warp (m).
const WARP_LAM: [f64; 2] = [14_000.0, 11_000.0];
const WARP_A: f64 = 4_000.0;

/// The atlas' dominant lithology at the texel nearest to `p` (one load; WGSL `mt_litho`).
pub(crate) fn litho_at(w: &World, p: DVec3) -> u32 {
    use crate::atlas::{slot, APRON};
    let a = w.atlas();
    let r = a.resolution();
    let rf = r as f64;
    let (f, x, y) = crate::atlas::geom::Grid::new(r).locate(p.normalize());
    let near = |x: f64| ((x.clamp(-0.5, rf - 0.5) + 0.5).floor() as usize).min(r - 1);
    (a.raw(f, near(x) + APRON, near(y) + APRON, slot::LITHO) & 7) as u32
}

/// The patch noises and the lithology at the warped position (WGSL `mt_fields`).
pub(crate) fn fields(w: &World, c: &Ctx) -> (u32, f64, f64) {
    let n1 = perlin3(SEED + 1, c.p / WARP_LAM[0]);
    let n2 = perlin3(SEED + 2, c.p / WARP_LAM[1]);
    let q = c.p + c.east * (WARP_A * n1) + c.north * (WARP_A * n2);
    (litho_at(w, q), n1, n2)
}

/// What the zones read (smooth fields at a pixel).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ZIn {
    pub moist: f64,
    pub temp: f64,
    pub mountain: f64,
    pub uplift: f64,
    pub cont: f64,
    pub coast_km: f64,
    pub sand: f64,
    /// a regional noise (180 km): which old belts are folded (0 in pass B, which does not draw them)
    pub style: f64,
}

/// Zone weights 0..1 of the landforms (and the drowning of the karst plain at the coast).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Zones {
    pub tower: f64,
    pub cockpit: f64,
    pub fold: f64,
    pub fault: f64,
    pub badland: f64,
    pub drown: f64,
}

/// The zones at a pixel: `lit` the lithology, `n1`, `n2` the patch noises.
pub(crate) fn zones(z: &ZIn, lit: u32, n1: f64, n2: f64) -> Zones {
    let carb = (lit == CARBONATE) as u32 as f64;
    let soft = (lit == SEDIMENTARY || lit == UNCONSOLIDATED) as u32 as f64;
    let hard = (lit == CRYSTALLINE || lit == VOLCANIC) as u32 as f64;
    let land = smoothstep(0.0, 0.08, z.cont);
    let shelf = smoothstep(-0.25, -0.1, z.cont) * smoothstep(-30.0, -12.0, z.coast_km);
    let patch_a = smoothstep(-0.45, -0.05, n1);
    let patch_b = smoothstep(-0.2, 0.2, n2);
    let humid = smoothstep(0.42, 0.58, z.moist) * smoothstep(15.0, 21.0, z.temp);
    let tower = carb * humid * (1.0 - smoothstep(0.35, 0.7, z.mountain)) * patch_a * shelf;
    let cockpit =
        carb * smoothstep(0.28, 0.42, z.moist) * smoothstep(4.0, 12.0, z.temp) * (1.0 - humid) * (1.0 - smoothstep(0.4, 0.7, z.mountain)) * patch_a * land;
    let u = z.uplift.max(0.0);
    let old = smoothstep(0.1, 0.4, z.mountain) * (1.0 - smoothstep(0.2, 0.5, u));
    let fore = smoothstep(0.04, 0.15, u) * (1.0 - smoothstep(0.3, 0.5, u)) * (1.0 - 0.7 * z.mountain);
    let fold = old.max(fore) * (1.0 - 0.5 * hard) * land * (1.0 - tower) * smoothstep(-0.1, 0.3, z.style);
    let fault = smoothstep(-0.04, -0.18, z.uplift) * smoothstep(0.08, 0.25, z.cont) * (1.0 - 0.6 * smoothstep(0.3, 0.7, z.mountain));
    let badland = soft
        * (1.0 - smoothstep(0.14, 0.28, z.moist))
        * smoothstep(2.0, 10.0, z.temp)
        * (1.0 - smoothstep(0.3, 0.6, z.mountain))
        * (1.0 - smoothstep(0.05, 0.3, z.sand))
        * patch_b
        * land;
    Zones { tower, cockpit, fold, fault, badland, drown: 1.0 - smoothstep(-1.0, 10.0, z.coast_km) }
}

// ============================================================================= karst

/// Karst cell (m), the share of cells with a tower, and the calibrated means (m) of the tower
/// field and of the cockpit ridge field (a test recomputes them).
const KARST_CELL: f64 = 420.0;
const TOWER_DENS: f64 = 0.7;
const TOWER_MEAN: f64 = 31.51;
const COCKPIT_MEAN: f64 = 15.24;

/// A cone's height (m) at `p`, `H (1 − F1/R)^e` with R varying with the azimuth (an oval, ragged
/// outline) and clipped at the bisector (no cut where two cells meet). `kind` 0: towers (70 % of
/// the cells, H 110–300 m, `√(1 − x^2.5)`: a round top, steep walls); 1: the cone hills of cockpit karst (every cell,
/// H 60–160 m, e 1.0: the star-shaped depressions lie between them).
#[inline]
fn cone_h(c: &Cell3, p: DVec3, east: DVec3, north: DVec3, kind: u32) -> f64 {
    let tower = kind == 0;
    if tower && u01k(c.id, 1) >= TOWER_DENS {
        return 0.0;
    }
    let hh = if tower { 110.0 + 190.0 * u01k(c.id, 2) } else { 60.0 + 100.0 * u01k(c.id, 4) };
    // (the outline from the azimuth's polynomial harmonics: `atan2` and `sin` are only good to
    // ~1e-4 on the GPU, which shows as centimetres in a tower's height)
    let d = p - c.point;
    let (dx, dy) = (d.dot(east), d.dot(north));
    let r = (dx * dx + dy * dy).sqrt().max(1e-6);
    let (co, si) = (dx / r, dy / r);
    let (c2, s2) = (co * co - si * si, 2.0 * co * si);
    let (c4, s4) = (c2 * c2 - s2 * s2, 2.0 * c2 * s2);
    let (sa, ca) = (TAU * u01k(c.id, 5)).sin_cos();
    let (sb, cb) = (TAU * u01k(c.id, 6)).sin_cos();
    let r0 = if tower { 0.75 + 0.5 * u01k(c.id, 3) } else { 1.1 + 0.4 * u01k(c.id, 3) };
    let rad = 0.5 * KARST_CELL * r0 * (1.0 + 0.2 * (s2 * ca + c2 * sa) + 0.1 * (s4 * cb + c4 * sb));
    let x = c.f1 / rad.min(0.9 * c.f2);
    if x >= 1.0 {
        0.0
    } else {
        // (a tower: a round top and walls that steepen towards the foot, `√(1 − x^2.5)`)
        if tower {
            hh * (1.0 - x * x * x.sqrt()).sqrt()
        } else {
            hh * (1.0 - x)
        }
    }
}

// ============================================================================= ridges

const RIDGE_CELL: f64 = 140_000.0;

/// Parallel ridges (m) of the 8 lattice corners around `p`, blended by their (sharpened)
/// trilinear weights: folds (crest `t^1.6`, λ 6-12 km, 520 m) and fault blocks (a tilted sawtooth,
/// λ 11-21 km, 900 m) mixed by the share `wf` of fault blocks (continuous: where both zones
/// overlap the profiles blend). Zero mean; `nw` bends the stripes (cycles), `pl` is the plunge
/// amplitude 0..1.
fn ridges(p: DVec3, wf: f64, nw: f64, pl: f64, belt: DVec3, wg: f64) -> f64 {
    let q = p / RIDGE_CELL;
    let i = q.floor();
    let f = q - i;
    let (ix, iy, iz) = (i.x as i64, i.y as i64, i.z as i64);
    let sm = |x: f64| x * x * (3.0 - 2.0 * x);
    let (sx, sy, sz) = (sm(f.x), sm(f.y), sm(f.z));
    let (mut v, mut sw) = (0.0, 0.0);
    for dz in 0..2i64 {
        for dy in 0..2i64 {
            for dx in 0..2i64 {
                let wt = (if dx == 0 { 1.0 - sx } else { sx }) * (if dy == 0 { 1.0 - sy } else { sy }) * (if dz == 0 { 1.0 - sz } else { sz });
                let w = wt * wt;
                let h = hash3(SEED + 10, ix + dx, iy + dy, iz + dz);
                let corner = DVec3::new((ix + dx) as f64, (iy + dy) as f64, (iz + dz) as f64) * RIDGE_CELL;
                let d = corner.normalize();
                let e = DVec3::Z.cross(d);
                let e = if e.length() < 1e-6 { DVec3::X } else { e.normalize() };
                let n = d.cross(e);
                // the strike: the belt's normal (the atlas' uplift gradient, `belt`, weight `wg`),
                // else a direction shared by blocks of 4 × 4 × 4 corners (560 km) with a little jitter
                let hb = hash3(SEED + 11, ((ix + dx) as f64 / 4.0).floor() as i64, ((iy + dy) as f64 / 4.0).floor() as i64, ((iz + dz) as f64 / 4.0).floor() as i64);
                let th = TAU * (u01k(hb, 1) + 0.18 * (u01k(h, 1) - 0.5));
                let nn = (e * th.cos() + n * th.sin()) * (1.0 - wg) + belt * wg;
                let lamk = 0.7 + 0.6 * u01k(h, 2);
                let along = (p - corner).dot(nn);
                let ph = u01k(h, 3) + 1.4 * nw;
                let xf = along / (9000.0 * lamk) + ph;
                let xk = along / (16000.0 * lamk) + ph;
                let xf = xf - xf.floor();
                let xk = xk - xk.floor();
                let t = 1.0 - (2.0 * xf - 1.0).abs();
                let pf = 700.0 * (t.powf(1.6) - 1.0 / 2.6);
                // (a flat basin floor on the first 30 % of the period, then the tilted block, then the scarp; the
                // constant 0.15 keeps the mean at zero)
                let pk = 1150.0
                    * if xk < 0.3 {
                        -0.35
                    } else if xk < 0.8 {
                        let s = (xk - 0.3) / 0.5;
                        s * s * (3.0 - 2.0 * s) - 0.35
                    } else {
                        let s = (1.0 - xk) / 0.2;
                        s * s * (3.0 - 2.0 * s) - 0.35
                    };
                v += w * (0.7 + 0.6 * u01k(h, 4)) * lerp(pf, pk, wf);
                sw += w;
            }
        }
    }
    v / sw * (0.25 + 0.75 * pl)
}

// ============================================================================= badlands

/// A smooth crease of a noise value: `|n|^0.7` with the V rounded off (`(n² + 0.04)^0.35`): a pit
/// at a crease would alias at a few metres per pixel. Its mean (a test recomputes it).
#[inline]
fn crease(n: f64) -> f64 {
    (n * n + 0.04).powf(0.35)
}

const CREASE_MEAN: f64 = 0.4097;

/// Creased gullies (m per unit amplitude): 300, 130 and 60 m octaves (weights 1, 0.55, 0.25: the
/// finer creases branch off the coarser ones), zero mean, band-limited.
fn badlands(p: DVec3, gsd: f64) -> f64 {
    let mut v = 0.0;
    for (k, (lam, wt)) in [(300.0, 1.0), (130.0, 0.55), (60.0, 0.25)].into_iter().enumerate() {
        let b = band(lam, gsd);
        if b > 0.0 {
            v += wt * (crease(perlin3(SEED + 20 + k as u64, p / lam)) - CREASE_MEAN) * b;
        }
    }
    v
}

// ============================================================================= fans

/// An alluvial fan at `p`: (sector weight 0..1 × radial edge, r / R, the azimuth of `p` from the
/// apex).
fn fan_at(it: &Instance, p: DVec3, east: DVec3, north: DVec3) -> Option<(f64, f64, f64)> {
    let rel = p - it.center;
    let (x, y) = (rel.dot(east), rel.dot(north));
    let r = (x * x + y * y).sqrt();
    let rr = it.v[0] as f64;
    if r >= 1.25 * rr {
        return None;
    }
    // (a lobed outline from the polynomial harmonics 3 and 6 of the azimuth)
    let th = y.atan2(x);
    let (co, si) = (x / r.max(1e-6), y / r.max(1e-6));
    let (c3, s3) = (4.0 * co * co * co - 3.0 * co, 3.0 * si - 4.0 * si * si * si);
    let (c6, s6) = (c3 * c3 - s3 * s3, 2.0 * s3 * c3);
    let (sa, ca) = (TAU * u01k(it.id, 7)).sin_cos();
    let (sb, cb) = (TAU * u01k(it.id, 8)).sin_cos();
    let k = 1.0 + 0.15 * (s3 * ca + c3 * sa) + 0.08 * (s6 * cb + c6 * sb);
    let rn = r / (rr * k);
    if rn >= 1.0 {
        return None;
    }
    let cs = (x * it.v[1] as f64 + y * it.v[2] as f64) / r.max(1.0);
    let sector = lerp(1.0, smoothstep(0.0, 0.5, cs), smoothstep(0.0, 0.05, rn));
    Some((sector * smoothstep(0.0, 0.1, 1.0 - rn), rn, th))
}

/// Fans exist at the fronts of dry mountains: the atlas' tectonic uplift is 0.12 … 0.55 and falls
/// by at least 0.03 over 24 km (the fan points down the gradient). v: R (m), direction east and
/// north, H (m). Atlas fields only (a macro-field sample costs 15 k instructions per call site).
fn fans_exist(w: &World, s: &InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 1) >= 0.4 {
        return None;
    }
    let d = 12_000.0;
    let mut up = [0.0f64; 5];
    for (k, u) in up.iter_mut().enumerate() {
        let (de, dn) = [(0.0, 0.0), (d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)][k];
        let a = w.atlas().sample((s.center + s.east * de + s.north * dn).normalize());
        if k == 0 && (a.coast_km < 5.0 || World::moisture_index(a.precip_mm, a.temp_c) > 0.33 || a.temp_c < 0.0 || a.uplift < 0.12 || a.uplift > 0.55) {
            return None;
        }
        *u = a.uplift;
    }
    let (gx, gy) = (up[1] - up[2], up[3] - up[4]);
    let gl = (gx * gx + gy * gy).sqrt();
    if gl < 0.03 {
        return None;
    }
    let rr = 2600.0 + 3000.0 * u01k(s.id, 2);
    let hh = rr * (0.016 + 0.014 * u01k(s.id, 3));
    Some([rr as f32, (-gx / gl) as f32, (-gy / gl) as f32, hh as f32, 0.0, 0.0, 0.0, 0.0])
}

fn fam_fans() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| crate::instances::families().position(|f| f.name == "mountains_fans").expect("the fans family"))
}

// ============================================================================= relief

/// The kit's relief: karst, folds, fault blocks, badlands, fans.
pub fn relief(w: &World, r: &ReliefIn, h: &mut f64) {
    let c = r.ctx;
    let m = r.m;
    let gsd = c.gsd;
    if m.cont < -0.3 {
        return;
    }
    let (lit, n1, n2) = fields(w, c);
    let zi = ZIn {
        moist: r.moist,
        temp: r.temp,
        mountain: r.mountain,
        uplift: m.uplift,
        cont: m.cont,
        coast_km: m.coast_km,
        sand: r.sand,
        style: perlin3(SEED + 5, c.p / 180_000.0),
    };
    let z = zones(&zi, lit, n1, n2);
    // the flattening target: the atlas' smooth elevation (exact and the same in every tile; the
    // pixel's own low-passed relief carries the tile grid's interpolation error: steps of ~10 m
    // between tiles once it is the target of a 85 % blend)
    // (and, in the folds and fault blocks, the belt's normal: the gradient of the atlas' uplift over
    // ±25 km, east and north)
    let want = z.fold + z.fault > 0.0;
    let mut ae = 0.0;
    let mut up = [0.0f64; 4];
    if z.tower + z.cockpit + z.fold + z.fault > 0.0 || r.instances(w, fam_fans()).iter().any(|_| true) {
        ae = w.atlas().sample(c.p.normalize()).elevation_m;
    }
    if want {
        for (k, u) in up.iter_mut().enumerate() {
            let (de, dn) = [(25_000.0, 0.0), (-25_000.0, 0.0), (0.0, 25_000.0), (0.0, -25_000.0)][k];
            *u = w.atlas().sample((c.p + c.east * de + c.north * dn).normalize()).uplift;
        }
    }
    // ---- karst: flattened floors, towers, dolines; drowned near the coast
    if z.tower + z.cockpit > 0.0 {
        let plain = lerp(ae, (-11.0f64).min(ae), z.drown);
        *h = lerp(*h, plain, 0.88 * z.tower);
        let e = smoothstep(1.2 * gsd, 3.0 * gsd, 0.45 * KARST_CELL);
        let (mut tw, mut cp) = (TOWER_MEAN, COCKPIT_MEAN);
        if e > 0.0 {
            let wc = worley3(SEED + 30, c.p, KARST_CELL, 0.85);
            tw = lerp(TOWER_MEAN, cone_h(&wc, c.p, c.east, c.north, 0), e);
            cp = lerp(COCKPIT_MEAN, cone_h(&wc, c.p, c.east, c.north, 1), e);
        }
        *h += z.tower * tw + z.cockpit * (cp - COCKPIT_MEAN);
    }
    // ---- folded ridges and fault blocks
    let zt = (z.fold + z.fault).min(1.0);
    if z.fold + z.fault > 0.0 {
        let wf = z.fault / (z.fold + z.fault);
        let b = lerp(band(6300.0, gsd), band(7700.0, gsd), wf);
        *h = lerp(*h, ae, 0.85 * zt);
        if b > 0.0 {
            let nw = perlin3(SEED + 3, c.p / 38_000.0);
            let pl = smoothstep(-0.55, 0.3, perlin3(SEED + 4, c.p / 26_000.0));
            let (gx, gy) = (up[0] - up[1], up[2] - up[3]);
            let gl = (gx * gx + gy * gy).sqrt();
            let wg = smoothstep(0.02, 0.06, gl);
            let belt = (c.east * gx + c.north * gy) / gl.max(1e-9);
            *h += zt * ridges(c.p, wf, nw, pl, belt, wg) * b;
        }
    }
    // ---- badlands
    if z.badland > 0.0 {
        *h += z.badland * (14.0 + 14.0 * smoothstep(-0.3, 0.5, n1)) * 2.4 * badlands(c.p, gsd);
    }
    // ---- alluvial fans
    for it in r.instances(w, fam_fans()).iter() {
        if let Some((sec, rn, _)) = fan_at(it, c.p, c.east, c.north) {
            let cone = it.v[3] as f64 * (1.0 - rn).powf(1.3);
            *h = lerp(*h, ae + cone, 0.7 * sec);
        }
    }
}

// ============================================================================= carve

/// The fjord weight 0..1 of a glaciated coast and the depth (m) of its floor: from the mouth (a
/// sill at −70 m) down to −330 … −530 m 12 km inland and up to the head 20-70 km from the coast.
fn fjord(m: &Macro, mountain: f64, glac: f64) -> (f64, f64) {
    let ck = m.coast_km;
    let head = 30.0 + 40.0 * m.glaciation;
    let fj = glac * smoothstep(0.2, 0.45, mountain) * (1.0 - smoothstep(head - 6.0, head, ck)) * smoothstep(-6.0, 0.0, ck);
    let prof = smoothstep(0.0, 12.0, ck) * (1.0 - smoothstep(head - 30.0, head, ck));
    (fj, 70.0 + (260.0 + 220.0 * mountain) * prof)
}

/// Glacial troughs, cirques with tarns, fjords and stepped canyons (the profile of the core's
/// valley carve; `mountains_carve` in WGSL).
pub fn carve(_w: &World, ci: &CarveIn, cv: &mut Carve) {
    if ci.level > 1 {
        return;
    }
    let m = ci.m;
    let lvl1 = ci.level == 1;
    let gsd = ci.ctx.gsd;
    // ---- glacial troughs: a U profile, wider floors, deeper (rivers 2.6, tributaries 8)
    let glac = smoothstep(0.3, 0.6, m.glaciation) * smoothstep(0.25, 0.55, ci.mountain);
    if glac > 0.0 {
        cv.u = cv.u.max(glac);
        cv.wall_k *= lerp(1.0, 0.75, glac);
        cv.fp_w *= lerp(1.0, 1.5, glac);
        cv.depth_cap *= lerp(1.0, if lvl1 { 8.0 } else { 2.0 }, glac);
        cv.floor = lerp(cv.floor, cv.floor.min((cv.floor - if lvl1 { 40.0 } else { 90.0 }).max(1.0)), glac);
        // cirques: a round floor at the head of a glaciated tributary, a tarn where the head is
        if lvl1 && ci.spring {
            let ch = glac * smoothstep(0.35, 0.7, ci.mountain);
            let cz = 1.0 - smoothstep(0.0, 0.14, ci.u);
            cv.fp_w = cv.fp_w.max(240.0 * ch * cz);
            cv.wall_k *= lerp(1.0, 0.8, ch * cz);
            let lr = 130.0 * ch * (1.0 - smoothstep(0.02, 0.06, ci.u));
            if lr > 0.4 * gsd {
                cv.lake_r = lr;
                cv.lake_level = cv.floor + 1.4;
            }
        }
    }
    // ---- fjords: the floor of a glaciated coastal river below sea level, for 40-70 km inland
    let (fj, dep) = fjord(m, ci.mountain, glac);
    let fj = fj * if lvl1 { 0.45 } else { 1.0 };
    if fj > 0.0 {
        cv.floor = lerp(cv.floor, -dep, fj);
        cv.depth_cap = lerp(cv.depth_cap, 1500.0, fj);
        cv.fp_min = 0.5 - 2000.0 * fj;
        // a narrow trough: a floor of ~1 km (the river's wide valley no longer sets the extent), steep walls
        cv.fp_w = lerp(cv.fp_w, (0.5 * cv.fp_w).min(450.0), fj);
        cv.wall_k *= 1.0 - 0.55 * fj;
        cv.narrow = fj;
        cv.u = cv.u.max(fj);
    }
    // ---- stepped canyons: arid plateaus (the weights are wide and the depth a constant: the
    // inputs differ by a few mm between the CPU and the GPU, and a canyon's depth multiplies them)
    let arid = 1.0 - smoothstep(0.15, 0.45, ci.moist);
    if arid > 0.0 {
        let cz = arid
            * smoothstep(100.0, 700.0, ci.h0)
            * (1.0 - smoothstep(0.3, 0.6, ci.mountain))
            * smoothstep(0.05, 0.3, ci.mesa)
            * smoothstep(0.0, 0.15, m.cont)
            * if lvl1 { 0.35 } else { 1.0 };
        if cz > 0.0 {
            let f0 = cv.floor;
            // (the depth by province, from the atlas' uplift and the plateau field: 400-1500 m)
            let prov = (0.6 * smoothstep(-0.05, 0.4, m.uplift) + 0.4 * smoothstep(0.2, 0.8, m.plateau)).clamp(0.0, 1.0);
            cv.floor = lerp(f0, (f0 - (400.0 + 1100.0 * prov)).max(f0.min(30.0)), cz);
            cv.depth_cap = lerp(cv.depth_cap, 1500.0, cz);
            cv.wall_k *= lerp(1.0, 0.3, cz);
            cv.fp_w *= lerp(1.0, 0.6, cz);
            cv.steps = 50.0;
            cv.step_mix = cv.step_mix.max(cz * band(160.0, gsd));
            // (the benches are counted from a fixed height above the floor, not from the ground)
            cv.top = cv.floor + 500.0;
        }
    }
}

// ============================================================================= pass B

/// Slot 6: banded badlands and strata of arid sedimentary walls, karst cliffs, glacial rock
/// walls, the pale aprons of the alluvial fans.
pub fn linear(s: &mut Stack) {
    let t = s.t;
    let ground = s.l.ground;
    let slope = s.l.slope;
    if ground < 0.5 {
        return;
    }
    let p = s.ctx.p;
    let (lit, n1, n2) = fields(s.world, s.ctx);
    let zi = ZIn { moist: t.moist, temp: t.temp, mountain: t.mountain, uplift: t.uplift, cont: t.cont, coast_km: t.coast_km, sand: t.sand, style: 0.0 };
    let z = zones(&zi, lit, n1, n2);
    let rn = s.pf.patch;
    // ---- badlands: pale clay floors, banded gully walls (and the strata of arid sedimentary walls)
    let arid = 1.0 - smoothstep(0.16, 0.34, t.moist);
    let soft = (lit == SEDIMENTARY || lit == CARBONATE || lit == UNCONSOLIDATED) as u32 as f64;
    let clay = z.badland * 0.85;
    if clay > 0.0 {
        let col = mixc(srgb(196.0, 184.0, 164.0), srgb(176.0, 160.0, 138.0), 0.5 + 0.5 * rn) * (1.0 + 0.05 * s.pf.detail);
        s.composite(Layer { cov: clay, albedo: col, cls: cls::BADLANDS, clear: 1.0, ..Default::default() });
        s.m.natural_ok *= 1.0 - clay;
    }
    let wall = smoothstep(0.42, 0.7, slope) * arid * soft;
    let bad = z.badland * smoothstep(0.2, 0.45, slope);
    let cb = bad.max(wall);
    if cb > 0.0 {
        let bandh = if z.badland > 0.2 { 9.0 } else { 14.0 };
        let kb = (ground / bandh + 0.6 * rn).floor();
        let h = hash1(SEED + 40, kb as i64);
        let pal = [
            srgb(204.0, 194.0, 176.0),
            srgb(168.0, 112.0, 84.0),
            srgb(216.0, 198.0, 160.0),
            srgb(168.0, 140.0, 118.0),
            srgb(184.0, 132.0, 108.0),
            srgb(188.0, 172.0, 148.0),
        ];
        let col = pal[(u01k(h, 1) * 5.999) as usize] * (0.9 + 0.2 * u01k(h, 2));
        let floor_col = srgb(190.0, 176.0, 150.0);
        let col = mixc(floor_col, col, 0.8 * smoothstep(0.25, 0.6, slope));
        s.composite(Layer { cov: cb, albedo: col, cls: if z.badland > 0.2 { cls::BADLANDS } else { cls::CLIFF }, clear: 1.0, ..Default::default() });
        s.m.natural_ok *= 1.0 - cb;
    }
    // ---- karst cliffs: pale grey rock on the steep flanks of towers and cones
    // ---- forest-clad towers: the core's rock on the walls (it starts at slope 0.55-0.85) is painted over
    // with a dark canopy colour, except on the steepest walls
    let cf = z.tower * smoothstep(0.3, 0.7, s.m.rock) * (1.0 - smoothstep(3.5, 6.0, slope + 0.8 * rn));
    if cf > 0.0 {
        let col = mixc(srgb(40.0, 60.0, 36.0), srgb(64.0, 86.0, 48.0), 0.5 + 0.5 * rn) * (1.0 + 0.15 * s.pf.detail);
        s.composite(Layer { cov: cf, albedo: col, cls: crate::landcover::BROADLEAF_FOREST, ..Default::default() });
    }
    // (towers are forest-clad: bare rock only on their steepest walls; cockpit cones keep the
    // lower threshold)
    let ck = (z.tower * smoothstep(3.5, 6.0, slope + 0.8 * rn) + z.cockpit * smoothstep(0.8, 1.4, slope + 0.2 * rn)).min(1.0);
    if ck > 0.0 {
        let col = mixc(srgb(168.0, 164.0, 150.0), srgb(120.0, 118.0, 108.0), 0.5 + 0.5 * rn) * (1.0 + 0.06 * s.pf.detail);
        s.composite(Layer { cov: ck, albedo: col, cls: cls::CLIFF, clear: 1.0, ..Default::default() });
        s.m.natural_ok *= 1.0 - ck;
    }
    // ---- glacial rock walls: scoured pale rock on the steep sides of glaciated mountains
    let glac = smoothstep(0.3, 0.6, t.glaciation) * smoothstep(0.25, 0.55, t.mountain);
    let cg = glac * smoothstep(0.75, 1.1, slope);
    if cg > 0.0 {
        let col = mixc(srgb(176.0, 174.0, 168.0), srgb(132.0, 130.0, 128.0), 0.5 + 0.5 * rn);
        s.composite(Layer { cov: cg, albedo: col, cls: cls::ROCK, clear: 1.0, ..Default::default() });
        s.m.natural_ok *= 1.0 - cg;
    }
    // ---- alluvial fans: pale gravel with radial streaks
    let inst = s.instances(fam_fans()).into_owned();
    for it in inst.iter() {
        if let Some((sec, rr, th)) = fan_at(it, p, s.ctx.east, s.ctx.north) {
            let streak = 0.5 + 0.5 * (18.0 * th + 6.0 * rn).sin();
            let v = sec * (1.0 - rr).powf(0.5) - 0.1 + 0.05 * s.pf.detail;
            let c = smoothstep(0.0, 0.05, v);
            if c > 0.0 {
                let col = mixc(srgb(206.0, 186.0, 152.0), srgb(176.0, 152.0, 122.0), 0.7 * streak + 0.2 * rr);
                s.composite(Layer { cov: c, albedo: col, cls: cls::GRAVEL, clear: 0.85, ..Default::default() });
                s.m.natural_ok *= 1.0 - c;
            }
            break;
        }
    }
}

// ============================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_ids() {
        use crate::registry::classes::id;
        assert_eq!(id("rock"), Some(cls::ROCK));
        assert_eq!(id("gravel"), Some(cls::GRAVEL));
        assert_eq!(id("badlands"), Some(cls::BADLANDS));
        assert_eq!(id("cliff"), Some(cls::CLIFF));
    }

    /// A point of a plane patch near the surface (Halton sequence, `n` of `size` m).
    fn plane(i: usize, size: f64) -> DVec3 {
        let x = crate::kernels::halton(i + 1, 2) * size;
        let y = crate::kernels::halton(i + 1, 3) * size;
        DVec3::new(6_371_000.0, 1_234_567.0 + x, 765_432.0 + y)
    }

    /// The calibrated means of the karst fields and of the creases are those of the explicit
    /// functions.
    #[test]
    fn means_match_explicit_averages() {
        let n = 120_000;
        let (mut t, mut c, mut b) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let p = plane(i, 60_000.0);
            let wc = worley3(SEED + 30, p, KARST_CELL, 0.85);
            let up = p.normalize();
            let east = DVec3::Z.cross(up).normalize();
            t += cone_h(&wc, p, east, up.cross(east), 0);
            c += cone_h(&wc, p, east, up.cross(east), 1);
            b += crease(perlin3(SEED + 20, p / 300.0));
        }
        let (t, c, b) = (t / n as f64, c / n as f64, b / n as f64);
        eprintln!("tower mean {t:.2} (const {TOWER_MEAN}), cockpit mean {c:.2} (const {COCKPIT_MEAN}), crease mean {b:.4} (const {CREASE_MEAN})");
        assert!((t - TOWER_MEAN).abs() < 0.04 * TOWER_MEAN, "tower mean {t}");
        assert!((c - COCKPIT_MEAN).abs() < 0.04 * COCKPIT_MEAN, "cockpit mean {c}");
        assert!((b - CREASE_MEAN).abs() < 0.02, "crease mean {b}");
    }

    /// Ridges have no mean, are bounded, and are continuous in the position (no step at the
    /// lattice cells, whatever the zoom).
    #[test]
    fn ridges_are_zero_mean_and_continuous() {
        for kind in [0u32, 1] {
            let n = 40_000;
            let (mut sum, mut lo, mut hi) = (0.0, f64::MAX, f64::MIN);
            for i in 0..n {
                let p = plane(i, 900_000.0);
                let v = ridges(p, kind as f64, 0.0, 1.0, DVec3::ZERO, 0.0) / [700.0, 1150.0][kind as usize];
                sum += v;
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let mean = sum / n as f64;
            eprintln!("ridges kind {kind}: mean {mean:.3}, range {lo:.2} .. {hi:.2}");
            assert!(mean.abs() < 0.05 && lo > -1.0 && hi < 1.2, "kind {kind}: mean {mean} range {lo} {hi}");
            // a line across several lattice cells: the largest step between points 20 m apart
            let mut worst: f64 = 0.0;
            let a = DVec3::new(6_371_000.0, 0.0, 0.0);
            let amp = [700.0, 1150.0][kind as usize];
            let mut prev = ridges(a, kind as f64, 0.0, 1.0, DVec3::ZERO, 0.0) / amp;
            for j in 1..30_000 {
                let v = ridges(a + DVec3::new(0.0, 20.0 * j as f64, 0.0), kind as f64, 0.0, 1.0, DVec3::ZERO, 0.0) / amp;
                worst = worst.max((v - prev).abs());
                prev = v;
            }
            // (the fault block's scarp falls over a fifth of 11 km: 0.02 per 20 m at most)
            assert!(worst < 0.1, "kind {kind}: step {worst} in 20 m");
        }
    }

    /// Zones are weights, and the lithology decides their kind.
    #[test]
    fn zones_are_weights() {
        for i in 0..4000u64 {
            let h = mix64(0x77 ^ i);
            let z = ZIn {
                moist: u01k(h, 1) * 1.2 - 0.1,
                temp: 40.0 * u01k(h, 2) - 10.0,
                mountain: u01k(h, 3),
                uplift: 2.0 * u01k(h, 4) - 1.0,
                cont: 0.8 * u01k(h, 5) - 0.4,
                coast_km: 200.0 * u01k(h, 6) - 60.0,
                sand: u01k(h, 7),
                style: 2.0 * u01k(h, 10) - 1.0,
            };
            for lit in 0..5 {
                let o = zones(&z, lit, 2.0 * u01k(h, 8) - 1.0, 2.0 * u01k(h, 9) - 1.0);
                for v in [o.tower, o.cockpit, o.fold, o.fault, o.badland, o.drown] {
                    assert!((0.0..=1.0 + 1e-9).contains(&v), "{o:?}");
                }
                if lit != CARBONATE {
                    assert!(o.tower == 0.0 && o.cockpit == 0.0);
                }
                if lit == CARBONATE || lit == CRYSTALLINE || lit == VOLCANIC {
                    assert!(o.badland == 0.0);
                }
            }
        }
    }

    /// The carve hook leaves the core's profile alone where nothing applies, and a fjord goes
    /// below sea level only in glaciated coastal mountains.
    #[test]
    fn carve_hook() {
        let w = World::new(crate::Config::default());
        let ctx = Ctx::new(0.8, 0.3, 20.0, &w.ell);
        let mut m = w.macro_at(ctx.p, 1000.0);
        let base = Carve {
            floor: 10.0,
            fp_w: 100.0,
            wall_k: 3.0,
            depth_cap: 400.0,
            fp_min: 0.5,
            u: 0.0,
            steps: 0.0,
            step_mix: 0.0,
            top: 600.0,
            lake_r: 0.0,
            lake_level: 0.0,
            narrow: 0.0,
        };
        fn ci<'a>(ctx: &'a Ctx, m: &'a Macro, level: u8, mountain: f64, moist: f64) -> CarveIn<'a> {
            CarveIn { ctx, m, level, ad: 50.0, u: 0.5, spring: false, hw: 30.0, h0: 600.0, temp: 5.0, moist, mountain, mesa: 0.0, smooth: 500.0 }
        }
        // humid lowland: unchanged
        m.glaciation = 0.0;
        let mut cv = base;
        carve(&w, &ci(&ctx, &m, 0, 0.0, 0.6), &mut cv);
        assert_eq!((cv.floor, cv.fp_w, cv.wall_k, cv.u, cv.lake_r), (10.0, 100.0, 3.0, 0.0, 0.0));
        // a glaciated coast: a fjord floor, a U
        m.glaciation = 0.9;
        m.coast_km = 20.0;
        let mut cv = base;
        carve(&w, &ci(&ctx, &m, 0, 0.8, 0.6), &mut cv);
        assert!(cv.floor < -60.0 && cv.u > 0.9 && cv.fp_min < -1000.0, "{cv:?}");
        // inland, a glaciated trough only: the floor stays above the sea
        m.coast_km = 300.0;
        let mut cv = base;
        carve(&w, &ci(&ctx, &m, 0, 0.8, 0.6), &mut cv);
        assert!(cv.floor >= 1.0 && cv.u > 0.9 && cv.depth_cap >= 400.0 * 2.0 - 1e-9, "{cv:?}");
        // streams are not touched
        let mut cv = base;
        carve(&w, &ci(&ctx, &m, 2, 0.8, 0.6), &mut cv);
        assert_eq!(cv.u, 0.0);
    }

    /// The relief is continuous: neighbouring points of a pixel size differ by no more than the
    /// slope allows (checked at the places where the zones are active, by a search).
    #[test]
    fn relief_is_continuous_and_bounded() {
        let w = World::new(crate::Config::default());
        let mut checked = 0;
        for i in 0..4000u64 {
            let h = mix64(0xCC ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = std::f64::consts::TAU * u01k(h, 2) - std::f64::consts::PI;
            let ctx = Ctx::new(lat, lon, 30.0, &w.ell);
            let m = w.macro_at(ctx.p, 30.0);
            let (mountain, _) = w.mountain_mask(&m);
            let (temp, moist) = w.climate(&m, lat, w.smooth_elevation(ctx.p).max(0.0));
            let r = ReliefIn { ctx: &ctx, m: &m, temp, moist, mountain, sand: 0.0, mesa: 0.0, smooth: w.smooth_elevation(ctx.p), inst: None };
            let (lit, n1, n2) = fields(&w, &ctx);
            let zi =
                ZIn { moist, temp, mountain, uplift: m.uplift, cont: m.cont, coast_km: m.coast_km, sand: 0.0, style: perlin3(SEED + 5, ctx.p / 180_000.0) };
            let z = zones(&zi, lit, n1, n2);
            if z.tower + z.cockpit + z.fold + z.fault + z.badland < 0.5 {
                continue;
            }
            checked += 1;
            let mut h0 = 0.0;
            relief(&w, &r, &mut h0);
            assert!(h0.is_finite() && h0.abs() < 6000.0, "({lat}, {lon}): {h0}");
            // 15 m to the east: the difference is bounded by the steepest feature (a tower flank)
            let c2 = Ctx::new(lat, lon + 15.0 / (6_371_000.0 * lat.cos().max(0.01)), 30.0, &w.ell);
            let r2 = ReliefIn { ctx: &c2, ..r };
            let mut h1 = 0.0;
            relief(&w, &r2, &mut h1);
            assert!((h1 - h0).abs() < 130.0, "({lat}, {lon}): {h0} -> {h1}");
        }
        eprintln!("{checked} pixels in zones");
    }

    /// Places of the default world where the kit draws: (name, lat, lon), for the tests and stills.
    pub(crate) const PLACES: &[(&str, f64, f64)] = &[
        ("tower_karst", -0.714, -96.231),
        ("karst_islands", 4.857, -93.234),
        ("cockpit_karst", 41.958, -21.188),
        ("folded_ridges", 27.899, 42.870),
        ("fault_blocks", 31.739, 137.721),
        ("badlands", -21.410, 36.504),
        ("canyon", 21.245, -121.297),
        ("u_valley", 49.423, -162.365),
        ("tarn", 40.874, 174.741),
        ("fjord", 63.382, -81.127),
        ("fan", 28.597, 67.059),
    ];

    fn tile_at(name: &str, z: u8) -> geodesy::tiles::TileId {
        let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == name).unwrap();
        geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)
    }

    /// Share of samples differing by more than `tol`, and the mean difference.
    #[cfg(feature = "gpu")]
    fn off<T: Copy>(a: &[T], b: &[T], tol: f64, f: impl Fn(T) -> f64) -> (f64, f64) {
        let (mut bad, mut sum) = (0usize, 0.0);
        for (x, y) in a.iter().zip(b) {
            let e = (f(*x) - f(*y)).abs();
            sum += e;
            bad += (e > tol) as usize;
        }
        (bad as f64 / a.len() as f64, sum / a.len() as f64)
    }

    /// The GPU's tiles are the CPU's where the kit draws.
    #[cfg(feature = "gpu")]
    #[test]
    fn tiles_match_the_cpu() {
        let cfg = crate::Config::default();
        let gen = match crate::gpu::GpuGenerator::new(cfg.clone()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU generator ({e}): skipped");
                return;
            }
        };
        let cpu = crate::Generator::new(cfg);
        for (name, z) in [
            ("tower_karst", 14u8),
            ("tower_karst", 16),
            ("karst_islands", 13),
            ("cockpit_karst", 14),
            ("folded_ridges", 12),
            ("folded_ridges", 14),
            ("fault_blocks", 12),
            ("badlands", 14),
            ("badlands", 16),
            ("canyon", 12),
            ("canyon", 14),
            ("u_valley", 12),
            ("u_valley", 14),
            ("tarn", 12),
            ("tarn", 14),
            ("fjord", 12),
            ("fjord", 14),
            ("fan", 12),
            ("fan", 14),
        ] {
            let id = tile_at(name, z);
            let g = gen.tiles(&[id]).unwrap().remove(0);
            let c = cpu.tile_cpu(id);
            let rgb = off(&g.rgb, &c.rgb, 2.0, |v| v as f64);
            let el = off(&g.elevation, &c.elevation, 0.05, |v| v as f64);
            let lc = off(&g.landcover, &c.landcover, 0.5, |v| v as f64);
            eprintln!("{name} z{z}: rgb off {:.3}% elevation off {:.3}% landcover off {:.3}%", rgb.0 * 100.0, el.0 * 100.0, lc.0 * 100.0);
            assert!(rgb.0 < 0.005 && el.0 < 0.005 && lc.0 < 0.005, "{name} z{z}");
        }
    }

    /// Parent ≈ mean of its children (elevation, albedo) where the kit draws.
    #[test]
    fn lod_consistent_at_places() {
        let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [
            ("tower_karst", 12u8),
            ("karst_islands", 12),
            ("cockpit_karst", 12),
            ("folded_ridges", 11),
            ("fault_blocks", 11),
            ("badlands", 12),
            ("canyon", 11),
            ("u_valley", 11),
            ("tarn", 11),
            ("fjord", 11),
            ("fan", 12),
        ] {
            let parent = tile_at(name, z);
            let p = g.tile(parent);
            let kids = parent.children().map(|c| g.tile(c));
            let (mut eh, mut ea, mut cnt) = (0.0, 0.0, 0.0);
            for (k, kid) in kids.iter().enumerate() {
                let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
                for j in (0..n).step_by(2) {
                    for i in (0..n).step_by(2) {
                        let ix = [j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1];
                        let m: f32 = ix.iter().map(|&q| kid.elevation[q]).sum::<f32>() / 4.0;
                        let pi = (oy + j / 2) * n + ox + i / 2;
                        eh += (m - p.elevation[pi]).abs() as f64;
                        for c in 0..3 {
                            let ma: f64 = ix.iter().map(|&q| kid.albedo[3 * q + c] as f64).sum::<f64>() / 4.0;
                            ea += (ma - p.albedo[3 * pi + c] as f64).abs();
                        }
                        cnt += 1.0;
                    }
                }
            }
            let (eh, ea) = (eh / cnt, ea / (3.0 * cnt));
            eprintln!("{name} z{z}: |parent − mean(children)| elevation {eh:.2} m, albedo {ea:.2} DN");
            assert!(eh < 12.0, "{name}: elevation {eh}");
            assert!(ea < 14.0, "{name}: albedo {ea}");
        }
    }

    /// No seam between neighbouring tiles where the kit draws (E-W and N-S): the relief is
    /// continuous across tiles.
    #[test]
    fn seamless_at_places() {
        let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [
            ("tower_karst", 14u8),
            ("karst_islands", 13),
            ("folded_ridges", 12),
            ("fault_blocks", 12),
            ("badlands", 14),
            ("canyon", 12),
            ("u_valley", 12),
            ("fjord", 12),
            ("fan", 13),
        ] {
            let id = tile_at(name, z);
            let a = g.tile(id);
            let e = g.tile(geodesy::tiles::TileId::new(z, id.x + 1, id.y));
            let s = g.tile(geodesy::tiles::TileId::new(z, id.x, id.y + 1));
            let (mut ea, mut ei, mut na, mut ni) = (0.0, 0.0, 0.0, 0.0);
            for j in 0..n {
                for c in 0..3 {
                    ea += (a.albedo[3 * (j * n + n - 1) + c] as f64 - e.albedo[3 * (j * n) + c] as f64).abs();
                    ei += (a.albedo[3 * (j * n + n - 2) + c] as f64 - a.albedo[3 * (j * n + n - 1) + c] as f64).abs();
                    na += (a.albedo[3 * ((n - 1) * n + j) + c] as f64 - s.albedo[3 * j + c] as f64).abs();
                    ni += (a.albedo[3 * ((n - 2) * n + j) + c] as f64 - a.albedo[3 * ((n - 1) * n + j) + c] as f64).abs();
                    ei += (e.albedo[3 * (j * n) + c] as f64 - e.albedo[3 * (j * n + 1) + c] as f64).abs();
                    ni += (s.albedo[3 * j + c] as f64 - s.albedo[3 * (n + j) + c] as f64).abs();
                }
                // elevation: the step across the seam is no larger than the slope inside (towers
                // and scarps are steep: a generous margin)
                let (h0, h1, h2) = (a.elevation[j * n + n - 2] as f64, a.elevation[j * n + n - 1] as f64, e.elevation[j * n] as f64);
                assert!((h2 - h1).abs() < 3.0 * (h1 - h0).abs() + 90.0, "{name}: E-W elevation seam {h0} {h1} {h2}");
                let (v0, v1, v2) = (a.elevation[(n - 2) * n + j] as f64, a.elevation[(n - 1) * n + j] as f64, s.elevation[j] as f64);
                assert!((v2 - v1).abs() < 3.0 * (v1 - v0).abs() + 90.0, "{name}: N-S elevation seam {v0} {v1} {v2}");
            }
            eprintln!("{name} z{z}: albedo E-W across {ea:.0} inside {ei:.0}; N-S across {na:.0} inside {ni:.0}");
            assert!(ea < 2.0 * ei + 2.0 * n as f64, "{name}: E-W across {ea} inside {ei}");
            assert!(na < 2.0 * ni + 2.0 * n as f64, "{name}: N-S across {na} inside {ni}");
        }
    }

    /// Tiles are a pure function of the world: the same tile twice is the same bits.
    #[test]
    fn deterministic_at_places() {
        let cfg = crate::Config { tile_supersample: 1, ..crate::Config::default() };
        let a = crate::Generator::new(cfg.clone());
        let b = crate::Generator::new(cfg);
        for (name, z) in [("tower_karst", 14u8), ("folded_ridges", 13), ("fjord", 13)] {
            let id = tile_at(name, z);
            let (x, y) = (a.tile(id), b.tile(id));
            assert!(x.elevation == y.elevation && x.albedo == y.albedo && x.landcover == y.landcover, "{name}");
        }
    }

    /// The fan family has instances at the fan place, each with a direction (a unit vector) and a
    /// radius within its reach.
    #[test]
    fn fans_exist_at_the_fan_place() {
        let w = World::new(crate::Config::default());
        let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == "fan").unwrap();
        let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &w.ell);
        let l = crate::instances::near(&w, fam_fans(), p, 40_000.0);
        assert!(!l.is_empty(), "no fans near the fan place");
        for it in &l {
            let (dx, dy) = (it.v[1] as f64, it.v[2] as f64);
            assert!(((dx * dx + dy * dy).sqrt() - 1.0).abs() < 1e-4 && it.v[0] > 2500.0 && (it.v[0] as f64) < 6200.0, "{it:?}");
        }
    }

    /// Every place generates on the GPU at all zooms (no CPU fallback: an overflowing family, a
    /// shader error): `Backend::Gpu` returns the error.
    #[cfg(feature = "gpu")]
    #[test]
    fn places_stay_on_the_gpu() {
        let Ok(g) = crate::gpu::shared() else { return };
        if g.check_generator().is_err() {
            return;
        }
        let gen = crate::Generator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        for &(name, lat, lon) in PLACES {
            let ids: Vec<_> = [6u8, 8, 10, 12, 14, 16].iter().map(|&z| geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)).collect();
            if let Err(e) = gen.tiles(&ids) {
                panic!("{name}: GPU generation failed: {e:#}");
            }
        }
        assert!(gen.backend_name().starts_with("GPU"), "{}", gen.backend_name());
    }

    /// Debug: places of each landform in the default world:
    /// `cargo test -p terragen --release --lib kits::mountains::tests::find_places -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_places() {
        use rayon::prelude::*;
        let w = World::new(crate::Config::default());
        let n = 200_000;
        struct Pt {
            lat: f64,
            lon: f64,
            z: Zones,
            lit: u32,
            t: crate::world::Terrain,
        }
        let pts: Vec<Pt> = (0..n)
            .into_par_iter()
            .map(|i| {
                let h = mix64(0x5EA ^ i as u64);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin().to_degrees();
                let lon = 360.0 * u01k(h, 2) - 180.0;
                let ctx = Ctx::new(lat.to_radians(), lon.to_radians(), 150.0, &w.ell);
                let t = w.terrain(&ctx);
                let (lit, n1, n2) = fields(&w, &ctx);
                let zi =
                    ZIn { moist: t.moist, temp: t.temp, mountain: t.mountain, uplift: t.uplift, cont: t.cont, coast_km: t.coast_km, sand: t.sand, style: 0.0 };
                Pt { lat, lon, z: zones(&zi, lit, n1, n2), lit, t }
            })
            .collect();
        let land = |p: &Pt| p.t.water_kind == 0 && p.t.ground > 2.0;
        #[allow(clippy::type_complexity)]
        let kinds: Vec<(&str, Box<dyn Fn(&Pt) -> bool + Sync>)> = vec![
            ("tower_karst", Box::new(|p| land(p) && p.z.tower > 0.95 && p.z.drown < 0.2)),
            ("karst_islands", Box::new(|p| p.z.tower > 0.9 && p.z.drown > 0.8 && p.t.coast_km.abs() < 12.0)),
            ("cockpit_karst", Box::new(|p| land(p) && p.z.cockpit > 0.95)),
            ("folded_ridges", Box::new(|p| land(p) && p.z.fold > 0.9)),
            ("fault_blocks", Box::new(|p| land(p) && p.z.fault > 0.9)),
            ("badlands", Box::new(|p| land(p) && p.z.badland > 0.95)),
            (
                "canyon",
                Box::new(|p| {
                    land(p) && p.lit <= 1 && p.t.moist < 0.2 && p.t.ground > 400.0 && p.t.mountain < 0.2 && p.t.mesa > 0.1 && p.t.river_d.abs() < 1500.0
                }),
            ),
            ("u_valley", Box::new(|p| land(p) && p.t.glaciation > 0.6 && p.t.mountain > 0.6 && p.t.coast_km > 150.0 && p.t.river_d.abs() < 1500.0)),
            ("fjord", Box::new(|p| p.t.glaciation > 0.6 && p.t.mountain > 0.5 && p.t.coast_km > 3.0 && p.t.coast_km < 30.0)),
        ];
        for (name, f) in &kinds {
            let v: Vec<&Pt> = pts.iter().filter(|p| f(p)).collect();
            let step = (v.len() / 10).max(1);
            let list: Vec<String> =
                v.iter().step_by(step).take(10).map(|p| format!("({:.3}, {:.3} h{:.0} ck{:.0})", p.lat, p.lon, p.t.ground, p.t.coast_km)).collect();
            eprintln!("{name}: {} samples: {}", v.len(), list.join(" "));
        }
        // fans: instances near dry piedmont points
        let mut seen = std::collections::BTreeMap::new();
        for p in pts.iter().filter(|p| land(p) && p.t.mountain > 0.1 && p.t.mountain < 0.7 && p.t.moist < 0.3) {
            let q = geodesy::geodetic2ecef(geodesy::Geodetic::new(p.lat.to_radians(), p.lon.to_radians(), p.t.ground), &w.ell);
            for it in crate::instances::near(&w, fam_fans(), q, 0.0) {
                seen.entry(it.id).or_insert(it);
            }
        }
        eprintln!("{} fans", seen.len());
        for it in seen.values().filter(|it| it.v[0] > 4300.0).take(40) {
            let g = geodesy::ecef2geodetic(it.center, &w.ell);
            eprintln!("fan ({:.3}, {:.3}) R{:.0} H{:.0} dir ({:.2}, {:.2})", g.lat.to_degrees(), g.lon.to_degrees(), it.v[0], it.v[3], it.v[1], it.v[2]);
        }
    }

    /// Debug: zones and relief along a line east of a place: `probe -- --ignored --nocapture`
    /// with `MT_PROBE=lat,lon` (default the first folded place).
    #[test]
    #[ignore]
    fn probe() {
        let w = World::new(crate::Config::default());
        let (lat0, lon0): (f64, f64) = std::env::var("MT_PROBE")
            .ok()
            .and_then(|v| {
                let mut it = v.split(',').map(|x| x.parse::<f64>().unwrap());
                Some((it.next()?, it.next()?))
            })
            .unwrap_or((51.411, -28.424));
        for j in 0..40 {
            let dx = (j as f64 - 20.0) * std::env::var("MT_STEP").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.5);
            let (la, lo) = (lat0, lon0 + dx / (111.0 * lat0.to_radians().cos()));
            let ctx = Ctx::new(la.to_radians(), lo.to_radians(), 40.0, &w.ell);
            let m = w.macro_at(ctx.p, 40.0);
            let (mountain, _) = w.mountain_mask(&m);
            let smooth = w.smooth_elevation(ctx.p);
            let (temp, moist) = w.climate(&m, ctx.lat, smooth.max(0.0));
            let (lit, n1, n2) = fields(&w, &ctx);
            let zi =
                ZIn { moist, temp, mountain, uplift: m.uplift, cont: m.cont, coast_km: m.coast_km, sand: 0.0, style: perlin3(SEED + 5, ctx.p / 180_000.0) };
            let z = zones(&zi, lit, n1, n2);
            let r = ReliefIn { ctx: &ctx, m: &m, temp, moist, mountain, sand: 0.0, mesa: 0.0, smooth, inst: None };
            let mut h = 0.0;
            relief(&w, &r, &mut h);
            let t = w.terrain(&ctx);
            eprintln!(
                "{dx:6.1} km lit {lit} mtn {mountain:.2} up {:.2} glac {:.2} moist {moist:.2} zones fold {:.2} fault {:.2} bad {:.2} tow {:.2} cock {:.2} | smooth {smooth:.0} relief {h:.1} ground {:.0} water {:.0} kind {} river {:.0} ck {:.0}",
                m.uplift, m.glaciation, z.fold, z.fault, z.badland, z.tower, z.cockpit, t.ground, t.water, t.water_kind, t.river_d, t.coast_km
            );
        }
    }

    /// Debug: fjords of the default world (flooded valleys): sample glaciated mountain coasts and
    /// look for ground below sea level inland: `find_fjords -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_fjords() {
        use rayon::prelude::*;
        let w = World::new(crate::Config::default());
        let n = 3_000_000;
        let cands: Vec<(f64, f64)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = mix64(0xF10 ^ i as u64);
                let lat = 72.0 * (2.0 * u01k(h, 1) - 1.0);
                let lon = 360.0 * u01k(h, 2) - 180.0;
                let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &w.ell);
                let a = w.atlas().sample(p.normalize());
                if a.coast_km < 3.0 || a.coast_km > 40.0 || a.glaciation < 0.45 {
                    return None;
                }
                let m = w.macro_at(p, 2000.0);
                let mountain = w.mountain_mask(&m).0;
                let glac = smoothstep(0.3, 0.6, m.glaciation) * smoothstep(0.25, 0.55, mountain);
                let (fj, _) = fjord(&m, mountain, glac);
                (fj > 0.4).then_some((lat, lon))
            })
            .collect();
        eprintln!("{} fjord candidates", cands.len());
        let found: Vec<(f64, f64, f64, f64)> = cands
            .par_iter()
            .take(400)
            .filter_map(|&(lat, lon)| {
                let mut lowest = f64::MAX;
                let mut at = (lat, lon);
                for k in 0..25 {
                    let (dx, dy) = ((k % 5) as f64 - 2.0, (k / 5) as f64 - 2.0);
                    let (la, lo) = (lat + dy * 3.0 / 111.0, lon + dx * 3.0 / (111.0 * lat.to_radians().cos()));
                    let t = w.terrain(&Ctx::new(la.to_radians(), lo.to_radians(), 80.0, &w.ell));
                    if t.ground < lowest {
                        lowest = t.ground;
                        at = (la, lo);
                    }
                }
                (lowest < -40.0).then_some((at.0, at.1, lowest, lat))
            })
            .collect();
        eprintln!("{} with ground below -40 m", found.len());
        for f in found.iter().take(25) {
            eprintln!("({:.3}, {:.3}) lowest {:.0} m", f.0, f.1, f.2);
        }
    }

    /// Debug: elevation statistics and class shares of the tile at a place: `MT_TILE=name,z`.
    #[test]
    #[ignore]
    fn tile_stats() {
        let v = std::env::var("MT_TILE").unwrap_or("fault_blocks,12".into());
        let (name, z) = v.split_once(',').unwrap();
        let z: u8 = z.parse().unwrap();
        let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let t = g.tile(tile_at(name, z));
        let n = crate::TILE_SIZE;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &e in &t.elevation {
            lo = lo.min(e);
            hi = hi.max(e);
        }
        eprintln!("{name} z{z}: elevation {lo:.0} .. {hi:.0}");
        let row: Vec<String> = (0..n).step_by(n / 16).map(|i| format!("{:.0}", t.elevation[(n / 2) * n + i])).collect();
        eprintln!("middle row: {}", row.join(" "));
        let col: Vec<String> = (0..n).step_by(n / 16).map(|j| format!("{:.0}", t.elevation[j * n + n / 2])).collect();
        eprintln!("middle column: {}", col.join(" "));
        let mut hist = std::collections::BTreeMap::new();
        for &c in &t.landcover {
            *hist.entry(c).or_insert(0usize) += 1;
        }
        let tot = t.landcover.len() as f64;
        let h: Vec<String> = hist.iter().map(|(c, k)| format!("{} {:.1}%", crate::registry::classes::name(*c), 100.0 * *k as f64 / tot)).collect();
        eprintln!("classes: {}", h.join(", "));
    }

    /// Debug: relief across the strike: the standard deviation of the ground along a north-south and
    /// an east-west line of 24 km through each place: `MT_LIST="lat,lon;lat,lon"`.
    #[test]
    #[ignore]
    fn strike_scan() {
        let w = World::new(crate::Config::default());
        let list = std::env::var("MT_LIST").unwrap_or_default();
        for item in list.split(';').filter(|s| !s.is_empty()) {
            let mut it = item.split(',').map(|x| x.parse::<f64>().unwrap());
            let (lat0, lon0) = (it.next().unwrap(), it.next().unwrap());
            let line = |dn: f64, de: f64| {
                let v: Vec<f64> = (-12..=12)
                    .map(|j| {
                        let (la, lo) = (lat0 + j as f64 * dn / 111.0, lon0 + j as f64 * de / (111.0 * lat0.to_radians().cos()));
                        w.terrain(&Ctx::new(la.to_radians(), lo.to_radians(), 150.0, &w.ell)).ground
                    })
                    .collect();
                let m = v.iter().sum::<f64>() / v.len() as f64;
                (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
            };
            eprintln!("({lat0}, {lon0}): north-south std {:.0} m, east-west std {:.0} m", line(1.0, 0.0), line(0.0, 1.0));
        }
    }

    /// Debug: the elevation step across the east seam of a tile: `MT_SEAM=lat,lon,z`.
    #[test]
    #[ignore]
    fn seam_debug() {
        let v = std::env::var("MT_SEAM").unwrap_or("-16.1256,53.6542,11".into());
        let mut it = v.split(',');
        let (lat, lon): (f64, f64) = (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap());
        let z: u8 = it.next().unwrap().parse().unwrap();
        let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let id = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
        let a = g.tile(id);
        let b = g.tile(geodesy::tiles::TileId::new(z, id.x + 1, id.y));
        let n = crate::TILE_SIZE;
        for j in (0..n).filter(|j| j % (n / 16) == 0 || (b.elevation[j * n] - a.elevation[j * n + n - 1]).abs() > 5.0) {
            let r = |t: &[f32], i: usize| t[j * n + i];
            eprintln!(
                "row {j:3}: a[n-3..n-1] {:.1} {:.1} {:.1} | b[0..2] {:.1} {:.1} {:.1}",
                r(&a.elevation, n - 3),
                r(&a.elevation, n - 2),
                r(&a.elevation, n - 1),
                r(&b.elevation, 0),
                r(&b.elevation, 1),
                r(&b.elevation, 2)
            );
        }
    }

    /// Debug: where the GPU's elevation differs from the CPU's by more than 5 cm in a tile:
    /// `MT_DIFF=lat,lon,z`.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn tile_diff() {
        let v = std::env::var("MT_DIFF").unwrap_or("-12.4109,-155.0993,13".into());
        let mut it = v.split(',');
        let (lat, lon): (f64, f64) = (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap());
        let z: u8 = it.next().unwrap().parse().unwrap();
        let cfg = crate::Config::default();
        let gpu = crate::gpu::GpuGenerator::new(cfg.clone()).unwrap();
        let cpu = crate::Generator::new(cfg);
        let id = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
        let g = gpu.tiles(&[id]).unwrap().remove(0);
        let c = cpu.tile_cpu(id);
        let n = crate::TILE_SIZE;
        let mut shown = 0;
        let mut rows = std::collections::BTreeMap::new();
        for (k, (a, b)) in g.elevation.iter().zip(&c.elevation).enumerate() {
            if (a - b).abs() > 0.05 {
                *rows.entry(k / n).or_insert(0) += 1;
                if shown < 12 {
                    eprintln!("({}, {}): gpu {a:.3} cpu {b:.3}", k % n, k / n);
                    if shown % 4 == 0 {
                        let px = glam::DVec2::new(id.x as f64 * n as f64 + (k % n) as f64 + 0.5, id.y as f64 * n as f64 + (k / n) as f64 + 0.5);
                        let (la, lo) = geodesy::tiles::pixel_to_latlon(px, z, n as u32);
                        let w = World::new(crate::Config::default());
                        let gsd = 156_543.03 * la.cos() / (1u64 << z) as f64 * 256.0 / n as f64;
                        let ctx = Ctx::new(la, lo, gsd, &w.ell);
                        let t = w.terrain(&ctx);
                        let m = w.macro_at(ctx.p, gsd);
                        eprintln!(
                            "   lat {:.5} lon {:.5} gsd {gsd:.1}: ground {:.3} river_d {:.1} hw {:.1} mountain {:.3} mesa {:.3} moist {:.3} cont {:.4} style {:?} water {} {}",
                            la.to_degrees(), lo.to_degrees(), t.ground, t.river_d, t.river_hw, t.mountain, t.mesa, t.moist, m.cont, m.style, t.water, t.water_kind
                        );
                    }
                    shown += 1;
                }
            }
        }
        eprintln!("rows with differences: {rows:?}");
    }
}
