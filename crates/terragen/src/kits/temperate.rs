//! The temperate, boreal and wetland kit (`docs/design/terrain-next.md` §5.7,
//! `docs/next/kits/temperate.md`).
//!
//! * **Biomes** (`biomes/temperate.yaml`): broadleaf deciduous, mixed forest, taiga, lichen
//!   woodland, maquis, garrigue, chaparral, feather-grass steppe, heathland, moorland, and the
//!   zones above the tree line (subalpine krummholz, alpine meadow) of the temperate biomes.
//! * **Layer functions** (here; the WGSL twins in `gpu/wgsl/kits/temperate.wgsl`, same names and
//!   order of operations), climate- and terrain-driven in every biome where they fit:
//!   * slot 3 (azonal) — wetlands: raised bogs with concentric pools, string bogs (aapa mires)
//!     with strings and flark pools, fens, reed marshes with open water, beaver ponds on small
//!     streams (with their ring of dead trees and beaver meadow), floodplain meadows with scroll
//!     swales;
//!   * slot 4 (disturbance) — managed forest: rectangular stands in the land-use region's frame,
//!     clear-cuts of different ages (slash and skid trails, fireweed, young plantations), logging
//!     roads along stand edges; windthrow (fallen trunks in the storm's direction); firebreaks;
//!   * canopy — autumn colours (deciduous crowns in yellow, gold, orange and red by the
//!     ecoregion's season), larches (light green, gold in autumn).
//!
//! The climate of the ecoregion's site and the atlas (temperature, precipitation, annual
//! range, precipitation regime; the `site_*` helpers) and the ecoregion's style
//! (season, culture archetype, the per-ecoregion draws of woodlots and shrubs) set where and how
//! much: forestry by culture, mires in the cool wet north and on oceanic coasts, firebreaks in
//! dry-summer climates.
//!
//! Kernels: the core's `scatter` with this kit's parameters ([`kernels`]); their parameters and
//! calibrated means are written into the WGSL file ([`wgsl_consts`], checked by a test).

use crate::kernels::{self, KIn, KMean, KOut, KParams};
use crate::kits::Kit;
use crate::landcover as lc;
use crate::noise::*;
use crate::registry::pal;
use crate::stack::{hmode, slot, Layer, Stack};
use crate::surface::{mixc, srgb};
use glam::{DVec2, DVec3};

pub static KIT: Kit = Kit {
    name: "temperate",
    biomes_yaml: include_str!("../../biomes/temperate.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/temperate.wgsl"),
    layers: &[(slot::AZONAL, azonal), (slot::DISTURBANCE, disturbance), (slot::CANOPY, canopy)],
    ..Kit::EMPTY
};

// ============================================================================= kernels

/// The kit's kernel instances: (name, core kernel, YAML parameters). White colours are
/// shading factors the layer functions multiply by their own colours.
const KERNELS: [(&str, &str, &str); 5] = [
    // deciduous crowns in autumn colours (over the canopy)
    ("autumn", "scatter", "{cell: 9, density: 1, shape: dome, radius: [3.0, 4.4], height: [13, 19], jitter: 0.85, colour_var: 0.22, shade: 0.55, colour: [255, 255, 255]}"),
    // larches
    ("larch", "scatter", "{cell: 6, density: 1, shape: cone, radius: [1.5, 2.3], height: [14, 21], jitter: 0.85, colour_var: 0.2, shade: 0.5, colour: [255, 255, 255]}"),
    // young trees of a regrowing clear-cut (planted: little jitter)
    ("young", "scatter", "{cell: 2.6, density: 1, shape: cone, radius: [0.55, 1.0], height: [0.85, 1.15], jitter: 0.35, colour_var: 0.45, shade: 0.7, colour: [255, 255, 255]}"),
    // fallen trunks of a windthrow, along the storm (`aux[0]`)
    ("logs", "scatter", "{cell: 7, density: 1, shape: rect, radius: [0.3, 0.45], aspect: 10, orient: aux, height: [0.3, 0.6], jitter: 0.9, colour_var: 0.2, shade: 0.2, colour: [178, 168, 152], colour2: [116, 92, 68], second: 0.35}"),
    // dead grey trees in a beaver pond's flooded ring
    ("snags", "scatter", "{cell: 6, density: 1, shape: dome, radius: [0.35, 0.7], height: [4, 9], jitter: 0.9, colour_var: 0.2, shade: 0.4, colour: [154, 150, 140]}"),
];
const K_AUTUMN: usize = 0;
const K_LARCH: usize = 1;
const K_YOUNG: usize = 2;
const K_LOGS: usize = 3;
const K_SNAGS: usize = 4;

/// A compiled kernel instance: parameters, calibrated mean, feature size.
pub struct KitKernel {
    pub k: KParams,
    pub mean: KMean,
    pub size: f64,
}

/// The kit's kernels (compiled and calibrated once).
pub fn kernels() -> &'static [KitKernel] {
    static K: std::sync::OnceLock<Vec<KitKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let pal = crate::registry::BiomePal::default_pal();
        KERNELS
            .iter()
            .enumerate()
            .map(|(i, (name, kernel, y))| {
                let spec = kernels::spec(kernel).expect("core kernel");
                let m: std::collections::BTreeMap<String, serde_yaml::Value> = serde_yaml::from_str(y).expect("kernel parameters");
                let k = kernels::compile_params(spec, &m, mix64(0x7E3A_0000 + i as u64), &pal).unwrap_or_else(|e| panic!("temperate kernel {name}: {e}"));
                kernels::check_reach(&k).unwrap_or_else(|e| panic!("temperate kernel {name}: {e}"));
                KitKernel { mean: kernels::calibrate(&k), size: kernels::size(&k), k }
            })
            .collect()
    })
}

#[cfg(test)]
const GEN_BEGIN: &str = "// ---- generated by kits::temperate::wgsl_consts (AERIALSYNTH_BLESS=1 cargo test -p terragen --lib temperate)";
#[cfg(test)]
const GEN_END: &str = "// ---- end of generated";

/// The WGSL constants of the kit's kernels (the generated block of `temperate.wgsl`).
#[cfg(test)]
pub fn wgsl_consts() -> String {
    let f = |x: f64| format!("{:?}", x as f32);
    let v3 = |c: DVec3| format!("vec3<f32>({}, {}, {})", f(c.x), f(c.y), f(c.z));
    let mut o = String::from(GEN_BEGIN);
    o.push('\n');
    for (i, kk) in kernels().iter().enumerate() {
        let n = KERNELS[i].0;
        let k = &kk.k;
        o.push_str(&format!("fn temperate_kp_{n}() -> KP {{\n    var k: KP;\n    k.kind = {}u;\n    k.seed = 0x{:x}lu;\n", k.kind, k.seed));
        o.push_str(&format!("    k.v = array<f32, 16>({});\n", k.v.iter().map(|x| f(*x)).collect::<Vec<_>>().join(", ")));
        o.push_str(&format!("    k.col = array<vec3<f32>, 3>({});\n    return k;\n}}\n", k.col.iter().map(|c| v3(*c)).collect::<Vec<_>>().join(", ")));
        let u = n.to_uppercase();
        o.push_str(&format!("const TEMPERATE_{u}_SIZE: f32 = {};\n", f(kk.size)));
        o.push_str(&format!("const TEMPERATE_{u}_MEAN: vec4<f32> = vec4<f32>({}, {}, {}, 0.0);\n", f(kk.mean.cov), f(kk.mean.dh), f(kk.mean.emit)));
        o.push_str(&format!("const TEMPERATE_{u}_MALB: vec3<f32> = {};\n", v3(kk.mean.albedo)));
    }
    o.push_str(GEN_END);
    o
}

/// Kernel `i` at the sample (local position `q`, `aux[0]`), its explicit and mean parts kept
/// apart (the layers colour each): the explicit weight, the explicit result, the mean coverage.
fn kernel_parts(s: &Stack, i: usize, amount: f64, q: DVec2, aux0: f64) -> Parts {
    let kk = &kernels()[i];
    let mut kin: KIn = crate::layers::kin(s, kk.k.kind, amount);
    kin.q = q;
    kin.aux[0] = aux0;
    let e = smoothstep(1.2 * kin.gsd, 3.0 * kin.gsd, kk.size);
    let x = if e > 0.0 { kernels::explicit(&kk.k, &kin) } else { KOut::default() };
    Parts { e, x, mc: kk.mean.cov * amount.clamp(0.0, 1.0) }
}

/// A kit kernel at a sample: the explicit weight, the explicit result, the mean coverage.
struct Parts {
    e: f64,
    x: KOut,
    mc: f64,
}

/// The layer of kernel `i`'s parts (explicit colour × `cx`, mean colour × `cm`, height × `hk`).
fn kernel_layer(s: &Stack, i: usize, k: &Parts, cx: DVec3, cm: DVec3, hk: f64) -> Layer {
    let kk = &kernels()[i];
    let (e, x, mc) = (k.e, &k.x, k.mc);
    let cov = e * x.cov + (1.0 - e) * mc;
    if cov <= 0.0 {
        return Layer::default();
    }
    let (we, wm) = (e * x.cov / cov, (1.0 - e) * mc / cov);
    Layer { cov, albedo: x.albedo * cx * we + kk.mean.albedo * cm * wm, dh: (x.dh * we + kk.mean.dh * wm) * hk, hmode: tree_hmode(s), ..Default::default() }
}

// ============================================================================= helpers

/// sRGB 0..255 → linear.
#[inline]
fn rgb(r: f64, g: f64, b: f64) -> DVec3 {
    srgb(r, g, b)
}

#[inline]
fn tree_hmode(s: &Stack) -> u8 {
    if s.world.cfg.vegetation.trees_in_dsm {
        hmode::MAX
    } else {
        hmode::NONE
    }
}

/// A per-ecoregion draw 0..1 (the style's woodlot share, uniform over 0.6..1.5).
#[inline]
fn eco_u1(s: &Stack) -> f64 {
    ((s.bio.style.woodlots - 0.6) / 0.9).clamp(0.0, 1.0)
}

/// Another per-ecoregion draw 0..1 (the style's shrub multiplier, uniform over 0.6..1.4).
#[inline]
fn eco_u2(s: &Stack) -> f64 {
    ((s.bio.style.shrubs - 0.6) / 0.8).clamp(0.0, 1.0)
}

/// Climate classes of the sample from the ecoregion's site and the atlas: cold continental
/// (boreal), oceanic (wet, small annual range), dry summers (Mediterranean regime).
#[inline]
fn site_boreal(s: &Stack) -> f64 {
    1.0 - smoothstep(1.0, 5.0, s.bio.site.temp_c)
}
#[inline]
fn site_oceanic(s: &Stack) -> f64 {
    let c = &s.bio.site;
    smoothstep(700.0, 1100.0, c.precip_mm)
        * (1.0 - smoothstep(13.0, 20.0, s.t.temp_range))
        * smoothstep(0.0, 4.0, c.temp_c)
        * (1.0 - smoothstep(12.0, 16.0, c.temp_c))
}
#[inline]
fn site_dry_summer(s: &Stack) -> f64 {
    let c = &s.bio.site;
    (1.0 - smoothstep(-0.45, -0.15, s.t.regime)) * smoothstep(9.0, 13.0, c.temp_c) * (1.0 - smoothstep(1100.0, 1600.0, c.precip_mm))
}

/// Forestry of the ecoregion's culture (0..1): boreal, survey-grid and temperate cultures
/// manage their forests, others hardly.
fn forestry(s: &Stack) -> f64 {
    let st = &s.bio.style;
    let a = match st.archetype {
        9 => 1.0,
        7 => 0.75,
        6 => 0.55,
        11 => 0.35,
        4 | 5 | 10 => 0.2,
        _ => 0.1,
    };
    // (no managed forest where the biome's trees are sparse: forest tundra, tundra, polar desert)
    let productive = smoothstep(0.6, 0.75, s.veg(|v| v.trees));
    a * (0.35 + 0.9 * eco_u1(s)) * (0.6 + 0.6 * st.development) * (1.0 - smoothstep(14.0, 19.0, s.bio.site.temp_c)) * productive
}

/// The natural forest at the sample as the canopy draws it (0..1, crisp at its edges): the
/// forest pattern under the climate's cover, farmland cleared, the tree line, what claims the
/// ground.
fn forest_est(s: &Stack) -> f64 {
    let t = s.t;
    let td = s.world.cfg.vegetation.tree_density;
    let trees_mult = s.veg(|v| v.trees) * s.bio.style.trees;
    let cold = (1.0 - smoothstep(0.0, 8.0, t.temp)) * (1.0 - smoothstep(0.15, 0.4, t.mountain));
    let base_cover = smoothstep(0.3 - 0.14 * cold, 0.68 - 0.24 * cold, t.moist) * smoothstep(-6.0, 2.0, t.temp) * td * trees_mult;
    let fpu = 0.5 + 0.5 * s.pf.forest;
    let forest = smoothstep(-0.03, 0.03, base_cover - fpu);
    let clear = 1.0 - 0.85 * smoothstep(0.05, 0.4, s.agri()) * (1.0 - smoothstep(0.45, 0.7, s.l.slope));
    let line = 1.0 - smoothstep(0.0, 0.6, t.mountain * smoothstep(-2.0, -6.0, t.temp));
    forest * clear * line * s.m.natural_ok * s.m.veg * (1.0 - s.m.field_cov)
}

/// A multi-scale noise (3 octaves) mapped to ~uniform 0..1 (the `patches` kernel's),
/// band-limited.
fn patch_u(seed: u64, p: DVec3, scale: f64, rough: f64, gsd: f64) -> f64 {
    let mut n = 0.0;
    let mut a = 1.0;
    let mut norm = 0.0;
    let mut lam = scale;
    for o in 0..3u64 {
        n += a * perlin3(seed ^ (0x9A7 + o), p / lam) * band(lam, gsd);
        norm += a * a;
        a *= rough;
        lam *= 0.43;
    }
    let s = n / (0.27 * norm.sqrt());
    1.0 / (1.0 + (-1.7 * s).exp())
}

/// Coverage of the patches `u < amount` of a [`patch_u`] noise of wavelength `scale`: crisp
/// where resolved, `amount` (their mean) where not.
#[inline]
fn patch_cov(u: f64, amount: f64, scale: f64, gsd: f64, fw: f64) -> f64 {
    let w = (2.0 * fw / scale).clamp(1e-3, 0.5);
    let e = smoothstep(1.2 * gsd, 3.0 * gsd, scale);
    e * smoothstep(-w, w, amount - u) + (1.0 - e) * amount
}

/// Distance beyond the river bank (m; 1e6: no river).
#[inline]
fn river_dist(s: &Stack) -> f64 {
    if s.l.river_hw > 0.0 {
        s.l.river_d.abs() - s.l.river_hw
    } else {
        1e6
    }
}

// ============================================================================= slot 3: wetlands

const SEED_MIRE: u64 = 0x7E3B_0001;
const SEED_MARSH: u64 = 0x7E3B_0002;
const SEED_BEAVER: u64 = 0x7E3B_0003;

fn azonal(s: &mut Stack) {
    let t = s.t;
    // cheap gates: wet enough, not frozen or tropical, not steep
    if t.moist < 0.38 || t.temp < -7.0 || t.temp > 24.0 || s.l.slope > 0.14 || s.m.natural_ok <= 0.0 {
        return;
    }
    floodplain_meadow(s);
    mires(s);
    marsh(s);
    beaver_ponds(s);
}

/// Floodplain meadows: lush wet grass with darker swales parallel to the channel (old scroll
/// bars), on the floodplains of the temperate rivers.
fn floodplain_meadow(s: &mut Stack) {
    let t = s.t;
    if t.floodplain < 0.35 || t.temp < 1.0 || t.temp > 21.0 || s.l.river_hw <= 0.0 {
        return;
    }
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let cov = smoothstep(0.42, 0.5, t.floodplain + 0.08 * s.pf.patch);
    if cov <= 0.0 {
        return;
    }
    let ph = s.l.river_d.abs() / 32.0 + 0.7 * perlin3(SEED_MIRE ^ 0x71, p / 260.0);
    let f = ph - ph.floor();
    let swale = (smoothstep(0.3, 0.42, f) - smoothstep(0.58, 0.7, f)) * band(32.0, gsd);
    let lush = rgb(94.0, 126.0, 52.0) * (1.0 + 0.1 * s.pf.detail);
    let col = mixc(lush, rgb(66.0, 96.0, 50.0), 0.75 * swale);
    s.composite(Layer::paint(cov, col));
}

/// Mires of the cool wet lowlands: raised bogs (ochre and red sphagnum, concentric pools,
/// stunted pines), string bogs (aapa: strings along the iso-lines of a slow flow, flark pools
/// between), fens (green sedge near streams and in water tracks).
fn mires(s: &mut Stack) {
    let t = s.t;
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    let wet = smoothstep(0.5, 0.75, t.moist);
    let cool = 1.0 - smoothstep(9.0, 14.0, t.temp);
    let flat = 1.0 - smoothstep(0.02, 0.07, s.l.slope);
    let low = 1.0 - smoothstep(0.3, 0.6, t.mountain);
    let region = (0.5 + 1.1 * site_boreal(s) + 0.7 * site_oceanic(s)) * (0.5 + 0.8 * eco_u2(s));
    let amount = (0.4 * wet * cool * flat * low * region).min(0.8);
    if amount <= 0.01 {
        return;
    }
    let u = patch_u(SEED_MIRE, p, 1100.0, 0.5, gsd);
    let cov = patch_cov(u, amount, 1100.0, gsd, fw);
    if cov <= 0.0 {
        return;
    }
    // depth into the mire: 0 at the margin .. 1
    let dpt = ((amount - u) / amount.max(0.05)).clamp(0.0, 1.0);
    let rd = river_dist(s);
    let tt = t.temp + 2.5 * perlin3(SEED_MIRE ^ 0x11, p / 3000.0);
    let fen = (1.0 - smoothstep(120.0, 380.0, rd)).max(smoothstep(0.3, 0.6, -t.gully));
    let strings = (1.0 - smoothstep(1.5, 3.5, tt)) * site_boreal(s).max(0.3);
    let mot = perlin3(SEED_MIRE ^ 0x21, p / 4.0) * band(4.0, gsd) + 0.6 * perlin3(SEED_MIRE ^ 0x22, p / 1.5) * band(1.5, gsd);
    let (col, pool, clear) = if fen > 0.5 {
        // fen: sedge and moss, tussocks, willow and birch scrub, a few pools
        let g = smoothstep(-0.3, 0.4, perlin3(SEED_MIRE ^ 0x31, p / 30.0) * band(30.0, gsd));
        let col = mixc(rgb(110.0, 114.0, 72.0), rgb(134.0, 128.0, 90.0), g) * (1.0 + 0.16 * mot);
        let e = band(40.0, gsd);
        let pu = perlin3(SEED_MIRE ^ 0x32, p / 22.0) + 0.5 * perlin3(SEED_MIRE ^ 0x33, p / 9.0) * band(9.0, gsd);
        let pool = e * smoothstep(0.5, 0.56, pu) * smoothstep(0.3, 0.5, dpt) + (1.0 - e) * 0.02 * smoothstep(0.3, 0.5, dpt);
        let clear = 0.55 + 0.3 * smoothstep(0.1, 0.4, dpt);
        (col, pool, clear)
    } else if strings > 0.5 {
        // string bog: iso-lines of a slow flow field F (|grad F| ~ 1): strings ~2.6 m wide
        // every ~15 m, flarks between them, pools where a second noise lets them
        let up = s.ctx.up;
        let (n1, g1) = perlin3_d(SEED_MIRE ^ 0x41, p / 600.0);
        let (n2, g2) = perlin3_d(SEED_MIRE ^ 0x42, p / 170.0);
        let fval = 600.0 * n1 + 0.35 * 170.0 * n2;
        let g = g1 + g2 * 0.35;
        let gt = (g - up * g.dot(up)).length().max(0.05);
        let x = fval / 15.0;
        let f = x - x.floor();
        let spacing = 15.0 / gt;
        let e = smoothstep(1.2 * gsd, 3.0 * gsd, spacing);
        let dist = f.min(1.0 - f) * spacing;
        let string = e * kernels::band_cov(dist, 1.3, fw) + (1.0 - e) * (2.6 / spacing).min(0.5);
        let mid = smoothstep(0.22, 0.3, f) * (1.0 - smoothstep(0.7, 0.78, f));
        let seg = perlin3(SEED_MIRE ^ 0x43, p / 28.0) + 0.4 * perlin3(SEED_MIRE ^ 0x44, p / 11.0) * band(11.0, gsd);
        let wetf = smoothstep(0.15, 0.5, dpt);
        let pool = e * mid * smoothstep(-0.12, -0.06, seg) * wetf + (1.0 - e) * 0.3 * wetf;
        let flark = mixc(rgb(156.0, 140.0, 84.0), rgb(130.0, 128.0, 72.0), smoothstep(-0.2, 0.3, seg)) * (1.0 + 0.12 * mot);
        let col = mixc(flark, rgb(92.0, 84.0, 52.0) * (1.0 + 0.2 * mot), string);
        let clear = 0.85 + 0.12 * smoothstep(0.1, 0.3, dpt);
        (col, pool, clear)
    } else {
        // raised bog: sphagnum mosaic (ochre, red-brown, green), a darker wet margin (lagg),
        // pools on concentric rings near the dome's top
        let a = perlin3(SEED_MIRE ^ 0x51, p / 45.0) * band(45.0, gsd);
        let b = perlin3(SEED_MIRE ^ 0x52, p / 110.0);
        let c0 = mixc(rgb(150.0, 132.0, 92.0), rgb(134.0, 98.0, 74.0), smoothstep(-0.25, 0.35, a));
        let c1 = mixc(c0, rgb(112.0, 118.0, 72.0), 0.5 * smoothstep(0.05, 0.45, b)) * (1.0 + 0.12 * mot);
        let lagg = 1.0 - smoothstep(0.04, 0.12, dpt);
        let col = mixc(c1, rgb(80.0, 88.0, 54.0), lagg);
        let ring = dpt * 7.0 + 0.25 * perlin3(SEED_MIRE ^ 0x53, p / 70.0);
        let fr = ring - ring.floor();
        let e = band(70.0, gsd);
        let blob = perlin3(SEED_MIRE ^ 0x54, p / 34.0) + 0.4 * perlin3(SEED_MIRE ^ 0x55, p / 13.0) * band(13.0, gsd);
        let zone = smoothstep(0.38, 0.48, dpt);
        let pool = e * zone * (1.0 - smoothstep(0.24, 0.32, fr)) * smoothstep(-0.1, -0.02, blob) + (1.0 - e) * 0.15 * zone;
        let clear = 0.5 + 0.42 * smoothstep(0.06, 0.25, dpt);
        (col, pool, clear)
    };
    s.m.natural_ok *= 1.0 - cov;
    s.composite(Layer { cov, albedo: col, cls: lc::BOG, clear, ..Default::default() });
    if pool > 0.0 {
        let water = rgb(34.0, 38.0, 36.0) * (1.0 + 0.15 * perlin3(SEED_MIRE ^ 0x61, p / 7.0) * band(7.0, gsd));
        s.composite(Layer { cov: pool * cov, albedo: water, cls: lc::LAKE, clear: 1.0, ..Default::default() });
    }
}

/// Reed marshes on the floodplains and along the rivers of the warmer temperate lowlands:
/// dense reed (green in summer, straw in autumn), clonal patches, open water in the middle.
fn marsh(s: &mut Stack) {
    let t = s.t;
    if t.temp < 3.0 || t.moist < 0.45 {
        return;
    }
    let rd = river_dist(s);
    if t.floodplain < 0.25 && rd > 120.0 {
        return;
    }
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    let near = (1.0 - smoothstep(30.0, 120.0, rd)) * 0.6;
    let flat = 1.0 - smoothstep(0.02, 0.06, s.l.slope);
    let amount = 0.5
        * smoothstep(0.25, 0.6, t.floodplain).max(near)
        * smoothstep(0.45, 0.7, t.moist)
        * smoothstep(3.0, 7.0, t.temp)
        * (1.0 - smoothstep(16.0, 21.0, t.temp))
        * flat
        // (farmed floodplains are drained: reed patches with dark open water all over the
        // fields read as holes)
        * (1.0 - 0.8 * smoothstep(0.3, 0.6, s.agri()));
    if amount <= 0.01 {
        return;
    }
    let u = patch_u(SEED_MARSH, p, 380.0, 0.55, gsd);
    let cov = patch_cov(u, amount, 380.0, gsd, fw);
    if cov <= 0.0 {
        return;
    }
    // reed of this year (two greens by stand) and last year's dead reed in clonal patches,
    // straw in autumn; willow scrub at the margins
    let dm = ((amount - u) / amount.max(0.05)).clamp(0.0, 1.0);
    let clone = perlin3(SEED_MARSH ^ 0x11, p / 22.0) * band(22.0, gsd);
    let stand = perlin3(SEED_MARSH ^ 0x13, p / 70.0) * band(70.0, gsd);
    let fine = perlin3(SEED_MARSH ^ 0x12, p / 2.5) * band(2.5, gsd) + 0.6 * perlin3(SEED_MARSH ^ 0x14, p / 0.9) * band(0.9, gsd);
    let straw = smoothstep(0.55, 0.85, s.bio.style.season).max(1.0 - smoothstep(4.0, 8.0, t.temp));
    let green = mixc(rgb(92.0, 108.0, 50.0), rgb(124.0, 132.0, 64.0), smoothstep(-0.3, 0.3, stand));
    let c0 = mixc(green, rgb(158.0, 146.0, 96.0), 0.55 * smoothstep(0.15, 0.45, clone));
    let col = mixc(c0, rgb(184.0, 162.0, 108.0), straw) * (1.0 + 0.16 * fine);
    s.m.natural_ok *= 1.0 - cov;
    s.composite(Layer { cov, albedo: col, cls: lc::MARSH, clear: 0.6 + 0.4 * smoothstep(0.04, 0.15, dm), ..Default::default() });
    // open water: the patch's core and a network of creeks (the zero line of a noise, ~6 m wide)
    let core = patch_cov(u, 0.22 * amount, 380.0, gsd, fw);
    let ch = perlin3(SEED_MARSH ^ 0x15, p / 140.0) + 0.3 * perlin3(SEED_MARSH ^ 0x16, p / 45.0) * band(45.0, gsd);
    let ec = band(24.0, gsd);
    let creek = (ec * kernels::band_cov(ch / 0.0107, 3.0, fw) + (1.0 - ec) * 0.05) * smoothstep(0.1, 0.3, dm);
    let water = core.max(creek) * cov;
    if water > 0.0 {
        s.composite(Layer { cov: water, albedo: rgb(40.0, 54.0, 50.0), cls: lc::LAKE, clear: 1.0, ..Default::default() });
    }
}

/// Beaver ponds: lens-shaped ponds dammed on small streams of the forest climates, a ring of
/// dead grey trees around them and the open beaver meadow beyond.
fn beaver_ponds(s: &mut Stack) {
    let t = s.t;
    let hw = s.l.river_hw;
    if hw < 0.6 || hw > 5.0 || t.river_wet < 0.5 || t.temp < -4.0 || t.temp > 12.0 || t.moist < 0.42 || s.l.slope > 0.08 {
        return;
    }
    let rd = river_dist(s);
    if rd > 90.0 {
        return;
    }
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    // one candidate pond per 450 m cell, its centre in the cell's inner half (it stays inside)
    let c = (p / 450.0).floor();
    let h = hash3(SEED_BEAVER, c.x as i64, c.y as i64, c.z as i64);
    let climate = smoothstep(0.42, 0.6, t.moist) * (1.0 - smoothstep(8.0, 12.0, t.temp)) * (0.4 + 0.6 * site_boreal(s));
    if u01k(h, 4) >= 0.6 * climate {
        return;
    }
    let site = (c + DVec3::new(0.25 + 0.5 * u01k(h, 1), 0.25 + 0.5 * u01k(h, 2), 0.25 + 0.5 * u01k(h, 3))) * 450.0;
    let r = 70.0 + 40.0 * u01k(h, 5);
    let ds = (p - site).length();
    if ds >= r {
        return;
    }
    let rho = ds / r;
    let wp = (10.0 + 26.0 * u01k(h, 6)) * (1.0 - rho * rho).sqrt();
    if 2.0 * wp < 0.05 * gsd {
        return;
    }
    let meadow = ((2.2 * wp + 20.0 - rd) / fw + 0.5).clamp(0.0, 1.0);
    if meadow <= 0.0 {
        return;
    }
    s.m.natural_ok *= 1.0 - meadow;
    let grass = rgb(114.0, 120.0, 76.0) * (1.0 + 0.12 * s.pf.detail);
    s.composite(Layer { cov: meadow, albedo: grass, cls: lc::GRASS, clear: 1.0, ..Default::default() });
    // the flooded ring: dead trees standing in shallow water and sedge
    let ring = ((1.5 * wp + 8.0 - rd) / fw + 0.5).clamp(0.0, 1.0);
    if ring > 0.0 {
        s.composite(Layer::paint(ring, rgb(96.0, 104.0, 70.0)));
        let kp = kernel_parts(s, K_SNAGS, ring, s.q_loc, 0.0);
        let ly = kernel_layer(s, K_SNAGS, &kp, DVec3::ONE, DVec3::ONE, 1.0);
        s.composite(ly);
    }
    let pond = ((wp - rd) / fw + 0.5).clamp(0.0, 1.0);
    if pond > 0.0 {
        s.composite(Layer { cov: pond, albedo: rgb(38.0, 46.0, 38.0), cls: lc::LAKE, clear: 1.0, ..Default::default() });
    }
}

// ============================================================================= slot 4: forestry

const SEED_STAND: u64 = 0x7E3C_0001;
const SEED_WIND: u64 = 0x7E3C_0002;
const SEED_FIRE: u64 = 0x7E3C_0003;

fn disturbance(s: &mut Stack) {
    let t = s.t;
    if t.moist < 0.2 || t.temp < -7.0 || s.region.is_none() {
        return;
    }
    let fe = forest_est(s);
    let dry = site_dry_summer(s);
    if fe < 0.05 && dry < 0.05 {
        return;
    }
    if fe > 0.05 {
        stands(s, fe);
        windthrow(s, fe);
    }
    firebreaks(s, fe, dry);
}

/// Managed forest: rectangular stands (a brick pattern of columns 150–300 m wide in the land-use
/// region's frame), some cut in the last 16 years — fresh slash with skid trails and a landing,
/// grass and fireweed, then dense young plantations — and logging roads along column edges.
fn stands(s: &mut Stack, fe: f64) {
    let fi = forestry(s);
    if fi < 0.08 {
        return;
    }
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    // (the stands' frame warped by ~25 m: straight-edged rectangles read as a plan, not a forest)
    // (the pixel fields' noise: no extra noise evaluations, pass B's compile time is at its limit)
    let q = s.q_rot + DVec2::new(s.pf.warp2, s.pf.patch) * 24.0;
    let seed = mix64(s.t.region.id ^ SEED_STAND);
    let wc = 150.0 + 150.0 * u01k(seed, 1);
    let xc = q.x / wc;
    let cx = xc.floor();
    let hc = hash1(seed, cx as i64);
    let hr = wc * (1.1 + 1.6 * u01k(hc, 1));
    let yr = (q.y + hr * u01k(hc, 2)) / hr;
    let cy = yr.floor();
    let fx = xc - cx;
    let fy = yr - cy;
    let bid = hash2(seed, cx as i64, cy as i64);
    let inside = smoothstep(0.35, 0.65, fe);
    // the stand's state: cut `age` years ago (0..16) with a probability by the forestry
    // (~0.5 % of the stands a year, most of the cuts years old and greening: 42 % cut within
    // 16 years, a fifth of them fresh, made the taiga a patchwork of bright rectangles)
    let pc = (0.07 * fi).min(0.09);
    let uc = u01k(bid, 3);
    if uc < pc && inside > 0.0 {
        let age = 16.0 * (uc / pc).powf(0.6);
        let ex = fx.min(1.0 - fx) * wc;
        let ey = fy.min(1.0 - fy) * hr;
        // ragged edges: seed trees and uncut corners left along the border
        let en = 18.0 * s.pf.detail;
        let inside = inside * ((ex.min(ey) - 8.0 + en) / fw + 0.5).clamp(0.0, 1.0);
        let mot = perlin3(SEED_STAND ^ 0x11, p / 6.0) * band(6.0, gsd) + 0.5 * perlin3(SEED_STAND ^ 0x12, p / 2.0) * band(2.0, gsd);
        let (col, clear) = if age < 3.0 {
            // fresh: slash (brown, reddish dead needles), skid trails along the stand, a landing
            // at its edge, a few trees left standing
            let c0 = mixc(rgb(92.0, 84.0, 70.0), rgb(106.0, 90.0, 72.0), smoothstep(-0.3, 0.4, mot));
            let c1 = mixc(c0, s.pal(pal::SOIL), 0.15) * (1.0 + 0.22 * mot);
            let tr = q.x / 19.0 + u01k(bid, 4);
            let trail = kernels::band_cov((tr - tr.floor() - 0.5) * 19.0, 1.6, fw) * band(19.0, gsd) * smoothstep(4.0, 14.0, ey);
            let c2 = mixc(c1, rgb(128.0, 116.0, 94.0), 0.7 * trail);
            let landing = kernels::band_cov(ey, 12.0, fw) * kernels::band_cov(ex - 0.5 * wc, 22.0, fw);
            let col = mixc(c2, rgb(138.0, 126.0, 104.0), landing);
            (col, 0.93)
        } else {
            // grass and herbs, fireweed in flower, then the young trees close in
            let g = smoothstep(3.0, 7.0, age);
            // (the regrowth's green differs by stand, and a 40 m patchwork runs through each)
            let gt = mixc(rgb(92.0, 102.0, 64.0), rgb(78.0, 92.0, 60.0), u01k(bid, 6));
            let mot2 = perlin3(SEED_STAND ^ 0x14, p / 40.0) * band(40.0, gsd);
            let c0 = mixc(rgb(110.0, 104.0, 70.0), gt, g) * (1.0 + 0.16 * mot + 0.14 * mot2);
            let fire =
                smoothstep(-0.1, 0.3, perlin3(SEED_STAND ^ 0x13, p / 14.0) * band(14.0, gsd)) * smoothstep(2.5, 4.0, age) * (1.0 - smoothstep(6.0, 9.0, age));
            let col = mixc(c0, rgb(166.0, 92.0, 132.0), 0.55 * fire * (0.6 + 0.4 * smoothstep(0.4, 0.7, s.bio.style.season)));
            (col, 1.0)
        };
        s.composite(Layer { cov: inside, albedo: col, cls: lc::CLEAR_CUT, clear, ..Default::default() });
        if age > 3.0 {
            // the young trees: planted in rows (the stand's frame), growing with the age
            let grow = ((age - 3.0) / 7.0).clamp(0.0, 1.0);
            let kp = kernel_parts(s, K_YOUNG, inside * (0.25 + 0.75 * grow), q, 0.0);
            let tc = if u01k(bid, 5) < 0.65 { rgb(44.0, 70.0, 42.0) } else { rgb(84.0, 112.0, 50.0) };
            let ly = kernel_layer(s, K_YOUNG, &kp, tc, tc, 0.6 * (age - 2.0));
            s.composite(ly);
        }
    }
    // logging roads along some column edges: a straight gravel track and its cleared verge
    let line = if fx < 0.5 { cx } else { cx + 1.0 };
    let dl = fx.min(1.0 - fx) * wc;
    if u01k(hash1(seed ^ 0x40AD, line as i64), 1) < 0.5 * fi.min(1.0) && 10.0 > 0.03 * gsd {
        let verge = kernels::band_cov(dl, 5.0, fw) * inside;
        if verge > 0.0 {
            s.composite(Layer { cov: verge, albedo: s.pal(pal::GRASS_WET) * 1.1, clear: 1.0, ..Default::default() });
            let road = kernels::band_cov(dl, 1.8, fw) * inside;
            s.composite(Layer { cov: road, albedo: rgb(156.0, 142.0, 116.0) * (1.0 + 0.08 * s.pf.detail), cls: lc::TRACK, clear: 1.0, ..Default::default() });
        }
    }
}

/// Windthrow: patches of forest blown down by a storm — dead needles and upturned roots,
/// trunks lying in the storm's direction (one per land-use region), a few trees standing.
fn windthrow(s: &mut Stack, fe: f64) {
    let c = &s.bio.site;
    let windy = smoothstep(400.0, 700.0, c.precip_mm) * (1.0 - smoothstep(10.0, 16.0, c.temp_c));
    let amount = 0.05 * windy * (0.3 + 0.9 * eco_u2(s)) * smoothstep(0.4, 0.7, fe);
    if amount <= 0.002 {
        return;
    }
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    let u = patch_u(SEED_WIND, p, 260.0, 0.55, gsd);
    let cov = patch_cov(u, amount, 260.0, gsd, fw);
    if cov <= 0.0 {
        return;
    }
    let mot = perlin3(SEED_WIND ^ 0x11, p / 5.0) * band(5.0, gsd);
    let col = mixc(rgb(124.0, 98.0, 70.0), rgb(98.0, 92.0, 66.0), smoothstep(-0.2, 0.3, perlin3(SEED_WIND ^ 0x12, p / 24.0))) * (1.0 + 0.12 * mot);
    s.composite(Layer { cov, albedo: col, cls: lc::CLEAR_CUT, clear: 0.82, ..Default::default() });
    let ang = std::f64::consts::TAU * u01k(mix64(s.t.region.id ^ SEED_WIND), 1);
    let kp = kernel_parts(s, K_LOGS, cov, s.q_loc, ang);
    let mut ly = kernel_layer(s, K_LOGS, &kp, DVec3::ONE, DVec3::ONE, 1.0);
    ly.hmode = hmode::ADD;
    s.composite(ly);
}

/// Firebreaks: straight cleared strips with a track through the forests and scrub of the
/// dry-summer climates (and the managed boreal forest), along the land-use region's frame.
fn firebreaks(s: &mut Stack, fe: f64, dry: f64) {
    let t = s.t;
    let scrub = dry * smoothstep(0.12, 0.2, t.moist) * (1.0 - smoothstep(0.6, 0.7, t.moist)) * s.m.natural_ok * s.m.veg;
    let veg = fe.max(scrub);
    let inten = dry.max(0.6 * site_boreal(s) * forestry(s));
    if veg < 0.3 || inten < 0.1 {
        return;
    }
    let (gsd, fw) = (s.ctx.gsd, s.l.fw);
    let seed = mix64(s.t.region.id ^ SEED_FIRE);
    let hw = 7.0 + 8.0 * dry;
    if 2.0 * hw < 0.03 * gsd {
        return;
    }
    let q = s.q_rot;
    let mut d = 1e9f64;
    for axis in 0..2u64 {
        let sp = 1300.0 + 1200.0 * u01k(seed, 2 + axis);
        let qa = if axis == 0 { q.x } else { q.y };
        let x = qa / sp + u01k(seed, 4 + axis);
        let k = (x + 0.5).floor();
        if u01k(hash1(seed ^ (0xF1 + axis), k as i64), 1) < 0.6 * inten {
            d = d.min((x - k).abs() * sp);
        }
    }
    let on = smoothstep(0.3, 0.5, veg);
    let cov = kernels::band_cov(d, hw, fw) * on;
    if cov <= 0.0 {
        return;
    }
    let ground = mixc(s.pal(pal::SOIL + 1), s.pal(pal::GRASS_DRY), 0.45) * (1.0 + 0.12 * s.pf.detail);
    s.composite(Layer { cov, albedo: ground, cls: lc::BARE, clear: 1.0, ..Default::default() });
    let track = kernels::band_cov(d, 1.6, fw) * on;
    if track > 0.0 {
        s.composite(Layer { cov: track, albedo: rgb(176.0, 160.0, 130.0), cls: lc::TRACK, ..Default::default() });
    }
}

// ============================================================================= canopy: autumn

/// Autumn colours of the deciduous crowns, by weight (yellow, gold, orange, red, brown): the
/// red share is the ecoregion's.
fn autumn_weights(red: f64) -> [f64; 5] {
    [0.3, 0.28, 0.2 - 0.5 * red, red, 0.22 - 0.5 * red]
}

fn autumn_colours() -> [DVec3; 5] {
    [rgb(176.0, 150.0, 62.0), rgb(184.0, 124.0, 48.0), rgb(172.0, 92.0, 40.0), rgb(138.0, 52.0, 40.0), rgb(112.0, 84.0, 54.0)]
}

/// Autumn colours: deciduous crowns in yellow, gold, orange, red and brown over the canopy, by
/// the ecoregion's season; larches (light green, gold in autumn) in the cool forests.
fn canopy(s: &mut Stack) {
    let t = s.t;
    if t.moist < 0.28 || t.temp < -6.0 || t.temp > 20.0 {
        return;
    }
    let aut = smoothstep(0.55, 0.85, s.bio.style.season);
    let larch = 0.32 * (1.0 - smoothstep(1.0, 7.0, t.temp)) * eco_u1(s) * site_boreal(s).max(0.25);
    if aut <= 0.0 && larch <= 0.01 {
        return;
    }
    let fe = forest_est(s);
    if fe <= 0.05 {
        return;
    }
    let conifer = ((1.0 - smoothstep(4.0, 13.0, t.temp)) + s.veg(|v| v.conifer) + s.bio.style.conifer).clamp(0.0, 1.0);
    let leaf = aut * fe * (1.0 - conifer) * 0.85;
    if leaf > 0.01 {
        let kp = kernel_parts(s, K_AUTUMN, leaf, s.q_loc, 0.0);
        let w = autumn_weights(0.06 + 0.3 * eco_u2(s));
        let cols = autumn_colours();
        let mut mean_c = DVec3::ZERO;
        for (c, wi) in cols.iter().zip(w) {
            mean_c += *c * wi;
        }
        let mut col_x = mean_c;
        if kp.x.cov > 0.0 {
            let u = u01k(kp.x.id, 0xA1);
            let mut acc = 0.0;
            for (c, wi) in cols.iter().zip(w) {
                acc += wi;
                if u < acc {
                    col_x = *c;
                    break;
                }
            }
        }
        let ly = kernel_layer(s, K_AUTUMN, &kp, col_x, mean_c, 1.0);
        s.composite(ly);
    }
    let la = larch * fe * conifer.max(0.3);
    if la > 0.01 {
        let kp = kernel_parts(s, K_LARCH, la, s.q_loc, 0.0);
        let c = mixc(rgb(98.0, 124.0, 62.0), rgb(204.0, 156.0, 56.0), aut);
        let ly = kernel_layer(s, K_LARCH, &kp, c, c, 1.0);
        s.composite(ly);
    }
}

#[cfg(test)]
mod tests {
    use crate::noise::*;
    use crate::registry::Registry;
    use crate::world::{Ctx, World};

    #[test]
    fn registry_compiles() {
        let r = Registry::builtin().unwrap();
        for id in [
            "broadleaf_deciduous",
            "mixed_forest",
            "taiga",
            "lichen_woodland",
            "maquis",
            "garrigue",
            "chaparral",
            "feather_grass_steppe",
            "heathland",
            "moorland",
            "subalpine_krummholz",
            "alpine_meadow",
        ] {
            assert!(r.index(id).is_some(), "{id}");
        }
    }

    /// The generated block of `temperate.wgsl` is up to date (`AERIALSYNTH_BLESS=1` rewrites it).
    #[test]
    fn wgsl_consts_up_to_date() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/wgsl/kits/temperate.wgsl");
        let text = std::fs::read_to_string(path).unwrap();
        let want = super::wgsl_consts();
        let (a, b) = (text.find(super::GEN_BEGIN), text.find(super::GEN_END));
        let have = match (a, b) {
            (Some(a), Some(b)) => &text[a..b + super::GEN_END.len()],
            _ => "",
        };
        if have != want {
            if std::env::var("AERIALSYNTH_BLESS").is_ok_and(|v| v == "1") {
                let new = match (a, b) {
                    (Some(a), Some(b)) => format!("{}{}{}", &text[..a], want, &text[b + super::GEN_END.len()..]),
                    _ => format!("{text}\n{want}\n"),
                };
                std::fs::write(path, new).unwrap();
            } else {
                panic!("temperate.wgsl: the generated kernel constants are out of date (AERIALSYNTH_BLESS=1 cargo test -p terragen --lib temperate)");
            }
        }
    }

    /// Places of the default world where the kit draws (lat, lon °): a raised bog, reed marsh
    /// along a river, maquis, managed forest with clear-cuts, lichen woodland.
    pub(super) const PLACES: [(&str, f64, f64); 5] =
        [("bog", 47.092, -2.889), ("marsh", 23.075, -115.365), ("maquis", 38.467, 5.375), ("clearcut", 48.760, 120.806), ("lichen", 62.488, -25.312)];

    fn tile_at(name: &str, z: u8) -> geodesy::tiles::TileId {
        let (_, lat, lon) = PLACES.iter().find(|p| p.0 == name).copied().unwrap();
        geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)
    }

    /// Share of samples differing by more than `tol`.
    #[cfg(feature = "gpu")]
    fn off<T: Copy>(a: &[T], b: &[T], tol: f64, f: impl Fn(T) -> f64) -> f64 {
        a.iter().zip(b).filter(|(x, y)| (f(**x) - f(**y)).abs() > tol).count() as f64 / a.len() as f64
    }

    /// The kit's places on both backends: at most 0.5 % of the pixels off (rgb > 2 DN, class,
    /// elevation > 5 cm).
    #[cfg(feature = "gpu")]
    #[test]
    fn tiles_match_the_cpu() {
        let cfg = crate::Config::default();
        let gen = match crate::gpu::shared().and_then(|g| g.check_generator()).and_then(|_| crate::gpu::GpuGenerator::new(cfg.clone())) {
            Ok(g) => g,
            Err(_) => {
                eprintln!("no GPU that runs the tile generator: skipped");
                return;
            }
        };
        let cpu = crate::NextGenerator::new(cfg);
        let mut worst = 0.0f64;
        for (name, _, _) in PLACES {
            for z in [16u8, 14, 12, 9] {
                let id = tile_at(name, z);
                let g = gen.tiles(&[id]).unwrap().remove(0);
                let c = cpu.tile_cpu(id);
                let rgb = off(&g.rgb, &c.rgb, 2.0, |v| v as f64);
                let cls = off(&g.landcover, &c.landcover, 0.5, |v| v as f64);
                let elev = off(&g.elevation, &c.elevation, 0.05, |v| v as f64);
                eprintln!("{name} z{z}: rgb {:.3}% class {:.3}% elevation {:.3}%", 100.0 * rgb, 100.0 * cls, 100.0 * elev);
                worst = worst.max(rgb).max(cls).max(elev);
            }
        }
        assert!(worst <= 0.005, "worst {:.3}% of pixels off", 100.0 * worst);
    }

    /// Parent ≈ mean of the children (albedo and class histogram) at the kit's places, and the
    /// tiles are deterministic and seamless there.
    #[test]
    fn lod_seams_determinism() {
        let cfg = crate::Config { tile_supersample: 1, ..crate::Config::default() };
        let g = crate::NextGenerator::new(cfg.clone());
        let n = crate::TILE_SIZE;
        for (name, _, _) in PLACES {
            for z in [12u8, 15] {
                let parent = tile_at(name, z);
                let p = g.tile(parent);
                let kids = parent.children().map(|c| g.tile(c));
                let (mut err, mut cnt) = (0.0, 0.0);
                let mut hp = vec![0f64; 256];
                let mut hk = vec![0f64; 256];
                for (k, kid) in kids.iter().enumerate() {
                    let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
                    for j in (0..n).step_by(2) {
                        for i in (0..n).step_by(2) {
                            for c in 0..3 {
                                let a = |ii: usize, jj: usize| kid.albedo[3 * (jj * n + ii) + c] as f64;
                                let m = (a(i, j) + a(i + 1, j) + a(i, j + 1) + a(i + 1, j + 1)) / 4.0;
                                err += (m - p.albedo[3 * ((oy + j / 2) * n + ox + i / 2) + c] as f64).abs();
                                cnt += 1.0;
                            }
                            for (jj, ii) in [(j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1)] {
                                hk[kid.landcover[jj * n + ii] as usize] += 0.25;
                            }
                            hp[p.landcover[(oy + j / 2) * n + ox + i / 2] as usize] += 1.0;
                        }
                    }
                }
                let hist: f64 = hp.iter().zip(&hk).map(|(a, b)| (a - b).abs()).sum::<f64>() / (2.0 * (n * n) as f64);
                eprintln!("{name} z{z}: albedo |parent - mean(children)| {:.2} DN, class histogram {:.3}", err / cnt, hist);
                // (the core alone, without the kit, at these places: up to 14.5 DN and 0.17 at
                // z15 — the canopy's crowns at the crossfade)
                assert!(err / cnt < 12.0, "{name} z{z}: albedo {}", err / cnt);
                assert!(hist < 0.1, "{name} z{z}: class histogram {hist}");
            }
            // determinism (a fresh generator) and the seam to the eastern neighbour
            let id = tile_at(name, 15);
            let a = g.tile(id);
            let b = crate::NextGenerator::new(cfg.clone()).tile(id);
            assert_eq!(a.rgb, b.rgb, "{name}: not deterministic");
            let r = g.tile(geodesy::tiles::TileId::new(id.z, id.x + 1, id.y));
            let (mut across, mut inside) = (0.0, 0.0);
            for j in 0..n {
                for c in 0..3 {
                    across += (a.rgb[3 * (j * n + n - 1) + c] as f64 - r.rgb[3 * (j * n) + c] as f64).abs();
                    inside += (a.rgb[3 * (j * n + n - 2) + c] as f64 - a.rgb[3 * (j * n + n - 1) + c] as f64).abs();
                }
            }
            eprintln!("{name}: seam {across:.0} vs inside {inside:.0}");
            assert!(across < 1.5 * inside + 3.0 * n as f64, "{name}: seam {across} inside {inside}");
        }
    }

    /// The GPU generates the kit's places at every zoom without falling back to the CPU (an
    /// instance family overflowing a block, a shader error): `Backend::Gpu` returns the error.
    #[cfg(feature = "gpu")]
    #[test]
    fn places_stay_on_the_gpu() {
        let Ok(g) = crate::gpu::shared() else { return };
        if g.check_generator().is_err() {
            return;
        }
        let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        for &(name, lat, lon) in &PLACES {
            let ids: Vec<_> = [6u8, 8, 10, 12, 14, 16].iter().map(|&z| geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)).collect();
            if let Err(e) = gen.tiles(&ids) {
                panic!("{name}: GPU generation failed: {e:#}");
            }
        }
        assert!(gen.backend_name().starts_with("GPU"), "{}", gen.backend_name());
    }

    /// GPU tiles/s at the kit's places (8 x 8 tiles per place and zoom; performance budget):
    /// `cargo test -p terragen --release --lib kits::temperate::tests::bench_places -- --ignored --nocapture`.
    /// For the baseline, run it with the kit's `KITS` entry commented out (the module stays).
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn bench_places() {
        let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        gen.tiles(&[geodesy::tiles::TileId::new(3, 4, 3)]).unwrap();
        for z in [8u8, 10, 12, 14, 16] {
            let mut ids = Vec::new();
            for &(_, lat, lon) in &PLACES {
                let c = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
                let n = 1i64 << z;
                for dy in 0..8i64 {
                    for dx in 0..8i64 {
                        ids.push(geodesy::tiles::TileId::new(z, (c.x as i64 + dx - 4).rem_euclid(n) as u32, (c.y as i64 + dy - 4).clamp(0, n - 1) as u32));
                    }
                }
            }
            let t0 = std::time::Instant::now();
            for chunk in ids.chunks(16) {
                gen.tiles(chunk).unwrap();
            }
            let dt = t0.elapsed().as_secs_f64();
            println!("bench z{z}: {} tiles {dt:.2} s {:.1} tiles/s", ids.len(), ids.len() as f64 / dt);
        }
    }

    /// Places of the default world where the kit's classes are (point probes at 8 m pixels;
    /// `cargo test --release -p terragen find_features -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn find_features() {
        use rayon::prelude::*;
        let g = crate::NextGenerator::new(crate::Config::default());
        let n: usize = std::env::var("FIND_N").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
        let hits: Vec<(f64, f64, u8)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = mix64(0xFEA7 ^ i as u64);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin();
                let lon = std::f64::consts::TAU * (u01k(h, 2) - 0.5);
                let (_, _, c) = g.probe(lat, lon, 8.0);
                [45u8, 53, 55, 56, 58, 59, 61, 103].contains(&c).then_some((lat.to_degrees(), lon.to_degrees(), c))
            })
            .collect();
        for c in [45u8, 53, 55, 56, 58, 59, 61, 103] {
            let v: Vec<String> = hits.iter().filter(|h| h.2 == c).take(8).map(|h| format!("{:.3},{:.3}", h.0, h.1)).collect();
            println!("{:10} {:6} {}", crate::landcover::name(c), hits.iter().filter(|h| h.2 == c).count(), v.join(" "));
        }
    }

    /// Places of the default world where the kit's biomes are (`cargo test --release -p terragen
    /// find_places -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn find_places() {
        use rayon::prelude::*;
        let w = World::new(crate::Config::default());
        let reg = Registry::builtin().unwrap();
        let n = 60_000usize;
        let hits: Vec<(f64, f64, String, String, f64, f64, f64)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = mix64(0xF1ED ^ i as u64);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin().to_degrees();
                let lon = 360.0 * u01k(h, 2) - 180.0;
                let ctx = Ctx::new(lat.to_radians(), lon.to_radians(), 100.0, &w.ell);
                let t = w.terrain(&ctx);
                if t.water_kind != 0 || t.ground < 2.0 {
                    return None;
                }
                let [(id, site), _] = crate::eco::sites(&w, ctx.p);
                let e = crate::eco::compute(&w, &reg, id, site);
                let mut b = e.biome;
                for &(below, zb) in &reg.biomes[e.biome as usize].zones {
                    if t.temp < below {
                        b = zb;
                    }
                }
                let k = format!(
                    "{} s{:.2} a{} p{:.0}",
                    crate::atlas::Koppen::ALL[e.climate.koppen as usize].code(),
                    e.style.season,
                    e.style.archetype,
                    e.climate.precip_mm
                );
                Some((lat, lon, reg.biomes[b as usize].name.clone(), k, t.temp, t.moist, t.ground))
            })
            .collect();
        let mut by: std::collections::BTreeMap<String, Vec<_>> = Default::default();
        for h in &hits {
            by.entry(h.2.clone()).or_default().push(h.clone());
        }
        let total = hits.len() as f64;
        let filt = std::env::var("FIND").unwrap_or_default();
        for (b, v) in &by {
            let v: Vec<_> = v
                .iter()
                .filter(|h| {
                    filt.is_empty()
                        || h.3.contains(&filt)
                        || (filt == "autumn" && h.3.split(' ').nth(1).is_some_and(|x| x[1..].parse::<f64>().unwrap_or(0.0) > 0.7) && h.5 > 0.45)
                })
                .collect();
            let s: Vec<String> = v.iter().take(6).map(|h| format!("{:.3},{:.3} {} T{:.0} M{:.2} h{:.0}", h.0, h.1, h.3, h.4, h.5, h.6)).collect();
            println!("{b:22} {:5.2}%  {}", 100.0 * v.len() as f64 / total, s.join(" | "));
        }
    }
}
