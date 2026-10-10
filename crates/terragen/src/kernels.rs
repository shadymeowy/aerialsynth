//! The kernel library (`docs/design/terrain-next.md` §3.7): small surface generators that
//! biomes compose (YAML `layers:`) and kits call from code. The WGSL twin is
//! `gpu/wgsl/kernels.wgsl` (same parameters, same hashes; `gpu::tests::kernels_match_the_cpu`
//! compares them on random inputs).
//!
//! Every kernel works in a local planar frame `q` (metres; the land-use region's east / north
//! frame for biome layers) with the ECEF point `p` for world-space noise, and has
//! * an explicit evaluation ([`explicit`]): coverage, albedo, height, emission;
//! * a feature size ([`size`]): the explicit result fades into the calibrated mean over
//!   `explicit = smoothstep(1.2·gsd, 3·gsd, size)` ([`eval`]);
//! * a calibrated mean ([`calibrate`], [`KMean`]): its explicit result integrated over 4096
//!   quasi-random points at a fine pixel size. A coarse pixel shows that mean, so a parent tile
//!   is the mean of its children per kernel.
//!
//! Coverage is linear in `amount` (instance densities, fill fractions and thresholds scale
//! with it), so a masked layer's mean is `amount · mean`.
//!
//! Hashes are those of [`crate::noise`] (`hash2`, `u01k`), lattices are floored in f64 (CPU) or
//! from f32 local coordinates (GPU): discrete choices only flip within ~1e-7 of a cell edge.

use crate::noise::*;
use glam::{DVec2, DVec3};

/// Kernel kinds (`KIND_*` in kernels.wgsl). Kits' kernels get ids from [`KIT_BASE`] on.
pub mod kind {
    pub const SCATTER: u32 = 0;
    pub const ROWS: u32 = 1;
    pub const CELLS: u32 = 2;
    pub const STRIPES: u32 = 3;
    pub const CONTOURS: u32 = 4;
    pub const RADIAL: u32 = 5;
    pub const CRESCENT: u32 = 6;
    pub const LOBES: u32 = 7;
    pub const PATCHES: u32 = 8;
    pub const LINEAR: u32 = 9;
    pub const STAMP: u32 = 10;
    pub const CANOPY: u32 = 11;
    pub const WATER: u32 = 12;
    /// settlements (the settlements kit provides it); the core evaluates nothing
    pub const CITY: u32 = 13;
}

/// First kind id of the kits' kernels (in the order of `kits::KITS`).
pub const KIT_BASE: u32 = 64;

/// Crown / instance shapes of `scatter` (and the canopy's crowns).
pub mod shape {
    pub const DISC: u32 = 0;
    pub const DOME: u32 = 1;
    pub const CONE: u32 = 2;
    pub const UMBRELLA: u32 = 3;
    pub const STAR: u32 = 4;
    pub const RECT: u32 = 5;
    pub const CRESCENT: u32 = 6;
    pub const RING: u32 = 7;
    pub const NAMES: [&str; 8] = ["disc", "dome", "cone", "umbrella", "star", "rect", "crescent", "ring"];
    pub fn id(n: &str) -> Option<u32> {
        NAMES.iter().position(|x| *x == n).map(|i| i as u32)
    }
}

/// A kernel instance: kind, seed, 16 numbers and 3 colours (linear RGB).
#[derive(Clone, Copy, Debug, Default)]
pub struct KParams {
    pub kind: u32,
    pub seed: u64,
    pub v: [f64; 16],
    pub col: [DVec3; 3],
}

/// Inputs of a kernel at a sample.
#[derive(Clone, Copy, Debug)]
pub struct KIn {
    /// local planar position (m)
    pub q: DVec2,
    /// ECEF position (world-space noise)
    pub p: DVec3,
    /// unit vectors of the frame's axes (ECEF)
    pub east: DVec3,
    pub north: DVec3,
    pub gsd: f64,
    /// filter width of the sub-sample (m)
    pub fw: f64,
    /// 0..1: densities / fill fractions scale with it (masks)
    pub amount: f64,
    /// kernel-specific: contours (height m, slope), stripes / rows / crescent / lobes (angle rad),
    /// linear (signed distance m, along m, half width m), water (depth m), stamp (u, v, half
    /// sizes m)
    pub aux: [f64; 4],
}

/// A kernel's result.
#[derive(Clone, Copy, Debug, Default)]
pub struct KOut {
    pub cov: f64,
    pub albedo: DVec3,
    /// height above the ground where covered (m)
    pub dh: f64,
    /// emission (× the third colour)
    pub emit: f64,
    /// the instance / cell hash (0: none), for kits
    pub id: u64,
}

/// The calibrated mean of a kernel instance (per unit `amount`).
#[derive(Clone, Copy, Debug, Default)]
pub struct KMean {
    pub cov: f64,
    pub albedo: DVec3,
    pub dh: f64,
    pub emit: f64,
}

/// One parameter of a kernel's YAML schema.
#[derive(Clone, Copy, Debug)]
pub enum P {
    /// a number into `v[slot]`, its range and default
    Num(&'static str, usize, f64, f64, f64),
    /// `[a, b]` into `v[slot], v[slot + 1]` (a number x means [x, x]), range, default
    Range(&'static str, usize, f64, f64, [f64; 2]),
    /// one of the names, its index into `v[slot]`, default index
    Enum(&'static str, usize, &'static [&'static str], usize),
    /// a colour into `col[i]` and its default
    Col(&'static str, usize, C),
}

/// A default colour: a palette entry or sRGB 0..255.
#[derive(Clone, Copy, Debug)]
pub enum C {
    N(&'static str),
    Rgb([f64; 3]),
}

/// A kernel: its YAML name, kind id, parameter schema, CPU evaluation and limits.
pub struct KernelSpec {
    pub name: &'static str,
    pub kind: u32,
    pub params: &'static [P],
    /// the slot of `v` holding the feature size (m) of the crossfade
    pub size_slot: usize,
    /// the size per sample instead (contours: bench width; linear: road width)
    pub dyn_size: Option<fn(&KParams, &KIn) -> f64>,
    /// how far instances reach from their site in cells (the slots of reach and cell size);
    /// checked against `max_reach`
    pub reach: Option<(usize, usize, f64)>,
    pub default_hmode: &'static str,
    pub eval: fn(&KParams, &KIn) -> KOut,
    /// the WGSL function of a kit kernel (`fn(li: u32, k: KIn) -> KOut`); "" for the core's
    pub wgsl: &'static str,
}

const SCATTER_P: &[P] = &[
    P::Num("cell", 0, 0.2, 5000.0, 10.0),
    P::Num("density", 1, 0.0, 1.0, 0.5),
    P::Enum("shape", 2, &shape::NAMES, 1),
    P::Range("radius", 3, 0.01, 5000.0, [2.0, 3.0]),
    P::Range("height", 5, -500.0, 5000.0, [1.0, 2.0]),
    P::Num("jitter", 7, 0.0, 1.0, 0.8),
    P::Num("colour_var", 8, 0.0, 1.0, 0.2),
    P::Num("shade", 9, 0.0, 1.0, 0.3),
    P::Num("aspect", 10, 0.1, 10.0, 1.0),
    P::Enum("orient", 11, &["random", "aux"], 0),
    P::Num("ring", 12, 0.05, 1.0, 0.3),
    P::Num("second", 13, 0.0, 1.0, 0.0),
    P::Col("colour", 0, C::N("crown_decid")),
    P::Col("colour2", 1, C::N("crown_dry")),
];
const ROWS_P: &[P] = &[
    P::Num("spacing", 0, 0.2, 500.0, 3.0),
    P::Num("along", 1, 0.0, 500.0, 0.0),
    P::Num("radius", 2, 0.02, 100.0, 0.6),
    P::Num("height", 3, -10.0, 100.0, 1.0),
    P::Num("angle", 4, -10.0, 10.0, 0.0),
    P::Enum("direction", 5, &["param", "aux"], 0),
    P::Num("gaps", 6, 0.0, 1.0, 0.05),
    P::Num("colour_var", 8, 0.0, 1.0, 0.15),
    P::Col("colour", 0, C::N("crown_decid")),
];
const CELLS_P: &[P] = &[
    P::Num("cell", 0, 0.5, 100000.0, 20.0),
    P::Num("jitter", 1, 0.0, 1.0, 0.85),
    P::Num("fill", 2, 0.0, 1.0, 1.0),
    P::Num("edge", 3, 0.0, 1000.0, 1.0),
    P::Num("edge_height", 4, -100.0, 100.0, 0.0),
    P::Num("colour_var", 5, 0.0, 1.0, 0.1),
    P::Num("aspect", 6, 0.1, 10.0, 1.0),
    P::Num("fill_height", 7, -100.0, 100.0, 0.0),
    P::Col("colour", 0, C::N("grass_dry")),
    P::Col("colour2", 1, C::N("grass_wet")),
    P::Col("edge_colour", 2, C::N("soil")),
];
const STRIPES_P: &[P] = &[
    P::Num("wavelength", 0, 0.5, 100000.0, 50.0),
    P::Num("angle", 1, -10.0, 10.0, 0.0),
    P::Enum("direction", 2, &["param", "aux"], 0),
    P::Num("sharpness", 3, 0.2, 8.0, 1.5),
    P::Num("height", 4, -1000.0, 1000.0, 0.0),
    P::Num("threshold", 5, 0.0, 1.0, 0.0),
    P::Col("colour", 0, C::N("grass_dry")),
    P::Col("colour2", 1, C::N("grass_wet")),
];
const CONTOURS_P: &[P] = &[
    P::Num("step", 0, 0.1, 1000.0, 3.0),
    P::Num("line", 1, 0.0, 100.0, 0.5),
    P::Enum("mode", 2, &["lines", "terraces"], 0),
    P::Num("phase", 3, 0.0, 1.0, 0.0),
    P::Num("riser", 4, 0.01, 1.0, 0.2),
    P::Col("colour", 0, C::N("soil")),
    P::Col("colour2", 1, C::N("grass_wet")),
];
const RADIAL_P: &[P] = &[
    P::Num("cell", 0, 1.0, 1.0e6, 1000.0),
    P::Num("density", 1, 0.0, 1.0, 0.3),
    P::Range("radius", 2, 0.1, 1.0e6, [200.0, 400.0]),
    P::Range("height", 4, -5000.0, 10000.0, [20.0, 60.0]),
    P::Num("exponent", 6, 0.1, 8.0, 1.5),
    P::Num("arms", 7, 0.0, 12.0, 0.0),
    P::Num("arm_amp", 8, 0.0, 0.9, 0.3),
    P::Num("crater", 9, 0.0, 0.9, 0.0),
    P::Num("crater_depth", 10, 0.0, 2.0, 0.3),
    P::Num("jitter", 11, 0.0, 1.0, 0.8),
    P::Col("colour", 0, C::N("rock")),
    P::Col("colour2", 1, C::N("soil")),
];
const CRESCENT_P: &[P] = &[
    P::Num("cell", 0, 1.0, 1.0e5, 300.0),
    P::Num("density", 1, 0.0, 1.0, 0.4),
    P::Range("radius", 2, 0.5, 1.0e5, [40.0, 90.0]),
    P::Num("height_ratio", 4, 0.0, 2.0, 0.15),
    P::Num("horns", 5, 0.1, 1.0, 0.55),
    P::Enum("direction", 6, &["param", "aux"], 0),
    P::Num("angle", 7, -10.0, 10.0, 0.0),
    P::Num("jitter", 8, 0.0, 1.0, 0.8),
    P::Col("colour", 0, C::N("sand")),
    P::Col("colour2", 1, C::N("soil")),
];
const LOBES_P: &[P] = &[
    P::Num("cell", 0, 1.0, 1.0e6, 3000.0),
    P::Num("density", 1, 0.0, 1.0, 0.3),
    P::Range("radius", 2, 1.0, 1.0e6, [400.0, 900.0]),
    P::Num("span", 4, 0.1, 6.3, 1.4),
    P::Num("margin", 5, 0.0, 0.9, 0.25),
    P::Num("channels", 6, 0.0, 40.0, 7.0),
    P::Num("height", 7, -100.0, 500.0, 5.0),
    P::Enum("direction", 8, &["random", "aux"], 0),
    P::Num("jitter", 9, 0.0, 1.0, 0.8),
    P::Col("colour", 0, C::N("sand")),
    P::Col("colour2", 1, C::N("soil")),
];
const PATCHES_P: &[P] = &[
    P::Num("scale", 0, 0.5, 1.0e6, 200.0),
    P::Num("fraction", 1, 0.0, 1.0, 0.3),
    P::Num("octaves", 2, 1.0, 3.0, 2.0),
    P::Num("roughness", 3, 0.0, 1.0, 0.45),
    P::Num("height", 4, -100.0, 100.0, 0.0),
    P::Num("colour_var", 5, 0.0, 1.0, 0.15),
    P::Col("colour", 0, C::N("soil")),
    P::Col("colour2", 1, C::N("grass_dry")),
];
const LINEAR_P: &[P] = &[
    P::Enum("profile", 0, &["flat", "crowned", "ridge", "ditch"], 0),
    P::Num("dash", 1, 0.0, 1000.0, 0.0),
    P::Num("duty", 2, 0.0, 1.0, 0.5),
    P::Num("shoulder", 3, 0.0, 100.0, 0.0),
    P::Num("height", 4, -100.0, 100.0, 0.0),
    P::Num("marking", 5, 0.0, 10.0, 0.0),
    P::Col("colour", 0, C::Rgb([78.0, 78.0, 82.0])),
    P::Col("shoulder_colour", 1, C::Rgb([162.0, 146.0, 120.0])),
    P::Col("marking_colour", 2, C::Rgb([230.0, 230.0, 225.0])),
];
const STAMP_P: &[P] = &[
    P::Enum("template", 0, &["pad", "strip", "blocks", "panels"], 0),
    P::Num("a", 1, 0.0, 1.0e4, 10.0),
    P::Num("b", 2, 0.0, 1.0e4, 2.0),
    P::Num("height", 3, -100.0, 500.0, 0.0),
    P::Num("lights", 4, 0.0, 16.0, 0.0),
    P::Col("colour", 0, C::Rgb([176.0, 174.0, 168.0])),
    P::Col("colour2", 1, C::Rgb([78.0, 78.0, 82.0])),
    P::Col("light_colour", 2, C::Rgb([255.0, 240.0, 220.0])),
];
const CANOPY_P: &[P] = &[
    P::Num("cell", 0, 2.0, 500.0, 40.0),
    P::Num("emergent", 1, 0.0, 1.0, 0.25),
    P::Num("height", 2, 0.0, 120.0, 30.0),
    P::Num("emergent_height", 3, 0.0, 150.0, 50.0),
    P::Num("clump", 4, 0.5, 100.0, 7.0),
    P::Num("colour_var", 5, 0.0, 1.0, 0.25),
    P::Num("flowering", 6, 0.0, 1.0, 0.03),
    P::Col("colour", 0, C::N("crown_tropic")),
    P::Col("colour2", 1, C::Rgb([150.0, 150.0, 60.0])),
    P::Col("flower_colour", 2, C::Rgb([170.0, 80.0, 70.0])),
];
const WATER_P: &[P] = &[
    P::Num("depth_scale", 0, 0.1, 1.0e4, 10.0),
    P::Num("sediment", 1, 0.0, 1.0, 0.06),
    P::Num("sediment_scale", 2, 1.0, 1.0e5, 300.0),
    P::Num("foam_depth", 3, 0.0, 100.0, 0.0),
    P::Num("foam_width", 4, 0.01, 100.0, 0.1),
    P::Col("colour", 0, C::Rgb([48.0, 104.0, 110.0])),
    P::Col("colour2", 1, C::Rgb([14.0, 36.0, 66.0])),
    P::Col("foam_colour", 2, C::Rgb([225.0, 232.0, 230.0])),
];
const CITY_P: &[P] = &[P::Num("block", 0, 10.0, 2000.0, 100.0)];

/// The core kernels.
pub static CORE: &[KernelSpec] = &[
    KernelSpec {
        name: "scatter",
        kind: kind::SCATTER,
        params: SCATTER_P,
        size_slot: 0,
        dyn_size: None,
        reach: Some((4, 0, 1.0)),
        default_hmode: "max",
        eval: scatter,
        wgsl: "",
    },
    KernelSpec { name: "rows", kind: kind::ROWS, params: ROWS_P, size_slot: 0, dyn_size: None, reach: None, default_hmode: "max", eval: rows, wgsl: "" },
    KernelSpec { name: "cells", kind: kind::CELLS, params: CELLS_P, size_slot: 0, dyn_size: None, reach: None, default_hmode: "add", eval: cells, wgsl: "" },
    KernelSpec {
        name: "stripes",
        kind: kind::STRIPES,
        params: STRIPES_P,
        size_slot: 0,
        dyn_size: None,
        reach: None,
        default_hmode: "add",
        eval: stripes,
        wgsl: "",
    },
    KernelSpec {
        name: "contours",
        kind: kind::CONTOURS,
        params: CONTOURS_P,
        size_slot: 0,
        dyn_size: Some(contours_size),
        reach: None,
        default_hmode: "add",
        eval: contours,
        wgsl: "",
    },
    KernelSpec {
        name: "radial",
        kind: kind::RADIAL,
        params: RADIAL_P,
        size_slot: 2,
        dyn_size: None,
        reach: Some((3, 0, 1.0)),
        default_hmode: "add",
        eval: radial,
        wgsl: "",
    },
    KernelSpec {
        name: "crescent",
        kind: kind::CRESCENT,
        params: CRESCENT_P,
        size_slot: 2,
        dyn_size: None,
        reach: Some((3, 0, 1.0)),
        default_hmode: "add",
        eval: crescent,
        wgsl: "",
    },
    KernelSpec {
        name: "lobes",
        kind: kind::LOBES,
        params: LOBES_P,
        size_slot: 2,
        dyn_size: None,
        reach: Some((3, 0, 1.0)),
        default_hmode: "add",
        eval: lobes,
        wgsl: "",
    },
    KernelSpec {
        name: "patches",
        kind: kind::PATCHES,
        params: PATCHES_P,
        size_slot: 0,
        dyn_size: None,
        reach: None,
        default_hmode: "add",
        eval: patches,
        wgsl: "",
    },
    KernelSpec {
        name: "linear",
        kind: kind::LINEAR,
        params: LINEAR_P,
        size_slot: 3,
        dyn_size: Some(linear_size),
        reach: None,
        default_hmode: "blend",
        eval: linear,
        wgsl: "",
    },
    KernelSpec { name: "stamp", kind: kind::STAMP, params: STAMP_P, size_slot: 1, dyn_size: None, reach: None, default_hmode: "blend", eval: stamp, wgsl: "" },
    KernelSpec {
        name: "canopy",
        kind: kind::CANOPY,
        params: CANOPY_P,
        size_slot: 4,
        dyn_size: None,
        reach: None,
        default_hmode: "max",
        eval: canopy,
        wgsl: "",
    },
    KernelSpec { name: "water", kind: kind::WATER, params: WATER_P, size_slot: 2, dyn_size: None, reach: None, default_hmode: "none", eval: water, wgsl: "" },
    KernelSpec { name: "city", kind: kind::CITY, params: CITY_P, size_slot: 0, dyn_size: None, reach: None, default_hmode: "blend", eval: city, wgsl: "" },
];

/// The kernel of a YAML name (core first, then the kits').
pub fn spec(name: &str) -> Option<&'static KernelSpec> {
    CORE.iter().find(|s| s.name == name).or_else(|| crate::kits::kernels().find(|s| s.name == name))
}

/// The kernel of a kind id.
pub fn spec_of(kind: u32) -> Option<&'static KernelSpec> {
    CORE.iter().find(|s| s.kind == kind).or_else(|| crate::kits::kernels().find(|s| s.kind == kind))
}

pub fn spec_names() -> Vec<&'static str> {
    CORE.iter().map(|s| s.name).chain(crate::kits::kernels().map(|s| s.name)).collect()
}

/// Parameters from YAML (`params:` of a layer); unknown names and out-of-range values are
/// errors naming the parameter.
pub fn compile_params(
    spec: &KernelSpec,
    y: &std::collections::BTreeMap<String, serde_yaml::Value>,
    seed: u64,
    pal: &crate::registry::BiomePal,
) -> Result<KParams, String> {
    let mut k = KParams { kind: spec.kind, seed, ..Default::default() };
    let num = |v: &serde_yaml::Value| v.as_f64();
    for p in spec.params {
        match *p {
            P::Num(_, s, _, _, d) => k.v[s] = d,
            P::Range(_, s, _, _, d) => {
                k.v[s] = d[0];
                k.v[s + 1] = d[1];
            }
            P::Enum(_, s, _, d) => k.v[s] = d as f64,
            P::Col(_, i, d) => k.col[i] = default_colour(d, pal),
        }
    }
    for (name, v) in y {
        let Some(p) = spec.params.iter().find(|p| pname(p) == name) else {
            return Err(format!("unknown parameter {name:?} of {} (one of {})", spec.name, spec.params.iter().map(pname).collect::<Vec<_>>().join(", ")));
        };
        match *p {
            P::Num(n, s, lo, hi, _) => match num(v) {
                Some(x) if (lo..=hi).contains(&x) => k.v[s] = x,
                _ => return Err(format!("{n}: a number in [{lo}, {hi}] (is {v:?})")),
            },
            P::Range(n, s, lo, hi, _) => {
                let (a, b) = match v {
                    serde_yaml::Value::Sequence(l) if l.len() == 2 => (l[0].as_f64(), l[1].as_f64()),
                    _ => (num(v), num(v)),
                };
                match (a, b) {
                    (Some(a), Some(b)) if a <= b && (lo..=hi).contains(&a) && (lo..=hi).contains(&b) => {
                        k.v[s] = a;
                        k.v[s + 1] = b;
                    }
                    _ => return Err(format!("{n}: [a, b] with {lo} <= a <= b <= {hi} (is {v:?})")),
                }
            }
            P::Enum(n, s, names, _) => match v.as_str().and_then(|x| names.iter().position(|m| *m == x)) {
                Some(i) => k.v[s] = i as f64,
                None => return Err(format!("{n}: one of {names:?} (is {v:?})")),
            },
            P::Col(n, i, _) => {
                let c = match v {
                    serde_yaml::Value::String(s) => named_colour(s, pal),
                    serde_yaml::Value::Sequence(l) if l.len() == 3 => {
                        let c: Vec<f64> = l.iter().filter_map(|x| x.as_f64()).collect();
                        (c.len() == 3 && c.iter().all(|x| (0.0..=255.0).contains(x))).then(|| crate::surface::srgb(c[0], c[1], c[2]))
                    }
                    _ => None,
                };
                match c {
                    Some(c) => k.col[i] = c,
                    None => return Err(format!("{n}: [r, g, b] (0..255) or a palette name (is {v:?})")),
                }
            }
        }
    }
    // the GPU reads the parameters as f32: both backends use those values (a noise wavelength
    // off by one f32 ulp shifts its phase by ~|p| / λ · 6e-8 cells at ECEF coordinates)
    k.v = k.v.map(|x| x as f32 as f64);
    k.col = k.col.map(|c| c.as_vec3().as_dvec3());
    Ok(k)
}

fn pname(p: &P) -> &'static str {
    match *p {
        P::Num(n, ..) | P::Range(n, ..) | P::Enum(n, ..) | P::Col(n, ..) => n,
    }
}

fn named_colour(n: &str, pal: &crate::registry::BiomePal) -> Option<DVec3> {
    crate::registry::pal::KEYS.iter().find(|k| k.0 == n && k.2 > 0).map(|k| pal.c[k.1])
}

fn default_colour(d: C, pal: &crate::registry::BiomePal) -> DVec3 {
    match d {
        C::N(n) => named_colour(n, pal).unwrap_or(DVec3::splat(0.5)),
        C::Rgb([r, g, b]) => crate::surface::srgb(r, g, b),
    }
}

/// The feature size (m) of an instance (the crossfade's).
pub fn size(k: &KParams) -> f64 {
    spec_of(k.kind).map_or(0.0, |s| k.v[s.size_slot])
}

/// The size at a sample (kernels whose features scale with the terrain).
pub fn size_at(k: &KParams, i: &KIn) -> f64 {
    match spec_of(k.kind) {
        Some(s) => match s.dyn_size {
            Some(f) => f(k, i),
            None => k.v[s.size_slot],
        },
        None => 0.0,
    }
}

/// Instances must stay within the cells the kernel looks at (bounded work, no cut shapes):
/// their reach from the site, in cells, at most the kernel's limit.
pub fn check_reach(k: &KParams) -> Result<(), String> {
    let Some((r, c, max)) = spec_of(k.kind).and_then(|s| s.reach) else { return Ok(()) };
    let reach = k.v[r] / k.v[c].max(1e-9);
    if reach > max + 1e-9 {
        return Err(format!("instances reach {reach:.2} cells from their site (at most {max}: shapes would be cut at cell borders)"));
    }
    Ok(())
}

/// Explicit evaluation (no crossfade).
pub fn explicit(k: &KParams, i: &KIn) -> KOut {
    match spec_of(k.kind) {
        Some(s) => (s.eval)(k, i),
        None => KOut::default(),
    }
}

/// The kernel at a sample: explicit where its features are resolved, the calibrated mean
/// where not, crossfaded over `smoothstep(1.2 gsd, 3 gsd, size)`.
pub fn eval(k: &KParams, mean: &KMean, i: &KIn) -> KOut {
    let e = smoothstep(1.2 * i.gsd, 3.0 * i.gsd, size_at(k, i));
    let mc = mean.cov * i.amount.clamp(0.0, 1.0);
    if e <= 0.0 {
        return KOut { cov: mc, albedo: mean.albedo, dh: mean.dh, emit: mean.emit * i.amount, id: 0 };
    }
    let x = explicit(k, i);
    if e >= 1.0 {
        return x;
    }
    let cov = e * x.cov + (1.0 - e) * mc;
    if cov <= 0.0 {
        return KOut { emit: e * x.emit + (1.0 - e) * mean.emit * i.amount, ..Default::default() };
    }
    let (we, wm) = (e * x.cov / cov, (1.0 - e) * mc / cov);
    KOut { cov, albedo: x.albedo * we + mean.albedo * wm, dh: x.dh * we + mean.dh * wm, emit: e * x.emit + (1.0 - e) * mean.emit * i.amount, id: x.id }
}

/// The mean of an instance: its explicit result over 4096 Halton points spread over 64 × 64
/// feature sizes, at a pixel size of 1/50 of the feature (everything resolved).
pub fn calibrate(k: &KParams) -> KMean {
    let sz = size(k).max(0.05);
    let span = 64.0 * sz;
    let n = 4096;
    let (mut c, mut a, mut h, mut e) = (0.0, DVec3::ZERO, 0.0, 0.0);
    let base = DVec3::new(6_371_000.0, 0.0, 0.0);
    for j in 0..n {
        let q = DVec2::new(halton(j + 1, 2) * span, halton(j + 1, 3) * span);
        let p = base + DVec3::new(0.0, q.x, q.y);
        let i = KIn {
            q,
            p,
            east: DVec3::Y,
            north: DVec3::Z,
            gsd: sz / 50.0,
            fw: sz / 50.0,
            amount: 1.0,
            // contours: a 30 % slope; linear: centred band of 2 sizes; water: 2 sizes deep
            aux: calib_aux(k, q, sz),
        };
        let o = explicit(k, &i);
        c += o.cov;
        a += o.albedo * o.cov;
        h += o.dh * o.cov;
        e += o.emit;
    }
    let nf = n as f64;
    KMean { cov: c / nf, albedo: if c > 0.0 { a / c } else { DVec3::ZERO }, dh: if c > 0.0 { h / c } else { 0.0 }, emit: e / nf }
}

fn calib_aux(k: &KParams, q: DVec2, sz: f64) -> [f64; 4] {
    match k.kind {
        kind::CONTOURS => [0.3 * q.x, 0.3, 0.0, 0.0],
        kind::LINEAR => [(q.y % (8.0 * sz)) - 4.0 * sz, q.x, sz, 0.0],
        kind::WATER => [2.0 * sz, 0.0, 0.0, 0.0],
        kind::STAMP => [(q.x % (16.0 * sz)) - 8.0 * sz, (q.y % (16.0 * sz)) - 8.0 * sz, 8.0 * sz, 8.0 * sz],
        _ => [0.0; 4],
    }
}

/// Radical inverse in base `b` (Halton).
pub fn halton(mut i: usize, b: usize) -> f64 {
    let (mut f, mut r) = (1.0, 0.0);
    while i > 0 {
        f /= b as f64;
        r += f * (i % b) as f64;
        i /= b;
    }
    r
}

// ============================================================================= helpers

/// Coverage of a band of half-width `hw` at distance `d`, box-filtered with width `fw`.
#[inline]
pub fn band_cov(d: f64, hw: f64, fw: f64) -> f64 {
    ((hw - d.abs()) / fw.max(1e-9) + 0.5).clamp(0.0, 1.0)
}

/// Inside test of a shape at `rel` (m, instance frame) of radius `r`: (coverage, x = d / r).
fn shape_at(sh: u32, rel: DVec2, r: f64, aspect: f64, ring: f64, fw: f64, phase: f64) -> (f64, f64) {
    match sh {
        shape::STAR => {
            let ang = rel.y.atan2(rel.x) + phase;
            let re = r * (0.82 + 0.18 * (8.0 * ang).cos());
            let d = rel.length();
            (band_cov(d, re, fw) * (d <= re + fw) as u8 as f64, (d / re.max(1e-6)).min(1.0))
        }
        shape::RECT => {
            let half = DVec2::new(r, r * aspect);
            let e = half - rel.abs();
            ((e.min_element() / fw.max(1e-9) + 0.5).clamp(0.0, 1.0), (rel.abs() / half).max_element().min(1.0))
        }
        shape::CRESCENT => {
            // outer disc minus an inner disc pushed downwind (+x)
            let d = rel.length();
            let di = (rel - DVec2::new(0.55 * r, 0.0)).length();
            let c = band_cov(d, r, fw) * (1.0 - band_cov(di, 0.85 * r, fw));
            (c, (d / r).min(1.0))
        }
        shape::RING => {
            let d = rel.length();
            let w = (ring * r).max(0.05);
            (band_cov(d - (r - 0.5 * w), 0.5 * w, fw), (d / r).min(1.0))
        }
        _ => {
            let q = DVec2::new(rel.x, rel.y / aspect.max(0.1));
            let d = q.length();
            (band_cov(d, r, fw) * (d <= r + fw) as u8 as f64, (d / r.max(1e-6)).min(1.0))
        }
    }
}

/// Height profile of a shape at x = d / r (0 centre, 1 rim).
pub fn profile(sh: u32, x: f64) -> f64 {
    match sh {
        shape::DISC | shape::RECT => 1.0,
        shape::CONE => 0.08 + 0.92 * (1.0 - x).powf(1.15),
        shape::UMBRELLA => 0.72 + 0.28 * (1.0 - x.powi(6)).max(0.0),
        shape::STAR => 0.6 + 0.4 * (1.0 - x),
        shape::CRESCENT => (1.0 - x * x).max(0.0).sqrt(),
        shape::RING => 1.0 - (2.0 * x - 1.0).abs().min(1.0) * 0.5,
        _ => 0.12 + 0.88 * (1.0 - x * x).max(0.0).sqrt(),
    }
}

fn rot(v: DVec2, a: f64) -> DVec2 {
    let (s, c) = a.sin_cos();
    DVec2::new(v.x * c + v.y * s, -v.x * s + v.y * c)
}

// ============================================================================= kernels

/// `scatter`: instances on a jittered grid (3 × 3 cells), the highest wins the colour.
/// v: cell, density, shape, radius [3, 4], height [5, 6], jitter, colour_var, shade, aspect,
/// orient, ring, second.
fn scatter(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let dens = v[1] * i.amount;
    let sh = v[2] as u32;
    let jit = v[7];
    let cq = i.q / cell;
    let cf = cq.floor();
    let (ix, iy) = (cf.x as i64, cf.y as i64);
    let mut out = KOut::default();
    let mut best = f64::NEG_INFINITY;
    if dens <= 0.0 {
        return out;
    }
    for dy in -1..=1i64 {
        for dx in -1..=1i64 {
            let h = hash2(k.seed, ix + dx, iy + dy);
            if u01k(h, 3) >= dens {
                continue;
            }
            let c = DVec2::new((ix + dx) as f64 + 0.5 + jit * (u01k(h, 1) - 0.5), (iy + dy) as f64 + 0.5 + jit * (u01k(h, 2) - 0.5)) * cell;
            let r = v[3] + (v[4] - v[3]) * u01k(h, 4);
            let mut rel = i.q - c;
            if rel.length_squared() > (r * v[10].max(1.0) + i.fw) * (r * v[10].max(1.0) + i.fw) {
                continue;
            }
            let ang = if v[11] >= 1.0 { i.aux[0] } else { u01k(h, 8) * std::f64::consts::TAU };
            rel = rot(rel, ang);
            let (cov, x) = shape_at(sh, rel, r, v[10], v[12], i.fw, u01k(h, 9) * 6.3);
            if cov <= 0.0 {
                continue;
            }
            let hh = (v[5] + (v[6] - v[5]) * u01k(h, 5)) * profile(sh, x);
            if hh > best {
                best = hh;
                let base = if u01k(h, 6) < v[13] { k.col[1] } else { k.col[0] };
                let tint = 1.0 + v[8] * (u01k(h, 7) - 0.5);
                out.albedo = base * tint * (1.0 - v[9] * 0.5 * x);
                out.dh = hh;
                out.id = h;
            }
            out.cov = out.cov.max(cov);
        }
    }
    out
}

/// `rows`: planting rows (along > 0: a grid of plants) in a direction.
/// v: spacing, along, radius, height, angle, direction, gaps, -, colour_var.
fn rows(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let ang = if v[5] >= 1.0 { i.aux[0] } else { v[4] };
    let q = rot(i.q, ang);
    let (sx, sy) = (v[0], v[1]);
    let row = (q.y / sx).round();
    let dv = q.y - row * sx;
    let (d, j) = if sy > 0.0 {
        let j = (q.x / sy).round();
        (DVec2::new(q.x - j * sy, dv).length(), j)
    } else {
        (dv.abs(), 0.0)
    };
    let h = hash2(k.seed, row as i64, j as i64);
    // a whole row (or plant) is missing with `gaps`; `amount` thins the rows
    if u01k(h, 1) < v[6] || u01k(hash1(k.seed ^ 0x5A, row as i64), 2) >= i.amount {
        return KOut::default();
    }
    let r = v[2];
    let cov = band_cov(d, r, i.fw);
    if cov <= 0.0 {
        return KOut::default();
    }
    let x = (d / r).min(1.0);
    let tint = 1.0 + v[8] * (u01k(h, 3) - 0.5);
    KOut { cov, albedo: k.col[0] * tint * (0.8 + 0.3 * (1.0 - x)), dh: v[3] * (1.0 - x * x).max(0.0).sqrt(), emit: 0.0, id: h }
}

/// `cells`: a Worley partition; filled cells take a colour between the two, edges a line.
/// v: cell, jitter, fill, edge, edge_height, colour_var, aspect, fill_height.
fn cells(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let asp = v[6];
    let qs = DVec2::new(i.q.x / asp, i.q.y);
    let wc = worley2(k.seed, qs, v[0] / asp.sqrt(), v[1]);
    let dn = wc.point2 - wc.point;
    let nrm = DVec2::new(dn.x / asp, dn.y).length() / dn.length().max(1e-9);
    let edge = worley2_edge_dist(&wc, qs) / nrm.max(1e-6);
    let filled = u01k(wc.id, 1) < v[2] * i.amount;
    let ec = if v[3] > 0.0 { band_cov(edge, 0.5 * v[3], i.fw) * i.amount.min(1.0) } else { 0.0 };
    let mut out = KOut { id: wc.id, ..Default::default() };
    if filled {
        let t = u01k(wc.id, 2);
        out.cov = 1.0;
        out.albedo = (k.col[0] + (k.col[1] - k.col[0]) * t) * (1.0 + v[5] * (u01k(wc.id, 3) - 0.5));
        out.dh = v[7];
    }
    if ec > 0.0 {
        let c0 = out.cov;
        out.albedo = if c0 > 0.0 { out.albedo + (k.col[2] - out.albedo) * ec } else { k.col[2] };
        out.dh = out.dh + (v[4] - out.dh) * ec;
        out.cov = c0.max(ec);
    }
    out
}

/// `stripes`: oriented stripes with per-lattice-point phases (`gully_octave`: no shearing,
/// R-wave). v: wavelength, angle, direction, sharpness, height, threshold.
fn stripes(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let ang = if v[2] >= 1.0 { i.aux[0] } else { v[1] };
    let dir = i.east * ang.cos() + i.north * ang.sin();
    let (s, _) = crate::world::World::gully_octave(k.seed, i.p / v[0], dir);
    let prof = (0.5 + 0.5 * s).clamp(0.0, 1.0).powf(v[3]);
    let cov = if v[5] > 0.0 {
        // crisp stripes: the profile above the threshold, antialiased over the filter width
        let w = (i.fw / v[0]).max(1e-3);
        smoothstep(v[5] - w, v[5] + w, prof) * i.amount.min(1.0)
    } else {
        i.amount.min(1.0)
    };
    KOut { cov, albedo: k.col[1] + (k.col[0] - k.col[1]) * prof, dh: v[4] * prof, emit: 0.0, id: 0 }
}

fn contours_size(k: &KParams, i: &KIn) -> f64 {
    // the horizontal spacing of the levels
    k.v[0] / i.aux[1].abs().max(1e-3)
}

/// `contours`: iso-lines (mode 0) or a staircase (mode 1, terraces) of the height `aux[0]` at
/// the slope `aux[1]`. v: step, line, mode, phase, riser.
fn contours(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let (h, slope) = (i.aux[0], i.aux[1].abs().max(1e-3));
    let x = h / v[0] + v[3];
    let f = x - x.floor();
    if v[2] >= 1.0 {
        // terraces: flat benches, risers over the last `riser` of each step
        let q = (x.floor() + smoothstep(1.0 - v[4], 1.0, f) - x) * v[0];
        let riser = smoothstep(1.0 - v[4], 1.0, f) * (1.0 - smoothstep(1.0 - 0.3 * v[4], 1.0, f));
        let cov = i.amount.min(1.0);
        return KOut { cov, albedo: k.col[1] + (k.col[0] - k.col[1]) * riser, dh: q, emit: 0.0, id: 0 };
    }
    let dist = f.min(1.0 - f) * v[0] / slope;
    let cov = band_cov(dist, v[1], i.fw) * i.amount.min(1.0);
    KOut { cov, albedo: k.col[0], dh: 0.0, emit: 0.0, id: 0 }
}

/// `radial`: instances with a radial profile h = H (1 − r/R)^k, arms and a crater.
/// v: cell, density, radius [2, 3], height [4, 5], exponent, arms, arm_amp, crater,
/// crater_depth, jitter.
fn radial(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let cq = i.q / cell;
    let cf = cq.floor();
    let (ix, iy) = (cf.x as i64, cf.y as i64);
    let mut out = KOut::default();
    let dens = v[1] * i.amount;
    for dy in -1..=1i64 {
        for dx in -1..=1i64 {
            let h = hash2(k.seed, ix + dx, iy + dy);
            if u01k(h, 3) >= dens {
                continue;
            }
            let c = DVec2::new((ix + dx) as f64 + 0.5 + v[11] * (u01k(h, 1) - 0.5), (iy + dy) as f64 + 0.5 + v[11] * (u01k(h, 2) - 0.5)) * cell;
            let rel = i.q - c;
            let r0 = v[2] + (v[3] - v[2]) * u01k(h, 4);
            let d = rel.length();
            if d > r0 * (1.0 + v[8]) + i.fw {
                continue;
            }
            let ang = rel.y.atan2(rel.x);
            let re = r0 * (1.0 + v[8] * if v[7] >= 1.0 { (v[7].round() * ang + u01k(h, 6) * 6.3).cos() } else { 0.0 });
            let x = (d / re.max(1e-6)).min(1.0);
            let cov = band_cov(d, re, i.fw);
            if cov <= 0.0 {
                continue;
            }
            let hh = v[4] + (v[5] - v[4]) * u01k(h, 5);
            let mut z = hh * (1.0 - x).powf(v[6]);
            if v[9] > 0.0 && x < v[9] {
                let xc = x / v[9];
                z -= v[10] * hh * (1.0 - xc * xc);
            }
            if z.abs() > out.dh.abs() || out.cov <= 0.0 {
                out.dh = z;
                out.albedo = k.col[1] + (k.col[0] - k.col[1]) * (1.0 - x);
                out.id = h;
            }
            out.cov = out.cov.max(cov);
        }
    }
    out
}

/// `crescent`: crescent instances (barchans) with horns downwind (`aux[0]` or the angle).
/// v: cell, density, radius [2, 3], height_ratio, horns, direction, angle, jitter.
fn crescent(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let cq = i.q / cell;
    let cf = cq.floor();
    let (ix, iy) = (cf.x as i64, cf.y as i64);
    let ang = if v[6] >= 1.0 { i.aux[0] } else { v[7] };
    let mut out = KOut::default();
    let dens = v[1] * i.amount;
    for dy in -1..=1i64 {
        for dx in -1..=1i64 {
            let h = hash2(k.seed, ix + dx, iy + dy);
            if u01k(h, 3) >= dens {
                continue;
            }
            let c = DVec2::new((ix + dx) as f64 + 0.5 + v[8] * (u01k(h, 1) - 0.5), (iy + dy) as f64 + 0.5 + v[8] * (u01k(h, 2) - 0.5)) * cell;
            let r = v[2] + (v[3] - v[2]) * u01k(h, 4);
            let rel = rot(i.q - c, ang);
            if rel.length() > r + i.fw {
                continue;
            }
            let d = rel.length();
            let di = (rel - DVec2::new(v[5] * r, 0.0)).length();
            let cov = band_cov(d, r, i.fw) * (1.0 - band_cov(di, (1.0 - 0.3 * v[5]) * r, i.fw));
            if cov <= 0.0 {
                continue;
            }
            // gentle windward (−x) slope, steep slip face at the inner edge
            let x = (d / r).min(1.0);
            let z = v[4] * r * (1.0 - x * x).max(0.0).sqrt() * smoothstep(-1.0, 0.2, -rel.x / r + 0.3);
            if z > out.dh {
                out.dh = z;
                out.albedo = k.col[0] * (0.9 + 0.2 * (1.0 - x));
                out.id = h;
            }
            out.cov = out.cov.max(cov);
        }
    }
    out
}

/// `lobes`: fan sectors from an apex with a noisy margin and radial channels.
/// v: cell, density, radius [2, 3], span, margin, channels, height, direction, jitter.
fn lobes(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let cq = i.q / cell;
    let cf = cq.floor();
    let (ix, iy) = (cf.x as i64, cf.y as i64);
    let mut out = KOut::default();
    let dens = v[1] * i.amount;
    for dy in -1..=1i64 {
        for dx in -1..=1i64 {
            let h = hash2(k.seed, ix + dx, iy + dy);
            if u01k(h, 3) >= dens {
                continue;
            }
            let c = DVec2::new((ix + dx) as f64 + 0.5 + v[9] * (u01k(h, 1) - 0.5), (iy + dy) as f64 + 0.5 + v[9] * (u01k(h, 2) - 0.5)) * cell;
            let r = v[2] + (v[3] - v[2]) * u01k(h, 4);
            let rel = i.q - c;
            let d = rel.length();
            if d > r * (1.0 + v[5]) + i.fw {
                continue;
            }
            let dir = if v[8] >= 1.0 { i.aux[0] } else { u01k(h, 6) * std::f64::consts::TAU };
            let mut da = rel.y.atan2(rel.x) - dir;
            da = (da + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
            let half = 0.5 * v[4];
            if da.abs() > half {
                continue;
            }
            // margin: lobate (a few lobes across the span) and noisy (noise in the local frame:
            // a per-instance wavelength at ECEF coordinates would differ between backends)
            let ql = (i.q - c) / (0.3 * r);
            let lobe = (da / half * 2.5 + u01k(h, 7) * 6.3).sin() * 0.5 + 0.5 * perlin3(h, DVec3::new(ql.x, ql.y, 0.5));
            let re = r * (1.0 - v[5] * 0.5 + v[5] * 0.5 * lobe) * (1.0 - (da / half).powi(4));
            let cov = band_cov(d - 0.5 * re, 0.5 * re, i.fw);
            if cov <= 0.0 {
                continue;
            }
            let x = (d / re.max(1e-6)).min(1.0);
            let ch = if v[6] > 0.0 { smoothstep(0.75, 0.95, (da / half * v[6] * std::f64::consts::PI).cos()) * (1.0 - x) } else { 0.0 };
            out.albedo = k.col[0] + (k.col[1] - k.col[0]) * ch;
            out.dh = v[7] * (1.0 - x);
            out.cov = out.cov.max(cov);
            out.id = h;
        }
    }
    out
}

/// A noise value of `patches` mapped to ~uniform 0..1 (so `fraction` is the coverage).
fn patch_u(seed: u64, p: DVec3, scale: f64, oct: f64, rough: f64, gsd: f64) -> f64 {
    let mut n = 0.0;
    let mut a = 1.0;
    let mut norm = 0.0;
    let mut lam = scale;
    for o in 0..3 {
        if o as f64 >= oct {
            break;
        }
        n += a * perlin3(seed ^ (0x9A7 + o), p / lam) * band(lam, gsd);
        norm += a * a;
        a *= rough;
        lam *= 0.43;
    }
    // perlin3 has a standard deviation ≈ 0.27: a logistic of n / σ is close to uniform
    let s = n / (0.27 * norm.sqrt());
    1.0 / (1.0 + (-1.7 * s).exp())
}

/// `patches`: crisp multi-scale noise threshold (forest / clearing, pools, bare patches).
/// v: scale, fraction, octaves, roughness, height, colour_var.
fn patches(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let u = patch_u(k.seed, i.p, v[0], v[2], v[3], i.gsd);
    let frac = v[1] * i.amount;
    // antialiased threshold: the noise changes by ~1/scale per metre
    let w = (2.0 * i.fw / v[0]).clamp(1e-3, 0.5);
    let cov = smoothstep(-w, w, frac - u);
    if cov <= 0.0 {
        return KOut::default();
    }
    let t = 0.5 + 0.5 * perlin3(k.seed ^ 0x77, i.p / (0.37 * v[0]));
    KOut { cov, albedo: (k.col[0] + (k.col[1] - k.col[0]) * t) * (1.0 + v[5] * (u - 0.5)), dh: v[4], emit: 0.0, id: 0 }
}

fn linear_size(_k: &KParams, i: &KIn) -> f64 {
    2.0 * i.aux[2]
}

/// `linear`: a cross-section across a segment (`aux`: signed distance, along, half width).
/// v: profile, dash, duty, shoulder, height, marking.
fn linear(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let (d, s, hw) = (i.aux[0], i.aux[1], i.aux[2]);
    let cov = band_cov(d, hw + v[3], i.fw) * i.amount.min(1.0);
    if cov <= 0.0 {
        return KOut::default();
    }
    let core = band_cov(d, hw, i.fw);
    let mut col = k.col[1] + (k.col[0] - k.col[1]) * core;
    let x = (d.abs() / hw.max(1e-3)).min(1.0);
    let dh = match v[0] as u32 {
        1 => v[4] * (1.0 - 0.3 * x * x),
        2 => v[4] * (1.0 - x * x).max(0.0).sqrt(),
        3 => -v[4] * (1.0 - x * x).max(0.0),
        _ => v[4],
    };
    if v[5] > 0.0 {
        let on = if v[1] > 0.0 { ((s / v[1]).rem_euclid(1.0) < v[2]) as u8 as f64 } else { 1.0 };
        let m = band_cov(d, 0.5 * v[5], i.fw) * on * band(v[1].max(4.0 * v[5]), i.gsd);
        col = col + (k.col[2] - col) * m;
    }
    KOut { cov, albedo: col, dh, emit: 0.0, id: 0 }
}

/// `stamp`: a template in an oriented box (`aux`: u, v (m from the centre), half sizes).
/// v: template (pad, strip, blocks, panels), a, b, height, lights.
fn stamp(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let (u, w, hu, hv) = (i.aux[0], i.aux[1], i.aux[2], i.aux[3]);
    let inside = ((hu - u.abs()).min(hv - w.abs()) / i.fw.max(1e-9) + 0.5).clamp(0.0, 1.0) * i.amount.min(1.0);
    if inside <= 0.0 {
        return KOut::default();
    }
    let mut col = k.col[0];
    let mut dh = 0.0;
    let mut emit = 0.0;
    match v[0] as u32 {
        1 => {
            // a strip (runway): dashed centre line every `a` m, edge lights every `b`·10 m
            let cl = band_cov(w, 0.5, i.fw) * ((u / v[1].max(1.0)).rem_euclid(1.0) < 0.5) as u8 as f64;
            col = col + (k.col[1] - col) * 0.85 + (k.col[2] - k.col[1]) * cl;
            if v[4] > 0.0 {
                let sp = 10.0 * v[2].max(1.0);
                let du = u - (u / sp).round() * sp;
                let dv = hv - w.abs();
                emit = crate::surface::point_light(du * du + dv * dv, v[4], 0.4, i.fw);
            }
        }
        2 => {
            // blocks: buildings of `a` m on a grid of `a + b`
            let pitch = v[1] + v[2];
            let gu = u - (u / pitch).round() * pitch;
            let gv = w - (w / pitch).round() * pitch;
            let b = ((0.5 * v[1] - gu.abs()).min(0.5 * v[1] - gv.abs()) / i.fw.max(1e-9) + 0.5).clamp(0.0, 1.0);
            col = col + (k.col[1] - col) * b;
            dh = v[3] * b;
        }
        3 => {
            // panel rows of width `a` with gaps `b`
            let pitch = v[1] + v[2];
            let g = w - (w / pitch).round() * pitch;
            let b = band_cov(g, 0.5 * v[1], i.fw);
            col = col + (k.col[1] - col) * b;
            dh = v[3] * b;
        }
        _ => {}
    }
    KOut { cov: inside, albedo: col, dh, emit, id: 0 }
}

/// `canopy`: a closed canopy: clumped crown texture with per-clump hue and emergent crowns.
/// v: cell, emergent, height, emergent_height, clump, colour_var, flowering.
fn canopy(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    if i.amount <= 0.0 {
        return KOut::default();
    }
    // crown clumps: a Worley partition of the canopy
    let wc = worley2(k.seed ^ 0xC1, i.q, v[4], 0.9);
    let x = (wc.f1 / (0.62 * v[4])).min(1.0);
    let hue = u01k(wc.id, 1);
    let mut col = k.col[0] * (1.0 + v[5] * (hue - 0.5)) * (0.72 + 0.4 * (1.0 - x * x));
    col = col + (k.col[1] - col) * (0.5 * v[5] * smoothstep(0.7, 1.0, u01k(wc.id, 2)));
    if u01k(wc.id, 3) < v[6] {
        col = k.col[2] * (0.85 + 0.3 * (1.0 - x));
    }
    let mut dh = v[2] * (0.88 + 0.12 * (1.0 - x * x).max(0.0).sqrt());
    // emergents on their own lattice
    let em = scatter(
        &KParams {
            kind: kind::SCATTER,
            seed: k.seed ^ 0xE3,
            v: [v[0], v[1], shape::DOME as f64, 0.25 * v[0], 0.38 * v[0], v[3] * 0.85, v[3], 0.8, v[5], 0.4, 1.0, 0.0, 0.3, 0.0, 0.0, 0.0],
            col: [k.col[0] * 1.08, k.col[1], k.col[2]],
        },
        &KIn { amount: 1.0, ..*i },
    );
    if em.cov > 0.0 && em.dh > dh {
        col = col + (em.albedo - col) * em.cov;
        dh = dh + (em.dh - dh) * em.cov;
    }
    KOut { cov: i.amount.min(1.0), albedo: col, dh, emit: 0.0, id: wc.id }
}

/// `water`: water colour by depth (`aux[0]`), sediment and a foam line.
/// v: depth_scale, sediment, sediment_scale, foam_depth, foam_width.
fn water(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let depth = i.aux[0].max(0.0);
    let t = 1.0 - (-depth / v[0]).exp();
    let mut col = k.col[0] + (k.col[1] - k.col[0]) * t;
    col *= 1.0 + v[1] * perlin3(k.seed, i.p / v[2]) * band(v[2], i.gsd);
    if v[3] > 0.0 {
        let a = (depth - v[3]) / v[4];
        let foam = (-a * a).exp() * band(3.0 * v[4].max(1.0), i.gsd);
        col = col + (k.col[2] - col) * foam.min(1.0);
    }
    KOut { cov: i.amount.min(1.0), albedo: col, dh: 0.0, emit: 0.0, id: 0 }
}

/// `city`: the settlements kit's (the core draws towns in `layers::built`).
fn city(_k: &KParams, _i: &KIn) -> KOut {
    KOut::default()
}

// ============================================================================= relief operators

/// Height-domain operators of pass A (`relief.wgsl` twins). Each maps a height (m) to a height.
pub mod relief {
    use crate::noise::smoothstep;

    /// Terraces: steps of `step` m, risers over the last `sharp` of each step.
    pub fn terrace(h: f64, step: f64, sharp: f64, phase: f64) -> f64 {
        let x = h / step + phase;
        let k = x.floor();
        (k + smoothstep(1.0 - sharp, 1.0, x - k) - phase) * step
    }

    /// A cliff at sea level: land above 0 raised by `c` m over `eps` m of height.
    pub fn cliff(h: f64, c: f64, eps: f64) -> f64 {
        if h <= 0.0 {
            h
        } else {
            h + c * smoothstep(0.0, eps.max(1e-6), h)
        }
    }

    /// A U-shaped valley cross-section at distance `d` from the axis: floor of half width `w`,
    /// walls rising as ((|d| − w)/k)² up to `depth`.
    pub fn u_profile(d: f64, w: f64, depth: f64, k: f64) -> f64 {
        let x = ((d.abs() - w) / k.max(1e-6)).max(0.0);
        -depth + depth * (x * x).min(1.0)
    }

    /// Flatten towards `target` with weight `w` (0..1).
    pub fn flatten(h: f64, target: f64, w: f64) -> f64 {
        h + (target - h) * w.clamp(0.0, 1.0)
    }

    /// A bowl of radius `r` and depth `depth` at distance `d`.
    pub fn bowl(d: f64, r: f64, depth: f64) -> f64 {
        let x = d / r.max(1e-6);
        if x >= 1.0 {
            0.0
        } else {
            -depth * (1.0 - x * x)
        }
    }

    /// A bump of height `b` centred on the contour `hb` (width `w` in height).
    pub fn contour_bump(h: f64, hb: f64, b: f64, w: f64) -> f64 {
        let x = (h - hb) / w.max(1e-6);
        h + b * (-x * x).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(name: &str, y: &str) -> KParams {
        let m: std::collections::BTreeMap<String, serde_yaml::Value> = serde_yaml::from_str(y).unwrap();
        compile_params(spec(name).unwrap(), &m, 0x1234, &crate::registry::BiomePal::default_pal()).unwrap()
    }

    fn kin(q: DVec2, gsd: f64) -> KIn {
        KIn { q, p: DVec3::new(6_371_000.0, q.x, q.y), east: DVec3::Y, north: DVec3::Z, gsd, fw: gsd, amount: 1.0, aux: [0.3 * q.x, 0.3, 4.0, 0.0] }
    }

    /// Every kernel's mean is what its explicit evaluation averages to over an area (a
    /// different point set than the calibration's), and a pixel far coarser than its
    /// features shows that mean.
    #[test]
    fn means_match_explicit_averages() {
        for (name, y) in [
            ("scatter", "{cell: 8, density: 0.6, radius: [2, 3], height: [3, 6]}"),
            ("rows", "{spacing: 3, along: 2, radius: 0.7, height: 2}"),
            ("cells", "{cell: 20, fill: 0.5, edge: 1.5}"),
            ("stripes", "{wavelength: 40, threshold: 0.4}"),
            ("radial", "{cell: 200, density: 0.5, radius: [40, 80], height: [10, 20], crater: 0.3}"),
            ("crescent", "{cell: 200, density: 0.5, radius: [40, 80]}"),
            ("lobes", "{cell: 400, density: 0.5, radius: [100, 180]}"),
            ("patches", "{scale: 80, fraction: 0.3}"),
            ("canopy", "{cell: 30}"),
        ] {
            let k = params(name, y);
            let m = calibrate(&k);
            let sz = size(&k);
            let n = 6000;
            let (mut c, mut a) = (0.0, DVec3::ZERO);
            for j in 0..n {
                let q = DVec2::new(halton(j + 7, 5) * 90.0 * sz + 1e4, halton(j + 7, 7) * 90.0 * sz - 3e3);
                let o = explicit(&k, &kin(q, sz / 50.0));
                c += o.cov;
                a += o.albedo * o.cov;
            }
            let c = c / n as f64;
            assert!((c - m.cov).abs() < 0.03 + 0.08 * m.cov, "{name}: mean coverage {} vs average {c}", m.cov);
            if c > 0.05 {
                assert!((a / (c * n as f64) - m.albedo).abs().max_element() < 0.02, "{name}: mean albedo");
            }
            // a coarse pixel: the mean, scaled by the amount
            let o = eval(&k, &m, &KIn { amount: 0.5, ..kin(DVec2::new(10.0, 20.0), 10.0 * sz) });
            assert!((o.cov - 0.5 * m.cov).abs() < 1e-9, "{name}: coarse coverage");
        }
    }

    #[test]
    fn kernels_are_deterministic_and_crossfade_continuously() {
        let k = params("scatter", "{cell: 6, density: 0.8, radius: [1.5, 2.5], height: [4, 8]}");
        let m = calibrate(&k);
        let q = DVec2::new(123.4, -56.7);
        let a = eval(&k, &m, &kin(q, 1.0));
        let b = eval(&k, &m, &kin(q, 1.0));
        assert_eq!(a.cov.to_bits(), b.cov.to_bits());
        // no jump across the crossfade
        let mut prev: Option<f64> = None;
        for s in 0..200 {
            let gsd = 0.5 + s as f64 * 0.03;
            let mut acc = 0.0;
            for j in 0..400 {
                let q = DVec2::new(halton(j + 1, 2) * 300.0, halton(j + 1, 3) * 300.0);
                acc += eval(&k, &m, &kin(q, gsd)).cov;
            }
            if let Some(p) = prev {
                assert!((acc / 400.0 - p).abs() < 0.05, "coverage jumps at gsd {gsd}");
            }
            prev = Some(acc / 400.0);
        }
    }

    #[test]
    fn relief_operators() {
        assert_eq!(relief::cliff(-3.0, 50.0, 2.0), -3.0);
        assert!((relief::cliff(5.0, 50.0, 2.0) - 55.0).abs() < 1e-9);
        assert!((relief::terrace(10.2, 3.0, 0.2, 0.0) - 9.0).abs() < 1e-9);
        assert_eq!(relief::bowl(10.0, 5.0, 3.0), 0.0);
        assert!((relief::bowl(0.0, 5.0, 3.0) + 3.0).abs() < 1e-12);
        assert!((relief::contour_bump(-6.0, -6.0, 4.0, 2.0) + 2.0).abs() < 1e-12);
        assert!((relief::u_profile(0.0, 100.0, 50.0, 200.0) + 50.0).abs() < 1e-12);
    }
}
