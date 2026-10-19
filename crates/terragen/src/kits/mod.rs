//! Kits: self-contained additions to the world (`docs/next/kit-guide.md`). A kit lives in its
//! own files:
//!
//! * `src/kits/<kit>.rs`: its [`Kit`] (`pub static KIT`), layer functions, kernels, relief;
//! * `src/gpu/wgsl/kits/<kit>.wgsl`: the WGSL twins (`<kit>_slot_<slot>`, kernels, relief);
//! * `biomes/<kit>.yaml`: its biomes (data);
//!
//! and is registered by one line in [`KITS`] (and its `mod` line below). Everything else (the
//! stack's slot calls, the kernel dispatch, the WGSL glue) is generated from [`KITS`].

use crate::kernels::KernelSpec;
use crate::stack::Stack;
use crate::world::{Ctx, Macro};
use glam::DVec3;

// ---- kit modules: replace your line (keep the blank lines between them: merges stay clean)

/// a test-only kit exercising every hook (the guide's worked example)
#[cfg(test)]
mod example;

mod desert;

mod volcanic;

mod mountains;

mod cold;

mod coast;

mod tropical;

mod temperate;

mod agriculture;

pub mod settlements;

pub mod transport;

/// The kits, in evaluation order within a slot. Kernel kind ids: each kit owns 16 from
/// `kernels::KIT_BASE` (64) on, by its line here: desert 64, volcanic 80, mountains 96,
/// cold 112, coast 128, tropical 144, temperate 160, agriculture 176, settlements 192,
/// transport 208.
// (the blank lines between the entries keep parallel kit merges conflict-free: not formatted)
#[rustfmt::skip]
pub static KITS: &[&Kit] = &[
    &desert::KIT,

    &volcanic::KIT,

    &mountains::KIT,

    &cold::KIT,

    &coast::KIT,

    &tropical::KIT,

    &temperate::KIT,

    &agriculture::KIT,

    &settlements::KIT,

    &transport::KIT,
    #[cfg(test)]
    &example::KIT,
];

/// Inputs of a pass-A relief operator (per pixel centre and per drainage lattice point).
#[derive(Clone, Copy, Debug)]
pub struct ReliefIn<'a> {
    pub ctx: &'a Ctx,
    pub m: &'a Macro,
    /// temperature (°C) at the smooth elevation, moisture 0..1
    pub temp: f64,
    pub moist: f64,
    /// mountain belt mask, sand-sea mask, mesa mask (0..1)
    pub mountain: f64,
    pub sand: f64,
    pub mesa: f64,
    /// smooth (≥ 5 km) elevation (m)
    pub smooth: f64,
    /// the instance lists of the area (per family; relief families only), if known
    pub inst: Option<&'a [Vec<crate::instances::Instance>]>,
}

impl ReliefIn<'_> {
    /// The instances of family `f` near the point (`FAM_*` index; WGSL `inst_list(f, r.blk,
    /// c.p)`).
    pub fn instances(&self, w: &crate::world::World, f: usize) -> std::borrow::Cow<'_, [crate::instances::Instance]> {
        match self.inst {
            Some(l) if f < l.len() => std::borrow::Cow::Borrowed(&l[f][..]),
            _ => std::borrow::Cow::Owned(crate::instances::near(w, f, self.ctx.p, 0.0)),
        }
    }
}

/// Inputs of a kit's carve hook (one drainage channel at a pixel, pass A: `World::terrain_impl`,
/// WGSL `terrain_rest`): where the core carves the valley of a channel, the hook may change the
/// profile's parameters ([`Carve`]).
#[derive(Clone, Copy, Debug)]
pub struct CarveIn<'a> {
    pub ctx: &'a Ctx,
    pub m: &'a Macro,
    /// drainage level (0 major rivers, 1 tributaries, 2 streams)
    pub level: u8,
    /// distance to the channel's centre line (m)
    pub ad: f64,
    /// position along the piece (0 at its start); on a spring piece 0 is the head of the channel
    pub u: f64,
    /// the piece is the first one of a channel (a source: a cirque and a tarn can sit at its head)
    pub spring: bool,
    /// channel half width (m)
    pub hw: f64,
    /// the ground height before the carve (m)
    pub h0: f64,
    pub temp: f64,
    pub moist: f64,
    pub mountain: f64,
    pub mesa: f64,
    pub smooth: f64,
}

/// The parameters of a channel's valley profile; the core sets its own values, a kit's hook
/// may change them. The floor and the floodplain are flat over `fp_w`, the walls rise to the
/// uncarved ground over `wall_k` m per m of depth.
#[derive(Clone, Copy, Debug)]
pub struct Carve {
    /// the valley floor (m; the drainage floor less the incision; below sea level: a fjord)
    pub floor: f64,
    /// floodplain half width (m)
    pub fp_w: f64,
    /// horizontal extent of the walls per metre of depth
    pub wall_k: f64,
    /// the deepest cut (m; the level's `max_depth_m`)
    pub depth_cap: f64,
    /// the floodplain surface is at least this high (m; −1e9 lets a fjord flood)
    pub fp_min: f64,
    /// 0..1: the walls' profile from the smooth S to a U (parabola, steep shoulders)
    pub u: f64,
    /// mean stratum thickness (m) of stepped walls (`noise::strata`), and their share 0..1 (0: smooth walls)
    pub steps: f64,
    pub step_mix: f64,
    /// a warp (m) of the distance to the channel in the walls: > 0 a spur, < 0 a gully
    pub dad: f64,
    /// a lake in the channel's head: radius (m, 0: none) and absolute water level (m)
    pub lake_r: f64,
    pub lake_level: f64,
    /// 0..1: the valley's extent from the drainage level's wide valley to what the walls alone need
    /// (`fp_w + wall_k` per metre of depth): a narrow trough (fjords)
    pub narrow: f64,
    /// a lateral shift (m, along the signed distance) of the channel and its whole valley: the
    /// meanders of an entrenched river (canyons)
    pub shift: f64,
    /// a factor of the channel's half width (an uneven bed)
    pub hw_k: f64,
    /// the depth (m) of the fine erosion texture of the walls (rills and rough rock; 0: none)
    pub rough: f64,
}

/// A host-built pass-A stamp (`Kit::astamps`): a landform that needs the drainage or exact
/// samples to be placed (a delta at the point where a river meets the drawn coast) and shapes
/// the ground after the drainage carve. On the GPU it travels in the sink-lake list as a
/// `Sink` with a negative radius: `c` = (centre, `bearing`), `id`, `rad` = −`r`, `level` = `v`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AStamp {
    /// the owning kit's kind of stamp (in `id`'s low byte: `id & 0xFF`), and its seed
    pub id: u64,
    /// centre: a surface point (ECEF)
    pub c: DVec3,
    /// its direction: radians from east, counter-clockwise, in the geocentric frame at `c`
    /// ([`astamp_frame`])
    pub bearing: f64,
    /// reach (m): it changes nothing farther than this from `c`
    pub r: f64,
    /// a parameter (f32 on the GPU)
    pub v: f32,
}

impl AStamp {
    pub fn kind(&self) -> u8 {
        (self.id & 0xFF) as u8
    }
}

/// The geocentric frame (east, north, up; f32 axes as in WGSL `astamp_frame`) at a stamp's
/// centre `c`, and its direction unit vector for `bearing`.
pub fn astamp_frame(c: DVec3, bearing: f64) -> [glam::Vec3; 4] {
    let up = c.as_vec3().normalize();
    let east = glam::Vec3::Z.cross(up).normalize();
    let north = up.cross(east);
    let (s, co) = (bearing as f32).sin_cos();
    [east, north, up, east * co + north * s]
}

/// What a pass-A stamp sees at a pixel (after the drainage carve).
#[derive(Clone, Copy, Debug)]
pub struct AStampIn<'a> {
    pub ctx: &'a Ctx,
    /// the nearest channel: signed distance to its centre line and its half width (m; 1e6 / 0:
    /// none)
    pub river_d: f64,
    pub river_hw: f64,
}

/// The kits' pass-A stamps that can reach the disc (`center`, `radius`) at pixel size `gsd`
/// (the kits' order, then theirs).
pub fn astamps(w: &crate::world::World, center: DVec3, radius: f64, gsd: f64) -> Vec<AStamp> {
    let mut out = Vec::new();
    for k in KITS {
        if let Some(f) = k.astamps {
            out.extend(f(w, center, radius, gsd));
        }
    }
    out
}

/// Do any kits build pass-A stamps?
pub fn has_astamps() -> bool {
    KITS.iter().any(|k| k.astamps.is_some())
}

/// Apply a pass-A stamp to the height `h`.
#[inline]
pub fn astamp(w: &crate::world::World, st: &AStamp, ai: &AStampIn, h: &mut f64) {
    for k in KITS {
        if let Some(f) = k.astamp {
            f(w, st, ai, h);
        }
    }
}

/// A kit's host function listing its pass-A stamps (`Kit::astamps`).
pub type AStampsFn = fn(&crate::world::World, DVec3, f64, f64) -> Vec<AStamp>;

/// A kit's layer function for one stack slot.
pub type LayerFn = fn(&mut Stack);

/// A kit.
pub struct Kit {
    /// lower_snake_case; prefixes its WGSL functions
    pub name: &'static str,
    /// its biomes (`include_str!("../../biomes/<kit>.yaml")`, or "")
    pub biomes_yaml: &'static str,
    /// its WGSL (`include_str!("../gpu/wgsl/kits/<kit>.wgsl")`, or "")
    pub wgsl: &'static str,
    /// layer functions by stack slot (`stack::slot`); the WGSL file defines
    /// `fn <name>_slot_<slot name>(s: ptr<function, Stack>)` for each
    pub layers: &'static [(usize, LayerFn)],
    /// its kernels (kinds in its range; WGSL: `fn <wgsl>(li: u32, k: KIn) -> KOut`)
    pub kernels: &'static [KernelSpec],
    /// a pass-A relief operator: changes the height `h` (m) before the drainage carves it.
    /// WGSL: `fn <name>_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>)`.
    /// Only analytic inputs (no host tables): it also shapes the drainage lattice heights.
    pub relief: Option<fn(&crate::world::World, &ReliefIn, &mut f64)>,
    /// the WGSL of the relief operator (included in every module with pass A: tiles, points,
    /// drainage; no tile-only bindings)
    pub relief_wgsl: &'static str,
    /// a hook in the drainage carve of pass A: glacial troughs, fjords, canyons change the valley
    /// profile of the channels (`CarveIn`, `Carve`). WGSL (in `relief_wgsl`): `fn <name>_carve(c:
    /// Ctx, m: Macro, ci: CarveIn, cv: ptr<function, Carve>)`.
    pub carve: Option<fn(&crate::world::World, &CarveIn, &mut Carve)>,
    /// host-built pass-A stamps (deltas at river mouths): the stamps that can reach the disc
    /// (centre, radius) at pixel size `gsd`; a pure function of the world (cached by the kit).
    /// They shape the ground after the drainage carve (`astamp`), in full pass-A evaluations.
    pub astamps: Option<AStampsFn>,
    /// a pass-A stamp's relief (`AStampIn`; the kit tells its own stamps by `AStamp::kind`).
    /// WGSL (in `relief_wgsl`): `fn <name>_astamp(c: Ctx, st: Sink, ai: AStampIn, h: ptr<function,
    /// f32>)` (the stamp arrives as a `Sink` with a negative radius, see `AStamp`).
    pub astamp: Option<fn(&crate::world::World, &AStamp, &AStampIn, &mut f64)>,
    /// its instance families (`crate::instances`)
    pub families: &'static [crate::instances::Family],
    /// its host-built linear features and stamps (`crate::features`)
    pub host: Option<crate::features::HostFn>,
    /// warm the memo tables `host` reads for a batch of areas (e.g. route every edge near them
    /// once, in parallel), before the areas' `host` calls run in parallel: concurrent calls
    /// would compute a shared item each (`crate::features::build_areas`). Must not change what
    /// `host` returns.
    pub prepare: Option<fn(&crate::world::World, &[crate::features::Area])>,
    /// the kit draws the settlements: the core's grid towns are off ([`legacy_towns`])
    pub replaces_towns: bool,
    /// the kit draws the road network: the core's noise-network roads are off ([`legacy_roads`];
    /// the farm tracks along land-use region borders stay)
    pub replaces_roads: bool,
    /// the kit's families share one WGSL existence test, `fn <kit>_exists(f: u32, site:
    /// InstSite) -> InstOut` (one call site in the dispatch: one inlined copy of what the tests
    /// share, e.g. the macro fields at the site), instead of `<family>_exists(site)` each
    pub shared_exists: bool,
}

impl Kit {
    pub const EMPTY: Kit = Kit {
        name: "",
        biomes_yaml: "",
        wgsl: "",
        layers: &[],
        kernels: &[],
        relief: None,
        relief_wgsl: "",
        carve: None,
        astamps: None,
        astamp: None,
        families: &[],
        host: None,
        prepare: None,
        replaces_towns: false,
        replaces_roads: false,
        shared_exists: false,
    };
}

/// Does the core draw its grid towns (`layers::built`, the town lattice of pass A)? Off when a
/// kit replaces them (the settlements kit).
pub fn legacy_towns() -> bool {
    !KITS.iter().any(|k| k.replaces_towns)
}

/// Does the core draw its noise-network roads (pass A's iso-line distances, `layers::linear`)?
/// Off when a kit replaces them (the transport kit).
pub fn legacy_roads() -> bool {
    !KITS.iter().any(|k| k.replaces_roads)
}

/// Run the kits' layers of `slot`.
#[inline]
pub fn slot(slot: usize, s: &mut Stack) {
    for k in KITS {
        for (sl, f) in k.layers {
            if *sl == slot && !s.done {
                f(s);
            }
        }
    }
}

/// The kits' kernels.
pub fn kernels() -> impl Iterator<Item = &'static KernelSpec> {
    KITS.iter().flat_map(|k| k.kernels.iter())
}

/// Run the kits' relief operators.
#[inline]
pub fn relief(w: &crate::world::World, r: &ReliefIn, h: &mut f64) {
    for k in KITS {
        if let Some(f) = k.relief {
            f(w, r, h);
        }
    }
}

/// Does a kit have a carve hook?
pub fn has_carve() -> bool {
    KITS.iter().any(|k| k.carve.is_some())
}

/// Run the kits' carve hooks.
#[inline]
pub fn carve(w: &crate::world::World, ci: &CarveIn, cv: &mut Carve) {
    for k in KITS {
        if let Some(f) = k.carve {
            f(w, ci, cv);
        }
    }
}

/// The kits' WGSL and the glue calling it (`kits_slot_<slot>`, `kits_kernel`, `kits_relief`; pass
/// B's stages dispatch to `kits_slot_<slot>`: `gpu::stages_wgsl`).
pub fn wgsl() -> String {
    let mut s = String::from("// ---------------------------------------------------------------- kits (generated)\n");
    for k in KITS {
        s.push_str(k.wgsl);
        s.push('\n');
    }
    for (si, name) in crate::stack::slot::NAMES.iter().enumerate() {
        s.push_str(&format!("fn kits_slot_{name}(s: ptr<function, Stack>) {{\n"));
        for k in KITS {
            if k.layers.iter().any(|l| l.0 == si) {
                s.push_str(&format!("    if (!(*s).done) {{ {}_slot_{name}(s); }}\n", k.name));
            }
        }
        s.push_str("}\n");
    }
    s.push_str("fn kits_kernel(kind: u32, li: u32, k: KIn) -> KOut {\n    switch kind {\n");
    for ks in kernels() {
        s.push_str(&format!("        case {}u: {{ return {}(li, k); }}\n", ks.kind, ks.wgsl));
    }
    s.push_str("        default: { return kout_none(); }\n    }\n}\n");
    s
}

/// The kits' relief WGSL glue (in every module that has the relief: tiles, points, drainage).
pub fn wgsl_relief() -> String {
    let mut s = String::new();
    for k in KITS {
        s.push_str(k.relief_wgsl);
        s.push('\n');
    }
    s.push_str(&format!("const KITS_LEGACY_TOWNS: bool = {};\n", legacy_towns()));
    s.push_str(&format!("const KITS_LEGACY_ROADS: bool = {};\n", legacy_roads()));
    s.push_str("fn kits_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {\n");
    // (the module's instance lists of the point: `relief_lists`, gpu::sources)
    s.push_str("    relief_lists(c.p, r.blk);\n");
    for k in KITS {
        if k.relief.is_some() {
            s.push_str(&format!("    {}_relief(c, m, r, h);\n", k.name));
        }
    }
    s.push_str("}\n");
    s.push_str("fn kits_astamp(c: Ctx, st: Sink, ai: AStampIn, h: ptr<function, f32>) {\n");
    for k in KITS {
        if k.astamp.is_some() {
            s.push_str(&format!("    {}_astamp(c, st, ai, h);\n", k.name));
        }
    }
    s.push_str("}\n");
    s.push_str("fn kits_carve(c: Ctx, m: Macro, ci: CarveIn, cv: ptr<function, Carve>) {\n");
    for k in KITS {
        if k.carve.is_some() {
            s.push_str(&format!("    {}_carve(c, m, ci, cv);\n", k.name));
        }
    }
    s.push_str("}\n");
    s
}

/// The agriculture kit's field system of land-use region `rid` in ecoregion `e` (its name).
pub fn agriculture_system(e: &crate::eco::EcoParams, rid: u64) -> &'static str {
    agriculture::sys::NAMES[agriculture::pick_system(&agriculture::EcoV::of(e), rid) as usize]
}

/// The agriculture kit's cadastral partition (the core's irregular fields): parcel id,
/// coordinates in the parcel's frame, distance to its border, its centre (`q`'s frame).
pub(crate) fn cadastral_parcel(seed: u64, q: glam::DVec2, cell: f64, aspect: f64) -> (u64, f64, f64, f64, glam::DVec2) {
    agriculture::cadastral_parcel(seed, q, cell, aspect)
}
