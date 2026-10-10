//! Kits: self-contained additions to the world (`docs/design/kit-guide.md`). A kit lives in its
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

// ---- kit modules: replace your line (keep the blank lines between them: merges stay clean)

/// a test-only kit exercising every hook (the guide's worked example)
#[cfg(test)]
mod example;

// mod desert;

// mod volcanic;

// mod mountains;

// mod cold;

// mod coast;

// mod tropical;

// mod temperate;

// mod agriculture;

// mod settlements;

// mod transport;

/// The kits, in evaluation order within a slot. Kernel kind ids: each kit owns 16 from
/// `kernels::KIT_BASE` (64) on, by its line here: desert 64, volcanic 80, mountains 96,
/// cold 112, coast 128, tropical 144, temperate 160, agriculture 176, settlements 192,
/// transport 208.
pub static KITS: &[&Kit] = &[
    // &desert::KIT,

    // &volcanic::KIT,

    // &mountains::KIT,

    // &cold::KIT,

    // &coast::KIT,

    // &tropical::KIT,

    // &temperate::KIT,

    // &agriculture::KIT,

    // &settlements::KIT,

    // &transport::KIT,
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
    /// its instance families (`crate::instances`)
    pub families: &'static [crate::instances::Family],
    /// its host-built linear features and stamps (`crate::features`)
    pub host: Option<crate::features::HostFn>,
}

impl Kit {
    pub const EMPTY: Kit = Kit { name: "", biomes_yaml: "", wgsl: "", layers: &[], kernels: &[], relief: None, relief_wgsl: "", families: &[], host: None };
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

/// The kits' WGSL and the glue calling it (`kits_slot_<slot>`, `kits_kernel`, `kits_relief`).
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
    s.push_str("fn kits_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {\n");
    for k in KITS {
        if k.relief.is_some() {
            s.push_str(&format!("    {}_relief(c, m, r, h);\n", k.name));
        }
    }
    s.push_str("}\n");
    s
}
