//! The composite stack of pass B (`docs/design/terrain-next.md` §3.2): every sub-sample is a
//! fixed sequence of layers, each returning a [`Layer`] that is composited over what lies below.
//!
//! ```text
//!  0 relief & water (pass A)   standing water returns early (its surface: slot 8 layers)
//!  1 zonal biome               ground of the ecoregion's biome (soil, grass, textures)
//!  2 altitudinal zone          zonation by lapse temperature (alpine meadow, nival …)
//!  3 azonal overrides          rock, sand seas, beaches, riparian, wetlands, coast kit …
//!  4 disturbance               burn scars, clear-cuts …
//!  5 agriculture               fields of the culture's field systems
//!  C canopy                    natural vegetation (trees, shrubs) of the biome, masked by
//!                              everything that claims the ground (`Masks::veg`)
//!  6 linear infrastructure     roads, tracks, rail …
//!  7 built & stamps            farmsteads, towns, sites …
//!  8 water surfaces & ice      rivers (and the surface of standing water)
//!  9 seasonal overlay          snow cover
//! ```
//!
//! Layers 1–5 are evaluated and composited in order. Layers 6 and 7 are evaluated before the
//! canopy (they may clear it: towns have no forest) and composited after it, in order: their
//! composite is kept as a transform of what lies below ([`Deferred`]: colour, height and light
//! are affine in it), applied after the canopy. There, height modes `max` act as `blend`, and
//! `water` is ignored. A layer that sets [`Layer::clear`] removes that fraction of the natural
//! vegetation where it covers.
//!
//! Kits add layers at any slot ([`crate::kits`]); the WGSL twin is `gpu/wgsl/stack.wgsl`.

use crate::kits;
use crate::surface::{Caches, Local, PixFields, RegionInfo, Surface, SurfaceModel, TownInfo};
use crate::world::{Ctx, Terrain, World};
use glam::{DVec2, DVec3};

/// Slots of the stack (composite order; [`CANOPY`] between 5 and 6).
pub mod slot {
    pub const WATER_BODY: usize = 0;
    pub const ZONAL: usize = 1;
    pub const ALTITUDINAL: usize = 2;
    pub const AZONAL: usize = 3;
    pub const DISTURBANCE: usize = 4;
    pub const AGRICULTURE: usize = 5;
    pub const LINEAR: usize = 6;
    pub const BUILT: usize = 7;
    pub const WATER: usize = 8;
    pub const SEASONAL: usize = 9;
    /// the canopy (natural vegetation), between 5 and 6
    pub const CANOPY: usize = 10;
    pub const N: usize = 11;
    pub const NAMES: [&str; N] =
        ["water_body", "zonal", "altitudinal", "azonal", "disturbance", "agriculture", "linear", "built", "water", "seasonal", "canopy"];
}

/// How a layer sets the DSM height where it covers.
pub mod hmode {
    /// unchanged
    pub const NONE: u8 = 0;
    /// blended towards ground + `dh` by the coverage (buildings, roads with `dh` 0)
    pub const BLEND: u8 = 1;
    /// raised to ground + `dh` where covered (tree crowns)
    pub const MAX: u8 = 2;
    /// `dh`·coverage added (field relief, small objects on the ground)
    pub const ADD: u8 = 3;
    /// blended towards the absolute height `dh` (water surfaces)
    pub const ABS: u8 = 4;
}

/// One layer at a sample.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    /// coverage 0..1 of the sample (crisp: a feature is there or not, R-crisp)
    pub cov: f64,
    /// linear albedo where covered
    pub albedo: DVec3,
    /// height (see `hmode`)
    pub dh: f64,
    pub hmode: u8,
    /// land-cover class where `cov` > 0.5 (0: keep the one below)
    pub cls: u8,
    /// emission added (already weighted by the coverage)
    pub emit: DVec3,
    /// cast-shadow factor (min-composited; 1: none)
    pub lit: f64,
    /// weight of resetting the shadow to fully lit (× `cov`; roads 0.5, water 1)
    pub relit: f64,
    /// material id (0: default; land-cover v2 material table)
    pub mat: u8,
    /// share of the natural vegetation (canopy) it removes where it covers (layers 1–7)
    pub clear: f64,
    /// a water surface where `cov` > 0.5 (the sample is water: glint, class)
    pub water: bool,
}

impl Default for Layer {
    fn default() -> Self {
        Layer { cov: 0.0, albedo: DVec3::ZERO, dh: 0.0, hmode: hmode::NONE, cls: 0, emit: DVec3::ZERO, lit: 1.0, relit: 0.0, mat: 0, clear: 0.0, water: false }
    }
}

impl Layer {
    /// A layer of colour `albedo` at coverage `cov` that only paints.
    pub fn paint(cov: f64, albedo: DVec3) -> Layer {
        Layer { cov, albedo, ..Default::default() }
    }
    /// … that also sets the class where it covers more than half.
    pub fn paint_cls(cov: f64, albedo: DVec3, cls: u8) -> Layer {
        Layer { cov, albedo, cls, ..Default::default() }
    }
}

/// Values the layers publish for the layers above them (masks), in evaluation order.
#[derive(Clone, Copy, Debug)]
pub struct Masks {
    /// grass cover of the zonal ground (0 bare soil .. 1)
    pub cover: f64,
    pub rock: f64,
    pub sand: f64,
    pub beach: f64,
    pub snow: f64,
    /// fields keep back from the shore
    pub shore_keep: f64,
    /// (1 − rock)(1 − snow)(1 − sand)(1 − beach): where natural land use can be
    pub natural_ok: f64,
    /// fields on gentle slopes only
    pub flat_ok: f64,
    /// micro-relief of the ground (m)
    pub micro: f64,
    pub riparian: f64,
    pub woodlot: f64,
    pub field_cov: f64,
    /// how built-up the selected town is (0..1) and its coverage at the sample
    pub town_urban: f64,
    pub town_cov: f64,
    pub river_clear: f64,
    pub road_major_cov: f64,
    pub road_cov: f64,
    pub river_cov: f64,
    /// allowance of natural vegetation (1: none removed); layers' `clear` lowers it
    pub veg: f64,
}

impl Default for Masks {
    fn default() -> Self {
        Masks {
            cover: 0.0,
            rock: 0.0,
            sand: 0.0,
            beach: 0.0,
            snow: 0.0,
            shore_keep: 1.0,
            natural_ok: 1.0,
            flat_ok: 1.0,
            micro: 0.0,
            riparian: 0.0,
            woodlot: 0.0,
            field_cov: 0.0,
            town_urban: 0.0,
            town_cov: 0.0,
            river_clear: 1.0,
            road_major_cov: 0.0,
            road_cov: 0.0,
            river_cov: 0.0,
            veg: 1.0,
        }
    }
}

/// A town's pixel at a sample: colour, height, coverage, class, shadow, emission
/// (`SurfaceModel::town`).
pub type TownPx = (DVec3, f64, f64, u8, f64, DVec3);

/// The state of one sub-sample going through the stack.
pub struct Stack<'a> {
    pub world: &'a World,
    pub sm: &'a SurfaceModel,
    pub cache: &'a mut Caches,
    pub ctx: &'a Ctx,
    pub t: &'a Terrain,
    pub l: &'a Local<'a>,
    pub pf: &'a PixFields,
    /// the sample's biome and ecoregion (resolved parameters)
    pub bio: crate::registry::BioSample,
    pub m: Masks,
    /// field-system region at the sample and the local frames: unrotated (east, north) and the
    /// region's rotated frame
    pub region: Option<RegionInfo>,
    pub q_loc: DVec2,
    pub q_rot: DVec2,
    /// the most built-up town at the sample (and how built-up), its pixel there
    pub town: Option<(TownInfo, f64)>,
    pub town_px: Option<TownPx>,
    /// the composite so far
    pub col: DVec3,
    pub height: f64,
    pub class: u8,
    pub lit: f64,
    pub emission: DVec3,
    pub is_water: bool,
    /// a layer ended the stack (a water surface): the rest is skipped
    pub done: bool,
    /// layers 6–7: evaluated before the canopy, composited after it
    deferring: bool,
    deferred: Deferred,
}

/// The composite of the layers evaluated before the canopy and drawn over it, as a function of
/// what lies below: colour `k·c + c0`, height `kh·h + ch`, light `min(la·l + lb, lmin)`, class.
#[derive(Clone, Copy, Debug)]
pub struct Deferred {
    k: f64,
    c: DVec3,
    kh: f64,
    ch: f64,
    la: f64,
    lb: f64,
    lmin: f64,
    cls: u8,
}

impl Default for Deferred {
    fn default() -> Self {
        Deferred { k: 1.0, c: DVec3::ZERO, kh: 1.0, ch: 0.0, la: 1.0, lb: 0.0, lmin: 1.0, cls: 0 }
    }
}

impl<'a> Stack<'a> {
    /// Composite `ly` over the stack (or keep it for after the canopy while layers 6–7 are
    /// evaluated). Its `clear` takes effect immediately.
    pub fn composite(&mut self, ly: Layer) {
        if ly.clear > 0.0 {
            self.m.veg *= 1.0 - ly.clear * ly.cov.clamp(0.0, 1.0);
        }
        if self.deferring {
            self.defer(&ly);
            return;
        }
        self.apply(&ly);
    }

    /// `ly` into the deferred composite (emission is added at once: it is order-independent).
    fn defer(&mut self, ly: &Layer) {
        self.emission += ly.emit;
        let a = ly.cov.clamp(0.0, 1.0);
        if a <= 0.0 {
            return;
        }
        let d = &mut self.deferred;
        d.k *= 1.0 - a;
        d.c = d.c * (1.0 - a) + ly.albedo * a;
        match ly.hmode {
            hmode::BLEND | hmode::MAX | hmode::ABS => {
                let target = if ly.hmode == hmode::ABS { ly.dh } else { self.l.ground + ly.dh };
                d.kh *= 1.0 - a;
                d.ch = d.ch * (1.0 - a) + target * a;
            }
            hmode::ADD => d.ch += ly.dh * a,
            _ => {}
        }
        if ly.relit > 0.0 {
            let w = ly.relit * a;
            d.la *= 1.0 - w;
            d.lb = d.lb * (1.0 - w) + w;
            d.lmin = d.lmin * (1.0 - w) + w;
        }
        d.lmin = d.lmin.min(ly.lit);
        if a > 0.5 && ly.cls != 0 {
            d.cls = ly.cls;
        }
    }

    /// Draw the deferred composite over the stack.
    fn flush(&mut self) {
        let d = self.deferred;
        self.col = self.col * d.k + d.c;
        self.height = self.height * d.kh + d.ch;
        self.lit = (self.lit * d.la + d.lb).min(d.lmin);
        if d.cls != 0 {
            self.class = d.cls;
        }
        self.deferred = Deferred::default();
    }

    fn apply(&mut self, ly: &Layer) {
        let a = ly.cov.clamp(0.0, 1.0);
        self.emission += ly.emit;
        if a <= 0.0 {
            return;
        }
        self.col = self.col + (ly.albedo - self.col) * a;
        match ly.hmode {
            hmode::BLEND => self.height = lerp(self.height, self.l.ground + ly.dh, a),
            hmode::MAX => self.height = self.height.max(self.l.ground + ly.dh),
            hmode::ADD => self.height += ly.dh * a,
            hmode::ABS => self.height = lerp(self.height, ly.dh, a),
            _ => {}
        }
        if ly.relit > 0.0 {
            self.lit = lerp(self.lit, 1.0, ly.relit * a);
        }
        self.lit = self.lit.min(ly.lit);
        if a > 0.5 && ly.cls != 0 {
            self.class = ly.cls;
        }
        if a > 0.5 && ly.water {
            self.is_water = true;
            self.done = true;
        }
    }

    /// Multiply the composite colour (a tint over everything below).
    pub fn tint(&mut self, k: DVec3) {
        self.col *= k;
    }

    pub fn p(&self) -> DVec3 {
        self.ctx.p
    }
    pub fn gsd(&self) -> f64 {
        self.ctx.gsd
    }
    /// filter width of the sub-sample (m)
    pub fn fw(&self) -> f64 {
        self.l.fw
    }

    /// The value of a named mask field (`registry::field`) at the sample.
    pub fn field(&self, f: usize) -> f64 {
        use crate::registry::field::*;
        let t = self.t;
        match f {
            TEMP => t.temp,
            MOIST => t.moist,
            SLOPE => self.l.slope,
            GULLY => t.gully,
            HEIGHT => self.l.ground,
            RIVER_DIST => {
                if self.l.river_hw > 0.0 {
                    self.l.river_d.abs() - self.l.river_hw
                } else {
                    1e6
                }
            }
            PATCH => self.pf.patch,
            DETAIL => self.pf.detail,
            LAND => self.pf.land,
            FOREST => self.pf.forest,
            SNOW_N => self.pf.snow,
            MOUNTAIN => t.mountain,
            FLOODPLAIN => t.floodplain,
            AGRI => t.agri,
            HABIT => t.habit,
            SAND => t.sand,
            ROCK_EXPECT => t.rock_expect,
            MESA => t.mesa,
            CONT => t.cont,
            LAT => self.ctx.lat.abs().to_degrees(),
            URBAN => self.m.town_urban,
            FIELD => self.m.field_cov,
            NATURAL => self.m.natural_ok,
            VEG => self.m.veg,
            STYLE0 => t.style[0],
            STYLE1 => t.style[1],
            STYLE2 => t.style[2],
            STYLE3 => t.style[3],
            ECO_EDGE => self.bio.edge_km,
            PRECIP => self.bio.site.precip_mm,
            TEMP_RANGE => self.bio.site.temp_range_c,
            DRY_MONTHS => self.bio.site.dry_months,
            _ => 0.0,
        }
    }
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Evaluate the surface at one sub-sample: the stack's layers in order.
pub fn eval(sm: &SurfaceModel, world: &World, cache: &mut Caches, ctx: &Ctx, l: &Local, pf: &PixFields) -> Surface {
    let bio = crate::registry::BioSample::at(world, sm, cache, ctx, l, pf);
    let mut s = Stack {
        world,
        sm,
        cache,
        ctx,
        t: l.t,
        l,
        pf,
        bio,
        m: Masks::default(),
        region: None,
        q_loc: DVec2::ZERO,
        q_rot: DVec2::ZERO,
        town: None,
        town_px: None,
        col: DVec3::ZERO,
        height: l.ground,
        class: crate::landcover::UNKNOWN,
        lit: 1.0,
        emission: DVec3::ZERO,
        is_water: false,
        done: false,
        deferring: false,
        deferred: Deferred::default(),
    };
    use crate::layers as core;
    use crate::layers::biome_layers as biome;

    // ---- 0: standing water (its surface: the water layers)
    if l.water > l.ground && l.water_kind != crate::world::water::NONE {
        core::water::standing(&mut s);
        biome(&mut s, slot::WATER);
        kits::slot(slot::WATER, &mut s);
        return s.finish();
    }
    // the masks some layers need before their own slot (snow: natural land use), the frame
    core::snow::prepare(&mut s);
    core::frame(&mut s);
    // ---- 1 .. 5
    core::ground::zonal(&mut s);
    biome(&mut s, slot::ZONAL);
    kits::slot(slot::ZONAL, &mut s);
    biome(&mut s, slot::ALTITUDINAL);
    kits::slot(slot::ALTITUDINAL, &mut s);
    core::azonal::layer(&mut s);
    biome(&mut s, slot::AZONAL);
    kits::slot(slot::AZONAL, &mut s);
    core::azonal::finish_ground(&mut s);
    biome(&mut s, slot::DISTURBANCE);
    kits::slot(slot::DISTURBANCE, &mut s);
    core::agriculture::layer(&mut s);
    biome(&mut s, slot::AGRICULTURE);
    kits::slot(slot::AGRICULTURE, &mut s);
    // ---- 6, 7 (evaluated now, composited after the canopy)
    s.deferring = true;
    core::built::select_town(&mut s);
    core::linear::layer(&mut s);
    biome(&mut s, slot::LINEAR);
    kits::slot(slot::LINEAR, &mut s);
    core::built::layer(&mut s);
    biome(&mut s, slot::BUILT);
    kits::slot(slot::BUILT, &mut s);
    s.deferring = false;
    // ---- canopy
    core::canopy::layer(&mut s);
    biome(&mut s, slot::CANOPY);
    kits::slot(slot::CANOPY, &mut s);
    s.flush();
    // ---- 8, 9
    core::water::rivers(&mut s);
    if !s.done {
        biome(&mut s, slot::WATER);
        kits::slot(slot::WATER, &mut s);
    }
    if !s.done {
        core::snow::layer(&mut s);
        biome(&mut s, slot::SEASONAL);
        kits::slot(slot::SEASONAL, &mut s);
    }
    s.finish()
}

impl Stack<'_> {
    fn finish(&self) -> Surface {
        Surface {
            albedo: if self.is_water { self.col } else { self.col.max(DVec3::ZERO) },
            height: self.height,
            class: self.class,
            lit: self.lit,
            is_water: self.is_water,
            emission: self.emission,
        }
    }
}
