# Writing a kit

A **kit** adds landscapes to the generator: biomes (data), and where data is not enough,
kernels, layer functions, relief operators, instance families and host-built features (code).
This guide is the contract between the generator core (`crates/terragen`, design:
[terrain-next.md](terrain-next.md)) and a kit: which files to write, how to register them, which
APIs to use, and what a kit must prove before it is merged.

Everything a kit adds lives in **its own files**; the only shared line is its entry in
`src/kits/mod.rs`. Kits written in parallel therefore merge without conflicts.

Contents: [Files](#files) · [The stack](#the-stack) · [Biomes](#biomes) ·
[Kernels](#kernels) · [Layer functions](#layer-functions) · [Relief](#relief-pass-a) ·
[Instance families](#instance-families) · [Linear features and stamps](#linear-features-and-stamps) ·
[Rules](#rules-invariants) · [Testing](#testing) · [Performance budget](#performance-budget) ·
[Checklist](#checklist)

---

## Files

For a kit named `desert` (lower_snake_case; it prefixes everything it defines):

| file | content |
|---|---|
| `crates/terragen/src/kits/desert.rs` | `pub static KIT: Kit`, its Rust layer functions, kernels, relief, families, host features, its tests |
| `crates/terragen/src/gpu/wgsl/kits/desert.wgsl` | the WGSL twins (`desert_slot_<slot>`, kernels, family existence) |
| `crates/terragen/src/gpu/wgsl/kits/desert_relief.wgsl` | (only with a relief operator or a relief family) WGSL included in every module with pass A |
| `crates/terragen/biomes/desert.yaml` | its biomes and patches of core biomes |
| `docs/kits/desert.md` (optional) | what it draws, where, and stills |

**Registration** (the only shared file): in `src/kits/mod.rs`, replace your two placeholder
lines, `// mod desert;` → `mod desert;` and `// &desert::KIT,` → `&desert::KIT,`. Keep the
blank lines between the placeholders. The order of `KITS` is the order kits run within a
slot.

```rust
// src/kits/desert.rs
use crate::kits::{Kit, ReliefIn};
use crate::stack::{slot, Layer, Stack};

pub static KIT: Kit = Kit {
    name: "desert",
    biomes_yaml: include_str!("../../biomes/desert.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/desert.wgsl"),
    layers: &[(slot::AZONAL, azonal)],
    // kernels, relief + relief_wgsl, families, host: as needed
    ..Kit::EMPTY
};

fn azonal(s: &mut Stack) {
    // …
}
```

```wgsl
// src/gpu/wgsl/kits/desert.wgsl
fn desert_slot_azonal(s: ptr<function, Stack>) {
    // …
}
```

The WGSL glue (`kits_slot_<slot>`, `kits_kernel`, `kits_relief`, the family dispatch) is
generated from `KITS` (`kits::wgsl()`); a WGSL function a kit's `Kit` names must exist with
exactly that name and signature.

**Kernel kind ids:** each kit owns 16 ids from `kernels::KIT_BASE` (64), by its position in
the placeholder list: desert 64, volcanic 80, mountains 96, cold 112, coast 128, tropical 144,
temperate 160, agriculture 176, settlements 192, transport 208.

---

## The stack

Pass B evaluates every sub-sample as a fixed stack of layers (`src/stack.rs`,
`gpu/wgsl/stack.wgsl`), each composited over what lies below:

| slot | `slot::` | core content | kits typically add |
|---|---|---|---|
| 0 | `WATER_BODY` | standing water returns early, after its surface and the slot-8 layers (biome and kits) | – |
| 1 | `ZONAL` | soil / grass / tundra / marsh ground of the biome, textures, drainage lines | ground textures of a biome |
| 2 | `ALTITUDINAL` | – | alpine meadows, krummholz, nival zones |
| 3 | `AZONAL` | rock on slopes, sand seas, beaches; then masks, micro-relief, riparian belts | coasts, playas, scree, glaciers, wetlands, ice-wedge polygons |
| 4 | `DISTURBANCE` | – | burn scars, clear-cuts, windthrow |
| 5 | `AGRICULTURE` | fields of the land-use regions | terraces, vineyards, greenhouses, paddies, orchards, ponds |
| C | `CANOPY` | trees and shrubs of the biome's crown layers | emergent crowns, palms, gallery forests |
| 6 | `LINEAR` | noise-network roads, farm tracks | roads, rail, power corridors (from host features) |
| 7 | `BUILT` | farmsteads, towns, their lights | settlements, airports, ports, mines, solar farms |
| 8 | `WATER` | rivers; the surface of oceans and lakes | reefs, sea ice, frozen lakes, sediment plumes |
| 9 | `SEASONAL` | snow cover | seasonal snow, autumn colours |

On standing water (oceans, lakes) the stack is: the water surface, then the slot-8 biome
layers and kits (`s.l.water > s.l.ground`, `s.l.water_kind`; depth `s.l.water - s.l.ground`);
a layer with `solid: true` (sea ice, frozen lakes) covering more than half makes the sample
land again (`is_water` false). On land the seasonal slot (9) is skipped where the sample is
water.

Order of evaluation: water body → snow mask → frame → 1 → 2 → 3 → (masks) → 4 → 5 → **6, 7**
→ canopy → 8 → 9. Layers 6 and 7 are evaluated *before* the canopy (so towns can clear it)
and composited *after* it, as an affine transform of what lies below: there `hmode` `max`
acts as `blend` and `water` is ignored. Within a slot: the core layer, then the biome's kernel
layers (YAML), then the kits in `KITS` order.

### `Layer`

```rust
pub struct Layer {
    pub cov: f64,        // 0..1 coverage (crisp: a feature is there or not)
    pub albedo: DVec3,   // linear albedo where covered
    pub dh: f64,         // height, by `hmode`
    pub hmode: u8,       // NONE | BLEND (to ground + dh) | MAX (crowns) | ADD (dh·cov) | ABS (to dh)
    pub cls: u8,         // land-cover class where cov > 0.5 (0: keep)
    pub emit: DVec3,     // emission added (already weighted by cov)
    pub lit: f64,        // cast-shadow factor, min-composited (1: none)
    pub relit: f64,      // weight resetting lit to 1 (× cov): roads 0.5, water 1
    pub mat: u8,         // material id (land-cover v2 material table)
    pub clear: f64,      // share of the natural vegetation it removes (× cov), layers before the canopy
    pub water: bool,     // the sample is a water surface where cov > 0.5
    pub solid: bool,     // a solid surface over water where cov > 0.5 (sea ice, frozen lakes): clears `is_water`
}
```

`Layer::paint(cov, albedo)`, `Layer::paint_cls(cov, albedo, cls)` and `..Default::default()`
for the rest; in WGSL `layer_none()`, `layer_paint(cov, albedo, cls)`, `layer_emit(e)`. Composite
with `s.composite(layer)` / `composite(s, ly)`. Class ids are the v2 ids of
`docs/design/terrain-next.md` §7.2 (`registry::classes::id("tidal_flat")`; WGSL `LC_TIDAL_FLAT`).

### What a layer function can read (`Stack`)

| CPU | WGSL | |
|---|---|---|
| `s.ctx` (`p`, `east`, `north`, `up`, `lat`, `gsd`) | `(*s).c` | the sample |
| `s.t: &Terrain` | `(*s).t` | pass A at the pixel: climate (`temp`, `moist`), `mountain`, `sand`, `mesa`, `floodplain`, `gully`, `agri`, `habit`, `style`, river, sites |
| `s.l: &Local` | `(*s).l` | interpolated: `ground`, `water`, `water_kind`, `river_d`, `river_hw`, `river_level`, `slope`, `grad` (east, north m/m: aspect, flow), `fw` (filter width), `eco_edge`, `blk` (WGSL: the block of instance / feature lists) |
| `s.pf` | `(*s).pf` | pixel fields (`detail`, `patch`, `land`, `forest`, `snow`; lazy: `s.sm.pf_lazy(s.pf, PF_…)` / `s_pf_lazy(s, PF_…)`) |
| `s.bio` | `(*s).bio` | the sample's biome(s) `a`, `b`, blend weight `w`, the ecoregion style `style` / `st`, site climate |
| `s.m: Masks` | `(*s).m` | what lower layers published: `cover`, `rock`, `sand`, `beach`, `snow`, `shore_keep`, `natural_ok`, `flat_ok`, `micro`, `riparian`, `woodlot`, `field_cov`, `town_urban`, `town_cov`, `river_clear`, `road_major_cov`, `road_cov`, `river_cov`, `veg` |
| `s.region`, `s.q_loc`, `s.q_rot` | `(*s).has_region`, `(*s).r`, `q_loc`, `q_rot` | the land-use region and the local frames (m): `q_loc` east/north at the region centre, `q_rot` the region's field frame |
| `s.pal(pal::X)` | `spal(s, BP_X)` | the biome palette with the ecoregion's style |
| `s.veg(\|v\| v.trees)` | `sveg(s, VEG_TREES)` | vegetation parameters of the biome(s) |
| `s.agri()` | `s_agri(s)` | agricultural intensity (pass A × biome × culture) |
| `s.field(field::X)` | `s_field(s, FIELD_X)` | a named field (below) |
| `s.col`, `s.height`, `s.class`, `s.lit`, `s.emission` | same | the composite so far |

A kit may lower `s.m.natural_ok` (no fields / farmsteads there) or `s.m.veg` (no trees), and
should do it through `Layer::clear` where it draws. Publish nothing else into `Masks`; if a kit
needs a new shared value, ask for a field in the core.

---

## Biomes

Biomes are YAML entries (`biomes/<kit>.yaml`), compiled at start-up by `src/registry.rs`.
Errors name the key: `biome registry: biomes.salt_flat.layers[1].mask.moist: window [0.5, 0.1]
must be increasing`. A kit's file may **add** biomes and **patch** core ones (`patch: true`
changes only the keys given).

```yaml
- id: hot_desert_reg                  # lower_snake_case, unique
  group: bare                         # land-cover group of its natural cover (reporting)
  koppen: [BWh]                       # Köppen classes at the ecoregion site (empty: any)
  envelope: { temp: [18, 35], precip_mm: [0, 250], dry_months: [9, 12],   # at the site
              regime: [-1, 1], temp_range: [0, 40] }
  lithology: [sedimentary, crystalline]   # or `any`
  weight: 1.0                         # pick weight among the biomes that fit
  min_share: 0.01                     # variety test: at least this share of land samples
  palette:                            # sRGB 0..255 or palette names; omitted: core default
    soil: [[120, 92, 70], [110, 84, 64], [96, 88, 80], [140, 116, 92]]   # 4
    rock: [[90, 70, 60], [110, 88, 70], [70, 62, 58]]                    # 3
    sand: [[210, 180, 140], [196, 160, 118], [222, 204, 170]]            # 3
    grass_dry: [180, 164, 120]        # also grass_wet, grass_cold, tundra, marsh, beach,
                                      # wet_sand, crown_conifer, crown_decid, crown_tropic,
                                      # crown_dry, shrub, laterite, floor
  vegetation: { trees: 0.3, shrubs: 1.4, conifer: 0.0, tropic: 0.0, savanna: 1.0, groves: 1.0,
                woodlots: 1.0, tall: 0.0, laterite: 1.0, meadow: 1.0, crown_scale: 1.0, gallery: 0.0 }
  canopy:                             # crown layers (≤ 6) replacing the default four
    - { share: savanna, shape: umbrella, cell: 18, height: [6, 3], colour: [88, 96, 58],
        density: 1.0, closure: false, stand: false }
  zonation:                           # other biomes below these sample temperatures (°C)
    - { below_c: 4, biome: montane_scrub }
  layers:                             # ≤ 8 kernel layers
    - slot: azonal                    # zonal | altitudinal | azonal | disturbance | agriculture
                                      # | canopy | linear | built | water | seasonal
      kernel: patches
      mask: { moist: [null, 0.15], slope: [null, 0.25] }   # ≤ 4 windows over named fields
      params: { scale: 300, fraction: 0.4, colour: [70, 58, 50], colour2: [96, 80, 66] }
      class: gravel                   # v2 class name where it covers > 0.5
      height: add                     # none | blend | max | add (default: the kernel's)
      clear: 0.5                      # removes half the vegetation where it covers
  landuse: { agriculture: 0.2, towns: 0.5 }
```

**How biomes are picked.** The world is tiled by ~100 km ecoregions (`src/eco.rs`). At each
ecoregion's site the planetary atlas (`src/atlas/`) gives the climate (Köppen class at the
site's smooth elevation, mean temperature, annual precipitation, dry months, precipitation
regime, temperature range), lithology and culture; the biomes whose `koppen`, `envelope` and
`lithology` fit are drawn by `weight` (none fits: the nearest envelope). A sample takes its
ecoregion's biome, then that biome's `zonation` entry by its own temperature (lapse rate), with
a dithered ±1 °C transition. Within 10 km of an ecoregion border each ~220 m patch belongs to
one side or the other (a mosaic, not a gradient). Where that mosaic is below the pixel the two
sides blend; a sample then has two biomes `a`, `b` with weight `w` (≤ 2 biomes per sample).

**Ecoregion style and culture.** Every ecoregion also draws a style (`eco::EcoStyle`): soil and
rock colours by its lithology, grass and crown tints, tree density, conifer bias, shrubs,
woodlots, season; and from its culture's archetype (12 archetypes, `eco::ARCHES`): field-system
weights (grid, irregular, pivots, strips), field size, hedges, roof palette, building height,
block size, lamp mix, agriculture and town density. Kits read them through `s.bio.style`
(WGSL `(*s).bio.st`: `soil`, `rock`, `grass`, `crown`, `veg` = trees/conifer/shrubs/woodlots,
`land` = season/agri/towns/field scale, `fields`, `town` = hedges/roof/height/block, `misc` =
sodium/development/site temperature/site precipitation, `arch`, `litho`).

**Masks** are products of smoothstep windows over these fields (`registry::field`, WGSL
`FIELD_*`):

| field | | field | |
|---|---|---|---|
| `temp` | °C at the sample | `mountain` | mountain-belt mask 0..1 |
| `moist` | moisture 0..1 | `floodplain` | 0..1 |
| `slope` | tan of the ground slope | `agri`, `habit` | land-use suitability (pass A) |
| `gully` | − channel … + spur | `sand`, `rock_expect`, `mesa`, `cont` | pass-A masks, continent value |
| `height` | ground (m) | `lat` | |latitude| (°) |
| `river_dist` | m from the river bank (1e6: none) | `urban`, `field`, `natural`, `veg` | masks published so far |
| `patch`, `detail`, `land`, `forest`, `snow_noise` | pixel noise fields ~[-1, 1] | `style0..3` | regional style channels 0..1 |
| `eco_edge` | km to the ecoregion border | `precip_mm`, `dry_months` | the ecoregion site's climate |
| `coast_km` | signed coast distance (km, > 0 land; atlas) | `wind_e`, `wind_n` | annual wind (m/s; atlas) |
| `volcanism`, `glaciation` | 0..1 (atlas: arcs, rifts, hotspots; last-glacial ice) | `uplift` | tectonic relief potential −1..1 |
| `population`, `development` | 0..1 (atlas) | `regime`, `temp_range` | precipitation regime −1..1, warmest − coldest month (°C) |

A window is `[lo, hi]` (soft edges of 10 % of its width), `[a0, a1, b0, b1]` (in over a0..a1,
out over b0..b1), `null` for an open end.

The **resolved registry** (core + kits + the config's `biomes.overrides`) is stored with the
world (`biomes.resolved` in the store's `generator_config`): changing a biome makes a different
world, and stores refuse it as they refuse any other config change.

---

## Kernels

The kernel library (`src/kernels.rs`, `gpu/wgsl/kernels.wgsl`) generates surface features in a
local frame `q` (metres; biome layers use `s.q_loc`). Every kernel has an explicit evaluation,
a feature size, and a **calibrated mean** (its explicit result integrated over 4096
quasi-random points at registry build): a sample blends them by
`explicit = smoothstep(1.2·gsd, 3·gsd, size)`, so a coarse pixel shows the mean and a parent
tile is the mean of its children *per kernel*. Coverage is linear in `amount` (densities, fill
fractions and thresholds scale with it); masks enter as `amount`.

| kernel | features | main parameters (YAML) | size |
|---|---|---|---|
| `scatter` | instances on a jittered grid (3 × 3 cells): disc, dome, cone, umbrella, star, rect, crescent, ring | `cell, density, shape, radius [a,b], height [a,b], jitter, colour_var, shade, aspect, orient (random/aux), ring, second, colour, colour2` | cell |
| `rows` | planting rows or a grid of plants along a direction | `spacing, along, radius, height, angle, direction (param/aux), gaps, colour_var, colour` | spacing |
| `cells` | Worley partition: per-cell colour, edge lines | `cell, jitter, fill, edge, edge_height, colour_var, aspect, fill_height, colour, colour2, edge_colour` | cell |
| `stripes` | oriented stripes with per-lattice-point phases (no shearing) | `wavelength, angle, direction, sharpness, height, threshold, colour, colour2` | wavelength |
| `contours` | iso-lines / terraces of the height (`aux` = ground, slope) | `step, line, mode (lines/terraces), phase, riser, colour, colour2` | step / slope |
| `radial` | radial profiles: cones, craters, star arms | `cell, density, radius, height, exponent, arms, arm_amp, crater, crater_depth, jitter, colour, colour2` | radius |
| `crescent` | crescents (barchans) along a direction | `cell, density, radius, height_ratio, horns, direction, angle, jitter, colour` | radius |
| `lobes` | fan sectors with lobate margins and channels | `cell, density, radius, span, margin, channels, height, direction, jitter, colour, colour2` | radius |
| `patches` | crisp multi-scale noise threshold | `scale, fraction, octaves, roughness, height, colour_var, colour, colour2` | scale |
| `linear` | cross-section over a segment (`aux` = signed distance, along, half width) | `profile (flat/crowned/ridge/ditch), dash, duty, shoulder, height, marking, colour, shoulder_colour, marking_colour` | 2 × half width |
| `stamp` | templates in an oriented box (`aux` = u, v, half sizes): pad, strip, blocks, panels | `template, a, b, height, lights, colour, colour2, light_colour` | a |
| `canopy` | closed canopy: clumped crowns, per-clump hue, emergents | `cell, emergent, height, emergent_height, clump, colour_var, flowering, colour, colour2, flower_colour` | clump |
| `water` | water colour by depth (`aux` = depth), sediment, foam line | `depth_scale, sediment, sediment_scale, foam_depth, foam_width, colour, colour2, foam_colour` | sediment_scale |
| `city` | hook for the settlements kit (draws nothing in the core) | | |

Relief operators (pass A, height in, height out; `kernels::relief`, WGSL `relief_*` in the
kit's relief file if needed): `terrace`, `cliff`, `u_profile`, `flatten`, `bowl`,
`contour_bump`.

### Calling kernels from code

```rust
use crate::kernels::{self, KIn};
let kin = crate::layers::kin(s, kernels::kind::SCATTER, amount);   // frame, gsd, fw, aux
let o = kernels::eval(&k, &mean, &kin);                            // k: KParams, mean: KMean
```

A kit that uses its own parameters (not from YAML) builds them once
(`kernels::compile_params(spec, &yaml_map, seed, &pal)`, `kernels::calibrate`) in a
`std::sync::OnceLock`; in WGSL it fills a `KP` and calls `kernel_mix(k, li, size, mean,
mean_albedo, kin)` with the same numbers (put the calibrated mean in WGSL constants generated
from Rust, or pass YAML layers and use `kernel_eval(li, kin)`).

### Adding a kernel

1. Rust: `fn my_kernel(k: &KParams, i: &KIn) -> KOut` and a `KernelSpec` in your `Kit`:

   ```rust
   KernelSpec { name: "desert_yardangs", kind: 64, params: YARDANG_P, size_slot: 0,
                dyn_size: None, reach: None, default_hmode: "add",
                eval: yardangs, wgsl: "desert_yardangs" }
   ```

   `params` is the YAML schema: `P::Num(name, slot, min, max, default)`, `P::Range`, `P::Enum`,
   `P::Col(name, colour index, C::N("palette name") | C::Rgb([r, g, b]))`.
2. WGSL: `fn desert_yardangs(li: u32, k: KIn) -> KOut { let p = kp_load(li); … }` in your
   kit's WGSL (dispatched by `kits_kernel`).
3. Contract: deterministic (hashes `hash2`/`hash3`/`u01k` of integer lattice coordinates);
   `cov` linear in `k.amount`; bounded loops; instances within `reach` of their site
   (`KernelSpec::reach = Some((radius slot, cell slot, max cells))`, checked at registry build);
   **noise in the local frame `q`**, or at ECEF `p` only with wavelengths computed in f64 from
   the (f32-rounded) parameters — an f32 wavelength shifts an ECEF noise by `|p|/λ · 6e-8` cells.
4. Parity: add it to a test calling `gpu::tests::kernel_parity` (below).

---

## Layer functions

A layer function draws what data cannot (`fn(&mut Stack)` / `fn <kit>_slot_<slot>(s:
ptr<function, Stack>)`), e.g. a coast that needs the ground height near sea level, sea ice that
needs the water depth. Write the Rust first, port it line by line; keep the two side by side
(same names, same order of operations).

```rust
fn azonal(s: &mut Stack) {
    let t = s.t;
    if t.moist > 0.2 || s.l.ground > 900.0 {
        return;                                     // cheap gates first
    }
    let k = playa_kernel();                          // OnceLock'ed KParams + KMean
    let kin = crate::layers::kin(s, crate::kernels::kind::CELLS, 1.0);
    let o = crate::kernels::eval(&k.0, &k.1, &kin);
    let salt = crate::registry::classes::id("salt_flat").unwrap();
    s.composite(Layer { cov: o.cov * flat, albedo: o.albedo, cls: salt, clear: 1.0, ..Default::default() });
}
```

```wgsl
fn desert_slot_azonal(s: ptr<function, Stack>) {
    let t = (*s).t;
    if (t.moist > 0.2 || (*s).l.ground > 900.0) {
        return;
    }
    // …
    var ly = layer_paint(o.cov * flat, o.albedo, LC_SALT_FLAT);
    ly.clear = 1.0;
    composite(s, ly);
}
```

**WGSL pitfalls** (naga → SPIR-V, NVIDIA): never pass a pointer to a member of the stack
(`&(*s).pf`) to a function — copy it to a local `var` and back (naga's SPIR-V backend panics:
"Expression is not cached"); `target`, `class`, `template`, `filter`, `sample`, `mod`, … are reserved words; no
recursion; every call site of a big function is inlined (keep one call site per heavy
function); 64-bit types (`u64`, `i64`, `f64`) are available; positions are `vec3<f64>`, local
coordinates `f32`. `cargo test -p terragen wgsl_compiles` parses, validates and compiles every
entry point to SPIR-V without a GPU.

---

## Relief (pass A)

A kit that shapes the ground (dunes, volcanoes, karst, cliffs, terraces) sets `relief` and
`relief_wgsl`. The operator runs after the core relief (continents, belts, hills, gullies,
mesas, dunes) and **before the drainage** carves valleys, in every context that computes the
ground: tile pixels, point queries and the drainage lattice heights. So its inputs are analytic
only — macro fields, climate, lattice hashes, instance families — never host tables.

```rust
fn relief(w: &crate::world::World, r: &ReliefIn, h: &mut f64) {
    // r.ctx (p, east, north, gsd), r.m (macro fields), r.temp, r.moist, r.mountain, r.sand,
    // r.mesa, r.smooth (≥ 5 km elevation)
}
```

```wgsl
// kits/desert_relief.wgsl (included in the tile, point and drainage modules)
fn desert_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {
    // r.temp, r.moist, r.mountain, r.sand, r.mesa, r.smooth_h
}
```

Height changes must be band-limited (`band(λ, gsd)`, or the operator's mean where its features
are below the pixel) and volume-preserving where widened (R-energy): the elevation LOD test
(parent ≈ mean of children) checks it.

---

## Instance families

Sparse landforms and objects with a
bounded reach — volcanoes, atolls, inselbergs, kettle lakes, cinder cones, quarries — are
instances of a **family**: the sites of a 3D jittered lattice (seamless on the sphere), each
existing by an analytic test at its centre.

```rust
pub struct Family {
    pub name: &'static str,     // "<kit>_<what>"; WGSL: fn <name>_exists(site: InstSite) -> InstOut
    pub cell: f64,              // lattice cell (m)
    pub jitter: f64,            // ≤ 0.9
    pub reach: f64,             // how far an instance reaches from its centre (m), ≤ cell
    pub relief: bool,           // pass A (also shapes the drainage) or pass B only
    pub exists: fn(&World, &InstSite) -> Option<[f32; 8]>,   // None: no instance; else its parameters
}
```

Each 16 × 16-pixel block lists the instances (≤ 8 per family, by id) that can reach it; a
sample evaluates only those (`s.instances(FAMILY)` / `inst_count(f, blk)`, `inst_get(f, blk,
k)`; in relief operators `instances::near(w, FAMILY, p)`). The existence test sees the
instance's id, centre (surface point below its site) and the analytic fields there — macro
fields, and the atlas (`w.atlas().sample(site.center.normalize())`; WGSL
`atlas_sample(vec3<f32>(normalize(site.center)))`: `boundary`, `volcanism`, `hotspot`,
`litho`, `glaciation` …). Its WGSL `<name>_exists` goes into the kit's relief WGSL (every
module), with the family constants `FAM_<NAME>`. A layer reads a family with
`inst_list(FAM_X, (*s).l.blk, (*s).c.p)` (CPU `s.instances(i)`), a relief operator with
`inst_list(FAM_X, r.blk, c.p)` (CPU `r.instances(w, i)`), returning an `InstList { n, items }`
of `Inst { id, center, v }`; its
decision must not depend on per-pixel values (R-site). More than 8 instances of a family in a
block is an error: choose a larger cell or a lower density (the density test in
`tests/invariants.rs` checks the default world).

---

## Linear features and stamps

Graphs and sites that need terrain samples
(roads, rail, power lines, dams, airports, ports, deltas) are built by **host** Rust code shared
by both backends and arrive per tile as binned lists:

```rust
pub struct FSeg {   // a segment: ECEF ends, heights, kind (kit-defined), class, half width,
    pub a: DVec3, pub b: DVec3, pub ha: f32, pub hb: f32,   // arc length at a, deck heights,
    pub kind: u32, pub class: u8, pub hw: f32, pub s0: f32, // flags, 4 parameters
    pub deck_a: f32, pub deck_b: f32, pub flags: u32, pub v: [f32; 4],
}
pub struct FStamp { // an oriented box: centre, axes, half sizes, template (kit-defined), parameters
    pub center: DVec3, pub ex: DVec3, pub ey: DVec3, pub half: [f32; 2],
    pub template: u32, pub h: f32, pub v: [f32; 8],   // (WGSL: `tmpl`, `cls`: reserved words)
}
pub host: Option<fn(&mut HostCtx, &Area) -> Features>   // in `Kit`
```

`HostCtx` gives the world and point evaluations of the terrain (`ctx.terrain(lat, lon, gsd)`:
`None` while the GPU evaluates it — return what you have; the host re-runs you), identical on
both backends. Lists are cached per area; a feature must be a pure function of the world and
its own neighbourhood (never of the tile). Per sample, `s.segments()` / `s.stamps()` (WGSL
`feat_seg_*`, `feat_stamp_*`) give the bin's features, and `features::seg_frame(seg, p)` the
signed distance, along-distance and fraction — the `aux` of the `linear` and `stamp` kernels.

---

## Rules (invariants)

The generator's invariants (`docs/design/terrain-next.md` §1) apply to every kit:

* **Pure function of (config, position).** No tile-local state; caches are memoization only.
* **Same world on both backends.** Discrete decisions (existence, class, topology) from integer
  hashes of lattice coordinates, shared host Rust, or values at instance centres — never from
  a per-pixel float near a threshold (R-site).
* **Band-limited, LOD-consistent.** A feature smaller than ~2 pixels shows its mean; use the
  kernels' crossfade, `band(λ, gsd)`, or a calibrated mean. Parent tile ≈ mean of children.
* **Crisp (R-crisp).** A feature is there or not; never fade a feature's opacity with a mask
  (translucent ghosts) — let masks thin instances (`amount`) or move a crisp edge.
* **Energy (R-energy).** Unresolved lights and pits widen with their energy / volume preserved.
* **Cut-off (R-cutoff).** Switch a feature off only where it covers ≤ a few % of a pixel.
* **Never modulate a wavelength or direction by a spatially varying field at ECEF coordinates
  (R-wave):** use per-lattice-point phases (`stripes`, `gully_octave`).
* **Seamless.** Grid/exact choices depend on the zoom only, never on the tile.
* **Bounded work.** Fixed loops; ≤ 8 kernel layers per biome; ≤ 2 biomes per sample.

---

## Testing

Every kit adds tests in its `kits/<kit>.rs` (`#[cfg(test)] mod tests`):

1. **Registry**: `registry::Registry::builtin()` compiles with your YAML (the core test
   `builtin_registry_compiles` runs it too).
2. **Kernel parity** (each kit kernel, and the core kernels with your parameters where you rely
   on unusual ones):
   ```rust
   #[cfg(feature = "gpu")]
   #[test]
   fn kernels_match_the_cpu() {
       let Some(r) = crate::gpu::tests::kernel_parity(&[("desert_yardangs", "{wavelength: 120}")]) else { return };
       for (name, bad, _) in r { assert!(bad < 0.005, "{name}"); }
   }
   ```
3. **Kernel means**: `kernels::calibrate` vs an explicit average (see
   `kernels::tests::means_match_explicit_averages`) for your kernels.
4. **Tiles**: places where your kit draws, both backends — `gpu::tests::compare_tiles(gpu_tile,
   cpu_tile)`, at most 0.5 % of pixels off (rgb > 2 DN, elevation > 5 cm), as
   `tiles_match_the_cpu`.
5. **Invariants**: `tests/invariants.rs` (determinism, seams E-W / N-S, parent ≈ mean of
   children for elevation, albedo and class histograms) runs on the default world; add your
   places to its lists if your features are local.
6. **Variety budget** (`tests/variety.rs`): every biome with `min_share` > 0 reaches it at the
   default seed; no legacy class group above 45 % of land.
7. **WGSL**: `cargo test -p terragen wgsl_compiles` (no GPU needed).
8. **Stills**: `cargo run --release -p terragen --example zoom_bench -- gpu --places … --png
   DIR` (or the survey tool, `terrain survey`), at z8, z12, z14, z16 — look at them, and put
   before/after pairs in your report.

Run on the farm (`rrun --gpu …`): the GPU tests skip themselves without a GPU.

---

## Performance budget

Per-zoom budgets (`docs/design/terrain-next.md` §9; measured with `zoom_bench`, GPU and CPU):
a kit may add at most **5 %** to the tiles/s of any zoom of the default bench places, and
**15 %** at places where it draws. Guidance:

* Gate early: test the cheap masks (climate, height, slope, biome) before any noise or loop.
* A noise octave (`perlin3`) costs ~8 hashes; a 3 × 3 kernel ~9–27; a Worley 3D query 27.
* Use `band(λ, gsd)` to skip octaves that cannot be seen; kernels skip their explicit part
  where `explicit = 0` (coarse pixels cost one table read).
* Per sample: ≤ 2 kernel evaluations per layer, ≤ 8 instances per family, ≤ 4 stamps per bin.
* Work that depends on an instance or a site only belongs in its existence test / host list,
  not per pixel.

---

## Checklist

- [ ] files: `kits/<kit>.rs`, `wgsl/kits/<kit>.wgsl` (and `_relief.wgsl`), `biomes/<kit>.yaml`
- [ ] registered: the two lines in `kits/mod.rs` (blank lines kept)
- [ ] kernel kinds in the kit's range; WGSL names prefixed with the kit
- [ ] Rust and WGSL side by side, same order of operations
- [ ] `cargo test -p terragen` (CPU) and on the farm with `--gpu`: parity, invariants,
      variety, `wgsl_compiles`
- [ ] stills before / after at z8, z12, z14, z16 of the places the kit changes
- [ ] `zoom_bench` before / after: within the performance budget
- [ ] `CHANGELOG.md` (Unreleased) line; `GENERATOR_VERSION` is the lead's (one bump per release)

---

## The planetary atlas

`src/atlas/` (branch `next-atlas`) precomputes per world, on a ~20 km cube map, what cannot be
computed locally: rain shadows and monsoons, temperature with currents and seasonality, plates
and boundary types, tectonic uplift and volcanism, lithology, glaciation, coast distance,
cultures, development and population. The core uses it for:

* **relief** — mountain ranges along plate boundaries (arcs, collision belts, rift shoulders:
  amplitude from `max(uplift, 0)`, `relief.tectonic_mountains`), the noise belts kept as old
  orogens (`relief.belt_mountains`), grabens from negative uplift;
* **climate** — temperature = atlas sea-level temperature − lapse × height; moisture from the
  aridity P / (20 T + 140) (Köppen's dry threshold) plus a little noise;
* **ecoregions** — biome by the site's Köppen class, envelope, lithology; culture and archetype
  from the atlas' culture areas;
* **land use** — town density by population, roads and lights by development.

Pass A samples it once per 16-pixel grid node (`Macro`, interpolated); the per-pixel values
reach the stack in `Terrain` (`uplift`, `coast_km`, `wind`, `volcanism`, `glaciation`,
`population`, `development`, `temp_range`, `regime`) and as mask fields. Kits sample it
directly only at instance centres or host sites.
