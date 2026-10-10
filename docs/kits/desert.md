# Desert kit

Hot and cold deserts from 100 m to 10 km above the ground: sand seas whose dunes follow the
wind, the stony deserts between them, playas and dry lakes, wadis, yardangs and inselbergs.
Design: [terrain-next.md §5.1](../design/terrain-next.md); contract:
[kit-guide.md](../design/kit-guide.md).

| file | content |
|---|---|
| `crates/terragen/src/kits/desert.rs` | the kit: proxies, ridge fields, slab lattices, relief, layers, families, tests |
| `crates/terragen/src/gpu/wgsl/kits/desert_relief.wgsl` | WGSL of the relief, the families' existence tests and the shared helpers (every module with pass A) |
| `crates/terragen/src/gpu/wgsl/kits/desert.wgsl` | WGSL of the layers (tile module) |
| `crates/terragen/biomes/desert.yaml` | six biomes and the kit's textures on the core's `desert` and `desert_red` |

## What it draws

**Dunes by wind regime (relief, pass A).** Where the core's sand-sea mask is, the core's
transverse dunes are taken out and the kit's put in, by the direction and the variability of
the wind:

* direction: the atlas' annual wind, sampled at the pixel (its cubic B-spline is smooth; the
  tile's grid-interpolated value kinks at the grid lines, which ridge phases turn into creases),
  with a trade / westerly belt where it is calm (`desert::wind`);
* variability ρ: high where the annual wind is weak (it comes from changing directions), with
  continental-scale noise (the atlas keeps no seasonal winds);
* ρ < 0.36 **transverse** ridges across the wind (λ 600 m, gentle stoss, a lee face of 14 % of the
  wavelength, sinuous crests); 0.36–0.68 **linear (seif)** ridges along the wind (λ 1.7 km, up to
  65 m, sharp crests, flat corridors, Y-junctions and terminations); ρ > 0.68 **star dunes**
  (3–5 sinuous arms, 750–1150 m radius, 100–300 m high) on a network of transverse ridges;
  **small dunes** (λ 240 m) on the linear and star dunes.

The ridge fields (`desert::ridges`) are oriented plane waves carried by the points of a jittered
3D lattice with anisotropic kernels and per-point phases (no direction or wavelength is
modulated at ECEF coordinates, R-wave), a smooth phase warp for sinuous crests, and a calibrated
mean where unresolved (`band(λ, gsd)`; `tests::ridge_means`). The relief keeps the core's dune
amplitude (`relief.dune_height_m`).

**Barchans (pass B, height added).** On the sand-poor fringe of the sand seas (sand mask
0.02–0.58), most where the wind is steady: crescents 70–190 m wide, 6–25 m high, horns
downwind, on a darker stony pavement with desert varnish and sand streaks along the wind.

**Playas and dry lakes.** Playas are an instance family (`desert_playas`, cell 36 km): they exist
at arid, warm, flat sites (no mountains, low hills, outside sand seas and mesas; the atlas and
the macro fields at the site), flatten their floor towards the smooth relief and are drawn as
white to beige salt crust with polygonal cracks (8–20 m), a darker wet floor, pink or green
brine pools in some, and pale paleo-shorelines on the slopes around. Lakes in arid climates
(the water-slot layers run on standing water) dry out from their margins inwards into salt
pans at the lake's level (`solid`: the sample is land again).

**Inselbergs.** An instance family (`desert_inselbergs`, cell 16 km) at dry sites off the
mountains and sand seas, mostly on crystalline rock (atlas lithology): domes 1–5 km across and
up to 600 m high, elongated and lobed, with a pediment; bare granite streaked by varnish;
widened with their volume kept where below the pixel.

**Yardangs.** Streamlined ridges along the wind (λ 180 m, 12 m) on hyper-arid, warm plains
outside the sand seas.

**Washes.** Dry channels in arid land wider than the channel (up to 2.4 × its half width), with
braided threads of gravel and sand bars parallel to the banks that merge and split; acacia and
tamarisk along them, oasis gardens and date palm groves on some stretches (biome layers).

**Colour and biomes.** Sand takes the biome's sand palette by the regional style, tinted by the
ecoregion's lithology: iron-red on crystalline rock, pale on carbonate, yellow on unconsolidated
sediment, dark grey to black (class `volcanic_ash`) on volcanic rock. Biomes (by Köppen class,
envelope and lithology at the ecoregion site): `hot_desert_erg` (Sahara and Arabia),
`hot_desert_hamada` (varnished rock pavement), `basalt_harrat` (black boulder fields),
`gypsum_desert` (white sands), `desert_scrubland` (BWh / BSh margins), `cold_desert_gobi` (BWk
gravel plains, black gravel, saxaul); the core's `desert` and `desert_red` get the kit's
varnish, gravel mottling, scrub, wash vegetation and oases.

Land-cover classes: `sand`, `gravel` (stony fringe, varnished pavement), `salt_flat`,
`dry_riverbed`, `volcanic_ash` (black sand), `rock` (inselbergs, hamada), `desert_scrub`,
`orchard` (palm groves), `crop` (oasis gardens).

## Instances without families

Barchans (cell 380 m) and star dunes (cell 2.6 km) are too dense for the per-block family
lists at coarse zooms (≤ 8 per block); they are sites of jittered 3D lattices kept within half
a cell of the ellipsoid (`desert::slab_sites`: about one site per cell² of surface, seamless,
decided by integer hashes), enumerated per sample (≤ 27 cells, ~6 on average) where their zone
is, and their calibrated mean where unresolved (`tests::shape_constants`).

## Stills

<!-- stills: docs/kits/images/ -->

## Tests

`cargo test --release -p terragen desert` (CPU) and with a GPU (`-- --test-threads=1`):

* `registry_compiles`, `class_ids`;
* `ridge_means`, `shape_constants`: the hard-coded means and shape integrals (Rust and WGSL
  constants) against their numerical values;
* `lod_consistency`: parent ≈ mean of children (elevation, albedo, the kit's classes) at the
  kit's places;
* `seamless_and_deterministic`: tile borders at z11 / z15 and the same tile twice;
* `tiles_match_the_cpu` (GPU): tiles at z10, z13, z16 of the kit's places, at most 0.5 % of the
  pixels off;
* ignored tools: `desert_places` (places of the default world by landform),
  `desert_previews` (`DESERT_PNG=dir`, nadir rgb and hillshade at the kit's places or
  `DESERT_AT="lat,lon;…"`, `DESERT_Z`, `DESERT_CPU`), `desert_probe`.
