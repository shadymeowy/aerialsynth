# Tropical and savanna kit

The wet and dry tropics from 100 m to 10 km above the ground: rainforest canopies with emergent
and flowering crowns, floodplains with meander scars, cloud forest and páramo on the mountains,
mangroves on low coasts; savannas in dry-season colours with umbrella acacias, baobabs, palm
groves, termite mounds, burn scars, waterholes and gallery forests; monsoon, dry deciduous and
miombo woodland; slash-and-burn frontiers. Design: [terrain-next.md §5.6](../design/terrain-next.md);
contract: [kit-guide.md](../design/kit-guide.md).

| file | content |
|---|---|
| `crates/terragen/src/kits/tropical.rs` | the kit: seven kernels, the instance lattice on a reference sphere, tests |
| `crates/terragen/src/gpu/wgsl/kits/tropical.wgsl` | the WGSL twins of the kernels |
| `crates/terragen/biomes/tropical.yaml` | eleven biomes (two of them patches of the core's `tropical_moist` and `tropical_dry`) |

The kit is data over kernels: no layer functions, no relief, no host features. Every feature is
a biome layer (`layers:` in the YAML) over one of the kit's kernels or a core kernel, so each is
evaluated identically on both backends, crossfaded with its calibrated mean where it is below
the pixel, and placed by the biome envelopes, the ecoregions and the masks.

## Kernels (kinds 144–150)

| kernel | draws | notes |
|---|---|---|
| `tropical_crowns` | tree crowns on a jittered grid: dome, umbrella, star (palms; leafless baobabs), per-crown hue, a second colour and a flowering colour by hash, leaf mottling | groves: a crown exists where the clump noise *at its centre* is below `clump_fraction` (crisp per crown); a mask thins crowns by shrinking them to nothing at the density threshold, never by clipping |
| `tropical_fire` | burn scars: a crisp threshold of 3-octave lobate noise at ECEF `p`; `ages` crisp age classes from a smoother noise (a younger fire over an older one): charred black with grey ash streaks and a dark fire front, brown, green regrowth | seamless (ECEF noise, f64 wavelengths) |
| `tropical_swidden` | slash-and-burn plots: a warped Worley partition; a share `fill` of the cells holds an irregular clearing inside a forest margin, burnt, cropped or in bushy fallow | local frame |
| `tropical_waterhole` | waterholes: a lobate pond (`part: pond`, class lake); a dark mud rim, trampled bare ground and 6–12 wavy, tapering game trails radiating out (`part: ground`) | instance lattice; the two parts share the `seed` |
| `tropical_meander` | meander scars: the horseshoe of a cut-off channel, open water over part of it (`part: lake`, an oxbow lake) and silted elsewhere; scroll bars (ridges and swales of young forest or grass) filling the loop (`part: scroll`) | instance lattice; bars band-limited to their mean duty |
| `tropical_mangrove` | mangrove forest: a crisp lobate extent moved by the mask, dense dark crown clumps of two species tones, lower along the creeks (`part: canopy`); tidal creeks along the edges of two warped Worley lattices, each edge its own width, some missing (`part: channels`) | ECEF lattices: seamless |
| `tropical_grass` | savanna grass in dry-season colours: golden and grey-straw tones, bare (laterite) patches, tussock mottling at metres | its extent a crisp threshold by the mask |

**Instance lattice.** Waterholes and meander scars are instances of a jittered 3D lattice in
ECEF coordinates on a reference sphere (`sphere_frame`, `sphere_site`): a site exists if it lies
in the slab of half a cell around the sphere (a decision of the site alone, one site per cell²
of surface), and the instance is drawn in the tangent frame of the sample. They are seamless on
the planet and independent of the land-use regions (whose frames the per-pixel kernels use).
The core's instance families (milestone 2) arrived after the kit; the lattice needs no host
lists and costs ~5 cell hashes per sample after the bound test.

## Biomes

| biome | Köppen / envelope | what it shows |
|---|---|---|
| `tropical_moist` (core, patched) | Af, Am | lowland rainforest: broad flat-topped crowns of stands, emergent giants (pale, some flowering yellow or pink-red), treefall gaps, a band of pale pioneer crowns along the rivers, meander scars on the floodplains, a slash-and-burn frontier where farming begins; agriculture × 0.2 |
| `flooded_forest` | Af, Am, ≥ 1800 mm | várzea / igapó: lower canopy with swamp palms, black-water pools, many meander scars, river-edge forest |
| `monsoon_forest` | Am, Aw, ≥ 1300 mm, 2–7 dry months | semi-deciduous forest: leafless crowns among green ones, flame-of-the-forest flowering orange, small burns, swidden |
| `tropical_dry_forest` | Aw, 800–1900 mm, 4–9 dry months | dry deciduous forest: grey-brown leafless and green crowns tree by tree, yellow flowering, fires, swidden, evergreen gallery forest in the gullies |
| `tropical_dry` (core, patched) | Aw | savanna: golden grass, umbrella acacias, burn scars, termite mounds, gallery forests along drainage lines and rivers, mangroves on its coasts |
| `savanna_acacia` | Aw, BSh, 300–1100 mm | acacia savanna with waterholes and game trails |
| `savanna_baobab` | Aw, BSh, 350–1000 mm | baobab savanna: wide leafless crowns, a few in leaf |
| `palm_savanna` | Aw, Am, 900–2400 mm | llanos: palm groves, gallery forests, meander scars with grassy scroll bars |
| `miombo_woodland` | Aw, Cwa, 650–1500 mm | open woodland with bronze new leaves, large wooded termitaria, dambos (treeless grassy drainage lines), fires, swidden |
| `tropical_cloud_forest` | zone below 17.5 °C | dense low mossy canopy, tree ferns, landslide scars on steep slopes |
| `tropical_paramo` | zone below 10 °C | tussock grass, silvery frailejón rosettes, tarns |

Laterite: red soils (`laterite` palette, `vegetation.laterite`) in every lowland biome. Cloud
forest and páramo are the altitudinal zones of the forest and savanna biomes (`zonation`); they
are never picked by an ecoregion (an impossible envelope keeps them out of the nearest-envelope
fallback too). Mangroves are two layers of the coastal biomes: on the sea where it is
shallower than ~5 m (water slot; the bed below sea level, so no inland lakes) and on the coastal
and delta flats below ~3 m (canopy slot), not on arid coasts. The default world has few such
places inside tropical ecoregions (its coasts rise quickly, and near-shore water often belongs
to an ecoregion whose site lies at sea), so mangroves are rare there; no still shows them yet.

Classes (v2): `tropical_rainforest`, `mangrove`, `broadleaf_forest`, `woodland`, `savanna`,
`burn_scar`, `clear_cut` (swidden), `lake` (oxbows, ponds, pools, tarns),
`bare`, `marsh`, `grass`.

## What needs the core

* **River geometry per pixel.** Point bars, cut banks and live meanders need the river's
  along-coordinate `s`, its curvature sign and the bend's inner side in `Local`; meandering
  channels need the drainage to meander. The kit draws meander *scars* beside the rivers
  (a mask on `river_dist` and `floodplain`), not the river's own bends.
* **River colour by ecoregion** (muddy white-water, black-water, sediment plumes): a palette
  entry for the river surface read by `layers::water::rivers`.
* **Coast distance at pixel resolution.** `coast_km` comes from ~20 km atlas texels and can miss
  the pixel coast by hundreds of km (237 km at a beach of the default world), so the mangroves
  use the height near sea level instead.
* **Water-slot layers that become land** (`solid` for YAML layers): mangroves over shallow sea
  keep the sample a water surface (the class is mangrove, `is_water` stays true).
* **Ecoregions at sea.** Near-shore water whose ecoregion site lies at sea gets that site's
  biome, not the coast's, so coastal water layers (mangroves, plumes) miss many coasts.
* **Envelope by latitude or temperature range** so a highland ecoregion in the tropics (Cwb,
  Cfb) can pick tropical montane biomes; today cloud forest is only a zone of lowland biomes.
* **Shadows of kit crowns.** The core canopy bakes cast shadows (`lit`) of its own crown layers
  only; kit crowns (emergents, baobabs, palms, gallery forests) rely on the renderer's DSM
  shadows.

## Stills

GPU-generated (`terrain survey` with `AERIALSYNTH_GPU_STRICT=1`), default world.

**Savanna** (22.50, −27.79): golden dry-season grass, umbrella acacias, burn scars of three ages
(charred with ash, brown, green regrowth), gallery forests along every drainage line.

![savanna 1.5 km](images/tropical_savanna_1500m.jpg)
![burn scar and gallery forest, nadir 1.5 km](images/tropical_burn_scar_1500m_nadir.jpg)
![savanna 10 km](images/tropical_savanna_10km.jpg)

Before (the same place without the kit, generator version 4 core):

![savanna 1.5 km, before](images/tropical_savanna_1500m_before.jpg)

**Rainforest** (12.20, −28.59): closed canopy with emergent crowns, yellow and pink-red flowering
crowns, the pale pioneer band along the streams, treefall gaps, meander scars on the
floodplain (in the core alone this place was mostly fields).

![rainforest 1.5 km](images/tropical_rainforest_1500m.jpg)
![rainforest canopy, nadir 300 m](images/tropical_rainforest_300m_nadir.jpg)

**Oxbow lake** (6.39, 150.99): a cut-off meander beside the river with its gallery forest.

![oxbow lake 300 m](images/tropical_oxbow_300m.jpg)

**Monsoon forest mosaic** (−17.95, −128.59): semi-deciduous forest and savanna glades, small
burns and swidden plots.

![monsoon forest 1.5 km](images/tropical_monsoon_1500m.jpg)

## Tests and performance

* `cargo test --release -p terragen` (CPU): registry, kernel means vs explicit averages, masks
  thin instances continuously, the instance lattice ignores the land-use frame, invariants,
  variety. With a GPU (`rrun --gpu`): `kernels_match_the_cpu` (12 cases, 0.000 % of samples
  off), `tiles_match_the_cpu` (7 places, z12–z16, ≤ 0.021 % of pixels off), `places_stay_on_the_gpu`
  (the places at z6–z16 with `Backend::Gpu`: no fallback).
* `kits::tropical::tests::bench_places` (ignored; GPU, RTX 6000 Ada, 448 tiles per zoom over
  the 7 places, two runs each, the baseline with the kit's `KITS` entry commented out):

  | zoom | with the kit (tiles/s) | without (tiles/s) | cost |
  |---|---|---|---|
  | z8 | 10.0–10.4 | 10.0–10.3 | ~0 % |
  | z10 | 55.6–57.4 | 59.2–59.7 | 4–6 % |
  | z12 | 115.8–118.4 | 123.6–123.8 | 4–6 % |
  | z14 | 105.9–108.7 | 115.5–116.9 | 7–9 % |
  | z16 | 109.6–114.5 | 120.5–120.6 | 5–9 % |

  Within the 15 % budget where the kit draws. On the default `zoom_bench` places (its
  "rainforest" and "savanna" places are no longer tropical biomes in the atlas world) the
  difference is within the run-to-run noise (±2 %).
* `places` and `find_places` (ignored) list where the biomes and features occur.
