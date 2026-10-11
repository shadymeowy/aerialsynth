# Volcanic kit: volcanoes, calderas, shields, cinder fields, lava and ash

`crates/terragen/src/kits/volcanic.rs` (families, relief, layer, tests), its WGSL twins
`src/gpu/wgsl/kits/volcanic_relief.wgsl` (families, shared geometry, pass-A relief) and
`volcanic.wgsl` (the layer), and its biomes `crates/terragen/biomes/volcanic.yaml` (design:
[terrain-next.md §5.2](../design/terrain-next.md), contract: [kit-guide.md](../design/kit-guide.md)).

## What it draws

**Three instance families** (existence from the atlas' volcanism at the site, plus its plate
boundary and hotspot flags; a cheap nearest-texel volcanism test runs before the full sample):

| family | cell | draws |
|---|---|---|
| `volcanic_volcanoes` | 40 km | stratovolcanoes (summit crater) and calderas (flat floor, ring wall, lake) on arcs, rifts and hotspot tracks; R 6-25 km. A few per 100 km of arc. |
| `volcanic_shields` | 110 km | shield volcanoes on hotspot tracks and in continental rifts, R 22-45 km, summit pit |
| `volcanic_fields` | 28 km | cinder-cone fields: inside one instance a per-pixel lattice of cones with craters, maars and lava tongues (dense objects are never a family) |

**Relief (pass A, so the DSM, drainage and lakes follow).** A volcano is an absolute shape: its
summit is E + H above the atlas' smooth elevation E at the site, so oceanic volcanoes grow from
the sea floor and make islands. Flank E + H (1 - r/R)^1.6 (shield ^1.3), blended over the
ground by a smooth max that fades at the foot; the rim is ragged by three azimuth harmonics.
The crater (bowl) or caldera (flat floor, steep wall) overrides the ground; its radius is widened
to 1.5 pixels with the volume kept, and a volcano smaller than 2 pixels is widened with its
volume kept. Radial gullies: three tiers of 14, 30 and 62 ravines, random depth per ravine,
zero mean, band-limited; flank noise.

**Pass B (slot 3).** Black-sand beaches (the core's beach, recoloured on volcanic coasts and
by the biome palette); ash and scoria on the upper cone and an ash-fall apron around active
volcanoes (noise threshold); lava tongues down the flanks (fresh black, weathered brown,
old moss-green by age; no trees, no fields on them); hydrothermal sulphur, rust and clay patches
at the crater; crater and caldera lakes at one flat water level of the instance; in fields the
cinder cones (dark crater, pale rim), tephra rings and lakes of maars, an ash apron, short lava
tongues. Snow caps come from the climate (lapse rate; the cold kit's snow model). No fields
in craters, on lava, ash or lakes (`Masks::natural_ok`), and no town on a cone (`Stack::no_town`,
set in slots 1-5: the town is not selected there; the active cone's ash apron counts).

**Biomes** `volcanic_scrub` (warm and temperate volcanic country; black beaches) and
`volcanic_barren` (cold and high), with old and fresh lava fans from the `lobes` kernel. Arid
volcanic land stays with the desert kit's `basalt_harrat`.

Land-cover classes: `lava`, `volcanic_ash` (also black sand), `lake`, `bare`.

## Tests

`registry_compiles_with_the_volcanic_biomes`, `class_ids_are_the_v2_table`, `cone_shape`,
`crater_volume_is_kept_when_widened`, `gullies_have_zero_mean`,
`field_means_match_explicit_averages` (also: the relief skips only cells that cannot reach the
point), `lod_consistent_at_volcanic_places`, `seamless_at_volcanic_places` (E-W and N-S, albedo
and elevation), `deterministic_at_volcanic_places`, `tiles_match_the_cpu` and
`places_stay_on_the_gpu` (GPU; z6-z16 of the places), `bench_places` (ignored). The kit has no
kernel of its own, so there is no kernel parity test; tile parity (rgb, elevation, landcover)
was at most 0.04 % of pixels off at nine places and zooms.

## Performance

`bench_places` (RTX 6000 Ada, 9 places x 8 x 8 tiles per zoom, median of three blocks), tiles/s,
the same build with the `KITS` entry commented out vs the kit; and `zoom_bench` at the 13
default places (4 x 4 tiles):

| zoom | places: without | with | default places: without | with |
|---|---|---|---|---|
| z10 | 73.8 | 73.8 | 69.0 | 62.0 |
| z12 | 134.8 | 132-145 | 134.8 | 135.1 |
| z14 | 127.3 | 128-129 | 127.7 | 128.7 |
| z16 | 131.6 | 129-132 | 133.4 | 133.3 |

(z8 is dominated by shader compilation.) A sample reads only the instances there are (an
`InstList` zero-initialises 512 bytes per list: the first version of the layer cost 10-20 % at
every place because of it).

Cold compile (RTX 6000 Ada, all six kits): pass B 24 / 26 / 26 s (three pipelines), pass A1 17 s,
point evaluation 19 s, drainage heights 15 s.

## Stills

GPU-generated (`terrain survey`, default world, 10:30 local on the equinox, all six kits). Oblique
views stand off from the cone along the view direction.

**Crater lake** (11.850, -106.320): oblique 300 m, nadir 1.5 km, oblique 1.5 km from 3 km.
![crater lake](images/volcanic_crater_lake_300m.jpg)
![crater lake, nadir](images/volcanic_crater_lake_1500m_nadir.jpg)
![dormant cone](images/volcanic_cone_1500m.jpg)

**Caldera lake** (-44.677, 9.339, oblique 10 km from 13 km): fields on the farmland around the ring.
![caldera](images/volcanic_caldera_10km.jpg)

**Snow-capped cone** (56.885, -101.647, oblique 1.5 km): gullies show as ridges in the snow.
![snow cone](images/volcanic_snow_cone_1500m.jpg)

**Active cone with ash apron** (14.069, -13.412, oblique 10 km); **island** (27.069, -174.181,
oblique 1.5 km); **cinder field** (21.948, 71.408, nadir 1.5 km: cones, ash apron).
![active](images/volcanic_active_10km.jpg)
![island](images/volcanic_island_1500m.jpg)
![cinder field](images/volcanic_cinder_field_1500m_nadir.jpg)

## Known limits

Black-sand beaches need a flat coast (steep cones meet the sea in cliffs), so they are rare.
Islands of oceanic ecoregions have no vegetation. Lava follows radial lines, not the drainage.
At globe zooms (z < 6) more than 8 instances reach a block and the first 8 by id are kept.
