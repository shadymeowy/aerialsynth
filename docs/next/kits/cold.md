# Cold kit: glacial, periglacial and ice

A kit of the experimental [next generator](../README.md) (`world.generator: next`). *The default
world* on this page is seed 1 of the next generator.

`crates/terragen/src/kits/cold.rs` (layer functions, kernel, drumlin family, tests), its WGSL
twin `src/gpu/wgsl/kits/cold.wgsl` (+ `cold_relief.wgsl`), and its biomes
`crates/terragen/biomes/cold.yaml` (design: [terrain-next.md §5.4](https://github.com/shadymeowy/aerialsynth/blob/terrain-design/docs/design/terrain-next.md),
contract: [kit-guide.md](../kit-guide.md)). Taiga, lichen woodland, krummholz and alpine
meadow belong to the temperate kit.

## What it draws

**Biomes:** `polygon_tundra` (ice-wedge polygons, sorted circles, wet sedge), `shrub_tundra`,
`polar_desert` (frost-shattered gravel, patterned ground), `forest_tundra` (tundra with scattered
stunted spruce).

**Layer functions** (climate: warmest-month temperature for snow line and glaciers, temperature at
the ecoregion's date for seasonal snow and frozen water):

| slot | draws |
|---|---|
| 1 zonal | perennial snow mask by the summer temperature (snow line lower on shaded, steep ground by aspect) |
| 3 azonal | thermokarst and kettle lakes, drained lake basins (sedge meadow), pingos (some collapsed) |
| 8 water | sea ice (fast ice, floes, leads, pressure ridges), frozen lakes, seasonal (frozen) rivers |
| 9 seasonal | glaciers (firn, bare ice, crevasses, ogives, medial moraines), lateral and terminal moraines, seasonal snow and névé |

**Drumlin fields with eskers** (instance family `cold_drumlins`, pass-A relief): glaciated lowlands
(atlas `glaciation`) get streamlined hills along one ice-flow azimuth per field and sinuous esker
ridges.

**Kernel** `cold_polygons` (3D Worley lattice, calibrated mean). The kernel's parameters and
calibrated means are generated into the WGSL file; `AERIALSYNTH_BLESS=1` rewrites it.

## Tests

`registry_compiles_with_the_cold_biomes`, `class_ids_are_the_v2_table`, `lakes_stay_within_reach`,
`lake_means_match_explicit_averages`, `polygon_mean_matches_explicit_average`, `dates`,
`lod_consistent_at_cold_places`, `seamless_at_cold_places`, `kernels_match_the_cpu` and
`tiles_match_the_cpu` (GPU), `places_stay_on_the_gpu` (z6 to z16 of the nine places under
`GPU_STRICT` semantics), `bench_places` (ignored; GPU tiles/s).

## Performance

`bench_places` (RTX 6000 Ada, 9 places x 8 x 8 tiles per zoom), tiles/s, same build with the
`KITS` entry commented out vs the kit:

| zoom | without | with | change |
|---|---|---|---|
| z10 | 124.4 | 116.6 | -6 % |
| z12 | 125.2 | 114.1 | -9 % |
| z14 | 131.4 | 115.7 | -12 % |
| z16 | 109.1 | 124.6 | noise (+14 %) |

(z8 is dominated by shader compilation and is not meaningful.) Within the 15 % budget where the
kit draws.

## Stills

GPU-generated (`terrain survey`, `AERIALSYNTH_GPU_STRICT=1`), default world, 10:30 local on the
equinox (glacier and polygon tundra: 12:00 at the December solstice). Lat, lon in brackets.

**Valley glacier** (-66.095, 123.288, oblique 1.5 km, summer solstice noon): firn, dirty ice, crevasse
fields, flow lines. The stills' grey snow is the sensor's fixed exposure, not the albedo.
![glacier](images/cold_glacier_1500m.jpg)

**Ice field** (-72.199, -89.333, oblique 10 km): outlet streams, moraine bands.
![ice field](images/cold_icefield_10km.jpg)

**Late-melt polygon tundra** (-72.339, -102.499, oblique 1.5 km): the bare patches are fragments
with a polygon net of snow in the troughs, not blobs.
![polygons](images/cold_polygons_1500m.jpg)

**Kettle lakes and snowy taiga** (58.609, -92.893, oblique 1.5 km); the pale rectangles are the
temperate kit's boreal clear-cuts (this is taiga at 2 C, not forest tundra).
![kettle](images/cold_kettle_1500m.jpg)

**Forest tundra** (-71.075, -168.603, oblique and nadir 300 m): ice-wedge polygons, thermokarst
ponds.
![tundra](images/cold_tundra_300m.jpg)
![tundra nadir](images/cold_tundra_300m_nadir.jpg)

**Sea ice** (-71.842, -70.019, oblique and nadir 10 km): fast ice, leads, floes.
![sea ice](images/cold_seaice_10km.jpg)
![sea ice nadir](images/cold_seaice_10km_nadir.jpg)

**Drumlin field under snow** (57.314, -26.177, nadir 10 km): the relief is subtle under the snow.
![drumlins](images/cold_drumlins_10km_nadir.jpg)

## Notes

* Seasonal snow's date fades to the annual mean over 10 km towards an ecoregion border (the
  mosaic of two ecoregions with different dates drew snow-free blobs next to snow). The melt edge
  is ragged with fine noise and a snow net along ~18 m polygon troughs.
* The temperate kit's `forestry` is zero where the biome's trees are below 0.6 (forest tundra,
  tundra, polar desert): no managed stands there.

## Known limits

* Every ecoregion's date lies in the warm season (mid May to late September, by its season):
  seasonal snow lies only where that date is still cold (high mountains, the polar regions),
  lakes freeze only there. Winter landscapes need render-time seasons (35 % of the ecoregions
  were dated in winter: snowy taiga and frozen grey lakes in summer datasets, with hard borders
  to their summer neighbours).
* Kettle lakes are lobed ellipsoid slices (shore noise 0.24 at 2.6 per radius): bays and points,
  but no islands, no chains along eskers.
* Glaciers, moraines and seasonal snow are pass-B surfaces (colour, class, height in the
  seasonal slot): the DTM and the drainage keep the valley below them; troughs and cirques come
  from the mountains kit's carve hook, not from the glaciers drawn here.
* Sea ice is a function of the atlas' temperature at the ecoregion's date and of the depth: no
  drift, no polynyas tied to winds or coasts; leads and pressure ridges are noise patterns.
* Kettle and thermokarst lakes are lattices tested per pixel: their existence near a lattice
  site's threshold is decided at the site (both backends agree), but the transport kit's routing
  only estimates their mask from pass A's values and the relief's slope.
* Drumlin fields take one ice-flow azimuth per field (no fanning around obstacles); eskers are
  sinuous ridges independent of the drainage.
* Pingos and polygons ignore the drainage: polygons continue across small streams.
* The snow line uses the warmest-month temperature and aspect, not avalanche paths or wind
  redistribution: snow fields have smooth edges on steep ground.

Polish round 6: perennial snow holds on slopes to ~45° and 40 % of it on walls steeper than ~60° (the cut was
35°–50° to 15 %).
