# Temperate, boreal and wetland kit

`crates/terragen/src/kits/temperate.rs` (layer functions, tests), its WGSL twin
`src/gpu/wgsl/kits/temperate.wgsl`, and its biomes `crates/terragen/biomes/temperate.yaml`
(design: [terrain-next.md §5.7](../design/terrain-next.md), contract:
[kit-guide.md](../design/kit-guide.md)).

## What it draws

**Biomes** (data; picked per ecoregion by Köppen class, envelope, lithology, weight):

| biome | Köppen | look | classes |
|---|---|---|---|
| `broadleaf_deciduous` | Cfa Cfb Dfa Dfb Dwa Cwa | oak / beech / hornbeam crowns (three crown layers of different size and green), few conifers | broadleaf forest |
| `mixed_forest` | Dfb Dwb Dsb Cfb | stands of spruce, pine, birch and oak side by side (needleleaf, broadleaf and mixed stands) | needleleaf / broadleaf / mixed forest |
| `taiga` | Dfc Dfd Dwc Dwd Dsc | dark narrow spruce spires, lighter pines, birch, a pale moss and lichen floor; open spire taiga where it is too dry for closed forest | needleleaf forest |
| `lichen_woodland` | Dfc Dfd Dwc Dwd, < 560 mm | widely spaced spires on cream reindeer lichen | woodland, needleleaf forest |
| `maquis` | Csa Csb | dense mottled olive-grey and dark green scrub, stone pines and holm oaks, terracotta soils, limestone outcrops | maquis |
| `garrigue` | Csa Csb on carbonate | low grey-green cushions on white limestone | maquis, rock |
| `chaparral` | Csa Csb | dense dark brush with burn scars of different ages | maquis, burn scar |
| `feather_grass_steppe` | BSk Dfa Dfb Dsa Dsb Dwa, 200–560 mm | silvery-buff tussock grass on black chernozem | steppe |
| `heathland` | Cfb Cfc Dfb on acid rocks | purple-brown heather, scattered birch and pine, blown sand | shrub, sand |
| `moorland` | Cfc Cfb Dfc (and above heathland) | heather burnt in strips of different ages, grass moor, peat hags | shrub, grass, bog |
| `subalpine_krummholz` | zone | dwarf pine mats, stunted spruce, meadows | shrub |
| `alpine_meadow` | zone | bright meadows with flowers, boulders and cattle terracettes | alpine meadow |

Zonation (by the sample's temperature): the temperate and Mediterranean biomes (also the core's
`temperate`, `temperate_broadleaf`, `temperate_conifer`, `mediterranean`, patched) climb through
mixed forest to krummholz (below 1 °C) and alpine meadow (below −1.5 °C). The core `steppe`
gets the steppe class.

**Layer functions** (code, in every biome where the climate and terrain fit):

* **Wetlands** (azonal slot): raised bogs — ochre, red-brown and green sphagnum, a darker wet
  margin, pools on concentric rings near the dome's top, a few stunted pines; string bogs (aapa
  mires, cold boreal) — narrow strings along the iso-lines of a slow flow field with flark pools
  between them; fens near streams and in water tracks — green sedge, willow and birch scrub;
  reed marshes on floodplains and river banks — two greens and last year's straw in clonal
  patches, open water in the core and a network of creeks; beaver ponds on small streams — a
  lens-shaped pond, a ring of dead grey trees, the beaver meadow; floodplain meadows with
  swales parallel to the channel. Pools and ponds are class *lake*; mires *bog*, reed *marsh*.
* **Managed forest** (disturbance slot): rectangular stands (a brick pattern of columns
  150–300 m wide in the land-use region's frame) where the culture manages its forests (boreal,
  survey-grid and temperate archetypes, by development and a per-ecoregion draw), some cut in the
  last 16 years — fresh slash with skid trails, a landing and a few seed trees, then grass and
  fireweed, then dense young plantations in rows — with logging roads along the column edges
  (class *clear-cut*, roads *track*); windthrow — patches of fallen trunks lying in the storm's
  direction; firebreaks — straight cleared strips with a track through the forests and scrub of
  dry-summer climates (and the managed boreal forest).
* **Autumn** (canopy slot): deciduous crowns in yellow, gold, orange, red and brown by the
  ecoregion's season (the red share is the ecoregion's: maple-red or beech-gold regions);
  larches in the cool forests, light green, gold in autumn.

Kernels: the core `scatter` with the kit's own parameters (autumn crowns, larches, young trees,
fallen trunks, dead snags); their parameters and calibrated means are generated into the WGSL
file (`wgsl_consts`, checked by `wgsl_consts_up_to_date`; `AERIALSYNTH_BLESS=1` rewrites it).

## Stills

STILLS

## Tests

* `registry_compiles`, `wgsl_consts_up_to_date`;
* `tiles_match_the_cpu` (GPU): five places (raised bog, reed marsh, maquis, managed forest with
  clear-cuts, lichen woodland) at z16, z14, z12, z9 — at most 0.5 % of the pixels off;
* `lod_seams_determinism`: parent vs mean of children (albedo, class histogram) at z12 and z15,
  determinism with a fresh generator, the E-W seam;
* `find_places` / `find_features` (ignored): where the kit's biomes and classes are.
