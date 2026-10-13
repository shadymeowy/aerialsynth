# Changelog

## Unreleased

- **Visual polish, round B** (see the kits' Known limits): tower karst on a rounder, taller profile with forest paint on
  the walls; fjords as narrow troughs (new `Carve::narrow` in the carve hook: the valley's extent from the walls alone);
  folds and fault blocks taller, striking along the atlas' uplift gradient, fault basins flat; canyon depth 400-1500 m
  by province; volcanic ash in rays, higher shaded cinder cones; settlement districts with wandering borders and fewer
  arterials; softer cotton, smallholder, flowering, fishbone, regrowth, varnish, burn-scar and road-verge colours.

- **Host work no longer stalls the GPU** (outputs bit-identical: `tilediff` of the 738 tiles of
  two views and 64 z16 tiles at 39.9N 32.8E against the previous master; parity and
  determinism tests pass). The kits' host features of a GPU batch are built in parallel on the
  host while the GPU runs pass A, with exact host point evaluations (the kits re-take close
  decisions exactly, so either backend's samples give the same features; the GPU rounds took
  20–30 settle rounds per cold batch), on a thread pool of the generator's own (`crate::pool`).
  A new `Kit::prepare` hook computes what a batch's tiles share once, in parallel (transport:
  every edge near the batch routed once; settlements: the places, top-down); a route's
  corridor probes and fine corridors run in parallel. Point queries of the drainage
  (`World::river_segments_of`, radius ≤ 100 m) and of instance families (`instances::near`,
  `r = 0`) filter short per-query-cell lists instead of scanning hundreds of lattice cells;
  wider drainage queries, flow points, node pieces, lake levels, places and `edges_near` are
  memoized in shared sharded tables (`memo::Memo`). Sites list the towns of an area only at
  pixel sizes where a site is drawn (a zoom-2 tile resolved every town of a continent). The
  tile store writes a batch while the next one is generated; the planner estimates a tile's
  children with it (one GPU evaluation instead of five). Fixed a deadlock when tiles are
  generated from rayon's pool while the generator's host work used it
  (`parallel_callers_do_not_deadlock`). Tesla P4, 960×540 nadir views at 39.9N 32.8E: 3 km
  834 → 33 s, 10 km 827 → 41–46 s (v0.2.0: 50 / 56 s), the GPU busy 90–95 % of the time;
  64 cold tiles z12 / z14 / z16: 86 / 39 / 31 → 11.2 / 6.5 / 5.6 s (v0.2.0: 3.8 / 3.4 /
  2.9 s; most of the rest is the kits' pass B on the GPU and the cold area's routes); survey
  of 3 places × 3/9/16 km views 253 → 161 s; `places_stay_on_the_gpu` > 60 min → 73 s,
  `glow_keeps_the_mean` 12 → 3 min.

- **Deltas** (coast kit, design §5.5): fan and bird's-foot deltas (2–12 km) at the mouths of the
  major rivers, found where the river meets the *drawn* coast (host, exact f64 samples along the
  meander-warped centre line, the same for both backends; `kits/coast/delta.rs`): lobate flat
  land 0.5–2 m extending the shoreline over the shelf, a shallow front, distributaries carved
  below the sea; reed marsh or mud on top (slot 5), mangroves in the tropics (the tropical
  biomes' land layer). New core hook for pass-A stamps (`Kit::astamps`, `Kit::astamp`: applied
  after the drainage carve in tiles and points; on the GPU in the sink-lake list with a negative
  radius). Tests `deltas_sit_at_river_mouths` (301 deltas at 24 coastal regions: 297 on their
  river at the mouth, 255 grown seaward), `deltas_are_tile_independent`; delta places in the
  coast kit's GPU/CPU, LOD and seam tests.
- **Confluences at an angle.** All the channels draining into a drainage node were bent into its
  outflow by the same quadratic curve, so they met tangentially and ran side by side for
  kilometres before joining (two rivers 100–200 m apart over 1–2 km). Now the main stem (the
  inflow that continues straightest into the outflow) keeps the bend, and every other channel's
  last reach is a cubic from the middle of its edge (entered along its own direction) to a
  junction on the main stem's bend, met at 35–80° to the main stem; the tributaries of one node
  join at different points (by rank). CPU `World::node_pieces` and WGSL `node_pieces` alike
  (`lat_main`, `trib_controls`); the pieces stay zoom-independent. Test
  `tributaries_join_at_an_angle` (no two channels of a level meet at < 12°; 26 of 421 before).
- **Road decks are backend-independent.** The engineered profiles' point samples (ground and
  water level) are quantized to 1 m, from the exact sample where the backend's is within 4 cm of
  a rounding boundary or near a water decision (`transport::build::deck_sample`), so GPU and CPU
  build the same decks, bridges, tunnels and portals; `sites_and_routes_match_the_cpu` now
  compares them too.
- **Towns and cities stay out of lakes.** The best candidate of a town, city or metropolis gives
  way to the next where a lattice lake or a closed-basin lake of the major rivers can reach its
  centre (cheap conservative screens over a few drainage lattice points, no drainage pieces,
  ~0.05 ms per town: `World::lattice_lake_near`, `World::sink_near_level`). Test
  `towns_avoid_lakes` (3 of 3000 centres in pass A's water, in streams' ponds; 7 before).

- **Qanats** (desert kit, design P2): dotted lines of shaft mounds (pale spoil rings around a
  dark shaft every 30–55 m) from the piedmont down to a common outlet, 1–3 lines of 3–8 km per
  group; host segments (`qanat_group`, 9 km lattice, by culture, aridity and the smooth relief's
  slope), drawn in slot 6 with their mean along the line below the pixel.
- **Ranches and feedlots** (agriculture kit, design P2): a `ranch` field system (2.2 × 1.5 km
  paddocks, stock trails to troughs, farm dams, savanna trees where warm), weighted to semi-arid
  steppe climates and pastoral, steppe and survey-grid cultures; feedlots in some ranch regions
  (slot 6): pens of dark manure along feed alleys and bunks, cattle, a feed mill with silage
  bunkers and silos, two manure lagoons.
- **Fishbone deforestation** (tropical kit, design P3): colonization projects in the rainforest
  lowlands with a trunk road, feeder roads every 4.4 km and lots of 250 m cleared from the roads
  to depths of their own (pasture, fresh burns, red soil, crops, regrowth), in the canopy slot.
- **Tropical river water** (tropical kit, design P0): white-water rivers in café au lait, black-
  water rivers (tea-black) in the very wet lowlands of a regional style channel's provinces,
  slot 8 over the core's river colour.
- **Known limits** sections in the desert, tropical, temperate, cold and agriculture kit docs.

- **Settlements avoid open water and like rivers.** Hamlet and village candidates whose centre
  lies in the sea, a lake or a level 0–1 river channel (pass A at the point) are dropped, water
  on a ring around a candidate lowers its score, a river within 2.5 km raises it (design §6.1
  water access); towns and larger keep the smooth-relief test (their candidates are visited
  over continents by the transport kit's trunk networks). Hamlets and villages flooded
  by a reservoir are drowned (not drawn), and a dam that would flood a town or a city is not
  built. Test `places_avoid_open_water` (0 of 362 centres in pass A's water).
- **Sites are backend-independent.** Airports, dams, ports, quarries, solar and wind farms and
  flares take close decisions (within 0.5 m of a threshold or of a competitor) with exact
  samples (`World::terrain`, the CPU backend's own) and keep exact heights, so GPU (f32) and CPU
  (f64) point samples build the same sites, and the transport kit the same network: new GPU test
  `sites_and_routes_match_the_cpu` (stamps and segments bit for bit at six places).
- **Settlement glow at low zooms.** Where a level's places are no longer listed (hamlets below
  z11, villages below z8, towns below z6, cities below z4), the settlements kit draws their
  expected built cover and street lights per pixel (`places::expected_places` × the place
  model's mean over a footprint, calibrated against the final places), so night views from
  altitude keep their lights and coarse pixels keep the finer zooms' mean.

- **Drainage: no straight trenches into inland basins.** A lattice point without a lower
  neighbour looked up to 3 cells farther for an outlet anywhere; inland that drew channels up to
  24 km long, straight across the higher ground between (canal-like water strips, several
  converging side by side on the next basin's lake; 7 level-1 and 42 level-2 such pieces within
  20 km of the 24 survey places). The farther search now only finds the sea (its original
  purpose): an inland point without a lower neighbour is a closed basin and holds its own
  closed-basin lake (lake share of land at the survey places 0.14 % → 0.37 %). CPU
  `World::flow_target` and WGSL `lat_target` alike.

- **Mountains kit** (`kits/mountains.rs`, `gpu/wgsl/kits/mountains{,_relief}.wgsl`,
  `docs/kits/mountains.md`): tower karst (a 3D Worley lattice of 420 m cells, oval towers 90-260 m
  over flattened floors, drowned to islands at the coast) and cockpit karst on carbonate (atlas
  lithology x warm humid / cooler climate); folded ridges (old orogens and foreland belts) and
  fault blocks (extensional provinces) as parallel ridges on an 8-corner lattice with per-corner
  strike, wavelength and phase (R-wave safe), bent and plunging by noise; creased badlands with
  banded strata on soft sediment x arid; alluvial fans at the fronts of dry mountains (family
  `mountains_fans`, pale radial apron); glacial U-troughs, cirques with tarns at the heads of
  glaciated tributaries, hanging tributaries, fjords (the floor of a glaciated coastal river goes
  to -70 .. -530 m and the ocean floods it, fading into the open sea) and stepped canyons with
  benches on arid sedimentary plateaus, all through a new **carve hook**. Core: `Kit::carve`
  (`kits::{Carve, CarveIn, has_carve, carve}`, WGSL `kits_carve`, `CarveIn`, `Carve`): the kit
  changes the valley profile of the drainage carve (floor, floodplain width, wall slope, depth
  cap, U profile, benches, a small lake at a channel's head); `RiverHit` gains `u`, `spring` and
  `reach`; the carve is bounded by the piece's reach (a no-op in the core). Pass B: banded
  badland and mesa strata, karst cliffs, glacial rock walls, fans, in slot 6 (the only stage with
  compile headroom).

- **Transport kit** (`kits/transport.rs`, `transport/{net,route,build,px}.rs`,
  `docs/kits/transport.md`): a host-built road, rail and power network over the settlements kit's
  places and sites — relative-neighbourhood graphs per network (motorways and trunk roads between
  cities, trunk roads between towns, secondary roads between villages, local roads to hamlets,
  farm and forest tracks, railways between cities and large towns, a high-voltage grid), spurs
  from towns to passing motorways, roads to airports, ports, dams, quarries, solar and wind farms,
  rail spurs and yards at ports, power lines from dams, solar and wind farms; routing on corridor
  grids over the analytic relief (refined per piece, smoothed, switchbacks on steep minor roads);
  engineered profiles from point samples (cut and fill with 1:2 slopes in the DSM, bridges with
  parapets and shadows, tunnels with portals); interchanges (cloverleaf, partial cloverleaf,
  diamond, trumpet), overpasses, roundabouts by culture and at town entries; per-pixel decks over
  rivers, reservoirs, lakes and the sea; motorway, trunk, secondary, local and track
  cross-sections with markings, cars and their shadows, lamps (thinned with their energy kept);
  railways with sleepers, rails, masts, trains, stations and yards; power lines with cleared
  corridors, pylons, wires and their shadows; parked cars in the settlements kit's parking lots
  and airport car parks. Core: `Kit::replaces_roads` / `kits::legacy_roads()` switch the
  noise-network roads off (pass A distances, the road pre-pass, WGSL `KITS_LEGACY_ROADS`); the
  farm tracks along region borders stay.

- **Transport routing** (`kits/transport/{route,build}.rs`, `docs/kits/transport.md`): routes go
  around water and sites and follow valleys. Each edge gets a scene along its corridor: the
  relief with the valleys of drainage levels 0–1 carved in (the core's valley profile), the sea,
  the core's lattice and closed-basin lakes, the cold kit's large kettle lakes, river crossings
  (a cost per crossing, up to ×3 when oblique); a second stage adds the settlements kit's sites
  near the route (airports, quarries, solar and wind farms as no-go footprints, reservoirs as
  water) and routes again where it touches one. Lakes are crossed only on bridges over narrow
  necks (500 m major, 160 m minor; reservoirs and the sea ×3), no more causeways; an edge that
  cannot go around is dropped; ends in water move to the shore. Roads end on a town's arterials
  (the settlements kit's district edges) instead of converging on its centre; power lines at its
  edge. New tests `routes_avoid_water_and_sites` (~400 routes over 20 regions: 101 points in
  water or sites before, 0 after) and `roads_end_at_town_arterials`; ignored `routing_time` and
  finders. Core: `World::relief_terrain`, `World::river_segments_of` (some levels only), host
  accessors of the lakes' levels.

- **Settlements kit** (`kits/settlements.rs`, `settlements/{places,town,sites}.rs`,
  `docs/kits/settlements.md`): a central-place hierarchy (hamlet, village, town, city,
  metropolis; host-side and analytic, `kits::settlements::places::settlements_near` for the
  transport kit) with culture-, climate- and terrain-driven density, sub-centres in
  conurbations; city morphology per pixel (zones, W2 districts with arterials, grid, radial,
  organic, curvilinear, superblock, informal and industrial patterns, hashed-BSP lots, houses,
  row houses, perimeter blocks, mid-rises, towers with podiums, warehouses, parks, plazas,
  parking lots, sports fields, cemeteries, stadiums, pools, street trees, cast shadows, zone-based
  night lights with energy-preserving lamp thinning); village forms by culture (nucleated,
  street villages, forest-strip Waldhufen villages with their strips, dispersed farms, kraals,
  perched hilltop villages; a plaza with church, mosque or temple); airports (runways by size
  along the wind, taxiways, apron, terminal, lights, graded plane), ports (container yards,
  cranes, piers, ships, breakwaters), dams and reservoirs on the drainage (upstream pieces,
  water level, bathtub ring), quarries and open-pit mines with tailings ponds, solar farms, wind
  farms, gas flares. Core: `Kit::replaces_towns` / `kits::legacy_towns()` switch the grid towns
  off (pass A site, host requests, WGSL `KITS_LEGACY_TOWNS`); pass B runs in four stages
  (`gpu::PB_STAGE_STEPS` `[0, 3, 5, 7]`).

- **Coast and ocean kit** (`kits/coast.rs`, `docs/kits/coast.md`): coast types per ~50 km
  stretch (atlas uplift, mountains, glaciation, sea-level temperature × noise); cliffs with
  wave-cut platforms and strata, skerries and archipelagos with kelp, tidal flats and salt marsh
  with dendritic creeks, barrier islands and lagoons, fringing and barrier reefs with coral
  heads, salt and aquaculture ponds, and an instance family of atolls with motus. All
  height-domain remaps near sea level in pass A (no distance to the coast); no core changes.
- **Low-zoom tiles in seconds again**: `configs/quick.yaml` (70 tiles, z0–z15) generated in
  187 s on an RTX 6000 Ada (2.7 s now) and 437 s on the CPU (33 s now, 6 threads). The
  instance lists of a zoom-0 block scanned millions of lattice cells per family (37 s per
  batch); a family is now left out of blocks whose finest pixel is at least its reach
  (`instances::sub_pixel`, both backends: its instances cover at most a pixel there). Bench
  stills: z5–z16 identical; z3–z4 deserts ≤ 0.2 % of the pixels change (≤ 0.013 % by more than
  8 DN); z0–z2 lose sub-pixel inselbergs, playas and drumlins, of which the lists kept 8
  arbitrary ones per block before. A batch whose land-use site requests outgrow their buffers
  grows them and repeats only the request passes, instead of halving the batch and generating
  it again (three halvings cost 110 s per run). z8–z16 throughput unchanged (`zoom_bench`).
- **Fewer spurious warnings**: "more than 8 instances of a family reach a 16-pixel block" only
  for blocks no wider than the family's cell (the kit's density problem, zoom ~8 and up; the
  desert playas and inselbergs and the cold kit's drumlins stay below it at the 13 bench
  places, z8–z16), and "ecoregion sites
  found only by a global search" not for tiles of zoom ≤ 2, whose corner discs do not bound
  them.
- **GPU stalls are actionable**: the error names the likely cause (another process using the
  GPU), `AERIALSYNTH_GPU` to pick another adapter, and the new `AERIALSYNTH_GPU_STALL_S` to
  wait longer than the default 120 s (`docs/gpu.md`, *Troubleshooting*).
- **Shader compile notice on Metal / DX12**: devices without a wgpu pipeline cache keep a
  marker file per adapter and shader version in `$XDG_CACHE_HOME/terrain/`, so later runs are
  announced as warm (a message only if the compile takes more than 5 s) instead of as a
  first-run compile every time.

- **First-run shader compile is announced, the pipeline cache is pruned**: the CLI, viewer
  (status bar), Python (`verbose=True`) and C (`as_set_verbose`) say when the GPU shaders
  are being compiled (~30 s once per GPU, driver and version) and how long it took; the cache
  file is keyed by adapter and shader hash, so stale versions are deleted instead of piling
  up (`terragen::gpu::set_compile_observer`, `docs/gpu.md`).
- **GPU compile time** (cold, RTX 6000 Ada, five kits): pass B 390 s -> three pipelines of
  17–23 s (pass B runs the stack in stages, compiled in parallel; one call site of the kernel
  library and of every kit layer instead of eleven), pass A1 16 s -> 12 s, point and drainage
  pipelines 18 / 13 s -> 15 / 10 s; start-up with empty caches ~30 s instead of ~6 min, and
  the volcanic kit no longer loses the device. Same tiles (52 bench stills: 50 identical, two
  with one pixel 1 DN off), same speed or faster (pass A 35 % faster: flat Worley loops; agriculture places z12–z16 ~10 %
  faster). Kits read
  instances in layers with the new `inst_count` / `inst_get` / `inst_list_bin`; the kit guide
  has compile-time rules and the `compile_times` / `inlined_sizes` measurements.
- **Volcanic kit** (`docs/kits/volcanic.md`): stratovolcanoes with craters, three tiers of radial
  gullies and rim wobble, calderas with flat floors and lakes, shield volcanoes, and cinder-cone
  fields with maars (three instance families placed by the atlas' volcanism, plate boundary and
  hotspot fields; volcanoes are pass-A relief, so the drainage follows the cones and oceanic
  ones make islands); ash and scoria on the upper cone and as an ash-fall apron, lava tongues
  from fresh black to weathered brown to moss-green, hydrothermal patches, crater and maar
  lakes at a flat water level, black-sand beaches, two volcanic-rock biomes with old lava fans.
- **Agriculture kit** (`docs/kits/agriculture.md`): field systems by culture x climate x terrain
  instead of one farmland look everywhere: 1-mile section grids with centre pivots and
  section-line roads, shelterbelt steppe, open-field strips (furlongs, reverse-S), bocage with
  hedgerows, smallholder mosaics with homestead compounds, rice paddies (flooded, green or ripe
  by season) and contour terraces, stone-walled dry terraces, vineyards, olive / almond / citrus /
  date groves, plantations on contours, greenhouse seas, paddocks with livestock trails and farm
  dams, irrigated pivots in deserts; each land-use region draws one system by its ecoregion's
  culture, climate and development.
- **Cold kit** (`docs/kits/cold.md`): polygon tundra, shrub tundra, polar desert and forest
  tundra; glaciers with firn, bare ice, crevasses, ogives and moraines by the summer
  temperature and aspect; thermokarst and kettle lakes, drained basins, pingos; drumlin fields
  with eskers (instance family); sea ice (fast ice, floes, leads, pressure ridges), frozen lakes
  and rivers; seasonal snow and nevé.
- **Temperate, boreal and wetland kit** (`docs/kits/temperate.md`): 12 biomes (deciduous,
  mixed and taiga forest, lichen woodland, maquis, garrigue, chaparral, feather-grass steppe,
  heath, moor, krummholz, alpine meadow); raised and string bogs, fens, reed marshes, beaver
  ponds; managed forest stands with clear-cuts, young plantations, logging roads, windthrow
  and firebreaks; autumn crowns and larches.
- **Tropical and savanna kit** (`docs/kits/tropical.md`): rainforest with emergent and
  flowering crowns, meander scars and oxbow lakes, mangroves, cloud forest and páramo zones,
  savannas with dry-season grass, burn scars, termite mounds, waterholes, gallery forests,
  baobabs and palms, monsoon, dry and miombo woodland, slash-and-burn frontiers.
- **Desert kit** (`docs/kits/desert.md`): sand seas whose dunes follow the atlas wind and its
  variability (transverse ridges, linear seif dunes with Y-junctions, star dunes, small dunes on
  the big ones), barchans on a stony fringe, playas with salt crust, cracks, brine pools and
  paleo-shorelines, arid lakes dried into salt pans, braided dry washes, yardangs, inselbergs;
  sand colours by palette and lithology; six desert biomes (erg, hamada, basalt harrat,
  gypsum, scrubland, Gobi) and the kit's textures on the two core deserts.
- **Lazy tile generation** (the bindings' cameras, `terrain run --lazy`):
  - The level of detail no longer over-refines a new place: tiles of unknown elevation range
    were refined on a default range (-100..6000 m), which put the camera inside their bounding
    spheres, so the first render generated the finest zoom around it (the C example: 2,900
    tiles to z18 for a 160 × 120 image whose pixels are ~14 m on the ground; > 6 GB, 13 min on
    4 CPU cores). The selection now generates the tiles whose range it needs, one zoom level
    per pass, and selects again until it knows them all (`lod::Selection::unknown`): 220 tiles
    to z15, 1 GB, ~3 min (the Python example: 1,900 tiles, 3.4 GB, 8.5 min before; 500
    tiles, 1.3 GB, 4.6 min now).
  - The same call gives the same image: the selection is the one a store holding every tile
    gives, so a render that generates its tiles and one that reads them select the same tiles
    (the first render differed by up to 1 DN before).
  - Missing tiles are generated in batches (`tiles.cache_tiles` bounds what the cache holds;
    before, one render's missing tiles were generated in a single batch).
  - Progress: `terrain run` prints the tiles a render generates; the bindings report
    generation on stderr with `as_set_verbose(w, 1)` (C) / `World(..., verbose=True)`,
    `World.verbose` (Python).
- **Planning:** the dry runs of `terrain run` / `terrain tiles` no longer add margin rings around
  tiles of unknown range (only around the tiles the renderer keeps), and the plan includes the
  tiles its selection refines: the previous `configs/quick.yaml` needed 357 + 280 tiles, now 357 + 2.
  A new cull drops tiles above the top of a downward view cone (the coarse tiles beside a low
  nadir camera, whose bounding spheres contain it).
- `configs/quick.yaml` is quick: 3 s at 5 fps, a 320 × 256 camera, zoom 15, margin 1 (~75 tiles).
- `terrain view` checks for a display and a GPU first ("terrain view needs a GPU (see
  docs/viewer.md): …") and creates no tile store when it cannot start.
- `terrain tiles --bbox` refuses lat_min >= lat_max (swapped bounds selected nothing) and
  out-of-range values; lon_min > lon_max stays a box across the antimeridian.
- A directory or a file that is not HDF5 given as a tile store fails with "… is a directory" /
  "… is not an HDF5 file" instead of an HDF5 error stack (CLI and bindings).
- Python: tile coordinates, zooms, seeds and sizes must be integers (`operator.index`: Python or
  numpy integers); `w.tile(3.5, 0, 0)` raises `TypeError` instead of truncating. The type stub
  `_native.pyi` covers the whole extension module.
- **Terrain generator version 4** (`docs/design/terrain-next.md`; stores of version 3 are
  refused for writing, as for any generator change):
  - **Composite stack:** pass B is a fixed stack of layers (zonal ground, altitudinal zone,
    azonal overrides, disturbance, agriculture, canopy, linear, built, water, seasonal), each a
    `Layer { cov, albedo, dh, cls, emit, lit, mat }`; the previous look is ported into it
    (`stack.rs`, `layers/`, `gpu/wgsl/stack.wgsl`).
  - **Biome registry:** biomes are YAML data (`crates/terragen/biomes/`): Köppen classes,
    climate envelopes, lithology, weights, palettes, vegetation parameters, crown layers,
    zonations and up to 8 kernel layers with masks over named fields. Compiled at start-up,
    validated with keyed errors, per-GSD-band layer lists; the resolved registry is stored in
    the world config (`biomes.resolved`), `world.biomes.overrides` changes weights.
  - **Kernel library** (`kernels.rs`, `gpu/wgsl/kernels.wgsl`): scatter, rows, cells, stripes,
    contours, radial, crescent, lobes, patches, linear, stamp, canopy, water and relief
    operators, each with a calibrated mean (coarse pixels show the mean of fine ones); a CPU /
    GPU parity harness for every kernel.
  - **Ecoregions and cultures** (`eco.rs`, `world.ecoregions`): ~100 km ecoregions pick their
    biome by the climate at their site and draw a style (lithology soils and rocks, tints, tree
    density and species mix, season); 12 culture archetypes set field systems, field size,
    hedges, roofs, building heights, blocks, lamps, agriculture and town density. Dithered
    ecotones (10 km).
  - **Planetary atlas wired in:** mountain ranges along plate boundaries
    (`relief.tectonic_mountains`; the noise belts become old orogens, `relief.belt_mountains`),
    climate from the atlas (rain shadows, currents, monsoons), biomes by Köppen class, cultures
    from the atlas' culture areas, town density by population, roads and lights by development.
    The atlas builds its elevation in two steps (pre-tectonic for the plates, then with their
    uplift); `ATLAS_VERSION` 3.
  - **Kits** (`src/kits/`, `docs/design/kit-guide.md`): biomes, kernels, layer functions,
    relief operators, **instance families** (sparse lattices listed per 16-pixel block on both
    backends) and **host-built linear features and stamps** (binned per tile), each kit in its
    own files and registered by one line.
  - Land cover: forest types (tropical, broadleaf, needleleaf, mixed), shrubs, hedgerows and
    tracks of the v2 classes.
  - Performance (RTX 6000 Ada, shared server, `examples/zoom_bench`, ms per tile, version 3 →
    4): z4 ~51 → ~63, z8 ~77 → ~87, z12 ~7 → ~9, z14 ~6 → ~8, z16 ~6 → ~7; pass B alone within
    +7 %, the rest pass A over more mountains and the atlas fields. CPU (6 threads): 0.86–0.98×
    of version 3's time. `TERRAGEN_PROFILE=passes` times each tile kernel.
  - Tests: tile parity asserted (< 0.1 % of pixels), kernel parity, LOD of albedo and
    land-cover groups, variety budget (`tests/variety.rs`), a CPU speed smoke test, and
    `wgsl_compiles` (every entry point to SPIR-V with naga, no GPU needed).

## 0.2.0 — 2026-10-10

- **`terrain show SEQ.h5`**: a viewer for sequence files (new crate `seqview`, `docs/show.md`).
  A timeline (play, pause, frame steps, speed, Space / arrow / Home / End keys) over the
  modalities of a camera in a grid: RGB, depth (turbo, auto or fixed range, log scale), optical
  flow (Middlebury wheel), land cover (class palette and legend), events (a window of ON / OFF on
  black or on the gray frame, or a time surface) and catalogue stars on the image. A side panel
  with the pose, IMU plots, the trajectory, display settings, the camera calibration and the
  scenario; the value of every panel under the pointer. Frames, event windows and IMU samples
  are read on demand in the background, so large files open at once; files with events-only
  cameras or without cameras work. `--snapshot out.png` renders the viewer without a window, on
  the CPU (no GPU or display).
- **`terrain export SEQ.h5`**: PNG sequences or videos (ffmpeg: `.mp4`, `.mkv`, `.mov`,
  `.webm`) of a camera's modalities, one per modality or `--side-by-side` with titles and
  legends, in the viewer's colours with ranges fixed over the export; `--start`, `--end`,
  `--every`, `--fps`, `--rate` (cameras without frames), `--window-ms`.
- **Viewer cold start:** on an empty store the first tiles are drawn after ~1 s instead of
  2 s (warm pipeline cache) or 137–157 s (cold), and the view's own tiles no longer wait for
  all 341 base tiles (a flight view's tiles: 51 s → 29 s on an RTX 2080 Ti).
  - GPU generator: pass B's pipeline compiles in ~25–30 s instead of ~150 s (the driver
    inlined the surface model at five call sites; now one, likewise the pixel-field fBm and
    the town shadow march). Tiles agree with before to 1 DN; no generator version change.
  - `Generator::prepare_gpu_in_background`: the viewer compiles the GPU generator on a
    background thread and generates on the CPU meanwhile.
  - Base levels: z0–z2 first (one level per batch), then the view's tiles, then z3..=base.
  - Snapshots report the first drawn tiles and the view's tiles in; `TERRAGEN_PROFILE=1` prints
    pipeline compile times and the viewer's batch times; `examples/base_levels` times z0..=Z.

- **Land-cover classes v2** (`terragen::landcover`, `docs/formats.md`): one table of 82 classes
  as data (id, name, group, legacy class, display colour, material). Ids 0–17 keep their
  meaning; the new classes (20–110: reservoirs, sea ice, glaciers, lava, forest types, savanna,
  rice paddies, vineyards, greenhouses, residential / industrial, motorways, runways, …) are
  reserved for the coming terrain kits (the generator still emits 0–17). 11 stable groups.
  - The shaders get the table as WGSL generated from it (`gpu/wgsl/classes.wgsl`, checked by a
    test): `LC_*` / `LG_*` constants, `lc_group`, `lc_material`, `lc_palette`.
  - Materials replace the water test in both renderers: glint weight, specular F0, roughness
    and self-emission per class (water renders as before).
  - The tile generator's majority vote of a pixel's class counts 128 classes (CPU and GPU).
  - Sequence files: `output.landcover: v2 | legacy | group` (default `v2`) chooses the values
    `landcover` datasets hold; new attributes `class_groups`, `class_legacy`, `class_mapping`
    next to `class_names`.
  - The viewer's and `terrain tiles --png`'s land-cover palettes come from the table.
- **`terrain survey`:** finds diverse places of a world quickly and renders stills of them.
  Random points on land and coasts (`--seed-places`) are classified with cheap point queries
  (class mix, elevation, relief, coast, river, town, climate); `--count` places are chosen by
  theme (coast, mountains, town, river, snow and ice, desert, forest, farmland, wetland,
  plateau, lake, tundra) and farthest-point sampling. Each place is rendered with the dataset
  renderer (GPU when available) in the `--views` (oblique from 1.5 km, nadir from 800 m,
  optionally high from 10 km, or custom), the sun at a fixed local time; tiles are generated
  into one store (`--tiles`). Outputs: stills, a labelled contact sheet `sheet.jpg` and
  `places.csv`; `--places FILE.csv` renders a fixed list again (regression stills).
  `docs/design/survey_places.csv` and `survey_places_seed2.csv` hold 24 places of worlds 1
  and 2 (the redesign's "before" baselines). ~20 s per place with two views on an RTX 6000 Ada.

- **Planetary atlas** (`terragen::atlas`, groundwork of the next generator; the generated world is
  unchanged): a deterministic cube map (6 × 512², ~20 km texels, 58 MB) of world-scale fields
  computed once per world: smooth elevation, signed coast distance, prevailing wind, annual
  precipitation from seasonal moisture advection with rain shadows and coastal deserts, sea-level
  temperature with ocean currents, seasonality, precipitation regime, tectonic plates (boundary
  kind, distance, closing speed, uplift, volcanism, hotspot tracks), lithology, glaciation,
  culture areas with archetypes, development and population potential. `Atlas::sample(dir)` and
  WGSL `atlas_sample(dir)` (bound to the GPU generator's kernels) read the same data; a Köppen
  classifier (`atlas::koppen`). Built in ~3–4 s on 8 threads, cached in the process and in
  `~/.cache/terrain/atlas-<key>.bin` (`TERRAGEN_ATLAS_CACHE`). New `world.atlas` settings
  (resolution, plates, hotspots, culture cell, rain shadow, currents, monsoon, precipitation,
  advection steps). Preview: `cargo run --release -p terragen --example atlas_preview -- OUT_DIR`.
  See `docs/simulation.md`.
- **Bindings** (`bindings/`): tile access from C and Python. Open a world's tile store (the
  world given like `terrain -c FILE --seed N`) and get a layer of tile z/x/y; missing tiles are
  generated (GPU if available, else CPU) and stored first.
  - C: `libaerialsynth` (shared and static) with the header `bindings/c/include/aerialsynth.h`
    (`as_open`, `as_tile`, `as_close`, layer descriptions, per-thread error messages).
  - Python: the `aerialsynth` package (maturin, one `cp310-abi3` wheel for CPython ≥ 3.10)
    returning numpy arrays; `World(tiles_file, config, seed).tile(z, x, y, layer)`.
- **Rendering in the bindings:** camera images of a world from a pose with the renderer of
  `terrain run` (CPU or GPU, `auto` by default), tiles in view generated into the store on
  demand. A camera is a pinhole (size, horizontal field of view, `forward` or `nadir` mount) or
  a camera of a scenario YAML (its intrinsics, mount, sensor and render settings). Pose: latitude,
  longitude, height above WGS84 and body roll / pitch / yaw (NED); time: UTC (sun, moon, stars).
  Outputs: RGB after the sensor model (deterministic noise), depth (z or range, inf = sky) and
  land cover (255 = sky), each only when asked for. Plus the surface height at a point.
  - C: `as_camera_pinhole`, `as_camera_open`, `as_render`, `as_camera_size`,
    `as_camera_backend`, `as_camera_close`, `as_surface_height`; `examples/render.c` writes a
    PPM.
  - Python: `World.camera(...)`, `Camera.render(lat, lon, height, roll, pitch, yaw, time=...,
    rgb=, depth=, landcover=)` returning a `Frame`, `World.surface_height(lat, lon)`; rendering
    releases the GIL.
- **Faster tile access in the bindings:**
  - An in-memory LRU cache of decoded tiles per world (default 256 MiB; tiles read or generated
    go into it): a cached tile is a copy (~0.02 ms) instead of a read and decompression
    (~0.8 ms for rgb). C `as_set_cache_mb`, Python `World(..., cache_mb=256)` /
    `World.cache_mb` / `World.cache_info()`; 0 turns it off.
  - Many tiles in one call: C `as_tiles(w, zxy, n, layer, out, out_len)`, Python
    `World.tiles(coords, layer) -> (n, 256, 256[, c])` array (GIL released). Coordinates are all
    checked first; stored tiles are decompressed in parallel (2–3× faster), missing ones
    generated together in batches of up to 64 and stored with one write per batch (2–3× the
    tiles per second of one call per tile); repeated tiles are made once.
  - Prefetch: C `as_prefetch`, Python `World.prefetch(bbox, zooms)` generate and store the
    missing tiles of a box over a zoom range (at most 10⁶ tiles).
  - `examples/tile.c` also reads a 2 × 2 block with `as_tiles`; `tile_bench` example of
    `aerialsynth-core` (timings in `bindings/README.md`).
- `render`: `LightingConfig::sun_at_utc`, the lighting at a UTC instant whatever the mode.
- **Platforms:** CI builds and tests on Linux, macOS (Apple silicon and Intel) and Windows
  (MSVC), including `cargo install --path crates/cli`. On Windows the C runtime is linked
  statically (no Visual C++ redistributable needed).
- **Release artifacts** (attached to each `v*` release):
  - Python wheels (`cp310-abi3`) for manylinux_2_28 x86_64 and aarch64, macOS arm64 and x86_64
    (macOS ≥ 11) and Windows x86_64;
  - `terrain-<version>-<target>` archives with the `terrain` CLI and the C library (shared and
    static, header, example) for the same platforms.

  All are self-contained: HDF5 and zlib are linked statically, only system libraries are needed.
- **GPU self-check:** with `render.backend: auto` the renderer first draws a 32 × 24 test frame
  on the GPU and on the CPU (once per process); a GPU that renders it clearly wrong is not used
  (a warning, then the CPU renders). Catches the virtual GPU of macOS VMs on Intel hosts, which
  renders nothing.
- `AERIALSYNTH_GPU=none`: no GPU (generation, rendering and the event sensor on the CPU).
- **Event sensor:** the luminance is clamped at 0 before its log (CPU and GPU): interpolated or
  flickering radiance below 0 gave NaN, which GPUs handle differently (on WARP and the macOS
  paravirtual GPU the GPU sensor gave half again as many events as the CPU).

- **GPU choice:** `AERIALSYNTH_GPU` selects the GPU (index, PCI bus id or part of the name;
  `none` for the CPU) for generation, rendering, the viewer and the bindings.
- **Robustness:**
  - Tile stores are created atomically, so a run killed at start no longer leaves a broken
    file; a store open in another process, an incomplete store or a half-written zoom level
    gives a one-line error instead of an HDF5 stack or a panic.
  - World configs, scenario rates and CLI arguments are validated, with the offending key
    named.
  - `terrain tiles --bbox` limits `--zooms` to `tiles.max_zoom` and refuses more than 4 million
    tiles; a deep zoom over a small box used to exhaust memory.
  - GPU device loss or out-of-memory is an error, not a panic; with `auto` a failed generator
    batch is redone on the CPU.

## 0.1.0 — 2026-10-09

First public release.

- **World generator** (`terragen`): a deterministic procedural planet on the WGS84 ellipsoid as
  256 × 256 Web-Mercator XYZ tiles (elevation, RGB, albedo, normals, land cover, night
  emission), generated on demand for any place and zoom, on the GPU (wgpu) or the CPU.
  Continents, eroded mountains, drainage-graph rivers and lakes, climate zones and biomes,
  dunes, glaciers, individual trees and forest stands, fields, farmsteads, towns with buildings,
  roads and street lamps, beaches and surf.
- **Tile store** (`tilestore`, `h5`): one HDF5 file per world, with compressed chunks, a
  crash-safe index and a record of the world it holds.
- **Renderer** (`render`): CPU and GPU (wgpu) backends with the same camera models (pinhole
  with radial-tangential or rational distortion, Kannala-Brandt fisheye, Mei, Scaramuzza), lighting for
  any date and time (sun, moon, stars and planets from a built-in ephemeris, atmosphere,
  shadows, night lights) and a sensor model.
- **Datasets** (`terrain run`): spline flights with wind, gusts, turbulence and engine vibration,
  or your own trajectory CSV; multi-camera rigs; ground truth (depth, optical flow, land cover,
  star positions); an event camera simulator and a synthetic IMU; all in one HDF5 sequence file.
- **Viewer** (`terrain view`): a map of the planet and the dataset camera flown live, with
  headless snapshots and keyframed recordings.
- **Tools**: builders of the bundled star catalogue and ephemeris (`scripts/`), and the
  showcase and long-haul flight videos (`showcase/`).
