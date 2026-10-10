# What is simulated

## Terrain (`crates/terragen`)

Generated on the GPU when it supports 64-bit shaders, else on the CPU (`tiles.generator`; both
build the same world, see `docs/gpu.md`).

- **Determinism:** a pure function of (seed, config, position), using f64 3D noise on the
  ellipsoid surface. There are no seams or pole problems.
- **Band limiting:** fields are band-limited by the pixel's ground sample distance (GSD), so coarse zooms approximate the
  average of fine ones. Large-scale fields are sampled on a 16-px grid aligned to tile corners.
  Whether a field comes from the grid or per pixel depends on the zoom only, so neighbouring
  tiles always take the same path.
- **Landforms:**
  - continents and shelves, home-region land bias
  - mountain belts (ridged multifractal with domain warp) carved by dendritic erosion gullies
  - hills with varying roughness, plateaus, arid mesas, dunes in sand seas
- **Water:**
  - rivers as a downhill flow graph with 3 levels (rivers, tributaries, streams; see
    `world.hydro.levels`): dendritic, always draining downhill, meandering
  - valleys, floodplains and riparian woods; wet or dry beds
  - lakes filled to their spill level, oceans with shallows and surf, beaches
- **Climate:** latitude, lapse rate, subtropical (Hadley-cell) dryness, coast and noise. It drives biomes: snow, rock,
  tundra, boreal, temperate and tropical forests, steppe, savanna, desert, wetlands.
- **Vegetation:** individual tree crowns in 3 layers plus shrubs, with explicit crowns and canopy
  heights in the DSM. They are prefiltered when unresolved and cast shadows in the baked
  imagery.
- **Farmland:**
  - per-region field systems: grid, irregular, strips, centre pivots
  - seasonal crop palettes, crop rows, tramlines, headlands, wet/bare patches
  - hedges and tracks along field edges
- **Farmsteads:** house, barn, gravel yard and yard lamp.
- **Settlements:**
  - towns and villages with street grids (organic in old centres)
  - lots, pitched or flat roofs, building heights in the DSM, parks and industry
  - street lamps in the emission layer
- **Roads:** curvy inter-town roads and region-border tracks.

### Planetary atlas

World-scale fields that cannot be computed locally are precomputed once per world
(`terragen::atlas`, design: `docs/design/terrain-next.md` §3.6). The generator does not use them
yet: both backends can sample them (CPU `Atlas::sample`, WGSL `atlas_sample` in
`gpu/wgsl/atlas.wgsl`, bound by the GPU generator), and the next generator will drive its
climate, biomes, landforms and cultures from them.

- **Geometry:** a cube map with a tangent warp (texels within ±30 % of a uniform size), 6 ×
  512² texels (`world.atlas.resolution`; ~20 km), a 2-texel apron per face. Continuous fields are
  read with a cubic B-spline (smooth, no overshoot); within half a texel of a face edge the
  faces meeting there are blended, so there are no seams. Discrete fields come from the
  nearest texel.
- **Fields** (16-bit each, 36 bytes per texel, 58 MB at 512):

| field | unit | how |
|---|---|---|
| smooth elevation | m | `World::smooth_relief`, band-limited at the texel size |
| coast distance | km, > 0 on land | jump flooding over the cube map from the texels next to the other class |
| prevailing wind (east, north) | m/s | trades, westerlies and polar easterlies shifted with the seasons (more over land), a stream function of pressure perturbations (3000 km), monsoon flow towards heated continents in summer and away in winter; annual mean of the two seasons |
| annual precipitation | mm/yr | moisture advected along the wind (semi-Lagrangian, half resolution, 100 steps per season, July and January): evaporation from the sea towards the saturation of the sea air, recycling over land, rain by a latitude rate (ITCZ, subtropical highs, storm tracks), summer convection over land, saturation of the air at the ground's elevation and orographic rain-out on ascent (on a ~100 km smoothed relief), less rain in descending air: **rain shadows** |
| annual mean temperature at sea level | °C | latitude (`world.climate`) and the existing temperature noise, cold currents and upwelling on subtropical west coasts (**coastal deserts**: cooler, more stable air), warm currents on subtropical east coasts and high-latitude west coasts, colder continental interiors at high latitudes |
| seasonality (warmest − coldest month) | °C | latitude × continentality (the maritime influence carried inland by the wind) |
| precipitation regime | −1 dry summers … +1 summer rains | (summer − winter) / (summer + winter) of the two seasons: Mediterranean climates, monsoons |
| plates, boundary distance, kind and closing speed | id, km, mm/yr | weighted, warped spherical Voronoi diagram of `world.atlas.plates` plates with Euler-pole velocities; boundary kind from the closing speed across the boundary and the crust on either side: overriding / subducting (ocean–continent), collision, island arc, rift, ridge, transform |
| uplift, volcanism | −1..1, 0..1 | relief and volcanism potential of the boundary kind by distance (coastal range and arc inland of subduction, wide collision belts, rift grabens and shoulders, ridges); hotspot tracks (`world.atlas.hotspots`) on the plates moving over them |
| lithology | dominant, secondary, share | sedimentary, carbonate, crystalline, volcanic, unconsolidated, from the tectonics, elevation, glaciation, aridity and noise |
| glaciation | 0..1 | ice at the last glacial maximum (~6 °C colder, more towards the poles): sheets and mountain glaciers |
| culture, archetype | id, 1 of 12 | warped Voronoi diagram of ~`world.atlas.culture_cell_km` areas; the archetype is drawn from the Köppen class of the area's land |
| development | 0..1 | per culture, ± regional noise and climate comfort |
| population potential | 0..1 | flatness × water (coast, rain) × climate comfort × soil fertility (lithology) |

- **Köppen classes:** `atlas::koppen(&sample, elevation)` gives the Köppen–Geiger class from the
  annual temperature (lowered by the lapse rate), the seasonality, the annual precipitation and
  the regime (monthly values modelled as sinusoids).
- **Determinism:** f64 on the CPU; every texel is a pure function of the config or the result of
  iterations that read only the previous buffer, so the bytes do not depend on the thread count.
- **Cost and caching:** ~3–4 s on 8 threads (shared 32-core server; mostly the elevation, the
  advection and the jump flooding). Kept in the process for the last 4 worlds and on disk in
  `$XDG_CACHE_HOME/terrain/atlas-<key>.bin` (default `~/.cache/terrain`; 58 MB each; written via a
  temporary file and a rename). `TERRAGEN_ATLAS_CACHE=DIR` moves the disk cache, `=off`
  disables it. The key hashes the settings the atlas depends on (seed, planet, home, continents,
  relief, climate, atlas) and its format version.
- **Settings** (`world.atlas`): `resolution` (512), `plates` (16), `hotspots` (6),
  `culture_cell_km` (1000), `rain_shadow` (1), `currents` (true), `monsoon` (1),
  `precipitation` (1, a multiplier), `advection_steps` (100). Preview them with
  `cargo run --release -p terragen --example atlas_preview -- OUT_DIR [SEED] [CONFIG.yaml]`
  (equirectangular PNGs of every field and a Köppen map).

## Rendering (`crates/render`)

- **Geometry:**
  - CPU reference rasterizer; tile meshes are built on the fly from the elevation layer
  - generic `CameraModel` trait with camodocal-style models, configured by the camera YAML
    schema (`model`, `width`, `height`, `intrinsics`, `distortion`, `xi`, `max_fov_deg`,
    `inv_poly`, `affine`, `center`; `docs/scenario.md`):
    - `pinhole` and `pinhole_full` (OpenCV rational)
    - `kannala_brandt`
    - `mei`
    - `scaramuzza`
  - fields of view beyond 180° are supported (range-based depth buffer)
  - quadtree LOD by projected texel size, with skirts against cracks
  - f64 transforms and an exact per-vertex camera model (incl. distortion)
- **Shading:**
  - supersampling; cameras with depth / flow / land cover need an odd factor (the GT sample is
    the central sub-sample, which is the pixel centre only for odd factors)
  - trilinear plus anisotropic pyramid texture filtering
- **Lighting:**
  - `relit` (albedo + normals + DSM ray-marched cast shadows) or `satellite` (baked imagery)
  - sun fixed or from date/time (NOAA), continuous twilight
  - moon position and phase with moonlight and the lunar disc
  - real stars (Hipparcos + Tycho-2, V ≤ 9), planets and the Moon (JPL DE440, 1990–2060) at
    their apparent positions for the date, time and place (≤ 0.01″ vs Skyfield), radiometric
    brightness and colour, planet discs, trails over the exposure, per-frame ground truth; see
    `docs/stars.md`
  - `render.backend`: `auto` (default: the GPU when there is one and it supports the cameras),
    `gpu` (headless wgpu renderer and event sensor: frames match the CPU reference to ~0.1 %,
    events are statistically equivalent, not bit-identical; see `docs/gpu.md`) or `cpu`
  - night lights from the generated emission layer: street lamps (sodium/LED), porch lights,
    farmsteads, lit main roads near towns, plazas and industry
  - light-pollution glow in the haze
  - aerial perspective, sky and water glint
- **Sensor** (per camera, `rgb.sensor`):
  - auto exposure as a 1st-order ODE, split into exposure time and gain
  - motion blur from sub-frame poses of the high-rate trajectory
  - defocus, chromatic aberration, vignetting, bloom, starburst spikes on bright lights
  - shot, read and PRNU noise, then tone curve
- **IMU** (`imu:`):
  - truth (specific force, inertial angular rate) computed inside the flight simulator from its
    1 kHz positions and attitudes (second difference + Coriolis − WGS84 normal gravity; attitude
    difference + Earth rate), so it is consistent with the poses by construction, and stored
    as trajectory columns
  - each IMU sample is the mean over its window [t − dt/2, t + dt/2] (no delay)
  - the sensor model adds extrinsics/lever arm, misalignment, scale factor, turn-on bias, bias
    random walk, white noise (Kalibr-style densities) and saturation
  - error-free values and biases are stored as GT
- **Lamp flicker** (`render.lighting.flicker`): lamps flicker at 2× mains frequency, with three
  supply phases and a share of LED lamps. Frames integrate it over their exposure; the event
  camera resolves it (the renderer returns the flicker as cos / sin images, so flicker steps
  need no extra renders).
- **Trajectories:**
  - arc-length spline path (line, circle, figure8, lawnmower, random, waypoints), C1 between
    its dense samples
  - AGL terrain following, crab angle into the crosswind
  - the path is laid out in the tangent plane at its origin; heading, velocity and IMU truth are
    expressed in the local NED frame along it (meridian convergence included)
  - Dryden-like turbulence and 1-cosine gusts acting on position/attitude through small ODEs
  - engine-harmonic and band-limited broadband vibration
  - optional stabilized gimbal (the whole sensor platform: `/pose` is then the gimbal frame)

## Performance

On the author's machine (Xeon W-2125, 8 threads; RTX 2080 Ti):

| task | speed |
|------|-------|
| generation | GPU (default, `docs/gpu.md`): ~60 tiles/s at z13–z15; CPU: ~7 tiles/s at z17, ~6 at z16, ~5 at z15, ~1.5 at z6–10 (8 threads, `world.tile_supersample` 2) |
| rendering 640×360, 2×2 supersampled, shadows | ~1.3 s per frame on the CPU, ~40× faster on the GPU ([`gpu.md`](gpu.md)) |

`terrain run --lazy` generates exactly the tiles each view needs while rendering: the
level-of-detail selection generates the tiles whose elevation range decides it, one zoom level
per pass, and selects again until it knows them all (so a view selects the same tiles whether
they were stored before or not), then the neighbours and ancestors of the selected tiles. The
bindings' cameras do the same. The live
camera view (`terrain view`) renders 640×360 at ~40 frames/s, 1280×720 at ~14 (`docs/viewer.md`).

Event simulation renders every step that moves the image by `max_px_per_step` (~100 s of
compute per simulated second at VGA with engine vibration); see `docs/events.md`.

## Tests

```
cargo test --release
```

- **Geodesy:** checked against pymap3d.
- **h5:** round trips, raw chunks, concurrency, error stacks.
- **Tile store:** round trip, rejection of invalid tiles and of tiles of another world.
- **Camera models and trajectories:** includes NED/ECEF CSV formats.
- **LOD and dynamics.**
- **Sensor:** auto-exposure ODE.
- **Scenario:** omitted modalities are off, overlapping HDF5 groups are rejected.
- **Events:** moving edge, background activity, low-light bandwidth.
- **IMU:** noise statistics, level-flight truth, lever arm and sample timing against the
  1 kHz simulator truth.
- **Solar position** (and the lighting at a UTC instant).
- **Planetary atlas:** f16 packing, the same bytes on 1, 3 and 8 threads, no seams across cube
  faces, the disk cache (round trip, other worlds and damaged files refused), climate belts,
  windward slopes wetter than leeward ones, Köppen classes, the build time, WGSL sampling
  against the CPU's.
- **Generator invariants:** determinism, seamless east-west and north-south tile borders, parent ≈
  mean of its children (on the default backend: the GPU).
- **GPU generator:** noise, pass A (the macro-scale world model: relief, drainage, sites) and
  whole tiles (z7–z16) against the CPU generator.
- **Flight camera:** level flight holds height and heading, banked turns at g·tan(bank)/V, the
  ground stops the camera.
- **Bindings** (`bindings/`): tiles generated once then read, stores of other worlds refused,
  bad arguments; rendering on the CPU (image shapes, depth and land cover, sky, time of day,
  determinism, mounts, scenario cameras, the store kept open by cameras) and GPU against CPU;
  the C API, its header and its examples (`tile.c`, `render.c`); the Python package by
  `bindings/python/tests` against a built wheel.
