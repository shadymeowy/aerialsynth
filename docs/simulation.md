# What is simulated

## Terrain (`crates/terragen`)

Generated on the GPU when it supports 64-bit shaders, else on the CPU (`tiles.generator`; both
build the same world, [`gpu.md`](gpu.md#tile-generation)). The generator is v0.2.0's, unchanged
(`world.generator: classic`, generator version 3): v0.2.0 tile stores open and grow with it.
(`world.generator: next` selects the experimental modular generator, [`next/`](next/README.md).)

- **Determinism:** a pure function of (seed, config, position), using f64 3D noise on the
  ellipsoid surface. There are no seams or pole problems.
- **Band limiting:** fields are band-limited by the pixel's ground sample distance (GSD), so
  coarse zooms approximate the average of fine ones. Large-scale fields are sampled on a 16-px
  grid aligned to tile corners; whether a field comes from the grid or per pixel depends on the
  zoom only, so neighbouring tiles always take the same path.
- **Landforms:**
  - continents and shelves, home-region land bias
  - mountain belts (ridged multifractal with domain warp) carved by dendritic erosion gullies
  - hills with varying roughness, plateaus, arid mesas, dunes in sand seas
- **Water:**
  - rivers as a downhill flow graph with 3 levels (rivers, tributaries, streams; see
    `world.hydro.levels`): dendritic, always draining downhill, meandering
  - valleys, floodplains and riparian woods; wet or dry beds
  - lakes filled to their spill level, oceans with shallows and surf, beaches
- **Climate:** latitude, lapse rate, subtropical (Hadley-cell) dryness, coast and noise. It
  drives biomes: snow, rock, tundra, boreal, temperate and tropical forests, steppe, savanna,
  desert, wetlands.
- **Vegetation:** individual tree crowns in 3 layers plus shrubs, with explicit crowns and
  canopy heights in the DSM. They are prefiltered when unresolved and cast shadows in the baked
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
- **Land cover:** the first 18 classes (0–17) of the land-cover table
  ([`formats.md`](formats.md#land-cover-classes)).

## Rendering (`crates/render`)

The CPU reference renderer and the GPU renderer ([`gpu.md`](gpu.md)) share the pipeline:
`render.backend: auto` (default) takes the GPU when there is one.

- **Geometry:**
  - tile meshes built on the fly from the elevation layer; quadtree level of detail by
    projected texel size (`render.texel_px`), with skirts against cracks
  - camodocal-style camera models: `pinhole`, `pinhole_full` (OpenCV rational),
    `kannala_brandt`, `mei`, `scaramuzza` ([`scenario.md`](scenario.md#cameras-and-modalities));
    fields of view beyond 180° (range-based depth buffer)
  - f64 transforms and an exact per-vertex camera model (incl. distortion)
- **Shading:**
  - supersampling; cameras with depth / flow / land cover need an odd factor (the ground truth
    is the central sub-sample, which is the pixel centre only for odd factors)
  - trilinear plus anisotropic pyramid texture filtering
  - per-class materials: Fresnel sun glint and reflections of the sky (or of the banks where the
    surface rises above the reflected ray) on water, ice, glass and solar panels, forward
    scattering on snow (`render.water_glint`)
  - wall materials on steep faces: facades with floors and lit windows, rock strata on cliffs,
    canopy sides (`render.walls`)
- **Lighting:**
  - `relit` (albedo + normals + DSM ray-marched cast shadows) or `satellite` (baked imagery)
  - sun fixed or from date/time (NOAA), continuous twilight; moon position and phase with
    moonlight and the lunar disc
  - real stars (Hipparcos + Tycho-2, V ≤ 9), planets and the Moon (JPL DE440, 1990–2060) at
    their apparent positions, with star ground truth ([`stars.md`](stars.md))
  - night lights from the generated emission layer (street lamps, porch lights, farmsteads,
    lit main roads near towns, plazas and industry), light-pollution glow in the haze
  - aerial perspective and sky
  - lamp flicker (`render.lighting.flicker`): 2× mains frequency, three supply phases, a share
    of LED lamps; frames integrate it over their exposure and event cameras resolve it
    ([`events.md`](events.md#lamp-flicker))
- **Sensor** (per camera, `rgb.sensor`):
  - auto exposure as a 1st-order ODE, split into exposure time and gain
  - motion blur from sub-frame poses of the high-rate trajectory
  - defocus, chromatic aberration, vignetting, bloom, starburst spikes on bright lights
  - shot, read and PRNU noise, then tone curve
- **Event cameras:** ESIM-style re-rendering with a DVS pixel model ([`events.md`](events.md)).
- **IMU** (`imu:`):
  - truth (specific force, inertial angular rate) computed inside the flight simulator from its
    1 kHz positions and attitudes (second difference + Coriolis − WGS84 normal gravity; attitude
    difference + Earth rate), so it is consistent with the poses by construction
  - each IMU sample is the mean over its window [t − dt/2, t + dt/2] (no delay)
  - the sensor model adds extrinsics/lever arm, misalignment, scale factor, turn-on bias, bias
    random walk, white noise (Kalibr-style densities) and saturation
  - error-free values and biases are stored as ground truth
- **Trajectories:**
  - arc-length spline path (line, circle, figure8, lawnmower, random, waypoints), C1 between
    its dense samples, or a trajectory CSV ([`scenario.md`](scenario.md#conventions))
  - AGL terrain following, crab angle into the crosswind
  - the path is laid out in the tangent plane at its origin; heading, velocity and IMU truth are
    expressed in the local NED frame along it (meridian convergence included)
  - Dryden-like turbulence and 1-cosine gusts acting on position/attitude through small ODEs
  - engine-harmonic and band-limited broadband vibration
  - optional stabilized gimbal (the whole sensor platform: `/pose` is then the gimbal frame)
- **Seasons:** the season of a render is its date (`render.lighting.mode: clock`: the sun's
  path, day length, twilight, when the night lights come on). The ground is the world's, one
  look per world (late summer): a winter date gives winter light over the summer ground.

## Performance

On the author's machine (Xeon W-2125, 8 threads; RTX 2080 Ti):

| task | speed |
|------|-------|
| tile generation | GPU: ~60 tiles/s at z13–z15; CPU: ~5–7 tiles/s at z15–z17, ~1.5 at z6–z10 |
| rendering 640×360, 2×2 supersampled, shadows | ~1.3 s per frame on the CPU, ~40× faster on the GPU |
| events (VGA, engine vibration) | ~100 s of CPU per simulated second; the GPU is 11–15× faster |
| `terrain view` camera | 640×360 at ~40 frames/s, 1280×720 at ~14 |

`terrain run --lazy` (and the bindings' cameras) generates exactly the tiles each view needs
while rendering: the level-of-detail selection generates the tiles whose elevation range it
needs, one zoom level per pass, until it knows them all, so a view selects the same tiles
whether they were stored before or not.

## Tests

```
cargo test --release
```

Unit and integration tests cover geodesy (against pymap3d), the HDF5 wrapper and tile store
(round trips, refusal of other worlds and generators, v0.2.0 stores), camera models and
trajectory formats, level of detail, the sensor, scenarios, events, the IMU (noise statistics,
truth against the 1 kHz simulator), solar position, the atlas, generator invariants
(determinism, seamless tile borders, parent ≈ mean of its children, for both generators), GPU
against CPU (noise, pass A, whole tiles z7–z16, renderer, event sensor), the flight camera of
the viewer, and the bindings (the C API, its header and examples; the Python package by
`bindings/python/tests` against a built wheel).

`scripts/classic_v020_check.sh V020_TERRAIN` (not part of `cargo test`: it needs a v0.2.0
build) generates 24 tiles (zooms 6–16 at four places) on the CPU and on the GPU with the v0.2.0
binary and with this checkout and compares them byte by byte.
