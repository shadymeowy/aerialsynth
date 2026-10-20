# Changelog

## Unreleased

### Highlights

- **The terrain of v0.2.0 stays the default**, bit-identical: v0.2.0 tile stores open and grow
  as before. A second, **experimental** terrain generator, `next`, is opt-in
  (`world.generator: next`, [`docs/next/`](docs/next/README.md)).
- **Renderer:** wall materials on steep faces, Fresnel glints with reflections of the banks,
  forward-scattering snow; land-cover classes v2 in the datasets.
- **`terrain survey`:** find diverse places of a world and render stills of them.
- **Faster first renders** from C and Python, with progress reports; clearer GPU errors and
  first-run messages.

### Terrain generators

- `world.generator: classic | next` (default `classic`), `--generator` on `terrain run`,
  `tiles`, `view`, `survey` and `config`; `terrain info` shows a store's generator. A store
  records its generator (version 3 classic, 4 next) and refuses the other's tiles; v0.2.0
  stores, which have no `generator` key, open with `classic`.
- `classic` is v0.2.0's code (`crates/terragen/src/classic`), unchanged:
  `scripts/classic_v020_check.sh` compares 24 tiles (zooms 6–16 at four places) of the v0.2.0
  binary and of this one byte by byte, on the CPU and on the GPU.
- `next` (experimental; status, known limits and kits in [`docs/next/`](docs/next/README.md)):
  a planetary atlas (climate from winds and moisture advection, tectonic plates, lithology,
  cultures), ecoregions with biomes and cultures, biomes as data over a kernel library, and ten
  landscape kits (agriculture, coast, cold, desert, mountains, settlements, temperate,
  transport, tropical, volcanic). Its settings: `world.atlas`, `world.ecoregions`,
  `world.biomes`, `world.relief.tectonic_mountains` / `belt_mountains`.
- `next` closer to v0.2.0's look: larger, softer fields with more woodlots; one street pattern
  per town; irregular dune fields; deltas that open from the river; mixed paddies; denser taiga.
- `next` polish round 4: organic deltas (lobate, wandering distributaries, silt blended into the
  land; no pale wedge); tree lines along the streams of dry farmland; closed lowland taiga, no
  clear-cuts in empty land; volcano ash without the dark star of rays; darker, smaller cloud-forest
  crowns; fewer big survey sections in wet climates; denser suburbs; red canyon walls.
- `next` polish round 5: temperate broadleaf forests with v0.2.0's crowns and gaps; canyon walls
  with side canyons, spurs and thick strata; houses of v0.2.0's size; delta silt in the land's
  hue; mixed farms and mown hay in pasture regions; softer desert varnish; A/B places of the
  same landscape for every v0.2.0 scene.
- `next` polish round 6: even desert floors (soft varnish and mottling) and calm arid deltas;
  town centres of detached boxes on lawns, fewer street trees; smaller, more varied parcels
  in wet climates, hedged mixed farms; no fishbone clearings over farmland; greenhouse seas
  only in warm climates; snow on steep high-mountain slopes.
- `next` polish round 7, from an audit at 30 random land places (`scripts/ab_v020.py audit`):
  ecotones blend their palettes and styles (no camouflage of two biomes' soils), barchans only
  near hyper-arid ergs, soft steppe and savanna mosaics, feathered forest edges, muted cold
  colours (glacier ice, crevasses, polar soils, snow nets) and autumn crowns; dense town centres
  of small lots with varied roofs; less steep 5–7 km ranges; irregular canyon benches, narrow
  road cuttings, muted salt pans, hedged pastures, grey-brown boreal heath.
- `next` polish round 8, against the audit's worst places: glaciers without contour stripes and
  only in mountain valleys, no snow nets, open (not frozen) lakes in summer datasets and no
  snowed-in ecoregion borders, maars without pale halos; vegetation and kernel layers blend over
  ecotones (no forest/steppe patchwork); shrubs and textured soil in semi-arid land; no canals
  or ripe crescents on paddy terraces; dirt roads and desert washes in the ground's colour (no
  pale ribbons up canyon walls); calmer deltas, burn scars, savanna grass and shrub speckle;
  autumn colours only with frost; hedged core farmland; like-for-like volcano and boreal places.
- `next` polish round 9, against the reviewer's worst like-for-like scenes: canyon walls of strata at
  absolute elevations with spurs, gullies and a narrow floor (no layer cake), rivers of dry lands as a
  low-water channel between bars, patchy washes; no straight karst cliffs at lithology texel borders; calm
  flat lakes and ponds without islet speckle or pale halos; lava tongues instead of black discs, cinder
  cones with scoria aprons; boreal heath and lichen openings; sabkha on arid tidal flats; even desert floors;
  softer power corridors; cities whose grid bends, skyline fades and estates are grey (no ring city, no tower
  cake), trees on car parks, frayed town edges; a desert coast and a volcanic valley as like-for-like places.
- `next` polish round 10: eroded canyon walls (rills, rough rock, talus aprons) over a river that meanders
  with its canyon, no stream trenches or pale beds on the canyon plateaus, sunken faint dry beds, silty
  desert rivers; no comb of ribs along high-mountain valleys (gullies per pixel, not from the tile grid);
  glaciers without contour arcs; one flat level per thermokarst and kettle lake (no tilted sheets) and
  lighter dark lakes; grown-over cinder cones and lobate lava flows in humid fields; boreal openings on the
  hills, calmer taiga floors and steppe crowns; lower barchans; cleaner lake shores on flat plains; routes off
  high-mountain ice; organic old city cores with one ring boulevard, curvilinear suburbs, estates on the town
  grid with green buffers, frayed town edges.
- `next` polish round 11: industrial estates of matte halls (skylights, HVAC, loading aprons), yards
  and car parks with light industry beside them (no glassy sheets of roof); old city cores on the
  town's grid, bent, cut by lanes into small blocks of terraced houses and fading into the grid (no
  crisp patch of rubble), an old quarter off the CBD's centre in big cities; ribbons along main streets
  and scattered houses at town edges; snowbeds and late snow where summers are cold (snowfields on
  high mountains, glaciers under the season's snow); cinder cones kept on fault-block uplands; long
  narrow rough lava flows; plants over steep rainforest slopes and a lighter rainforest canopy;
  semi-arid tussocks and larger shrubs; fewer, sandier barchans; calmer barely resolved crowns.
- `next` polish round 12: old quarters of small blocks and single houses with an irregular outline
  fading into the town's grid (no stamped star), beside the CBD in big cities (no pale disc among the
  towers); grid streets that end at crossings and built edge lots (no empty stubs); a town's grid turned
  to its relief; red, deep, dissected badlands where warm (`canyon@type` now shows them); far canyon
  walls without hairy rills; capped valley carves that taper off (no scarps along forest streams);
  volcanic relief kept in big valleys; firn-filled gullies under perennial snow and snow on steeper
  slopes; polar moss and lichen over polygons with later snowbeds; fewer barchans; semi-arid ground
  in clumps; a lighter pass-B stage (light industry beside estates no longer re-evaluates the district).
- `next` polish round 13: no "bunch of small trees scattered" over open land: trees in groves, clumps
  and closed stands (a pixel-filtered clump density, `noise::clumped`), lone trees only in savannas,
  shrubs, tussocks, stones and spires drawn one by one only from ~4–10 pixels (their mean below),
  shaded gaps in closed canopies (isolated dark dots of the audit 2.24× classic's → 0.92×); a macro
  tone over 600 m – 1.5 km on the ground and in fields; less orange night lamps (38 % white LED);
  softer polygon troughs, shallower crevasses, small streams snowed over in winter; tower karst walls
  bare in streaks between vegetation; airports graded over 220 m and fading into the land, no tracks
  across them; drained mires in farmland; towers in six tints; the kit places of the forest-strip
  village and the rice paddies moved to where the world now has them.
- `next` polish round 14: whole buildings at district edges (a block's district, zone and street
  width decided at its centre; estates as whole groups of grid blocks; lots along an arterial or
  the ring boulevard built only where wholly clear of it, else a green verge; the old quarter by
  block: no lot split between districts, 185 of 2280 before); rice terraces along the hillside
  (plots on the benches with cross-bunds, stages in patches, no parcel rectangles; flooded paddies
  sky-coloured); `next`'s buildings at dusk and night: street light falling off with the height
  above the ground, sparse warm and cool lit windows varying by building, glass reflecting the
  ground below the horizon (no white towers at dusk, no grey ghosts at night; classic's renders
  unchanged); airfields culvert small streams; fewer, duller flowering crowns; tower glass in its
  roof's tint; a pixel's channel by its distance relative to the half width (a braided wash's band
  ended square on a tributary's bisector); fainter creeks on arid flats, near the sea only.
- **Optional tile layers** (tile store format 2; `next` only): `tiles.layers: [dtm, instance]`
  adds the bare ground and instance ids (building lots, fields, forest stands) to a new store;
  the viewer's map shadings *Height above ground* and *Instances* show them. Format-1 stores
  open as before.
- Rust API: `terragen::Generator` is the world's generator (classic or next, from the config);
  `terragen::NextGenerator`, `terragen::classic::Generator`, `GeneratorKind`,
  `Generator::ell()` (replaces `gen.world.ell`), `CLASSIC_GENERATOR_VERSION` (3;
  `GENERATOR_VERSION` 4 is next's).

### Rendering

- **Walls** (`render.walls`, on by default, both renderers): steep faces get a wall material
  instead of the top-down texture stretched down them: facades with floors, windows and lit
  windows at night, glass curtain walls, rock strata on cliffs, canopy sides; lit with the
  geometric normal. About +3–5 % per GPU frame.
- **Materials:** the sun glint is Fresnel-weighted and normalized; reflections show the banks
  where the surface rises above the reflected ray (narrow rivers in forest were silvery lines of
  horizon sky); snow and ice scatter forward at grazing views.
- **Land-cover classes v2** ([`docs/formats.md`](docs/formats.md#land-cover-classes)): one
  table of 82 classes with stable ids (0–17 keep their meaning), 11 groups, a legacy class,
  colour and material each. `output.landcover: v2 | legacy | group` chooses the values the
  `landcover` datasets hold, described by new attributes `class_groups`, `class_legacy`,
  `class_mapping`. The default generator emits classes 0–17, as before.
- Seasons: the season of a render is its date (documented; there is no `world.season`).

### Tools

- **`terrain survey`:** random points on land and coasts are classified with cheap point
  queries and `--count` places chosen by theme (coast, mountains, town, river, desert, …);
  each is rendered in `--views` (oblique, nadir, high, custom) into stills, a labelled contact
  sheet and `places.csv`; `--places FILE.csv` renders a list again (regression stills).
- Planning: the dry runs of `run` / `tiles` no longer add margins around tiles of unknown
  range, and tiles above a downward view cone are culled (the old `configs/quick.yaml` needed
  357 + 280 tiles, now 357 + 2). `configs/quick.yaml` is a 3 s smoke test (320 × 256, ~75
  tiles).
- `terrain view` checks for a display and a GPU first and creates no store when it cannot
  start.
- New example `examples/camera-footprint`: camera ↔ geodetic / ECEF / ENU ↔ XYZ tiles, a
  camera's ground footprint on the stitched map, the map warped into the camera.

### C and Python bindings

- **First renders at a new place:** the level-of-detail selection learns the elevation ranges
  it needs one zoom level at a time instead of refining unknown tiles to the finest zoom: the C
  example generates 220 tiles in ~3 min on 4 CPU cores (was 2,900 tiles, 13 min), the Python
  example 500 tiles in 4.6 min (was 1,900, 8.5 min). A render that generates its tiles now
  gives exactly the image of one that reads them. The same applies to `terrain run --lazy`.
- Progress on stderr: C `as_set_verbose`, Python `World(..., verbose=True)` / `World.verbose`.
- The terrain generator: Python `World(..., generator=)`, `World.generator`,
  `aerialsynth.GENERATORS`; C `as_open_generator`, `as_generator`.
- Optional layers: `AS_LAYER_DTM`, `AS_LAYER_INSTANCE`, `as_has_layer`, `World.layers`.
- Python: tile coordinates, zooms, seeds and sizes must be integers (`TypeError` instead of
  truncating floats); the type stub covers the whole extension.
- The release archives ship the documents the README links to.

### GPU

- The first-run shader compile is announced (CLI, viewer status bar, Python / C with verbose
  on) with its duration; the pipeline cache is keyed by GPU, driver and shader version, and
  stale files are deleted. Metal / DX12 keep a marker so later runs stay quiet.
- A stalled GPU fails with an actionable message (another process using it,
  `AERIALSYNTH_GPU`), and `AERIALSYNTH_GPU_STALL_S` sets the bound (default 120 s).
- `AERIALSYNTH_GPU_STRICT=1` makes a failed GPU tile batch an error instead of a switch to
  the CPU (development, CI).
- `next`: shorter first-run shader compile (Tesla P4 51 → 28 s, RTX 2080 Ti desktop 45 → 36 s):
  pass B in six pipelines that compile only their slots' kernels, point queries in three;
  pass B ~5 % faster.

### Fixes

- The `ground` example (terrain heights at points, used by `showcase/route.py`) always queried the next generator, so with the classic default it reported another world's heights (up to ~1.6 km off). It now uses the configured generator (`--generator classic|next`, default classic), and `route.py` uses the showcase world's seed and generator.
- Rock walls (`render.walls`) on terraced canyon faces drew vertical stripes (the wall colour was a nearest coarse texel, whose skewed grid made teeth): bilinear colour, soft strata keyed on elevation only, rare faint fractures, scree at the foot, and rock takes the wall material from ~40° of slope (before | after: `docs/images/render_walls_canyon.jpg`, `render_walls_alpine.jpg`).
- A GPU generator that fails to set up (device or pipeline compile) is an error with
  `backend: gpu` and under `AERIALSYNTH_GPU_STRICT=1`, as a failed batch was; `auto` still falls back to the CPU.
- Worlds set up at the same time share one pipeline compile (the cache is read under the compile lock).
- The atlas disk cache works on Windows (`%LOCALAPPDATA%`), like the pipeline cache.
- `terrain tiles --png` uses the configured `world.generator` and `tiles.generator`.
- A lazily generated tile that fails to generate is reported once instead of leaving a silent hole.
- `cargo check -p terragen --no-default-features` is warning-free (checked in CI).
- A classic world says once when the config sets `next`-only keys (`atlas`, `ecoregions`, `biomes`,
  `relief.tectonic_mountains`, `relief.belt_mountains`).
- On the DX12 GPU backend the renderer is built without the new wall shading (DX12's shader
  compilers miscompile it; Windows' WARP device rendered no terrain); a notice says so.
- The GPU renderer submits its G-buffer pass before the shading pass (software rasterizers such
  as lavapipe shaded an undrawn G-buffer); a shader the backend rejects is a start-up error.

- The shared GPU device could hang when several generators compiled at once (parallel tests,
  threads of the bindings), and crashed at exit on some drivers.
- A flaky GPU read-back panic (`TargetAlignmentGreaterAndInputNotAligned`).
- A directory or a file that is not HDF5 given as a tile store fails with that reason instead
  of an HDF5 error stack (CLI and bindings).
- `terrain tiles --bbox` rejects swapped latitudes and out-of-range values.
- An empty store left by `next` with optional layers (e.g. interrupted during the first shader
  compile) refused classic tiles ("layer dtm has 0 bytes"); it is now made again.
- `AERIALSYNTH_GPU` naming no adapter (or a GPU failing in strict mode) was silent with
  `backend: auto`; a warning now gives the reason and the adapter list.
- The pipeline cache works on Windows (`%LOCALAPPDATA%\terrain` without `HOME`; every run
  compiled cold) and ignores an empty `XDG_CACHE_HOME`.
- `terrain view --mode height|instance` on a store without that layer is an error instead of
  an empty map.

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
