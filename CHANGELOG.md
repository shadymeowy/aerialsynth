# Changelog

## Unreleased

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
