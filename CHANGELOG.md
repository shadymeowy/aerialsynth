# Changelog

## Unreleased

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
