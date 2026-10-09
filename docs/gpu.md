# GPU backend (wgpu, headless)

The GPU generates the tiles (below, [Tile generation](#tile-generation)) and renders the
frames, on one shared headless device.

`render.backend: gpu` renders frames with wgpu (Vulkan / Metal / DX12) instead of the CPU
reference renderer. The default, `auto`, takes the GPU when there is one and the CPU otherwise;
it also renders a camera the GPU cannot take on the CPU (a Scaramuzza model with more than 10
`inv_poly` coefficients, or a supersampled image beyond the GPU's texture or buffer size
limits), where `gpu` is an error when the scenario is loaded. It is headless (no window, no
surface) and plugs into the same pipeline:
`Renderer::render` returns the same `FrameOut` (radiance, depth, 3D points, land cover, lamp
flicker split), so the sensor model, motion blur, events, IMU and writers are unchanged.

```yaml
render:
  backend: gpu   # auto (default) | gpu | cpu
```

Build: the `gpu` feature of the `render` and `terragen` crates (on by default).

## Split of the work

| stage | where | why |
|-------|-------|-----|
| LOD unit selection, tile set (units, neighbours, ancestors) | CPU | tiny; shared code |
| tile fetch / lazy generation | CPU (`TileCache`); missing tiles from the tile generator (GPU or CPU, below) | |
| mesh geometry per unit and vertex stride: f64 ECEF vertices (shared `unit_geometry`) | CPU, once; cached on the GPU relative to a per-unit origin | camera-independent |
| projection through the camera model (pinhole ± distortion, rational, Kannala–Brandt, Mei, Scaramuzza) | GPU vertex shader (`mesh.wgsl`) | per frame only a per-unit offset (f64 on the CPU) is uploaded |
| rasterization into a supersampled G-buffer (unit, u, v, range) | GPU render pass | |
| deferred shading: pyramid texture filtering, relighting, DSM shadows, lamps + flicker, sky, aerial perspective | GPU compute pass (`shade.wgsl`, a port of `raster.rs` / `atmo.rs`) | the expensive part |
| read-back, 3D points from depth and rays | CPU | |

A unit's mesh is fixed by the unit, its vertex stride (from the camera distance, as on the CPU)
and which neighbour tiles exist; it is built once and cached on the GPU (LRU within
`gpu::MESH_BUDGET`), with one shared index buffer per grid size. The vertex shader projects with
the same validity rules as the CPU mesh builder (triangles touching a non-imageable vertex are
dropped) and sets `clip.w = range`, so u, v and range are interpolated perspective-correctly with
weights 1/range, exactly like the CPU rasterizer.

## Performance

On the author's machine (RTX 2080 Ti; CPU: Xeon W-2125, 8 threads), warm caches:

| frame | CPU (8 threads) | GPU |
|-------|-----------------|-----|
| 1920×1080, supersample 2 | 5.0 s | 0.12 s |
| 640×360, supersample 2 | 1.3 s | 0.03 s |

Over a flight each tile and each mesh crosses the bus once; per frame only small tables, the
per-unit offsets and the read-back (radiance, depth, land cover) move.

## Event cameras

With `backend: gpu` the event simulation renders its keyframes on the GPU and also runs the
pixel model of the event sensor there (`gpu/events.rs`, `gpu/events.wgsl`, a port of
`EventSensor::step`): photoreceptor low-pass, high-pass, leak, threshold crossings with the
refractory period and shot noise, with the pixel state resident on the GPU.

* Each keyframe (radiance and, with flickering lamps, the cos / sin split) is uploaded once. The
  sensor steps between two keyframes are interpolated on the GPU and run in one submission;
  only the events come back (12 bytes each).
* Events are appended with an atomic counter; on overflow the batch is rerun from a copy of the
  pixel state with a larger buffer.
* Hot pixels, timestamp jitter, sorting and the rate controller stay on the CPU and share the
  CPU sensor's code. The fixed pattern (per-pixel thresholds, leak rates) and hot pixels are
  the CPU sensor's, so both backends simulate the same sensor.
* Random numbers come from a per-(step, pixel) hash: statistically equivalent to the CPU
  streams, not bit-identical. The output is deterministic (events are put in row-major order
  before the time sort).

Validation (`cargo test -p render gpu::events`): the deterministic moving edge gives the CPU's
events to within 1 µs. Flicker interpolation, noise rates and low-light bandwidth match the CPU
sensor.

On a 1 s, 640×360 test flight, the GPU path gives 2,441,653 events against 2,440,883 on the CPU
(+0.03%), with correlation 1.0000 in both the 10 ms rate histogram and the spatial histogram.
The sensor steps take 3.4 s instead of 37 s, and the keyframe renders 10 s instead of 150 s.

## Tile cache on the GPU

* A fixed pool of slots (`gpu::POOL_SLOTS`: 2048 ≈ 2.2 GB, or the adapter's texture-array limit): texture arrays with one layer per
  slot for colour (albedo or satellite RGB, sRGB-decoded by the hardware), normals, emission,
  elevation, land cover and 16×16 block maxima of the elevation (shadow-ray skipping).
* A tile is uploaded once and stays resident; when the pool is full the least recently used tile
  not needed by the current frame is evicted. Over a flight, each tile crosses the bus once.
* Per frame, an open-addressing table (storage buffer) maps the frame's tile ids (the same set
  the CPU renderer uses) to their slots, so the shader's neighbour and ancestor fallbacks behave
  exactly as on the CPU.

## Precision

Global pixel coordinates reach 2^27 at zoom 19, beyond f32. The shader keeps positions on the
pyramid as (zoom, integer global pixel, fraction) per axis. 64-bit hashes (lamp flicker cells,
stars) are emulated with pairs of u32. Shading runs in f32; geometry stays f64 on the CPU.

## Validation

`cargo run --release -p render --example gpu_compare -- SCENARIO.yaml [t] [camera_index] [out.png]`
renders one frame with both backends, prints the radiance / depth / land-cover differences and
writes CPU | GPU | |difference| side by side.

## Tile generation

`tiles.generator: auto` (default) generates tiles on the GPU when it has 64-bit float and
integer shaders with 64-bit atomics (`SHADER_F64`, `SHADER_INT64`,
`SHADER_INT64_ATOMIC_ALL_OPS`: Vulkan on NVIDIA and recent AMD), else on the CPU;
`gpu` / `cpu` force one. The GPU generator (`terragen::gpu`) is the CPU generator ported to WGSL
and builds the same world:

* **Same noise:** the hashes are the CPU's (u64 in the shader), the octave frames and the
  gradient table are uploaded from the CPU's own tables, and noise lattice coordinates are f64
  (ECEF metres over wavelengths down to decimetres; only `+ − × floor` run in f64, everything
  inside a lattice cell in f32).
* **Agreement:** tiles agree with the CPU's to f32 precision. On test tiles from z3 to z16, at
  most 0.01% of the pixels differ by more than 2 DN or 5 cm (`cargo test -p terragen
  gpu::tests`), so both write generator version 3 and can share a store.
* **Split of the work:** everything per pixel and per point runs on the GPU: macro fields, the
  coarse grid, relief, river carving, lakes, land use, the surface with its trees, fields, towns
  and lights, the canopy opening and the output layers.
  * **Drainage network** (`drain.wgsl`): the jittered lattice points of every level live in a
    GPU hash table kept across batches (8M points, ~0.5 GB, emptied when 60% full). The table
    holds their heights, flow targets (steepest descent) and sources; each query's channel
    pieces are gathered from it in the CPU's lattice order.
  * **Host** (`gpu/host.rs`): only site lists remain: lake levels (minimum over the rim), sink
    lakes, land-use regions, towns and their overlaps. Their inputs come from batched GPU point
    evaluations, cached across batches. The GPU reports which lakes, regions and town cells a
    batch needs.
* **Batches:** 16 tiles per batch (~25 MB of GPU memory per tile). Kernels: grid nodes → relief
  (and lake requests) → drainage pieces per 16-px bin → rest of pass A (and site requests) →
  pass B (adaptive supersampling) → canopy opening → output layers.
* **Startup:** the compiled pipelines are kept in `$XDG_CACHE_HOME/terrain/` (default
  `~/.cache/terrain/`; the driver's own shader cache is per executable); after the first run the generator is ready in ~0.1 s.

| | CPU (8 threads) | GPU (RTX 2080 Ti) |
|---|---|---|
| 64 tiles at z15 / z13 (cold caches) | 12.7 s / 13.5 s (5 tiles/s) | 1.1 s (60 tiles/s) |
| `configs/quick.yaml` planned tiles (357, z0–z17) | 57.6 s | 5.7 s |
| its completion (280 tiles, mostly z1–z8 near the poles) | 148 s | 13 s |
| `terrain view` snapshot from an empty store (592 tiles, the whole globe at z3–z4) | 323 s | 33 s |

Low-zoom tiles are the most work per tile on both backends, for two reasons:
* **Polar pixels:** a Mercator pixel of z3 at 80° is 3.4 km, so lakes and land-use regions
  switch on over thousands of kilometres.
* **Lake levels:** every lake's level takes 11 terrain evaluations with the finest drainage
  network around it.
