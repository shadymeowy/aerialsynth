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
* **Agreement:** tiles agree with the CPU's to f32 precision. On test tiles from z7 to z16, at
  most 0.03% of the pixels differ by more than 2 DN or 5 cm (`cargo test -p terragen
  gpu::tests`, asserted below 0.1%), so both write generator version 4 and can share a store.
  Every kernel of the library is compared on 20,000 random inputs (`kernel_parity`: no sample
  off); its parameters are rounded to f32 on both sides, and noise wavelengths derived from
  them are computed in f64 (an f32 wavelength shifts an ECEF-domain noise by ~|p|/λ · 6e-8
  cells).
* **Same discrete decisions:** what is decided per site is computed by shared host Rust for
  both backends: ecoregions (biome, style, culture: `eco::compute` from the exact lattice site,
  which the host finds from the ids the GPU reports), land-use regions, towns, host-built
  features (`features.rs`). Kernel instances, instance families and ecotone patches are integer
  hashes of lattice coordinates.
* **Split of the work:** everything per pixel and per point runs on the GPU: macro fields (with
  the atlas sampled at the grid nodes), the coarse grid, relief (with the kits' relief
  operators and instance families), river carving, lakes, land use, the surface — the
  composite stack (`stack.wgsl`, the core layers, the biome registry's kernel layers from
  `registry.wgsl` / `kernels.wgsl`, the kits' WGSL and the generated glue) — the canopy opening
  and the output layers.
  * **Drainage network** (`drain.wgsl`): the jittered lattice points of every level live in a
    GPU hash table kept across batches (8M points, ~0.5 GB, emptied when 60% full). The table
    holds their heights, flow targets (steepest descent) and sources; each query's channel
    pieces are gathered from it in the CPU's lattice order.
  * **Host** (`gpu/host.rs`): only site lists remain: lake levels (minimum over the rim), sink
    lakes, land-use regions, towns and their overlaps, ecoregions, the kits' linear features
    and stamps. Their inputs come from batched GPU point evaluations, cached across batches.
    The GPU reports which lakes, regions, town cells and ecoregions a batch needs.
  * **Tables** (world-constant, group 0): the atlas (binding 5) and the biome registry —
    biomes, crown layers, zones, kernel layers with their calibrated means, per-GSD-band layer
    lists (bindings 6–11). Per batch (group 1): ecoregions, features and their bins.
* **Batches:** 16 tiles per batch (~26 MB of GPU memory per tile). Kernels: grid nodes →
  instance lists per 16-px bin → relief (and lake requests) → drainage pieces per bin → rest of
  pass A (and site requests: regions, town cells, ecoregions) → pass B (adaptive supersampling)
  → canopy opening → output layers. `TERRAGEN_PROFILE=passes` times each kernel on its own.
* **Startup:** the compiled pipelines are kept in `$XDG_CACHE_HOME/terrain/` (default
  `~/.cache/terrain/`; the driver's own shader cache is per executable); after the first run the
  generator is ready in ~0.1 s. The first compilation (new install, new driver, changed shaders)
  takes ~25–30 s, set by pass B (`TERRAGEN_PROFILE=1` prints each pipeline's time). The driver
  inlines every call, so the shaders call the big functions (the surface model, the pixel-field
  fBm, a town) from one site each: with five call sites of the surface model pass B took
  ~150 s. `terrain view` compiles in the background and generates on the CPU meanwhile
  (`Generator::prepare_gpu_in_background`).

| | CPU (8 threads) | GPU (RTX 2080 Ti) |
|---|---|---|
| 64 tiles at z15 / z13 (cold caches) | 12.7 s / 13.5 s (5 tiles/s) | 1.1 s (60 tiles/s) |
| the 0.2.0 `configs/quick.yaml` (10 s, 640 × 512): planned tiles (357, z0–z17) | 57.6 s | 5.7 s |
| its completion (280 tiles: margins around coarse tiles of unknown range, which the dry runs no longer add: now 2 tiles) | 148 s | 13 s |
| `terrain view` snapshot from an empty store (592 tiles: the whole globe at z0–z4 and the view's tiles) | 323 s | 33 s |
| base levels z0–z4 (341 tiles; `examples/base_levels`, loaded machine) | ~620 s (z3–z4 ~0.5 tiles/s) | 25 s (z3–z4 9–14 tiles/s) |
| `terrain view`, empty store: first tiles drawn (warm / cold pipeline cache) | z0 in 0.4–0.8 s | ~1 s / ~1 s (was 2 s / 137–157 s) |
| `terrain view --view fly:39.9,32.8,600,0,-12`, empty store: the view's 1321 tiles in | | 29 s (was 51 s: the base levels went first) |

On the RTX 6000 Ada of a shared server, cold pipeline cache, globe snapshot from an empty store:
first tiles 157 s → 0.7 s, the view's tiles 172 s → 29 s, all base levels 188 s → 45 s.

Generator version 4 (the composite stack, registry, ecoregions, atlas-driven relief and
climate) against version 3 on the RTX 6000 Ada of a shared server (`examples/zoom_bench`, 52
tiles per zoom at 13 places, interleaved runs; see the table in `CHANGELOG.md`): pass B costs
the same (the stack with its layers is as fast as the hand-written surface model); pass A
costs more where the world now has more mountains (the plate-boundary ranges) and carries the
atlas fields. Low zooms are dominated by pass A and the drainage, so the per-band layer lists
do not make them faster yet.

Low-zoom tiles are the most work per tile on both backends (`TERRAGEN_PROFILE=1`: a batch of
four polar z3/z4 tiles takes 0.3–0.8 s on the GPU, ~0.6 s of it for 30,000–40,000 land-use
regions, each needing a point evaluation with the finest drainage network), for two reasons:
* **Polar pixels:** a Mercator pixel of z3 at 80° is 3.4 km, so lakes and land-use regions
  switch on over thousands of kilometres.
* **Lake levels:** every lake's level takes 11 terrain evaluations with the finest drainage
  network around it.

## Troubleshooting

- **Several GPUs:** `AERIALSYNTH_GPU` picks the one to use, for the CLI, the viewer and the C /
  Python bindings: an index, a PCI bus id (`0000:83:00.0`) or part of the adapter name
  (`AERIALSYNTH_GPU=6000`). Without it the first high-performance adapter is used. A value
  that matches no single GPU is an error that lists the adapters (`AERIALSYNTH_GPU=list`).
- **No GPU at all:** `AERIALSYNTH_GPU=none` hides every GPU: tile generation, rendering and the
  event sensor run on the CPU (`backend: gpu` and the viewer then fail). For a GPU whose driver
  misbehaves, or to compare against the CPU reference.
- **A GPU that renders wrongly:** with `render.backend: auto` the renderer first draws a 32 × 24
  test frame of a synthetic tile on the GPU and on the CPU (once per process: 15–30 ms after
  the GPU renderer's start-up, ~0.3 s on Windows' software WARP device). A GPU whose frame
  clearly differs (no terrain, wrong depth, land cover or brightness) is not used: a one-line
  warning (`render.backend auto: the GPU (…) renders a test frame wrongly (…); rendering on the
  CPU`) and the CPU renders. Seen on the virtual GPU of macOS VMs on Intel hosts ("Apple
  Paravirtual device"), which renders nothing. `backend: gpu` skips the check.

- **No suitable GPU:** with `backend: auto` (the default) generation and rendering fall back to
  the CPU; `backend: gpu` makes a missing GPU an error instead. Tile generation
  needs 64-bit float and integer shaders and 256 MiB storage buffers; adapters without them
  (many integrated GPUs, software Vulkan, Metal on macOS) use the CPU generator, while
  rendering may still run on the GPU.
- **A GPU process hangs at start over SSH:** with X forwarding (`DISPLAY` set to a remote
  display), some NVIDIA Vulkan drivers block while creating the device. Run headless work with
  `env -u DISPLAY terrain …` (the same for Python and C programs).
