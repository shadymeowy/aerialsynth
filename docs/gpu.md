# The GPU

One headless wgpu device (Vulkan, Metal or DX12) generates the tiles, renders the frames and
runs the event sensor; the CPU does all three too, more slowly, and is the reference.

```yaml
tiles:
  generator: auto   # tile generation: auto (default) | gpu | cpu
render:
  backend: auto     # rendering and the event sensor: auto (default) | gpu | cpu
```

- **Tile generation** (`tiles.generator`; not to be confused with `world.generator`, the terrain
  generator) runs on the GPU when it has 64-bit float and integer shaders with 64-bit atomics
  (`SHADER_F64`, `SHADER_INT64`, `SHADER_INT64_ATOMIC_ALL_OPS`: Vulkan on NVIDIA and recent AMD;
  not Metal) and binds 256 MiB storage buffers, else on the CPU. Both build the same world
  ([below](#tile-generation)).
- **Rendering** (`render.backend`) takes any GPU. `auto` also renders on the CPU a camera the GPU
  cannot take (a Scaramuzza model with more than 10 `inv_poly` coefficients, or a supersampled
  image beyond the GPU's texture or buffer limits); with `gpu` that is an error when the
  scenario loads. Both backends return the same `FrameOut`, so the sensor model, motion blur,
  events, IMU and writers are shared; frames match the CPU reference to ~0.1 %.
- With `auto`, no usable GPU means the CPU, and a GPU tile batch that fails switches
  generation to the CPU with a one-line warning; `gpu` makes either an error. `terrain view`
  needs a GPU.

## First run

The first run on a GPU compiles the generator's compute shaders: about 30 s on a fast desktop
GPU, longer on laptops and integrated GPUs. `terrain` says so on stderr (`compiling GPU shaders
for <GPU> ...`, then `GPU shaders ready in N s`), the viewer in its status bar, Python / C with
`verbose` on. The viewer generates on the CPU meanwhile.

The compiled pipelines are cached per GPU, driver and aerialsynth version in
`$XDG_CACHE_HOME/terrain/` (default `~/.cache/terrain/`; on Windows without `HOME`,
`%LOCALAPPDATA%\terrain\`): later runs start in about a second, and
files of older versions are deleted automatically. Delete the directory to clear the cache. On
Metal and DX12, which keep their own driver caches, a marker file there records that this GPU
compiled these shaders, and later runs say nothing unless the compile takes more than 5 s.
Library users can watch the compile with `terragen::gpu::set_compile_observer` (silent by
default; `stderr_observer()` gives the CLI's messages).

## Choosing a GPU, troubleshooting

| environment variable | |
|---|---|
| `AERIALSYNTH_GPU` | the GPU to use (CLI, viewer, C and Python): an index, a PCI bus id (`0000:83:00.0`) or part of the adapter name (`6000`); `none` hides every GPU (everything on the CPU). Unset: the first high-performance adapter. A value that matches no single GPU is an error listing the adapters (`AERIALSYNTH_GPU=list`) |
| `AERIALSYNTH_GPU_STRICT=1` | a failed GPU tile batch, or a GPU generator that fails to set up (device, pipeline compile), is an error instead of a switch to the CPU (development, CI) |
| `AERIALSYNTH_GPU_STALL_S` | seconds without a completed read-back before the GPU counts as stalled (default 120) |
| `TERRAGEN_PROFILE=1` | prints each pipeline's compile time and each generated batch's time |

- **No suitable GPU:** many integrated GPUs, software Vulkan and Metal lack the 64-bit shaders:
  tiles are generated on the CPU while frames may still render on the GPU.
- **A GPU that renders wrongly:** with `render.backend: auto` the renderer first draws a
  32 × 24 test frame on the GPU and on the CPU (once per process, 15–30 ms). A GPU whose frame
  clearly differs is not used (`render.backend auto: the GPU (…) renders a test frame wrongly
  (…); rendering on the CPU`); seen on the virtual GPU of macOS VMs on Intel hosts.
  `backend: gpu` skips the check.
- **No walls on DX12 (Windows):** the GPU renderer's shading kernel is built without the wall
  shading on DX12 (both of its shader compilers, FXC and DXC, fail on it), with a one-line
  notice; steep faces then take the top-down texture. For walls on Windows use the CPU renderer
  (`render.backend: cpu`) or a Vulkan adapter (`AERIALSYNTH_GPU`).
- **"the GPU stalled (no read-back completed in 120 s)":** the GPU finished no work for two
  minutes, usually because another process (a training job) is using it heavily. Pick another
  adapter with `AERIALSYNTH_GPU`, or wait longer with `AERIALSYNTH_GPU_STALL_S`. A driver that
  stopped executing the queue never recovers, so without the bound the run would hang.
- **A GPU process hangs at start over SSH:** with X forwarding (`DISPLAY` set to a remote
  display), some NVIDIA Vulkan drivers block while creating the device. Run headless work with
  `env -u DISPLAY terrain …` (the same for Python and C programs).

## Rendering

| stage | where |
|-------|-------|
| level-of-detail selection, tile fetch and lazy generation | CPU |
| mesh geometry per tile unit (f64 ECEF vertices) | CPU, once per unit; cached on the GPU relative to a per-unit origin |
| projection through the camera model (all models, with distortion) | GPU vertex shader (`mesh.wgsl`); per frame only per-unit offsets are uploaded |
| rasterization into a supersampled G-buffer (unit, u, v, range) | GPU |
| deferred shading: texture filtering, relighting, DSM shadows, lamps and flicker, materials, walls, sky, aerial perspective | GPU compute (`shade.wgsl`, a port of the CPU renderer) |
| read-back, 3D points from depth and rays | CPU |

The vertex shader uses the CPU mesh builder's validity rules and interpolates u, v and range
perspective-correctly like the CPU rasterizer. Tiles live in a fixed pool of texture-array
slots (`POOL_SLOTS`: 2048 ≈ 2.2 GB, or the adapter's limit), each uploaded once and evicted
least-recently-used; meshes likewise (`MESH_BUDGET`). Positions on the pyramid are kept as
(zoom, integer global pixel, fraction) per axis, since global pixels reach 2^27 at zoom 19;
shading runs in f32, geometry in f64 on the CPU.

`cargo run --release -p render --example gpu_compare -- SCENARIO.yaml [t] [camera_index] [out.png]`
renders one frame with both backends and writes CPU | GPU | difference side by side.

| frame (RTX 2080 Ti; Xeon W-2125, 8 threads; warm caches) | CPU | GPU |
|-------|-----|-----|
| 1920×1080, supersample 2 | 5.0 s | 0.12 s |
| 640×360, supersample 2 | 1.3 s | 0.03 s |

## Event cameras

With a GPU the event simulation renders its keyframes there and runs the pixel model of the
sensor there too (photoreceptor low-pass, high-pass, leak, threshold crossings with refractory
period and shot noise; `gpu/events.wgsl`), with the pixel state resident on the GPU; only the
events come back. Hot pixels, timestamp jitter, sorting and the rate controller stay on the
CPU, and the fixed pattern is the CPU sensor's, so both backends simulate the same sensor. The
random numbers differ: the events are statistically equivalent, not bit-identical, and
deterministic.

On a 1 s, 640×360 test flight the GPU gives 2,441,653 events against 2,440,883 on the CPU
(+0.03 %; rate and spatial histograms correlate at 1.0000); the sensor steps take 3.4 s instead
of 37 s and the keyframe renders 10 s instead of 150 s (`cargo test -p render gpu::events`
checks the moving edge to within 1 µs).

## Tile generation

The GPU generator is the CPU generator ported to WGSL and builds the same world: the same
hashes (u64 in the shader), the CPU's octave frames and gradient tables, and f64 noise lattice
coordinates. Tiles agree with the CPU's to f32 precision (at most 0.1 % of the pixels differ by
more than 2 DN or 5 cm; `cargo test -p terragen gpu::tests`), so both backends write the same
generator version and can share a store. Everything per pixel and per point runs on the GPU,
in batches of 16 tiles (~25 MB of GPU memory each); the drainage network's lattice points live
in a GPU hash table (~0.5 GB) kept across batches; only the site lists (lake levels, land-use
regions, towns) are made on the host from batched GPU point evaluations. The classic generator's
GPU code is v0.2.0's, unchanged: its tiles are bit-identical to v0.2.0's on the same GPU. (The
experimental next generator has its own: [`next/`](next/README.md#how-it-works).)

| (RTX 2080 Ti; CPU: 8 threads) | CPU | GPU |
|---|---|---|
| 64 tiles at z15 / z13 (cold caches) | 12.7 s / 13.5 s | 1.1 s (60 tiles/s) |
| `terrain view` snapshot from an empty store (592 tiles: the globe at z0–z4 and the view's tiles) | 323 s | 33 s |
| base levels z0–z4 (341 tiles; terragen's `base_levels` example) | ~620 s | 25 s |

Low-zoom tiles are the most work per tile on both backends: a Mercator pixel of z3 at 80° is
3.4 km, so lakes and land-use regions switch on over thousands of kilometres, and every lake's
level takes 11 terrain evaluations with the finest drainage network.
