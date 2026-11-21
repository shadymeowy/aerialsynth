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

The first run on a GPU compiles the generator's compute shaders: about 35 s for `next` on a
desktop (RTX 2080 Ti, 8 CPU threads: the driver compiles the pipelines in parallel on the CPU),
longer on laptops and integrated GPUs. `terrain` says so on stderr (`compiling GPU shaders
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
| `TERRAGEN_VRAM=1` | prints the GPU memory the generators and the renderer allocate ([below](#gpu-memory)); `=smi` adds the GPU's memory in use (`nvidia-smi`) at each step, `=peak` the live allocations by label at each new peak |

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

## GPU memory

A typical `terrain run` (generation and rendering in one process) fits a 4 GB GPU. Peak GPU
memory in use (`nvidia-smi`, the desktop's 97 MiB included) on an RTX 2080 Ti, for a
1280 × 720 frame at supersample 2 over a mountain valley from 800 m (944 + 27 tiles, z0–17):

| | 0.3.2 | 0.3.3 |
|---|---|---|
| next, `terrain run` (all steps) | 4011 MiB | 2630 MiB |
| next, `--step tiles` / `--step render` | 3305 / 2992 MiB | 2620 / 1710–1770 MiB |
| classic, `terrain run` | 3063 MiB | 1790 MiB |
| classic, `--step tiles` / `--step render` | 1910 / 2992 MiB | 1150–1190 / 1770–1850 MiB |
| next, `terrain run --lazy` / `--step render --lazy` from an empty store | 5547 / 5224 MiB | 3100 / 3067 MiB |
| classic, `terrain run --lazy` / `--step render --lazy` from an empty store | 4214 / 3702 MiB | 2100–2130 / 2119 MiB |
| `terrain view --snapshot` of the valley, next / classic (store empty) | 4654 / 3159 MiB | 3840 / 2580 MiB |
| `terrain view --camera-view` (960 × 540, a window), next / classic, 150 s from an empty store | 7517 / 6540 MiB | 3377 / 2044 MiB |
| the same, switching between map and camera every 25 s (the map's own peak) | | 3925 / 2357 MiB |

Generating while rendering (`tiles.lazy`, `terrain view`) puts the generator next to the
renderer on one device, with the next generator's kernels' local memory (below); on a GPU the
size of an RTX 2080 Ti that does not fit 2.8 GB with next. The generators free their batch
memory after each burst of tiles (`Generator::trim_gpu`), and `terrain run` frees the
renderer's tiles before generating the neighbourhood of lazily generated tiles
(`render::gpu::trim`).

Where it goes:

- **The generators' kernels:** the driver reserves local memory for every thread the GPU can
  run at once, sized by the kernel that needs the most: next's take ~16 KiB per thread, i.e.
  ~1.1 GB on an RTX 2080 Ti (68 SMs × 1024 threads), 640 MiB on a Tesla P4, ~270 MiB on a
  GTX 1650 Ti (16 SMs); classic's ~90 MiB. The largest is pass A1 with the kits' relief
  operators inlined (`TERRAGEN_VRAM=smi` creates the pipelines one at a time and prints the
  GPU's memory after each), then the point evaluation of the kits' operators and the drainage
  lattice's heights (~11 KiB). The driver keeps it until the device is gone,
  so `terrain run` renders on a new device after generating
  (`terragen::gpu::device::release_shared`).
- **Generation:** the drainage lattice (both generators: 144 MiB at first, made larger when a
  batch needs it, up to 576 MiB), pass B's work buffer (172 MiB), pass A's buffers (next: 24 MiB
  per tile of the largest batch so far, 388 MiB for 16 tiles) and per batch the drainage pieces
  of its queries (up to ~0.3 GB each), site requests and the tiles' output. Device memory comes
  in blocks of 32 MiB, larger buffers get their own.
- **Rendering:** the tile pool holds the frame's tiles plus a quarter (made again larger when a
  frame needs more, or when a frame evicts tiles the frame before used: up to twice the most
  tiles a frame needed, at most 2048 slots ≈ 2.2 GB); the scene above uses 814 tiles, a pool
  of 1024 slots (1.1 GB). Then the cached meshes (121 MB here, at most 512 MiB:
  `Renderer::mesh_budget`; the viewer's camera view keeps 128 MiB), the
  supersampled G-buffer and depth (20 B per sample: 70 MiB at 2560 × 1440) and the unit rays
  (16 B per sample).

- **The viewer** (`terrain view`): the map's tiles (1 MiB each; 1024 slots made by the map's
  first frame, growing up to `--gpu-tiles`, 1536) on eframe's device; the generator (batches of
  half the cores) and the camera view's renderer on the shared one. Only the view shown keeps
  its GPU memory: the map's tiles, meshes and image are freed while the camera view is shown
  (and loaded from the store again), the camera view's tile pool and meshes while the map is.
  A 960 × 540 camera view of the valley needs ~460 tiles (a pool of 512 slots, 0.55 GB) and
  ~110 MB of meshes.

`TERRAGEN_VRAM=1` prints the allocator's live and reserved memory after each large buffer and at
each step.

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
perspective-correctly like the CPU rasterizer. Tiles live in a pool of texture-array
slots (~1.1 MB each) sized to the frames' need ([GPU memory](#gpu-memory)), each uploaded once
and evicted least-recently-used; meshes likewise, up to 512 MiB (`Renderer::mesh_budget`). Positions on the pyramid are kept as
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
in batches of 16 tiles (next: ~1.1 GB of GPU memory at their peak besides the lattice); the
drainage network's lattice points live in a GPU hash table kept across batches (144 MiB, made
larger when a batch needs it, up to 576 MiB); only the site lists (lake levels, land-use
regions, towns) are made on the host from batched GPU point evaluations. The classic generator's
GPU kernels are v0.2.0's, unchanged: its tiles are bit-identical to v0.2.0's on the same GPU. (The
experimental next generator has its own: [`next/`](next/README.md#how-it-works).)

| (RTX 2080 Ti; CPU: 8 threads) | CPU | GPU |
|---|---|---|
| 64 tiles at z15 / z13 (cold caches) | 12.7 s / 13.5 s | 1.1 s (60 tiles/s) |
| `terrain view` snapshot from an empty store (592 tiles: the globe at z0–z4 and the view's tiles) | 323 s | 33 s |
| base levels z0–z4 (341 tiles; terragen's `base_levels` example) | ~620 s | 25 s |

Low-zoom tiles are the most work per tile on both backends: a Mercator pixel of z3 at 80° is
3.4 km, so lakes and land-use regions switch on over thousands of kilometres, and every lake's
level takes 11 terrain evaluations with the finest drainage network.
