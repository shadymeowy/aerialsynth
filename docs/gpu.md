# GPU backend (wgpu, headless)

`render.backend: gpu` renders frames with wgpu (Vulkan / Metal / DX12) instead of the CPU
reference renderer. The default, `auto`, takes the GPU when there is one and the CPU otherwise. It is headless (no window, no surface) and plugs into the same pipeline:
`Renderer::render` returns the same `FrameOut` (radiance, depth, 3D points, land cover, lamp
flicker split), so the sensor model, motion blur, events, IMU and writers are unchanged.

```yaml
render:
  backend: gpu   # auto (default) | gpu | cpu
```

Build: the `gpu` feature of the `render` crate (on by default).

## Split of the work

| stage | where | why |
|-------|-------|-----|
| LOD unit selection, tile set (units, neighbours, ancestors) | CPU | tiny; shared code |
| tile fetch / lazy generation | CPU (`TileCache`) | the generator is CPU code |
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

## Performance (RTX 2080 Ti, warm caches)

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

`cargo run --release -p render --example gpu_compare SCENARIO.yaml [t] [camera] [out.png]`
renders one frame with both backends, prints the radiance / depth / land-cover differences and
writes CPU | GPU | |difference| side by side.
