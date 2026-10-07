# GPU backend (wgpu, headless)

`render.backend: gpu` renders frames with wgpu (Vulkan / Metal / DX12) instead of the CPU
reference renderer. It is headless (no window, no surface) and plugs into the same pipeline:
`Renderer::render` returns the same `FrameOut` (radiance, depth, 3D points, land cover, lamp
flicker split), so the sensor model, motion blur, events, IMU and writers are unchanged.

```yaml
render:
  backend: gpu   # default: cpu
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
