# Bindings: tiles and camera images from C and Python

Two things: **a layer of tile z/x/y of a world**, and **a camera image of the world from a pose**.
A world's tiles live in its HDF5 tile store ([`docs/formats.md`](../docs/formats.md)); a tile that
is not stored yet is generated (on the GPU when it has 64-bit shaders, else on the CPU, see
[`docs/gpu.md`](../docs/gpu.md)), written to the store and returned. Cameras render with the
renderer of `terrain run` ([rendering](#rendering)), generating the tiles in view the same way.
Flights, IMU, event cameras and sequence files are not exposed (use `terrain run`).

```
bindings/core     aerialsynth-core: the implementation, shared by both bindings (Rust)
bindings/c        aerialsynth-capi: libaerialsynth (.so / .dylib / .dll, .a / .lib) and include/aerialsynth.h
bindings/python   the aerialsynth Python package (maturin; extension aerialsynth._native)
```

Both bindings call the same Rust code (`aerialsynth-core`), so they behave identically.

## The world

A world is given like the CLI's `terrain -c CONFIG --seed N`:

- **config:** the path of a scenario YAML (its `world:` section is the world, its
  `tiles.max_zoom` the zoom limit; the other sections are ignored), or of a bare world config
  (the contents of a `world:` section), or none for the default world;
- **seed:** overrides the config's seed.

A tile store holds exactly one world. Opening a store made by another world (or by another
generator version) fails, naming the settings that differ; a missing store is created (with its
directory), and an empty one is stamped with the world.

## Tiles

Web-Mercator XYZ tiles of 256 × 256 pixels: `z` at most the world's max zoom (`tiles.max_zoom`,
default 18), `x, y < 2^z`, `y = 0` at the north. Pixels are row-major, row 0 the north edge,
pixel-centre registered, channels last, little-endian:

| layer | type | channels | |
|---|---|---|---|
| rgb | u8 | 3 | satellite look (fixed sun, haze), sRGB |
| albedo | u8 | 3 | unlit surface colour, sRGB encoded |
| elevation | f32 | 1 | DSM (ground, canopy, buildings, water surface), metres above the WGS84 ellipsoid, at pixel centres |
| normal | i8 | 3 | unit surface normal (east, north, up) × 127 |
| landcover | u8 | 1 | class id: 0 unknown, 1 ocean, 2 lake, 3 river, 4 beach, 5 sand, 6 rock, 7 snow, 8 grass, 9 shrub, 10 forest, 11 crop, 12 building, 13 road, 14 wetland, 15 tundra, 16 bare, 17 urban |
| emission | u8 | 3 | night-time artificial light, linear radiance = 16 (v/255)³ |

A generated tile is stored with all its layers, so the other layers of it are read, not
generated. Tiles are deterministic: the same world gives the same bytes.

**Many tiles at once** (`as_tiles` / `World.tiles`): every coordinate is checked before any
work (one out of range fails the call and nothing is read or generated); cached tiles are
copied, stored ones are read and decompressed in parallel, and the missing ones are generated
together, in batches of up to 64 tiles (the GPU generator dispatches many tiles at once; ~70 MB
per batch), each batch stored with one write. A tile listed several times is made once.
**Prefetch** (`as_prefetch` / `World.prefetch`) generates and stores the missing tiles of a
latitude / longitude box over a zoom range without returning them (at most 10⁶ tiles, checked
before any work).

**Cache:** a handle keeps the decoded tiles it reads or generates in memory: one entry per layer
of a tile (as returned), the least recently used dropped beyond the cache size (default 256 MiB;
a generated tile goes in with all its layers, ~1.1 MiB; an `rgb` layer is 192 KiB). A cached
tile is copied instead of read and decompressed. One lock guards the index and is held only to
look up, insert or evict (no copying under it). Size 0 turns the cache off (`as_set_cache_mb` /
`World(..., cache_mb=)`, `World.cache_mb`). Prefetched tiles are not cached.

**Threads:** a handle (`as_world *` / `aerialsynth.World`) may be used by several threads at
once. Store reads run in parallel, writes are serialized, the cache is shared; two threads asking for the same missing
tile may both generate it (identical result, stored once). Close a handle only when no other
thread uses it. A tiles file can be open by only one handle per process (a second open fails
while the first is alive): share the handle.

## Rendering

A **camera** over a world renders frames from poses with the renderer of `terrain run`
(`crates/render`: level-of-detail tile selection, CPU or GPU rasterizer, sky and atmosphere, sun /
moon / stars, night lights, the camera sensor model, depth and land-cover ground truth), so its
images match the CLI's datasets. The tiles in view are read from the world's store, or generated
and stored when missing (as `terrain run` with `tiles.lazy: true`).

A camera is either

- a **pinhole** camera: width, height, horizontal field of view (degrees), optionally the
  principal point (Python), mounted `forward` (default) or `nadir`, with the default render
  settings (Python: or those of a scenario), or
- a **scenario camera**: one of the `cameras` of a scenario YAML (by HDF5 path such as `/cam0` or
  index; default the first), with the scenario's `render` section (backend, supersample,
  shading, lighting, atmosphere, stars), the camera's `intrinsics` (any model of the camera YAML
  schema), `extrinsics` (mount), `rgb.sensor` and `depth.kind`, and the `tiles` zoom range and
  cache size. The scenario's `world:` section is ignored: the world is the one of the handle the
  camera is made from.

Conventions (as in the datasets, [`docs/scenario.md`](../docs/scenario.md)):

| | |
|---|---|
| position | geodetic latitude, longitude (degrees) and height above the WGS84 ellipsoid (m); `World.surface_height` / `as_surface_height` give the surface height below (DSM) for heights above ground |
| attitude | of the **body** (FRD: x forward, y right, z down): aerospace Z-Y-X Euler angles (degrees) in the local NED frame: yaw = heading clockwise from north, pitch nose-up positive, roll right-wing-down positive |
| mount | the camera on the body: the scenario camera's `extrinsics` (default `nadir`); a pinhole camera's `forward` mount (optical axis = body forward, image top = up) makes the angles the camera's own: yaw = heading of the optical axis, pitch = its elevation (−30 looks 30° down); `nadir` (optical axis = body down, image top = body forward) looks straight down when level |
| camera frame | OpenCV: x right, y down, z = optical axis; pixel (0, 0) = centre of the top-left pixel; images row-major, row 0 = top |
| time | UTC as Unix seconds (Python also `datetime`, naive = UTC, and ISO 8601 strings): places the sun, moon and stars at that instant (the lighting `clock` mode); none (C: `NAN`) = the scenario's lighting (by default a fixed sun at 52° elevation) |
| rgb | u8 × 3 sRGB after the camera sensor model of the scenario camera (`rgb.sensor`; default settings otherwise): auto exposure converged on the frame (as a sequence's first frame), optics, noise, tone curve; no motion blur. The noise is deterministic, a function of the camera, the pose and the time: the same call gives the same image |
| depth | f32 metres: z-depth along the optical axis (the dataset default), or the range along the pixel ray when the scenario camera has `depth: { kind: range }`; +inf = sky. Needs an odd supersample (the default 3 is) |
| landcover | u8 class ids of the `landcover` layer, 255 = sky |
| backend | `auto` (the GPU when there is a usable one that takes the camera model, else the CPU), `cpu` or `gpu` (an error without a usable GPU); default: the scenario's `render.backend` (auto). CPU and GPU images agree closely, not bit for bit |

Only the images asked for are made (no RGB: shading is skipped). A camera's tile cache holds up to
`tiles.cache_tiles` tiles (default 2000, ~0.7 MB each) in memory.

**Threads:** renders of one camera are serialized (it holds one renderer); different cameras,
also of the same world, render concurrently (GPU work is serialized by the device). Python
releases the GIL while rendering. A camera keeps its world's store open: in C the world may be
closed first (the store closes with its last camera); in Python closing the world closes its
cameras.

## C

**Prebuilt:** each [GitHub release](https://github.com/shadymeowy/aerialsynth/releases) has
`terrain-<version>-<target>.tar.gz` (`.zip` on Windows) for `x86_64-unknown-linux-gnu`,
`aarch64-unknown-linux-gnu` (glibc ≥ 2.28), `aarch64-apple-darwin`, `x86_64-apple-darwin`
(macOS ≥ 11) and `x86_64-pc-windows-msvc`, with the `terrain` CLI and the C library. HDF5,
zlib (and on Windows the C runtime) are linked in: the libraries need only system libraries.

```
bin/terrain[.exe]
lib/libaerialsynth.so, libaerialsynth.a                      Linux
    libaerialsynth.dylib, libaerialsynth.a                   macOS
    aerialsynth.dll, aerialsynth.dll.lib (import library),   Windows
    aerialsynth.lib (static, /MT)
    native-static-libs.txt     system libraries to link with the static library
include/aerialsynth.h
examples/tile.c
```

```sh
cc -std=c99 examples/tile.c -I include -L lib -laerialsynth -Wl,-rpath,$PWD/lib -o tile        # shared
cc -std=c99 examples/tile.c -I include lib/libaerialsynth.a $(cat lib/native-static-libs.txt) -o tile  # static
```

```bat
cl /W4 examples\tile.c /I include lib\aerialsynth.dll.lib          :: shared: aerialsynth.dll next to tile.exe or on the PATH
cl /W4 /MT examples\tile.c /I include lib\aerialsynth.lib <native-static-libs.txt>   :: static
```

The release workflow (`.github/workflows/release.yml`) builds both on every platform from the
unpacked archive and runs them.

**From source:**

```sh
cargo build --release -p aerialsynth-capi
# target/release/libaerialsynth.so (and libaerialsynth.a), header bindings/c/include/aerialsynth.h
cc -std=c99 bindings/c/examples/tile.c -I bindings/c/include -L target/release -laerialsynth \
   -Wl,-rpath,$PWD/target/release -o tile
./tile out/world.h5 - 12 2200 1500     # TILES.h5 [CONFIG.yaml|- [Z X Y]]
cc -std=c99 bindings/c/examples/render.c -I bindings/c/include -L target/release -laerialsynth \
   -Wl,-rpath,$PWD/target/release -o render
./render out/world.h5 - 45 10 1500 view.ppm   # TILES.h5 [CONFIG.yaml|- [LAT LON HEIGHT_ABOVE_GROUND [OUT.ppm]]]
```

```c
#include "aerialsynth.h"

as_world *w = as_open("out/world.h5", "my.yaml" /* or NULL */, -1 /* seed: < 0 keeps the config's */);
if (!w) { fprintf(stderr, "%s\n", as_last_error()); return 1; }
size_t n = as_layer_size(AS_LAYER_ELEVATION);          /* 256 * 256 * 4 */
float *h = malloc(n);
int rc = as_tile(w, 12, 2200, 1500, AS_LAYER_ELEVATION, h, n);
if (rc != AS_OK) fprintf(stderr, "%d: %s\n", rc, as_last_error());

uint32_t zxy[] = {12, 2200, 1500,  12, 2201, 1500,  12, 2200, 1501,  12, 2201, 1501};
float *block = malloc(4 * n);                          /* tile i at block + i * n bytes */
rc = as_tiles(w, zxy, 4, AS_LAYER_ELEVATION, block, 4 * n);
as_close(w);
```

| function | |
|---|---|
| `as_world *as_open(const char *tiles_file, const char *config_yaml, int64_t seed)` | open / create the store; NULL on error |
| `int as_tile(const as_world *w, uint32_t z, uint32_t x, uint32_t y, as_layer layer, void *out, size_t out_len)` | write a layer of a tile (generated and stored if missing); `AS_OK` or `AS_ERR_*` |
| `int as_tiles(const as_world *w, const uint32_t *zxy, size_t n, as_layer layer, void *out, size_t out_len)` | a layer of `n` tiles (`zxy`: n × (z, x, y)), tile `i` at `out + i * as_layer_size(layer)` |
| `int as_prefetch(const as_world *w, double lat_min, double lon_min, double lat_max, double lon_max, uint32_t z_min, uint32_t z_max, size_t *generated)` | generate and store the missing tiles of a box (at most `AS_MAX_PREFETCH_TILES`); the count into `*generated` (may be NULL) |
| `int as_set_cache_mb(const as_world *w, size_t mb)` | size of the decoded-tile cache in MiB (default `AS_DEFAULT_CACHE_MB` = 256; 0 = off) |
| `void as_close(as_world *w)` | close (NULL is ignored) |
| `size_t as_layer_size(as_layer layer)` | bytes of a tile of `layer` (0: unknown layer) |
| `int as_layer_describe(as_layer layer, as_layer_info *info)` | name, dtype, channels, element size, size |
| `int as_max_zoom(const as_world *w)` | the world's max zoom |
| `int as_surface_height(const as_world *w, double lat_deg, double lon_deg, double *height_m)` | DSM height at a point (m above WGS84) |
| `as_camera *as_camera_pinhole(const as_world *w, uint32_t width, uint32_t height, double hfov_deg, as_mount mount, as_backend backend)` | a pinhole camera (`AS_MOUNT_FORWARD` / `AS_MOUNT_NADIR`); NULL on error |
| `as_camera *as_camera_open(const as_world *w, const char *scenario_yaml, const char *camera, as_backend backend)` | a scenario camera (`camera`: "/cam0", "0" or NULL = the first); NULL on error |
| `int as_camera_size(const as_camera *c, uint32_t *width, uint32_t *height)` | image size |
| `int as_camera_backend(const as_camera *c)` | `AS_BACKEND_CPU` or `AS_BACKEND_GPU` |
| `int as_render(const as_camera *c, const as_pose *pose, double unix_time, uint8_t *rgb, size_t rgb_len, float *depth, size_t depth_len, uint8_t *landcover, size_t landcover_len)` | render a frame: the non-NULL buffers (lengths in elements: w·h·3, w·h, w·h); `unix_time` `NAN` = the scenario's lighting |
| `void as_camera_close(as_camera *c)` | close a camera (NULL is ignored) |
| `const char *as_last_error(void)` | message of this thread's last failure; valid until its next failure |
| `const char *as_version(void)` | library version |

Rendering (`examples/render.c` writes a PPM):

```c
as_camera *cam = as_camera_pinhole(w, 640, 480, 90.0, AS_MOUNT_FORWARD, AS_BACKEND_DEFAULT);
if (!cam) { fprintf(stderr, "%s\n", as_last_error()); return 1; }
double ground;
as_surface_height(w, 45.0, 10.0, &ground);
as_pose pose = {45.0, 10.0, ground + 300.0, 0.0 /* roll */, -30.0 /* pitch */, 90.0 /* yaw */};
uint8_t *rgb = malloc(640 * 480 * 3);
float *depth = malloc(640 * 480 * sizeof(float));
int rc = as_render(cam, &pose, 1782027000.0 /* 2026-06-21T07:30:00Z */, rgb, 640 * 480 * 3, depth, 640 * 480, NULL, 0);
as_camera_close(cam);
```

Errors: `AS_ERR_INVALID_ARGUMENT` (-1: NULL pointer, unknown layer, tile out of range, a pose or
camera setting out of range),
`AS_ERR_BUFFER_TOO_SMALL` (-2), `AS_ERR_FAILED` (-3: store, config or generation),
`AS_ERR_PANIC` (-4: an internal error, caught). Rust panics never cross the API.

The header is generated by [cbindgen](https://github.com/mozilla/cbindgen) from
`c/src/lib.rs` (`c/cbindgen.toml`) and committed; `cargo test -p aerialsynth-capi` checks that it
is up to date (`AERIALSYNTH_BLESS=1` regenerates it) and compiles and runs `examples/tile.c` and
`examples/render.c` with the system C compiler (`cc` / `$CC`; MSVC's `cl.exe` on Windows; skipped
without one). Linking the static library needs the system libraries Rust reports (`cargo rustc -p
aerialsynth-capi --release --crate-type staticlib -- --print native-static-libs`;
`lib/native-static-libs.txt` in the release archives); on Linux `-ldl -lgcc_s -lutil -lrt
-lpthread -lm -lc`. On Windows the build is `target\release\aerialsynth.dll` (import library
`aerialsynth.dll.lib`) and `aerialsynth.lib`, on macOS `libaerialsynth.dylib` and
`libaerialsynth.a`.

## Performance

Measured with `cargo run --release -p aerialsynth-core --example tile_bench -- DIR 64` (CPU only:
`VK_DRIVER_FILES=/nonexistent.json VK_ICD_FILENAMES=/nonexistent.json`, 32 tiles) on an RTX 2080
Ti and a Xeon W-2125 (8 threads); the default world at zoom 12; milliseconds per tile. The
machine was shared with other jobs (about 3 of the 8 threads busy), so the numbers are rough;
compare within a column. "Store" is the first read after opening (decompression from the HDF5
store; the OS file cache is warm), "cached" a second read.

| ms per tile | GPU | CPU only |
|---|---|---|
| generate missing tiles, one `as_tile` / `tile` call each | 85 | 395 |
| generate missing tiles, one `as_tiles` / `tiles` call | **32** | **188** |
| rgb from the store, one call each | 0.87–1.05 | 0.83 |
| rgb from the store, one batch call | **0.41** | **0.26** |
| rgb cached, one call each / one batch call | **0.026 / 0.016** | **0.019 / 0.011** |
| elevation: store one call each / store batch / cached | 0.79 / **0.35** / **0.036** | 0.65 / **0.18** / **0.032** |
| landcover: store one call each / store batch / cached | 0.084 / **0.043** / **0.008** | 0.12 / **0.04** / **0.005** |

Before the cache and the batch calls, every read cost the "one call each, from the store" time
(rgb 0.85 ms, elevation 0.7 ms, landcover 0.08 ms on a quiet machine) and a missing tile was
generated alone (48 ms per tile on the GPU, ~250 ms on the CPU on a quiet machine; `terrain
tiles` makes about 60 tiles/s on the GPU). Batch decompression is limited by the HDF5 reads,
which are serialized.

## Python

See [`python/README.md`](python/README.md).

```python
import aerialsynth

with aerialsynth.World("out/world.h5", config=None, seed=None) as w:   # created if missing
    rgb = w.tile(12, 2200, 1500, "rgb")          # np.uint8 (256, 256, 3)
    h = w.tile(12, 2200, 1500, "elevation")      # np.float32 (256, 256)
    hs = w.tiles([(12, 2200, 1500), (12, 2201, 1500)], "elevation")  # np.float32 (2, 256, 256)
    cam = w.camera(width=640, height=480, hfov=90)               # or w.camera(config="scenario.yaml", camera="/cam0")
    f = cam.render(lat=45.0, lon=10.0, height=w.surface_height(45.0, 10.0) + 300,
                   roll=0, pitch=-30, yaw=90, time="2026-06-21T07:30:00Z", depth=True)
    f.rgb, f.depth                               # np.uint8 (480, 640, 3), np.float32 (480, 640)
aerialsynth.LAYERS["elevation"]                  # LayerInfo(name, dtype, channels, shape, size, description)
```

```sh
cd bindings/python
maturin build --release -o dist          # dist/aerialsynth-0.1.0-cp310-abi3-<platform>.whl
pip install dist/aerialsynth-*.whl pytest && pytest tests
```

Release wheels are built by `.github/workflows/wheels.yml` and attached to the GitHub release
of each `v*` tag: `manylinux_2_28` x86_64 and aarch64 (in the manylinux container: glibc 2.28 or
newer), macOS arm64 and x86_64 (macOS 11 or newer) and Windows x86_64. Each is checked to link
only system libraries (`auditwheel`, `delocate` / `otool -L`, the PE imports) and tested on
CPython 3.10 and 3.13. The Linux build runs locally with
`docker run quay.io/pypa/manylinux_2_28_x86_64` and `maturin build --release --manylinux 2_28`.

The extension uses the stable ABI (PyO3 `abi3-py310`): one wheel for CPython 3.10 and newer.
It returns tiles and images as `bytearray`s and the Python layer views them with `np.frombuffer`
(writable arrays, no copy), so the extension needs no numpy C API; numpy is the only dependency.

### Cargo and libpython

The extension crate (`bindings/python`, `aerialsynth-python`) is a workspace member, so
`cargo build / clippy / test --workspace` build it too. PyO3's `extension-module` feature is on,
so it never links libpython (the interpreter provides the symbols when it loads the module), and
the crate has no Rust test targets (`test = false`, `doctest = false`): a test executable could
not link without libpython. Its logic is tested in `aerialsynth-core`, the module itself by
`python/tests` against a built wheel. maturin builds it from the same workspace (the shared
`target/`).
