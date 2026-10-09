# aerialsynth (Python)

Tiles and camera images of the [aerialsynth](https://github.com/shadymeowy/aerialsynth) procedural
planet as numpy arrays. A `World` is an HDF5 tile store of one world (seed + world config);
`World.tile` reads a layer of a Web-Mercator XYZ tile (256 × 256 pixels) from the store, and a
tile that is not stored yet is generated (on the GPU when there is a suitable one, else on the
CPU), stored and returned. A `Camera` renders images of the world from a pose
([rendering](#rendering)).

```python
import aerialsynth

with aerialsynth.World("out/world.h5", config=None, seed=None) as w:   # created if missing
    rgb = w.tile(12, 2200, 1500, "rgb")          # uint8 (256, 256, 3)
    h = w.tile(12, 2200, 1500, "elevation")      # float32 (256, 256), m above the WGS84 ellipsoid
    block = w.tiles([(12, x, y) for y in range(1500, 1508) for x in range(2200, 2208)], "rgb")
                                                 # uint8 (64, 256, 256, 3): many tiles in one call
    w.prefetch((45.0, 10.0, 45.2, 10.3), (10, 14))  # generate and store a box's missing tiles

aerialsynth.LAYERS["normal"]   # LayerInfo(name='normal', dtype=dtype('int8'), channels=3, shape=(256, 256, 3), ...)
```

- **The world** is given like the CLI's `terrain -c CONFIG --seed N`: `config` is a scenario YAML
  (its `world:` section is the world and its `tiles.max_zoom` the zoom limit, default 18; the
  other sections are ignored) or a bare world config; `None` is the default world. `seed`
  overrides the config's seed.
- **One world per store:** opening a store made by another world (or generator version) raises
  `RuntimeError`, naming the settings that differ. Give each world its own tiles file.
- **Tiles:** `z` at most `World.max_zoom`, `x, y < 2**z`; row 0 is the north edge. The arrays are
  writable and own their memory (no copy is made from the extension's buffer).
- **Many tiles:** `w.tiles(coords, layer="rgb")` takes `(z, x, y)` triples (a list or an
  `(n, 3)` integer array) and returns one array of shape `(n,) + LAYERS[layer].shape`. It is
  much faster than `tile` in a loop: all coordinates are checked first (one bad one raises
  `ValueError` and nothing is made), stored tiles are decompressed in parallel, missing ones are
  generated together (batches of up to 64: 2–3× the tiles per second of single calls, see
  [the timings](../README.md#performance)) and stored, and repeated tiles are made once. The GIL is released meanwhile.
- **Prefetch:** `w.prefetch((lat_min, lon_min, lat_max, lon_max), zooms)` (`zooms` a zoom or a
  `(z_min, z_max)` range) generates and stores the missing tiles of a box without returning them,
  and returns how many it made. Boxes over `MAX_PREFETCH_TILES` (10⁶) tiles are refused before
  any work. It cannot be interrupted (Ctrl-C waits for it).
- **Cache:** a `World` keeps the decoded tiles it reads or generates in memory, the least
  recently used dropped beyond `cache_mb` (default 256 MiB; a generated tile with all its layers
  is ~1.1 MiB, an `rgb` layer 192 KiB). A cached tile is a copy (~0.02 ms) instead of a store
  read and decompression (~0.8 ms for `rgb`). `World(..., cache_mb=0)` or `w.cache_mb = 0` turns
  it off; `w.cache_info()` gives its size, bytes and entries held, hits and misses.
- **Errors:** `ValueError` for bad coordinates, layer names, seeds or a closed world;
  `FileNotFoundError` / `OSError` for an unreadable config; `RuntimeError` for a store of another
  world, an invalid config, or a failed read / generation.
- **Threads:** a `World` can be used from several threads; reading and generation release the
  GIL. A tiles
  file can be open by only one `World` per process at a time (a second one raises
  `RuntimeError`): share it, or close the first.

| layer | dtype | shape | |
|---|---|---|---|
| `rgb` | uint8 | 256 × 256 × 3 | satellite look (fixed sun, haze), sRGB |
| `albedo` | uint8 | 256 × 256 × 3 | unlit surface colour, sRGB encoded |
| `elevation` | float32 | 256 × 256 | DSM (ground, canopy, buildings, water), m above the WGS84 ellipsoid, at pixel centres |
| `normal` | int8 | 256 × 256 × 3 | unit normal (east, north, up) × 127 |
| `landcover` | uint8 | 256 × 256 | class id: 0 unknown, 1 ocean, 2 lake, 3 river, 4 beach, 5 sand, 6 rock, 7 snow, 8 grass, 9 shrub, 10 forest, 11 crop, 12 building, 13 road, 14 wetland, 15 tundra, 16 bare, 17 urban; classes v2 add 20–110 (`docs/formats.md`) |
| `emission` | uint8 | 256 × 256 × 3 | night lights, linear radiance = 16 (v/255)³ |

## Rendering

```python
import aerialsynth

with aerialsynth.World("out/world.h5") as w:
    cam = w.camera(width=640, height=480, hfov=90)          # a pinhole camera, forward mount
    ground = w.surface_height(45.0, 10.0)                   # DSM height below, m above WGS84
    f = cam.render(lat=45.0, lon=10.0, height=ground + 300,
                   roll=0, pitch=-30, yaw=90,               # looking east, 30° down
                   time="2026-06-21T07:30:00Z", depth=True, landcover=True)
    f.rgb        # uint8 (480, 640, 3), sRGB after the sensor model
    f.depth      # float32 (480, 640), z-depth in m, inf = sky
    f.landcover  # uint8 (480, 640), 255 = sky
    f.exposure, f.time, f.sun_elevation, f.position_ecef, f.r_ecef_cam
    cam.K        # 3 × 3 camera matrix (pinhole models)

    # a camera of a scenario (its render settings, intrinsics, mount, sensor), on the CPU
    cam0 = w.camera(config="configs/dataset.yaml", camera="/cam0", backend="cpu")
```

`Camera.render` uses the renderer of `terrain run`, so the images match the CLI's datasets; the
tiles in view are generated into the store when missing (like `terrain run` with lazy tiles).

- **Cameras:** `w.camera(width, height, hfov, cx=None, cy=None, mount="forward")` is a
  distortion-free pinhole camera (principal point default the image centre); `mount="forward"`
  makes the pose's angles the camera's own, `"nadir"` looks straight down when level.
  `w.camera(config=SCENARIO, camera="/cam0" | 0 | None)` is one of a scenario's `cameras` (default
  the first) with its `intrinsics` (any camera model), `extrinsics`, `rgb.sensor` and
  `depth.kind`; `config` with `width/height/hfov` is a pinhole camera with the scenario's render
  settings. The scenario's `render` section (supersample, shading, lighting, atmosphere,
  backend) and `tiles` zoom range apply; its `world` section is ignored (the world is `w`).
- **Pose:** `lat`, `lon` (degrees) and `height` (m above the WGS84 ellipsoid; see
  `World.surface_height`); `roll`, `pitch`, `yaw` (degrees) of the body in the local
  north-east-down frame (aerospace Z-Y-X: yaw = heading clockwise from north, pitch nose-up,
  roll right wing down). The camera sits on the body by its mount. Camera frame: OpenCV (x right,
  y down, z forward); `f.r_ecef_cam` maps camera vectors to ECEF.
- **Time:** UTC as Unix seconds, a `datetime` (naive = UTC) or an ISO 8601 string; it places the
  sun, moon and stars. `None` uses the scenario's lighting (by default a fixed sun at 52°).
- **Images:** only those asked for (`rgb=True`, `depth=False`, `landcover=False`) are made.
  The RGB image is a sequence's first frame (auto exposure converged on it, no motion blur); its
  noise is deterministic (the same call gives the same image). Depth is the z-depth (or the range
  with a scenario camera's `depth: { kind: range }`), `inf` for sky.
- **Backend:** `backend="auto"` (the GPU when there is a usable one, else the CPU), `"cpu"` or
  `"gpu"` (`RuntimeError` without a usable GPU); `None` (default) takes the scenario's
  `render.backend` (auto). `cam.backend` tells which one renders.
- **Threads:** rendering releases the GIL. One camera renders one frame at a time (calls from
  several threads wait for each other); several cameras render concurrently. A camera keeps the
  store open until it is closed (`cam.close()`, `with`, or closing the world, which closes its
  cameras).
- **Errors:** `ValueError` for a pose out of range or not finite, bad camera settings, an unknown
  scenario camera, a bad time or a closed camera; `RuntimeError` for an invalid scenario or a
  failed render.

## Install

From the wheels attached to each
[GitHub release](https://github.com/shadymeowy/aerialsynth/releases): one wheel per platform,
for CPython 3.10 and newer (`cp310-abi3`, the stable ABI). Each is self-contained (HDF5 and
zlib are linked statically; nothing else to install besides numpy):

| platform | wheel |
|---|---|
| Linux x86_64 (glibc ≥ 2.28) | `aerialsynth-<version>-cp310-abi3-manylinux_2_28_x86_64.whl` |
| Linux aarch64 (glibc ≥ 2.28) | `aerialsynth-<version>-cp310-abi3-manylinux_2_28_aarch64.whl` |
| macOS Apple silicon (≥ 11) | `aerialsynth-<version>-cp310-abi3-macosx_11_0_arm64.whl` |
| macOS Intel (≥ 11) | `aerialsynth-<version>-cp310-abi3-macosx_11_0_x86_64.whl` |
| Windows x86_64 | `aerialsynth-<version>-cp310-abi3-win_amd64.whl` |

```sh
pip install https://github.com/shadymeowy/aerialsynth/releases/download/v0.2.0/aerialsynth-0.2.0-cp310-abi3-manylinux_2_28_x86_64.whl
```

A GPU is optional: without a suitable one, images are rendered on the CPU, and tiles are
generated on the CPU (on macOS always: Metal has no 64-bit floats). Other platforms: build from
source (below).

## Build

One wheel (`cp310-abi3`, the stable ABI) serves CPython 3.10 and newer. It needs the Rust
toolchain of the workspace, a C compiler and CMake (HDF5 is built from source; per platform see
the [main README](../../README.md#requirements)), and
[maturin](https://www.maturin.rs):

```sh
cd bindings/python
maturin build --release -o dist        # or: pip install .
pip install dist/aerialsynth-*.whl
pip install pytest && pytest tests
```

The only Python dependency is numpy.
