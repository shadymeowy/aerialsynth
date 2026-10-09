# Changelog

## Unreleased

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
