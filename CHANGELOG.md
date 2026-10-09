# Changelog

## Unreleased

- **Bindings** (`bindings/`): tile access from C and Python. Open a world's tile store (the
  world given like `terrain -c FILE --seed N`) and get a layer of tile z/x/y; missing tiles are
  generated (GPU if available, else CPU) and stored first.
  - C: `libaerialsynth` (shared and static) with the header `bindings/c/include/aerialsynth.h`
    (`as_open`, `as_tile`, `as_close`, layer descriptions, per-thread error messages).
  - Python: the `aerialsynth` package (maturin, one `cp310-abi3` wheel for CPython ≥ 3.10)
    returning numpy arrays; `World(tiles_file, config, seed).tile(z, x, y, layer)`.
- **Platforms:** CI builds and tests on Linux, macOS (Apple silicon and Intel) and Windows
  (MSVC), including `cargo install --path crates/cli`. On Windows the C runtime is linked
  statically (no Visual C++ redistributable needed).
- **Release artifacts** (attached to each `v*` release):
  - Python wheels (`cp310-abi3`) for manylinux_2_28 x86_64 and aarch64, macOS arm64 and x86_64
    (macOS ≥ 11) and Windows x86_64;
  - `terrain-<version>-<target>` archives with the `terrain` CLI and the C library (shared and
    static, header, example) for the same platforms.

  All are self-contained: HDF5 and zlib are linked statically, only system libraries are needed.

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
