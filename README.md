# aerialsynth

A procedural planet for aerial vision research. `terrain` generates a deterministic world as
Web-Mercator XYZ tiles, flies cameras and an IMU over it, and writes datasets with exact
ground truth. A viewer flies over the world in realtime.

- **World:** continents, mountains carved by erosion, rivers, lakes, climate and biomes,
  forests of individual trees, fields, towns with buildings, roads, night lights. Any place,
  any zoom, generated on demand (on the GPU) into one HDF5 tile store.
- **Flights:** spline paths with wind, gusts, turbulence and engine vibration; or your own
  trajectory CSV.
- **Sensors:**
  - cameras of any model (pinhole, fisheye, omnidirectional) with lighting for the date and
    time (sun, moon, stars, planets, night lights), atmosphere, shadows and a sensor model;
  - ground truth: depth, optical flow, land cover, star positions;
  - event cameras and an IMU.
- **Viewer:** a map of the tile store and the camera through the dataset renderer, flown
  live.

## Start

```sh
cargo build --release                       # the binary: target/release/terrain
terrain config > my.yaml                    # a commented scenario template; edit it
terrain run -c my.yaml                      # trajectory → tiles → render (→ events)
terrain view -c my.yaml                     # look around: map, camera (M), flying (F)
python scripts/check_gt.py out/seq.h5       # check the ground truth of every camera
```

`configs/quick.yaml` is a 10 s smoke test and `configs/dataset.yaml` a fuller dataset.
`configs/examples/` has night flights, a full moon, an event camera rig, a star tracker, a
fisheye, a 10 km cruise, a sunset and an IMU check.

## Commands

| command | |
|---|---|
| `terrain run` | make a dataset: trajectory → tiles → render → events. `--step traj,tiles,render,events` runs single steps (an existing trajectory file is kept unless `traj` is asked for); `--lazy` generates missing tiles while rendering |
| `terrain tiles` | plan and generate the flight's tiles; or a region's (`--bbox … --zooms 6-14`), a list's (`--list`); `--dry-run [-o FILE]` only lists them; `--png PREFIX --zoom Z` previews a mosaic without a store |
| `terrain view` | the viewer: map and camera, flown live; generates as you go (`docs/viewer.md`) |
| `terrain info FILE` | summarize a tile store or a sequence file |
| `terrain config` | the scenario template; `--all` every setting; `-c my.yaml` a scenario with its defaults filled in |

`run`, `tiles`, `view` and `config` read one scenario (`-c`) and take `--seed` (a different
world) and `-j` (threads); `info` takes just the file.

## Outputs

- **Tile store** (`tiles.file`): the world's tiles, layers rgb, albedo, elevation (DSM),
  normal, land cover, night lights. It holds one world and grows as needed.
- **Sequence file** (`output.file`):
  - body poses, the IMU, and every camera's frames, depth, flow, land cover, events and
    stars;
  - i64 µs timestamps, self-describing datasets (units, descriptions);
  - optional PNG / NPY export.

Layouts in `docs/formats.md`. Python helpers in `scripts/`: `check_gt.py`, `check_imu.py`,
`check_events.py` (validation), `view_seq.py`, `view_events.py` (contact sheets),
`contact.py` (generator previews).

## Documentation

| | |
|---|---|
| `docs/scenario.md` | writing scenarios: sections, cameras and modalities, conventions, look |
| `docs/formats.md` | the tile store and sequence file layouts, validation scripts |
| `docs/simulation.md` | what is simulated: terrain, rendering, lighting, sensor, IMU, flights; performance, tests |
| `docs/viewer.md` | the viewer: map and camera views, controls, how it works |
| `docs/events.md` | event cameras |
| `docs/stars.md` | stars, planets, Moon: catalogue, astrometry, star ground truth |
| `docs/gpu.md` | the GPU backend: tile generation and rendering |

## Code

```
crates/geodesy    ellipsoid, geodetic/ECEF/ENU/NED/AER, XYZ tile math, attitude
crates/h5         safe wrapper over hdf5-sys (HDF5 2.2.0 bundled)
crates/tilestore  the HDF5 tile store
crates/terragen   the world generator
crates/render     cameras, flights, level of detail, renderer (CPU / GPU), lighting, sensor, events, IMU, writers
crates/viewer     terrain view: map and camera views
crates/cli        the terrain command
```

`cargo test --release` runs the tests (`docs/simulation.md` lists them).
