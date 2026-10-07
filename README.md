# aerialsynth — procedural XYZ terrain tiles + onboard camera simulator

Rust toolchain for aerial visual odometry research (command-line tool: `terrain`). It covers
five stages:

1. **Plan:** list the Web-Mercator XYZ tiles a trajectory needs.
2. **Generate:** produce those tiles procedurally (deterministic, lazily, at any zoom) into one
   HDF5 tile pyramid.
3. **Synthesize:** create realistic flight trajectories (spline path plus small ODE disturbances
   such as wind, gusts and vibration).
4. **Render:** produce onboard camera images from the tiles with exact ground truth (poses,
   depth, optical flow, land cover).
5. **Validate:** check the GT with a Python script.

```
cargo build --release
./target/release/terrain config > my.yaml             # full default scenario, every option documented by name
./target/release/terrain run -c configs/quick.yaml    # traj → plan → gen → render
python scripts/check_gt.py out/quick/seq.h5           # validate every camera's GT
python scripts/check_imu.py out/quick/seq.h5          # IMU vs /pose, noise statistics
python scripts/view_seq.py out/quick/seq.h5 view.png  # rgb | depth | flow | validity | landcover
```

`configs/dataset.yaml` is a fuller example:

- 60 s of 600 m AGL flight with wind, gusts and vibration
- forward-oblique distorted camera
- clock-driven sun
- full sensor model

## Subcommands

| command   | what it does |
|-----------|--------------|
| `config`  | print the (merged) scenario YAML with all defaults |
| `traj`    | synthesize a flight record → `trajectory.file` (CSV) |
| `plan`    | tiles needed for all cameras along the trajectory, plus `tiles.margin` (default 2) rings of neighbours per zoom, so a consumer whose pose estimate is slightly off still finds the surrounding tiles (`-o tiles.txt`) |
| `gen`     | generate tiles into `tiles.file` (from the plan, `--tiles list`, or `--bbox … --zooms a-b`); existing tiles are skipped, so the store grows lazily |
| `render`  | create the sequence file: body poses, IMU, every camera's frame modalities (`--lazy` generates missing tiles on the fly and stores them) |
| `events`  | add the event streams of cameras with an `events` modality (ESIM-style, realistic sensor noise); see `docs/events.md` |
| `run`     | `traj` (if missing) → `plan` → `gen` → `render` (→ `events` if any camera has events) |
| `info`    | summarize a tile store / sequence file |
| `preview` | generate tiles straight into a PNG mosaic (`--layers rgb,albedo,elevation,normal,landcover,hillshade`) |

All subcommands read **one scenario YAML** (`-c`) with the sections `world`, `tiles`,
`trajectory`, `render`, `cameras`, `imu` and `output`. See `configs/*.yaml`, or run
`terrain config` for the complete list with defaults.

## Cameras and modalities

`cameras` is a list. Each camera has intrinsics (camodocal schema), extrinsics, an HDF5 group
`path`, and the modalities it produces as subsections. An omitted subsection means the camera
does not produce that output; `{}` turns it on with defaults.

```yaml
cameras:
  - path: /cam0                 # everything of this camera goes under /cam0
    intrinsics: { model: pinhole, width: 640, height: 480, intrinsics: [420, 420, 319.5, 239.5], distortion: [-0.05, 0.01, 0, 0] }
    extrinsics: { mount: forward, pitch_deg: -30, translation: [0.4, 0, 0.1] }
    frame_rate: 10              # rate of the frame modalities
    rgb: { gray: false, sensor: { exposure: {...}, motion_blur: {...}, noise: {...} } }
    depth: { kind: z }          # or range
    flow: {}
    landcover: {}
  - path: /dvs                  # an event camera; it can also carry depth / flow ground truth
    intrinsics: { model: pinhole, width: 640, height: 480, intrinsics: [400, 400, 319.5, 239.5] }
    extrinsics: { mount: nadir, translation: [0, 0.08, 0] }
    frame_rate: 20
    depth: {}
    flow: {}
    events: { contrast_pos: 0.25, contrast_neg: 0.25 }
imu: { path: /imu, rate_hz: 200 }   # omitted = no IMU
output: { file: out/seq.h5, pose: { path: /pose, rate_hz: 200 } }
```

Cameras without `rgb` are rendered geometry-only (no shading), so depth / flow for an event
camera cost little. The group paths are free-form; nothing in the toolchain assumes a
particular dataset layout.

## Conventions

- **Earth model:** WGS84 ellipsoid (configurable: `world.planet`). Heights are **meters above
  the ellipsoid**; the vertical datum is the ellipsoid, i.e. the geoid undulation is 0. Real
  SRTM data is referenced to EGM96 instead.
- **Tiles:** Google/OSM XYZ on Web Mercator (EPSG:3857), 256×256 px, y grows southwards, and
  pixel centres are registered (pixel (i, j) of tile (z, x, y) is global pixel
  (256x+i, 256y+j)).
- **Geodesy:** `crates/geodesy` mirrors pymap3d: `geodetic2ecef`, `ecef2geodetic`, `ned2ecef`,
  `geodetic2enu`, `aer2geodetic` and the rest. Angles are in radians and all maths is f64.
  Tests compare against pymap3d.
- **Frames:**
  - **Body:** FRD. Attitude `q_ned_body` (Hamilton, `[w,x,y,z]` in files) maps body vectors into
    the local NED frame at the body.
  - **Camera:** OpenCV (x right, y down, z forward). Pixel centres sit on integer coordinates.
  - **Extrinsics:** a camera's `extrinsics` give the mount (`nadir`/`forward`), roll/pitch/yaw
    offsets and the translation in the body frame, or an explicit `q_body_cam`.
- **Trajectory CSV:** columns are matched by name. Supported layouts:
  - `t,lat,lon,h,qw,qx,qy,qz`
  - `t,lat,lon,h,roll,pitch,yaw`
  - `t,x,y,z,qw..` (ECEF, q = body→ECEF)
  - `t,n,e,d,qw..` with a `# origin: lat,lon,h` line (pymap3d `ned2geodetic` semantics)

## Tile store (`tiles.file`, HDF5)

```
/levels/<z>/index        i32 [N,2]   (x, y)
/levels/<z>/elev_range   f32 [N,2]
/levels/<z>/rgb          u8  [N,256,256,3]  satellite look (baked sun, haze)
/levels/<z>/albedo       u8  [N,256,256,3]  unlit surface colour (sRGB encoded)
/levels/<z>/elevation    f32 [N,256,256]    DSM (canopy + buildings), m above ellipsoid
/levels/<z>/normal       i8  [N,256,256,3]  ENU unit normal * 127
/levels/<z>/landcover    u8  [N,256,256]    class id (terragen::landcover::NAMES)
/levels/<z>/emission     u8  [N,256,256,3]  night lights, linear = 16 (v/255)^3
```

- **Chunks:** one tile per chunk, compressed with shuffle + deflate. Compression runs in parallel
  and is written with HDF5 direct chunk I/O. The bundled HDF5 is 2.2.0 (via `hdf5-metno-sys`).
- **Readers:** files open in h5py or any HDF5 reader.
- **Growth:** rows can be appended in any order, so a store grows lazily.

## Sequence file (`output.file`)

One HDF5 file per sequence. Group paths come from the scenario; dataset names are fixed. All
timestamps are i64 µs since the sequence start (`output.start` into the trajectory; the root
attribute `t0` holds it in trajectory seconds).

```
<output.pose.path>/          body ground truth at output.pose.rate_hz
    t                        i64 [M]
    position_ecef, q_ecef_body               f64 [M,3], [M,4]
    lla, q_ned_body                          lat°, lon°, h; body → local NED
    position_ned0, q_ned0_body               in the NED frame at the first pose (attr ned0_origin_lla)
    sun_azimuth_deg, sun_elevation_deg, lights
<camera.path>/
    calib/                   intrinsics [4], distortion_coeffs, resolution [W,H] (i64),
                             T_body_cam [4,4] (camera → body), attrs model + camera_yaml
    t                        i64 [N] frame times (mid-exposure)
    pose/                    position_ecef [N,3], q_ecef_cam [N,4] at the frame times
    rgb, exposure            u8 [N,H,W,3] (or [N,H,W] gray); exposure time, gain, EV   ← rgb
    depth                    f32 [N,H,W], m, +inf = sky (attr kind: z | range)         ← depth
    flow, flow_valid         f32 [N,H,W,2] to the next frame in px; u8 [N,H,W]         ← flow
    landcover                u8 [N,H,W] (255 = sky)                                    ← landcover
    events/                  x, y u16; t i64 µs; p i8 (1 = ON); ms_index u64           ← events
<imu.path>/
    t, accel, gyro           i64 [K]; f64 [K,3] specific force (m/s²), angular rate (rad/s)
    gt_accel, gt_gyro, gt_bias_accel, gt_bias_gyro
    calib/T_body_imu         [4,4] IMU → body, noise parameters as attributes
```

- **Poses:** a camera's pose at any time is the `/pose` body pose composed with
  `calib/T_body_cam` (`<camera>/pose` holds it at the frame times for convenience). The body
  frame is the common rig frame of all sensors.
- **PNG export:** `output.png_dir` additionally writes `<png_dir>/<camera>/{rgb,depth,flow,
  flow_valid,landcover}/NNNNNN.*`, `frames.csv` and `camera.yaml` per camera.
- **camodocal:** its H5 reader expects the M3ED layout (`/ovc/left/data`, `/ovc/ts`, ...). The
  data types are the same (i64 µs timestamps, u16/i64/i8 events, f64 calibration); the reader
  needs to take the group / dataset names from its config, and to read RGB or set `rgb.gray`.
- **Compression:** `output.compression` sets the deflate level (shuffle + gzip, as in h5py) and
  `float_keep_bits`, an optional lossy rounding of depth/flow mantissas. 16 bits gives a max
  relative error of 7.6e-6 and shrinks depth/flow by ~35–40%. Flow is exactly recomputable from
  depth + poses (`scripts/check_gt.py`), so omitting `flow` roughly halves the file.
- **Validation:**
  - `scripts/check_gt.py` checks every camera: flow against depth reprojected with the poses
    (agrees to ~1e-5 px), photometric warping (~2 DN: sensor noise plus motion blur), and
    camera poses against `/pose` ∘ `T_body_cam`.
  - `scripts/check_imu.py` rebuilds the IMU truth from `/pose` independently (needs `/pose`
    finer than the IMU, e.g. `configs/imu_check.yaml`: agrees to 1e-4 m/s² and 3e-10 rad/s
    without lever arm; with a lever arm under 84 Hz engine vibration to the simulation step's
    discretization, ~5% of the lever term at 1 ms, 0.2% at 0.2 ms) and checks white noise and
    bias walk against the configured densities.
  - `scripts/check_events.py` checks the event format invariants and prints rate statistics.

## What is simulated

**Terrain (`crates/terragen`)**
- **Determinism:** a pure function of (seed, config, position), using f64 3D noise on the
  ellipsoid surface. There are no seams or pole problems.
- **Band limiting:** fields are band-limited by the pixel GSD, so coarse zooms approximate the
  average of fine ones. Large-scale fields are sampled on a 16-px grid aligned to tile corners.
- **Landforms:**
  - continents and shelves, home-region land bias
  - mountain belts (ridged multifractal with domain warp) carved by dendritic erosion gullies
  - hills with varying roughness, plateaus, arid mesas, dunes in sand seas
- **Water:**
  - rivers as a downhill flow graph with 3 levels (rivers, tributaries, streams; see
    `world.hydro.levels`): dendritic, always draining downhill, meandering
  - valleys, floodplains and riparian woods; wet or dry beds
  - lakes filled to their spill level, oceans with shallows and surf, beaches
- **Climate:** latitude, lapse rate, Hadley dryness, coast and noise. It drives biomes: snow, rock,
  tundra, boreal, temperate and tropical forests, steppe, savanna, desert, wetlands.
- **Vegetation:** individual tree crowns in 3 layers plus shrubs, with explicit crowns and canopy
  heights in the DSM. They are prefiltered when unresolved and cast shadows in the baked
  imagery.
- **Farmland:**
  - per-region field systems: grid, irregular, strips, centre pivots
  - seasonal crop palettes, crop rows, tramlines, headlands, wet/bare patches
  - hedges and tracks along field edges
- **Farmsteads:** house, barn, gravel yard and yard lamp.
- **Settlements:**
  - towns and villages with street grids (organic in old centres)
  - lots, pitched or flat roofs, building heights in the DSM, parks and industry
  - street lamps in the emission layer
- **Roads:** curvy inter-town roads and region-border tracks.

**Rendering (`crates/render`)**
- **Geometry:**
  - CPU reference rasterizer; tile meshes are built on the fly from the elevation layer
  - generic `CameraModel` trait implementing all of camodocal's `calib/camera.cpp` models, with
    the same YAML schema (`model`, `width`, `height`, `intrinsics`, `distortion`, `xi`,
    `max_fov_deg`, `inv_poly`, `affine`, `center`):
    - `pinhole` and `pinhole_full` (OpenCV rational)
    - `kannala_brandt`
    - `mei`
    - `scaramuzza`
  - fields of view beyond 180° are supported (range-based depth buffer)
  - quadtree LOD by projected texel size, with skirts against cracks
  - f64 transforms and an exact per-vertex camera model (incl. distortion)
- **Shading:**
  - supersampling; cameras with depth / flow / land cover need an odd factor (the GT sample is
    the central sub-sample, which is the pixel centre only for odd factors)
  - trilinear plus anisotropic pyramid texture filtering
- **Lighting:**
  - `relit` (albedo + normals + DSM ray-marched cast shadows) or `satellite` (baked imagery)
  - sun fixed or from date/time (NOAA), continuous twilight
  - moon position and phase with moonlight and the lunar disc
  - real stars (Hipparcos + Tycho-2, V ≤ 9) at their apparent positions for the date, time and
    place (star-tracker accuracy: 0.01″ vs Skyfield), radiometric brightness and colour, per-frame
    star ground truth; see `docs/stars.md`
  - `render.backend: gpu`: headless wgpu renderer and event sensor, same output as the CPU
    reference; see `docs/gpu.md`
  - night lights from the generated emission layer: street lamps (sodium/LED), porch lights,
    farmsteads, lit main roads near towns, plazas and industry
  - light-pollution glow in the haze
  - aerial perspective, sky and water glint
- **Sensor** (per camera, `rgb.sensor`):
  - auto exposure as a 1st-order ODE, split into exposure time and gain
  - motion blur from sub-frame poses of the high-rate trajectory
  - defocus, chromatic aberration, vignetting, bloom, starburst spikes on bright lights
  - shot, read and PRNU noise, then tone curve
- **IMU** (`imu:`):
  - truth (specific force, inertial angular rate) computed inside the flight simulator from its
    1 kHz positions and attitudes (second difference + Coriolis − WGS84 normal gravity; attitude
    difference + Earth rate), so it is consistent with the poses by construction, and stored
    as trajectory columns
  - each IMU sample is the mean over its window [t − dt/2, t + dt/2] (no delay)
  - the sensor model adds extrinsics/lever arm, misalignment, scale factor, turn-on bias, bias
    random walk, white noise (Kalibr-style densities) and saturation
  - error-free values and biases are stored as GT
- **Lamp flicker** (`render.lighting.flicker`): lamps flicker at 2× mains frequency, with three
  supply phases and a share of LED lamps. Frames integrate it over their exposure; the event
  camera resolves it (the renderer returns the flicker as cos / sin images, so flicker steps
  need no extra renders).
- **Trajectories:**
  - arc-length spline path (line, circle, figure8, lawnmower, random, waypoints), C1 between
    its dense samples
  - AGL terrain following, crab angle into the crosswind
  - the path is laid out in the tangent plane at its origin; heading, velocity and IMU truth are
    expressed in the local NED frame along it (meridian convergence included)
  - Dryden-like turbulence and 1-cosine gusts acting on position/attitude through small ODEs
  - engine-harmonic and band-limited broadband vibration
  - optional stabilized gimbal (the whole sensor platform: `/pose` is then the gimbal frame)

## Workspace layout

```
crates/geodesy    ellipsoid, geodetic/ECEF/ENU/NED/AER, XYZ tile math, attitude
crates/h5         safe wrapper over hdf5-sys (hdf5-metno-sys 0.12, HDF5 2.2.0 bundled)
crates/tilestore  HDF5 tile pyramid (layout above), parallel codec
crates/terragen   procedural terrain generator
crates/render     camera, trajectories/dynamics, LOD, rasterizer, lighting, sensor, writers, pipeline
crates/cli        `terrain` binary (the aerialsynth CLI)
scripts/          contact.py (generator contact sheets), check_gt.py + cammodels.py (camera GT, all models), check_imu.py, check_events.py, seqio.py, view_seq.py, view_events.py
docs/events.md    event camera modality: options, sensor model, format
docs/stars.md     star catalogue, astrometry, brightness, star ground truth
docs/gpu.md       GPU backend (wgpu, headless)
configs/          example scenarios (quick, dataset, fisheye, events, oblique_sunset, night, night_moon, cruise, imu_check)
```

## Performance (8 cores)

| task | speed |
|------|-------|
| generation | ~1 s CPU per 256² tile at zooms 13–17 (supersample 2) |
| rendering 640×512, 3×3 supersampled, shadows | ~0.5 s per frame |

`--lazy` rendering generates exactly the tiles each view needs.

Event simulation renders every step that moves the image by `max_px_per_step` (~100 s of
compute per simulated second at VGA with engine vibration); see `docs/events.md`.

## Tests

```
cargo test --release
```

- **Geodesy:** checked against pymap3d.
- **h5:** round trips, raw chunks, concurrency, error stacks.
- **Tile store:** round trip, rejection of invalid tiles.
- **Camera models and trajectories:** includes NED/ECEF CSV formats.
- **LOD and dynamics.**
- **Sensor:** auto-exposure ODE.
- **Scenario:** omitted modalities are off, overlapping HDF5 groups are rejected.
- **Events:** moving edge, background activity, low-light bandwidth.
- **IMU:** noise statistics, level-flight truth, lever arm and sample timing against the
  1 kHz simulator truth.
- **Solar position.**
- **Generator invariants:** determinism, seamless tile borders, parent ≈ mean of its children.

## Look / colour knobs

| knob | effect |
|------|--------|
| `world.look.albedo_saturation`, `albedo_brightness` | the generated surface colours |
| `world.look.*` | sun, ambient, haze of the baked satellite layer |
| `render.atmosphere.visibility_km`, `inscatter` | haze |
| `cameras[].rgb.sensor.tone` | `saturation`, `white_balance`, `curve` (`srgb`/`filmic`/`gamma`) |
| `cameras[].rgb.sensor.exposure.target` | overall brightness |
| `render.lighting` | `lights_intensity`, `moon_intensity`, `night_sky`, `light_pollution` |
