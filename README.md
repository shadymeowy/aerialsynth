# terrain — procedural XYZ terrain tiles + onboard camera simulator

Rust toolchain for aerial visual odometry research. It covers five stages:

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
python scripts/check_gt.py out/quick/seq.h5           # validate flow vs depth + poses
python scripts/view_seq.py out/quick/seq.h5 view.png  # rgb | depth | flow | validity
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
| `plan`    | tiles needed for the trajectory/camera (`-o tiles.txt`) |
| `gen`     | generate tiles into `tiles.file` (from the plan, `--tiles list`, or `--bbox … --zooms a-b`); existing tiles are skipped, so the store grows lazily |
| `render`  | render the sequence (`--lazy` generates missing tiles on the fly and stores them) |
| `run`     | `traj` (if missing) → `plan` → `gen` → `render` |
| `info`    | summarize a tile store / sequence file |
| `preview` | generate tiles straight into a PNG mosaic (`--layers rgb,albedo,elevation,normal,landcover,hillshade`) |

All subcommands read **one scenario YAML** (`-c`). Its sections are `world`, `tiles`, `camera`,
`extrinsics`, `trajectory`, `render`, `sensor` and `output`. Every field is optional. See
`configs/*.yaml`, or run `terrain config` for the complete list.

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
  - **Extrinsics:** `extrinsics` gives the mount (`nadir`/`forward`), the roll/pitch/yaw offsets
    and the translation, or an explicit `q_body_cam`.
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
/levels/<z>/emission     u8  [N,256,256,3]  night lights, linear = 4 (v/255)^2.2
```

- **Chunks:** one tile per chunk, compressed with shuffle + deflate. Compression runs in parallel
  and is written with HDF5 direct chunk I/O. The bundled HDF5 is 2.2.0 (via `hdf5-metno-sys`).
- **Readers:** files open in h5py or any HDF5 reader.
- **Growth:** rows can be appended in any order, so a store grows lazily.

## Sequence output (`output.h5` and/or `output.dir`)

| dataset                                                             | type / shape                                        |
|---------------------------------------------------------------------|-----------------------------------------------------|
| `rgb`                                                               | u8 [N,H,W,3]                                        |
| `depth`                                                             | f32 [N,H,W], z-depth in m, inf = sky                |
| `flow`                                                              | f32 [N,H,W,2], forward flow k → k+1 in px           |
| `flow_valid`                                                        | u8 [N,H,W], visibility in frame k+1                 |
| `landcover`                                                         | u8 [N,H,W]                                          |
| `t`                                                                 | timestamps                                          |
| `pose/cam_position_ecef`, `pose/cam_q_ecef`                         | camera pose in ECEF                                 |
| `pose/cam_lla`                                                      | camera position, lat/lon/h                          |
| `pose/cam_position_ned0`, `pose/cam_q_ned0`                         | camera pose in the NED frame of the first camera position |
| `pose/body_q_ned`, `pose/body_lla`                                  | body attitude and position                          |
| `exposure/{time,gain,ev}`, `sun/{azimuth,elevation}_deg`            | per-frame exposure and sun                          |
| `camera/` attrs `K`, `dist_radtan`, `q_body_cam`, `t_body_cam`      | intrinsics and extrinsics                           |

- **Directory output:** `rgb/*.png`, `depth/*.npy`, `flow/*.npy`, `flow_valid/*.png`,
  `landcover/*.png`, `poses.csv`, `camera.yaml` and `scenario.yaml`.
- **GT timing:** all GT refers to the mid-exposure pose.
- **Compression:** `output.compression` sets the deflate level (shuffle + gzip, as in h5py) and
  `float_keep_bits`, an optional lossy rounding of depth/flow mantissas. 16 bits gives a max
  relative error of 7.6e-6 and shrinks depth/flow by ~35–40%. Flow is exactly recomputable from
  depth + poses (`scripts/check_gt.py`), so `output.flow: false` halves the file.
- **Validation:** `scripts/check_gt.py` reprojects depth with the poses. Flow agrees to about
  1e-5 px, and photometric warping errors are about 2 DN (sensor noise plus motion blur).

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
  - generic `CameraModel` trait: pinhole radtan and Kannala-Brandt fisheye included (`configs/fisheye.yaml`)
  - quadtree LOD by projected texel size, with skirts against cracks
  - f64 transforms and an exact per-vertex camera model (incl. distortion)
- **Shading:**
  - supersampling (an odd factor gives an exact GT sample)
  - trilinear plus anisotropic pyramid texture filtering
- **Lighting:**
  - `relit` (albedo + normals + DSM ray-marched cast shadows) or `satellite` (baked imagery)
  - sun fixed or from date/time (NOAA), continuous twilight
  - moon position and phase with moonlight and the lunar disc, stars
  - night lights from the generated emission layer: street lamps (sodium/LED), porch lights,
    farmsteads, lit main roads near towns, plazas and industry
  - light-pollution glow in the haze
  - aerial perspective, sky and water glint
- **Sensor:**
  - auto exposure as a 1st-order ODE, split into exposure time and gain
  - motion blur from sub-frame poses of the high-rate trajectory
  - defocus, chromatic aberration, vignetting, bloom, starburst spikes on bright lights
  - shot, read and PRNU noise, then tone curve
- **Trajectories:**
  - arc-length spline path (line, circle, figure8, lawnmower, random, waypoints)
  - AGL terrain following, crab angle into the crosswind
  - Dryden-like turbulence and 1-cosine gusts acting on position/attitude through small ODEs
  - engine-harmonic and broadband vibration, optional stabilized gimbal

## Workspace layout

```
crates/geodesy    ellipsoid, geodetic/ECEF/ENU/NED/AER, XYZ tile math, attitude
crates/h5         safe wrapper over hdf5-sys (hdf5-metno-sys 0.12, HDF5 2.2.0 bundled)
crates/tilestore  HDF5 tile pyramid (layout above), parallel codec
crates/terragen   procedural terrain generator
crates/render     camera, trajectories/dynamics, LOD, rasterizer, lighting, sensor, writers, pipeline
crates/cli        `terrain` binary
scripts/          contact.py (generator contact sheets), check_gt.py (GT validation), view_seq.py
configs/          example scenarios (quick, dataset, fisheye, oblique_sunset, night, night_moon, cruise)
```

## Performance (8 cores)

| task | speed |
|------|-------|
| generation | ~1 s CPU per 256² tile at zooms 13–17 (supersample 2) |
| rendering 640×512, 3×3 supersampled, shadows | ~0.5 s per frame |

`--lazy` rendering generates exactly the tiles each view needs.

## Tests

```
cargo test --release
```

- **Geodesy:** checked against pymap3d.
- **h5:** round trips, raw chunks, concurrency.
- **Tile store:** round-trip test.
- **Camera models and trajectories:** includes NED/ECEF CSV formats.
- **LOD and dynamics.**
- **Sensor:** auto-exposure ODE.
- **Solar position.**
- **Generator invariants:** determinism, seamless tile borders, parent ≈ mean of its children.

## Look / colour knobs

| knob | effect |
|------|--------|
| `world.look.albedo_saturation`, `albedo_brightness` | the generated surface colours |
| `world.look.*` | sun, ambient, haze of the baked satellite layer |
| `render.atmosphere.visibility_km`, `inscatter` | haze |
| `sensor.tone` | `saturation`, `white_balance`, `curve` (`srgb`/`filmic`/`gamma`) |
| `sensor.exposure.target` | overall brightness |
| `render.lighting` | `lights_intensity`, `moon_intensity`, `night_sky`, `light_pollution` |
