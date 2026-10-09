# File formats

Both files are HDF5 and open in h5py or any HDF5 reader. `terrain info FILE` summarizes either.

## Tile store (`tiles.file`)

```
/                        attrs: format, format_version, generator_config (world YAML), seed,
                         generator_version, ellipsoid_a/b, ...
/levels/<z>/index        i32 [N,2]   (x, y)
/levels/<z>/elev_range   f32 [N,2]
/levels/<z>/rgb          u8  [N,256,256,3]  satellite look (baked sun, haze)
/levels/<z>/albedo       u8  [N,256,256,3]  unlit surface colour (sRGB encoded)
/levels/<z>/elevation    f32 [N,256,256]    DSM (canopy + buildings), m above ellipsoid
/levels/<z>/normal       i8  [N,256,256,3]  ENU unit normal * 127
/levels/<z>/landcover    u8  [N,256,256]    class id (land-cover classes below)
/levels/<z>/emission     u8  [N,256,256,3]  night lights, linear = 16 (v/255)^3
```

- **Chunks:** one tile per chunk, compressed with shuffle + deflate. Compression runs in parallel
  and is written with HDF5 direct chunk I/O. The bundled HDF5 is 2.2.0 (via `hdf5-metno-sys`).
- **Growth:** rows can be appended in any order, so a store grows lazily. A new store is
  written as `<file>.tmp` and renamed once its metadata is on disk, and every write ends with
  a flush, so a killed process loses at most its last batch.
- **Locking:** HDF5 locks the file: a store open for writing in one process (`terrain view`,
  `terrain tiles`) cannot be opened by another, and one open for reading cannot be written;
  such an open fails with an error saying so.
- **One world per store:** a store records the world config and generator version of its
  tiles. Every command refuses a store of another world, naming the settings that differ.
  It also refuses to add tiles to a store of another generator version; reading such a store
  only warns (`run` and `tiles` open a store for writing only when tiles are missing, or with
  `--lazy`; `terrain view` unless `--no-generate`). `terrain info` shows both.
- **Making tiles:** `terrain tiles` plans and generates the flight's tiles. `--bbox` covers a
  region, `--list` a tile list. Existing tiles are skipped, so the store grows lazily.
  `tiles.lazy` (or `terrain run --lazy`) generates what the renderer misses while rendering.
  `terrain view` generates what it shows.
- **Planning:** the plan covers every camera along the trajectory, plus `tiles.margin`
  (default 2) rings of neighbours per zoom, so a consumer whose pose estimate is slightly off
  still finds the surrounding tiles. `terrain tiles --dry-run -o tiles.txt` writes the plan.

## Sequence file (`output.file`)

One HDF5 file per sequence (format version 3). Group paths come from the scenario; dataset
names are fixed. All timestamps are i64 µs since the sequence start (`output.start` into the
trajectory; the root attribute `t0` holds it in trajectory seconds). Every dataset carries
`units` and `description` attributes, multi-column ones also `columns` (e.g. `exposure`:
`exposure_time_s,gain,ev`), and `landcover` its classes (`class_names`, `class_groups`,
`class_legacy`, `class_mapping`; see Land-cover classes).

```
<output.pose.path>/          body ground truth at output.pose.rate_hz
    t                        i64 [M]
    position_ecef, q_ecef_body               f64 [M,3], [M,4]
    lla, q_ned_body                          lat°, lon°, h; body → local NED
    position_ned0, q_ned0_body               in the NED frame at the first pose (attr ned0_origin_lla)
    sun_azimuth_deg, sun_elevation_deg, lights (artificial lights on, 0..1)
<camera.path>/
    calib/                   intrinsics [4], distortion_coeffs (always; empty = none), resolution [W,H] (i64),
                             T_body_cam [4,4] (camera → body), attrs model + camera_yaml
    t                        i64 [N] frame times (mid-exposure)
    pose/                    position_ecef [N,3], q_ecef_cam [N,4] at the frame times
    rgb, exposure            u8 [N,H,W,3] (or [N,H,W] gray); exposure time, gain, EV   ← rgb
    depth                    f32 [N,H,W], m, +inf = sky (attr kind: z | range)         ← depth
    flow, flow_valid         f32 [N,H,W,2] to the next frame in px; u8 [N,H,W]         ← flow
    landcover                u8 [N,H,W] class (output.landcover), 255 = sky            ← landcover
    events/                  x, y u16; t i64 µs; p i8 (1 = ON); ms_index u64           ← events
    stars/                   catalogue stars per frame (frame k: rows index[k]..index[k+1]): ← stars
                             index u64 [N+1]; id, x, y (px at the frame time), xm, ym (trail
                             centroid over the exposure), v (mag), irradiance, visible (pixel
                             shows sky); docs/stars.md
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
- **Reading:** everything is plain HDF5 with fixed dataset names under configurable group
  paths. The data types follow common event / VIO datasets such as M3ED (i64 µs timestamps,
  u16 / i64 / i8 events, f64 calibration), but the layout is this project's own: a reader
  written for another layout needs the group and dataset names above (h5py reads it as is).
- **Compression:** `output.compression` sets the deflate level (shuffle + gzip, as in h5py) and
  `float_keep_bits`, an optional lossy rounding of depth/flow mantissas. 16 bits gives a max
  relative error of 7.6e-6 and shrinks depth/flow by ~35–40%. Flow is exactly recomputable from
  depth + poses, so omitting `flow` roughly halves the file.
- **Accuracy** (measured against independent reconstructions from the stored data): flow
  agrees with depth reprojected with the poses to ~5e-6 px median, ~3e-5 px max; photometric
  warping to ~2 DN (sensor noise plus motion blur); camera poses with `/pose` ∘ `T_body_cam`.
  The IMU truth rebuilt from a `/pose` finer than the IMU (e.g. `configs/examples/imu_check.yaml`)
  agrees to 3e-10 rad/s (gyro) and 1e-4 m/s² (accel, without a lever arm). The example's
  lever arm (12 cm) under the default engine vibration (38 Hz and its 76 Hz harmonic) adds
  ~5.8 m/s² RMS, which the simulation step discretizes: 0.27 m/s² (~5%) off at the 1 ms step, ~0.2% at 0.2 ms.

## Land-cover classes

The `landcover` layer of the tile store and the `landcover` modality of cameras hold class ids
of one table, `terragen::landcover::CLASSES` (classes v2): id, name, group, legacy class,
display colour and material of every class. The shaders get it as generated WGSL
(`crates/terragen/src/gpu/wgsl/classes.wgsl`; a test keeps it in sync).

- **Ids are stable:** 0–17 are the classes of the first release and keep their meaning; new
  classes are appended in per-group ranges and never renumbered. Ids are below 128; 255 is the
  sky in sequence outputs. The generator emits classes 0–17 today; the others are reserved for
  the coming terrain kits (their data is in the table already).
- **Groups** (stable ids): 0 unknown, 1 water, 2 wetland, 3 bare, 4 snow-ice, 5 vegetation,
  6 forest, 7 agriculture, 8 disturbed, 9 built, 10 transport.
- **Legacy class:** every class names the class among 0–17 that stands for it, so readers of
  the first classes keep working (`output.landcover: legacy`).
- **Material** (used by both renderers): glint weight (sky reflection with Schlick Fresnel and
  sun glint; 1 on open water, as before v2), specular F0, roughness (glint exponent
  2 / roughness² − 2, 300 on water) and night-time self-emission. Classes not listed below as
  glinting or emissive are matte.

| id | name | label | group | legacy | colour | material |
|---|---|---|---|---|---|---|
| 0 | `unknown` | unknown | – | 0 | `#ff00ff` |  |
| 1 | `ocean` | ocean | water | 1 | `#14326e` | water glint |
| 2 | `lake` | lake | water | 2 | `#285aa0` | water glint |
| 3 | `river` | river | water | 3 | `#3c82c8` | water glint |
| 4 | `beach` | beach | bare | 4 | `#f0dca0` |  |
| 5 | `sand` | sand | bare | 5 | `#dcbe78` |  |
| 6 | `rock` | rock | bare | 6 | `#82786e` |  |
| 7 | `snow` | snow | snow-ice | 7 | `#fafaff` |  |
| 8 | `grass` | grass | vegetation | 8 | `#8cbe50` |  |
| 9 | `shrub` | shrub | vegetation | 9 | `#969646` |  |
| 10 | `forest` | forest | forest | 10 | `#1e6428` |  |
| 11 | `crop` | crop | agriculture | 11 | `#e6c83c` |  |
| 12 | `building` | building | built | 12 | `#c83c3c` |  |
| 13 | `road` | road | transport | 13 | `#3c3c3c` |  |
| 14 | `wetland` | wetland | wetland | 14 | `#468c82` |  |
| 15 | `tundra` | tundra | vegetation | 15 | `#a0a082` |  |
| 16 | `bare` | bare | bare | 16 | `#a0785a` |  |
| 17 | `urban` | urban | built | 17 | `#b49696` |  |
| 20 | `reservoir` | reservoir | water | 2 | `#3264aa` | water glint |
| 21 | `lagoon` | lagoon | water | 1 | `#287896` | water glint |
| 22 | `canal` | canal | water | 3 | `#468cd2` | water glint |
| 23 | `aquaculture` | aquaculture / salt pond | water | 2 | `#5a96aa` | water glint |
| 24 | `tidal_flat` | tidal flat | wetland | 14 | `#968c6e` | glint 0.5 |
| 25 | `coral_reef` | coral reef (shallow) | water | 1 | `#3cbebe` | water glint |
| 26 | `sea_ice` | sea ice | snow-ice | 7 | `#dcebf5` | glint 0.4 |
| 27 | `glacier` | glacier | snow-ice | 7 | `#c8e6fa` | glint 0.25 |
| 28 | `frozen_water` | frozen water | snow-ice | 7 | `#b4d2eb` | glint 0.6 |
| 29 | `dry_riverbed` | dry riverbed / wash | bare | 5 | `#c8b48c` |  |
| 30 | `salt_flat` | salt flat / playa | bare | 16 | `#f5f0e1` |  |
| 31 | `lava` | lava | bare | 6 | `#281e1e` | emissive 1.0 |
| 32 | `volcanic_ash` | volcanic ash / black sand | bare | 5 | `#464141` |  |
| 33 | `gravel` | gravel / alluvial fan | bare | 16 | `#afa591` |  |
| 34 | `badlands` | badlands | bare | 16 | `#be7850` |  |
| 35 | `scree` | scree / talus | bare | 6 | `#96918c` |  |
| 36 | `moraine` | moraine | bare | 16 | `#a5a096` |  |
| 37 | `cliff` | cliff | bare | 6 | `#645a55` |  |
| 40 | `tropical_rainforest` | tropical rainforest | forest | 10 | `#0a5a1e` |  |
| 41 | `mangrove` | mangrove | forest | 10 | `#1e6e50` |  |
| 42 | `broadleaf_forest` | broadleaf forest | forest | 10 | `#287828` |  |
| 43 | `needleleaf_forest` | needleleaf forest | forest | 10 | `#145032` |  |
| 44 | `mixed_forest` | mixed forest | forest | 10 | `#23642d` |  |
| 45 | `woodland` | woodland (open trees) | forest | 10 | `#5a8c3c` |  |
| 52 | `savanna` | savanna (grass + trees) | vegetation | 9 | `#beb45a` |  |
| 53 | `steppe` | steppe grassland | vegetation | 8 | `#bebe78` |  |
| 54 | `desert_scrub` | desert scrub | vegetation | 9 | `#b4a06e` |  |
| 55 | `maquis` | maquis / chaparral | vegetation | 9 | `#78823c` |  |
| 56 | `alpine_meadow` | alpine meadow | vegetation | 8 | `#82b46e` |  |
| 57 | `polygon_tundra` | polygon tundra | vegetation | 15 | `#969b7d` |  |
| 58 | `bog` | bog / peatland | wetland | 14 | `#6e6e50` |  |
| 59 | `marsh` | marsh / reed | wetland | 14 | `#5a966e` |  |
| 60 | `burn_scar` | burn scar | disturbed | 16 | `#322823` |  |
| 61 | `clear_cut` | clear-cut / regrowth | disturbed | 9 | `#aa9664` |  |
| 70 | `rice_paddy` | rice paddy | agriculture | 11 | `#78c8aa` |  |
| 71 | `orchard` | orchard / grove | agriculture | 11 | `#6ea03c` |  |
| 72 | `vineyard` | vineyard | agriculture | 11 | `#8c5a8c` |  |
| 73 | `plantation` | plantation | agriculture | 11 | `#468232` |  |
| 74 | `pasture` | pasture | agriculture | 8 | `#aad264` |  |
| 75 | `greenhouse` | greenhouse | agriculture | 12 | `#dce6f0` | glint 0.6, emissive 0.3 |
| 76 | `fallow` | fallow / ploughed | agriculture | 11 | `#966e46` |  |
| 77 | `hedgerow` | hedgerow / shelterbelt | forest | 10 | `#3c7832` |  |
| 78 | `farmyard` | farmyard | built | 17 | `#be8c6e` |  |
| 80 | `residential` | residential | built | 17 | `#dc826e` |  |
| 81 | `commercial` | commercial / CBD | built | 17 | `#e6505a` |  |
| 82 | `industrial` | industrial | built | 17 | `#aa82aa` |  |
| 83 | `building_tall` | building, tall (> 30 m) | built | 12 | `#961e28` | glint 0.3 |
| 84 | `park` | park / urban green | vegetation | 8 | `#64c864` |  |
| 85 | `sports_field` | sports field / stadium | built | 17 | `#a0d28c` |  |
| 86 | `parking` | parking / paved | built | 17 | `#787878` |  |
| 87 | `solar_farm` | solar farm | built | 17 | `#28325a` | glint 0.7 |
| 88 | `port` | port / dock | built | 17 | `#646e8c` |  |
| 89 | `cemetery` | cemetery | built | 17 | `#78966e` |  |
| 90 | `quarry` | quarry / mine | bare | 16 | `#c8aa96` |  |
| 100 | `motorway` | motorway | transport | 13 | `#e67828` |  |
| 101 | `road_major` | road, major | transport | 13 | `#f0b450` |  |
| 102 | `road_minor` | road, minor / street | transport | 13 | `#5a5a5a` |  |
| 103 | `track` | track (unpaved) | transport | 13 | `#96785a` |  |
| 104 | `railway` | railway | transport | 13 | `#6e3c6e` |  |
| 105 | `runway` | runway | transport | 13 | `#282832` |  |
| 106 | `taxiway` | taxiway / apron | transport | 13 | `#50505f` |  |
| 107 | `bridge` | bridge | transport | 13 | `#c8c83c` |  |
| 108 | `dam` | dam | built | 12 | `#a0a0b4` |  |
| 110 | `seasonal_snow` | seasonal snow | snow-ice | 7 | `#ebf0fa` |  |

**In sequence files:** `output.landcover` (`v2` by default, `legacy`, `group`) chooses the values
the `landcover` datasets hold. Their attributes describe the values written:

| attribute | |
|---|---|
| `class_names` | comma-separated names indexed by value (empty where no class has that value) |
| `class_groups` | comma-separated group name of each value |
| `class_legacy` | u8 array: the legacy class (0–17) of each value (not written for `group`) |
| `class_mapping` | `v2`, `legacy` or `group` |
