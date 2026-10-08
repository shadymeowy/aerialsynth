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
/levels/<z>/landcover    u8  [N,256,256]    class id (terragen::landcover::NAMES)
/levels/<z>/emission     u8  [N,256,256,3]  night lights, linear = 16 (v/255)^3
```

- **Chunks:** one tile per chunk, compressed with shuffle + deflate. Compression runs in parallel
  and is written with HDF5 direct chunk I/O. The bundled HDF5 is 2.2.0 (via `hdf5-metno-sys`).
- **Readers:** files open in h5py or any HDF5 reader.
- **Growth:** rows can be appended in any order, so a store grows lazily.
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
`exposure_time_s,gain,ev`), and `landcover` its `class_names`.

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
    landcover                u8 [N,H,W] (255 = sky)                                    ← landcover
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
- **camodocal:** its H5 reader expects the M3ED layout (`/ovc/left/data`, `/ovc/ts`, ...). The
  data types are the same (i64 µs timestamps, u16/i64/i8 events, f64 calibration); the reader
  needs to take the group / dataset names from its config, and to read RGB or set `rgb.gray`.
- **Compression:** `output.compression` sets the deflate level (shuffle + gzip, as in h5py) and
  `float_keep_bits`, an optional lossy rounding of depth/flow mantissas. 16 bits gives a max
  relative error of 7.6e-6 and shrinks depth/flow by ~35–40%. Flow is exactly recomputable from
  depth + poses (`scripts/check_gt.py`), so omitting `flow` roughly halves the file.
- **Validation:** the scripts print PASS / FAIL per check and exit non-zero when a check fails
  or nothing could be checked; their tolerances are in their docstrings.
  - `scripts/check_gt.py` checks every camera: flow against depth reprojected with the poses
    (agrees to ~5e-6 px median, ~3e-5 px max), photometric warping (~2 DN: sensor noise plus
    motion blur), and camera poses against `/pose` ∘ `T_body_cam` (interpolated for frames
    between `/pose` samples).
  - `scripts/check_imu.py` rebuilds the IMU truth from `/pose` independently (needs `/pose`
    finer than the IMU, e.g. `configs/examples/imu_check.yaml`) and checks white noise and bias
    walk against the configured densities. The gyro truth agrees to 3e-10 rad/s, the accel
    truth to 1e-4 m/s² without a lever arm. The example's lever arm (12 cm) under the default
    engine vibration (38 Hz and its 76 Hz harmonic) adds ~5.8 m/s² RMS, which the simulation
    step discretizes: 0.27 m/s² (~5%) off at the 1 ms step, ~0.2% at 0.2 ms.
  - `scripts/check_events.py` checks the event format invariants and prints rate statistics.
