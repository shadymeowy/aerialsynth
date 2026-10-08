# Scenarios

One YAML file drives everything (`-c` of `run`, `tiles`, `view` and `config`). `terrain config` prints a commented
template of the main settings; `terrain config --all` lists every setting with its default;
`terrain config -c my.yaml` shows a scenario with its defaults filled in (and checks it).

```
world        the planet: seed, continents, relief, rivers, climate, vegetation, land use
tiles        the tile store: file, zoom range, generation while rendering
trajectory   the flight: a CSV, synthesized from `synth` (path, speed, wind, vibration, gimbal)
render       backend, supersampling, level of detail, lighting, atmosphere, stars
cameras      any number of cameras: intrinsics, mounting, modalities
imu          an IMU (omit for none)
output       the sequence file: body pose rate, time window, PNG export, compression
```

- **Optional:** every section and nearly every key is optional; a camera needs its `path` and
  `intrinsics` (model, width, height, intrinsics). Unknown keys are errors, so a typo is not
  silently ignored.
- **Units** are in the key: `altitude_m`, `speed_mps`, `duration_s`, `rate_hz`, `tau_s`,
  `visibility_km`, `sun_elevation_deg`, … The exceptions are `lat` / `lon` (degrees) and
  dimensionless factors.
- **Zoom range:** `tiles.min_zoom` / `max_zoom` is the zoom range of everything (planning,
  generation, rendering). `render.texel_px` sets the level of detail within it; planning
  refines to 0.8× that.
- **Paths:** relative paths are relative to the working directory.

Examples: `configs/quick.yaml` (10 s smoke test), `configs/dataset.yaml` (60 s with wind,
vibration, a forward camera, clock sun, sensor model, IMU), `configs/view.yaml` (for
`terrain view`), and `configs/examples/` (night, full moon, event camera rig, star tracker,
fisheye, 10 km cruise, sunset, IMU check).

## Cameras and modalities

`cameras` is a list. Each camera has:
- intrinsics (camodocal schema);
- extrinsics;
- an HDF5 group `path`;
- the modalities it produces, as subsections. An omitted subsection means the camera does
  not produce that output; `{}` turns it on with defaults.

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

- **Geometry-only cameras:** cameras without `rgb` are rendered geometry-only (no shading), so
  depth / flow for an event camera cost little.
- **Supersampling:** cameras with depth / flow / land cover need an odd `render.supersample`
  (the ground truth is the central sub-sample).
- **Group paths** are free-form; nothing assumes a particular dataset layout.
- **Camera models:** `pinhole`, `pinhole_full` (OpenCV rational), `kannala_brandt`, `mei`,
  `scaramuzza`. Fields of view beyond 180° are supported.
- **More:** star ground truth: `stars: {}` (`docs/stars.md`); event cameras: `docs/events.md`.

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
  - **Body:** FRD. Attitude `q_ned_body` (Hamilton, `[w,x,y,z]` in files) maps body vectors
    into the local NED frame at the body.
  - **Camera:** OpenCV (x right, y down, z forward). Pixel centres sit on integer coordinates.
  - **Extrinsics:** a camera's `extrinsics` give the mount (`nadir`/`forward`), roll/pitch/yaw
    offsets and the translation in the body frame, or an explicit `q_body_cam`.
- **Trajectory CSV:** columns are matched by name. Supported layouts:
  - `t,lat,lon,h,qw,qx,qy,qz`
  - `t,lat,lon,h,roll,pitch,yaw`
  - `t,x,y,z,qw..` (ECEF, q = body→ECEF)
  - `t,n,e,d,qw..` with a `# origin: lat,lon,h` line (pymap3d `ned2geodetic` semantics)

## Look and colour

| knob | effect |
|------|--------|
| `world.albedo.saturation`, `brightness` | the generated surface colours (every layer) |
| `world.satellite.*` | sun, ambient, haze of the baked `rgb` tile layer only (camera images use `render.lighting`) |
| `render.lighting` | `mode: clock` (sun / moon from `date`, `time_utc`), `lights_intensity`, `moon_intensity`, `night_sky`, `light_pollution` |
| `render.atmosphere.visibility_km`, `inscatter` | haze |
| `cameras[].rgb.sensor.exposure.target` | overall brightness |
| `cameras[].rgb.sensor.tone` | `saturation`, `white_balance`, `curve` (`srgb`/`filmic`/`gamma`) |
