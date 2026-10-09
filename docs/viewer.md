# The viewer (`terrain view`)

One window, two views of the scenario's world, flown in realtime:

- **Map:** the tile store as a globe. Browse the planet, see what is stored, and generate as
  you go. It draws with its own light shader (lit albedo, simple haze and sky).
- **Camera:** a scenario camera rendered by the dataset renderer and developed by its sensor
  model, live. You get the lighting for the date and time (sun, moon, twilight, night lights,
  stars), the atmosphere and shadows, auto exposure, noise and the tone curve: what a dataset
  frame from that pose and time looks like.

To look at a dataset `terrain run` wrote, see `terrain show` ([`show.md`](show.md)).

```sh
terrain view -c configs/view.yaml                  # the map: orbit the planet (M: camera)
terrain view -c configs/view.yaml --fly free       # fly over home on the map
terrain view -c configs/view.yaml --camera-view    # fly the camera through the dataset renderer
```

Both views share one flight and one tile store. By default the tiles a view wants are
generated in the background and stored (`--no-generate`: stored tiles only, the store opened
read-only, so a store of another generator version can be viewed). On start, levels
z0..=`--base-zoom` (4) are completed for the whole planet (341 tiles; low zooms are the
slowest tiles to generate, see `docs/gpu.md`): z0–z2 first (the first frame waits for z0
only), then the tiles the view wants, and the rest of the base levels whenever the view wants
nothing generated. `--base-zoom` is at most 8 (87,381 tiles) and `tiles.max_zoom`. While the
GPU generator's pipelines compile (the first run on a machine, ~30 s), tiles are generated on
the CPU.

The viewer needs a GPU (wgpu: Vulkan, Metal or DX12). Run it on the machine's own display;
over SSH X forwarding it runs, but slowly.

## Controls

| | |
|---|---|
| **M** | switch Map / Camera |
| **F** | next flight mode: orbit (map only) → free flight → plane |
| free flight | **WASD** move · **Space / C** up / down · drag: look · scroll: speed · **Shift** ×5 · **Ctrl** ×0.2 |
| plane | always flying · **W/S** nose down / up · **A/D** roll (banked turns) · **Q/E** rudder · **Shift / Ctrl** throttle · drag: look |
| orbit | drag: move · right drag: turn / tilt · scroll: zoom · double click: fly there |

- **Top bar:** the view (Map / Camera), the flight mode and speed, the frame rate and
  generation activity.
- **Side panel:**
  - Map: shading (surface, elevation, land cover, relief) and relief exaggeration; a *Display*
    group with the level-of-detail bias, finest level, tile borders and light; and, when
    orbiting, *Whole planet / Home / North up / Top down*.
  - Camera: the camera and its resolution, *Look forward* (15° down instead of the scenario's
    mounting), a time-of-day offset (± 12 h, with `lighting.mode: clock`), and frame rate,
    exposure, sun elevation and lights.
  - Below: *Generate missing tiles*, then the tile statistics and the deepest generated level,
    and a key reference.
- **Readout:** the view shows speed, height above the ellipsoid and the ground, attitude and
  position. The body cannot go below the terrain.

## Options

| option | default | |
|---|---|---|
| `--camera-view` | | open the camera view |
| `--fly free\|plane` | | start flying instead of orbiting |
| `--start lat,lon,agl` | home, 600 m | where to start (deg, deg, m above the ground) |
| `--no-generate` | | stored tiles only (read-only) |
| `--max-zoom`, `--base-zoom` | `tiles.max_zoom`, 4 (at most 8) | deepest generated level (map and camera; the camera renders no deeper than `tiles.max_zoom`); levels completed on start |
| `--camera PATH` | first camera with `rgb` | the scenario camera; without one, a 960×540 forward camera |
| `--scale F` | ≤ 960 px wide | camera resolution factor (pinhole-type models) |
| `--supersample N` | 1 | camera supersampling |
| `--exag`, `--mode` | 1, surface | map relief exaggeration and shading |
| `--gpu-tiles` | 1536 | map tiles on the GPU (768 KB each) |
| `--snapshot PNG --view … --wait S --size WxH` | `--view 20,10,16000,0,0`, `--wait 600`, `--size 1280x800` | a headless map view into a PNG (below) |
| `--record DIR --path FILE --fps N --until S` | 25 fps | a headless keyframed map flight into PNG frames (below) |

## How it works

Code: `crates/viewer`:
- `lib.rs`: the app and its options;
- `globe.rs` + `globe.wgsl`: the map renderer;
- `camera.rs`: the camera view;
- `fly.rs`: the flight;
- `tiles.rs`: the tile service;
- `snapshot.rs`: headless snapshots and recordings.

It uses wgpu 30 (shared with the dataset renderer) and egui / eframe 0.36.

- **Flight (`fly.rs`):**
  - A position (ECEF) with heading / pitch / roll relative to the local horizon, so level
    flight stays level anywhere on the planet.
  - The plane is arcade-style: stick rates, wings that level themselves, a coordinated turn
    rate g·tan(bank)/V, and a throttle.
  - Ground clearance is 2 m: under the map, from the finest resident tile (exaggerated as
    drawn); under the camera, from the finest stored tile's DSM.
- **Tile service (`tiles.rs`):**
  - Two loader threads read tiles from the store. A generator thread generates in batches of
    half the cores and writes them to the store: base levels z0–z2 first (one level per
    batch), then the view's wishes, then the deeper base levels.
  - The GPU generator is set up on a background thread; the CPU generates until it is ready.
  - The active view replaces the wish lists every frame.
- **Map (`globe.rs`):**
  - **Level of detail:** a quadtree from z0, with frustum and horizon culling. A tile is
    refined while its texels look larger than *pixels per texel*, and only if a finer tile can
    be had.
  - **No holes:** a tile that has not arrived is drawn with the region of its nearest resident
    ancestor. Requests are ordered by how blurry that stand-in looks on screen, so the ground
    near the camera sharpens first.
  - **Geometry:**
    - Each tile is a 33×33 grid (+ skirts) relative to its own centre in f64, drawn
      camera-relative, so it is precise from orbit to a few metres. Reversed-Z infinite
      projection.
    - Heights are box-filtered over the grid spacing. The full-resolution slope shades,
      capped at ~50° so building walls do not turn into needles.
  - **Residency:** texture arrays with least-recently-used eviction.
  - **Sky:** a full-screen pass gives the limb glow, a sky near the ground and the stars.
    Polar caps cover latitudes beyond the Mercator limit.
- **Camera (`camera.rs`):**
  - **Threads:** a render thread renders the newest pose with `render::Renderer`; a sensor
    thread develops frame *k* while frame *k* + 1 renders (the UI stays at the display rate).
  - **Read-back:** `Renderer::radiance_only` skips reading back depth, points and land cover
    (the stars still need them at night).
  - **Tiles:** the renderer uses stored tiles only. Every 0.3 s its own tile selection runs
    with an oracle in which every tile exists. The tiles it would want that are not stored,
    with their missing ancestors, are generated coarse to fine.
  - **Time:** the lighting runs on the scenario clock plus the slider offset; the auto exposure
    runs on wall-clock time.
  - **Not drawn:** motion blur and the ground truth modalities. To record, write a trajectory
    and use `terrain run`.

## Snapshots

`--snapshot` renders a map view without a window, once the view's tiles are in (or after
`--wait` seconds). `--view` is either `lat,lon,km,heading,tilt` (orbit) or
`fly:lat,lon,agl_m,heading,pitch`. It reports when the first tiles were drawn and when the
view's tiles were all in (`TERRAGEN_PROFILE=1` also times each generated batch).

```sh
terrain view -c configs/view.yaml --snapshot out/fly.png --view fly:39.9,32.8,600,0,-12 --wait 45 --size 1280x720
```

## Recordings

`--record DIR --path FILE` flies the map camera along keyframes and saves every frame
(`DIR/frame_00000.png`, …, at `--fps`, `--size`), each once its tiles are in (or after `--wait`
seconds). `DIR/frames.csv` gives each frame's time, the camera pose as a trajectory (`lat`,
`lon`, `h`, and `roll`, `pitch`, `yaw` of a forward-looking body, the columns `terrain run`
reads), the distance to the target and the finest zoom level drawn: the dataset renderer can fly
the same path. The showcase opens with one (`showcase/globe/dive.yaml`), handed over to the
dataset renderer 6 km above the ground. `--until S` saves the frames up to S seconds only; the
camera path in `frames.csv` goes on to the last key.

```yaml
keys:     # the orbit camera: target lat / lon (deg), distance (km), heading / tilt / fov (deg), exag
  - { t: 0,  lat: 20, lon: 40,   km: 15000 }
  - { t: 8,  lat: 10, lon: -70,  km: 13000 }
  - { t: 20, lat: 7,  lon: -102, km: 3, heading: 90, tilt: 63 }
look:     # switches of the shading mode and tile borders, dissolving over `fade` seconds
  - { t: 0,   mode: surface }
  - { t: 2.2, mode: elevation, fade: 0.7 }
  - { t: 8.2, borders: true, fade: 0.6 }
```

- **Keys:** interpolated with Catmull-Rom splines: the position as a unit vector, the distance
  logarithmically (an even zoom from orbit to the ground), easing in at the first key and out at
  the last. Omitted: heading and tilt 0 (north up, straight down), exaggeration 1, fov 40°.
- **Look:** a switch holds until the next one that sets the same thing.
- **Target height:** the orbit target sits on the z12 DSM (bilinear; tiles the store lacks are
  generated in memory), faded in below 400 km: the recorded path is the same in every run,
  whatever tiles have streamed in. Looking straight down, `frames.csv` gives roll 0 and the yaw
  of the image's up direction.

```sh
terrain view -c configs/view.yaml --record out/dive --path showcase/globe/dive.yaml --size 1920x1080
```

## Performance

On the author's machine (RTX 2080 Ti, Xeon W-2125):

| | |
|---|---|
| map | ~16–17 ms per frame (vsync) at 1200×950, ~600 patches in low flight |
| camera, 640×360 | ~42 frames/s (render ~24 ms, frame ~35 ms) |
| camera, 960×540 (default) | ~21 frames/s |
| camera, 1280×720 | ~14 frames/s (render ~55 ms: GPU 20 + read-back 20–40) |
| generation | ~6.5 tiles/s in low flight with the CPU generator (z13–z17); z5–z12 are slower |

- **Camera limits:** the GPU → CPU read-back and the CPU sensor (bloom, chromatic aberration,
  noise, tone). A GPU sensor writing straight into the displayed texture would make 720p run
  at ≥ 30 frames/s.
- **Unexplored ground:** distant terrain stays coarse for the first minute or so of low flight,
  until its tiles are generated.
