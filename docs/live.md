# Live renderer (`terrain live`)

`terrain live` flies a camera of the scenario in realtime. Every frame is rendered by the
dataset renderer and developed by the camera's sensor model, the same code that writes the
sequence files. So the window shows what a dataset frame from that pose and time looks like:
- lighting from the scenario: sun / moon for the date and time, twilight, night lights,
  shadows, stars;
- the atmosphere and water glint;
- the sensor: auto exposure, noise, optics, tone curve.

`terrain explore` is different: a globe with its own lightweight shader, for browsing the tile
store (`docs/explorer.md`).

```sh
terrain live -c configs/live.yaml --dynamic              # 1280×720, ~14 frames/s
terrain live -c configs/live.yaml --dynamic --scale 0.5  # 640×360, ~40 frames/s
```

`configs/live.yaml` uses the seed-1 world and the explorer's store (`out/explorer/world.h5`).
Its camera is a 1280×720 forward camera tilted 15° down, with a clock sun on 21 June at
07:30 UTC. Any scenario with an `rgb` camera works: its `render` settings, lighting and sensor
are used.

## Controls

| mode | keys and mouse |
|---|---|
| free flight | **W/S** forward / back · **A/D** sideways · **Space / C** up / down · drag: look · **Shift** ×5 · **Ctrl** ×0.2 |
| plane | always flying · **W/S** nose down / up · **A/D** roll (banked turns) · **Q/E** rudder · **Shift / Ctrl** throttle · drag: look |
| both | **P**: switch free flight / plane |

The panel has:
- the speed;
- *Camera looking forward*: a 15° down forward mount instead of the scenario's mounting (a
  nadir camera looks straight down);
- background tile generation on/off;
- a time-of-day offset (± 12 h; with `lighting.mode: clock`), which shows dawn, dusk and the
  night lights and stars;
- live statistics: resolution, frames/s, render and frame time, exposure, sun elevation,
  lights, tiles generated and wanted.

## Options

| option | default | meaning |
|---|---|---|
| `-c`, `--seed`, `-j` | | scenario, seed override, threads |
| `--camera PATH` | first camera with `rgb` | the scenario camera to fly |
| `--forward` | off | start with the forward mount |
| `--scale F` | 1 | render at F × the camera's resolution (pinhole-type models) |
| `--supersample N` | 1 | supersampling per axis |
| `--dynamic` | off | generate the tiles the view selects in the background |
| `--plane` | off | start as a plane |
| `--start lat,lon,agl` | home, 300 m | start position: lat, lon (deg), height above the ground (m) |

## How it works

- **Three threads:**
  - the UI flies the camera body (heading / pitch / roll relative to the local horizon) and
    shows the latest frame;
  - a render thread renders the newest pose with `render::Renderer` (GPU backend when there
    is one);
  - a sensor thread develops frame *k* while frame *k* + 1 renders.
  The controls stay at the display rate whatever the render rate.
- **Frame:** `Renderer::render` (terrain, sky, lights, stars) → `Sensor::develop` (auto exposure
  from the previous frames' metering, noise, optics, tone curve). The camera pose comes from
  the body pose and the camera extrinsics. The lighting time is the scenario clock plus the
  slider offset; the auto exposure runs on wall-clock time.
- **Tiles:**
  - The renderer uses stored tiles only, so a frame never waits for generation.
  - Every 0.3 s the render thread runs the renderer's own tile selection with an oracle in
    which every tile to `tiles.max_zoom` exists. The tiles it wants that are not stored, with
    their missing ancestors (the renderer descends into a tile only through stored parents),
    are generated coarse to fine in the background. They show up as they are written.
  - The terrain height under the camera comes from the finest stored tile.
- **Radiance-only read-back:** `Renderer::radiance_only` makes the GPU backend skip reading
  back depth, 3D points and land cover. At night they are still read back, since the stars
  mask the sky with them.

## Performance (RTX 2080 Ti, Xeon W-2125)

| resolution | frames/s | render | frame (render + develop) |
|---|---|---|---|
| 1280×720 | ~14 | ~55 ms (GPU 20 ms + read-back 20–40 ms) | ~105 ms |
| 640×360 | ~42 | ~24 ms | ~35 ms |

The limits are the GPU → CPU read-back and the CPU sensor development (bloom, chromatic
aberration, noise, tone), which compete for the cores. A GPU sensor writing straight into the
displayed texture, with no read-back, would make 720p run at ≥ 30 frames/s.

## Not in the live view

- motion blur (one sample per frame) and the ground truth modalities (depth, flow, land cover,
  events, IMU);
- tiles that are not generated yet: the view uses coarser stored tiles until they arrive;
- no recording: for a sequence, write the trajectory and use `terrain render`.
