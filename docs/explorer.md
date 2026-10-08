# Globe explorer and realtime flight (`terrain explore`)

`terrain explore` opens the scenario's tile store (`tiles.file`) as a globe. You can orbit the
whole planet, or fly over it in realtime with free flight (WASD) or a simple plane. Tiles stream
in from the store as you move; with dynamic generation on, missing tiles are generated in the
background and written to the store, so the store grows wherever you go.

```sh
terrain explore -c configs/explore.yaml --dynamic             # orbit the planet
terrain explore -c configs/explore.yaml --dynamic --fly free  # fly over home (WASD)
terrain explore -c configs/explore.yaml --dynamic --fly plane # fly a plane over home
```

`configs/explore.yaml` is the seed-1 world with its store at `out/explorer/world.h5`. Any
scenario works: the explorer uses its `world`, `tiles.file` and `tiles.max_zoom`.

Run it on the machine's own display. On this machine the local X server is `:1`
(`DISPLAY=:1`). Over SSH X forwarding it runs, but every frame goes through the forwarded X
server and is slow.

## Controls

| mode | keys and mouse |
|---|---|
| all | **F**: next mode (orbit → free flight → plane → orbit), starting from the current view |
| orbit | drag: move · right drag: turn / tilt · scroll: zoom · double click: fly to the point |
| free flight | **W/S** forward / back · **A/D** sideways · **Space / C** up / down · drag: look · scroll: speed · **Shift** ×5 · **Ctrl** ×0.2 |
| plane | always flying forward · **W/S** nose down / up (stick) · **A/D** roll (the bank turns the plane) · **Q/E** rudder · **Shift / Ctrl** throttle · drag: look |

- **Readout:** the top-left of the view shows the mode, speed, height above the ellipsoid and
  above the ground, heading, pitch, roll and position.
- **Ground:** the camera cannot go below the terrain (2 m clearance). A plane that touches the
  ground pulls its nose up.
- **Panel controls:**
  - camera mode and speed (km/h);
  - generation on/off and the deepest level generated;
  - view mode: surface (lit albedo), elevation (hypsometric tint), land cover (class colours),
    relief (shading only);
  - relief exaggeration (1–100×);
  - level-of-detail bias (pixels per texel) and the finest level shown;
  - tile borders coloured by level;
  - light from the upper left of the view, or a fixed sun.
- **Panel readouts:** tiles stored per level, generation rate, the load / generate queues,
  GPU tiles in use and the frame time.

## Options

| option | default | meaning |
|---|---|---|
| `-c`, `--seed`, `-j` | | scenario, world seed override, threads (as every subcommand) |
| `--dynamic` | off | start with *Generate missing tiles as you fly* on |
| `--max-zoom` | `tiles.max_zoom` | deepest level generated dynamically |
| `--base-zoom` | 4 | levels 0..=this are completed for the whole planet on start |
| `--fly free\|plane` | | start in the air: 600 m above the ground at the world's home, heading north |
| `--gpu-tiles` | 1536 | tiles kept on the GPU (768 KB each, ~1.2 GB; capped by the GPU's array-layer limit) |
| `--exag`, `--mode` | 1, surface | relief exaggeration and view mode at start |
| `--snapshot PNG` | | render one view headless into a PNG and exit (below) |
| `--view` | `20,10,16000,0,0` | snapshot view: `lat,lon,km,heading,tilt` (orbit) or `fly:lat,lon,agl_m,heading,pitch` |
| `--wait S` | 600 | snapshot: capture when the view's tiles are in, or after S seconds |
| `--size WxH` | 1280x800 | snapshot size |

## Store and world

- **One world per store:** the store must hold the scenario's world, the same rule as `gen` and
  `render`. A store of another seed or setting is refused with the settings that differ, and
  so is a store written by another generator version (new tiles would not match its old
  ones). A missing store is created.
- **Base levels:** on start, z0..=`--base-zoom` is completed for the whole planet in the
  background, while the view shows what exists. A progress bar shows it. z0–z4 is 341 tiles,
  about 4 minutes on 8 threads.
- **Dynamic generation:** with it on, the tiles the view wants that are not in the store are
  generated, down to `--max-zoom`, and written to the store. With it off, only stored tiles
  (and the base levels) are shown, and detail stops where the store does.

## How it works

Code: `crates/explorer` (`lib.rs` app and options, `globe.rs` renderer and level of detail,
`globe.wgsl` shaders, `tiles.rs` tile service, `fly.rs` flight camera). wgpu 30 (shared with
the renderer) with egui / eframe 0.36.

- **Tile service (`tiles.rs`):**
  - Two loader threads read tiles from the store (albedo, elevation, land cover). A generator
    thread generates missing tiles in batches of half the cores (each tile is itself parallel
    over its rows) and writes them to the store. The base levels go first.
  - Each tile becomes a GPU payload: colour (albedo + class), the elevation box-filtered over
    the mesh spacing, and the full-resolution slope (Rg16F) for shading.
  - The view replaces its wish lists every frame; workers take the first tiles not in flight.
- **Level of detail (`globe.rs`):**
  - A quadtree walk from z0, with frustum and horizon culling.
  - A tile is refined while its texels look larger than `pixels per texel` on screen, and only
    if a finer tile can be had: resident on the GPU, in the store, or generated.
  - A tile whose data has not arrived is drawn with the matching region of its nearest
    resident ancestor, so there are no holes while tiles stream in.
- **Request order:** the tile whose stand-in looks blurriest on screen comes first. Blur is
  the ancestor's texel size seen from the camera; tiles with nothing to draw come first of all.
  This sharpens the ground near the camera within seconds. Ordering by (zoom, distance)
  instead spent the first minutes on slow, distant, low-zoom tiles.
- **Geometry:**
  - Each tile is a 33×33 vertex grid (+ a skirt against cracks between levels) relative to its
    own centre in f64, drawn with the centre minus the eye, so positions are camera-relative
    and precise from orbit to a few metres.
  - Reversed-Z infinite projection.
  - Elevation (exaggerated) is displaced in the vertex shader from the box-filtered height.
    The fragment shader shades with the full-resolution slope, capped at ~50° so building walls
    do not turn into needles.
- **Residency:** colour, height and slope live in three texture arrays, one layer per tile.
  Least-recently-used tiles are evicted (z0–z2 stay).
- **Sky:** a full-screen pass gives the limb glow from orbit, a sky near the ground and a star
  field. Aerial perspective follows the path through the lower atmosphere, so the disc stays
  clear from orbit and the limb hazes. Polar caps cover the latitudes beyond the Mercator limit
  (±85.05°).
- **Flight camera (`fly.rs`):**
  - It keeps a position (ECEF) and heading / pitch / roll relative to the local horizon, so
    level flight stays level anywhere on the planet.
  - The plane is arcade-style: stick rates, wings that level themselves, a coordinated turn
    rate g·tan(bank)/V, and a throttle.
  - Terrain height under the camera comes from the finest resident tile (a 32×32 copy of each
    tile's heights).

## Snapshots (headless)

`--snapshot` renders a view without a window: with `--view fly:...`, from the flight camera.
It is used to check views: whole planet, oblique, z17 close-ups, low flight.

```sh
terrain explore -c configs/explore.yaml --dynamic --snapshot out/fly.png \
    --view fly:39.9,32.8,600,0,-12 --wait 45 --size 1280x720
```

The capture waits until nothing the view wants is loading, generating or uploading, or for
`--wait` seconds. A low, level view sees the horizon and wants hundreds of tiles: from an
unexplored start it does not settle for minutes. `--wait 45` shows what the window shows after
45 s of flight.

## Performance (RTX 2080 Ti, Xeon W-2125 8 threads)

| | |
|---|---|
| frame time | ~16–17 ms at 1200×950 (vsync), 600 patches in low flight |
| generation | ~6.5 tiles/s while flying low (z13–z17), slower for z5–z12 |
| descent from orbit to z16 at 30°N 20°E | 433 tiles generated in 143 s |
| spawn over unexplored home | near field at z17 after ~45 s (96 tiles) |

## Limitations

- **Unexplored ground:** the distance stays coarse for the first minute or so of low flight,
  since low-zoom tiles are slow to generate. Normal flying speeds are kept up with.
- **Simplified lighting:** no cast shadows, no night lights (the emission layer is not used),
  no clouds. The plane is not a flight model.
- **Shading:** single buildings and trees shade as bumps (the mesh height is box-filtered;
  slopes are capped).
