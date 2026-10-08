# Showcase video

A promotional video of aerialsynth, fully reproducible from this folder:

```
cargo build --release
python showcase/make_showcase.py          # renders every shot, writes out/showcase/showcase.mp4
```

| file | what |
|------|------|
| `base.yaml` | base scenario of every shot: one world (seed 1), one shared tile store, 1280×720 camera, sensor look |
| `storyboard.yaml` | the shots: caption (bottom left), description (bottom right), duration, layout and the scenario overrides (deep-merged onto `base.yaml`) |
| `make_showcase.py` | renders the shots with `terrain run`, composes the video (captions, cross-fades, title collage, 2×2 panels, tile map, outro) and pipes it into ffmpeg (H.264) |
| `fonts/` | Noto Sans Light / Medium (SIL Open Font License, `fonts/OFL.txt`) |

Everything is rendered from generated tiles — there is no imagery or elevation data anywhere.
All shots share one planet: the places (farmland at 39.9°N 32.8°E, the snow-capped massif at
39.2°N 33.8°E, the coast near 40.3°N 36.9°E, …) are where seed 1 put them.

## Shots

| layout | shows |
|--------|-------|
| single | farmland, river valley, mountains, town, coast, forest, golden hour, day→night time-lapse, night towns, full moon, 10 km cruise, 60 m treetop flight, steep turns, vibration motion blur, 200° fisheye |
| grid | camera models from one pose (pinhole, distorted pinhole, Kannala–Brandt 190°, Mei 200°); a four-camera rig (nadir, forward, left / right oblique) |
| modalities | RGB, depth (normalised to the frame's maximum), optical flow (Middlebury colour wheel), events (10 ms, ON red / OFF blue) of one camera |
| globe | the opening: the planet turning through its map layers (surface, elevation, land cover), then a dive through the tile pyramid (borders coloured by zoom) down to the first shot; a keyframed map flight (`globe/dive.yaml`) recorded by `terrain view --record` into its own tile store |
| map | a sped-up flight next to a 2D mosaic of its XYZ tiles: whole trajectory, current position, camera footprint (from depth), the planned LOD tiles coloured by zoom, altitude profile |

## Options

```
python showcase/make_showcase.py --stills          # framing check: 3 small frames per shot → out/showcase/stills.png
python showcase/make_showcase.py --only coast      # (re)render one shot
python showcase/make_showcase.py --compose-only    # re-compose from rendered shots (captions, timing)
python showcase/make_showcase.py --title-preview   # title-card frames from the stills
```

Each shot is rendered into `out/showcase/<id>/` (`scenario.yaml`, `traj.csv`, `seq.h5`, log) and
is re-rendered only when its resolved scenario changes, so editing a caption only needs
`--compose-only`.

Requirements: Python 3 with numpy, h5py, pyyaml, pillow, matplotlib; ffmpeg with libx264.
Rendering everything takes a few hours on 8 cores, most of it generating the tiles of the
horizon views; the tile store is shared, so later shots reuse the tiles of earlier ones.
