# Showcase video

A promotional video of aerialsynth, fully reproducible from this folder:

```
cargo build --release
python showcase/make_showcase.py          # renders every shot, writes out/showcase/showcase.mp4
~/.venv/bin/python showcase/showcase_score.py   # + its score (needs scipy) → out/showcase/showcase_music.mp4
```

| file | what |
|------|------|
| `base.yaml` | base scenario of every shot: one world (seed 1), one shared tile store, 1920×1080 camera, sensor look |
| `storyboard.yaml` | the shots: caption (bottom left), description (bottom right), duration, layout and the scenario overrides (deep-merged onto `base.yaml`) |
| `make_showcase.py` | renders the shots with `terrain run`, composes the video (captions, cross-fades, title collage, 2×2 panels, tile map, outro) and pipes it into ffmpeg (H.264) |
| `showcase_score.py` | the video's score, synthesised here (no samples): electronic, 120 bpm in A minor (kick, hats, clap, a rolling bass pumped by the kick, a pluck arpeggio, pads, bells, a lead motif) on a beat grid through the cuts; its sections follow the storyboard (quiet under the title and globe, the groove from the first landscape, a sunset drop, a drum-less night section, the peak through the steep turns, settling into the outro) |
| `fonts/` | Noto Sans Light / Medium (SIL Open Font License, `fonts/OFL.txt`) |

Everything is rendered from generated tiles — there is no imagery or elevation data anywhere.
All shots share one planet: the places (farmland at 39.9°N 32.8°E, the snow-capped massif at
39.2°N 33.8°E, the coast near 40.3°N 36.9°E, …) are where seed 1 put them.

## Shots

| layout | shows |
|--------|-------|
| single | farmland, river valley, mountains, town, coast, forest, golden hour, day→night time-lapse, night towns, full moon, 10 km cruise, 120 m low flight, steep turns, vibration motion blur, 200° fisheye |
| grid | camera models from one pose (pinhole, distorted pinhole, Kannala–Brandt 190°, Mei 200°); a four-camera rig (nadir, forward, left / right oblique) |
| modalities | RGB, depth (normalised to the frame's 98th percentile), optical flow (Middlebury colour wheel), events (10 ms, ON red / OFF blue) of one camera |
| events | the event camera of another shot full screen (`source`: that shot; `offset`: seconds into its flight, i.e. the continuation after the source's own segment, which renders `extend` seconds more for it; `window_ms`, `camera`) |
| globe | the opening: the planet turning through its map layers (surface, elevation, land cover), then a dive through the tile pyramid (borders coloured by zoom) down to the first shot; a keyframed map flight (`globe/dive.yaml`) recorded by `terrain view --record` into its own tile store |
| follows | the globe's recorded camera path flown on by the dataset renderer (`descent`): the hand-off from the map to the camera, 6 km above the ground |
| map | a sped-up flight next to a 2D mosaic of its XYZ tiles: whole trajectory, current position, camera footprint (from depth), the planned LOD tiles coloured by zoom, altitude profile |

## The long-haul flight

A second video, `out/airliner/airliner.mp4` (about 4 min): one airliner flight from a tropical
lake district on the equator to a tundra basin inside an arctic ice sheet, 8,100 km in 9 h 47 min,
shown as one continuous time-lapse.

```
python showcase/make_airliner.py            # route, render, compose → out/airliner/airliner.mp4
python showcase/make_airliner.py --stills   # framing check per camera stretch → out/airliner/stills.png
~/.venv/bin/python showcase/soundtrack.py   # sound (needs scipy) → out/airliner/airliner_sound.mp4
```

| file | what |
|------|------|
| `airliner.yaml` | the route (airports, waypoints, speeds, climb and descent), the take-off time, the globe opening, the flight timeline (playback speed along the route, camera switches, captions) and the outro |
| `route.py` | flies the route: great circles with fly-by turns, an airliner's climb / cruise / 3° descent and flare, attitude from the flight path; the airports' ground from the world (`terragen` example `ground`) → `out/airliner/route.csv` (`t,lat,lon,h,roll,pitch,yaw`, 10 Hz) |
| `make_airliner.py` | the video: video time ↔ flight time from the playback speed (real time at take-off and landing, ×20 over each place, up to ×3000 in between), one `terrain run` per camera stretch over the time-warped trajectory (`render.lighting.time_map` keeps the sun on the flight's clock), the globe with the route drawn over it, captions, the speed, a readout and a route inset |
| `globe/route.yaml` | the globe opening's keyframes |
| `soundtrack.py` | the sound, all synthesised here: an original ambient score (D dorian, 76 bpm: pads, a plucked arpeggio, sub bass, bells, reverb) whose sections follow the video, and the aircraft from the flight itself (turbofan roar, buzz-saw and whine from the thrust schedule, airflow from the indicated airspeed, gear, touchdown, reversers); the faster the playback, the more the aircraft recedes to a cabin hum; near real time the music ducks under the engines |

## Options

```
python showcase/make_showcase.py --stills          # framing check: small frames per shot → out/showcase/stills.png
python showcase/make_showcase.py --only coast      # (re)render one shot and its clip (no final video)
python showcase/make_showcase.py --compose-only    # no rendering: re-compose changed clips, assemble the video
python showcase/make_showcase.py --only coast --compose-only   # the same for one shot's clip
python showcase/make_showcase.py --full-compose    # compose every frame from the sequences instead of joining clips
python showcase/make_showcase.py --force           # re-render even if up to date
python showcase/make_showcase.py --story showcase/scout.yaml --stills   # another storyboard (candidate places)
python showcase/make_showcase.py --title-preview   # title-card frames from the stills
```

Each shot is rendered into `out/showcase/<id>/` (`scenario.yaml`, `traj.csv`, `seq.h5`, log) and
is re-rendered only when its resolved scenario changes. Its captioned clip
(`out/showcase/clips/NN_<id>.mp4`) is re-composed only when the shot's entry or the video
settings change; the video joins the clips with cross-fades. So editing a caption, a note or a
duration only needs `--compose-only`. (`scout.yaml` reuses some storyboard ids, and the output
directory is `out/showcase/<id>` for either storyboard: a scout run overwrites those shots.)

Requirements: Python 3 with numpy, h5py, pyyaml, pillow, matplotlib; ffmpeg with libx264.
Rendering everything takes a few hours on 8 cores, most of it generating the tiles of the
horizon views; the tile store is shared, so later shots reuse the tiles of earlier ones.
