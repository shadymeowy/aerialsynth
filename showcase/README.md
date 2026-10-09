# Showcase videos

Two videos of aerialsynth, both fully reproducible from this folder. Everything in them is
rendered from generated tiles: there is no imagery or elevation data anywhere, and the sound is
synthesised by the scripts here. Both are also attached to the
[v0.1.0 release](https://github.com/shadymeowy/aerialsynth/releases/tag/v0.1.0).

- **The feature showcase** ([YouTube](https://youtu.be/SkTq4ph64l4); `out/showcase/showcase_music.mp4`, 4:35): 46 short
  shots of the planet of seed 1, one feature each (landscapes, light, night sky, sensor
  modalities, camera models, tiles on demand), with an original electronic score.
  Made by `make_showcase.py` (video) and `showcase_score.py` (music).
- **The long-haul flight** ([YouTube](https://youtu.be/ByafGs-iNug); `out/airliner/airliner_sound.mp4`, 3:49): one airliner flight
  from a tropical lake district on the equator to a tundra basin inside an arctic ice sheet,
  8,100 km in 9 h 46 min, shown as one continuous time-lapse with an ambient score, engine
  sound and cockpit callouts. Made by `route.py` (trajectory), `make_airliner.py` (video) and
  `soundtrack.py` (sound).

## Requirements

```
cargo build --release                       # the terrain binary (target/release/terrain)
pip install -r showcase/requirements.txt    # numpy, h5py, PyYAML, Pillow, matplotlib, scipy, piper-tts
```

- **ffmpeg** and **ffprobe** with libx264 (system package, e.g. `apt install ffmpeg`).
- **Piper voice**, only for the airliner's callouts: download `en_US-ljspeech-high.onnx` and
  `en_US-ljspeech-high.onnx.json` from
  [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ljspeech/high)
  into `~/.cache/piper/`, or point `PIPER_VOICE` at the `.onnx` file. The `piper` command is
  looked up next to the Python interpreter (a virtualenv's `bin/`), then on `PATH`.

Run the commands below from the repository root. The renders use the GPU backend
(`render.backend: gpu` in `base.yaml`; `cpu` gives the same output, slower).

## The feature showcase

```
python showcase/make_showcase.py      # renders every shot, writes out/showcase/showcase.mp4
python showcase/showcase_score.py     # its score → out/showcase/showcase_music.wav, showcase_music.mp4
```

Rendering everything takes a few hours, most of it generating the tiles of the horizon views;
the tile store (`out/showcase/world.h5`) is shared, so later shots reuse the tiles of earlier
ones. All shots share one planet: the places (farmland at 39.9°N 32.8°E, the snow-capped massif
at 39.2°N 33.8°E, the coast near 40.3°N 36.9°E, …) are where seed 1 put them.

### Shots

| layout | shows |
|--------|-------|
| globe | the opening: the planet turning through its map layers (surface, elevation, land cover), then a dive through the tile pyramid (borders coloured by zoom) down to the first shot; a keyframed map flight (`globe/dive.yaml`) recorded by `terrain view --record` into its own tile store |
| follows | the globe's recorded camera path flown on by the dataset renderer (`descent`): the hand-off from the map to the camera |
| single | rainforest peak, full moon, boreal forest, 10 km cruise, 120 m low flight, golden hour, day → night time-lapse, night towns, real stars and planets, a star tracker with ground-truth rings, rivers, vibration motion blur, mountains, lakes, tropics, 200° fisheye, dunes, steep turns, irrigated desert, town, coasts, taiga, glacial valley, forest, farmland |
| map | a sped-up flight next to a 2D mosaic of its XYZ tiles: whole trajectory, current position, camera footprint (from depth), the planned LOD tiles coloured by zoom, altitude profile |
| modalities | RGB, depth (normalised to the frame's 98th percentile), optical flow (Middlebury colour wheel, camera rotation removed), events (10 ms, ON red / OFF blue) of one camera |
| events | the event camera of another shot full screen (`source`: that shot; `offset`: seconds into its flight, i.e. the continuation after the source's own segment, which renders `extend` seconds more for it; `window_ms`, `camera`) |
| grid | camera models from one pose (pinhole, distorted pinhole, Kannala–Brandt, Mei); a four-camera rig (nadir, forward, left / right oblique) |

### Options

```
python showcase/make_showcase.py --stills          # framing check: small frames per shot → out/showcase/stills.png
python showcase/make_showcase.py --only coast      # (re)render one shot and its clip (no final video)
python showcase/make_showcase.py --compose-only    # no rendering: re-compose changed clips, assemble the video
python showcase/make_showcase.py --only coast --compose-only   # the same for one shot's clip
python showcase/make_showcase.py --full-compose    # compose every frame from the sequences instead of joining clips
python showcase/make_showcase.py --force           # re-render even if up to date
python showcase/make_showcase.py --story showcase/scout.yaml --stills   # another storyboard (candidate places)
python showcase/make_showcase.py --title-preview   # title-card frames from the stills
python showcase/showcase_score.py VIDEO [OUT]      # the score under another video
```

Each shot is rendered into `out/showcase/<id>/` (`scenario.yaml`, `traj.csv`, `seq.h5`, log) and
is re-rendered only when its resolved scenario changes. Its captioned clip
(`out/showcase/clips/NN_<id>.mp4`) is re-composed only when the shot's entry or the video
settings change; the video joins the clips with cross-fades. So editing a caption, a note or a
duration only needs `--compose-only`. (`scout.yaml` reuses some storyboard ids, and the output
directory is `out/showcase/<id>` for either storyboard: a scout run overwrites those shots.)
The score follows the cuts it finds in `out/showcase/clips`; under another video its sections
are spread over the video's length.

## The long-haul flight

```
python showcase/make_airliner.py            # route, render, compose → out/airliner/airliner.mp4
python showcase/soundtrack.py               # its sound → out/airliner/soundtrack.wav, airliner_sound.mp4
```

```
python showcase/make_airliner.py --stills   # framing check per camera stretch → out/airliner/stills.png
python showcase/make_airliner.py --only s03_wide   # (re)render some camera stretches (ids as printed), no video
python showcase/make_airliner.py --compose-only    # no rendering: compose and assemble
python showcase/route.py                    # only the trajectory → out/airliner/route.csv, route.json
```

The seven camera stretches render in about 1.2 h on one GPU. Each stretch flies over its own
part of the planet, so its tile store is deleted after it is rendered (`drop_tiles`; the whole
flight's tiles would take tens of GB). `make_airliner.py` builds the `terragen` example
`ground` (the airports' ground height) and runs `route.py` itself when needed.

## Files

| file | what |
|------|------|
| `requirements.txt` | the Python packages of these scripts (`pip install -r showcase/requirements.txt`) |
| `base.yaml` | base scenario of every shot of both videos: one world (seed 1), one shared tile store, 1920×1080 camera, sensor look |
| `storyboard.yaml` | the showcase's shots: caption (bottom left), description (bottom right), duration, layout and the scenario overrides (deep-merged onto `base.yaml`), the title and outro |
| `scout.yaml` | an alternative storyboard of candidate places, for `--stills` checks before they go into `storyboard.yaml` |
| `make_showcase.py` | renders the shots with `terrain run` (the globe with `terrain view --record`), composes the video (captions, cross-fades, title collage, 2×2 panels, tile map, outro) and pipes it into ffmpeg (H.264); its drawing helpers are shared with `make_airliner.py` |
| `showcase_score.py` | the showcase's score, synthesised here (no samples): electronic, 120 bpm in A minor (kick, hats, clap, a rolling bass pumped by the kick, a pluck arpeggio, pads, bells, a lead motif) on a beat grid through the cuts; its sections follow the storyboard (quiet under the title and globe, the groove from the first landscape, a sunset drop, a drum-less night section, the peak through the steep turns, settling into the outro) |
| `airliner.yaml` | the flight: the route (airports, waypoints, speeds, climb and descent), the take-off time, the globe opening, the flight timeline (playback speed along the route, camera switches, captions) and the outro |
| `route.py` | flies the route: great circles with fly-by turns, an airliner's climb / cruise / 3° descent and flare, attitude from the flight path; the airports' ground from the world (`terragen` example `ground`) → `out/airliner/route.csv` (`t,lat,lon,h,roll,pitch,yaw`, 10 Hz) and `route.json` |
| `make_airliner.py` | the flight video: video time ↔ flight time from the playback speed (real time at take-off and landing, ×20 over each place, up to ×3000 in between), one `terrain run` per camera stretch over the time-warped trajectory (`render.lighting.time_map` keeps the sun on the flight's clock), the globe with the route drawn over it, captions, the speed, a readout and a route inset |
| `soundtrack.py` | the flight's sound, all synthesised here: an original ambient score (D dorian, 76 bpm: pads, a plucked arpeggio, sub bass, bells, reverb) whose sections follow the video; the aircraft from the flight itself (turbofan roar, buzz-saw and whine from the thrust schedule, airflow from the indicated airspeed, gear, touchdown, reversers); the callouts ("positive rate", "gear up", the radio-altitude callouts on landing) spoken by Piper through a cockpit-speaker filter. The faster the playback, the more the aircraft recedes to a cabin hum; near real time the music ducks under the engines |
| `globe/dive.yaml` | the showcase opening's map flight (keyframes and map layers) for `terrain view --record` |
| `globe/route.yaml` | the airliner opening's map flight |
| `fonts/` | Noto Sans Light / Medium, used for every caption (SIL Open Font License, `fonts/OFL.txt`) |
| `media/` | stills used by the repository README (`mosaic.jpg`: a 3×3 grid of showcase stills; `ground_truth.jpg`: an RGB / depth / flow / events panel frame) |

## Outputs

| path | what |
|------|------|
| `out/showcase/showcase.mp4`, `showcase_music.mp4` | the showcase without / with its score (`showcase_music.wav`: the score alone) |
| `out/showcase/<id>/` | each shot's scenario, trajectory, sequence (`seq.h5`) and render log |
| `out/showcase/clips/` | the captioned clip of each shot, the title and the outro |
| `out/showcase/world.h5` | the shared tile store |
| `out/airliner/airliner.mp4`, `airliner_sound.mp4` | the flight without / with its sound (`soundtrack.wav`: the sound alone) |
| `out/airliner/route.csv`, `route.json` | the flown trajectory and its phases |
| `out/airliner/s*_<camera>/`, `clips/`, `voice/` | the camera stretches, the composed clips, the cached callouts |

## Licensing of the assets

- **Code and configuration** in this folder: Apache-2.0, like the rest of the repository.
- **Fonts**: Noto Sans (Light, Medium) under the SIL Open Font License 1.1, see `fonts/OFL.txt`.
- **Music and sound effects**: synthesised by `showcase_score.py` and `soundtrack.py` from
  oscillators and filtered noise; no samples or recordings are used.
- **Voice callouts**: generated at render time with [Piper](https://github.com/OHF-Voice/piper1-gpl)
  and its LJSpeech voice (`en_US-ljspeech-high`), trained on the
  [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), whose recordings are in the
  public domain. Neither Piper nor the voice model is part of this repository: Piper is an
  external program run as a subprocess (the original rhasspy/piper releases are MIT, the
  current `piper-tts` releases from OHF-Voice/piper1-gpl are GPL-3.0); check the voice's model
  card in rhasspy/piper-voices before redistributing rendered audio.
