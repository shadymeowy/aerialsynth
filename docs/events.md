# Event camera modality

## What an event camera produces

Each pixel independently reports an event `e = (x, y, t, p)` whenever its log intensity has
changed by a contrast threshold `C` since that pixel's last event. Polarity `p` is ON for
brighter and OFF for darker. Timestamps have µs resolution, and data rates run from ~0.1 to
tens of Mev/s.

Datasets store events as parallel arrays:

- **M3ED / DSEC:** `x, y (u16), t (µs, i64), p (0/1)` plus an `ms_map_idx` index.
- **camodocal:** reads the same layout through `io/h5_source.cpp`.

## Options considered

| approach | idea | pros | cons |
|---|---|---|---|
| **A. Adaptive re-rendering (ESIM)**, *implemented* | render log intensity at times chosen so image motion between renders ≤ `max_px_per_step`; interpolate per pixel between renders | exact geometry, occlusions and lighting; any camera model; the trajectory's vibration is handled naturally (dense sampling only where the image moves) | cost ∝ image motion (~0.06–0.2 s per internal render) |
| B. Frame interpolation (v2e style) | render at frame rate, synthesize intermediate frames | cheaper | interpolation artifacts at occlusions; our exact flow would make warping easy, but disocclusions still need content |
| C. Linearized brightness constancy | `dL/dt = −∇L · flow` from one render + exact flow | very cheap | ignores occlusions and non-linear changes; poor for large motion |
| D. GPU renderer at kHz | same as A on the GPU | real-time-ish | needs the GPU renderer (planned by the user) |

A is the reference implementation. B and C could be added later as fast modes on top of the
same sensor model.

## Sensor model (`crates/render/src/events.rs`)

Per pixel, the processing chain is the following. Prophesee bias names are in brackets.

1. **Photoreceptor input:** `L = ln(gain · lum + eps)`. `eps` acts as dark current: it limits
   contrast in the dark and makes shot noise dominate there.
2. **Photoreceptor bandwidth** (`bias_fo`): a first-order low-pass on `L` whose 3 dB cutoff
   grows with photocurrent and saturates: `cutoff_hz · I/(I + cutoff_half_lum)`, floored at
   `cutoff_min_hz` (with 1e-5 ≈ 1 lux). It is ~kHz under street lights and slow under starlight,
   so events in the dark are delayed and smeared.
3. **High-pass** (`bias_hpf`): the reference level relaxes towards the signal (`hpf_hz`),
   suppressing slow illumination changes.
4. **Change detection** (`bias_diff_on` / `bias_diff_off`): ON/OFF thresholds `contrast_pos` and
   `contrast_neg`, with per-pixel fixed-pattern mismatch `contrast_sigma`.
   - Several thresholds can be crossed in one step.
   - Crossing times are linearly interpolated within the step.
5. **Refractory period** (`bias_refr`): `refractory_us`.
6. **Leak events:** the reference drifts down at a per-pixel rate (log-normal spread around
   `leak_hz`), producing mostly ON background events, as in real DVS pixels.
7. **Shot-noise background activity:** `shot_noise_hz` at `noise_ref_lum`, scaled by
   `sqrt(ref / lum)` (darker means noisier, up to `shot_noise_dark_gain`), random polarity.
8. **Hot pixels:** a fraction `hot_pixel_fraction` of pixels fire at ~`hot_pixel_hz` with a
   polarity bias.
9. **Timestamp jitter:** Gaussian with σ `timestamp_jitter_us`.
10. **Event rate controller:** at most `max_rate_mev_s` events per second. The excess within
    each 1 ms window is dropped at random, like Prophesee's ERC under bus saturation.

Unit tests (`cargo test -p render events`):

- **Moving step edge:** produces exactly `⌊ln 4 / C⌋` events per crossed pixel.
- **Background activity:** rates match the configured values, and are much higher in the dark.
- **Low-light bandwidth:** low light delays the response to a step.

## Using it

```
terrain events -c configs/events.yaml        # or `run` with events.enabled: true
python scripts/view_events.py out/events/events.h5 view.png --window-ms 5
```

Everything is set in the scenario `events:` section, including an optional separate event
`camera` (camodocal schema) and `extrinsics`.

Output: into `output.h5` by default (`events.h5` gives a separate file), with group and dataset
names from `output.layout`. The defaults are:

- `/prophesee/left/{x,y,t,p}`: `t` in µs relative to `/prophesee/left.attrs.t0_us`; `p` is 1 for
  ON and 0 for OFF.
- `/prophesee/left/ms_map_idx`: index of the first event of every ms, plus one closing entry.
- `/prophesee/left/calib/{intrinsics, distortion_coeffs, resolution, T_to_prophesee_left}`,
  plus the full camodocal camera YAML in an attribute.
- `/gt_events/{t, cam_position_ecef, cam_q_ecef}`: the camera pose at every internal render step.
  Poses at any other time come from the trajectory file.

Performance on 8 cores (VGA, supersample 2, `max_px_per_step` 0.5, ~1000 m AGL flight with
vibration): about 60 s of compute per simulated second, at ~300 internal renders/s. Two knobs
trade fidelity for speed:

- **`max_px_per_step`:** step size limit; larger is faster and less exact.
- **`supersample`:** 1 is faster but adds aliasing events.

## Lamp flicker

`render.lighting.flicker` makes artificial lights flicker at twice the mains frequency (default
50 Hz mains), with three supply phases and a fraction of LED lamps that barely flicker. When
lights are on, the event simulation renders at least `flicker_steps_per_period` times per
flicker period. At night this produces the periodic ON/OFF bursts at lamps that real event
cameras show (activity spectrum peaks at 100/200/300 Hz). It costs ~2.4k renders/s of simulated
time.

## Possible next steps

- **Stereo events:** a second event camera via an extra `events.camera` / `extrinsics` pair
  (camodocal's reader expects `left`/`right` groups for stereo).
- **Fast modes B/C** for long sequences.
