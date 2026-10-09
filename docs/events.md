# Event camera modality

## What an event camera produces

Each pixel independently reports an event `e = (x, y, t, p)` whenever its log intensity has
changed by a contrast threshold `C` since that pixel's last event. Polarity `p` is ON for
brighter and OFF for darker. Timestamps have µs resolution, and data rates run from ~0.1 to
tens of Mev/s.

Datasets (M3ED, DSEC, ...) store events as parallel arrays `x, y (u16), t (µs, i64), p (0/1)`
plus a per-millisecond index; we use the same data types.

## Options considered

| approach | idea | pros | cons |
|---|---|---|---|
| **A. Adaptive re-rendering (ESIM)**, *implemented* | render log intensity at times chosen so image motion between renders ≤ `max_px_per_step`; interpolate per pixel between renders | exact geometry, occlusions and lighting; any camera model; the trajectory's vibration is handled naturally (dense sampling only where the image moves) | cost ∝ image motion (~0.2 s per internal render) |
| B. Frame interpolation (v2e style), *tried and rejected* | render keyframes every few px of motion, reproject them with the exact per-pixel geometry for the steps in between | 5–7x faster | ~37% fewer events than A in textured terrain, also with 2x-resolution keyframes (see below) |
| C. Linearized brightness constancy | `dL/dt = −∇L · flow` from one render + exact flow | very cheap | ignores occlusions and non-linear changes; poor for large motion |
| D. A on the GPU, *implemented* | A with the keyframes rendered and the pixel model run on the GPU (`render.backend: gpu`, the default `auto` when there is a GPU; `docs/gpu.md`) | ~11–15x faster than A on the CPU; same events statistically | random numbers not bit-identical to the CPU |

A is the implementation, on the CPU or (D) the GPU. B was implemented and measured: the reprojection
itself is exact (image shifts match the geometric flow, identity warps are exact), but a
pixel is a box integral of texture with detail near its Nyquist frequency, and under a
sub-pixel shift that integral changes in ways no interpolation of the integrated image can
predict (the best shift explains only 5–25% of the change between renders 0.15 px apart).
Interpolated images are too smooth in time, so B systematically loses events (per-pixel event
maps correlate at 0.92 with A, at 0.63x the count). C has the same problem in a stronger form.

## Sensor model (`crates/render/src/events.rs`)

Per pixel, the processing chain is the following. Prophesee bias names are in brackets.

1. **Photoreceptor input:** `L = ln(gain · lum + log_eps)`. `log_eps` acts as dark current: it limits
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

Any camera becomes an event camera by giving it an `events` subsection (all fields optional;
`terrain config -c FILE` on a scenario with `events: {}` shows them with their defaults, since
`terrain config --all` lists only the sections that are on by default). It keeps its own intrinsics and extrinsics, and can carry `depth` and
`flow` ground truth at its `frame_rate` as well:

```yaml
cameras:
  - path: /dvs
    intrinsics: { model: pinhole, width: 640, height: 480, intrinsics: [400, 400, 319.5, 239.5] }
    extrinsics: { mount: nadir, translation: [0, 0.08, 0] }
    frame_rate: 20
    depth: {}
    flow: {}
    events: { contrast_pos: 0.25, contrast_neg: 0.25, supersample: 2 }
```

```
terrain run -c configs/examples/events.yaml           # render (poses, IMU, frames), then events
terrain run -c configs/examples/events.yaml --step events   # (re)simulate only the events into the existing file
python scripts/check_events.py out/events/seq.h5    # format invariants and event rates
```

Output, in the sequence file:

- `<camera>/events/{x, y, t, p}`: `t` in µs since the sequence start (the clock shared with
  frames, IMU and `/pose`); `p` is 1 for ON and 0 for OFF.
- `<camera>/events/ms_index`: index of the first event of every ms, plus one closing entry, so
  the events of ms `m` are `[ms_index[m], ms_index[m+1])`.
- `<camera>/calib/`: intrinsics, distortion, resolution and `T_body_cam`; the event camera's pose
  at any time is the `/pose` body pose composed with `T_body_cam`.

The depth / flow of an event camera come from a separate geometry-only render at its
`frame_rate` (no shading, one sample per pixel centre). The sensor noise (threshold mismatch,
hot pixels, background activity) is seeded from `seed` mixed with the camera path, so two event
cameras with the same settings get independent noise.

Performance of the CPU backend on the author's machine (Xeon W-2125, 8 threads; VGA, supersample 2, `max_px_per_step` 0.5,
~1000 m AGL flight with engine vibration): about 100 s of compute per simulated second, for
~520 renders per simulated second; the sensor model runs in parallel (~3 ms per step) and is
not the bottleneck. The GPU backend renders the keyframes ~15x and runs the sensor steps ~11x
faster (`docs/gpu.md`). The events step prints renders, sensor steps and their times. Two
knobs trade fidelity for speed:

- **`max_px_per_step`:** step size limit; larger is faster and less exact.
- **`supersample`:** 1 is faster but adds aliasing events.

## Lamp flicker

`render.lighting.flicker` makes artificial lights flicker at twice the mains frequency (default
50 Hz mains), with three supply phases and a fraction of LED lamps that barely flicker. When
lights are on, the event sensor steps at least `flicker_steps_per_period` times per flicker
period. At night this produces the periodic ON/OFF bursts at lamps that real event cameras
show (activity spectrum peaks at 100/200/300 Hz). These steps need no extra renders: the
renderer returns the lamp light as `radiance + cos(ωt)·A + sin(ωt)·B` (exact; checked against
direct renders to 1e-11), so only image motion triggers renders, plus one every 0.25 s for a
still camera (so a hover keeps its batches of steps short and follows lighting changes).
