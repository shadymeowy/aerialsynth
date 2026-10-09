#!/usr/bin/env python3
"""An original score for the showcase video (out/showcase/showcase.mp4), synthesised here (no
samples) and muxed under it:

    python showcase/showcase_score.py              # → out/showcase/showcase_music.wav, showcase_music.mp4
    python showcase/showcase_score.py VIDEO [OUT]  # under another video (OUT: default VIDEO_music.mp4)

Electronic, 120 bpm in A minor: four-on-the-floor kick, hats and clap, a rolling bass pumped by the
kick, a 16th-note pluck arpeggio, pads, bells and a lead motif. The beat grid is laid on the cuts
(the shots are ~6 s long, the cuts fall on beats) and the sections follow the storyboard: a quiet
intro under the title and the globe, the groove from the first landscape, a sunset drop for golden
hour and dusk, a sparse night section (no drums) for the night, stars and event cameras, a build
back into the full groove with the lead for the rivers, the peak through the steep turns, and a
settling end over the last landscapes into the outro. The cuts come from the clips in
out/showcase/clips (make_showcase.py); without them the sections are spread over the video.
Needs numpy, scipy (through soundtrack.py, whose instruments it shares) and ffmpeg.
"""
import argparse, math, os, subprocess, sys, wave
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soundtrack import SR, add, band, bell, duration, env_adsr, midi, pad, reverb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "showcase")
STORY = os.path.join(ROOT, "showcase", "storyboard.yaml")
RNG = np.random.default_rng(7)
BEAT = 0.5  # 120 bpm
PHASE = 0.2  # (the grid: beats at PHASE + k·BEAT, through the middle of most cross-fades)


def on_grid(t):
    """The beat nearest to the middle of a cross-fade starting at `t`."""
    return round((t + 0.3 - PHASE) / BEAT) * BEAT + PHASE

# sections: the shot (storyboard id) each starts on, and its layers (0..1)
SECTIONS = [
    ("title",        "intro",  dict(pad=0.7, bell=1.0)),
    ("globe",        "intro",  dict(pad=1.0, bell=0.7, arp=0.35, bright=0.4, sub=0.5, riser=4.0)),
    ("descent",      "main",   dict(pad=0.8, kick=1.0, hat=0.7, bass=1.0, arp=0.7, bright=0.7, crash=1.0)),
    ("map",          "main",   dict(pad=0.8, kick=1.0, hat=1.0, clap=1.0, bass=1.0, arp=0.8, bright=0.9, crash=0.7)),
    ("golden",       "main",   dict(pad=1.0, hat=0.6, bass=0.5, arp=0.6, bright=0.35, bell=0.6, crash=0.5)),
    ("night",        "night",  dict(pad=1.0, bell=1.0, sub=0.8, arp=0.3, bright=0.25, tick=0.6)),
    ("star_tracker", "night",  dict(pad=1.0, bell=0.8, sub=0.9, arp=0.5, bright=0.4, tick=0.8, hat=0.3, riser=4.0)),
    ("braided",      "main",   dict(pad=0.8, kick=1.0, hat=1.0, clap=1.0, bass=1.0, arp=0.8, bright=0.9, lead=1.0, crash=1.0)),
    ("alpine_south", "lift",   dict(pad=0.9, kick=1.0, hat=1.0, hat16=1.0, clap=1.0, bass=1.0, arp=1.0, bright=1.0, lead=1.0, crash=1.0)),
    ("town",         "main",   dict(pad=0.8, kick=1.0, hat=0.8, clap=0.8, bass=0.9, arp=0.8, bright=0.8, bell=0.4, crash=0.7)),
    ("farmland",     "main",   dict(pad=1.0, arp=0.5, bright=0.5, bell=0.8, sub=0.6, crash=0.6)),
    ("outro",        "end",    dict(pad=1.0, bell=1.0)),
]

# chord progressions (MIDI chord tones, bass note), two bars per chord
PROGS = {
    "intro": [([57, 60, 64, 67, 71], 45), ([53, 57, 60, 64, 67], 41), ([55, 59, 62, 64, 67], 43), ([52, 55, 59, 62, 66], 40)],
    "main":  [([53, 57, 60, 64, 67], 41), ([55, 59, 62, 64, 67], 43), ([57, 60, 64, 67, 71], 45), ([52, 55, 59, 62, 66], 40)],
    "lift":  [([53, 57, 60, 64, 69], 41), ([55, 59, 62, 67, 71], 43), ([57, 60, 64, 69, 72], 45), ([55, 59, 62, 66, 71], 43)],
    "night": [([57, 60, 64, 67, 71], 45), ([53, 57, 60, 64, 71], 41), ([50, 53, 57, 60, 64], 38), ([52, 57, 59, 62, 64], 40)],
    "end":   [([57, 60, 64, 67, 71, 76], 45)],
}
# the lead motif over eight bars: (beat, MIDI, beats)
LEAD = [(0, 76, 1.5), (1.5, 74, 0.5), (2, 72, 1), (3, 74, 1), (4, 76, 3),
        (8, 79, 1.5), (9.5, 76, 0.5), (10, 74, 2), (12, 72, 1), (13, 71, 1), (14, 72, 2),
        (16, 76, 1.5), (17.5, 74, 0.5), (18, 72, 1), (19, 74, 1), (20, 76, 2), (22, 79, 2),
        (24, 81, 2), (26, 79, 1), (27, 76, 1), (28, 74, 4)]


# ----------------------------------------------------------------------------- instruments
def kick():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 46 + 110 * np.exp(-t * 28)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    x[: int(0.004 * SR)] += RNG.standard_normal(int(0.004 * SR)) * 0.3  # (click)
    return x * 0.9


def clap():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    noise = band(RNG.standard_normal(n), 900, 6000)
    e = np.exp(-t * 16)
    for d in (0.011, 0.023):  # (the flams of a clap)
        e += 0.7 * np.exp(-np.maximum(t - d, 0) * 60) * (t >= d)
    return (noise * e + 0.4 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30)) * 0.32


def hat(open_=False):
    n = int((0.3 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    x = band(RNG.standard_normal(n), 7000, 16000)
    return x * np.exp(-t * (9 if open_ else 60)) * (0.16 if open_ else 0.13)


def tick():  # a soft high click: the night section's pulse (the event cameras)
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 180) * 0.12


def crash():
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    x = band(RNG.standard_normal(n), 3500, 15000) * np.exp(-t * 1.6)
    return x * env_adsr(n, 0.002, 0.3) * 0.22


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    noise = band(RNG.standard_normal(n), 1500, 12000) * u ** 2.5
    f = 220 * 2 ** (3 * u)  # (three octaves up)
    sweep = np.sin(2 * np.pi * np.cumsum(f) / SR) * u ** 2
    return (noise * 0.18 + sweep * 0.05) * env_adsr(n, 0.05, 0.02)


def bass_note(freq, dur, bright):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(1, 12):
        if freq * k > 2500:
            break
        x += (1.0 / k) * np.sin(2 * np.pi * freq * k * t) * np.exp(-t * (3 + k * (6 - 3 * bright)))
    return x * env_adsr(n, 0.003, 0.03) * 0.3


def pluck(freq, dur, bright):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.18), (5, 0.1), (7, 0.05)):
        x += a * bright ** (0.5 * (k - 1)) * np.sin(2 * np.pi * freq * k * t) * np.exp(-t * (5 + 2.5 * k))
    return x * env_adsr(n, 0.002, 0.04) * 0.2


def lead_note(freq, dur):
    n = int((dur + 0.4) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t) * np.clip(t / 0.3, 0, 1)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph) + 0.08 * np.sin(5 * ph)
    e = env_adsr(n, 0.02, 0.4) * np.where(t < dur, 1.0, np.exp(-(t - dur) * 8))
    return x * e * 0.09


# ----------------------------------------------------------------------------- the score
def cut_times(video_in):
    """(start of each part in the video, part ids): the title, the storyboard's shots and the
    outro, from the clips as make_showcase.py joined them; None without the clips."""
    d = os.path.join(OUT, "clips")
    if not os.path.isdir(d):
        return None
    story = yaml.safe_load(open(STORY))
    xf = story["video"]["crossfade"]

    def clip_of(sid):
        c = [os.path.join(d, n) for n in os.listdir(d) if n[:2].isdigit() and n[2:3] == "_" and n[3:] == f"{sid}.mp4"]
        return max(c, key=os.path.getmtime) if c else None
    ids = ["title"] + [s["id"] for s in story["shots"]] + (["outro"] if story.get("outro") else [])
    clips = [os.path.join(d, "title.mp4")] + [clip_of(s["id"]) for s in story["shots"]] + ([os.path.join(d, "outro.mp4")] if story.get("outro") else [])
    if any(c is None or not os.path.exists(c) for c in clips):
        return None
    starts, t = [], 0.0
    for c in clips:
        starts.append(t)
        t += duration(c) - xf
    if abs(t + xf - duration(video_in)) > 1.0:  # (not this video)
        return None
    return starts, ids


def score(total, cuts):
    """The score over `total` seconds; `cuts`: (start times, part ids) of the video's parts."""
    starts, ids = cuts
    at = dict(zip(ids, starts))
    # each section from the cut it starts on, snapped to the beat grid through the cuts
    secs = [(on_grid(at[sid]) if sid != "title" else PHASE, prog, lay) for sid, prog, lay in SECTIONS if sid in at]
    secs.append((total + 4.0, None, None))
    drums = np.zeros((2, int((total + 6) * SR)))
    mus = np.zeros_like(drums)
    pump = np.ones(drums.shape[1])  # sidechain from the kick
    K, C, HC, HO, T = kick(), clap(), hat(), hat(True), tick()
    pump_curve = 1 - 0.6 * np.exp(-np.arange(int(0.4 * SR)) / SR / 0.09)
    cut_set = {round(on_grid(s) / BEAT * 2) for s in starts[1:]}
    for (t0, prog, L), (t1, _, _) in zip(secs[:-1], secs[1:]):
        g = lambda k: L.get(k, 0.0)  # noqa: E731
        if g("crash") and t0 > 0:
            add(drums, t0, crash(), gain=g("crash"))
        if g("riser"):
            add(mus, t1 - g("riser"), riser(g("riser")), gain=1.0)
        chords = PROGS[prog]
        nb = int(round((t1 - t0) / BEAT))
        for b in range(nb):
            t = t0 + b * BEAT
            if t >= total:
                break
            bar, pos = divmod(b, 4)
            notes, root = chords[(bar // 2) % len(chords)]
            if pos == 0 and bar % 2 == 0:  # a chord: pads, sub, bells
                span = min(8 * BEAT, t1 - t)
                for i, m in enumerate(notes):
                    add(mus, t, pad(midi(m), span + 2.0, bright=0.8 + 0.6 * g("bright")), pan=-0.6 + 0.25 * i, gain=g("pad") * 0.8)
                if g("sub"):
                    add(mus, t, bass_note(midi(root - 12), span, 0.0), gain=g("sub") * 0.9)
                if g("bell"):
                    for _ in range(2):
                        if RNG.uniform() < 0.75:
                            add(mus, t + RNG.integers(0, 8) * BEAT, bell(midi(RNG.choice(notes[1:]) + 24)), pan=RNG.uniform(-0.8, 0.8), gain=g("bell") * 1.4)
            # drums
            if g("kick"):
                add(drums, t, K, gain=g("kick"))
                i = int(t * SR)
                m = min(len(pump_curve), len(pump) - i)
                pump[i:i + m] = np.minimum(pump[i:i + m], pump_curve[:m])
            if g("clap") and pos in (1, 3):
                add(drums, t, C, pan=0.05, gain=g("clap"))
            if g("hat"):
                add(drums, t + BEAT / 2, HO if bar % 2 == 1 and pos == 3 else HC, pan=0.3, gain=g("hat"))
                if g("hat16"):
                    for q in (0.25, 0.75):
                        add(drums, t + q * BEAT, HC, pan=-0.3, gain=0.5 * g("hat16"))
            if g("tick"):
                add(drums, t, T, pan=0.4 * math.sin(b * 0.7), gain=g("tick") * (1.0 if pos == 0 else 0.5))
            if round(t / BEAT * 2) in cut_set and g("hat") and b > 0:  # (an open hat on each cut)
                add(drums, t, HO, pan=-0.2, gain=0.6)
            # bass: rolling 8ths (root, root, octave, root ...)
            if g("bass"):
                for q, o in ((0.0, 0), (0.5, 12 if pos % 2 else 0)):
                    add(mus, t + q * BEAT, bass_note(midi(root + o), BEAT * 0.48, 0.5), gain=g("bass") * (0.8 if q else 1.0))
            # arpeggio: 16ths up and down the chord, an octave up
            if g("arp"):
                seq = notes[1:] + notes[-2:0:-1]
                for q in range(4):
                    j = b * 4 + q
                    add(mus, t + q * BEAT / 4, pluck(midi(seq[j % len(seq)] + 12), 0.35, g("bright")),
                        pan=0.45 * math.sin(j * 0.9), gain=g("arp") * (1.0 if q == 0 else 0.65))
            # lead motif
            if g("lead") and pos == 0 and bar % 8 == 0:
                for bb, m, dd in LEAD:
                    if t + bb * BEAT < t1:
                        add(mus, t + bb * BEAT, lead_note(midi(m), min(dd * BEAT, t1 - t - bb * BEAT)), pan=0.1, gain=g("lead"))
        if prog == "end":  # (the last chord rings out)
            break
    mus = mus * pump[None, :]
    # a dotted-8th echo on the music, then reverb (drums drier)
    d = int(0.75 * BEAT * SR)
    echo = np.zeros_like(mus)
    echo[0, d:] = mus[1, :-d] * 0.2
    echo[1, 2 * d:] = mus[0, :-2 * d] * 0.14
    mus = reverb(mus + echo, seconds=3.0, decay=1.0, wet=0.28)
    drums = reverb(drums, seconds=1.2, decay=0.4, wet=0.12)
    mix = mus + 0.9 * drums
    return mix[:, : int(total * SR)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", default=os.path.join(OUT, "showcase.mp4"), help="the video (default: out/showcase/showcase.mp4)")
    ap.add_argument("out", nargs="?", default=None, help="output video (default: VIDEO_music.mp4; the WAV next to it)")
    a = ap.parse_args()
    src = a.video
    out = a.out or os.path.splitext(src)[0] + "_music.mp4"
    total = duration(src)
    cuts = cut_times(src)
    if cuts is None:  # (another video: the sections spread over it)
        ids = [s[0] for s in SECTIONS]
        frac = [0, 0.02, 0.08, 0.15, 0.27, 0.32, 0.43, 0.47, 0.73, 0.81, 0.96, 0.98]
        cuts = ([f * total for f in frac], ids)
    mix = score(total, cuts)
    n = mix.shape[1]
    fade = np.ones(n)
    fade[: int(1.0 * SR)] = np.linspace(0, 1, int(1.0 * SR))
    fade[-int(4.0 * SR):] = np.linspace(1, 0, int(4.0 * SR)) ** 1.5
    mix = mix / (np.sqrt(np.mean(mix ** 2)) + 1e-12) * 0.14
    mix = np.tanh(mix * fade * 1.3) / 1.3
    mix *= 0.89 / np.max(np.abs(mix))
    wav = os.path.splitext(out)[0] + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix.T * 32767).astype("<i2").tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out], check=True)
    print(f"wrote {wav} and {out} ({total:.1f} s)")


if __name__ == "__main__":
    main()
