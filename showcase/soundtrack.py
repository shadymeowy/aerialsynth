#!/usr/bin/env python3
"""The airliner video's sound: an original ambient score and the aircraft, synthesised from the
flight itself, mixed and muxed into the video.

    ~/.venv/bin/python showcase/soundtrack.py      # → out/airliner/soundtrack.wav, airliner_sound.mp4

Music (generated here, no samples): D dorian at 76 bpm, slow pads, a plucked arpeggio with a
ping-pong echo, sub bass and sparse bells in a long reverb; sections follow the video (globe,
take-off, cruise, descent, landing, outro). Callouts: "positive rate", "gear up" after lift-off and
the automatic height callouts on the landing (from the radio altitude along the trajectory),
spoken by Piper TTS with the LJSpeech voice (public-domain recordings) through a cockpit speaker
filter. Aircraft: turbofan roar, buzz-saw and fan whine whose
level and pitch follow the thrust schedule of the flight phase, airflow noise from the indicated
airspeed, gear up / down, the gear rumble on final, touchdown (thump, tyre chirps) and the
reversers. Sound follows the moment shown: the faster the playback, the more the aircraft
recedes to a cabin hum and the music leads; at real time the music ducks under the engines.
Needs numpy, scipy and piper-tts (voice: ~/.cache/piper/en_US-ljspeech-high.onnx from
huggingface.co/rhasspy/piper-voices).
"""
import json, math, os, subprocess, sys
import numpy as np
import yaml
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_airliner as ma  # noqa: E402
import route as rt  # noqa: E402

SR = 48000
RNG = np.random.default_rng(11)


# ----------------------------------------------------------------------------- helpers
def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True, check=True).stdout)


def band(x, lo, hi, order=4):
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos") if lo > 0 else signal.butter(order, hi, btype="low", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def norm(x):
    return x / (np.sqrt(np.mean(x * x)) + 1e-12)


def env_adsr(n, a, r):
    e = np.ones(n)
    na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    if na:
        e[:na] = np.linspace(0, 1, na) ** 2
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(buf, start, x, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i >= buf.shape[1] or i + len(x) <= 0:
        return
    j0 = max(0, -i)
    x = x[j0:]
    i = max(i, 0)
    n = min(len(x), buf.shape[1] - i)
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    buf[0, i:i + n] += gain * l * x[:n]
    buf[1, i:i + n] += gain * r * x[:n]


def reverb(buf, seconds=3.6, decay=1.1, wet=0.32):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.empty_like(buf)
    for c in range(2):
        ir = RNG.standard_normal(n) * np.exp(-t / decay * 3.0)
        ir = band(ir, 0, 7000)  # (darker tail)
        ir[: int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))
        ir /= np.sqrt(np.sum(ir ** 2))
        out[c] = signal.fftconvolve(buf[c], ir)[: buf.shape[1]]
    return buf * (1 - wet) + out * wet * 2.2


# ----------------------------------------------------------------------------- music
def pad(freq, dur, bright=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for det in (-0.006, 0.0, 0.0065):
        f = freq * (1 + det)
        ph = RNG.uniform(0, 2 * np.pi)
        for k in range(1, 9):
            if f * k > 5000:
                break
            a = (1.0 / k ** 1.4) * math.exp(-k / (3.5 * bright))
            x += a * np.sin(2 * np.pi * f * k * t + ph * k)
    # slow breathing of the brightness
    x *= 1 + 0.15 * np.sin(2 * np.pi * 0.11 * t + RNG.uniform(0, 6))
    return x * env_adsr(n, min(2.5, dur * 0.4), min(3.0, dur * 0.5)) * 0.18


def pluck(freq, dur=1.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k, a in ((1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (6, 0.05)):
        x += a * np.sin(2 * np.pi * freq * k * t) * np.exp(-t * (2.6 + 1.6 * k))
    return x * env_adsr(n, 0.004, 0.05) * 0.22


def bell(freq, dur=4.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for r, a, d in ((1.0, 1.0, 1.6), (2.76, 0.5, 0.9), (5.4, 0.25, 0.5), (8.93, 0.12, 0.3)):
        x += a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / d)
    return x * env_adsr(n, 0.003, 0.2) * 0.07


def sub(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)) * env_adsr(n, 0.6, 1.2) * 0.22


# chords (MIDI), each two bars: D dorian colours
CHORDS = [
    ([50, 57, 60, 64, 65], 38),   # Dm9 (D A C E F)
    ([46, 53, 57, 62, 64], 34),   # Bbmaj9(#11-ish)
    ([41, 48, 57, 60, 64], 41),   # Fmaj7/add9
    ([48, 55, 57, 62, 64], 36),   # C6/9
    ([43, 50, 58, 62, 65], 43),   # Gm9
    ([45, 52, 55, 60, 64], 45),   # Am7(add11)
    ([46, 53, 57, 60, 65], 34),   # Bbmaj7
    ([45, 52, 57, 61, 64], 45),   # A (sus → major: tension back to D)
]


def music(total, cues):
    """The score over `total` seconds; `cues`: video times of the sections."""
    buf = np.zeros((2, int((total + 6) * SR)))
    beat = 60.0 / 76
    bar = 4 * beat
    span = 2 * bar
    # intensity curves (0..1) of the layers over video time
    def level(name, t):
        c = cues
        if name == "arp":
            return np.interp(t, [0, c["takeoff"] - 2, c["takeoff"] + 6, c["cruise"], c["descent"], c["final"], c["final"] + 8, c["landing"], c["outro"] + 2, total],
                             [0, 0, 0.55, 1.0, 0.8, 0.45, 0.25, 0.15, 0.0, 0.0])
        if name == "sub":
            return np.interp(t, [0, c["globe_end"] - 4, c["takeoff"] + 2, c["final"], c["landing"], c["outro"], total], [0, 0.2, 0.9, 0.8, 0.4, 0.7, 0.5])
        if name == "bell":
            return np.interp(t, [0, 3, c["takeoff"], c["cruise"], c["descent"], c["outro"], total], [0.6, 1.0, 0.5, 0.7, 0.4, 1.0, 0.8])
        return np.interp(t, [0, 2, total - 4, total], [0.4, 1.0, 1.0, 0.8])  # pad
    k = 0
    t0 = 0.0
    while t0 < total:
        notes, bass = CHORDS[k % len(CHORDS)]
        if t0 >= cues["outro"] - span / 2:
            notes, bass = [50, 57, 62, 64, 69], 38  # D(add9) to close
        for i, m in enumerate(notes):
            add(buf, t0, pad(midi(m), span + 2.5, bright=1.2 if t0 > cues["cruise"] else 0.9), pan=-0.6 + 0.3 * i, gain=float(level("pad", t0)))
        add(buf, t0, sub(midi(bass), span + 1.0), gain=float(level("sub", t0)))
        # arpeggio: 8ths over the chord tones (up / down), an octave above
        la = float(level("arp", t0))
        if la > 0.01:
            seq = notes[1:] + notes[1:-1][::-1]
            for j in range(16):
                tt = t0 + j * beat / 2
                m = seq[j % len(seq)] + 12
                acc = 1.0 if j % 4 == 0 else 0.7
                add(buf, tt, pluck(midi(m)), pan=0.35 * math.sin(j * 1.3), gain=acc * float(level("arp", tt)))
        # bells: a high chord tone now and then
        lb = float(level("bell", t0))
        for j in range(2):
            if RNG.uniform() < 0.7:
                add(buf, t0 + RNG.integers(0, 8) * beat, bell(midi(RNG.choice(notes[2:]) + 24)), pan=RNG.uniform(-0.8, 0.8), gain=lb)
        k += 1
        t0 += span
    # ping-pong echo on everything above the pads (cheap: on the whole mix, quiet)
    d = int(0.75 * beat * SR)
    echo = np.zeros_like(buf)
    echo[0, d:] = buf[1, :-d] * 0.22
    echo[1, 2 * d:] = buf[0, :-2 * d] * 0.16
    buf = buf + echo
    buf = reverb(buf)
    return buf[:, : int(total * SR)]


# ----------------------------------------------------------------------------- the aircraft
def aircraft(total, off, tl, route, video):
    """Aircraft sound over the video (`off`: video time of the flight's first frame)."""
    n = int(total * SR)
    T = route.meta["duration_s"]
    ph = route.meta["phases"]
    fs = 200  # control rate
    tv = np.arange(0, total, 1 / fs)
    tau = tv - off
    tr = np.where(tau < 0, -1.0, tl.t_at(np.clip(tau, 0, tl.end)))
    flying = (tau >= 0) & (tau <= tl.end)
    after = np.clip(tau - tl.end, 0, None)  # seconds after touchdown (the outro)
    speed = np.where(flying, np.interp(np.clip(tr, 0, T), tl.t, tl.v), 1.0)
    d = route.data
    h = np.interp(np.clip(tr, 0, T), d[:, 0], d[:, 3])
    v = np.array([route.at(x)[3] for x in np.clip(tr[:: fs // 4], 1, T - 1)])  # ground speed, 4 Hz
    v = np.interp(tv, tv[:: fs // 4], v)
    ias = v * np.sqrt(rt.isa_density_ratio(h))
    # thrust (N1) schedule from the phase
    toc, tod = ph["top_of_climb"], ph["top_of_descent"]
    n1 = np.interp(tr, [0, 300, toc - 200, toc, tod - 30, tod + 30, 34500, 34560, 34900, 35000, T - 7, T - 3, T],
                   [0.95, 0.92, 0.9, 0.84, 0.84, 0.32, 0.36, 0.5, 0.45, 0.58, 0.55, 0.32, 0.3])
    n1 = np.where(flying, n1, 0.0)
    # presence: real time → full; sped up → a cabin hum
    pres = np.clip(1.0 - 0.5 * np.log10(np.maximum(speed, 1.0)), 0.06, 1.0)
    pres = np.where(flying, pres, 0.0)
    # (fade in from the globe; the reversers carry it into the outro)
    rev = np.where(after > 0, np.interp(after, [0, 0.6, 1.2, 5.0, 7.5], [0, 0.6, 1.0, 0.7, 0.0]), 0.0)
    pres = pres * np.clip((tau + 1.0) / 2.0, 0, 1)
    up = lambda x: np.interp(np.arange(n) / SR, tv, x)
    N1, PRES, IAS, REV = up(n1), up(pres), up(ias), up(rev)
    # broadband sources
    w = RNG.standard_normal(n)
    roar = norm(band(w, 60, 900)) * 0.6 + norm(band(w, 900, 3000)) * 0.25
    rumble = norm(band(RNG.standard_normal(n), 25, 160))
    air = norm(band(RNG.standard_normal(n), 250, 2500))
    hiss = norm(band(RNG.standard_normal(n), 2500, 9000))
    # tonal: buzz-saw (shaft harmonics) and fan whine
    f_shaft = 78.0 * np.maximum(N1, 0.2)
    phase = 2 * np.pi * np.cumsum(f_shaft) / SR
    buzz = np.zeros(n)
    amps = RNG.uniform(0.3, 1.0, 24)
    for k in range(4, 24):
        buzz += amps[k] / k ** 0.7 * np.sin(k * phase + k * 1.7)
    buzz = norm(buzz)
    whine = norm(np.sin(22 * phase) + 0.4 * np.sin(44 * phase + 1.0))
    eng = (N1 ** 2) * (roar * 0.55 + rumble * 0.45) + np.clip((N1 - 0.78) / 0.15, 0, 1) * buzz * 0.10 + (N1 ** 2) * whine * 0.04
    # airflow: hiss only near real time (sped up, the cabin is a low rumble)
    wind = (np.clip(IAS / 140.0, 0, 2.0) ** 2) * (air * 0.14 + hiss * 0.05 * PRES)
    cabin = rumble * 0.10 + air * 0.03  # the floor: the hum heard at any speed
    mono = PRES ** 1.5 * (eng + wind) + up(np.where(flying, 1.0, 0.0)) * cabin * 0.35
    mono += REV * (roar * 0.6 + rumble * 0.5 + hiss * 0.1)
    out = np.stack([mono, mono])
    # decorrelate a little (stereo width)
    out[1] = np.roll(out[1], int(0.011 * SR))
    # events (route time → video time)
    def at(t_route):
        return off + tl.tau_at(t_route)
    def thud(f=70, dec=0.12, g=1.0):
        m = int(0.8 * SR)
        t = np.arange(m) / SR
        return g * (np.sin(2 * np.pi * f * t) * np.exp(-t / dec) + 0.3 * band(RNG.standard_normal(m), 200, 2500) * np.exp(-t / 0.03))
    def hydraulic(dur, f=380):
        m = int(dur * SR)
        t = np.arange(m) / SR
        return (0.6 * np.sin(2 * np.pi * (f + 25 * t) * t) + 0.4 * band(RNG.standard_normal(m), 300, 1200)) * env_adsr(m, 0.3, 0.8) * 0.12
    def p(tr_):
        return float(np.interp(tr_, tl.t, np.clip(1.0 - 0.62 * np.log10(np.maximum(tl.v, 1.0)), 0.06, 1.0)))
    # gear up after lift-off, gear down on final
    for t_ev, kind in ((3.2, "up"), (34960.0, "down")):
        g = p(t_ev)
        add(out, at(t_ev), hydraulic(5.0), gain=g)
        add(out, at(t_ev) + (4.6 if kind == "up" else 3.8), thud(60, 0.15, 0.5), gain=g)
    # gear rumble while it is down
    gd = up(np.where(flying & (tr > 34965), 1.0, 0.0)) * PRES
    out += gd * norm(band(RNG.standard_normal(n), 40, 400)) * 0.12
    # touchdown: the mains (thump + chirps), then the nose gear
    td = off + tl.end
    for dt, f, g in ((0.0, 52, 1.0), (0.12, 48, 0.8), (1.6, 70, 0.5)):
        add(out, td + dt, thud(f, 0.18, g * 0.9))
        m = int(0.14 * SR)
        add(out, td + dt, band(RNG.standard_normal(m), 1800, 6500) * np.exp(-np.arange(m) / SR / 0.04) * 0.5 * g)
    # ground roll rumble into the outro
    roll = up(np.interp(after, [0, 0.1, 6, 9], [0, 1, 0.6, 0])) * up(np.where(after > 0, 1.0, 0.0))
    out += roll * norm(band(RNG.standard_normal(n), 20, 250)) * 0.3
    return out


# ----------------------------------------------------------------------------- callouts
VOICE = os.path.expanduser("~/.cache/piper/en_US-ljspeech-high.onnx")  # LJSpeech (public domain), Piper model (MIT)
PIPER = os.path.join(os.path.dirname(sys.executable), "piper")


def phrase(text):
    """A callout as audio at SR (Piper TTS, cached), through a cockpit speaker: 300–3400 Hz,
    slightly driven."""
    d = os.path.join(ma.OUT, "voice")
    os.makedirs(d, exist_ok=True)
    wav = os.path.join(d, "".join(c if c.isalnum() else "_" for c in text.lower()) + ".wav")
    if not os.path.exists(wav):
        subprocess.run([PIPER, "-m", VOICE, "--length-scale", "0.82", "-f", wav], input=text, text=True, check=True, capture_output=True)
    import wave
    with wave.open(wav) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768
        sr = w.getframerate()
    x = signal.resample_poly(x, SR, sr)
    nz = np.nonzero(np.abs(x) > 0.02)[0]  # (trim the silence around it)
    x = x[max(0, nz[0] - 200): nz[-1] + 2000] if len(nz) else x
    x = band(x, 300, 3400)
    x = np.tanh(3.0 * x / (np.max(np.abs(x)) + 1e-9)) / np.tanh(3.0)
    return x


def callouts(total, off, tl, route):
    """Take-off: "positive rate", "gear up"; landing: the automatic height callouts from the
    height of the gear above the ground (radio altitude) along the trajectory."""
    buf = np.zeros((2, int(total * SR)))
    def at(t_route):
        return off + tl.tau_at(t_route)
    add(buf, at(0.6), phrase("Positive rate."), gain=0.9)
    add(buf, at(1.9), phrase("Gear up."), gain=0.9)
    # radio altitude over the last minutes (gear ≈ 2.5 m below the body reference)
    d = route.data
    T = route.meta["duration_s"]
    tt = np.arange(T - 400.0, T, 0.25)
    lat, lon, h = (np.interp(tt, d[:, 0], d[:, i]) for i in (1, 2, 3))
    ra = (h - rt.ground_heights(list(zip(lat, lon))) - 2.5) / 0.3048  # feet
    ra = np.minimum.accumulate(ra)  # (each height called once, on the way down)
    calls = [(1000, "One thousand."), (500, "Five hundred."), (200, "Minimums."), (100, "One hundred."), (50, "Fifty."),
             (40, "Forty."), (30, "Thirty."), (20, "Twenty."), (15, "Retard."), (10, "Ten.")]
    for ft, text in calls:
        i = np.nonzero(ra <= ft)[0]
        if len(i):
            add(buf, at(float(tt[i[0]])), phrase(text), gain=0.9)
    return buf


def main():
    story_path = sys.argv[1] if len(sys.argv) > 1 else ma.STORY
    story = yaml.safe_load(open(story_path))
    ma.STORY = story_path
    video = story["video"]
    route = ma.Route(story)
    tl = ma.Timeline(story["flight"], route)
    d = os.path.join(ma.OUT, "clips")
    parts = [os.path.join(d, n) for n in ("title.mp4", "globe.mp4", "flight.mp4", "outro.mp4")]
    # (before the clips exist: their planned lengths, a preview without the video)
    preview = not all(os.path.exists(x) for x in parts)
    dur = [5.0 + 2 * video["crossfade"], story["globe"]["seconds"] + video["crossfade"], tl.end, story["outro"].get("seconds", 8.0)] if preview else [duration(x) for x in parts]
    xf = video["crossfade"]
    starts = np.concatenate([[0.0], np.cumsum(np.array(dur[:-1]) - xf)])
    total = float(starts[-1] + dur[-1])
    off = float(starts[2])
    cues = {"globe_end": float(starts[2]), "takeoff": off, "cruise": off + tl.tau_at(route.meta["phases"]["top_of_climb"]),
            "descent": off + tl.tau_at(route.meta["phases"]["top_of_descent"]), "final": off + tl.tau_at(34900),
            "landing": off + tl.end - 12, "outro": float(starts[3])}
    print("cues", {k: round(v, 1) for k, v in cues.items()}, "total", round(total, 1), flush=True)
    mus = music(total, cues)
    air = aircraft(total, off, tl, route, video)
    voice = callouts(total, off, tl, route)
    # loudness: the aircraft at take-off ≈ -16 dBFS RMS, the music ≈ -20 dBFS; near real time
    # (take-off, landing) the music ducks ~10 dB under the engines
    i0, i1 = int(off * SR), int((off + 8.0) * SR)
    air = air * (0.22 / (np.sqrt(np.mean(air[:, i0:i1] ** 2)) + 1e-12))
    mus = mus / (np.sqrt(np.mean(mus ** 2)) + 1e-12) * 0.10
    tv = np.arange(mus.shape[1]) / SR
    tau = tv - off
    sp = np.where((tau >= 0) & (tau <= tl.end), np.interp(tl.t_at(np.clip(tau, 0, tl.end)), tl.t, tl.v), 1e4)
    rt_ = np.clip(1.0 - 0.6 * np.log10(np.maximum(sp, 1.0)), 0.0, 1.0)
    rt_ = signal.sosfiltfilt(signal.butter(2, 0.5, fs=SR, output="sos"), rt_)
    duck = 1.0 - 0.85 * np.clip(rt_, 0, 1)  # (-16 dB at real time)
    mix = mus * duck + air + voice * 0.5
    if os.environ.get("SOUND_STEMS"):  # (levels of both stems at 20 Hz, for checking the mix)
        m = mus.shape[1] // 2400 * 2400
        np.save(os.path.join(ma.OUT, "stems_rms.npy"), np.stack([np.sqrt(np.mean(x[:, :m].reshape(2, -1, 2400) ** 2, axis=(0, 2))) for x in (mus * duck, air)]))
    # fades, soft limit, -1 dBFS peak
    n = mix.shape[1]
    fade = np.ones(n)
    fade[: int(1.0 * SR)] = np.linspace(0, 1, int(1.0 * SR))
    fade[-int(3.0 * SR):] = np.linspace(1, 0, int(3.0 * SR)) ** 1.5
    mix = np.tanh(mix * fade * 1.4) / 1.4
    mix *= 0.89 / np.max(np.abs(mix))
    wav = os.path.join(ma.OUT, "soundtrack.wav")
    pcm = (mix.T * 32767).astype("<i2")
    import wave
    with wave.open(wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    if preview:
        print(f"wrote {wav} ({total:.1f} s; preview: no video yet)")
        return
    src = os.path.join(ma.OUT, "airliner.mp4")
    dst = os.path.join(ma.OUT, "airliner_sound.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", dst], check=True)
    print(f"wrote {wav} and {dst} ({total:.1f} s)")


if __name__ == "__main__":
    main()
