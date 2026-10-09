#!/usr/bin/env python3
"""Render and compose the terrain showcase video.

    python showcase/make_showcase.py                  # render every shot, compose out/showcase/showcase.mp4
    python showcase/make_showcase.py --only coast     # (re)render one shot and its clip
    python showcase/make_showcase.py --compose-only   # no rendering: re-compose changed clips, assemble
    python showcase/make_showcase.py --stills         # quick framing check: small frames per shot
    python showcase/make_showcase.py --help           # every option (--force, --full-compose, --story, …)

Shots are described in storyboard.yaml (scenario overrides on base.yaml). Each shot is rendered
by `terrain run` into out/showcase/<id>/ (scenario.yaml, traj.csv, seq.h5) — the globe shot by
`terrain view --record` into out/showcase/<id>/frames/ — and is re-rendered only when its
resolved scenario changed. Composition (captions, cross-fades, 2x2 panels, the
tile map) is done here and piped into ffmpeg (H.264). Needs: numpy, h5py, pyyaml, pillow,
matplotlib (colour maps), ffmpeg, and the release build of `terrain` (cargo build --release).
"""
import argparse, copy, csv, hashlib, os, subprocess, time
import numpy as np, h5py, yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "showcase")
TERRAIN = os.path.join(ROOT, "target", "release", "terrain")
FONT_LIGHT = os.path.join(HERE, "fonts", "NotoSans-Light.ttf")
FONT_MEDIUM = os.path.join(HERE, "fonts", "NotoSans-Medium.ttf")


# ----------------------------------------------------------------------------- scenarios
def deep_merge(a, b):
    """b over a: mappings merge recursively, anything else replaces."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = copy.deepcopy(a)
        for k, v in b.items():
            out[k] = deep_merge(a[k], v) if k in a else copy.deepcopy(v)
        return out
    return copy.deepcopy(b)


def shot_scenario(base, shot, video, stills=False):
    """Resolved terrain scenario of a shot."""
    over = copy.deepcopy(shot.get("scenario", {}))
    cams_over = over.pop("cameras", [{}])
    scn = deep_merge(base, over)
    # cameras: every camera of the shot is merged onto the base camera (sensor look etc.)
    template = base["cameras"][0]
    # (a key set to null removes the template's value, e.g. `rgb: null` for an event-only camera)
    scn["cameras"] = [{k: v for k, v in deep_merge(template, c).items() if v is not None} for c in cams_over]
    fps = video["fps"]
    speedup = shot.get("speedup", 1)
    # flight time shown (+ `extend`: a continuation shown by another shot, e.g. its event camera)
    span = (shot["seconds"] + video["crossfade"] + shot.get("extend", 0.0)) * speedup
    start = scn["output"]["start"]
    sid = shot["id"]
    d = os.path.join(OUT, sid)
    scn["trajectory"]["file"] = os.path.join(d, "traj.csv")
    scn["trajectory"]["synth"]["duration_s"] = start + span + 1.0
    scn["output"]["file"] = os.path.join(d, "seq.h5")
    scn["output"]["end"] = start + span
    for c in scn["cameras"]:
        c["frame_rate"] = fps / speedup
    if not stills and any(k in c for c in scn["cameras"] for k in ("depth", "flow", "landcover")):
        # geometry ground truth needs an odd supersample; at 1080p one sample per pixel (the
        # panels show it downscaled)
        scn["render"]["supersample"] = 1
    if stills:
        # look check: 3 frames, half resolution, light supersampling, no events
        scn["output"]["file"] = os.path.join(d, "stills.h5")
        # event-only cameras have nothing to show in stills
        scn["cameras"] = [c for c in scn["cameras"] if any(k in c for k in ("rgb", "depth", "flow", "landcover"))]
        # (geometry ground truth needs an odd supersample)
        ss = 3 if any(k in c for c in scn["cameras"] for k in ("depth", "flow", "landcover")) else 2
        scn["render"]["supersample"] = ss
        for c in scn["cameras"]:
            c["frame_rate"] = 3.0 / span
            c.pop("events", None)
            if "rgb" in c:
                c["rgb"] = deep_merge(c["rgb"], {"supersample": ss})
            i = c["intrinsics"]
            if i.get("model", "pinhole") in ("pinhole", "kannala_brandt", "mei", "pinhole_full"):
                i["width"], i["height"] = i["width"] // 2, i["height"] // 2
                f = i["intrinsics"]
                i["intrinsics"] = [f[0] / 2, f[1] / 2, (f[2] + 0.5) / 2 - 0.5, (f[3] + 0.5) / 2 - 0.5]
    return scn


def globe_scenario(base, shot):
    """Scenario of a globe shot (`layout: globe`): the base world, its own tile store (the
    whole planet's coarse levels and the dive's tiles); `output.file` stands for the recording
    (its frames.csv, written last)."""
    d = os.path.join(OUT, shot["id"])
    scn = deep_merge(base, shot.get("scenario", {}))
    scn["tiles"]["file"] = os.path.join(d, "world.h5")
    scn["output"]["file"] = os.path.join(d, "frames", "frames.csv")
    return scn


def render_globe(base, shot, video, force=False):
    """A globe shot: the keyframed map flight `shot.path` recorded by `terrain view --record`."""
    sid = shot["id"]
    d = os.path.join(OUT, sid)
    os.makedirs(d, exist_ok=True)
    scn = globe_scenario(base, shot)
    path = os.path.join(HERE, shot["path"])
    size = f"{video['width']}x{video['height']}"
    text = yaml.safe_dump(scn, sort_keys=False)
    # (the frames shown; the camera path goes on to the end of the flight, for a `follows` shot)
    until = shot["seconds"] + video["crossfade"]
    digest = hashlib.sha1((text + open(path).read() + f"{size} {video['fps']} {until}").encode()).hexdigest()
    stamp = os.path.join(d, "scenario.done")
    if not force and os.path.exists(stamp) and open(stamp).read() == digest and os.path.exists(scn["output"]["file"]):
        print(f"[{sid}] up to date")
        return scn
    with open(os.path.join(d, "scenario.yaml"), "w") as f:
        f.write(text)
    frames = os.path.dirname(scn["output"]["file"])
    if os.path.isdir(frames):
        for n in os.listdir(frames):  # (a shorter recording leaves no stale frames)
            if n.startswith("frame_") and n.endswith(".png") or n == "frames.csv":
                os.remove(os.path.join(frames, n))
    t0 = time.time()
    print(f"[{sid}] recording the globe flight …", flush=True)
    log = open(os.path.join(d, "scenario.log"), "w")
    cmd = [TERRAIN, "view", "-c", os.path.join(d, "scenario.yaml"), "--record", frames,
           "--path", path, "--fps", str(video["fps"]), "--size", size, "--wait", "120", "--until", f"{until:.3f}"]
    if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode != 0:
        raise SystemExit(f"[{sid}] terrain view failed, see {log.name}")
    with open(stamp, "w") as f:
        f.write(digest)
    print(f"[{sid}] done in {time.time() - t0:.0f} s", flush=True)
    return scn


def follow_scenario(base, shot, video, lead):
    """Scenario of a shot that `follows` a globe shot (`lead`): its camera flies the globe's
    recorded camera path on, over the globe's tile store."""
    scn = shot_scenario(base, shot, video)
    scn["tiles"]["file"] = globe_scenario(base, lead)["tiles"]["file"]
    scn["tiles"]["lazy"] = True
    return scn


def follow_trajectory(base, lead, start):
    """The camera path recorded by the globe shot `lead`, from the end of its shown part on, as
    a trajectory CSV: that moment is output time `start`; the last pose is held after it."""
    rows = list(csv.DictReader(open(globe_scenario(base, lead)["output"]["file"])))
    t0 = lead["seconds"] - start
    cols = ("lat", "lon", "h", "roll", "pitch", "yaw")
    out = ["t," + ",".join(cols)]
    rows = [r for r in rows if float(r["t"]) >= t0] + [dict(rows[-1], t=float(rows[-1]["t"]) + 60.0)]
    for r in rows:
        out.append(f"{float(r['t']) - t0:.4f}," + ",".join(r[c] for c in cols))
    return "\n".join(out) + "\n"


def shot_by_id(story, sid):
    return next(x for x in story["shots"] if x["id"] == sid)


def planned_scenario(base, story, shot, stills=False):
    """A shot's resolved scenario (where its output is or will be), without rendering."""
    if shot.get("layout") == "globe":
        return globe_scenario(base, shot)
    if "follows" in shot:
        return follow_scenario(base, shot, story["video"], shot_by_id(story, shot["follows"]))
    if "source" in shot:
        return shot_scenario(base, shot_by_id(story, shot["source"]), story["video"], stills)
    return shot_scenario(base, shot, story["video"], stills)


def render_shot(base, shot, video, stills=False, force=False, scn=None, traj=None):
    """Render a shot by `terrain run` (`scn`: its scenario, if not the default one; `traj`: its
    trajectory CSV, if not synthesized)."""
    sid = shot["id"]
    d = os.path.join(OUT, sid)
    os.makedirs(d, exist_ok=True)
    scn = scn or shot_scenario(base, shot, video, stills)
    text = yaml.safe_dump(scn, sort_keys=False)
    tag = "stills" if stills else "scenario"
    path = os.path.join(d, f"{tag}.yaml")
    stamp = os.path.join(d, f"{tag}.done")
    # the render backend (cpu / gpu) gives the same output: not part of the up-to-date check
    keyed = copy.deepcopy(scn)
    keyed.get("render", {}).pop("backend", None)
    digest = hashlib.sha1((yaml.safe_dump(keyed, sort_keys=False) + (traj or "")).encode()).hexdigest()
    if not force and os.path.exists(stamp) and open(stamp).read() == digest and os.path.exists(scn["output"]["file"]):
        print(f"[{sid}] up to date")
        return scn
    with open(path, "w") as f:
        f.write(text)
    traj_file = scn["trajectory"]["file"]
    if os.path.exists(traj_file):
        os.remove(traj_file)  # the flight belongs to the scenario
    if traj is not None:
        with open(traj_file, "w") as f:
            f.write(traj)
    t0 = time.time()
    print(f"[{sid}] rendering ({tag}) …", flush=True)
    log = open(os.path.join(d, f"{tag}.log"), "w")
    r = subprocess.run([TERRAIN, "run", "-c", path], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[{sid}] terrain failed, see {log.name}")
    with open(stamp, "w") as f:
        f.write(digest)
    print(f"[{sid}] done in {time.time() - t0:.0f} s", flush=True)
    return scn


# ----------------------------------------------------------------------------- drawing
class Fonts:
    def __init__(self):
        self.cache = {}

    def get(self, which, size):
        key = (which, size)
        if key not in self.cache:
            self.cache[key] = ImageFont.truetype(FONT_MEDIUM if which == "medium" else FONT_LIGHT, size)
        return self.cache[key]


FONTS = Fonts()

# UI scale: layouts are designed for 1280x720 and scaled to the video width
UI = 1.0


def u(x):
    return int(round(x * UI))


def text_layer(size, items):
    """RGBA overlay with soft-shadowed text. items: (xy, text, font, alpha, anchor)."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    ds, dl = ImageDraw.Draw(shadow), ImageDraw.Draw(layer)
    for xy, txt, font, alpha, anchor in items:
        if alpha <= 0:
            continue
        ds.text((xy[0] + 1, xy[1] + 2), txt, font=font, fill=(0, 0, 0, int(200 * alpha)), anchor=anchor)
        dl.text(xy, txt, font=font, fill=(255, 255, 255, int(235 * alpha)), anchor=anchor)
    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    return Image.alpha_composite(shadow, layer)


def bottom_band(w, h, height=110, strength=0.55):
    """Vertical gradient that darkens the bottom of the frame for legible captions."""
    height = u(height)
    a = np.zeros((h, w), np.float32)
    y = np.arange(height, dtype=np.float32) / height
    a[h - height:, :] = (strength * y ** 1.6)[:, None]
    return a[..., None]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def caption(frame, label, text, t, dur, w, h, band):
    """Label bottom-left, description bottom-right; fade in/out."""
    a = ease((t - 0.35) / 0.45) * ease((dur - t - 0.15) / 0.45)
    if a <= 0 or (not label and not text):
        return frame
    f = frame.astype(np.float32) * (1 - band * a)
    img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
    m = u(36)
    items = [((m, h - m), label, FONTS.get("medium", u(24)), a, "ls"), ((w - m, h - m), text, FONTS.get("light", u(22)), a * 0.95, "rs")]
    img = Image.alpha_composite(img, text_layer(img.size, items))
    return np.asarray(img.convert("RGB"))


def panel_label(img, txt, xy=None, size=18, scale=None):
    """Small label at the top-left of a panel (PIL image, in place)."""
    k = UI if scale is None else scale
    xy = (round(14 * k), round(12 * k)) if xy is None else xy
    lay = text_layer(img.size, [(xy, txt, FONTS.get("medium", round(size * k)), 1.0, "la")])
    return Image.alpha_composite(img.convert("RGBA"), lay).convert("RGB")


# ----------------------------------------------------------------------------- visualisations
def flow_wheel():
    """Middlebury colour wheel (Baker et al.), as in flowlib / flow_vis."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    cols = []
    for n, (a, b) in zip([RY, YG, GC, CB, BM, MR], [((255, 0, 0), (255, 255, 0)), ((255, 255, 0), (0, 255, 0)), ((0, 255, 0), (0, 255, 255)),
                                                    ((0, 255, 255), (0, 0, 255)), ((0, 0, 255), (255, 0, 255)), ((255, 0, 255), (255, 0, 0))]):
        for i in range(n):
            cols.append([a[k] + (b[k] - a[k]) * i / n for k in range(3)])
    return np.array(cols) / 255.0


WHEEL = flow_wheel()


def flow_to_rgb(flow, rad_max):
    u, v = flow[..., 0], flow[..., 1]
    rad = np.sqrt(u * u + v * v) / max(rad_max, 1e-6)
    ang = np.arctan2(-v, -u) / np.pi
    fk = (ang + 1) / 2 * (len(WHEEL) - 1)
    k0 = np.floor(fk).astype(int)
    k1 = (k0 + 1) % len(WHEEL)
    f = (fk - k0)[..., None]
    col = (1 - f) * WHEEL[k0] + f * WHEEL[k1]
    r = np.clip(rad, 0, 1)[..., None]
    col = 1 - r * (1 - col)
    return (np.clip(col, 0, 1) * 255).astype(np.uint8)


def depth_to_rgb(depth, cmap):
    """Normalised to [0, 98th percentile of the frame's depth] and colour mapped; sky black
    (the plain maximum is the far horizon in oblique views, hundreds of km, which flattened
    all the terrain in front into one colour)."""
    fin = np.isfinite(depth) & (depth > 0)
    out = np.zeros(depth.shape + (3,), np.uint8)
    if fin.any():
        dmax = float(np.percentile(depth[fin], 98))
        c = cmap(np.clip(depth / dmax, 0, 1))[..., :3]
        out[fin] = (c[fin] * 255).astype(np.uint8)
        return out, dmax
    return out, 0.0


def events_to_rgb(ev, t_us, window_us, w, h):
    """Events in (t - window, t]: ON red, OFF blue on a dark background (captions stay legible)."""
    x, y, t, p = ev
    i1 = np.searchsorted(t, t_us, "right")
    i0 = np.searchsorted(t, t_us - window_us, "right")
    img = np.full((h, w, 3), (16, 18, 22), np.uint8)
    xs, ys, ps = x[i0:i1], y[i0:i1], p[i0:i1]
    on = ps > 0
    img[ys[~on], xs[~on]] = (70, 150, 255)
    img[ys[on], xs[on]] = (255, 70, 60)
    return img, i1 - i0


def ecef2lla(p):
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    e2, ep2 = f * (2 - f), (a * a - b * b) / (b * b)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.hypot(x, y)
    th = np.arctan2(z * a, r * b)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, r - e2 * a * np.cos(th) ** 3)
    lon = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), r / np.cos(lat) - n


def merc_px(lat, lon, z):
    """Global Web-Mercator pixel coordinates at zoom z."""
    n = 256 * 2 ** z
    x = (np.asarray(lon) + 180) / 360 * n
    la = np.radians(np.asarray(lat))
    y = (1 - np.log(np.tan(la) + 1 / np.cos(la)) / np.pi) / 2 * n
    return x, y


def quat_to_R(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


# ----------------------------------------------------------------------------- shot frames
def cam_frames(f, path):
    g = f[path]
    return g, g["rgb"].shape[0]


def fit(img, w, h):
    """Letterbox / pillarbox an image into w x h."""
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    s = min(w / iw, h / ih)
    im = Image.fromarray(img).resize((max(1, round(iw * s)), max(1, round(ih * s))), Image.LANCZOS)
    out = Image.new("RGB", (w, h))
    out.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.asarray(out)


# catalogue ids (HIP; planets 1<<30 | NAIF) of the named objects drawn by `star_overlay`
STAR_NAMES = {32349: "Sirius", 24436: "Rigel", 27989: "Betelgeuse", 25336: "Bellatrix", 26311: "Alnilam",
              26727: "Alnitak", 25930: "Mintaka", 27366: "Saiph", 26207: "Meissa", 37279: "Procyon", 37826: "Pollux",
              36850: "Castor", 21421: "Aldebaran", 24608: "Capella", 30438: "Canopus", 91262: "Vega", 97649: "Altair",
              102098: "Deneb", 11767: "Polaris", 69673: "Arcturus", 65474: "Spica", 49669: "Regulus", 80763: "Antares",
              (1 << 30) | 199: "Mercury", (1 << 30) | 299: "Venus", (1 << 30) | 499: "Mars", (1 << 30) | 599: "Jupiter",
              (1 << 30) | 699: "Saturn", (1 << 30) | 799: "Uranus", (1 << 30) | 899: "Neptune"}


def star_overlay(img, g, k, mag, scale):
    """Ground truth of frame k drawn on the image: a ring around every catalogue star brighter
    than `mag` (and every planet) at its recorded position, names for the bright ones."""
    st = g["stars"]
    i0, i1 = st["index"][k], st["index"][k + 1]
    ids, x, y, v, vis = st["id"][i0:i1], st["x"][i0:i1], st["y"][i0:i1], st["v"][i0:i1], st["visible"][i0:i1]
    im = Image.fromarray(img).convert("RGBA")
    ring = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ring)
    labels = []
    for j in np.argsort(v):
        planet = (int(ids[j]) >> 30) == 1
        if not vis[j] or (v[j] > mag and not planet):
            continue
        r = (6 + 2.2 * max(0.0, mag - v[j])) * scale
        col = (255, 200, 90, 230) if planet else (120, 220, 255, 200)
        d.ellipse([x[j] * scale - r, y[j] * scale - r, x[j] * scale + r, y[j] * scale + r], outline=col, width=max(1, round(1.5 * scale)))
        name = STAR_NAMES.get(int(ids[j]))
        if name:
            labels.append(((round(x[j] * scale + r + 4 * scale), round(y[j] * scale - 8 * scale)), name, FONTS.get("medium", u(15)), 0.9, "la"))
    im = Image.alpha_composite(im, ring)
    if labels:
        im = Image.alpha_composite(im, text_layer(im.size, labels))
    return np.asarray(im.convert("RGB"))


def single_frames(shot, f, video):
    g, n = cam_frames(f, shot.get("camera", "/cam0"))
    W, H = video["width"], video["height"]
    overlay = shot.get("star_overlay")
    for k in range(n):
        img = g["rgb"][k]
        if overlay and "stars" in g:
            img = star_overlay(img, g, k, overlay.get("mag", 4.5), 1.0)
        yield fit(img, W, H)


def globe_frames(shot, scn, video):
    d = os.path.dirname(scn["output"]["file"])
    for n in sorted(x for x in os.listdir(d) if x.startswith("frame_") and x.endswith(".png")):
        yield fit(np.asarray(Image.open(os.path.join(d, n)).convert("RGB")), video["width"], video["height"])


def grid_frames(shot, f, video, scn):
    W, H = video["width"], video["height"]
    paths = [c["path"] for c in scn["cameras"]]
    n = min(f[p]["rgb"].shape[0] for p in paths)
    labels = shot.get("panels", paths)
    for k in range(n):
        canvas = Image.new("RGB", (W, H))
        for i, p in enumerate(paths[:4]):
            im = Image.fromarray(fit(f[p]["rgb"][k], W // 2, H // 2))
            im = panel_label(im, labels[i])
            canvas.paste(im, ((i % 2) * W // 2, (i // 2) * H // 2))
        d = ImageDraw.Draw(canvas)
        d.line([(W // 2, 0), (W // 2, H)], fill=(0, 0, 0), width=u(2))
        d.line([(0, H // 2), (W, H // 2)], fill=(0, 0, 0), width=u(2))
        yield np.asarray(canvas)


def modality_frames(shot, f, video):
    import matplotlib
    cmap = matplotlib.colormaps["turbo"]
    W, H = video["width"], video["height"]
    g = f["/cam0"]
    n = g["rgb"].shape[0]
    w, h = g["rgb"].shape[2], g["rgb"].shape[1]
    ts = g["t"][:]
    ge = f[shot.get("events_camera", "/events")] if shot.get("events_camera", "/events") in f else g
    ev = ge["events"]
    evs = (ev["x"][:], ev["y"][:], ev["t"][:], ev["p"][:])
    ew, eh = (int(v) for v in ge["calib/resolution"][:])
    fl = g["flow"]
    # The camera's attitude motion shifts the whole image by a few px per frame, while the
    # depth-dependent parallax is a fraction of a pixel: shown raw, the flow is a smooth wash. The
    # panel shows the flow minus the rotational flow of the recorded poses (pinhole cameras), i.e.
    # the parallax, which carries the 3D structure.
    pin = g["calib"].attrs.get("model", "pinhole") in ("pinhole", b"pinhole") if "calib" in g else False
    fx, fy, cx, cy = (float(v) for v in g["calib/intrinsics"][:4])
    q = g["pose/q_ecef_cam"][:]

    def rot(qq):
        w_, x_, y_, z_ = qq
        return np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
                         [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)],
                         [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rays = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], -1)

    def parallax(k):
        f_ = fl[k]
        if not pin or k + 1 >= len(q):
            return f_
        rr = (rot(q[k + 1]).T @ rot(q[k])).astype(np.float32)
        d_ = rays @ rr.T
        rf = np.stack([fx * d_[..., 0] / d_[..., 2] + cx - xx, fy * d_[..., 1] / d_[..., 2] + cy - yy], -1)
        return f_ - rf

    # colour scale: robust maximum over the shot (stable colours)
    rad_max = max(1e-3, np.percentile(np.hypot(*np.moveaxis(parallax(n // 2), -1, 0)), 99))
    flow_label = "Optical flow · camera rotation removed (parallax)" if pin else "Optical flow"
    for k in range(n):
        rgb = g["rgb"][k]
        dep, dmax = depth_to_rgb(g["depth"][k], cmap)
        flo = flow_to_rgb(parallax(min(k, n - 2)), rad_max)
        evi, _ = events_to_rgb(evs, ts[k], 10_000, ew, eh)
        tiles = [(rgb, "RGB"), (dep, f"Depth · 0 – {dmax / 1000:.1f} km" if dmax >= 1000 else f"Depth · 0 – {dmax:.0f} m"),
                 (flo, flow_label), (evi, "Events · 10 ms · ON red / OFF blue")]
        canvas = Image.new("RGB", (W, H))
        for i, (im, lab) in enumerate(tiles):
            pim = panel_label(Image.fromarray(fit(im, W // 2, H // 2)), lab)
            canvas.paste(pim, ((i % 2) * W // 2, (i // 2) * H // 2))
        d = ImageDraw.Draw(canvas)
        d.line([(W // 2, 0), (W // 2, H)], fill=(0, 0, 0), width=u(2))
        d.line([(0, H // 2), (W, H // 2)], fill=(0, 0, 0), width=u(2))
        yield np.asarray(canvas)


def events_frames(shot, f, video):
    """Full-frame event camera view (10 ms windows, ON red / OFF blue), starting `offset` seconds
    into the source shot's flight (the continuation after the source's own segment)."""
    W, H, fps = video["width"], video["height"], video["fps"]
    g = f[shot.get("camera", "/events")]
    ts = f["/cam0"]["t"][:]  # the frame clock of the source shot
    ev = g["events"]
    evs = (ev["x"][:], ev["y"][:], ev["t"][:], ev["p"][:])
    w, h = (int(v) for v in g["calib/resolution"][:])
    k0 = int(round(shot.get("offset", 0.0) * fps))
    for k in range(k0, len(ts)):
        evi, _ = events_to_rgb(evs, ts[k], shot.get("window_ms", 10) * 1000, w, h)
        yield fit(evi, W, H)


def read_tile_mosaic(store, z, x0, y0, x1, y1):
    """RGB mosaic of tiles [x0, x1] x [y0, y1] at zoom z from the tile store (missing: grey)."""
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    img = np.full((ny * 256, nx * 256, 3), 60, np.uint8)
    with h5py.File(store, "r") as s:
        lv = s[f"levels/{z}"]
        idx = lv["index"][:]
        rows = {(int(a), int(b)): i for i, (a, b) in enumerate(idx)}
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                r = rows.get((tx, ty))
                if r is not None:
                    img[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = lv["rgb"][r]
    return img


def map_frames(shot, f, video, scn):
    """Camera view + 2D tile map with the whole flight, the current position, the camera footprint
    and the tiles planned for the flight (outlines coloured by zoom); altitude profile below."""
    import matplotlib
    W, H = 1280, 720  # laid out at 1280x720, upscaled in shot_stream
    z = shot.get("map_zoom", 14)
    d = os.path.join(OUT, shot["id"])
    g = f["/cam0"]
    n = g["rgb"].shape[0]
    P = f["/pose"]
    lla = P["lla"][:]
    tp = P["t"][:]
    # map extent: the flight + margin, in tiles
    gx, gy = merc_px(lla[:, 0], lla[:, 1], z)
    pad = 300
    x0, x1 = int((gx.min() - pad) // 256), int((gx.max() + pad) // 256)
    y0, y1 = int((gy.min() - pad) // 256), int((gy.max() + pad) // 256)
    store = scn["tiles"]["file"]
    need = [f"{z}/{x}/{y}" for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    lst = os.path.join(d, "map_tiles.txt")
    with open(lst, "w") as fh:
        fh.write("\n".join(need) + "\n")
    scen = os.path.join(d, "scenario.yaml")
    subprocess.run([TERRAIN, "tiles", "-c", scen, "--list", lst], cwd=ROOT, check=True, capture_output=True)
    plan_file = os.path.join(d, "plan.txt")
    subprocess.run([TERRAIN, "tiles", "-c", scen, "--dry-run", "-o", plan_file], cwd=ROOT, check=True, capture_output=True)
    plan = [tuple(map(int, l.split("/"))) for l in open(plan_file) if l.strip()]
    mosaic = read_tile_mosaic(store, z, x0, y0, x1, y1)
    ox, oy = x0 * 256, y0 * 256
    # map panel geometry
    cam_w, cam_h = g["rgb"].shape[2], g["rgb"].shape[1]
    mp = H - 2 * 40 - 120 - 16                     # map side
    mx, my = W - mp - 40, 40
    s = mp / max(mosaic.shape[0], mosaic.shape[1])
    base = Image.fromarray(mosaic).resize((round(mosaic.shape[1] * s), round(mosaic.shape[0] * s)), Image.LANCZOS)
    base = Image.fromarray((np.asarray(base).astype(np.float32) * 0.85).astype(np.uint8)).convert("RGBA")
    # planned tile outlines, coloured by zoom
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    zs = sorted({t[0] for t in plan if t[0] >= z - 2})
    cm = matplotlib.colormaps["plasma"]
    for (tz, tx, ty) in plan:
        if tz < z - 2:
            continue
        k = 2 ** (tz - z)
        ax, ay = (tx * 256 / k - ox) * s, (ty * 256 / k - oy) * s
        size = 256 / k * s
        if ax + size < 0 or ay + size < 0 or ax > base.width or ay > base.height:
            continue
        c = cm(0.15 + 0.75 * zs.index(tz) / max(1, len(zs) - 1))
        od.rectangle([ax, ay, ax + size, ay + size], outline=tuple(int(255 * v) for v in c[:3]) + (150,), width=1)
    base = Image.alpha_composite(base, over)
    tx_, ty_ = (gx - ox) * s, (gy - oy) * s
    alt = lla[:, 2]
    # camera footprint from the depth of the image border
    Rq = quat_to_R(g["pose/q_ecef_cam"][:])
    cpos = g["pose/position_ecef"][:]
    calib = yaml.safe_load(g["calib"].attrs["camera_yaml"])
    fx, fy, cx, cy = calib["intrinsics"]
    border = [(u, 0) for u in np.linspace(0, cam_w - 1, 24)] + [(cam_w - 1, v) for v in np.linspace(0, cam_h - 1, 14)] + \
             [(u, cam_h - 1) for u in np.linspace(cam_w - 1, 0, 24)] + [(0, v) for v in np.linspace(cam_h - 1, 0, 14)]
    ts = g["t"][:]
    for k in range(n):
        canvas = Image.new("RGB", (W, H), (18, 20, 24))
        cam = Image.fromarray(g["rgb"][k])
        cs = (W - mp - 40 * 3) / cam_w
        cam = cam.resize((round(cam_w * cs), round(cam_h * cs)), Image.LANCZOS)
        canvas.paste(cam, (40, 40))
        m = base.copy()
        md = ImageDraw.Draw(m)
        md.line(list(zip(tx_, ty_)), fill=(255, 255, 255, 110), width=2)
        j = min(np.searchsorted(tp, ts[k]), len(tp) - 1)
        md.line(list(zip(tx_[:j + 1], ty_[:j + 1])), fill=(255, 210, 60, 255), width=3)
        # footprint
        dep = g["depth"][k]
        pts = []
        for (u, v) in border:
            zz = dep[int(round(v)), int(round(u))]
            if np.isfinite(zz) and zz > 0:
                pc = np.array([(u - cx) / fx * zz, (v - cy) / fy * zz, zz])
                pts.append(Rq[k] @ pc + cpos[k])
        if len(pts) > 3:
            la, lo, _ = ecef2lla(np.array(pts))
            fx_, fy_ = merc_px(la, lo, z)
            poly = list(zip((fx_ - ox) * s, (fy_ - oy) * s))
            fp = Image.new("RGBA", m.size, (0, 0, 0, 0))
            ImageDraw.Draw(fp).polygon(poly, fill=(80, 200, 255, 60), outline=(80, 200, 255, 220))
            m = Image.alpha_composite(m, fp)
            md = ImageDraw.Draw(m)
        px, py = tx_[j], ty_[j]
        md.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(255, 210, 60, 255), outline=(0, 0, 0, 255), width=2)
        m = m.convert("RGB")
        m = panel_label(m, f"XYZ tiles · z{z} mosaic · planned LOD tiles z{zs[0]}–z{zs[-1]}", size=15, scale=1.0)
        canvas.paste(m, (mx, my))
        # altitude profile
        ap_y, ap_h = my + mp + 16, 120
        ad = ImageDraw.Draw(canvas)
        ad.rectangle([mx, ap_y, mx + mp, ap_y + ap_h], fill=(28, 31, 36))
        tt = (tp - tp[0]) / max(tp[-1] - tp[0], 1)
        lo_, hi_ = alt.min() - 20, alt.max() + 20
        prof = [(mx + 8 + (mp - 16) * a, ap_y + ap_h - 10 - (ap_h - 34) * (h_ - lo_) / (hi_ - lo_)) for a, h_ in zip(tt[::20], alt[::20])]
        ad.line(prof, fill=(150, 160, 175), width=2)
        ad.ellipse([prof[min(j // 20, len(prof) - 1)][0] - 4, prof[min(j // 20, len(prof) - 1)][1] - 4,
                    prof[min(j // 20, len(prof) - 1)][0] + 4, prof[min(j // 20, len(prof) - 1)][1] + 4], fill=(255, 210, 60))
        canvas = panel_label(canvas, f"Altitude above the ellipsoid · {alt[j]:.0f} m", xy=(mx + 10, ap_y + 6), size=14, scale=1.0)
        canvas = panel_label(canvas, f"Flight time {(ts[k] - ts[0]) * 1e-6:5.1f} s  (×{shot.get('speedup', 1)})", xy=(50, 50 + cam.height - 30), size=16, scale=1.0)
        yield np.asarray(canvas)


# ----------------------------------------------------------------------------- cards
def collage_sources(shots_scn, video):
    """One striking frame per shot (the middle one of its first camera) for the title collage."""
    ims = []
    for shot, scn in shots_scn:
        if shot.get("layout") in ("modalities", "map", "events", "globe") or "follows" in shot:
            continue
        with h5py.File(scn["output"]["file"], "r") as f:
            rgb = f[scn["cameras"][0]["path"]]["rgb"]
            ims.append(rgb[rgb.shape[0] // 2])
    return ims


def title_frames(video, seconds, sources):
    """Title over a drifting, tilted collage of frames from the whole video: tiles fade in one by
    one, the collage slowly zooms and pans, blurred and darkened under the title."""
    W, H, fps = video["width"], video["height"], video["fps"]
    tw, th, gap = u(384), u(216), u(10)
    cols, rows = 6, 5
    big = Image.new("RGB", (cols * (tw + gap), rows * (th + gap)), (8, 9, 11))
    rng = np.random.default_rng(7)
    order = rng.permutation(cols * rows)
    def cover(src, w, h):
        """Crop to the w:h aspect (centre), then resize."""
        ih, iw = src.shape[:2]
        if iw / ih > w / h:
            cw = int(ih * w / h)
            src = src[:, (iw - cw) // 2:(iw - cw) // 2 + cw]
        else:
            chh = int(iw * h / w)
            src = src[(ih - chh) // 2:(ih - chh) // 2 + chh]
        return Image.fromarray(np.ascontiguousarray(src)).resize((w, h), Image.LANCZOS)

    if not sources:
        sources = [np.zeros((th, tw, 3), np.uint8)]
    cells = [cover(sources[(c * 7) % len(sources)], tw, th) for c in range(cols * rows)]
    n = int(seconds * fps)
    yy, xx = np.mgrid[0:H, 0:W]
    vign = 1 - 0.55 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.2
    vign = np.clip(vign, 0.25, 1)[..., None]
    for k in range(n):
        t = k / fps
        frame = Image.new("RGB", big.size, (8, 9, 11))
        for c, im in enumerate(cells):
            a = ease((t - 0.08 * order[c] / 3) / 0.6)
            if a <= 0:
                continue
            x, y = (c % cols) * (tw + gap), (c // cols) * (th + gap)
            frame.paste(Image.blend(Image.new("RGB", im.size, (8, 9, 11)), im, a), (x, y))
        # slow zoom + pan, slight tilt
        z = 1.18 - 0.10 * ease(t / seconds)
        rot = frame.rotate(-7, resample=Image.BICUBIC, expand=False, fillcolor=(8, 9, 11))
        cw, ch = W * z * 1.05, H * z * 1.05
        cx = rot.width / 2 + u(60) * (t / seconds - 0.5)
        cy = rot.height / 2 - u(25) * (t / seconds - 0.5)
        crop = rot.crop((int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))).resize((W, H), Image.BICUBIC)
        blur = (5.0 - 2.5 * ease((t - 1.2) / 2.0)) * UI
        crop = crop.filter(ImageFilter.GaussianBlur(blur))
        f = np.asarray(crop).astype(np.float32) * 0.5 * vign
        img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        a1, a2, a3 = ease((t - 0.6) / 0.9), ease((t - 1.3) / 0.8), ease((t - 2.0) / 0.8)
        items = [((W // 2, H // 2 - u(34)), video["title"], FONTS.get("light", u(104)), a1, "ms"),
                 ((W // 2, H // 2 + u(20)), video["subtitle"], FONTS.get("light", u(32)), a2, "ms"),
                 ((W // 2, H // 2 + u(70)), video["tagline"], FONTS.get("light", u(20)), a3 * 0.85, "ms")]
        img = Image.alpha_composite(img, text_layer(img.size, items))
        yield np.asarray(img.convert("RGB"))


def outro_frames(video, outro, seconds=6.0):
    W, H, fps = video["width"], video["height"], video["fps"]
    lines = outro["lines"]
    for k in range(int(seconds * fps)):
        t = k / fps
        img = Image.new("RGBA", (W, H), (10, 11, 14, 255))
        items = []
        y0 = H // 2 - u(30) * len(lines)
        for i, l in enumerate(lines):
            items.append(((W // 2, y0 + u(56) * i), l, FONTS.get("light", u(30)), ease((t - 0.3 - 0.45 * i) / 0.6), "ms"))
        items.append(((W // 2, H - u(70)), outro["footer"], FONTS.get("medium", u(20)), ease((t - 0.4 - 0.45 * len(lines)) / 0.6) * 0.8, "ms"))
        img = Image.alpha_composite(img, text_layer(img.size, items))
        yield np.asarray(img.convert("RGB"))


# ----------------------------------------------------------------------------- composition
def shot_stream(shot, scn, video):
    lay = shot.get("layout", "single")
    if lay == "globe":
        return globe_frames(shot, scn, video)
    f = h5py.File(scn["output"]["file"], "r")
    if lay == "grid":
        gen = grid_frames(shot, f, video, scn)
    elif lay == "modalities":
        gen = modality_frames(shot, f, video)
    elif lay == "map":
        gen = (np.asarray(Image.fromarray(fr).resize((video["width"], video["height"]), Image.LANCZOS)) for fr in map_frames(shot, f, video, scn))
    elif lay == "events":
        gen = events_frames(shot, f, video)
    else:
        gen = single_frames(shot, f, video)
    return gen


def assemble(base, story, out_mp4, shots_scn):
    """The final video from the per-shot clips (out/showcase/clips): title and outro are encoded
    as clips too, then everything is joined in one ffmpeg pass with `crossfade`-second fades
    (xfade) — minutes instead of re-composing every frame from the sequences."""
    video = story["video"]
    fps, xf = video["fps"], video["crossfade"]
    d = os.path.join(OUT, "clips")
    ids = [s["id"] for s in story["shots"]]
    # each shot's clip by name (any position number; the newest if several)
    def clip_of(sid):
        c = [os.path.join(d, n) for n in os.listdir(d) if n[:2].isdigit() and n[2:3] == "_" and n[3:] == f"{sid}.mp4"]
        return max(c, key=os.path.getmtime) if c else os.path.join(d, f"??_{sid}.mp4")
    clips = [clip_of(sid) for sid in ids]
    missing = [c for c in clips if not os.path.exists(c)]
    if missing:
        raise SystemExit(f"assemble: missing clips {missing}")

    def encode(path, frames):
        ff = ffmpeg_writer(path, video)
        for fr in frames:
            ff.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        ff.stdin.close()
        ff.wait()

    title, outro = os.path.join(d, "title.mp4"), os.path.join(d, "outro.mp4")
    encode(title, title_frames(video, 5.0 + xf, collage_sources(shots_scn, video)))
    parts = [title] + clips
    if story.get("outro"):
        encode(outro, outro_frames(video, story["outro"], 6.0))
        parts.append(outro)
    dur = [float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", c],
                                capture_output=True, text=True, check=True).stdout) for c in parts]
    graph, prev, t = [], "[0:v]", 0.0
    for i in range(1, len(parts)):
        t += dur[i - 1] - xf
        out = f"[v{i}]"
        graph.append(f"{prev}[{i}:v]xfade=transition=fade:duration={xf}:offset={t:.4f}{out}")
        prev = out
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for c in parts:
        cmd += ["-i", c]
    cmd += ["-filter_complex", ";".join(graph), "-map", prev, "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True)
    total = t + dur[-1]
    print(f"wrote {out_mp4}: {len(parts)} parts, {total:.1f} s")


def compose(base, story, out_mp4, shots_scn):
    video = story["video"]
    W, H, fps = video["width"], video["height"], video["fps"]
    xf = int(round(video["crossfade"] * fps))
    band = bottom_band(W, H)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    written = 0

    def emit(fr):
        nonlocal written
        ff.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        written += 1

    tail = []  # last frames of the previous segment, for the cross-fade

    def segment(frames):
        nonlocal tail
        frames = list(frames)
        for i, fr in enumerate(frames):
            if i < len(tail):
                a = ease((i + 1) / (len(tail) + 1))
                fr = (tail[i].astype(np.float32) * (1 - a) + fr.astype(np.float32) * a).astype(np.uint8)
            if i < len(frames) - xf:
                emit(fr)
        tail = frames[-xf:] if len(frames) > xf else []

    # title over a collage of the whole video
    segment(title_frames(video, 5.0 + xf / fps, collage_sources(shots_scn, video)))
    for shot, scn in shots_scn:
        frames = shot_frames(shot, scn, video, band)
        print(f"composed {shot['id']}: {len(frames)} frames", flush=True)
        segment(frames)
    if story.get("outro"):
        segment(outro_frames(video, story["outro"], 6.0))
    for fr in tail:
        emit(fr)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out_mp4}: {written} frames, {written / fps:.1f} s")


def geo_readout(scn, shot, video):
    """Per output frame k: 'lat, lon · height · UTC' of the flight at that frame (pose of the
    sequence, the scenario's clock incl. its time-lapse factor), or None."""
    import datetime
    if shot.get("layout") == "globe":
        return globe_readout(scn)
    f = h5py.File(scn["output"]["file"], "r")
    if "/cam0" not in f or "pose" not in f:
        return lambda k: None
    ts = f["/cam0"]["t"][:]
    pt, lla = f["pose/t"][:], f["pose/lla"][:]
    t0 = float(f.attrs.get("t0", 0.0))
    light = yaml.safe_load(f.attrs["scenario"])["render"]["lighting"]
    day = light["date"] if isinstance(light["date"], datetime.date) else datetime.date.fromisoformat(str(light["date"]))
    tu = light["time_utc"]
    if not isinstance(tu, (int, float)):
        hh, mm, *ss = (float(v) for v in str(tu).split(":"))
        tu = hh * 3600 + mm * 60 + (ss[0] if ss else 0.0)
    base = datetime.datetime.combine(day, datetime.time()) + datetime.timedelta(seconds=float(tu))
    scale = float(light.get("time_scale", 1.0)) if light.get("mode", "fixed") == "clock" else 0.0
    k0 = int(round(shot.get("offset", 0.0) * video["fps"])) if shot.get("layout") == "events" else 0

    def text(k):
        t = ts[min(k0 + k, len(ts) - 1)]
        lat, lon, h = (np.interp(t, pt, lla[:, i]) for i in range(3))
        utc = base + datetime.timedelta(seconds=(t0 + t / 1e6) * scale)
        return (f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}  {abs(lon):.4f}°{'E' if lon >= 0 else 'W'}  ·  {h:,.0f} m  ·  "
                f"{utc:%Y-%m-%d %H:%M:%S} UTC")
    return text


def globe_readout(scn):
    """Per frame of a globe shot: 'lat, lon · height · finest tiles' of the recorded flight."""
    rows = list(csv.DictReader(open(scn["output"]["file"])))

    def text(k):
        r = rows[min(k, len(rows) - 1)]
        lat, lon, h = float(r["lat"]), float(r["lon"]), float(r["h"])
        alt = f"{h / 1000:,.0f} km" if h >= 20000 else f"{h:,.0f} m"
        return (f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}  {abs(lon):.4f}°{'E' if lon >= 0 else 'W'}  ·  {alt}  ·  "
                f"finest tiles z{r['max_zoom']}")
    return text


def notes_layer(shot, t, dur, W, H):
    """A shot's time-coded `notes` ({t, until, text}): top left, fading in and out."""
    items = []
    for n in shot.get("notes", []):
        end = min(n.get("until", dur), dur - 0.15)
        a = ease((t - n["t"]) / 0.4) * ease((end - t) / 0.4)
        if a > 0:
            items.append(((u(36), u(30)), n["text"], FONTS.get("light", u(24)), a, "la"))
    return text_layer((W, H), items) if items else None


def shot_frames(shot, scn, video, band):
    W, H, fps = video["width"], video["height"], video["fps"]
    dur = shot["seconds"] + video["crossfade"]
    geo = geo_readout(scn, shot, video) if video.get("coords") else (lambda k: None)
    frames = []
    for k, fr in enumerate(shot_stream(shot, scn, video)):
        if k >= int(round(dur * fps)):
            break
        txt = geo(k)
        if txt:
            # small readout, top right, fading with the caption
            a = ease((k / fps - 0.35) / 0.45) * ease((dur - k / fps - 0.15) / 0.45)
            if a > 0:
                lay = text_layer((W, H), [((W - u(14), u(12)), txt, FONTS.get("medium", u(13)), 0.85 * a, "ra")])
                fr = np.asarray(Image.alpha_composite(Image.fromarray(fr).convert("RGBA"), lay).convert("RGB"))
        notes = notes_layer(shot, k / fps, dur, W, H)
        if notes is not None:
            fr = np.asarray(Image.alpha_composite(Image.fromarray(fr).convert("RGBA"), notes).convert("RGB"))
        frames.append(caption(fr, shot["label"], shot["text"], k / fps, dur, W, H, band))
    return frames


def ffmpeg_writer(path, video):
    W, H, fps = video["width"], video["height"], video["fps"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_clip(index, shot, scn, video, so_far=True):
    """The shot as its own captioned clip (out/showcase/clips/NN_<id>.mp4), and (`so_far`) all
    clips so far joined into out/showcase/showcase_so_far.mp4, for review while the rest renders."""
    d = os.path.join(OUT, "clips")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{index:02d}_{shot['id']}.mp4")
    # reuse the shot's clip (also from another position) when the shot, the video settings and
    # the rendered sequence are unchanged
    digest = hashlib.sha1(yaml.safe_dump([shot, video], sort_keys=True).encode()).hexdigest()
    if not os.path.exists(scn["output"]["file"]):
        raise SystemExit(f"[{shot['id']}] not rendered yet ({scn['output']['file']}): run without --compose-only")
    src_mtime = os.path.getmtime(scn["output"]["file"])
    old = [c for c in os.listdir(d) if c[:2].isdigit() and c[2:3] == "_" and c[3:] == f"{shot['id']}.mp4"]
    reused = False
    for c in old:
        cp, st = os.path.join(d, c), os.path.join(d, c[:-4] + ".done")
        if os.path.exists(st) and open(st).read() == digest and os.path.getmtime(cp) > src_mtime:
            if cp != path:
                os.replace(cp, path)
                os.replace(st, path[:-4] + ".done")
            reused = True
            break
    for c in old:
        cp = os.path.join(d, c)
        if cp != path and os.path.exists(cp):
            os.remove(cp)
            if os.path.exists(cp[:-4] + ".done"):
                os.remove(cp[:-4] + ".done")
    if not reused:
        ff = ffmpeg_writer(path, video)
        for fr in shot_frames(shot, scn, video, bottom_band(video["width"], video["height"])):
            ff.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        ff.stdin.close()
        ff.wait()
        with open(path[:-4] + ".done", "w") as fh:
            fh.write(digest)
    if so_far:
        clips = sorted(c for c in os.listdir(d) if c.endswith(".mp4"))
        lst = os.path.join(d, "list.txt")
        with open(lst, "w") as fh:
            fh.writelines(f"file '{c}'\n" for c in clips)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        os.path.join(OUT, "showcase_so_far.mp4")], check=True)
    print(f"[{shot['id']}] clip {path}" + (" (reused)" if reused else ""), flush=True)


def export_stills(shot, scn):
    """Every still frame as its own PNG: out/showcase/stills/<shot>[_<camera>]_<k>.png."""
    d = os.path.join(OUT, "stills")
    os.makedirs(d, exist_ok=True)
    with h5py.File(scn["output"]["file"], "r") as f:
        cams = [c["path"] for c in scn["cameras"]]
        for c in cams:
            rgb = f[c]["rgb"]
            tag = shot["id"] if len(cams) == 1 else f"{shot['id']}_{c.strip('/').replace('/', '_')}"
            for k in range(rgb.shape[0]):
                Image.fromarray(rgb[k]).save(os.path.join(d, f"{tag}_{k}.png"))


def stills_sheet(base, story, shots_scn, path):
    rows = []
    for shot, scn in shots_scn:
        export_stills(shot, scn)
        f = h5py.File(scn["output"]["file"], "r")
        ims = []
        for c in scn["cameras"][:4]:
            rgb = f[c["path"]]["rgb"]
            for k in range(min(3, rgb.shape[0])):
                ims.append(fit(rgb[k], 320, 180))
            if shot.get("layout") not in ("grid",):
                break
        row = Image.new("RGB", (320 * max(3, len(ims)) + 220, 180), (20, 20, 20))
        for i, im in enumerate(ims):
            row.paste(Image.fromarray(im), (220 + 320 * i, 0))
        ImageDraw.Draw(row).text((10, 80), shot["id"], font=FONTS.get("medium", 22), fill=(255, 255, 255))
        rows.append(row)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, 180 * len(rows)))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, 180 * i))
    sheet.save(path)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="render only these shot ids")
    ap.add_argument("--compose-only", action="store_true",
                    help="render nothing: re-compose the clips whose shot or video settings changed, then assemble")
    ap.add_argument("--full-compose", action="store_true", help="re-compose every frame from the sequences instead of joining the clips")
    ap.add_argument("--stills", action="store_true", help="framing check (3 small frames per shot) → out/showcase/stills.png")
    ap.add_argument("--force", action="store_true", help="re-render even if up to date")
    ap.add_argument("--title-preview", action="store_true", help="save a few title-card frames (from the stills) → out/showcase/title_*.png")
    ap.add_argument("--out", default=os.path.join(OUT, "showcase.mp4"))
    ap.add_argument("--story", default=os.path.join(HERE, "storyboard.yaml"), help="storyboard (e.g. showcase/scout.yaml: candidate places)")
    a = ap.parse_args()
    if not os.path.exists(TERRAIN):
        raise SystemExit("build terrain first: cargo build --release")
    base = yaml.safe_load(open(os.path.join(HERE, "base.yaml")))
    story = yaml.safe_load(open(a.story))
    global UI
    UI = story["video"]["width"] / 1280
    shots = story["shots"]
    if a.only:
        shots = [s for s in shots if s["id"] in a.only]
    os.makedirs(OUT, exist_ok=True)
    if a.title_preview:
        scns = [(s, shot_scenario(base, s, story["video"], True)) for s in shots]
        scns = [(s, c) for s, c in scns if os.path.exists(c["output"]["file"])]
        fr = list(title_frames(story["video"], 5.0, collage_sources(scns, story["video"])))
        for t in (0.6, 1.6, 3.0, 4.8):
            Image.fromarray(fr[int(t * story["video"]["fps"])]).save(os.path.join(OUT, f"title_{t:.1f}.png"))
        print("wrote title previews")
        return
    done = []
    by_id = {}
    all_ids = [s["id"] for s in story["shots"]]
    # event simulation is the slowest part: shots with event cameras (and the shots shown from
    # them) are rendered last; the video keeps the storyboard order
    has_events = lambda s: "source" in s or any("events" in c for c in s.get("scenario", {}).get("cameras", []))
    order = [s for s in shots if not has_events(s)] + [s for s in shots if has_events(s)]
    for s in order:
        if a.stills and (s.get("layout") == "globe" or "follows" in s or "source" in s):
            # (no stills: a map recording, or the flight of another shot; preview a globe with
            # `terrain view --record` at a small --size)
            continue
        if a.compose_only:
            scn = planned_scenario(base, story, s, a.stills)
        elif s.get("layout") == "globe":
            scn = render_globe(base, s, story["video"], a.force)
        elif "follows" in s:
            lead = shot_by_id(story, s["follows"])
            scn = follow_scenario(base, s, story["video"], lead)
            scn = render_shot(base, s, story["video"], force=a.force, scn=scn,
                              traj=follow_trajectory(base, lead, scn["output"]["start"]))
        elif "source" in s:
            # shown from another shot's sequence (e.g. its event camera, later in the same flight)
            scn = by_id.get(s["source"]) or render_shot(base, shot_by_id(story, s["source"]), story["video"], force=a.force)
        else:
            scn = render_shot(base, s, story["video"], a.stills, a.force)
            if a.stills:
                export_stills(s, scn)  # PNGs per shot as soon as it is done, for feedback
        by_id[s["id"]] = scn
        if not a.stills and not (a.compose_only and a.full_compose):
            # (compose-only: nothing is rendered, but clips whose shot or video settings changed
            # are re-composed from their sequences; unchanged ones are reused)
            write_clip(all_ids.index(s["id"]) + 1, s, scn, story["video"], so_far=not a.compose_only)
        done.append((s, scn))
    done.sort(key=lambda x: all_ids.index(x[0]["id"]))
    if a.stills:
        name = os.path.splitext(os.path.basename(a.story))[0]
        stills_sheet(base, story, done, os.path.join(OUT, "stills.png" if name == "storyboard" else f"stills_{name}.png"))
    elif not a.only or a.compose_only:
        # the whole storyboard (with --only: the other shots as rendered before), so the title
        # collage always samples every shot
        every = [(s, by_id.get(s["id"]) or planned_scenario(base, story, s)) for s in story["shots"]]
        if a.full_compose:
            compose(base, story, a.out, every)
        else:
            assemble(base, story, a.out, every)


if __name__ == "__main__":
    main()
