#!/usr/bin/env python3
"""Render and compose the long-haul flight video (storyboard: showcase/airliner.yaml).

    python showcase/make_airliner.py                   # route, every segment, out/airliner/airliner.mp4
    python showcase/make_airliner.py --stills          # framing check: 3 small frames per segment → out/airliner/stills.png
    python showcase/make_airliner.py --only s03_nadir  # (re)render some segments (ids as printed), no video
    python showcase/make_airliner.py --compose-only    # no rendering: compose and assemble

One flight (`route.py`: a trajectory CSV from lift-off to touchdown), shown as one continuous
time-lapse: `flight.speed` sets the playback speed along the route (real time at take-off and
landing, ×20 over each place worth a look, up to ×3000 in between, eased log-linearly), so the
video never jumps in time. `flight.cameras` switches the view at route times (front: ahead;
nadir: straight down; wide: a 100° view ahead, level, for the curvature of the horizon), with a
short dissolve. Each camera's stretch is one `terrain run` over a time-warped trajectory
(video time → flight time; `lighting.time_map` keeps the sun on the flight's clock). The
opening is a map flight recorded by `terrain view --record` (`globe`) with the route drawn over
it, the outro the route on a map. Captions, the speed, the readout and a route inset are
composed here in the showcase's style (make_showcase.py, whose helpers are used).
"""
import argparse, copy, csv, datetime, hashlib, json, math, os, subprocess, sys
import numpy as np, h5py, yaml
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_showcase as ms  # noqa: E402
import route as rt  # noqa: E402

HERE, ROOT = ms.HERE, ms.ROOT
OUT = os.path.join(ROOT, "out", "airliner")
ms.OUT = OUT
FONTS, u, ease, text_layer = ms.FONTS, ms.u, ms.ease, ms.text_layer
STORY = os.path.join(HERE, "airliner.yaml")

# camera presets (1920x1080); `pitch` is relative to the body (forward mount: up > 0)
PRESETS = {
    "front": {"extrinsics": {"mount": "forward", "pitch_deg": -16},
              "intrinsics": {"model": "pinhole", "width": 1920, "height": 1080, "intrinsics": [1371.0, 1371.0, 959.5, 539.5]}},
    "nadir": {"extrinsics": {"mount": "nadir", "pitch_deg": 0},
              "intrinsics": {"model": "pinhole", "width": 1920, "height": 1080, "intrinsics": [1371.0, 1371.0, 959.5, 539.5]}},
    "wide": {"extrinsics": {"mount": "forward", "pitch_deg": -3},
             "intrinsics": {"model": "pinhole", "width": 1920, "height": 1080, "intrinsics": [805.5, 805.5, 959.5, 539.5]}},
}


# ----------------------------------------------------------------------------- the flight
class Route:
    def __init__(self, story):
        cfg = story["route"]
        d = os.path.join(ROOT, cfg.get("out", "out/airliner"))
        self.csv = os.path.join(d, "route.csv")
        self.meta_path = os.path.join(d, "route.json")
        digest = hashlib.sha1((yaml.safe_dump(cfg) + open(os.path.join(HERE, "route.py")).read()).encode()).hexdigest()
        stamp = os.path.join(d, "route.done")
        if not (os.path.exists(stamp) and open(stamp).read() == digest and os.path.exists(self.csv)):
            subprocess.run([sys.executable, os.path.join(HERE, "route.py"), STORY], check=True, cwd=ROOT)
            with open(stamp, "w") as f:
                f.write(digest)
        self.meta = json.load(open(self.meta_path))
        self.data = np.loadtxt(self.csv, delimiter=",", skiprows=1)  # t, lat, lon, h, roll, pitch, yaw
        self.lines = open(self.csv).read().splitlines()
        self.takeoff = datetime.datetime.fromisoformat(str(story["route"]["takeoff_utc"]))
        tr = np.array(self.meta["track"])
        self.track = tr  # lat, lon, t, h (sparse)

    def window(self, t_first, t_last):
        """CSV text of the rows in [t_first, t_last] (route time), times from 0."""
        i0 = int(round(t_first * 10)) + 1
        i1 = min(int(round(t_last * 10)) + 2, len(self.lines))
        out = [self.lines[0]]
        for line in self.lines[i0:i1]:
            t, rest = line.split(",", 1)
            out.append(f"{float(t) - t_first:.2f},{rest}")
        return "\n".join(out) + "\n"

    def at(self, t):
        """(lat, lon, h, ground speed) at route time t."""
        d = self.data
        lat, lon, h = (float(np.interp(t, d[:, 0], d[:, i])) for i in (1, 2, 3))
        k = min(max(int(t * 10), 1), len(d) - 2)
        dn, de = rt.ellipsoid_steps(d[k - 1:k + 2, 1], d[k - 1:k + 2, 2])
        v = float(np.hypot(dn, de).sum() / (d[k + 1, 0] - d[k - 1, 0]))
        return lat, lon, h, v



def region_map(story, base):
    """The route's region as a Web-Mercator mosaic (z5 surface preview): image and the global
    pixel origin at that zoom."""
    m = story["map"]
    d = os.path.join(OUT, "map")
    os.makedirs(d, exist_ok=True)
    prefix = os.path.join(d, "region")
    png = prefix + "_rgb.png"
    if not os.path.exists(png):
        subprocess.run([ms.TERRAIN, "tiles", "--seed", str(base["world"]["seed"]), "--png", prefix, "--zoom", str(m["zoom"]),
                        "--size", str(m["size"]), "--at", f"{m['at'][0]},{m['at'][1]}", "--layers", "rgb"], cwd=ROOT, check=True)
    z, n = m["zoom"], m["size"]
    cx, cy = ms.merc_px(m["at"][0], m["at"][1], z)
    x0, y0 = (math.floor(cx / 256) - n // 2) * 256, (math.floor(cy / 256) - n // 2) * 256
    return Image.open(png).convert("RGB"), (x0, y0), z


class Inset:
    """The route on a small map, the flown part highlighted, the aircraft as a dot."""

    def __init__(self, story, base, route, size):
        img, (x0, y0), z = region_map(story, base)
        tr = route.track
        gx, gy = ms.merc_px(tr[:, 0], tr[:, 1], z)
        gx, gy = gx - x0, gy - y0
        pad = 0.18 * max(gx.max() - gx.min(), gy.max() - gy.min())
        cx, cy = (gx.max() + gx.min()) / 2, (gy.max() + gy.min()) / 2
        side = max(gx.max() - gx.min(), gy.max() - gy.min()) + 2 * pad
        box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        self.size = size
        k = size / side
        crop = img.crop(tuple(int(round(v)) for v in box)).resize((size, size), Image.LANCZOS)
        crop = Image.fromarray((np.asarray(crop).astype(np.float32) * 0.8).astype(np.uint8))
        self.bg = crop.convert("RGBA")
        self.px = (gx - box[0]) * k
        self.py = (gy - box[1]) * k
        self.t = tr[:, 2]
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=max(4, size // 18), fill=255)
        self.mask = mask

    def draw(self, t, alpha=1.0):
        im = self.bg.copy()
        d = ImageDraw.Draw(im)
        pts = list(zip(self.px, self.py))
        w = max(2, self.size // 110)
        # the route ahead thin and faint, the part flown thick orange on a dark outline (it
        # must read over ice as well as over sea)
        d.line(pts, fill=(255, 255, 255, 110), width=max(1, w - 1))
        j = int(np.searchsorted(self.t, t))
        if j > 0:
            p = pts[:j] + [(float(np.interp(t, self.t, self.px)), float(np.interp(t, self.t, self.py)))]
            d.line(p, fill=(0, 0, 0, 200), width=3 * w + 2, joint="curve")
            d.line(p, fill=(255, 140, 0, 255), width=2 * w + 1, joint="curve")
        x, y = float(np.interp(t, self.t, self.px)), float(np.interp(t, self.t, self.py))
        r = max(4, self.size // 45)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 140, 0, 255), outline=(0, 0, 0, 255), width=2)
        for i in (0, -1):
            d.ellipse([pts[i][0] - r * 0.6, pts[i][1] - r * 0.6, pts[i][0] + r * 0.6, pts[i][1] + r * 0.6], fill=(255, 255, 255, 230))
        a = Image.new("L", im.size, 0)
        a.paste(Image.eval(self.mask, lambda v: int(v * 0.92 * alpha)), (0, 0))
        im.putalpha(a)
        return im


# ----------------------------------------------------------------------------- composition
def fmt_dur(s):
    s = int(round(s))
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s % 60:02d} s" if m < 10 else f"{m} min"


def readout(route, t):
    lat, lon, h, v = route.at(t)
    clock = route.takeoff + datetime.timedelta(seconds=t)
    return (f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'}  {abs(lon):.3f}°{'E' if lon >= 0 else 'W'}  ·  {h:,.0f} m  ·  "
            f"{v * 3.6:,.0f} km/h  ·  {clock:%H:%M} UTC  ·  flight time {fmt_dur(t)}")


def globe_frames(shot, scn, video, route, band):
    """The recorded map flight with the route drawn over it: the line grows from `draw` [t0, t1]
    (shot seconds), the airports and the `labels` along it fade in."""
    W, H, fps = video["width"], video["height"], video["fps"]
    dur = shot["seconds"] + video["crossfade"]
    d = os.path.dirname(scn["output"]["file"])
    rows = list(csv.DictReader(open(scn["output"]["file"])))
    fov = math.radians(shot.get("fov", 40.0))
    f = (H / 2) / math.tan(fov / 2)
    tr = route.track
    P = geo2ecef(tr[:, 0], tr[:, 1], np.maximum(tr[:, 3], 0) * 0 + 2000.0)
    T = tr[:, 2] / tr[-1, 2]
    d0, d1 = shot.get("draw", [2.0, 8.0])
    labels = shot.get("labels", [])
    frames = []
    names = sorted(x for x in os.listdir(d) if x.startswith("frame_") and x.endswith(".png"))
    for k, name in enumerate(names[:int(round(dur * fps))]):
        tt = k / fps
        fr = ms.fit(np.asarray(Image.open(os.path.join(d, name)).convert("RGB")), W, H)
        r = rows[min(k, len(rows) - 1)]
        cam = geo2ecef(float(r["lat"]), float(r["lon"]), float(r["h"]))
        R = body_to_ecef(float(r["lat"]), float(r["lon"]), float(r["roll"]), float(r["pitch"]), float(r["yaw"]))
        pb = (P - cam) @ R  # body frame (forward, right, down)
        vis = (np.sum((cam - P) * P, -1) > 0) & (pb[:, 0] > 1.0)
        x = W / 2 + f * pb[:, 1] / np.maximum(pb[:, 0], 1e-6)
        y = H / 2 + f * pb[:, 2] / np.maximum(pb[:, 0], 1e-6)
        grow = ease((tt - d0) / (d1 - d0)) if d1 > d0 else 1.0
        img = Image.fromarray(fr).convert("RGBA")
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(lay)
        a_line = ease((tt - d0 + 0.3) / 0.3) * ease((dur - tt - 0.1) / 0.4)
        if a_line > 0:
            seg = []
            for i in range(len(P)):
                if vis[i] and T[i] <= grow:
                    seg.append((x[i], y[i]))
                else:
                    if len(seg) > 1:
                        route_line(dr, seg, a_line, u(4))
                    seg = []
            if len(seg) > 1:
                route_line(dr, seg, a_line, u(4))
            for i in (0, len(P) - 1):
                if vis[i] and (i == 0 or grow >= 1.0):
                    rr = u(5)
                    dr.ellipse([x[i] - rr, y[i] - rr, x[i] + rr, y[i] + rr], fill=(255, 255, 255, int(240 * a_line)), outline=(0, 0, 0, int(200 * a_line)), width=2)
        img = Image.alpha_composite(img, lay)
        items, places = [], []
        for lb in labels:
            i = int(np.argmin(np.abs(tr[:, 2] - lb["t"]))) if "t" in lb else None
            if i is None or not vis[i]:
                continue
            b = ease((grow - T[i] + 0.03) / 0.06) * ease((tt - lb.get("from", 0)) / 0.4) * ease((min(lb.get("until", dur), dur - 0.15) - tt) / 0.4)
            if b > 0:
                places.append((x[i], y[i], lb["text"], b, lb.get("dx", 14)))
        img = place_labels(img, places)
        for nt in shot.get("notes", []):
            end = min(nt.get("until", dur), dur - 0.15)
            b = ease((tt - nt["t"]) / 0.4) * ease((end - tt) / 0.4)
            if b > 0:
                items.append(((u(36), u(30)), nt["text"], FONTS.get("light", u(24)), b, "la"))
        if items:
            img = Image.alpha_composite(img, text_layer((W, H), items))
        fr = np.asarray(img.convert("RGB"))
        frames.append(ms.caption(fr, shot.get("label", ""), shot.get("text", ""), tt, dur, W, H, band))
    return frames


def place_labels(img, labels):
    """Place names on the route: a marker dot, and the name on a dark translucent pill beside it.
    labels: (x, y, text, alpha, dx) in pixels (dx < 0: to the left)."""
    if not labels:
        return img
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    font = FONTS.get("medium", u(18))
    for x, y, text, a, dx in labels:
        r = u(4)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, int(255 * a)), outline=(0, 0, 0, int(220 * a)), width=2)
        anchor = "lm" if dx >= 0 else "rm"
        tx = x + (u(dx) if dx >= 0 else -u(-dx))
        l, t, rr, b = d.textbbox((tx, y), text, font=font, anchor=anchor)
        pad = u(6)
        d.rounded_rectangle([l - pad, t - pad * 0.7, rr + pad, b + pad * 0.7], radius=u(7), fill=(12, 14, 18, int(170 * a)))
        d.text((tx, y), text, font=font, fill=(255, 255, 255, int(250 * a)), anchor=anchor)
    return Image.alpha_composite(img, lay)


def route_line(d, pts, a, w):
    """The route: orange on a dark outline."""
    d.line(pts, fill=(0, 0, 0, int(190 * a)), width=w + u(3), joint="curve")
    d.line(pts, fill=(255, 140, 0, int(255 * a)), width=w, joint="curve")


def geo2ecef(lat, lon, h):
    a, e2 = rt.A_WGS, rt.E2
    la, lo = np.radians(lat), np.radians(lon)
    n = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo), (n * (1 - e2) + h) * np.sin(la)], -1)


def body_to_ecef(lat, lon, roll, pitch, yaw):
    """Columns: body forward / right / down in ECEF (aerospace ZYX Euler angles in NED)."""
    la, lo = math.radians(lat), math.radians(lon)
    ned = np.array([[-math.sin(la) * math.cos(lo), -math.sin(lo), -math.cos(la) * math.cos(lo)],
                    [-math.sin(la) * math.sin(lo), math.cos(lo), -math.cos(la) * math.sin(lo)],
                    [math.cos(la), 0.0, -math.sin(la)]])  # columns: N, E, D in ECEF
    r, p, y = (math.radians(v) for v in (roll, pitch, yaw))
    cz, sz, cy, sy, cx, sx = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return ned @ (rz @ ry @ rx)


def map_card_frames(story, base, route, video, card, seconds):
    """The outro: the whole route on the region map, the place names along it, the numbers."""
    W, H, fps = video["width"], video["height"], video["fps"]
    img, (x0, y0), z = region_map(story, base)
    tr = route.track
    gx, gy = ms.merc_px(tr[:, 0], tr[:, 1], z)
    gx, gy = gx - x0, gy - y0
    # fit the route (with margin) into the frame's left 60 %
    side_w, side_h = (gx.max() - gx.min()) * 1.5, (gy.max() - gy.min()) * 1.25
    s = min(W * 0.56 / side_w, H * 0.9 / side_h)
    cw, ch = W * 0.56 / s, H / s
    cx, cy = (gx.max() + gx.min()) / 2, (gy.max() + gy.min()) / 2
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    bg = img.crop(tuple(int(round(v)) for v in box)).resize((int(W * 0.56), H), Image.LANCZOS)
    px, py = (gx - box[0]) * s, (gy - box[1]) * s
    T = tr[:, 2] / tr[-1, 2]
    n = int(seconds * fps)
    stats = card["lines"]
    for k in range(n):
        tt = k / fps
        canvas = Image.new("RGBA", (W, H), (10, 11, 14, 255))
        m = Image.fromarray((np.asarray(bg).astype(np.float32) * (0.55 + 0.25 * ease(tt / 1.0))).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(m)
        grow = ease((tt - 0.4) / 2.2)
        pts = [(a, b) for a, b, c in zip(px, py, T) if c <= grow]
        if len(pts) > 1:
            route_line(d, pts, 1.0, u(4))
        r = u(5)
        for i in ([0] + ([-1] if grow >= 1 else [])):
            d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
        places = []
        for lb in card.get("labels", []):
            i = int(np.argmin(np.abs(tr[:, 2] - lb["t"])))
            b = ease((grow - T[i] + 0.03) / 0.05)
            if b > 0:
                places.append((px[i], py[i], lb["text"], b, lb.get("dx", 14)))
        m = place_labels(m, places)
        canvas.alpha_composite(m, (0, 0))
        items = []
        x_text = int(W * 0.56) + u(40)
        items.append(((x_text, u(150)), card["title"], FONTS.get("light", u(40)), ease((tt - 0.3) / 0.6), "ls"))
        for i, l in enumerate(stats):
            items.append(((x_text, u(210) + u(38) * i), l, FONTS.get("light", u(21)), ease((tt - 0.8 - 0.25 * i) / 0.5), "ls"))
        items.append(((x_text, H - u(60)), card.get("footer", ""), FONTS.get("medium", u(16)), ease((tt - 2.5) / 0.6) * 0.8, "ls"))
        canvas = Image.alpha_composite(canvas, text_layer((W, H), items))
        yield np.asarray(canvas.convert("RGB"))



def encode(path, frames, video):
    ff = ms.ffmpeg_writer(path, video)
    for fr in frames:
        ff.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()


def assemble(parts, video, out_mp4):
    """Join the clips with `crossfade`-second fades (one ffmpeg pass)."""
    xf, fps = video["crossfade"], video["fps"]
    dur = [float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", c],
                                capture_output=True, text=True, check=True).stdout) for c in parts]
    graph, prev, t = [], "[0:v]", 0.0
    for i in range(1, len(parts)):
        t += dur[i - 1] - xf
        graph.append(f"{prev}[{i}:v]xfade=transition=fade:duration={xf}:offset={t:.4f}[v{i}]")
        prev = f"[v{i}]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for c in parts:
        cmd += ["-i", c]
    cmd += ["-filter_complex", ";".join(graph), "-map", prev, "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True)
    print(f"wrote {out_mp4}: {len(parts)} parts, {t + dur[-1]:.1f} s")



class Timeline:
    """Video time τ ↔ route time t (keys and captions at t < 0: seconds before touchdown): the
    playback speed v(t) is interpolated log-linearly between `flight.speed` keys ([t, v], or {slow: t, speed, width, up, ramp}: a slow pass
    centred on t, `width` route seconds at ×`speed`, eased over `ramp` route seconds from and to
    ×`up`), and τ(t) = ∫ dt / v."""

    def __init__(self, flight, route):
        T = route.meta["duration_s"]
        at = lambda t: T + t if t < 0 else t  # (< 0: seconds before touchdown)
        keys = []
        for k in flight["speed"]:
            if isinstance(k, dict):
                c, v, w, up, r = at(k["slow"]), k.get("speed", 20), k.get("width", 120), k.get("up", 300), k.get("ramp", 110)
                keys += [(c - w / 2 - r, up), (c - w / 2, v), (c + w / 2, v), (c + w / 2 + r, up)]
            else:
                keys.append((float(at(k[0])), float(k[1])))
        keys.sort()
        kt = np.array([k[0] for k in keys])
        if np.any(np.diff(kt) <= 0):
            raise SystemExit(f"flight.speed: overlapping keys near t = {kt[1:][np.diff(kt) <= 0]}")
        self.T = T
        self.t = np.append(np.arange(0.0, T, 0.05), T)
        self.v = np.exp(np.interp(self.t, kt, np.log([k[1] for k in keys])))
        dt = np.diff(self.t)
        self.tau = np.concatenate([[0.0], np.cumsum(dt * 0.5 * (1 / self.v[1:] + 1 / self.v[:-1]))])
        self.end = float(self.tau[-1])

    def t_at(self, tau):
        return np.interp(tau, self.tau, self.t)

    def tau_at(self, t):
        return float(np.interp(t, self.t, self.tau))

    def speed(self, t):
        return float(np.interp(t, self.t, self.v))


def warped_pose(route, tl, taus):
    """Poses at video times `taus` (rows t, lat, lon, h, roll, pitch, yaw of the route at
    t(τ)); the attitude averaged over 0.2 s of video, so turbulence does not flicker when sped
    up (turns, slower, stay)."""
    d = route.data
    t = tl.t_at(taus)
    out = [t] + [np.interp(t, d[:, 0], d[:, i]) for i in (1, 2, 3)]
    w = np.maximum(0.2 * np.interp(t, tl.t, tl.v), 0.1)
    for i in (4, 5, 6):
        x = np.degrees(np.unwrap(np.radians(d[:, i]))) if i == 6 else d[:, i]
        c = np.concatenate([[0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(d[:, 0]))])
        a, b = np.clip(t - w / 2, 0, d[-1, 0]), np.clip(t + w / 2, 0, d[-1, 0])
        avg = (np.interp(b, d[:, 0], c) - np.interp(a, d[:, 0], c)) / np.maximum(b - a, 1e-9)
        out.append(np.where(b - a > 1e-6, avg, np.interp(t, d[:, 0], x)))
    out[6] = out[6] % 360.0
    return np.stack(out, -1)


def segments(story, tl, video):
    """The camera stretches: [{id, camera, pitch, tau0, tau1}] (video seconds, on the frame
    grid; each extends half a dissolve into its neighbours)."""
    fps = video["fps"]
    d = story["flight"].get("dissolve", 0.6)
    cams = story["flight"]["cameras"]
    snap = lambda x: round(x * fps) / fps
    bounds = [0.0] + [snap(tl.tau_at(c["from"])) for c in cams[1:]] + [snap(tl.end)]
    out = []
    for i, c in enumerate(cams):
        a = max(0.0, bounds[i] - (d / 2 if i else 0.0))
        b = min(snap(tl.end - 1.0 / fps), bounds[i + 1] + (d / 2 if i < len(cams) - 1 else 0.0))
        out.append({"id": f"s{i + 1:02d}_{c['camera']}", "camera": c["camera"], "pitch": c.get("pitch"), "scenario": c.get("scenario", {}),
                    "tau0": snap(a), "tau1": snap(b), "cut": bounds[i]})
    return out


def segment_scenario(base, seg, video, route, tl, stills=False):
    """A camera stretch: its scenario (frames every 1/fps of video time) and warped trajectory."""
    cam = copy.deepcopy(PRESETS[seg["camera"]])
    # (no motion blur: the exposure, in video time, would span kilometres of the sped-up flight)
    cam["rgb"] = {"sensor": {"motion_blur": {"enabled": False}}}
    if seg.get("pitch") is not None:
        cam["extrinsics"]["pitch_deg"] = seg["pitch"]
    over = copy.deepcopy(seg["scenario"])
    over["cameras"] = [ms.deep_merge(cam, c) for c in over.get("cameras", [{}])]
    start = base["output"]["start"]
    span = seg["tau1"] - seg["tau0"] + 0.5 / video["fps"]
    shot = {"id": seg["id"], "seconds": span - video["crossfade"], "scenario": over}
    scn = ms.shot_scenario(base, shot, video, stills)
    scn["output"]["end"] = start + span
    tau_first = seg["tau0"] - start
    taus = np.arange(tau_first, seg["tau1"] + 1.0, 0.01)
    P = warped_pose(route, tl, taus)
    t_first = float(P[0, 0])
    rows = ["t,lat,lon,h,roll,pitch,yaw"] + [f"{tau - tau_first:.3f},{p[1]:.8f},{p[2]:.8f},{p[3]:.2f},{p[4]:.3f},{p[5]:.3f},{p[6]:.3f}" for tau, p in zip(taus, P)]
    clock = route.takeoff + datetime.timedelta(seconds=t_first)
    light = scn["render"]["lighting"]
    light["date"] = clock.strftime("%Y-%m-%d")
    light["time_utc"] = clock.strftime("%H:%M:%S.%f")[:-3]
    k = slice(None, None, 10)  # (every 0.1 s of video)
    light["time_map"] = [[round(float(a - tau_first), 3), round(float(b - t_first), 3)] for a, b in zip(taus[k], P[k, 0])]
    scn["trajectory"].pop("synth", None)
    return scn, "\n".join(rows) + "\n"


def nice_speed(v):
    """The speed shown: real time, or × a round number."""
    if v < 1.25:
        return "real time"
    steps = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600, 800,
             1000, 1500, 2000, 2500, 3000, 4000, 5000]
    n = min(steps, key=lambda s: abs(math.log(s / v)))
    return f"speed ×{n:,}"


def flight_frames(story, segs, scns, video, route, tl, inset, band):
    """The continuous flight: every video frame from its camera stretch (dissolving at a
    switch), the readout, the speed, the caption of the moment, the route inset."""
    W, H, fps = video["width"], video["height"], video["fps"]
    d = story["flight"].get("dissolve", 0.6)
    files = [h5py.File(s["output"]["file"], "r")["/cam0"]["rgb"] for s in scns]
    caps = [dict(c, tau=tl.tau_at(c["at"] if c["at"] >= 0 else tl.T + c["at"])) for c in story["flight"]["captions"]]
    n = int(round(tl.end * fps))
    dur = n / fps

    def cam_frame(i, tau):
        g = files[i]
        k = min(max(int(round((tau - segs[i]["tau0"]) * fps)), 0), g.shape[0] - 1)
        return g[k].astype(np.float32)

    for K in range(n):
        tau = K / fps
        t = float(tl.t_at(tau))
        # the stretch of this moment, and its neighbour within half a dissolve of the cut
        i = max(j for j, s in enumerate(segs) if s["cut"] <= tau + 1e-9)
        fr = cam_frame(i, tau)
        for j in (i - 1, i + 1):
            if 0 <= j < len(segs):
                cut = segs[max(i, j)]["cut"]
                x = (tau - cut) / d + 0.5  # 0 → 1 across the dissolve
                if 0 < x < 1 and segs[j]["tau0"] <= tau <= segs[j]["tau1"]:
                    a = ease(x) if j > i else 1 - ease(x)
                    fr = fr * (1 - a) + cam_frame(j, tau) * a
        img = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)).convert("RGBA")
        a = ease((tau - 0.35) / 0.45) * ease((dur - tau - 0.15) / 0.45)
        items = [((W - u(14), u(12)), readout(route, t), FONTS.get("medium", u(13)), 0.85 * a, "ra"),
                 ((u(36), u(30)), nice_speed(tl.speed(t)), FONTS.get("medium", u(24)), a, "la")]
        img = Image.alpha_composite(img, text_layer((W, H), items))
        if inset is not None and a > 0:
            img.alpha_composite(inset.draw(t, a), (W - u(36) - inset.size, H - u(78) - inset.size))
        frame = np.asarray(img.convert("RGB"))
        for c in caps:
            if c["tau"] <= tau < c["tau"] + c.get("seconds", 6.0):
                frame = ms.caption(frame, c.get("label", ""), c.get("text", ""), tau - c["tau"], c.get("seconds", 6.0), W, H, band)
        yield frame


def stamp_ok(path, digest, src_mtime=0.0):
    st = path + ".done"
    return os.path.exists(path) and os.path.exists(st) and open(st).read() == digest and os.path.getmtime(path) > src_mtime


def write_stamp(path, digest):
    with open(path + ".done", "w") as f:
        f.write(digest)


def clip(path, digest, frames_fn, video, src_mtime=0.0):
    if stamp_ok(path, digest, src_mtime):
        print(f"{os.path.basename(path)} up to date")
        return path
    encode(path, frames_fn(), video)
    write_stamp(path, digest)
    print(f"wrote {path}", flush=True)
    return path


def stills_sheet(done, path):
    rows = []
    for seg, scn in done:
        with h5py.File(scn["output"]["file"], "r") as f:
            rgb = f["/cam0"]["rgb"]
            ims = [ms.fit(rgb[k], 480, 270) for k in range(min(3, rgb.shape[0]))]
        row = Image.new("RGB", (480 * 3 + 220, 270), (20, 20, 20))
        for i, im in enumerate(ims):
            row.paste(Image.fromarray(im), (220 + 480 * i, 0))
        ImageDraw.Draw(row).text((10, 120), seg["id"], font=FONTS.get("medium", 22), fill=(255, 255, 255))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, 270 * len(rows)))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, 270 * i))
    sheet.save(path)
    print("wrote", path)


def main():
    global STORY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="render only these segments (or globe)")
    ap.add_argument("--compose-only", action="store_true")
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--story", default=STORY)
    ap.add_argument("--out", default=os.path.join(OUT, "airliner.mp4"))
    a = ap.parse_args()
    STORY = a.story
    story = yaml.safe_load(open(STORY))
    video = story["video"]
    ms.UI = video["width"] / 1280
    base = ms.deep_merge(yaml.safe_load(open(os.path.join(HERE, "base.yaml"))), story.get("base", {}))
    os.makedirs(OUT, exist_ok=True)
    route = Route(story)
    tl = Timeline(story["flight"], route)
    segs = segments(story, tl, video)
    print(f"flight: {tl.end:.1f} s of video for {tl.T / 3600:.2f} h; segments: " +
          ", ".join(f"{s['id']} {s['tau0']:.1f}–{s['tau1']:.1f} s" for s in segs), flush=True)
    globe = story["globe"]
    # ---- render
    scns = []
    for s in segs:
        scn, traj = segment_scenario(base, s, video, route, tl, a.stills)
        if not a.compose_only and (not a.only or s["id"] in a.only):
            ms.render_shot(base, s, video, stills=a.stills, force=a.force, scn=scn, traj=traj)
            if story.get("drop_tiles"):
                # (each stretch flies over its own part of the planet: the store is regenerated
                # on the GPU in minutes, and the whole flight's would be tens of GB)
                store = os.path.join(ROOT, scn["tiles"]["file"])
                if os.path.exists(store):
                    os.remove(store)
        scns.append(scn)
    if a.stills:
        stills_sheet([(s, c) for s, c in zip(segs, scns) if os.path.exists(c["output"]["file"])], os.path.join(OUT, "stills.png"))
        return
    if not a.compose_only and (not a.only or "globe" in a.only):
        gscn = ms.render_globe(base, globe, video, a.force)
    else:
        gscn = ms.globe_scenario(base, globe)
    if a.only:
        return
    # ---- compose
    d = os.path.join(OUT, "clips")
    os.makedirs(d, exist_ok=True)
    inset = Inset(story, base, route, ms.u(story.get("inset_px", 150)))
    band = ms.bottom_band(video["width"], video["height"])
    meta = json.dumps(route.meta["phases"])
    newest = lambda files: max(os.path.getmtime(f) for f in files)
    g = clip(os.path.join(d, "globe.mp4"), hashlib.sha1((yaml.safe_dump([globe, video]) + meta).encode()).hexdigest(),
             lambda: globe_frames(globe, gscn, video, route, band), video, newest([gscn["output"]["file"]]))
    files = [s["output"]["file"] for s in scns]
    f = clip(os.path.join(d, "flight.mp4"), hashlib.sha1((yaml.safe_dump([story["flight"], video]) + meta).encode()).hexdigest(),
             lambda: flight_frames(story, segs, scns, video, route, tl, inset, band), video, newest(files))
    # title over a collage of the flight (the middle frame of every camera stretch)
    src = []
    for s in scns:
        with h5py.File(s["output"]["file"], "r") as h:
            rgb = h["/cam0"]["rgb"]
            src.append(rgb[rgb.shape[0] // 2])
    title = os.path.join(d, "title.mp4")
    encode(title, ms.title_frames(video, 5.0 + video["crossfade"], src), video)
    outro = os.path.join(d, "outro.mp4")
    encode(outro, map_card_frames(story, base, route, video, story["outro"], story["outro"].get("seconds", 8.0)), video)
    assemble([title, g, f, outro], video, a.out)


if __name__ == "__main__":
    main()
