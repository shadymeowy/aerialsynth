#!/usr/bin/env python3
"""A gallery of stills of the next generator's world (one 1280 x 720 still per scene).

`scripts/gallery/<scene>.yaml` are complete scenarios (`world.generator: next`), one per image,
named `gNN_<name>`. Their header comments describe the image for the index:

    # title: alpine peaks at golden hour
    # category: mountains
    # landcover: snow-covered high mountains, temperate conifer valleys
    # look: what to look at in the image
    # target: LAT LON (the ground point at the image centre; `new` placed the camera from it)
    # status: notes on the last render (free text, not parsed)

The camera flies a short straight line (`trajectory.synth`, altitude above the ground) and the
still is the frame at t = 2 s. Usage:

    # a new scene: the camera looks at TARGET from ALT m above the ground, PITCH down, along HEADING
    # (or --at LAT LON: the camera's own position); the local solar time sets time_utc
    scripts/gallery.py new g01_alpine --target 48.5 -171.0 --alt 1200 --heading 60 --pitch -25 \\
        --date 2026-06-21 --local 19:30 --mode golden --title ... --category ... --landcover ... --look ...
    # render (needs only the `terrain` binary; GPU: run it on a machine with one)
    scripts/gallery.py render --terrain target/release/terrain --out /tmp/gallery [SCENE ...]
    # the JPEGs (gNN_name.jpg, 1280 x 720), contact sheets and index (needs numpy, h5py, Pillow)
    scripts/gallery.py jpeg --run /tmp/gallery --out gallery/ [SCENE ...]
    scripts/gallery.py sheet --out gallery/
    scripts/gallery.py index --out gallery/ [--notes notes.csv]

`render` and `jpeg` report scenes whose run.log says the renderer fell back to the CPU.
`index` writes index.csv and index.md from the scene headers and poses; `--notes` (CSV with
columns scene,artefacts) adds the artefacts noticed per image.
"""
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "scripts" / "gallery"
W, H, FX = 1280, 720, 914.0  # 70 deg horizontal field of view
T_STILL = 2.0  # s: the still's time on the trajectory (auto exposure settled)

# sensor exposure per lighting mode (night: the night@type still's; dusk: dusk@type's)
EXPOSURE = {
    "day": "        mode: auto\n        tau_s: 0.4\n        target: 0.13\n",
    "golden": "        mode: auto\n        tau_s: 0.4\n        target: 0.13\n",
    "dusk": "        mode: auto\n        tau_s: 0.25\n        target: 0.13\n        max_gain: 160\n        max_time_s: 0.04\n",
    "night": "        mode: auto\n        tau_s: 0.4\n        target: 0.012\n        max_gain: 64\n        max_time_s: 0.008\n",
}
OPTICS = {
    "night": "        defocus_px: 0.0\n        chromatic_aberration_px: 0.0\n        vignette: 0.25\n        bloom: 0.02\n",
    "other": "        defocus_px: 0.25\n        chromatic_aberration_px: 0.3\n        vignette: 0.25\n        bloom: 0.06\n",
}
NOISE = {"night": "        read: 0.04\n        shot: 0.06\n", "other": "        read: 0.08\n        shot: 0.1\n"}

TEMPLATE = """\
{header}world:
  seed: 1
  generator: next
tiles:
  file: out/world.h5
  max_zoom: 18
  margin: 0
  plan_every: 5
trajectory:
  synth:
    rate_hz: 400
    speed_mps: {speed}
    altitude_m: {alt}
    heading_deg: {heading}
    kind: line
    lat: {lat:.5f}
    lon: {lon:.5f}
    duration_s: 3.0
render:
  supersample: 2
  shading: relit
  lighting:
    mode: clock
    date: '{date}'
    time_utc: '{utc}'
    time_scale: 1.0
    sky_intensity: 0.55
  atmosphere:
    visibility_km: {vis}
    inscatter: 1.0
cameras:
- path: /cam0
  intrinsics:
    model: pinhole
    width: {w}
    height: {h}
    intrinsics:
    - {fx}
    - {fx}
    - {cx}
    - {cy}
  extrinsics:
    mount: {mount}
    pitch_deg: {pitch}
  frame_rate: 1.0
  rgb:
    sensor:
      exposure:
{exposure}      motion_blur:
        enabled: true
        max_samples: 24
      optics:
{optics}      noise:
        enabled: true
{noise}      tone:
        curve: filmic
        saturation: 0.92
    supersample: 2
output:
  start: {t}
  end: {t}
  max_frames: 1
  compression:
    level: 1
"""

HEAD_KEYS = ["title", "category", "landcover", "look", "target"]


def scenes(names):
    all_ = sorted(p.stem for p in CFG.glob("*.yaml"))
    if not names:
        return all_
    out = []
    for n in names:
        m = [s for s in all_ if s == n or s.split("_")[0] == n]
        if not m:
            sys.exit(f"no scene {n!r} (have: {', '.join(all_)})")
        out += m
    return out


def offset(lat, lon, dist_m, az_deg):
    """The point dist_m from (lat, lon) along the azimuth az_deg (spherical, short distances)."""
    r = 6371000.0
    la, lo, az, d = math.radians(lat), math.radians(lon), math.radians(az_deg), dist_m / r
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(az))
    lo2 = lo + math.atan2(math.sin(az) * math.sin(d) * math.cos(la), math.cos(d) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), (math.degrees(lo2) + 540.0) % 360.0 - 180.0


def new(a):
    nadir = a.pitch <= -89.5
    if a.target:
        tl, tn = a.target
        d = 0.0 if nadir else a.alt / math.tan(math.radians(-a.pitch))
        # the still is taken T_STILL s along the line: start that much earlier
        lat, lon = offset(tl, tn, d + a.speed * T_STILL, a.heading + 180.0)
    else:
        lat, lon = a.at
        tl = tn = None
    hh, mm = map(int, a.local.split(":"))
    utc_h = (hh + mm / 60.0 - lon / 15.0) % 24.0
    date = a.date
    day = (hh + mm / 60.0 - lon / 15.0) // 24  # local date given: the UTC date may differ
    if day:
        import datetime

        date = (datetime.date.fromisoformat(a.date) + datetime.timedelta(days=int(day))).isoformat()
    utc = f"{int(utc_h):02d}:{int(round((utc_h % 1) * 60)) % 60:02d}:00"
    if int(round((utc_h % 1) * 60)) == 60:
        utc = f"{(int(utc_h) + 1) % 24:02d}:00:00"
    head = {
        "title": a.title,
        "category": a.category,
        "landcover": a.landcover,
        "look": a.look,
        "target": f"{tl:.5f} {tn:.5f}" if tl is not None else "",
    }
    header = "".join(f"# {k}: {v}\n" for k, v in head.items() if v)
    header += f"# local solar time {a.local} on {a.date}\n"
    mode = a.mode
    text = TEMPLATE.format(
        header=header,
        speed=a.speed,
        alt=a.alt,
        heading=a.heading,
        lat=lat,
        lon=lon,
        date=date,
        utc=utc,
        vis=a.visibility,
        w=W,
        h=H,
        fx=FX,
        cx=(W - 1) / 2,
        cy=(H - 1) / 2,
        mount="nadir" if nadir else "forward",
        pitch=0 if nadir else a.pitch,
        exposure=EXPOSURE[mode],
        optics=OPTICS["night" if mode == "night" else "other"],
        noise=NOISE["night" if mode == "night" else "other"],
        t=T_STILL,
    )
    CFG.mkdir(parents=True, exist_ok=True)
    p = CFG / f"{a.name}.yaml"
    p.write_text(text)
    print(f"{p.relative_to(ROOT)}: camera {lat:.5f} {lon:.5f}, {utc} UTC {date}")


def render(a):
    out = Path(a.out).resolve()
    terrain = str(Path(a.terrain).resolve())
    for s in scenes(a.scenes):
        d = out / s
        if (d / "out" / "seq.h5").exists() and not a.force:
            print(f"{s}: done")
            continue
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        shutil.copy(CFG / f"{s}.yaml", d / "cfg.yaml")
        with open(d / "run.log", "w") as log:
            r = subprocess.run([terrain, "run", "-c", "cfg.yaml"], cwd=d, stdout=log, stderr=subprocess.STDOUT)
        (d / "out" / "world.h5").unlink(missing_ok=True)  # the tiles are not kept
        cpu = " (RENDERED ON THE CPU)" if "on the CPU" in (d / "run.log").read_text(errors="replace") else ""
        print(f"{s}: exit {r.returncode}{cpu}", flush=True)


def header(s):
    h = {}
    for line in (CFG / f"{s}.yaml").read_text().splitlines():
        m = re.match(r"# (\w+): (.*)", line)
        if m and m.group(1) in HEAD_KEYS:
            h[m.group(1)] = m.group(2)
    return h


def jpeg(a):
    import h5py
    from PIL import Image

    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in scenes(a.scenes):
        h5 = run / s / "out" / "seq.h5"
        if not h5.exists():
            print(f"{s}: missing")
            continue
        log = (run / s / "run.log").read_text(errors="replace")
        with h5py.File(h5, "r") as f:
            im = f["cam0/rgb"][0]
        Image.fromarray(im).save(out / f"{s}.jpg", quality=a.quality)
        print(f"{s}.jpg{' (RENDERED ON THE CPU)' if 'on the CPU' in log else ''}")


def label(im, text, size):
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox((6, 4), text, font=font)
    d.rectangle([0, 0, x1 + 6, y1 + 4], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 230, 0), font=font)


def sheet(a):
    from PIL import Image

    out = Path(a.out)
    names = [s for s in scenes(a.scenes) if (out / f"{s}.jpg").exists()]
    w, h = a.width, a.width * H // W
    per, ncol = a.per_sheet, a.cols
    for k in range(0, len(names), per):
        chunk = names[k : k + per]
        nrow = (len(chunk) + ncol - 1) // ncol
        sh = Image.new("RGB", (w * ncol, h * nrow), (20, 20, 20))
        for i, s in enumerate(chunk):
            t = Image.open(out / f"{s}.jpg").convert("RGB").resize((w, h), Image.LANCZOS)
            title = header(s).get("title", "")
            label(t, f"{s.split('_')[0]}  {title}", max(12, w // 32))
            sh.paste(t, ((i % ncol) * w, (i // ncol) * h))
        p = out / f"sheet{k // per + 1}.jpg"
        sh.save(p, quality=88)
        print(p)


def pose(s):
    import yaml

    c = yaml.safe_load((CFG / f"{s}.yaml").read_text())
    syn, light, cam = c["trajectory"]["synth"], c["render"]["lighting"], c["cameras"][0]["extrinsics"]
    pitch = -90.0 if cam.get("mount") == "nadir" else float(cam["pitch_deg"])
    hh, mm, _ = map(int, str(light["time_utc"]).split(":"))
    loc = (hh + mm / 60.0 + float(syn["lon"]) / 15.0) % 24.0
    return {
        "lat": f"{syn['lat']:.4f}",
        "lon": f"{syn['lon']:.4f}",
        "alt_agl_m": syn["altitude_m"],
        "heading_deg": syn["heading_deg"],
        "pitch_deg": pitch,
        "date_utc": light["date"],
        "time_utc": light["time_utc"],
        "local_solar": f"{int(loc):02d}:{int(round(loc % 1 * 60)) % 60:02d}",
    }


def index(a):
    out = Path(a.out)
    notes = {}
    if a.notes:
        with open(a.notes) as f:
            notes = {r["scene"]: r["artefacts"] for r in csv.DictReader(f)}
    rows = []
    for s in scenes(a.scenes):
        h, p = header(s), pose(s)
        tgt = h.get("target", "").split()
        rows.append(
            {
                "n": s.split("_")[0][1:],
                "scene": s,
                "title": h.get("title", ""),
                "category": h.get("category", ""),
                **p,
                "target_lat": tgt[0] if tgt else "",
                "target_lon": tgt[1] if tgt else "",
                "landcover": h.get("landcover", ""),
                "look": h.get("look", ""),
                "artefacts": notes.get(s, ""),
            }
        )
    with open(out / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "index.md", "w") as f:
        f.write("# Gallery of the next generator\n\n")
        f.write("Stills of `world.generator: next` (1280 x 720, supersample 2, GPU renders). Camera position (lat, lon), ")
        f.write("altitude above the ground, heading / pitch (-90: nadir), date and local solar time; the target is the ")
        f.write("ground point at the image centre. Configs: `scripts/gallery/<scene>.yaml`.\n\n")
        f.write("| # | image | category | lat, lon | alt AGL | hdg / pitch | date, local time | biome / landcover | look at | artefacts noticed |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['n']} | [{r['title']}]({r['scene']}.jpg) | {r['category']} | {r['lat']}, {r['lon']} | {r['alt_agl_m']} m | "
                f"{r['heading_deg']} / {r['pitch_deg']:g} | {r['date_utc']} {r['local_solar']} | {r['landcover']} | {r['look']} | {r['artefacts']} |\n"
            )
    print(out / "index.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="write scripts/gallery/NAME.yaml")
    n.add_argument("name")
    g = n.add_mutually_exclusive_group(required=True)
    g.add_argument("--target", nargs=2, type=float, metavar=("LAT", "LON"), help="the ground point at the image centre")
    g.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"), help="the camera's position")
    n.add_argument("--alt", type=float, required=True, help="m above the ground")
    n.add_argument("--heading", type=float, required=True)
    n.add_argument("--pitch", type=float, required=True, help="deg (-90: nadir)")
    n.add_argument("--date", required=True, help="local date YYYY-MM-DD")
    n.add_argument("--local", required=True, help="local solar time HH:MM")
    n.add_argument("--mode", choices=list(EXPOSURE), default="day")
    n.add_argument("--speed", type=float, default=50.0)
    n.add_argument("--visibility", type=float, default=40.0, help="km")
    for k in HEAD_KEYS[:4]:
        n.add_argument(f"--{k}", default="")
    r = sub.add_parser("render")
    r.add_argument("--terrain", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--force", action="store_true")
    r.add_argument("scenes", nargs="*")
    j = sub.add_parser("jpeg")
    j.add_argument("--run", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--quality", type=int, default=92)
    j.add_argument("scenes", nargs="*")
    s = sub.add_parser("sheet")
    s.add_argument("--out", required=True, help="the directory of the JPEGs")
    s.add_argument("--per-sheet", type=int, default=9)
    s.add_argument("--cols", type=int, default=3)
    s.add_argument("--width", type=int, default=640)
    s.add_argument("scenes", nargs="*")
    i = sub.add_parser("index")
    i.add_argument("--out", required=True)
    i.add_argument("--notes")
    i.add_argument("scenes", nargs="*")
    a = ap.parse_args()
    {"new": new, "render": render, "jpeg": jpeg, "sheet": sheet, "index": index}[a.cmd](a)


if __name__ == "__main__":
    os.umask(0o022)
    main()
