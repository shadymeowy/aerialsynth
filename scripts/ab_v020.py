#!/usr/bin/env python3
"""A/B of the generator against v0.2.0 at the v0.2.0 showcase poses.

The v0.2.0 showcase stills (`showcase/stills.yaml` of v0.2.0, one frame per scene, 640 x 360)
are kept as small JPEGs in `docs/next/ab/v020/<scene>.jpg`; `scripts/ab_v020/<scene>.yaml` are the
same scenes with today's config keys. A scene `<scene>@<tag>.yaml` is the same camera at another
place: a place of the same landscape type in today's world (its world differs from v0.2.0's,
so the identical pose often shows another landscape). The scenes set `world.generator: next`
(the modular generator; `classic` is the v0.2.0 one).

    # render (needs only the `terrain` binary; GPU: run it on a machine with one)
    scripts/ab_v020.py render --terrain target/release/terrain --out /tmp/ab [SCENE ...]
    # side-by-side sheets and image statistics (needs numpy, h5py, Pillow)
    scripts/ab_v020.py sheet --run after=/tmp/ab [--run before=/tmp/ab0] --out sheets/
    # the reference JPEGs from a v0.2.0 showcase output (out/showcase/<scene>/stills.h5)
    scripts/ab_v020.py refs --showcase out/showcase

Audit (unbiased): `scripts/ab_v020/audit_places.csv` lists 30 deterministic pseudo-random land
places of the next world (`probe_px audit 30 7`: area-uniform, every Köppen group, >= 400 km
apart, no hand picking) with the nearest classic-world land point. `audit` renders them with the
camera of the `farmland` still (700 m above the ground, 42° down, 640 x 360, June, the farmland
still's local solar time 10:41; December south of the equator) with either generator; `sheet --audit` puts them side by side.

    scripts/ab_v020.py audit --terrain target/release/terrain --generator classic --out /tmp/audit_c
    scripts/ab_v020.py audit --terrain target/release/terrain --generator next --out /tmp/audit_n
    scripts/ab_v020.py sheet --audit --run classic=/tmp/audit_c --run next=/tmp/audit_n --out sheets/

`sheet` writes one JPEG per 6 scenes (columns: v0.2.0, then each run) and `stats.csv` /
`stats.md`: per image, the mean saturation, the high-frequency energy (mean |Laplacian| of
the luma), the share of flat 8 x 8 blocks (luma std < 1.5 DN) and the edge contrast (99th /
50th percentile of the gradient), over the lower two thirds of the frame (mostly ground).
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "scripts" / "ab_v020"
REF = ROOT / "docs" / "next" / "ab" / "v020"
AUDIT = CFG / "audit_places.csv"
FRAME = 1  # the frame of each sequence shown (of 4)


def scenes(names):
    all_ = sorted(p.stem for p in CFG.glob("*.yaml"))
    if not names:
        return all_
    out = []
    for n in names:
        m = [s for s in all_ if s == n or s.startswith(n + "@")]
        if not m:
            sys.exit(f"no scene {n!r} (have: {', '.join(all_)})")
        out += m
    return out


def audit_places():
    with open(AUDIT) as f:
        return list(csv.DictReader(f))


def audit_cfg(place, generator):
    """The farmland still's config at an audit place (local solar time kept)."""
    import re

    lat, lon = (place["classic_lat"], place["classic_lon"]) if generator == "classic" else (place["lat"], place["lon"])
    text = (CFG / "farmland.yaml").read_text()
    h = (8.5 + (32.775 - float(lon)) / 15.0) % 24.0
    hh, mm = int(h), int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = (hh + 1) % 24, 0
    text = text.replace("generator: next", f"generator: {generator}")
    text = re.sub(r"\n    lat: .*\n", f"\n    lat: {lat}\n", text)
    text = re.sub(r"\n    lon: .*\n", f"\n    lon: {lon}\n", text)
    text = text.replace("time_utc: 08:30:00", f"time_utc: '{hh:02d}:{mm:02d}:00'")
    if float(lat) < 0:  # the same season: southern summer
        text = text.replace("date: '2026-06-21'", "date: '2026-12-21'")
    return f"# audit place {place['id']} ({place['koppen']}, {place['biome']})\n" + text


def audit(a):
    places = [p for p in audit_places() if not a.scenes or p["id"] in a.scenes]
    run_all(a, [(p["id"], audit_cfg(p, a.generator)) for p in places])


def render(a):
    run_all(a, [(s, (CFG / f"{s}.yaml").read_text()) for s in scenes(a.scenes)])


def run_all(a, jobs):
    out = Path(a.out).resolve()
    terrain = str(Path(a.terrain).resolve())
    for s, cfg in jobs:
        d = out / s
        if (d / "out" / "seq.h5").exists() and not a.force:
            print(f"{s}: done")
            continue
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        (d / "cfg.yaml").write_text(cfg)
        with open(d / "run.log", "w") as log:
            r = subprocess.run([terrain, "run", "-c", "cfg.yaml"], cwd=d, stdout=log, stderr=subprocess.STDOUT)
        # the tiles are not kept (hundreds of MB per scene)
        (d / "out" / "world.h5").unlink(missing_ok=True)
        print(f"{s}: exit {r.returncode}", flush=True)


def load_frame(h5):
    import h5py

    with h5py.File(h5, "r") as f:
        rgb = f["cam0/rgb"]
        return rgb[min(FRAME, rgb.shape[0] - 1)]


def stats(img):
    import numpy as np

    a = img.astype(np.float64)
    a = a[a.shape[0] // 3 :]
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0).mean()
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = np.abs(4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]).mean()
    h, w = (y.shape[0] // 8) * 8, (y.shape[1] // 8) * 8
    b = y[:h, :w].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3))
    flat = (b < 1.5).mean()
    gx, gy = np.diff(y, axis=1)[:-1], np.diff(y, axis=0)[:, :-1]
    g = np.hypot(gx, gy)
    edge = np.percentile(g, 99) / max(np.percentile(g, 50), 0.25)
    return {"sat": sat, "hf": lap, "flat": flat, "edge": edge}


def label(im, text):
    from PIL import ImageDraw

    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 6 * len(text), 13], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 0))


def sheet(a):
    import numpy as np
    from PIL import Image

    runs = []
    for r in a.run:
        tag, _, path = r.partition("=")
        runs.append((tag, Path(path)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.audit:
        names = [p["id"] for p in audit_places() if not a.scenes or p["id"] in a.scenes]
        desc = {p["id"]: f"{p['koppen']} {p['biome']}" for p in audit_places()}
    else:
        names = scenes(a.scenes)
    rows, table = [], []
    for s in names:
        base = s.split("@")[0]
        ref = REF / f"{base}.jpg"
        cells = []
        if ref.exists() and not a.audit:
            im = np.asarray(Image.open(ref).convert("RGB"))
            cells.append(("v0.2.0", im))
        for tag, path in runs:
            h5 = path / s / "out" / "seq.h5"
            if h5.exists():
                cells.append((tag, load_frame(h5)))
            else:
                cells.append((tag + " (missing)", None))
        tiles = []
        for tag, im in cells:
            if im is None:
                t = Image.new("RGB", (640, 360), (40, 40, 40))
            else:
                t = Image.fromarray(im).resize((640, 360))
                st = stats(np.asarray(t))
                table.append({"scene": s, "run": tag, **{k: round(v, 3) for k, v in st.items()}})
            label(t, f"{s} {desc[s]} | {tag}" if a.audit else f"{s} | {tag}")
            tiles.append(t)
        rows.append(tiles)
    ncol = max(len(r) for r in rows)
    per = a.per_sheet
    for k in range(0, len(rows), per):
        chunk = rows[k : k + per]
        w = int(640 * a.scale)
        h = int(360 * a.scale)
        sh = Image.new("RGB", (w * ncol, h * len(chunk)), (20, 20, 20))
        for i, row in enumerate(chunk):
            for j, t in enumerate(row):
                sh.paste(t.resize((w, h), Image.LANCZOS), (j * w, i * h))
        p = out / f"sheet{k // per + 1}.jpg"
        sh.save(p, quality=85)
        print(p)
    with open(out / "stats.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["scene", "run", "sat", "hf", "flat", "edge"])
        wr.writeheader()
        wr.writerows(table)
    with open(out / "stats.md", "w") as f:
        f.write("| scene | run | saturation | hf energy | flat blocks | edge contrast |\n|---|---|---|---|---|---|\n")
        for r in table:
            f.write(f"| {r['scene']} | {r['run']} | {r['sat']:.3f} | {r['hf']:.2f} | {r['flat']:.3f} | {r['edge']:.1f} |\n")
    print(out / "stats.md")


def small_jpeg(im, dst, limit):
    from PIL import Image

    for q in range(88, 20, -2):
        b = io.BytesIO()
        im.save(b, "JPEG", quality=q, optimize=True)
        if b.tell() <= limit:
            dst.write_bytes(b.getvalue())
            return q
    raise RuntimeError(f"{dst}: cannot reach {limit} bytes")


def refs(a):
    from PIL import Image

    REF.mkdir(parents=True, exist_ok=True)
    for s in scenes(a.scenes):
        if "@" in s:
            continue
        h5 = Path(a.showcase) / s / "stills.h5"
        if not h5.exists():
            print(f"{s}: no {h5}")
            continue
        q = small_jpeg(Image.fromarray(load_frame(h5)), REF / f"{s}.jpg", a.limit)
        print(f"{s}: q{q}")


def thumbs(a):
    """Small JPEG copies of the sheets (for the repo docs)."""
    from PIL import Image

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in a.sheets:
        im = Image.open(p).convert("RGB")
        im = im.resize((a.width, round(im.height * a.width / im.width)), Image.LANCZOS)
        q = small_jpeg(im, out / Path(p).name, a.limit)
        print(out / Path(p).name, q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--terrain", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--force", action="store_true")
    r.add_argument("scenes", nargs="*")
    s = sub.add_parser("sheet")
    s.add_argument("--run", action="append", default=[], help="TAG=DIR of a render")
    s.add_argument("--out", required=True)
    s.add_argument("--per-sheet", type=int, default=6)
    s.add_argument("--scale", type=float, default=0.5)
    s.add_argument("--audit", action="store_true", help="the audit places (scenes = place ids)")
    s.add_argument("scenes", nargs="*")
    u = sub.add_parser("audit", help="render the audit places")
    u.add_argument("--terrain", required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--generator", choices=["next", "classic"], default="next")
    u.add_argument("--force", action="store_true")
    u.add_argument("scenes", nargs="*", help="place ids (default: all)")
    f = sub.add_parser("refs")
    f.add_argument("--showcase", required=True)
    f.add_argument("--limit", type=int, default=40000)
    f.add_argument("scenes", nargs="*")
    t = sub.add_parser("thumbs")
    t.add_argument("--out", required=True)
    t.add_argument("--width", type=int, default=640)
    t.add_argument("--limit", type=int, default=60000)
    t.add_argument("sheets", nargs="+")
    a = ap.parse_args()
    {"render": render, "audit": audit, "sheet": sheet, "refs": refs, "thumbs": thumbs}[a.cmd](a)


if __name__ == "__main__":
    os.umask(0o022)
    main()
