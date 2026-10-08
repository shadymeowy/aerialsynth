#!/usr/bin/env python3
"""Contact sheet of generator previews: rows = locations, columns = zoom levels.

usage: contact.py OUT.png --locs "lat,lon;lat,lon" --zooms 9,12,14,16 [--tiles 2] [--layer rgb] [--config scenario.yaml]
"""
import argparse, subprocess, os, tempfile
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--locs", default="39.9,32.8;38.5,30.0;41.0,35.5")
ap.add_argument("--zooms", default="9,12,14,16")
ap.add_argument("--tiles", type=int, default=2)
ap.add_argument("--layer", default="rgb")
ap.add_argument("--cell", type=int, default=384)
ap.add_argument("--config", default=None)
ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..", "target", "release", "terrain"))
a = ap.parse_args()
locs = [tuple(map(float, l.split(","))) for l in a.locs.split(";")]
zooms = [int(z) for z in a.zooms.split(",")]
tmp = tempfile.mkdtemp()
sheet = Image.new("RGB", (a.cell * len(zooms), a.cell * len(locs)))
for r, (lat, lon) in enumerate(locs):
    for c, z in enumerate(zooms):
        prefix = os.path.join(tmp, f"{r}_{z}")
        cmd = [a.bin, "preview", "-z", str(z), "--tiles", str(a.tiles), "--lat", str(lat), "--lon", str(lon),
               "--layers", a.layer, "-o", prefix]
        if a.config:
            cmd += ["--config", a.config]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        im = Image.open(f"{prefix}_{a.layer}.png").resize((a.cell, a.cell), Image.LANCZOS)
        sheet.paste(im, (c * a.cell, r * a.cell))
sheet.save(a.out)
print("wrote", a.out)
