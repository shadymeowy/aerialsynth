#!/usr/bin/env python3
"""Contact sheet of tile generator previews (`terrain tiles --png`): rows = locations,
columns = zoom levels. Needs the release build (cargo build --release).

    python scripts/contact.py OUT.png --locs "lat,lon;lat,lon" --zooms 9,12,14,16 [--tiles 2] [--layer rgb] [--config scenario.yaml]
"""
import argparse, os, subprocess, tempfile
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", help="output PNG")
ap.add_argument("--locs", default="39.9,32.8;38.5,30.0;41.0,35.5", help="semicolon-separated lat,lon (deg), one row each")
ap.add_argument("--zooms", default="9,12,14,16", help="comma-separated zoom levels, one column each")
ap.add_argument("--tiles", type=int, default=2, help="mosaic size in tiles per side")
ap.add_argument("--layer", default="rgb", help="preview layer (rgb, albedo, elevation, normal, landcover, hillshade)")
ap.add_argument("--cell", type=int, default=384, help="cell size in the sheet (px)")
ap.add_argument("--config", default=None, help="scenario YAML (world settings)")
ap.add_argument("--bin", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "terrain"),
                help="terrain binary (default: target/release/terrain)")
a = ap.parse_args()
locs = [tuple(map(float, loc.split(","))) for loc in a.locs.split(";")]
zooms = [int(z) for z in a.zooms.split(",")]
sheet = Image.new("RGB", (a.cell * len(zooms), a.cell * len(locs)))
with tempfile.TemporaryDirectory() as tmp:
    for r, (lat, lon) in enumerate(locs):
        for c, z in enumerate(zooms):
            prefix = os.path.join(tmp, f"{r}_{z}")
            cmd = [a.bin, "tiles", "--png", prefix, "--zoom", str(z), "--size", str(a.tiles), "--at", f"{lat},{lon}",
                   "--layers", a.layer]
            if a.config:
                cmd += ["-c", a.config]
            subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
            im = Image.open(f"{prefix}_{a.layer}.png").resize((a.cell, a.cell), Image.LANCZOS)
            sheet.paste(im, (c * a.cell, r * a.cell))
sheet.save(a.out)
print("wrote", a.out)
