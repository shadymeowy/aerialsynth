#!/usr/bin/env python3
"""Render event windows of a camera's event stream (`<camera>/events`, sliced with ms_index).

Windows side by side; positive events red, negative blue on white, like common event
visualizers. Prints the event count and rate per window.

    python scripts/view_events.py SEQ.h5 OUT.png [--camera /events] [--at 100,500,1500] [--window-ms 5]
"""
import argparse, os, sys
import numpy as np, h5py
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqio import camera_with

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("seq", help="sequence file (HDF5)")
ap.add_argument("out", help="output PNG")
ap.add_argument("--camera", default=None, help="camera group (default: the first with events/)")
ap.add_argument("--at", default=None, help="comma-separated window starts in ms (default: at 1/6, 1/2, 5/6 of the stream)")
ap.add_argument("--window-ms", type=int, default=5, help="window length in ms")
a = ap.parse_args()
f = h5py.File(a.seq, "r")
path = a.camera or camera_with(f, "events")
cam = f[path]; g = cam["events"]
W, H = cam["calib/resolution"][:]
mm = g["ms_index"][:]
n_ms = len(mm) - 1
at = [int(x) for x in a.at.split(",")] if a.at else [n_ms // 6, n_ms // 2, (5 * n_ms) // 6]
tiles = []
for ms in at:
    i0, i1 = mm[ms], mm[min(ms + a.window_ms, n_ms)]
    x, y, p = g["x"][i0:i1], g["y"][i0:i1], g["p"][i0:i1]
    img = np.full((H, W, 3), 255, np.uint8)
    pos = p > 0
    img[y[pos], x[pos]] = (220, 40, 40)
    img[y[~pos], x[~pos]] = (40, 70, 220)
    tiles.append(img)
    print(f"t={ms} ms: {i1 - i0} events in {a.window_ms} ms ({(i1 - i0) / a.window_ms / 1e3:.2f} Mev/s), ON {pos.mean() * 100:.0f}%")
Image.fromarray(np.concatenate(tiles, axis=1)).save(a.out)
print(f"{path}/events: {g['t'].shape[0]} events over {n_ms} ms; mean rate {g['t'].shape[0] / n_ms / 1e3:.2f} Mev/s")
