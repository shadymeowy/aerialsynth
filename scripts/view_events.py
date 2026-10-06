#!/usr/bin/env python3
"""Render event windows of a camera's event stream (`<camera>/events`, sliced with ms_index).

usage: view_events.py SEQ.h5 OUT.png [--camera /dvs] [--at 100,500,1500] [--window-ms 5]
Positive events red, negative blue (white background), like common event visualizers.
"""
import argparse, os, sys
import numpy as np, h5py
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqio import camera_with

ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("out")
ap.add_argument("--camera", default=None); ap.add_argument("--at", default=None)
ap.add_argument("--window-ms", type=int, default=5)
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
