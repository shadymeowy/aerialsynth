#!/usr/bin/env python3
"""Render event windows of an M3ED-layout event file (as camodocal reads it: ms_map_idx slices).

usage: view_events.py EVENTS.h5 OUT.png [--group /prophesee/left] [--at 100,500,1500] [--window-ms 5]
Positive events red, negative blue (white background), like common event visualizers.
"""
import argparse, numpy as np, h5py
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("events"); ap.add_argument("out")
ap.add_argument("--group", default="/prophesee/left"); ap.add_argument("--at", default=None)
ap.add_argument("--window-ms", type=int, default=5)
a = ap.parse_args()
f = h5py.File(a.events, "r"); g = f[a.group]
W, H = g["calib/resolution"][:]
mm = g["ms_map_idx"][:]
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
t = g["t"]
print(f"total {t.shape[0]} events over {n_ms} ms; mean rate {t.shape[0] / n_ms / 1e3:.2f} Mev/s")
