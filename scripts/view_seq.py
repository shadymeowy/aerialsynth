#!/usr/bin/env python3
"""Visualize frames of a rendered sequence (HDF5): rgb | depth | flow (HSV) | flow validity.

usage: view_seq.py SEQ.h5 OUT.png [--frames 0,10,20]
"""
import argparse, colorsys, numpy as np, h5py
from PIL import Image

def flow_to_rgb(f, maxmag=None):
    mag = np.hypot(f[..., 0], f[..., 1]); ang = np.arctan2(f[..., 1], f[..., 0])
    m = maxmag or (np.percentile(mag, 99) + 1e-6)
    h = (ang / (2 * np.pi)) % 1.0; s = np.clip(mag / m, 0, 1); v = np.ones_like(s)
    rgb = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)
    return (np.stack(rgb, -1) * 255).astype(np.uint8)

def depth_to_rgb(d):
    fin = np.isfinite(d)
    out = np.zeros(d.shape + (3,), np.uint8)
    if fin.any():
        lo, hi = np.percentile(np.log(d[fin]), [1, 99])
        t = np.clip((np.log(np.where(fin, d, 1)) - lo) / (hi - lo + 1e-9), 0, 1)
        out[..., 0] = (255 * (1 - t)).astype(np.uint8); out[..., 1] = (255 * (1 - abs(t - 0.5) * 2)).astype(np.uint8)
        out[..., 2] = (255 * t).astype(np.uint8); out[~fin] = (40, 40, 60)
    return out

ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("out"); ap.add_argument("--frames", default=None)
a = ap.parse_args()
f = h5py.File(a.seq, "r")
n = f["rgb"].shape[0]
frames = [int(x) for x in a.frames.split(",")] if a.frames else [0, n // 2, n - 2]
rows = []
for k in frames:
    rgb = f["rgb"][k]
    parts = [rgb]
    if "depth" in f: parts.append(depth_to_rgb(f["depth"][k]))
    if "flow" in f:
        parts.append(flow_to_rgb(f["flow"][k]))
        parts.append(np.repeat((f["flow_valid"][k] * 255)[..., None], 3, -1))
    rows.append(np.concatenate(parts, axis=1))
img = np.concatenate(rows, axis=0)
Image.fromarray(img).save(a.out)
t = f["t"][:]; e = f["exposure/time"][:]
print(f"{n} frames, t {t[0]:.2f}..{t[-1]:.2f} s, exposure {e.min()*1e3:.2f}..{e.max()*1e3:.2f} ms; wrote {a.out}")
