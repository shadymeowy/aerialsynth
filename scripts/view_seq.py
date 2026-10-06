#!/usr/bin/env python3
"""Visualize frames of one camera of a sequence file: rgb | depth | flow (HSV) | flow validity
(whichever the camera has).

usage: view_seq.py SEQ.h5 OUT.png [--camera /cam0] [--frames 0,10,20]
"""
import argparse, colorsys, os, sys
import numpy as np, h5py
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqio import cameras

def flow_to_rgb(f, maxmag=None):
    mag = np.hypot(f[..., 0], f[..., 1]); ang = np.arctan2(f[..., 1], f[..., 0])
    m = maxmag or (np.percentile(mag, 99) + 1e-6)
    h = (ang / (2 * np.pi)) % 1.0; s = np.clip(mag / m, 0, 1); v = np.ones_like(s)
    rgb = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)
    return (np.stack(rgb, -1) * 255).astype(np.uint8)

def depth_to_rgb(d):
    d = np.abs(d)  # z-depth is negative beyond 90° on wide-angle models
    fin = np.isfinite(d) & (d > 0)
    out = np.zeros(d.shape + (3,), np.uint8)
    if fin.any():
        lo, hi = np.percentile(np.log(d[fin]), [1, 99])
        t = np.clip((np.log(np.where(fin, d, 1)) - lo) / (hi - lo + 1e-9), 0, 1)
        out[..., 0] = (255 * (1 - t)).astype(np.uint8); out[..., 1] = (255 * (1 - abs(t - 0.5) * 2)).astype(np.uint8)
        out[..., 2] = (255 * t).astype(np.uint8); out[~fin] = (40, 40, 60)
    return out

ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("out"); ap.add_argument("--camera", default=None); ap.add_argument("--frames", default=None)
a = ap.parse_args()
f = h5py.File(a.seq, "r")
path = a.camera or next(p for p in cameras(f) if "t" in f[p])
g = f[path]
n = g["t"].shape[0]
if n == 0:
    raise SystemExit(f"{path}: no frames")
frames = [int(x) for x in a.frames.split(",")] if a.frames else sorted({0, n // 2, max(n - 2, 0)})
rows = []
for k in frames:
    parts = []
    if "rgb" in g:
        im = g["rgb"][k]
        parts.append(np.repeat(im[..., None], 3, -1) if im.ndim == 2 else im)
    if "depth" in g: parts.append(depth_to_rgb(g["depth"][k]))
    if "flow" in g:
        parts.append(flow_to_rgb(g["flow"][k]))
        parts.append(np.repeat((g["flow_valid"][k] * 255)[..., None], 3, -1))
    if "landcover" in g:
        lc = g["landcover"][k].astype(np.uint32)
        parts.append(np.stack([(lc * 67) % 256, (lc * 139) % 256, (lc * 211) % 256], -1).astype(np.uint8))
    rows.append(np.concatenate(parts, axis=1))
Image.fromarray(np.concatenate(rows, axis=0)).save(a.out)
t = g["t"][:] * 1e-6
msg = f"{path}: {n} frames, t {t[0]:.3f}..{t[-1]:.3f} s"
if "exposure" in g:
    e = g["exposure"][:, 0]
    msg += f", exposure {e.min()*1e3:.2f}..{e.max()*1e3:.2f} ms"
print(msg + f"; wrote {a.out}")
