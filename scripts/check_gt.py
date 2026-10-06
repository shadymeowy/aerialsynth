#!/usr/bin/env python3
"""Validate rendered GT: warp frame k+1 into frame k with the GT flow and measure photometric error,
and check depth/flow consistency with the poses (reprojection of depth with pose deltas).

usage: check_gt.py SEQ.h5 [--frames 0,10,20]
"""
import argparse, numpy as np, h5py

def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("--frames", default=None)
a = ap.parse_args()
f = h5py.File(a.seq, "r")
rgb = f["rgb"]; flow = f["flow"]; valid = f["flow_valid"]; depth = f["depth"]
K = np.array(f["camera"].attrs["K"]).reshape(3, 3)
dist = np.array(f["camera"].attrs["dist_radtan"])
pos = f["pose/cam_position_ecef"][:]; q = f["pose/cam_q_ecef"][:]
n = rgb.shape[0]
frames = [int(x) for x in a.frames.split(",")] if a.frames else list(range(0, n - 1, max(1, (n - 1) // 6)))
H, W = rgb.shape[1:3]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

def undistort(u, v):
    xd = (u - K[0, 2]) / K[0, 0]; yd = (v - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    k1, k2, p1, p2, k3 = dist
    for _ in range(20):
        r2 = x * x + y * y; rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x); dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / rad; y = (yd - dy) / rad
    return x, y

def project(Pc):
    x = Pc[..., 0] / Pc[..., 2]; y = Pc[..., 1] / Pc[..., 2]
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y; rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]

xn, yn = undistort(xx, yy)
for k in frames:
    I0 = rgb[k].astype(np.float64); I1 = rgb[k + 1].astype(np.float64)
    fl = flow[k]; v = valid[k] > 0
    warped = bilinear(I1, xx + fl[..., 0], yy + fl[..., 1])
    err = np.abs(warped - I0).mean(axis=2)
    base = np.abs(I1 - I0).mean(axis=2)
    # flow from depth + poses
    d = depth[k]
    Pc = np.stack([xn * d, yn * d, d], axis=-1)
    R0 = quat_to_R(q[k]); R1 = quat_to_R(q[k + 1])
    Pw = Pc @ R0.T + pos[k]
    Pc1 = (Pw - pos[k + 1]) @ R1
    u1, v1 = project(Pc1)
    fe = np.hypot(u1 - xx - fl[..., 0], v1 - yy - fl[..., 1])
    print(f"frame {k:4d}: valid {v.mean()*100:5.1f}%  photometric |I0 - warp(I1)| = {err[v].mean():5.2f} (no warp {base[v].mean():5.2f}) DN;"
          f" flow vs depth+pose reprojection: median {np.median(fe[v]):.2e} px, max {fe[v].max():.2e} px; mean |flow| {np.hypot(fl[...,0], fl[...,1])[v].mean():.2f} px")
