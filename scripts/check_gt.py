#!/usr/bin/env python3
"""Validate the ground truth of every camera in a sequence file:
* flow vs reprojection of depth with the camera poses (should be ~1e-5 px),
* photometric error of frame k+1 warped into frame k with the flow (cameras with rgb),
* camera poses vs body poses (/pose) composed with calib/T_body_cam.

usage: check_gt.py SEQ.h5 [--camera /cam0] [--frames 0,10,20]
"""
import argparse, os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cammodels import Camera
from seqio import cameras, quat_to_R, scenario


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def check_poses(f, g, pose_path):
    """Camera pose at its frame times == body pose ∘ T_body_cam (frames on the /pose grid)."""
    body = f[pose_path]
    tb = body["t"][:]
    T = g["calib/T_body_cam"][:]
    errs_p, errs_r = [], []
    for k, t in enumerate(g["t"][:]):
        i = np.searchsorted(tb, t)
        if i >= len(tb) or tb[i] != t:
            continue
        Rb = quat_to_R(body["q_ecef_body"][i]); pb = body["position_ecef"][i]
        Rc = Rb @ T[:3, :3]; pc = pb + Rb @ T[:3, 3]
        errs_p.append(np.linalg.norm(pc - g["pose/position_ecef"][k]))
        errs_r.append(np.abs(Rc - quat_to_R(g["pose/q_ecef_cam"][k])).max())
    if errs_p:
        print(f"  poses: {len(errs_p)} frames on the {pose_path} grid; |cam - body∘T_body_cam| max {max(errs_p):.1e} m, {max(errs_r):.1e} (rotation)")


def check_camera(f, path, frames_arg, pose_path):
    g = f[path]
    cam = Camera.from_calib(g["calib"])
    print(f"{path}: model {cam.model}, {len(g['t'])} frames, modalities: "
          + ", ".join(m for m in ("rgb", "depth", "flow", "landcover", "events") if m in g))
    check_poses(f, g, pose_path)
    if "flow" not in g or "depth" not in g:
        print("  (no depth + flow: flow check skipped)")
        return
    if g["depth"].attrs.get("kind", b"z") not in (b"z", "z"):
        print("  (range depth: flow check skipped)")
        return
    flow, valid, depth = g["flow"], g["flow_valid"], g["depth"]
    pos, q = g["pose/position_ecef"][:], g["pose/q_ecef_cam"][:]
    n, H, W = depth.shape
    frames = [int(x) for x in frames_arg.split(",")] if frames_arg else list(range(0, n - 1, max(1, (n - 1) // 6)))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = cam.unproject(xx, yy)  # unit rays (camera frame)
    rgb = g["rgb"] if "rgb" in g else None
    for k in frames:
        fl = flow[k]; v = valid[k] > 0
        d = depth[k]
        with np.errstate(all="ignore"):
            Pc = rays * (d / rays[..., 2])[..., None]  # z-depth → point on the pixel ray
            Pw = Pc @ quat_to_R(q[k]).T + pos[k]
            Pc1 = (Pw - pos[k + 1]) @ quat_to_R(q[k + 1])
            u1, v1 = cam.project(Pc1)
        fe = np.hypot(u1 - xx - fl[..., 0], v1 - yy - fl[..., 1])
        v = v & np.isfinite(fe)
        msg = f"  frame {k:4d}: valid {v.mean()*100:5.1f}%"
        if rgb is not None:
            I0 = rgb[k].astype(np.float64); I1 = rgb[k + 1].astype(np.float64)
            if I0.ndim == 2:
                I0, I1 = I0[..., None], I1[..., None]
            err = np.abs(bilinear(I1, xx + fl[..., 0], yy + fl[..., 1]) - I0).mean(axis=2)
            base = np.abs(I1 - I0).mean(axis=2)
            msg += f"  photometric |I0 - warp(I1)| {err[v].mean():5.2f} (no warp {base[v].mean():5.2f}) DN;"
        print(msg + f"  flow vs depth+pose: median {np.median(fe[v]):.2e} px, max {fe[v].max():.2e} px;"
              f" mean |flow| {np.hypot(fl[..., 0], fl[..., 1])[v].mean():.2f} px")


ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("--camera", default=None); ap.add_argument("--frames", default=None)
a = ap.parse_args()
f = h5py.File(a.seq, "r")
pose_path = (scenario(f).get("output") or {}).get("pose", {}).get("path", "/pose")
for p in [a.camera] if a.camera else cameras(f):
    check_camera(f, p, a.frames, pose_path)
