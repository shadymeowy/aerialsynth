#!/usr/bin/env python3
"""Validate the ground truth of every camera in a sequence file; exits non-zero if a check
fails or nothing could be checked.

* camera poses vs body poses (/pose) composed with calib/T_body_cam. Frames on the /pose grid
  are compared exactly (≤ 1e-4 m, ≤ 1e-6 rotation-matrix entries; measured 0 / 1e-15); other
  frames against the /pose interpolation (≤ 1e-2 m, ≤ 1e-4; measured 2e-5 m, 2e-7 at 200 Hz),
* flow vs reprojection of depth (z or range) with the camera poses: median ≤ 1e-3 px and
  max ≤ 1e-2 px over the valid pixels (measured ~5e-6 / ~3e-5 px),
* photometric error of frame k+1 warped into frame k with the flow (cameras with rgb): the warp
  must help, |I0 - warp(I1)| ≤ max(4 DN, the unwarped difference) (measured ~2 DN: sensor
  noise plus motion blur).

usage: check_gt.py SEQ.h5 [--camera /cam0] [--frames 0,10,20]
"""
import argparse, os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cammodels import Camera
from seqio import cameras, quat_to_R, scenario

POSE_EXACT_M, POSE_EXACT_R = 1e-4, 1e-6
POSE_INTERP_M, POSE_INTERP_R = 1e-2, 1e-4
FLOW_MEDIAN_PX, FLOW_MAX_PX = 1e-3, 1e-2
PHOTO_DN = 4.0

failures, checked = [], 0


def verdict(name, ok, detail):
    global checked
    checked += 1
    print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def slerp(q0, q1, s):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


def check_poses(f, path, g, pose_path):
    """Camera pose at its frame times == body pose ∘ T_body_cam."""
    body = f[pose_path]
    tb = body["t"][:]
    pb_all, qb_all = body["position_ecef"][:], body["q_ecef_body"][:]
    T = g["calib/T_body_cam"][:]
    exact, interp = [], []
    for k, t in enumerate(g["t"][:]):
        i = np.searchsorted(tb, t)
        if i < len(tb) and tb[i] == t:
            pb, Rb, out = pb_all[i], quat_to_R(qb_all[i]), exact
        elif 0 < i < len(tb):
            s = (t - tb[i - 1]) / (tb[i] - tb[i - 1])
            pb = pb_all[i - 1] + s * (pb_all[i] - pb_all[i - 1])
            Rb, out = quat_to_R(slerp(qb_all[i - 1], qb_all[i], s)), interp
        else:
            continue
        Rc = Rb @ T[:3, :3]; pc = pb + Rb @ T[:3, 3]
        out.append((np.linalg.norm(pc - g["pose/position_ecef"][k]), np.abs(Rc - quat_to_R(g["pose/q_ecef_cam"][k])).max()))
    for name, rows, tol_m, tol_r in [("poses on the /pose grid", exact, POSE_EXACT_M, POSE_EXACT_R),
                                     ("poses between /pose samples (interpolated)", interp, POSE_INTERP_M, POSE_INTERP_R)]:
        if rows:
            em, er = max(r[0] for r in rows), max(r[1] for r in rows)
            verdict(f"{path} {name}", em <= tol_m and er <= tol_r,
                    f"{len(rows)} frames; |cam - body∘T_body_cam| max {em:.1e} m (≤ {tol_m:g}), {er:.1e} rotation (≤ {tol_r:g})")
    if not exact and not interp:
        verdict(f"{path} poses", False, f"no frame time inside {pose_path}")


def check_camera(f, path, frames_arg, pose_path):
    g = f[path]
    cam = Camera.from_calib(g["calib"])
    print(f"{path}: model {cam.model}, {len(g['t'])} frames, modalities: "
          + ", ".join(m for m in ("rgb", "depth", "flow", "landcover", "events") if m in g))
    check_poses(f, path, g, pose_path)
    if "flow" not in g or "depth" not in g:
        print("  (no depth + flow: flow check not applicable)")
        return
    kind = g["depth"].attrs.get("kind", b"z")
    kind = kind.decode() if isinstance(kind, bytes) else kind
    flow, valid, depth = g["flow"], g["flow_valid"], g["depth"]
    pos, q = g["pose/position_ecef"][:], g["pose/q_ecef_cam"][:]
    n, H, W = depth.shape
    if n < 2:
        print("  (fewer than 2 frames: flow check not applicable)")
        return
    frames = [int(x) for x in frames_arg.split(",")] if frames_arg else list(range(0, n - 1, max(1, (n - 1) // 6)))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = cam.unproject(xx, yy)  # unit rays (camera frame)
    rgb = g["rgb"] if "rgb" in g else None
    for k in frames:
        fl = flow[k]; v = valid[k] > 0
        d = depth[k]
        with np.errstate(all="ignore"):
            # depth → point on the pixel ray (z: along the optical axis; range: along the ray)
            Pc = rays * ((d / rays[..., 2]) if kind == "z" else d)[..., None]
            Pw = Pc @ quat_to_R(q[k]).T + pos[k]
            Pc1 = (Pw - pos[k + 1]) @ quat_to_R(q[k + 1])
            u1, v1 = cam.project(Pc1)
        fe = np.hypot(u1 - xx - fl[..., 0], v1 - yy - fl[..., 1])
        v = v & np.isfinite(fe)
        if not v.any():
            verdict(f"{path} frame {k} flow vs depth+pose", False, "no valid flow pixel")
            continue
        med, mx = float(np.median(fe[v])), float(fe[v].max())
        verdict(f"{path} frame {k} flow vs depth+pose", med <= FLOW_MEDIAN_PX and mx <= FLOW_MAX_PX,
                f"valid {v.mean()*100:5.1f}%, median {med:.2e} px (≤ {FLOW_MEDIAN_PX:g}), max {mx:.2e} px (≤ {FLOW_MAX_PX:g}),"
                f" mean |flow| {np.hypot(fl[..., 0], fl[..., 1])[v].mean():.2f} px")
        if rgb is not None:
            I0 = rgb[k].astype(np.float64); I1 = rgb[k + 1].astype(np.float64)
            if I0.ndim == 2:
                I0, I1 = I0[..., None], I1[..., None]
            err = float(np.abs(bilinear(I1, xx + fl[..., 0], yy + fl[..., 1]) - I0).mean(axis=2)[v].mean())
            base = float(np.abs(I1 - I0).mean(axis=2)[v].mean())
            verdict(f"{path} frame {k} photometric", err <= max(PHOTO_DN, base),
                    f"|I0 - warp(I1)| {err:5.2f} DN (≤ max({PHOTO_DN:g}, unwarped {base:5.2f}))")


ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("--camera", default=None); ap.add_argument("--frames", default=None)
a = ap.parse_args()
f = h5py.File(a.seq, "r")
pose_path = (scenario(f).get("output") or {}).get("pose", {}).get("path", "/pose")
cams = [a.camera] if a.camera else cameras(f)
if not cams:
    raise SystemExit(f"FAIL: no camera in {a.seq}")
for p in cams:
    if p not in f:
        raise SystemExit(f"FAIL: no camera group {p} in {a.seq}")
    check_camera(f, p, a.frames, pose_path)
if failures:
    print(f"FAIL: {len(failures)} of {checked} checks failed")
    sys.exit(1)
if checked == 0:
    print("FAIL: nothing was checked")
    sys.exit(1)
print(f"PASS: {checked} checks")
