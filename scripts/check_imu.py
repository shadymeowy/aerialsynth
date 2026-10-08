#!/usr/bin/env python3
"""Validate the synthetic IMU of a sequence file against the body ground truth (/pose),
independently of the simulator:

* truth: angular rate from attitude differences, specific force from the twice-differentiated
  position of the IMU point (lever arm included) + Coriolis − WGS84 normal gravity, both
  averaged over each IMU sample window [t - dt/2, t + dt/2] and rotated into the IMU frame;
  compared with gt_gyro / gt_accel,
* sensor model: white noise and bias random walk of the measurements against the configured
  densities.

The truth rebuild needs /pose finer than the IMU window (e.g. output.pose.rate_hz: 1000 with a
1 kHz trajectory); at the IMU rate it is only approximate under engine vibration.

Tolerances (exit 1 if one fails; measured with configs/examples/imu_check.yaml, 1 kHz /pose):
* gyro truth: RMS ≤ 1e-6 rad/s (measured 3e-10); with /pose coarser than 4 samples per IMU
  window, ≤ 1e-6 + 1% of the dynamic rate (measured 3e-5 at 200 Hz),
* accel truth: RMS ≤ 2e-3 m/s² + 10% of the lever-arm term (the simulation step discretizes
  it: ~5% at 1 ms; measured 1e-4 without, 0.27 m/s² with the example's lever arm under engine
  vibration); with a coarse /pose, + 0.02 m/s² + 5% of the dynamic part (measured 5e-3 at
  200 Hz: the unresolved vibration),
* white noise and bias walk per sample: 0.8 to 1.25 times the configured σ.

usage: check_imu.py SEQ.h5
"""
import os, sys
import numpy as np, h5py, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqio import quat_to_R, scenario

EARTH_RATE = 7.292115e-5


def normal_gravity(lat, h):
    s2 = np.sin(lat) ** 2
    g0 = 9.7803253359 * (1 + 0.00193185265241 * s2) / np.sqrt(1 - 0.00669437999013 * s2)
    a, f, m = 6378137.0, 1 / 298.257223563, 0.00344978600308
    return g0 * (1 - 2 / a * (1 + f + m - 2 * f * s2) * h + 3 * h * h / a ** 2)


def up_vector(lat, lon):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def rot_log(R):
    """Rotation vectors of rotation matrices (..., 3, 3)."""
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    k = np.where(th < 1e-12, 0.5, th / (2 * np.sin(np.maximum(th, 1e-12))))
    return v * k[..., None]


f = h5py.File(sys.argv[1], "r")
scn = scenario(f)
imu_cfg = scn.get("imu")
if not imu_cfg:
    raise SystemExit("no imu in this sequence")
pose_path = (scn.get("output") or {}).get("pose", {}).get("path", "/pose")
P, I = f[pose_path], f[imu_cfg["path"]]
tp = P["t"][:] * 1e-6
pos = P["position_ecef"][:]
R_eb = quat_to_R(P["q_ecef_body"][:])
lla = np.radians(P["lla"][:, :2])
ti = I["t"][:] * 1e-6
dt = float(np.median(np.diff(ti)))
T = I["calib/T_body_imu"][:]
R_bi, lever = T[:3, :3], T[:3, 3]
print(f"{imu_cfg['path']}: {len(ti)} samples at {1/dt:.1f} Hz (configured {imu_cfg['rate_hz']}); "
      f"/pose at {1/np.median(np.diff(tp)):.0f} Hz; lever arm {lever} m")
failures, checked = [], 0


def verdict(name, ok, detail):
    global checked
    checked += 1
    print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


verdict("timestamps", bool(np.all(np.diff(ti) > 0) and np.allclose(np.diff(ti), dt, atol=2e-6)), "increasing and uniform")

# ---- truth per /pose interval (τ[j-1], τ[j]] (the IMU semantics: means over intervals)
h = np.median(np.diff(tp))
Om = np.array([0, 0, EARTH_RATE])
lat, lon, hgt = lla[1:-1, 0], lla[1:-1, 1], P["lla"][1:-1, 2]


def specific_force(p):
    """Specific force of the ECEF track p per /pose interval, in the body frame (instantaneous
    at the interior samples, then averaged over the intervals)."""
    v = (p[2:] - p[:-2]) / (2 * h)
    a = (p[2:] - 2 * p[1:-1] + p[:-2]) / h ** 2
    f_e = a + 2 * np.cross(Om, v) + normal_gravity(lat, hgt)[:, None] * up_vector(lat, lon)
    f_s = np.einsum("nji,nj->ni", R_eb[1:-1], f_e)          # samples 1..n-2
    return 0.5 * (f_s[1:] + f_s[:-1])                        # intervals (2..n-2]


f_int = specific_force(pos + np.einsum("nij,j->ni", R_eb, lever))   # the IMU point (lever arm)
f_int0 = specific_force(pos)                                          # the body origin
# body rate: attitude difference over each interval + Earth rate
w_all = rot_log(np.einsum("nji,njk->nik", R_eb[:-1], R_eb[1:])) / h + np.einsum("nji,j->ni", R_eb[1:], Om)
w_int = w_all[1:-1]                                          # same intervals as f_int
t_end = tp[2:-1]                                             # interval j ends at t_end[j]

def window_mean(x, t):
    lo, hi = t - dt / 2, t + dt / 2
    if lo < t_end[0] - h or hi > t_end[-1]:
        return None
    wgt = np.clip(np.minimum(t_end, hi) - np.maximum(t_end - h, lo), 0, None)
    return (x * wgt[:, None]).sum(0) / wgt.sum()

rows = []
for k, t in enumerate(ti):
    fw, ww, f0 = window_mean(f_int, t), window_mean(w_int, t), window_mean(f_int0, t)
    if fw is not None:
        rows.append((k, R_bi.T @ fw, R_bi.T @ ww, R_bi.T @ (fw - f0)))
if not rows:
    raise SystemExit("/pose too coarse for windowed truth (need several samples per IMU window)")
k = np.array([r[0] for r in rows])
f_ref = np.array([r[1] for r in rows])
w_ref = np.array([r[2] for r in rows])
lever_term = np.array([r[3] for r in rows])
ga, gg = I["gt_accel"][:][k], I["gt_gyro"][:][k]
rms = lambda x: float(np.sqrt(np.mean(np.sum(x ** 2, -1))))
dyn = lambda x: x - x.mean(0)
pose_rate = 1 / np.median(np.diff(tp))
coarse = pose_rate < 4 / dt
print(f"truth over {len(k)} samples:" + (f"  [/pose at {pose_rate:.0f} Hz is too coarse for an exact rebuild (want ≥ {4 / dt:.0f} Hz): indicative only under vibration]" if coarse else ""))
tol_a = 2e-3 + 0.10 * rms(lever_term) + (0.02 + 0.05 * rms(dyn(f_ref)) if coarse else 0.0)
tol_w = 1e-6 + (0.01 * rms(dyn(w_ref)) if coarse else 0.0)
verdict("gt_accel vs rebuilt", rms(ga - f_ref) <= tol_a,
        f"RMS {rms(ga - f_ref):.4f} m/s² (≤ {tol_a:.4f}; signal RMS {rms(f_ref):.3f}, dynamic part {rms(dyn(f_ref)):.3f}, "
        f"lever-arm term {rms(lever_term):.3f})")
verdict("gt_gyro vs rebuilt", rms(gg - w_ref) <= tol_w,
        f"RMS {rms(gg - w_ref):.2e} rad/s (≤ {tol_w:.1e}; signal RMS {rms(w_ref):.2e}, dynamic part {rms(dyn(w_ref)):.2e})")
g_norm = np.linalg.norm(f_ref, axis=1).mean()
print(f"  mean |f| {g_norm:.4f} m/s² (normal gravity {normal_gravity(lla[:, 0].mean(), P['lla'][:, 2].mean()):.4f})")

# ---- sensor model: noise and bias walk
cfg = yaml.safe_load(I["calib"].attrs["imu_yaml"])
for name, meas, gt, bias, c in [("accel", "accel", "gt_accel", "gt_bias_accel", cfg["accel"]), ("gyro", "gyro", "gt_gyro", "gt_bias_gyro", cfg["gyro"])]:
    m, x, b = I[meas][:], I[gt][:], I[bias][:]
    res = m - x - b                        # white noise + scale / misalignment (small)
    wn = np.diff(res, axis=0).std(0) / np.sqrt(2)   # differencing removes slow scale/misalignment terms
    want = c["noise_density"] / np.sqrt(dt)
    rw = np.diff(b, axis=0).std(0)
    want_rw = c["random_walk"] * np.sqrt(dt)
    verdict(f"{name} white noise", 0.8 <= wn.mean() / want <= 1.25, f"σ {wn.mean():.3e} (configured {want:.3e})")
    if want_rw > 0:
        verdict(f"{name} bias walk", 0.8 <= rw.mean() / want_rw <= 1.25, f"per sample σ {rw.mean():.3e} (configured {want_rw:.3e})")
if failures:
    print(f"FAIL: {len(failures)} of {checked} checks failed")
    sys.exit(1)
print(f"PASS: {checked} checks")
