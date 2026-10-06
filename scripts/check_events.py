#!/usr/bin/env python3
"""Validate the event streams of a sequence file: per camera with `events/`, check the format
invariants (coordinates inside the sensor, binary polarity, sorted timestamps inside the
sequence, ms_index consistent with t) and print rate / polarity / hot-pixel statistics.

usage: check_events.py SEQ.h5 [--camera /dvs]
"""
import argparse, os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seqio import cameras, scenario

ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("--camera", default=None)
a = ap.parse_args()
f = h5py.File(a.seq, "r")
pose_path = (scenario(f).get("output") or {}).get("pose", {}).get("path", "/pose")
t_end = int(f[pose_path]["t"][-1])
ok = True
for path in [a.camera] if a.camera else [c for c in cameras(f) if "events" in f[c]]:
    g = f[path]["events"]
    W, H = f[path]["calib/resolution"][:]
    x, y, t, p, mi = g["x"][:], g["y"][:], g["t"][:], g["p"][:], g["ms_index"][:]
    n = len(t)
    checks = {
        "x < width": bool((x < W).all()),
        "y < height": bool((y < H).all()),
        "p in {0,1}": bool(np.isin(p, [0, 1]).all()),
        "t sorted": bool((np.diff(t) >= 0).all()),
        "t inside sequence": bool(n == 0 or (t[0] >= 0 and t[-1] <= t_end)),
        "ms_index[m] = first event at/after m ms": bool(np.array_equal(mi[:-1], np.searchsorted(t, np.arange(len(mi) - 1) * 1000, "left"))),
        "ms_index closes at n": bool(mi[-1] == n),
        "ms_index covers the stream": bool(n == 0 or (len(mi) - 1) * 1000 > t[-1]),
    }
    ok &= all(checks.values())
    dur = max(t_end, 1) * 1e-6
    cnt = np.bincount(y.astype(np.int64) * W + x, minlength=W * H)
    active = cnt[cnt > 0]
    hot = int((cnt > 20 * np.median(active)).sum()) if len(active) else 0
    per_ms = np.diff(mi)
    print(f"{path}/events: {n} events over {dur:.3f} s = {n / dur / 1e6:.2f} Mev/s; ON {p.mean() * 100:.1f}%; "
          f"active pixels {len(active) / (W * H) * 100:.1f}%; per ms p50/p99/max {np.percentile(per_ms, 50):.0f}/{np.percentile(per_ms, 99):.0f}/{per_ms.max()}; "
          f"hot-like pixels {hot}")
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
sys.exit(0 if ok else 1)
