"""Small helpers for terrain sequence files (format "terrain-sequence", see crates/render/src/output.rs)."""
import numpy as np


def quat_to_R(q):
    """[w, x, y, z] (..., 4) -> rotation matrices (..., 3, 3)."""
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def scenario(f):
    import yaml
    s = f.attrs["scenario"]
    return yaml.safe_load(s.decode() if isinstance(s, bytes) else s)


def cameras(f):
    """Camera group paths of the sequence, in scenario order."""
    return [c["path"] for c in scenario(f).get("cameras") or []]


def camera_with(f, *datasets):
    """First camera group that has all of `datasets`."""
    for p in cameras(f):
        if p in f and all(d in f[p] for d in datasets):
            return p
    raise SystemExit(f"no camera with {', '.join(datasets)} in {f.filename} (cameras: {cameras(f)})")
