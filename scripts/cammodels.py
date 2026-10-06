"""Camera models of camodocal's calib/camera.cpp (camodocal formulas), vectorized with numpy.

Independent re-implementation used to validate rendered ground truth. `Camera.from_calib` reads
a camera's `calib` group of a sequence HDF5 (its `camera_yaml` attribute, camodocal schema:
model, intrinsics, distortion, xi, max_fov_deg, inv_poly, affine, center).
project(P[...,3]) -> (u, v); unproject(u, v) -> unit rays.
"""
import numpy as np


def _radtan4(x, y, k1, k2, p1, p2):
    x2, y2, xy = x * x, y * y, x * y
    r2 = x2 + y2
    rad = k1 * r2 + k2 * r2 * r2
    return x * rad + 2 * p1 * xy + p2 * (r2 + 2 * x2), y * rad + 2 * p2 * xy + p1 * (r2 + 2 * y2)


def _invert2(f, tx, ty, iters=40):
    """Newton inversion of an elementwise 2D map f(x, y) -> (u, v) with numeric Jacobian."""
    x, y = tx.copy(), ty.copy()
    e = 1e-7
    for _ in range(iters):
        u, v = f(x, y)
        rx, ry = u - tx, v - ty
        ux1, vx1 = f(x + e, y); ux0, vx0 = f(x - e, y)
        uy1, vy1 = f(x, y + e); uy0, vy0 = f(x, y - e)
        a, b = (ux1 - ux0) / (2 * e), (uy1 - uy0) / (2 * e)
        c, d = (vx1 - vx0) / (2 * e), (vy1 - vy0) / (2 * e)
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-18, 1e-18, det)
        x = x - (d * rx - b * ry) / det
        y = y - (-c * rx + a * ry) / det
    return x, y


class Camera:
    def __init__(self, model, intrinsics=(), distortion=(), xi=0.0, max_fov_deg=0.0, inv_poly=(), affine=(), center=()):
        self.model = model
        self.i = np.asarray(intrinsics, float)
        self.d = np.asarray(distortion, float)
        self.xi = float(xi)
        self.max_theta = max_fov_deg * np.pi / 360 if max_fov_deg > 0 else np.pi
        self.inv_poly = np.asarray(inv_poly, float)
        self.affine = np.asarray(affine, float)
        self.center = np.asarray(center, float)

    @classmethod
    def from_dict(cls, c):
        d = c.get("distortion") or [0.0] * (8 if c["model"] == "pinhole_full" else 4)  # omitted = none
        return cls(c["model"], c.get("intrinsics", ()), d, c.get("xi", 0.0), c.get("max_fov_deg", 0.0),
                   c.get("inv_poly", ()), c.get("affine", ()), c.get("center", ()))

    @classmethod
    def from_calib(cls, calib):
        """`calib` = an h5py group `<camera>/calib`."""
        import yaml
        y = calib.attrs["camera_yaml"]
        return cls.from_dict(yaml.safe_load(y.decode() if isinstance(y, bytes) else y))

    # ---------------------------------------------------------------- forward
    def project(self, P):
        X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
        m = self.model
        if m in ("pinhole", "pinhole_full"):
            fx, fy, cx, cy = self.i
            x, y = X / Z, Y / Z
            if m == "pinhole":
                dx, dy = _radtan4(x, y, *self.d)
                x, y = x + dx, y + dy
            else:
                x, y = self._full(x, y)
            return fx * x + cx, fy * y + cy
        if m == "kannala_brandt":
            mu, mv, u0, v0 = self.i
            n = np.sqrt(X * X + Y * Y + Z * Z)
            th = np.arccos(np.clip(Z / n, -1, 1))
            r = self._kb(th)
            phi = np.arctan2(Y, X)
            return mu * r * np.cos(phi) + u0, mv * r * np.sin(phi) + v0
        if m == "mei":
            g1, g2, u0, v0 = self.i
            z = Z + self.xi * np.sqrt(X * X + Y * Y + Z * Z)
            x, y = X / z, Y / z
            dx, dy = _radtan4(x, y, *self.d)
            return g1 * (x + dx) + u0, g2 * (y + dy) + v0
        if m == "scaramuzza":
            n = np.hypot(X, Y)
            th = np.arctan2(-Z, n)
            rho = np.polyval(self.inv_poly[::-1], th)
            xn, yn = X / n * rho, Y / n * rho
            C, D, E = self.affine
            return xn * C + yn * D + self.center[0], xn * E + yn + self.center[1]
        raise ValueError(m)

    def _full(self, x, y):
        k1, k2, p1, p2, k3, k4, k5, k6 = self.d
        r2 = x * x + y * y; r4 = r2 * r2; r6 = r4 * r2
        a1, a2, a3 = 2 * x * y, r2 + 2 * x * x, r2 + 2 * y * y
        c = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        return x * c + p1 * a1 + p2 * a2, y * c + p1 * a3 + p2 * a1

    def _kb(self, th):
        k2, k3, k4, k5 = self.d
        t2 = th * th
        return th * (1 + t2 * (k2 + t2 * (k3 + t2 * (k4 + t2 * k5))))

    # ---------------------------------------------------------------- inverse
    def unproject(self, u, v):
        m = self.model
        if m in ("pinhole", "pinhole_full"):
            fx, fy, cx, cy = self.i
            xd, yd = (u - cx) / fx, (v - cy) / fy
            if m == "pinhole":
                f = lambda x, y: tuple(a + b for a, b in zip((x, y), _radtan4(x, y, *self.d)))
            else:
                f = self._full
            x, y = _invert2(f, xd, yd)
            r = np.stack([x, y, np.ones_like(x)], -1)
        elif m == "kannala_brandt":
            mu, mv, u0, v0 = self.i
            mx, my = (u - u0) / mu, (v - v0) / mv
            rd = np.hypot(mx, my)
            th = rd.copy()
            for _ in range(50):
                e = 1e-7
                df = (self._kb(th + e) - self._kb(th - e)) / (2 * e)
                th = th - (self._kb(th) - rd) / df
            s = np.where(rd > 1e-12, np.sin(th) / np.maximum(rd, 1e-12), 0.0)
            r = np.stack([mx * s, my * s, np.cos(th)], -1)
        elif m == "mei":
            g1, g2, u0, v0 = self.i
            md, nd = (u - u0) / g1, (v - v0) / g2
            f = lambda x, y: tuple(a + b for a, b in zip((x, y), _radtan4(x, y, *self.d)))
            x, y = _invert2(f, md, nd)
            r2 = x * x + y * y
            lam = (self.xi + np.sqrt(1 + (1 - self.xi ** 2) * r2)) / (1 + r2)
            r = np.stack([lam * x, lam * y, lam - self.xi], -1)
        elif m == "scaramuzza":
            C, D, E = self.affine
            du, dv = u - self.center[0], v - self.center[1]
            det = C - D * E
            xn, yn = (du - D * dv) / det, (-E * du + C * dv) / det
            rho = np.hypot(xn, yn)
            lo = np.full_like(rho, -np.pi / 2 + 1e-9); hi = np.full_like(rho, np.pi / 2 - 1e-9)
            p = self.inv_poly[::-1]
            sgn = np.sign(np.polyval(p, hi) - rho)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                up = np.sign(np.polyval(p, mid) - rho) == sgn
                hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
            th = 0.5 * (lo + hi)
            rr = np.maximum(rho, 1e-12)
            r = np.stack([np.cos(th) * xn / rr, np.cos(th) * yn / rr, -np.sin(th)], -1)
        else:
            raise ValueError(m)
        return r / np.linalg.norm(r, axis=-1, keepdims=True)
