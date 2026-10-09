#!/usr/bin/env python3
"""An airliner's flight as a trajectory CSV: lateral path, speed / altitude profile and attitude.

    python showcase/route.py showcase/airliner.yaml     # → out/airliner/route.csv, route.json

The route (`route:` in the storyboard) is a list of waypoints flown as great circles with
fly-by turns (bank ≤ 25°). The flight starts `start_agl_m` above the departure (just after
lift-off) on the runway heading, climbs and accelerates on an airliner's schedule to cruise,
and descends on a 3° path to a touchdown at the arrival point on the final course, with a
flare. Attitude: yaw = track (no wind), pitch = flight path angle + angle of attack (from the
indicated airspeed, flaps at low speed), roll from the turn rate, plus light turbulence near
the ground. Heights are above the WGS84 ellipsoid; the airports' ground comes from the world
itself (`cargo run --release -p terragen --example ground`).

The CSV (`t,lat,lon,h,roll,pitch,yaw`, 10 Hz) is what `terrain run` reads as `trajectory.file`;
route.json holds the phases, the sparse polyline and the ground clearance for the composer.
make_airliner.py runs this itself when the route section or this file changed.
"""
import argparse, json, math, os, subprocess
import numpy as np, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GROUND = os.path.join(ROOT, "target", "release", "examples", "ground")

A_WGS, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)
G = 9.80665


def ground_heights(pts, seed=1):
    """Terrain surface (m above the ellipsoid; water level over water) at (lat, lon) degrees."""
    if not os.path.exists(GROUND):
        subprocess.run(["cargo", "build", "--release", "-p", "terragen", "--example", "ground"], cwd=ROOT, check=True)
    inp = "".join(f"{a:.7f} {b:.7f} 60\n" for a, b in pts)
    out = subprocess.run([GROUND, str(seed)], input=inp, capture_output=True, text=True, check=True).stdout.split("\n")
    return np.array([float(line.split()[0]) for line in out if line.strip()])


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def latlon(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))), np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def destination(lat, lon, bearing_deg, dist_m):
    """Point `dist_m` from (lat, lon) on the initial bearing (sphere of radius 6371 km)."""
    d = dist_m / 6371000.0
    la, lo, b = math.radians(lat), math.radians(lon), math.radians(bearing_deg)
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(b))
    lo2 = lo + math.atan2(math.sin(b) * math.sin(d) * math.cos(la), math.cos(d) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), (math.degrees(lo2) + 540) % 360 - 180


def ellipsoid_steps(lat, lon):
    """North / east steps (m) between consecutive points."""
    la = np.radians(lat)
    m = A_WGS * (1 - E2) / (1 - E2 * np.sin(la) ** 2) ** 1.5
    n = A_WGS / np.sqrt(1 - E2 * np.sin(la) ** 2)
    dn = np.diff(np.radians(lat)) * 0.5 * (m[1:] + m[:-1])
    dlon = (np.diff(np.radians(lon)) + np.pi) % (2 * np.pi) - np.pi
    de = dlon * 0.5 * (n[1:] * np.cos(la[1:]) + n[:-1] * np.cos(la[:-1]))
    return dn, de


def isa_density_ratio(h):
    """ISA density / sea-level density (troposphere and the isothermal layer above 11 km)."""
    h = np.asarray(h, float)
    t = 288.15 - 0.0065 * np.minimum(h, 11000.0)
    r = (t / 288.15) ** 4.2559
    return np.where(h > 11000.0, r * np.exp(-(h - 11000.0) / 6341.6), r)


def lateral_path(wps, turn_v, step=100.0, bank_deg=25.0):
    """Dense (lat, lon) along great circles through `wps` with fly-by turns: around each inner
    waypoint the legs are joined by a quadratic Bézier (in 3D, back on the sphere) whose half
    length is R tan(Δ/2), R = v² / (g tan(bank)) at that waypoint's speed `turn_v[i]`."""
    u = unit(np.array([w[0] for w in wps]), np.array([w[1] for w in wps]))
    pts = []
    for i in range(len(wps) - 1):
        a, b = u[i], u[i + 1]
        w = math.acos(np.clip(a @ b, -1, 1))
        n = max(2, int(math.ceil(w * 6371000.0 / step)))
        t = np.linspace(0, 1, n + 1)[:-1] if i < len(wps) - 2 else np.linspace(0, 1, n + 1)
        seg = (np.sin((1 - t) * w)[:, None] * a + np.sin(t * w)[:, None] * b) / math.sin(w)
        pts.append(seg)
    p = np.concatenate(pts)
    # arc length (sphere) of the raw polyline, and each inner waypoint's index
    d = np.concatenate([[0.0], np.cumsum(np.arccos(np.clip(np.sum(p[1:] * p[:-1], -1), -1, 1)) * 6371000.0)])
    idx = [int(np.argmin(np.sum((p - u[i]) ** 2, -1))) for i in range(len(wps))]
    out = p.copy()
    for k in range(1, len(wps) - 1):
        i = idx[k]
        din, dout = p[i] - p[i - 1], p[i + 1] - p[i]
        cos_d = np.clip(din @ dout / (np.linalg.norm(din) * np.linalg.norm(dout)), -1, 1)
        delta = math.acos(cos_d)
        r = turn_v[k] ** 2 / (G * math.tan(math.radians(bank_deg)))
        half = r * math.tan(delta / 2)
        j0 = int(np.searchsorted(d, d[i] - half))
        j1 = int(np.searchsorted(d, d[i] + half))
        if j1 - j0 < 3:
            continue
        p0, p1, c = p[j0], p[j1], p[i]
        for j in range(j0, j1 + 1):
            # Bézier parameter by the raw arc length (both halves are equally long)
            s = (d[j] - d[j0]) / (d[j1] - d[j0])
            q = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s * s * p1
            out[j] = q / np.linalg.norm(q)
    return latlon(out)


def climb_profile(h0, cfg):
    """Forward from lift-off: (s, h, v) until cruise height and speed (dt = 0.5 s)."""
    hc, vc = cfg["cruise_m"], cfg["cruise_mps"]
    hs, rates = zip(*cfg["climb_rates"])  # [[height_m, climb m/s], ...]
    s, h, v, t = 0.0, h0, cfg["liftoff_mps"], 0.0
    out = [(s, h, v)]
    dt = 0.5
    while h < hc - 0.01 or v < vc - 0.01:
        vt = float(np.interp(h, [0, 3000, 3001, 10000], [cfg["low_mps"], cfg["low_mps"], cfg["low_mps"], vc]))
        a = np.clip((vt - v) / 15.0, -0.8, 1.0)
        vs = float(np.interp(h, hs, rates))
        vs = min(vs, max(0.4, (hc - h) / 40.0))
        v += a * dt
        h = min(hc, h + vs * dt)
        s += v * math.cos(math.atan2(vs, v)) * dt
        t += dt
        out.append((s, h, v))
    return np.array(out)


def descent_profile(h_td, ground_at, cfg):
    """Backward from the touchdown: (distance before touchdown, h, v) up to cruise height.
    A 3° path from `glide_from_m` above the ground (the approach) and the descent above it;
    the flare rounds the last ~300 m (path angle 0.7° at touchdown)."""
    hc, vc = cfg["cruise_m"], cfg["cruise_mps"]
    gam = math.radians(cfg.get("glide_deg", 3.0))
    x, h = 0.0, h_td
    out = [(x, h, cfg["approach_mps"])]
    dx = 20.0
    while h < hc:
        agl = h - h_td
        flare = min(1.0, x / 300.0)
        g = math.radians(0.7) + (gam - math.radians(0.7)) * (flare * flare * (3 - 2 * flare))
        h = min(hc, h + math.tan(g) * dx)
        x += dx
        v = float(np.interp(agl, [0, 300, 900, 3000, 10000], [cfg["approach_mps"], cfg["approach_mps"] + 3, cfg["approach_mps"] + 18, cfg["low_mps"], vc]))
        out.append((x, h, v))
    return np.array(out)


def smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    # (odd reflection at the ends: keeps the end slopes, e.g. the climb right after lift-off)
    xp = np.concatenate([2 * x[0] - x[r:0:-1], x, 2 * x[-1] - x[-2:-r - 2:-1]])
    return np.convolve(xp, k, "valid")


def build(cfg, seed=1):
    dep, arr = cfg["departure"], cfg["arrival"]
    # waypoints: lift-off, runway heading for `straight_m`, the route, the final approach fix
    w = [(dep["lat"], dep["lon"]), destination(dep["lat"], dep["lon"], dep["heading_deg"], dep.get("straight_m", 6000.0))]
    w += [tuple(p[:2]) for p in cfg["waypoints"]]
    back = (arr["course_deg"] + 180.0) % 360.0
    w += [destination(arr["lat"], arr["lon"], back, arr.get("final_m", 30000.0)), (arr["lat"], arr["lon"])]
    g_dep, g_arr = ground_heights([w[0], w[-1]], seed)
    h0, h_td = g_dep + cfg["start_agl_m"], g_arr + cfg.get("touchdown_agl_m", 3.0)
    # (a waypoint's third value: its turn speed; default the cruise speed)
    turn_v = [cfg["liftoff_mps"] + 10] * 2 + [p[2] if len(p) > 2 else cfg["cruise_mps"] for p in cfg["waypoints"]] + [cfg["low_mps"], cfg["approach_mps"]]
    lat, lon = lateral_path(w, turn_v)
    dn, de = ellipsoid_steps(lat, lon)
    ds = np.hypot(dn, de)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = s[-1]
    # ---- vertical and speed profile along s
    cl = climb_profile(h0, cfg)
    dc = descent_profile(h_td, None, cfg)
    s_toc, s_tod = cl[-1, 0], total - dc[-1, 0]
    if s_tod <= s_toc:
        raise SystemExit("route too short for its climb and descent")
    h = np.where(s <= s_toc, np.interp(s, cl[:, 0], cl[:, 1]), np.where(s >= s_tod, np.interp(total - s, dc[:, 0], dc[:, 1]), cfg["cruise_m"]))
    v = np.where(s <= s_toc, np.interp(s, cl[:, 0], cl[:, 2]), np.where(s >= s_tod, np.interp(total - s, dc[:, 0], dc[:, 2]), cfg["cruise_mps"]))
    v = smooth(v, 20)  # (100 m samples: the deceleration at the top of descent spreads over ~4 km)
    t = np.concatenate([[0.0], np.cumsum(ds / (0.5 * (v[1:] + v[:-1])))])
    # ---- attitude
    track = np.degrees(np.arctan2(de, dn))
    track = np.concatenate([track, track[-1:]])
    track = np.degrees(np.unwrap(np.radians(track)))
    track_s = smooth(track, 5)
    dpsi = np.gradient(np.radians(track_s), s)
    # roll: coordinated turn, rolled in over a few seconds (smoothed in time)
    roll = np.degrees(np.arctan(v * v * dpsi / G))
    dt_s = np.gradient(t)
    roll = np.clip(roll, -27, 27)
    roll = smooth(roll, max(1.0, 2.0 / float(np.median(dt_s[s < 20000])))) if len(roll) > 10 else roll
    gamma = np.degrees(np.arctan(np.gradient(smooth(h, 3), s)))
    ias = v * np.sqrt(isa_density_ratio(h))
    aoa = 2.3 * (135.0 / np.maximum(ias, 50.0)) ** 2
    aoa -= np.clip((110.0 - ias) / 30.0, 0.0, 1.0) * 2.5  # flaps / slats out at low speed
    pitch = gamma + aoa
    # ---- resample at 10 Hz, light turbulence near the ground
    rate = cfg.get("rate_hz", 10.0)
    tt = np.arange(0.0, t[-1], 1.0 / rate)
    S = np.interp(tt, t, s)
    out = {"t": tt, "s": S}
    for k, arr_ in (("lat", lat), ("lon", lon), ("h", h), ("roll", roll), ("pitch", pitch), ("yaw", track_s), ("v", v)):
        out[k] = np.interp(S, s, arr_)
    rng = np.random.default_rng(seed)
    agl_dep = out["h"] - g_dep
    agl_arr = out["h"] - g_arr
    # (each half of the flight from its own airport: the departure's turbulence does not
    # depend on the arrival)
    first = tt < 0.5 * tt[-1]
    near = np.exp(-np.maximum(np.where(first, agl_dep, agl_arr), 0) / 1500.0)
    for k, amp, tau in (("roll", 0.8, 1.6), ("pitch", 0.35, 1.2), ("yaw", 0.25, 2.0)):
        a = math.exp(-1.0 / (rate * tau))
        n = rng.standard_normal(len(tt)) * amp * math.sqrt(1 - a * a)
        x = np.zeros(len(tt))
        for i in range(1, len(tt)):
            x[i] = a * x[i - 1] + n[i]
        out[k] = out[k] + x * near
    out["yaw"] = (out["yaw"] + 360.0) % 360.0
    phases = {
        "liftoff": 0.0,
        "top_of_climb": float(np.interp(s_toc, s, t)),
        "top_of_descent": float(np.interp(s_tod, s, t)),
        "touchdown": float(t[-1]),
    }
    meta = {
        "waypoints": w, "distance_m": float(total), "duration_s": float(t[-1]), "phases": phases,
        "ground_departure_m": float(g_dep), "ground_arrival_m": float(g_arr), "start_h_m": float(h0),
        # sparse polyline (every ~5 km) with the flight time: maps, overlays, insets
        "track": [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(lat[::50], lon[::50], t[::50], h[::50])] + [[float(lat[-1]), float(lon[-1]), float(t[-1]), float(h[-1])]],
    }
    return out, meta


def clearance(out, seed=1, every_s=20.0):
    """Lowest height above the terrain along the flight (sampled every `every_s` seconds)."""
    k = max(1, int(every_s * 10))
    idx = np.arange(0, len(out["t"]), k)
    g = ground_heights(list(zip(out["lat"][idx], out["lon"][idx])), seed)
    return out["t"][idx], out["h"][idx] - g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("story", nargs="?", default=os.path.join(HERE, "airliner.yaml"), help="storyboard with a `route:` section")
    story = yaml.safe_load(open(ap.parse_args().story))
    cfg = story["route"]
    d = os.path.join(ROOT, cfg.get("out", "out/airliner"))
    os.makedirs(d, exist_ok=True)
    out, meta = build(cfg)
    tc, agl = clearance(out)
    meta["min_clearance"] = [[float(a), float(b)] for a, b in zip(tc, agl)]
    with open(os.path.join(d, "route.csv"), "w") as f:
        f.write("t,lat,lon,h,roll,pitch,yaw\n")
        for i in range(len(out["t"])):
            f.write(f"{out['t'][i]:.2f},{out['lat'][i]:.8f},{out['lon'][i]:.8f},{out['h'][i]:.2f},{out['roll'][i]:.3f},{out['pitch'][i]:.3f},{out['yaw'][i]:.3f}\n")
    with open(os.path.join(d, "route.json"), "w") as f:
        json.dump(meta, f)
    ph = meta["phases"]
    print(f"route: {meta['distance_m'] / 1000:,.0f} km, {meta['duration_s'] / 3600:.2f} h; top of climb {ph['top_of_climb'] / 60:.1f} min, "
          f"top of descent {ph['top_of_descent'] / 60:.1f} min; departure ground {meta['ground_departure_m']:.0f} m, arrival {meta['ground_arrival_m']:.0f} m")
    away = (tc > 120) & (tc < meta["duration_s"] - 200)
    i = int(np.argmin(np.where(away, agl, 1e9)))
    print(f"lowest clearance away from the airports: {agl[i]:.0f} m at t = {tc[i]:.0f} s")
    # a sparse table for choosing shots
    for tt in list(range(0, 1800, 60)) + list(range(1800, int(meta["duration_s"]) - 1800, 600)) + list(range(int(meta["duration_s"]) - 1800, int(meta["duration_s"]), 60)):
        j = min(int(tt * 10), len(out["t"]) - 1)
        print(f"  t {tt:6d} s  {out['lat'][j]:8.3f} {out['lon'][j]:9.3f}  h {out['h'][j]:7.0f}  v {out['v'][j]:5.1f}  "
              f"roll {out['roll'][j]:5.1f} pitch {out['pitch'][j]:5.1f} yaw {out['yaw'][j]:5.1f}  s {out['s'][j] / 1000:7.1f} km")


if __name__ == "__main__":
    main()
