#!/usr/bin/env python3
"""Build the renderer's star catalogue (binary, see crates/render/src/stars/catalog.rs).

Sources (CDS / VizieR, ESA; free to use with acknowledgement):
  I/311  Hipparcos, the new reduction (van Leeuwen 2007): hip2.dat.gz       — astrometry
  I/239  Hipparcos main catalogue (ESA 1997): hip_main.dat                 — Johnson V, B-V
  I/259  Tycho-2 (Høg et al. 2000): tyc2.dat.00.gz … 19.gz, suppl_1.dat.gz — stars not in Hipparcos

Every star gets its ICRS position at epoch J2000.0 (Hipparcos positions are propagated from
1991.25 with their proper motion), proper motion (mas/yr, μα* = μα cos δ), parallax (mas),
Johnson V and B-V (Tycho: V = VT - 0.090 (BT-VT), B-V = 0.850 (BT-VT)). Records are sorted by
V, so a magnitude limit is a prefix.

    python scripts/build_stars.py CATDIR --vmax 9 -o crates/render/data/stars_v9.bin
    python scripts/build_stars.py CATDIR --vmax 12 -o tycho2.stars      # deeper, optional

Download (into CATDIR):
    B=https://cdsarc.cds.unistra.fr/ftp
    curl -O $B/I/311/hip2.dat.gz -O $B/I/239/hip_main.dat -O $B/I/259/suppl_1.dat.gz
    for i in $(seq -w 0 19); do curl -O $B/I/259/tyc2.dat.$i.gz; done
"""
import argparse, gzip, math, os, struct, sys
import numpy as np

MAGIC = b"STARCAT1"
EPOCH = 2000.0
UNIT = 2 * math.pi / 2**32  # rad per position unit (≈ 0.3 mas)


def fnum(s):
    s = s.strip()
    return float(s) if s else math.nan


def hipparcos(d):
    v, bv1 = {}, {}
    with open(os.path.join(d, "hip_main.dat"), encoding="latin-1") as f:
        for ln in f:
            hip = int(ln[8:14])
            v[hip] = fnum(ln[41:46])
            bv1[hip] = fnum(ln[245:251])
    rows = []
    with gzip.open(os.path.join(d, "hip2.dat.gz"), "rt") as f:
        for ln in f:
            hip = int(ln[0:6])
            ra, de = float(ln[15:28]), float(ln[29:42])
            plx, pmra, pmde = fnum(ln[43:50]), fnum(ln[51:59]), fnum(ln[60:68])
            hp, bv = fnum(ln[129:136]), fnum(ln[152:158])
            vm = v.get(hip, math.nan)
            if math.isnan(bv):
                bv = bv1.get(hip, math.nan)
            if math.isnan(bv):
                bv = 0.6
            if math.isnan(vm):
                # Hp → V (ESA 1997, vol. 1, approximate for normal colours)
                vm = hp - 0.304 * bv + 0.202 * bv * bv - 0.107 * bv**3 + 0.045 if not math.isnan(hp) else math.nan
            if math.isnan(vm):
                continue
            rows.append((hip, ra, de, 1991.25, pmra, pmde, max(plx, 0.0), vm, bv))
    return rows


def tycho(d):
    rows = []
    files = [os.path.join(d, f"tyc2.dat.{i:02d}.gz") for i in range(20)]
    for fn in files + [os.path.join(d, "suppl_1.dat.gz")]:
        suppl = fn.endswith("suppl_1.dat.gz")
        with gzip.open(fn, "rt") as f:
            for ln in f:
                hip = ln[115:121] if suppl else ln[142:148]
                if hip.strip():
                    continue  # Hipparcos star: from Hipparcos
                t1, t2, t3 = int(ln[0:4]), int(ln[5:10]), int(ln[11:12])
                tid = (1 << 31) | (t1 << 17) | (t2 << 3) | t3
                if suppl:
                    ra, de, ep = fnum(ln[15:27]), fnum(ln[28:40]), 1991.25
                    pmra, pmde = fnum(ln[41:48]), fnum(ln[49:56])
                    bt, vt = fnum(ln[83:89]), fnum(ln[96:102])
                else:
                    if ln[13] == "X":  # no mean position: observed position (epoch ≈ 1991.5), no proper motion
                        ra, de = fnum(ln[152:164]), fnum(ln[165:177])
                        ep = 1990.0 + 0.5 * (fnum(ln[178:182]) + fnum(ln[183:187]))
                        pmra = pmde = 0.0
                    else:
                        ra, de, ep = fnum(ln[15:27]), fnum(ln[28:40]), 2000.0
                        pmra, pmde = fnum(ln[41:48]), fnum(ln[49:56])
                    bt, vt = fnum(ln[110:116]), fnum(ln[123:129])
                if math.isnan(pmra):
                    pmra = pmde = 0.0
                if math.isnan(vt) and math.isnan(bt):
                    continue
                if math.isnan(bt):
                    vm, bv = vt, 0.6
                elif math.isnan(vt):
                    vm, bv = bt - 0.6, 0.6
                else:
                    vm, bv = vt - 0.090 * (bt - vt), 0.850 * (bt - vt)
                rows.append((tid, math.radians(ra), math.radians(de), ep, pmra, pmde, 0.0, vm, bv))
    return rows


def to_j2000(ra, de, ep, pmra, pmde):
    """Linear space motion on the sphere to epoch J2000 (vectorised)."""
    dt = EPOCH - ep
    mas = math.pi / 180 / 3600e3
    u = np.stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)], -1)
    ea = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], -1)
    ed = np.stack([-np.sin(de) * np.cos(ra), -np.sin(de) * np.sin(ra), np.cos(de)], -1)
    u = u + (dt * pmra * mas)[:, None] * ea + (dt * pmde * mas)[:, None] * ed
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return np.arctan2(u[:, 1], u[:, 0]) % (2 * math.pi), np.arcsin(np.clip(u[:, 2], -1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catdir")
    ap.add_argument("--vmax", type=float, default=9.0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    rows = hipparcos(a.catdir)
    print(f"Hipparcos: {len(rows)}", file=sys.stderr)
    ty = tycho(a.catdir)
    print(f"Tycho-2 (not in Hipparcos): {len(ty)}", file=sys.stderr)
    rows += ty
    arr = np.array([r[1:] for r in rows], dtype=np.float64)
    ids = np.array([r[0] for r in rows], dtype=np.uint32)
    keep = arr[:, 6] <= a.vmax
    arr, ids = arr[keep], ids[keep]
    order = np.argsort(arr[:, 6], kind="stable")
    arr, ids = arr[order], ids[order]
    ra, de = to_j2000(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])
    rec = np.zeros(len(ids), dtype=[("id", "<u4"), ("ra", "<u4"), ("de", "<i4"), ("pmra", "<f4"), ("pmde", "<f4"),
                                    ("plx", "<f4"), ("v", "<i2"), ("bv", "<i2")])
    rec["id"] = ids
    rec["ra"] = np.round(ra / UNIT).astype(np.int64) % 2**32
    rec["de"] = np.round(de / UNIT).astype(np.int64)
    rec["pmra"], rec["pmde"], rec["plx"] = arr[:, 3], arr[:, 4], arr[:, 5]
    rec["v"] = np.round(arr[:, 6] * 1000).astype(np.int16)
    rec["bv"] = np.round(np.clip(arr[:, 7], -0.5, 3.0) * 1000).astype(np.int16)
    with open(a.out, "wb") as f:
        f.write(MAGIC + struct.pack("<Id", len(rec), EPOCH))
        f.write(rec.tobytes())
    hip = (ids & 0x80000000) == 0
    print(f"wrote {a.out}: {len(rec)} stars (V ≤ {a.vmax}; {hip.sum()} Hipparcos, {(~hip).sum()} Tycho-2), "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB", file=sys.stderr)
    for m in (4, 6, 7, 8, 9, 10, 11, 12):
        if m <= a.vmax + 1:
            print(f"  V ≤ {m}: {int((arr[:, 6] <= m).sum())}", file=sys.stderr)


if __name__ == "__main__":
    main()
