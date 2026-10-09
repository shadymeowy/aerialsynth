#!/usr/bin/env python3
"""Build the renderer's compact planetary ephemeris (crates/render/src/stars/ephem.rs) from JPL
DE440 (the 'de440s.bsp' kernel, public domain): the Chebyshev segments of the Sun, the planet
barycentres, the Earth-Moon barycentre (EMB) and the Moon relative to the EMB, cut to a date
range. Earth relative to the EMB is -Moon / EMRAT and is not stored.

    pip install jplephem
    curl -O https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440s.bsp
    python scripts/build_planets.py de440s.bsp -o crates/render/data/planets.bin

Format (little endian): b"PLANETS1", u32 body count, f64 EMRAT; per body: i32 centre, i32
target, f64 JD (TDB) of the first interval, f64 interval (days), u32 intervals, u32 coefficients
per axis; then per interval and axis the constant term as f64 and the others as f32 (km).
"""
import argparse, struct, sys
import numpy as np

BODIES = [(0, 10), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (3, 301)]
EMRAT = 81.30056822149722  # DE440 Earth/Moon mass ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bsp", help="JPL SPK kernel (de440s.bsp)")
    ap.add_argument("--start", type=float, default=2447892.5, help="first JD (TDB; default 1990-01-01)")
    ap.add_argument("--end", type=float, default=2473459.5, help="last JD (TDB; default 2060-01-01)")
    ap.add_argument("-o", "--out", required=True, help="output file (crates/render/data/planets.bin)")
    a = ap.parse_args()
    from jplephem.spk import SPK  # (pip install jplephem)
    k = SPK.open(a.bsp)
    out = [b"PLANETS1", struct.pack("<Id", len(BODIES), EMRAT)]
    worst = 0.0
    for c, t in BODIES:
        seg = k[c, t]
        init, intlen, coef = seg.load_array()  # coef: (3, n, ncoef), km
        i0 = int((a.start - init) // intlen)
        i1 = int(np.ceil((a.end - init) / intlen))
        sub = coef[:, i0:i1, :]
        n, nc = sub.shape[1], sub.shape[2]
        out.append(struct.pack("<iiddII", c, t, init + i0 * intlen, intlen, n, nc))
        blob = bytearray()
        for i in range(n):
            for ax in range(3):
                blob += struct.pack("<d", sub[ax, i, 0])
                blob += sub[ax, i, 1:].astype("<f4").tobytes()
        out.append(bytes(blob))
        # precision check: f32 rounding of the higher coefficients (km, worst case sum)
        err = np.abs(sub[:, :, 1:] - sub[:, :, 1:].astype(np.float32).astype(np.float64)).sum(axis=2).max()
        worst = max(worst, err)
        print(f"{c:>2} -> {t:<3}: {n} intervals x {nc} coefficients, rounding <= {err * 1000:.1f} m", file=sys.stderr)
    data = b"".join(out)
    open(a.out, "wb").write(data)
    print(f"wrote {a.out}: {len(data) / 1e6:.2f} MB, JD {a.start}..{a.end}, worst rounding {worst * 1000:.1f} m", file=sys.stderr)


if __name__ == "__main__":
    main()
