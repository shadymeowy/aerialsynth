#!/usr/bin/env bash
# Bit-identity check of the classic generator against v0.2.0: generates the same tiles (seed 1,
# the default world) with the v0.2.0 `terrain` binary and with this checkout's, on the CPU and
# on the GPU, and compares the stores layer by layer (`tilestore` example `tilediff`).
#
#   scripts/classic_v020_check.sh V020_TERRAIN [OUT_DIR] [cpu|gpu|both]
#
# V020_TERRAIN: a `terrain` binary built from the tag, e.g.
#   git worktree add --detach ../v020 v0.2.0 && (cd ../v020 && CARGO_TARGET_DIR=/tmp/v020-target \
#     cargo build --release -p terrain)   →   /tmp/v020-target/release/terrain
# This checkout's `terrain` and `tilediff` are built by the script (release). The GPU half needs
# a GPU with 64-bit shaders (both binaries pick the same adapter, AERIALSYNTH_GPU). Tiles: zooms
# 6–16 (incl. 8, 12, 16) at four places (home, boreal 53°N, the Black Sea coast, southern
# Africa), 24 per backend. Exits non-zero if any tile differs.
set -euo pipefail
v020=${1:?usage: classic_v020_check.sh V020_TERRAIN [OUT_DIR] [cpu|gpu|both]}
out=${2:-out/classic_v020}
which=${3:-both}
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)
target=${CARGO_TARGET_DIR:-$root/target}
(cd "$root" && cargo build --release -q -p terrain && cargo build --release -q -p tilestore --example tilediff)
cur=$target/release/terrain
diff_bin=$target/release/examples/tilediff

# the tile list: z/x/y of each place at each zoom (web-mercator XYZ)
python3 - "$out/tiles.txt" <<'EOF'
import math, sys
places = [(39.9, 32.79), (52.97924, -16.41426), (41.562, 32.955), (-26.00742, 32.99572)]
lines = []
for lat, lon in places:
    for z in (6, 8, 10, 12, 14, 16):
        n = 2 ** z
        x = int((lon + 180.0) / 360.0 * n)
        r = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * n)
        lines.append(f"{z}/{x}/{y}")
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
EOF

fail=0
for backend in cpu gpu; do
  [[ $which == both || $which == "$backend" ]] || continue
  for who in v020 classic; do
    bin=$cur; [[ $who == v020 ]] && bin=$v020
    store=$out/${who}_$backend.h5
    rm -f "$store"
    cat > "$out/${who}_$backend.yaml" <<EOF
world: { seed: 1 }
tiles: { file: $store, generator: $backend, max_zoom: 18 }
EOF
    echo "== $who ($backend): $(wc -l < "$out/tiles.txt") tiles"
    rc=0
    "$bin" tiles -c "$out/${who}_$backend.yaml" --list "$out/tiles.txt" || rc=$?
    if (( rc == 139 )) && [[ $who == v020 && $backend == gpu ]]; then
      # v0.2.0's NVIDIA exit segfault (the driver unloaded under its threads; fixed since): it
      # comes after the tiles are written and the store is closed
      echo "(v0.2.0 crashed at exit, after writing the store: the exit segfault fixed since v0.2.0)"
    elif (( rc != 0 )); then
      exit $rc
    fi
  done
  echo "== v0.2.0 vs classic ($backend)"
  "$diff_bin" "$out/v020_$backend.h5" "$out/classic_$backend.h5" || fail=1
done
exit $fail
