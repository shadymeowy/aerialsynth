#!/usr/bin/env bash
# Build the `terrain` CLI and the C library for the host target and package them
# (.github/scripts/package.sh). Run from the repository root: release-build.sh TARGET VERSION
set -euo pipefail
target=$1
version=$2
# no debug info in the release binaries (the workspace's release profile keeps line tables)
export CARGO_PROFILE_RELEASE_DEBUG=0
cargo build --release --locked -p terrain -p aerialsynth-capi
# the system libraries a program linking the static library needs (rebuilds the static library)
cargo rustc --color never --release --locked -p aerialsynth-capi --lib --crate-type staticlib -- --print native-static-libs 2>&1 \
  | sed -n 's/.*native-static-libs: //p' | tail -n 1 | tr -d '\r' > target/release/native-static-libs.txt
echo "native-static-libs: $(cat target/release/native-static-libs.txt)"
test -s target/release/native-static-libs.txt
.github/scripts/package.sh "$target" "$version"
