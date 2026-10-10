#!/usr/bin/env bash
# Package the release build of target/release (the `terrain` CLI and the C library) as
# dist/terrain-<version>-<target>.tar.gz (.zip on Windows). Run from the repository root after
#   cargo build --release --locked -p terrain -p aerialsynth-capi
#
#   terrain-<version>-<target>/
#     bin/terrain[.exe]
#     lib/libaerialsynth.so  + libaerialsynth.a         Linux
#         libaerialsynth.dylib + libaerialsynth.a       macOS
#         aerialsynth.dll + aerialsynth.dll.lib (import library) + aerialsynth.lib (static)  Windows
#         native-static-libs.txt (the system libraries to link with the static library)
#     include/aerialsynth.h, examples/{tile,render}.c, configs/, docs/, bindings/README.md,
#     bindings/python/README.md, README.md, LICENSE, NOTICE, CHANGELOG.md
set -euo pipefail
target=$1
version=$2
name="terrain-$version-$target"
stage="dist/$name"
rel=target/release
rm -rf "$stage"
mkdir -p "$stage/bin" "$stage/lib" "$stage/include" "$stage/examples"
case "$target" in
  *windows*)
    cp "$rel/terrain.exe" "$stage/bin/"
    cp "$rel/aerialsynth.dll" "$rel/aerialsynth.dll.lib" "$rel/aerialsynth.lib" "$stage/lib/" ;;
  *apple*)
    cp "$rel/terrain" "$stage/bin/"
    cp "$rel/libaerialsynth.dylib" "$rel/libaerialsynth.a" "$stage/lib/" ;;
  *)
    cp "$rel/terrain" "$stage/bin/"
    cp "$rel/libaerialsynth.so" "$rel/libaerialsynth.a" "$stage/lib/" ;;
esac
if [ -f "$rel/native-static-libs.txt" ]; then cp "$rel/native-static-libs.txt" "$stage/lib/"; fi
cp bindings/c/include/aerialsynth.h "$stage/include/"
cp bindings/c/examples/*.c "$stage/examples/"
cp -r configs "$stage/"
cp README.md LICENSE NOTICE CHANGELOG.md "$stage/"
# the documents README.md links to
cp -r docs "$stage/"
mkdir -p "$stage/bindings/python"
cp bindings/README.md "$stage/bindings/"
cp bindings/python/README.md "$stage/bindings/python/"
cd dist
case "$target" in
  *windows*) 7z a -tzip -bso0 "$name.zip" "$name" && echo "dist/$name.zip" ;;
  *) tar czf "$name.tar.gz" "$name" && echo "dist/$name.tar.gz" ;;
esac
