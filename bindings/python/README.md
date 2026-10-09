# aerialsynth (Python)

Tiles of the [aerialsynth](https://github.com/shadymeowy/aerialsynth) procedural planet as numpy
arrays. A `World` is an HDF5 tile store of one world (seed + world config); `World.tile` reads a
layer of a Web-Mercator XYZ tile (256 × 256 pixels) from the store, and a tile that is not stored
yet is generated (on the GPU when there is a suitable one, else on the CPU), stored and returned.

```python
import aerialsynth

with aerialsynth.World("out/world.h5", config=None, seed=None) as w:   # created if missing
    rgb = w.tile(12, 2200, 1500, "rgb")          # uint8 (256, 256, 3)
    h = w.tile(12, 2200, 1500, "elevation")      # float32 (256, 256), m above the WGS84 ellipsoid

aerialsynth.LAYERS["normal"]   # LayerInfo(name='normal', dtype=dtype('int8'), channels=3, shape=(256, 256, 3), ...)
```

- **The world** is given like the CLI's `terrain -c CONFIG --seed N`: `config` is a scenario YAML
  (its `world:` section is the world and its `tiles.max_zoom` the zoom limit, default 18; the
  other sections are ignored) or a bare world config; `None` is the default world. `seed`
  overrides the config's seed.
- **One world per store:** opening a store made by another world (or generator version) raises
  `RuntimeError`, naming the settings that differ. Give each world its own tiles file.
- **Tiles:** `z` at most `World.max_zoom`, `x, y < 2**z`; row 0 is the north edge. The arrays are
  writable and own their memory (no copy is made from the extension's buffer).
- **Errors:** `ValueError` for bad coordinates, layer names, seeds or a closed world;
  `FileNotFoundError` / `OSError` for an unreadable config; `RuntimeError` for a store of another
  world, an invalid config, or a failed read / generation.
- **Threads:** a `World` can be used from several threads; generation releases the GIL. A tiles
  file can be open by only one `World` per process at a time (a second one raises
  `RuntimeError`): share it, or close the first.

| layer | dtype | shape | |
|---|---|---|---|
| `rgb` | uint8 | 256 × 256 × 3 | satellite look (fixed sun, haze), sRGB |
| `albedo` | uint8 | 256 × 256 × 3 | unlit surface colour, sRGB encoded |
| `elevation` | float32 | 256 × 256 | DSM (ground, canopy, buildings, water), m above the WGS84 ellipsoid, at pixel centres |
| `normal` | int8 | 256 × 256 × 3 | unit normal (east, north, up) × 127 |
| `landcover` | uint8 | 256 × 256 | class id: 0 unknown, 1 ocean, 2 lake, 3 river, 4 beach, 5 sand, 6 rock, 7 snow, 8 grass, 9 shrub, 10 forest, 11 crop, 12 building, 13 road, 14 wetland, 15 tundra, 16 bare, 17 urban |
| `emission` | uint8 | 256 × 256 × 3 | night lights, linear radiance = 16 (v/255)³ |

## Install

Linux x86_64 (glibc 2.28 or newer, CPython 3.10 or newer): the wheel attached to each
[GitHub release](https://github.com/shadymeowy/aerialsynth/releases):

```sh
pip install aerialsynth-0.1.0-cp310-abi3-manylinux_2_28_x86_64.whl
```

A GPU is optional: without a suitable one, tiles are generated on the CPU.

## Build

One wheel (`cp310-abi3`, the stable ABI) serves CPython 3.10 and newer. It needs the Rust
toolchain of the workspace, a C compiler and CMake (HDF5 is built from source), and
[maturin](https://www.maturin.rs):

```sh
cd bindings/python
maturin build --release -o dist        # or: pip install .
pip install dist/aerialsynth-*.whl
pip install pytest && pytest tests
```

The only Python dependency is numpy.
