"""Tiles of a procedural planet as numpy arrays.

A :class:`World` is an HDF5 tile store of one world (seed + world config). :meth:`World.tile`
reads a layer of a Web-Mercator XYZ tile (256 x 256 pixels) from the store; a tile that is not
stored yet is generated (on the GPU when there is a suitable one, else on the CPU), stored, and
returned::

    import aerialsynth
    with aerialsynth.World("out/world.h5") as w:          # created if missing
        rgb = w.tile(12, 2200, 1500, "rgb")              # uint8 (256, 256, 3)
        h = w.tile(12, 2200, 1500, "elevation")          # float32 (256, 256), m above WGS84

Arrays are writable and own their memory (a view of a fresh ``bytearray``; no copy is made).
Row 0 is the north edge, column 0 the west edge.
"""

from __future__ import annotations

import os
from typing import Literal, NamedTuple, Optional, Union

import numpy as np

from . import _native

__all__ = ["World", "LayerInfo", "LAYERS", "TILE_SIZE", "MAX_ZOOM", "DEFAULT_MAX_ZOOM", "__version__"]

__version__: str = _native.__version__
#: Width and height of a tile in pixels.
TILE_SIZE: int = _native.TILE_SIZE
#: Highest zoom of the XYZ scheme (a world's own limit is :attr:`World.max_zoom`).
MAX_ZOOM: int = _native.MAX_ZOOM
#: Zoom limit of a world whose config sets no ``tiles.max_zoom``.
DEFAULT_MAX_ZOOM: int = _native.DEFAULT_MAX_ZOOM

LayerName = Literal["rgb", "albedo", "elevation", "normal", "landcover", "emission"]
PathLike = Union[str, "os.PathLike[str]"]


class LayerInfo(NamedTuple):
    """Pixel format and meaning of a layer."""

    name: str
    dtype: np.dtype
    channels: int
    #: array shape of a tile: (256, 256) or (256, 256, channels)
    shape: tuple
    #: bytes of one tile
    size: int
    description: str


def _layer(name: str, dtype: str, channels: int, size: int, description: str) -> LayerInfo:
    shape = (TILE_SIZE, TILE_SIZE) if channels == 1 else (TILE_SIZE, TILE_SIZE, channels)
    return LayerInfo(name, np.dtype(dtype), channels, shape, size, description)


#: Every layer by name: rgb, albedo, elevation, normal, landcover, emission.
LAYERS: dict[str, LayerInfo] = {t[0]: _layer(*t) for t in _native.layers()}


class World:
    """The tile store ``tiles_file`` of one world, generating missing tiles on demand.

    :param tiles_file: the HDF5 tile store; created (with its directory) if missing.
    :param config: a scenario YAML (its ``world:`` section is the world and its
        ``tiles.max_zoom`` the zoom limit, default 18; other sections are ignored) or a bare world
        config; ``None`` is the default world. Like ``terrain -c``.
    :param seed: overrides the config's seed (like ``terrain --seed``).
    :raises RuntimeError: the store holds another world (or generator version), or the config is
        invalid.
    :raises OSError: the config file cannot be read.

    A store holds exactly one world: open it with the config and seed that made it. A ``World``
    can be used from several threads; tile generation releases the GIL. A tiles file can be open
    by one ``World`` per process at a time (a second one raises ``RuntimeError``).
    """

    def __init__(self, tiles_file: PathLike, config: Optional[PathLike] = None, seed: Optional[int] = None) -> None:
        if seed is not None:
            seed = int(seed)
            if not 0 <= seed < 2**64:
                raise ValueError(f"seed {seed} is out of range 0 .. 2**64 - 1")
        self._w = _native.World(os.fspath(tiles_file), None if config is None else os.fspath(config), seed)

    def tile(self, z: int, x: int, y: int, layer: LayerName = "rgb") -> np.ndarray:
        """A layer of tile ``z/x/y`` (generated and stored if missing).

        ``z`` must be at most :attr:`max_zoom`, ``x`` and ``y`` less than ``2**z`` (XYZ scheme,
        ``y = 0`` at the north). The array has the layer's dtype and shape (see :data:`LAYERS`).

        :raises ValueError: bad coordinates or layer name, or the world is closed.
        :raises RuntimeError: reading, generating or storing the tile failed.
        """
        info = LAYERS.get(layer)
        if info is None:
            raise ValueError(f"unknown layer {layer!r} (layers: {', '.join(LAYERS)})")
        buf = self._w.tile(int(z), int(x), int(y), layer)
        return np.frombuffer(buf, dtype=info.dtype).reshape(info.shape)

    def close(self) -> None:
        """Close the store (idempotent)."""
        self._w.close()

    @property
    def closed(self) -> bool:
        return self._w.closed

    @property
    def path(self) -> str:
        """The tiles file."""
        return os.fspath(self._w.path)

    @property
    def seed(self) -> int:
        """The world's seed."""
        return self._w.seed

    @property
    def max_zoom(self) -> int:
        """The highest zoom served (the config's ``tiles.max_zoom``, default 18)."""
        return self._w.max_zoom

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else f"seed {self.seed}, max zoom {self.max_zoom}"
        return f"aerialsynth.World({self.path!r}, {state})"
