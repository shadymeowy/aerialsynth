"""Tiles and camera images of a procedural planet as numpy arrays.

A :class:`World` is an HDF5 tile store of one world (seed + world config). :meth:`World.tile`
reads a layer of a Web-Mercator XYZ tile (256 x 256 pixels) from the store; a tile that is not
stored yet is generated (on the GPU when there is a suitable one, else on the CPU), stored, and
returned::

    import aerialsynth
    with aerialsynth.World("out/world.h5") as w:          # created if missing
        rgb = w.tile(12, 2200, 1500, "rgb")              # uint8 (256, 256, 3)
        h = w.tile(12, 2200, 1500, "elevation")          # float32 (256, 256), m above WGS84
        block = w.tiles([(12, x, y) for y in range(1500, 1504) for x in range(2200, 2204)])
                                                          # uint8 (16, 256, 256, 3), one call

Decoded tiles are kept in an in-memory cache (``cache_mb``, default 256 MiB) of the world.

A :class:`Camera` renders images of the world from a pose, with the renderer of ``terrain run``
(the tiles in view are generated into the store when missing)::

        cam = w.camera(width=640, height=480, hfov=90)
        ground = w.surface_height(45.0, 10.0)
        f = cam.render(45.0, 10.0, ground + 300, pitch=-30, yaw=90,
                       time="2026-06-21T07:30:00Z", depth=True)
        f.rgb, f.depth                                    # uint8 (480, 640, 3), float32 (480, 640)

The first render at a new place generates the tiles it needs: on the CPU (no GPU that
generates tiles) that takes minutes; ``World(..., verbose=True)`` reports it on stderr, and
:meth:`World.prefetch` makes an area's tiles ahead. Later renders read them from the store.

Arrays are writable and own their memory (a view of a fresh ``bytearray``; no copy is made).
Tile row 0 is the north edge, column 0 the west edge; image row 0 is the image top.
"""

from __future__ import annotations

import datetime as _dt
import operator
import os
from typing import Literal, NamedTuple, Optional, Union

import numpy as np

from . import _native

__all__ = [
    "World",
    "Camera",
    "Frame",
    "LayerInfo",
    "LAYERS",
    "TILE_SIZE",
    "MAX_ZOOM",
    "DEFAULT_MAX_ZOOM",
    "DEFAULT_CACHE_MB",
    "MAX_PREFETCH_TILES",
    "CacheInfo",
    "MAX_IMAGE_SIZE",
    "SKY",
    "__version__",
]

__version__: str = _native.__version__
#: Width and height of a tile in pixels.
TILE_SIZE: int = _native.TILE_SIZE
#: Highest zoom of the XYZ scheme (a world's own limit is :attr:`World.max_zoom`).
MAX_ZOOM: int = _native.MAX_ZOOM
#: Zoom limit of a world whose config sets no ``tiles.max_zoom``.
DEFAULT_MAX_ZOOM: int = _native.DEFAULT_MAX_ZOOM
#: Default size of a world's tile cache in MiB.
DEFAULT_CACHE_MB: int = _native.DEFAULT_CACHE_MB
#: Most tiles :meth:`World.prefetch` takes (its box over its zooms).
MAX_PREFETCH_TILES: int = _native.MAX_PREFETCH_TILES

#: Largest image width or height of a camera.
MAX_IMAGE_SIZE: int = _native.MAX_IMAGE_SIZE
#: Land-cover class of sky pixels in a rendered frame.
SKY: int = 255

LayerName = Literal["rgb", "albedo", "elevation", "normal", "landcover", "emission"]
BackendName = Literal["auto", "cpu", "gpu"]
PathLike = Union[str, "os.PathLike[str]"]
TimeLike = Union[float, int, str, _dt.datetime, None]


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


class CacheInfo(NamedTuple):
    """Usage of a world's tile cache (:meth:`World.cache_info`)."""

    #: size limit in MiB (0: off)
    size_mb: int
    #: bytes held
    bytes: int
    #: layers of tiles held
    entries: int
    #: lookups that found the layer since the world was opened
    hits: int
    #: lookups that did not
    misses: int


class World:
    """The tile store ``tiles_file`` of one world, generating missing tiles on demand.

    :param tiles_file: the HDF5 tile store; created (with its directory) if missing.
    :param config: a scenario YAML (its ``world:`` section is the world and its
        ``tiles.max_zoom`` the zoom limit, default 18; other sections are ignored) or a bare world
        config; ``None`` is the default world. Like ``terrain -c``.
    :param seed: overrides the config's seed (like ``terrain --seed``).
    :param cache_mb: size of the in-memory cache of decoded tiles in MiB (0: no cache; see
        :attr:`cache_mb`).
    :param verbose: report tile generation on stderr (see :attr:`verbose`).
    :raises RuntimeError: the store holds another world (or generator version), or the config is
        invalid.
    :raises OSError: the config file cannot be read.

    A store holds exactly one world: open it with the config and seed that made it. A ``World``
    can be used from several threads; tile generation releases the GIL. A tiles file can be open
    by one ``World`` per process at a time (a second one raises ``RuntimeError``).
    """

    def __init__(
        self,
        tiles_file: PathLike,
        config: Optional[PathLike] = None,
        seed: Optional[int] = None,
        cache_mb: int = DEFAULT_CACHE_MB,
        verbose: bool = False,
    ) -> None:
        if seed is not None:
            seed = _int(seed, "seed")
            if not 0 <= seed < 2**64:
                raise ValueError(f"seed {seed} is out of range 0 .. 2**64 - 1")
        cache_mb = _index(cache_mb, "cache_mb")
        self._w = _native.World(os.fspath(tiles_file), None if config is None else os.fspath(config), seed, cache_mb)
        self._w.verbose = bool(verbose)

    def tile(self, z: int, x: int, y: int, layer: LayerName = "rgb") -> np.ndarray:
        """A layer of tile ``z/x/y`` (generated and stored if missing).

        ``z`` must be at most :attr:`max_zoom`, ``x`` and ``y`` less than ``2**z`` (XYZ scheme,
        ``y = 0`` at the north). The array has the layer's dtype and shape (see :data:`LAYERS`).

        :raises TypeError: a coordinate is not an integer (Python or numpy).
        :raises ValueError: bad coordinates or layer name, or the world is closed.
        :raises RuntimeError: reading, generating or storing the tile failed.
        """
        info = LAYERS.get(layer)
        if info is None:
            raise ValueError(f"unknown layer {layer!r} (layers: {', '.join(LAYERS)})")
        buf = self._w.tile(_int(z, "z"), _int(x, "x"), _int(y, "y"), layer)
        return np.frombuffer(buf, dtype=info.dtype).reshape(info.shape)

    def tiles(self, coords, layer: LayerName = "rgb") -> np.ndarray:
        """A layer of many tiles at once: ``coords`` is a sequence (or an array) of ``(z, x, y)``,
        shape ``(n, 3)``; the result has shape ``(n,) + LAYERS[layer].shape``, tile ``i`` at
        index ``i``.

        Faster than :meth:`tile` in a loop: every coordinate is checked before any work (one out
        of range raises ``ValueError`` and nothing is read or generated), cached tiles are copied,
        stored ones are read and decompressed in parallel, and missing ones are generated
        together (in batches of up to 64 tiles: on the GPU, many tiles per dispatch) and stored.
        A tile listed several times is read or generated once. The GIL is released meanwhile.

        :raises TypeError: ``coords`` is not an integer array.
        :raises ValueError: bad coordinates or layer name, or the world is closed.
        :raises RuntimeError: reading, generating or storing a tile failed (the tiles generated
            before the failure are stored).
        """
        info = LAYERS.get(layer)
        if info is None:
            raise ValueError(f"unknown layer {layer!r} (layers: {', '.join(LAYERS)})")
        c = np.asarray(coords)
        if c.size == 0:
            c = c.reshape(0, 3)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"coords: (z, x, y) triples expected (shape (n, 3)), got shape {c.shape}")
        if c.dtype.kind not in "iu" and c.size:
            raise TypeError(f"coords must be integers, not {c.dtype}")
        if c.size and (c.min() < 0 or c.max() >= 2**32):
            raise ValueError("coords: z, x and y must be in 0 .. 2**32 - 1")
        zxy = np.ascontiguousarray(c, dtype="<u4").tobytes()
        buf = self._w.tiles(zxy, layer)
        return np.frombuffer(buf, dtype=info.dtype).reshape((len(c),) + info.shape)

    def prefetch(self, bbox: tuple, zooms: Union[int, tuple]) -> int:
        """Generate and store the missing tiles of a box, without returning them.

        :param bbox: ``(lat_min, lon_min, lat_max, lon_max)`` in degrees (``lon_min > lon_max``
            is a box across the antimeridian); the tiles intersecting it are taken.
        :param zooms: a zoom or a ``(z_min, z_max)`` range (inclusive), at most :attr:`max_zoom`.
        :returns: the number of tiles generated (the stored ones are skipped).
        :raises ValueError: a bad box or zoom range, or more than :data:`MAX_PREFETCH_TILES`
            tiles over those zooms (stored ones included); nothing is generated then.
        :raises RuntimeError: generating or storing failed (the batches done are stored).

        Tiles are generated in batches of up to 64 (as :meth:`tiles`); the GIL is released, but
        the call cannot be interrupted.
        """
        lat_min, lon_min, lat_max, lon_max = (float(v) for v in bbox)
        z0, z1 = (zooms, zooms) if not isinstance(zooms, (tuple, list)) else zooms
        return self._w.prefetch(lat_min, lon_min, lat_max, lon_max, _index(z0, "z_min"), _index(z1, "z_max"))

    @property
    def cache_mb(self) -> int:
        """Size of the in-memory cache of decoded tiles in MiB (default 256): :meth:`tile` and
        :meth:`tiles` copy cached tiles instead of reading and decompressing them from the store.
        Tiles read or generated are cached (a generated tile with all its layers, ~1.1 MiB; an
        rgb layer is 192 KiB), the least recently used dropped beyond the size. Settable; 0 turns
        the cache off and frees it."""
        return self.cache_info().size_mb

    @cache_mb.setter
    def cache_mb(self, mb: int) -> None:
        self._w.set_cache_mb(_index(mb, "cache_mb"))

    def cache_info(self) -> CacheInfo:
        """Usage of the tile cache."""
        return CacheInfo(*self._w.cache_info())

    @property
    def verbose(self) -> bool:
        """Report tile generation on stderr: a line per batch of tiles generated (count, zooms,
        CPU or GPU, time) by :meth:`tile`, :meth:`tiles`, :meth:`prefetch` and the renders of
        this world's cameras. Settable; off by default."""
        return self._w.verbose

    @verbose.setter
    def verbose(self, on: bool) -> None:
        self._w.verbose = bool(on)

    def camera(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        hfov: Optional[float] = None,
        *,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        mount: Literal["forward", "nadir"] = "forward",
        config: Optional[PathLike] = None,
        camera: Union[str, int, None] = None,
        backend: Optional[BackendName] = None,
    ) -> "Camera":
        """A camera over this world, for :meth:`Camera.render`.

        Either a distortion-free pinhole camera, ``width`` x ``height`` pixels with a horizontal
        field of view of ``hfov`` degrees (principal point ``cx, cy`` in pixels, default the image
        centre; ``mount`` "forward": the pose's angles are the camera's own, or "nadir": a level
        pose looks straight down), or a camera of the scenario ``config`` (``camera``: its HDF5
        path such as ``"/cam0"`` or its index; default the first one).

        :param config: a scenario YAML (as for ``terrain run``). Its ``render`` section (backend,
            supersample, shading, lighting, atmosphere, sensor of its cameras...), ``tiles`` zoom
            range and cache size and ``cameras`` are used; its ``world`` section is ignored (the
            world is this one). ``None``: the default settings.
        :param backend: "auto" (the GPU when there is a usable one, else the CPU), "cpu" or "gpu";
            default: the scenario's ``render.backend`` (auto).
        :raises ValueError: bad camera settings, an unknown camera, or the world is closed.
        :raises RuntimeError: an invalid scenario, or ``backend="gpu"`` without a usable GPU.
        :raises OSError: the config file cannot be read.

        The camera keeps the store open until it is closed; closing the world closes its cameras.
        """
        if camera is not None and not isinstance(camera, str):
            camera = str(_index(camera, "camera"))
        size = [None if v is None else _index(v, name) for v, name in ((width, "width"), (height, "height"))]
        for v, name in zip(size, ("width", "height")):
            if v is not None and not 0 < v <= MAX_IMAGE_SIZE:
                raise ValueError(f"{name} {v} is out of range 1 .. {MAX_IMAGE_SIZE}")
        c = self._w.camera(
            size[0],
            size[1],
            None if hfov is None else float(hfov),
            None if cx is None else float(cx),
            None if cy is None else float(cy),
            mount,
            None if config is None else os.fspath(config),
            camera,
            backend,
        )
        return Camera(c)

    def surface_height(self, lat: float, lon: float) -> float:
        """The DSM height (ground, canopy, buildings, water surface) at ``lat``, ``lon`` (degrees)
        in metres above the WGS84 ellipsoid: about what the tiles of the max zoom hold there
        (evaluated by the generator; no tile is made). Add a height above ground to it for
        :meth:`Camera.render`.
        """
        return self._w.surface_height(float(lat), float(lon))

    def close(self) -> None:
        """Close the store and the cameras of this world (idempotent)."""
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


def _int(v: object, name: str) -> int:
    """An integer argument: an ``int`` or a numpy integer (anything with ``__index__``), not a
    float or a bool (no silent truncation)."""
    if isinstance(v, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not {type(v).__name__}")
    try:
        return operator.index(v)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{name} must be an integer, not {type(v).__name__}") from None


def _index(v: object, name: str) -> int:
    """A non-negative integer argument."""
    i = _int(v, name)
    if i < 0:
        raise ValueError(f"{name} {i} must be >= 0")
    return i


def _unix_time(time: TimeLike) -> Optional[float]:
    """UTC Unix seconds of a time argument (None stays None)."""
    if time is None:
        return None
    if isinstance(time, str):
        s = time.strip()
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        try:
            time = _dt.datetime.fromisoformat(s)
        except ValueError:
            raise ValueError(f"time {time!r}: an ISO 8601 date and time such as 2026-06-21T07:30:00Z expected") from None
    if isinstance(time, _dt.datetime):
        if time.tzinfo is None:
            time = time.replace(tzinfo=_dt.timezone.utc)  # naive: UTC
        return time.timestamp()
    if isinstance(time, (int, float, np.integer, np.floating)) and not isinstance(time, bool):
        return float(time)
    raise TypeError(f"time must be Unix seconds, a datetime or an ISO 8601 string, not {type(time).__name__}")


class Frame(NamedTuple):
    """A rendered frame. Images are row-major, row 0 = image top; ``None`` when not asked for."""

    #: (H, W, 3) uint8 sRGB after the camera sensor model
    rgb: Optional[np.ndarray]
    #: (H, W) float32 metres: z-depth along the optical axis (or the range along the pixel ray
    #: when the scenario camera's ``depth.kind`` is ``range``); +inf = sky
    depth: Optional[np.ndarray]
    #: (H, W) uint8 land-cover class ids (as the ``landcover`` layer); 255 (:data:`SKY`) = sky
    landcover: Optional[np.ndarray]
    #: (exposure time s, gain, EV) of the RGB image, else None
    exposure: Optional[tuple]
    #: (3,) camera centre in ECEF metres
    position_ecef: np.ndarray
    #: (3, 3) rotation camera -> ECEF (columns: the camera's x right, y down, z forward axes)
    r_ecef_cam: np.ndarray
    #: UTC of the lighting (Unix seconds): the time asked for, or the scenario's
    time: float
    #: sun azimuth (clockwise from north) and elevation at the camera, degrees
    sun_azimuth: float
    sun_elevation: float

    @property
    def datetime(self) -> _dt.datetime:
        """:attr:`time` as an aware UTC datetime."""
        return _dt.datetime.fromtimestamp(self.time, _dt.timezone.utc)


class Camera:
    """A camera over a :class:`World` (made by :meth:`World.camera`), rendering frames from poses.

    One camera renders one frame at a time (calls from several threads are serialized; rendering
    releases the GIL); several cameras render concurrently. A camera keeps its world's store open
    until it is closed (``close``, ``with``, or closing the world).
    """

    def __init__(self, native: "_native.Camera") -> None:
        self._c = native

    def render(
        self,
        lat: float,
        lon: float,
        height: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        *,
        time: TimeLike = None,
        rgb: bool = True,
        depth: bool = False,
        landcover: bool = False,
    ) -> Frame:
        """Render a frame from a pose; tiles in view are generated and stored when missing (the
        first render at a new place: minutes on the CPU, see :attr:`World.verbose` and
        :meth:`World.prefetch`).

        :param lat, lon: geodetic position in degrees (``|lat| <= 90``).
        :param height: metres above the WGS84 ellipsoid (see :meth:`World.surface_height`).
        :param roll, pitch, yaw: attitude of the body in degrees, aerospace Z-Y-X Euler angles
            in the local north-east-down frame: yaw = heading clockwise from north, pitch nose-up
            positive, roll right-wing-down positive. The camera sits on the body by its mount: a
            "forward" pinhole camera looks along the heading (pitch -30 looks 30 degrees down),
            a "nadir" one straight down when level; scenario cameras by their ``extrinsics``.
        :param time: UTC of the sun, moon and stars: Unix seconds, a ``datetime`` (naive = UTC)
            or an ISO 8601 string ("2026-06-21T07:30:00Z"); None: the scenario's lighting (a
            fixed sun by default).
        :param rgb, depth, landcover: the images to make (only those are computed).
        :raises ValueError: a pose out of range or non-finite, a bad time, or the camera is
            closed.
        :raises RuntimeError: rendering failed (e.g. the GPU device was lost).

        The RGB image is that of the first frame of a ``terrain run`` sequence at this pose: the
        auto exposure converged on it, no motion blur. Its noise is deterministic (a function of
        the camera, the pose and the time): the same call gives the same image.
        """
        t = _unix_time(time)
        r = self._c.render(float(lat), float(lon), float(height), float(roll), float(pitch), float(yaw), t, bool(rgb), bool(depth), bool(landcover))
        h, w = self._c.height, self._c.width
        img, dep, lc, exposure, pos, rot, unix, sun_az, sun_el = r
        return Frame(
            rgb=None if img is None else np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3),
            depth=None if dep is None else np.frombuffer(dep, dtype="<f4").reshape(h, w),
            landcover=None if lc is None else np.frombuffer(lc, dtype=np.uint8).reshape(h, w),
            exposure=exposure,
            position_ecef=np.array(pos, dtype=np.float64),
            r_ecef_cam=np.array(rot, dtype=np.float64).reshape(3, 3),
            time=unix,
            sun_azimuth=sun_az,
            sun_elevation=sun_el,
        )

    @property
    def width(self) -> int:
        return self._c.width

    @property
    def height(self) -> int:
        return self._c.height

    @property
    def shape(self) -> tuple:
        """(height, width) of the images."""
        return (self._c.height, self._c.width)

    @property
    def backend(self) -> str:
        """Where frames are rendered: "cpu" or "gpu" (auto resolved)."""
        return self._c.backend

    @property
    def supersample(self) -> int:
        """Samples per pixel and axis of the RGB image."""
        return self._c.supersample

    @property
    def path(self) -> str:
        """The camera's HDF5 path in the scenario ("/cam0" for a pinhole camera)."""
        return self._c.path

    @property
    def model(self) -> str:
        """Camera model of the camera YAML schema: pinhole, pinhole_full, kannala_brandt, mei,
        scaramuzza."""
        return self._c.model

    @property
    def intrinsics(self) -> tuple:
        """The model's ``intrinsics`` values (pinhole: fx, fy, cx, cy in pixels; OpenCV pixel
        coordinates, (0, 0) = centre of the top-left pixel)."""
        return tuple(self._c.intrinsics)

    @property
    def distortion(self) -> tuple:
        """The model's ``distortion`` values (pinhole: k1, k2, p1, p2)."""
        return tuple(self._c.distortion)

    @property
    def K(self) -> Optional[np.ndarray]:
        """3 x 3 camera matrix of a pinhole model (None for the other models)."""
        if self.model not in ("pinhole", "pinhole_full"):
            return None
        fx, fy, cx, cy = self.intrinsics[:4]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def depth_is_range(self) -> bool:
        """Is :attr:`Frame.depth` the range along the pixel ray (scenario ``depth.kind: range``)
        rather than the z-depth?"""
        return self._c.depth_is_range

    def close(self) -> None:
        """Close the camera (idempotent)."""
        self._c.close()

    @property
    def closed(self) -> bool:
        return self._c.closed

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else self.backend
        return f"aerialsynth.Camera({self.path!r}, {self.model} {self.width}x{self.height}, {state})"
