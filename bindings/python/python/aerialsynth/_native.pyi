# Type stub of the extension module (bindings/python/src/lib.rs). The public API is the wrapper
# in __init__.py; this describes what it calls.
import os
import pathlib

__version__: str
TILE_SIZE: int
MAX_ZOOM: int
MAX_IMAGE_SIZE: int
DEFAULT_MAX_ZOOM: int
DEFAULT_CACHE_MB: int
MAX_PREFETCH_TILES: int
GENERATOR_VERSION: int

class World:
    def __init__(
        self,
        tiles_file: str | os.PathLike[str],
        config: str | os.PathLike[str] | None = None,
        seed: int | None = None,
        cache_mb: int = ...,
    ) -> None: ...
    def camera(
        self,
        width: int | None = None,
        height: int | None = None,
        hfov: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        mount: str = "forward",
        config: str | os.PathLike[str] | None = None,
        camera: str | None = None,
        backend: str | None = None,
    ) -> Camera: ...
    def surface_height(self, lat: float, lon: float) -> float: ...
    def tile(self, z: int, x: int, y: int, layer: str) -> bytearray: ...
    def tiles(self, zxy: bytes, layer: str) -> bytearray: ...
    def prefetch(self, lat_min: float, lon_min: float, lat_max: float, lon_max: float, z_min: int, z_max: int) -> int: ...
    def set_cache_mb(self, mb: int) -> None: ...
    def cache_info(self) -> tuple[int, int, int, int, int]: ...
    @property
    def verbose(self) -> bool: ...
    @verbose.setter
    def verbose(self, on: bool) -> None: ...
    def close(self) -> None: ...
    @property
    def closed(self) -> bool: ...
    @property
    def path(self) -> pathlib.Path: ...
    @property
    def seed(self) -> int: ...
    @property
    def max_zoom(self) -> int: ...

class Camera:
    def render(
        self,
        lat: float,
        lon: float,
        height: float,
        roll: float,
        pitch: float,
        yaw: float,
        time: float | None,
        rgb: bool,
        depth: bool,
        landcover: bool,
    ) -> tuple[
        bytearray | None,
        bytearray | None,
        bytearray | None,
        tuple[float, float, float] | None,
        list[float],
        list[float],
        float,
        float,
        float,
    ]: ...
    def close(self) -> None: ...
    @property
    def closed(self) -> bool: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def backend(self) -> str: ...
    @property
    def supersample(self) -> int: ...
    @property
    def path(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def intrinsics(self) -> list[float]: ...
    @property
    def distortion(self) -> list[float]: ...
    @property
    def depth_is_range(self) -> bool: ...

def layers() -> list[tuple[str, str, int, int, str]]: ...
