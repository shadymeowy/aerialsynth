"""Tests of the aerialsynth package (tile stores are made in pytest's temporary directories)."""

import threading

import numpy as np
import pytest

import aerialsynth


@pytest.fixture
def config(tmp_path):
    """A cheap world (one sample per pixel) served up to zoom 8."""
    p = tmp_path / "world.yaml"
    p.write_text("world:\n  tile_supersample: 1\ntiles:\n  max_zoom: 8\n")
    return p


def test_layers():
    assert list(aerialsynth.LAYERS) == ["rgb", "albedo", "elevation", "normal", "landcover", "emission"]
    expect = {"rgb": ("uint8", 3), "albedo": ("uint8", 3), "elevation": ("float32", 1), "normal": ("int8", 3),
              "landcover": ("uint8", 1), "emission": ("uint8", 3)}
    for name, info in aerialsynth.LAYERS.items():
        assert (info.dtype.name, info.channels) == expect[name]
        assert info.size == 256 * 256 * info.channels * info.dtype.itemsize
        assert info.shape == ((256, 256) if info.channels == 1 else (256, 256, info.channels))
    assert aerialsynth.TILE_SIZE == 256 and aerialsynth.DEFAULT_MAX_ZOOM == 18


def test_tiles_are_generated_then_read(tmp_path, config):
    path = tmp_path / "store" / "world.h5"
    with aerialsynth.World(path, config=config, seed=5) as w:
        assert (w.seed, w.max_zoom, w.path) == (5, 8, str(path))
        h = w.tile(4, 9, 6, "elevation")
        assert h.dtype == np.float32 and h.shape == (256, 256)
        assert np.isfinite(h).all() and np.abs(h).max() < 12_000
        np.testing.assert_array_equal(w.tile(4, 9, 6, "elevation"), h)  # now read from the store
        tiles = {name: w.tile(4, 9, 6, name) for name in aerialsynth.LAYERS}
        for name, a in tiles.items():
            info = aerialsynth.LAYERS[name]
            assert a.dtype == info.dtype and a.shape == info.shape
        assert tiles["landcover"].max() < 18
        n = tiles["normal"].astype(np.float32) / 127
        assert np.allclose(np.linalg.norm(n, axis=-1), 1, atol=0.03) and (n[..., 2] > 0).all()
        assert tiles["rgb"].flags.writeable  # an array of its own
        tiles["rgb"][0, 0] = 0
    assert w.closed
    # one handle per file and process: a second one is refused while the first is open
    with aerialsynth.World(path, config=config, seed=5):
        with pytest.raises(RuntimeError, match="already open"):
            aerialsynth.World(tmp_path / "store" / ".." / "store" / "world.h5", config=config, seed=5)
    # reopened: the same world and tile; another seed is another world
    with aerialsynth.World(path, config=config, seed=5) as w:
        np.testing.assert_array_equal(w.tile(4, 9, 6, "elevation"), h)
    with pytest.raises(RuntimeError, match=r"world\.seed \(5 → 6\)"):
        aerialsynth.World(path, config=config, seed=6)


def test_bad_arguments(tmp_path, config):
    w = aerialsynth.World(str(tmp_path / "w.h5"), config=str(config))
    for z, x, y in [(9, 0, 0), (3, 8, 0), (3, 0, 8), (-1, 0, 0), (3, -1, 0), (2**40, 0, 0)]:
        with pytest.raises(ValueError):
            w.tile(z, x, y, "rgb")
    with pytest.raises(ValueError, match="unknown layer"):
        w.tile(3, 0, 0, "height")
    # no silent truncation: integers only (numpy integers too)
    for z, x, y in [(3.5, 0, 0), (3, 0.0, 0), (3, 0, "1"), (True, 0, 0), (3, np.float32(1), 0)]:
        with pytest.raises(TypeError, match="must be an integer"):
            w.tile(z, x, y, "landcover")
    np.testing.assert_array_equal(w.tile(np.int64(3), np.uint8(1), np.int32(2), "landcover"), w.tile(3, 1, 2, "landcover"))
    with pytest.raises(TypeError):
        w.prefetch((44.0, 9.0, 46.0, 11.0), (0, 2.5))
    w.close()
    w.close()
    with pytest.raises(ValueError, match="closed"):
        w.tile(3, 0, 0, "rgb")
    with pytest.raises(FileNotFoundError):
        aerialsynth.World(tmp_path / "x.h5", config=tmp_path / "missing.yaml")
    with pytest.raises(ValueError):
        aerialsynth.World(tmp_path / "x.h5", seed=-1)
    with pytest.raises(TypeError):
        aerialsynth.World(tmp_path / "x.h5", seed=1.5)
    # not a tile store: the reason first, no HDF5 error stack
    (tmp_path / "notes.txt").write_text("hello")
    for p, why in [(tmp_path, "is a directory"), (tmp_path / "notes.txt", "is not an HDF5 file")]:
        with pytest.raises(RuntimeError, match=why) as e:
            aerialsynth.World(p)
        assert str(e.value).startswith(str(p)) and "\n" not in str(e.value)
    bad = tmp_path / "bad.yaml"
    bad.write_text("world: { no_such_setting: 1 }\n")
    with pytest.raises(RuntimeError, match="no_such_setting"):
        aerialsynth.World(tmp_path / "x.h5", config=bad)


def test_verbose(tmp_path, config, capfd):
    with aerialsynth.World(tmp_path / "w.h5", config=config, verbose=True) as w:
        assert w.verbose
        w.tile(2, 1, 1, "rgb")
        assert "generated tile 2/1/1" in capfd.readouterr().err
        w.verbose = False
        w.tile(2, 2, 1, "rgb")
        assert capfd.readouterr().err == ""
    with aerialsynth.World(tmp_path / "w.h5", config=config) as w:
        assert not w.verbose


def test_threads(tmp_path, config):
    """Tiles are made in parallel (the GIL is released) and match the ones made serially."""
    ids = [(3, x, 3) for x in range(4)]
    with aerialsynth.World(tmp_path / "a.h5", config=config) as w:
        serial = {i: w.tile(*i, "elevation") for i in ids}
    out, errors = {}, []
    with aerialsynth.World(tmp_path / "b.h5", config=config) as w:
        def work(i):
            try:
                out[i] = w.tile(*i, "elevation")
            except Exception as e:  # pragma: no cover
                errors.append(e)
        threads = [threading.Thread(target=work, args=(i,)) for i in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors
    for i in ids:
        np.testing.assert_array_equal(out[i], serial[i])


def test_tiles_batch(tmp_path, config):
    """World.tiles: the tiles of World.tile in one array; duplicates, stored and missing ones."""
    ids = [(3, 4, 2), (3, 5, 2), (3, 4, 2), (4, 9, 6), (2, 1, 1)]
    with aerialsynth.World(tmp_path / "one.h5", config=config) as w:
        w.tile(3, 4, 2, "rgb")  # stored before the batch
        one = {i: w.tile(*i, "elevation") for i in ids}
    with aerialsynth.World(tmp_path / "many.h5", config=config) as w:
        w.tile(3, 4, 2, "landcover")  # stored (and cached) before the batch
        a = w.tiles(ids, "elevation")
        assert a.dtype == np.float32 and a.shape == (5, 256, 256) and a.flags.writeable
        for k, i in enumerate(ids):
            np.testing.assert_array_equal(a[k], one[i])
        # an array of coordinates, default layer rgb
        rgb = w.tiles(np.array(ids, dtype=np.uint32))
        assert rgb.dtype == np.uint8 and rgb.shape == (5, 256, 256, 3)
        np.testing.assert_array_equal(rgb[1], w.tile(3, 5, 2, "rgb"))
        assert w.tiles([], "normal").shape == (0, 256, 256, 3)
        # a bad coordinate is refused before anything is made
        for bad in [[(3, 6, 2), (3, 8, 0)], [(9, 0, 0)], [(3, -1, 0)], [(3, 0, 2**32)]]:
            with pytest.raises(ValueError):
                w.tiles(bad, "rgb")
        with pytest.raises(ValueError, match="shape"):
            w.tiles([3, 6, 2])
        with pytest.raises(TypeError):
            w.tiles([(3.5, 6, 2)])
        with pytest.raises(ValueError, match="unknown layer"):
            w.tiles(ids, "height")
    with aerialsynth.World(tmp_path / "many.h5", config=config, cache_mb=0) as w:
        np.testing.assert_array_equal(w.tiles([(3, 6, 2)], "landcover")[0], w.tile(3, 6, 2, "landcover"))
        assert w.cache_info().entries == 0


def test_cache(tmp_path, config):
    with aerialsynth.World(tmp_path / "w.h5", config=config, cache_mb=1) as w:
        assert w.cache_mb == 1
        a = w.tile(3, 4, 2, "rgb")  # generated: its layers go to the cache (as far as they fit)
        info = w.cache_info()
        assert info.size_mb == 1 and 0 < info.bytes <= 2**20
        b = w.tile(3, 4, 2, "rgb")
        np.testing.assert_array_equal(a, b)
        assert w.cache_info().hits == info.hits + 1
        b[0, 0] = 0  # arrays are copies: the cache is not changed through them
        np.testing.assert_array_equal(w.tile(3, 4, 2, "rgb"), a)
        w.cache_mb = 0
        assert w.cache_info() == (0, 0, 0, w.cache_info().hits, w.cache_info().misses)
        np.testing.assert_array_equal(w.tile(3, 4, 2, "rgb"), a)  # from the store
        assert w.cache_info().entries == 0
        w.cache_mb = 64
        w.tiles([(3, x, 2) for x in range(4)], "rgb")
        assert w.cache_info().entries >= 4
        with pytest.raises(ValueError):
            w.cache_mb = -1
    with pytest.raises(ValueError, match="closed"):
        w.cache_info()


def test_prefetch(tmp_path, config):
    with aerialsynth.World(tmp_path / "w.h5", config=config) as w:
        n = w.prefetch((44.0, 9.0, 46.0, 11.0), (0, 4))
        assert n == 5  # one tile per zoom
        assert w.prefetch((44.0, 9.0, 46.0, 11.0), 4) == 0  # stored now
        assert w.cache_info().entries == 0  # not cached
        with pytest.raises(ValueError):
            w.prefetch((44.0, 9.0, 46.0, 11.0), (0, 9))  # above max_zoom
        with pytest.raises(ValueError):
            w.prefetch((46.0, 9.0, 44.0, 11.0), 3)  # lat_min > lat_max
        with pytest.raises(ValueError):
            w.prefetch((44.0, 9.0, 95.0, 11.0), 3)
    big = tmp_path / "big.yaml"
    big.write_text("world:\n  tile_supersample: 1\n")
    with aerialsynth.World(tmp_path / "big.h5", config=big) as w:
        with pytest.raises(ValueError, match="at most"):
            w.prefetch((-80, -180, 80, 180), (0, 12))  # millions of tiles: refused
