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
    w.close()
    w.close()
    with pytest.raises(ValueError, match="closed"):
        w.tile(3, 0, 0, "rgb")
    with pytest.raises(FileNotFoundError):
        aerialsynth.World(tmp_path / "x.h5", config=tmp_path / "missing.yaml")
    with pytest.raises(ValueError):
        aerialsynth.World(tmp_path / "x.h5", seed=-1)
    bad = tmp_path / "bad.yaml"
    bad.write_text("world: { no_such_setting: 1 }\n")
    with pytest.raises(RuntimeError, match="no_such_setting"):
        aerialsynth.World(tmp_path / "x.h5", config=bad)


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
