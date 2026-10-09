"""Tests of camera rendering (on the CPU: one world for the module, its tiles are made once)."""

import datetime as dt
import math
import threading

import numpy as np
import pytest

import aerialsynth

LAT, LON = 39.9, 32.8
#: 2026-06-21T08:30:00Z
T = 1_782_030_600.0


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("render")
    cfg = d / "world.yaml"
    cfg.write_text("world:\n  tile_supersample: 1\ntiles:\n  max_zoom: 7\n")
    with aerialsynth.World(d / "world.h5", config=cfg, seed=3) as w:
        yield w


@pytest.fixture(scope="module")
def ground(world):
    return world.surface_height(LAT, LON)


@pytest.fixture
def cam(world):
    with world.camera(64, 48, 70, backend="cpu") as c:
        yield c


def test_frame(world, cam, ground):
    assert (cam.width, cam.height, cam.shape, cam.backend, cam.supersample) == (64, 48, (48, 64), "cpu", 3)
    assert (cam.path, cam.model, len(cam.intrinsics), cam.distortion) == ("/cam0", "pinhole", 4, (0.0,) * 4)
    fx = 32 / math.tan(math.radians(35))
    np.testing.assert_allclose(cam.K, [[fx, 0, 31.5], [0, fx, 23.5], [0, 0, 1]])
    assert not cam.depth_is_range and "pinhole 64x48, cpu" in repr(cam)
    assert math.isfinite(ground) and abs(ground) < 9000
    f = cam.render(LAT, LON, ground + 3000, pitch=-60, time=T, depth=True, landcover=True)
    assert isinstance(f, aerialsynth.Frame)
    assert f.rgb.dtype == np.uint8 and f.rgb.shape == (48, 64, 3) and f.rgb.flags.writeable
    assert f.depth.dtype == np.float32 and f.depth.shape == (48, 64)
    assert f.landcover.dtype == np.uint8 and f.landcover.shape == (48, 64)
    # 60 degrees down from 3 km: terrain everywhere, a few km away
    assert np.isfinite(f.depth).all() and (f.depth > 500).all() and (f.depth < 20_000).all()
    assert (f.landcover < 18).all()
    assert 20 < f.rgb.mean() < 235
    t, gain, ev = f.exposure
    assert t > 0 and gain >= 1
    assert f.time == T and f.datetime == dt.datetime(2026, 6, 21, 8, 30, tzinfo=dt.timezone.utc)
    assert f.sun_elevation > 30
    # the camera pose: the optical axis (third column) points 60 degrees down
    assert f.position_ecef.shape == (3,) and f.r_ecef_cam.shape == (3, 3)
    np.testing.assert_allclose(f.r_ecef_cam @ f.r_ecef_cam.T, np.eye(3), atol=1e-12)
    up = f.position_ecef / np.linalg.norm(f.position_ecef)
    assert abs(f.r_ecef_cam[:, 2] @ up + math.sin(math.radians(60))) < 0.01
    # deterministic; only the images asked for
    g = cam.render(LAT, LON, ground + 3000, pitch=-60, time=T, depth=True, landcover=True)
    np.testing.assert_array_equal(g.rgb, f.rgb)
    np.testing.assert_array_equal(g.depth, f.depth)
    d = cam.render(LAT, LON, ground + 3000, pitch=-60, time=T, rgb=False, depth=True)
    assert d.rgb is None and d.landcover is None and d.exposure is None
    np.testing.assert_array_equal(d.depth, f.depth)


def test_sky_and_time(cam, ground):
    up = dict(lat=LAT, lon=LON, height=ground + 3000, pitch=60)  # looking up: sky only
    f = cam.render(**up, time=T, depth=True, landcover=True)
    assert np.isposinf(f.depth).all() and (f.landcover == aerialsynth.SKY).all()
    # the same instant in every form
    for t in [T, int(T), dt.datetime(2026, 6, 21, 8, 30, tzinfo=dt.timezone.utc), dt.datetime(2026, 6, 21, 8, 30),
              dt.datetime(2026, 6, 21, 11, 30, tzinfo=dt.timezone(dt.timedelta(hours=3))), "2026-06-21T08:30:00Z",
              "2026-06-21T11:30:00+03:00", "2026-06-21 08:30:00", np.float64(T)]:
        g = cam.render(**up, time=t)
        assert g.time == T, t
        np.testing.assert_array_equal(g.rgb, f.rgb)
    # night (local midnight) needs a much longer exposure
    night = cam.render(**up, time="2026-06-21T22:00:00Z")
    assert night.sun_elevation < -10 and night.exposure[2] > f.exposure[2] + 5
    # no time: the scenario's lighting (by default a fixed sun at 52 degrees elevation)
    assert cam.render(**up).sun_elevation == pytest.approx(52)


def test_mounts(world, ground):
    """A nadir camera looks straight down when level; a forward one along the heading."""
    with world.camera(32, 24, 60, mount="nadir", backend="cpu") as nadir:
        d = nadir.render(LAT, LON, ground + 3000, yaw=40, rgb=False, depth=True).depth
        assert abs(d[12, 16] - 3000) < 800
    with world.camera(32, 24, 60, cx=10, cy=5, backend="cpu") as c:
        assert c.intrinsics[2:] == (10, 5)
        f = c.render(LAT, LON, ground + 3000, pitch=-90, yaw=0, rgb=False, depth=True)
        # the principal point's ray is the optical axis: straight down
        assert abs(f.depth[5, 10] - 3000) < 800


def test_scenario_cameras(world, tmp_path, ground):
    scn = tmp_path / "scenario.yaml"
    scn.write_text(
        "render: { supersample: 1, lighting: { mode: clock, date: 2026-03-20, time_utc: '09:49:00' } }\n"
        "cameras:\n"
        "  - path: /left\n"
        "    intrinsics: { model: pinhole, width: 48, height: 32, intrinsics: [40, 40, 23.5, 15.5] }\n"
        "    rgb: {}\n"
        "  - path: /down\n"
        "    intrinsics: { model: kannala_brandt, width: 40, height: 40, intrinsics: [15, 15, 19.5, 19.5], distortion: [0, 0, 0, 0] }\n"
        "    depth: { kind: range }\n"
    )
    left = world.camera(config=scn, backend="cpu")
    assert (left.path, left.shape, left.supersample) == ("/left", (32, 48), 1)
    for sel in ["/down", 1]:
        down = world.camera(config=scn, camera=sel, backend="cpu")
        assert (down.path, down.model, down.depth_is_range, down.K) == ("/down", "kannala_brandt", True, None)
    f = down.render(LAT, LON, ground + 3000, rgb=False, depth=True)
    # the scenario's clock: about local noon at the equinox, the sun at 90 - latitude
    assert f.sun_elevation == pytest.approx(90 - LAT, abs=1.5)
    assert abs(f.depth[20, 20] - 3000) < 800 and f.depth[20, 1] > 2 * f.depth[20, 20]
    # a pinhole camera with the scenario's render settings
    assert world.camera(16, 16, 50, config=scn, backend="cpu").supersample == 1
    with pytest.raises(ValueError, match="no camera /up"):
        world.camera(config=scn, camera="/up")
    with pytest.raises(ValueError, match="out of range"):
        world.camera(config=scn, camera=2)
    with pytest.raises(FileNotFoundError):
        world.camera(config=tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("render: { no_such_setting: 1 }\n")
    with pytest.raises(RuntimeError, match="no_such_setting"):
        world.camera(config=bad)


def test_bad_arguments(world, cam, ground):
    ok = dict(lat=LAT, lon=LON, height=ground + 3000)
    for bad in [dict(lat=91), dict(lat=float("nan")), dict(lon=float("inf")), dict(height=1e9), dict(roll=float("nan"))]:
        with pytest.raises(ValueError):
            cam.render(**{**ok, **bad})
    with pytest.raises(ValueError, match="ISO 8601"):
        cam.render(**ok, time="noon")
    with pytest.raises(TypeError):
        cam.render(**ok, time=[1, 2])
    with pytest.raises(ValueError):
        cam.render(**ok, time=float("nan"))
    for kw in [dict(width=64, height=48), dict(width=0, height=48, hfov=60), dict(width=64, height=48, hfov=180),
               dict(width=64, height=48, hfov=60, backend="vulkan"), dict(width=64, height=48, hfov=60, mount="up"),
               dict(width=64, height=48, hfov=60, cx=3), dict(mount="nadir"), dict(width=64, height=48, hfov=60, camera="/cam0"),
               dict(width=-1, height=48, hfov=60), dict(width=aerialsynth.MAX_IMAGE_SIZE + 1, height=48, hfov=60)]:
        with pytest.raises(ValueError):
            world.camera(**kw)
    with pytest.raises(TypeError):
        world.camera(64.5, 48, 60)


def test_close(tmp_path):
    cfg = tmp_path / "world.yaml"
    cfg.write_text("world:\n  tile_supersample: 1\ntiles:\n  max_zoom: 4\n")
    w = aerialsynth.World(tmp_path / "w.h5", config=cfg)
    a, b = w.camera(16, 12, 60, backend="cpu"), w.camera(16, 12, 60, backend="cpu")
    b.close()
    b.close()
    assert b.closed and not a.closed
    with pytest.raises(ValueError, match="closed"):
        b.render(0, 0, 1000)
    # closing the world closes its cameras (and the store: it can be opened again)
    w.close()
    assert a.closed
    with pytest.raises(ValueError, match="closed"):
        a.render(0, 0, 1000)
    with pytest.raises(ValueError, match="closed"):
        w.camera(16, 12, 60)
    aerialsynth.World(tmp_path / "w.h5", config=cfg).close()


def test_threads(world, ground):
    """Cameras render in parallel (the GIL is released), the same frames as serially."""
    poses = [dict(lat=LAT, lon=LON, height=ground + 3000, pitch=-60, yaw=y, time=T) for y in (0, 90, 180)]
    cams = [world.camera(32, 24, 60, backend="cpu") for _ in poses]
    serial = [c.render(**p).rgb for c, p in zip(cams, poses)]
    out, errors = [None] * len(poses), []

    def work(i):
        try:
            out[i] = cams[i].render(**poses[i]).rgb
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(poses))]
    for t in threads:
        t.start()
    # one camera from several threads: serialized
    same = [cams[0].render(**poses[0]).rgb for _ in range(2)]
    for t in threads:
        t.join()
    assert not errors
    for a, b in zip(out, serial):
        np.testing.assert_array_equal(a, b)
    for a in same:
        np.testing.assert_array_equal(a, serial[0])
