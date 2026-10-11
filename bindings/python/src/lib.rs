//! `aerialsynth._native`: the extension module behind the `aerialsynth` Python package
//! (`python/aerialsynth/__init__.py` wraps it and returns numpy arrays).
//!
//! Tiles and images are returned as `bytearray`s (raw little-endian pixels) so that the module
//! needs only the stable ABI and no numpy C API: the Python layer views them with
//! `np.frombuffer` (writable, no copy). The implementation is `aerialsynth-core`, shared with the
//! C API.

use aerialsynth_core as core;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyPermissionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

fn to_py(e: core::Error) -> PyErr {
    let msg = e.to_string();
    match e {
        core::Error::InvalidArgument(_) => PyValueError::new_err(msg),
        core::Error::Io { source, .. } => match source.kind() {
            std::io::ErrorKind::NotFound => PyFileNotFoundError::new_err(msg),
            std::io::ErrorKind::PermissionDenied => PyPermissionError::new_err(msg),
            _ => PyOSError::new_err(msg),
        },
        core::Error::Failed(_) => PyRuntimeError::new_err(msg),
    }
}

fn coord(v: i64, what: &str) -> PyResult<u32> {
    u32::try_from(v).map_err(|_| PyValueError::new_err(format!("{what} = {v} is out of range")))
}

/// A slot that is emptied on close. The value is an `Arc`, so that work in progress (without
/// the GIL) keeps it alive while another thread closes it.
type Slot<T> = Mutex<Option<Arc<T>>>;

fn get<T>(slot: &Slot<T>, what: &str) -> PyResult<Arc<T>> {
    slot.lock().unwrap_or_else(|e| e.into_inner()).clone().ok_or_else(|| PyValueError::new_err(format!("the {what} is closed")))
}

fn take<T>(slot: &Slot<T>) -> Option<Arc<T>> {
    slot.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// A tile store opened for one world (see `aerialsynth.World`).
#[pyclass(frozen, module = "aerialsynth._native")]
struct World {
    /// `None` once closed.
    inner: Slot<core::World>,
    /// The cameras made from this world: closed with it (a camera keeps the store open).
    cameras: Mutex<Vec<Weak<Slot<core::Camera>>>>,
    path: PathBuf,
    seed: u64,
    max_zoom: u32,
}

impl World {
    fn get(&self) -> PyResult<Arc<core::World>> {
        get(&self.inner, "world")
    }
}

#[pymethods]
impl World {
    #[new]
    #[pyo3(signature = (tiles_file, config=None, seed=None, cache_mb=core::DEFAULT_CACHE_MB))]
    fn new(py: Python<'_>, tiles_file: PathBuf, config: Option<PathBuf>, seed: Option<u64>, cache_mb: usize) -> PyResult<Self> {
        let w = py.detach(|| core::World::open(&tiles_file, config.as_deref(), seed)).map_err(to_py)?;
        w.set_cache_mb(cache_mb);
        Ok(World {
            path: w.path().to_path_buf(),
            seed: w.seed(),
            max_zoom: w.max_zoom(),
            inner: Mutex::new(Some(Arc::new(w))),
            cameras: Mutex::new(Vec::new()),
        })
    }

    /// A camera over this world: a pinhole camera (`width`, `height`, `hfov`; optional principal
    /// point `cx`, `cy`, `mount` "forward" | "nadir") with the render settings of `config`, or a
    /// camera of the scenario `config` (`camera`: HDF5 path or index, None = the first).
    #[pyo3(signature = (width=None, height=None, hfov=None, cx=None, cy=None, mount="forward", config=None, camera=None, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn camera(
        &self,
        py: Python<'_>,
        width: Option<u32>,
        height: Option<u32>,
        hfov: Option<f64>,
        cx: Option<f64>,
        cy: Option<f64>,
        mount: &str,
        config: Option<PathBuf>,
        camera: Option<String>,
        backend: Option<&str>,
    ) -> PyResult<Camera> {
        let w = self.get()?;
        let backend = backend.map(core::Backend::from_name).transpose().map_err(to_py)?;
        let mount = core::Mount::from_name(mount).map_err(to_py)?;
        let def = match (width, height, hfov) {
            (Some(width), Some(height), Some(hfov_deg)) => {
                if camera.is_some() {
                    return Err(PyValueError::new_err("camera selects a camera of the scenario: not with width, height and hfov"));
                }
                let principal_point = match (cx, cy) {
                    (Some(x), Some(y)) => Some((x, y)),
                    (None, None) => None,
                    _ => return Err(PyValueError::new_err("give both cx and cy (or neither)")),
                };
                core::CameraDef::Pinhole(core::Pinhole { width, height, hfov_deg, principal_point, mount })
            }
            (None, None, None) => {
                if cx.is_some() || cy.is_some() || mount != core::Mount::Forward {
                    return Err(PyValueError::new_err("cx, cy and mount are settings of a pinhole camera (width, height, hfov)"));
                }
                core::CameraDef::Scenario(camera)
            }
            _ => return Err(PyValueError::new_err("a pinhole camera needs width, height and hfov")),
        };
        let cam = py.detach(|| core::Camera::new(w, config.as_deref(), def, backend)).map_err(to_py)?;
        let info = CameraInfo::of(&cam);
        let slot = Arc::new(Mutex::new(Some(Arc::new(cam))));
        {
            let mut cams = self.cameras.lock().unwrap_or_else(|e| e.into_inner());
            cams.retain(|c| c.strong_count() > 0);
            cams.push(Arc::downgrade(&slot));
        }
        // (`close` empties `inner` before it closes the listed cameras: a world closed while this
        // camera was made is seen here, else the camera is in the list it closes)
        if self.inner.lock().unwrap_or_else(|e| e.into_inner()).is_none() {
            let c = take(&slot);
            py.detach(move || drop(c));
            return Err(PyValueError::new_err("the world is closed"));
        }
        Ok(Camera { inner: slot, info })
    }

    /// DSM height (m above the WGS84 ellipsoid) at a point (degrees).
    fn surface_height(&self, py: Python<'_>, lat: f64, lon: f64) -> PyResult<f64> {
        let w = self.get()?;
        py.detach(move || w.surface_height(lat, lon)).map_err(to_py)
    }

    /// Raw little-endian pixels of a layer of tile z/x/y (generated and stored if missing).
    fn tile<'py>(&self, py: Python<'py>, z: i64, x: i64, y: i64, layer: &str) -> PyResult<Bound<'py, PyByteArray>> {
        let l = core::layer_by_name(layer).map_err(to_py)?;
        let (z, x, y) = (coord(z, "z")?, coord(x, "x")?, coord(y, "y")?);
        let w = self.get()?;
        filled(py, l.tile_bytes(), |buf| w.tile_into(z, x, y, l, buf))
    }

    /// Raw pixels of a layer of n tiles, one after the other: `zxy` is n × 3 little-endian
    /// uint32 (z, x, y of each tile).
    fn tiles<'py>(&self, py: Python<'py>, zxy: &[u8], layer: &str) -> PyResult<Bound<'py, PyByteArray>> {
        let l = core::layer_by_name(layer).map_err(to_py)?;
        if !zxy.len().is_multiple_of(12) {
            return Err(PyValueError::new_err("zxy: n × 3 uint32 expected"));
        }
        let coords: Vec<[u32; 3]> =
            zxy.as_chunks::<12>().0.iter().map(|c| [0, 4, 8].map(|o| u32::from_le_bytes([c[o], c[o + 1], c[o + 2], c[o + 3]]))).collect();
        let size = coords.len().checked_mul(l.tile_bytes()).ok_or_else(|| PyValueError::new_err("too many tiles"))?;
        let w = self.get()?;
        filled(py, size, |buf| w.tiles_into(&coords, l, buf))
    }

    /// Generate and store the missing tiles of a latitude / longitude box (degrees) at zooms
    /// z_min..=z_max; the number generated.
    #[allow(clippy::too_many_arguments)]
    fn prefetch(&self, py: Python<'_>, lat_min: f64, lon_min: f64, lat_max: f64, lon_max: f64, z_min: i64, z_max: i64) -> PyResult<usize> {
        let (z0, z1) = (coord(z_min, "z_min")?, coord(z_max, "z_max")?);
        let w = self.get()?;
        py.detach(move || w.prefetch([lat_min, lon_min, lat_max, lon_max], z0, z1)).map_err(to_py)
    }

    /// Set the size of the decoded-tile cache in MiB (0: off).
    fn set_cache_mb(&self, mb: usize) -> PyResult<()> {
        self.get()?.set_cache_mb(mb);
        Ok(())
    }

    /// Report tile generation on stderr.
    #[getter]
    fn verbose(&self) -> PyResult<bool> {
        Ok(self.get()?.verbose())
    }

    #[setter]
    fn set_verbose(&self, on: bool) -> PyResult<()> {
        self.get()?.set_verbose(on);
        Ok(())
    }

    /// `(size in MiB, bytes held, layers of tiles held, hits, misses)` of the tile cache.
    fn cache_info(&self) -> PyResult<(usize, usize, usize, u64, u64)> {
        let s = self.get()?.cache_stats();
        Ok((s.capacity >> 20, s.bytes, s.entries, s.hits, s.misses))
    }

    /// Close the store and the world's cameras (idempotent). A tile or frame being made in
    /// another thread finishes first.
    fn close(&self, py: Python<'_>) {
        let w = take(&self.inner);
        let cams: Vec<_> = std::mem::take(&mut *self.cameras.lock().unwrap_or_else(|e| e.into_inner()));
        let cams: Vec<_> = cams.iter().filter_map(Weak::upgrade).filter_map(|s| take(&s)).collect();
        py.detach(move || {
            drop(cams);
            drop(w)
        });
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).is_none()
    }
    #[getter]
    fn path(&self) -> PathBuf {
        self.path.clone()
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }
    #[getter]
    fn max_zoom(&self) -> u32 {
        self.max_zoom
    }
}

/// What does not change about a camera (readable after it is closed).
#[derive(Clone)]
struct CameraInfo {
    width: u32,
    height: u32,
    backend: &'static str,
    supersample: u32,
    path: String,
    model: String,
    intrinsics: Vec<f64>,
    distortion: Vec<f64>,
    depth_range: bool,
}

impl CameraInfo {
    fn of(c: &core::Camera) -> Self {
        let (model, intrinsics, distortion) = c.model();
        CameraInfo {
            width: c.width(),
            height: c.height(),
            backend: c.backend().name(),
            supersample: c.supersample(),
            path: c.path().to_string(),
            model: model.to_string(),
            intrinsics: intrinsics.to_vec(),
            distortion: distortion.to_vec(),
            depth_range: c.depth_kind_range(),
        }
    }
}

/// `(rgb, depth, landcover, exposure (time, gain, ev) | None, position_ecef, r_ecef_cam (row-major
/// 9), unix_time, sun_azimuth_deg, sun_elevation_deg)` of a frame; images as raw bytes.
type FrameTuple<'py> = (
    Option<Bound<'py, PyByteArray>>,
    Option<Bound<'py, PyByteArray>>,
    Option<Bound<'py, PyByteArray>>,
    Option<(f64, f64, f64)>,
    [f64; 3],
    [f64; 9],
    f64,
    f64,
    f64,
);

/// A camera over a world (see `aerialsynth.Camera`).
#[pyclass(frozen, module = "aerialsynth._native")]
struct Camera {
    /// `None` once closed (by itself or by its world).
    inner: Arc<Slot<core::Camera>>,
    info: CameraInfo,
}

#[pymethods]
impl Camera {
    /// Render a frame: images as raw bytes (see `FrameTuple`).
    #[pyo3(signature = (lat, lon, height, roll, pitch, yaw, time, rgb, depth, landcover))]
    #[allow(clippy::too_many_arguments)]
    fn render<'py>(
        &self,
        py: Python<'py>,
        lat: f64,
        lon: f64,
        height: f64,
        roll: f64,
        pitch: f64,
        yaw: f64,
        time: Option<f64>,
        rgb: bool,
        depth: bool,
        landcover: bool,
    ) -> PyResult<FrameTuple<'py>> {
        let cam = get(&self.inner, "camera")?;
        let pose = core::Pose { lat_deg: lat, lon_deg: lon, height_m: height, roll_deg: roll, pitch_deg: pitch, yaw_deg: yaw };
        let f = py.detach(move || cam.render(&pose, time, core::Outputs { rgb, depth, landcover })).map_err(to_py)?;
        let depth = f.depth.map(|d| {
            let b: Vec<u8> = d.iter().flat_map(|v| v.to_le_bytes()).collect();
            PyByteArray::new(py, &b)
        });
        let r = f.r_ecef_cam;
        Ok((
            f.rgb.map(|v| PyByteArray::new(py, &v)),
            depth,
            f.landcover.map(|v| PyByteArray::new(py, &v)),
            f.exposure.map(|e| (e.time, e.gain, e.ev)),
            f.position_ecef,
            [r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]],
            f.unix_time,
            f.sun_azimuth_deg,
            f.sun_elevation_deg,
        ))
    }

    /// Close the camera (idempotent). A frame being rendered in another thread finishes first.
    fn close(&self, py: Python<'_>) {
        let c = take(&self.inner);
        py.detach(move || drop(c));
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).is_none()
    }
    #[getter]
    fn width(&self) -> u32 {
        self.info.width
    }
    #[getter]
    fn height(&self) -> u32 {
        self.info.height
    }
    #[getter]
    fn backend(&self) -> &'static str {
        self.info.backend
    }
    #[getter]
    fn supersample(&self) -> u32 {
        self.info.supersample
    }
    #[getter]
    fn path(&self) -> String {
        self.info.path.clone()
    }
    #[getter]
    fn model(&self) -> String {
        self.info.model.clone()
    }
    #[getter]
    fn intrinsics(&self) -> Vec<f64> {
        self.info.intrinsics.clone()
    }
    #[getter]
    fn distortion(&self) -> Vec<f64> {
        self.info.distortion.clone()
    }
    #[getter]
    fn depth_is_range(&self) -> bool {
        self.info.depth_range
    }
}

/// A new `bytearray` of `len` bytes filled by `fill` without the GIL.
fn filled<'py>(py: Python<'py>, len: usize, fill: impl FnOnce(&mut [u8]) -> core::Result<()> + Send) -> PyResult<Bound<'py, PyByteArray>> {
    let b = PyByteArray::new_with(py, len, |_| Ok(()))?;
    // SAFETY: the bytearray was just made and no Python code holds a reference to it (it is
    // returned only afterwards), so nothing else accesses its buffer while it is filled without
    // the GIL, and it is not resized.
    let buf = unsafe { b.as_bytes_mut() };
    py.detach(move || fill(buf)).map_err(to_py)?;
    Ok(b)
}

/// `[(name, numpy dtype str, channels, bytes per tile, description)]` of every layer.
#[pyfunction]
fn layers() -> Vec<(&'static str, &'static str, usize, usize, &'static str)> {
    core::layers().iter().map(|l| (l.name, l.dtype.numpy(), l.channels, l.size, l.description)).collect()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    m.add_class::<Camera>()?;
    m.add_function(wrap_pyfunction!(layers, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TILE_SIZE", core::TILE_PX)?;
    m.add("MAX_ZOOM", core::MAX_ZOOM)?;
    m.add("MAX_IMAGE_SIZE", core::MAX_IMAGE_SIZE)?;
    m.add("DEFAULT_MAX_ZOOM", core::DEFAULT_MAX_ZOOM)?;
    m.add("DEFAULT_CACHE_MB", core::DEFAULT_CACHE_MB)?;
    m.add("MAX_PREFETCH_TILES", core::MAX_PREFETCH_TILES)?;
    m.add("GENERATOR_VERSION", core::GENERATOR_VERSION)?;
    Ok(())
}
