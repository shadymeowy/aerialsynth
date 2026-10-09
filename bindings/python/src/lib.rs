//! `aerialsynth._native`: the extension module behind the `aerialsynth` Python package
//! (`python/aerialsynth/__init__.py` wraps it and returns numpy arrays).
//!
//! Tiles are returned as `bytearray`s (raw little-endian pixels) so that the module needs only
//! the stable ABI and no numpy C API: the Python layer views them with `np.frombuffer` (writable,
//! no copy). The implementation is `aerialsynth-core`, shared with the C API.

use aerialsynth_core as core;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyPermissionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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

/// A tile store opened for one world (see `aerialsynth.World`).
#[pyclass(frozen, module = "aerialsynth._native")]
struct World {
    /// `None` once closed. An `Arc`, so that a tile being made (without the GIL) keeps the store
    /// open while another thread closes the world.
    inner: Mutex<Option<Arc<core::World>>>,
    path: PathBuf,
    seed: u64,
    max_zoom: u32,
}

impl World {
    fn get(&self) -> PyResult<Arc<core::World>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clone().ok_or_else(|| PyValueError::new_err("the world is closed"))
    }
}

#[pymethods]
impl World {
    #[new]
    #[pyo3(signature = (tiles_file, config=None, seed=None))]
    fn new(py: Python<'_>, tiles_file: PathBuf, config: Option<PathBuf>, seed: Option<u64>) -> PyResult<Self> {
        let w = py.detach(|| core::World::open(&tiles_file, config.as_deref(), seed)).map_err(to_py)?;
        Ok(World { path: w.path().to_path_buf(), seed: w.seed(), max_zoom: w.max_zoom(), inner: Mutex::new(Some(Arc::new(w))) })
    }

    /// Raw little-endian pixels of a layer of tile z/x/y (generated and stored if missing).
    fn tile<'py>(&self, py: Python<'py>, z: i64, x: i64, y: i64, layer: &str) -> PyResult<Bound<'py, PyByteArray>> {
        let l = core::layer_by_name(layer).map_err(to_py)?;
        let (z, x, y) = (coord(z, "z")?, coord(x, "x")?, coord(y, "y")?);
        let w = self.get()?;
        let v = py.detach(move || w.tile(z, x, y, l)).map_err(to_py)?;
        Ok(PyByteArray::new(py, &v))
    }

    /// Close the store (idempotent). A tile being made in another thread finishes first.
    fn close(&self, py: Python<'_>) {
        let w = self.inner.lock().unwrap_or_else(|e| e.into_inner()).take();
        py.detach(move || drop(w));
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

/// `[(name, numpy dtype str, channels, bytes per tile, description)]` of every layer.
#[pyfunction]
fn layers() -> Vec<(&'static str, &'static str, usize, usize, &'static str)> {
    core::layers().iter().map(|l| (l.name, l.dtype.numpy(), l.channels, l.size, l.description)).collect()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    m.add_function(wrap_pyfunction!(layers, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TILE_SIZE", core::TILE_PX)?;
    m.add("MAX_ZOOM", core::MAX_ZOOM)?;
    m.add("DEFAULT_MAX_ZOOM", core::DEFAULT_MAX_ZOOM)?;
    m.add("GENERATOR_VERSION", core::GENERATOR_VERSION)?;
    Ok(())
}
