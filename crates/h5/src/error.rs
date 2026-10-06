/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An HDF5 library call failed. `stack` is the HDF5 error stack
    /// (outermost API function first), `target` is `file:/object/path`
    /// (plus the child name for operations on a named member).
    #[error("HDF5 {op} failed on {target}: {stack}")]
    Hdf5 { op: String, target: String, stack: String },
    /// The arguments were rejected before calling into HDF5 (rank mismatch,
    /// wrong buffer length, out-of-bounds selection, NUL in a name, ...).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}
