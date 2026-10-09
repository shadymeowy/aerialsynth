//! A small, safe wrapper over the raw HDF5 C bindings (`hdf5-sys`).
//!
//! Scope: files, groups, numeric N-d datasets (chunked / resizable /
//! deflate+shuffle), hyperslab I/O, direct (pre-filtered) chunk I/O and
//! numeric / string attributes.
//!
//! # Thread safety
//!
//! The bundled HDF5 is not built thread-safe, so every call into the
//! library is serialised through one global reentrant lock
//! (`hdf5_sys::LOCK`, see [`lock`]). All handle types are therefore `Send +
//! Sync`, but HDF5 work never runs in parallel. To parallelise compression,
//! do it outside the lock and use [`Dataset::write_chunk_raw`] /
//! [`Dataset::read_chunk_raw`].
//!
//! All `unsafe` code lives in this crate's modules next to the FFI calls it
//! guards; ids are owned by an internal RAII handle that decrements the
//! HDF5 reference count on drop.

pub use hdf5_sys as sys;

mod attr;
mod dataset;
mod error;
mod file;
mod group;
mod raw;
mod types;

pub use attr::Attrs;
pub use dataset::{Dataset, DatasetBuilder};
pub use error::{Error, Result};
pub use file::File;
pub use group::Group;
pub use raw::lock;
pub use sys::h5i::hid_t;
pub use types::H5Type;

// Compile-time check that the handle types are `Send + Sync`.
const _: fn() = || {
    fn f<T: Send + Sync>() {}
    f::<File>();
    f::<Group>();
    f::<Dataset>();
    f::<Error>();
};
