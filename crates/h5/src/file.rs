use std::ffi::CString;
use std::ops::Deref;
use std::path::Path;

use hdf5_sys as sys;
use sys::h5f::{H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_TRUNC};
use sys::h5i::hid_t;
use sys::h5p::H5P_DEFAULT;

use crate::error::{Error, Result};
use crate::group::Group;
use crate::raw::{check, lock, Handle};

/// An open HDF5 file. Dereferences to its root [`Group`], so all group
/// operations (and attributes, which then live on the root group) are
/// available directly on `File`.
///
/// Dropping a `File` releases its handle; objects opened from it (groups,
/// datasets) keep the underlying file open until they are dropped too
/// (HDF5's default "weak" close degree).
#[derive(Clone, Debug)]
pub struct File {
    /// A `Group` wrapping the *file* id: HDF5 accepts a file id anywhere a
    /// location id is expected and resolves it to the root group.
    root: Group,
}

fn path_cstr(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("non UTF-8 path {path:?}")))?
        .as_bytes()
        .to_vec();
    CString::new(bytes).map_err(|_| Error::InvalidArgument(format!("path contains NUL: {path:?}")))
}

impl File {
    fn open_with(path: &Path, flags: u32, op: &str) -> Result<File> {
        let c = path_cstr(path)?;
        let _g = lock();
        // SAFETY: under the lock; `c` is a valid NUL-terminated string.
        let id = unsafe {
            if flags & H5F_ACC_TRUNC != 0 {
                sys::h5f::H5Fcreate(c.as_ptr(), flags, H5P_DEFAULT, H5P_DEFAULT)
            } else {
                sys::h5f::H5Fopen(c.as_ptr(), flags, H5P_DEFAULT)
            }
        };
        let h = Handle::check(id, op, || path.display().to_string())?;
        Ok(File { root: Group { h } })
    }

    /// Create a new file, truncating any existing file at `path`.
    pub fn create(path: impl AsRef<Path>) -> Result<File> {
        Self::open_with(path.as_ref(), H5F_ACC_TRUNC, "H5Fcreate")
    }

    /// Open an existing file read-only.
    pub fn open(path: impl AsRef<Path>) -> Result<File> {
        Self::open_with(path.as_ref(), H5F_ACC_RDONLY, "H5Fopen(read-only)")
    }

    /// Open an existing file for reading and writing.
    pub fn open_rw(path: impl AsRef<Path>) -> Result<File> {
        Self::open_with(path.as_ref(), H5F_ACC_RDWR, "H5Fopen(read-write)")
    }

    /// Flush all buffers of this file to disk.
    pub fn flush(&self) -> Result<()> {
        let _g = lock();
        // SAFETY: valid file id under the lock.
        let r = unsafe { sys::h5f::H5Fflush(self.id(), sys::h5f::H5F_scope_t::H5F_SCOPE_LOCAL) };
        check(r, "H5Fflush", || crate::raw::describe(self.id()))?;
        Ok(())
    }

    /// Open the root group "/" as a standalone [`Group`] handle.
    /// (Not normally needed: `File` derefs to its root group.)
    pub fn root(&self) -> Result<Group> {
        self.root.group("/")
    }

    /// The file name as HDF5 knows it.
    pub fn filename(&self) -> String {
        crate::raw::file_name(self.id())
    }

    /// Raw HDF5 file id (for use with [`crate::sys`] under [`crate::lock`]).
    pub fn id(&self) -> hid_t {
        self.root.id()
    }
}

impl Deref for File {
    type Target = Group;
    fn deref(&self) -> &Group {
        &self.root
    }
}
