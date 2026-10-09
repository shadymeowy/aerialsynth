use std::ffi::{c_char, c_void, CStr};
use std::ptr;

use hdf5_sys as sys;
use sys::h5::herr_t;
use sys::h5i::hid_t;
use sys::h5p::H5P_DEFAULT;

use crate::dataset::{Dataset, DatasetBuilder};
use crate::error::Result;
use crate::raw::{check, clear_errors, cstr, describe, describe_child, lock, Handle};
use crate::types::H5Type;

/// An HDF5 group (or the root group of a [`crate::File`]).
#[derive(Clone, Debug)]
pub struct Group {
    pub(crate) h: Handle,
}

/// Split "a/b/c" (or "/a/b/c") into the cumulative prefixes
/// ["a", "a/b", "a/b/c"] (or ["/a", "/a/b", "/a/b/c"]).
fn path_prefixes(path: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    if path.starts_with('/') {
        cur.push('/');
    }
    for comp in path.split('/').filter(|c| !c.is_empty()) {
        if !cur.is_empty() && !cur.ends_with('/') {
            cur.push('/');
        }
        cur.push_str(comp);
        out.push(cur.clone());
    }
    out
}

unsafe extern "C" fn collect_link(_group: hid_t, name: *const c_char, _info: *const sys::h5l::H5L_info2_t, data: *mut c_void) -> herr_t {
    // SAFETY: `data` is the &mut Vec<String> passed to H5Literate2, `name` a
    // valid C string for the duration of the callback.
    unsafe {
        let v = &mut *(data as *mut Vec<String>);
        v.push(CStr::from_ptr(name).to_string_lossy().into_owned());
    }
    0
}

impl Group {
    /// Raw HDF5 id (for use with [`crate::sys`] under [`crate::lock`]).
    pub fn id(&self) -> hid_t {
        self.h.id()
    }

    /// Absolute path of this group inside its file.
    pub fn name(&self) -> String {
        crate::raw::obj_path(self.id())
    }

    /// Create a new child group. Fails if it already exists or if an
    /// intermediate group is missing (use [`Group::ensure_group`] for that).
    pub fn create_group(&self, name: &str) -> Result<Group> {
        let c = cstr(name)?;
        let _g = lock();
        // SAFETY: valid location id and C string, under the lock.
        let id = unsafe { sys::h5g::H5Gcreate2(self.id(), c.as_ptr(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) };
        Ok(Group { h: Handle::check(id, "H5Gcreate2", || describe_child(self.id(), name))? })
    }

    /// Open an existing group; `name` may be a relative ("a/b") or absolute
    /// ("/a/b") path.
    pub fn group(&self, name: &str) -> Result<Group> {
        let c = cstr(name)?;
        let _g = lock();
        // SAFETY: valid location id and C string, under the lock.
        let id = unsafe { sys::h5g::H5Gopen2(self.id(), c.as_ptr(), H5P_DEFAULT) };
        Ok(Group { h: Handle::check(id, "H5Gopen2", || describe_child(self.id(), name))? })
    }

    /// Open the group at `path`, creating it and any missing intermediate
    /// groups. Returns `self` (re-opened) for an empty path.
    pub fn ensure_group(&self, path: &str) -> Result<Group> {
        let _g = lock();
        let mut cur = if path.starts_with('/') { self.group("/")? } else { self.clone() };
        for comp in path.split('/').filter(|c| !c.is_empty()) {
            cur = if cur.exists(comp) { cur.group(comp)? } else { cur.create_group(comp)? };
        }
        Ok(cur)
    }

    /// Whether a link `name` exists. Nested paths are checked component by
    /// component, so a missing intermediate group yields `false` rather than
    /// an error. The empty path and "/" always exist.
    pub fn exists(&self, name: &str) -> bool {
        let _g = lock();
        for prefix in path_prefixes(name) {
            let Ok(c) = cstr(&prefix) else { return false };
            // SAFETY: valid location id and C string, under the lock.
            let r = unsafe { sys::h5l::H5Lexists(self.id(), c.as_ptr(), H5P_DEFAULT) };
            if r <= 0 {
                if r < 0 {
                    clear_errors();
                }
                return false;
            }
        }
        true
    }

    /// Names of the direct members (links) of this group, in name order.
    pub fn member_names(&self) -> Result<Vec<String>> {
        let _g = lock();
        let mut names: Vec<String> = Vec::new();
        // SAFETY: valid group id; the callback only pushes into `names`,
        // which outlives the call.
        let r = unsafe {
            sys::h5l::H5Literate2(
                self.id(),
                sys::h5::H5_index_t::H5_INDEX_NAME,
                sys::h5::H5_iter_order_t::H5_ITER_INC,
                ptr::null_mut(),
                Some(collect_link),
                &mut names as *mut Vec<String> as *mut c_void,
            )
        };
        check(r, "H5Literate2", || describe(self.id()))?;
        Ok(names)
    }

    /// Open an existing dataset (relative or absolute path).
    pub fn dataset(&self, name: &str) -> Result<Dataset> {
        let c = cstr(name)?;
        let _g = lock();
        // SAFETY: valid location id and C string, under the lock.
        let id = unsafe { sys::h5d::H5Dopen2(self.id(), c.as_ptr(), H5P_DEFAULT) };
        Ok(Dataset { h: Handle::check(id, "H5Dopen2", || describe_child(self.id(), name))? })
    }

    /// Start building a new dataset with element type `T`.
    pub fn new_dataset<T: H5Type>(&self) -> DatasetBuilder<'_, T> {
        DatasetBuilder::new(self)
    }

    /// Delete (unlink) a member. Note HDF5 does not reclaim file space.
    pub fn delete(&self, name: &str) -> Result<()> {
        let c = cstr(name)?;
        let _g = lock();
        // SAFETY: valid location id and C string, under the lock.
        let r = unsafe { sys::h5l::H5Ldelete(self.id(), c.as_ptr(), H5P_DEFAULT) };
        check(r, "H5Ldelete", || describe_child(self.id(), name))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::path_prefixes;
    #[test]
    fn prefixes() {
        assert_eq!(path_prefixes("a/b//c/"), vec!["a", "a/b", "a/b/c"]);
        assert_eq!(path_prefixes("/a/b"), vec!["/a", "/a/b"]);
        assert!(path_prefixes("/").is_empty());
        assert!(path_prefixes("").is_empty());
    }
}
