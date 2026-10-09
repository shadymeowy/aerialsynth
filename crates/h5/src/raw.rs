//! Low-level plumbing shared by all safe wrappers: the global library lock,
//! one-time initialisation, the RAII [`Handle`] for HDF5 ids, error-stack
//! capture and small conversion helpers.
//!
//! # Locking discipline
//!
//! The bundled HDF5 is built *without* thread-safety, so every FFI call into
//! the library must be serialised. We use `hdf5_sys::LOCK` (a
//! `parking_lot::ReentrantMutex<()>` exported by the sys crate itself) so that
//! any other code in the process that talks to `hdf5-sys` directly and honours
//! that lock is serialised with us too. Every public entry point of this crate
//! takes the lock via [`lock`] for the whole duration of its FFI work; the
//! lock is reentrant, so nested helpers may take it again.

use std::ffi::{c_char, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::Once;

use hdf5_sys as sys;
use parking_lot::ReentrantMutexGuard;
use sys::h5::herr_t;
use sys::h5i::hid_t;

use crate::error::{Error, Result};

static INIT: Once = Once::new();

/// Acquire the global (reentrant) HDF5 lock, initialising the library on
/// first use.
///
/// All safe APIs in this crate take this lock internally; you only need it
/// yourself when calling raw functions from [`crate::sys`] (e.g. using the
/// `id()` of a handle). Holding it across a long computation blocks every
/// other thread's HDF5 access, so keep critical sections short.
pub fn lock() -> ReentrantMutexGuard<'static, ()> {
    let guard = sys::LOCK.lock();
    INIT.call_once(|| {
        // SAFETY: we hold the global lock. H5open initialises the library and
        // the global ids (H5T_NATIVE_*, H5P_CLS_*) that are read afterwards.
        // Disabling the automatic error printer is a plain library setting.
        unsafe {
            sys::h5::H5open();
            sys::h5e::H5Eset_auto2(sys::h5e::H5E_DEFAULT, None, ptr::null_mut());
        }
    });
    guard
}

/// Owned reference to an HDF5 identifier (file, group, dataset, attribute,
/// dataspace, datatype, property list...). Decrements the id's reference
/// count on drop, which closes the object when it reaches zero.
///
/// Invariant: `self.0` is a valid id whose reference we own. Never wrap a
/// library-owned predefined id (e.g. `H5T_NATIVE_UINT8`).
#[derive(Debug)]
pub(crate) struct Handle(hid_t);

impl Handle {
    /// Wrap a freshly returned id, converting a negative id into an error that
    /// carries the HDF5 error stack. Caller must hold the lock.
    pub(crate) fn check(id: hid_t, op: &str, target: impl FnOnce() -> String) -> Result<Handle> {
        if id < 0 {
            Err(hdf5_error(op, target))
        } else {
            Ok(Handle(id))
        }
    }

    #[inline]
    pub(crate) fn id(&self) -> hid_t {
        self.0
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        let _g = lock();
        // SAFETY: self.0 is a valid id (struct invariant); we take one more
        // reference which the clone will own.
        unsafe { sys::h5i::H5Iinc_ref(self.0) };
        Handle(self.0)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        let _g = lock();
        // SAFETY: we own exactly one reference to a valid id.
        unsafe { sys::h5i::H5Idec_ref(self.0) };
    }
}

/// Check an `herr_t`/`htri_t` style return code (negative = failure).
/// Caller must hold the lock.
pub(crate) fn check(ret: herr_t, op: &str, target: impl FnOnce() -> String) -> Result<herr_t> {
    if ret < 0 {
        Err(hdf5_error(op, target))
    } else {
        Ok(ret)
    }
}

/// Clear the current error stack (after an expected/ignored failure).
pub(crate) fn clear_errors() {
    let _g = lock();
    // SAFETY: plain library call under the lock.
    unsafe { sys::h5e::H5Eclear2(sys::h5e::H5E_DEFAULT) };
}

struct RawErr {
    func: String,
    desc: String,
    maj: hid_t,
    min: hid_t,
}

unsafe extern "C" fn walk_cb(_n: c_uint, err: *const sys::h5e::H5E_error2_t, data: *mut c_void) -> herr_t {
    // SAFETY: HDF5 passes a valid error record; `data` is the &mut Vec we
    // handed to H5Ewalk2. Strings are only valid during the callback, so copy.
    unsafe {
        let out = &mut *(data as *mut Vec<RawErr>);
        let e = &*err;
        let s = |p: *const c_char| {
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        out.push(RawErr { func: s(e.func_name), desc: s(e.desc), maj: e.maj_num, min: e.min_num });
    }
    0
}

fn err_msg(id: hid_t) -> String {
    let mut buf = [0 as c_char; 256];
    // SAFETY: buffer is valid for 256 bytes; H5Eget_msg NUL-terminates.
    let n = unsafe { sys::h5e::H5Eget_msg(id, ptr::null_mut(), buf.as_mut_ptr(), buf.len()) };
    if n <= 0 {
        return String::new();
    }
    // SAFETY: NUL-terminated by HDF5 within buf.
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

/// Capture (and clear) the current HDF5 error stack as a human readable
/// string. Caller must hold the lock.
pub(crate) fn error_stack() -> String {
    let _g = lock();
    let mut recs: Vec<RawErr> = Vec::new();
    // SAFETY: under the lock. H5Eget_current_stack copies and clears the
    // default stack; we walk the copy with a callback that only copies data
    // out, then close the copy.
    unsafe {
        let stack = sys::h5e::H5Eget_current_stack();
        if stack < 0 {
            return "<unable to retrieve HDF5 error stack>".into();
        }
        sys::h5e::H5Ewalk2(stack, sys::h5e::H5E_direction_t::H5E_WALK_DOWNWARD, Some(walk_cb), &mut recs as *mut Vec<RawErr> as *mut c_void);
        let msgs: Vec<String> = recs
            .iter()
            .map(|r| {
                let (maj, min) = (err_msg(r.maj), err_msg(r.min));
                format!("{}(): {} [{}: {}]", r.func, r.desc, maj, min)
            })
            .collect();
        sys::h5e::H5Eclose_stack(stack);
        if msgs.is_empty() {
            "<empty HDF5 error stack>".into()
        } else {
            msgs.join("; ")
        }
    }
}

/// Build an error from the current HDF5 error stack. The stack is captured before `target`
/// runs: describing the target calls HDF5 API functions, which clear the stack on entry.
pub(crate) fn hdf5_error(op: &str, target: impl FnOnce() -> String) -> Error {
    let stack = error_stack();
    Error::Hdf5 { op: op.to_string(), target: target(), stack }
}

/// Convert a Rust string to a C string, rejecting interior NULs.
pub(crate) fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::InvalidArgument(format!("name contains NUL byte: {s:?}")))
}

/// Read a string via the HDF5 "call with NULL to get length" convention.
fn get_string(f: impl Fn(*mut c_char, usize) -> isize) -> Option<String> {
    let n = f(ptr::null_mut(), 0);
    if n < 0 {
        return None;
    }
    let mut buf = vec![0u8; n as usize + 1];
    let n2 = f(buf.as_mut_ptr() as *mut c_char, buf.len());
    if n2 < 0 {
        return None;
    }
    buf.truncate(n2 as usize);
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Path of an object inside its file (e.g. "/tiles/z3/rgb"), or "?" if
/// unavailable. Never fails; clears any error it causes.
pub(crate) fn obj_path(id: hid_t) -> String {
    let _g = lock();
    // SAFETY: under the lock, buffer pointer/size pairs come from get_string.
    let r = get_string(|p, n| unsafe { sys::h5i::H5Iget_name(id, p, n) });
    r.unwrap_or_else(|| {
        clear_errors();
        "?".into()
    })
}

/// File name an object lives in, or "?" if unavailable.
pub(crate) fn file_name(id: hid_t) -> String {
    let _g = lock();
    // SAFETY: as above.
    let r = get_string(|p, n| unsafe { sys::h5f::H5Fget_name(id, p, n) });
    r.unwrap_or_else(|| {
        clear_errors();
        "?".into()
    })
}

/// "file.h5:/group/path" description of an object, for error context.
pub(crate) fn describe(id: hid_t) -> String {
    format!("{}:{}", file_name(id), obj_path(id))
}

/// "file.h5:/group/path » child" description for operations on a child name.
pub(crate) fn describe_child(id: hid_t, child: &str) -> String {
    format!("{} -> {:?}", describe(id), child)
}

pub(crate) fn to_hsize(v: &[usize]) -> Vec<sys::h5::hsize_t> {
    v.iter().map(|&x| x as sys::h5::hsize_t).collect()
}
