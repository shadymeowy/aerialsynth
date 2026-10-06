use std::ffi::{c_char, c_void, CStr};
use std::ptr;

use hdf5_sys as sys;
use sys::h5::herr_t;
use sys::h5i::hid_t;
use sys::h5p::H5P_DEFAULT;

use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::file::File;
use crate::group::Group;
use crate::raw::{check, clear_errors, cstr, describe_child, lock, Handle};
use crate::types::H5Type;

unsafe extern "C" fn collect_attr(
    _loc: hid_t,
    name: *const c_char,
    _info: *const sys::h5a::H5A_info_t,
    data: *mut c_void,
) -> herr_t {
    // SAFETY: `data` is the &mut Vec<String> passed to H5Aiterate2, `name` a
    // valid C string for the duration of the callback.
    unsafe {
        let v = &mut *(data as *mut Vec<String>);
        v.push(CStr::from_ptr(name).to_string_lossy().into_owned());
    }
    0
}

/// Create (replacing any existing) attribute `name` on `loc` with file type
/// `ftype`, dataspace `space`, and write `buf` (memory type `mtype`).
///
/// # Safety
/// Caller holds the lock; `buf` must point to data matching `mtype`×npoints(space).
unsafe fn write_attr(
    loc: hid_t,
    name: &str,
    ftype: hid_t,
    mtype: hid_t,
    space: &Handle,
    buf: *const c_void,
) -> Result<()> {
    let c = cstr(name)?;
    let ctx = || describe_child(loc, name);
    unsafe {
        if sys::h5a::H5Aexists(loc, c.as_ptr()) > 0 {
            check(sys::h5a::H5Adelete(loc, c.as_ptr()), "H5Adelete", ctx)?;
        }
        let a = Handle::check(
            sys::h5a::H5Acreate2(loc, c.as_ptr(), ftype, space.id(), H5P_DEFAULT, H5P_DEFAULT),
            "H5Acreate2",
            ctx,
        )?;
        check(sys::h5a::H5Awrite(a.id(), mtype, buf), "H5Awrite", ctx)?;
    }
    Ok(())
}

/// Open attribute `name` and return (attr, npoints).
fn open_attr(loc: hid_t, name: &str) -> Result<(Handle, Handle, usize)> {
    let c = cstr(name)?;
    let ctx = || describe_child(loc, name);
    let _g = lock();
    // SAFETY: valid ids and C string under the lock.
    unsafe {
        let a = Handle::check(sys::h5a::H5Aopen(loc, c.as_ptr(), H5P_DEFAULT), "H5Aopen", ctx)?;
        let s = Handle::check(sys::h5a::H5Aget_space(a.id()), "H5Aget_space", ctx)?;
        let n = sys::h5s::H5Sget_simple_extent_npoints(s.id());
        if n < 0 {
            return Err(crate::raw::hdf5_error("H5Sget_simple_extent_npoints", ctx));
        }
        Ok((a, s, n as usize))
    }
}

/// A variable-length UTF-8 C string datatype (caller must hold the lock).
fn vlen_str_type(cset: sys::h5t::H5T_cset_t, ctx: impl Fn() -> String) -> Result<Handle> {
    // SAFETY: under the lock; H5T_C_S1 is a valid predefined type after H5open.
    unsafe {
        let t = Handle::check(sys::h5t::H5Tcopy(*sys::h5t::H5T_C_S1), "H5Tcopy", &ctx)?;
        check(sys::h5t::H5Tset_size(t.id(), sys::h5t::H5T_VARIABLE), "H5Tset_size", &ctx)?;
        check(sys::h5t::H5Tset_cset(t.id(), cset), "H5Tset_cset", &ctx)?;
        Ok(t)
    }
}

/// Attribute access, implemented for [`File`] (root group), [`Group`] and
/// [`Dataset`].
pub trait Attrs {
    /// The HDF5 object id attributes are attached to.
    fn attr_loc(&self) -> hid_t;

    /// Set a scalar attribute, replacing any existing one with that name.
    fn set_attr<T: H5Type>(&self, name: &str, value: T) -> Result<()> {
        let loc = self.attr_loc();
        let _g = lock();
        // SAFETY: under the lock; `value` is one T matching its native type.
        unsafe {
            let s = Handle::check(
                sys::h5s::H5Screate(sys::h5s::H5S_class_t::H5S_SCALAR),
                "H5Screate",
                || describe_child(loc, name),
            )?;
            write_attr(loc, name, T::native_type(), T::native_type(), &s, &value as *const T as *const c_void)
        }
    }

    /// Set a 1-D array attribute, replacing any existing one.
    fn set_attr_array<T: H5Type>(&self, name: &str, values: &[T]) -> Result<()> {
        let loc = self.attr_loc();
        let _g = lock();
        let dims = [values.len() as sys::h5::hsize_t];
        // SAFETY: under the lock; `values` has dims[0] elements of T.
        unsafe {
            let s = Handle::check(
                sys::h5s::H5Screate_simple(1, dims.as_ptr(), ptr::null()),
                "H5Screate_simple",
                || describe_child(loc, name),
            )?;
            write_attr(loc, name, T::native_type(), T::native_type(), &s, values.as_ptr() as *const c_void)
        }
    }

    /// Set a scalar variable-length UTF-8 string attribute (h5py reads it as
    /// `str`), replacing any existing one.
    fn set_attr_str(&self, name: &str, value: &str) -> Result<()> {
        let loc = self.attr_loc();
        let cval = cstr(value)?;
        let _g = lock();
        let ctx = || describe_child(loc, name);
        let t = vlen_str_type(sys::h5t::H5T_cset_t::H5T_CSET_UTF8, ctx)?;
        let p: *const c_char = cval.as_ptr();
        // SAFETY: under the lock; a vlen string buffer is an array of char*
        // (here one pointer, valid for the call).
        unsafe {
            let s = Handle::check(
                sys::h5s::H5Screate(sys::h5s::H5S_class_t::H5S_SCALAR),
                "H5Screate",
                ctx,
            )?;
            write_attr(loc, name, t.id(), t.id(), &s, &p as *const *const c_char as *const c_void)
        }
    }

    /// Read a scalar (or single-element) attribute.
    fn attr<T: H5Type>(&self, name: &str) -> Result<T> {
        let loc = self.attr_loc();
        let _g = lock();
        let (a, _s, n) = open_attr(loc, name)?;
        if n != 1 {
            return Err(Error::InvalidArgument(format!(
                "{}: attribute has {n} elements, expected 1",
                describe_child(loc, name)
            )));
        }
        let mut v = T::default();
        // SAFETY: one element of T's native type.
        let r = unsafe { sys::h5a::H5Aread(a.id(), T::native_type(), &mut v as *mut T as *mut c_void) };
        check(r, "H5Aread", || describe_child(loc, name))?;
        Ok(v)
    }

    /// Read all elements of an attribute (any rank, C order).
    fn attr_array<T: H5Type>(&self, name: &str) -> Result<Vec<T>> {
        let loc = self.attr_loc();
        let _g = lock();
        let (a, _s, n) = open_attr(loc, name)?;
        let mut v = vec![T::default(); n];
        if n == 0 {
            return Ok(v);
        }
        // SAFETY: n elements of T's native type.
        let r = unsafe { sys::h5a::H5Aread(a.id(), T::native_type(), v.as_mut_ptr() as *mut c_void) };
        check(r, "H5Aread", || describe_child(loc, name))?;
        Ok(v)
    }

    /// Read a scalar string attribute (variable- or fixed-length; fixed
    /// strings are trimmed at the first NUL and, if space-padded, of
    /// trailing spaces). Invalid UTF-8 is replaced lossily.
    fn attr_str(&self, name: &str) -> Result<String> {
        let loc = self.attr_loc();
        let _g = lock();
        let ctx = || describe_child(loc, name);
        let (a, s, n) = open_attr(loc, name)?;
        if n != 1 {
            return Err(Error::InvalidArgument(format!(
                "{}: string attribute has {n} elements, expected 1",
                ctx()
            )));
        }
        // SAFETY: all under the lock. For vlen strings we read one char*
        // (allocated by HDF5) and release it with H5Treclaim; for fixed
        // strings we read into a buffer of exactly H5Tget_size bytes.
        unsafe {
            let ft = Handle::check(sys::h5a::H5Aget_type(a.id()), "H5Aget_type", ctx)?;
            if sys::h5t::H5Tget_class(ft.id()) != sys::h5t::H5T_class_t::H5T_STRING {
                return Err(Error::InvalidArgument(format!("{}: attribute is not a string", ctx())));
            }
            let is_vlen = check(sys::h5t::H5Tis_variable_str(ft.id()), "H5Tis_variable_str", ctx)? > 0;
            if is_vlen {
                let mt = vlen_str_type(sys::h5t::H5Tget_cset(ft.id()), ctx)?;
                let mut p: *mut c_char = ptr::null_mut();
                check(
                    sys::h5a::H5Aread(a.id(), mt.id(), &mut p as *mut *mut c_char as *mut c_void),
                    "H5Aread(vlen str)",
                    ctx,
                )?;
                let out = if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                };
                sys::h5t::H5Treclaim(
                    mt.id(),
                    s.id(),
                    H5P_DEFAULT,
                    &mut p as *mut *mut c_char as *mut c_void,
                );
                Ok(out)
            } else {
                let size = sys::h5t::H5Tget_size(ft.id());
                let mt = Handle::check(sys::h5t::H5Tcopy(ft.id()), "H5Tcopy", ctx)?;
                let mut buf = vec![0u8; size];
                check(
                    sys::h5a::H5Aread(a.id(), mt.id(), buf.as_mut_ptr() as *mut c_void),
                    "H5Aread(fixed str)",
                    ctx,
                )?;
                if let Some(z) = buf.iter().position(|&b| b == 0) {
                    buf.truncate(z);
                }
                if sys::h5t::H5Tget_strpad(ft.id()) == sys::h5t::H5T_str_t::H5T_STR_SPACEPAD {
                    while buf.last() == Some(&b' ') {
                        buf.pop();
                    }
                }
                Ok(String::from_utf8_lossy(&buf).into_owned())
            }
        }
    }

    /// Whether an attribute named `name` exists.
    fn has_attr(&self, name: &str) -> bool {
        let loc = self.attr_loc();
        let Ok(c) = cstr(name) else { return false };
        let _g = lock();
        // SAFETY: valid id and C string under the lock.
        let r = unsafe { sys::h5a::H5Aexists(loc, c.as_ptr()) };
        if r < 0 {
            clear_errors();
        }
        r > 0
    }

    /// Delete an attribute.
    fn delete_attr(&self, name: &str) -> Result<()> {
        let loc = self.attr_loc();
        let c = cstr(name)?;
        let _g = lock();
        // SAFETY: valid id and C string under the lock.
        let r = unsafe { sys::h5a::H5Adelete(loc, c.as_ptr()) };
        check(r, "H5Adelete", || describe_child(loc, name))?;
        Ok(())
    }

    /// Names of all attributes, in name order.
    fn attr_names(&self) -> Result<Vec<String>> {
        let loc = self.attr_loc();
        let _g = lock();
        let mut names: Vec<String> = Vec::new();
        // SAFETY: valid id; callback only pushes into `names`.
        let r = unsafe {
            sys::h5a::H5Aiterate2(
                loc,
                sys::h5::H5_index_t::H5_INDEX_NAME,
                sys::h5::H5_iter_order_t::H5_ITER_INC,
                ptr::null_mut(),
                Some(collect_attr),
                &mut names as *mut Vec<String> as *mut c_void,
            )
        };
        check(r, "H5Aiterate2", || crate::raw::describe(loc))?;
        Ok(names)
    }
}

impl Attrs for Group {
    fn attr_loc(&self) -> hid_t {
        self.id()
    }
}

impl Attrs for Dataset {
    fn attr_loc(&self) -> hid_t {
        self.id()
    }
}

impl Attrs for File {
    fn attr_loc(&self) -> hid_t {
        self.id()
    }
}
