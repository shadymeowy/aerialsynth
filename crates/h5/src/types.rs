use hdf5_sys as sys;
use sys::h5i::hid_t;

/// Rust element types that map 1:1 onto an HDF5 native datatype.
///
/// # Safety
///
/// `native_type()` must return a (library-owned, never closed) HDF5 datatype
/// whose in-memory size and representation are exactly those of `Self`, since
/// buffers of `Self` are handed to HDF5 as raw bytes of that type.
pub unsafe trait H5Type: Copy + Default + Send + Sync + 'static {
    /// The HDF5 native datatype id for `Self` (initialises the library if
    /// needed).
    fn native_type() -> hid_t;
}

macro_rules! impl_h5type {
    ($($t:ty => $g:ident),* $(,)?) => {$(
        // SAFETY: the H5T_NATIVE_* type with the same width/signedness/float
        // format as the Rust primitive.
        unsafe impl H5Type for $t {
            #[inline]
            fn native_type() -> hid_t {
                // Ensures H5open() has run, which populates the global id.
                let _g = crate::raw::lock();
                *sys::h5t::$g
            }
        }
    )*};
}

impl_h5type! {
    u8 => H5T_NATIVE_UINT8,
    i8 => H5T_NATIVE_INT8,
    u16 => H5T_NATIVE_UINT16,
    i16 => H5T_NATIVE_INT16,
    u32 => H5T_NATIVE_UINT32,
    i32 => H5T_NATIVE_INT32,
    u64 => H5T_NATIVE_UINT64,
    i64 => H5T_NATIVE_INT64,
    f32 => H5T_NATIVE_FLOAT,
    f64 => H5T_NATIVE_DOUBLE,
}
