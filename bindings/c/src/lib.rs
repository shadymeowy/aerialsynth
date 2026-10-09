//! C API of aerialsynth: a layer of a tile of a world, read from its HDF5 tile store and generated
//! (and stored) when missing. `include/aerialsynth.h` is generated from this file by cbindgen
//! (`tests/header.rs` checks it is up to date; `AERIALSYNTH_BLESS=1` rewrites it).
//!
//! The implementation is `aerialsynth-core`, shared with the Python bindings.

#![allow(non_camel_case_types)]

use aerialsynth_core::{Error, Layer, World};
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

/// A layer of a tile: one of the `AS_LAYER_*` values.
pub type as_layer = u32;
/// Satellite look: the surface lit by a fixed sun, with haze. u8 x 3, sRGB.
pub const AS_LAYER_RGB: as_layer = 0;
/// Unlit surface colour. u8 x 3, sRGB encoded.
pub const AS_LAYER_ALBEDO: as_layer = 1;
/// DSM (ground, canopy, buildings, water surface) in metres above the WGS84 ellipsoid, at pixel
/// centres. f32 x 1.
pub const AS_LAYER_ELEVATION: as_layer = 2;
/// Unit surface normal (east, north, up) * 127. i8 x 3.
pub const AS_LAYER_NORMAL: as_layer = 3;
/// Land-cover class id (0 unknown, 1 ocean, 2 lake, 3 river, 4 beach, 5 sand, 6 rock, 7 snow,
/// 8 grass, 9 shrub, 10 forest, 11 crop, 12 building, 13 road, 14 wetland, 15 tundra, 16 bare,
/// 17 urban). u8 x 1.
pub const AS_LAYER_LANDCOVER: as_layer = 4;
/// Night-time artificial light: linear radiance = 16 * (v/255)^3. u8 x 3.
pub const AS_LAYER_EMISSION: as_layer = 5;
/// Number of layers (valid layers are 0 .. AS_LAYER_COUNT - 1).
pub const AS_LAYER_COUNT: u32 = 6;

/// Element type of a layer's pixels: one of the `AS_DTYPE_*` values.
pub type as_dtype = u32;
/// unsigned 8-bit
pub const AS_DTYPE_U8: as_dtype = 0;
/// signed 8-bit
pub const AS_DTYPE_I8: as_dtype = 1;
/// little-endian IEEE 754 binary32
pub const AS_DTYPE_F32: as_dtype = 2;

/// Success.
pub const AS_OK: c_int = 0;
/// A bad argument: NULL pointer, unknown layer, tile out of range (z above the world's max zoom,
/// x or y >= 2^z).
pub const AS_ERR_INVALID_ARGUMENT: c_int = -1;
/// The output buffer is smaller than the layer's tile size (`as_layer_size`).
pub const AS_ERR_BUFFER_TOO_SMALL: c_int = -2;
/// Reading, generating or storing the tile failed (see `as_last_error`).
pub const AS_ERR_FAILED: c_int = -3;
/// An internal error (a Rust panic, caught at the API boundary; see `as_last_error`).
pub const AS_ERR_PANIC: c_int = -4;

/// Width and height of a tile in pixels.
pub const AS_TILE_SIZE: u32 = 256;

/// A tile store opened for one world. Opaque; made by `as_open`, freed by `as_close`.
pub struct as_world {
    world: World,
}

/// Pixel format of a layer (`as_layer_describe`).
#[repr(C)]
pub struct as_layer_info {
    /// Layer name ("rgb", "albedo", "elevation", "normal", "landcover", "emission"); static.
    pub name: *const c_char,
    /// An `AS_DTYPE_*` value.
    pub dtype: as_dtype,
    /// Channels per pixel: 1 or 3.
    pub channels: u32,
    /// Bytes per channel: 1 or 4.
    pub elem_size: u32,
    /// Bytes of one tile of this layer: 256 * 256 * channels * elem_size.
    pub size: usize,
}

const NAMES: [&CStr; 6] = [c"rgb", c"albedo", c"elevation", c"normal", c"landcover", c"emission"];

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: impl Into<String>) {
    let msg = msg.into().replace('\0', " ");
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(msg).unwrap_or_default());
}

/// Run `f`, turning errors and panics into a status code (and the thread's last error).
fn guard(f: impl FnOnce() -> Result<(), (c_int, String)>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => AS_OK,
        Ok(Err((code, msg))) => {
            set_error(msg);
            code
        }
        Err(p) => {
            set_error(format!("internal error: {}", panic_message(&*p)));
            AS_ERR_PANIC
        }
    }
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
}

fn status(e: Error) -> (c_int, String) {
    let code = match e {
        Error::InvalidArgument(_) => AS_ERR_INVALID_ARGUMENT,
        _ => AS_ERR_FAILED,
    };
    (code, e.to_string())
}

fn invalid(msg: impl Into<String>) -> (c_int, String) {
    (AS_ERR_INVALID_ARGUMENT, msg.into())
}

fn layer(l: as_layer) -> Result<Layer, (c_int, String)> {
    Layer::ALL.get(l as usize).copied().ok_or_else(|| invalid(format!("unknown layer {l} (0..{AS_LAYER_COUNT})")))
}

/// # Safety
/// `s` is NULL or a NUL-terminated string.
unsafe fn path(s: *const c_char, what: &str) -> Result<Option<PathBuf>, (c_int, String)> {
    if s.is_null() {
        return Ok(None);
    }
    let bytes = unsafe { CStr::from_ptr(s) }.to_bytes();
    #[cfg(unix)]
    let p = PathBuf::from(<std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(bytes));
    #[cfg(not(unix))]
    let p = PathBuf::from(std::str::from_utf8(bytes).map_err(|_| invalid(format!("{what} is not UTF-8")))?);
    if p.as_os_str().is_empty() {
        return Err(invalid(format!("{what} is empty")));
    }
    Ok(Some(p))
}

/// The library version ("0.1.0"); a static string.
#[no_mangle]
pub extern "C" fn as_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// The message of the last failed call on this thread ("" if none failed yet). The string is
/// owned by the library and valid until the next failing call on the same thread.
#[no_mangle]
pub extern "C" fn as_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Open the tile store `tiles_file` (created, with its directory, if missing) for a world.
///
/// The world is defined like `terrain -c CONFIG --seed SEED`: `config_yaml` is the path of a
/// scenario YAML (its `world:` section is the world, its `tiles.max_zoom` the zoom limit, default
/// 18; other sections are ignored) or of a bare world config, or NULL for the default world.
/// `seed` >= 0 overrides the config's seed (< 0 keeps it).
///
/// A store holds one world: a store of another world (or generator version) is refused. A tiles
/// file can be open only once per process: share the handle (it is thread-safe) instead.
/// Returns NULL on error (see `as_last_error`). Close the handle with `as_close`.
///
/// # Safety
/// `tiles_file` is a NUL-terminated path; `config_yaml` is NULL or a NUL-terminated path.
#[no_mangle]
pub unsafe extern "C" fn as_open(tiles_file: *const c_char, config_yaml: *const c_char, seed: i64) -> *mut as_world {
    let mut out = std::ptr::null_mut();
    guard(|| {
        let Some(tiles) = (unsafe { path(tiles_file, "tiles_file") })? else { return Err(invalid("tiles_file is NULL")) };
        let config = unsafe { path(config_yaml, "config_yaml") }?;
        let seed = u64::try_from(seed).ok();
        let world = World::open(&tiles, config.as_deref(), seed).map_err(status)?;
        out = Box::into_raw(Box::new(as_world { world }));
        Ok(())
    });
    out
}

/// Close a world (flushing its store). NULL is ignored. The handle must not be in use by another
/// thread, and is invalid afterwards.
///
/// # Safety
/// `w` is NULL or a handle from `as_open` not closed yet.
#[no_mangle]
pub unsafe extern "C" fn as_close(w: *mut as_world) {
    if !w.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(w) })));
    }
}

/// Write the pixels of `layer` of tile `z/x/y` to `out`: `as_layer_size(layer)` bytes, row-major,
/// 256 rows (row 0 = north) x 256 columns x channels, little-endian. A tile that is not stored
/// yet is generated (all layers; on the GPU when there is a suitable one, else on the CPU) and
/// stored first.
///
/// `z` must be at most the world's max zoom, `x` and `y` less than 2^z (XYZ / Web-Mercator
/// scheme, y = 0 at the north edge). A handle can be used from several threads at once.
/// Returns `AS_OK` or a negative `AS_ERR_*` code (see `as_last_error`).
///
/// # Safety
/// `w` is a handle from `as_open`; `out` points to `out_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn as_tile(w: *const as_world, z: u32, x: u32, y: u32, layer: as_layer, out: *mut c_void, out_len: usize) -> c_int {
    guard(|| {
        let l = self::layer(layer)?;
        let Some(w) = (unsafe { w.as_ref() }) else { return Err(invalid("world handle is NULL")) };
        if out.is_null() {
            return Err(invalid("output buffer is NULL"));
        }
        if out_len < l.tile_bytes() {
            return Err((AS_ERR_BUFFER_TOO_SMALL, format!("buffer of {out_len} bytes is too small for layer {} ({} bytes)", l.name(), l.tile_bytes())));
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out as *mut u8, l.tile_bytes()) };
        w.world.tile_into(z, x, y, l, buf).map_err(status)
    })
}

/// The highest zoom the world serves (`tiles.max_zoom` of its config, default 18), or -1 if `w`
/// is NULL.
///
/// # Safety
/// `w` is NULL or a handle from `as_open`.
#[no_mangle]
pub unsafe extern "C" fn as_max_zoom(w: *const as_world) -> c_int {
    match unsafe { w.as_ref() } {
        Some(w) => w.world.max_zoom() as c_int,
        None => -1,
    }
}

/// Bytes of one tile of `layer` (256 * 256 * channels * element size), or 0 for an unknown layer.
#[no_mangle]
pub extern "C" fn as_layer_size(layer: as_layer) -> usize {
    self::layer(layer).map(|l| l.tile_bytes()).unwrap_or(0)
}

/// Describe `layer` into `*info`. Returns `AS_OK` or `AS_ERR_INVALID_ARGUMENT`.
///
/// # Safety
/// `info` points to a writable `as_layer_info`.
#[no_mangle]
pub unsafe extern "C" fn as_layer_describe(layer: as_layer, info: *mut as_layer_info) -> c_int {
    guard(|| {
        let l = self::layer(layer)?;
        if info.is_null() {
            return Err(invalid("info is NULL"));
        }
        let d = aerialsynth_core::layer_info(l);
        let dtype = match d.dtype {
            aerialsynth_core::Dtype::U8 => AS_DTYPE_U8,
            aerialsynth_core::Dtype::I8 => AS_DTYPE_I8,
            aerialsynth_core::Dtype::F32 => AS_DTYPE_F32,
        };
        let i = as_layer_info { name: NAMES[layer as usize].as_ptr(), dtype, channels: d.channels as u32, elem_size: d.dtype.size() as u32, size: d.size };
        unsafe { info.write(i) };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last_error() -> String {
        unsafe { CStr::from_ptr(as_last_error()) }.to_string_lossy().into_owned()
    }

    #[test]
    fn layers_match_the_core() {
        assert_eq!(AS_LAYER_COUNT as usize, Layer::ALL.len());
        for (i, l) in Layer::ALL.iter().enumerate() {
            let mut info = std::mem::MaybeUninit::<as_layer_info>::uninit();
            assert_eq!(unsafe { as_layer_describe(i as as_layer, info.as_mut_ptr()) }, AS_OK);
            let info = unsafe { info.assume_init() };
            assert_eq!(unsafe { CStr::from_ptr(info.name) }.to_str().unwrap(), l.name());
            assert_eq!((info.channels as usize, info.elem_size as usize, info.size), (l.channels(), l.elem_size(), l.tile_bytes()));
            assert_eq!(as_layer_size(i as as_layer), l.tile_bytes());
        }
        assert_eq!(as_layer_size(AS_LAYER_COUNT), 0);
        assert_eq!(as_layer_size(AS_LAYER_ELEVATION), 256 * 256 * 4);
        assert_eq!(unsafe { as_layer_describe(AS_LAYER_COUNT, std::ptr::null_mut()) }, AS_ERR_INVALID_ARGUMENT);
        assert_eq!(unsafe { as_layer_describe(AS_LAYER_RGB, std::ptr::null_mut()) }, AS_ERR_INVALID_ARGUMENT);
        assert_eq!(unsafe { CStr::from_ptr(as_version()) }.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn bad_arguments() {
        unsafe {
            assert!(as_open(std::ptr::null(), std::ptr::null(), -1).is_null());
            assert!(last_error().contains("tiles_file is NULL"), "{}", last_error());
            assert!(as_open(c"".as_ptr(), std::ptr::null(), -1).is_null());
            let missing = c"/nonexistent/aerialsynth/config.yaml";
            assert!(as_open(c"/nonexistent/aerialsynth/w.h5".as_ptr(), missing.as_ptr(), -1).is_null());
            assert!(last_error().contains("config.yaml"), "{}", last_error());
            let mut buf = vec![0u8; 16];
            assert_eq!(as_tile(std::ptr::null(), 0, 0, 0, AS_LAYER_RGB, buf.as_mut_ptr() as *mut c_void, buf.len()), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_tile(std::ptr::null(), 0, 0, 0, 99, buf.as_mut_ptr() as *mut c_void, buf.len()), AS_ERR_INVALID_ARGUMENT);
            assert!(last_error().contains("unknown layer 99"));
            assert_eq!(as_max_zoom(std::ptr::null()), -1);
            as_close(std::ptr::null_mut());
        }
    }

    #[test]
    fn tiles_through_the_c_api() {
        let dir = std::env::temp_dir().join(format!("aerialsynth-capi-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("w.yaml");
        std::fs::write(&cfg, "world: { tile_supersample: 1 }\ntiles: { max_zoom: 5 }\n").unwrap();
        let c = |p: &std::path::Path| CString::new(p.to_str().unwrap()).unwrap();
        let (tiles, cfgc) = (c(&dir.join("w.h5")), c(&cfg));
        unsafe {
            let w = as_open(tiles.as_ptr(), cfgc.as_ptr(), 2);
            assert!(!w.is_null(), "{}", last_error());
            assert_eq!(as_max_zoom(w), 5);
            let n = as_layer_size(AS_LAYER_ELEVATION);
            let (mut a, mut b) = (vec![0u8; n], vec![0u8; n + 7]);
            assert_eq!(as_tile(w, 3, 5, 3, AS_LAYER_ELEVATION, a.as_mut_ptr() as *mut c_void, n), AS_OK, "{}", last_error());
            assert_eq!(as_tile(w, 3, 5, 3, AS_LAYER_ELEVATION, b.as_mut_ptr() as *mut c_void, n + 7), AS_OK);
            assert_eq!(a[..], b[..n]);
            assert_eq!(as_tile(w, 3, 5, 3, AS_LAYER_ELEVATION, b.as_mut_ptr() as *mut c_void, n - 1), AS_ERR_BUFFER_TOO_SMALL);
            assert_eq!(as_tile(w, 3, 5, 3, AS_LAYER_ELEVATION, std::ptr::null_mut(), n), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_tile(w, 3, 8, 3, AS_LAYER_ELEVATION, a.as_mut_ptr() as *mut c_void, n), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_tile(w, 6, 0, 0, AS_LAYER_ELEVATION, a.as_mut_ptr() as *mut c_void, n), AS_ERR_INVALID_ARGUMENT);
            assert!(last_error().contains("max_zoom"), "{}", last_error());
            as_close(w);
            // another seed: refused
            assert!(as_open(tiles.as_ptr(), cfgc.as_ptr(), 3).is_null());
            assert!(last_error().contains("world.seed (2 → 3)"), "{}", last_error());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
