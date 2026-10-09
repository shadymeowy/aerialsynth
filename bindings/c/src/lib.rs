//! C API of aerialsynth: a layer of a tile of a world, read from its HDF5 tile store and generated
//! (and stored) when missing; camera images of the world rendered from a pose.
//! `include/aerialsynth.h` is generated from this file by cbindgen (`tests/header.rs` checks it
//! is up to date; `AERIALSYNTH_BLESS=1` rewrites it).
//!
//! The implementation is `aerialsynth-core`, shared with the Python bindings.

#![allow(non_camel_case_types)]

use aerialsynth_core::{Backend, Camera, CameraDef, Error, Layer, Mount, Outputs, Pinhole, Pose, World};
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;

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

/// Default size of a world's tile cache in MiB (`as_set_cache_mb`).
pub const AS_DEFAULT_CACHE_MB: usize = 256;

/// Most tiles `as_prefetch` takes (its box over its zooms).
pub const AS_MAX_PREFETCH_TILES: u64 = 1000000;

/// A tile store opened for one world. Opaque; made by `as_open`, freed by `as_close`.
pub struct as_world {
    world: Arc<World>,
}

/// Where a camera renders: one of the `AS_BACKEND_*` values.
pub type as_backend = u32;
/// The scenario's `render.backend` (`AS_BACKEND_AUTO` without a scenario).
pub const AS_BACKEND_DEFAULT: as_backend = 0;
/// The GPU when there is a usable one (that can take the camera model), else the CPU.
pub const AS_BACKEND_AUTO: as_backend = 1;
/// The CPU reference renderer.
pub const AS_BACKEND_CPU: as_backend = 2;
/// The GPU: opening the camera fails without a usable GPU.
pub const AS_BACKEND_GPU: as_backend = 3;

/// How a pinhole camera sits on the body: one of the `AS_MOUNT_*` values.
pub type as_mount = u32;
/// Optical axis = body forward, image top = body up: the pose's angles are the camera's own (yaw
/// = heading of the optical axis, pitch = its elevation, negative looks down).
pub const AS_MOUNT_FORWARD: as_mount = 0;
/// Optical axis = body down, image top = body forward (the scenario default mount): a level
/// pose looks straight down.
pub const AS_MOUNT_NADIR: as_mount = 1;

/// A camera over a world, rendering images from poses. Opaque; made by `as_camera_open` or
/// `as_camera_pinhole`, freed by `as_camera_close`.
pub struct as_camera {
    camera: Camera,
}

/// Position and body attitude of a render.
///
/// Position: geodetic latitude and longitude (degrees) and height above the WGS84 ellipsoid
/// (metres). Attitude: aerospace Z-Y-X Euler angles (degrees) of the body (x forward, y right,
/// z down) in the local north-east-down frame: yaw = heading clockwise from north, pitch nose-up
/// positive, roll right-wing-down positive. The camera sits on the body by its mount (the
/// scenario camera's `extrinsics`, or `as_mount` for a pinhole camera).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct as_pose {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    pub roll_deg: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
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
        out = Box::into_raw(Box::new(as_world { world: Arc::new(world) }));
        Ok(())
    });
    out
}

/// Close a world (flushing its store). NULL is ignored. The handle must not be in use by another
/// thread, and is invalid afterwards. Cameras of the world stay usable: the store is closed when
/// the world and all its cameras are closed.
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
/// 256 rows (row 0 = north) x 256 columns x channels, little-endian. A tile in the world's cache
/// (`as_set_cache_mb`) is copied from it; else it is read from the store, or, when it is not
/// stored yet, generated (all layers; on the GPU when there is a suitable one, else on the CPU)
/// and stored first. Many tiles at once: `as_tiles`.
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

/// Write the pixels of `layer` of `n` tiles to `out`: tile `i` (zoom, x, y = `zxy[3i]`,
/// `zxy[3i + 1]`, `zxy[3i + 2]`) at byte offset `i * as_layer_size(layer)`, as `as_tile` writes
/// it. `out_len` must be at least `n * as_layer_size(layer)`.
///
/// Faster than `n` calls of `as_tile`: every coordinate is checked before any work (one out of
/// range: `AS_ERR_INVALID_ARGUMENT`, nothing is read or written), cached tiles are copied, the
/// stored ones are read and decompressed in parallel, and the missing ones are generated
/// together (in batches of up to 64 tiles: on the GPU, many tiles per dispatch) and stored with
/// one write per batch. A tile listed several times is read or generated once. `n` = 0 does
/// nothing (`zxy` and `out` may then be NULL).
///
/// Returns `AS_OK` or a negative `AS_ERR_*` code (see `as_last_error`); on an error after the
/// checks (`AS_ERR_FAILED`) the contents of `out` are unspecified, but the tiles generated before
/// the failure are stored.
///
/// # Safety
/// `w` is a handle from `as_open`; `zxy` points to `3 * n` readable `uint32_t`s; `out` points to
/// `out_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn as_tiles(w: *const as_world, zxy: *const u32, n: usize, layer: as_layer, out: *mut c_void, out_len: usize) -> c_int {
    guard(|| {
        let l = self::layer(layer)?;
        let Some(w) = (unsafe { w.as_ref() }) else { return Err(invalid("world handle is NULL")) };
        if n == 0 {
            return Ok(());
        }
        if zxy.is_null() {
            return Err(invalid("zxy is NULL"));
        }
        if out.is_null() {
            return Err(invalid("output buffer is NULL"));
        }
        let need = match (n.checked_mul(3).and_then(|v| v.checked_mul(4)), n.checked_mul(l.tile_bytes())) {
            (Some(c), Some(need)) if c <= isize::MAX as usize && need <= isize::MAX as usize => need,
            _ => return Err(invalid(format!("n = {n} tiles is too many"))),
        };
        if out_len < need {
            return Err((AS_ERR_BUFFER_TOO_SMALL, format!("buffer of {out_len} bytes is too small for {n} tiles of layer {} ({need} bytes)", l.name())));
        }
        let coords = unsafe { std::slice::from_raw_parts(zxy as *const [u32; 3], n) };
        let buf = unsafe { std::slice::from_raw_parts_mut(out as *mut u8, need) };
        w.world.tiles_into(coords, l, buf).map_err(status)
    })
}

/// Generate and store the missing tiles of a latitude / longitude box (degrees) at zooms `z_min`
/// to `z_max`, without returning them: the tiles intersecting the box (`lon_min > lon_max` is a
/// box across the antimeridian), in batches of up to 64 tiles. The number of tiles generated is
/// written to `*generated` (may be NULL).
///
/// The box must hold at most `AS_MAX_PREFETCH_TILES` tiles over those zooms (stored ones
/// included), and `z_min <= z_max <= ` the world's max zoom; else `AS_ERR_INVALID_ARGUMENT`
/// before any work. Returns `AS_OK` or a negative `AS_ERR_*` code; after a failure the batches
/// done are stored.
///
/// # Safety
/// `w` is a handle from `as_open`; `generated` is NULL or points to a writable `size_t`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn as_prefetch(
    w: *const as_world,
    lat_min: f64,
    lon_min: f64,
    lat_max: f64,
    lon_max: f64,
    z_min: u32,
    z_max: u32,
    generated: *mut usize,
) -> c_int {
    guard(|| {
        let Some(w) = (unsafe { w.as_ref() }) else { return Err(invalid("world handle is NULL")) };
        let n = w.world.prefetch([lat_min, lon_min, lat_max, lon_max], z_min, z_max).map_err(status)?;
        if !generated.is_null() {
            unsafe { generated.write(n) };
        }
        Ok(())
    })
}

/// Set the size of world `w`'s in-memory cache of decoded tiles to `mb` MiB (default
/// `AS_DEFAULT_CACHE_MB`, 256; about 1.1 MiB per generated tile with all its layers, 192 KiB per
/// rgb layer of a tile). `as_tile` and `as_tiles` copy cached tiles instead of reading and
/// decompressing them from the store; tiles read or generated are cached, the least recently used
/// dropped beyond the size. 0 turns the cache off (and frees it). Returns `AS_OK` or
/// `AS_ERR_INVALID_ARGUMENT` (`w` is NULL).
///
/// # Safety
/// `w` is NULL or a handle from `as_open`.
#[no_mangle]
pub unsafe extern "C" fn as_set_cache_mb(w: *const as_world, mb: usize) -> c_int {
    guard(|| {
        let Some(w) = (unsafe { w.as_ref() }) else { return Err(invalid("world handle is NULL")) };
        w.world.set_cache_mb(mb);
        Ok(())
    })
}

/// Report tile generation on stderr (`on` != 0) or not (0, the default): a line per batch of
/// tiles generated (count, zooms, CPU or GPU, time) by `as_tile`, `as_tiles`, `as_prefetch` and
/// the renders of the world's cameras (`as_render`). The first render at a new place generates
/// the tiles in view, which on the CPU can take minutes. Returns `AS_OK` or
/// `AS_ERR_INVALID_ARGUMENT` (`w` is NULL).
///
/// # Safety
/// `w` is NULL or a handle from `as_open`.
#[no_mangle]
pub unsafe extern "C" fn as_set_verbose(w: *const as_world, on: c_int) -> c_int {
    guard(|| {
        let Some(w) = (unsafe { w.as_ref() }) else { return Err(invalid("world handle is NULL")) };
        w.world.set_verbose(on != 0);
        Ok(())
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

/// The DSM height (ground, canopy, buildings, water surface; metres above the WGS84 ellipsoid) at
/// latitude / longitude `lat_deg`, `lon_deg` into `*height_m`: about what the tiles of the max
/// zoom hold there (evaluated by the generator; no tile is made). For heights above ground.
/// Returns `AS_OK` or a negative `AS_ERR_*` code.
///
/// # Safety
/// `w` is a handle from `as_open`; `height_m` points to a writable double.
#[no_mangle]
pub unsafe extern "C" fn as_surface_height(w: *const as_world, lat_deg: f64, lon_deg: f64, height_m: *mut f64) -> c_int {
    guard(|| {
        let Some(w) = (unsafe { w.as_ref() }) else { return Err(invalid("world handle is NULL")) };
        if height_m.is_null() {
            return Err(invalid("height_m is NULL"));
        }
        let h = w.world.surface_height(lat_deg, lon_deg).map_err(status)?;
        unsafe { height_m.write(h) };
        Ok(())
    })
}

fn backend(b: as_backend) -> Result<Option<Backend>, (c_int, String)> {
    match b {
        AS_BACKEND_DEFAULT => Ok(None),
        AS_BACKEND_AUTO => Ok(Some(Backend::Auto)),
        AS_BACKEND_CPU => Ok(Some(Backend::Cpu)),
        AS_BACKEND_GPU => Ok(Some(Backend::Gpu)),
        _ => Err(invalid(format!("unknown backend {b} (AS_BACKEND_DEFAULT, _AUTO, _CPU or _GPU)"))),
    }
}

/// A camera of a scenario over world `w`.
///
/// `scenario_yaml` is the path of a scenario YAML (as for `terrain run`) or NULL for the default
/// settings: its `render` section (backend, supersample, shading, lighting, atmosphere, ...),
/// `tiles` zoom range and cache size and `cameras` are used; its `world` section is ignored (the
/// world is `w`'s). `camera` picks one of its `cameras` by HDF5 path ("/cam0") or index ("0");
/// NULL is the first one (the default camera, 640 x 512 with a 70 degree field of view on a nadir
/// mount, when there is none). `backend` overrides `render.backend` (`AS_BACKEND_DEFAULT` keeps
/// it).
///
/// The camera keeps the world's store open (`as_close` of the world may come first). Returns NULL
/// on error (see `as_last_error`). Close the camera with `as_camera_close`.
///
/// # Safety
/// `w` is a handle from `as_open`; `scenario_yaml` and `camera` are NULL or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn as_camera_open(w: *const as_world, scenario_yaml: *const c_char, camera: *const c_char, backend: as_backend) -> *mut as_camera {
    let mut out = std::ptr::null_mut();
    guard(|| {
        let Some(w) = (unsafe { w.as_ref() }) else { return Err(invalid("world handle is NULL")) };
        let config = unsafe { path(scenario_yaml, "scenario_yaml") }?;
        let sel =
            if camera.is_null() { None } else { Some(unsafe { CStr::from_ptr(camera) }.to_str().map_err(|_| invalid("camera is not UTF-8"))?.to_string()) };
        let b = self::backend(backend)?;
        let camera = Camera::new(w.world.clone(), config.as_deref(), CameraDef::Scenario(sel), b).map_err(status)?;
        out = Box::into_raw(Box::new(as_camera { camera }));
        Ok(())
    });
    out
}

/// A distortion-free pinhole camera over world `w`: `width` x `height` pixels, horizontal field
/// of view `hfov_deg` (0 < hfov < 180), principal point at the image centre, mounted by `mount`
/// (`AS_MOUNT_FORWARD` or `AS_MOUNT_NADIR`), with the default render settings; `backend`
/// (`AS_BACKEND_DEFAULT` = `AS_BACKEND_AUTO`).
///
/// The camera keeps the world's store open. Returns NULL on error (see `as_last_error`). Close
/// the camera with `as_camera_close`.
///
/// # Safety
/// `w` is a handle from `as_open`.
#[no_mangle]
pub unsafe extern "C" fn as_camera_pinhole(w: *const as_world, width: u32, height: u32, hfov_deg: f64, mount: as_mount, backend: as_backend) -> *mut as_camera {
    let mut out = std::ptr::null_mut();
    guard(|| {
        let Some(w) = (unsafe { w.as_ref() }) else { return Err(invalid("world handle is NULL")) };
        let mount = match mount {
            AS_MOUNT_FORWARD => Mount::Forward,
            AS_MOUNT_NADIR => Mount::Nadir,
            m => return Err(invalid(format!("unknown mount {m} (AS_MOUNT_FORWARD or AS_MOUNT_NADIR)"))),
        };
        let b = self::backend(backend)?;
        let p = Pinhole { mount, ..Pinhole::new(width, height, hfov_deg) };
        let camera = Camera::new(w.world.clone(), None, CameraDef::Pinhole(p), b).map_err(status)?;
        out = Box::into_raw(Box::new(as_camera { camera }));
        Ok(())
    });
    out
}

/// The image size of camera `c` into `*width` and `*height` (either may be NULL). Returns `AS_OK`
/// or `AS_ERR_INVALID_ARGUMENT` (`c` is NULL).
///
/// # Safety
/// `c` is NULL or a camera handle; `width` and `height` are NULL or point to writable `uint32_t`s.
#[no_mangle]
pub unsafe extern "C" fn as_camera_size(c: *const as_camera, width: *mut u32, height: *mut u32) -> c_int {
    guard(|| {
        let Some(c) = (unsafe { c.as_ref() }) else { return Err(invalid("camera handle is NULL")) };
        if !width.is_null() {
            unsafe { width.write(c.camera.width()) };
        }
        if !height.is_null() {
            unsafe { height.write(c.camera.height()) };
        }
        Ok(())
    })
}

/// The backend camera `c` renders on: `AS_BACKEND_CPU` or `AS_BACKEND_GPU` (auto resolved), or
/// `AS_ERR_INVALID_ARGUMENT` if `c` is NULL.
///
/// # Safety
/// `c` is NULL or a camera handle.
#[no_mangle]
pub unsafe extern "C" fn as_camera_backend(c: *const as_camera) -> c_int {
    let mut out = AS_ERR_INVALID_ARGUMENT;
    guard(|| {
        let Some(c) = (unsafe { c.as_ref() }) else { return Err(invalid("camera handle is NULL")) };
        out = match c.camera.backend() {
            Backend::Gpu => AS_BACKEND_GPU,
            _ => AS_BACKEND_CPU,
        } as c_int;
        Ok(())
    });
    out
}

/// Render a frame of camera `c` from `pose` at `unix_time` (UTC, Unix seconds; NAN: the
/// scenario's lighting time). Tiles the view needs are read from the store, or generated and
/// stored when missing. Only the images whose buffer is not NULL are made; each `*_len` is the
/// buffer's length in elements and must be at least:
///
/// - `rgb`: width * height * 3 bytes, row-major (row 0 = image top), sRGB after the camera
///   sensor model (auto exposure converged on the frame, optics, noise, tone curve; no motion
///   blur). The noise is deterministic: the same camera, pose and time give the same image.
/// - `depth`: width * height floats, metres: the z-depth along the optical axis (OpenCV camera
///   frame: x right, y down, z forward), or the range along the pixel ray when the scenario
///   camera's `depth.kind` is `range`; +infinity where there is no terrain (sky).
/// - `landcover`: width * height class ids (as `AS_LAYER_LANDCOVER`), 255 = sky.
///
/// The time places the sun, moon and stars (as the lighting `clock` mode at that instant).
/// Renders of one camera are serialized; different cameras render concurrently. Returns `AS_OK`
/// or a negative `AS_ERR_*` code (`AS_ERR_INVALID_ARGUMENT`: a NULL camera or pose, a pose out
/// of range (|lat| > 90, non-finite values), a non-finite time; `AS_ERR_BUFFER_TOO_SMALL`).
///
/// # Safety
/// `c` is a camera handle; `pose` points to an `as_pose`; each buffer is NULL or points to
/// `*_len` writable elements.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn as_render(
    c: *const as_camera,
    pose: *const as_pose,
    unix_time: f64,
    rgb: *mut u8,
    rgb_len: usize,
    depth: *mut f32,
    depth_len: usize,
    landcover: *mut u8,
    landcover_len: usize,
) -> c_int {
    guard(|| {
        let Some(c) = (unsafe { c.as_ref() }) else { return Err(invalid("camera handle is NULL")) };
        let Some(p) = (unsafe { pose.as_ref() }) else { return Err(invalid("pose is NULL")) };
        let n = c.camera.width() as usize * c.camera.height() as usize;
        for (what, null, len, need) in
            [("rgb", rgb.is_null(), rgb_len, 3 * n), ("depth", depth.is_null(), depth_len, n), ("landcover", landcover.is_null(), landcover_len, n)]
        {
            if !null && len < need {
                return Err((AS_ERR_BUFFER_TOO_SMALL, format!("{what} buffer of {len} elements is too small ({need} needed)")));
            }
        }
        let pose = Pose { lat_deg: p.lat_deg, lon_deg: p.lon_deg, height_m: p.height_m, roll_deg: p.roll_deg, pitch_deg: p.pitch_deg, yaw_deg: p.yaw_deg };
        let want = Outputs { rgb: !rgb.is_null(), depth: !depth.is_null(), landcover: !landcover.is_null() };
        let time = (!unix_time.is_nan()).then_some(unix_time);
        let f = c.camera.render(&pose, time, want).map_err(status)?;
        if let Some(v) = f.rgb {
            unsafe { std::slice::from_raw_parts_mut(rgb, v.len()) }.copy_from_slice(&v);
        }
        if let Some(v) = f.depth {
            unsafe { std::slice::from_raw_parts_mut(depth, v.len()) }.copy_from_slice(&v);
        }
        if let Some(v) = f.landcover {
            unsafe { std::slice::from_raw_parts_mut(landcover, v.len()) }.copy_from_slice(&v);
        }
        Ok(())
    })
}

/// Close a camera. NULL is ignored. The handle must not be in use by another thread, and is
/// invalid afterwards.
///
/// # Safety
/// `c` is NULL or a camera handle not closed yet.
#[no_mangle]
pub unsafe extern "C" fn as_camera_close(c: *mut as_camera) {
    if !c.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(c) })));
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
        assert_eq!(AS_DEFAULT_CACHE_MB, aerialsynth_core::DEFAULT_CACHE_MB);
        assert_eq!(AS_MAX_PREFETCH_TILES, aerialsynth_core::MAX_PREFETCH_TILES);
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
            // many tiles at once: stored (3/5/3), missing, repeated; the bytes of as_tile
            let ids: [u32; 12] = [3, 5, 3, 3, 6, 3, 3, 5, 3, 2, 1, 1];
            let mut many = vec![0u8; 4 * n + 3];
            let rc = as_tiles(w, ids.as_ptr(), 4, AS_LAYER_ELEVATION, many.as_mut_ptr() as *mut c_void, many.len());
            assert_eq!(rc, AS_OK, "{}", last_error());
            for (i, t) in ids.chunks(3).enumerate() {
                assert_eq!(as_tile(w, t[0], t[1], t[2], AS_LAYER_ELEVATION, b.as_mut_ptr() as *mut c_void, n), AS_OK);
                assert!(many[i * n..(i + 1) * n] == b[..n], "tile {i}");
            }
            assert_eq!(many[..n], a[..]);
            let rgb = as_layer_size(AS_LAYER_RGB);
            let mut out = vec![0u8; 2 * rgb];
            let o = out.as_mut_ptr() as *mut c_void;
            assert_eq!(as_tiles(w, ids.as_ptr(), 2, AS_LAYER_RGB, o, 2 * rgb - 1), AS_ERR_BUFFER_TOO_SMALL);
            assert!(last_error().contains("too small"), "{}", last_error());
            let bad: [u32; 6] = [1, 0, 0, 3, 8, 0]; // 1/0/0 valid and missing, 3/8/0 out of range
            assert_eq!(as_tiles(w, bad.as_ptr(), 2, AS_LAYER_RGB, o, 2 * rgb), AS_ERR_INVALID_ARGUMENT);
            assert!(last_error().contains("3/8/0"), "{}", last_error());
            assert_eq!(as_tiles(w, ids.as_ptr(), 2, 99, o, 2 * rgb), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_tiles(w, std::ptr::null(), 2, AS_LAYER_RGB, o, 2 * rgb), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_tiles(w, ids.as_ptr(), 2, AS_LAYER_RGB, std::ptr::null_mut(), 2 * rgb), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_tiles(w, ids.as_ptr(), usize::MAX / 2, AS_LAYER_RGB, o, 2 * rgb), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_tiles(std::ptr::null(), ids.as_ptr(), 2, AS_LAYER_RGB, o, 2 * rgb), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_tiles(w, std::ptr::null(), 0, AS_LAYER_RGB, std::ptr::null_mut(), 0), AS_OK);
            // the cache: off, then on again; the same bytes
            assert_eq!(as_set_cache_mb(w, 0), AS_OK);
            assert_eq!(as_tiles(w, ids.as_ptr(), 4, AS_LAYER_ELEVATION, b.as_mut_ptr() as *mut c_void, n), AS_ERR_BUFFER_TOO_SMALL);
            let mut again = vec![0u8; 4 * n];
            assert_eq!(as_tiles(w, ids.as_ptr(), 4, AS_LAYER_ELEVATION, again.as_mut_ptr() as *mut c_void, 4 * n), AS_OK);
            assert!(again[..] == many[..4 * n]);
            assert_eq!(as_set_cache_mb(w, AS_DEFAULT_CACHE_MB), AS_OK);
            assert_eq!(as_set_cache_mb(std::ptr::null(), 1), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_set_verbose(w, 1), AS_OK);
            assert_eq!(as_set_verbose(w, 0), AS_OK);
            assert_eq!(as_set_verbose(std::ptr::null(), 1), AS_ERR_INVALID_ARGUMENT);
            // prefetch: z0..=2 of a box: 0/0/0, 1/1/0, 2/2/1
            let mut g = 99usize;
            assert_eq!(as_prefetch(w, 44.0, 9.0, 46.0, 11.0, 0, 2, &mut g), AS_OK, "{}", last_error());
            assert_eq!(g, 3);
            // across the antimeridian (from 9 W east to 11 W): 1/0/0 and 1/1/0, which is stored;
            // 1/0/0 is missing (the refused batch did not make it)
            assert_eq!(as_prefetch(w, 44.0, -9.0, 46.0, -11.0, 1, 1, &mut g), AS_OK);
            assert_eq!(g, 1);
            assert_eq!(as_prefetch(w, 44.0, 9.0, 46.0, 11.0, 0, 2, std::ptr::null_mut()), AS_OK);
            assert_eq!(as_prefetch(w, 44.0, 9.0, 46.0, 11.0, 0, 6, &mut g), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_prefetch(w, 44.0, 9.0, 95.0, 11.0, 0, 1, &mut g), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(as_prefetch(std::ptr::null(), 44.0, 9.0, 46.0, 11.0, 0, 1, &mut g), AS_ERR_INVALID_ARGUMENT);
            as_close(w);
            // another seed: refused
            assert!(as_open(tiles.as_ptr(), cfgc.as_ptr(), 3).is_null());
            assert!(last_error().contains("world.seed (2 → 3)"), "{}", last_error());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rendering_through_the_c_api() {
        let dir = std::env::temp_dir().join(format!("aerialsynth-capi-render-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("w.yaml");
        std::fs::write(&cfg, "world: { tile_supersample: 1 }\ntiles: { max_zoom: 8 }\n").unwrap();
        let scn = dir.join("scn.yaml");
        std::fs::write(&scn, "cameras:\n  - path: /down\n    intrinsics: { model: pinhole, width: 24, height: 16, intrinsics: [20, 20, 11.5, 7.5] }\n")
            .unwrap();
        let c = |p: &std::path::Path| CString::new(p.to_str().unwrap()).unwrap();
        let (tiles, cfgc, scnc) = (c(&dir.join("w.h5")), c(&cfg), c(&scn));
        unsafe {
            let w = as_open(tiles.as_ptr(), cfgc.as_ptr(), 1);
            assert!(!w.is_null(), "{}", last_error());
            let mut ground = 0.0;
            assert_eq!(as_surface_height(w, 45.0, 10.0, &mut ground), AS_OK, "{}", last_error());
            assert!(ground.is_finite() && ground.abs() < 9000.0);
            assert_eq!(as_surface_height(w, 95.0, 10.0, &mut ground), AS_ERR_INVALID_ARGUMENT);
            let cam = as_camera_pinhole(w, 32, 24, 60.0, AS_MOUNT_FORWARD, AS_BACKEND_CPU);
            assert!(!cam.is_null(), "{}", last_error());
            let (mut cw, mut ch) = (0u32, 0u32);
            assert_eq!(as_camera_size(cam, &mut cw, &mut ch), AS_OK);
            assert_eq!((cw, ch), (32, 24));
            assert_eq!(as_camera_backend(cam), AS_BACKEND_CPU as c_int);
            let n = 32 * 24;
            let (mut rgb, mut depth, mut lc) = (vec![0u8; 3 * n], vec![0f32; n], vec![0u8; n]);
            let pose = as_pose { lat_deg: 45.0, lon_deg: 10.0, height_m: ground + 2000.0, roll_deg: 0.0, pitch_deg: -45.0, yaw_deg: 30.0 };
            let rc = as_render(cam, &pose, 1.7e9, rgb.as_mut_ptr(), rgb.len(), depth.as_mut_ptr(), depth.len(), lc.as_mut_ptr(), lc.len());
            assert_eq!(rc, AS_OK, "{}", last_error());
            assert!(depth.iter().all(|z| z.is_finite() && *z > 0.0));
            assert!(lc.iter().all(|c| *c < 18) && rgb.iter().any(|v| *v > 0));
            // depth only, the scenario's time (NAN)
            let mut d2 = vec![0f32; n];
            let rc = as_render(cam, &pose, f64::NAN, std::ptr::null_mut(), 0, d2.as_mut_ptr(), n, std::ptr::null_mut(), 0);
            assert_eq!(rc, AS_OK, "{}", last_error());
            assert_eq!(d2, depth);
            // errors
            let r = |p: &as_pose, t: f64, d: &mut [f32]| as_render(cam, p, t, std::ptr::null_mut(), 0, d.as_mut_ptr(), d.len(), std::ptr::null_mut(), 0);
            assert_eq!(r(&pose, 0.0, &mut d2[..n - 1]), AS_ERR_BUFFER_TOO_SMALL);
            assert_eq!(r(&as_pose { lat_deg: -90.5, ..pose }, 0.0, &mut d2), AS_ERR_INVALID_ARGUMENT);
            assert!(last_error().contains("latitude"), "{}", last_error());
            assert_eq!(r(&pose, f64::INFINITY, &mut d2), AS_ERR_INVALID_ARGUMENT);
            assert_eq!(
                as_render(cam, std::ptr::null(), 0.0, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0),
                AS_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                as_render(std::ptr::null(), &pose, 0.0, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0),
                AS_ERR_INVALID_ARGUMENT
            );
            assert!(as_camera_pinhole(w, 32, 24, 60.0, 7, AS_BACKEND_CPU).is_null());
            assert!(as_camera_pinhole(w, 32, 24, 60.0, AS_MOUNT_NADIR, 9).is_null());
            assert!(as_camera_pinhole(w, 0, 24, 60.0, AS_MOUNT_NADIR, AS_BACKEND_CPU).is_null());
            assert!(as_camera_pinhole(std::ptr::null(), 32, 24, 60.0, AS_MOUNT_NADIR, AS_BACKEND_CPU).is_null());
            assert_eq!(as_camera_backend(std::ptr::null()), AS_ERR_INVALID_ARGUMENT);
            // a scenario camera by path; the world closed first: the camera keeps the store open
            let down = as_camera_open(w, scnc.as_ptr(), c"/down".as_ptr(), AS_BACKEND_CPU);
            assert!(!down.is_null(), "{}", last_error());
            assert!(as_camera_open(w, scnc.as_ptr(), c"/up".as_ptr(), AS_BACKEND_CPU).is_null());
            assert!(last_error().contains("no camera /up"), "{}", last_error());
            as_close(w);
            assert_eq!(as_camera_size(down, &mut cw, std::ptr::null_mut()), AS_OK);
            assert_eq!(cw, 24);
            let mut d3 = vec![0f32; 24 * 16];
            let level = as_pose { pitch_deg: 0.0, ..pose };
            assert_eq!(as_render(down, &level, 0.0, std::ptr::null_mut(), 0, d3.as_mut_ptr(), d3.len(), std::ptr::null_mut(), 0), AS_OK, "{}", last_error());
            assert!(d3.iter().all(|z| z.is_finite() && *z > 1000.0), "nadir view from 2 km");
            as_camera_close(down);
            as_camera_close(cam);
            as_camera_close(std::ptr::null_mut());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
