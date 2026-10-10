//! Camera rendering: images of a [`World`] from a pose, with the renderer of `terrain run`
//! (`render` crate: LOD tile selection, CPU or GPU rasterizer, sky, sun / moon / stars, the
//! camera sensor model and the depth / land-cover ground truth). Tiles the view needs are read
//! from the world's store, or generated and stored when missing (like `terrain run` with
//! `tiles.lazy`).
//!
//! Conventions (as the CLI's datasets):
//! * pose: geodetic position (degrees, metres above the WGS84 ellipsoid) and body attitude as
//!   aerospace Z-Y-X Euler angles (degrees) of the body (FRD: x forward, y right, z down)
//!   relative to the local NED frame: yaw = heading clockwise from north, pitch nose-up
//!   positive, roll right-wing-down positive. The camera sits on the body by its mount
//!   (scenario `extrinsics`); with the [`Mount::Forward`] mount of a [`Pinhole`] camera the
//!   angles are the camera's own (yaw = heading of the optical axis, pitch = its elevation).
//! * camera frame: OpenCV (x right, y down, z = optical axis); pixel (0, 0) is the centre of
//!   the top-left pixel.
//! * depth: z-depth along the optical axis in metres (or the range along the pixel ray when the
//!   scenario camera's `depth.kind` is `range`), `+inf` where there is no terrain (sky).
//! * land cover: the tiles' class ids at the pixel centres, 255 = sky.
//! * time: UTC as Unix seconds. It places the sun, moon and stars (as the lighting `clock`
//!   mode at that instant); without a time the scenario's lighting applies (`fixed` sun by
//!   default, or its `date` / `time_utc` in `clock` mode).

use crate::{Error, Result, World};
use render::camera::{CameraConfig, Extrinsics};
use render::raster::Renderer;
use render::scenario::{CameraSpec, DepthKind, DepthModality, RgbModality, Scenario};
use render::sensor::{Sensor, SensorSettings};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Largest image width or height (pixels).
pub const MAX_IMAGE_SIZE: u32 = 16384;

/// Where frames are rendered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The GPU when there is a usable one (and it can take the camera model), else the CPU.
    Auto,
    /// The CPU reference renderer.
    Cpu,
    /// The GPU (wgpu): an error when there is no usable GPU.
    Gpu,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }

    /// `"auto"`, `"cpu"` or `"gpu"`.
    pub fn from_name(s: &str) -> Result<Backend> {
        match s {
            "auto" => Ok(Backend::Auto),
            "cpu" => Ok(Backend::Cpu),
            "gpu" => Ok(Backend::Gpu),
            _ => Err(Error::InvalidArgument(format!("unknown backend {s:?} (auto, cpu or gpu)"))),
        }
    }

    fn of(b: render::raster::Backend) -> Backend {
        match b {
            render::raster::Backend::Auto => Backend::Auto,
            render::raster::Backend::Cpu => Backend::Cpu,
            render::raster::Backend::Gpu => Backend::Gpu,
        }
    }

    fn to_render(self) -> render::raster::Backend {
        match self {
            Backend::Auto => render::raster::Backend::Auto,
            Backend::Cpu => render::raster::Backend::Cpu,
            Backend::Gpu => render::raster::Backend::Gpu,
        }
    }
}

/// How a [`Pinhole`] camera sits on the body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mount {
    /// Optical axis = body forward, image top = body up: the pose's angles are the camera's
    /// (yaw = heading of the optical axis, pitch = its elevation, negative looks down).
    #[default]
    Forward,
    /// Optical axis = body down, image top = body forward (the scenario default mount): level
    /// flight looks straight down.
    Nadir,
}

impl Mount {
    pub fn from_name(s: &str) -> Result<Mount> {
        match s {
            "forward" => Ok(Mount::Forward),
            "nadir" => Ok(Mount::Nadir),
            _ => Err(Error::InvalidArgument(format!("unknown mount {s:?} (forward or nadir)"))),
        }
    }
}

/// A distortion-free pinhole camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pinhole {
    pub width: u32,
    pub height: u32,
    /// Horizontal field of view (degrees, 0 < hfov < 180); square pixels.
    pub hfov_deg: f64,
    /// Principal point (cx, cy) in pixels (OpenCV: (0, 0) = centre of the top-left pixel);
    /// `None`: the image centre ((width - 1) / 2, (height - 1) / 2).
    pub principal_point: Option<(f64, f64)>,
    pub mount: Mount,
}

impl Pinhole {
    pub fn new(width: u32, height: u32, hfov_deg: f64) -> Pinhole {
        Pinhole { width, height, hfov_deg, principal_point: None, mount: Mount::Forward }
    }

    fn spec(&self) -> Result<CameraSpec> {
        check_size(self.width, self.height)?;
        if !(self.hfov_deg.is_finite() && self.hfov_deg > 0.0 && self.hfov_deg < 180.0) {
            return Err(Error::InvalidArgument(format!("hfov {}° must be in (0, 180)", self.hfov_deg)));
        }
        let mut intrinsics = CameraConfig::pinhole_hfov(self.width, self.height, self.hfov_deg);
        if let Some((cx, cy)) = self.principal_point {
            if !(cx.is_finite() && cy.is_finite()) {
                return Err(Error::InvalidArgument(format!("principal point ({cx}, {cy}) must be finite")));
            }
            (intrinsics.intrinsics[2], intrinsics.intrinsics[3]) = (cx, cy);
        }
        let mount = match self.mount {
            Mount::Forward => render::camera::Mount::Forward,
            Mount::Nadir => render::camera::Mount::Nadir,
        };
        Ok(CameraSpec {
            intrinsics,
            extrinsics: Extrinsics { mount, ..Default::default() },
            rgb: Some(RgbModality::default()),
            depth: Some(DepthModality::default()),
            flow: None,
            landcover: None,
            events: None,
            stars: None,
            ..CameraSpec::example()
        })
    }
}

fn check_size(w: u32, h: u32) -> Result<()> {
    if w == 0 || h == 0 || w > MAX_IMAGE_SIZE || h > MAX_IMAGE_SIZE {
        return Err(Error::InvalidArgument(format!("image size {w} x {h}: width and height must be in 1..={MAX_IMAGE_SIZE}")));
    }
    Ok(())
}

/// Which camera a [`Camera`] renders.
#[derive(Clone, Debug)]
pub enum CameraDef {
    /// A pinhole camera (with the scenario's render settings, if one is given).
    Pinhole(Pinhole),
    /// A camera of the scenario's `cameras` list: by HDF5 path (`"/cam0"`) or index (`"0"`);
    /// `None` is the first one (the default camera, 640 x 512 with a 70° field of view, nadir
    /// mount, when the scenario has none).
    Scenario(Option<String>),
}

/// Camera position and body attitude (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    /// Latitude (degrees, -90 ..= 90).
    pub lat_deg: f64,
    /// Longitude (degrees, east positive).
    pub lon_deg: f64,
    /// Height above the WGS84 ellipsoid (metres).
    pub height_m: f64,
    pub roll_deg: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
}

impl Pose {
    fn check(&self) -> Result<()> {
        let v = [self.lat_deg, self.lon_deg, self.height_m, self.roll_deg, self.pitch_deg, self.yaw_deg];
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidArgument(format!("pose {v:?}: every value must be finite")));
        }
        if self.lat_deg.abs() > 90.0 {
            return Err(Error::InvalidArgument(format!("latitude {}° out of range -90 ..= 90", self.lat_deg)));
        }
        if !(-1e5..=1e8).contains(&self.height_m) {
            return Err(Error::InvalidArgument(format!("height {} m out of range -1e5 ..= 1e8", self.height_m)));
        }
        Ok(())
    }

    /// Seed of the sensor noise of a render at this pose and time (FNV-1a of the bits).
    fn noise_index(&self, unix: Option<f64>) -> u64 {
        let v = [self.lat_deg, self.lon_deg, self.height_m, self.roll_deg, self.pitch_deg, self.yaw_deg, unix.unwrap_or(f64::NAN)];
        v.iter().flat_map(|x| x.to_bits().to_le_bytes()).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
    }
}

/// The images to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outputs {
    pub rgb: bool,
    pub depth: bool,
    pub landcover: bool,
}

impl Default for Outputs {
    fn default() -> Self {
        Outputs { rgb: true, depth: false, landcover: false }
    }
}

/// Exposure of an RGB image (the sensor's auto exposure, converged on the frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exposure {
    /// exposure time (s)
    pub time: f64,
    /// analogue gain (>= 1)
    pub gain: f64,
    /// exposure value relative to the base exposure
    pub ev: f64,
}

/// A rendered frame: row-major images (row 0 = image top), `width` x `height` pixels.
#[derive(Clone, Debug)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// sRGB u8 x 3 after the sensor model (exposure, optics, noise, tone curve)
    pub rgb: Option<Vec<u8>>,
    /// metres (z-depth, or range: [`Camera::depth_kind_range`]), +inf = sky
    pub depth: Option<Vec<f32>>,
    /// class ids, 255 = sky
    pub landcover: Option<Vec<u8>>,
    pub exposure: Option<Exposure>,
    /// Camera centre in ECEF (m).
    pub position_ecef: [f64; 3],
    /// Rotation camera → ECEF, row-major (columns = camera axes in ECEF).
    pub r_ecef_cam: [[f64; 3]; 3],
    /// UTC of the lighting (Unix seconds): the time asked for, or the scenario's.
    pub unix_time: f64,
    /// Sun azimuth (clockwise from north) and elevation at the camera (degrees).
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
}

/// A camera over a world: renders frames from poses.
///
/// `Camera` is `Send + Sync`. Renders of one camera are serialized (a mutex; the renderer itself
/// runs in parallel on the CPU or on the GPU); several cameras, also of the same world, render
/// concurrently (GPU work is serialized by the device). A camera keeps its world's store open
/// (it holds the [`World`]): the tiles file is closed when the world and all its cameras are
/// dropped.
pub struct Camera {
    /// (before `world`: the renderer's tile cache shares the world's store)
    renderer: Mutex<Renderer>,
    spec: CameraSpec,
    scn: Scenario,
    sensor: SensorSettings,
    supersample: u32,
    width: u32,
    height: u32,
    world: Arc<World>,
}

const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<Camera>()
};

impl Camera {
    /// A camera over `world`. `config` is a scenario YAML whose `render` section (backend,
    /// supersample, shading, lighting, atmosphere, ...), `tiles` zoom range and cache size and
    /// `cameras` are used; its `world` section is ignored (the world is `world`'s). `None`: the
    /// default settings. `backend` overrides `render.backend`.
    pub fn new(world: Arc<World>, config: Option<&Path>, def: CameraDef, backend: Option<Backend>) -> Result<Camera> {
        let mut scn = match config {
            None => Scenario::default(),
            Some(p) => load_scenario(p)?,
        };
        let spec = match def {
            CameraDef::Pinhole(p) => p.spec()?,
            CameraDef::Scenario(sel) => select(&scn, sel.as_deref())?,
        };
        let model = spec.intrinsics.build().map_err(|e| Error::InvalidArgument(format!("camera {}: {e:#}", spec.path)))?;
        check_size(model.width(), model.height())?;
        if let Some(b) = backend {
            scn.render.backend = b.to_render();
        }
        let rgb = spec.rgb.clone().unwrap_or_default();
        let supersample = rgb.supersample.unwrap_or(scn.render.supersample).max(1);
        if supersample > 9 {
            return Err(Error::InvalidArgument(format!("camera {}: supersample {supersample} > 9", spec.path)));
        }
        if scn.render.backend == render::raster::Backend::Gpu {
            render::gpu::supports(&*model, supersample).map_err(|e| Error::Failed(anyhow::anyhow!("render backend gpu: {e}")))?;
        }
        // the world's zoom limit; tiles missing from the store are generated and stored
        scn.tiles.max_zoom = scn.tiles.max_zoom.min(world.max_zoom() as u8);
        scn.tiles.min_zoom = scn.tiles.min_zoom.min(scn.tiles.max_zoom);
        scn.tiles.lazy = true;
        let ell = world.store.meta().ellipsoid();
        let cache = render::pipeline::tile_cache(&scn, world.store.clone(), Some(world.gen.clone()));
        cache.set_log(Some(world.log()));
        let renderer = render::pipeline::renderer(&scn, model.clone(), supersample, ell, cache);
        let mut sensor = rgb.sensor;
        sensor.noise.seed ^= spec.seed_mix(); // as `terrain run`
        Ok(Camera { renderer: Mutex::new(renderer), spec, scn, sensor, supersample, width: model.width(), height: model.height(), world })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The backend frames are rendered on: [`Backend::Cpu`] or [`Backend::Gpu`] (`auto`
    /// resolved).
    pub fn backend(&self) -> Backend {
        Backend::of(self.lock().settings.backend)
    }

    /// Supersampling per axis of the RGB image.
    pub fn supersample(&self) -> u32 {
        self.supersample
    }

    /// The camera's HDF5 path in the scenario (`"/cam0"` for a pinhole camera).
    pub fn path(&self) -> &str {
        &self.spec.path
    }

    /// Is the depth the range along the pixel ray (scenario `depth.kind: range`) rather than the
    /// z-depth?
    pub fn depth_kind_range(&self) -> bool {
        self.spec.depth.as_ref().is_some_and(|d| d.kind == DepthKind::Range)
    }

    /// The camera model (camera YAML schema: `pinhole`, `kannala_brandt`, ...) and its
    /// `intrinsics` and `distortion` values (pinhole: fx, fy, cx, cy and k1, k2, p1, p2).
    pub fn model(&self) -> (&str, &[f64], &[f64]) {
        let c = &self.spec.intrinsics;
        (&c.model, &c.intrinsics, &c.distortion)
    }

    /// The world the camera renders.
    pub fn world(&self) -> &Arc<World> {
        &self.world
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Renderer> {
        self.renderer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Render a frame from `pose` at `unix_time` (UTC, Unix seconds; `None`: the scenario's
    /// lighting time). Only the `want`ed images are made.
    ///
    /// The RGB image is that of the first frame of a `terrain run` sequence at this pose: the
    /// auto exposure converged on the frame, no motion blur; the sensor noise is deterministic
    /// (a function of the camera's noise seed, the pose and the time), so the same pose and time
    /// give the same image.
    pub fn render(&self, pose: &Pose, unix_time: Option<f64>, want: Outputs) -> Result<Frame> {
        pose.check()?;
        if let Some(t) = unix_time {
            // (±1e11 s: ±3000 years)
            if !(t.is_finite() && t.abs() < 1e11) {
                return Err(Error::InvalidArgument(format!("time {t}: finite Unix seconds expected")));
            }
        }
        if (want.depth || want.landcover) && self.supersample.is_multiple_of(2) {
            return Err(Error::InvalidArgument(format!(
                "camera {}: supersample {} is even; depth and land cover need an odd supersample (a sample at the pixel centre)",
                self.spec.path, self.supersample
            )));
        }
        let ell = self.world.store.meta().ellipsoid();
        let body = render::Pose {
            t: 0.0,
            geo: geodesy::Geodetic::from_deg(pose.lat_deg, pose.lon_deg, pose.height_m),
            q_ned_body: geodesy::euler_zyx_to_quat(pose.yaw_deg.to_radians(), pose.pitch_deg.to_radians(), pose.roll_deg.to_radians()),
        };
        let cam = body.camera(&self.spec.extrinsics, &ell);
        let lighting = &self.scn.render.lighting;
        let mut sun = match unix_time {
            Some(t) => lighting.sun_at_utc(t, body.geo.lat, body.geo.lon),
            None => lighting.sun_at(0.0, body.geo.lat, body.geo.lon),
        };
        // the exposure of a sequence's first frame (lamp flicker is averaged over it)
        sun.exposure = self.sensor.exposure.base_time;
        let frame = {
            let mut r = self.lock();
            r.geometry_only = !want.rgb;
            r.radiance_only = want.rgb && !want.depth && !want.landcover;
            r.stars_in_render = true;
            r.try_render(&cam, &sun).map_err(|e| Error::Failed(e.context("rendering")))?
        };
        let (w, h) = (self.width as usize, self.height as usize);
        let (rgb, exposure) = if want.rgb {
            let mut s = Sensor::new(self.sensor.clone(), w, h);
            s.meter(&frame.radiance);
            let ex = s.exposure_for(0.0);
            (Some(s.develop(&frame.radiance, &ex, pose.noise_index(unix_time))), Some(Exposure { time: ex.time, gain: ex.gain, ev: ex.ev }))
        } else {
            (None, None)
        };
        let depth = match (want.depth, self.depth_kind_range()) {
            (false, _) => None,
            (true, true) => Some(frame.points.iter().map(|q| q.map(|q| (q - cam.pos).length() as f32).unwrap_or(f32::INFINITY)).collect()),
            (true, false) => Some(frame.depth),
        };
        let landcover = want.landcover.then_some(frame.landcover);
        let sizes = [
            ("rgb", rgb.as_ref().map(|v| v.len()), 3 * w * h),
            ("depth", depth.as_ref().map(Vec::len), w * h),
            ("land cover", landcover.as_ref().map(Vec::len), w * h),
        ];
        for (what, n, want) in sizes {
            if n.is_some_and(|n| n != want) {
                return Err(Error::Failed(anyhow::anyhow!("rendering: the {what} image has {} values, expected {want}", n.unwrap_or(0))));
            }
        }
        let r = cam.r_ecef_cam.transpose().to_cols_array_2d(); // rows of r_ecef_cam
        Ok(Frame {
            width: self.width,
            height: self.height,
            rgb,
            depth,
            landcover,
            exposure,
            position_ecef: cam.pos.to_array(),
            r_ecef_cam: r,
            unix_time: sun.unix,
            sun_azimuth_deg: sun.azimuth.to_degrees(),
            sun_elevation_deg: sun.elevation.to_degrees(),
        })
    }
}

/// A scenario YAML (validated); its world section is not used.
fn load_scenario(p: &Path) -> Result<Scenario> {
    let s = std::fs::read_to_string(p).map_err(|source| Error::Io { context: format!("reading {}", p.display()), source })?;
    if s.trim().is_empty() {
        return Ok(Scenario { cameras: vec![], ..Scenario::default() });
    }
    let scn: Scenario = serde_yaml::from_str(&s).map_err(|e| Error::Failed(anyhow::anyhow!("parsing {}: {e}", p.display())))?;
    // (`backend: gpu` is checked for the chosen camera, unless overridden)
    let mut check = scn.clone();
    check.render.backend = render::raster::Backend::Auto;
    check.validate().map_err(|e| Error::Failed(e.context(format!("checking {}", p.display()))))?;
    Ok(scn)
}

/// A camera of the scenario by HDF5 path or index (`None`: the first; the default camera when
/// there is none).
fn select(scn: &Scenario, sel: Option<&str>) -> Result<CameraSpec> {
    let paths = || scn.cameras.iter().map(|c| c.path.as_str()).collect::<Vec<_>>().join(", ");
    match sel {
        None => Ok(scn.cameras.first().cloned().unwrap_or_else(CameraSpec::example)),
        Some(s) if s.starts_with('/') => {
            let want = s.trim_end_matches('/');
            scn.cameras
                .iter()
                .find(|c| c.path.trim_end_matches('/') == want)
                .cloned()
                .ok_or_else(|| Error::InvalidArgument(format!("no camera {s} in the scenario (cameras: {})", paths())))
        }
        Some(s) => {
            match s.parse::<usize>() {
                Ok(i) => scn.cameras.get(i).cloned().ok_or_else(|| {
                    Error::InvalidArgument(format!("camera index {i} out of range: the scenario has {} cameras ({})", scn.cameras.len(), paths()))
                }),
                Err(_) => Err(Error::InvalidArgument(format!("camera {s:?}: an HDF5 path such as /cam0 or an index expected"))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TempDir;

    /// A cheap world (one sample per pixel) up to zoom `max_zoom`.
    fn world(d: &TempDir, name: &str, max_zoom: u32) -> Arc<World> {
        let cfg = d.0.join(format!("{name}.yaml"));
        std::fs::write(&cfg, format!("world:\n  tile_supersample: 1\ntiles:\n  max_zoom: {max_zoom}\n")).unwrap();
        Arc::new(World::open(&d.0.join(format!("{name}.h5")), Some(&cfg), Some(3)).unwrap())
    }

    /// A small, cheap camera on the CPU.
    fn cpu_cam(w: &Arc<World>, mount: Mount) -> Camera {
        let p = Pinhole { mount, ..Pinhole::new(80, 60, 70.0) };
        Camera::new(w.clone(), None, CameraDef::Pinhole(p), Some(Backend::Cpu)).unwrap()
    }

    const LAT: f64 = 39.9;
    const LON: f64 = 32.8;

    /// A pose 3 km above the surface, looking 60° down (forward mount).
    fn pose(w: &World) -> Pose {
        let g = w.surface_height(LAT, LON).unwrap();
        Pose { lat_deg: LAT, lon_deg: LON, height_m: g + 3000.0, pitch_deg: -60.0, ..Default::default() }
    }

    const ALL: Outputs = Outputs { rgb: true, depth: true, landcover: true };
    /// 2026-06-21 08:30 UTC
    const T: f64 = 1_782_030_600.0;

    /// Rendering on the CPU. One world for every check: CPU tile generation is the slow part
    /// (~1 s per tile), and the views share tiles.
    #[test]
    fn renders_on_the_cpu() {
        let d = TempDir::new("render");
        let w = world(&d, "w", 7);
        frames(&w);
        sky_and_time(&w);
        scenario_cameras(&w, &d);
    }

    fn frames(w: &Arc<World>) {
        let cam = cpu_cam(w, Mount::Forward);
        assert_eq!((cam.width(), cam.height(), cam.backend(), cam.supersample()), (80, 60, Backend::Cpu, 3));
        let (model, k, _) = cam.model();
        assert_eq!((model, k.len()), ("pinhole", 4));
        assert!(w.store.is_empty());
        let p = pose(w);
        let f = cam.render(&p, Some(T), ALL).unwrap();
        assert!(!w.store.is_empty(), "the tiles in view were generated and stored");
        let (rgb, depth, lc) = (f.rgb.as_ref().unwrap(), f.depth.as_ref().unwrap(), f.landcover.as_ref().unwrap());
        assert_eq!((rgb.len(), depth.len(), lc.len()), (80 * 60 * 3, 80 * 60, 80 * 60));
        // looking 60° down from 3 km: terrain everywhere, at a few km
        assert!(depth.iter().all(|z| z.is_finite() && *z > 500.0 && *z < 20_000.0), "{:?}", &depth[..8]);
        assert!(lc.iter().all(|c| *c < 18));
        let mean = rgb.iter().map(|v| *v as f64).sum::<f64>() / rgb.len() as f64;
        assert!(mean > 20.0 && mean < 235.0, "mean {mean}");
        assert!(f.exposure.unwrap().time > 0.0);
        assert!(f.sun_elevation_deg > 30.0, "{}", f.sun_elevation_deg);
        assert_eq!(f.unix_time, T);
        // the camera centre is the pose's position; the optical axis points 60° down
        let ecef = geodesy::geodetic2ecef(geodesy::Geodetic::from_deg(p.lat_deg, p.lon_deg, p.height_m), &w.store.meta().ellipsoid());
        assert!((geodesy::DVec3::from_array(f.position_ecef) - ecef).length() < 1e-6);
        let up = ecef.normalize();
        let axis = geodesy::DVec3::new(f.r_ecef_cam[0][2], f.r_ecef_cam[1][2], f.r_ecef_cam[2][2]);
        assert!((axis.dot(up) + 60f64.to_radians().sin()).abs() < 0.01, "{}", axis.dot(up));
        // deterministic: the same pose and time give the same images
        let g = cam.render(&p, Some(T), ALL).unwrap();
        assert_eq!((&g.rgb, &g.depth, &g.landcover), (&f.rgb, &f.depth, &f.landcover));
        // only what is asked for
        let only = cam.render(&p, Some(T), Outputs { rgb: false, depth: true, landcover: false }).unwrap();
        assert!(only.rgb.is_none() && only.landcover.is_none() && only.exposure.is_none());
        assert_eq!(only.depth, f.depth);
        // a nadir mount looks straight down when level: z-depth ≈ the height above the surface
        let nadir = cpu_cam(w, Mount::Nadir);
        let level = Pose { pitch_deg: 0.0, yaw_deg: 123.0, ..p };
        let f = nadir.render(&level, Some(T), Outputs { rgb: false, depth: true, landcover: false }).unwrap();
        let centre = f.depth.unwrap()[30 * 80 + 40];
        assert!((centre - 3000.0).abs() < 800.0, "{centre}");
    }

    fn sky_and_time(w: &Arc<World>) {
        let cam = cpu_cam(w, Mount::Forward);
        let mut p = pose(w);
        p.pitch_deg = 60.0; // looking up: sky only
        let f = cam.render(&p, Some(T), ALL).unwrap();
        assert!(f.depth.as_ref().unwrap().iter().all(|z| *z == f32::INFINITY));
        assert!(f.landcover.as_ref().unwrap().iter().all(|c| *c == 255));
        // the night sky is darker than the day sky (midnight local time): a higher exposure
        let night = cam.render(&p, Some(T + 13.5 * 3600.0), Outputs::default()).unwrap();
        assert!(night.sun_elevation_deg < -10.0);
        let ev = |f: &Frame| f.exposure.unwrap().ev;
        assert!(ev(&night) > ev(&f) + 5.0, "night ev {} day ev {}", ev(&night), ev(&f));
        // without a time: the scenario's lighting (default: fixed sun at 52° elevation)
        let fixed = cam.render(&p, None, Outputs::default()).unwrap();
        assert!((fixed.sun_elevation_deg - 52.0).abs() < 1e-9);
    }

    fn scenario_cameras(w: &Arc<World>, d: &TempDir) {
        let scn = d.0.join("scn.yaml");
        std::fs::write(
            &scn,
            "world: { seed: 99 }\n\
             render: { supersample: 1, backend: cpu, lighting: { mode: clock, date: 2026-03-20, time_utc: '09:49:00' } }\n\
             cameras:\n\
             \x20 - path: /left\n\
             \x20   intrinsics: { model: pinhole, width: 48, height: 32, intrinsics: [40, 40, 23.5, 15.5] }\n\
             \x20   rgb: {}\n\
             \x20 - path: /down\n\
             \x20   intrinsics: { model: kannala_brandt, width: 40, height: 40, intrinsics: [15, 15, 19.5, 19.5], distortion: [0, 0, 0, 0] }\n\
             \x20   depth: { kind: range }\n",
        )
        .unwrap();
        let open = |sel: Option<&str>| Camera::new(w.clone(), Some(&scn), CameraDef::Scenario(sel.map(String::from)), None);
        let first = open(None).unwrap();
        assert_eq!((first.path(), first.width(), first.height(), first.backend(), first.supersample()), ("/left", 48, 32, Backend::Cpu, 1));
        assert!(!first.depth_kind_range());
        let down = open(Some("/down")).unwrap();
        assert_eq!((down.path(), down.width(), down.model().0), ("/down", 40, "kannala_brandt"));
        assert!(down.depth_kind_range());
        assert_eq!(open(Some("1")).unwrap().path(), "/down");
        for bad in ["/cam9", "2", "left"] {
            assert!(matches!(open(Some(bad)), Err(Error::InvalidArgument(_))), "{bad}");
        }
        // the scenario's lighting (clock mode, about local noon at the equinox): sun at 90° - lat
        let p = Pose { pitch_deg: 0.0, ..pose(w) };
        let f = down.render(&p, None, Outputs { rgb: false, depth: true, landcover: false }).unwrap();
        assert!((f.sun_elevation_deg - (90.0 - LAT)).abs() < 1.5, "{}", f.sun_elevation_deg);
        // range depth (nadir mount; a fisheye): about the height above the surface at the
        // centre, much more towards the edges
        let depth = f.depth.unwrap();
        let centre = depth[20 * 40 + 20];
        assert!((centre - 3000.0).abs() < 800.0, "{centre}");
        assert!(depth[20 * 40 + 1] > 2.0 * centre, "{}", depth[20 * 40 + 1]);
        // a pinhole camera with the scenario's render settings
        let pin = Camera::new(w.clone(), Some(&scn), CameraDef::Pinhole(Pinhole::new(32, 24, 60.0)), None).unwrap();
        assert_eq!((pin.supersample(), pin.backend()), (1, Backend::Cpu));
        // a missing file, an invalid scenario
        let missing = Camera::new(w.clone(), Some(&d.0.join("missing.yaml")), CameraDef::Scenario(None), None);
        assert!(matches!(missing, Err(Error::Io { .. })));
        std::fs::write(&scn, "render: { no_such_setting: 1 }\n").unwrap();
        assert!(open(None).err().unwrap().to_string().contains("no_such_setting"));
    }

    /// The same call gives the same image whether its tiles are generated (the first render) or
    /// read from the store (later ones), and whatever else the store holds; the first render
    /// generates the zooms the view needs, not the world's max zoom (with unknown elevation
    /// ranges the camera, 3 km up, used to be inside the tiles' assumed volumes: refined to the
    /// max zoom around it).
    #[test]
    fn renders_do_not_depend_on_what_the_store_held() {
        let d = TempDir::new("same");
        let small = |w: &Arc<World>| Camera::new(w.clone(), None, CameraDef::Pinhole(Pinhole::new(48, 32, 70.0)), Some(Backend::Cpu)).unwrap();
        let what = Outputs { rgb: true, depth: true, landcover: true };
        let a = world(&d, "a", 16);
        let p = pose(&a); // 3 km up, 60° down
        let first = small(&a).render(&p, Some(T), what).unwrap();
        // (z14 right below the camera: the level of detail measures from the tiles' bounding
        // spheres)
        let zmax = a.store.zooms().into_iter().max().unwrap();
        assert!(zmax <= 14, "zoom {zmax} generated");
        let n = a.store.len();
        let again = small(&a).render(&p, Some(T), what).unwrap();
        assert_eq!(a.store.len(), n, "the second render generated tiles");
        let same = |f: &Frame, g: &Frame| {
            assert_eq!(f.rgb, g.rgb);
            assert_eq!(
                f.depth.as_ref().map(|d| d.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                g.depth.as_ref().map(|d| d.iter().map(|v| v.to_bits()).collect())
            );
            assert_eq!(f.landcover, g.landcover);
            assert_eq!(f.exposure.map(|e| e.time.to_bits()), g.exposure.map(|e| e.time.to_bits()));
        };
        same(&first, &again);
        // a store that also holds other tiles around the view (finer ones too): the same image
        drop(a); // (closes the store)
        std::fs::copy(d.0.join("a.h5"), d.0.join("b.h5")).unwrap();
        let b = Arc::new(World::open(&d.0.join("b.h5"), Some(&d.0.join("a.yaml")), Some(3)).unwrap());
        assert!(b.prefetch([LAT - 0.015, LON - 0.015, LAT + 0.015, LON + 0.015], 12, 15).unwrap() > 0);
        same(&first, &small(&b).render(&p, Some(T), what).unwrap());
    }

    #[test]
    fn bad_arguments_are_rejected() {
        let d = TempDir::new("render-args");
        let w = world(&d, "w", 6);
        let cam = cpu_cam(&w, Mount::Nadir);
        let ok = Pose { lat_deg: 10.0, lon_deg: 20.0, height_m: 5000.0, ..Default::default() };
        for bad in [
            Pose { lat_deg: 91.0, ..ok },
            Pose { lat_deg: f64::NAN, ..ok },
            Pose { lon_deg: f64::INFINITY, ..ok },
            Pose { height_m: 1e9, ..ok },
            Pose { yaw_deg: f64::NAN, ..ok },
        ] {
            assert!(matches!(cam.render(&bad, None, Outputs::default()), Err(Error::InvalidArgument(_))), "{bad:?}");
        }
        assert!(matches!(cam.render(&ok, Some(f64::NAN), Outputs::default()), Err(Error::InvalidArgument(_))));
        let pin = |p: Pinhole| Camera::new(w.clone(), None, CameraDef::Pinhole(p), Some(Backend::Cpu)).err();
        for p in [Pinhole::new(0, 10, 60.0), Pinhole::new(10, MAX_IMAGE_SIZE + 1, 60.0), Pinhole::new(10, 10, 180.0), Pinhole::new(10, 10, f64::NAN)] {
            assert!(matches!(pin(p), Some(Error::InvalidArgument(_))), "{p:?}");
        }
        assert!(matches!(pin(Pinhole { principal_point: Some((f64::NAN, 1.0)), ..Pinhole::new(10, 10, 60.0) }), Some(Error::InvalidArgument(_))));
        assert!(Backend::from_name("vulkan").is_err() && Mount::from_name("up").is_err());
        assert_eq!((Backend::from_name("gpu").unwrap(), Mount::from_name("nadir").unwrap()), (Backend::Gpu, Mount::Nadir));
        // the camera holds the world: the store stays open until the camera is dropped
        let path = w.path().to_path_buf();
        drop(w);
        assert!(World::open(&path, None, Some(3)).err().unwrap().to_string().contains("already open"));
        drop(cam);
        let cfg = d.0.join("w.yaml");
        World::open(&path, Some(&cfg), Some(3)).unwrap();
    }

    /// GPU and CPU renders agree roughly (skipped without a usable GPU).
    #[test]
    fn gpu_matches_cpu() {
        let d = TempDir::new("render-gpu");
        let w = world(&d, "w", 9);
        let p = Pinhole::new(96, 72, 70.0);
        let gpu = match Camera::new(w.clone(), None, CameraDef::Pinhole(p), Some(Backend::Gpu)) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no usable GPU, skipped: {e}");
                return;
            }
        };
        assert_eq!(gpu.backend(), Backend::Gpu);
        let cpu = Camera::new(w.clone(), None, CameraDef::Pinhole(p), Some(Backend::Cpu)).unwrap();
        let pose = pose(&w);
        let (a, b) = (cpu.render(&pose, Some(T), ALL).unwrap(), gpu.render(&pose, Some(T), ALL).unwrap());
        let (ra, rb) = (a.rgb.unwrap(), b.rgb.unwrap());
        let mad = ra.iter().zip(&rb).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / ra.len() as f64;
        assert!(mad < 6.0, "mean |rgb difference| {mad}");
        let (da, db) = (a.depth.unwrap(), b.depth.unwrap());
        let rel = da.iter().zip(&db).map(|(x, y)| ((x - y).abs() / x) as f64).sum::<f64>() / da.len() as f64;
        assert!(rel < 0.01, "mean relative depth difference {rel}");
        let (la, lb) = (a.landcover.unwrap(), b.landcover.unwrap());
        let same = la.iter().zip(&lb).filter(|(x, y)| x == y).count() as f64 / la.len() as f64;
        assert!(same > 0.9, "land cover agrees on {same}");
    }
}
