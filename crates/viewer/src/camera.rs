//! The camera view of `terrain view`: a scenario camera, flown live, rendered by the dataset
//! renderer and developed by its sensor model every frame (`render::Renderer`, GPU backend when
//! there is one; `render::sensor::Sensor`): lighting for the date and time (sun, moon,
//! twilight, night lights, stars), atmosphere, shading, auto exposure, noise, optics, tone
//! curve. Motion blur and the ground truth modalities are left out.
//!
//! Three threads: the UI flies the body; a render thread renders the newest pose; a sensor
//! thread develops frame k while frame k + 1 renders. The renderer uses stored tiles only; the
//! tiles its view selects (with their missing ancestors) are generated in the background.

use crate::tiles::Service;
use anyhow::{Context, Result};
use eframe::egui;
use geodesy::tiles::{latlon_to_pixel, tile_for_latlon, TileId};
use glam::DVec2;
use parking_lot::Mutex;
use render::cache::TileCache;
use render::camera::{CameraConfig, Extrinsics, Mount};
use render::lod::{LodParams, Selector, TileOracle};
use render::scenario::{CameraSpec, Scenario};
use render::sensor::Sensor;
use render::{pipeline, Pose};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tilestore::TileStore;

/// What the UI asks the render thread for.
#[derive(Clone)]
struct Request {
    pose: Pose,
    /// scenario time (s): lighting / clock (moved by the time-of-day slider)
    t: f64,
    /// wall-clock time (s) since the start: the sensor's auto exposure
    wall: f64,
    forward: bool,
    dynamic: bool,
    /// deepest level to generate (the viewer's `--max-zoom` / slider)
    max_zoom: u8,
}

/// The latest developed frame.
struct Output {
    rgb: Vec<u8>,
    w: usize,
    h: usize,
    render_ms: f64,
    total_ms: f64,
    exposure: [f64; 3],
    sun_elevation: f64,
    lights: f64,
}

/// A rendered frame on its way to the sensor.
struct Developing {
    radiance: Vec<f32>,
    /// wall-clock time (s): the auto-exposure ODE
    t: f64,
    render_ms: f64,
    started: Instant,
    sun_elevation: f64,
    lights: f64,
}

#[derive(Default)]
struct Shared {
    req: Mutex<Option<Request>>,
    out: Mutex<Option<Output>>,
    /// terrain height under the camera (m above the ellipsoid)
    ground: Mutex<Option<f64>>,
    stop: AtomicBool,
    frames: AtomicU64,
    wanted: AtomicU64,
}

/// The scenario camera at `scale` of its resolution.
fn scaled_camera(c: &CameraConfig, scale: f64) -> CameraConfig {
    if (scale - 1.0).abs() < 1e-9 || c.intrinsics.len() != 4 {
        return c.clone();
    }
    let mut c = c.clone();
    c.width = ((c.width as f64 * scale).round() as u32).max(16);
    c.height = ((c.height as f64 * scale).round() as u32).max(16);
    let f = c.intrinsics.clone();
    c.intrinsics = vec![f[0] * scale, f[1] * scale, (f[2] + 0.5) * scale - 0.5, (f[3] + 0.5) * scale - 0.5];
    c
}

/// Oracle of the tiles the renderer would like: everything to the max zoom exists.
struct WantOracle<'a> {
    store: &'a TileStore,
    max_zoom: u8,
}
impl TileOracle for WantOracle<'_> {
    fn range(&self, id: TileId) -> Option<(f32, f32)> {
        self.store.elev_range(id)
    }
    fn exists(&self, id: TileId) -> bool {
        id.z <= self.max_zoom || self.store.contains(id)
    }
    fn refine_unknown(&self) -> bool {
        false
    }
}

/// Terrain height (DSM, m) at (lat, lon) from the finest stored tile.
fn ground_at(cache: &TileCache, lat: f64, lon: f64, max_z: u8) -> Option<f64> {
    for z in (0..=max_z).rev() {
        let id = tile_for_latlon(lat, lon, z);
        if !cache.store.contains(id) {
            continue;
        }
        let t = cache.get(id)?;
        if t.elevation.is_empty() {
            return None;
        }
        let px = latlon_to_pixel(lat, lon, z, 256) - DVec2::new(id.x as f64, id.y as f64) * 256.0 - DVec2::splat(0.5);
        let (i, j) = (px.x.floor() as isize, px.y.floor() as isize);
        let (fx, fy) = (px.x - i as f64, px.y - j as f64);
        let e = |a: isize, b: isize| t.elev(a, b) as f64;
        let top = e(i, j) + (e(i + 1, j) - e(i, j)) * fx;
        let bot = e(i, j + 1) + (e(i + 1, j + 1) - e(i, j + 1)) * fx;
        return Some(top + (bot - top) * fy);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn render_thread(
    sh: Arc<Shared>,
    scn: Scenario,
    spec: CameraSpec,
    scale: f64,
    ss: u32,
    store: Arc<TileStore>,
    svc: Arc<Service>,
    repaint: egui::Context,
) -> Result<()> {
    let ell = store.meta().ellipsoid();
    let cam_cfg = scaled_camera(&spec.intrinsics, scale);
    let model = cam_cfg.build()?;
    let (w, h) = (model.width() as usize, model.height() as usize);
    let cache = pipeline::tile_cache(&scn, store.clone(), None);
    let mut renderer = pipeline::renderer(&scn, model.clone(), ss, ell, cache.clone());
    renderer.radiance_only = true;
    let rgb_spec = spec.rgb.clone().unwrap_or_default();
    let mut sensor_cfg = rgb_spec.sensor.clone();
    sensor_cfg.motion_blur.enabled = false;
    // the sensor develops frame k on its own thread while frame k + 1 renders
    let (tx, rx) = std::sync::mpsc::sync_channel::<Developing>(1);
    let exposure_time = Arc::new(AtomicU64::new(rgb_spec.sensor.exposure.base_time.to_bits()));
    let developer = {
        let (sh, repaint, exposure_time) = (sh.clone(), repaint.clone(), exposure_time.clone());
        std::thread::Builder::new().name("view-sensor".into()).spawn(move || {
            let mut sensor = Sensor::new(sensor_cfg, w, h);
            for (k, d) in rx.into_iter().enumerate() {
                if !sensor.has_metering() {
                    sensor.meter(&d.radiance); // start converged
                }
                let ex = sensor.exposure_for(d.t);
                let rgb = sensor.develop(&d.radiance, &ex, k as u64);
                sensor.meter(&d.radiance);
                exposure_time.store(ex.time.to_bits(), Ordering::Relaxed);
                *sh.out.lock() = Some(Output {
                    rgb,
                    w,
                    h,
                    render_ms: d.render_ms,
                    total_ms: d.started.elapsed().as_secs_f64() * 1000.0,
                    exposure: [ex.time, ex.gain, ex.ev],
                    sun_elevation: d.sun_elevation,
                    lights: d.lights,
                });
                sh.frames.fetch_add(1, Ordering::Relaxed);
                repaint.request_repaint();
            }
        })?
    };
    let forward_ext = Extrinsics { mount: Mount::Forward, pitch_deg: -15.0, ..Extrinsics::default() };
    let mut last_want = Instant::now() - std::time::Duration::from_secs(10);
    while !sh.stop.load(Ordering::Relaxed) {
        let Some(req) = sh.req.lock().clone() else {
            std::thread::sleep(std::time::Duration::from_millis(5));
            continue;
        };
        let t0 = Instant::now();
        let ext = if req.forward { &forward_ext } else { &spec.extrinsics };
        let cam = req.pose.camera(ext, &ell);
        // (the renderer itself goes no deeper than the scenario's tiles.max_zoom)
        let max_zoom = req.max_zoom.min(scn.tiles.max_zoom);
        *sh.ground.lock() = ground_at(&cache, req.pose.geo.lat, req.pose.geo.lon, max_zoom);
        // what this view would select, generated in the background (ancestors first: the
        // renderer descends into a tile only through its stored parents)
        if req.dynamic && last_want.elapsed().as_secs_f64() > 0.3 {
            last_want = Instant::now();
            let oracle = WantOracle { store: &store, max_zoom };
            let want_params = LodParams { min_zoom: scn.tiles.min_zoom, max_zoom, texel_px: scn.render.texel_px, ..Default::default() };
            let mut want: BTreeSet<TileId> = BTreeSet::new();
            for u in Selector::new(&cam, model.as_ref(), ell, &want_params, &oracle).select() {
                let mut a = u.id;
                while !store.contains(a) {
                    want.insert(a);
                    match a.parent() {
                        Some(p) if p.z >= scn.tiles.min_zoom => a = p,
                        _ => break,
                    }
                }
            }
            let mut v: Vec<TileId> = want.into_iter().collect();
            v.sort_by_key(|t| t.z);
            sh.wanted.store(v.len() as u64, Ordering::Relaxed);
            svc.want(vec![], v);
        }
        let _ = svc.take_results();
        let mut sun = scn.render.lighting.sun_at(req.t, req.pose.geo.lat, req.pose.geo.lon);
        // (the exposure matters to the render for lamp flicker: the latest one)
        sun.exposure = f64::from_bits(exposure_time.load(Ordering::Relaxed));
        let t1 = Instant::now();
        let frame = match renderer.try_render(&cam, &sun) {
            Ok(f) => f,
            Err(e) => {
                drop(tx);
                let _ = developer.join();
                return Err(e);
            }
        };
        let render_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let d = Developing { radiance: frame.radiance, t: req.wall, render_ms, started: t0, sun_elevation: sun.elevation.to_degrees(), lights: sun.lights };
        if tx.send(d).is_err() {
            break;
        }
    }
    drop(tx);
    let _ = developer.join();
    Ok(())
}

/// The live camera view: owns the render and sensor threads.
pub struct CameraView {
    sh: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
    tex: Option<egui::TextureHandle>,
    shown_frame: u64,
    fps: f64,
    fps_mark: (Instant, u64),
    stats: Option<Output>,
    /// the camera's HDF5 path and the rendered size
    pub name: String,
    pub size: [u32; 2],
    /// a forward mount (15° down) instead of the scenario's
    pub forward: bool,
    /// time-of-day offset (h) on the scenario clock
    pub time_offset_h: f64,
    /// the scenario lighting runs on a clock (else the time offset does nothing)
    pub clock: bool,
}

impl CameraView {
    /// Start the threads for camera `spec` of `scn` at `scale` of its resolution.
    #[allow(clippy::too_many_arguments)]
    pub fn start(ctx: &egui::Context, scn: &Scenario, spec: &CameraSpec, scale: f64, ss: u32, store: Arc<TileStore>, svc: Arc<Service>) -> Result<CameraView> {
        let sh = Arc::new(Shared::default());
        let size = {
            let c = scaled_camera(&spec.intrinsics, scale);
            [c.width, c.height]
        };
        let thread = {
            let (sh, ctx, spec, scn) = (sh.clone(), ctx.clone(), spec.clone(), scn.clone());
            std::thread::Builder::new().name("view-render".into()).spawn(move || render_thread(sh, scn, spec, scale, ss, store, svc, ctx))?
        };
        Ok(CameraView {
            sh,
            thread: Some(thread),
            tex: None,
            shown_frame: 0,
            fps: 0.0,
            fps_mark: (Instant::now(), 0),
            stats: None,
            name: spec.path.clone(),
            size,
            forward: false,
            time_offset_h: 0.0,
            clock: scn.render.lighting.mode == render::lighting::SunMode::Clock,
        })
    }

    /// Render `pose` next (`wall`: seconds since the start, for the auto exposure; `dynamic`:
    /// generate the missing tiles it wants, down to `max_zoom`).
    pub fn request(&self, pose: Pose, wall: f64, dynamic: bool, max_zoom: u8) {
        let t = pose.t + self.time_offset_h * 3600.0;
        *self.sh.req.lock() = Some(Request { pose: Pose { t, ..pose }, t, wall, forward: self.forward, dynamic, max_zoom });
    }

    /// Stop rendering until the next request (the view is hidden).
    pub fn pause(&self) {
        *self.sh.req.lock() = None;
    }

    /// Terrain height (DSM, m above the ellipsoid) under the last rendered pose.
    pub fn ground(&self) -> Option<f64> {
        *self.sh.ground.lock()
    }

    /// Draw the latest frame into `rect` (fitted, the camera's aspect kept).
    pub fn show(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let frames = self.sh.frames.load(Ordering::Relaxed);
        if frames != self.shown_frame {
            if let Some(o) = self.sh.out.lock().take() {
                let img = egui::ColorImage::from_rgb([o.w, o.h], &o.rgb);
                match &mut self.tex {
                    Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                    None => self.tex = Some(ui.ctx().load_texture("camera view", img, egui::TextureOptions::LINEAR)),
                }
                self.stats = Some(Output { rgb: Vec::new(), ..o });
            }
            self.shown_frame = frames;
        }
        let el = self.fps_mark.0.elapsed().as_secs_f64();
        if el > 0.5 {
            self.fps = (frames - self.fps_mark.1) as f64 / el;
            self.fps_mark = (Instant::now(), frames);
        }
        ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
        match &self.tex {
            Some(t) => {
                let a = self.size[0] as f32 / self.size[1].max(1) as f32;
                let (mut iw, mut ih) = (rect.width(), rect.width() / a);
                if ih > rect.height() {
                    (iw, ih) = (rect.height() * a, rect.height());
                }
                let r = egui::Rect::from_center_size(rect.center(), egui::vec2(iw, ih));
                ui.painter().image(t.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            }
            None => {
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "rendering the first frame…",
                    egui::FontId::proportional(16.0),
                    egui::Color32::GRAY,
                );
            }
        }
    }

    /// The camera section of the side panel.
    pub fn panel(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new(format!("{} · {} × {} px · the dataset renderer and sensor", self.name, self.size[0], self.size[1])).small().weak());
        ui.checkbox(&mut self.forward, "Look forward (15° down) instead of the scenario's mount");
        ui.add_enabled_ui(self.clock, |ui| {
            ui.add(egui::Slider::new(&mut self.time_offset_h, -12.0..=12.0).text("time of day ± h"))
                .on_disabled_hover_text("the scenario's render.lighting.mode is fixed (no clock)");
        });
        if let Some(o) = &self.stats {
            ui.label(
                egui::RichText::new(format!(
                    "{:.0} frames/s · render {:.0} ms · frame {:.0} ms\nexposure {:.2} ms × gain {:.1} (EV {:.1})\nsun {:+.1}° · lights {:.0}% · tiles wanted {}",
                    self.fps,
                    o.render_ms,
                    o.total_ms,
                    o.exposure[0] * 1000.0,
                    o.exposure[1],
                    o.exposure[2],
                    o.sun_elevation,
                    o.lights * 100.0,
                    self.sh.wanted.load(Ordering::Relaxed)
                ))
                .small()
                .monospace(),
            );
        }
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }
}

impl Drop for CameraView {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            if let Ok(Err(e)) = t.join() {
                eprintln!("camera view: {e:#}");
            }
        }
    }
}

/// The camera the view flies: `path`, else the scenario's first camera with `rgb`, else a
/// forward camera of 960 × 540 px and 75° horizontal field of view.
pub fn pick_camera(scn: &Scenario, path: Option<&str>) -> Result<CameraSpec> {
    if let Some(p) = path {
        return scn.cameras.iter().find(|c| c.path == p).cloned().with_context(|| format!("no camera {p} in the scenario"));
    }
    if let Some(c) = scn.cameras.iter().find(|c| c.rgb.is_some()) {
        return Ok(c.clone());
    }
    let mut c = CameraSpec::example();
    c.intrinsics = CameraConfig::pinhole_hfov(960, 540, 75.0);
    c.extrinsics = Extrinsics { mount: Mount::Forward, pitch_deg: -15.0, ..Extrinsics::default() };
    (c.depth, c.flow) = (None, None);
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scenario_without_an_rgb_camera_gets_a_forward_camera() {
        let scn: Scenario = serde_yaml::from_str("cameras: []").unwrap();
        let c = pick_camera(&scn, None).unwrap();
        assert!(c.rgb.is_some() && c.depth.is_none());
        assert_eq!((c.intrinsics.width, c.intrinsics.height), (960, 540));
        assert!(pick_camera(&scn, Some("/nope")).is_err());
        let half = scaled_camera(&CameraConfig::pinhole_hfov(1280, 720, 80.0), 0.5);
        assert_eq!((half.width, half.height), (640, 360));
    }
}
