//! `terrain live`: the dataset renderer in realtime. A camera of the scenario is flown with the
//! flight controls and rendered by the actual pipeline every frame: `render::Renderer` (GPU
//! backend when there is one) with the scenario's lighting (sun, moon, twilight, night lights,
//! stars), atmosphere and shading, developed by the camera's sensor model (auto exposure, noise,
//! optics, tone curve). Motion blur and the ground truth modalities are left out.
//!
//! The renderer runs on its own thread with the latest pose (the controls stay smooth when a
//! frame takes longer). It only uses stored tiles; the tiles its view selects are generated in the
//! background (with `--dynamic`) and show up as they are written.

use crate::fly::{FlyCam, FlyInput, FlyMode};
use crate::tiles::Service;
use anyhow::{bail, Context, Result};
use eframe::egui;
use geodesy::tiles::{latlon_to_pixel, tile_for_latlon, TileId};
use geodesy::{euler_zyx_to_quat, geodetic2ecef, Ellipsoid, Geodetic};
use glam::{DVec2, DVec3};
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
use terragen::Generator;
use tilestore::TileStore;

/// Options of `terrain live`.
#[derive(clap::Args, Clone, Debug)]
pub struct LiveOptions {
    /// Camera of the scenario to fly (its HDF5 path; default: the first with `rgb`).
    #[arg(long)]
    pub camera: Option<String>,
    /// Mount the camera looking forward (15° down) instead of the scenario's mounting.
    #[arg(long)]
    pub forward: bool,
    /// Render at this fraction of the camera's resolution (pinhole-type models).
    #[arg(long, default_value_t = 1.0)]
    pub scale: f64,
    /// Supersampling per axis (default 1: one sample per pixel).
    #[arg(long, default_value_t = 1)]
    pub supersample: u32,
    /// Generate the tiles the view selects in the background (else only stored tiles).
    #[arg(long)]
    pub dynamic: bool,
    /// Start as a plane instead of free flight.
    #[arg(long)]
    pub plane: bool,
    /// Start position: lat,lon (deg),height above the ground (m) (default: the world's home,
    /// 300 m up).
    #[arg(long, allow_hyphen_values = true)]
    pub start: Option<String>,
}

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
fn render_thread(sh: Arc<Shared>, scn: Scenario, spec: CameraSpec, scale: f64, ss: u32, store: Arc<TileStore>, svc: Arc<Service>, repaint: egui::Context) -> Result<()> {
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
        std::thread::Builder::new().name("live-sensor".into()).spawn(move || {
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
    let max_zoom = scn.tiles.max_zoom;
    let want_params = LodParams { min_zoom: scn.tiles.min_zoom, max_zoom, texel_px: scn.render.texel_px, ..Default::default() };
    let mut last_want = Instant::now() - std::time::Duration::from_secs(10);
    while !sh.stop.load(Ordering::Relaxed) {
        let Some(req) = sh.req.lock().clone() else {
            std::thread::sleep(std::time::Duration::from_millis(5));
            continue;
        };
        let t0 = Instant::now();
        let ext = if req.forward { &forward_ext } else { &spec.extrinsics };
        let cam = req.pose.camera(ext, &ell);
        *sh.ground.lock() = ground_at(&cache, req.pose.geo.lat, req.pose.geo.lon, max_zoom);
        // what this view would select, generated in the background (ancestors first: the
        // renderer descends into a tile only through its stored parents)
        if req.dynamic && last_want.elapsed().as_secs_f64() > 0.3 {
            last_want = Instant::now();
            let oracle = WantOracle { store: &store, max_zoom };
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
        let frame = renderer.render(&cam, &sun);
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

struct LiveApp {
    sh: Arc<Shared>,
    svc: Arc<Service>,
    fly: FlyCam,
    ell: Ellipsoid,
    tex: Option<egui::TextureHandle>,
    img_size: [usize; 2],
    shown_frame: u64,
    t: f64,
    time_offset_h: f64,
    clock: bool,
    forward: bool,
    dynamic: bool,
    last: Instant,
    fps: f64,
    fps_mark: (Instant, u64),
    spawn_agl: Option<f64>,
    stats: (f64, f64, [f64; 3], f64, f64),
    camera_name: String,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
}

impl LiveApp {
    fn pose(&self) -> Pose {
        let g = self.fly.geodetic(&self.ell);
        Pose { t: self.t, geo: g, q_ned_body: euler_zyx_to_quat(self.fly.heading, self.fly.pitch, self.fly.roll) }
    }
}

impl eframe::App for LiveApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        self.t += dt;
        let typing = ui.ctx().egui_wants_keyboard_input();

        egui::Panel::left("live").resizable(true).default_size(260.0).show(ui, |ui| {
            ui.add_space(6.0);
            ui.heading("Live renderer");
            ui.label(egui::RichText::new(format!("{} · the dataset renderer and sensor", self.camera_name)).small().weak());
            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.fly.mode, FlyMode::Free, "Free flight");
                ui.selectable_value(&mut self.fly.mode, FlyMode::Plane, "Plane");
            });
            let mut kmh = self.fly.speed * 3.6;
            ui.add(egui::Slider::new(&mut kmh, 5.0..=5000.0).logarithmic(true).text("speed km/h"));
            self.fly.speed = kmh / 3.6;
            ui.checkbox(&mut self.forward, "Camera looking forward (else the scenario's mount)");
            ui.checkbox(&mut self.dynamic, "Generate missing tiles");
            if self.clock {
                ui.add(egui::Slider::new(&mut self.time_offset_h, -12.0..=12.0).text("time of day ± h"));
            } else {
                ui.label(egui::RichText::new("lighting.mode: fixed (no clock)").small().weak());
            }
            ui.separator();
            let (rms, tms, ex, sun_el, lights) = self.stats;
            ui.label(egui::RichText::new(format!(
                "{} × {} px · {:.1} frames/s\nrender {:.0} ms · frame {:.0} ms\nexposure {:.2} ms × {:.1} gain (EV {:.1})\nsun {:+.1}° · lights {:.0}%\ntiles generated {} · wanted {}",
                self.img_size[0],
                self.img_size[1],
                self.fps,
                rms,
                tms,
                ex[0] * 1000.0,
                ex[1],
                ex[2],
                sun_el,
                lights * 100.0,
                self.svc.stats().generated.load(Ordering::Relaxed),
                self.sh.wanted.load(Ordering::Relaxed),
            )).small().monospace());
            ui.separator();
            ui.label(egui::RichText::new(match self.fly.mode {
                FlyMode::Free => "WASD move · Space/C up/down · drag: look\nShift ×5 · Ctrl ×0.2 · P: plane",
                FlyMode::Plane => "W/S nose down/up · A/D roll · Q/E rudder\nShift/Ctrl throttle · drag: look · P: free",
            }).small().weak());
        });

        egui::CentralPanel::no_frame().show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
            // ---- controls
            if !typing && ui.input(|i| i.key_pressed(egui::Key::P)) {
                self.fly.mode = if self.fly.mode == FlyMode::Free { FlyMode::Plane } else { FlyMode::Free };
            }
            let key = |k: egui::Key| if !typing && ui.input(|i| i.key_down(k)) { 1.0 } else { 0.0 };
            let d = if resp.dragged() { resp.drag_delta() } else { egui::Vec2::ZERO };
            let inp = FlyInput {
                forward: key(egui::Key::W) - key(egui::Key::S),
                right: key(egui::Key::D) - key(egui::Key::A),
                up: key(egui::Key::Space) - key(egui::Key::C),
                yaw: key(egui::Key::E) - key(egui::Key::Q),
                boost: ui.input(|i| i.modifiers.shift),
                slow: ui.input(|i| i.modifiers.ctrl),
                look_yaw: d.x as f64 * 0.003,
                look_pitch: -d.y as f64 * 0.003,
                speed_steps: 0.0,
            };
            let ground = self.sh.ground.lock().unwrap_or(0.0).max(0.0);
            if let Some(agl) = self.spawn_agl {
                let touched = inp.forward != 0.0 || inp.right != 0.0 || inp.up != 0.0 || inp.yaw != 0.0 || d != egui::Vec2::ZERO;
                if touched {
                    self.spawn_agl = None;
                } else {
                    let g = self.fly.geodetic(&self.ell);
                    self.fly.pos = geodetic2ecef(Geodetic { h: ground + agl, ..g }, &self.ell);
                }
            }
            self.fly.update(dt, &inp, &self.ell, ground);
            let t_scn = self.t + self.time_offset_h * 3600.0;
            *self.sh.req.lock() = Some(Request { pose: Pose { t: t_scn, ..self.pose() }, t: t_scn, wall: self.t, forward: self.forward, dynamic: self.dynamic });

            // ---- latest frame
            let frames = self.sh.frames.load(Ordering::Relaxed);
            if frames != self.shown_frame {
                if let Some(o) = self.sh.out.lock().take() {
                    let img = egui::ColorImage::from_rgb([o.w, o.h], &o.rgb);
                    match &mut self.tex {
                        Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                        None => self.tex = Some(ui.ctx().load_texture("live", img, egui::TextureOptions::LINEAR)),
                    }
                    self.img_size = [o.w, o.h];
                    self.stats = (o.render_ms, o.total_ms, o.exposure, o.sun_elevation, o.lights);
                }
                self.shown_frame = frames;
            }
            let el = self.fps_mark.0.elapsed().as_secs_f64();
            if el > 0.5 {
                self.fps = (frames - self.fps_mark.1) as f64 / el;
                self.fps_mark = (Instant::now(), frames);
            }
            ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
            if let Some(t) = &self.tex {
                // fit, keeping the camera's aspect
                let a = self.img_size[0] as f32 / self.img_size[1].max(1) as f32;
                let (mut iw, mut ih) = (rect.width(), rect.width() / a);
                if ih > rect.height() {
                    (iw, ih) = (rect.height() * a, rect.height());
                }
                let r = egui::Rect::from_center_size(rect.center(), egui::vec2(iw, ih));
                ui.painter().image(t.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            } else {
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "rendering the first frame…", egui::FontId::proportional(16.0), egui::Color32::GRAY);
            }
            let g = self.fly.geodetic(&self.ell);
            let hud = format!(
                "{}   {:.0} km/h\nalt {:.0} m  above ground {:.0} m\nhdg {:03.0}°  pitch {:+.0}°  roll {:+.0}°\n{:.4}°, {:.4}°",
                if self.fly.mode == FlyMode::Plane { "PLANE" } else { "FREE FLIGHT" },
                self.fly.speed * 3.6,
                g.h,
                g.h - ground,
                self.fly.heading.to_degrees(),
                self.fly.pitch.to_degrees(),
                self.fly.roll.to_degrees(),
                g.lat.to_degrees(),
                g.lon.to_degrees()
            );
            let pos = rect.left_top() + egui::vec2(12.0, 10.0);
            let font = egui::FontId::monospace(13.0);
            ui.painter().text(pos + egui::vec2(1.0, 1.0), egui::Align2::LEFT_TOP, &hud, font.clone(), egui::Color32::from_black_alpha(200));
            ui.painter().text(pos, egui::Align2::LEFT_TOP, &hud, font, egui::Color32::from_rgb(230, 240, 230));
        });
        ui.ctx().request_repaint();
    }

    fn on_exit(&mut self) {
        self.sh.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            if let Ok(Err(e)) = t.join() {
                eprintln!("renderer: {e:#}");
            }
        }
    }
}

/// Fly a camera of `scn` over its tile store with the actual renderer.
pub fn run(scn: Scenario, store: Arc<TileStore>, gen: Arc<Generator>, opts: LiveOptions) -> Result<()> {
    let spec = match &opts.camera {
        Some(p) => scn.cameras.iter().find(|c| &c.path == p).with_context(|| format!("no camera {p} in the scenario"))?.clone(),
        None => match scn.cameras.iter().find(|c| c.rgb.is_some()) {
            Some(c) => c.clone(),
            None => bail!("the scenario has no camera with `rgb` (see configs/live.yaml)"),
        },
    };
    let ell = gen.world.ell;
    let (lat, lon, agl) = match &opts.start {
        Some(s) => {
            let v: Vec<f64> = s.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--start lat,lon,agl")?;
            if v.len() != 3 {
                bail!("--start lat,lon,agl");
            }
            (v[0], v[1], v[2])
        }
        None => {
            let h = gen.config().home.clone().unwrap_or_default();
            (h.lat, h.lon, 300.0)
        }
    };
    let p = geodetic2ecef(Geodetic { lat: lat.to_radians(), lon: lon.to_radians(), h: agl }, &ell);
    let mut fly = FlyCam::at(p, DVec3::X, &ell, 0.8, if opts.plane { FlyMode::Plane } else { FlyMode::Free });
    (fly.heading, fly.pitch, fly.speed) = (0.0, 0.0, 60.0);
    eprintln!("live: camera {}, store {} ({} tiles), seed {}", spec.path, store.path().display(), store.len(), gen.config().seed);

    let options = eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 860.0]).with_title("Live renderer"), ..Default::default() };
    let clock = scn.render.lighting.mode == render::lighting::SunMode::Clock;
    eframe::run_native(
        "terrain live",
        options,
        Box::new(move |cc| {
            let sh = Arc::new(Shared::default());
            let batch = (rayon::current_num_threads() / 2).max(1);
            let svc = Arc::new(Service::start(store.clone(), gen, vec![], batch, || {}));
            let thread = {
                let (sh, svc, ctx, spec, scn) = (sh.clone(), svc.clone(), cc.egui_ctx.clone(), spec.clone(), scn.clone());
                let (scale, ss) = (opts.scale, opts.supersample);
                std::thread::Builder::new().name("live-render".into()).spawn(move || render_thread(sh, scn, spec, scale, ss, store, svc, ctx))?
            };
            Ok(Box::new(LiveApp {
                sh,
                svc,
                fly,
                ell,
                tex: None,
                img_size: [0, 0],
                shown_frame: 0,
                t: 0.0,
                time_offset_h: 0.0,
                clock,
                forward: opts.forward,
                dynamic: opts.dynamic,
                last: Instant::now(),
                fps: 0.0,
                fps_mark: (Instant::now(), 0),
                spawn_agl: Some(agl),
                stats: (0.0, 0.0, [0.0; 3], 0.0, 0.0),
                camera_name: spec.path.clone(),
                thread: Some(thread),
            }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
