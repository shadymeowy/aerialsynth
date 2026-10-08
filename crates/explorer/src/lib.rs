//! Globe explorer (`terrain explore`): the scenario's tile store as a globe, generated as you go.
//!
//! On start the base levels (z0..=`--base-zoom`) are completed in the background. Then the view
//! streams the tiles it needs from the store; with dynamic generation on, missing ones are
//! generated (and written to the store) as you fly.

mod globe;
mod tiles;

use anyhow::{Context, Result};
use eframe::{egui, egui_wgpu, wgpu};
use geodesy::tiles::TileId;
use glam::DVec2;
use globe::{Camera, Globe, Mode, Settings};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use terragen::Generator;
use tiles::Service;
use tilestore::TileStore;

/// Options of `terrain explore`.
#[derive(clap::Args, Clone, Debug)]
pub struct Options {
    /// Levels 0..=this are generated for the whole planet on start.
    #[arg(long, default_value_t = 4)]
    pub base_zoom: u8,
    /// Start with dynamic generation on.
    #[arg(long)]
    pub dynamic: bool,
    /// Deepest level generated dynamically (default: tiles.max_zoom).
    #[arg(long)]
    pub max_zoom: Option<u8>,
    /// Tiles kept on the GPU (768 KB each).
    #[arg(long, default_value_t = 512)]
    pub gpu_tiles: u32,
    /// Relief exaggeration at start.
    #[arg(long, default_value_t = 1.0)]
    pub exag: f32,
    /// View mode at start: surface, elevation, landcover or relief.
    #[arg(long, default_value = "surface")]
    pub mode: String,
    /// Render one view without a window into this PNG (once its tiles are in) and exit.
    #[arg(long)]
    pub snapshot: Option<PathBuf>,
    /// Snapshot view: lat,lon (deg), distance (km), heading, tilt (deg).
    #[arg(long, default_value = "20,10,16000,0,0", allow_hyphen_values = true)]
    pub view: String,
    /// Snapshot size, WxH.
    #[arg(long, default_value = "1280x800")]
    pub size: String,
}

/// Explore `store` (which holds `gen`'s world): the window, or one `--snapshot`.
/// `max_zoom` is the deepest level generated unless the options set one.
pub fn run(store: Arc<TileStore>, gen: Arc<Generator>, mut opts: Options, max_zoom: u8) -> Result<()> {
    opts.max_zoom.get_or_insert(max_zoom);
    eprintln!("store {} ({} tiles), seed {}", store.path().display(), store.len(), gen.config().seed);
    if let Some(out) = opts.snapshot.clone() {
        return snapshot(&opts, store, gen, &out);
    }
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    // texture arrays of hundreds of tiles: the adapter's own limits, not the portable defaults
    setup.device_descriptor = Arc::new(|adapter: &wgpu::Adapter| wgpu::DeviceDescriptor {
        label: Some("explorer"),
        required_limits: adapter.limits(),
        ..Default::default()
    });
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 950.0]).with_title("Terrain explorer"),
        wgpu_options: egui_wgpu::WgpuConfiguration { wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(setup), ..Default::default() },
        ..Default::default()
    };
    eframe::run_native("terrain explore", options, Box::new(move |cc| Ok(Box::new(App::new(cc, opts, store, gen)?))))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn base_tiles(base_zoom: u8) -> Vec<TileId> {
    let mut base = Vec::new();
    for z in 0..=base_zoom {
        let n = 1u32 << z;
        for y in 0..n {
            for x in 0..n {
                base.push(TileId::new(z, x, y));
            }
        }
    }
    base
}

fn settings(args: &Options) -> Settings {
    Settings {
        exaggeration: args.exag,
        mode: match args.mode.as_str() {
            "elevation" => Mode::Elevation,
            "landcover" => Mode::Landcover,
            "relief" => Mode::Relief,
            _ => Mode::Albedo,
        },
        borders: false,
        lod_bias: 1.0,
        dynamic: args.dynamic,
        gen_max_zoom: args.max_zoom.unwrap_or(17),
        view_max_zoom: 20,
        base_zoom: args.base_zoom,
        sun_follows_view: true,
        sun_lat: 15.0,
        sun_lon: 30.0,
    }
}

/// Headless: render the `--view` once every tile it wants is in, save a PNG.
fn snapshot(args: &Options, store: Arc<TileStore>, gen: Arc<Generator>, out: &PathBuf) -> Result<()> {
    let v: Vec<f64> = args.view.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--view lat,lon,km,heading,tilt")?;
    let (w, h) = args.size.split_once('x').map(|(a, b)| (a.parse::<u32>(), b.parse::<u32>())).context("--size WxH")?;
    let (w, h) = (w?, h?);
    let (device, queue) = pollster::block_on(async {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }).await?;
        anyhow::Ok(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("explorer"), required_limits: adapter.limits(), ..Default::default() }).await?)
    })?;
    let ell = gen.world.ell;
    let svc = Service::start(store, gen, base_tiles(args.base_zoom), (rayon::current_num_threads() / 2).max(1), || {});
    let mut globe = Globe::new(&device, &queue, ell, args.gpu_tiles);
    let s = settings(args);
    let mut cam = Camera { lat: v[0].to_radians(), lon: v[1].to_radians(), dist: v[2] * 1000.0, heading: v[3].to_radians(), tilt: v[4].to_radians(), fov_y: 40f64.to_radians(), target_h: 0.0 };
    let t0 = Instant::now();
    let mut calm = 0;
    loop {
        cam.target_h = globe.height_at(cam.lat, cam.lon, 22).unwrap_or(0.0).max(0.0) * s.exaggeration as f64;
        globe.render(&cam, &s, &svc, w, h, None);
        calm = if globe.settled(&svc) { calm + 1 } else { 0 };
        if calm >= 5 || t0.elapsed().as_secs() > 600 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let (w, h, px) = globe.read_image().context("reading the image back")?;
    image::save_buffer(out, &px, w, h, image::ExtendedColorType::Rgba8)?;
    let g = &globe.stats;
    eprintln!(
        "wrote {} after {:.1}s: {} patches, finest z{}, {} GPU tiles, {} generated",
        out.display(),
        t0.elapsed().as_secs_f64(),
        g.drawn,
        g.max_zoom_drawn,
        g.resident,
        svc.stats().generated.load(std::sync::atomic::Ordering::Relaxed)
    );
    Ok(())
}

struct App {
    svc: Service,
    globe: Globe,
    cam: Camera,
    s: Settings,
    store_path: PathBuf,
    counts: Vec<(u8, usize)>,
    counts_at: Instant,
    hover: Option<(f64, f64)>,
    eye_alt: f64,
    rate: f64,
    rate_mark: (Instant, u64),
    frame_ms: f64,
    last_frame: Instant,
    home: Option<(f64, f64)>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, args: Options, store: Arc<TileStore>, gen: Arc<Generator>) -> Result<App, Box<dyn std::error::Error + Send + Sync>> {
        let rs = cc.wgpu_render_state.as_ref().ok_or("wgpu is not available")?;
        let ell = gen.world.ell;
        let home = gen.config().home.as_ref().map(|h| (h.lat.to_radians(), h.lon.to_radians()));
        let base = base_tiles(args.base_zoom);
        let store_path = store.path().to_path_buf();
        let ctx = cc.egui_ctx.clone();
        let batch = (rayon::current_num_threads() / 2).max(1);
        let svc = Service::start(store, gen, base, batch, move || ctx.request_repaint());
        let globe = Globe::new(&rs.device, &rs.queue, ell, args.gpu_tiles);
        let (lat, lon) = home.unwrap_or((20f64.to_radians(), 10f64.to_radians()));
        Ok(App {
            svc,
            globe,
            cam: Camera { lat, lon, dist: 2.6 * ell.a, heading: 0.0, tilt: 0.0, fov_y: 40f64.to_radians(), target_h: 0.0 },
            s: settings(&args),
            store_path,
            counts: Vec::new(),
            counts_at: Instant::now() - std::time::Duration::from_secs(10),
            hover: None,
            eye_alt: 0.0,
            rate: 0.0,
            rate_mark: (Instant::now(), 0),
            frame_ms: 0.0,
            last_frame: Instant::now(),
            home,
        })
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        let st = self.svc.stats();
        let ord = std::sync::atomic::Ordering::Relaxed;
        ui.add_space(6.0);
        ui.heading("Terrain explorer");
        ui.label(egui::RichText::new(format!("{}  ·  seed {}", self.store_path.display(), self.svc.gen.config().seed)).small().weak());
        ui.separator();

        // ---- store
        if self.counts_at.elapsed().as_secs_f64() > 1.0 {
            self.counts = self.svc.store.zooms().into_iter().map(|z| (z, self.svc.store.tiles_at(z).len())).collect();
            self.counts_at = Instant::now();
        }
        ui.strong("Tiles in the store");
        egui::Grid::new("counts").num_columns(2).spacing([14.0, 2.0]).show(ui, |ui| {
            for (k, (z, n)) in self.counts.iter().enumerate() {
                ui.monospace(format!("z{z:<2} {n:>7}"));
                if k % 2 == 1 {
                    ui.end_row();
                }
            }
        });
        let (bt, bd) = (st.base_total.load(ord), st.base_done.load(ord));
        if bt > 0 && bd < bt {
            ui.add(egui::ProgressBar::new(bd as f32 / bt as f32).text(format!("base levels z0–z{}: {bd} / {bt}", self.s.base_zoom)));
        }
        ui.separator();

        // ---- generation
        ui.strong("Generation");
        ui.checkbox(&mut self.s.dynamic, "Generate missing tiles as you fly");
        ui.add_enabled(self.s.dynamic, egui::Slider::new(&mut self.s.gen_max_zoom, self.s.base_zoom..=19).text("deepest level"));
        let generated = st.generated.load(ord);
        let dt = self.rate_mark.0.elapsed().as_secs_f64();
        if dt > 1.0 {
            let r = (generated - self.rate_mark.1) as f64 / dt;
            self.rate = 0.5 * self.rate + 0.5 * r;
            self.rate_mark = (Instant::now(), generated);
        }
        ui.label(format!("{generated} generated  ·  {:.1} tiles/s  ·  {} loaded", self.rate, st.loaded.load(ord)));
        ui.label(
            egui::RichText::new(format!(
                "in flight {}  ·  wanted: load {}, generate {}",
                self.svc.in_flight(),
                self.globe.stats.want_load,
                if self.s.dynamic { self.globe.stats.want_gen } else { 0 }
            ))
            .small()
            .weak(),
        );
        let busy = st.gen_busy.load(ord);
        if busy > 0 {
            ui.label(egui::RichText::new(format!("generating {busy} tiles…")).small());
        }
        let errors = st.errors.load(ord);
        if errors > 0 {
            ui.colored_label(egui::Color32::from_rgb(230, 120, 90), format!("{errors} tile errors (see the terminal)"));
        }
        ui.separator();

        // ---- view
        ui.strong("View");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.s.mode, Mode::Albedo, "Surface");
            ui.selectable_value(&mut self.s.mode, Mode::Elevation, "Elevation");
            ui.selectable_value(&mut self.s.mode, Mode::Landcover, "Land cover");
            ui.selectable_value(&mut self.s.mode, Mode::Relief, "Relief");
        });
        ui.add(egui::Slider::new(&mut self.s.exaggeration, 1.0..=100.0).logarithmic(true).text("relief ×"));
        ui.add(egui::Slider::new(&mut self.s.lod_bias, 0.5..=4.0).text("pixels per texel"));
        ui.add(egui::Slider::new(&mut self.s.view_max_zoom, 0..=22).text("finest level shown"));
        ui.checkbox(&mut self.s.borders, "Tile borders (coloured by level)");
        ui.checkbox(&mut self.s.sun_follows_view, "Light from the upper left");
        if !self.s.sun_follows_view {
            ui.add(egui::Slider::new(&mut self.s.sun_lon, -180.0..=180.0).text("sun longitude"));
            ui.add(egui::Slider::new(&mut self.s.sun_lat, -23.44..=23.44).text("sun latitude"));
        }
        ui.separator();

        // ---- camera
        ui.strong("Camera");
        ui.monospace(format!(
            "{:>9.4}°  {:>9.4}°\nalt {}  hdg {:>3.0}°  tilt {:>2.0}°",
            self.cam.lat.to_degrees(),
            self.cam.lon.to_degrees(),
            fmt_dist(self.eye_alt),
            self.cam.heading.to_degrees().rem_euclid(360.0),
            self.cam.tilt.to_degrees()
        ));
        if let Some((la, lo)) = self.hover {
            ui.label(egui::RichText::new(format!("cursor {:.4}°, {:.4}°", la.to_degrees(), lo.to_degrees())).small().weak());
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Whole planet").clicked() {
                self.cam.dist = 2.6 * self.globe.ellipsoid().a;
                self.cam.tilt = 0.0;
            }
            if let Some((la, lo)) = self.home {
                if ui.button("Home").clicked() {
                    (self.cam.lat, self.cam.lon, self.cam.dist) = (la, lo, 40_000.0);
                }
            }
            if ui.button("North up").clicked() {
                self.cam.heading = 0.0;
            }
            if ui.button("Top down").clicked() {
                self.cam.tilt = 0.0;
            }
        });
        ui.separator();

        // ---- statistics
        let g = &self.globe.stats;
        ui.label(
            egui::RichText::new(format!(
                "patches {}  ·  finest z{}\nGPU tiles {} / {}  ·  uploads {}\nframe {:.1} ms",
                g.drawn, g.max_zoom_drawn, g.resident, g.capacity, g.uploads, self.frame_ms
            ))
            .small()
            .weak(),
        );
        ui.add_space(4.0);
        ui.label(egui::RichText::new("drag: move  ·  right drag: turn / tilt\nscroll: zoom  ·  double click: fly there").small().weak());
    }
}

fn fmt_dist(m: f64) -> String {
    if m.abs() >= 10_000.0 {
        format!("{:.0} km", m / 1000.0)
    } else {
        format!("{m:.0} m")
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt_ms = now.duration_since(self.last_frame).as_secs_f64() * 1000.0;
        self.frame_ms = if self.frame_ms == 0.0 { dt_ms } else { 0.9 * self.frame_ms + 0.1 * dt_ms };
        self.last_frame = now;

        egui::Panel::left("controls").resizable(true).default_size(300.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.panel(ui));
        });
        egui::CentralPanel::no_frame().show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
            let ppp = ui.ctx().pixels_per_point();
            let (w, h) = ((rect.width() * ppp).round() as u32, (rect.height() * ppp).round() as u32);
            let ell = self.globe.ellipsoid();

            // ---- input
            let m_per_pt = self.cam.dist * self.cam.fov_y / rect.height().max(1.0) as f64;
            if resp.dragged_by(egui::PointerButton::Primary) {
                let d = resp.drag_delta();
                self.cam.pan(d.x as f64, d.y as f64, m_per_pt, &ell);
            }
            if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
                let d = resp.drag_delta();
                self.cam.heading -= d.x as f64 * 0.005;
                self.cam.tilt = (self.cam.tilt + d.y as f64 * 0.004).clamp(0.0, 1.48);
            }
            if resp.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y) as f64;
                if scroll != 0.0 {
                    self.cam.dist = (self.cam.dist * (-scroll * 0.003).exp()).clamp(40.0, 6.0e7);
                }
            }
            // keep the target on the terrain (smoothly: finer tiles change it as they arrive)
            let th = self.globe.height_at(self.cam.lat, self.cam.lon, 22).unwrap_or(0.0).max(0.0) * self.s.exaggeration as f64;
            self.cam.target_h += (th - self.cam.target_h) * 0.25;

            // ---- render
            let rs = frame.wgpu_render_state().expect("wgpu").clone();
            let cf = {
                let mut r = rs.renderer.write();
                self.globe.render(&self.cam, &self.s, &self.svc, w, h, Some(&mut r))
            };
            self.eye_alt = geodesy::ecef2geodetic(cf.eye, &ell).h;
            if let Some(id) = self.globe.texture_id {
                ui.painter().image(id, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            }
            let ndc_of = |p: egui::Pos2| DVec2::new(((p.x - rect.left()) / rect.width() * 2.0 - 1.0) as f64, (1.0 - (p.y - rect.top()) / rect.height() * 2.0) as f64);
            self.hover = resp.hover_pos().and_then(|p| self.globe.pick(&cf, ndc_of(p)));
            if resp.double_clicked() {
                if let Some(p) = resp.interact_pointer_pos() {
                    if let Some((la, lo)) = self.globe.pick(&cf, ndc_of(p)) {
                        (self.cam.lat, self.cam.lon) = (la, lo);
                        self.cam.dist = (self.cam.dist * 0.35).max(200.0);
                    }
                }
            }
        });
        ui.ctx().request_repaint();
    }
}
