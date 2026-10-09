//! Looking at sequence files: `terrain show` (an interactive viewer, or a headless snapshot of
//! it) and `terrain export` (PNG sequences and MP4 videos of the modalities).
//!
//! * [`seq`] reads a sequence file lazily, [`loader`] caches frames read in the background;
//! * [`colorize`] and [`view`] turn depth, flow, land cover and events into pictures with
//!   legends, the same for the viewer and the exporter;
//! * [`app`] is the viewer, [`draw`] its painting, [`soft`] a CPU renderer of egui output for
//!   snapshots and the exporter's labelled grids (no GPU or display needed);
//! * [`export`] writes PNGs and, through ffmpeg, videos.

pub mod app;
pub mod colorize;
pub mod draw;
pub mod export;
pub mod loader;
pub mod seq;
pub mod soft;
pub mod view;

pub use export::{export, ExportOptions};

use anyhow::{bail, Context, Result};
use colorize::EventStyle;
use eframe::{egui, egui_wgpu};
use seq::{Modality, Sequence};
use std::path::PathBuf;
use std::sync::Arc;
use view::Style;

/// Frame cache of the viewer (bytes).
const CACHE_BYTES: usize = 512 << 20;

/// Options of `terrain show`.
#[derive(clap::Args, Clone, Debug)]
pub struct ShowOptions {
    /// Sequence file (HDF5, written by `terrain run`).
    pub file: PathBuf,
    /// Camera (HDF5 path such as /cam0, or its name; default: the first with frames).
    #[arg(long)]
    pub camera: Option<String>,
    /// Start at this frame of the camera.
    #[arg(long, conflicts_with = "time")]
    pub frame: Option<usize>,
    /// Start at this time (s on the sequence clock).
    #[arg(long, allow_hyphen_values = true)]
    pub time: Option<f64>,
    /// Modalities shown, comma separated: rgb, depth, flow, landcover, events (default: all
    /// the camera has).
    #[arg(long)]
    pub modalities: Option<String>,
    #[command(flatten)]
    pub style: StyleArgs,
    /// Start without the side panel (pose, IMU, trajectory, settings, metadata).
    #[arg(long)]
    pub no_side_panel: bool,

    /// Render the viewer once, without a window, into this PNG and exit (CPU only: no GPU or
    /// display needed).
    #[arg(long, help_heading = "Snapshot (headless)")]
    pub snapshot: Option<PathBuf>,
    /// Snapshot size in points, WxH.
    #[arg(long, default_value = "1600x1000", help_heading = "Snapshot (headless)")]
    pub size: String,
    /// Snapshot pixels per point (2: a sharper image twice the size).
    #[arg(long, default_value_t = 1.0, help_heading = "Snapshot (headless)")]
    pub scale: f32,
    /// Snapshot with the pointer at x,y (points): the pixel readout of that place.
    #[arg(long, help_heading = "Snapshot (headless)")]
    pub pointer: Option<String>,
}

/// How the modalities are drawn (`terrain show` and `terrain export`).
#[derive(clap::Args, Clone, Debug)]
pub struct StyleArgs {
    /// Events: the accumulation window (ms) ending at the time shown.
    #[arg(long, default_value_t = 10.0, help_heading = "Display")]
    pub window_ms: f64,
    /// Events drawn ON red / OFF blue on black, on the camera's gray frame, or as a time
    /// surface (the latest event per pixel, fading over the window).
    #[arg(long, default_value = "black", value_parser = ["black", "gray", "surface"], help_heading = "Display")]
    pub events: String,
    /// Depth colour range near,far in m (default: show: each frame's own; export: the
    /// sequence's, from sampled frames).
    #[arg(long, help_heading = "Display")]
    pub depth_range: Option<String>,
    /// Depth colour scale: auto (log when the range spans more than a factor 4, as in oblique
    /// views), linear or log.
    #[arg(long, default_value = "auto", value_parser = ["auto", "linear", "log"], help_heading = "Display")]
    pub depth_scale: String,
    /// Flow magnitude (px) at full colour (default like --depth-range: the 99th percentile).
    #[arg(long, help_heading = "Display")]
    pub flow_max: Option<f32>,
    /// Catalogue stars (cameras with `stars`) ringed on the image up to this magnitude.
    #[arg(long, default_value_t = 6.5, allow_hyphen_values = true, help_heading = "Display")]
    pub star_mag: f32,
    /// Without the catalogue stars on the image.
    #[arg(long, help_heading = "Display")]
    pub no_stars: bool,
}

impl StyleArgs {
    pub fn style(&self) -> Result<Style> {
        if !(self.window_ms > 0.0 && self.window_ms.is_finite()) {
            bail!("--window-ms must be > 0, got {}", self.window_ms);
        }
        let depth_range = match &self.depth_range {
            Some(s) => {
                let v: Vec<f32> = s.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--depth-range near,far (m)")?;
                if v.len() != 2 || !(v[0] > 0.0 && v[1] > v[0] && v[1].is_finite()) {
                    bail!("--depth-range near,far: two numbers with 0 < near < far, got {s:?}");
                }
                Some((v[0], v[1]))
            }
            None => None,
        };
        if !self.star_mag.is_finite() {
            bail!("--star-mag must be a number");
        }
        if let Some(m) = self.flow_max {
            if !(m > 0.0 && m.is_finite()) {
                bail!("--flow-max must be > 0, got {m}");
            }
        }
        Ok(Style {
            depth_range,
            depth_log: match self.depth_scale.as_str() {
                "auto" => None,
                "linear" => Some(false),
                "log" => Some(true),
                s => bail!("--depth-scale {s:?}: auto, linear or log"),
            },
            flow_max: self.flow_max,
            window_us: (self.window_ms * 1000.0).round().max(1.0) as i64,
            events: EventStyle::parse(&self.events).with_context(|| format!("--events {:?}: black, gray or surface", self.events))?,
            stars: (!self.no_stars).then_some(self.star_mag),
        })
    }
}

/// `WxH`, each at least `min`.
pub fn parse_size(s: &str, min: u32) -> Result<(u32, u32)> {
    let wh = s.split_once('x').and_then(|(a, b)| Some((a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?)));
    match wh {
        Some((w, h)) if w >= min && h >= min && w <= 16384 && h <= 16384 => Ok((w, h)),
        _ => bail!("--size WxH (each {min}..=16384), got {s:?}"),
    }
}

/// The start of the viewer from the options, checked against the file.
pub fn start(seq: &Sequence, o: &ShowOptions) -> Result<app::Start> {
    let camera = o.camera.as_deref().map(|c| seq.camera_index(c)).transpose()?;
    let modalities = o.modalities.as_deref().map(Modality::parse_list).transpose()?;
    if let (Some(ms), Some(c)) = (&modalities, camera.or_else(|| seq.cameras.iter().position(|c| !c.t.is_empty())).and_then(|i| seq.cameras.get(i))) {
        let missing: Vec<&str> = ms.iter().filter(|m| !c.has(**m)).map(|m| m.name()).collect();
        if !missing.is_empty() {
            let have: Vec<&str> = c.modalities().iter().map(|m| m.name()).collect();
            bail!("{} has no {} (it has: {})", c.path, missing.join(", "), if have.is_empty() { "nothing".into() } else { have.join(", ") });
        }
    }
    if let Some(k) = o.frame {
        let c = camera.or_else(|| seq.cameras.iter().position(|c| !c.t.is_empty())).and_then(|i| seq.cameras.get(i));
        match c {
            Some(c) if k < c.t.len() => {}
            Some(c) => bail!("--frame {k}: {} has {} frames", c.path, c.t.len()),
            None => bail!("--frame: the file has no camera with frames (use --time)"),
        }
    }
    if let Some(t) = o.time {
        if !t.is_finite() {
            bail!("--time must be a number of seconds");
        }
    }
    if o.snapshot.is_some() {
        parse_size(&o.size, 64)?;
        if !(o.scale >= 0.5 && o.scale <= 4.0) {
            bail!("--scale: 0.5..=4, got {}", o.scale);
        }
        if let Some(p) = &o.pointer {
            parse_pointer(p)?;
        }
    }
    Ok(app::Start { camera, frame: o.frame, time: o.time, modalities, style: o.style.style()? })
}

fn parse_pointer(p: &str) -> Result<egui::Pos2> {
    let v: Vec<f32> = p.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--pointer x,y")?;
    if v.len() != 2 || !v.iter().all(|x| x.is_finite()) {
        bail!("--pointer x,y (points), got {p:?}");
    }
    Ok(egui::pos2(v[0], v[1]))
}

/// Render the viewer once, headlessly, into a PNG.
pub fn snapshot(seq: Arc<Sequence>, start: app::Start, o: &ShowOptions, out: &std::path::Path) -> Result<()> {
    let (w, h) = parse_size(&o.size, 64)?;
    let loader = loader::Loader::blocking(seq.clone(), CACHE_BYTES);
    let mut app = app::ShowApp::new(seq, loader, start);
    app.set_side_panel(!o.no_side_panel);
    let events = o.pointer.as_deref().map(parse_pointer).transpose()?.map(|p| vec![egui::Event::PointerMoved(p)]).unwrap_or_default();
    let img = soft::Headless::default().run((w as f32, h as f32), o.scale, 4, events, |ui| app.ui(ui));
    if let Some(p) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p)?;
    }
    image::save_buffer(out, &img.rgb(), img.w as u32, img.h as u32, image::ExtendedColorType::Rgb8).with_context(|| format!("writing {}", out.display()))?;
    eprintln!("wrote {} ({} × {} px)", out.display(), img.w, img.h);
    Ok(())
}

/// `terrain show`: open the viewer (or write a snapshot).
pub fn show(o: ShowOptions) -> Result<()> {
    let seq = Arc::new(Sequence::open(&o.file)?);
    eprintln!("{}", seq.summary());
    let st = start(&seq, &o)?;
    if let Some(out) = o.snapshot.clone() {
        return snapshot(seq, st, &o, &out);
    }
    let title = format!("terrain show · {}", o.file.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 1000.0]).with_title(&title),
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew::without_display_handle()),
            ..Default::default()
        },
        ..Default::default()
    };
    let side = !o.no_side_panel;
    eframe::run_native(
        "terrain show",
        options,
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            let loader = loader::Loader::background(seq.clone(), CACHE_BYTES, move || ctx.request_repaint());
            let mut app = app::ShowApp::new(seq, loader, st);
            app.set_side_panel(side);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e} (`--snapshot out.png` renders without a window)"))
}
