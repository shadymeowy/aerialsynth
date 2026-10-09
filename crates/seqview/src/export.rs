//! `terrain export`: the modalities of a camera as PNG sequences or (through ffmpeg) videos,
//! one per modality or side by side in a labelled grid. The pictures are those of the viewer
//! (`view`), with fixed colour ranges for the whole export (stable colours).

use crate::colorize;
use crate::draw;
use crate::loader::Loader;
use crate::seq::{FrameData, Key, Modality, Sequence};
use crate::soft::Headless;
use crate::view::{self, Picture, Plan, Style};
use crate::StyleArgs;
use anyhow::{bail, Context, Result};
use eframe::egui::{self, pos2, vec2, Rect};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

/// Options of `terrain export`.
#[derive(clap::Args, Clone, Debug)]
pub struct ExportOptions {
    /// Sequence file (HDF5, written by `terrain run`).
    pub file: PathBuf,
    /// Camera (HDF5 path such as /cam0, or its name; default: the first with frames).
    #[arg(long)]
    pub camera: Option<String>,
    /// Modalities, comma separated: rgb, depth, flow, landcover, events (default: all the
    /// camera has).
    #[arg(long, short)]
    pub modality: Option<String>,
    /// Output: a directory for PNGs (DIR/<modality>/NNNNNN.png, DIR/grid/… with
    /// --side-by-side, and DIR/frames.csv) or a video file (.mp4, .mkv, .mov, .webm; needs
    /// ffmpeg; one per modality, FILE_<modality>.mp4, unless --side-by-side).
    #[arg(long, short)]
    pub out: PathBuf,
    /// All modalities side by side in one labelled picture per frame (with legends).
    #[arg(long)]
    pub side_by_side: bool,
    /// Start (s on the sequence clock).
    #[arg(long, allow_hyphen_values = true)]
    pub start: Option<f64>,
    /// End (s on the sequence clock, inclusive).
    #[arg(long, allow_hyphen_values = true)]
    pub end: Option<f64>,
    /// Every n-th frame only.
    #[arg(long, default_value_t = 1)]
    pub every: usize,
    /// Pictures per second of sequence time for a camera without frames (events only).
    #[arg(long)]
    pub rate: Option<f64>,
    /// Video frame rate (default: the camera's, i.e. real time; with --every, slower).
    #[arg(long)]
    pub fps: Option<f64>,
    /// Video quality (x264 CRF: lower is better and larger).
    #[arg(long, default_value_t = 18)]
    pub crf: u8,
    #[command(flatten)]
    pub style: StyleArgs,
}

const VIDEO_EXT: [&str; 4] = ["mp4", "mkv", "mov", "webm"];

/// Where the pictures go.
#[derive(Debug, PartialEq)]
enum Sink {
    Pngs(PathBuf),
    Video(PathBuf),
}

impl Sink {
    fn of(out: &Path) -> Sink {
        match out.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()) {
            Some(e) if VIDEO_EXT.contains(&e.as_str()) => Sink::Video(out.to_path_buf()),
            _ => Sink::Pngs(out.to_path_buf()),
        }
    }
}

/// The checked export plan.
pub struct Job {
    pub cam: usize,
    pub mods: Vec<Modality>,
    /// (index in the output, camera frame, time µs)
    pub times: Vec<(usize, Option<usize>, i64)>,
    pub fps: f64,
    pub style: Style,
}

/// Check the options against the file and list the times to export.
pub fn plan(seq: &Sequence, o: &ExportOptions) -> Result<Job> {
    let cam = match &o.camera {
        Some(c) => seq.camera_index(c)?,
        None => match seq.cameras.iter().position(|c| !c.t.is_empty()).or((!seq.cameras.is_empty()).then_some(0)) {
            Some(i) => i,
            None => bail!("{} has no cameras: nothing to export", seq.path.display()),
        },
    };
    let c = &seq.cameras[cam];
    let have = c.modalities();
    let mods = match &o.modality {
        Some(s) => Modality::parse_list(s)?,
        None => have.clone(),
    };
    let missing: Vec<&str> = mods.iter().filter(|m| !c.has(**m)).map(|m| m.name()).collect();
    if !missing.is_empty() || mods.is_empty() {
        let h: Vec<&str> = have.iter().map(|m| m.name()).collect();
        bail!(
            "{} has no {} (it has: {})",
            c.path,
            if missing.is_empty() { "modalities".into() } else { missing.join(", ") },
            if h.is_empty() { "nothing".into() } else { h.join(", ") }
        );
    }
    if o.every == 0 {
        bail!("--every must be at least 1");
    }
    for (v, name) in [(o.rate, "--rate"), (o.fps, "--fps")] {
        if let Some(v) = v {
            if !(v > 0.0 && v.is_finite()) {
                bail!("{name} must be > 0, got {v}");
            }
        }
    }
    let style = o.style.style()?;
    let us = |s: f64| (s * 1e6).round() as i64;
    let (r0, r1) = seq.time_range();
    let t0 = o.start.map(us).unwrap_or(i64::MIN);
    let t1 = o.end.map(us).unwrap_or(i64::MAX);
    if t1 < t0 {
        bail!("--end before --start");
    }
    let (times, native) = if !c.t.is_empty() {
        let v: Vec<(usize, Option<usize>, i64)> =
            c.t.iter().enumerate().filter(|(_, t)| **t >= t0 && **t <= t1).step_by(o.every).map(|(k, t)| (k, Some(k), *t)).collect();
        (v, c.frame_rate().unwrap_or(10.0) / o.every as f64)
    } else {
        // events only: pictures at --rate (default 30 Hz), the first one window in
        let rate = o.rate.unwrap_or(30.0);
        let dt = (1e6 / rate).round().max(1.0) as i64;
        let (a, b) = (t0.max(r0 + style.window_us), t1.min(r1));
        let mut v = Vec::new();
        let mut t = a;
        while t <= b {
            v.push((v.len(), None, t));
            t += dt * o.every as i64;
        }
        (v, rate / o.every as f64)
    };
    if times.is_empty() {
        bail!(
            "nothing to export: no frames of {} between {:.3} s and {:.3} s (the sequence spans {:.3}..{:.3} s)",
            c.path,
            t0.max(r0) as f64 / 1e6,
            t1.min(r1) as f64 / 1e6,
            r0 as f64 / 1e6,
            r1 as f64 / 1e6
        );
    }
    Ok(Job { cam, mods, times, fps: o.fps.unwrap_or(native), style })
}

/// Colour ranges for the whole export (where not given): from up to 8 frames spread over it.
fn fix_ranges(seq: &Sequence, loader: &Loader, job: &mut Job) {
    let frames: Vec<usize> = {
        let ks: Vec<usize> = job.times.iter().filter_map(|t| t.1).collect();
        let n = ks.len();
        (0..n.min(8)).map(|i| ks[if n <= 1 { 0 } else { i * (n - 1) / (n.min(8) - 1).max(1) }]).collect()
    };
    let read = |m: Modality| -> Vec<Arc<Result<FrameData, String>>> { frames.iter().filter_map(|&k| loader.get(Key::Frame { cam: job.cam, m, k })).collect() };
    let c = &seq.cameras[job.cam];
    if job.mods.contains(&Modality::Depth) && job.style.depth_range.is_none() && c.depth.is_some() {
        let d = read(Modality::Depth);
        let v: Vec<&[f32]> = d
            .iter()
            .filter_map(|x| match x.as_ref() {
                Ok(FrameData::Depth(v)) => Some(v.as_slice()),
                _ => None,
            })
            .collect();
        job.style.depth_range = colorize::depth_range(&v);
    }
    if job.mods.contains(&Modality::Flow) && job.style.flow_max.is_none() && c.flow {
        let d = read(Modality::Flow);
        let v: Vec<(&[f32], Option<&[u8]>)> = d
            .iter()
            .filter_map(|x| match x.as_ref() {
                Ok(FrameData::Flow { flow, valid }) => Some((flow.as_slice(), valid.as_deref())),
                _ => None,
            })
            .collect();
        if !v.is_empty() {
            job.style.flow_max = Some(colorize::flow_max(&v));
        }
    }
}

/// A picture with its stars drawn in (RGB).
fn flat(p: &Picture) -> Vec<u8> {
    let mut img = p.rgb.clone();
    for s in &p.stars {
        let c = if s.visible { [255, 220, 60] } else { [140, 140, 140] };
        colorize::draw_ring(&mut img, p.w, p.h, s.x, s.y, colorize::star_radius(s.v), c);
    }
    img
}

/// Lays pictures out side by side with titles and legends (egui, rendered on the CPU).
struct Composer {
    h: Headless,
}

impl Composer {
    fn new() -> Composer {
        Composer { h: Headless::default() }
    }

    fn layout(n: usize, w: usize, h: usize) -> (usize, usize, f32, f32) {
        let cols = match n {
            0..=3 => n.max(1),
            4 => 2,
            _ => 3,
        };
        let rows = n.div_ceil(cols);
        let cw = w as f32 + 8.0;
        let ch = h as f32 + draw::TITLE_H + draw::LEGEND_H + 8.0;
        (cols, rows, cw, ch)
    }

    /// The grid of `pics` (all `w` × `h`) as RGB (width, height, pixels).
    fn compose(&mut self, pics: &[Picture]) -> (usize, usize, Vec<u8>) {
        let (w, h) = (pics[0].w, pics[0].h);
        let (cols, rows, cw, ch) = Self::layout(pics.len(), w, h);
        let texs: Vec<egui::TextureHandle> =
            pics.iter().map(|p| self.h.ctx.load_texture("export", egui::ColorImage::from_rgb([p.w, p.h], &p.rgb), egui::TextureOptions::NEAREST)).collect();
        let img = self.h.run((cols as f32 * cw, rows as f32 * ch), 1.0, 1, vec![], |ui| {
            let painter = ui.painter().clone();
            painter.rect_filled(ui.max_rect(), 0.0, egui::Color32::from_gray(12));
            for (i, p) in pics.iter().enumerate() {
                let cell = Rect::from_min_size(pos2((i % cols) as f32 * cw, (i / cols) as f32 * ch), vec2(cw, ch));
                let r = draw::image_rect(cell, w, h, 1.0);
                // whole pixels: no resampling
                let r = Rect::from_min_size(r.min.round(), r.size());
                draw::cell(&painter, cell, r, p, texs[i].id(), None);
            }
        });
        drop(texs);
        (img.w, img.h, img.rgb())
    }
}

/// An ffmpeg process encoding raw RGB frames from its stdin.
struct Video {
    child: Child,
    path: PathBuf,
    size: (usize, usize),
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

impl Video {
    fn start(path: &Path, w: usize, h: usize, fps: f64, crf: u8) -> Result<Video> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("mp4").to_ascii_lowercase();
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &format!("{fps}"), "-i", "-"]);
        // yuv420p needs even sizes
        cmd.args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]);
        if ext == "webm" {
            cmd.args(["-c:v", "libvpx-vp9", "-crf", &crf.to_string(), "-b:v", "0"]);
        } else {
            cmd.args(["-c:v", "libx264", "-preset", "medium", "-crf", &crf.to_string()]);
        }
        cmd.args(["-pix_fmt", "yuv420p"]);
        if ext == "mp4" || ext == "mov" {
            cmd.args(["-movflags", "+faststart"]);
        }
        cmd.arg(path).stdin(Stdio::piped());
        let child = cmd.spawn().with_context(|| format!("starting ffmpeg for {}", path.display()))?;
        Ok(Video { child, path: path.to_path_buf(), size: (w, h) })
    }

    fn write(&mut self, w: usize, h: usize, rgb: &[u8]) -> Result<()> {
        if (w, h) != self.size {
            bail!("{}: picture size changed from {:?} to {:?}", self.path.display(), self.size, (w, h));
        }
        self.child.stdin.as_mut().unwrap().write_all(rgb).with_context(|| format!("ffmpeg writing {} stopped (see its message above)", self.path.display()))
    }

    fn finish(mut self) -> Result<()> {
        drop(self.child.stdin.take());
        let st = self.child.wait()?;
        if !st.success() {
            bail!("ffmpeg failed writing {} ({st})", self.path.display());
        }
        Ok(())
    }
}

/// Output streams: one per modality, or the grid.
struct Outputs {
    sink: Sink,
    videos: Vec<Option<Video>>,
    names: Vec<String>,
    fps: f64,
    crf: u8,
}

impl Outputs {
    fn video_path(&self, i: usize) -> PathBuf {
        let Sink::Video(p) = &self.sink else { unreachable!() };
        if self.names.len() == 1 {
            return p.clone();
        }
        let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let ext = p.extension().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        p.with_file_name(format!("{stem}_{}.{ext}", self.names[i]))
    }

    fn put(&mut self, i: usize, index: usize, w: usize, h: usize, rgb: &[u8]) -> Result<()> {
        match &self.sink {
            Sink::Pngs(dir) => {
                let p = dir.join(&self.names[i]).join(format!("{index:06}.png"));
                image::save_buffer(&p, rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8).with_context(|| format!("writing {}", p.display()))?;
            }
            Sink::Video(_) => {
                if self.videos[i].is_none() {
                    let p = self.video_path(i);
                    self.videos[i] = Some(Video::start(&p, w, h, self.fps, self.crf)?);
                }
                self.videos[i].as_mut().unwrap().write(w, h, rgb)?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        for v in self.videos.into_iter().flatten() {
            out.push(v.path.clone());
            v.finish()?;
        }
        Ok(out)
    }
}

/// `terrain export`.
pub fn export(o: ExportOptions) -> Result<()> {
    let seq = Arc::new(Sequence::open(&o.file)?);
    let mut job = plan(&seq, &o)?;
    let sink = Sink::of(&o.out);
    if let Sink::Video(p) = &sink {
        if !ffmpeg_available() {
            bail!(
                "writing {} needs ffmpeg on the PATH, which was not found: install it (apt install ffmpeg, brew install ffmpeg, winget install ffmpeg), or write PNGs with --out DIR",
                p.display()
            );
        }
    }
    let loader = Loader::blocking(seq.clone(), 256 << 20);
    fix_ranges(&seq, &loader, &mut job);
    let c = &seq.cameras[job.cam];
    let names: Vec<String> = if o.side_by_side { vec!["grid".into()] } else { job.mods.iter().map(|m| m.name().to_string()).collect() };
    if let Sink::Pngs(dir) = &sink {
        for n in &names {
            std::fs::create_dir_all(dir.join(n)).with_context(|| format!("creating {}", dir.join(n).display()))?;
        }
    } else if let Some(p) = o.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p)?;
    }
    let mut csv = match &sink {
        Sink::Pngs(dir) => {
            let mut f = std::io::BufWriter::new(std::fs::File::create(dir.join("frames.csv"))?);
            writeln!(f, "index,frame,t_us")?;
            Some(f)
        }
        Sink::Video(_) => None,
    };
    let mut outs = Outputs { sink, videos: names.iter().map(|_| None).collect(), names, fps: job.fps, crf: o.crf };
    let mut composer = o.side_by_side.then(Composer::new);
    let t_start = std::time::Instant::now();
    let n = job.times.len();
    eprintln!("exporting {} × [{}] of {} ({}×{}) → {}", n, job.mods.iter().map(|m| m.name()).collect::<Vec<_>>().join(", "), c.path, c.w, c.h, o.out.display());
    for (i, &(index, frame, t)) in job.times.iter().enumerate() {
        let mut pics = Vec::new();
        for &m in &job.mods {
            let plan = Plan::new(&seq, job.cam, m, t, &job.style).map_err(anyhow::Error::msg)?;
            let pic = view::render(&seq, &loader, &plan, &job.style).expect("blocking loader").map_err(anyhow::Error::msg)?;
            pics.push(pic);
        }
        match composer.as_mut() {
            Some(comp) => {
                let (w, h, rgb) = comp.compose(&pics);
                outs.put(0, index, w, h, &rgb)?;
            }
            None => {
                for (j, p) in pics.iter().enumerate() {
                    outs.put(j, index, p.w, p.h, &flat(p))?;
                }
            }
        }
        if let Some(f) = csv.as_mut() {
            writeln!(f, "{index},{},{t}", frame.map(|k| k.to_string()).unwrap_or_default())?;
        }
        if (i + 1) % 50 == 0 || i + 1 == n {
            eprintln!("  {}/{n} ({:.1} s)", i + 1, t_start.elapsed().as_secs_f64());
        }
    }
    if let Some(mut f) = csv {
        f.flush()?;
    }
    let sink_desc = match &outs.sink {
        Sink::Pngs(d) => format!("{}/{{{}}}/NNNNNN.png", d.display(), outs.names.join(",")),
        Sink::Video(_) => String::new(),
    };
    let videos = outs.finish()?;
    if videos.is_empty() {
        eprintln!("wrote {n} pictures per stream to {sink_desc} in {:.1} s", t_start.elapsed().as_secs_f64());
    } else {
        for v in videos {
            eprintln!("wrote {} ({n} frames at {:.2} fps)", v.display(), job.fps);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinks() {
        assert_eq!(Sink::of(Path::new("out/a.MP4")), Sink::Video("out/a.MP4".into()));
        assert_eq!(Sink::of(Path::new("out/a.webm")), Sink::Video("out/a.webm".into()));
        assert_eq!(Sink::of(Path::new("out/frames")), Sink::Pngs("out/frames".into()));
        assert_eq!(Sink::of(Path::new("out/x.png")), Sink::Pngs("out/x.png".into()));
        let o = Outputs { sink: Sink::of(Path::new("v/seq.mp4")), videos: vec![None, None], names: vec!["rgb".into(), "depth".into()], fps: 10.0, crf: 18 };
        assert_eq!(o.video_path(1), PathBuf::from("v/seq_depth.mp4"));
    }

    #[test]
    fn grid_layouts() {
        assert_eq!(Composer::layout(1, 10, 10).0, 1);
        assert_eq!(Composer::layout(3, 10, 10).0, 3);
        let (c, r, _, _) = Composer::layout(4, 10, 10);
        assert_eq!((c, r), (2, 2));
    }
}
