//! The viewer of `terrain show`: a timeline over the sequence, the modalities of one camera in a
//! grid, and a side panel with the pose, IMU, trajectory, display settings and metadata.

use crate::colorize::EventStyle;
use crate::draw;
use crate::loader::Loader;
use crate::seq::{FrameData, Key, Modality, Sequence};
use crate::view::{self, Picture, Plan, Style};
use eframe::egui::{self, vec2, Color32, Rect, RichText};
use std::collections::HashMap;
use std::sync::Arc;

/// Where the viewer starts.
#[derive(Clone, Debug, Default)]
pub struct Start {
    pub camera: Option<usize>,
    /// frame of that camera
    pub frame: Option<usize>,
    /// time (s since the sequence start)
    pub time: Option<f64>,
    pub modalities: Option<Vec<Modality>>,
    pub style: Style,
}

struct Cell {
    sig: u64,
    pic: Option<Result<Picture, String>>,
    tex: Option<egui::TextureHandle>,
}

const SPEEDS: [f64; 9] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 30.0];

pub struct ShowApp {
    seq: Arc<Sequence>,
    loader: Loader,
    cam: usize,
    /// current time (µs)
    t: i64,
    range: (i64, i64),
    playing: bool,
    speed: f64,
    looping: bool,
    shown: Vec<Modality>,
    style: Style,
    /// the manual values behind the auto switches
    depth_manual: (f32, f32),
    flow_manual: f32,
    imu_window_s: f64,
    star_mag: f32,
    cells: HashMap<Modality, Cell>,
    /// image pixel under the pointer (all panels show it)
    hover: Option<(usize, usize)>,
    /// fractional µs carried between frames while playing
    carry: f64,
    side: bool,
    imu_shown: Option<Arc<Result<FrameData, String>>>,
}

fn scale_name(log: Option<bool>) -> &'static str {
    match log {
        None => "auto scale",
        Some(false) => "linear",
        Some(true) => "log",
    }
}

fn fmt_s(us: i64) -> String {
    format!("{:.3} s", us as f64 / 1e6)
}

impl ShowApp {
    pub fn new(seq: Arc<Sequence>, loader: Loader, start: Start) -> ShowApp {
        let range = seq.time_range();
        let cam = start.camera.unwrap_or_else(|| seq.cameras.iter().position(|c| !c.t.is_empty()).unwrap_or(0));
        let mut style = start.style;
        let mut t = range.0;
        if let Some(c) = seq.cameras.get(cam) {
            t = c.t.first().copied().unwrap_or(range.0);
            if c.t.is_empty() {
                // an events-only camera: one window in
                t = (range.0 + style.window_us).min(range.1);
            }
            if let Some(k) = start.frame {
                t = c.t.get(k.min(c.t.len().saturating_sub(1))).copied().unwrap_or(t);
            }
        }
        if let Some(s) = start.time {
            t = ((s * 1e6).round() as i64).clamp(range.0, range.1);
        }
        let shown = start.modalities.unwrap_or_else(|| Modality::ALL.to_vec());
        let depth_manual = style.depth_range.unwrap_or((100.0, 2000.0));
        let flow_manual = style.flow_max.unwrap_or(5.0);
        let star_mag = style.stars.unwrap_or(6.5);
        style.window_us = style.window_us.max(1);
        ShowApp {
            seq,
            loader,
            cam,
            t,
            range,
            playing: false,
            speed: 1.0,
            looping: true,
            shown,
            style,
            depth_manual,
            flow_manual,
            imu_window_s: 4.0,
            star_mag,
            cells: HashMap::new(),
            hover: None,
            carry: 0.0,
            side: true,
            imu_shown: None,
        }
    }

    pub fn set_side_panel(&mut self, on: bool) {
        self.side = on;
    }

    pub fn loader(&self) -> &Loader {
        &self.loader
    }

    fn camera(&self) -> Option<&crate::seq::Camera> {
        self.seq.cameras.get(self.cam)
    }

    /// The modalities shown: those picked that the camera has.
    fn visible(&self) -> Vec<Modality> {
        match self.camera() {
            Some(c) => self.shown.iter().copied().filter(|m| c.has(*m)).collect(),
            None => vec![],
        }
    }

    fn frame_index(&self) -> Option<usize> {
        self.camera().and_then(|c| c.frame_at(self.t))
    }

    /// Step `n` frames (or event windows, for a camera without frames).
    fn step(&mut self, n: i64) {
        self.playing = false;
        match self.camera() {
            Some(c) if !c.t.is_empty() => {
                let k = c.frame_at(self.t).unwrap_or(0) as i64;
                let k = (k + n).clamp(0, c.t.len() as i64 - 1) as usize;
                self.t = c.t[k];
            }
            _ => {
                let dt = if self.camera().is_some() { self.style.window_us } else { 100_000 };
                self.t = (self.t + n * dt).clamp(self.range.0, self.range.1);
            }
        }
    }

    fn home(&mut self) {
        self.playing = false;
        self.t = match self.camera() {
            Some(c) if !c.t.is_empty() => c.t[0],
            Some(_) => (self.range.0 + self.style.window_us).min(self.range.1),
            None => self.range.0,
        };
    }

    fn end(&mut self) {
        self.playing = false;
        self.t = match self.camera() {
            Some(c) if !c.t.is_empty() => *c.t.last().unwrap(),
            _ => self.range.1,
        };
    }

    fn keys(&mut self, ui: &mut egui::Ui) {
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        // consumed, so a focused button or slider does not act on them too
        let (space, left, right, left10, right10, home, end) = ui.input_mut(|i| {
            use egui::{Key, Modifiers};
            (
                i.consume_key(Modifiers::NONE, Key::Space),
                i.consume_key(Modifiers::NONE, Key::ArrowLeft),
                i.consume_key(Modifiers::NONE, Key::ArrowRight),
                i.consume_key(Modifiers::SHIFT, Key::ArrowLeft),
                i.consume_key(Modifiers::SHIFT, Key::ArrowRight),
                i.consume_key(Modifiers::NONE, Key::Home),
                i.consume_key(Modifiers::NONE, Key::End),
            )
        });
        if space {
            self.toggle_play();
        }
        if left || left10 {
            self.step(if left10 { -10 } else { -1 });
        }
        if right || right10 {
            self.step(if right10 { 10 } else { 1 });
        }
        if home {
            self.home();
        }
        if end {
            self.end();
        }
    }

    fn toggle_play(&mut self) {
        self.playing = !self.playing;
        if self.playing && self.t >= self.range.1 {
            self.home();
            self.playing = true;
        }
        self.carry = 0.0;
    }

    fn advance(&mut self, dt: f64) {
        if !self.playing {
            return;
        }
        self.carry += dt.clamp(0.0, 0.1) * self.speed * 1e6;
        let d = self.carry.floor();
        self.carry -= d;
        self.t += d as i64;
        if self.t > self.range.1 {
            if self.looping {
                self.home();
                self.playing = true;
            } else {
                self.t = self.range.1;
                self.playing = false;
            }
        }
    }

    /// The whole window.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.keys(ui);
        let dt = ui.input(|i| i.stable_dt) as f64;
        self.advance(dt);
        egui::Panel::top("show-bar").show(ui, |ui| self.top_bar(ui));
        egui::Panel::bottom("show-timeline").show(ui, |ui| self.timeline(ui));
        if self.side {
            egui::Panel::right("show-side").resizable(true).default_size(360.0).min_size(260.0).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.side_panel(ui));
            });
        }
        egui::CentralPanel::no_frame().show(ui, |ui| self.central(ui));
        // what to read next: the pictures shown, then (playing) the frames ahead
        let mut want: Vec<Key> = Vec::new();
        for m in self.visible() {
            if let Ok(p) = Plan::new(&self.seq, self.cam, m, self.t, &self.style) {
                want.extend(p.keys());
            }
        }
        if let Some(k) = self.imu_key() {
            want.push(k);
        }
        if let (true, Some(c), Some(k)) = (self.playing, self.camera(), self.frame_index()) {
            for j in k + 1..(k + 4).min(c.t.len()) {
                for m in self.visible() {
                    if m != Modality::Events {
                        want.push(Key::Frame { cam: self.cam, m, k: j });
                    }
                }
            }
        }
        self.loader.want(want);
        if self.playing {
            ui.ctx().request_repaint();
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let name = self.seq.path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            ui.label(RichText::new(name).strong()).on_hover_text(self.seq.path.display().to_string());
            ui.separator();
            if self.seq.cameras.is_empty() {
                ui.label(RichText::new("no cameras").weak());
            } else {
                let label = |c: &crate::seq::Camera| format!("{} · {}×{}", c.path, c.w, c.h);
                let cur = self.camera().map(label).unwrap_or_default();
                let mut cam = self.cam;
                egui::ComboBox::from_id_salt("camera").selected_text(cur).show_ui(ui, |ui| {
                    for (i, c) in self.seq.cameras.iter().enumerate() {
                        let mods: Vec<&str> = c.modalities().iter().map(|m| m.name()).collect();
                        ui.selectable_value(&mut cam, i, format!("{} · {}×{} · {}", c.path, c.w, c.h, mods.join(", ")));
                    }
                });
                if cam != self.cam {
                    self.cam = cam;
                    self.cells.clear();
                    self.hover = None;
                }
                ui.separator();
                let mods = self.camera().map(|c| c.modalities()).unwrap_or_default();
                for m in mods {
                    let mut on = self.shown.contains(&m);
                    if ui.toggle_value(&mut on, m.label()).changed() {
                        if on {
                            self.shown.push(m);
                            self.shown.sort();
                        } else {
                            self.shown.retain(|x| *x != m);
                        }
                    }
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.toggle_value(&mut self.side, "Side panel");
                if self.loader.pending() {
                    ui.spinner();
                }
            });
        });
    }

    fn timeline(&mut self, ui: &mut egui::Ui) {
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            if ui.button("⏮").on_hover_text("first frame (Home)").clicked() {
                self.home();
            }
            if ui.button("⏴").on_hover_text("previous frame (Left; Shift: 10)").clicked() {
                self.step(-1);
            }
            let play = if self.playing { "⏸" } else { "▶" };
            if ui.button(play).on_hover_text("play / pause (Space)").clicked() {
                self.toggle_play();
            }
            if ui.button("⏵").on_hover_text("next frame (Right; Shift: 10)").clicked() {
                self.step(1);
            }
            if ui.button("⏭").on_hover_text("last frame (End)").clicked() {
                self.end();
            }
            egui::ComboBox::from_id_salt("speed").width(64.0).selected_text(format!("{}×", self.speed)).show_ui(ui, |ui| {
                for s in SPEEDS {
                    ui.selectable_value(&mut self.speed, s, format!("{s}×"));
                }
            });
            ui.checkbox(&mut self.looping, "loop");
            let mut s = self.t as f64 / 1e6;
            let (a, b) = (self.range.0 as f64 / 1e6, (self.range.1 as f64 / 1e6).max(self.range.0 as f64 / 1e6 + 1e-6));
            ui.spacing_mut().slider_width = (ui.available_width() - 90.0).max(80.0);
            let r = ui.add(
                egui::Slider::new(&mut s, a..=b)
                    .show_value(true)
                    .custom_formatter(|v, _| format!("{v:.3} s"))
                    .custom_parser(|t| t.trim_end_matches('s').trim().parse().ok()),
            );
            if r.changed() {
                self.t = (s * 1e6).round() as i64;
                // snap to the camera's frame times while dragging
                if let Some(c) = self.camera().filter(|c| !c.t.is_empty()) {
                    let k = c.t.partition_point(|&x| x < self.t);
                    let near = [k.saturating_sub(1), k.min(c.t.len() - 1)].into_iter().min_by_key(|&j| (c.t[j] - self.t).abs()).unwrap();
                    self.t = c.t[near];
                }
            }
        });
        ui.horizontal(|ui| {
            let mut s = format!("t {} of {}..{}", fmt_s(self.t), fmt_s(self.range.0), fmt_s(self.range.1));
            if let Some(c) = self.camera() {
                match c.frame_at(self.t) {
                    Some(k) => s += &format!(" · frame {} / {} of {}", k, c.t.len(), c.path),
                    None => s += &format!(" · {} (events only)", c.path),
                }
            }
            ui.label(RichText::new(s).monospace());
            if let Some((x, y)) = self.hover {
                let mut parts = vec![format!("x {x} y {y}")];
                for m in self.visible() {
                    if let Some(Some(Ok(p))) = self.cells.get(&m).map(|c| c.pic.as_ref()) {
                        if let Some(r) = view::readout(&self.seq, p, x, y) {
                            parts.push(r);
                        }
                    }
                }
                ui.separator();
                ui.label(RichText::new(parts.join(" · ")).monospace().color(Color32::from_rgb(240, 220, 150)));
            }
        });
        ui.add_space(2.0);
    }

    fn central(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        ui.painter().rect_filled(rect, 0.0, Color32::from_gray(12));
        let resp = ui.allocate_rect(rect, egui::Sense::hover());
        let mods = self.visible();
        if self.seq.cameras.is_empty() {
            // an IMU / pose-only file: the plots fill the view
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink(16.0)));
            child.label(RichText::new("No cameras in this file: poses and IMU").strong());
            self.imu_section(&mut child, 220.0);
            self.track_section(&mut child, 320.0);
            return;
        }
        let Some(c) = self.camera() else { return };
        let (w, h) = (c.w, c.h);
        if mods.is_empty() {
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "pick a modality in the top bar", egui::FontId::proportional(16.0), draw::WEAK);
            return;
        }
        let (cols, rows, scale) = draw::grid(mods.len(), w, h, rect.size());
        // the grid centred in the view
        let cs = draw::cell_size(w, h, scale);
        let (cw, ch) = ((rect.width() / cols as f32).min(cs.x), (rect.height() / rows as f32).min(cs.y));
        let origin = rect.center() - vec2(cw * cols as f32, ch * rows as f32) / 2.0;
        // rendering: only where what a picture depends on changed
        for &m in &mods {
            let plan = match Plan::new(&self.seq, self.cam, m, self.t, &self.style) {
                Ok(p) => p,
                Err(e) => {
                    self.cells.insert(m, Cell { sig: 0, pic: Some(Err(e)), tex: None });
                    continue;
                }
            };
            let sig = plan.signature(&self.style, &self.loader);
            if self.cells.get(&m).is_some_and(|c| c.sig == sig) {
                continue;
            }
            let Some(pic) = view::render(&self.seq, &self.loader, &plan, &self.style) else { continue }; // loading: keep the last
            let cell = self.cells.entry(m).or_insert(Cell { sig: 0, pic: None, tex: None });
            cell.sig = sig;
            if let Ok(p) = &pic {
                let img = egui::ColorImage::from_rgb([p.w, p.h], &p.rgb);
                let opts = egui::TextureOptions { magnification: egui::TextureFilter::Nearest, ..egui::TextureOptions::LINEAR };
                match &mut cell.tex {
                    Some(t) => t.set(img, opts),
                    None => cell.tex = Some(ui.ctx().load_texture(format!("show-{}", m.name()), img, opts)),
                }
            }
            cell.pic = Some(pic);
        }
        let pointer = resp.hover_pos();
        let mut hover = None;
        let painter = ui.painter_at(rect);
        for (i, m) in mods.iter().enumerate() {
            let cell_rect = Rect::from_min_size(origin + vec2((i % cols) as f32 * cw, (i / cols) as f32 * ch), vec2(cw, ch));
            let img = draw::image_rect(cell_rect, w, h, scale);
            if let Some(p) = pointer.filter(|p| img.contains(*p)) {
                let x = ((p.x - img.left()) / scale).floor().clamp(0.0, w as f32 - 1.0) as usize;
                let y = ((p.y - img.top()) / scale).floor().clamp(0.0, h as f32 - 1.0) as usize;
                hover = Some((x, y));
            }
            match self.cells.get(m) {
                Some(Cell { pic: Some(Ok(p)), tex: Some(t), .. }) => draw::cell(&painter, cell_rect, img, p, t.id(), self.hover),
                Some(Cell { pic: Some(Err(e)), .. }) => {
                    painter.text(
                        img.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("{}: {e}", m.label()),
                        egui::FontId::proportional(14.0),
                        Color32::from_rgb(230, 120, 90),
                    );
                }
                _ => {
                    painter.rect_filled(img, 0.0, Color32::from_gray(20));
                    painter.text(img.center(), egui::Align2::CENTER_CENTER, format!("{} loading…", m.label()), egui::FontId::proportional(14.0), draw::WEAK);
                }
            }
        }
        self.hover = hover;
    }

    fn imu_key(&self) -> Option<Key> {
        self.seq.imu.as_ref()?;
        // windows on a grid of 1/50 of their width (fewer reads while playing)
        let half = (self.imu_window_s * 1e6 / 2.0) as i64;
        let q = (half / 25).max(1);
        let c = (self.t / q) * q;
        Some(Key::Imu { t0: c - half - q, t1: c + half + q })
    }

    fn imu_section(&mut self, ui: &mut egui::Ui, h: f32) {
        let Some(imu) = &self.seq.imu else {
            ui.label(RichText::new("no IMU in this file").weak());
            return;
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{} · {} samples", imu.path, imu.t.len())).small().weak());
            ui.add(egui::Slider::new(&mut self.imu_window_s, 0.1..=60.0).logarithmic(true).suffix(" s").text("window"));
        });
        if let Some(k) = self.imu_key() {
            if let Some(d) = self.loader.get(k) {
                self.imu_shown = Some(d);
            }
        }
        let half = self.imu_window_s / 2.0;
        let tc = self.t as f64 / 1e6;
        let xr = (tc - half, tc + half);
        match self.imu_shown.as_deref() {
            Some(Ok(FrameData::Imu(w))) => {
                let ser = |v: &Vec<[f64; 3]>, i: usize| -> Vec<(f64, f64)> { w.t.iter().zip(v).map(|(t, a)| (*t as f64 / 1e6, a[i])).collect() };
                let gyro = [("x", draw::AXES[0], ser(&w.gyro, 0)), ("y", draw::AXES[1], ser(&w.gyro, 1)), ("z", draw::AXES[2], ser(&w.gyro, 2))];
                draw::plot(ui, "gyro rad/s", h, xr, &gyro, Some(tc));
                let acc = [("x", draw::AXES[0], ser(&w.accel, 0)), ("y", draw::AXES[1], ser(&w.accel, 1)), ("z", draw::AXES[2], ser(&w.accel, 2))];
                draw::plot(ui, "accel m/s²", h, xr, &acc, Some(tc));
            }
            Some(Err(e)) => {
                ui.colored_label(Color32::from_rgb(230, 120, 90), e);
            }
            _ => {
                ui.label("loading…");
            }
        }
    }

    fn track_section(&self, ui: &mut egui::Ui, h: f32) {
        let Some(p) = &self.seq.pose else {
            ui.label(RichText::new("no poses in this file").weak());
            return;
        };
        let pts: Vec<[f64; 2]> = if p.ned0.len() == p.t.len() && !p.ned0.is_empty() {
            p.ned0.iter().map(|q| [q[1], q[0]]).collect()
        } else {
            // local metres around the first position (equirectangular)
            let (lat0, lon0) = p.lla.first().map(|q| (q[0], q[1])).unwrap_or((0.0, 0.0));
            let k = 111_320.0;
            p.lla.iter().map(|q| [(q[1] - lon0) * k * lat0.to_radians().cos(), (q[0] - lat0) * k]).collect()
        };
        let i = p.index_at(self.t);
        draw::track(ui, h, &pts, i, i.and_then(|i| p.rpy.get(i)).map(|r| r[2]));
    }

    fn pose_section(&self, ui: &mut egui::Ui) {
        let Some(p) = &self.seq.pose else {
            ui.label(RichText::new("no poses in this file").weak());
            return;
        };
        let Some(i) = p.index_at(self.t) else { return };
        egui::Grid::new("pose").num_columns(4).spacing([10.0, 2.0]).show(ui, |ui| {
            let l = p.lla[i];
            ui.label("lat");
            ui.monospace(format!("{:.6}°", l[0]));
            ui.label("lon");
            ui.monospace(format!("{:.6}°", l[1]));
            ui.end_row();
            ui.label("h");
            ui.monospace(format!("{:.1} m", l[2]));
            ui.label("sun");
            ui.monospace(p.sun_elevation.get(i).map(|s| format!("{s:+.1}°")).unwrap_or_else(|| "–".into()));
            ui.end_row();
            if let Some(r) = p.rpy.get(i) {
                ui.label("roll");
                ui.monospace(format!("{:+.2}°", r[0]));
                ui.label("pitch");
                ui.monospace(format!("{:+.2}°", r[1]));
                ui.end_row();
                ui.label("yaw");
                ui.monospace(format!("{:.2}°", r[2].rem_euclid(360.0)));
                ui.label("t");
                ui.monospace(fmt_s(p.t[i]));
                ui.end_row();
            }
        });
        ui.label(RichText::new(format!("{} · body (FRD) in local NED; h above the ellipsoid", p.path)).small().weak());
    }

    fn display_section(&mut self, ui: &mut egui::Ui) {
        let mods = self.camera().map(|c| c.modalities()).unwrap_or_default();
        if mods.contains(&Modality::Depth) {
            ui.horizontal(|ui| {
                ui.label("depth");
                let mut auto = self.style.depth_range.is_none();
                ui.checkbox(&mut auto, "auto");
                if !auto {
                    let (mut a, mut b) = self.style.depth_range.unwrap_or(self.depth_manual);
                    ui.add(egui::DragValue::new(&mut a).speed(1.0).range(0.01..=1e7).suffix(" m"));
                    ui.label("–");
                    ui.add(egui::DragValue::new(&mut b).speed(1.0).range(0.02..=1e7).suffix(" m"));
                    b = b.max(a * 1.001);
                    self.depth_manual = (a, b);
                    self.style.depth_range = Some((a, b));
                } else {
                    // the range shown now, as the start of a manual one
                    if let Some(Some(Ok(p))) = self.cells.get(&Modality::Depth).map(|c| c.pic.as_ref()) {
                        if let Some(d) = p.depth {
                            self.depth_manual = (d.near, d.far);
                        }
                    }
                    self.style.depth_range = None;
                }
                egui::ComboBox::from_id_salt("depth-scale").width(70.0).selected_text(scale_name(self.style.depth_log)).show_ui(ui, |ui| {
                    for v in [None, Some(false), Some(true)] {
                        ui.selectable_value(&mut self.style.depth_log, v, scale_name(v)).on_hover_text("auto: log when the range spans more than a factor 4");
                    }
                });
            });
        }
        if mods.contains(&Modality::Flow) {
            ui.horizontal(|ui| {
                ui.label("flow");
                let mut auto = self.style.flow_max.is_none();
                ui.checkbox(&mut auto, "auto");
                if !auto {
                    let mut m = self.style.flow_max.unwrap_or(self.flow_manual);
                    ui.add(egui::DragValue::new(&mut m).speed(0.05).range(0.01..=1e4).prefix("rim ").suffix(" px"));
                    self.flow_manual = m;
                    self.style.flow_max = Some(m);
                } else {
                    if let Some(Some(Ok(p))) = self.cells.get(&Modality::Flow).map(|c| c.pic.as_ref()) {
                        if let Some(m) = p.flow_max {
                            self.flow_manual = m;
                        }
                    }
                    self.style.flow_max = None;
                }
            });
        }
        if mods.contains(&Modality::Events) {
            let mut ms = self.style.window_us as f64 / 1000.0;
            ui.add(egui::Slider::new(&mut ms, 0.1..=1000.0).logarithmic(true).suffix(" ms").text("event window"));
            self.style.window_us = ((ms * 1000.0).round() as i64).max(1);
            ui.horizontal(|ui| {
                for s in EventStyle::ALL {
                    let ok = s != EventStyle::Gray || self.camera().is_some_and(|c| c.rgb.is_some());
                    ui.add_enabled_ui(ok, |ui| ui.radio_value(&mut self.style.events, s, s.label())).inner.on_disabled_hover_text("this camera has no frames");
                }
            });
        }
        if self.camera().is_some_and(|c| c.stars.is_some()) {
            ui.horizontal(|ui| {
                let mut on = self.style.stars.is_some();
                ui.checkbox(&mut on, "stars up to mag");
                let mut v = self.style.stars.unwrap_or(self.star_mag);
                ui.add_enabled(on, egui::DragValue::new(&mut v).speed(0.05).range(-2.0..=12.0));
                self.star_mag = v;
                self.style.stars = on.then_some(v);
            })
            .response
            .on_hover_text("catalogue stars ringed on the image: yellow on sky pixels, gray behind the terrain");
        }
        if mods.is_empty() {
            ui.label(RichText::new("no camera").weak());
        }
    }

    fn camera_section(&self, ui: &mut egui::Ui) {
        let Some(c) = self.camera() else { return };
        let mut lines = vec![format!("path        {}", c.path), format!("model       {}", c.model), format!("resolution  {} × {}", c.w, c.h)];
        if c.intrinsics.len() == 4 {
            let k = &c.intrinsics;
            lines.push(format!("fx fy       {:.3} {:.3}", k[0], k[1]));
            lines.push(format!("cx cy       {:.3} {:.3}", k[2], k[3]));
        } else if !c.intrinsics.is_empty() {
            lines.push(format!("intrinsics  {:?}", c.intrinsics));
        }
        if !c.distortion.is_empty() {
            lines.push(format!("distortion  {}", c.distortion.iter().map(|d| format!("{d:.5}")).collect::<Vec<_>>().join(" ")));
        }
        if c.t_body_cam.len() == 16 {
            let m = &c.t_body_cam;
            lines.push("T_body_cam".into());
            for r in 0..3 {
                lines.push(format!("  {:+.4} {:+.4} {:+.4}  {:+.3}", m[4 * r], m[4 * r + 1], m[4 * r + 2], m[4 * r + 3]));
            }
        }
        match c.frame_rate() {
            Some(f) => lines.push(format!("frames      {} at {:.2} Hz", c.t.len(), f)),
            None => lines.push(format!("frames      {}", c.t.len())),
        }
        if let Some(k) = &c.depth {
            lines.push(format!("depth       {k}"));
        }
        if let Some(e) = &c.events {
            lines.push(format!("events      {}", e.n));
        }
        ui.label(RichText::new(lines.join("\n")).monospace().size(11.5));
        if !c.camera_yaml.is_empty() {
            egui::CollapsingHeader::new("camera YAML").id_salt(("cam-yaml", self.cam)).show(ui, |ui| {
                ui.label(RichText::new(&c.camera_yaml).monospace().size(11.0));
            });
        }
        if let Some(e) = c.events.as_ref().filter(|e| !e.yaml.is_empty()) {
            egui::CollapsingHeader::new("event simulator").id_salt(("ev-yaml", self.cam)).show(ui, |ui| {
                ui.label(RichText::new(&e.yaml).monospace().size(11.0));
            });
        }
    }

    fn sequence_section(&self, ui: &mut egui::Ui) {
        let s = &self.seq;
        let (a, b) = self.range;
        ui.label(
            RichText::new(format!(
                "{}\nformat {} v{} · t0 {:.3} s · {}\n{} camera(s){}{}",
                s.path.display(),
                if s.format.is_empty() { "?" } else { &s.format },
                s.format_version,
                s.t0,
                fmt_s(b - a),
                s.cameras.len(),
                if s.imu.is_some() { " · IMU" } else { "" },
                if s.pose.is_some() { " · poses" } else { "" }
            ))
            .small(),
        );
        if !s.conventions.is_empty() {
            egui::CollapsingHeader::new("conventions").show(ui, |ui| {
                ui.label(RichText::new(&s.conventions).small());
            });
        }
        if !s.scenario_yaml.is_empty() {
            egui::CollapsingHeader::new("scenario YAML").show(ui, |ui| {
                egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                    ui.label(RichText::new(&s.scenario_yaml).monospace().size(11.0));
                });
            });
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        egui::CollapsingHeader::new(RichText::new("Pose").strong()).default_open(true).show(ui, |ui| self.pose_section(ui));
        if self.seq.imu.is_some() {
            egui::CollapsingHeader::new(RichText::new("IMU").strong()).default_open(true).show(ui, |ui| self.imu_section(ui, 120.0));
        }
        if self.seq.pose.is_some() {
            egui::CollapsingHeader::new(RichText::new("Trajectory").strong()).default_open(true).show(ui, |ui| self.track_section(ui, 200.0));
        }
        if !self.seq.cameras.is_empty() {
            egui::CollapsingHeader::new(RichText::new("Display").strong()).default_open(true).show(ui, |ui| self.display_section(ui));
            egui::CollapsingHeader::new(RichText::new("Camera").strong()).default_open(false).show(ui, |ui| self.camera_section(ui));
        }
        egui::CollapsingHeader::new(RichText::new("Sequence").strong()).default_open(false).show(ui, |ui| self.sequence_section(ui));
        egui::CollapsingHeader::new(RichText::new("Keys").strong()).default_open(false).show(ui, |ui| {
            ui.label(
                RichText::new("Space play / pause · Left / Right: frame (Shift: 10) · Home / End\nhover an image: values of every panel at that pixel").small(),
            );
        });
        let mb = self.loader.cached_bytes() as f64 / 1e6;
        ui.label(RichText::new(format!("cache {mb:.0} MB")).small().weak());
    }
}

impl eframe::App for ShowApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ShowApp::ui(self, ui);
    }
}
