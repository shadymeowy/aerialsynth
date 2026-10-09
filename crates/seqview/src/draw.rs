//! Painting with egui: picture cells with their titles and legends, line plots and the
//! trajectory. Used by the viewer and by the exporter's labelled grids.

use crate::colorize;
use crate::view::{fmt_ms, Legend, Picture};
use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Rect, Stroke};

/// Height of a cell's title (points).
pub const TITLE_H: f32 = 20.0;
/// Height of a cell's legend (points).
pub const LEGEND_H: f32 = 40.0;
const PAD: f32 = 4.0;

pub const TEXT: Color32 = Color32::from_gray(225);
pub const WEAK: Color32 = Color32::from_gray(150);
pub const AXES: [Color32; 3] = [Color32::from_rgb(235, 90, 80), Color32::from_rgb(110, 200, 90), Color32::from_rgb(90, 150, 245)];

/// The grid (columns, rows) for `n` pictures of `w` × `h` px in `avail` that shows them
/// largest, and that scale (points per pixel).
pub fn grid(n: usize, w: usize, h: usize, avail: egui::Vec2) -> (usize, usize, f32) {
    let mut best = (1, n.max(1), 0.0f32);
    for cols in 1..=n.max(1) {
        let rows = n.max(1).div_ceil(cols);
        let cw = avail.x / cols as f32 - 2.0 * PAD;
        let ch = avail.y / rows as f32 - TITLE_H - LEGEND_H - 2.0 * PAD;
        let s = (cw / w.max(1) as f32).min(ch / h.max(1) as f32);
        if s > best.2 {
            best = (cols, rows, s);
        }
    }
    best
}

/// The size of a cell showing a `w` × `h` px picture at `scale`.
pub fn cell_size(w: usize, h: usize, scale: f32) -> egui::Vec2 {
    vec2(w as f32 * scale + 2.0 * PAD + 8.0, h as f32 * scale + TITLE_H + LEGEND_H + 2.0 * PAD)
}

/// The image rect of a cell (centred under its title).
pub fn image_rect(cell: Rect, w: usize, h: usize, scale: f32) -> Rect {
    let size = vec2(w as f32 * scale, h as f32 * scale);
    let top = cell.top() + PAD + TITLE_H;
    Rect::from_min_size(pos2(cell.center().x - size.x / 2.0, top), size)
}

/// Paint a picture's cell: title, image (texture `tex`), stars, crosshair at `cursor` (image
/// px) and legend.
pub fn cell(painter: &egui::Painter, cell: Rect, img: Rect, pic: &Picture, tex: egui::TextureId, cursor: Option<(usize, usize)>) {
    painter.text(pos2(img.left(), cell.top() + PAD + TITLE_H / 2.0), Align2::LEFT_CENTER, &pic.title, FontId::proportional(14.0), TEXT);
    painter.image(tex, img, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    let s = img.width() / pic.w.max(1) as f32;
    let at = |x: f32, y: f32| pos2(img.left() + (x + 0.5) * s, img.top() + (y + 0.5) * s);
    for st in &pic.stars {
        let r = colorize::star_radius(st.v) * s.max(0.6);
        let c = if st.visible { Color32::from_rgb(255, 220, 60) } else { Color32::from_rgba_unmultiplied(160, 160, 160, 120) };
        if img.expand(r).contains(at(st.x, st.y)) {
            painter.circle_stroke(at(st.x, st.y), r, Stroke::new(1.2, c));
        }
    }
    if let Some((x, y)) = cursor {
        let p = at(x as f32, y as f32);
        let st = Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 200));
        let shadow = Stroke::new(3.0, Color32::from_black_alpha(140));
        for (a, b) in
            [(vec2(-9.0, 0.0), vec2(-3.0, 0.0)), (vec2(3.0, 0.0), vec2(9.0, 0.0)), (vec2(0.0, -9.0), vec2(0.0, -3.0)), (vec2(0.0, 3.0), vec2(0.0, 9.0))]
        {
            painter.line_segment([p + a, p + b], shadow);
            painter.line_segment([p + a, p + b], st);
        }
    }
    let lg = Rect::from_min_max(pos2(img.left(), img.bottom() + PAD), pos2(img.right().max(img.left() + 200.0), img.bottom() + PAD + LEGEND_H));
    legend(painter, lg, &pic.legend);
}

/// The flow colour wheel as a mesh (centre white, the rim at full colour).
fn wheel(painter: &egui::Painter, c: egui::Pos2, r: f32) {
    let mut mesh = egui::Mesh::default();
    let (segs, rings) = (48, 4);
    let col = |dx: f32, dy: f32| {
        let [r, g, b] = colorize::flow_color(dx, dy, 1.0);
        Color32::from_rgb(r, g, b)
    };
    mesh.colored_vertex(c, Color32::WHITE);
    for ring in 1..=rings {
        let f = ring as f32 / rings as f32;
        for s in 0..segs {
            let a = s as f32 / segs as f32 * std::f32::consts::TAU;
            let (dx, dy) = (a.cos() * f, a.sin() * f);
            mesh.colored_vertex(c + vec2(dx, dy) * r, col(dx, dy));
        }
    }
    let idx = |ring: u32, s: u32| if ring == 0 { 0 } else { 1 + (ring - 1) * segs + s % segs };
    for ring in 0..rings {
        for s in 0..segs {
            if ring == 0 {
                mesh.add_triangle(0, idx(1, s), idx(1, s + 1));
            } else {
                mesh.add_triangle(idx(ring, s), idx(ring + 1, s), idx(ring + 1, s + 1));
                mesh.add_triangle(idx(ring, s), idx(ring + 1, s + 1), idx(ring, s + 1));
            }
        }
    }
    painter.add(mesh);
}

fn swatch(painter: &egui::Painter, p: egui::Pos2, c: [u8; 3]) -> Rect {
    let r = Rect::from_min_size(p, vec2(11.0, 11.0));
    painter.rect_filled(r, 2.0, Color32::from_rgb(c[0], c[1], c[2]));
    painter.rect_stroke(r, 2.0, Stroke::new(0.5, Color32::from_gray(90)), egui::StrokeKind::Inside);
    r
}

/// Paint a legend into `r`.
pub fn legend(painter: &egui::Painter, r: Rect, l: &Legend) {
    let small = FontId::proportional(12.0);
    let row1 = r.top() + 8.0;
    match l {
        Legend::None => {}
        Legend::Colorbar { colors, lo, hi, note } => {
            if colors.is_empty() {
                painter.text(pos2(r.left(), row1), Align2::LEFT_CENTER, note, small, WEAK);
                return;
            }
            let bar = Rect::from_min_size(pos2(r.left(), r.top() + 2.0), vec2((r.width() * 0.6).clamp(120.0, 360.0), 12.0));
            let n = colors.len();
            for (i, c) in colors.iter().enumerate() {
                let x0 = bar.left() + bar.width() * i as f32 / n as f32;
                let x1 = bar.left() + bar.width() * (i + 1) as f32 / n as f32 + 0.5;
                painter.rect_filled(Rect::from_min_max(pos2(x0, bar.top()), pos2(x1, bar.bottom())), 0.0, Color32::from_rgb(c[0], c[1], c[2]));
            }
            painter.text(pos2(bar.left(), bar.bottom() + 8.0), Align2::LEFT_CENTER, lo, small.clone(), TEXT);
            painter.text(pos2(bar.right(), bar.bottom() + 8.0), Align2::RIGHT_CENTER, hi, small.clone(), TEXT);
            painter.text(pos2(bar.right() + 10.0, bar.center().y), Align2::LEFT_CENTER, note, small, WEAK);
        }
        Legend::Wheel { max } => {
            let c = pos2(r.left() + 14.0, r.top() + 15.0);
            wheel(painter, c, 13.0);
            painter.text(
                pos2(r.left() + 34.0, row1),
                Align2::LEFT_CENTER,
                format!("rim = {max:.2} px · hue = direction: right red, down yellow, left cyan, up violet"),
                small.clone(),
                TEXT,
            );
            painter.text(pos2(r.left() + 34.0, row1 + 15.0), Align2::LEFT_CENTER, "white: still · dimmed: target hidden", small, WEAK);
        }
        Legend::Classes(cls) => {
            let (mut x, mut y) = (r.left(), r.top() + 1.0);
            for (i, (name, c, share)) in cls.iter().enumerate() {
                let text = format!("{name} {:.0}%", share * 100.0);
                let g = painter.layout_no_wrap(text, small.clone(), TEXT);
                let wdt = 15.0 + g.size().x + 10.0;
                if x + wdt > r.right() && x > r.left() {
                    x = r.left();
                    y += 16.0;
                }
                if y + 12.0 > r.bottom() {
                    painter.text(pos2(x, y + 6.0), Align2::LEFT_CENTER, format!("+{} more", cls.len() - i), small.clone(), WEAK);
                    break;
                }
                swatch(painter, pos2(x, y), *c);
                painter.galley(pos2(x + 15.0, y + 5.5 - g.size().y / 2.0), g, TEXT);
                x += wdt;
            }
        }
        Legend::Events { style, window_ms, count } => {
            let mut x = r.left();
            let (on_label, off_label) = match style {
                colorize::EventStyle::TimeSurface => ("ON (latest)", "OFF (latest)"),
                _ => ("ON", "OFF"),
            };
            for (c, t) in [(colorize::ON, on_label), (colorize::OFF, off_label)] {
                swatch(painter, pos2(x, r.top() + 2.0), c);
                let rr = painter.text(pos2(x + 15.0, row1), Align2::LEFT_CENTER, t, small.clone(), TEXT);
                x = rr.right() + 12.0;
            }
            let rate = *count as f64 / (window_ms / 1000.0).max(1e-9);
            painter.text(
                pos2(x, row1),
                Align2::LEFT_CENTER,
                format!(
                    "{count} events in {} ms ({}){}",
                    fmt_ms(*window_ms),
                    fmt_rate(rate),
                    if *style == colorize::EventStyle::Black { String::new() } else { format!(" · {}", style.label()) }
                ),
                small,
                WEAK,
            );
        }
    }
}

fn fmt_rate(r: f64) -> String {
    if r >= 1e6 {
        format!("{:.1} Mev/s", r / 1e6)
    } else if r >= 1e3 {
        format!("{:.0} kev/s", r / 1e3)
    } else {
        format!("{r:.0} ev/s")
    }
}

/// Round tick spacing (1, 2, 5 × 10^k) for about `n` ticks over `span`.
pub fn tick_step(span: f64, n: f64) -> f64 {
    if !(span > 0.0 && span.is_finite()) {
        return 1.0;
    }
    let raw = span / n.max(1.0);
    let p = 10f64.powf(raw.log10().floor());
    let m = raw / p;
    p * if m < 1.5 {
        1.0
    } else if m < 3.5 {
        2.0
    } else if m < 7.5 {
        5.0
    } else {
        10.0
    }
}

fn fmt_tick(v: f64, step: f64) -> String {
    let d = (-step.log10().floor()).clamp(0.0, 6.0) as usize;
    // no "-0"
    let v = if v.abs() < step * 1e-6 { 0.0 } else { v };
    format!("{v:.d$}")
}

/// A line of a plot: name, colour, points (x, y).
pub type Series<'a> = (&'a str, Color32, Vec<(f64, f64)>);

/// A line plot of `series` over x in `xr`, with a vertical line
/// at `cursor`. Allocates `height` points of the ui's width.
pub fn plot(ui: &mut egui::Ui, title: &str, height: f32, xr: (f64, f64), series: &[Series], cursor: Option<f64>) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_gray(22));
    let small = FontId::proportional(11.0);
    let area = Rect::from_min_max(rect.min + vec2(44.0, 16.0), rect.max - vec2(6.0, 16.0));
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for (_, _, pts) in series {
        for &(_, y) in pts {
            if y.is_finite() {
                lo = lo.min(y);
                hi = hi.max(y);
            }
        }
    }
    if lo > hi {
        painter.text(rect.center(), Align2::CENTER_CENTER, "no samples here", small, WEAK);
        painter.text(rect.left_top() + vec2(6.0, 2.0), Align2::LEFT_TOP, title, FontId::proportional(12.0), TEXT);
        return;
    }
    let pad = ((hi - lo) * 0.08).max(1e-6);
    let (lo, hi) = (lo - pad, hi + pad);
    let (x0, x1) = (xr.0, xr.1.max(xr.0 + 1e-9));
    let px = |x: f64, y: f64| pos2(area.left() + ((x - x0) / (x1 - x0)) as f32 * area.width(), area.bottom() - ((y - lo) / (hi - lo)) as f32 * area.height());
    let grid = Stroke::new(0.5, Color32::from_gray(55));
    let ys = tick_step(hi - lo, 4.0);
    let mut y = (lo / ys).ceil() * ys;
    while y <= hi {
        let p = px(x0, y);
        painter.line_segment([pos2(area.left(), p.y), pos2(area.right(), p.y)], grid);
        painter.text(pos2(area.left() - 4.0, p.y), Align2::RIGHT_CENTER, fmt_tick(y, ys), small.clone(), WEAK);
        y += ys;
    }
    let xs = tick_step(x1 - x0, 5.0);
    let mut x = (x0 / xs).ceil() * xs;
    while x <= x1 {
        let p = px(x, lo);
        painter.line_segment([pos2(p.x, area.top()), pos2(p.x, area.bottom())], grid);
        painter.text(pos2(p.x, area.bottom() + 2.0), Align2::CENTER_TOP, fmt_tick(x, xs), small.clone(), WEAK);
        x += xs;
    }
    for (_, c, pts) in series {
        let line: Vec<egui::Pos2> = pts.iter().filter(|p| p.1.is_finite()).map(|&(x, y)| px(x, y)).collect();
        if line.len() >= 2 {
            painter.add(egui::Shape::line(line, Stroke::new(1.2, *c)));
        }
    }
    if let Some(t) = cursor.filter(|t| *t >= x0 && *t <= x1) {
        let p = px(t, lo);
        painter.line_segment([pos2(p.x, area.top()), pos2(p.x, area.bottom())], Stroke::new(1.0, Color32::from_gray(200)));
    }
    let mut lx = rect.left() + 6.0;
    let r = painter.text(pos2(lx, rect.top() + 2.0), Align2::LEFT_TOP, title, FontId::proportional(12.0), TEXT);
    lx = r.right() + 10.0;
    for (name, c, _) in series {
        let r = painter.text(pos2(lx, rect.top() + 3.0), Align2::LEFT_TOP, *name, small.clone(), *c);
        lx = r.right() + 8.0;
    }
}

/// The trajectory seen from above: `pts` (east, north) in m, the position `cur` and heading
/// `yaw` (deg from north).
pub fn track(ui: &mut egui::Ui, height: f32, pts: &[[f64; 2]], cur: Option<usize>, yaw: Option<f64>) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_gray(22));
    let small = FontId::proportional(11.0);
    if pts.is_empty() {
        painter.text(rect.center(), Align2::CENTER_CENTER, "no poses", small, WEAK);
        return;
    }
    let (mut e0, mut e1, mut n0, mut n1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in pts {
        (e0, e1, n0, n1) = (e0.min(p[0]), e1.max(p[0]), n0.min(p[1]), n1.max(p[1]));
    }
    let area = rect.shrink(14.0);
    let span = (e1 - e0).max(n1 - n0).max(1.0);
    let s = (area.width().min(area.height()) as f64 / span) as f32;
    let (ce, cn) = ((e0 + e1) / 2.0, (n0 + n1) / 2.0);
    let px = |p: [f64; 2]| area.center() + vec2(((p[0] - ce) as f32) * s, -((p[1] - cn) as f32) * s);
    let step = pts.len().div_ceil(4000).max(1);
    let line: Vec<egui::Pos2> = pts.iter().step_by(step).map(|p| px(*p)).collect();
    painter.add(egui::Shape::line(line, Stroke::new(1.5, Color32::from_rgb(120, 170, 230))));
    painter.circle_filled(px(pts[0]), 3.0, Color32::from_rgb(110, 200, 90));
    if let Some(i) = cur.filter(|i| *i < pts.len()) {
        let p = px(pts[i]);
        if let Some(y) = yaw {
            let d = vec2(y.to_radians().sin() as f32, -y.to_radians().cos() as f32);
            painter.arrow(p, d * 18.0, Stroke::new(2.0, Color32::from_rgb(255, 200, 60)));
        }
        painter.circle_filled(p, 4.0, Color32::from_rgb(255, 200, 60));
    }
    // scale bar
    let bar_m = tick_step(span, 3.0);
    let bl = pos2(rect.left() + 8.0, rect.bottom() - 8.0);
    painter.line_segment([bl, bl + vec2((bar_m as f32) * s, 0.0)], Stroke::new(2.0, TEXT));
    let label = if bar_m >= 1000.0 { format!("{} km", bar_m / 1000.0) } else { format!("{bar_m} m") };
    painter.text(bl + vec2(0.0, -4.0), Align2::LEFT_BOTTOM, label, small.clone(), TEXT);
    let n = rect.right_top() + vec2(-12.0, 22.0);
    painter.arrow(n, vec2(0.0, -12.0), Stroke::new(1.0, WEAK));
    painter.text(n + vec2(-6.0, -6.0), Align2::RIGHT_CENTER, "N", small, WEAK);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_picks_the_largest_layout() {
        // four 4:3 pictures in a wide area: 2 × 2 beats 4 × 1 and 1 × 4
        let (c, r, s) = grid(4, 640, 480, vec2(1400.0, 1100.0));
        assert_eq!((c, r), (2, 2));
        assert!(s > 0.9 && s < 1.1, "{s}");
        // a very wide area: one row
        assert_eq!(grid(3, 640, 480, vec2(3000.0, 600.0)).0, 3);
        assert_eq!(grid(1, 10, 10, vec2(100.0, 100.0)).0, 1);
    }

    #[test]
    fn ticks_are_round() {
        assert_eq!(tick_step(10.0, 5.0), 2.0);
        assert_eq!(tick_step(0.9, 4.0), 0.2);
        assert_eq!(tick_step(3000.0, 3.0), 1000.0);
        assert_eq!(tick_step(0.0, 3.0), 1.0);
        assert_eq!(fmt_tick(0.2, 0.2), "0.2");
        assert_eq!(fmt_tick(1000.0, 1000.0), "1000");
        assert_eq!(fmt_tick(-1e-17, 0.5), "0.0");
    }
}
