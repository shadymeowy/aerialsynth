//! A small CPU renderer of egui output (triangles with vertex colours and textures, clip rects,
//! premultiplied alpha blended in gamma space, as egui-wgpu does): headless snapshots of the
//! viewer and the labelled grids of the exporter, without a GPU or a display.

use eframe::egui::{self, epaint};
use std::collections::HashMap;

struct Texture {
    w: usize,
    h: usize,
    px: Vec<[u8; 4]>,
    linear: bool,
}

/// The textures egui has sent, by id.
#[derive(Default)]
pub struct SoftRenderer {
    textures: HashMap<egui::TextureId, Texture>,
}

/// An RGBA8 image (premultiplied colour, opaque where the background was).
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[u8; 4]>,
}

impl Image {
    /// RGB8, dropping alpha.
    pub fn rgb(&self) -> Vec<u8> {
        self.px.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
    }
}

impl SoftRenderer {
    /// Apply the texture uploads and frees of a frame.
    pub fn update(&mut self, delta: &egui::TexturesDelta) {
        for (id, deltas) in &delta.set {
            for d in deltas {
                let egui::ImageData::Color(img) = &d.image;
                let [w, h] = img.size;
                let px: Vec<[u8; 4]> = img.pixels.iter().map(|c| c.to_array()).collect();
                let linear = d.options.magnification == egui::TextureFilter::Linear;
                match d.pos {
                    None => {
                        self.textures.insert(*id, Texture { w, h, px, linear });
                    }
                    Some([x0, y0]) => {
                        if let Some(t) = self.textures.get_mut(id) {
                            for y in 0..h.min(t.h.saturating_sub(y0)) {
                                for x in 0..w.min(t.w.saturating_sub(x0)) {
                                    t.px[(y0 + y) * t.w + x0 + x] = px[y * w + x];
                                }
                            }
                        }
                    }
                }
            }
        }
        for id in &delta.free {
            self.textures.remove(id);
        }
    }

    /// Rasterize `prims` into a `w` × `h` px image cleared to `clear`.
    pub fn render(&self, prims: &[epaint::ClippedPrimitive], ppp: f32, w: usize, h: usize, clear: egui::Color32) -> Image {
        let mut img = Image { w, h, px: vec![clear.to_array(); w * h] };
        for p in prims {
            let epaint::Primitive::Mesh(mesh) = &p.primitive else { continue };
            let c = p.clip_rect;
            let clip = [
                ((c.min.x * ppp).round().max(0.0) as usize).min(w),
                ((c.min.y * ppp).round().max(0.0) as usize).min(h),
                ((c.max.x * ppp).round().max(0.0) as usize).min(w),
                ((c.max.y * ppp).round().max(0.0) as usize).min(h),
            ];
            if clip[0] >= clip[2] || clip[1] >= clip[3] {
                continue;
            }
            let tex = self.textures.get(&mesh.texture_id);
            for tri in mesh.indices.as_chunks::<3>().0 {
                let v = [&mesh.vertices[tri[0] as usize], &mesh.vertices[tri[1] as usize], &mesh.vertices[tri[2] as usize]];
                raster_triangle(&mut img, clip, ppp, v, tex);
            }
        }
        img
    }
}

fn sample(tex: &Texture, u: f32, v: f32) -> [f32; 4] {
    let (w, h, px, linear) = (tex.w, tex.h, &tex.px, tex.linear);
    if w == 0 || h == 0 {
        return [1.0; 4];
    }
    let get = |x: isize, y: isize| -> [f32; 4] {
        let p = px[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
        [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
    };
    let (x, y) = (u * w as f32 - 0.5, v * h as f32 - 0.5);
    if !linear {
        let p = get(x.round() as isize, y.round() as isize);
        return p.map(|c| c / 255.0);
    }
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let (a, b, c, d) = (get(x0, y0), get(x0 + 1, y0), get(x0, y0 + 1), get(x0 + 1, y0 + 1));
    std::array::from_fn(|k| ((a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy) / 255.0)
}

/// Fill one triangle (pixel centres inside, top-left rule on shared edges).
fn raster_triangle(img: &mut Image, clip: [usize; 4], ppp: f32, v: [&epaint::Vertex; 3], tex: Option<&Texture>) {
    let p: [(f32, f32); 3] = v.map(|v| (v.pos.x * ppp, v.pos.y * ppp));
    let area = (p[1].0 - p[0].0) * (p[2].1 - p[0].1) - (p[1].1 - p[0].1) * (p[2].0 - p[0].0);
    if area.abs() < 1e-12 {
        return;
    }
    // orient counter-clockwise in y-down screen space (positive area)
    let (p, v) = if area < 0.0 { ([p[0], p[2], p[1]], [v[0], v[2], v[1]]) } else { (p, v) };
    let area = area.abs();
    let minx = p.iter().map(|q| q.0).fold(f32::MAX, f32::min).floor().max(clip[0] as f32) as usize;
    let maxx = (p.iter().map(|q| q.0).fold(f32::MIN, f32::max).ceil() as usize).min(clip[2]);
    let miny = p.iter().map(|q| q.1).fold(f32::MAX, f32::min).floor().max(clip[1] as f32) as usize;
    let maxy = (p.iter().map(|q| q.1).fold(f32::MIN, f32::max).ceil() as usize).min(clip[3]);
    if minx >= maxx || miny >= maxy {
        return;
    }
    let edge = |a: (f32, f32), b: (f32, f32), x: f32, y: f32| (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0);
    // top-left rule: an edge owns the pixel centres exactly on it when it is a top or left edge
    let owns = |a: (f32, f32), b: (f32, f32)| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        (dy == 0.0 && dx < 0.0) || dy > 0.0
    };
    let tl = [owns(p[1], p[2]), owns(p[2], p[0]), owns(p[0], p[1])];
    let col: [[f32; 4]; 3] = v.map(|v| v.color.to_array().map(|c| c as f32 / 255.0));
    for y in miny..maxy {
        let yc = y as f32 + 0.5;
        for x in minx..maxx {
            let xc = x as f32 + 0.5;
            let w = [edge(p[1], p[2], xc, yc), edge(p[2], p[0], xc, yc), edge(p[0], p[1], xc, yc)];
            if (0..3).any(|i| w[i] < 0.0 || (w[i] == 0.0 && !tl[i])) {
                continue;
            }
            let b = w.map(|x| x / area);
            let mut src: [f32; 4] = std::array::from_fn(|k| b[0] * col[0][k] + b[1] * col[1][k] + b[2] * col[2][k]);
            if let Some(t) = tex {
                let u = b[0] * v[0].uv.x + b[1] * v[1].uv.x + b[2] * v[2].uv.x;
                let vv = b[0] * v[0].uv.y + b[1] * v[1].uv.y + b[2] * v[2].uv.y;
                let s = sample(t, u, vv);
                src = std::array::from_fn(|k| src[k] * s[k]);
            }
            let dst = &mut img.px[y * img.w + x];
            let a = src[3].clamp(0.0, 1.0);
            for k in 0..4 {
                let d = dst[k] as f32 / 255.0;
                dst[k] = ((src[k] + d * (1.0 - a)).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
}

/// An egui context rendered on the CPU.
#[derive(Default)]
pub struct Headless {
    pub ctx: egui::Context,
    renderer: SoftRenderer,
}

impl Headless {
    /// Run `ui` for a `w` × `h` point screen at `ppp` pixels per point (`passes` passes, so
    /// egui settles its layout) and rasterize the last. `events`: input of every pass (e.g. a
    /// pointer position).
    pub fn run(&mut self, (w, h): (f32, f32), ppp: f32, passes: usize, events: Vec<egui::Event>, ui: impl FnMut(&mut egui::Ui)) -> Image {
        run_headless(&self.ctx, &mut self.renderer, (w, h, ppp), passes, events, ui)
    }
}

fn run_headless(
    ctx: &egui::Context,
    renderer: &mut SoftRenderer,
    (w, h, ppp): (f32, f32, f32),
    passes: usize,
    events: Vec<egui::Event>,
    mut ui: impl FnMut(&mut egui::Ui),
) -> Image {
    let mut out = None;
    for pass in 0..passes.max(1) {
        let last = pass + 1 == passes.max(1);
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h))),
            time: Some(pass as f64 * 0.05),
            predicted_dt: 0.05,
            ..Default::default()
        };
        input.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
        // the pointer arrives in every pass (hover state builds up over passes)
        input.events = events.clone();
        let full = ctx.run_ui(input, &mut ui);
        renderer.update(&full.textures_delta);
        if last {
            let prims = ctx.tessellate(full.shapes, full.pixels_per_point);
            let (pw, ph) = ((w * full.pixels_per_point).round() as usize, (h * full.pixels_per_point).round() as usize);
            out = Some(renderer.render(&prims, full.pixels_per_point, pw, ph, ctx.global_style().visuals.panel_fill));
        }
    }
    out.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rect_and_text_are_drawn() {
        let img = Headless::default().run((200.0, 100.0), 1.0, 2, vec![], |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(50.0, 40.0)), 0.0, egui::Color32::from_rgb(255, 0, 0));
                ui.painter().text(egui::pos2(100.0, 50.0), egui::Align2::LEFT_TOP, "Hello", egui::FontId::proportional(20.0), egui::Color32::WHITE);
            });
        });
        assert_eq!((img.w, img.h), (200, 100));
        assert_eq!(img.px[20 * 200 + 20], [255, 0, 0, 255]);
        // no doubled diagonal on the rectangle (top-left rule)
        assert_eq!(img.px[30 * 200 + 30], [255, 0, 0, 255]);
        // text: some bright pixels where it is, none far from it
        let bright = |x0: usize, x1: usize, y0: usize, y1: usize| {
            (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| img.px[y * 200 + x][1] > 200).count()
        };
        assert!(bright(100, 160, 50, 75) > 20);
        assert_eq!(bright(10, 60, 60, 95), 0);
    }

    #[test]
    fn textured_quads_sample_the_image() {
        let mut tex = None;
        let img = Headless::default().run((64.0, 64.0), 1.0, 2, vec![], |ui| {
            let t = tex.get_or_insert_with(|| {
                let px: Vec<u8> = (0..4 * 4).flat_map(|i| if i % 2 == 0 { [0, 0, 255] } else { [0, 255, 0] }).collect();
                ui.ctx().load_texture("t", egui::ColorImage::from_rgb([4, 4], &px), egui::TextureOptions::NEAREST)
            });
            ui.painter().image(
                t.id(),
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(64.0, 64.0)),
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        });
        // texel (0, 0) blue, (1, 0) green; 16 px each
        assert_eq!(img.px[8 * 64 + 8], [0, 0, 255, 255]);
        assert_eq!(img.px[8 * 64 + 24], [0, 255, 0, 255]);
    }
}
