//! Colour maps of the ground truth (shared by the viewer, its snapshots and the exporter): depth
//! (turbo, near = red; sky black), optical flow (the Middlebury colour wheel), land cover (the
//! generator's class palette; sky black) and events (accumulated over a window: ON red / OFF
//! blue on black or on the gray frame, or a time surface). Every image is packed RGB8, row-major.

/// Colour of the sky (depth +inf, land cover 255) and of pixels without a value.
pub const SKY: [u8; 3] = [0, 0, 0];
/// ON events.
pub const ON: [u8; 3] = [255, 48, 40];
/// OFF events.
pub const OFF: [u8; 3] = [40, 110, 255];
/// Land cover class id of the sky.
pub const LANDCOVER_SKY: u8 = 255;

/// Turbo colour map (Mikhailov 2019, polynomial fit) of `t` in [0, 1].
pub fn turbo(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = 0.135_721_38 + t * (4.615_392_6 + t * (-42.660_32 + t * (132.131_08 + t * (-152.942_4 + t * 59.286_38))));
    let g = 0.091_402_61 + t * (2.194_188_4 + t * (4.842_966_6 + t * (-14.185_033 + t * (4.277_298_6 + t * 2.829_566))));
    let b = 0.106_673_3 + t * (12.641_946 + t * (-60.582_05 + t * (110.362_77 + t * (-89.903_11 + t * 27.348_25))));
    [(r.clamp(0.0, 1.0) * 255.0).round() as u8, (g.clamp(0.0, 1.0) * 255.0).round() as u8, (b.clamp(0.0, 1.0) * 255.0).round() as u8]
}

/// The `q`-quantile (0..=1) of `v` (sorted in place). None when empty.
fn quantile(v: &mut [f32], q: f64) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    let i = ((v.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    let (_, x, _) = v.select_nth_unstable_by(i, |a, b| a.total_cmp(b));
    Some(*x)
}

/// At most ~`n` of the values selected by `f` from `len` items, evenly strided.
fn sample(len: usize, n: usize, f: impl Fn(usize) -> Option<f32>) -> Vec<f32> {
    let step = len.div_ceil(n.max(1)).max(1);
    (0..len).step_by(step).filter_map(f).collect()
}

/// How depth maps to colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthStyle {
    /// Depth (m) shown red and blue.
    pub near: f32,
    pub far: f32,
    /// Logarithmic scale (oblique views: the horizon is far away).
    pub log: bool,
}

impl DepthStyle {
    /// Position of depth `d` on the colour bar (0 = near, 1 = far); None for the sky.
    pub fn t(&self, d: f32) -> Option<f32> {
        if !(d.is_finite() && d > 0.0) {
            return None;
        }
        let (a, b) = (self.near.max(1e-3), self.far.max(self.near.max(1e-3) * (1.0 + 1e-6)));
        Some(if self.log { (d.max(1e-3).ln() - a.ln()) / (b.ln() - a.ln()) } else { (d - a) / (b - a) }.clamp(0.0, 1.0))
    }

    /// The colour of depth `d` (the sky: [`SKY`]).
    pub fn color(&self, d: f32) -> [u8; 3] {
        self.t(d).map_or(SKY, |t| turbo(1.0 - t))
    }
}

/// A robust depth range: the 2nd and 98th percentiles of the finite depths of the frames
/// (sampled; the plain maximum is the far horizon in oblique views). None when all is sky.
pub fn depth_range(frames: &[&[f32]]) -> Option<(f32, f32)> {
    let mut v: Vec<f32> = Vec::new();
    for d in frames {
        v.extend(sample(d.len(), 100_000 / frames.len().max(1), |i| Some(d[i]).filter(|x| x.is_finite() && *x > 0.0)));
    }
    let lo = quantile(&mut v, 0.02)?;
    let hi = quantile(&mut v, 0.98)?;
    Some(if hi > lo { (lo, hi) } else { (lo * 0.99, lo * 1.01 + 1e-3) })
}

/// Depth map (m; +inf = sky) to RGB.
pub fn depth_rgb(depth: &[f32], style: &DepthStyle) -> Vec<u8> {
    depth.iter().flat_map(|&d| style.color(d)).collect()
}

/// The Middlebury colour wheel (Baker et al. 2011; as in flowlib / flow_vis): 55 hues.
fn wheel() -> &'static [[f32; 3]] {
    static W: std::sync::OnceLock<Vec<[f32; 3]>> = std::sync::OnceLock::new();
    W.get_or_init(|| {
        let seg: [(usize, [f32; 3], [f32; 3]); 6] = [
            (15, [255.0, 0.0, 0.0], [255.0, 255.0, 0.0]),
            (6, [255.0, 255.0, 0.0], [0.0, 255.0, 0.0]),
            (4, [0.0, 255.0, 0.0], [0.0, 255.0, 255.0]),
            (11, [0.0, 255.0, 255.0], [0.0, 0.0, 255.0]),
            (13, [0.0, 0.0, 255.0], [255.0, 0.0, 255.0]),
            (6, [255.0, 0.0, 255.0], [255.0, 0.0, 0.0]),
        ];
        let mut v = Vec::new();
        for (n, a, b) in seg {
            for i in 0..n {
                let f = i as f32 / n as f32;
                v.push(std::array::from_fn(|k| (a[k] + (b[k] - a[k]) * f) / 255.0));
            }
        }
        v
    })
}

/// Colour of the flow vector (dx, dy) (px) with `max` (px) at full saturation: the hue is the
/// direction (Middlebury: right red, down yellow, left cyan, up violet), white is no motion;
/// beyond `max` the colour darkens.
pub fn flow_color(dx: f32, dy: f32, max: f32) -> [u8; 3] {
    if !(dx.is_finite() && dy.is_finite()) {
        return SKY;
    }
    let w = wheel();
    let rad = (dx * dx + dy * dy).sqrt() / max.max(1e-6);
    let ang = (-dy).atan2(-dx) / std::f32::consts::PI;
    let fk = (ang + 1.0) / 2.0 * (w.len() - 1) as f32;
    let k0 = (fk.floor() as usize).min(w.len() - 1);
    let k1 = (k0 + 1) % w.len();
    let f = fk - k0 as f32;
    std::array::from_fn(|c| {
        let col = (1.0 - f) * w[k0][c] + f * w[k1][c];
        let col = if rad <= 1.0 { 1.0 - rad * (1.0 - col) } else { col * 0.75 };
        (col.clamp(0.0, 1.0) * 255.0).round() as u8
    })
}

/// A robust flow scale: the 99th percentile of the valid flow magnitudes of the frames (px,
/// at least 0.01).
pub fn flow_max(frames: &[(&[f32], Option<&[u8]>)]) -> f32 {
    let mut v: Vec<f32> = Vec::new();
    for (f, valid) in frames {
        let n = f.len() / 2;
        v.extend(sample(n, 100_000 / frames.len().max(1), |i| {
            let m = f[2 * i].hypot(f[2 * i + 1]);
            (m.is_finite() && valid.is_none_or(|v| v[i] != 0)).then_some(m)
        }));
    }
    quantile(&mut v, 0.99).unwrap_or(1.0).max(0.01)
}

/// Flow (dx, dy per pixel, px) to RGB; pixels whose target is not visible (`valid` 0) dimmed.
pub fn flow_rgb(flow: &[f32], valid: Option<&[u8]>, max: f32) -> Vec<u8> {
    let n = flow.len() / 2;
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let c = flow_color(flow[2 * i], flow[2 * i + 1], max);
        if valid.is_some_and(|v| v[i] == 0) {
            out.extend(c.map(|x| x / 4));
        } else {
            out.extend(c);
        }
    }
    out
}

/// A `size`² image of the colour wheel (legend): the flow at a pixel is its offset from the
/// centre, the rim at `max`; outside the disc transparent (alpha 0). RGBA.
pub fn flow_wheel_rgba(size: usize) -> Vec<u8> {
    let c = (size as f32 - 1.0) / 2.0;
    let mut out = Vec::with_capacity(size * size * 4);
    for y in 0..size {
        for x in 0..size {
            let (dx, dy) = (x as f32 - c, y as f32 - c);
            let r = (dx * dx + dy * dy).sqrt() / c.max(1.0);
            let col = flow_color(dx, dy, c.max(1.0));
            out.extend(col);
            out.push(if r <= 1.0 { 255 } else { 0 });
        }
    }
    out
}

/// Colour of land cover class `c` (the generator's palette; 255 = sky).
pub fn landcover_color(c: u8) -> [u8; 3] {
    if c == LANDCOVER_SKY {
        SKY
    } else {
        terragen::landcover::palette(c)
    }
}

/// Name of class `c`: from the file's `class_names` when it has one, else the generator's.
pub fn class_name(names: &[String], c: u8) -> String {
    if c == LANDCOVER_SKY {
        return "sky".into();
    }
    names.get(c as usize).cloned().or_else(|| terragen::landcover::NAMES.get(c as usize).map(|s| s.to_string())).unwrap_or_else(|| format!("class {c}"))
}

/// Land cover classes (255 = sky) to RGB.
pub fn landcover_rgb(lc: &[u8]) -> Vec<u8> {
    let lut: Vec<[u8; 3]> = (0..=255u8).map(landcover_color).collect();
    lc.iter().flat_map(|&c| lut[c as usize]).collect()
}

/// Pixel count of each class in `lc` (index = class id).
pub fn class_histogram(lc: &[u8]) -> [usize; 256] {
    let mut h = [0usize; 256];
    for &c in lc {
        h[c as usize] += 1;
    }
    h
}

/// Gray (1 channel) or RGB (3 channels) pixels as RGB.
pub fn to_rgb(px: &[u8], channels: usize) -> Vec<u8> {
    if channels == 3 {
        px.to_vec()
    } else {
        px.iter().flat_map(|&g| [g, g, g]).collect()
    }
}

/// Events as columns (x, y, t µs, p: 1 = ON).
#[derive(Clone, Copy, Debug, Default)]
pub struct Events<'a> {
    pub x: &'a [u16],
    pub y: &'a [u16],
    pub t: &'a [i64],
    pub p: &'a [i8],
}

/// Events of one window accumulated per pixel.
#[derive(Clone, Debug, Default)]
pub struct EventFrame {
    pub w: usize,
    pub h: usize,
    /// window (t_end - window, t_end], µs
    pub t_end: i64,
    pub window: i64,
    pub on: Vec<u32>,
    pub off: Vec<u32>,
    /// time (µs) and polarity of the latest event of each pixel (i64::MIN: none)
    pub last_t: Vec<i64>,
    pub last_on: Vec<bool>,
    /// events in the window
    pub count: usize,
}

impl EventFrame {
    /// Accumulate the events with t in (t_end - window, t_end]; events outside the w × h sensor
    /// are ignored.
    pub fn accumulate(w: usize, h: usize, t_end: i64, window: i64, ev: Events) -> EventFrame {
        let Events { x, y, t, p } = ev;
        let n = w * h;
        let mut f = EventFrame { w, h, t_end, window, on: vec![0; n], off: vec![0; n], last_t: vec![i64::MIN; n], last_on: vec![false; n], count: 0 };
        let t0 = t_end - window;
        for i in 0..x.len().min(y.len()).min(t.len()).min(p.len()) {
            let (xi, yi, ti) = (x[i] as usize, y[i] as usize, t[i]);
            if ti <= t0 || ti > t_end || xi >= w || yi >= h {
                continue;
            }
            let j = yi * w + xi;
            if p[i] > 0 {
                f.on[j] += 1;
            } else {
                f.off[j] += 1;
            }
            if ti >= f.last_t[j] {
                f.last_t[j] = ti;
                f.last_on[j] = p[i] > 0;
            }
            f.count += 1;
        }
        f
    }

    /// Bytes held (for the cache).
    pub fn bytes(&self) -> usize {
        self.on.len() * 17
    }
}

/// How events are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventStyle {
    /// ON red / OFF blue on black
    Black,
    /// ON red / OFF blue on the camera's gray frame (dimmed)
    Gray,
    /// time surface: the latest event per pixel, fading with its age over the window
    TimeSurface,
}

impl EventStyle {
    pub const ALL: [EventStyle; 3] = [EventStyle::Black, EventStyle::Gray, EventStyle::TimeSurface];

    pub fn name(self) -> &'static str {
        match self {
            EventStyle::Black => "black",
            EventStyle::Gray => "gray",
            EventStyle::TimeSurface => "surface",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            EventStyle::Black => "ON / OFF on black",
            EventStyle::Gray => "ON / OFF on the gray frame",
            EventStyle::TimeSurface => "time surface",
        }
    }

    pub fn parse(s: &str) -> Option<EventStyle> {
        EventStyle::ALL.into_iter().find(|e| e.name() == s)
    }
}

fn mix(a: [u8; 3], b: [u8; 3], f: f32) -> [u8; 3] {
    std::array::from_fn(|k| (a[k] as f32 + (b[k] as f32 - a[k] as f32) * f).round() as u8)
}

/// Accumulated events to RGB. `background`: the gray frame (RGB, w × h) for
/// [`EventStyle::Gray`] (black without one). A pixel's colour is the mix of ON red and OFF blue
/// by its counts, brighter with more events (full at 3).
pub fn events_rgb(ev: &EventFrame, style: EventStyle, background: Option<&[u8]>) -> Vec<u8> {
    let n = ev.w * ev.h;
    let mut out = Vec::with_capacity(n * 3);
    let tau = (ev.window as f64 / 3.0).max(1.0);
    for i in 0..n {
        let bg = match (style, background) {
            (EventStyle::Gray, Some(b)) if b.len() >= 3 * n => {
                let g = ((b[3 * i] as u32 * 299 + b[3 * i + 1] as u32 * 587 + b[3 * i + 2] as u32 * 114) / 1000) as f32;
                let g = (g * 0.6) as u8;
                [g, g, g]
            }
            _ => [0, 0, 0],
        };
        let c = if style == EventStyle::TimeSurface {
            if ev.last_t[i] == i64::MIN {
                bg
            } else {
                let a = (-((ev.t_end - ev.last_t[i]) as f64) / tau).exp() as f32;
                mix(bg, if ev.last_on[i] { ON } else { OFF }, a)
            }
        } else {
            let (on, off) = (ev.on[i], ev.off[i]);
            let k = on + off;
            if k == 0 {
                bg
            } else {
                let col = mix(OFF, ON, on as f32 / k as f32);
                mix(bg, col, 0.55 + 0.45 * (k.min(3) as f32 - 1.0) / 2.0)
            }
        };
        out.extend(c);
    }
    out
}

/// Draw a ring of radius `r` (px) around (cx, cy) (pixel centres at integers) into an RGB image.
pub fn draw_ring(img: &mut [u8], w: usize, h: usize, cx: f32, cy: f32, r: f32, color: [u8; 3]) {
    let (x0, x1) = ((cx - r - 1.0).floor().max(0.0) as usize, ((cx + r + 1.0).ceil().max(0.0) as usize).min(w.saturating_sub(1)));
    let (y0, y1) = ((cy - r - 1.0).floor().max(0.0) as usize, ((cy + r + 1.0).ceil().max(0.0) as usize).min(h.saturating_sub(1)));
    if !(cx.is_finite() && cy.is_finite()) || cx + r + 1.0 < 0.0 || cy + r + 1.0 < 0.0 {
        return;
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            let a = (1.0 - (d - r).abs()).clamp(0.0, 1.0);
            if a > 0.0 {
                let j = 3 * (y * w + x);
                let c = mix([img[j], img[j + 1], img[j + 2]], color, a);
                img[j..j + 3].copy_from_slice(&c);
            }
        }
    }
}

/// Ring radius (px) of a star of magnitude `v`.
pub fn star_radius(v: f32) -> f32 {
    (7.0 - v).clamp(2.5, 9.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_ends_and_clamps() {
        assert_eq!(turbo(-1.0), turbo(0.0));
        assert_eq!(turbo(2.0), turbo(1.0));
        // dark at 0, blue at 0.15, green in the middle, dark red at 1
        let (a, q, m, b) = (turbo(0.0), turbo(0.15), turbo(0.5), turbo(1.0));
        assert!(a.iter().all(|c| *c < 60), "{a:?}");
        assert!(q[2] > q[0] && q[2] > q[1], "{q:?}");
        assert!(m[1] > m[0] && m[1] > m[2], "{m:?}");
        assert!(b[0] > 100 && b[0] > 4 * b[2], "{b:?}");
    }

    #[test]
    fn depth_colours_near_red_far_blue_sky_black() {
        let s = DepthStyle { near: 100.0, far: 200.0, log: false };
        let near = s.color(100.0);
        let far = s.color(200.0);
        let mid = s.color(170.0);
        assert!(near[0] > 4 * near[2] && mid[2] > mid[0] && far.iter().all(|c| *c < 60), "{near:?} {mid:?} {far:?}");
        assert_eq!(s.color(f32::INFINITY), SKY);
        assert_eq!(s.color(f32::NAN), SKY);
        assert_eq!(s.color(50.0), near); // clamped
        assert_eq!(s.t(150.0), Some(0.5));
        let l = DepthStyle { log: true, near: 10.0, far: 1000.0 };
        assert!((l.t(100.0).unwrap() - 0.5).abs() < 1e-5);
        assert_eq!(depth_rgb(&[100.0, f32::INFINITY], &s), [near, SKY].concat());
    }

    #[test]
    fn depth_range_is_robust() {
        let mut d: Vec<f32> = (0..1000).map(|i| 100.0 + i as f32 * 0.1).collect();
        d.extend([f32::INFINITY; 500]);
        d[0] = 1e7; // a far outlier
        let (lo, hi) = depth_range(&[&d]).unwrap();
        assert!((lo - 102.0).abs() < 0.5 && (hi - 198.0).abs() < 0.5, "{lo} {hi}");
        assert!(depth_range(&[&[f32::INFINITY; 4]]).is_none());
        let (lo, hi) = depth_range(&[&[5.0; 10]]).unwrap();
        assert!(lo < hi);
    }

    #[test]
    fn flow_wheel_directions() {
        // no motion: white; right red, left cyan; the saturation grows with the magnitude
        assert_eq!(flow_color(0.0, 0.0, 1.0), [255, 255, 255]);
        let right = flow_color(1.0, 0.0, 1.0);
        assert!(right[0] > 200 && right[1] < 60 && right[2] < 60, "{right:?}");
        let left = flow_color(-1.0, 0.0, 1.0);
        assert!(left[2] > 150 && left[0] < 100, "{left:?}");
        let half = flow_color(0.5, 0.0, 1.0);
        assert!(half[1] > right[1] && half[0] >= 250, "{half:?}");
        // beyond the scale: darker
        let over = flow_color(2.0, 0.0, 1.0);
        assert!(over[0] < right[0]);
        assert_eq!(flow_color(f32::NAN, 0.0, 1.0), SKY);
        // invalid pixels are dimmed
        let img = flow_rgb(&[1.0, 0.0, 1.0, 0.0], Some(&[1, 0]), 1.0);
        assert_eq!(&img[..3], &right);
        assert_eq!(&img[3..], &right.map(|x| x / 4));
    }

    #[test]
    fn flow_scale_is_the_99th_percentile() {
        let f: Vec<f32> = (0..1000).flat_map(|i| [i as f32 / 100.0, 0.0]).collect();
        let m = flow_max(&[(&f, None)]);
        assert!((m - 9.89).abs() < 0.05, "{m}");
        // invalid pixels do not count
        let valid: Vec<u8> = (0..1000).map(|i| (i < 500) as u8).collect();
        let m = flow_max(&[(&f, Some(&valid))]);
        assert!((m - 4.94).abs() < 0.05, "{m}");
        assert_eq!(flow_max(&[]), 1.0);
    }

    #[test]
    fn wheel_legend_is_a_disc() {
        let img = flow_wheel_rgba(33);
        assert_eq!(img.len(), 33 * 33 * 4);
        assert_eq!(img[3], 0); // corner transparent
        let c = 4 * (16 * 33 + 16);
        assert_eq!(&img[c..c + 4], &[255, 255, 255, 255]); // centre: no motion
    }

    #[test]
    fn landcover_palette_and_names() {
        assert_eq!(landcover_color(LANDCOVER_SKY), SKY);
        assert_eq!(landcover_color(terragen::landcover::FOREST), terragen::landcover::palette(terragen::landcover::FOREST));
        assert_eq!(landcover_rgb(&[255, 10]), [SKY, terragen::landcover::palette(10)].concat());
        assert_eq!(class_name(&[], 10), "forest");
        assert_eq!(class_name(&["a".into(), "b".into()], 1), "b");
        assert_eq!(class_name(&[], 255), "sky");
        assert_eq!(class_name(&[], 200), "class 200");
        let h = class_histogram(&[1, 1, 255]);
        assert_eq!((h[1], h[255], h[0]), (2, 1, 0));
    }

    #[test]
    fn events_accumulate_in_the_window() {
        let (x, y) = ([0u16, 1, 1, 2, 9], [0u16, 0, 0, 0, 9]);
        let t = [100i64, 150, 160, 50, 120];
        let p = [1i8, 1, 0, 0, 1];
        // window (60, 160]: the event at 50 is out, the one at (9, 9) off the sensor
        let f = EventFrame::accumulate(3, 1, 160, 100, Events { x: &x, y: &y, t: &t, p: &p });
        assert_eq!(f.count, 3);
        assert_eq!((f.on[0], f.off[0], f.on[1], f.off[1], f.on[2] + f.off[2]), (1, 0, 1, 1, 0));
        assert_eq!((f.last_t[1], f.last_on[1]), (160, false));
        let img = events_rgb(&f, EventStyle::Black, None);
        assert_eq!(&img[6..9], &[0, 0, 0]); // no events: black
        assert!(img[0] > img[2]); // ON: red
        let mixed = &img[3..6]; // one ON + one OFF: between
        assert!(mixed[0] > 60 && mixed[2] > 60);
        // on the gray frame: the background shows through
        let gray = events_rgb(&f, EventStyle::Gray, Some(&[200; 9]));
        assert_eq!(&gray[6..9], &[120, 120, 120]);
        // time surface: the newest is brightest
        let ts = events_rgb(&f, EventStyle::TimeSurface, None);
        assert!(ts[5] > 200 && ts[0] < 220 && ts[0] > 0, "{ts:?}");
        for s in EventStyle::ALL {
            assert_eq!(EventStyle::parse(s.name()), Some(s));
        }
    }

    #[test]
    fn rings_stay_inside_the_image() {
        let mut img = vec![0u8; 10 * 8 * 3];
        draw_ring(&mut img, 10, 8, 0.0, 0.0, 3.0, [255, 255, 0]);
        draw_ring(&mut img, 10, 8, 50.0, -50.0, 3.0, [255, 255, 0]);
        draw_ring(&mut img, 10, 8, f32::NAN, 1.0, 3.0, [255, 255, 0]);
        let j = 3 * 3; // (3, 0) on the ring
        assert!(img[j] > 200);
        assert_eq!(img[3 * (7 * 10 + 9)], 0);
    }
}
