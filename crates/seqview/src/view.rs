//! One modality of one camera at one time as a picture: the colourised image, its legend and
//! the data under it (for the pixel readout). Shared by the viewer, its snapshots and the
//! exporter.

use crate::colorize::{self, DepthStyle, EventStyle};
use crate::loader::{Loaded, Loader};
use crate::seq::{FrameData, Key, Modality, Sequence, Star};
use std::hash::{Hash, Hasher};

/// Display settings of the modalities.
#[derive(Clone, Debug, PartialEq)]
pub struct Style {
    /// depth range (m); None: each frame's own (2nd–98th percentile)
    pub depth_range: Option<(f32, f32)>,
    /// logarithmic depth scale; None: when the range spans more than a factor 4 (oblique
    /// views reaching to the horizon)
    pub depth_log: Option<bool>,
    /// flow magnitude (px) at full colour; None: each frame's 99th percentile
    pub flow_max: Option<f32>,
    /// events: the accumulation window (µs) ending at the time shown
    pub window_us: i64,
    pub events: EventStyle,
    /// catalogue stars on the image (cameras with `stars`) up to this magnitude; None: none
    pub stars: Option<f32>,
}

impl Default for Style {
    fn default() -> Self {
        Style { depth_range: None, depth_log: None, flow_max: None, window_us: 10_000, events: EventStyle::Black, stars: Some(6.5) }
    }
}

/// A legend under a picture.
#[derive(Clone, Debug, PartialEq)]
pub enum Legend {
    None,
    /// a colour bar from `lo` to `hi` (text)
    Colorbar {
        colors: Vec<[u8; 3]>,
        lo: String,
        hi: String,
        note: String,
    },
    /// the flow colour wheel, its rim at `max` px
    Wheel {
        max: f32,
    },
    /// the classes in view (name, colour, share of the pixels)
    Classes(Vec<(String, [u8; 3], f32)>),
    /// ON / OFF colours, the window and the event count
    Events {
        style: EventStyle,
        window_ms: f64,
        count: usize,
    },
}

/// What a panel shows: the reads it needs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Plan {
    pub cam: usize,
    pub m: Modality,
    /// the frame shown (frame modalities; for events: the frame under them, if any)
    pub frame: Option<usize>,
    pub main: Key,
    /// the gray frame under the events, the stars on the image
    pub extra: Option<Key>,
}

impl Plan {
    /// What camera `cam` shows of `m` at time `t` (µs), or why nothing.
    pub fn new(seq: &Sequence, cam: usize, m: Modality, t: i64, style: &Style) -> Result<Plan, String> {
        let c = &seq.cameras[cam];
        if !c.has(m) {
            return Err(format!("{} has no {}", c.path, m.name()));
        }
        let frame = c.frame_at(t);
        if m == Modality::Events {
            let extra = match (style.events, frame, c.rgb) {
                (EventStyle::Gray, Some(k), Some(_)) => Some(Key::Frame { cam, m: Modality::Rgb, k }),
                _ => None,
            };
            return Ok(Plan { cam, m, frame, main: Key::Events { cam, t_end: t, window: style.window_us.max(1) }, extra });
        }
        let k = frame.ok_or_else(|| format!("{} has no frames", c.path))?;
        let extra = (m == Modality::Rgb && style.stars.is_some() && c.stars.is_some()).then_some(Key::Stars { cam, k });
        Ok(Plan { cam, m, frame: Some(k), main: Key::Frame { cam, m, k }, extra })
    }

    /// The reads, the main one first.
    pub fn keys(&self) -> Vec<Key> {
        std::iter::once(self.main).chain(self.extra).collect()
    }

    /// A signature of what the picture depends on: the plan, the settings that matter for the
    /// modality and which reads are in.
    pub fn signature(&self, style: &Style, loader: &Loader) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        match self.m {
            Modality::Depth => {
                style.depth_range.map(|(a, b)| (a.to_bits(), b.to_bits())).hash(&mut h);
                style.depth_log.hash(&mut h);
            }
            Modality::Flow => style.flow_max.map(f32::to_bits).hash(&mut h),
            Modality::Events => style.events.hash(&mut h),
            Modality::Rgb => style.stars.map(f32::to_bits).hash(&mut h),
            Modality::Landcover => {}
        }
        for k in self.keys() {
            loader.get(k).is_some().hash(&mut h);
        }
        h.finish()
    }
}

/// A rendered panel.
pub struct Picture {
    pub plan: Plan,
    pub title: String,
    pub w: usize,
    pub h: usize,
    /// RGB8
    pub rgb: Vec<u8>,
    pub legend: Legend,
    /// the data under the picture (readout)
    pub data: Loaded,
    pub stars: Vec<Star>,
    /// depth / flow settings used (auto ranges resolved)
    pub depth: Option<DepthStyle>,
    pub flow_max: Option<f32>,
}

fn fmt_m(d: f32) -> String {
    if d >= 10_000.0 {
        format!("{:.1} km", d / 1000.0)
    } else if d >= 100.0 {
        format!("{d:.0} m")
    } else {
        format!("{d:.1} m")
    }
}

/// Render `plan` from what the loader has; None while its main read is not in. A failed read
/// gives `Err`.
pub fn render(seq: &Sequence, loader: &Loader, plan: &Plan, style: &Style) -> Option<Result<Picture, String>> {
    let data = loader.get(plan.main)?;
    let d = match data.as_ref() {
        Ok(d) => d,
        Err(e) => return Some(Err(e.clone())),
    };
    let extra = plan.extra.and_then(|k| loader.get(k));
    let c = &seq.cameras[plan.cam];
    let (w, h) = (c.w, c.h);
    let frame_note = plan.frame.map(|k| format!(" · frame {k}")).unwrap_or_default();
    let mut pic = Picture {
        plan: plan.clone(),
        title: format!("{}{frame_note}", plan.m.label()),
        w,
        h,
        rgb: vec![],
        legend: Legend::None,
        data: data.clone(),
        stars: vec![],
        depth: None,
        flow_max: None,
    };
    match d {
        FrameData::Image { channels, px } => {
            pic.rgb = colorize::to_rgb(px, *channels);
            if *channels == 1 {
                pic.title = format!("Gray{frame_note}");
            }
            if let Some(FrameData::Stars(s)) = extra.as_ref().and_then(|e| e.as_ref().as_ref().ok()) {
                let limit = style.stars.unwrap_or(f32::INFINITY);
                pic.stars = s.iter().filter(|s| s.v <= limit).copied().collect();
                let vis = s.iter().filter(|s| s.visible).count();
                pic.legend = Legend::Colorbar {
                    colors: vec![],
                    lo: String::new(),
                    hi: String::new(),
                    note: format!("{} catalogue stars ({} to mag {limit:.1} ringed), {vis} on sky pixels", s.len(), pic.stars.len()),
                };
            }
            if let Some(k) = plan.frame {
                if let Some(e) = c.exposure.get(k) {
                    let note = format!("exposure {:.2} ms · gain {:.2} · EV {:.1}", e[0] * 1e3, e[1], e[2]);
                    pic.legend = match pic.legend {
                        Legend::Colorbar { note: n, .. } => {
                            Legend::Colorbar { colors: vec![], lo: String::new(), hi: String::new(), note: format!("{note} · {n}") }
                        }
                        _ => Legend::Colorbar { colors: vec![], lo: String::new(), hi: String::new(), note },
                    };
                }
            }
        }
        FrameData::Depth(dep) => {
            let (near, far) = style.depth_range.or_else(|| colorize::depth_range(&[dep])).unwrap_or((1.0, 1000.0));
            let ds = DepthStyle { near, far, log: style.depth_log.unwrap_or(far > 4.0 * near) };
            pic.rgb = colorize::depth_rgb(dep, &ds);
            let kind = c.depth.as_deref().unwrap_or("z");
            pic.title = format!("Depth ({kind}){frame_note}");
            let colors = (0..64).map(|i| colorize::turbo(1.0 - i as f32 / 63.0)).collect();
            let sky = dep.iter().filter(|d| !(d.is_finite() && **d > 0.0)).count() as f64 / dep.len().max(1) as f64;
            let mut note = if style.depth_range.is_none() { "auto range".to_string() } else { "fixed range".to_string() };
            if ds.log {
                note += " · log";
            }
            if sky > 0.0 {
                note += &format!(" · black: sky {:.0}%", sky * 100.0);
            }
            pic.legend = Legend::Colorbar { colors, lo: fmt_m(near), hi: fmt_m(far), note };
            pic.depth = Some(ds);
        }
        FrameData::Flow { flow, valid } => {
            let max = style.flow_max.unwrap_or_else(|| colorize::flow_max(&[(flow, valid.as_deref())]));
            pic.rgb = colorize::flow_rgb(flow, valid.as_deref(), max);
            pic.title = format!("Optical flow to the next frame{frame_note}");
            pic.legend = Legend::Wheel { max };
            pic.flow_max = Some(max);
        }
        FrameData::Landcover(lc) => {
            pic.rgb = colorize::landcover_rgb(lc);
            let hist = colorize::class_histogram(lc);
            let names = c.landcover.as_deref().unwrap_or(&[]);
            let n = lc.len().max(1) as f32;
            let mut cls: Vec<(String, [u8; 3], f32)> = (0..=255u8)
                .filter(|&i| hist[i as usize] > 0)
                .map(|i| (colorize::class_name(names, i), colorize::landcover_color(i), hist[i as usize] as f32 / n))
                .collect();
            cls.sort_by(|a, b| b.2.total_cmp(&a.2));
            pic.legend = Legend::Classes(cls);
        }
        FrameData::Events(ev) => {
            let bg = extra.as_ref().and_then(|e| match e.as_ref() {
                Ok(FrameData::Image { channels, px }) => Some(colorize::to_rgb(px, *channels)),
                _ => None,
            });
            // on the gray frame only where there is one
            let es = if style.events == EventStyle::Gray && c.rgb.is_none() { EventStyle::Black } else { style.events };
            pic.rgb = colorize::events_rgb(ev, es, bg.as_deref());
            let ms = ev.window as f64 / 1000.0;
            pic.title = format!("Events · {} ms{}", fmt_ms(ms), if c.t.is_empty() { String::new() } else { frame_note });
            pic.legend = Legend::Events { style: es, window_ms: ms, count: ev.count };
        }
        FrameData::Stars(_) | FrameData::Imu(_) => return Some(Err("not a picture".into())),
    }
    Some(Ok(pic))
}

pub fn fmt_ms(ms: f64) -> String {
    if ms >= 10.0 || ms.fract() == 0.0 {
        format!("{ms:.0}")
    } else {
        format!("{ms:.1}")
    }
}

/// The value under pixel (x, y) of a picture, as text.
pub fn readout(seq: &Sequence, p: &Picture, x: usize, y: usize) -> Option<String> {
    if x >= p.w || y >= p.h {
        return None;
    }
    let i = y * p.w + x;
    let d = p.data.as_ref().as_ref().ok()?;
    Some(match d {
        FrameData::Image { channels: 3, px } => format!("RGB {} {} {}", px[3 * i], px[3 * i + 1], px[3 * i + 2]),
        FrameData::Image { px, .. } => format!("gray {}", px[i]),
        FrameData::Depth(dep) => {
            let v = dep[i];
            if v.is_finite() {
                format!("depth {v:.2} m")
            } else {
                "depth inf (sky)".into()
            }
        }
        FrameData::Flow { flow, valid } => {
            let (dx, dy) = (flow[2 * i], flow[2 * i + 1]);
            let v = valid.as_ref().map(|v| if v[i] != 0 { "" } else { " (target hidden)" }).unwrap_or("");
            format!("flow {dx:+.3} {dy:+.3} px |{:.3}|{v}", dx.hypot(dy))
        }
        FrameData::Landcover(lc) => format!("class {} ({})", colorize::class_name(seq.cameras[p.plan.cam].landcover.as_deref().unwrap_or(&[]), lc[i]), lc[i]),
        FrameData::Events(ev) => {
            let last = if ev.last_t[i] == i64::MIN {
                String::new()
            } else {
                format!(", latest {} {:.2} ms ago", if ev.last_on[i] { "ON" } else { "OFF" }, (ev.t_end - ev.last_t[i]) as f64 / 1000.0)
            };
            format!("events ON {} OFF {}{last}", ev.on[i], ev.off[i])
        }
        FrameData::Stars(_) | FrameData::Imu(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metres() {
        assert_eq!(fmt_m(12.34), "12.3 m");
        assert_eq!(fmt_m(1234.0), "1234 m");
        assert_eq!(fmt_m(123_456.0), "123.5 km");
        assert_eq!(fmt_ms(10.0), "10");
        assert_eq!(fmt_ms(2.5), "2.5");
    }
}
