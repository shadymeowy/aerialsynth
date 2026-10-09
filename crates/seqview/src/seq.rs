//! Reading a sequence file (`docs/formats.md`) lazily: opening reads only the small per-sequence
//! data (attributes, calibration, timestamps, the body poses and the events' millisecond index);
//! frames, events and IMU samples are read when asked for.
//!
//! Groups are found by their contents, not by the scenario's paths: a camera has
//! `calib/resolution`, the IMU `accel` + `gyro`, the body poses `lla` + `q_ned_body`. Any of
//! them may be missing, as may any modality of a camera.

use crate::colorize::{EventFrame, Events};
use anyhow::{bail, Context, Result};
use h5::Attrs;
use std::path::{Path, PathBuf};

/// The image-like modalities of a camera (and its events).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Modality {
    Rgb,
    Depth,
    Flow,
    Landcover,
    Events,
}

impl Modality {
    pub const ALL: [Modality; 5] = [Modality::Rgb, Modality::Depth, Modality::Flow, Modality::Landcover, Modality::Events];

    pub fn name(self) -> &'static str {
        match self {
            Modality::Rgb => "rgb",
            Modality::Depth => "depth",
            Modality::Flow => "flow",
            Modality::Landcover => "landcover",
            Modality::Events => "events",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Modality::Rgb => "RGB",
            Modality::Depth => "Depth",
            Modality::Flow => "Flow",
            Modality::Landcover => "Land cover",
            Modality::Events => "Events",
        }
    }

    pub fn parse(s: &str) -> Result<Modality> {
        let s = s.trim().to_ascii_lowercase();
        Ok(match s.as_str() {
            "rgb" | "gray" | "image" => Modality::Rgb,
            "depth" => Modality::Depth,
            "flow" => Modality::Flow,
            "landcover" | "land_cover" | "lc" => Modality::Landcover,
            "events" | "ev" => Modality::Events,
            _ => bail!("unknown modality {s:?} (rgb, depth, flow, landcover, events)"),
        })
    }

    /// A comma-separated list (duplicates dropped, order kept).
    pub fn parse_list(s: &str) -> Result<Vec<Modality>> {
        let mut v = Vec::new();
        for m in s.split(',').filter(|x| !x.trim().is_empty()) {
            let m = Modality::parse(m)?;
            if !v.contains(&m) {
                v.push(m);
            }
        }
        if v.is_empty() {
            bail!("no modality given (rgb, depth, flow, landcover, events)");
        }
        Ok(v)
    }
}

/// The events of a camera.
#[derive(Debug)]
pub struct EventsInfo {
    pub n: usize,
    /// `ms_index[m]`: index of the first event at or after m ms (plus a closing entry)
    pub ms_index: Vec<u64>,
    /// time of the first and the last event (µs)
    pub t_range: Option<(i64, i64)>,
    /// the simulator settings (attr `events_yaml`)
    pub yaml: String,
}

/// One camera group.
#[derive(Debug)]
pub struct Camera {
    /// HDF5 path, e.g. `/cam0`
    pub path: String,
    pub w: usize,
    pub h: usize,
    /// frame times (µs; empty for an events-only camera)
    pub t: Vec<i64>,
    pub model: String,
    pub intrinsics: Vec<f64>,
    pub distortion: Vec<f64>,
    /// row-major 4x4, camera → body
    pub t_body_cam: Vec<f64>,
    pub camera_yaml: String,
    /// `Some(channels)`: 3 = RGB, 1 = gray
    pub rgb: Option<usize>,
    /// exposure time (s), gain, EV per frame
    pub exposure: Vec<[f64; 3]>,
    /// `Some(kind)`: "z" or "range"
    pub depth: Option<String>,
    pub flow: bool,
    pub flow_valid: bool,
    /// `Some(class names)`
    pub landcover: Option<Vec<String>>,
    pub events: Option<EventsInfo>,
    /// `stars/index` (frame k: rows index[k]..index[k+1])
    pub stars: Option<Vec<u64>>,
}

impl Camera {
    pub fn has(&self, m: Modality) -> bool {
        match m {
            Modality::Rgb => self.rgb.is_some(),
            Modality::Depth => self.depth.is_some(),
            Modality::Flow => self.flow,
            Modality::Landcover => self.landcover.is_some(),
            Modality::Events => self.events.is_some(),
        }
    }

    /// The modalities this camera has.
    pub fn modalities(&self) -> Vec<Modality> {
        Modality::ALL.into_iter().filter(|m| self.has(*m)).collect()
    }

    /// The frame shown at time `t` (µs): the last at or before it (the first before the first).
    pub fn frame_at(&self, t: i64) -> Option<usize> {
        if self.t.is_empty() {
            return None;
        }
        Some(self.t.partition_point(|&x| x <= t).saturating_sub(1))
    }

    /// The frame rate from the frame times (Hz).
    pub fn frame_rate(&self) -> Option<f64> {
        let n = self.t.len();
        (n >= 2 && self.t[n - 1] > self.t[0]).then(|| (n - 1) as f64 * 1e6 / (self.t[n - 1] - self.t[0]) as f64)
    }

    /// Short name: the last component of the path.
    pub fn name(&self) -> &str {
        self.path.rsplit('/').find(|s| !s.is_empty()).unwrap_or(&self.path)
    }
}

/// The body ground truth (`lla`, attitude), all samples.
#[derive(Debug, Default)]
pub struct PoseTrack {
    pub path: String,
    pub t: Vec<i64>,
    /// lat°, lon°, h (m above the ellipsoid)
    pub lla: Vec<[f64; 3]>,
    /// roll, pitch, yaw (deg) of the body in the local NED frame (aerospace ZYX)
    pub rpy: Vec<[f64; 3]>,
    /// north, east, down (m) in the NED frame at the first pose (empty if absent)
    pub ned0: Vec<[f64; 3]>,
    pub sun_elevation: Vec<f64>,
}

impl PoseTrack {
    /// The sample shown at time `t`: the last at or before it.
    pub fn index_at(&self, t: i64) -> Option<usize> {
        (!self.t.is_empty()).then(|| self.t.partition_point(|&x| x <= t).saturating_sub(1))
    }
}

/// The IMU: timestamps read on opening, samples on demand.
#[derive(Debug)]
pub struct Imu {
    pub path: String,
    pub t: Vec<i64>,
    accel: h5::Dataset,
    gyro: h5::Dataset,
}

/// IMU samples of a time window.
#[derive(Clone, Debug, Default)]
pub struct ImuWindow {
    pub t: Vec<i64>,
    pub accel: Vec<[f64; 3]>,
    pub gyro: Vec<[f64; 3]>,
}

impl Imu {
    /// The samples with t in [t0, t1] (at most `max` of them, evenly strided).
    pub fn window(&self, t0: i64, t1: i64, max: usize) -> Result<ImuWindow> {
        let a = self.t.partition_point(|&x| x < t0);
        let b = self.t.partition_point(|&x| x <= t1);
        if b <= a {
            return Ok(ImuWindow::default());
        }
        let n = b - a;
        let acc: Vec<f64> = self.accel.read_slice(&[a, 0], &[n, 3])?;
        let gyr: Vec<f64> = self.gyro.read_slice(&[a, 0], &[n, 3])?;
        let step = n.div_ceil(max.max(1)).max(1);
        let mut w = ImuWindow::default();
        for i in (0..n).step_by(step) {
            w.t.push(self.t[a + i]);
            w.accel.push([acc[3 * i], acc[3 * i + 1], acc[3 * i + 2]]);
            w.gyro.push([gyr[3 * i], gyr[3 * i + 1], gyr[3 * i + 2]]);
        }
        Ok(w)
    }
}

/// A star of a frame.
#[derive(Clone, Copy, Debug)]
pub struct Star {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub v: f32,
    pub visible: bool,
}

/// One frame of one modality, decoded.
#[derive(Debug)]
pub enum FrameData {
    /// `channels` 1 (gray) or 3
    Image {
        channels: usize,
        px: Vec<u8>,
    },
    Depth(Vec<f32>),
    Flow {
        flow: Vec<f32>,
        valid: Option<Vec<u8>>,
    },
    Landcover(Vec<u8>),
    Events(EventFrame),
    Stars(Vec<Star>),
    Imu(ImuWindow),
}

impl FrameData {
    pub fn bytes(&self) -> usize {
        match self {
            FrameData::Image { px, .. } => px.len(),
            FrameData::Depth(d) => d.len() * 4,
            FrameData::Flow { flow, valid } => flow.len() * 4 + valid.as_ref().map_or(0, |v| v.len()),
            FrameData::Landcover(l) => l.len(),
            FrameData::Events(e) => e.bytes(),
            FrameData::Stars(s) => s.len() * std::mem::size_of::<Star>(),
            FrameData::Imu(w) => w.t.len() * 56,
        }
    }
}

/// What to read: a frame of a modality, the events of a window, or the stars of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Frame {
        cam: usize,
        m: Modality,
        k: usize,
    },
    /// events with t in (t_end - window, t_end] (µs)
    Events {
        cam: usize,
        t_end: i64,
        window: i64,
    },
    Stars {
        cam: usize,
        k: usize,
    },
    /// IMU samples with t in [t0, t1] (µs)
    Imu {
        t0: i64,
        t1: i64,
    },
}

/// An open sequence file.
pub struct Sequence {
    pub path: PathBuf,
    file: h5::File,
    pub format: String,
    pub format_version: i32,
    /// trajectory time of the sequence start (s)
    pub t0: f64,
    pub scenario_yaml: String,
    pub conventions: String,
    pub pose: Option<PoseTrack>,
    pub imu: Option<Imu>,
    pub cameras: Vec<Camera>,
}

fn read_vec<T: h5::H5Type>(g: &h5::Group, name: &str) -> Result<Vec<T>> {
    Ok(g.dataset(name)?.read_all::<T>()?)
}

fn rows<const N: usize>(v: &[f64]) -> Vec<[f64; N]> {
    v.as_chunks::<N>().0.to_vec()
}

/// Groups of `g` (recursively, not below a recognised sensor group): (path, group).
fn walk(g: &h5::Group, prefix: &str, depth: usize, out: &mut Vec<(String, h5::Group)>) -> Result<()> {
    for name in g.member_names()? {
        let Ok(sub) = g.group(&name) else { continue }; // a dataset
        let path = format!("{prefix}/{name}");
        let sensor = sub.exists("calib/resolution") || (sub.exists("accel") && sub.exists("gyro")) || sub.exists("lla");
        out.push((path.clone(), sub.clone()));
        if !sensor && depth < 4 {
            walk(&sub, &path, depth + 1, out)?;
        }
    }
    Ok(())
}

impl Sequence {
    pub fn open(path: &Path) -> Result<Sequence> {
        if !path.exists() {
            bail!("{} does not exist", path.display());
        }
        let file = h5::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let format = file.attr_str("format").unwrap_or_default();
        if format == "terrain-tiles" || file.exists("levels") {
            bail!("{} is a tile store, not a sequence file (`terrain view -c …` shows the world)", path.display());
        }
        let mut groups = Vec::new();
        walk(&file, "", 0, &mut groups)?;
        let mut seq = Sequence {
            path: path.to_path_buf(),
            format_version: file.attr::<i32>("format_version").unwrap_or(0),
            t0: file.attr::<f64>("t0").unwrap_or(0.0),
            scenario_yaml: file.attr_str("scenario").unwrap_or_default(),
            conventions: file.attr_str("conventions").unwrap_or_default(),
            format,
            pose: None,
            imu: None,
            cameras: Vec::new(),
            file: file.clone(),
        };
        for (p, g) in &groups {
            let ctx = || format!("reading {p} of {}", path.display());
            if g.exists("calib/resolution") {
                seq.cameras.push(Self::camera(p, g).with_context(ctx)?);
            } else if g.exists("accel") && g.exists("gyro") && g.exists("t") && seq.imu.is_none() {
                seq.imu = Some(Imu { path: p.clone(), t: read_vec(g, "t").with_context(ctx)?, accel: g.dataset("accel")?, gyro: g.dataset("gyro")? });
            } else if g.exists("lla") && g.exists("t") && seq.pose.is_none() {
                seq.pose = Some(Self::pose(p, g).with_context(ctx)?);
            }
        }
        if seq.cameras.is_empty() && seq.imu.is_none() && seq.pose.is_none() {
            bail!("{}: no cameras, IMU or poses found (a sequence file of `terrain run`?)", path.display());
        }
        Ok(seq)
    }

    fn pose(path: &str, g: &h5::Group) -> Result<PoseTrack> {
        let t: Vec<i64> = read_vec(g, "t")?;
        let lla = rows::<3>(&read_vec::<f64>(g, "lla")?);
        let rpy = if g.exists("q_ned_body") {
            rows::<4>(&read_vec::<f64>(g, "q_ned_body")?)
                .iter()
                .map(|q| {
                    let (yaw, pitch, roll) = geodesy::quat_to_euler_zyx(glam::DQuat::from_xyzw(q[1], q[2], q[3], q[0]).normalize());
                    [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
                })
                .collect()
        } else {
            vec![]
        };
        let ned0 = if g.exists("position_ned0") { rows::<3>(&read_vec::<f64>(g, "position_ned0")?) } else { vec![] };
        let sun_elevation = if g.exists("sun_elevation_deg") { read_vec(g, "sun_elevation_deg")? } else { vec![] };
        Ok(PoseTrack { path: path.to_string(), t, lla, rpy, ned0, sun_elevation })
    }

    fn camera(path: &str, g: &h5::Group) -> Result<Camera> {
        let c = g.group("calib")?;
        let res: Vec<i64> = read_vec(&c, "resolution")?;
        if res.len() != 2 || res[0] <= 0 || res[1] <= 0 {
            bail!("calib/resolution: [W, H] expected, got {res:?}");
        }
        let (w, h) = (res[0] as usize, res[1] as usize);
        let opt_f64 = |name: &str| -> Result<Vec<f64>> {
            if c.exists(name) {
                read_vec(&c, name)
            } else {
                Ok(vec![])
            }
        };
        let t: Vec<i64> = if g.exists("t") { read_vec(g, "t")? } else { vec![] };
        let n = t.len();
        // a frame dataset counts when it holds every frame at the calibrated size
        let frames = |name: &str, tail: &[usize]| -> Result<bool> {
            if !g.exists(name) || n == 0 {
                return Ok(false);
            }
            let s = g.dataset(name)?.shape()?;
            Ok(s.len() == 3 + tail.len() && s[0] >= n && s[1] == h && s[2] == w && s[3..] == *tail)
        };
        let rgb = if frames("rgb", &[3])? {
            Some(3)
        } else if frames("rgb", &[])? {
            Some(1)
        } else {
            None
        };
        let depth = frames("depth", &[])?.then(|| g.dataset("depth").ok().and_then(|d| d.attr_str("kind").ok()).unwrap_or_else(|| "z".into()));
        let flow = frames("flow", &[2])?;
        let flow_valid = flow && frames("flow_valid", &[])?;
        let landcover = frames("landcover", &[])?.then(|| {
            g.dataset("landcover")
                .ok()
                .and_then(|d| d.attr_str("class_names").ok())
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
                .unwrap_or_default()
        });
        let exposure = if rgb.is_some() && g.exists("exposure") { rows::<3>(&read_vec::<f64>(g, "exposure")?) } else { vec![] };
        let events = if g.exists("events/ms_index") && g.exists("events/t") {
            let e = g.group("events")?;
            let td = e.dataset("t")?;
            let ne = td.shape()?.first().copied().unwrap_or(0);
            let t_range = if ne > 0 { Some((td.read_slice::<i64>(&[0], &[1])?[0], td.read_slice::<i64>(&[ne - 1], &[1])?[0])) } else { None };
            Some(EventsInfo { n: ne, ms_index: read_vec(&e, "ms_index")?, t_range, yaml: e.attr_str("events_yaml").unwrap_or_default() })
        } else {
            None
        };
        let stars = if g.exists("stars/index") && n > 0 {
            let idx: Vec<u64> = read_vec(g, "stars/index")?;
            (idx.len() == n + 1).then_some(idx)
        } else {
            None
        };
        Ok(Camera {
            path: path.to_string(),
            w,
            h,
            t,
            model: c.attr_str("model").unwrap_or_default(),
            intrinsics: opt_f64("intrinsics")?,
            distortion: opt_f64("distortion_coeffs")?,
            t_body_cam: opt_f64("T_body_cam")?,
            camera_yaml: c.attr_str("camera_yaml").unwrap_or_default(),
            rgb,
            exposure,
            depth,
            flow,
            flow_valid,
            landcover,
            events,
            stars,
        })
    }

    /// The camera at `path` (or named so: `cam0` finds `/cam0`).
    pub fn camera_index(&self, path: &str) -> Result<usize> {
        let p = path.trim_end_matches('/');
        self.cameras.iter().position(|c| c.path == p || c.name() == p.trim_start_matches('/')).ok_or_else(|| {
            let have: Vec<&str> = self.cameras.iter().map(|c| c.path.as_str()).collect();
            anyhow::anyhow!("no camera {path:?} in {} (cameras: {})", self.path.display(), if have.is_empty() { "none".into() } else { have.join(", ") })
        })
    }

    /// The time span of everything in the file (µs).
    pub fn time_range(&self) -> (i64, i64) {
        let mut r: Option<(i64, i64)> = None;
        let mut add = |a: i64, b: i64| r = Some(r.map_or((a, b), |(x, y)| (x.min(a), y.max(b))));
        if let Some(p) = &self.pose {
            if let (Some(a), Some(b)) = (p.t.first(), p.t.last()) {
                add(*a, *b);
            }
        }
        if let Some(i) = &self.imu {
            if let (Some(a), Some(b)) = (i.t.first(), i.t.last()) {
                add(*a, *b);
            }
        }
        for c in &self.cameras {
            if let (Some(a), Some(b)) = (c.t.first(), c.t.last()) {
                add(*a, *b);
            }
            if let Some((a, b)) = c.events.as_ref().and_then(|e| e.t_range) {
                add(a, b);
            }
        }
        r.unwrap_or((0, 0))
    }

    fn group(&self, cam: usize) -> Result<h5::Group> {
        Ok(self.file.group(&self.cameras[cam].path)?)
    }

    /// Read what `key` names.
    pub fn read(&self, key: Key) -> Result<FrameData> {
        match key {
            Key::Frame { cam, m, k } => self.read_frame(cam, m, k),
            Key::Events { cam, t_end, window } => Ok(FrameData::Events(self.read_events(cam, t_end, window)?)),
            Key::Stars { cam, k } => Ok(FrameData::Stars(self.read_stars(cam, k)?)),
            Key::Imu { t0, t1 } => match &self.imu {
                Some(imu) => Ok(FrameData::Imu(imu.window(t0, t1, 4000)?)),
                None => bail!("{} has no IMU", self.path.display()),
            },
        }
    }

    fn read_frame(&self, cam: usize, m: Modality, k: usize) -> Result<FrameData> {
        let c = &self.cameras[cam];
        if !c.has(m) || m == Modality::Events {
            bail!("{} has no {} frames", c.path, m.name());
        }
        if k >= c.t.len() {
            bail!("{}: frame {k} of {}", c.path, c.t.len());
        }
        let g = self.group(cam)?;
        let (w, h) = (c.w, c.h);
        Ok(match m {
            Modality::Rgb => {
                let ch = c.rgb.unwrap_or(3);
                let px =
                    if ch == 3 { g.dataset("rgb")?.read_slice(&[k, 0, 0, 0], &[1, h, w, 3])? } else { g.dataset("rgb")?.read_slice(&[k, 0, 0], &[1, h, w])? };
                FrameData::Image { channels: ch, px }
            }
            Modality::Depth => FrameData::Depth(g.dataset("depth")?.read_slice(&[k, 0, 0], &[1, h, w])?),
            Modality::Flow => FrameData::Flow {
                flow: g.dataset("flow")?.read_slice(&[k, 0, 0, 0], &[1, h, w, 2])?,
                valid: if c.flow_valid { Some(g.dataset("flow_valid")?.read_slice(&[k, 0, 0], &[1, h, w])?) } else { None },
            },
            Modality::Landcover => FrameData::Landcover(g.dataset("landcover")?.read_slice(&[k, 0, 0], &[1, h, w])?),
            Modality::Events => unreachable!(),
        })
    }

    /// The events of camera `cam` with t in (t_end - window, t_end] (µs), accumulated.
    pub fn read_events(&self, cam: usize, t_end: i64, window: i64) -> Result<EventFrame> {
        let c = &self.cameras[cam];
        let Some(ev) = &c.events else { bail!("{} has no events", c.path) };
        let window = window.max(1);
        let t0 = t_end - window;
        let last = ev.ms_index.len().saturating_sub(1); // the closing entry
        let ms = |t: i64| (t.max(0) / 1000) as usize;
        let a = ev.ms_index.get(ms(t0).min(last)).copied().unwrap_or(0) as usize;
        let b = if t_end < 0 { a } else { ev.ms_index.get((ms(t_end) + 1).min(last)).copied().unwrap_or(ev.n as u64) as usize };
        let (a, b) = (a.min(ev.n), b.min(ev.n).max(a.min(ev.n)));
        if b == a {
            return Ok(EventFrame::accumulate(c.w, c.h, t_end, window, Events::default()));
        }
        let g = self.group(cam)?.group("events")?;
        let n = b - a;
        let x: Vec<u16> = g.dataset("x")?.read_slice(&[a], &[n])?;
        let y: Vec<u16> = g.dataset("y")?.read_slice(&[a], &[n])?;
        let t: Vec<i64> = g.dataset("t")?.read_slice(&[a], &[n])?;
        let p: Vec<i8> = g.dataset("p")?.read_slice(&[a], &[n])?;
        Ok(EventFrame::accumulate(c.w, c.h, t_end, window, Events { x: &x, y: &y, t: &t, p: &p }))
    }

    /// The catalogue stars of frame `k`.
    pub fn read_stars(&self, cam: usize, k: usize) -> Result<Vec<Star>> {
        let c = &self.cameras[cam];
        let Some(idx) = &c.stars else { return Ok(vec![]) };
        let (a, b) = (idx[k] as usize, idx[k + 1] as usize);
        if b <= a {
            return Ok(vec![]);
        }
        let g = self.group(cam)?.group("stars")?;
        let n = b - a;
        let f = |name: &str| -> Result<Vec<f32>> { Ok(g.dataset(name)?.read_slice(&[a], &[n])?) };
        let (x, y, v) = (f("x")?, f("y")?, f("v")?);
        let id: Vec<u32> = g.dataset("id")?.read_slice(&[a], &[n])?;
        let vis: Vec<u8> = g.dataset("visible")?.read_slice(&[a], &[n])?;
        Ok((0..n).map(|i| Star { id: id[i], x: x[i], y: y[i], v: v[i], visible: vis[i] != 0 }).collect())
    }

    /// One line per part of the file, for the terminal.
    pub fn summary(&self) -> String {
        let (a, b) = self.time_range();
        let mut s = format!("{}: {:.2} s", self.path.display(), (b - a) as f64 / 1e6);
        if let Some(p) = &self.pose {
            s += &format!("; poses {} ({} samples)", p.path, p.t.len());
        }
        if let Some(i) = &self.imu {
            s += &format!("; IMU {} ({} samples)", i.path, i.t.len());
        }
        for c in &self.cameras {
            let mods: Vec<&str> = c.modalities().iter().map(|m| m.name()).collect();
            s += &format!("; {} {}×{} {} frames [{}]", c.path, c.w, c.h, c.t.len(), mods.join(","));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_lists() {
        assert_eq!(Modality::parse_list("rgb, depth,rgb,events").unwrap(), vec![Modality::Rgb, Modality::Depth, Modality::Events]);
        assert!(Modality::parse_list("rgb,normals").is_err());
        assert!(Modality::parse_list(",").is_err());
        for m in Modality::ALL {
            assert_eq!(Modality::parse(m.name()).unwrap(), m);
        }
    }

    #[test]
    fn frame_lookup() {
        let mut c = Camera {
            path: "/a/cam1".into(),
            w: 1,
            h: 1,
            t: vec![100, 200, 300],
            model: String::new(),
            intrinsics: vec![],
            distortion: vec![],
            t_body_cam: vec![],
            camera_yaml: String::new(),
            rgb: None,
            exposure: vec![],
            depth: None,
            flow: false,
            flow_valid: false,
            landcover: None,
            events: None,
            stars: None,
        };
        assert_eq!(c.frame_at(50), Some(0));
        assert_eq!(c.frame_at(200), Some(1));
        assert_eq!(c.frame_at(299), Some(1));
        assert_eq!(c.frame_at(10_000), Some(2));
        assert_eq!(c.frame_rate(), Some(1e4));
        assert_eq!(c.name(), "cam1");
        c.t.clear();
        assert_eq!(c.frame_at(0), None);
        assert!(c.modalities().is_empty());
    }
}
