//! Sequence file writer. One HDF5 file per sequence; group paths come from the scenario,
//! dataset names are fixed:
//!
//! ```text
//! /                       attrs: format, scenario (YAML), conventions, t0 (trajectory time, s)
//! <output.pose.path>/     body ground truth at output.pose.rate_hz
//!     t                   i64 [M]     µs since the sequence start
//!     position_ecef       f64 [M,3]   m
//!     q_ecef_body         f64 [M,4]   [w,x,y,z], body (FRD) → ECEF
//!     lla                 f64 [M,3]   lat°, lon°, h (m above the ellipsoid)
//!     q_ned_body          f64 [M,4]   body → local NED
//!     position_ned0       f64 [M,3]   in the NED frame at the first pose (attr ned0_origin_lla)
//!     q_ned0_body         f64 [M,4]
//!     sun_azimuth_deg, sun_elevation_deg, lights   f64 [M]   scene lighting (lights: 0..1 on)
//! <camera.path>/
//!     calib/              intrinsics [4] (4-parameter models), distortion_coeffs, resolution
//!                         i64 [2] = (W, H), T_body_cam f64 [4,4] row-major camera → body;
//!                         attrs model, camera_yaml (full camera description, camera YAML schema)
//!     t                   i64 [N]     frame timestamps (µs since the sequence start)
//!     pose/               camera pose at the frame times: position_ecef [N,3], q_ecef_cam [N,4]
//!     rgb                 u8  [N,H,W,3] (or [N,H,W] gray)     ← rgb
//!     exposure            f64 [N,3]   exposure time (s), gain, EV  ← rgb
//!     depth               f32 [N,H,W] m, +inf = sky           ← depth
//!     flow                f32 [N,H,W,2] to the next frame (px) ← flow
//!     flow_valid          u8  [N,H,W]
//!     landcover           u8  [N,H,W] (255 = sky)              ← landcover
//!     events/             x, y u16, t i64 µs, p i8 (1 = ON), ms_index u64  ← events (events.rs)
//!     stars/              catalogue stars per frame: index u64 [N+1] (frame k: [index[k],
//!                         index[k+1])), id u32 (HIP number; Tycho-2 1<<31|TYC1<<17|TYC2<<3|TYC3),
//!                         x, y f32 (px at the frame time, pixel centres at integers), xm, ym
//!                         f32 (averaged over the exposure: the trail centroid), v f32 (catalogue V),
//!                         irradiance f32 (V band, relative to the Sun outside the atmosphere,
//!                         after extinction), visible u8 (pixel shows sky)  ← stars
//! <imu.path>/             t, accel, gyro, gt_*, calib/T_body_imu  (imu.rs)
//! ```
//! Frames are stamped at mid-exposure. Camera frame: OpenCV (x right, y down, z forward).
//! Every dataset carries `units`, `description` and (multi-column) `columns` attributes
//! ([`describe`]).

use crate::camera::CameraConfig;
use crate::scenario::{h5path, Compression, DepthKind, Scenario};
use crate::trajectory::{CamPose, Pose};
use anyhow::Result;
use geodesy::Ellipsoid;
use glam::{DMat3, DQuat, DVec3};
use h5::Attrs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "terrain-sequence";
pub const FORMAT_VERSION: i32 = 3;

pub fn q4(q: DQuat) -> [f64; 4] {
    [q.w, q.x, q.y, q.z]
}

/// Sequence clock: µs since `t0` (trajectory time of the sequence start).
pub fn to_us(t: f64, t0: f64) -> i64 {
    ((t - t0) * 1e6).round() as i64
}

/// Rigid transform (rotation `r`, translation `t`) as a row-major 4x4.
pub fn transform_4x4(r: DMat3, t: DVec3) -> [f64; 16] {
    let c = r.to_cols_array_2d();
    [c[0][0], c[1][0], c[2][0], t.x, c[0][1], c[1][1], c[2][1], t.y, c[0][2], c[1][2], c[2][2], t.z, 0.0, 0.0, 0.0, 1.0]
}

/// Round f32 values to `keep` mantissa bits (round-to-nearest; inf/nan untouched). Relative
/// error ≤ 2^-(keep+1); the zeroed low bits make shuffle+deflate far more effective.
pub fn round_mantissa(v: &mut [f32], keep: u8) {
    if keep >= 23 {
        return;
    }
    let drop = 23 - keep as u32;
    let half = 1u32 << (drop - 1);
    let mask = !((1u32 << drop) - 1);
    for x in v.iter_mut() {
        if x.is_finite() {
            *x = f32::from_bits((x.to_bits().wrapping_add(half)) & mask);
        }
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    Ok(())
}

/// Create (truncate) the sequence file and write the root attributes.
pub fn create_file(scn: &Scenario, t0: f64) -> Result<h5::File> {
    let path = &scn.output.file;
    ensure_parent(path)?;
    let f = h5::File::create(path)?;
    f.set_attr_str("format", FORMAT)?;
    f.set_attr("format_version", FORMAT_VERSION)?;
    f.set_attr_str("scenario", &scn.to_yaml())?;
    f.set_attr("t0", t0)?;
    f.set_attr("float_keep_bits", scn.output.compression.float_keep_bits.map(|b| b as i32).unwrap_or(23))?;
    f.set_attr_str(
        "conventions",
        "t: i64 µs since the sequence start (root attr t0 = trajectory time in s); frames stamped at mid-exposure; \
         camera frame OpenCV (x right, y down, z forward), body frame FRD; q_a_b = [w,x,y,z] rotating b-frame vectors into a; \
         T_body_x = row-major 4x4 mapping x-frame points into the body frame; depth in m (+inf = sky); \
         flow[k] = forward flow from frame k to k+1 of the same camera in px (dx, dy); events p: 1 = ON, 0 = OFF",
    )?;
    Ok(f)
}

/// Units and descriptions of the datasets, by group kind: (dataset path in the group, units,
/// description, column names).
const POSE_DOC: &[(&str, &str, &str, &str)] = &[
    ("t", "us", "time since the sequence start", ""),
    ("position_ecef", "m", "body position, ECEF", "x,y,z"),
    ("q_ecef_body", "", "attitude: body (FRD) to ECEF, Hamilton", "w,x,y,z"),
    ("lla", "deg,deg,m", "geodetic position, height above the ellipsoid", "lat,lon,h"),
    ("q_ned_body", "", "attitude: body (FRD) to the local NED frame at the body", "w,x,y,z"),
    ("position_ned0", "m", "position in the NED frame at the first pose (attr ned0_origin_lla)", "n,e,d"),
    ("q_ned0_body", "", "attitude: body (FRD) to the NED frame at the first pose", "w,x,y,z"),
    ("sun_azimuth_deg", "deg", "sun azimuth, clockwise from north", ""),
    ("sun_elevation_deg", "deg", "sun elevation above the horizon", ""),
    ("lights", "", "artificial lights on (0 = off, 1 = fully on)", ""),
];
const CAMERA_DOC: &[(&str, &str, &str, &str)] = &[
    ("t", "us", "frame times (mid-exposure) since the sequence start", ""),
    ("calib/intrinsics", "px", "focal lengths and principal point (4-parameter models)", "fx,fy,cx,cy"),
    ("calib/distortion_coeffs", "", "distortion coefficients of the model (empty: none)", ""),
    ("calib/resolution", "px", "image size", "width,height"),
    ("calib/T_body_cam", "m", "row-major 4x4: camera points into the body frame", ""),
    ("pose/position_ecef", "m", "camera position at the frame times, ECEF", "x,y,z"),
    ("pose/q_ecef_cam", "", "camera (OpenCV) to ECEF at the frame times, Hamilton", "w,x,y,z"),
    ("rgb", "DN", "developed 8-bit image, sRGB", ""),
    ("exposure", "s,,EV", "exposure of each frame", "exposure_time_s,gain,ev"),
    ("depth", "m", "z along the optical axis or range (attr kind); +inf = sky", ""),
    ("flow", "px", "forward optical flow to the next frame of this camera", "dx,dy"),
    ("flow_valid", "", "1 = the flow target is visible", ""),
    ("landcover", "", "class id (attr class_names); 255 = sky", ""),
    ("events/x", "px", "event column", ""),
    ("events/y", "px", "event row", ""),
    ("events/t", "us", "event time since the sequence start", ""),
    ("events/p", "", "polarity: 1 = ON, 0 = OFF", ""),
    ("events/ms_index", "", "index of the first event at or after each millisecond (+1 closing entry)", ""),
    ("stars/index", "", "frame k holds stars [index[k], index[k+1])", ""),
    ("stars/id", "", "HIP number; Tycho-2: 1<<31 | TYC1<<17 | TYC2<<3 | TYC3; planets 1<<30 | NAIF id", ""),
    ("stars/x", "px", "position at the frame time (pixel centres at integers)", ""),
    ("stars/y", "px", "position at the frame time (pixel centres at integers)", ""),
    ("stars/xm", "px", "position averaged over the exposure (trail centroid)", ""),
    ("stars/ym", "px", "position averaged over the exposure (trail centroid)", ""),
    ("stars/v", "mag", "catalogue V magnitude", ""),
    ("stars/irradiance", "", "V-band irradiance relative to the Sun outside the atmosphere, after extinction", ""),
    ("stars/visible", "", "1 = the pixel shows sky", ""),
];
const IMU_DOC: &[(&str, &str, &str, &str)] = &[
    ("t", "us", "sample times since the sequence start", ""),
    ("accel", "m/s^2", "measured specific force, IMU frame", "x,y,z"),
    ("gyro", "rad/s", "measured angular rate, IMU frame", "x,y,z"),
    ("gt_accel", "m/s^2", "true specific force, IMU frame", "x,y,z"),
    ("gt_gyro", "rad/s", "true angular rate, IMU frame", "x,y,z"),
    ("gt_bias_accel", "m/s^2", "accelerometer bias", "x,y,z"),
    ("gt_bias_gyro", "rad/s", "gyroscope bias", "x,y,z"),
    ("calib/T_body_imu", "m", "row-major 4x4: IMU points into the body frame", ""),
];

/// Attach `units`, `description` and `columns` attributes to every dataset of the sequence
/// file that exists (so the file describes itself).
pub fn describe(file: &h5::File, scn: &Scenario) -> Result<()> {
    let annotate = |group: &str, doc: &[(&str, &str, &str, &str)]| -> Result<()> {
        let root = file.root()?;
        for (name, units, description, columns) in doc {
            let p = format!("{}/{}", h5path(group).trim_end_matches('/'), name);
            if !root.exists(&p) {
                continue;
            }
            let ds = root.dataset(&p)?;
            ds.set_attr_str("units", units)?;
            ds.set_attr_str("description", description)?;
            if !columns.is_empty() {
                ds.set_attr_str("columns", columns)?;
            }
        }
        Ok(())
    };
    annotate(&scn.output.pose.path, POSE_DOC)?;
    for c in &scn.cameras {
        annotate(&c.path, CAMERA_DOC)?;
        let lc = format!("{}/landcover", h5path(&c.path).trim_end_matches('/'));
        if file.root()?.exists(&lc) {
            file.root()?.dataset(&lc)?.set_attr_str("class_names", &terragen::landcover::NAMES.join(","))?;
        }
    }
    if let Some(imu) = &scn.imu {
        annotate(&imu.path, IMU_DOC)?;
    }
    Ok(())
}

/// Open an existing sequence file for adding to it (events).
pub fn open_file(path: &Path) -> Result<h5::File> {
    let f = h5::File::open_rw(path)?;
    if f.attr_str("format").unwrap_or_default() != FORMAT {
        anyhow::bail!("{} is not a terrain sequence file (make it with `terrain run`)", path.display());
    }
    Ok(f)
}

/// Replace `path` by an empty group.
pub fn fresh_group(f: &h5::File, path: &str) -> Result<h5::Group> {
    let p = h5path(path);
    if f.exists(p) {
        f.delete(p)?;
    }
    Ok(f.ensure_group(p)?)
}

/// `calib/` of a camera group.
pub fn write_camera_calib(g: &h5::Group, cam: &CameraConfig, t_body_cam: [f64; 16]) -> Result<()> {
    if g.exists("calib") {
        g.delete("calib")?;
    }
    let c = g.ensure_group("calib")?;
    if cam.intrinsics.len() == 4 {
        c.new_dataset::<f64>().shape(&[4]).create("intrinsics")?.write_all(&cam.intrinsics)?;
    }
    // always present (empty for a model without distortion)
    c.new_dataset::<f64>().shape(&[cam.distortion.len()]).create("distortion_coeffs")?.write_all(&cam.distortion)?;
    c.new_dataset::<i64>().shape(&[2]).create("resolution")?.write_all(&[cam.width as i64, cam.height as i64])?;
    c.new_dataset::<f64>().shape(&[4, 4]).create("T_body_cam")?.write_all(&t_body_cam)?;
    c.set_attr_str("model", &cam.model)?;
    c.set_attr_str("camera_yaml", &serde_yaml::to_string(cam)?)?;
    Ok(())
}

fn put_f64(g: &h5::Group, name: &str, cols: usize, data: &[f64]) -> Result<()> {
    let n = data.len().checked_div(cols).unwrap_or(data.len());
    let shape: Vec<usize> = if cols == 0 { vec![n] } else { vec![n, cols] };
    let ds = g.new_dataset::<f64>().shape(&shape).create(name)?;
    if n > 0 {
        ds.write_all(data)?;
    }
    Ok(())
}

fn put_i64(g: &h5::Group, name: &str, data: &[i64]) -> Result<()> {
    let ds = g.new_dataset::<i64>().shape(&[data.len()]).create(name)?;
    if !data.is_empty() {
        ds.write_all(data)?;
    }
    Ok(())
}

/// One body ground-truth sample.
pub struct BodySample {
    pub t: f64,
    pub pose: Pose,
    pub sun: crate::lighting::SunState,
}

/// The body ground-truth group.
pub fn write_body_pose(f: &h5::File, path: &str, t0: f64, s: &[BodySample], ell: &Ellipsoid) -> Result<()> {
    let g = fresh_group(f, path)?;
    let ecef: Vec<DVec3> = s.iter().map(|b| b.pose.ecef(ell)).collect();
    let ned0 = s.first().map(|b| geodesy::LocalFrame::new(b.pose.geo, geodesy::LocalConvention::Ned, *ell));
    put_i64(&g, "t", &s.iter().map(|b| to_us(b.t, t0)).collect::<Vec<_>>())?;
    put_f64(&g, "position_ecef", 3, &ecef.iter().flat_map(|p| p.to_array()).collect::<Vec<_>>())?;
    put_f64(&g, "q_ecef_body", 4, &s.iter().flat_map(|b| q4(DQuat::from_mat3(&b.pose.r_ecef_body()).normalize())).collect::<Vec<_>>())?;
    put_f64(&g, "lla", 3, &s.iter().flat_map(|b| [b.pose.geo.lat.to_degrees(), b.pose.geo.lon.to_degrees(), b.pose.geo.h]).collect::<Vec<_>>())?;
    put_f64(&g, "q_ned_body", 4, &s.iter().flat_map(|b| q4(b.pose.q_ned_body)).collect::<Vec<_>>())?;
    if let Some(ned0) = &ned0 {
        let r0 = geodesy::rot_ecef2ned(ned0.origin.lat, ned0.origin.lon);
        put_f64(&g, "position_ned0", 3, &ecef.iter().flat_map(|p| geodesy::ecef2ned(*p, ned0.origin, ell).to_array()).collect::<Vec<_>>())?;
        put_f64(&g, "q_ned0_body", 4, &s.iter().flat_map(|b| q4(DQuat::from_mat3(&(r0 * b.pose.r_ecef_body())).normalize())).collect::<Vec<_>>())?;
        g.set_attr_array("ned0_origin_lla", &[ned0.origin.lat.to_degrees(), ned0.origin.lon.to_degrees(), ned0.origin.h])?;
    }
    put_f64(&g, "sun_azimuth_deg", 0, &s.iter().map(|b| b.sun.azimuth.to_degrees()).collect::<Vec<_>>())?;
    put_f64(&g, "sun_elevation_deg", 0, &s.iter().map(|b| b.sun.elevation.to_degrees()).collect::<Vec<_>>())?;
    put_f64(&g, "lights", 0, &s.iter().map(|b| b.sun.lights).collect::<Vec<_>>())?;
    Ok(())
}

/// Everything produced for one frame of one camera (absent modalities are `None`).
pub struct Frame<'a> {
    pub index: usize,
    pub t: f64,
    pub cam: CamPose,
    /// developed RGB (always RGB; converted to gray by the writer when configured)
    pub rgb: Option<&'a [u8]>,
    /// exposure time (s), gain, EV
    pub exposure: Option<[f64; 3]>,
    pub depth: Option<&'a [f32]>,
    pub flow: Option<(&'a [f32], &'a [u8])>,
    pub landcover: Option<&'a [u8]>,
    pub stars: Option<&'a [crate::stars::StarObs]>,
}

fn luma(rgb: &[u8]) -> Vec<u8> {
    rgb.as_chunks::<3>().0.iter().map(|c| (0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64).round() as u8).collect()
}

/// Writer of one camera group (datasets sized for `n` frames up front, chunk = one frame).
pub struct CameraWriter {
    group: h5::Group,
    w: usize,
    h: usize,
    gray: bool,
    rgb: Option<h5::Dataset>,
    depth: Option<h5::Dataset>,
    flow: Option<(h5::Dataset, h5::Dataset)>,
    landcover: Option<h5::Dataset>,
    stars: Option<StarsWriter>,
    t0: f64,
    t: Vec<i64>,
    pos: Vec<f64>,
    q: Vec<f64>,
    exposure: Vec<f64>,
}

impl CameraWriter {
    /// Creates `<spec.path>` with its calibration and the frame datasets of its modalities.
    pub fn new(f: &h5::File, spec: &crate::scenario::CameraSpec, n: usize, comp: &Compression, t0: f64) -> Result<Self> {
        let cam = &spec.intrinsics;
        let (w, h) = (cam.width as usize, cam.height as usize);
        let lvl = comp.level.min(9);
        let g = fresh_group(f, &spec.path)?;
        write_camera_calib(&g, cam, transform_4x4(spec.extrinsics.r_body_cam(), spec.extrinsics.t_body_cam()))?;
        let gray = spec.rgb.as_ref().is_some_and(|r| r.gray);
        let rgb = match &spec.rgb {
            Some(_) if gray => Some(g.new_dataset::<u8>().shape(&[n, h, w]).chunk(&[1, h, w]).deflate(lvl).create("rgb")?),
            Some(_) => Some(g.new_dataset::<u8>().shape(&[n, h, w, 3]).chunk(&[1, h, w, 3]).shuffle(true).deflate(lvl).create("rgb")?),
            None => None,
        };
        let depth = match &spec.depth {
            Some(d) => {
                let ds = g.new_dataset::<f32>().shape(&[n, h, w]).chunk(&[1, h, w]).shuffle(true).deflate(lvl).create("depth")?;
                ds.set_attr_str("kind", if d.kind == DepthKind::Range { "range" } else { "z" })?;
                Some(ds)
            }
            None => None,
        };
        let flow = match &spec.flow {
            Some(_) => Some((
                g.new_dataset::<f32>().shape(&[n, h, w, 2]).chunk(&[1, h, w, 2]).shuffle(true).deflate(lvl).create("flow")?,
                g.new_dataset::<u8>().shape(&[n, h, w]).chunk(&[1, h, w]).deflate(lvl).create("flow_valid")?,
            )),
            None => None,
        };
        let landcover = match &spec.landcover {
            Some(_) => Some(g.new_dataset::<u8>().shape(&[n, h, w]).chunk(&[1, h, w]).deflate(lvl).create("landcover")?),
            None => None,
        };
        let stars = match &spec.stars {
            Some(_) => Some(StarsWriter::new(&g, lvl)?),
            None => None,
        };
        Ok(CameraWriter { group: g, w, h, gray, rgb, depth, flow, landcover, stars, t0, t: vec![], pos: vec![], q: vec![], exposure: vec![] })
    }

    pub fn write(&mut self, fr: &Frame) -> Result<()> {
        let (k, w, h) = (fr.index, self.w, self.h);
        if let (Some(ds), Some(rgb)) = (&self.rgb, fr.rgb) {
            if self.gray {
                ds.write_slice(&luma(rgb), &[k, 0, 0], &[1, h, w])?;
            } else {
                ds.write_slice(rgb, &[k, 0, 0, 0], &[1, h, w, 3])?;
            }
        }
        if let (Some(ds), Some(d)) = (&self.depth, fr.depth) {
            ds.write_slice(d, &[k, 0, 0], &[1, h, w])?;
        }
        if let (Some((fl, fv)), Some((flow, valid))) = (&self.flow, fr.flow) {
            fl.write_slice(flow, &[k, 0, 0, 0], &[1, h, w, 2])?;
            fv.write_slice(valid, &[k, 0, 0], &[1, h, w])?;
        }
        if let (Some(ds), Some(l)) = (&self.landcover, fr.landcover) {
            ds.write_slice(l, &[k, 0, 0], &[1, h, w])?;
        }
        if let Some(sw) = self.stars.as_mut() {
            sw.write(fr.stars.unwrap_or(&[]))?;
        }
        self.t.push(to_us(fr.t, self.t0));
        self.pos.extend(fr.cam.pos.to_array());
        self.q.extend(q4(fr.cam.q_ecef_cam()));
        if let Some(e) = fr.exposure {
            self.exposure.extend(e);
        }
        Ok(())
    }

    /// Writes the per-frame vectors; returns the number of frames.
    pub fn finish(self) -> Result<usize> {
        let g = &self.group;
        put_i64(g, "t", &self.t)?;
        let p = g.ensure_group("pose")?;
        put_f64(&p, "position_ecef", 3, &self.pos)?;
        put_f64(&p, "q_ecef_cam", 4, &self.q)?;
        if self.rgb.is_some() {
            put_f64(g, "exposure", 3, &self.exposure)?;
        }
        if let Some(sw) = self.stars {
            sw.finish()?;
        }
        Ok(self.t.len())
    }
}

/// Streaming writer of `<camera>/stars` (frames in order).
struct StarsWriter {
    group: h5::Group,
    ds: [h5::Dataset; 8],
    n: usize,
    index: Vec<u64>,
}

impl StarsWriter {
    fn new(g: &h5::Group, lvl: u8) -> Result<Self> {
        let s = g.ensure_group("stars")?;
        let chunk = 1 << 14;
        let mk = |name: &str, kind: u8| -> Result<h5::Dataset> {
            let d = match kind {
                0 => s.new_dataset::<u32>().shape(&[0]).max_shape(&[None]).chunk(&[chunk]).shuffle(true).deflate(lvl).create(name)?,
                1 => s.new_dataset::<f32>().shape(&[0]).max_shape(&[None]).chunk(&[chunk]).shuffle(true).deflate(lvl).create(name)?,
                _ => s.new_dataset::<u8>().shape(&[0]).max_shape(&[None]).chunk(&[chunk]).deflate(lvl).create(name)?,
            };
            Ok(d)
        };
        let ds = [mk("id", 0)?, mk("x", 1)?, mk("y", 1)?, mk("v", 1)?, mk("irradiance", 1)?, mk("visible", 2)?, mk("xm", 1)?, mk("ym", 1)?];
        Ok(StarsWriter { group: s, ds, n: 0, index: vec![] })
    }

    fn write(&mut self, st: &[crate::stars::StarObs]) -> Result<()> {
        self.index.push(self.n as u64);
        if st.is_empty() {
            return Ok(());
        }
        let n1 = self.n + st.len();
        for d in &self.ds {
            d.resize(&[n1])?;
        }
        let (o, c) = (&[self.n][..], &[st.len()][..]);
        self.ds[0].write_slice(&st.iter().map(|s| s.id).collect::<Vec<_>>(), o, c)?;
        self.ds[1].write_slice(&st.iter().map(|s| s.x).collect::<Vec<_>>(), o, c)?;
        self.ds[2].write_slice(&st.iter().map(|s| s.y).collect::<Vec<_>>(), o, c)?;
        self.ds[3].write_slice(&st.iter().map(|s| s.v).collect::<Vec<_>>(), o, c)?;
        self.ds[4].write_slice(&st.iter().map(|s| s.irradiance).collect::<Vec<_>>(), o, c)?;
        self.ds[5].write_slice(&st.iter().map(|s| s.visible as u8).collect::<Vec<_>>(), o, c)?;
        self.ds[6].write_slice(&st.iter().map(|s| s.xm).collect::<Vec<_>>(), o, c)?;
        self.ds[7].write_slice(&st.iter().map(|s| s.ym).collect::<Vec<_>>(), o, c)?;
        self.n = n1;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.index.push(self.n as u64);
        self.group.new_dataset::<u64>().shape(&[self.index.len()]).create("index")?.write_all(&self.index)?;
        Ok(())
    }
}

fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    let shape_s = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_s}, }}");
    let total = 10 + header.len() + 1;
    let pad = (64 - total % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Optional per-camera PNG / NPY export: `<dir>/{rgb,depth,flow,flow_valid,landcover}/NNNNNN.*`,
/// `camera.yaml` and `frames.csv` (t µs, camera pose in ECEF, exposure).
pub struct PngWriter {
    dir: PathBuf,
    w: u32,
    h: u32,
    gray: bool,
    t0: f64,
    csv: std::io::BufWriter<std::fs::File>,
}

impl PngWriter {
    pub fn new(dir: &Path, spec: &crate::scenario::CameraSpec, t0: f64) -> Result<Self> {
        for (on, sub) in [
            (spec.rgb.is_some(), "rgb"),
            (spec.depth.is_some(), "depth"),
            (spec.flow.is_some(), "flow"),
            (spec.flow.is_some(), "flow_valid"),
            (spec.landcover.is_some(), "landcover"),
        ] {
            if on {
                std::fs::create_dir_all(dir.join(sub))?;
            }
        }
        std::fs::write(dir.join("camera.yaml"), serde_yaml::to_string(spec)?)?;
        let mut csv = std::io::BufWriter::new(std::fs::File::create(dir.join("frames.csv"))?);
        writeln!(csv, "frame,t_us,x_ecef,y_ecef,z_ecef,qw_ecef_cam,qx_ecef_cam,qy_ecef_cam,qz_ecef_cam,exposure_time,gain,ev")?;
        let gray = spec.rgb.as_ref().is_some_and(|r| r.gray);
        Ok(PngWriter { dir: dir.to_path_buf(), w: spec.intrinsics.width, h: spec.intrinsics.height, gray, t0, csv })
    }

    pub fn write(&mut self, fr: &Frame) -> Result<()> {
        let name = format!("{:06}", fr.index);
        let (w, h) = (self.w as usize, self.h as usize);
        if let Some(rgb) = fr.rgb {
            let p = self.dir.join("rgb").join(format!("{name}.png"));
            if self.gray {
                image::save_buffer(p, &luma(rgb), self.w, self.h, image::ExtendedColorType::L8)?;
            } else {
                image::save_buffer(p, rgb, self.w, self.h, image::ExtendedColorType::Rgb8)?;
            }
        }
        if let Some(d) = fr.depth {
            write_npy_f32(&self.dir.join("depth").join(format!("{name}.npy")), &[h, w], d)?;
        }
        if let Some((flow, valid)) = fr.flow {
            write_npy_f32(&self.dir.join("flow").join(format!("{name}.npy")), &[h, w, 2], flow)?;
            let v: Vec<u8> = valid.iter().map(|v| v * 255).collect();
            image::save_buffer(self.dir.join("flow_valid").join(format!("{name}.png")), &v, self.w, self.h, image::ExtendedColorType::L8)?;
        }
        if let Some(l) = fr.landcover {
            image::save_buffer(self.dir.join("landcover").join(format!("{name}.png")), l, self.w, self.h, image::ExtendedColorType::L8)?;
        }
        let (p, q) = (fr.cam.pos, q4(fr.cam.q_ecef_cam()));
        let e = fr.exposure.unwrap_or([f64::NAN; 3]);
        writeln!(
            self.csv,
            "{},{},{:.4},{:.4},{:.4},{:.9},{:.9},{:.9},{:.9},{:.7},{:.4},{:.4}",
            fr.index,
            to_us(fr.t, self.t0),
            p.x,
            p.y,
            p.z,
            q[0],
            q[1],
            q[2],
            q[3],
            e[0],
            e[1],
            e[2]
        )?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.csv.flush()?;
        Ok(())
    }
}
