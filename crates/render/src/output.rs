//! Dataset writers: PNG/NPY directory and a single HDF5 file.
//!
//! Per frame k (all at the mid-exposure pose):
//! * `rgb`         u8  [H,W,3]  developed camera image
//! * `depth`       f32 [H,W]    z-depth along the optical axis (m), +inf = sky
//! * `flow`        f32 [H,W,2]  forward optical flow k → k+1 (px, (dx, dy)); last frame = 0
//! * `flow_valid`  u8  [H,W]    1 where the flow target is visible in frame k+1
//! * `landcover`   u8  [H,W]    class id (255 = sky)
//! Pose conventions: camera frame OpenCV (x right, y down, z forward); `q_*` are [w, x, y, z]
//! Hamilton quaternions rotating camera-frame vectors into the named frame.

use crate::camera::{CameraConfig, Extrinsics};
use anyhow::Result;
use glam::{DQuat, DVec3};
use h5::Attrs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Everything written for one frame.
pub struct FrameRecord<'a> {
    pub index: usize,
    pub t: f64,
    pub rgb: &'a [u8],
    pub depth: &'a [f32],
    pub flow: &'a [f32],
    pub flow_valid: &'a [u8],
    pub landcover: &'a [u8],
    pub pose: PoseRecord,
    pub exposure_time: f64,
    pub gain: f64,
    pub ev: f64,
    pub sun_elevation: f64,
    pub sun_azimuth: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PoseRecord {
    pub cam_ecef: DVec3,
    pub q_ecef_cam: DQuat,
    /// camera position (lat deg, lon deg, h m)
    pub cam_lla: [f64; 3],
    /// camera position in the local NED frame of the first camera position
    pub cam_ned0: DVec3,
    pub q_ned0_cam: DQuat,
    /// body attitude (body → local NED at the body)
    pub q_ned_body: DQuat,
    pub body_lla: [f64; 3],
}

fn q4(q: DQuat) -> [f64; 4] {
    [q.w, q.x, q.y, q.z]
}

/// Per-sensor calibration under `<sensor group>/<layout.calib_group>`: intrinsics (4, when the
/// model has them), distortion, resolution, sensor → rig transform (row-major 4x4), and the full
/// camera description (camodocal YAML schema) as attributes.
pub fn write_sensor_calib(g: &h5::Group, layout: &crate::scenario::Layout, cam: &CameraConfig, t_rig: [f64; 16]) -> Result<()> {
    if g.exists(&layout.calib_group) {
        g.delete(&layout.calib_group)?;
    }
    let c = g.ensure_group(&layout.calib_group)?;
    if cam.intrinsics.len() == 4 {
        c.new_dataset::<f64>().shape(&[4]).create(&layout.calib_intrinsics)?.write_all(&cam.intrinsics)?;
    }
    if !cam.distortion.is_empty() {
        c.new_dataset::<f64>().shape(&[cam.distortion.len()]).create(&layout.calib_distortion)?.write_all(&cam.distortion)?;
    }
    c.new_dataset::<i64>().shape(&[2]).create(&layout.calib_resolution)?.write_all(&[cam.width as i64, cam.height as i64])?;
    c.new_dataset::<f64>().shape(&[4, 4]).create(&layout.calib_transform)?.write_all(&t_rig)?;
    c.set_attr_str("camera_model", &cam.model)?;
    c.set_attr_str("camera_yaml", &serde_yaml::to_string(cam)?)?;
    c.set_attr_str("transform_convention", "row-major 4x4: sensor frame → rig frame (rig = body FRD)")?;
    Ok(())
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

pub struct PngWriter {
    dir: PathBuf,
    w: u32,
    h: u32,
    poses: std::io::BufWriter<std::fs::File>,
    depth: bool,
    flow: bool,
    landcover: bool,
}

impl PngWriter {
    pub fn new(dir: &Path, w: u32, h: u32, cam: &CameraConfig, ext: &Extrinsics, scenario_yaml: &str, depth: bool, flow: bool, landcover: bool) -> Result<Self> {
        std::fs::create_dir_all(dir.join("rgb"))?;
        if depth {
            std::fs::create_dir_all(dir.join("depth"))?;
        }
        if flow {
            std::fs::create_dir_all(dir.join("flow"))?;
            std::fs::create_dir_all(dir.join("flow_valid"))?;
        }
        if landcover {
            std::fs::create_dir_all(dir.join("landcover"))?;
        }
        #[derive(serde::Serialize)]
        struct Rig<'a> {
            camera: &'a CameraConfig,
            extrinsics: &'a Extrinsics,
        }
        std::fs::write(dir.join("camera.yaml"), serde_yaml::to_string(&Rig { camera: cam, extrinsics: ext })?)?;
        std::fs::write(dir.join("scenario.yaml"), scenario_yaml)?;
        let mut poses = std::io::BufWriter::new(std::fs::File::create(dir.join("poses.csv"))?);
        writeln!(poses, "# camera frame OpenCV; q_ecef_cam / q_ned0_cam rotate camera vectors into ECEF / NED(first camera position); q_ned_body body(FRD)->local NED")?;
        writeln!(
            poses,
            "frame,t,x_ecef,y_ecef,z_ecef,qw_ecef_cam,qx_ecef_cam,qy_ecef_cam,qz_ecef_cam,lat,lon,h,n0,e0,d0,qw_ned0_cam,qx_ned0_cam,qy_ned0_cam,qz_ned0_cam,qw_ned_body,qx_ned_body,qy_ned_body,qz_ned_body,exposure_time,gain,ev,sun_az_deg,sun_el_deg"
        )?;
        Ok(PngWriter { dir: dir.to_path_buf(), w, h, poses, depth, flow, landcover })
    }

    pub fn write(&mut self, f: &FrameRecord) -> Result<()> {
        let name = format!("{:06}", f.index);
        image::save_buffer(self.dir.join("rgb").join(format!("{name}.png")), f.rgb, self.w, self.h, image::ExtendedColorType::Rgb8)?;
        let (w, h) = (self.w as usize, self.h as usize);
        if self.depth {
            write_npy_f32(&self.dir.join("depth").join(format!("{name}.npy")), &[h, w], f.depth)?;
        }
        if self.flow {
            write_npy_f32(&self.dir.join("flow").join(format!("{name}.npy")), &[h, w, 2], f.flow)?;
            let v: Vec<u8> = f.flow_valid.iter().map(|v| v * 255).collect();
            image::save_buffer(self.dir.join("flow_valid").join(format!("{name}.png")), &v, self.w, self.h, image::ExtendedColorType::L8)?;
        }
        if self.landcover {
            image::save_buffer(self.dir.join("landcover").join(format!("{name}.png")), f.landcover, self.w, self.h, image::ExtendedColorType::L8)?;
        }
        let p = &f.pose;
        let (a, b, c) = (q4(p.q_ecef_cam), q4(p.q_ned0_cam), q4(p.q_ned_body));
        writeln!(
            self.poses,
            "{},{:.6},{:.4},{:.4},{:.4},{:.9},{:.9},{:.9},{:.9},{:.10},{:.10},{:.4},{:.4},{:.4},{:.4},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.7},{:.4},{:.4},{:.3},{:.3}",
            f.index,
            f.t,
            p.cam_ecef.x,
            p.cam_ecef.y,
            p.cam_ecef.z,
            a[0],
            a[1],
            a[2],
            a[3],
            p.cam_lla[0],
            p.cam_lla[1],
            p.cam_lla[2],
            p.cam_ned0.x,
            p.cam_ned0.y,
            p.cam_ned0.z,
            b[0],
            b[1],
            b[2],
            b[3],
            c[0],
            c[1],
            c[2],
            c[3],
            f.exposure_time,
            f.gain,
            f.ev,
            f.sun_azimuth.to_degrees(),
            f.sun_elevation.to_degrees()
        )?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.poses.flush()?;
        Ok(())
    }
}

/// HDF5 sequence writer (datasets sized for `n` frames up front, chunk = one frame).
pub struct H5Writer {
    file: h5::File,
    w: usize,
    h: usize,
    n: usize,
    rgb: h5::Dataset,
    depth: Option<h5::Dataset>,
    flow: Option<(h5::Dataset, h5::Dataset)>,
    landcover: Option<h5::Dataset>,
    /// per-sensor frame view (layout.frames_group / frames_data)
    gray: Option<h5::Dataset>,
    layout: crate::scenario::Layout,
    t: Vec<f64>,
    poses: Vec<PoseRecord>,
    expo: Vec<[f64; 5]>,
}

impl H5Writer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(path: &Path, w: u32, h: u32, n: usize, cam: &CameraConfig, ext: &Extrinsics, scenario_yaml: &str, depth: bool, flow: bool, landcover: bool, comp: &crate::scenario::Compression, layout: &crate::scenario::Layout) -> Result<Self> {
        let lvl = comp.level.min(9);
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p)?;
            }
        }
        let (w, h) = (w as usize, h as usize);
        let file = h5::File::create(path)?;
        file.set_attr_str("format", "terrain-sequence")?;
        file.set_attr("format_version", 1i32)?;
        file.set_attr_str("scenario", scenario_yaml)?;
        file.set_attr("float_keep_bits", comp.float_keep_bits.map(|b| b as i32).unwrap_or(23))?;
        file.set_attr_str("camera", &serde_yaml::to_string(cam)?)?;
        file.set_attr_str("extrinsics", &serde_yaml::to_string(ext)?)?;
        file.set_attr_str(
            "conventions",
            "camera frame OpenCV (x right, y down, z forward); pixel centres at integer coords; \
             q_* = [w,x,y,z] rotating camera (or body FRD) vectors into the named frame; \
             ned0 = local NED at the first camera position; depth = z along the optical axis (m), inf = sky; \
             flow[k] = forward flow from frame k to k+1 in px (dx, dy); poses at mid-exposure",
        )?;
        let rgb = file.new_dataset::<u8>().shape(&[n, h, w, 3]).chunk(&[1, h, w, 3]).shuffle(true).deflate(lvl).create("rgb")?;
        let depth = if depth {
            Some(file.new_dataset::<f32>().shape(&[n, h, w]).chunk(&[1, h, w]).shuffle(true).deflate(lvl).create("depth")?)
        } else {
            None
        };
        let flow = if flow {
            Some((
                file.new_dataset::<f32>().shape(&[n, h, w, 2]).chunk(&[1, h, w, 2]).shuffle(true).deflate(lvl).create("flow")?,
                file.new_dataset::<u8>().shape(&[n, h, w]).chunk(&[1, h, w]).deflate(lvl).create("flow_valid")?,
            ))
        } else {
            None
        };
        let landcover = if landcover {
            Some(file.new_dataset::<u8>().shape(&[n, h, w]).chunk(&[1, h, w]).deflate(lvl).create("landcover")?)
        } else {
            None
        };
        // camera intrinsics in camodocal's schema (model, intrinsics, distortion, xi, ...)
        {
            let g = file.ensure_group("camera")?;
            g.set_attr_str("model", &cam.model)?;
            g.set_attr("width", w as i32)?;
            g.set_attr("height", h as i32)?;
            for (k, v) in [("intrinsics", &cam.intrinsics), ("distortion", &cam.distortion), ("inv_poly", &cam.inv_poly), ("affine", &cam.affine), ("center", &cam.center)] {
                if !v.is_empty() {
                    g.set_attr_array(k, v)?;
                }
            }
            g.set_attr("xi", cam.xi)?;
            g.set_attr("max_fov_deg", cam.max_fov_deg)?;
            g.set_attr_str("yaml", &serde_yaml::to_string(cam)?)?;
            if cam.intrinsics.len() == 4 {
                let i = &cam.intrinsics;
                g.set_attr_array("K", &[i[0], 0.0, i[2], 0.0, i[1], i[3], 0.0, 0.0, 1.0])?;
            }
            g.set_attr_array("q_body_cam", &q4(DQuat::from_mat3(&ext.r_body_cam())))?;
            g.set_attr_array("t_body_cam", &ext.translation)?;
        }
        let gray = if layout.enabled {
            use crate::scenario::h5path;
            let g = file.ensure_group(h5path(&layout.frames_group))?;
            let ds = if layout.frames_gray {
                g.new_dataset::<u8>().shape(&[n, h, w]).chunk(&[1, h, w]).deflate(lvl).create(&layout.frames_data)?
            } else {
                g.new_dataset::<u8>().shape(&[n, h, w, 3]).chunk(&[1, h, w, 3]).deflate(lvl).create(&layout.frames_data)?
            };
            write_sensor_calib(&g, layout, cam, crate::imu::t_body_sensor(ext.r_body_cam(), ext.t_body_cam()))?;
            Some(ds)
        } else {
            None
        };
        Ok(H5Writer { file, w, h, n, rgb, depth, flow, landcover, gray, layout: layout.clone(), t: vec![], poses: vec![], expo: vec![] })
    }

    pub fn write(&mut self, f: &FrameRecord) -> Result<()> {
        let k = f.index;
        let (w, h) = (self.w, self.h);
        self.rgb.write_slice(f.rgb, &[k, 0, 0, 0], &[1, h, w, 3])?;
        if let Some(d) = &self.depth {
            d.write_slice(f.depth, &[k, 0, 0], &[1, h, w])?;
        }
        if let Some((fl, fv)) = &self.flow {
            fl.write_slice(f.flow, &[k, 0, 0, 0], &[1, h, w, 2])?;
            fv.write_slice(f.flow_valid, &[k, 0, 0], &[1, h, w])?;
        }
        if let Some(l) = &self.landcover {
            l.write_slice(f.landcover, &[k, 0, 0], &[1, h, w])?;
        }
        if let Some(g) = &self.gray {
            if self.layout.frames_gray {
                let gray: Vec<u8> = f.rgb.chunks_exact(3).map(|c| (0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64).round() as u8).collect();
                g.write_slice(&gray, &[k, 0, 0], &[1, h, w])?;
            } else {
                g.write_slice(f.rgb, &[k, 0, 0, 0], &[1, h, w, 3])?;
            }
        }
        self.t.push(f.t);
        self.poses.push(f.pose);
        self.expo.push([f.exposure_time, f.gain, f.ev, f.sun_azimuth.to_degrees(), f.sun_elevation.to_degrees()]);
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        let n = self.t.len();
        let f = &self.file;
        let put = |name: &str, cols: usize, data: Vec<f64>| -> Result<()> {
            let shape: Vec<usize> = if cols == 1 { vec![n] } else { vec![n, cols] };
            let ds = f.new_dataset::<f64>().shape(&shape).create(name)?;
            if n > 0 {
                ds.write_all(&data)?;
            }
            Ok(())
        };
        put("t", 1, self.t.clone())?;
        if self.gray.is_some() {
            let t0 = self.t.first().copied().unwrap_or(0.0);
            let ts: Vec<i64> = self.t.iter().map(|t| ((t - t0) * 1e6).round() as i64).collect();
            let ds = f.new_dataset::<i64>().shape(&[n]).create(crate::scenario::h5path(&self.layout.frames_ts))?;
            if n > 0 {
                ds.write_all(&ts)?;
            }
            f.set_attr("t0", t0)?;
            f.set_attr_str("time_base", "per-sensor view (output.layout): µs since t0 (trajectory time of the first frame); /t and /pose use trajectory time in seconds")?;
        }
        let p = &self.poses;
        put("pose/cam_position_ecef", 3, p.iter().flat_map(|p| p.cam_ecef.to_array()).collect())?;
        put("pose/cam_q_ecef", 4, p.iter().flat_map(|p| q4(p.q_ecef_cam)).collect())?;
        put("pose/cam_lla", 3, p.iter().flat_map(|p| p.cam_lla).collect())?;
        put("pose/cam_position_ned0", 3, p.iter().flat_map(|p| p.cam_ned0.to_array()).collect())?;
        put("pose/cam_q_ned0", 4, p.iter().flat_map(|p| q4(p.q_ned0_cam)).collect())?;
        put("pose/body_q_ned", 4, p.iter().flat_map(|p| q4(p.q_ned_body)).collect())?;
        put("pose/body_lla", 3, p.iter().flat_map(|p| p.body_lla).collect())?;
        put("exposure/time", 1, self.expo.iter().map(|e| e[0]).collect())?;
        put("exposure/gain", 1, self.expo.iter().map(|e| e[1]).collect())?;
        put("exposure/ev", 1, self.expo.iter().map(|e| e[2]).collect())?;
        put("sun/azimuth_deg", 1, self.expo.iter().map(|e| e[3]).collect())?;
        put("sun/elevation_deg", 1, self.expo.iter().map(|e| e[4]).collect())?;
        if let Some(p0) = p.first() {
            let g = f.group("pose")?;
            g.set_attr_array("ned0_origin_lla", &p0.cam_lla)?;
        }
        if n < self.n {
            f.set_attr("frames_written", n as i32)?;
        }
        f.flush()?;
        Ok(())
    }
}
