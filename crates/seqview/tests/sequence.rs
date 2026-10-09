//! A tiny sequence written with the dataset writer (`render::output`, events, IMU) and read
//! back by the viewer's reader, its snapshot and the exporter. CPU only.

use clap::Parser;
use glam::{DMat3, DQuat, DVec3};
use render::output::{self, BodySample, CameraWriter, Frame};
use render::scenario::Scenario;
use seqview::seq::{FrameData, Key, Modality, Sequence};
use std::path::{Path, PathBuf};

const W: usize = 32;
const H: usize = 24;
const N: usize = 5;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("seqview-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rgb(k: usize) -> Vec<u8> {
    (0..W * H).flat_map(|i| [(i % 256) as u8, (k * 40) as u8, 200]).collect()
}

fn depth(k: usize) -> Vec<f32> {
    // the top rows are sky
    (0..W * H).map(|i| if i < 2 * W { f32::INFINITY } else { 100.0 + (i / W) as f32 * 10.0 + k as f32 }).collect()
}

/// Writes the sequence: /cam0 (rgb, depth, flow, land cover, stars; 5 frames at 10 Hz),
/// /rig/dvs (events only), /imu and /pose.
fn write(dir: &Path) -> PathBuf {
    let file = dir.join("seq.h5");
    let yaml = format!(
        r#"
cameras:
  - path: /cam0
    intrinsics: {{ model: pinhole, width: {W}, height: {H}, intrinsics: [20.0, 20.0, 15.5, 11.5] }}
    frame_rate: 10
    rgb: {{}}
    depth: {{}}
    flow: {{}}
    landcover: {{}}
    stars: {{}}
  - path: /rig/dvs
    intrinsics: {{ model: pinhole, width: 16, height: 12, intrinsics: [10.0, 10.0, 7.5, 5.5] }}
    events: {{}}
imu: {{ path: /imu, rate_hz: 100 }}
output: {{ file: {} }}
"#,
        file.display()
    );
    let cfg = dir.join("scenario.yaml");
    std::fs::write(&cfg, yaml).unwrap();
    let scn = Scenario::load(&cfg).unwrap();
    let ell = geodesy::Ellipsoid::WGS84;
    let f = output::create_file(&scn, 0.0).unwrap();
    // body poses: 21 samples over 2 s, flying north
    let samples: Vec<BodySample> = (0..21)
        .map(|i| {
            let t = i as f64 * 0.1;
            let geo = geodesy::Geodetic { lat: (40.0 + t * 1e-4).to_radians(), lon: 30f64.to_radians(), h: 1000.0 };
            BodySample { t, pose: render::Pose { t, geo, q_ned_body: DQuat::IDENTITY }, sun: scn.render.lighting.sun_at(t, geo.lat, geo.lon) }
        })
        .collect();
    output::write_body_pose(&f, &scn.output.pose.path, 0.0, &samples, &ell).unwrap();
    // the frame camera
    let mut cw = CameraWriter::new(&f, &scn.cameras[0], N, &scn.output.compression, 0.0).unwrap();
    for k in 0..N {
        let t = k as f64 * 0.1;
        let (img, dep) = (rgb(k), depth(k));
        let flow: Vec<f32> = (0..W * H).flat_map(|_| [1.0, -0.5 * k as f32]).collect();
        let valid: Vec<u8> = (0..W * H).map(|i| (i % 7 != 0) as u8).collect();
        let lc: Vec<u8> = (0..W * H)
            .map(|i| {
                if i < 2 * W {
                    255
                } else if i % 2 == 0 {
                    terragen::landcover::FOREST
                } else {
                    terragen::landcover::CROP
                }
            })
            .collect();
        let stars = [render::stars::StarObs { id: 7, x: 5.0, y: 1.0, xm: 5.0, ym: 1.0, v: 2.0 + k as f32, irradiance: 1e-12, visible: true }];
        let cam = render::CamPose { t, pos: DVec3::new(ell.a + 1000.0, 0.0, 0.0), r_ecef_cam: DMat3::IDENTITY };
        cw.write(&Frame {
            index: k,
            t,
            cam,
            rgb: Some(&img),
            exposure: Some([0.001, 1.0, 0.0]),
            depth: Some(&dep),
            flow: Some((&flow, &valid)),
            landcover: Some(&lc),
            stars: Some(if k == 1 { &[] } else { &stars }),
        })
        .unwrap();
    }
    assert_eq!(cw.finish().unwrap(), N);
    output::describe(&f, &scn).unwrap();
    // the event camera: calibration and events only, 1 kHz of events over 1 s
    let spec = &scn.cameras[1];
    let g = f.ensure_group("rig/dvs").unwrap();
    output::write_camera_calib(&g, &spec.intrinsics, output::transform_4x4(spec.extrinsics.r_body_cam(), spec.extrinsics.t_body_cam())).unwrap();
    let mut ew = render::events::EventWriter::new(&g, 0.0, 4).unwrap();
    let ev: Vec<render::events::Event> =
        (0..1000).map(|i| render::events::Event { t_us: i * 1000 + 500, x: (i % 16) as u16, y: 3, p: (i % 2) as i8 }).collect();
    ew.write(&ev).unwrap();
    ew.finish(spec.events.as_ref().unwrap()).unwrap();
    // the IMU
    let imu = scn.imu.as_ref().unwrap();
    let n = 201;
    let v = |x: f64| vec![DVec3::new(x, 0.0, -9.8); n];
    let data = render::imu::ImuData {
        t: (0..n).map(|i| i as f64 * 0.01).collect(),
        accel: v(0.1),
        omega: v(0.01),
        gt_accel: v(0.0),
        gt_omega: v(0.0),
        bias_accel: v(0.0),
        bias_gyro: v(0.0),
        from_truth_columns: false,
    };
    render::imu::write_h5(&f, imu, &data, 0.0, 4).unwrap();
    drop(g);
    drop(f);
    file
}

#[test]
fn reads_what_the_writer_wrote() {
    let dir = scratch("read");
    let file = write(&dir);
    let seq = Sequence::open(&file).unwrap();
    assert_eq!(seq.format, "terrain-sequence");
    assert!(seq.scenario_yaml.contains("/rig/dvs"));
    let paths: Vec<&str> = seq.cameras.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(paths, ["/cam0", "/rig/dvs"]);
    let c = &seq.cameras[0];
    assert_eq!((c.w, c.h, c.t.len()), (W, H, N));
    assert_eq!(c.modalities(), [Modality::Rgb, Modality::Depth, Modality::Flow, Modality::Landcover]);
    assert_eq!(c.t, [0, 100_000, 200_000, 300_000, 400_000]);
    assert_eq!(c.landcover.as_ref().unwrap()[terragen::landcover::FOREST as usize], "forest");
    assert_eq!(c.intrinsics, [20.0, 20.0, 15.5, 11.5]);
    assert_eq!(c.model, "pinhole");
    assert!(c.stars.is_some());
    let dvs = &seq.cameras[1];
    assert!(dvs.t.is_empty());
    assert_eq!(dvs.modalities(), [Modality::Events]);
    assert_eq!(dvs.events.as_ref().unwrap().n, 1000);
    assert_eq!(seq.camera_index("dvs").unwrap(), 1);
    assert_eq!(seq.camera_index("/cam0").unwrap(), 0);
    assert!(seq.camera_index("/nope").unwrap_err().to_string().contains("/cam0, /rig/dvs"));
    assert_eq!(seq.imu.as_ref().unwrap().t.len(), 201);
    let p = seq.pose.as_ref().unwrap();
    assert_eq!(p.t.len(), 21);
    assert!((p.lla[0][0] - 40.0).abs() < 1e-9 && p.rpy[0].iter().all(|a| a.abs() < 1e-9));
    assert!(p.ned0[20][0] > 10.0); // flew north
    assert_eq!(seq.time_range(), (0, 2_000_000));

    // frames
    match seq.read(Key::Frame { cam: 0, m: Modality::Rgb, k: 3 }).unwrap() {
        FrameData::Image { channels: 3, px } => assert_eq!(px, rgb(3)),
        d => panic!("{d:?}"),
    }
    match seq.read(Key::Frame { cam: 0, m: Modality::Depth, k: 2 }).unwrap() {
        FrameData::Depth(d) => {
            assert!(d[0].is_infinite());
            assert_eq!(d[5 * W], depth(2)[5 * W]);
        }
        d => panic!("{d:?}"),
    }
    match seq.read(Key::Frame { cam: 0, m: Modality::Flow, k: 4 }).unwrap() {
        FrameData::Flow { flow, valid } => {
            assert_eq!(&flow[..2], &[1.0, -2.0]);
            assert_eq!(valid.unwrap()[7], 0);
        }
        d => panic!("{d:?}"),
    }
    match seq.read(Key::Frame { cam: 0, m: Modality::Landcover, k: 0 }).unwrap() {
        FrameData::Landcover(l) => assert_eq!((l[0], l[2 * W]), (255, terragen::landcover::FOREST)),
        d => panic!("{d:?}"),
    }
    assert!(seq.read(Key::Frame { cam: 0, m: Modality::Rgb, k: N }).is_err());
    assert!(seq.read(Key::Frame { cam: 1, m: Modality::Depth, k: 0 }).is_err());
    // stars: frame 1 has none
    match seq.read(Key::Stars { cam: 0, k: 1 }).unwrap() {
        FrameData::Stars(s) => assert!(s.is_empty()),
        d => panic!("{d:?}"),
    }
    match seq.read(Key::Stars { cam: 0, k: 2 }).unwrap() {
        FrameData::Stars(s) => assert_eq!((s.len(), s[0].id, s[0].v, s[0].visible), (1, 7, 4.0, true)),
        d => panic!("{d:?}"),
    }
    // events: (10 ms, 20 ms] holds the events at 10.5 .. 19.5 ms
    let e = seq.read_events(1, 20_000, 10_000).unwrap();
    assert_eq!(e.count, 10);
    assert_eq!(e.on.iter().sum::<u32>() + e.off.iter().sum::<u32>(), 10);
    assert_eq!(e.on[3 * 16 + 11], 1); // event 11: x 11, ON
    assert_eq!(seq.read_events(1, -5, 1000).unwrap().count, 0);
    assert_eq!(seq.read_events(1, 10_000_000, 1000).unwrap().count, 0);
    assert_eq!(seq.read_events(1, 2_000_000, 2_000_000).unwrap().count, 1000);
    // IMU window
    match seq.read(Key::Imu { t0: 100_000, t1: 200_000 }).unwrap() {
        FrameData::Imu(w) => {
            assert_eq!(w.t.len(), 11);
            assert_eq!(w.accel[0], [0.1, 0.0, -9.8]);
        }
        d => panic!("{d:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn opens_nothing_else() {
    let dir = scratch("other");
    assert!(Sequence::open(&dir.join("missing.h5")).err().unwrap().to_string().contains("does not exist"));
    let f = h5::File::create(dir.join("empty.h5")).unwrap();
    drop(f);
    assert!(Sequence::open(&dir.join("empty.h5")).err().unwrap().to_string().contains("no cameras"));
    std::fs::write(dir.join("text.h5"), "not hdf5").unwrap();
    assert!(Sequence::open(&dir.join("text.h5")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[derive(Parser)]
struct ShowCli {
    #[command(flatten)]
    o: seqview::ShowOptions,
}

#[derive(Parser)]
struct ExportCli {
    #[command(flatten)]
    o: seqview::ExportOptions,
}

fn png_size(p: &Path) -> (u32, u32) {
    let img = image::open(p).unwrap();
    (img.width(), img.height())
}

#[test]
fn snapshots_without_a_gpu() {
    let dir = scratch("snap");
    let file = write(&dir);
    let out = dir.join("snap.png");
    let f = file.to_str().unwrap();
    let o = ShowCli::try_parse_from(["show", f, "--snapshot", out.to_str().unwrap(), "--frame", "2", "--size", "900x600", "--pointer", "100,100"]).unwrap().o;
    seqview::show(o).unwrap();
    assert_eq!(png_size(&out), (900, 600));
    // the events-only camera, and pixels per point 2
    let o = ShowCli::try_parse_from(["show", f, "--snapshot", out.to_str().unwrap(), "--camera", "dvs", "--time", "0.5", "--size", "400x300", "--scale", "2"])
        .unwrap()
        .o;
    seqview::show(o).unwrap();
    assert_eq!(png_size(&out), (800, 600));
    // checked options
    for bad in [
        vec!["--frame", "5"],
        vec!["--camera", "/nope"],
        vec!["--modalities", "rgb,normals"],
        vec!["--camera", "dvs", "--modalities", "rgb"],
        vec!["--depth-range", "10,5"],
        vec!["--window-ms", "0"],
        vec!["--size", "10x10"],
        vec!["--pointer", "1"],
    ] {
        let mut args = vec!["show", f, "--snapshot", out.to_str().unwrap()];
        args.extend(&bad);
        let o = ShowCli::try_parse_from(&args).unwrap().o;
        assert!(seqview::show(o).is_err(), "{bad:?} accepted");
    }
    // clap: --frame and --time exclude each other, --events takes known styles
    assert!(ShowCli::try_parse_from(["show", f, "--frame", "1", "--time", "0.1"]).is_err());
    assert!(ShowCli::try_parse_from(["show", f, "--events", "rainbow"]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

fn count_pngs(d: &Path) -> usize {
    std::fs::read_dir(d).map(|r| r.filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "png")).count()).unwrap_or(0)
}

#[test]
fn exports_png_sequences() {
    let dir = scratch("export");
    let file = write(&dir);
    let f = file.to_str().unwrap();
    let out = dir.join("pngs");
    let o = ExportCli::try_parse_from(["export", f, "--out", out.to_str().unwrap()]).unwrap().o;
    seqview::export(o).unwrap();
    for m in ["rgb", "depth", "flow", "landcover"] {
        assert_eq!(count_pngs(&out.join(m)), N, "{m}");
        assert_eq!(png_size(&out.join(m).join("000000.png")), (W as u32, H as u32));
    }
    // the exported RGB is the stored RGB, with the star (mag 5 at (5, 1)) ringed in yellow
    let img = image::open(out.join("rgb/000003.png")).unwrap().to_rgb8().into_raw();
    let want = rgb(3);
    assert!(img[3 * 8 * W..] == want[3 * 8 * W..], "rows below the star differ");
    let ring = 3 * (W + 5 + 3); // 3 px right of the star: on its ring (radius 2.5 .. 3.5)
    assert!(img[ring..ring + 3] != want[ring..ring + 3]);
    let csv = std::fs::read_to_string(out.join("frames.csv")).unwrap();
    assert_eq!(csv.lines().count(), N + 1);
    assert_eq!(csv.lines().nth(2).unwrap(), "1,1,100000");

    // a range, every other frame, side by side
    let grid = dir.join("grid");
    let o = ExportCli::try_parse_from([
        "export",
        f,
        "-m",
        "rgb,depth,flow",
        "--side-by-side",
        "--start",
        "0.1",
        "--end",
        "0.4",
        "--every",
        "2",
        "--out",
        grid.to_str().unwrap(),
    ])
    .unwrap()
    .o;
    seqview::export(o).unwrap();
    assert_eq!(count_pngs(&grid.join("grid")), 2);
    assert!(grid.join("grid/000001.png").exists() && grid.join("grid/000003.png").exists());
    let (gw, gh) = png_size(&grid.join("grid/000001.png"));
    assert!(gw > 3 * W as u32 && gh > H as u32);

    // the events-only camera at 20 Hz: pictures from one window in to the end (2 s)
    let ev = dir.join("ev");
    let o = ExportCli::try_parse_from(["export", f, "--camera", "/rig/dvs", "--rate", "20", "--window-ms", "50", "--out", ev.to_str().unwrap()]).unwrap().o;
    seqview::export(o).unwrap();
    assert_eq!(count_pngs(&ev.join("events")), 40);

    // checked options
    for bad in [
        vec!["--camera", "/nope"],
        vec!["-m", "events"],
        vec!["-m", "rgb,normals"],
        vec!["--every", "0"],
        vec!["--start", "1", "--end", "0.5"],
        vec!["--start", "5"],
        vec!["--fps", "0"],
        vec!["--depth-range", "1"],
    ] {
        let mut args = vec!["export", f, "--out", ev.to_str().unwrap()];
        args.extend(&bad);
        let o = ExportCli::try_parse_from(&args).unwrap().o;
        assert!(seqview::export(o).is_err(), "{bad:?} accepted");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
