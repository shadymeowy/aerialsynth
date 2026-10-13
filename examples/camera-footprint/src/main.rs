//! Camera footprint: camera ↔ geodesy ↔ XYZ tile coordinates, step by step.
//!
//! One camera position, two orientations:
//! * an **oblique** camera (looking 32° below the horizon, heading 60°), and
//! * a **nadir** camera at the same position: straight down, north up, east right.
//!
//! The example renders both, casts the rays of the oblique camera's 4 image corners onto the
//! terrain, converts those ground points to fractional Web-Mercator XYZ tile coordinates,
//! stitches the map tiles under them (albedo and elevation), draws the footprint on the maps,
//! and warps the map into the oblique camera's image with the homography of the 4 corners.
//!
//! Frames (every conversion below names its source and target):
//!
//! ```text
//!  geodetic (lat, lon, h)  ──geodetic2ecef──▶  ECEF (x, y, z)  ──R_enu_ecef, minus origin──▶  ENU (e, n, u)
//!   WGS84, h above ellipsoid                  Earth-centred, metres      local tangent plane at the camera
//!
//!  ENU ──R_cam_enu──▶ camera (x right, y down, z optical axis; OpenCV) ──K──▶ image pixel (u, v)
//!                                                                           (0, 0) = centre of the top-left pixel
//!
//!  (lat, lon) ──Web Mercator──▶ fractional tile (x, y) at zoom LOD ──minus the first tile, ×256──▶ stitched-map pixel
//!                               integer part = XYZ tile, x east, y south   pixel (i, j) covers [i, i+1) × [j, j+1)
//! ```
//!
//! Run from the workspace root: `cargo run --release -p camera-footprint [OUT_DIR] [GENERATOR] [ALT]`
//! (default `out/camera-footprint`; the tiles are generated into `OUT_DIR/world.h5`; GENERATOR:
//! `classic`, the default, or `next`; ALT: the camera height in metres above the ellipsoid
//! instead of [`ALT`], e.g. for another generator's ground).

use aerialsynth_core::{Camera, CameraDef, Frame, Layer, Mount, Outputs, Pinhole, Pose, World, TILE_PX};
use anyhow::{bail, ensure, Context, Result};
use geodesy::{ecef2geodetic, geodetic2ecef, latlon_to_uv, uv_to_latlon, DMat3, DVec2, DVec3, Ellipsoid, Geodetic, LocalConvention, LocalFrame};
use image::{Rgb, RgbImage};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ============================================================================================
// The scene: change these
// ============================================================================================

/// Capture time, UTC (places the sun): midsummer noon (the v0.2.0 showcase's "boreal" shot).
const TIME: &str = "2026-06-21T12:00:00Z";
/// Camera position: geodetic latitude / longitude (degrees, WGS84) and height above the WGS84
/// ellipsoid (metres). The pose of the v0.2.0 showcase's "boreal" shot (spruce, bogs and rivers
/// at 53°N): 900 m above the ground, which is at 1751.7 m here in the default (classic) world
/// (the surface height below the camera, `World::surface_height`).
const LAT: f64 = 52.97924;
const LON: f64 = -16.41426;
const ALT: f64 = 2651.7;
/// Orientation of the oblique camera (degrees), as rotations away from the nadir camera.
/// (0, 0, 0) = looking straight down with north at the image top and east at the image right.
/// The angles are the aerospace yaw-pitch-roll (Z-Y-X) of an aircraft whose camera looks out
/// of its belly with the image top towards the nose (see [`r_enu_cam`]):
/// * YAW: heading of the image top, clockwise from north;
/// * PITCH: nose up, i.e. the view tilts from nadir towards the image top (58° = 32° below the horizon);
/// * ROLL: right wing down (the view tilts to the left of the image).
///
/// The showcase's camera: heading 60°, 32° below the horizon, level.
const YAW: f64 = 60.0;
const PITCH: f64 = 58.0;
const ROLL: f64 = 0.0;
/// Pinhole camera without distortion: image size (pixels) and focal length (pixels); the
/// principal point is the image centre.
const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;
const FOCAL_PX: f64 = 900.0;
/// Zoom level (LOD) of the XYZ map tiles that are stitched (z15: ~2.4 m pixels here).
const LOD: u8 = 15;
/// Zoom level of the elevation tiles the corner rays are intersected with (coarser = fewer tiles).
const ELEV_LOD: u8 = 13;
/// The ray / terrain intersection: march along the ray in steps of this fraction of an
/// elevation pixel (so no ridge between two samples is missed), then bisect the step that
/// crosses the ground until the ray is this close to the terrain (metres), giving up beyond a
/// range (metres).
const RAY_STEP_PX: f64 = 0.25;
const RAY_TOLERANCE_M: f64 = 0.01;
const RAY_MAX_RANGE_M: f64 = 200_000.0;
/// World seed: 1 is the default world (terragen's default config).
const SEED: u64 = 1;
/// Terrain generator when the command line names none: the default, `classic` (`next`: the
/// modular generator, a preview).
const GENERATOR: &str = "classic";
/// Width (map pixels) of the red footprint outline.
const LINE_PX: f64 = 3.0;

fn main() -> Result<()> {
    let out_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "out/camera-footprint".into()));
    let generator = std::env::args().nth(2).unwrap_or_else(|| GENERATOR.into());
    // (another world's ground is elsewhere: its camera height can be given, metres above the ellipsoid)
    let alt: f64 = std::env::args().nth(3).map(|a| a.parse().with_context(|| format!("ALT {a:?}: metres expected"))).transpose()?.unwrap_or(ALT);
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let ell = Ellipsoid::WGS84;

    // ----------------------------------------------------------------------------------------
    println!("== 0. World and time");
    let world = Arc::new(World::open_generator(&out_dir.join("world.h5"), None, Some(SEED), Some(&generator))?);
    world.set_verbose(true);
    let unix_time = parse_time(TIME)?;
    println!("world seed {} (default config), generator {}, tiles in {}", world.seed(), world.generator(), world.path().display());
    println!("time {TIME} = {unix_time:.0} s Unix");

    // ----------------------------------------------------------------------------------------
    println!("\n== 1. Intrinsics: camera frame -> pixels");
    let hfov_deg = 2.0 * (0.5 * WIDTH as f64 / FOCAL_PX).atan().to_degrees();
    let pinhole = Pinhole { mount: Mount::Nadir, ..Pinhole::new(WIDTH, HEIGHT, hfov_deg) };
    let camera = Camera::new(world.clone(), None, CameraDef::Pinhole(pinhole), None)?;
    let (_, k, _) = camera.model(); // fx, fy, cx, cy as the renderer uses them
    let intr = Intrinsics { fx: k[0], fy: k[1], cx: k[2], cy: k[3] };
    println!(
        "{WIDTH} x {HEIGHT} px, fx = fy = {:.3} px, (cx, cy) = ({:.1}, {:.1}), hfov {hfov_deg:.2}°, renders on {:?}",
        intr.fx,
        intr.cx,
        intr.cy,
        camera.backend()
    );

    // ----------------------------------------------------------------------------------------
    println!("\n== 2. Extrinsics: one position, two rotations");
    let position = Geodetic::from_deg(LAT, LON, alt);
    let oblique = CameraPose::new(position, r_enu_cam(YAW, PITCH, ROLL), ell);
    let nadir = CameraPose::new(position, r_enu_cam(0.0, 0.0, 0.0), ell);
    let ground_below = world.surface_height(LAT, LON)?;
    println!("position  geodetic {}  ({:.1} m above the ground at {ground_below:.1} m)", fmt_geo(position), alt - ground_below);
    println!("          ECEF     {}", fmt_v(oblique.ecef));
    for (name, cam) in [("nadir", &nadir), ("oblique", &oblique)] {
        let r = cam.r_enu_cam;
        println!("{name:8}  camera axes in ENU: x(right) {}  y(down) {}  z(optical) {}", fmt_dir(r.x_axis), fmt_dir(r.y_axis), fmt_dir(r.z_axis));
    }

    // ----------------------------------------------------------------------------------------
    println!("\n== 3. Render both cameras");
    let pose = |yaw, pitch, roll| Pose { lat_deg: LAT, lon_deg: LON, height_m: alt, yaw_deg: yaw, pitch_deg: pitch, roll_deg: roll };
    let t0 = std::time::Instant::now();
    let img_oblique = camera.render(&pose(YAW, PITCH, ROLL), Some(unix_time), Outputs { rgb: true, depth: true, landcover: false })?;
    let img_nadir = camera.render(&pose(0.0, 0.0, 0.0), Some(unix_time), Outputs::default())?;
    println!(
        "rendered in {:.1} s; sun azimuth {:.1}°, elevation {:.1}°",
        t0.elapsed().as_secs_f64(),
        img_oblique.sun_azimuth_deg,
        img_oblique.sun_elevation_deg
    );
    // the renderer reports its camera pose: it must be ours
    for (name, cam, frame) in [("oblique", &oblique, &img_oblique), ("nadir", &nadir, &img_nadir)] {
        let (dp, dr) = cam.difference_to(frame);
        println!("{name:8}  renderer's pose vs ours: position {dp:.2e} m, rotation {dr:.2e}");
        ensure!(dp < 1e-3 && dr < 1e-9, "the renderer's {name} camera pose differs from the one computed here");
    }
    let oblique_rgb = frame_image(&img_oblique)?;
    save(&oblique_rgb, &out_dir, "1_oblique_camera.png")?;
    save(&frame_image(&img_nadir)?, &out_dir, "2_nadir_camera.png")?;

    // ----------------------------------------------------------------------------------------
    let mut elev = ElevationSampler::new(&world, ELEV_LOD);
    let step = RAY_STEP_PX * elev.pixel_size(position.lat, &ell);
    println!(
        "\n== 4. Corner rays of the oblique camera ∩ terrain (elevation tiles at zoom {ELEV_LOD}: steps of {step:.2} m, bisection to {RAY_TOLERANCE_M} m)"
    );
    let corners_px = image_corners();
    let mut ground = Vec::new();
    for (name, px) in CORNER_NAMES.iter().zip(corners_px) {
        let ray = oblique.ray_ecef(intr.pixel_to_ray(px));
        let hit = intersect_terrain(oblique.ecef, ray, step, &ell, |g| elev.height_at(g))?;
        let enu = oblique.enu.geodetic_to_local(hit.point);
        println!("{name:12} pixel ({:7.1}, {:6.1})", px.x, px.y);
        println!(
            "             marched {} steps to {:.1} m (ray {:.2} m above the terrain) .. {:.1} m ({:.2} m below), {} bisections: residual {:+.4} m",
            hit.steps, hit.bracket[0].0, hit.bracket[0].1, hit.bracket[1].0, -hit.bracket[1].1, hit.bisections, hit.residual
        );
        println!("             GPS  {}", fmt_geo(hit.point));
        println!("             ECEF {}", fmt_v(geodetic2ecef(hit.point, &ell)));
        println!("             ENU  {}  (range {:.0} m)", fmt_v(enu), enu.length());
        ground.push(hit.point);
    }

    // ----------------------------------------------------------------------------------------
    println!("\n== 5. Ground corners -> fractional XYZ tiles at zoom {LOD}");
    let tiles: Vec<TileXY> = ground.iter().map(|g| TileXY::from_geodetic(*g, LOD)).collect();
    for (name, t) in CORNER_NAMES.iter().zip(&tiles) {
        let (tx, ty) = t.tile();
        println!(
            "{name:12} tile (x, y) = ({:.4}, {:.4})  -> in tile {LOD}/{tx}/{ty} at ({:.1}, {:.1}) px",
            t.x,
            t.y,
            (t.x - tx as f64) * 256.0,
            (t.y - ty as f64) * 256.0
        );
    }
    let grid = MapGrid::covering(&tiles, LOD);
    println!(
        "coverage at zoom {LOD}: x {}..={} , y {}..={}  ({} x {} tiles, {} x {} px)",
        grid.x0,
        grid.x0 + grid.nx - 1,
        grid.y0,
        grid.y0 + grid.ny - 1,
        grid.nx,
        grid.ny,
        grid.width(),
        grid.height()
    );

    // ----------------------------------------------------------------------------------------
    println!("\n== 6. Stitch the tiles (albedo, elevation)");
    let albedo = grid.stitch_albedo(&world)?;
    let elevation = grid.stitch_elevation(&world)?;
    let (emin, emax) = elevation.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    println!("elevation range {emin:.1} .. {emax:.1} m (drawn black .. white)");
    let elevation_img = grayscale(&elevation, grid.width(), grid.height(), emin, emax);
    save(&albedo, &out_dir, "3_map_albedo.png")?;
    save(&elevation_img, &out_dir, "6_map_elevation.png")?;

    // ----------------------------------------------------------------------------------------
    println!("\n== 7. The footprint on the maps (stitched-map pixels)");
    let quad: Vec<DVec2> = tiles.iter().map(|t| grid.pixel_of(*t)).collect();
    for (name, q) in CORNER_NAMES.iter().zip(&quad) {
        println!("{name:12} map pixel ({:8.2}, {:8.2})", q.x, q.y);
    }
    let red = Rgb([255, 0, 0]);
    let mut albedo_quad = albedo.clone();
    draw_polygon(&mut albedo_quad, &quad, red, LINE_PX);
    save(&albedo_quad, &out_dir, "4_map_albedo_footprint.png")?;
    let mut elevation_quad = elevation_img.clone();
    draw_polygon(&mut elevation_quad, &quad, red, LINE_PX);
    save(&elevation_quad, &out_dir, "7_map_elevation_footprint.png")?;

    // ----------------------------------------------------------------------------------------
    println!("\n== 8. Homography map pixels -> oblique image pixels, warp the map");
    let h = Homography::from_4_points(&quad, &corners_px)?;
    println!("H = {}", fmt_h(&h));
    for (q, c) in quad.iter().zip(corners_px) {
        println!("H ({:8.2}, {:8.2}) = ({:7.2}, {:6.2})  corner ({:7.1}, {:6.1})", q.x, q.y, h.apply(*q).x, h.apply(*q).y, c.x, c.y);
    }
    let warped = warp_into_camera(&albedo, &h, WIDTH, HEIGHT);
    save(&warped, &out_dir, "5_map_warped_into_oblique.png")?;

    // ----------------------------------------------------------------------------------------
    println!("\n== 9. Check: does the warped map line up with the rendered image?");
    check_alignment(&img_oblique, &oblique, &intr, &grid, &h, camera.depth_kind_range(), &ell)?;
    save(&checkerboard(&oblique_rgb, &warped, 128), &out_dir, "8_check_oblique_vs_warped.png")?;
    println!("\nwrote the images to {}", out_dir.display());
    Ok(())
}

// ============================================================================================
// Time
// ============================================================================================

/// "YYYY-MM-DDTHH:MM:SSZ" (UTC) → Unix seconds.
fn parse_time(s: &str) -> Result<f64> {
    let (date, time) = s.trim_end_matches('Z').split_once('T').context("TIME: YYYY-MM-DDTHH:MM:SSZ expected")?;
    render::lighting::parse_utc(date, time)
}

// ============================================================================================
// Camera: intrinsics (camera frame ↔ pixels) and extrinsics (camera frame ↔ ENU ↔ ECEF)
// ============================================================================================

/// Pinhole intrinsics `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` (no distortion).
/// Camera frame: x right, y down, z along the optical axis (OpenCV). Pixel (0, 0) is the centre
/// of the top-left pixel, so the image spans [-0.5, w - 0.5] × [-0.5, h - 0.5].
#[derive(Clone, Copy, Debug)]
struct Intrinsics {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

impl Intrinsics {
    /// Image pixel → unit ray in the camera frame: `normalize(K⁻¹ (u, v, 1))`.
    fn pixel_to_ray(&self, px: DVec2) -> DVec3 {
        DVec3::new((px.x - self.cx) / self.fx, (px.y - self.cy) / self.fy, 1.0).normalize()
    }

    /// Camera-frame point (z > 0) → image pixel: `K (x/z, y/z, 1)`.
    #[cfg_attr(not(test), allow(dead_code))]
    fn project(&self, p: DVec3) -> DVec2 {
        DVec2::new(self.fx * p.x / p.z + self.cx, self.fy * p.y / p.z + self.cy)
    }
}

/// Image corners (outer edges of the corner pixels), clockwise from the top left.
fn image_corners() -> [DVec2; 4] {
    let (w, h) = (WIDTH as f64, HEIGHT as f64);
    [DVec2::new(-0.5, -0.5), DVec2::new(w - 0.5, -0.5), DVec2::new(w - 0.5, h - 0.5), DVec2::new(-0.5, h - 0.5)]
}
const CORNER_NAMES: [&str; 4] = ["top-left", "top-right", "bottom-right", "bottom-left"];

/// Rotation camera → ENU (columns: the camera's x, y, z axes in east-north-up) for the angle
/// convention of the constants (degrees):
///
/// ```text
/// R_enu_cam = R_enu_ned · Rz(yaw) · Ry(pitch) · Rx(roll) · R_body_cam
/// ```
/// * `R_body_cam`: the camera in the body (x forward, y right, z down) of an aircraft, looking
///   out of its belly: camera x = body right, camera y = body backward, camera z = body down
///   (the renderer's `nadir` mount);
/// * `Rz(yaw) · Ry(pitch) · Rx(roll)`: body → NED (north, east, down), aerospace Z-Y-X angles;
/// * `R_enu_ned`: swaps north/east and flips down to up.
///
/// At (0, 0, 0): camera x = east, camera y = south, camera z = down, i.e. a nadir image with
/// north up and east right.
fn r_enu_cam(yaw_deg: f64, pitch_deg: f64, roll_deg: f64) -> DMat3 {
    let r_body_cam = DMat3::from_cols(DVec3::Y, -DVec3::X, DVec3::Z);
    let r_ned_body =
        DMat3::from_rotation_z(yaw_deg.to_radians()) * DMat3::from_rotation_y(pitch_deg.to_radians()) * DMat3::from_rotation_x(roll_deg.to_radians());
    let r_enu_ned = DMat3::from_cols(DVec3::Y, DVec3::X, -DVec3::Z);
    r_enu_ned * r_ned_body * r_body_cam
}

/// A camera's pose: its position (ECEF), the ENU frame at that position, and its
/// rotation into ENU and ECEF.
struct CameraPose {
    /// camera centre in ECEF (m)
    ecef: DVec3,
    /// local east-north-up frame with its origin at the camera
    enu: LocalFrame,
    /// camera → ENU
    r_enu_cam: DMat3,
    /// camera → ECEF = (ENU → ECEF) · (camera → ENU)
    r_ecef_cam: DMat3,
}

impl CameraPose {
    fn new(position: Geodetic, r_enu_cam: DMat3, ell: Ellipsoid) -> Self {
        let enu = LocalFrame::new(position, LocalConvention::Enu, ell);
        CameraPose { ecef: enu.origin_ecef, enu, r_enu_cam, r_ecef_cam: enu.rot_local2ecef() * r_enu_cam }
    }

    /// A camera-frame direction → ECEF direction.
    fn ray_ecef(&self, d_cam: DVec3) -> DVec3 {
        self.r_ecef_cam * d_cam
    }

    /// A camera-frame point → ECEF point.
    fn cam_to_ecef(&self, p_cam: DVec3) -> DVec3 {
        self.ecef + self.r_ecef_cam * p_cam
    }

    /// (position difference in m, largest rotation-matrix element difference) to the pose the
    /// renderer reports for a frame.
    fn difference_to(&self, f: &Frame) -> (f64, f64) {
        let r = DMat3::from_cols_array_2d(&f.r_ecef_cam).transpose(); // (rows given)
        let dr = (0..3).map(|i| (r.col(i) - self.r_ecef_cam.col(i)).abs().max_element()).fold(0.0, f64::max);
        ((DVec3::from_array(f.position_ecef) - self.ecef).length(), dr)
    }
}

// ============================================================================================
// Ray ∩ terrain
// ============================================================================================

/// Elevation (DSM, metres above the ellipsoid) from the elevation tiles of one zoom, bilinear
/// between pixel centres. Tiles are fetched from the world (generated if missing) and kept.
struct ElevationSampler<'a> {
    world: &'a World,
    z: u8,
    tiles: HashMap<(u32, u32), Vec<f32>>,
}

impl<'a> ElevationSampler<'a> {
    fn new(world: &'a World, z: u8) -> Self {
        ElevationSampler { world, z, tiles: HashMap::new() }
    }

    /// Elevation of global pixel (gx, gy) at zoom `z` (tile gx / 256, row-major inside).
    fn pixel(&mut self, gx: i64, gy: i64) -> Result<f32> {
        let n = 1i64 << self.z;
        let ts = TILE_PX as i64;
        let (gx, gy) = (gx.rem_euclid(n * ts), gy.clamp(0, n * ts - 1));
        let key = ((gx / ts) as u32, (gy / ts) as u32);
        if !self.tiles.contains_key(&key) {
            let bytes = self.world.tile(self.z as u32, key.0, key.1, Layer::Elevation)?;
            self.tiles.insert(key, bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect());
        }
        Ok(self.tiles[&key][(gy % ts * ts + gx % ts) as usize])
    }

    /// East-west size (m) of an elevation pixel at latitude `lat` (radians): the equator's
    /// circumference over the pixels around it, times cos(lat).
    fn pixel_size(&self, lat: f64, ell: &Ellipsoid) -> f64 {
        2.0 * std::f64::consts::PI * ell.a * lat.cos() / (TILE_PX as f64 * (1u64 << self.z) as f64)
    }

    /// Elevation at (lat, lon): fractional tile → global pixel → bilinear between pixel centres
    /// (pixel (i, j) covers [i, i+1) × [j, j+1), its value sits at (i + ½, j + ½)).
    fn height_at(&mut self, g: Geodetic) -> Result<f64> {
        let t = TileXY::from_geodetic(g, self.z);
        let p = DVec2::new(t.x, t.y) * TILE_PX as f64 - 0.5;
        let (i, j) = (p.x.floor() as i64, p.y.floor() as i64);
        let (fx, fy) = (p.x - i as f64, p.y - j as f64);
        let (a, b) = (self.pixel(i, j)? as f64, self.pixel(i + 1, j)? as f64);
        let (c, d) = (self.pixel(i, j + 1)? as f64, self.pixel(i + 1, j + 1)? as f64);
        Ok((a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy)
    }
}

/// A ray / terrain intersection: the ground point, and how it was found.
struct TerrainHit {
    point: Geodetic,
    /// steps marched along the ray before it went below the terrain
    steps: usize,
    /// the step that crossed the ground: (distance along the ray in m, ray height minus terrain
    /// height in m) before and after
    bracket: [(f64, f64); 2],
    /// bisections of that step, and the final ray height minus terrain height (m)
    bisections: usize,
    residual: f64,
}

/// Intersect the ray `origin + t·dir` (ECEF, `dir` a unit vector, t ≥ 0 in metres) with the
/// terrain `height(lat, lon)` (metres above the ellipsoid): the first point where the ray's
/// geodetic height drops to the terrain's.
///
/// 1. **March**: f(t) = (ray height − terrain height) at t = 0, step, 2·step, … until f ≤ 0.
///    The step is a fraction of an elevation pixel, so the bilinear terrain cannot rise above
///    the ray and fall again between two samples unnoticed (steep or not).
/// 2. **Bisect** the last step [t₀, t₁] (f(t₀) > 0 ≥ f(t₁)) until |f| < `RAY_TOLERANCE_M`.
fn intersect_terrain(origin: DVec3, dir: DVec3, step: f64, ell: &Ellipsoid, mut height: impl FnMut(Geodetic) -> Result<f64>) -> Result<TerrainHit> {
    // ray height minus terrain height at distance t, and the ray's point there
    let mut f = |t: f64| -> Result<(f64, Geodetic)> {
        let g = ecef2geodetic(origin + t * dir, ell);
        Ok((g.h - height(g)?, g))
    };
    let (f0, _) = f(0.0)?;
    ensure!(f0 > 0.0, "the camera is {:.1} m below the terrain", -f0);
    // 1. march until the ray is at or below the terrain
    let (mut t0, mut f_t0) = (0.0, f0);
    let mut steps = 0;
    let (mut t1, f_t1) = loop {
        steps += 1;
        let t = steps as f64 * step;
        if t > RAY_MAX_RANGE_M {
            bail!("a corner ray does not reach the ground within {RAY_MAX_RANGE_M} m (it points above the horizon)");
        }
        let (ft, _) = f(t)?;
        if ft <= 0.0 {
            break (t, ft);
        }
        (t0, f_t0) = (t, ft);
    };
    let bracket = [(t0, f_t0), (t1, f_t1)];
    // 2. bisect [t0, t1], keeping f(t0) > 0 >= f(t1)
    let (mut fm, mut point) = f(t1)?;
    let mut bisections = 0;
    while fm.abs() >= RAY_TOLERANCE_M && bisections < 64 {
        bisections += 1;
        let tm = 0.5 * (t0 + t1);
        (fm, point) = f(tm)?;
        if fm > 0.0 {
            t0 = tm;
        } else {
            t1 = tm;
        }
    }
    Ok(TerrainHit { point, steps, bracket, bisections, residual: fm })
}

// ============================================================================================
// XYZ tiles and the stitched map
// ============================================================================================

/// A fractional Web-Mercator XYZ tile coordinate at zoom `z`: `(x, y) = (u, v) · 2^z` with
/// `u = (lon + π) / 2π` (east), `v = ½ − asinh(tan lat) / 2π` (south). Its integer part is
/// the XYZ tile (`tiles_xyz`, as `z/x/y` in the tile store), the fraction × 256 the pixel
/// inside the tile.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TileXY {
    x: f64,
    y: f64,
    z: u8,
}

impl TileXY {
    fn from_geodetic(g: Geodetic, z: u8) -> Self {
        let uv = latlon_to_uv(g.lat, g.lon) * (1u64 << z) as f64;
        TileXY { x: uv.x, y: uv.y, z }
    }

    /// (lat, lon) in radians.
    #[cfg_attr(not(test), allow(dead_code))]
    fn to_latlon(self) -> (f64, f64) {
        uv_to_latlon(DVec2::new(self.x, self.y) / (1u64 << self.z) as f64)
    }

    /// The integer XYZ tile containing this point.
    fn tile(self) -> (u32, u32) {
        (self.x.floor() as u32, self.y.floor() as u32)
    }
}

/// A block of whole tiles `x0 .. x0 + nx`, `y0 .. y0 + ny` at zoom `z`, stitched into one image
/// with tile (x0, y0) at the top left. Stitched-map pixel (i, j) covers [i, i+1) × [j, j+1)
/// (its centre is at (i + ½, j + ½)), as pixels inside a tile.
#[derive(Clone, Copy, Debug)]
struct MapGrid {
    z: u8,
    x0: u32,
    y0: u32,
    nx: u32,
    ny: u32,
}

impl MapGrid {
    /// The smallest block of tiles holding every point.
    fn covering(points: &[TileXY], z: u8) -> Self {
        let (xs, ys): (Vec<u32>, Vec<u32>) = points.iter().map(|p| p.tile()).unzip();
        let (x0, x1) = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap());
        let (y0, y1) = (*ys.iter().min().unwrap(), *ys.iter().max().unwrap());
        MapGrid { z, x0, y0, nx: x1 - x0 + 1, ny: y1 - y0 + 1 }
    }

    fn width(&self) -> u32 {
        self.nx * TILE_PX as u32
    }

    fn height(&self) -> u32 {
        self.ny * TILE_PX as u32
    }

    /// Fractional tile → stitched-map pixel: `((x − x0) · 256, (y − y0) · 256)`.
    fn pixel_of(&self, t: TileXY) -> DVec2 {
        assert_eq!(t.z, self.z, "tile coordinate of another zoom");
        DVec2::new(t.x - self.x0 as f64, t.y - self.y0 as f64) * TILE_PX as f64
    }

    /// Stitched-map pixel → fractional tile (inverse of [`MapGrid::pixel_of`]).
    #[cfg_attr(not(test), allow(dead_code))]
    fn tile_of(&self, px: DVec2) -> TileXY {
        TileXY { x: self.x0 as f64 + px.x / TILE_PX as f64, y: self.y0 as f64 + px.y / TILE_PX as f64, z: self.z }
    }

    /// The tiles, row by row, as `[z, x, y]`.
    fn tile_ids(&self) -> Vec<[u32; 3]> {
        (0..self.ny).flat_map(|dy| (0..self.nx).map(move |dx| [self.z as u32, self.x0 + dx, self.y0 + dy])).collect()
    }

    /// Fetch a layer of every tile (generated if missing) and lay them out: (pixels, channels).
    fn stitch(&self, world: &World, layer: Layer, bytes_per_px: usize) -> Result<Vec<u8>> {
        let ids = self.tile_ids();
        let tile_bytes = TILE_PX * TILE_PX * bytes_per_px;
        let mut raw = vec![0u8; ids.len() * tile_bytes];
        world.tiles_into(&ids, layer, &mut raw)?;
        let row_bytes = self.width() as usize * bytes_per_px;
        let mut out = vec![0u8; row_bytes * self.height() as usize];
        for (k, tile) in raw.chunks_exact(tile_bytes).enumerate() {
            let (dx, dy) = (k % self.nx as usize, k / self.nx as usize);
            for row in 0..TILE_PX {
                let dst = (dy * TILE_PX + row) * row_bytes + dx * TILE_PX * bytes_per_px;
                out[dst..dst + TILE_PX * bytes_per_px].copy_from_slice(&tile[row * TILE_PX * bytes_per_px..][..TILE_PX * bytes_per_px]);
            }
        }
        Ok(out)
    }

    /// The albedo layer (unlit surface colour, sRGB).
    fn stitch_albedo(&self, world: &World) -> Result<RgbImage> {
        let px = self.stitch(world, Layer::Albedo, 3)?;
        RgbImage::from_raw(self.width(), self.height(), px).context("albedo image size")
    }

    /// The elevation layer (DSM: ground, trees, buildings; metres above the ellipsoid).
    fn stitch_elevation(&self, world: &World) -> Result<Vec<f32>> {
        let px = self.stitch(world, Layer::Elevation, 4)?;
        Ok(px.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }
}

// ============================================================================================
// Homography
// ============================================================================================

/// A plane-to-plane projective map `p' ~ H · (x, y, 1)`.
#[derive(Clone, Copy, Debug)]
struct Homography(DMat3);

impl Homography {
    /// The homography mapping `src[i]` to `dst[i]` (4 point pairs, no three collinear), by the
    /// direct linear transform: each pair gives two equations linear in the 9 entries of H,
    ///
    /// ```text
    /// h11 x + h12 y + h13 − x' (h31 x + h32 y + h33) = 0
    /// h21 x + h22 y + h23 − y' (h31 x + h32 y + h33) = 0
    /// ```
    /// and with h33 = 1 the 4 pairs give a square 8 × 8 system. Both point sets are first
    /// normalised (centroid at 0, mean distance √2; Hartley) for a well-conditioned solve.
    fn from_4_points(src: &[DVec2], dst: &[DVec2]) -> Result<Self> {
        ensure!(src.len() == 4 && dst.len() == 4, "4 point pairs expected");
        let (ts, s) = normalize_points(src);
        let (td, d) = normalize_points(dst);
        let mut a = [[0.0; 8]; 8];
        let mut b = [0.0; 8];
        for i in 0..4 {
            let (x, y, u, v) = (s[i].x, s[i].y, d[i].x, d[i].y);
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
            b[2 * i] = u;
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];
            b[2 * i + 1] = v;
        }
        let h = solve8(a, b).context("degenerate corners (three on a line?)")?;
        let hn = DMat3::from_cols_array(&[h[0], h[3], h[6], h[1], h[4], h[7], h[2], h[5], 1.0]);
        Ok(Homography(td.inverse() * hn * ts))
    }

    fn apply(&self, p: DVec2) -> DVec2 {
        let q = self.0 * p.extend(1.0);
        q.truncate() / q.z
    }

    fn inverse(&self) -> Homography {
        Homography(self.0.inverse())
    }
}

/// Hartley normalisation: the similarity T moving the centroid to 0 and the mean distance to √2,
/// and the moved points.
fn normalize_points(p: &[DVec2]) -> (DMat3, Vec<DVec2>) {
    let c = p.iter().copied().sum::<DVec2>() / p.len() as f64;
    let mean = p.iter().map(|q| (*q - c).length()).sum::<f64>() / p.len() as f64;
    let s = std::f64::consts::SQRT_2 / mean.max(1e-300);
    let t = DMat3::from_cols_array(&[s, 0.0, 0.0, 0.0, s, 0.0, -s * c.x, -s * c.y, 1.0]);
    (t, p.iter().map(|q| (*q - c) * s).collect())
}

/// Solve the 8 × 8 system `a · x = b` (Gaussian elimination with partial pivoting).
fn solve8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let piv = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..8 {
            let f = a[row][col] / a[col][col];
            let pivot = a[col];
            for (x, p) in a[row][col..].iter_mut().zip(&pivot[col..]) {
                *x -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 8];
    for row in (0..8).rev() {
        let s: f64 = (row + 1..8).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    Some(x)
}

/// The map seen through the homography (map pixels → image pixels): each image pixel centre
/// (u, v) takes the map at H⁻¹ (u, v) (bilinear; black outside the map).
fn warp_into_camera(map: &RgbImage, h: &Homography, w: u32, hgt: u32) -> RgbImage {
    let inv = h.inverse();
    RgbImage::from_fn(w, hgt, |u, v| {
        let m = inv.apply(DVec2::new(u as f64, v as f64));
        bilinear(map, m - 0.5).unwrap_or(Rgb([0, 0, 0]))
    })
}

/// Bilinear sample at continuous pixel-index coordinates (pixel (i, j) at (i, j)).
fn bilinear(img: &RgbImage, p: DVec2) -> Option<Rgb<u8>> {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (i, j) = (p.x.floor() as i64, p.y.floor() as i64);
    if i < -1 || j < -1 || i >= w || j >= h {
        return None;
    }
    let (fx, fy) = (p.x - i as f64, p.y - j as f64);
    let at = |x: i64, y: i64| img.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32).0;
    let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
    let mut out = [0u8; 3];
    for k in 0..3 {
        let top = a[k] as f64 * (1.0 - fx) + b[k] as f64 * fx;
        let bottom = c[k] as f64 * (1.0 - fx) + d[k] as f64 * fx;
        out[k] = (top * (1.0 - fy) + bottom * fy).round() as u8;
    }
    Some(Rgb(out))
}

// ============================================================================================
// Check: the rendered depth gives every pixel's true ground point; the homography should
// map that point's map pixel back to the pixel (exact for flat ground)
// ============================================================================================

fn check_alignment(frame: &Frame, cam: &CameraPose, intr: &Intrinsics, grid: &MapGrid, h: &Homography, depth_is_range: bool, ell: &Ellipsoid) -> Result<()> {
    let depth = frame.depth.as_ref().context("the oblique frame has no depth")?;
    let mut err: Vec<(f64, f64, DVec2)> = Vec::new(); // (pixel error, ground height, pixel)
    for gy in 0..=12 {
        for gx in 0..=16 {
            let px = DVec2::new(gx as f64 * (WIDTH - 1) as f64 / 16.0, gy as f64 * (HEIGHT - 1) as f64 / 12.0).round();
            let z = depth[px.y as usize * WIDTH as usize + px.x as usize] as f64;
            if !z.is_finite() {
                continue; // sky
            }
            // pixel + depth → camera-frame point → ECEF → geodetic → tile → map pixel → H → pixel
            let ray = intr.pixel_to_ray(px);
            let p_cam = if depth_is_range { ray * z } else { ray * (z / ray.z) };
            let g = ecef2geodetic(cam.cam_to_ecef(p_cam), ell);
            let back = h.apply(grid.pixel_of(TileXY::from_geodetic(g, grid.z)));
            err.push(((back - px).length(), g.h, px));
        }
    }
    ensure!(!err.is_empty(), "no terrain in the oblique image");
    err.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (hmin, hmax) = err.iter().fold((f64::MAX, f64::MIN), |(a, b), e| (a.min(e.1), b.max(e.1)));
    let worst = err.last().unwrap();
    println!(
        "{} grid pixels: homography vs rendered ground point: median {:.1} px, 90% {:.1} px, max {:.1} px (at ({:.0}, {:.0}))",
        err.len(),
        err[err.len() / 2].0,
        err[err.len() * 9 / 10].0,
        worst.0,
        worst.2.x,
        worst.2.y
    );
    println!("(the plane through the 4 corners stands in for terrain of {hmin:.0} .. {hmax:.0} m: relief, trees and buildings shift points)");
    Ok(())
}

// ============================================================================================
// Images
// ============================================================================================

fn frame_image(f: &Frame) -> Result<RgbImage> {
    RgbImage::from_raw(f.width, f.height, f.rgb.clone().context("no rgb image")?).context("rgb image size")
}

fn save(img: &RgbImage, dir: &Path, name: &str) -> Result<()> {
    let p = dir.join(name);
    img.save(&p).with_context(|| format!("writing {}", p.display()))?;
    println!("  wrote {} ({} x {})", p.display(), img.width(), img.height());
    Ok(())
}

/// Elevation → grey: `lo` black, `hi` white.
fn grayscale(v: &[f32], w: u32, h: u32, lo: f32, hi: f32) -> RgbImage {
    let s = 255.0 / (hi - lo).max(1e-6);
    RgbImage::from_fn(w, h, |x, y| {
        let g = ((v[(y * w + x) as usize] - lo) * s).round().clamp(0.0, 255.0) as u8;
        Rgb([g, g, g])
    })
}

/// A closed polygon (stitched-map pixel coordinates) drawn `width` pixels wide: every pixel whose
/// centre is within width/2 of an edge.
fn draw_polygon(img: &mut RgbImage, pts: &[DVec2], color: Rgb<u8>, width: f64) {
    let r = 0.5 * width;
    for k in 0..pts.len() {
        let (a, b) = (pts[k], pts[(k + 1) % pts.len()]);
        let (lo, hi) = (a.min(b) - r - 1.0, a.max(b) + r + 1.0);
        for y in lo.y.max(0.0) as u32..(hi.y.max(0.0) as u32).min(img.height()) {
            for x in lo.x.max(0.0) as u32..(hi.x.max(0.0) as u32).min(img.width()) {
                let c = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
                let t = ((c - a).dot(b - a) / (b - a).length_squared().max(1e-12)).clamp(0.0, 1.0);
                if (a + t * (b - a) - c).length() <= r {
                    img.put_pixel(x, y, color);
                }
            }
        }
    }
}

/// Squares of `a` and `b` alternating (`n` pixels): edges continue across the squares where the
/// two images agree.
fn checkerboard(a: &RgbImage, b: &RgbImage, n: u32) -> RgbImage {
    RgbImage::from_fn(a.width(), a.height(), |x, y| if (x / n + y / n).is_multiple_of(2) { *a.get_pixel(x, y) } else { *b.get_pixel(x, y) })
}

// ============================================================================================
// Printing
// ============================================================================================

fn fmt_geo(g: Geodetic) -> String {
    format!("lat {:.6}°, lon {:.6}°, h {:.2} m", g.lat_deg(), g.lon_deg(), g.h)
}

fn fmt_v(v: DVec3) -> String {
    format!("({:.2}, {:.2}, {:.2}) m", v.x, v.y, v.z)
}

fn fmt_dir(v: DVec3) -> String {
    format!("({:+.3}, {:+.3}, {:+.3})", v.x, v.y, v.z)
}

fn fmt_h(h: &Homography) -> String {
    let m = h.0 * (1.0 / h.0.z_axis.z); // scaled to h33 = 1
    let r = |i: usize| format!("[{:+.5e} {:+.5e} {:+.5e}]", m.col(0)[i], m.col(1)[i], m.col(2)[i]);
    format!("[{}, {}, {}]", r(0), r(1), r(2))
}

// ============================================================================================
// Tests
// ============================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_maps_the_4_points_both_ways() {
        let src = [DVec2::new(812.3, 410.9), DVec2::new(1650.2, 302.7), DVec2::new(1201.4, 1188.0), DVec2::new(905.6, 1030.5)];
        let dst = image_corners();
        let h = Homography::from_4_points(&src, &dst).unwrap();
        let inv = h.inverse();
        for (s, d) in src.iter().zip(dst) {
            assert!((h.apply(*s) - d).length() < 1e-8, "{s} -> {} != {d}", h.apply(*s));
            assert!((inv.apply(d) - *s).length() < 1e-8);
        }
        // an affine map is recovered exactly, also away from the 4 points
        let aff = |p: DVec2| DVec2::new(2.0 * p.x - 0.5 * p.y + 7.0, 0.3 * p.x + 1.5 * p.y - 3.0);
        let h = Homography::from_4_points(&src, &src.map(aff)).unwrap();
        let p = DVec2::new(1000.0, 700.0);
        assert!((h.apply(p) - aff(p)).length() < 1e-8);
        // collinear points are refused
        let line = [DVec2::new(0.0, 0.0), DVec2::new(1.0, 1.0), DVec2::new(2.0, 2.0), DVec2::new(3.0, 3.0)];
        assert!(Homography::from_4_points(&line, &dst).is_err());
    }

    #[test]
    fn geodetic_ecef_enu_round_trip() {
        let ell = Ellipsoid::WGS84;
        let origin = Geodetic::from_deg(LAT, LON, ALT);
        let enu = LocalFrame::new(origin, LocalConvention::Enu, ell);
        let g = Geodetic::from_deg(LAT + 0.01, LON - 0.02, 612.5);
        let ecef = geodetic2ecef(g, &ell);
        let back = ecef2geodetic(ecef, &ell);
        assert!((back.lat - g.lat).abs() < 1e-12 && (back.lon - g.lon).abs() < 1e-12 && (back.h - g.h).abs() < 1e-6);
        let l = enu.ecef_to_local(ecef);
        assert!(l.y > 1000.0 && l.x < -1000.0 && l.z < 0.0, "north-west and below the origin: {l}");
        assert!((enu.local_to_ecef(l) - ecef).length() < 1e-6);
    }

    #[test]
    fn tiles_and_map_pixels() {
        let g = Geodetic::from_deg(LAT, LON, 0.0);
        let t = TileXY::from_geodetic(g, LOD);
        let (lat, lon) = t.to_latlon();
        assert!((lat - g.lat).abs() < 1e-12 && (lon - g.lon).abs() < 1e-12);
        // the integer tile is the geodesy crate's
        let id = geodesy::tile_for_latlon(g.lat, g.lon, LOD);
        assert_eq!(t.tile(), (id.x, id.y));
        // map pixel (0, 0) is the north-west corner of the first tile; pixel_of / tile_of invert
        let far = TileXY { x: t.x + 2.5, y: t.y + 1.2, z: LOD };
        let grid = MapGrid::covering(&[t, far], LOD);
        assert_eq!((grid.x0, grid.y0), t.tile());
        assert_eq!((grid.x0 + grid.nx - 1, grid.y0 + grid.ny - 1), far.tile());
        assert_eq!(grid.pixel_of(TileXY { x: grid.x0 as f64, y: grid.y0 as f64, z: LOD }), DVec2::ZERO);
        let p = grid.pixel_of(t);
        assert!((grid.tile_of(p).x - t.x).abs() < 1e-9 && (grid.tile_of(p).y - t.y).abs() < 1e-9);
        assert!((p.x - t.x.fract() * 256.0).abs() < 1e-6);
    }

    #[test]
    fn nadir_camera_is_north_up() {
        let r = r_enu_cam(0.0, 0.0, 0.0);
        assert!(r.x_axis.abs_diff_eq(DVec3::X, 1e-15)); // image right = east
        assert!(r.y_axis.abs_diff_eq(-DVec3::Y, 1e-15)); // image down = south
        assert!(r.z_axis.abs_diff_eq(-DVec3::Z, 1e-15)); // optical axis = down
                                                         // pitch tilts the view towards the heading: yaw 90 → east, 50° from nadir
        let r = r_enu_cam(90.0, 50.0, 0.0);
        let z = r.z_axis;
        assert!((z.x - 50f64.to_radians().sin()).abs() < 1e-12 && z.y.abs() < 1e-12 && (z.z + 50f64.to_radians().cos()).abs() < 1e-12);
        // a proper rotation
        assert!((r.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pixel_ray_pixel_round_trip() {
        let k = Intrinsics { fx: FOCAL_PX, fy: FOCAL_PX, cx: (WIDTH as f64 - 1.0) / 2.0, cy: (HEIGHT as f64 - 1.0) / 2.0 };
        for px in image_corners() {
            let ray = k.pixel_to_ray(px);
            assert!((ray.length() - 1.0).abs() < 1e-12);
            assert!((k.project(ray * 1234.0) - px).length() < 1e-9);
        }
    }

    #[test]
    fn ray_finds_the_first_crossing_of_steep_terrain() {
        // a sawtooth of 400 m ridges every 300 m: the first ridge the ray meets is the hit, even
        // where the ray passes over lower terrain behind it
        let ell = Ellipsoid::WGS84;
        let cam = CameraPose::new(Geodetic::from_deg(LAT, LON, 2000.0), r_enu_cam(YAW, PITCH, ROLL), ell);
        let terrain = |g: Geodetic| -> Result<f64> {
            let x = (g.lon.to_degrees() - LON) * 111_000.0 * LAT.to_radians().cos() + (g.lat.to_degrees() - LAT) * 111_000.0;
            Ok(1000.0 + 400.0 * (x / 300.0).rem_euclid(1.0))
        };
        let dir = cam.ray_ecef(DVec3::new(0.0, 0.0, 1.0));
        let hit = intersect_terrain(cam.ecef, dir, 2.0, &ell, terrain).unwrap();
        assert!(hit.residual.abs() < RAY_TOLERANCE_M, "{}", hit.residual);
        assert!((hit.point.h - terrain(hit.point).unwrap()).abs() < RAY_TOLERANCE_M);
        // nothing earlier on the ray is below the terrain (fine sampling)
        let t_hit = (geodetic2ecef(hit.point, &ell) - cam.ecef).length();
        let mut t = 0.0;
        while t < t_hit - 0.05 {
            let g = ecef2geodetic(cam.ecef + t * dir, &ell);
            assert!(g.h > terrain(g).unwrap(), "the ray is below the terrain at {t} m, before the hit at {t_hit} m");
            t += 0.05;
        }
        // a camera below the terrain is an error
        let low = CameraPose::new(Geodetic::from_deg(LAT, LON, 900.0), r_enu_cam(YAW, PITCH, ROLL), ell);
        assert!(intersect_terrain(low.ecef, dir, 2.0, &ell, terrain).is_err());
    }
}
