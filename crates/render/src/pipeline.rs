//! High-level operations used by the CLI: plan tiles, generate tiles, render a sequence
//! (every camera's frame modalities, body poses, IMU), simulate event cameras.

use crate::cache::TileCache;
use crate::camera::CameraModel;
use crate::lod::{LodParams, PlanOracle, Selector, TileOracle};
use crate::output::{self, BodySample, CameraWriter, Frame, PngWriter};
use crate::raster::Renderer;
use crate::scenario::{CameraSpec, DepthKind, Scenario};
use crate::sensor::Sensor;
use crate::trajectory::{self, CamPose, Pose};
use anyhow::{bail, Context, Result};
use geodesy::tiles::TileId;
use geodesy::Ellipsoid;
use glam::{DVec2, DVec3};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use terragen::Generator;
use tilestore::{Layer, StoreMeta, TileStore};

/// Time window of the sequence in trajectory time: [t0, t1]. t0 is the zero of the sequence
/// clock (all timestamps in the output are µs since t0).
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub t0: f64,
    pub t1: f64,
}

impl Window {
    pub fn new(scn: &Scenario, poses: &[Pose]) -> Result<Self> {
        let (Some(first), Some(last)) = (poses.first(), poses.last()) else { bail!("empty trajectory") };
        let o = &scn.output;
        let t0 = first.t + o.start.max(0.0);
        let t1 = o.end.map(|e| first.t + e).unwrap_or(last.t).min(last.t);
        if t1 < t0 {
            bail!("output window [{}, {:?}] is outside the trajectory ({:.1} s long)", o.start, o.end, last.t - first.t);
        }
        Ok(Window { t0, t1 })
    }

    pub fn duration(&self) -> f64 {
        self.t1 - self.t0
    }
}

/// `t0 + offset + k / rate` within the window, at most `max` samples.
fn uniform_times(win: Window, offset: f64, rate: f64, max: Option<usize>) -> Vec<f64> {
    let dt = 1.0 / rate.max(1e-6);
    let mut v = Vec::new();
    loop {
        let t = win.t0 + offset + v.len() as f64 * dt;
        if t > win.t1 + 1e-9 || max.is_some_and(|m| v.len() >= m) {
            break;
        }
        v.push(t);
    }
    v
}

/// Frame times of a camera (empty without frame modalities).
pub fn frame_times(scn: &Scenario, spec: &CameraSpec, win: Window) -> Vec<f64> {
    if !spec.has_frames() {
        return vec![];
    }
    uniform_times(win, spec.time_offset, spec.frame_rate, scn.output.max_frames)
}

/// Rough per-tile elevation range from the generator (pass A at a coarse grid) + margins.
pub fn generator_range_estimator(gen: &Generator) -> impl Fn(TileId) -> (f32, f32) + Sync + '_ {
    let cache: Mutex<HashMap<TileId, (f32, f32)>> = Mutex::new(HashMap::new());
    move |id: TileId| {
        if let Some(r) = cache.lock().get(&id) {
            return *r;
        }
        let b = id.bounds();
        let size = (b.lon_max - b.lon_min).abs() * gen.world.ell.a * ((b.lat_min + b.lat_max) * 0.5).cos();
        let gsd = (size / 4.0).max(30.0);
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for j in 0..4 {
            for i in 0..4 {
                let v = (j as f64 + 0.5) / 4.0;
                let y = id.y as f64 + v;
                let lat = ((std::f64::consts::PI * (1.0 - 2.0 * y / (1u64 << id.z) as f64)).sinh()).atan();
                let lon = b.lon_min + (b.lon_max - b.lon_min) * (i as f64 + 0.5) / 4.0;
                let t = gen.world.terrain(&terragen::world::Ctx::new(lat, lon, gsd, &gen.world.ell));
                let g = if t.water_kind != 0 { t.water.max(t.ground) } else { t.ground };
                lo = lo.min(g);
                hi = hi.max(g);
            }
        }
        let span = hi - lo;
        let r = ((lo - 40.0 - 0.3 * span) as f32, (hi + 60.0 + 0.3 * span + 0.02 * size) as f32);
        cache.lock().insert(id, r);
        r
    }
}

/// Tiles needed to render every camera of the sequence (with margins and all ancestors).
/// Frame cameras are planned at their frame times, event cameras every 1 / frame_rate.
pub fn plan(scn: &Scenario, poses: &[Pose], gen: Option<&Generator>) -> Result<BTreeSet<TileId>> {
    let ell = Ellipsoid::from_a_invf(scn.world.planet.a, scn.world.planet.inv_f);
    let win = Window::new(scn, poses)?;
    let params = LodParams {
        min_zoom: scn.tiles.min_zoom,
        max_zoom: scn.tiles.max_zoom,
        texel_px: scn.tiles.plan_texel_px,
        cone_margin: 0.08,
        ..Default::default()
    };
    let est = gen.map(generator_range_estimator);
    let oracle = match &est {
        Some(f) => PlanOracle { estimate: Some(f), fixed: (-100.0, 5000.0) },
        None => PlanOracle::fixed((-100.0, 5000.0)),
    };
    let every = scn.tiles.plan_every.max(1);
    let mut set: BTreeSet<TileId> = BTreeSet::new();
    for spec in &scn.cameras {
        let model = spec.intrinsics.build()?;
        let mut times = frame_times(scn, spec, win);
        if spec.events.is_some() {
            times.extend(uniform_times(win, 0.0, spec.frame_rate, None));
            times.sort_by(f64::total_cmp);
            times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        }
        for (k, t) in times.iter().enumerate() {
            if k % every != 0 && k + 1 != times.len() {
                continue;
            }
            let cam = trajectory::interpolate(poses, *t).camera(&spec.extrinsics, &ell);
            set.extend(Selector::new(&cam, model.as_ref(), ell, &params, &oracle).select().into_iter().map(|u| u.id));
        }
    }
    with_margin_and_ancestors(scn, &mut set);
    Ok(set)
}

/// Margin rings (`tiles.margin` neighbours at each zoom) and all ancestors of a tile set.
pub fn with_margin_and_ancestors(scn: &Scenario, set: &mut BTreeSet<TileId>) {
    let m = scn.tiles.margin as i32;
    let base: Vec<TileId> = set.iter().copied().collect();
    for id in base {
        for dy in -m..=m {
            for dx in -m..=m {
                if let Some(n) = id.neighbor(dx, dy) {
                    set.insert(n);
                }
            }
        }
    }
    let base: Vec<TileId> = set.iter().copied().collect();
    for id in base {
        let mut a = id;
        while let Some(p) = a.parent() {
            a = p;
            set.insert(a);
            if a.z <= scn.tiles.min_zoom.saturating_sub(2) {
                break;
            }
        }
    }
}

/// Dry-run oracle: what the renderer sees with lazy generation — every tile up to the maximum
/// zoom exists, elevation ranges are known for stored tiles only (the selector falls back to the
/// nearest stored ancestor's, exactly as when rendering).
struct DryRunOracle<'a> {
    store: &'a TileStore,
    max_zoom: u8,
}
impl TileOracle for DryRunOracle<'_> {
    fn range(&self, id: TileId) -> Option<(f32, f32)> {
        self.store.elev_range(id)
    }
    fn exists(&self, id: TileId) -> bool {
        id.z <= self.max_zoom || self.store.contains(id)
    }
    fn refine_unknown(&self) -> bool {
        false
    }
}

/// Tiles the renderer will select that the store lacks: a dry run of the renderer's own LOD
/// selection (its zoom limits and texel threshold) at every frame of every camera (event cameras
/// at 4× their frame rate), plus margins and ancestors. Run after generating a plan, it finds
/// the tiles that the plan's estimated elevation ranges missed; repeated until empty, the render
/// needs no lazy generation.
pub fn plan_missing(scn: &Scenario, poses: &[Pose], store: &TileStore) -> Result<BTreeSet<TileId>> {
    let ell = store.meta().ellipsoid();
    let win = Window::new(scn, poses)?;
    let max_zoom = scn.render.max_zoom.min(scn.tiles.max_zoom);
    let params = LodParams {
        min_zoom: scn.render.min_zoom.max(scn.tiles.min_zoom),
        max_zoom,
        texel_px: scn.render.texel_px,
        ..Default::default()
    };
    let oracle = DryRunOracle { store, max_zoom };
    let mut want: BTreeSet<TileId> = BTreeSet::new();
    for spec in &scn.cameras {
        let model = spec.intrinsics.build()?;
        let mut times = frame_times(scn, spec, win);
        if spec.events.is_some() {
            times.extend(uniform_times(win, 0.0, 4.0 * spec.frame_rate, None));
        }
        let sel: Vec<Vec<TileId>> = times
            .par_iter()
            .map(|t| {
                let cam = trajectory::interpolate(poses, *t).camera(&spec.extrinsics, &ell);
                Selector::new(&cam, model.as_ref(), ell, &params, &oracle).select().into_iter().map(|u| u.id).collect()
            })
            .collect();
        want.extend(sel.into_iter().flatten());
    }
    with_margin_and_ancestors(scn, &mut want);
    want.retain(|id| !store.contains(*id));
    Ok(want)
}

/// Generate tiles into the store (skipping existing ones unless `force`). Calls `progress(done, total)`.
pub fn generate(gen: &Generator, store: &TileStore, tiles: &[TileId], force: bool, progress: &(dyn Fn(usize, usize) + Sync)) -> Result<usize> {
    let todo: Vec<TileId> = tiles.iter().copied().filter(|t| force || !store.contains(*t)).collect();
    let total = todo.len();
    let batch = (rayon::current_num_threads() * 4).max(8);
    let mut done = 0;
    progress(0, total);
    for chunk in todo.chunks(batch) {
        let data: Vec<tilestore::TileData> = chunk.par_iter().map(|id| gen.tile(*id)).collect();
        store.write_tiles(&data)?;
        done += chunk.len();
        progress(done, total);
    }
    store.flush()?;
    Ok(total)
}

pub fn open_or_create_store(scn: &Scenario, gen: &Generator) -> Result<TileStore> {
    let meta = StoreMeta {
        ellipsoid_a: gen.world.ell.a,
        ellipsoid_b: gen.world.ell.b,
        generator_config: gen.config().to_yaml(),
        seed: gen.config().seed,
        layers: Layer::ALL.to_vec(),
    };
    Ok(TileStore::open_or_create(&scn.tiles.file, meta)?)
}

/// Tile cache with the layers the shading mode needs (~40% less memory per cached tile), lazily
/// generating missing tiles when `tiles.lazy`.
pub fn tile_cache(scn: &Scenario, store: Arc<TileStore>, gen: Option<Arc<Generator>>) -> Arc<TileCache> {
    let mut layers = vec![Layer::Elevation, Layer::Landcover, Layer::Emission];
    match scn.render.shading {
        crate::raster::Shading::Relit => layers.extend([Layer::Albedo, Layer::Normal]),
        crate::raster::Shading::Satellite => layers.push(Layer::Rgb),
    }
    let mut cache = TileCache::new(store, layers, scn.tiles.cache_tiles);
    if scn.tiles.lazy {
        if let Some(g) = gen {
            cache = cache.with_generator(g, scn.tiles.max_zoom, true);
        }
    }
    Arc::new(cache)
}

pub fn renderer(scn: &Scenario, model: Arc<dyn CameraModel>, supersample: u32, ell: Ellipsoid, cache: Arc<TileCache>) -> Renderer {
    let mut rs = scn.render.clone();
    rs.supersample = supersample.max(1);
    rs.min_zoom = rs.min_zoom.max(scn.tiles.min_zoom);
    rs.max_zoom = rs.max_zoom.min(scn.tiles.max_zoom);
    Renderer::new(model, rs, ell, cache)
}

/// Forward flow from frame A (points) to frame B (pose + z-depth). Returns (flow, valid).
fn compute_flow(points: &[Option<DVec3>], w: usize, h: usize, cam_b: &CamPose, depth_b: &[f32], model: &dyn CameraModel) -> (Vec<f32>, Vec<u8>) {
    let mut flow = vec![0f32; w * h * 2];
    let mut valid = vec![0u8; w * h];
    flow.par_chunks_mut(w * 2).zip(valid.par_chunks_mut(w)).enumerate().for_each(|(y, (fr, vr))| {
        for x in 0..w {
            let Some(p) = points[y * w + x] else { continue };
            let pc = cam_b.world_to_cam(p);
            let Some(px) = model.project(pc) else { continue };
            fr[2 * x] = (px.x - x as f64) as f32;
            fr[2 * x + 1] = (px.y - y as f64) as f32;
            let (u, v) = (px.x.round(), px.y.round());
            if u >= 0.0 && v >= 0.0 && (u as usize) < w && (v as usize) < h {
                // visible if the target pixel's depth matches; the tolerance covers the rounding
                // to the nearest pixel (local depth gradient) plus a small relative margin, so
                // occluders (buildings, trees) of a few metres are detected even at 1 km
                let (u, v) = (u as usize, v as usize);
                let d = depth_b[v * w + u] as f64;
                let mut grad: f64 = 0.0;
                for (du, dv) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (uu, vv) = (u as i64 + du, v as i64 + dv);
                    if uu >= 0 && vv >= 0 && (uu as usize) < w && (vv as usize) < h {
                        let dn = depth_b[vv as usize * w + uu as usize] as f64;
                        if dn.is_finite() {
                            grad = grad.max((dn - d).abs());
                        }
                    }
                }
                if (pc.z - d).abs() < 0.5 + 0.003 * pc.z.abs() + 0.6 * grad.min(0.05 * pc.z.abs()) {
                    vr[x] = 1;
                }
            }
        }
    });
    (flow, valid)
}

/// Motion blur: re-project the frame's points with sub-frame poses across the exposure window
/// and integrate the radiance along the image-space paths.
#[allow(clippy::too_many_arguments)]
fn motion_blur(sensor: &Sensor, radiance: Vec<f32>, points: &[Option<DVec3>], sample_offset: f64, cam: &CamPose, poses: &[Pose], spec: &CameraSpec, t: f64, exposure: f64, ell: &Ellipsoid, model: &dyn CameraModel) -> Vec<f32> {
    let Some(rgb) = &spec.rgb else { return radiance };
    let mb = &rgb.sensor.motion_blur;
    if !mb.enabled || exposure <= 0.0 {
        return radiance;
    }
    let (w, h) = (model.width() as usize, model.height() as usize);
    let span = exposure * mb.shutter;
    let cam_at = |dt: f64| trajectory::interpolate(poses, t + dt).camera(&spec.extrinsics, ell);
    let (ca, cb) = (cam_at(-0.5 * span), cam_at(0.5 * span));
    // blur length on a coarse grid decides the number of sub-poses
    let mut maxd: f64 = 0.0;
    for gy in 0..8 {
        for gx in 0..8 {
            let (x, y) = (gx * w / 8 + w / 16, gy * h / 8 + h / 16);
            if let Some(p) = points[y * w + x] {
                if let (Some(a), Some(b)) = (model.project(ca.world_to_cam(p)), model.project(cb.world_to_cam(p))) {
                    maxd = maxd.max((a - b).length());
                }
            }
        }
    }
    let ksub = ((maxd * 1.5).ceil() as usize + 1).clamp(1, (mb.max_samples as usize).max(1));
    if ksub <= 1 || maxd <= 0.25 {
        return radiance;
    }
    let sub: Vec<CamPose> = (0..ksub).map(|i| cam_at(span * (i as f64 / (ksub - 1) as f64 - 0.5))).collect();
    let far = 1e7; // sky: a point "at infinity" along the pixel ray
    let mut disp = vec![0f32; w * h * 2 * ksub];
    disp.par_chunks_mut(w * 2 * ksub).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            // terrain points belong to pixel + sample_offset (even supersampling); the sky ray is
            // taken at the pixel itself
            let (p, o) = match points[y * w + x] {
                Some(p) => (p, sample_offset),
                None => (cam.cam_to_world(model.unproject(DVec2::new(x as f64, y as f64)).unwrap_or(DVec3::Z) * far), 0.0),
            };
            for (i, c) in sub.iter().enumerate() {
                if let Some(px) = model.project(c.world_to_cam(p)) {
                    row[(x * ksub + i) * 2] = (px.x - x as f64 - o) as f32;
                    row[(x * ksub + i) * 2 + 1] = (px.y - y as f64 - o) as f32;
                }
            }
        }
    });
    sensor.motion_blur(&radiance, &disp, ksub)
}

/// A rendered frame waiting for the next one (its flow target).
struct Pending {
    index: usize,
    t: f64,
    cam: CamPose,
    rgb: Option<Vec<u8>>,
    exposure: Option<[f64; 3]>,
    depth_z: Vec<f32>,
    points: Vec<Option<DVec3>>,
    landcover: Vec<u8>,
    stars: Vec<crate::stars::StarObs>,
}

/// The frame modalities of one camera. `progress(done, total)` in frames.
#[allow(clippy::too_many_arguments)]
fn render_camera(scn: &Scenario, spec: &CameraSpec, poses: &[Pose], win: Window, cache: Arc<TileCache>, ell: Ellipsoid, file: &h5::File, progress: &dyn Fn(usize, usize)) -> Result<usize> {
    let model = spec.intrinsics.build()?;
    let (w, h) = (model.width() as usize, model.height() as usize);
    let times = frame_times(scn, spec, win);
    let n = times.len();
    let mut renderer = renderer(scn, model.clone(), spec.supersample(&scn.render), ell, cache);
    renderer.geometry_only = spec.rgb.is_none();
    // stars are drawn along the exposure track after motion blur (below)
    renderer.stars_in_render = false;
    let mut sensor = spec.rgb.as_ref().map(|r| {
        let mut cfg = r.sensor.clone();
        cfg.noise.seed ^= spec.seed_mix();
        Sensor::new(cfg, w, h)
    });
    let mut writer = CameraWriter::new(file, spec, n, &scn.output.compression, win.t0)?;
    let mut png = match &scn.output.png_dir {
        Some(d) => Some(PngWriter::new(&d.join(spec.slug()), spec, win.t0)?),
        None => None,
    };
    let keep_bits = scn.output.compression.float_keep_bits;
    let mut emit = |p: Pending, mut flow: Option<(Vec<f32>, Vec<u8>)>| -> Result<()> {
        let mut depth = spec.depth.as_ref().map(|d| match d.kind {
            DepthKind::Z => p.depth_z.clone(),
            DepthKind::Range => p.points.iter().map(|q| q.map(|q| (q - p.cam.pos).length() as f32).unwrap_or(f32::INFINITY)).collect(),
        });
        if let Some(k) = keep_bits {
            if let Some(d) = depth.as_mut() {
                output::round_mantissa(d, k);
            }
            if let Some((f, _)) = flow.as_mut() {
                output::round_mantissa(f, k);
            }
        }
        let fr = Frame {
            index: p.index,
            t: p.t,
            cam: p.cam,
            rgb: p.rgb.as_deref(),
            exposure: p.exposure,
            depth: depth.as_deref(),
            flow: flow.as_ref().map(|(f, v)| (f.as_slice(), v.as_slice())),
            landcover: spec.landcover.as_ref().map(|_| p.landcover.as_slice()),
            stars: spec.stars.as_ref().map(|_| p.stars.as_slice()),
        };
        writer.write(&fr)?;
        if let Some(pw) = png.as_mut() {
            pw.write(&fr)?;
        }
        Ok(())
    };

    let tr0 = poses[0].t;
    let mut pending: Option<Pending> = None;
    progress(0, n);
    for (k, &t) in times.iter().enumerate() {
        let pose = trajectory::interpolate(poses, t);
        let cam = pose.camera(&spec.extrinsics, &ell);
        let mut sun = scn.render.lighting.sun_at(t - tr0, pose.geo.lat, pose.geo.lon);
        // the exposure is decided from the previous frames' metering before capturing this one
        // (its window matters for lamp flicker)
        let ex_pre = sensor.as_mut().filter(|s| s.has_metering()).map(|s| s.exposure_for(t));
        if let Some(r) = &spec.rgb {
            sun.exposure = ex_pre.map(|e| e.time).unwrap_or(r.sensor.exposure.base_time);
        }
        let frame = renderer.render(&cam, &sun);
        let mut stars_gt = vec![];
        // camera poses across the open shutter, for star trails
        let star_track = |span: f64| -> Vec<CamPose> {
            let k = if span > 0.0 { 17 } else { 1 };
            (0..k).map(|i| if k == 1 { cam } else { trajectory::interpolate(poses, t + span * (i as f64 / (k - 1) as f64 - 0.5)).camera(&spec.extrinsics, &ell) }).collect()
        };
        let (rgb, exposure) = match sensor.as_mut() {
            Some(s) => {
                if !s.has_metering() {
                    s.meter(&frame.radiance); // start converged
                }
                let ex = ex_pre.unwrap_or_else(|| s.exposure_for(t));
                let mut radiance = motion_blur(s, frame.radiance, &frame.points, frame.sample_offset, &cam, poses, spec, t, ex.time, &ell, model.as_ref());
                if sun.stars {
                    let mb = &spec.rgb.as_ref().unwrap().sensor.motion_blur;
                    let span = if mb.enabled { ex.time * mb.shutter } else { 0.0 };
                    stars_gt = renderer.stars().render_track(&mut radiance, &frame.points, (w, h), model.as_ref(), &star_track(span), &cam, sun.unix, &ell, &scn.render.atmosphere);
                }
                let rgb = s.develop(&radiance, &ex, k as u64);
                s.meter(&radiance);
                (Some(rgb), Some([ex.time, ex.gain, ex.ev]))
            }
            None => {
                if sun.stars && spec.stars.is_some() {
                    stars_gt = renderer.stars().render_track(&mut [], &frame.points, (w, h), model.as_ref(), &[cam], &cam, sun.unix, &ell, &scn.render.atmosphere);
                }
                (None, None)
            }
        };
        // flow of the previous frame, now that this frame's depth is known
        if let Some(prev) = pending.take() {
            let flow = spec.flow.as_ref().map(|_| compute_flow(&prev.points, w, h, &cam, &frame.depth, model.as_ref()));
            emit(prev, flow)?;
        }
        let stars_gt = match &spec.stars {
            Some(m) => stars_gt.into_iter().filter(|s| s.v as f64 <= m.mag_limit).collect(),
            None => vec![],
        };
        pending = Some(Pending { index: k, t, cam, rgb, exposure, depth_z: frame.depth, points: frame.points, landcover: frame.landcover, stars: stars_gt });
        progress(k + 1, n);
    }
    if let Some(prev) = pending.take() {
        let flow = spec.flow.as_ref().map(|_| (vec![0f32; w * h * 2], vec![0u8; w * h]));
        emit(prev, flow)?;
    }
    if let Some(pw) = png {
        pw.finish()?;
    }
    writer.finish()
}

/// Synthesize the IMU over the window into `<imu.path>`. Returns the number of samples.
fn write_imu(scn: &Scenario, imu: &crate::imu::ImuConfig, poses: &[Pose], win: Window, ell: &Ellipsoid, file: &h5::File) -> Result<usize> {
    let truth = trajectory::load_imu_truth(&scn.trajectory.file)?;
    if truth.is_none() {
        eprintln!("imu: trajectory has no IMU truth columns (f_*, w_*); deriving from poses numerically (lower fidelity)");
    }
    let d = crate::imu::synthesize(imu, poses, truth.as_deref(), ell, win.t0, win.t1)?;
    crate::imu::write_h5(file, imu, &d, win.t0, scn.output.compression.level)?;
    Ok(d.t.len())
}

/// Summary of a `render_sequence` run.
#[derive(Debug, Default)]
pub struct RenderReport {
    /// (camera path, frames written) for cameras with frame modalities
    pub frames: Vec<(String, usize)>,
    pub pose_samples: usize,
    pub imu_samples: Option<usize>,
}

/// Create the sequence file: body ground truth, every camera's calibration and frame
/// modalities, the IMU. Event streams are added by [`render_events`].
/// `progress(camera path, done, total)` in frames.
pub fn render_sequence(scn: &Scenario, poses: &[Pose], store: Arc<TileStore>, gen: Option<Arc<Generator>>, progress: &dyn Fn(&str, usize, usize)) -> Result<RenderReport> {
    let ell = store.meta().ellipsoid();
    let win = Window::new(scn, poses)?;
    let cache = tile_cache(scn, store, gen);
    let file = output::create_file(scn, win.t0).with_context(|| format!("creating {}", scn.output.file.display()))?;
    let mut report = RenderReport::default();

    let tr0 = poses[0].t;
    let body: Vec<BodySample> = uniform_times(win, 0.0, scn.output.pose.rate_hz, None)
        .into_iter()
        .map(|t| {
            let pose = trajectory::interpolate(poses, t);
            BodySample { t, pose, sun: scn.render.lighting.sun_at(t - tr0, pose.geo.lat, pose.geo.lon) }
        })
        .collect();
    output::write_body_pose(&file, &scn.output.pose.path, win.t0, &body, &ell)?;
    report.pose_samples = body.len();

    if let Some(imu) = &scn.imu {
        report.imu_samples = Some(write_imu(scn, imu, poses, win, &ell, &file)?);
    }
    for spec in &scn.cameras {
        if spec.has_frames() {
            let n = render_camera(scn, spec, poses, win, cache.clone(), ell, &file, &|d, t| progress(&spec.path, d, t))?;
            report.frames.push((spec.path.clone(), n));
        } else {
            // event-only camera: calibration now, events from `render_events`
            let g = output::fresh_group(&file, &spec.path)?;
            output::write_camera_calib(&g, &spec.intrinsics, output::transform_4x4(spec.extrinsics.r_body_cam(), spec.extrinsics.t_body_cam()))?;
        }
        file.flush()?;
    }
    let g = cache.flush_generated()?;
    if g > 0 {
        eprintln!("lazily generated and stored {g} tiles");
    }
    Ok(report)
}

/// Simulate every camera with an `events` modality into the existing sequence file.
/// `progress(camera path, simulated s, total s)`. Returns (camera path, statistics) per camera.
pub fn render_events(scn: &Scenario, poses: &[Pose], store: Arc<TileStore>, gen: Option<Arc<Generator>>, progress: &dyn Fn(&str, f64, f64)) -> Result<Vec<(String, crate::events::EventStats)>> {
    let ell = store.meta().ellipsoid();
    let win = Window::new(scn, poses)?;
    let cache = tile_cache(scn, store, gen);
    let path = &scn.output.file;
    let file = output::open_file(path).with_context(|| format!("opening {} (run `terrain render` first)", path.display()))?;
    let mut out = vec![];
    for spec in scn.cameras.iter().filter(|c| c.events.is_some()) {
        let n = crate::events::simulate(scn, spec, poses, cache.clone(), ell, &file, (win.t0, win.t1), &|d, t| progress(&spec.path, d, t))?;
        file.flush()?;
        out.push((spec.path.clone(), n));
    }
    cache.flush_generated()?;
    Ok(out)
}
