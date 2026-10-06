//! High-level operations used by the CLI: plan tiles, generate tiles, render a sequence.

use crate::cache::TileCache;
use crate::camera::CameraModel;
use crate::lod::{LodParams, PlanOracle, Selector};
use crate::output::{FrameRecord, H5Writer, PngWriter, PoseRecord};
use crate::raster::Renderer;
use crate::scenario::Scenario;
use crate::sensor::Sensor;
use crate::trajectory::{self, CamPose, Pose};
use anyhow::{bail, Result};
use geodesy::tiles::TileId;
use geodesy::{Ellipsoid, Geodetic};
use glam::{DQuat, DVec2, DVec3};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use terragen::Generator;
use tilestore::{Layer, StoreMeta, TileStore};

/// Frame times of the output sequence.
pub fn frame_times(scn: &Scenario, poses: &[Pose]) -> Vec<f64> {
    let t0 = poses.first().map(|p| p.t).unwrap_or(0.0);
    let t1 = poses.last().map(|p| p.t).unwrap_or(0.0);
    let o = &scn.output;
    let start = t0 + o.start;
    let end = o.end.map(|e| t0 + e).unwrap_or(t1).min(t1);
    let dt = 1.0 / o.frame_rate.max(1e-6);
    let mut v = Vec::new();
    let mut k = 0;
    loop {
        let t = start + k as f64 * dt;
        if t > end + 1e-9 {
            break;
        }
        v.push(t);
        k += 1;
        if let Some(m) = o.max_frames {
            if v.len() >= m {
                break;
            }
        }
    }
    v
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

/// Tiles needed to render the sequence (with margins and all ancestors).
pub fn plan(scn: &Scenario, poses: &[Pose], gen: Option<&Generator>, model: &dyn CameraModel) -> BTreeSet<TileId> {
    let ell = Ellipsoid::from_a_invf(scn.world.planet.a, scn.world.planet.inv_f);
    let params = LodParams {
        min_zoom: scn.tiles.min_zoom,
        max_zoom: scn.tiles.max_zoom,
        texel_px: scn.tiles.plan_texel_px,
        cone_margin: 0.08,
        ..Default::default()
    };
    let times = frame_times(scn, poses);
    let est = gen.map(generator_range_estimator);
    let oracle = match &est {
        Some(f) => PlanOracle { estimate: Some(f), fixed: (-100.0, 5000.0) },
        None => PlanOracle::fixed((-100.0, 5000.0)),
    };
    let every = scn.tiles.plan_every.max(1);
    let mut set: BTreeSet<TileId> = BTreeSet::new();
    for (k, t) in times.iter().enumerate() {
        if k % every != 0 && k + 1 != times.len() {
            continue;
        }
        let cam = trajectory::interpolate(poses, *t).camera(&scn.extrinsics, &ell);
        let sel = Selector::new(&cam, model, ell, &params, &oracle);
        for u in sel.select() {
            set.insert(u.id);
        }
    }
    // neighbours (margin) at each zoom
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
    // all ancestors down to the coarsest level
    let base: Vec<TileId> = set.iter().copied().collect();
    for id in base {
        let mut a = id;
        while a.z > 0 {
            a = a.parent().unwrap();
            set.insert(a);
            if a.z <= scn.tiles.min_zoom.saturating_sub(2) {
                break;
            }
        }
    }
    set
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
    let s = TileStore::open_or_create(&scn.tiles.file, meta)?;
    if !s.meta().generator_config.is_empty() && s.meta().seed != gen.config().seed {
        eprintln!("warning: tile store seed {} differs from scenario seed {}", s.meta().seed, gen.config().seed);
    }
    Ok(s)
}

fn pose_record(p: &Pose, cam: &CamPose, ned0: &geodesy::LocalFrame, ell: &Ellipsoid) -> PoseRecord {
    let g = geodesy::ecef2geodetic(cam.pos, ell);
    let r_ned0_ecef = geodesy::rot_ecef2ned(ned0.origin.lat, ned0.origin.lon);
    PoseRecord {
        cam_ecef: cam.pos,
        q_ecef_cam: cam.q_ecef_cam(),
        cam_lla: [g.lat.to_degrees(), g.lon.to_degrees(), g.h],
        cam_ned0: geodesy::ecef2ned(cam.pos, ned0.origin, ell),
        q_ned0_cam: DQuat::from_mat3(&(r_ned0_ecef * cam.r_ecef_cam)).normalize(),
        q_ned_body: p.q_ned_body,
        body_lla: [p.geo.lat.to_degrees(), p.geo.lon.to_degrees(), p.geo.h],
    }
}

/// Forward flow from frame A (points) to frame B (pose + depth). Returns (flow, valid).
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
                let d = depth_b[v as usize * w + u as usize] as f64;
                if (pc.z - d).abs() < 0.02 * pc.z.abs() + 0.5 {
                    vr[x] = 1;
                }
            }
        }
    });
    (flow, valid)
}

/// Render the whole sequence. `progress(done, total)`.
pub fn render_sequence(scn: &Scenario, poses: &[Pose], store: Arc<TileStore>, gen: Option<Arc<Generator>>, progress: &dyn Fn(usize, usize)) -> Result<()> {
    if poses.is_empty() {
        bail!("empty trajectory");
    }
    let ell = store.meta().ellipsoid();
    let model = scn.camera.build()?;
    let (w, h) = (model.width() as usize, model.height() as usize);
    // load only the layers the shading mode needs (~40% less memory per cached tile)
    let mut layers = vec![Layer::Elevation, Layer::Landcover, Layer::Emission];
    match scn.render.shading {
        crate::raster::Shading::Relit => layers.extend([Layer::Albedo, Layer::Normal]),
        crate::raster::Shading::Satellite => layers.push(Layer::Rgb),
    }
    let mut cache = TileCache::new(store.clone(), layers, scn.tiles.cache_tiles);
    if scn.tiles.lazy {
        if let Some(g) = gen.clone() {
            cache = cache.with_generator(g, scn.tiles.max_zoom, true);
        }
    }
    let cache = Arc::new(cache);
    let mut rs = scn.render.clone();
    rs.min_zoom = scn.tiles.min_zoom;
    rs.max_zoom = rs.max_zoom.min(scn.tiles.max_zoom);
    let renderer = Renderer::new(model.clone(), rs, ell, cache.clone());
    let mut sensor = Sensor::new(scn.sensor.clone(), w, h);
    let times = frame_times(scn, poses);
    let n = times.len();
    let yaml = scn.to_yaml();
    let o = &scn.output;
    let mut png = if o.png { Some(PngWriter::new(&o.dir, w as u32, h as u32, &scn.camera, &scn.extrinsics, &yaml, o.depth, o.flow, o.landcover)?) } else { None };
    let mut h5w = match &o.h5 {
        Some(p) => Some(H5Writer::new(p, w as u32, h as u32, n, &scn.camera, &scn.extrinsics, &yaml, o.depth, o.flow, o.landcover, &o.compression)?),
        None => None,
    };
    let first_cam = trajectory::interpolate(poses, times[0]).camera(&scn.extrinsics, &ell);
    let g0 = geodesy::ecef2geodetic(first_cam.pos, &ell);
    let ned0 = geodesy::LocalFrame::new(Geodetic::new(g0.lat, g0.lon, g0.h), geodesy::LocalConvention::Ned, ell);

    struct Pending {
        index: usize,
        t: f64,
        rgb: Vec<u8>,
        depth: Vec<f32>,
        points: Vec<Option<DVec3>>,
        landcover: Vec<u8>,
        pose: PoseRecord,
        ex: crate::sensor::Exposure,
        sun: crate::lighting::SunState,
    }
    let mut pending: Option<Pending> = None;
    let keep_bits = o.compression.float_keep_bits;
    let flush = |mut p: Pending, mut flow: Vec<f32>, valid: Vec<u8>, png: &mut Option<PngWriter>, h5w: &mut Option<H5Writer>| -> Result<()> {
        if let Some(k) = keep_bits {
            crate::output::round_mantissa(&mut p.depth, k);
            crate::output::round_mantissa(&mut flow, k);
        }
        let rec = FrameRecord {
            index: p.index,
            t: p.t,
            rgb: &p.rgb,
            depth: &p.depth,
            flow: &flow,
            flow_valid: &valid,
            landcover: &p.landcover,
            pose: p.pose,
            exposure_time: p.ex.time,
            gain: p.ex.gain,
            ev: p.ex.ev,
            sun_elevation: p.sun.elevation,
            sun_azimuth: p.sun.azimuth,
        };
        if let Some(w) = png {
            w.write(&rec)?;
        }
        if let Some(w) = h5w {
            w.write(&rec)?;
        }
        Ok(())
    };

    progress(0, n);
    for (k, &t) in times.iter().enumerate() {
        let pose = trajectory::interpolate(poses, t);
        let cam = pose.camera(&scn.extrinsics, &ell);
        let sun = scn.render.lighting.sun_at(t - poses[0].t, pose.geo.lat, pose.geo.lon);
        let frame = renderer.render(&cam, &sun);
        if !sensor.has_metering() {
            sensor.meter(&frame.radiance); // start converged
        }
        let ex = sensor.exposure_for(t);
        // ---- motion blur from sub-frame poses across the exposure window
        let mb = &scn.sensor.motion_blur;
        let mut radiance = frame.radiance;
        if mb.enabled && ex.time > 0.0 {
            let span = ex.time * mb.shutter;
            let cams: Vec<CamPose> = (0..2).map(|i| trajectory::interpolate(poses, t + span * (i as f64 - 0.5)).camera(&scn.extrinsics, &ell)).collect();
            // estimate the blur length on a coarse grid to choose the number of samples
            let mut maxd: f64 = 0.0;
            for gy in 0..8 {
                for gx in 0..8 {
                    let (x, y) = ((gx * w / 8 + w / 16), (gy * h / 8 + h / 16));
                    if let Some(p) = frame.points[y * w + x] {
                        if let (Some(a), Some(b)) = (model.project(cams[0].world_to_cam(p)), model.project(cams[1].world_to_cam(p))) {
                            maxd = maxd.max((a - b).length());
                        }
                    }
                }
            }
            let ksub = ((maxd * 1.5).ceil() as usize + 1).clamp(1, mb.max_samples as usize);
            if ksub > 1 && maxd > 0.25 {
                let sub: Vec<CamPose> = (0..ksub)
                    .map(|i| trajectory::interpolate(poses, t + span * (i as f64 / (ksub - 1) as f64 - 0.5)).camera(&scn.extrinsics, &ell))
                    .collect();
                let rays_far = 1e7;
                let mut disp = vec![0f32; w * h * 2 * ksub];
                disp.par_chunks_mut(w * 2 * ksub).enumerate().for_each(|(y, row)| {
                    for x in 0..w {
                        let p = match frame.points[y * w + x] {
                            Some(p) => p,
                            None => {
                                // sky: direction only (point at "infinity")
                                let r = model.unproject(DVec2::new(x as f64, y as f64)).unwrap_or(DVec3::Z);
                                cam.cam_to_world(r * rays_far)
                            }
                        };
                        for (i, c) in sub.iter().enumerate() {
                            if let Some(px) = model.project(c.world_to_cam(p)) {
                                row[(x * ksub + i) * 2] = (px.x - x as f64) as f32;
                                row[(x * ksub + i) * 2 + 1] = (px.y - y as f64) as f32;
                            }
                        }
                    }
                });
                radiance = sensor.motion_blur(&radiance, &disp, ksub);
            }
        }
        let rgb = sensor.develop(&radiance, &ex, k as u64);
        sensor.meter(&radiance);
        let rec = pose_record(&pose, &cam, &ned0, &ell);
        // ---- flow for the previous frame, now that this frame's depth is known
        if let Some(prev) = pending.take() {
            let (flow, valid) = if o.flow { compute_flow(&prev.points, w, h, &cam, &frame.depth, model.as_ref()) } else { (vec![], vec![]) };
            flush(prev, flow, valid, &mut png, &mut h5w)?;
        }
        pending = Some(Pending { index: k, t, rgb, depth: frame.depth, points: frame.points, landcover: frame.landcover, pose: rec, ex, sun });
        progress(k + 1, n);
    }
    if let Some(prev) = pending.take() {
        let (flow, valid) = if o.flow { (vec![0f32; w * h * 2], vec![0u8; w * h]) } else { (vec![], vec![]) };
        flush(prev, flow, valid, &mut png, &mut h5w)?;
    }
    if let Some(w) = png {
        w.finish()?;
    }
    if let Some(w) = h5w {
        w.finish()?;
    }
    let g = cache.flush_generated()?;
    if g > 0 {
        eprintln!("lazily generated and stored {g} tiles");
    }
    Ok(())
}

/// Event-camera simulation over the scenario (see `events.rs`). Returns the number of events.
pub fn render_events(scn: &Scenario, poses: &[Pose], store: Arc<TileStore>, gen: Option<Arc<Generator>>, progress: &dyn Fn(f64, f64)) -> Result<usize> {
    let ell = store.meta().ellipsoid();
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
    let cache = Arc::new(cache);
    let n = crate::events::simulate_events(scn, poses, cache.clone(), ell, progress)?;
    cache.flush_generated()?;
    Ok(n)
}
