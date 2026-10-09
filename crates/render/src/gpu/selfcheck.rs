//! Self-check of the GPU renderer for `render.backend: auto`: before the GPU is chosen, it renders
//! a tiny synthetic frame (32 × 24 px, one flat, chequered tile seen from 2000 km) on the GPU and
//! on the CPU and compares them. A GPU that renders it clearly wrong (no terrain, wrong depth or
//! land cover, black or wrong radiance) is not used: auto renders on the CPU for the rest of the
//! process, with a one-line warning. Some virtual GPUs render nothing at all (the macOS Intel
//! runners' "Apple Paravirtual device"). `backend: gpu` is not checked.
//!
//! The verdict is computed once per process. A check that cannot run (no temporary directory for
//! its scratch tile store, a render error) is inconclusive and keeps the GPU.

use crate::cache::TileCache;
use crate::camera::CameraConfig;
use crate::lighting::LightingConfig;
use crate::raster::{Backend, FrameOut, RenderSettings, Renderer};
use crate::trajectory::CamPose;
use geodesy::{Ellipsoid, Geodetic, TileId};
use glam::{DMat3, DVec3};
use std::sync::{Arc, OnceLock};
use tilestore::{Layer, StoreMeta, TileData, TileStore};

const W: u32 = 32;
const H: u32 = 24;
/// The scene: a camera 2000 km above (30° N, 45° E), looking down, 30° field of view, at
/// 09:00 UTC on an equinox (the sun ~60° up); the zoom-2 tile under it is synthetic.
const LAT: f64 = 30.0;
const LON: f64 = 45.0;
const ALT: f64 = 2.0e6;
const UNIX: f64 = 1_710_925_200.0; // 2024-03-20 09:00 UTC
const ZOOM: u8 = 2;

/// Does the GPU render correctly (`true` also when the check is inconclusive)? Computed on the
/// first call; prints a warning when the GPU fails.
pub fn passed() -> bool {
    static VERDICT: OnceLock<bool> = OnceLock::new();
    *VERDICT.get_or_init(|| {
        let t0 = std::time::Instant::now();
        let r = run();
        if std::env::var_os("RENDER_PROFILE").is_some() {
            eprintln!("[gpu self-check] {:.1} ms: {r:?}", t0.elapsed().as_secs_f64() * 1e3);
        }
        match r {
            Ok(Ok(())) => true,
            Ok(Err(why)) => {
                let name = super::device::shared().map(|g| g.info.name.clone()).unwrap_or_default();
                eprintln!("warning: render.backend auto: the GPU ({name}) renders a test frame wrongly ({why}); rendering on the CPU");
                false
            }
            Err(_) => true,
        }
    })
}

/// Render the test frame on both backends: `Ok(verdict)`, or `Err` when the check could not run.
pub fn run() -> anyhow::Result<Result<(), String>> {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("aerialsynth-gpu-check-{}-{n}.h5", std::process::id()));
    let id = geodesy::tile_for_latlon(LAT.to_radians(), LON.to_radians(), ZOOM);
    let result = (|| {
        let ell = Ellipsoid::WGS84;
        let store = Arc::new(TileStore::create(&path, StoreMeta::default())?);
        let cache = Arc::new(TileCache::new(store, Layer::ALL.to_vec(), 16));
        cache.insert(id, Arc::new(tile(id)));
        let model = CameraConfig::pinhole_hfov(W, H, 30.0).build()?;
        let settings = |backend| {
            let mut s = RenderSettings { supersample: 1, min_zoom: ZOOM, max_zoom: ZOOM, backend, ..Default::default() };
            s.lighting = LightingConfig { stars: false, ..Default::default() };
            s
        };
        let render = |backend| -> anyhow::Result<FrameOut> {
            let mut r = Renderer::new(model.clone(), settings(backend), ell, cache.clone());
            r.stars_in_render = false;
            let sun = r.settings.lighting.sun_at_utc(UNIX, LAT.to_radians(), LON.to_radians());
            r.try_render(&camera(&ell), &sun)
        };
        let cpu = render(Backend::Cpu)?;
        let gpu = render(Backend::Gpu)?;
        Ok(compare(&cpu, &gpu))
    })();
    // the GPU keeps tiles and meshes by tile id: forget the synthetic ones
    super::forget_tiles(&[id]);
    let _ = std::fs::remove_file(&path);
    result
}

/// The synthetic tile: flat (elevation 0), a chequerboard of 32-px squares of grey albedo
/// (60 / 200) and land cover (grass / forest), normals straight up, no lights.
fn tile(id: TileId) -> TileData {
    let n = 256 * 256;
    let check: Vec<bool> = (0..n).map(|k| ((k % 256) / 32 + (k / 256) / 32) % 2 == 0).collect();
    let grey: Vec<u8> = check.iter().flat_map(|&c| [if c { 200 } else { 60 }; 3]).collect();
    TileData {
        id,
        rgb: grey.clone(),
        albedo: grey,
        elevation: vec![0.0; n],
        normal: (0..n).flat_map(|_| [0i8, 0, 127]).collect(),
        landcover: check.iter().map(|&c| if c { 8 } else { 10 }).collect(),
        emission: vec![0; 3 * n],
        elev_min: 0.0,
        elev_max: 0.0,
    }
}

/// The camera: at (LAT, LON, ALT), looking straight down, image x east, image y south.
fn camera(ell: &Ellipsoid) -> CamPose {
    let (lat, lon) = (LAT.to_radians(), LON.to_radians());
    let pos = geodesy::geodetic2ecef(Geodetic::new(lat, lon, ALT), ell);
    let east = geodesy::enu2ecefv(DVec3::X, lat, lon);
    let down = -geodesy::enu2ecefv(DVec3::Z, lat, lon);
    let y = down.cross(east);
    CamPose { t: 0.0, pos, r_ecef_cam: DMat3::from_cols(east, y, down) }
}

/// The verdict on a GPU frame against the CPU's of the same scene: `Err(why)` when it is clearly
/// wrong. Lenient (the backends differ in sampling and rounding): the terrain must cover the same
/// pixels (90 %), at the same depth (2 % on average), with the same land cover (90 %) and a mean
/// radiance within 25 % of the CPU's. A CPU frame without terrain is no test (passes).
pub fn compare(cpu: &FrameOut, gpu: &FrameOut) -> Result<(), String> {
    let n = cpu.depth.len();
    if gpu.depth.len() != n || gpu.landcover.len() != n || gpu.radiance.len() != cpu.radiance.len() || cpu.landcover.len() != n {
        return Err(format!("image sizes differ: {} / {} pixels", gpu.depth.len(), n));
    }
    let hits = |f: &FrameOut| f.depth.iter().filter(|d| d.is_finite()).count();
    let (hc, hg) = (hits(cpu), hits(gpu));
    if hc < n / 10 {
        return Ok(());
    }
    let same_hit = cpu.depth.iter().zip(&gpu.depth).filter(|(a, b)| a.is_finite() == b.is_finite()).count();
    if (same_hit as f64) < 0.9 * n as f64 {
        return Err(format!("terrain in {hg} of {n} pixels, the CPU's in {hc}"));
    }
    let both: Vec<(f32, f32)> = cpu.depth.iter().zip(&gpu.depth).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).collect();
    let rel = both.iter().map(|(a, b)| ((a - b).abs() / a.abs().max(1.0)) as f64).sum::<f64>() / both.len().max(1) as f64;
    if rel.is_nan() || rel >= 0.02 {
        return Err(format!("depth off by {:.1} % on average", rel * 100.0));
    }
    let same_lc = cpu.landcover.iter().zip(&gpu.landcover).filter(|(a, b)| a == b).count();
    if (same_lc as f64) < 0.9 * n as f64 {
        return Err(format!("land cover agrees in {same_lc} of {n} pixels"));
    }
    let mean = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64;
    let (mc, mg) = (mean(&cpu.radiance), mean(&gpu.radiance));
    if mc > 1e-3 && (mg.is_nan() || (mg - mc).abs() > 0.25 * mc) {
        return Err(format!("mean radiance {mg:.4}, the CPU's {mc:.4}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(depth: Vec<f32>, landcover: Vec<u8>, radiance: f32) -> FrameOut {
        let n = depth.len();
        FrameOut {
            width: n as u32,
            height: 1,
            radiance: vec![radiance; 3 * n],
            depth,
            points: vec![None; n],
            sample_offset: 0.0,
            landcover,
            flicker_cos: vec![],
            flicker_sin: vec![],
            stars: vec![],
            units: vec![],
        }
    }

    #[test]
    fn verdicts() {
        let n = 100;
        let inf = f32::INFINITY;
        // terrain in 80 pixels, sky in 20
        let depth: Vec<f32> = (0..n).map(|k| if k < 80 { 2.0e6 + k as f32 } else { inf }).collect();
        let lc: Vec<u8> = (0..n).map(|k| if k < 80 { 8 + (k % 2) as u8 * 2 } else { 255 }).collect();
        let cpu = frame(depth.clone(), lc.clone(), 0.3);
        assert_eq!(compare(&cpu, &frame(depth.clone(), lc.clone(), 0.3)), Ok(()));
        // small differences pass
        let d2: Vec<f32> = depth.iter().map(|d| d * 1.005).collect();
        let mut lc2 = lc.clone();
        lc2[0] = 10;
        assert_eq!(compare(&cpu, &frame(d2, lc2, 0.33)), Ok(()));
        // the macOS Intel runner's GPU: no terrain, black
        let e = compare(&cpu, &frame(vec![inf; n], vec![255; n], 0.0)).unwrap_err();
        assert!(e.contains("terrain in 0 of 100"), "{e}");
        // NaN depth counts as no terrain
        assert!(compare(&cpu, &frame(vec![f32::NAN; n], lc.clone(), 0.3)).is_err());
        // wrong depth, land cover, radiance
        assert!(compare(&cpu, &frame(depth.iter().map(|d| d * 1.5).collect(), lc.clone(), 0.3)).unwrap_err().contains("depth"));
        assert!(compare(&cpu, &frame(depth.clone(), (0..n).map(|k| if k < 80 { 1 } else { 255 }).collect(), 0.3)).unwrap_err().contains("land cover"));
        assert!(compare(&cpu, &frame(depth.clone(), lc.clone(), 0.0)).unwrap_err().contains("radiance"));
        assert!(compare(&cpu, &frame(depth.clone(), lc.clone(), f32::NAN)).unwrap_err().contains("radiance"));
        // a CPU frame without terrain is no test; mismatched sizes fail
        assert_eq!(compare(&frame(vec![inf; n], vec![255; n], 0.1), &frame(vec![1.0; n], vec![0; n], 0.0)), Ok(()));
        assert!(compare(&cpu, &frame(vec![1.0; 10], vec![0; 10], 0.3)).is_err());
    }

    /// The check on this machine's GPU (skipped without one). `AERIALSYNTH_EXPECT_GPU_CHECK=pass`
    /// / `fail` asserts the verdict (CI: the macOS Intel runner's GPU fails, the others pass).
    #[test]
    fn self_check_on_this_gpu() {
        if super::super::device::shared().is_err() {
            eprintln!("no GPU, skipped");
            return;
        }
        // (the first run includes the GPU renderer's start-up: shaders, pipelines)
        let mut r = Ok(());
        for k in 0..2 {
            let t0 = std::time::Instant::now();
            r = run().expect("the self-check ran");
            eprintln!("gpu self-check {k}: {r:?} in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        match std::env::var("AERIALSYNTH_EXPECT_GPU_CHECK").as_deref() {
            Ok("pass") => assert_eq!(r, Ok(())),
            Ok("fail") => assert!(r.is_err(), "the GPU passed"),
            _ => {}
        }
    }
}
