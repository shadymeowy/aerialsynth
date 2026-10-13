//! Times one camera render over tiles already in a store (e.g. a `terrain survey --tiles` store):
//! the default scenario's renderer and sensor (as `terrain survey`), a pinhole camera of 70°
//! looking `PITCH` degrees down, `AGL` metres above the highest tile at the place (zoom 14), the
//! sun at 10:30 local solar time on the equinox; the frame is rendered `N` times (the first
//! uploads and builds the meshes) and the min / median of the others are printed.
//!
//!     cargo run --release -p render --example render_bench -- STORE.h5 LAT LON [AGL_M] [PITCH_DEG] [N] [WxH] [cpu|gpu] [OUT.png]
//!
//! `RENDER_PROFILE=1` prints the renderer's timings per frame, `BENCH_WALLS=0` and
//! `BENCH_GLINT=0` turn `render.walls` and `render.water_glint` off.
use render::camera::{CameraConfig, Extrinsics, Mount};
use render::raster::Backend;
use render::scenario::{CameraSpec, Scenario};
use render::{pipeline, Pose};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        anyhow::bail!("usage: render_bench STORE.h5 LAT LON [AGL_M] [PITCH_DEG] [N] [WxH] [cpu|gpu] [OUT.png]");
    }
    let arg = |i: usize, d: &str| a.get(i).cloned().unwrap_or_else(|| d.to_string());
    let (lat_d, lon_d): (f64, f64) = (a[2].parse()?, a[3].parse()?);
    let agl: f64 = arg(4, "1500").parse()?;
    let pitch: f64 = arg(5, "-25").parse()?;
    let n: usize = arg(6, "6").parse::<usize>()?.max(2);
    let size = arg(7, "960x540");
    let (w, h) = size.split_once('x').ok_or_else(|| anyhow::anyhow!("size WxH"))?;
    let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
    let store = Arc::new(tilestore::TileStore::open(&a[1])?);
    let ell = store.meta().ellipsoid();
    let (lat, lon) = (lat_d.to_radians(), lon_d.to_radians());
    let top = store.elev_range(geodesy::tile_for_latlon(lat, lon, 14)).map_or(0.0, |r| r.1 as f64);
    let mut scn = Scenario::default();
    scn.render.backend = match arg(8, "gpu").as_str() {
        "cpu" => Backend::Cpu,
        _ => Backend::Gpu,
    };
    // (BENCH_WALLS=0: without the wall materials)
    scn.render.walls = std::env::var("BENCH_WALLS").map_or(true, |v| v != "0");
    scn.render.water_glint = std::env::var("BENCH_GLINT").map_or(true, |v| v != "0");
    let mut spec = CameraSpec::example();
    spec.intrinsics = CameraConfig::pinhole_hfov(w, h, 70.0);
    spec.extrinsics = Extrinsics { mount: Mount::Forward, pitch_deg: pitch, ..Extrinsics::default() };
    let model = spec.intrinsics.build()?;
    let cache = pipeline::tile_cache(&scn, store, None);
    let mut r = pipeline::renderer(&scn, model, spec.supersample(&scn.render), ell, cache);
    r.radiance_only = true;
    let unix = 1_773_964_800.0 + (10.5 - lon_d / 15.0) * 3600.0; // 2026-03-20 10:30 local solar time
    let sun = scn.render.lighting.sun_at_utc(unix, lat, lon);
    let pose = Pose { t: 0.0, geo: geodesy::Geodetic::new(lat, lon, top + agl), q_ned_body: geodesy::euler_zyx_to_quat(sun.azimuth + 150f64.to_radians(), 0.0, 0.0) };
    let cam = pose.camera(&spec.extrinsics, &ell);
    let mut times = Vec::new();
    let mut frame = None;
    for _ in 0..n {
        let t = std::time::Instant::now();
        frame = Some(r.try_render(&cam, &sun)?);
        times.push(t.elapsed().as_secs_f64());
    }
    let first = times.remove(0);
    times.sort_by(f64::total_cmp);
    println!(
        "render_bench {lat_d} {lon_d} {w}x{h} {:?}: first {:.1} ms, min {:.1} ms, median {:.1} ms ({} runs)",
        r.settings.backend,
        first * 1e3,
        times[0] * 1e3,
        times[times.len() / 2] * 1e3,
        times.len()
    );
    if let Some(out) = a.get(9) {
        let f = frame.unwrap();
        let mut sensor = render::sensor::Sensor::new(spec.rgb.clone().unwrap_or_default().sensor, w as usize, h as usize);
        sensor.meter(&f.radiance);
        let ex = sensor.exposure_for(0.0);
        image::RgbImage::from_raw(w, h, sensor.develop(&f.radiance, &ex, 0)).unwrap().save(out)?;
    }
    Ok(())
}
