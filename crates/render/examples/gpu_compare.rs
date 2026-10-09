//! Renders one frame with the CPU and the GPU backend and compares them (radiance, depth, land
//! cover, timings); writes CPU | GPU | |difference| as a PNG. The scenario's tiles and trajectory
//! must exist (e.g. after `terrain run`). `SPLIT=1` also compares the lamp-flicker split.
//!
//!     cargo run --release -p render --example gpu_compare -- SCENARIO.yaml [T_SECONDS] [CAMERA_INDEX] [OUT.png]
use render::pipeline;
use render::raster::Backend;
use render::scenario::Scenario;
use std::path::Path;
use std::sync::Arc;

fn tone(x: f32) -> u8 {
    ((x * 2.5).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let scn = Scenario::load(Path::new(&a[1]))?;
    let dt: f64 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let ci: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(0);
    let out = a.get(4).cloned().unwrap_or_else(|| "gpu_compare.png".into());
    let store = Arc::new(tilestore::TileStore::open(&scn.tiles.file)?);
    let ell = store.meta().ellipsoid();
    let poses = render::trajectory::load(&scn.trajectory.file, &ell)?;
    let t = poses[0].t + dt;
    let pose = render::trajectory::interpolate(&poses, t);
    let spec = &scn.cameras[ci];
    let cam = pose.camera(&spec.extrinsics, &ell);
    let mut sun = scn.render.lighting.sun_at(t - poses[0].t, pose.geo.lat, pose.geo.lon);
    sun.exposure = 0.0;
    let model = spec.intrinsics.build()?;
    let cache = pipeline::tile_cache(&scn, store, None);
    let ss = spec.supersample(&scn.render);
    let cpu = pipeline::renderer(&scn, model.clone(), ss, ell, cache.clone());
    let mut gpu = pipeline::renderer(&scn, model.clone(), ss, ell, cache.clone());
    gpu.settings.backend = Backend::Gpu;
    let mut cpu = cpu;
    if std::env::var_os("SPLIT").is_some() {
        cpu.split_flicker = true;
        gpu.split_flicker = true;
    }
    if spec.rgb.is_none() {
        cpu.geometry_only = true;
        gpu.geometry_only = true;
    }
    let t0 = std::time::Instant::now();
    let fc = cpu.render(&cam, &sun);
    let tc = t0.elapsed().as_secs_f64();
    let _warm = gpu.render(&cam, &sun); // first frame: device, pipelines, tile uploads
    let t1 = std::time::Instant::now();
    let fg = gpu.render(&cam, &sun);
    let tg = t1.elapsed().as_secs_f64();
    let (w, h) = (fc.width as usize, fc.height as usize);
    let n = w * h;
    let (mut sad, mut sref, mut maxd) = (0.0f64, 0.0f64, 0.0f32);
    let mut big = 0usize;
    for k in 0..n * 3 {
        let d = (fc.radiance[k] - fg.radiance[k]).abs();
        sad += d as f64;
        sref += fc.radiance[k].abs() as f64;
        maxd = maxd.max(d);
        if d > 0.05 * fc.radiance[k].abs().max(0.02) {
            big += 1;
        }
    }
    let (mut dd, mut dn, mut lc_eq, mut lc_n) = (0.0f64, 0usize, 0usize, 0usize);
    for k in 0..n {
        if fc.depth[k].is_finite() && fg.depth[k].is_finite() {
            dd += ((fc.depth[k] - fg.depth[k]).abs() / fc.depth[k]) as f64;
            dn += 1;
        }
        if fc.landcover[k] != 255 || fg.landcover[k] != 255 {
            lc_n += 1;
            lc_eq += (fc.landcover[k] == fg.landcover[k]) as usize;
        }
    }
    if !fc.flicker_cos.is_empty() || !fg.flicker_cos.is_empty() {
        let d =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / a.iter().map(|x| x.abs() as f64).sum::<f64>().max(1e-12);
        println!(
            "flicker split: cos {:.2}% sin {:.2}% (lens {} {})",
            100.0 * d(&fc.flicker_cos, &fg.flicker_cos),
            100.0 * d(&fc.flicker_sin, &fg.flicker_sin),
            fc.flicker_cos.len(),
            fg.flicker_cos.len()
        );
    }
    let nan = |f: &render::FrameOut| f.radiance.iter().filter(|v| !v.is_finite()).count();
    println!("non-finite radiance: cpu {} gpu {}", nan(&fc), nan(&fg));
    let mut pairs: std::collections::BTreeMap<(u8, u8), usize> = Default::default();
    for k in 0..n {
        if fc.landcover[k] != fg.landcover[k] {
            *pairs.entry((fc.landcover[k], fg.landcover[k])).or_default() += 1;
        }
    }
    let mut pv: Vec<_> = pairs.into_iter().collect();
    pv.sort_by_key(|x| std::cmp::Reverse(x.1));
    println!("land cover mismatches (cpu, gpu): {:?}", &pv[..pv.len().min(8)]);
    println!("{}x{} ss{}: cpu {:.2}s gpu {:.3}s (×{:.0})", w, h, ss, tc, tg, tc / tg);
    println!(
        "radiance: mean |d| {:.2}% of mean, max |d| {:.3}, samples off by >5%: {:.2}%",
        100.0 * sad / sref.max(1e-12),
        maxd,
        100.0 * big as f64 / (3 * n) as f64
    );
    println!("depth: mean rel |d| {:.2e} over {} px; land cover agrees on {:.2}%", dd / dn.max(1) as f64, dn, 100.0 * lc_eq as f64 / lc_n.max(1) as f64);
    let mut img = image::RgbImage::new(3 * w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let k = y * w + x;
            for (i, f) in [&fc, &fg].iter().enumerate() {
                img.put_pixel((i * w + x) as u32, y as u32, image::Rgb([tone(f.radiance[3 * k]), tone(f.radiance[3 * k + 1]), tone(f.radiance[3 * k + 2])]));
            }
            let d = |c: usize| ((fc.radiance[3 * k + c] - fg.radiance[3 * k + c]).abs() * 2000.0).min(255.0) as u8;
            img.put_pixel((2 * w + x) as u32, y as u32, image::Rgb([d(0), d(1), d(2)]));
        }
    }
    img.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
