//! Headless snapshots of the map view (`terrain view --snapshot`).

use crate::fly::{FlyCam, FlyMode};
use crate::globe::{Camera, Globe};
use crate::tiles::Service;
use crate::{base_tiles, map_settings, ViewOptions};
use anyhow::{Context, Result};
use eframe::wgpu;
use glam::DVec3;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use terragen::Generator;
use tilestore::TileStore;

/// Headless: render the `--view` once every tile it wants is in, save a PNG.
pub(crate) fn snapshot(args: &ViewOptions, store: Arc<TileStore>, gen: Arc<Generator>, out: &Path) -> Result<()> {
    let fly_view = args.view.starts_with("fly:");
    let v: Vec<f64> = args.view.trim_start_matches("fly:").split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--view lat,lon,km,heading,tilt (or fly:lat,lon,agl_m,heading,pitch)")?;
    let (w, h) = args.size.split_once('x').map(|(a, b)| (a.parse::<u32>(), b.parse::<u32>())).context("--size WxH")?;
    let (w, h) = (w?, h?);
    let (device, queue) = pollster::block_on(async {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }).await?;
        anyhow::Ok(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("viewer"), required_limits: adapter.limits(), ..Default::default() }).await?)
    })?;
    let ell = gen.world.ell;
    let svc = Service::start(store, gen, base_tiles(args.base_zoom), (rayon::current_num_threads() / 2).max(1), || {});
    let mut globe = Globe::new(&device, &queue, ell, args.gpu_tiles);
    let s = map_settings(args);
    let mut cam = Camera { lat: v[0].to_radians(), lon: v[1].to_radians(), dist: v[2] * 1000.0, heading: v[3].to_radians(), tilt: v[4].to_radians(), fov_y: 40f64.to_radians(), target_h: 0.0 };
    let t0 = Instant::now();
    let mut calm = 0;
    loop {
        cam.target_h = globe.height_at(cam.lat, cam.lon, 22).unwrap_or(0.0).max(0.0) * s.exaggeration as f64;
        let cf = if fly_view {
            // v[2] m above the ground under the camera (as known so far)
            let p = geodesy::geodetic2ecef(geodesy::Geodetic { lat: cam.lat, lon: cam.lon, h: cam.target_h + v[2] }, &ell);
            let mut f = FlyCam::at(p, DVec3::X, &ell, cam.fov_y, FlyMode::Free);
            (f.heading, f.pitch) = (v[3].to_radians(), v[4].to_radians());
            f.frame(&ell, w as f64 / h as f64, cam.target_h)
        } else {
            cam.frame(&ell, w as f64 / h as f64)
        };
        globe.render(&cf, &s, &svc, w, h, None);
        calm = if globe.settled(&svc) { calm + 1 } else { 0 };
        if calm >= 5 || t0.elapsed().as_secs_f64() > args.wait {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let (w, h, px) = globe.read_image().context("reading the image back")?;
    image::save_buffer(out, &px, w, h, image::ExtendedColorType::Rgba8)?;
    let g = &globe.stats;
    eprintln!(
        "wrote {} after {:.1}s: {} patches, finest z{}, {} GPU tiles, {} generated",
        out.display(),
        t0.elapsed().as_secs_f64(),
        g.drawn,
        g.max_zoom_drawn,
        g.resident,
        svc.stats().generated.load(std::sync::atomic::Ordering::Relaxed)
    );
    Ok(())
}

