//! Headless snapshots of the map view (`terrain view --snapshot`).

use crate::fly::{FlyCam, FlyMode};
use crate::globe::{CamFrame, Camera, Globe};
use crate::tiles::Service;
use crate::{base_tiles, map_settings, ViewOptions};
use anyhow::{Context, Result};
use eframe::wgpu;
use geodesy::ecef2geodetic;
use glam::DVec3;
use std::io::Write;
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
    let (device, queue) = headless_device()?;
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

/// A keyframe of a recorded camera flight (`terrain view --record DIR --path FILE`): the orbit
/// camera (target lat / lon in degrees, distance in km, heading / tilt / field of view in
/// degrees) and the relief exaggeration, interpolated smoothly between keyframes (Catmull-Rom,
/// easing in at the first and out at the last).
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {
    pub t: f64,
    pub lat: f64,
    pub lon: f64,
    pub km: f64,
    #[serde(default)]
    pub heading: f64,
    #[serde(default)]
    pub tilt: f64,
    #[serde(default = "one")]
    pub exag: f64,
    #[serde(default = "forty")]
    pub fov: f64,
}

/// A switch of the map's look at time `t`: shading mode and / or tile borders (held until the
/// next switch that sets them), dissolving from the previous look over `fade` seconds.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Look {
    pub t: f64,
    pub mode: Option<String>,
    pub borders: Option<bool>,
    #[serde(default)]
    pub fade: f64,
}

fn one() -> f64 {
    1.0
}
fn forty() -> f64 {
    40.0
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Flight {
    keys: Vec<Key>,
    #[serde(default)]
    look: Vec<Look>,
}

/// The camera and relief exaggeration at time `t` (s) of a keyframed flight.
fn at_time(keys: &[Key], t: f64) -> (Camera, f64) {
    let n = keys.len();
    let i = keys.iter().rposition(|k| k.t <= t).unwrap_or(0).min(n.saturating_sub(2));
    let (a, b) = (&keys[i], &keys[(i + 1).min(n - 1)]);
    let span = (b.t - a.t).max(1e-9);
    let s = ((t - a.t) / span).clamp(0.0, 1.0);
    let unit = |k: &Key| {
        let (la, lo) = (k.lat.to_radians(), k.lon.to_radians());
        DVec3::new(la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin())
    };
    // channels: position (unit vector), log distance, heading, tilt, exaggeration, fov
    let ch = |k: &Key| -> [f64; 8] {
        let u = unit(k);
        [u.x, u.y, u.z, k.km.ln(), k.heading, k.tilt, k.exag, k.fov]
    };
    let tangent = |j: usize| -> [f64; 8] {
        if j == 0 || j + 1 >= n {
            return [0.0; 8]; // ease in / out at the ends
        }
        let (p, q) = (ch(&keys[j - 1]), ch(&keys[j + 1]));
        let dt = (keys[j + 1].t - keys[j - 1].t).max(1e-9);
        std::array::from_fn(|c| (q[c] - p[c]) / dt)
    };
    let (pa, pb, ma, mb) = (ch(a), ch(b), tangent(i), tangent((i + 1).min(n - 1)));
    let (s2, s3) = (s * s, s * s * s);
    let (h00, h10, h01, h11) = (2.0 * s3 - 3.0 * s2 + 1.0, s3 - 2.0 * s2 + s, -2.0 * s3 + 3.0 * s2, s3 - s2);
    let v: [f64; 8] = std::array::from_fn(|c| h00 * pa[c] + h10 * span * ma[c] + h01 * pb[c] + h11 * span * mb[c]);
    let u = DVec3::new(v[0], v[1], v[2]).normalize();
    let cam = Camera {
        lat: u.z.clamp(-1.0, 1.0).asin(),
        lon: u.y.atan2(u.x),
        dist: v[3].exp() * 1000.0,
        heading: v[4].to_radians(),
        tilt: v[5].to_radians().clamp(0.0, 1.5),
        fov_y: v[7].to_radians(),
        target_h: 0.0,
    };
    (cam, v[6])
}

/// The look (shading mode, borders) set by the switches before `t`, or up to and including it.
fn look_at(look: &[Look], t: f64, inclusive: bool) -> (Option<String>, bool) {
    let past = || look.iter().filter(move |k| k.t < t || (inclusive && k.t == t));
    (past().filter_map(|k| k.mode.clone()).next_back(), past().filter_map(|k| k.borders).next_back().unwrap_or(false))
}

/// While a switch dissolves at `t`: the look before it and the weight of the new one.
fn fading(look: &[Look], t: f64) -> Option<((Option<String>, bool), f64)> {
    let k = look.iter().rev().find(|k| k.t <= t && t < k.t + k.fade)?;
    let x = (t - k.t) / k.fade;
    Some((look_at(look, k.t, false), x * x * (3.0 - 2.0 * x)))
}

/// Roll, pitch, yaw (deg, aerospace ZYX) of an FRD body looking along the camera (x: the view
/// direction, z: down the image) in the NED frame at (lat, lon) (rad).
fn attitude(f: &CamFrame, lat: f64, lon: f64) -> (f64, f64, f64) {
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    let (n, e, d) = (DVec3::new(-sl * co, -sl * so, cl), DVec3::new(-so, co, 0.0), DVec3::new(-cl * co, -cl * so, -sl));
    let (x, z) = (f.dir, -f.cam_up);
    let y = z.cross(x);
    (y.dot(d).atan2(z.dot(d)).to_degrees(), (-x.dot(d)).clamp(-1.0, 1.0).asin().to_degrees(), x.dot(e).atan2(x.dot(n)).to_degrees())
}

pub(crate) fn headless_device() -> Result<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }).await?;
        anyhow::Ok(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("viewer"), required_limits: adapter.limits(), ..Default::default() }).await?)
    })
}

/// Record a keyframed map flight into `dir/frame_00000.png`, … at `fps`: every frame is
/// captured once its tiles are in (or after `args.wait` seconds). `dir/frames.csv` gives each
/// frame's time, the camera pose as a trajectory (position in deg / m above the ellipsoid;
/// roll / pitch / yaw in deg of a forward-looking FRD body, as `terrain run` reads them), the
/// distance to the target (km), the shading mode and the finest zoom level drawn.
pub(crate) fn record(args: &ViewOptions, store: Arc<TileStore>, gen: Arc<Generator>, dir: &Path, path: &Path, fps: f64) -> Result<()> {
    let flight = serde_yaml::from_str::<Flight>(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))?;
    let (keys, look) = (flight.keys, flight.look);
    if keys.len() < 2 || keys.windows(2).any(|w| w[1].t <= w[0].t) {
        anyhow::bail!("{}: at least two keys with increasing t", path.display());
    }
    for m in look.iter().filter_map(|k| k.mode.as_deref()) {
        if !["surface", "elevation", "landcover", "relief"].contains(&m) {
            anyhow::bail!("{}: unknown mode {m:?} (surface, elevation, landcover, relief)", path.display());
        }
    }
    let (w, h) = args.size.split_once('x').map(|(a, b)| (a.parse::<u32>(), b.parse::<u32>())).context("--size WxH")?;
    let (w, h) = (w?, h?);
    std::fs::create_dir_all(dir)?;
    let mut csv = std::io::BufWriter::new(std::fs::File::create(dir.join("frames.csv"))?);
    writeln!(csv, "frame,t,lat,lon,h,roll,pitch,yaw,km,mode,max_zoom")?;
    let (device, queue) = headless_device()?;
    let ell = gen.world.ell;
    let svc = Service::start(store, gen, base_tiles(args.base_zoom), (rayon::current_num_threads() / 2).max(1), || {});
    let mut globe = Globe::new(&device, &queue, ell, args.gpu_tiles);
    let base = map_settings(args);
    let settings = |(mode, borders): (Option<String>, bool), exag: f64| {
        let mut s = base.clone();
        s.exaggeration = exag as f32;
        s.borders = borders;
        if let Some(m) = mode {
            s.mode = crate::parse_mode(&m);
        }
        s
    };
    let n = (keys.last().unwrap().t * fps).round() as usize + 1;
    let t0 = Instant::now();
    for k in 0..n {
        let t = k as f64 / fps;
        let (mut cam, exag) = at_time(&keys, t);
        let now = look_at(&look, t, true);
        let mode = now.0.clone().unwrap_or_else(|| args.mode.clone());
        // the frame (and while a switch dissolves, the frame in the look before it)
        let layers = match fading(&look, t) {
            Some((before, a)) => vec![(settings(before, exag), 1.0 - a), (settings(now, exag), a)],
            None => vec![(settings(now, exag), 1.0)],
        };
        let mut px = vec![0f32; (w * h * 4) as usize];
        let mut max_zoom = 0;
        for (s, weight) in &layers {
            let tf = Instant::now();
            let mut calm = 0;
            loop {
                cam.target_h = globe.height_at(cam.lat, cam.lon, 22).unwrap_or(0.0).max(0.0) * exag;
                globe.render(&cam.frame(&ell, w as f64 / h as f64), s, &svc, w, h, None);
                calm = if globe.settled(&svc) { calm + 1 } else { 0 };
                if calm >= 3 || tf.elapsed().as_secs_f64() > args.wait {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let (_, _, img) = globe.read_image().context("reading the image back")?;
            for (o, &v) in px.iter_mut().zip(&img) {
                *o += v as f32 * *weight as f32;
            }
            max_zoom = max_zoom.max(globe.stats.max_zoom_drawn);
        }
        let img: Vec<u8> = px.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect();
        image::save_buffer(dir.join(format!("frame_{k:05}.png")), &img, w, h, image::ExtendedColorType::Rgba8)?;
        let f = cam.frame(&ell, w as f64 / h as f64);
        let eye = ecef2geodetic(f.eye, &ell);
        let (roll, pitch, yaw) = attitude(&f, eye.lat, eye.lon);
        writeln!(
            csv,
            "{k},{t:.4},{:.8},{:.8},{:.2},{roll:.5},{pitch:.5},{yaw:.5},{:.4},{mode},{max_zoom}",
            eye.lat.to_degrees(),
            eye.lon.to_degrees(),
            eye.h,
            cam.dist / 1000.0
        )?;
        if k % 25 == 0 || k + 1 == n {
            eprintln!(
                "frame {k}/{n} ({:.0} s): {:.1} km, finest z{max_zoom}, {} generated",
                t0.elapsed().as_secs_f64(),
                cam.dist / 1000.0,
                svc.stats().generated.load(std::sync::atomic::Ordering::Relaxed)
            );
        }
    }
    csv.flush()?;
    eprintln!("recorded {n} frames into {} in {:.0} s", dir.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(t: f64, lat: f64, lon: f64, km: f64) -> Key {
        Key { t, lat, lon, km, heading: 0.0, tilt: 0.0, exag: 1.0, fov: 40.0 }
    }

    #[test]
    fn keys_are_hit_and_distance_is_geometric() {
        let keys = [key(0.0, 10.0, 20.0, 10000.0), key(2.0, 0.0, 40.0, 100.0), key(4.0, -5.0, 45.0, 1.0)];
        for k in &keys {
            let (c, _) = at_time(&keys, k.t);
            assert!((c.lat.to_degrees() - k.lat).abs() < 1e-9 && (c.lon.to_degrees() - k.lon).abs() < 1e-9);
            assert!((c.dist / 1000.0 - k.km).abs() < 1e-6 * k.km);
        }
        // eased ends: the start and the end are at rest
        let (a, _) = at_time(&keys, 0.0);
        let (b, _) = at_time(&keys, 0.01);
        assert!((b.dist - a.dist).abs() / a.dist < 1e-3);
        // beyond the last key the camera holds
        let (c, _) = at_time(&keys, 9.0);
        assert!((c.dist / 1000.0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn attitude_of_the_orbit_camera() {
        let ell = geodesy::Ellipsoid::WGS84;
        let mut c = at_time(&[key(0.0, 7.0, -102.0, 3.0), key(1.0, 7.0, -102.0, 3.0)], 0.0).0;
        // looking east, 63 deg from straight down: yaw 90, pitch -27, level wings
        c.heading = 90f64.to_radians();
        c.tilt = 63f64.to_radians();
        let f = c.frame(&ell, 16.0 / 9.0);
        let eye = ecef2geodetic(f.eye, &ell);
        let (roll, pitch, yaw) = attitude(&f, eye.lat, eye.lon);
        assert!(roll.abs() < 0.05 && (pitch + 27.0).abs() < 0.05 && (yaw - 90.0).abs() < 0.05, "{roll} {pitch} {yaw}");
    }

    #[test]
    fn looks_hold_and_dissolve() {
        let l = |t: f64, mode: Option<&str>, borders: Option<bool>, fade: f64| Look { t, mode: mode.map(String::from), borders, fade };
        let look = [l(0.0, Some("surface"), None, 0.0), l(1.0, Some("elevation"), None, 0.5), l(2.0, None, Some(true), 0.0)];
        assert_eq!(look_at(&look, 0.5, true), (Some("surface".into()), false));
        assert_eq!(look_at(&look, 2.5, true), (Some("elevation".into()), true));
        // the switch at 1 s dissolves from surface over 0.5 s
        let (before, a) = fading(&look, 1.25).unwrap();
        assert_eq!(before, (Some("surface".into()), false));
        assert!((a - 0.5).abs() < 1e-12);
        assert!(fading(&look, 1.5).is_none() && fading(&look, 2.0).is_none());
    }
}
