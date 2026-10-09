//! Mosaic preview of generated tiles (`terrain tiles --png`).

use anyhow::{bail, Result};
use geodesy::tiles::{tile_for_latlon, TileId};
use terragen::{Generator, TileData, TILE_SIZE};

pub fn layer_rgb(t: &TileData, layer: &str, emin: f32, emax: f32) -> Vec<u8> {
    let n = TILE_SIZE * TILE_SIZE;
    let mut out = vec![0u8; n * 3];
    for k in 0..n {
        let c: [u8; 3] = match layer {
            "rgb" => [t.rgb[3 * k], t.rgb[3 * k + 1], t.rgb[3 * k + 2]],
            "albedo" => [t.albedo[3 * k], t.albedo[3 * k + 1], t.albedo[3 * k + 2]],
            "normal" => [(t.normal[3 * k] as i32 + 128) as u8, (t.normal[3 * k + 1] as i32 + 128) as u8, (t.normal[3 * k + 2] as i32 + 128) as u8],
            "landcover" => terragen::landcover::palette(t.landcover[k]),
            // the stored code (radiance = 16 (v/255)^3) is a perceptual scale already
            "emission" => [t.emission[3 * k], t.emission[3 * k + 1], t.emission[3 * k + 2]],
            "hillshade" => {
                let nx = t.normal[3 * k] as f32 / 127.0;
                let ny = t.normal[3 * k + 1] as f32 / 127.0;
                let nz = t.normal[3 * k + 2] as f32 / 127.0;
                let s = (-0.5 * nx + 0.5 * ny + 0.707 * nz).max(0.0);
                let v = (s * 255.0) as u8;
                [v, v, v]
            }
            _ => {
                // elevation: terrain colormap
                let e = t.elevation[k];
                let u = ((e - emin) / (emax - emin).max(1e-3)).clamp(0.0, 1.0);
                if e <= 0.0 {
                    [30, 60, 140]
                } else {
                    let stops = [(0.0, [40, 110, 50]), (0.35, [170, 170, 90]), (0.7, [130, 90, 60]), (1.0, [250, 250, 250])];
                    let mut c = [0u8; 3];
                    for w in stops.windows(2) {
                        let (a, ca) = w[0];
                        let (b, cb) = w[1];
                        if u >= a && u <= b {
                            let f = (u - a) / (b - a);
                            for i in 0..3 {
                                c[i] = (ca[i] as f32 + (cb[i] as f32 - ca[i] as f32) * f) as u8;
                            }
                        }
                    }
                    c
                }
            }
        };
        out[3 * k..3 * k + 3].copy_from_slice(&c);
    }
    out
}

/// The layers a preview can show.
pub const LAYERS: [&str; 7] = ["rgb", "albedo", "elevation", "normal", "landcover", "hillshade", "emission"];

/// The largest mosaic side (tiles): the mosaic is generated in memory (~1.1 MB per tile, plus
/// 0.2 MB per tile and layer of image).
pub const MAX_SIZE: u32 = 32;

/// Check the preview arguments; returns the layers.
pub fn check_args(zoom: u8, max_zoom: u8, size: u32, layers: &str, at: Option<(f64, f64)>) -> Result<Vec<String>> {
    if zoom > max_zoom {
        bail!("--zoom {zoom}: at most tiles.max_zoom ({max_zoom})");
    }
    if !(1..=MAX_SIZE).contains(&size) {
        bail!("--size {size}: 1..={MAX_SIZE} tiles per side");
    }
    if let Some((lat, lon)) = at {
        if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() {
            bail!("--at {lat},{lon}: needs a latitude in [-90, 90] and a finite longitude");
        }
    }
    let v: Vec<String> = layers.split(',').map(|l| l.trim().to_string()).collect();
    for l in &v {
        if !LAYERS.contains(&l.as_str()) {
            bail!("--layers: unknown layer {l:?} (one of {})", LAYERS.join(", "));
        }
    }
    Ok(v)
}

/// A mosaic of `size` × `size` tiles of zoom `zoom` around (lat, lon) (deg; default: the
/// world's home), generated straight into `<out>_<layer>.png` (no tile store). The arguments
/// are checked by [`check_args`].
pub fn mosaic(cfg: terragen::Config, at: Option<(f64, f64)>, zoom: u8, size: u32, layers: &[String], out: &std::path::Path) -> Result<()> {
    let home = cfg.home.clone().unwrap_or_default();
    let (lat, lon) = at.unwrap_or((home.lat, home.lon));
    let (lat, lon) = (lat.to_radians(), lon.to_radians());
    let gen = Generator::try_new(cfg)?;
    let c = tile_for_latlon(lat, lon, zoom);
    let n = size as i64;
    let max = 1i64 << zoom;
    let mut ids = Vec::new();
    for dy in 0..n {
        for dx in 0..n {
            let x = (c.x as i64 + dx - n / 2).rem_euclid(max);
            let y = (c.y as i64 + dy - n / 2).clamp(0, max - 1);
            ids.push((dx, dy, TileId::new(zoom, x as u32, y as u32)));
        }
    }
    let t0 = std::time::Instant::now();
    let data = gen.tiles(&ids.iter().map(|i| i.2).collect::<Vec<_>>())?;
    let tiles: Vec<(i64, i64, TileData)> = ids.iter().zip(data).map(|(&(dx, dy, _), t)| (dx, dy, t)).collect();
    let dt = t0.elapsed().as_secs_f64();
    eprintln!("generated {} tiles in {:.2}s ({:.3}s/tile wall)", tiles.len(), dt, dt / tiles.len() as f64);
    let emin = tiles.iter().map(|t| t.2.elev_min).fold(f32::MAX, f32::min);
    let emax = tiles.iter().map(|t| t.2.elev_max).fold(f32::MIN, f32::max);
    eprintln!("elevation range {emin:.1} .. {emax:.1} m");
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p)?;
    }
    let w = (n as usize) * TILE_SIZE;
    for layer in layers {
        let mut img = image::RgbImage::new(w as u32, w as u32);
        for (dx, dy, t) in &tiles {
            let px = layer_rgb(t, layer, emin, emax);
            for j in 0..TILE_SIZE {
                for i in 0..TILE_SIZE {
                    let k = 3 * (j * TILE_SIZE + i);
                    img.put_pixel((*dx as usize * TILE_SIZE + i) as u32, (*dy as usize * TILE_SIZE + j) as u32, image::Rgb([px[k], px[k + 1], px[k + 2]]));
                }
            }
        }
        let path = format!("{}_{layer}.png", out.display());
        img.save(&path)?;
        eprintln!("wrote {path}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_args_are_checked() {
        assert_eq!(check_args(14, 18, 4, "rgb,emission", None).unwrap(), vec!["rgb", "emission"]);
        assert!(check_args(18, 18, 1, "elevation", Some((-90.0, 180.0))).is_ok());
        assert!(check_args(31, 18, 4, "rgb", None).unwrap_err().to_string().contains("--zoom 31"));
        assert!(check_args(14, 18, 0, "rgb", None).is_err());
        assert!(check_args(14, 18, MAX_SIZE + 1, "rgb", None).is_err());
        let e = check_args(14, 18, 4, "rgb,bogus", None).unwrap_err().to_string();
        assert!(e.contains("\"bogus\"") && e.contains("hillshade"), "{e}");
        assert!(check_args(14, 18, 4, "rgb", Some((91.0, 0.0))).is_err());
    }
}
