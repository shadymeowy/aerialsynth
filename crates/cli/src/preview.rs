//! Mosaic preview of generated (or stored) tiles.

use anyhow::Result;
use clap::Args as ClapArgs;
use geodesy::tiles::{tile_for_latlon, TileId};
use rayon::prelude::*;
use std::path::PathBuf;
use terragen::{Config, Generator, TileData, TILE_SIZE};

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Generator config (YAML). Defaults are used if omitted.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Override the seed from the config.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Centre latitude (deg). Defaults to the config's home.
    #[arg(long, allow_hyphen_values = true)]
    pub lat: Option<f64>,
    /// Centre longitude (deg).
    #[arg(long, allow_hyphen_values = true)]
    pub lon: Option<f64>,
    #[arg(long, short)]
    pub zoom: u8,
    /// Mosaic size in tiles per side.
    #[arg(long, default_value_t = 4)]
    pub tiles: u32,
    /// Layers to write: rgb, albedo, elevation, normal, landcover, hillshade (comma separated).
    #[arg(long, default_value = "rgb")]
    pub layers: String,
    /// Output PNG prefix (a suffix `_<layer>.png` is appended).
    #[arg(long, short, default_value = "out/preview")]
    pub out: PathBuf,
}

pub fn load_config(path: &Option<PathBuf>, seed: Option<u64>) -> Result<Config> {
    let mut cfg = match path {
        Some(p) => Config::from_file(p)?,
        None => Config::default(),
    };
    if let Some(s) = seed {
        cfg.seed = s;
    }
    Ok(cfg)
}

pub fn layer_rgb(t: &TileData, layer: &str, emin: f32, emax: f32) -> Vec<u8> {
    let n = TILE_SIZE * TILE_SIZE;
    let mut out = vec![0u8; n * 3];
    for k in 0..n {
        let c: [u8; 3] = match layer {
            "rgb" => [t.rgb[3 * k], t.rgb[3 * k + 1], t.rgb[3 * k + 2]],
            "albedo" => [t.albedo[3 * k], t.albedo[3 * k + 1], t.albedo[3 * k + 2]],
            "normal" => [
                (t.normal[3 * k] as i32 + 128) as u8,
                (t.normal[3 * k + 1] as i32 + 128) as u8,
                (t.normal[3 * k + 2] as i32 + 128) as u8,
            ],
            "landcover" => terragen::landcover::palette(t.landcover[k]),
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

pub fn run(a: Args) -> Result<()> {
    let cfg = load_config(&a.config, a.seed)?;
    let home = cfg.home.clone().unwrap_or_default();
    let lat = a.lat.unwrap_or(home.lat).to_radians();
    let lon = a.lon.unwrap_or(home.lon).to_radians();
    let gen = Generator::new(cfg);
    let c = tile_for_latlon(lat, lon, a.zoom);
    let n = a.tiles as i64;
    let max = 1i64 << a.zoom;
    let mut ids = Vec::new();
    for dy in 0..n {
        for dx in 0..n {
            let x = (c.x as i64 + dx - n / 2).rem_euclid(max);
            let y = (c.y as i64 + dy - n / 2).clamp(0, max - 1);
            ids.push((dx, dy, TileId::new(a.zoom, x as u32, y as u32)));
        }
    }
    let t0 = std::time::Instant::now();
    let tiles: Vec<(i64, i64, TileData)> = ids.par_iter().map(|&(dx, dy, id)| (dx, dy, gen.tile(id))).collect();
    let dt = t0.elapsed().as_secs_f64();
    eprintln!("generated {} tiles in {:.2}s ({:.3}s/tile wall)", tiles.len(), dt, dt / tiles.len() as f64);
    let emin = tiles.iter().map(|t| t.2.elev_min).fold(f32::MAX, f32::min);
    let emax = tiles.iter().map(|t| t.2.elev_max).fold(f32::MIN, f32::max);
    eprintln!("elevation range {emin:.1} .. {emax:.1} m");
    if let Some(p) = a.out.parent() {
        std::fs::create_dir_all(p)?;
    }
    let w = (n as usize) * TILE_SIZE;
    for layer in a.layers.split(',') {
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
        let path = format!("{}_{}.png", a.out.display(), layer);
        img.save(&path)?;
        eprintln!("wrote {path}");
    }
    Ok(())
}
