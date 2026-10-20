//! Forests keep their brightness across the zooms where their crowns become explicit (z13–z16;
//! round 15: forests stepped from light to dark where a view passed from z14 to z15 tiles).
//!
//! The brightness is a shaded model of the renderer (`render::raster::pixel_shade`, relit) seen
//! from nadir: the albedo times direct sun on the tile's normal with the shadow cast by its DSM,
//! plus sky light, averaged over the area of one z15 tile, from that tile's ancestors (z13, z14)
//! and descendants (z16), compared with the mean of z15 and z16, where most crowns are drawn one
//! by one. The unresolved crowns' shading (`layers::canopy::crown_relief`) is a compromise
//! between nadir and oblique views (where resolved crowns hide their shadows and look less
//! dark), so a broadleaf forest's coarse zooms stay ~1.35x as bright as its fine ones from
//! nadir (round 14: 1.9x, with a step of 1.8x from z14 to z15), a conifer forest's within
//! ~12 % (round 14: 1.45x).
use geodesy::tiles::{gsd_ew, tile_for_latlon, TileId};
use terragen::{Config, NextGenerator, TileData, TILE_SIZE};

/// A grid of a zoom's pixels: linear albedo luminance, DSM height (m), normal (ENU).
struct Grid {
    n: usize,
    texel: f64,
    lum: Vec<f64>,
    h: Vec<f64>,
    nrm: Vec<[f64; 3]>,
}

fn lin(v: u8) -> f64 {
    let c = v as f64 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The pixels of zoom `z` over the area of the z15 tile `t`.
fn grid(g: &NextGenerator, t: TileId, z: u8, lat: f64) -> Grid {
    let n0 = TILE_SIZE;
    let texel = gsd_ew(lat, z, n0 as u32, &g.world.ell);
    let push = |gr: &mut Grid, td: &TileData, x0: usize, y0: usize, x1: usize, y1: usize, ox: usize, oy: usize| {
        for j in y0..y1 {
            for i in x0..x1 {
                let k = j * n0 + i;
                let o = (oy + j - y0) * gr.n + ox + i - x0;
                gr.lum[o] = 0.2126 * lin(td.albedo[3 * k]) + 0.7152 * lin(td.albedo[3 * k + 1]) + 0.0722 * lin(td.albedo[3 * k + 2]);
                gr.h[o] = td.elevation[k] as f64;
                let v = |c: usize| td.normal[3 * k + c] as f64 / 127.0;
                gr.nrm[o] = [v(0), v(1), v(2)];
            }
        }
    };
    if z <= 15 {
        let k = 15 - z;
        let a = TileId::new(z, t.x >> k, t.y >> k);
        let s = n0 >> k;
        let (x0, y0) = ((t.x as usize - ((a.x as usize) << k)) * s, (t.y as usize - ((a.y as usize) << k)) * s);
        let mut gr = Grid { n: s, texel, lum: vec![0.0; s * s], h: vec![0.0; s * s], nrm: vec![[0.0; 3]; s * s] };
        let td = g.tile(a);
        push(&mut gr, &td, x0, y0, x0 + s, y0 + s, 0, 0);
        gr
    } else {
        let k = z - 15;
        let m = 1usize << k;
        let s = n0 * m;
        let mut gr = Grid { n: s, texel, lum: vec![0.0; s * s], h: vec![0.0; s * s], nrm: vec![[0.0; 3]; s * s] };
        for dy in 0..m {
            for dx in 0..m {
                let td = g.tile(TileId::new(z, (t.x << k) + dx as u32, (t.y << k) + dy as u32));
                push(&mut gr, &td, 0, 0, n0, n0, dx * n0, dy * n0);
            }
        }
        gr
    }
}

/// Mean shaded luminance of a grid under a sun at azimuth / elevation (deg), with the
/// renderer's shadow ray (bias 0.4 texel + 0.3 m) over its DSM (up to 150 m; lit beyond the
/// grid).
fn shaded(gr: &Grid, az: f64, el: f64) -> f64 {
    let (az, el) = (az.to_radians(), el.to_radians());
    let sun = [az.sin() * el.cos(), az.cos() * el.cos(), el.sin()];
    let tan_e = el.tan();
    // grid units: x east, y south
    let dir = [az.sin(), -az.cos()];
    let n = gr.n;
    let h_at = |x: f64, y: f64| -> Option<f64> {
        let (x, y) = (x - 0.5, y - 0.5);
        if x < 0.0 || y < 0.0 || x >= (n - 1) as f64 || y >= (n - 1) as f64 {
            return None;
        }
        let (i, j) = (x.floor() as usize, y.floor() as usize);
        let (fx, fy) = (x - i as f64, y - j as f64);
        let v = |i: usize, j: usize| gr.h[j * n + i];
        Some((v(i, j) * (1.0 - fx) + v(i + 1, j) * fx) * (1.0 - fy) + (v(i, j + 1) * (1.0 - fx) + v(i + 1, j + 1) * fx) * fy)
    };
    let steps = (150.0 / gr.texel).ceil() as usize;
    let bias = 0.4 * gr.texel + 0.3;
    let mut acc = 0.0;
    for j in 0..n {
        for i in 0..n {
            let o = j * n + i;
            let nr = gr.nrm[o];
            let ndl = (nr[0] * sun[0] + nr[1] * sun[1] + nr[2] * sun[2]).max(0.0);
            let mut vis = 1.0;
            if ndl > 0.0 {
                let h0 = gr.h[o] + bias;
                for s in 1..=steps {
                    let d = s as f64;
                    match h_at(i as f64 + 0.5 + dir[0] * d, j as f64 + 0.5 + dir[1] * d) {
                        Some(hh) if hh > h0 + d * gr.texel * tan_e => {
                            vis = 0.0;
                            break;
                        }
                        Some(_) => {}
                        None => break,
                    }
                }
            }
            acc += gr.lum[o] * (1.25 * ndl * vis + 0.16 * (0.6 + 0.4 * nr[2]));
        }
    }
    acc / (n * n) as f64
}

#[test]
fn forest_brightness_across_zooms() {
    let g = NextGenerator::new(Config { tile_supersample: 1, ..Config::default() });
    let backend = g.backend_name();
    // a broadleaf forest (39 N) and a conifer forest (49 N), low and high sun
    let mut bad = vec![];
    // (the largest step between neighbouring zooms, the spread over z13-z16 of the z15-16 mean)
    for &(name, lat, lon, step_max, spread_max) in &[("broadleaf", 39.05f64, 28.48f64, 1.5, 1.6), ("conifer", 49.0, -151.0, 1.3, 1.18)] {
        let t = tile_for_latlon(lat.to_radians(), lon.to_radians(), 15);
        let grids: Vec<(u8, Grid)> = (13u8..=16).map(|z| (z, grid(&g, t, z, lat.to_radians()))).collect();
        for &(az, el) in &[(110.0, 30.0), (180.0, 65.0)] {
            let b: Vec<(u8, f64)> = grids.iter().map(|(z, gr)| (*z, shaded(gr, az, el))).collect();
            let fine = 0.5 * (b[2].1 + b[3].1);
            let line: Vec<String> = b.iter().map(|(z, v)| format!("z{z} {v:.4} ({:.3})", v / fine)).collect();
            println!("{backend} {name} sun {el}°: {}", line.join("  "));
            for w in b.windows(2) {
                let step = (w[0].1 / w[1].1).max(w[1].1 / w[0].1);
                if step > step_max {
                    bad.push(format!("{name} sun {el}° z{} -> z{}: step {step:.3}", w[0].0, w[1].0));
                }
            }
            for (z, v) in &b {
                if (v / fine).max(fine / v) > spread_max {
                    bad.push(format!("{name} sun {el}° z{z}: {:.3} of z15-16's", v / fine));
                }
            }
        }
    }
    assert!(bad.is_empty(), "{backend}: forest brightness off across zooms: {bad:?}");
}
