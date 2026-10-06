//! Tile generation: combines pass A (per pixel centre, with a 1-pixel apron) and pass B
//! (supersampled detail), then derives normals and the baked "satellite" rgb.

use crate::config::Config;
use crate::surface::{l2s, Caches, Local, SurfaceModel};
use crate::world::{water, Ctx, Terrain, World};
use geodesy::tiles::{gsd_ew, pixel_to_latlon, TileId};
use glam::{DVec2, DVec3};
use rayon::prelude::*;

pub use tilestore::TILE_SIZE;

pub use tilestore::TileData;

pub struct Generator {
    pub world: World,
    pub surface: SurfaceModel,
}

#[inline]
fn bilerp(v: [f64; 4], fx: f64, fy: f64) -> f64 {
    let a = v[0] + (v[1] - v[0]) * fx;
    let b = v[2] + (v[3] - v[2]) * fx;
    a + (b - a) * fy
}

impl Generator {
    pub fn new(cfg: Config) -> Self {
        let world = World::new(cfg);
        let surface = SurfaceModel::new(&world);
        Generator { world, surface }
    }

    pub fn config(&self) -> &Config {
        &self.world.cfg
    }

    /// Evaluate pass A + B at a single geodetic point (radians) for a given pixel size.
    /// Useful for planning/inspection. Returns (pass A terrain, DSM height, landcover).
    pub fn probe(&self, lat: f64, lon: f64, gsd: f64) -> (Terrain, f64, u8) {
        let ell = self.world.ell;
        let ctx = Ctx::new(lat, lon, gsd, &ell);
        let t = self.world.terrain(&ctx);
        let local = Local {
            t: &t,
            ground: t.ground,
            water: t.water,
            water_kind: t.water_kind,
            river_d: t.river_d.min(1e7),
            river_hw: t.river_hw,
            river_level: t.river_level,
            road_major: t.road_major.min(1e7),
            road_minor: t.road_minor.min(1e7),
            slope: 0.0,
            fw: gsd,
        };
        let mut caches = Caches::default();
        let s = self.surface.eval(&self.world, &mut caches, &ctx, &local);
        (t, s.height, s.class)
    }

    /// Generate one tile (parallel over rows internally).
    pub fn tile(&self, id: TileId) -> TileData {
        let n = TILE_SIZE;
        let na = n + 2; // with apron
        let z = id.z;
        let ell = self.world.ell;
        let ss = self.world.cfg.supersample.max(1) as usize;
        let ox = id.x as f64 * n as f64;
        let oy = id.y as f64 * n as f64;

        // ---------------- pass A on pixel centres incl. 1px apron
        let rows_a: Vec<Vec<Terrain>> = (0..na)
            .into_par_iter()
            .map(|j| {
                let mut row = Vec::with_capacity(na);
                let py = oy + j as f64 - 1.0 + 0.5;
                let (lat, _) = pixel_to_latlon(DVec2::new(ox, py), z, n as u32);
                let gsd = gsd_ew(lat, z, n as u32, &ell);
                for i in 0..na {
                    let px = ox + i as f64 - 1.0 + 0.5;
                    let (lat, lon) = pixel_to_latlon(DVec2::new(px, py), z, n as u32);
                    let ctx = Ctx::new(lat, lon, gsd, &ell);
                    row.push(self.world.terrain(&ctx));
                }
                row
            })
            .collect();
        let grid: Vec<Terrain> = rows_a.into_iter().flatten().collect();
        let at = |i: isize, j: isize| -> &Terrain {
            let i = i.clamp(0, na as isize - 1) as usize;
            let j = j.clamp(0, na as isize - 1) as usize;
            &grid[j * na + i]
        };

        // per-row gsd (mercator scale varies with latitude)
        let row_gsd: Vec<f64> = (0..na)
            .map(|j| {
                let py = oy + j as f64 - 1.0 + 0.5;
                let (lat, _) = pixel_to_latlon(DVec2::new(ox, py), z, n as u32);
                gsd_ew(lat, z, n as u32, &ell)
            })
            .collect();

        // ---------------- slope of the bare ground at pixel scale
        let slope: Vec<f64> = (0..na * na)
            .map(|k| {
                let (i, j) = ((k % na) as isize, (k / na) as isize);
                let g = row_gsd[j as usize];
                let dx = (at(i + 1, j).ground - at(i - 1, j).ground) / (2.0 * g);
                let dy = (at(i, j + 1).ground - at(i, j - 1).ground) / (2.0 * g);
                (dx * dx + dy * dy).sqrt()
            })
            .collect();

        // ---------------- pass B, supersampled, on the apron grid
        struct PixB {
            albedo: DVec3,
            height: f64,
            lit: f64,
            class: u8,
        }
        let rows_b: Vec<Vec<PixB>> = (0..na)
            .into_par_iter()
            .map(|j| {
                let mut caches = Caches::default();
                let mut row = Vec::with_capacity(na);
                let gsd = row_gsd[j];
                let fw = gsd / ss as f64;
                for i in 0..na {
                    let mut acc_a = DVec3::ZERO;
                    let mut acc_h = 0.0;
                    let mut acc_l = 0.0;
                    let mut counts = [0u8; 32];
                    for sy in 0..ss {
                        for sx in 0..ss {
                            let fxo = (sx as f64 + 0.5) / ss as f64 - 0.5;
                            let fyo = (sy as f64 + 0.5) / ss as f64 - 0.5;
                            // neighbours for bilinear interpolation
                            let (ii, jj) = (i as isize, j as isize);
                            let i0 = if fxo < 0.0 { ii - 1 } else { ii };
                            let j0 = if fyo < 0.0 { jj - 1 } else { jj };
                            let fx = if fxo < 0.0 { 1.0 + fxo } else { fxo };
                            let fy = if fyo < 0.0 { 1.0 + fyo } else { fyo };
                            let nb = [at(i0, j0), at(i0 + 1, j0), at(i0, j0 + 1), at(i0 + 1, j0 + 1)];
                            let t = at(ii, jj);
                            let ground = bilerp(nb.map(|t| t.ground), fx, fy);
                            let mut wl = f64::NEG_INFINITY;
                            let mut wk = water::NONE;
                            for t in nb {
                                if t.water_kind != water::NONE && t.water > wl {
                                    wl = t.water;
                                    wk = t.water_kind;
                                }
                            }
                            let rd = bilerp(nb.map(|t| t.river_d.clamp(-1e6, 1e6)), fx, fy);
                            let rhw = nb.iter().map(|t| t.river_hw).fold(0.0, f64::max);
                            let rl = bilerp(nb.map(|t| if t.river_hw > 0.0 { t.river_level } else { ground }), fx, fy);
                            let road_major = bilerp(nb.map(|t| t.road_major.clamp(-1e6, 1e6)), fx, fy);
                            let road_minor = bilerp(nb.map(|t| t.road_minor.clamp(-1e6, 1e6)), fx, fy);
                            let mut tt = *t;
                            tt.region.edge = bilerp(nb.map(|t| t.region.edge.min(1e6)), fx, fy);
                            let local = Local {
                                t: &tt,
                                ground,
                                water: wl,
                                water_kind: wk,
                                river_d: rd,
                                river_hw: rhw,
                                river_level: rl,
                                road_major,
                                road_minor,
                                slope: slope[j * na + i],
                                fw,
                            };
                            let px = ox + i as f64 - 1.0 + 0.5 + fxo;
                            let py = oy + j as f64 - 1.0 + 0.5 + fyo;
                            let (lat, lon) = pixel_to_latlon(DVec2::new(px, py), z, n as u32);
                            let ctx = Ctx::new(lat, lon, gsd, &ell);
                            let s = self.surface.eval(&self.world, &mut caches, &ctx, &local);
                            acc_a += s.albedo;
                            acc_h += s.height;
                            acc_l += s.lit;
                            counts[(s.class as usize).min(31)] += 1;
                        }
                    }
                    let inv = 1.0 / (ss * ss) as f64;
                    let class = counts.iter().enumerate().max_by_key(|(_, c)| **c).map(|(k, _)| k as u8).unwrap_or(0);
                    row.push(PixB { albedo: acc_a * inv, height: acc_h * inv, lit: acc_l * inv, class });
                }
                row
            })
            .collect();
        let pb: Vec<PixB> = rows_b.into_iter().flatten().collect();

        // ---------------- outputs
        let look = &self.world.cfg.look;
        let az = look.sun_azimuth_deg.to_radians();
        let el = look.sun_elevation_deg.to_radians();
        let sun = DVec3::new(az.sin() * el.cos(), az.cos() * el.cos(), el.sin());
        let l0 = look.ambient + look.direct * el.sin();
        let haze_col = DVec3::new(0.50, 0.58, 0.70);

        let mut rgb = vec![0u8; n * n * 3];
        let mut albedo = vec![0u8; n * n * 3];
        let mut elevation = vec![0f32; n * n];
        let mut normal = vec![0i8; n * n * 3];
        let mut landcover = vec![0u8; n * n];
        let mut emin = f32::MAX;
        let mut emax = f32::MIN;
        let hb = |i: usize, j: usize| pb[j * na + i].height;
        for j in 0..n {
            let g = row_gsd[j + 1];
            for i in 0..n {
                let (ia, ja) = (i + 1, j + 1);
                let px = &pb[ja * na + ia];
                let dhdx = (hb(ia + 1, ja) - hb(ia - 1, ja)) / (2.0 * g);
                let dhdn = -(hb(ia, ja + 1) - hb(ia, ja - 1)) / (2.0 * g);
                let nrm = DVec3::new(-dhdx, -dhdn, 1.0).normalize();
                let k = j * n + i;
                let h = px.height as f32;
                elevation[k] = h;
                emin = emin.min(h);
                emax = emax.max(h);
                landcover[k] = px.class;
                normal[3 * k] = (nrm.x * 127.0).round() as i8;
                normal[3 * k + 1] = (nrm.y * 127.0).round() as i8;
                normal[3 * k + 2] = (nrm.z * 127.0).round() as i8;
                let light = look.ambient * (0.55 + 0.45 * nrm.z) + look.direct * nrm.dot(sun).max(0.0) * px.lit;
                let mut c = px.albedo * (light / l0) * look.exposure;
                c = c * (1.0 - look.haze) + haze_col * look.haze;
                for ch in 0..3 {
                    albedo[3 * k + ch] = (l2s(px.albedo[ch]) * 255.0).round() as u8;
                    rgb[3 * k + ch] = (l2s(c[ch]) * 255.0).round() as u8;
                }
            }
        }
        TileData { id, rgb, albedo, elevation, normal, landcover, elev_min: emin, elev_max: emax }
    }
}
