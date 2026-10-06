//! Tile generation: combines pass A (per pixel centre, with a 2-pixel apron) and pass B
//! (supersampled detail, 1-pixel apron), then derives normals and the baked "satellite" rgb.
//! Pass B interpolates pass A between pixel centres, so its apron pixels need pass-A values one
//! pixel further out; the normals of the edge pixels use pass B's apron.

use crate::config::Config;
use crate::surface::{l2s, Caches, Local, SurfaceModel};
use crate::world::{water, Ctx, Macro, Terrain, World};
use geodesy::tiles::{gsd_ew, gsd_ns, pixel_to_latlon, TileId};
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
        let pf = self.surface.pixel_fields(ctx.p, gsd);
        let s = self.surface.eval(&self.world, &mut caches, &ctx, &local, &pf);
        (t, s.height, s.class)
    }

    /// Generate one tile (parallel over rows internally).
    pub fn tile(&self, id: TileId) -> TileData {
        let n = TILE_SIZE;
        let na = n + 2; // pass B, 1-px apron
        let na2 = n + 4; // pass A, 2-px apron
        let z = id.z;
        let ell = self.world.ell;
        let ss = self.world.cfg.supersample.max(1) as usize;
        let ox = id.x as f64 * n as f64;
        let oy = id.y as f64 * n as f64;

        let prof = std::env::var_os("TERRAGEN_PROFILE").is_some();
        let t_start = std::time::Instant::now();
        // ---------------- macro fields: on a coarse grid aligned to global multiples of 16 px
        // (shared by neighbouring tiles → seamless), or exactly at low zooms where the grid would
        // be too coarse for the macro wavelengths.
        let (lat_c, _) = pixel_to_latlon(DVec2::new(ox + 128.0, oy + 128.0), z, n as u32);
        let use_grid = 16.0 * gsd_ew(lat_c, z, n as u32, &ell) <= 2000.0;
        const G: f64 = 16.0;
        let gk0x = (ox / G) as i64 - 1;
        let gk0y = (oy / G) as i64 - 1;
        let ng = (n as f64 / G) as usize + 3;
        let macro_grid: Vec<Macro> = if use_grid {
            (0..ng * ng)
                .into_par_iter()
                .map(|k| {
                    let gx = (gk0x + (k % ng) as i64) as f64 * G;
                    let gy = (gk0y + (k / ng) as i64) as f64 * G;
                    let (lat, lon) = pixel_to_latlon(DVec2::new(gx, gy), z, n as u32);
                    let gsd = gsd_ew(lat, z, n as u32, &ell);
                    self.world.macro_at(Ctx::new(lat, lon, gsd, &ell).p, gsd)
                })
                .collect()
        } else {
            vec![]
        };
        let macro_at = |px: f64, py: f64, ctx: &Ctx| -> Macro {
            if !use_grid {
                return self.world.macro_at(ctx.p, ctx.gsd);
            }
            let u = px / G - gk0x as f64;
            let v = py / G - gk0y as f64;
            let (i0, j0) = ((u.floor() as usize).min(ng - 2), (v.floor() as usize).min(ng - 2));
            let (fx, fy) = (u - i0 as f64, v - j0 as f64);
            let g = |i: usize, j: usize| &macro_grid[j * ng + i];
            let mut m = Macro::bilerp(g(i0, j0), g(i0 + 1, j0), g(i0, j0 + 1), g(i0 + 1, j0 + 1), fx, fy);
            m.mtn_warp = self.world.mtn_warp_at(ctx.p, ctx.gsd);
            m
        };

        // ---------------- drainage segments that can affect this tile
        let segs = {
            let (lat_c, lon_c) = pixel_to_latlon(DVec2::new(ox + 128.0, oy + 128.0), z, n as u32);
            let c = Ctx::new(lat_c, lon_c, 1.0, &ell);
            let (lat0, lon0) = pixel_to_latlon(DVec2::new(ox - 2.0, oy - 2.0), z, n as u32);
            let corner = Ctx::new(lat0, lon0, 1.0, &ell).p;
            let gsd_c = gsd_ew(lat_c, z, n as u32, &ell);
            self.world.river_segments(c.p, (corner - c.p).length(), gsd_c)
        };

        // ---------------- pass A on pixel centres incl. 2px apron
        let rows_a: Vec<Vec<Terrain>> = (0..na2)
            .into_par_iter()
            .map(|j| {
                let mut row = Vec::with_capacity(na2);
                let py = oy + j as f64 - 2.0 + 0.5;
                let (lat, _) = pixel_to_latlon(DVec2::new(ox, py), z, n as u32);
                let gsd = gsd_ew(lat, z, n as u32, &ell);
                for i in 0..na2 {
                    let px = ox + i as f64 - 2.0 + 0.5;
                    let (lat, lon) = pixel_to_latlon(DVec2::new(px, py), z, n as u32);
                    let ctx = Ctx::new(lat, lon, gsd, &ell);
                    let m = macro_at(px, py, &ctx);
                    row.push(self.world.terrain_with(&ctx, &m, &segs));
                }
                row
            })
            .collect();
        let grid: Vec<Terrain> = rows_a.into_iter().flatten().collect();
        // pass-A value at pass-B grid coordinates (i, j) ∈ [-1, na]
        let at = |i: isize, j: isize| -> &Terrain {
            let i = (i + 1).clamp(0, na2 as isize - 1) as usize;
            let j = (j + 1).clamp(0, na2 as isize - 1) as usize;
            &grid[j * na2 + i]
        };

        // per-row pixel size, east-west and north-south (mercator scale varies with latitude;
        // on the ellipsoid the N-S size is smaller by M/N)
        let row_lat: Vec<f64> = (0..na)
            .map(|j| {
                let py = oy + j as f64 - 1.0 + 0.5;
                pixel_to_latlon(DVec2::new(ox, py), z, n as u32).0
            })
            .collect();
        let row_gsd: Vec<f64> = row_lat.iter().map(|&lat| gsd_ew(lat, z, n as u32, &ell)).collect();
        let row_gsd_ns: Vec<f64> = row_lat.iter().map(|&lat| gsd_ns(lat, z, n as u32, &ell)).collect();

        // ---------------- slope of the bare ground at pixel scale
        let slope: Vec<f64> = (0..na * na)
            .map(|k| {
                let (i, j) = ((k % na) as isize, (k / na) as isize);
                let dx = (at(i + 1, j).ground - at(i - 1, j).ground) / (2.0 * row_gsd[j as usize]);
                let dy = (at(i, j + 1).ground - at(i, j - 1).ground) / (2.0 * row_gsd_ns[j as usize]);
                (dx * dx + dy * dy).sqrt()
            })
            .collect();

        let t_a = t_start.elapsed().as_secs_f64();
        // ---------------- pass B, supersampled, on the apron grid
        struct PixB {
            emission: DVec3,
            albedo: DVec3,
            height: f64,
            /// bare ground under the pixel (for the canopy clean-up below)
            ground: f64,
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
                    let pf = {
                        let px = ox + i as f64 - 1.0 + 0.5;
                        let py = oy + j as f64 - 1.0 + 0.5;
                        let (lat, lon) = pixel_to_latlon(DVec2::new(px, py), z, n as u32);
                        self.surface.pixel_fields(Ctx::new(lat, lon, gsd, &ell).p, gsd)
                    };
                    let mut acc_a = DVec3::ZERO;
                    let mut acc_e = DVec3::ZERO;
                    let mut acc_h = 0.0;
                    let mut acc_l = 0.0;
                    let mut acc_g = 0.0;
                    let mut counts = [0u16; 32];
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
                            acc_g += ground;
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
                            let s = self.surface.eval(&self.world, &mut caches, &ctx, &local, &pf);
                            acc_a += s.albedo;
                            acc_e += s.emission;
                            acc_h += s.height;
                            acc_l += s.lit;
                            counts[(s.class as usize).min(31)] += 1;
                        }
                    }
                    let inv = 1.0 / (ss * ss) as f64;
                    let class = counts.iter().enumerate().max_by_key(|(_, c)| **c).map(|(k, _)| k as u8).unwrap_or(0);
                    row.push(PixB { emission: acc_e * inv, albedo: acc_a * inv, height: acc_h * inv, ground: acc_g * inv, lit: acc_l * inv, class });
                }
                row
            })
            .collect();
        let mut pb: Vec<PixB> = rows_b.into_iter().flatten().collect();

        // ---------------- canopy clean-up: morphological opening of the height above ground with
        // a ~1 m radius. Tree / land-use densities are evaluated per pixel, so wherever an input
        // mask changes sharply (slope stripes on gully walls, field edges) a crown can be clipped
        // into a sliver: a needle in the DSM. Whole crowns, hedges and buildings are wider than
        // the structuring element and survive.
        let r_open = (0.9 / row_gsd[na / 2]).floor() as usize;
        if r_open >= 1 {
            let r = r_open.min(4);
            let canopy: Vec<f64> = pb.iter().map(|p| (p.height - p.ground).max(0.0)).collect();
            let filt = |src: &[f64], max: bool| -> Vec<f64> {
                // separable square min / max filter over (2r+1)^2, clamped at the grid border
                let pick = |a: f64, b: f64| if max { a.max(b) } else { a.min(b) };
                let mut tmp = vec![0.0; na * na];
                for j in 0..na {
                    for i in 0..na {
                        let (lo, hi) = (i.saturating_sub(r), (i + r).min(na - 1));
                        tmp[j * na + i] = (lo..=hi).map(|k| src[j * na + k]).fold(if max { f64::MIN } else { f64::MAX }, pick);
                    }
                }
                let mut out = vec![0.0; na * na];
                for j in 0..na {
                    for i in 0..na {
                        let (lo, hi) = (j.saturating_sub(r), (j + r).min(na - 1));
                        out[j * na + i] = (lo..=hi).map(|k| tmp[k * na + i]).fold(if max { f64::MIN } else { f64::MAX }, pick);
                    }
                }
                out
            };
            let opened = filt(&filt(&canopy, false), true);
            for (k, p) in pb.iter_mut().enumerate() {
                if opened[k] < canopy[k] - 0.5 {
                    p.height = p.ground + opened[k];
                }
            }
        }

        if prof {
            eprintln!("tile {id}: pass A {:.3}s, pass B {:.3}s", t_a, t_start.elapsed().as_secs_f64() - t_a);
        }
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
        let mut emission = vec![0u8; n * n * 3];
        let mut emin = f32::MAX;
        let mut emax = f32::MIN;
        let hb = |i: usize, j: usize| pb[j * na + i].height;
        for j in 0..n {
            let (g, gn) = (row_gsd[j + 1], row_gsd_ns[j + 1]);
            for i in 0..n {
                let (ia, ja) = (i + 1, j + 1);
                let px = &pb[ja * na + ia];
                let dhdx = (hb(ia + 1, ja) - hb(ia - 1, ja)) / (2.0 * g);
                let dhdn = -(hb(ia, ja + 1) - hb(ia, ja - 1)) / (2.0 * gn);
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
                let alb = {
                    let a = px.albedo;
                    let lum = 0.2126 * a.x + 0.7152 * a.y + 0.0722 * a.z;
                    ((DVec3::splat(lum) + (a - DVec3::splat(lum)) * look.albedo_saturation) * look.albedo_brightness).max(DVec3::ZERO)
                };
                let mut c = alb * (light / l0) * look.exposure;
                c = c * (1.0 - look.haze) + haze_col * look.haze;
                for ch in 0..3 {
                    albedo[3 * k + ch] = (l2s(alb[ch]) * 255.0).round() as u8;
                    rgb[3 * k + ch] = (l2s(c[ch]) * 255.0).round() as u8;
                    emission[3 * k + ch] = ((px.emission[ch] / 4.0).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                }
            }
        }
        TileData { id, rgb, albedo, elevation, normal, landcover, emission, elev_min: emin, elev_max: emax }
    }
}
