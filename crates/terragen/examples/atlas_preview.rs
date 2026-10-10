//! Equirectangular previews of the planetary atlas, one PNG per field, and a Köppen map:
//!
//!     cargo run --release -p terragen --example atlas_preview -- OUT_DIR [SEED] [CONFIG.yaml]
//!
//! Also prints the build time, the land share of each Köppen class and how well the world's
//! mountain belts (`world.rs` noise) line up with the convergent plate boundaries.

use glam::DVec3;
use rayon::prelude::*;
use terragen::atlas::{koppen_with_lapse, Atlas, AtlasSample, Boundary, Koppen};
use terragen::world::World;
use terragen::Config;

const W: usize = 1440;
const H: usize = 720;

type Rgb = [u8; 3];

/// Piecewise-linear colour map over (value, colour) stops.
fn ramp(stops: &[(f64, Rgb)], v: f64) -> Rgb {
    if v <= stops[0].0 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        let ((a, ca), (b, cb)) = (w[0], w[1]);
        if v <= b {
            let t = (v - a) / (b - a);
            return [0, 1, 2].map(|i| (ca[i] as f64 + (cb[i] as f64 - ca[i] as f64) * t).round() as u8);
        }
    }
    stops[stops.len() - 1].1
}

fn mix(a: Rgb, b: Rgb, t: f64) -> Rgb {
    [0, 1, 2].map(|i| (a[i] as f64 + (b[i] as f64 - a[i] as f64) * t.clamp(0.0, 1.0)).round() as u8)
}

fn hash_colour(x: u64) -> Rgb {
    let h = terragen::noise::mix64(x.wrapping_add(0x51));
    [(60 + (h & 0x9F)) as u8, (60 + ((h >> 8) & 0x9F)) as u8, (60 + ((h >> 16) & 0x9F)) as u8]
}

const VIRIDIS: [(f64, Rgb); 5] = [(0.0, [68, 1, 84]), (0.25, [59, 82, 139]), (0.5, [33, 145, 140]), (0.75, [94, 201, 98]), (1.0, [253, 231, 37])];

struct Px {
    s: AtlasSample,
    amp_m: f64,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = std::path::PathBuf::from(args.first().map(String::as_str).unwrap_or("atlas_preview"));
    let mut cfg = match args.get(2) {
        Some(p) => Config::from_file(std::path::Path::new(p))?,
        None => Config::default(),
    };
    if let Some(s) = args.get(1) {
        cfg.seed = s.parse()?;
    }
    cfg.validate()?;
    std::fs::create_dir_all(&out)?;
    let world = World::new(cfg.clone());
    let t = std::time::Instant::now();
    let atlas = Atlas::build(&world);
    println!(
        "seed {}: atlas {}² × 6 built in {:.2} s ({} threads), {} MB, {} cultures",
        cfg.seed,
        atlas.resolution(),
        t.elapsed().as_secs_f64(),
        rayon::current_num_threads(),
        (atlas.words().len() * 4) >> 20,
        atlas.cultures().len()
    );
    let lapse = cfg.climate.lapse_rate_c_per_km;
    let gsd = std::f64::consts::FRAC_PI_2 / atlas.resolution() as f64 * world.ell.a;
    let dir_of = |x: usize, y: usize| {
        let lat = 90.0 - (y as f64 + 0.5) * 180.0 / H as f64;
        let lon = -180.0 + (x as f64 + 0.5) * 360.0 / W as f64;
        geodesy::geodetic2ecef(geodesy::Geodetic::from_deg(lat, lon, 0.0), &world.ell)
    };
    let px: Vec<Px> = (0..W * H)
        .into_par_iter()
        .map(|k| {
            let p = dir_of(k % W, k / W);
            Px { s: atlas.sample(p), amp_m: world.smooth_relief(p, gsd).1 }
        })
        .collect();
    let lat_of = |y: usize| 90.0 - (y as f64 + 0.5) * 180.0 / H as f64;
    let land = |k: usize| px[k].s.coast_km > 0.0;
    // coastline: a land pixel next to a sea pixel
    let coastline = |k: usize| {
        let (x, y) = (k % W, k / W);
        let nb = [((x + 1) % W, y), ((x + W - 1) % W, y), (x, y.saturating_sub(1)), (x, (y + 1).min(H - 1))];
        land(k) && nb.iter().any(|&(a, b)| !land(b * W + a))
    };
    let save = |name: &str, f: &(dyn Fn(usize) -> Rgb + Sync), coast: bool| -> anyhow::Result<()> {
        let buf: Vec<u8> = (0..W * H)
            .into_par_iter()
            .flat_map_iter(|k| {
                let c = if coast && coastline(k) { [20, 20, 20] } else { f(k) };
                c.into_iter()
            })
            .collect();
        image::RgbImage::from_raw(W as u32, H as u32, buf).unwrap().save(out.join(format!("{name}.png")))?;
        Ok(())
    };
    // hillshade of the smooth elevation (from the east-west and north-south differences)
    let shade = |k: usize| {
        let (x, y) = (k % W, k / W);
        let e = |a: usize, b: usize| px[b * W + a].s.elevation_m.max(0.0);
        let dx = e((x + 1) % W, y) - e((x + W - 1) % W, y);
        let dy = e(x, y.saturating_sub(1)) - e(x, (y + 1).min(H - 1));
        (1.0 - 0.004 * (dx + dy)).clamp(0.55, 1.35)
    };
    let shaded = |c: Rgb, s: f64| c.map(|v| (v as f64 * s).clamp(0.0, 255.0) as u8);
    let hyps = |h: f64| {
        if h < 0.0 {
            ramp(&[(-5000.0, [10, 30, 80]), (-1000.0, [30, 70, 140]), (0.0, [90, 150, 200])], h)
        } else {
            ramp(&[(0.0, [90, 140, 70]), (500.0, [150, 170, 90]), (1200.0, [180, 150, 100]), (2500.0, [140, 110, 90]), (4000.0, [250, 250, 250])], h)
        }
    };
    save("elevation", &|k| if land(k) { shaded(hyps(px[k].s.elevation_m), shade(k)) } else { hyps(px[k].s.elevation_m) }, false)?;
    save(
        "coast_distance",
        &|k| {
            let d = px[k].s.coast_km;
            let c = ramp(&[(-2500.0, [20, 40, 120]), (0.0, [240, 240, 240]), (2500.0, [140, 40, 20])], d);
            if (d / 500.0).fract().abs() < 0.04 {
                mix(c, [0, 0, 0], 0.5)
            } else {
                c
            }
        },
        true,
    )?;
    // wind: speed with direction ticks
    let mut ticks = vec![false; W * H];
    for y in (12..H).step_by(24) {
        for x in (12..W).step_by(24) {
            let s = &px[y * W + x].s;
            let sp = (s.wind_e * s.wind_e + s.wind_n * s.wind_n).sqrt().max(1e-6);
            for t in 0..=(sp * 1.6) as i64 {
                let xx = (x as f64 + s.wind_e / sp * t as f64 / (lat_of(y).to_radians().cos().max(0.2))).round() as i64;
                let yy = (y as f64 - s.wind_n / sp * t as f64).round() as i64;
                if (0..W as i64).contains(&xx) && (0..H as i64).contains(&yy) {
                    ticks[yy as usize * W + xx as usize] = true;
                }
            }
        }
    }
    save(
        "wind",
        &|k| {
            if ticks[k] {
                return [0, 0, 0];
            }
            let s = &px[k].s;
            ramp(&VIRIDIS, (s.wind_e * s.wind_e + s.wind_n * s.wind_n).sqrt() / 12.0)
        },
        true,
    )?;
    let precip_map = [
        (0.0, [120, 70, 30]),
        (150.0, [200, 150, 80]),
        (400.0, [240, 220, 120]),
        (800.0, [120, 200, 90]),
        (1600.0, [30, 140, 70]),
        (3000.0, [20, 90, 160]),
        (5000.0, [60, 20, 120]),
    ];
    save("precipitation", &|k| ramp(&precip_map, px[k].s.precip_mm), true)?;
    let temp_map =
        [(-30.0, [80, 0, 120]), (-10.0, [40, 80, 220]), (0.0, [200, 230, 255]), (10.0, [140, 220, 120]), (20.0, [250, 210, 60]), (30.0, [200, 40, 20])];
    save("temperature", &|k| ramp(&temp_map, px[k].s.temp_c - if land(k) { lapse * px[k].s.elevation_m / 1000.0 } else { 0.0 }), true)?;
    save("temperature_sea_level", &|k| ramp(&temp_map, px[k].s.temp_c), true)?;
    save("temperature_range", &|k| ramp(&VIRIDIS, px[k].s.temp_range_c / 50.0), true)?;
    save("precip_regime", &|k| ramp(&[(-1.0, [170, 90, 20]), (0.0, [240, 240, 240]), (1.0, [20, 120, 60])], px[k].s.regime), true)?;
    let boundary_colour = |b: Boundary| match b {
        Boundary::Overriding => [230, 30, 30],
        Boundary::Subducting => [120, 0, 0],
        Boundary::Collision => [255, 140, 0],
        Boundary::IslandArc => [255, 0, 200],
        Boundary::Rift => [0, 200, 255],
        Boundary::Ridge => [0, 90, 255],
        Boundary::Transform => [255, 255, 255],
        Boundary::None => [0, 0, 0],
    };
    let line_km = 0.9 * gsd / 1000.0 + 15.0;
    save(
        "plates",
        &|k| {
            let s = &px[k].s;
            if s.plate_dist_km < line_km {
                return boundary_colour(s.boundary);
            }
            if s.hotspot {
                return [255, 255, 160];
            }
            let c = hash_colour(s.plate as u64);
            if land(k) {
                c
            } else {
                mix(c, [0, 0, 40], 0.45)
            }
        },
        true,
    )?;
    save(
        "tectonics",
        &|k| {
            let s = &px[k].s;
            let c = ramp(&[(-0.6, [20, 60, 200]), (0.0, [235, 235, 235]), (1.0, [120, 60, 10])], s.uplift);
            mix(c, [255, 0, 0], s.volcanism)
        },
        true,
    )?;
    save("convergence", &|k| ramp(&[(-80.0, [0, 120, 255]), (0.0, [240, 240, 240]), (80.0, [220, 30, 0])], px[k].s.convergence_mm_yr), true)?;
    let litho_colour = [[200, 170, 120], [230, 230, 210], [200, 90, 110], [60, 60, 60], [240, 210, 100]];
    save(
        "lithology",
        &|k| {
            let s = &px[k].s;
            let c = mix(litho_colour[s.litho as usize], litho_colour[s.litho2 as usize], s.litho2_frac);
            if land(k) {
                c
            } else {
                mix(c, [0, 0, 60], 0.7)
            }
        },
        true,
    )?;
    save("glaciation", &|k| mix(if land(k) { [110, 120, 100] } else { [30, 50, 90] }, [255, 255, 255], px[k].s.glaciation), true)?;
    save(
        "culture",
        &|k| {
            let s = &px[k].s;
            let c = hash_colour(s.culture as u64 * 7919);
            if land(k) {
                c
            } else {
                mix(c, [0, 0, 40], 0.75)
            }
        },
        true,
    )?;
    let arch_colour: [Rgb; 12] = [
        [0, 120, 40],
        [120, 200, 0],
        [210, 180, 60],
        [240, 220, 150],
        [190, 150, 90],
        [160, 160, 40],
        [60, 160, 200],
        [230, 120, 40],
        [0, 200, 160],
        [40, 80, 60],
        [210, 220, 255],
        [150, 90, 160],
    ];
    save("archetype", &|k| if land(k) { arch_colour[px[k].s.archetype as usize] } else { [20, 30, 60] }, true)?;
    save("development", &|k| if land(k) { ramp(&VIRIDIS, px[k].s.development) } else { [20, 30, 60] }, true)?;
    save("population", &|k| if land(k) { ramp(&VIRIDIS, px[k].s.population) } else { [20, 30, 60] }, true)?;
    let kp = |k: usize| koppen_with_lapse(&px[k].s, px[k].s.elevation_m, lapse);
    save("koppen", &|k| if land(k) { kp(k).colour() } else { [25, 35, 70] }, true)?;
    // the world's mountain belts (`world.rs`) against the plate boundaries
    let mh = cfg.relief.mountain_height_m.max(1.0);
    save(
        "mountains_vs_plates",
        &|k| {
            let s = &px[k].s;
            if s.plate_dist_km < line_km && s.boundary.is_convergent() {
                return boundary_colour(s.boundary);
            }
            let base = if land(k) { [200, 200, 190] } else { [40, 60, 100] };
            mix(base, [90, 40, 10], px[k].amp_m / mh)
        },
        false,
    )?;

    // statistics: area-weighted over the land
    let wt = |k: usize| lat_of(k / W).to_radians().cos();
    let land_w: f64 = (0..W * H).filter(|&k| land(k)).map(wt).sum();
    let total_w: f64 = (0..W * H).map(wt).sum();
    println!("land {:.1} % of the surface", 100.0 * land_w / total_w);
    let mut shares: Vec<(Koppen, f64)> = Koppen::ALL.iter().map(|&c| (c, 0.0)).collect();
    for k in (0..W * H).filter(|&k| land(k)) {
        let c = kp(k);
        shares[c as usize].1 += wt(k);
    }
    let line: Vec<String> = shares.iter().filter(|s| s.1 > 0.0).map(|(c, w)| format!("{} {:.1}", c.code(), 100.0 * w / land_w)).collect();
    println!("Köppen classes (% of land): {}", line.join(", "));
    // mountains near convergent boundaries vs the land near them
    let near_conv = |k: usize| px[k].s.boundary.is_convergent() && px[k].s.plate_dist_km < 400.0;
    let mtn = |k: usize| px[k].amp_m > 0.3 * mh;
    let (mut m_all, mut m_near, mut l_near) = (0.0, 0.0, 0.0);
    for k in (0..W * H).filter(|&k| land(k)) {
        let w = wt(k);
        if mtn(k) {
            m_all += w;
            if near_conv(k) {
                m_near += w;
            }
        }
        if near_conv(k) {
            l_near += w;
        }
    }
    println!(
        "mountain land (amplitude > 30 %): {:.1} % of land; {:.1} % of it within 400 km of a convergent boundary, which covers {:.1} % of all land",
        100.0 * m_all / land_w,
        100.0 * m_near / m_all.max(1e-9),
        100.0 * l_near / land_w
    );
    // rain shadows: windward vs leeward land
    let (mut up, mut down) = ((0.0, 0.0), (0.0, 0.0));
    for y in 0..H {
        for x in 0..W {
            let k = y * W + x;
            let s = &px[k].s;
            if !land(k) || lat_of(y).abs() > 60.0 {
                continue;
            }
            let sp = (s.wind_e * s.wind_e + s.wind_n * s.wind_n).sqrt();
            if sp < 2.0 {
                continue;
            }
            let p = dir_of(x, y);
            let (lat, lon) = (lat_of(y).to_radians(), (-180.0 + (x as f64 + 0.5) * 360.0 / W as f64).to_radians());
            let east = DVec3::new(-lon.sin(), lon.cos(), 0.0);
            let north = DVec3::new(-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos());
            let back = p - (east * s.wind_e + north * s.wind_n) / sp * 250e3;
            let rise = s.elevation_m.max(0.0) - atlas.sample(back).elevation_m.max(0.0);
            if rise > 300.0 {
                up.0 += s.precip_mm;
                up.1 += 1.0;
            } else if rise < -300.0 {
                down.0 += s.precip_mm;
                down.1 += 1.0;
            }
        }
    }
    println!("windward slopes {:.0} mm/yr, leeward {:.0} mm/yr", up.0 / up.1, down.0 / down.1);
    println!("wrote {}", out.display());
    Ok(())
}
