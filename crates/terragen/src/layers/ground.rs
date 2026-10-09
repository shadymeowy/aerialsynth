//! Slot 1, zonal ground: soil and grass of the biome by the climate, tundra in the cold, marsh on
//! wet floodplains, the ground textures and the drainage lines.

use crate::landcover as lc;
use crate::noise::*;
use crate::registry::pal;
use crate::stack::{Layer, Stack};
use crate::surface::mixc;
use glam::DVec3;

pub fn zonal(s: &mut Stack) {
    let t = s.t;
    let pf = s.pf;
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let st = &t.style;
    let detail = pf.detail;
    let patch = pf.patch;
    let wet = t.moist;
    let temp = t.temp;
    let soil_i = st[0] * 3.0;
    let i0 = (soil_i.floor() as usize).min(2);
    let soil = mixc(s.pal(pal::SOIL + i0), s.pal(pal::SOIL + i0 + 1), soil_i - i0 as f64);
    // red laterite soils in hot, wet climates
    let lat_w = (0.75 * s.veg(|v| v.laterite)).min(1.0);
    let soil = mixc(soil, s.pal(pal::LATERITE), lat_w * smoothstep(19.0, 25.0, temp) * smoothstep(0.45, 0.7, wet));
    let grass_green = mixc(s.pal(pal::GRASS_DRY), s.pal(pal::GRASS_WET), smoothstep(0.25, 0.75, wet + 0.15 * patch));
    let grass = mixc(s.pal(pal::GRASS_COLD), grass_green, smoothstep(-2.0, 8.0, temp));
    // hue drift per region so neighbouring areas differ
    let grass = grass * DVec3::new(1.0 + 0.10 * (st[1] - 0.5), 1.0 + 0.06 * (st[3] - 0.5), 1.0 - 0.08 * (st[1] - 0.5));
    let land_n = pf.land;
    let cover = smoothstep(0.08, 0.45, wet + 0.25 * patch + 0.2 * land_n) * smoothstep(-9.0, -1.0, temp);
    let mut col = mixc(soil, grass, cover) * (1.0 + 0.16 * land_n);
    let mut class = if cover > 0.5 { lc::GRASS } else { lc::BARE };
    if temp < 0.0 && cover > 0.3 {
        col = mixc(col, s.pal(pal::TUNDRA), smoothstep(0.0, -6.0, temp));
        class = lc::TUNDRA;
    }
    if t.floodplain > 0.3 && wet > 0.55 {
        let m = smoothstep(0.3, 0.9, t.floodplain) * smoothstep(0.55, 0.8, wet) * smoothstep(-0.1, 0.3, patch);
        col = mixc(col, s.pal(pal::MARSH), m);
        if m > 0.5 {
            class = lc::WETLAND;
        }
    }
    col *= 1.0 + 0.22 * detail;
    // meadow texture: dry straw-coloured patches and mottling (tussocks, growth) at a few
    // metres to tens of metres, band-limited (the mean is unchanged at coarse zooms)
    let meadow = s.veg(|v| v.meadow);
    {
        let dry_p = smoothstep(0.05, 0.55, perlin3(0x3EAD, p / 60.0) + 0.5 * perlin3(0x3EAE, p / 22.0)) * band(30.0, gsd) * cover;
        col = mixc(col, col * DVec3::new(1.16, 1.06, 0.80), 0.45 * dry_p * meadow);
        col *= 1.0
            + (0.10 * perlin3(0x3EB1, p / 12.0) * band(12.0, gsd) + 0.08 * perlin3(0x3EB2, p / 4.0) * band(4.0, gsd) + 0.06 * perlin3(0x3EB3, p / 1.3) * band(1.3, gsd))
                * cover
                * meadow;
    }
    // drainage lines: moister, greener, darker channels; dry bright spurs
    if t.gully != 0.0 {
        let ch = smoothstep(0.1, 0.8, -t.gully);
        col = mixc(col, mixc(col * 0.8, s.pal(pal::GRASS_WET) * 0.85, 0.5 * smoothstep(-6.0, 4.0, temp)), 0.6 * ch);
        col *= 1.0 + 0.06 * smoothstep(0.2, 1.0, t.gully);
    }
    s.m.cover = cover;
    s.composite(Layer { cov: 1.0, albedo: col, cls: class, ..Default::default() });
}
