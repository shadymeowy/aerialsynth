//! Slot 5, agriculture: the fields of the land-use regions.

use crate::landcover as lc;
use crate::noise::*;
use crate::surface::*;
use crate::world::Terrain;
use glam::{DVec2, DVec3};

impl SurfaceModel {
    /// Agricultural field at rotated local coords. Returns (colour, extra height, coverage, kind)
    /// where kind 0 = crop, 1 = hedge, 2 = track.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn field(
        &self,
        cache: &mut Caches,
        r: &RegionInfo,
        t: &Terrain,
        agri: f64,
        q: DVec2,
        p: DVec3,
        gsd: f64,
        fw: f64,
        pf: &PixFields,
    ) -> Option<(DVec3, f64, f64, u8)> {
        let pal = &self.pal;
        let cult = smoothstep(0.02, 0.45, agri);
        let tint = DVec3::new(1.0 + 0.08 * (r.palette - 0.5), 1.0, 1.0 - 0.06 * (r.palette - 0.5));
        // Always evaluate the actual field; when fields approach the pixel size, their contrast is
        // reduced towards the mean, mimicking the variance reduction of box-filtering a mosaic.
        let fsize = if r.fh > 0.0 { r.fw.min(r.fh) } else { r.fw };
        let k = fsize / (fsize * fsize + 4.0 * gsd * gsd).sqrt();
        let tropic = smoothstep(19.0, 25.0, t.temp) * smoothstep(0.5, 0.7, t.moist);
        let mean = mixc(pal.crop_mean, srgb(78.0, 100.0, 58.0), tropic) * tint * (1.0 + 0.10 * self.pf_lazy(pf, PF_FIELD_VAR));
        let fwe = fw.max(0.6 * gsd); // edge filter never sharper than ~half a pixel
        let (c, h, cov, kind) = self.field_explicit(cache, r, t, q, p, gsd, fwe, cult, tint, pf).unwrap_or((mean, 0.0, 0.0, 0));
        let cult_mean = cult * 0.9;
        let cov_m = lerp(cult_mean, cov, k);
        if cov_m <= 0.0 {
            return None;
        }
        let col = if cov > 0.0 { mixc(mean, c, k) } else { mean };
        Some((col, h * k, cov_m, kind))
    }

    /// Contiguous cultivation zones: a smooth mask (compared with the cultivated fraction).
    pub(crate) fn cultivated_mask(&self, c3: DVec3, gsd: f64) -> f64 {
        0.5 + 0.5 * self.cult_n.eval(c3, gsd.min(100.0)) * self.cult_n.norm() * 2.2
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn field_explicit(
        &self,
        cache: &mut Caches,
        r: &RegionInfo,
        t: &Terrain,
        q: DVec2,
        p: DVec3,
        gsd: f64,
        fw: f64,
        cult: f64,
        tint: DVec3,
        pf: &PixFields,
    ) -> Option<(DVec3, f64, f64, u8)> {
        let pal = &self.pal;
        // field id, within-field coords (along, across), distance to boundary
        let (id, fx, fy, edge, inside, fc) = match r.style {
            0 | 3 => {
                let wq = DVec2::new(perlin3(r.split.to_bits(), p / 900.0), perlin3(r.split.to_bits() ^ 1, p / 900.0));
                let q = q + wq * (0.08 * r.fw);
                let (w, h) = (r.fw, r.fh.max(r.fw));
                let seed = r.split.to_bits();
                // rows with jittered heights; within a row, field boundaries are jittered lines
                // (widths vary ~0.55-1.45 w) so the grid does not look like a regular quilt
                let row_line = |j: i64| (j as f64 + 0.32 * (u01(hash2(seed ^ 0x40, 7, j)) - 0.5)) * h;
                let mut j = (q.y / h).floor() as i64;
                if q.y < row_line(j) {
                    j -= 1;
                } else if q.y >= row_line(j + 1) {
                    j += 1;
                }
                let (y0, y1) = (row_line(j), row_line(j + 1));
                let shift = u01(hash1(seed, j)) * w;
                let x = q.x + shift;
                let col_line = |i: i64| (i as f64 + 0.45 * (u01(hash2(seed ^ 0x41, j, i)) - 0.5)) * w;
                let mut i = (x / w).floor() as i64;
                if x < col_line(i) {
                    i -= 1;
                } else if x >= col_line(i + 1) {
                    i += 1;
                }
                let (x0, x1) = (col_line(i), col_line(i + 1));
                let (cw, ch) = (x1 - x0, y1 - y0);
                let fx = x - x0;
                let fy = q.y - y0;
                // split some cells into strips
                let hc = hash2(seed ^ 0x55, i, j);
                let nstrip = 1 + (u01k(hc, 1) * if r.style == 3 { 1.0 } else { 3.5 }) as i64;
                let sh = ch / nstrip as f64;
                let k = (fy / sh).floor().min(nstrip as f64 - 1.0);
                let fy2 = fy - k * sh;
                let edge = fx.min(cw - fx).min(fy2).min(sh - fy2);
                let fc = DVec2::new(x0 + 0.5 * cw - shift, y0 + k * sh + 0.5 * sh);
                (mix64(hc ^ k as u64), fx, fy2, edge, 1.0, fc)
            }
            1 => {
                // irregular fields: Voronoi cells in a stretched frame (elongated fields), cell
                // size modulated across the region
                let aspect = 1.0 + 2.2 * u01k(r.split.to_bits(), 21);
                let qs = DVec2::new(q.x / aspect, q.y);
                let scale = 0.7 + 0.6 * (0.5 + 0.5 * perlin3(r.split.to_bits() ^ 0x5CA1, p / 2500.0));
                let cell = r.fw / aspect.sqrt() * scale;
                let wc = worley2(r.split.to_bits(), qs, cell, 0.85);
                // edge distance back in metres: the bisector normal is stretched by the aspect
                let dn = wc.point2 - wc.point;
                let nrm = DVec2::new(dn.x / aspect, dn.y).length() / dn.length().max(1e-9);
                let e = worley2_edge_dist(&wc, qs) / nrm.max(1e-6) * 1.0;
                let rel = q - DVec2::new(wc.point.x * aspect, wc.point.y);
                (wc.id, rel.x, rel.y, e, 1.0, DVec2::new(wc.point.x * aspect, wc.point.y))
            }
            _ => {
                let s = r.fw;
                let cq = (q / s).floor();
                let c = (cq + 0.5) * s;
                let rel = q - c;
                let d = rel.length();
                let rad = s * 0.48;
                let hc = hash2(r.split.to_bits(), cq.x as i64, cq.y as i64);
                let ins = band_cov(d, rad, fw);
                // pie sectors
                let nsec = 1 + (u01k(hc, 1) * 3.0) as i64;
                let ang = rel.y.atan2(rel.x) + std::f64::consts::PI + u01k(hc, 2) * 6.0;
                let sec = ((ang / std::f64::consts::TAU * nsec as f64).floor() as i64).rem_euclid(nsec);
                let edge_c = (rad - d).abs();
                (mix64(hc ^ sec as u64), rel.x, rel.y, edge_c, ins, c)
            }
        };
        if inside <= 0.0 {
            return None;
        }
        // is this field cultivated?
        let c3 = r.center + r.ex * fc.x + r.ey * fc.y;
        // both depend on the field only (the mask's shortest octave is resolved at any gsd ≤
        // 100 m), so they are kept per field instead of being evaluated for every sample;
        // neighbouring fields often grow the same crop: drawn from a coarse crop-cluster cell
        let (mask, cluster) = *cache
            .fields
            .entry([c3.x.to_bits(), c3.y.to_bits(), c3.z.to_bits(), r.split.to_bits()])
            .or_insert_with(|| (self.cultivated_mask(c3, gsd), worley2(r.split.to_bits() ^ 0xC1C, fc, 700.0, 1.0).id));
        if mask >= cult || u01k(id, 5) < 0.06 {
            return None;
        }
        let u_crop = if u01k(id, 16) < 0.5 { u01k(cluster, 6) } else { u01k(id, 6) };
        let kind = crop_kind(r.season, u_crop, t.moist < 0.33 && r.style != 2);
        // hot, wet climates grow other crops: rice paddies, oil-palm plantations, sugarcane and
        // bare red laterite between plantings (not the golden cereals of temperate farmland)
        let tropic = smoothstep(19.0, 25.0, t.temp) * smoothstep(0.5, 0.7, t.moist);
        let mut col;
        let mut extra_h = 0.0;
        if u01k(id, 30) < tropic {
            let tk = u_crop;
            let fine = 1.0 + 0.08 * perlin3(id ^ 0x7A1, p / 6.0) * band(6.0, gsd) + 0.06 * perlin3(id ^ 0x7A2, p / 1.5) * band(1.5, gsd);
            if tk < 0.35 {
                // rice: young green or flooded paddies, cut into small plots by earth bunds
                col = mixc(srgb(74.0, 106.0, 62.0), srgb(58.0, 82.0, 74.0), u01k(id, 31)) * fine;
                let sx = 18.0 + 22.0 * u01k(id, 32);
                let sy = sx * (1.2 + 0.8 * u01k(id, 33));
                let (mx, my) = (fx.rem_euclid(sx), fy.rem_euclid(sy));
                let d = mx.min(sx - mx).min(my).min(sy - my);
                let bund = band_cov(d, 0.6, fw) * band(sx, gsd);
                col = mixc(col, srgb(108.0, 106.0, 74.0), bund);
                extra_h = 0.3 * bund;
            } else if tk < 0.75 {
                // oil palm: star-shaped crowns on a triangular grid; stands of different age
                let age = 0.35 + 0.65 * u01k(id, 34);
                let (sx, sy) = (9.0, 7.8);
                let j = (fy / sy).round();
                let off = if (j as i64).rem_euclid(2) == 0 { 0.0 } else { 0.5 * sx };
                let i = ((fx - off) / sx).round();
                let rel = DVec2::new(fx - off - i * sx, fy - j * sy);
                let ang = rel.y.atan2(rel.x) + u01k(mix64(id ^ (i as i64 as u64) ^ ((j as i64 as u64) << 20)), 1) * 6.3;
                let r_eff = 4.4 * age * (0.82 + 0.18 * (8.0 * ang).cos());
                let d = rel.length();
                let explicit = band(sx, gsd);
                let crown = band_cov(d, r_eff, fw) * explicit + (1.0 - explicit) * (0.75 * age);
                let ground = srgb(92.0, 100.0, 60.0) * fine;
                let shade = 0.85 + 0.25 * (1.0 - (d / r_eff.max(0.1)).min(1.0));
                col = mixc(ground, srgb(58.0, 92.0, 44.0) * shade * fine, crown);
                extra_h = (3.0 + 9.0 * age) * crown;
            } else if tk < 0.92 {
                // sugarcane / banana: dense vivid green in rows
                col = srgb(86.0, 116.0, 60.0) * fine;
                let along = fx;
                col *= 1.0 + 0.08 * (along * std::f64::consts::TAU / 1.5).sin() * band(1.5, gsd);
                extra_h = 2.5;
            } else {
                // freshly ploughed red laterite
                col = srgb(152.0, 90.0, 62.0) * fine;
                col *= 1.0 + 0.1 * (fx * std::f64::consts::TAU / 0.9).sin() * band(0.9, gsd);
            }
            col *= tint;
        } else {
            col = pal.crop[kind];
            col *= 0.94 + 0.12 * u01k(id, 8);
            col *= tint;
            // within-field variation (soil moisture, growth, management) at several scales
            col *= 1.0 + 0.10 * self.pf_lazy(pf, PF_FIELD_VAR) + 0.12 * perlin3(id, p / 35.0) * band(35.0, gsd) + 0.08 * perlin3(id ^ 1, p / (0.8 * r.fw));
            // growth zones (soil, moisture): greener / yellower patches of tens of metres
            let gz = perlin3(id ^ 0x6A0, p / 55.0) * band(55.0, gsd);
            col = mixc(col, col * DVec3::new(1.12, 1.04, 0.82), 0.5 * smoothstep(0.0, 0.6, gz));
            col = mixc(col, col * DVec3::new(0.86, 0.93, 0.88), 0.5 * smoothstep(0.0, 0.6, -gz));
            // management direction: rows / tramlines along one field axis; in the headland (strip
            // along the field edge where the tractor turns) the pattern runs parallel to the edge
            let row_ang = if u01k(id, 9) < 0.7 { 0.0 } else { std::f64::consts::FRAC_PI_2 };
            let headland_w = 8.0 + 10.0 * u01k(id, 11);
            let in_headland = edge < headland_w && matches!(kind, 0..=4);
            let row_ang = if in_headland { row_ang + std::f64::consts::FRAC_PI_2 } else { row_ang };
            let (sa, ca) = row_ang.sin_cos();
            let along = fx * ca + fy * sa;
            // soil / growth texture at several scales (band-limited)
            let tex = 0.18 * perlin3(id ^ 0x7E1, p / 7.0) * band(7.0, gsd)
                + 0.13 * perlin3(id ^ 0x7E2, p / 2.5) * band(2.5, gsd)
                + 0.07 * perlin3(id ^ 0x7E5, p / 0.9) * band(0.9, gsd)
                + 0.12 * perlin3(id ^ 0x7E4, p / 22.0) * band(22.0, gsd)
                + 0.10 * perlin3(id ^ 0x7E3, p / 90.0) * band(90.0, gsd);
            col *= 1.0 + tex;
            // wet hollows / bare patches inside some fields
            if u01k(id, 12) < 0.35 {
                let wp = perlin3(id ^ 0x5A7, p / (40.0 + 60.0 * u01k(id, 13)));
                let m = smoothstep(0.25, 0.45, wp) * band(30.0, gsd).max(0.3);
                col = mixc(col, col * DVec3::new(0.82, 0.86, 0.80), m);
            }
            if in_headland {
                col *= 0.96 + 0.03 * u01k(id, 14);
            }
            match kind {
                0 | 1 => {
                    let sp = 0.8 + 0.8 * u01k(id, 10);
                    col *= 1.0 + 0.12 * (along * std::f64::consts::TAU / sp).sin() * band(sp, gsd);
                }
                3 => {
                    let sp = 6.0 + 4.0 * u01k(id, 10);
                    col *= 1.0 + 0.07 * ((along / sp * std::f64::consts::TAU).sin()).signum() * band(sp, gsd);
                }
                4 => {
                    let sp = 0.45;
                    col *= 1.0 + 0.15 * (along * std::f64::consts::TAU / sp).sin() * band(sp, gsd);
                    col *= 1.0 - 0.15 * smoothstep(0.2, 0.6, self.pf_lazy(pf, PF_FIELD_VAR2));
                }
                8 => {
                    // orchard: rows of small trees
                    let (sx, sy) = (5.0 + 2.0 * u01k(id, 10), 4.0);
                    let gx = (fx / sx).round() * sx;
                    let gy = (fy / sy).round() * sy;
                    let d = DVec2::new(fx - gx, fy - gy).length();
                    let explicit = band(sx, gsd);
                    let cov = band_cov(d, 1.7, fw) * explicit + (1.0 - explicit) * 0.4;
                    col = mixc(col, pal.crown_decid * 1.1, cov);
                    extra_h = 4.0 * cov;
                }
                _ => {}
            }
            // tramlines (wheel tracks every ~18-36 m) in cereals / green crops / stubble
            if matches!(kind, 0 | 1 | 2 | 3 | 7) && !in_headland {
                let sp = 18.0 + 18.0 * (u01k(id, 15) * 2.0).floor() / 2.0;
                let m = along.rem_euclid(sp);
                let tl = band_cov(m - 0.9, 0.22, fw) + band_cov(m - 2.7, 0.22, fw);
                let vis = band(1.2, gsd).max(0.35 * band(sp, gsd));
                col = mixc(col, col * DVec3::new(0.78, 0.76, 0.74), tl * vis);
            }
        }
        // field borders: hedges (trees) or tracks or simply a thin margin
        let bw = r.border_w;
        let bcov = band_cov(edge, bw, fw.max(gsd * 0.4));
        let mut kind_out = 0;
        if bcov > 0.0 {
            let hb = mix64(id ^ 0xED6E);
            if u01k(hb, 1) < r.hedge {
                let hc = pal.crown_decid * (0.8 + 0.3 * self.pf_lazy(pf, PF_FIELD_VAR3));
                col = mixc(col, hc, bcov);
                // rounded cross-section and a height varying along the hedge (a row of shrubs and
                // small trees, not a flat-topped wall)
                let across = (1.0 - (edge.abs() / bw.max(0.1)).powi(2)).max(0.0).sqrt();
                let along = 0.45 + 0.55 * (0.5 + 0.5 * perlin3(hb, p / 6.0));
                extra_h = lerp(extra_h, (2.2 + 2.3 * u01k(hb, 2)) * across * along, bcov);
                if bcov > 0.5 {
                    kind_out = 1;
                }
            } else if u01k(hb, 3) < r.track * 0.4 {
                col = mixc(col, pal.gravel, bcov);
                if bcov > 0.5 {
                    kind_out = 2;
                }
            } else {
                col = mixc(col, mixc(pal.grass_dry, pal.grass_wet, t.moist), bcov * 0.7);
            }
        }
        Some((col, extra_h, inside, kind_out))
    }
}

/// Slot 5: the fields of the sample's land-use region. Farmland keeps forest on its most
/// forest-prone patches (woodlots, up to ~1/7 of the land by default, the biome's and culture's
/// share), fields stay on gentle slopes, off riparian belts and back from the shore.
pub fn layer(s: &mut crate::stack::Stack) {
    use crate::stack::{hmode, Layer};
    let world = s.world;
    let t = s.t;
    let fpu = 0.5 + 0.5 * s.pf.forest;
    let cover = smoothstep(0.12, 0.45, t.moist.max(0.0)) * smoothstep(-6.0, 2.0, t.temp) * world.cfg.vegetation.tree_density;
    let share = (0.34 * s.veg(|v| v.woodlots) * s.bio.style.woodlots).min(0.95);
    s.m.woodlot = smoothstep(-0.02, 0.02, share * cover - fpu) * s.m.natural_ok;
    let agri = s.agri();
    let m = s.m;
    let Some(r) = s.region else { return };
    if !(agri > 0.02 && m.natural_ok * m.flat_ok > 0.3 && world.cfg.landuse.agriculture > 0.0) {
        return;
    }
    let (p, gsd, fw, q) = (s.ctx.p, s.ctx.gsd, s.l.fw, s.q_rot);
    let Some((fcol, fh, cov, edge_kind)) = s.sm.field(s.cache, &r, t, agri, q, p, gsd, fw, s.pf) else { return };
    // a field is there or not: a crisp cutoff (fields faded out over gentle valley sides left
    // washed, half-transparent bands; a noisy threshold left specks of open ground)
    let keep = m.natural_ok * m.flat_ok * (1.0 - m.riparian) * (1.0 - m.woodlot) * m.shore_keep;
    let a = cov * smoothstep(0.45, 0.55, keep);
    s.m.field_cov = a;
    let cls = match edge_kind {
        1 => crate::registry::classes::id("hedgerow").unwrap_or(lc::FOREST),
        2 => crate::registry::classes::id("track").unwrap_or(lc::ROAD),
        _ => lc::CROP,
    };
    // tilled fields are smoother than natural ground
    s.composite(Layer { cov: a, albedo: fcol, dh: fh - 0.6 * m.micro, hmode: hmode::ADD, cls, ..Default::default() });
}
