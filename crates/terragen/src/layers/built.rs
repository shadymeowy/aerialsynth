//! Slot 7, built: farmsteads, towns and their lights.

use crate::landcover as lc;
use crate::noise::*;
use crate::surface::*;
use glam::{DVec2, DVec3};

impl SurfaceModel {
    /// Farmstead: a small cluster (house, barn, gravel yard, yard lamp) on a sparse lattice in the
    /// agricultural regions. Returns (colour, height, coverage, class, emission).
    pub(crate) fn farmstead(&self, r: &RegionInfo, agri: f64, q: DVec2, gsd: f64, fw: f64) -> Option<(DVec3, f64, f64, u8, DVec3)> {
        let pal = &self.pal;
        let wc = worley2(r.split.to_bits() ^ 0xFA4, q, 650.0, 0.8);
        let fid = wc.id;
        if u01k(fid, 1) > 0.55 * smoothstep(0.05, 0.4, agri) {
            return None;
        }
        let rel0 = q - wc.point;
        if rel0.length() > 60.0 {
            return None;
        }
        let ang = (u01k(fid, 2) - 0.5) * 0.4;
        let (sa, ca) = ang.sin_cos();
        let rel = DVec2::new(rel0.x * ca + rel0.y * sa, -rel0.x * sa + rel0.y * ca);
        let fwe = fw.max(0.3 * gsd);
        let boxc = |c: DVec2, half: DVec2| -> f64 {
            let d = (half - (rel - c).abs()).min_element();
            (d / fwe + 0.5).clamp(0.0, 1.0)
        };
        let yard_half = DVec2::new(20.0 + 10.0 * u01k(fid, 3), 14.0 + 8.0 * u01k(fid, 4));
        let yard = boxc(DVec2::ZERO, yard_half);
        // lamp light pool (also lights the surrounding field a little)
        let lamp = rel - DVec2::new(2.0, 6.0);
        let d2 = lamp.length_squared();
        let lamp_col = if u01k(fid, 9) < 0.6 { DVec3::new(1.0, 0.72, 0.38) } else { DVec3::new(0.86, 0.92, 1.0) };
        let res = band(10.0, gsd);
        let emission = lamp_col * (0.04 * (-d2 / (2.0 * 7.0 * 7.0)).exp() + point_light(d2, 5.0, 0.4, fw)) * res
            + lamp_col * 0.02 * (1.0 - res) * (1.0 - smoothstep(20.0, 60.0, rel.length()));
        if yard <= 0.0 {
            return if emission.max_element() > 1e-4 { Some((DVec3::ZERO, 0.0, 0.0, lc::CROP, emission)) } else { None };
        }
        let mut col = mixc(pal.gravel, pal.concrete, 0.3 * u01k(fid, 5)) * (0.9 + 0.2 * u01k(fid, 6));
        let mut h = 0.0;
        let mut class = lc::URBAN;
        // house (pitched roof) and barn (long, low-pitched metal roof)
        let hc = DVec2::new(-yard_half.x * 0.45, -yard_half.y * 0.3);
        let hh = DVec2::new(5.5, 4.5);
        let house = boxc(hc, hh);
        if house > 0.0 {
            let roof = pal.roofs[(u01k(fid, 7) * 2.99) as usize] * (0.9 + 0.2 * u01k(fid, 8));
            let ridge = ((hh.y - (rel.y - hc.y).abs()) / hh.y).clamp(0.0, 1.0);
            col = mixc(col, roof, house);
            h = (5.5 + 2.0 * ridge) * house;
            class = lc::BUILDING;
        }
        let bc = DVec2::new(yard_half.x * 0.3, yard_half.y * 0.35);
        let bh = DVec2::new(11.0 + 5.0 * u01k(fid, 10), 6.0);
        let barn = boxc(bc, bh);
        if barn > 0.0 {
            let roof = mixc(pal.roofs[4], pal.roofs[2], u01k(fid, 11));
            let ridge = ((bh.y - (rel.y - bc.y).abs()) / bh.y).clamp(0.0, 1.0);
            col = mixc(col, roof, barn);
            h = h.max((6.0 + 1.5 * ridge) * barn);
            class = lc::BUILDING;
        }
        let cov = yard * smoothstep(1.0 * gsd, 3.0 * gsd, 30.0).max(0.25);
        Some((col, h, cov, class, emission))
    }

    /// How built-up the town is at `p` (1 in the centre, 0 outside its irregular footprint), and
    /// the relative distance from the centre.
    /// `clear`: 0 where the town must not be (rivers), 1 elsewhere.
    pub(crate) fn town_urban(&self, town: &TownInfo, p: DVec3, gsd: f64, slope: f64, clear: f64, pf: &PixFields) -> (f64, f64) {
        let d = p - town.center;
        let q0 = DVec2::new(d.dot(town.ex), d.dot(town.ey));
        let r = town.radius;
        // far outside: the test below without its square roots (|qa|² >= |q0|² min(e, 1/e))
        let (e, q2, r2) = (town.elong, q0.length_squared(), 4.0 * r * r * (1.0 + 1e-9));
        if if e >= 1.0 { q2 > r2 * e } else { e * q2 > r2 } {
            return (0.0, 2.0);
        }
        // elongated, irregular footprint (noise relative to the town size)
        let qa = DVec2::new(q0.x / town.elong.sqrt(), q0.y * town.elong.sqrt());
        if qa.length() > r * 2.0 {
            return (0.0, 2.0);
        }
        let n1 = 0.32 * perlin3(town.seed ^ 0x71, p / (0.9 * r)) + 0.18 * perlin3(town.seed ^ 0x72, p / (0.35 * r)) * band(0.35 * r, gsd) + 0.08 * pf.warp2;
        let rel = qa.length() / (r * (1.0 + n1)).max(1.0);
        ((1.0 - smoothstep(0.3, 1.0, rel)) * (1.0 - smoothstep(0.45, 0.8, slope)) * clear, rel)
    }

    /// Town at point p. Returns (colour, height above ground, coverage, class, shadow, emission).
    /// Coverage is crisp: streets and built lots cover the ground fully, open land in the
    /// outskirts shows the underlying fields / nature.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn town(
        &self,
        town: &TownInfo,
        p: DVec3,
        gsd: f64,
        fw: f64,
        slope: f64,
        clear: f64,
        shadows: bool,
        pf: &PixFields,
    ) -> Option<(DVec3, f64, f64, u8, f64, DVec3)> {
        let pal = &self.pal;
        let d = p - town.center;
        let q0 = DVec2::new(d.dot(town.ex), d.dot(town.ey));
        let (urban, rel) = self.town_urban(town, p, gsd, slope, clear, pf);
        if urban <= 0.0 {
            return None;
        }
        // organic (curved) streets in old towns
        let warp = DVec2::new(perlin3(town.seed, p / 350.0), perlin3(town.seed ^ 9, p / 350.0)) * (35.0 * town.organic);
        let q = q0 + warp;
        // street grid with jittered (irregularly spaced) street lines
        let b = town.block;
        let line = |axis: u64, i: i64| (i as f64 + 0.36 * (u01(hash2(town.seed ^ 0x5EE7, axis as i64, i)) - 0.5)) * b;
        let cell = |axis: u64, x: f64| -> (i64, f64, f64) {
            let mut i = (x / b).floor() as i64;
            if x < line(axis, i) {
                i -= 1;
            } else if x >= line(axis, i + 1) {
                i += 1;
            }
            let (a0, a1) = (line(axis, i), line(axis, i + 1));
            (i, x - a0, a1 - a0)
        };
        let (bix, bqx, bsx) = cell(0, q.x);
        let (biy, bqy, bsy) = cell(1, q.y);
        let sw = town.street * 0.7; // lots start beyond the widest street (a verge along narrower ones)
                                    // streets exist where the town is dense enough; outskirts keep only some of them. Decided
                                    // per street segment (axis, line, and the segment along it), so the blocks on both sides
                                    // agree, and per axis, so streets end square at a crossing; crisp: a street is there or
                                    // not (a fading street left half-transparent asphalt with half-masked trees on it)
        let near_x = bix + (bqx > bsx - bqx) as i64; // nearest line across x and across y
        let near_y = biy + (bqy > bsy - bqy) as i64;
        // each street segment exists or not as a whole: decided with the town density at its
        // midpoint (per-pixel density faded streets in and out along a density contour, and the
        // houses of the blocks beside them were cut along it)
        let urban_at = |dq: DVec2| self.town_urban(town, p + town.ex * dq.x + town.ey * dq.y, gsd, slope, clear, pf).0;
        let seg_here = |h: u64, u: f64| -> f64 {
            if u > 0.15 + 0.12 * u01k(h, 4) && !(u < 0.45 && u01k(h, 3) < 0.4) {
                1.0
            } else {
                0.0
            }
        };
        let ymid = 0.5 * (line(1, biy) + line(1, biy + 1));
        let xmid = 0.5 * (line(0, bix) + line(0, bix + 1));
        let seg_x = |i: i64| seg_here(hash2(town.seed ^ 0x57, i, biy), urban_at(DVec2::new(line(0, i), ymid) - q));
        let seg_y = |i: i64| seg_here(hash2(town.seed ^ 0x58, i, bix), urban_at(DVec2::new(xmid, line(1, i)) - q));
        let (x0, x1, y0, y1) = (seg_x(bix), seg_x(bix + 1), seg_y(biy), seg_y(biy + 1));
        let here_x = if near_x == bix { x0 } else { x1 };
        let here_y = if near_y == biy { y0 } else { y1 };
        // a block is built on only if a street runs along at least one of its sides (houses stood
        // in the fields of the outskirts with no street anywhere near)
        let access = x0.max(x1).max(y0).max(y1);
        let sw_of = |line: i64| if line.rem_euclid(4) == 0 { town.street * 1.4 } else { town.street } * 0.5;
        let (sw_x, sw_y) = (sw_of(near_x), sw_of(near_y));
        let fws = fw.max(gsd * 0.5);
        let street = f64::max(band_cov(bqx.min(bsx - bqx), sw_x, fws) * here_x, band_cov(bqy.min(bsy - bqy), sw_y, fws) * here_y);
        // sidewalks between the carriageway and the lots (that verge was left as bare ground,
        // unlit at night: dark lines along every street)
        let walk_w = 0.2 * town.street + 0.5;
        let walk = f64::max(band_cov(bqx.min(bsx - bqx), sw_x + walk_w, fws) * here_x, band_cov(bqy.min(bsy - bqy), sw_y + walk_w, fws) * here_y);
        let bh = hash2(town.seed ^ 0xB10C, bix, biy);
        let block_kind = u01k(bh, 1);
        let mut col = pal.asphalt;
        let mut height = 0.0;
        let mut class = lc::URBAN;
        let mut shadow = 0.0;
        let mut cov_lot = 0.0;
        let mut porch = 0.0;
        let mut roof_frac = 0.0; // building roof coverage of this sample
        let mut win_col = DVec3::new(1.0, 0.74, 0.44);
        let mut windows = 0.0; // lit windows on the walls (the footprint rim, which the renderer
                               // stretches into the facades)
        let inner = DVec2::new(bqx - sw, bqy - sw);
        let (bw, bd) = (bsx - 2.0 * sw, bsy - 2.0 * sw);
        let central = (1.0 - rel).max(0.0);
        let buildings_resolved = band(town.lot, gsd);
        if urban > 0.5 && block_kind < 0.07 {
            // park
            col = pal.grass_wet * (1.0 + 0.15 * pf.detail);
            class = lc::GRASS;
            cov_lot = 1.0;
        } else if urban > 0.45 && block_kind < 0.12 {
            col = pal.concrete * (0.82 + 0.1 * u01k(bh, 4)); // parking / plaza
            cov_lot = 1.0;
        } else if inner.x >= 0.0 && inner.y >= 0.0 && inner.x < bw && inner.y < bd {
            let industrial = block_kind > 0.92 && central < 0.5;
            let lot_w = if industrial { bw } else { town.lot * (0.7 + 0.6 * u01k(bh, 5)) };
            let rows = if industrial || bd < 2.6 * town.lot { 1.0 } else { 2.0 };
            let li = (inner.x / lot_w).floor();
            let lj = (inner.y / (bd / rows)).floor();
            let lx = inner.x - li * lot_w;
            let ly = inner.y - lj * (bd / rows);
            let lh = hash2(bh, li as i64, lj as i64);
            // decided with the town density at the lot's centre, not per pixel: houses were cut
            // in half where the density contour crossed them
            let lot_c = DVec2::new(li * lot_w + 0.5 * lot_w - inner.x, (lj + 0.5) * (bd / rows) - inner.y);
            let urban_lot = self.town_urban(town, p + town.ex * lot_c.x + town.ey * lot_c.y, gsd, slope, clear, pf).0;
            let built = u01k(lh, 3) < urban_lot.powf(0.7) * 1.05 && access > 0.5;
            if built || urban_lot > 0.65 {
                cov_lot = 1.0;
                let yard = mixc(mixc(pal.grass_wet, pal.soil[0], 0.3 + 0.4 * u01k(lh, 10)), pal.concrete, 0.25 * central) * (1.0 + 0.2 * pf.detail);
                col = yard;
                if built {
                    // building footprint inside the lot (sometimes L-shaped)
                    let setb = if industrial { 6.0 } else { 1.5 + 3.5 * u01k(lh, 1) };
                    let fwid = lot_w - 2.0 * setb.min(lot_w * 0.3);
                    let fdep = (bd / rows - setb - 2.0 - 5.0 * u01k(lh, 2)).max(0.0);
                    if fwid > 3.0 && fdep > 3.0 {
                        let cx = lx - lot_w * 0.5;
                        let cy = if (lj as i64) % 2 == 0 { ly - setb - fdep * 0.5 } else { ly - (bd / rows - setb - fdep * 0.5) };
                        let ex = fwid * 0.5 - cx.abs();
                        let ey = fdep * 0.5 - cy.abs();
                        let mut inside = (ex.min(ey) / fw + 0.5).clamp(0.0, 1.0);
                        if !industrial && u01k(lh, 11) < 0.3 {
                            // L-shape: remove a corner quadrant
                            let qx = if u01k(lh, 12) < 0.5 { cx } else { -cx };
                            let qy = if u01k(lh, 13) < 0.5 { cy } else { -cy };
                            let cut = (qx - fwid * 0.1).min(qy - fdep * 0.1);
                            inside *= 1.0 - (cut / fw + 0.5).clamp(0.0, 1.0);
                        }
                        let tall = central.powf(2.0) * town.height;
                        let hb = if industrial { 7.0 + 7.0 * u01k(lh, 4) } else { 3.5 + 4.0 * u01k(lh, 4) + 40.0 * tall * u01k(lh, 5) };
                        let flat_roof = industrial || hb > 12.0 || u01k(lh, 6) < 0.2 + 0.3 * central;
                        let ri = ((town.roof_style * 3.0 + u01k(lh, 7) * 4.0) as usize) % 7;
                        let mut roof = if industrial { mixc(pal.roofs[4], pal.roofs[2], u01k(lh, 8)) } else { pal.roofs[ri] };
                        roof *= 0.82 + 0.36 * u01k(lh, 9);
                        let mut h_here = hb;
                        if !flat_roof {
                            // pitched roof along the longer axis
                            let (half, dperp) = if fwid > fdep { (fdep * 0.5, ey) } else { (fwid * 0.5, ex) };
                            h_here = hb + 0.35 * half * (dperp / half).clamp(0.0, 1.0);
                        } else {
                            roof *= 1.0 - 0.15 * band_cov(ex.min(ey), 0.6, fw); // parapet edge
                        }
                        if inside > 0.0 {
                            col = mixc(col, roof, inside);
                            height = h_here * inside;
                            roof_frac = inside * buildings_resolved;
                            // windows every ~2.6 m along the walls, a share of them lit (more in
                            // the centre, offices and flats included)
                            let rim = (1.0 - smoothstep(0.0, 0.9, ex.min(ey))) * inside;
                            if rim > 0.0 {
                                let along = if ex < ey { cy } else { cx };
                                let wsp = 2.6;
                                let wi = (along / wsp).floor();
                                let wf = along / wsp - wi;
                                let lit_frac = 0.08 + 0.2 * central + 0.12 * u01k(lh, 15);
                                // incandescent / warm LED in most homes, some neutral and cool
                                // (offices, screens)
                                let wc = u01k(lh, 16);
                                win_col = if wc < 0.6 {
                                    DVec3::new(1.0, 0.70, 0.40)
                                } else if wc < 0.85 {
                                    DVec3::new(1.0, 0.86, 0.66)
                                } else {
                                    DVec3::new(0.82, 0.90, 1.0)
                                };
                                let lit = u01k(hash2(lh ^ 0x3D0, wi as i64, (ex < ey) as i64), 1) < lit_frac;
                                let explicit = band(wsp, gsd);
                                let pane = if lit && (0.3..0.65).contains(&wf) { 1.0 } else { 0.0 };
                                windows = rim * (pane * explicit + 0.55 * lit_frac * (1.0 - explicit));
                            }
                            if inside > 0.5 {
                                class = lc::BUILDING;
                            }
                        }
                        // porch / yard light in front of some houses
                        if !industrial && u01k(lh, 14) < 0.6 {
                            let front_y = if (lj as i64) % 2 == 0 { setb * 0.5 } else { bd / rows - setb * 0.5 };
                            let d2 = (lx - lot_w * 0.5).powi(2) + (ly - front_y).powi(2);
                            // a small bright lamp by the door, not a soft blob over the yard
                            porch = point_light(d2, 2.0, 0.4, fw);
                        }
                    }
                }
            }
            // mean appearance when lots are unresolved
            if buildings_resolved < 1.0 {
                let mean_roof = mixc(pal.roofs[(town.roof_style * 6.99) as usize], pal.concrete, 0.3);
                let mean = mixc(mixc(pal.grass_wet, pal.soil[0], 0.4), mean_roof, 0.5);
                col = mixc(mean, col, buildings_resolved);
                height = lerp(3.0 * urban, height, buildings_resolved);
                cov_lot = lerp(urban.powf(0.7), cov_lot, buildings_resolved);
            }
            // cast shadows of buildings onto the ground (march toward the sun)
            if shadows && buildings_resolved > 0.0 && height < 0.5 && cov_lot > 0.0 {
                for k in 1..=4 {
                    let dist_s = k as f64 * 3.5;
                    let need = dist_s * self.sun_tan;
                    let sp = p + town.sun * dist_s;
                    let hh = self.building_height_at(town, sp, gsd, pf);
                    if hh > need {
                        shadow = 0.75 * buildings_resolved;
                        break;
                    }
                }
            }
        }
        let cov = cov_lot.max(walk);
        if cov <= 0.0 {
            return None;
        }
        col = mixc(col, pal.concrete * 0.9, (walk - street).max(0.0) * (1.0 - cov_lot) / cov);
        col = mixc(col, pal.asphalt * (1.0 + 0.05 * pf.detail), street / cov);
        if street > 0.5 {
            class = lc::ROAD;
        }
        height *= 1.0 - street;

        // ---- night lights: sharp lamp heads with soft pools under them along every street,
        // lit plazas / industrial yards, windows and porch lights
        let lamp_sp = 18.0 + 8.0 * town.organic;
        // lamp types: high-pressure sodium (amber), warm LED (3000 K) and neutral LED (4000 K),
        // chosen per street with a town-specific mix (main streets lean to neutral LED)
        const LAMPS: [DVec3; 3] = [DVec3::new(1.0, 0.48, 0.12), DVec3::new(1.0, 0.72, 0.38), DVec3::new(0.86, 0.92, 1.0)];
        let mix_sodium = 0.25 + 0.4 * u01k(town.seed, 20);
        let dominant = if mix_sodium > 0.45 { 0 } else { 1 };
        let lamp_type = |axis: u64, line: i64| -> DVec3 {
            let u = u01k(hash2(town.seed ^ 0x1A3C ^ (axis << 40), line, 0), 1);
            let main = line.rem_euclid(4) == 0;
            let t = if main && u < 0.45 {
                2
            } else if u < mix_sodium {
                0
            } else if u < 0.85 {
                1
            } else {
                2
            };
            LAMPS[t]
        };
        let lamp_col = LAMPS[dominant];
        // lamps stay points down to the scale of the street grid: where they would merge (sample
        // spacing above half their spacing) only every m-th lamp is drawn, m times brighter. An
        // area mean there was clipped by the camera very differently from the points it averaged:
        // the far part of a town glowed as a flat slab next to the point-lit near part.
        let lamp_res = band(town.block, gsd);
        let fwl = fw.max(0.5 * gsd);
        let m = (2.0 * fwl / lamp_sp).max(1.0).log2().ceil().exp2();
        let sp_eff = lamp_sp * m;
        let mut emission = DVec3::ZERO;
        if lamp_res > 0.0 && here_x.max(here_y) > 0.0 {
            // lamps every lamp_sp along each street, alternating sides of the carriageway
            for axis in 0..2u64 {
                let (bi, bq, bs, along) = if axis == 0 { (bix, bqx, bsx, q.y) } else { (biy, bqy, bsy, q.x) };
                // signed offset of the pixel from the nearest street centreline, and that line's index
                let (off, li) = if bq < bs - bq { (bq, bi) } else { (-(bs - bq), bi + 1) };
                let (here, swa) = if axis == 0 { (here_x, sw_x) } else { (here_y, sw_y) };
                if here <= 0.0 || off.abs() > swa + 22.0 + 3.0 * fwl {
                    continue;
                }
                // index of the lamp (in units of lamp_sp; a multiple of m when thinned)
                let k = (along / sp_eff).round() * m;
                let side = if (k as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                let lamp_off = side * swa * 0.85;
                let (dp, da) = (off - lamp_off, along - k * lamp_sp);
                let d2 = dp * dp + da * da;
                let lh = hash2(town.seed ^ 0x1A3B ^ (axis << 40), li, k as i64);
                if u01k(lh, 1) < 0.93 {
                    let sa = 0.28 * lamp_sp;
                    // a soft pool on the street and a sharp, bright head (a point from the air)
                    // (the head carries most of the light seen from the air: with a bright pool
                    // every lamp was a soft disc)
                    let pool = 0.03 * (-(dp * dp) / (2.0 * 3.5 * 3.5) - (da * da) / (2.0 * sa * sa)).exp();
                    let core = point_light(d2, 6.0, 0.4, fw.max(0.5 * gsd));
                    emission += lamp_type(axis, li) * ((pool + core) * m) * (0.75 + 0.5 * u01k(lh, 2));
                }
            }
        }
        // the lamps light the ground below them, not the roofs (lit roofs read as glowing spikes)
        let ground_lit = 1.0 - roof_frac * (1.0 - street);
        emission *= ground_lit;
        porch *= ground_lit;
        // prefiltered mean when lamps are unresolved (town glow)
        // unresolved lamps: their mean over the area (lamp energy: head 2π·6·0.4² + pool
        // 2π·0.03·3.5·0.28·sp, 93% present, every sp along streets every ~block on both axes); a
        // fixed 0.07 was ~7x the mean of the resolved lamps, so the far part of a town glowed as a
        // flat orange slab next to the dark-roofed near part
        let lamp_e = 0.93 * (2.0 * std::f64::consts::PI * (6.0 * 0.16 + 0.03 * 3.5 * 0.28 * lamp_sp));
        let lamp_mean = lamp_e * 2.0 / (town.block * lamp_sp);
        emission = emission * lamp_res + lamp_col * lamp_mean * (1.0 - lamp_res) * smoothstep(0.15, 0.4, urban);
        // windows: only where the buildings are resolved (a rim smeared over coarse pixels lit
        // whole blocks; the lamps carry the town's light at coarse zooms)
        let windows = windows * buildings_resolved;
        emission += win_col * (0.55 * windows);
        // industrial yards: a dim base (plazas and parking have only the street lamps around
        // them: a lit base drew uniform grey slabs)

        if block_kind > 0.92 && central < 0.5 {
            emission += DVec3::new(1.0, 0.88, 0.7) * 0.02 * band(lamp_sp, gsd);
        }
        emission += DVec3::new(1.0, 0.72, 0.42) * porch;
        Some((col, height, cov, class, shadow * (1.0 - street * 0.5), emission / cov.max(0.05)))
    }

    pub(crate) fn building_height_at(&self, town: &TownInfo, p: DVec3, gsd: f64, pf: &PixFields) -> f64 {
        match self.town(town, p, gsd, 0.01, 0.0, 1.0, false, pf) {
            Some((_, h, cov, _, _, _)) => h * cov,
            None => 0.0,
        }
    }
}

/// Before the canopy: the most built-up town at the sample and its pixel there (towns have no
/// trees: they clear the canopy, with a short fade at their outer edge). Every existing town
/// of the surrounding lattice cells is tried; towns end at the banks of rivers.
pub fn select_town(s: &mut crate::stack::Stack) {
    let world = s.world;
    let (t, l) = (s.t, s.l);
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, l.fw);
    let river_clear = if l.river_hw > 0.0 {
        // a narrow bank (a few metres plus a tenth of the width), not a wide green strip
        let bank = 2.0 + 0.1 * l.river_hw;
        1.0 - (1.0 - smoothstep(l.river_hw + bank, l.river_hw + 2.0 * bank + 2.0, l.river_d.abs())) * smoothstep(0.3, 0.6, t.river_wet)
    } else {
        1.0
    };
    s.m.river_clear = river_clear;
    // towns avoid steep relief, judged from the relief type (smooth), not the slope of each
    // pixel (that cut houses in half along every terrace edge and gully wall)
    let town_slope = 0.12 + 0.75 * t.rock_expect;
    let sel = if t.town.id != 0 && world.cfg.landuse.towns > 0.0 { s.sm.select_town(world, s.cache, p, gsd, town_slope, river_clear, s.pf, 2) } else { None };
    let urban = sel.map_or(0.0, |x| x.1);
    let px = sel.and_then(|(town, _)| s.sm.town(&town, p, gsd, fw, town_slope, river_clear, world.cfg.satellite.shadows, s.pf));
    let cov = px.map_or(0.0, |x| x.2);
    s.town = sel;
    s.town_px = px;
    s.m.town_urban = urban;
    s.m.town_cov = cov;
    s.m.veg *= (1.0 - smoothstep(0.0, 0.08, urban)) * (1.0 - cov);
}

/// Slot 7: farmsteads, lit main roads near towns, embankment lamps, the town.
pub fn layer(s: &mut crate::stack::Stack) {
    use crate::stack::{hmode, Layer};
    let world = s.world;
    let (t, l) = (s.t, s.l);
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, l.fw);
    let in_dsm = if world.cfg.landuse.buildings_in_dsm { hmode::BLEND } else { hmode::NONE };
    let m = s.m;
    // ---- farmsteads
    let agri = s.agri();
    if let Some(r) = s.region {
        if agri > 0.06 && m.flat_ok > 0.3 && m.natural_ok > 0.3 && world.cfg.landuse.towns > 0.0 {
            if let Some((fcol, fh, fcov, fcls, fem)) = s.sm.farmstead(&r, agri, s.q_rot, gsd, fw) {
                s.composite(Layer { cov: fcov, albedo: fcol, dh: fh, hmode: in_dsm, cls: fcls, emit: fem, ..Default::default() });
            }
        }
    }
    // ---- lit main roads near towns
    if m.road_major_cov > 0.0 {
        if let Some((town, _)) = s.town {
            let dist = (p - town.center).length();
            let near = 1.0 - smoothstep(1.2 * town.radius, 2.2 * town.radius, dist);
            let sp = 38.0;
            let res = band(sp, gsd);
            if near > 0.0 {
                let q_loc = s.q_loc;
                let ql = (q_loc / sp).round() * sp;
                let d2 = (q_loc - ql).length_squared();
                let lh = hash2(town.seed ^ 0x40AD, (ql.x / sp) as i64, (ql.y / sp) as i64);
                let lamp_col = if u01k(lh, 1) < 0.5 { DVec3::new(1.0, 0.48, 0.12) } else { DVec3::new(0.86, 0.92, 1.0) };
                let pool = (0.04 * (-d2 / (2.0 * 6.0 * 6.0)).exp() + point_light(d2, 6.0, 0.4, fw)) * res + 0.04 * (1.0 - res);
                s.composite(Layer { emit: lamp_col * pool * near * m.road_major_cov, ..Default::default() });
            }
        }
    }
    // ---- embankment lamps: a string of lights along the banks of rivers through towns (the
    // river was a long dark gap in the lit town)
    if m.town_urban > 0.25 && l.river_hw > 0.0 && t.river_wet > 0.5 {
        let bank = 2.0 + 0.1 * l.river_hw;
        let dl = l.river_d.abs() - (l.river_hw + 0.5 * bank);
        let dots = smoothstep(0.35, 0.6, perlin3(0xE3B, p / 4.0)) * band(4.0, gsd) + 0.3 * (1.0 - band(4.0, gsd));
        let e = DVec3::new(1.0, 0.80, 0.55) * (4.0 * (-(dl * dl) / (2.0 * 0.5 * 0.5)).exp() * dots * smoothstep(0.25, 0.45, m.town_urban));
        s.composite(Layer { emit: e, ..Default::default() });
    }
    // ---- the town (its lights by the atlas' development)
    if let Some((tcol, th, cov, cls, shadow, em)) = s.town_px {
        let dev = 0.55 + 0.9 * t.development;
        s.composite(Layer { cov, albedo: tcol, dh: th, hmode: in_dsm, cls, emit: em * (cov * dev), lit: 1.0 - shadow, ..Default::default() });
    }
}
