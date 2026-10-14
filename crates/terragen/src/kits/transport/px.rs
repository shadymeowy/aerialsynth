//! The look of the transport features at a sample (WGSL twin: `tr_*` in
//! `gpu/wgsl/kits/transport.wgsl`, same names and order of operations): road cross-sections
//! (carriageway, markings, cars, verges, embankments and cuttings, bridge decks and their
//! shadows, lamps), railways (ballast, sleepers, rails, masts, trains), power lines (cleared
//! corridors, pylons, wires and their shadows), and the stamps (tunnel portals, stations, yards,
//! roundabouts). Coverages are exact box filters of the sub-sample footprint (`fw`), so a parent
//! pixel is the mean of its children.

use super::{cls, flag, tpl};
use crate::features::{FSeg, FStamp};
use crate::noise::*;
use crate::surface::{mixc, point_light, srgb};
use glam::{DVec2, DVec3};

/// What a sample sees.
#[derive(Clone, Copy, Debug)]
pub struct Env {
    pub p: DVec3,
    pub ground: f64,
    pub gsd: f64,
    pub fw: f64,
    /// horizontal unit vector toward the sun (ECEF), tan of its elevation
    pub sun: DVec3,
    pub sun_tan: f64,
    /// the composite below (verges and slopes are tinted from it)
    pub below: DVec3,
}

/// A feature's look at a sample.
#[derive(Clone, Copy, Debug)]
pub struct Px {
    pub cov: f64,
    pub col: DVec3,
    /// DSM height where covered (absolute)
    pub h: f64,
    pub cls: u8,
    pub emit: DVec3,
    /// cast shadow 0..1 (cars, pylons, portals), a bridge deck's shadow on the ground below
    /// (not on road surfaces), vegetation cleared 0..1, share of road surface
    pub shade: f64,
    pub shade_low: f64,
    pub clear: f64,
    pub road: f64,
    /// z-order (the top surface wins)
    pub z: f64,
}

impl Px {
    pub fn none() -> Px {
        Px { cov: 0.0, col: DVec3::ZERO, h: 0.0, cls: 0, emit: DVec3::ZERO, shade: 0.0, shade_low: 0.0, clear: 0.0, road: 0.0, z: -1e9 }
    }
}

/// `c` with coverage `a` and height `h` over what `o` holds so far.
#[inline]
fn over(o: &mut Px, a: f64, c: DVec3, h: f64) {
    let a = a.clamp(0.0, 1.0);
    if a <= 0.0 {
        return;
    }
    let c0 = o.cov;
    let c1 = 1.0 - (1.0 - c0) * (1.0 - a);
    let k = a / c1;
    o.col = o.col + (c - o.col) * k;
    o.h = o.h + (h - o.h) * k;
    o.cov = c1;
}

/// Exact box-filtered coverage of the band |d| ≤ hw by a footprint of width `fw` around `d`.
#[inline]
pub fn boxcov(d: f64, hw: f64, fw: f64) -> f64 {
    let lo = (d - 0.5 * fw).max(-hw);
    let hi = (d + 0.5 * fw).min(hw);
    ((hi - lo) / fw).clamp(0.0, 1.0)
}

/// Antialiased inside test of a signed distance (> 0 inside).
#[inline]
fn aa(sd: f64, fw: f64) -> f64 {
    (sd / fw + 0.5).clamp(0.0, 1.0)
}

/// Box-filtered share of a periodic dash pattern (on for `on` of every `period` m) around `s`.
#[inline]
pub fn dash(s: f64, period: f64, on: f64, fw: f64) -> f64 {
    let f = |x: f64| (x / period).floor() * on + (x - (x / period).floor() * period).min(on);
    ((f(s + 0.5 * fw) - f(s - 0.5 * fw)) / fw).clamp(0.0, 1.0)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// A car's paint by its hash.
pub fn car_col(h: u64) -> DVec3 {
    match (u01k(h, 2) * 5.99) as u32 {
        0 => srgb(230.0, 230.0, 228.0),
        1 => srgb(30.0, 30.0, 34.0),
        2 => srgb(150.0, 152.0, 156.0),
        3 => srgb(140.0, 30.0, 30.0),
        4 => srgb(40.0, 60.0, 120.0),
        _ => srgb(90.0, 92.0, 96.0),
    }
}

/// Cars in a parking lot (the settlements kit's parcels): stalls 2.6 m along `u`, rows of
/// 16 m in `v` (two rows of stalls, a lane between). (coverage, paint)
pub fn parked(h: u64, u: f64, v: f64, fw: f64, gsd: f64) -> (f64, DVec3) {
    let row = v / 16.0 - (v / 16.0).floor();
    let stall = (u / 2.6).floor();
    let lane = (v / 16.0).floor();
    let ch = hash2(h ^ 0xCA2, stall as i64, lane as i64);
    if (row - 0.5).abs() <= 0.2 || u01k(ch, 1) >= 0.65 {
        return (0.0, DVec3::ZERO);
    }
    let cu = u - (stall + 0.5) * 2.6;
    let cv = (row - if row > 0.5 { 0.85 } else { 0.15 }) * 16.0;
    let car = boxcov(cu, 0.9, fw) * boxcov(cv, 2.2, fw);
    (car * band(2.0, gsd), car_col(ch))
}

/// A segment's frame at `p`: signed distance (+ left), along-distance (+ `s0`), fraction, and the
/// sun's components along the segment and to its left.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub d: f64,
    pub along: f64,
    pub t: f64,
    pub len: f64,
    pub su: f64,
    pub sl: f64,
    /// along-distance and signed distance to the segment's line, not clamped to its ends
    /// (patterns along the road continue across the joints)
    pub ar: f64,
    pub dl: f64,
}

pub fn frame(sg: &FSeg, p: DVec3, sun: DVec3) -> Frame {
    let ab = sg.b - sg.a;
    let l2 = ab.length_squared().max(1e-9);
    let len = l2.sqrt();
    let tr = (p - sg.a).dot(ab) / l2;
    let t = tr.clamp(0.0, 1.0);
    let foot = sg.a + ab * t;
    let up = foot.normalize_or_zero();
    let u = ab / len;
    let left = up.cross(ab).normalize_or_zero();
    let fp = p - foot;
    let side = if fp.dot(left) < 0.0 { -1.0 } else { 1.0 };
    let s0 = sg.s0 as f64;
    Frame { d: side * fp.length(), along: s0 + t * len, t, len, su: sun.dot(u), sl: sun.dot(left), ar: s0 + tr * len, dl: (p - sg.a).dot(left) }
}

const LAMP_SODIUM: DVec3 = DVec3::new(1.0, 0.48, 0.12);
const LAMP_LED: DVec3 = DVec3::new(0.86, 0.92, 1.0);

/// Street lamps every `sp` m on the segment's own stretch (heads at lateral `off`, both sides
/// alternating unless `median`), thinned with their energy kept where they would merge.
fn lamps(f: &Frame, s0: f64, sp: f64, off: f64, median: bool, seed: u64, fw: f64) -> DVec3 {
    let m = (2.0 * fw / sp).max(1.0).log2().ceil().exp2();
    let sp_eff = sp * m;
    // (lamps half a spacing off the profile's 40 m vertices: never on a segment's end)
    let k = (f.ar / sp_eff).floor();
    let sk = (k + 0.5) * sp_eff;
    if sk < s0 || sk >= s0 + f.len {
        return DVec3::ZERO;
    }
    let side = if median || (k as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
    let dp = f.dl - if median { 0.0 } else { side * off };
    let da = f.ar - sk;
    let d2 = dp * dp + da * da;
    let pool = 0.035 * (-(dp * dp) / (2.0 * 5.0 * 5.0) - (da * da) / (2.0 * 0.3 * sp * 0.3 * sp)).exp();
    let core = point_light(d2, 6.0, 0.4, fw);
    let c = if u01k(seed, 7) < 0.5 { LAMP_LED } else { LAMP_SODIUM };
    c * ((pool + core) * m)
}

/// The deck / carriageway surface of a road or railway across its width (also slot 8 over
/// water): colour, markings, rails. `hs` is its height.
fn surface(sg: &FSeg, f: &Frame, class: usize, fw: f64, o: &mut Px, hs: f64) {
    let hw = sg.v[0] as f64;
    let fl = sg.flags;
    let dd = f.d.abs();
    let seed = sg.v[3] as u64;
    let a = boxcov(f.d, hw, fw);
    if a <= 0.0 {
        return;
    }
    let var = 0.94 + 0.12 * u01k(seed, 1);
    if class == cls::RAIL {
        let tracks = ((fl >> flag::LANES_SHIFT) & 15).max(1) as f64;
        let mut c = srgb(128.0, 118.0, 106.0) * var;
        // the nearest track: sleepers and rails
        let tc = if tracks > 1.5 {
            if f.d > 0.0 {
                2.25
            } else {
                -2.25
            }
        } else {
            0.0
        };
        let dt = f.d - tc;
        let concrete = u01k(seed, 3) < 0.6;
        let sl = dash(f.ar, 0.6, 0.26, fw) * boxcov(dt, 1.3, fw);
        c = mixc(c, if concrete { srgb(150.0, 148.0, 142.0) } else { srgb(92.0, 78.0, 64.0) }, sl);
        let rail = boxcov(dt.abs() - 0.7175, 0.036, fw);
        c = mixc(c, srgb(178.0, 178.0, 184.0), rail);
        over(o, a, c, hs);
        o.cls = cls::LANDCOVER[cls::RAIL];
        return;
    }
    let paved = fl & flag::PAVED != 0;
    let mut c = if class == cls::TRACK {
        srgb(146.0, 126.0, 100.0)
    } else if !paved {
        srgb(160.0, 146.0, 122.0)
    } else if class <= cls::TRUNK || class == cls::RAMP {
        srgb(66.0, 66.0, 68.0)
    } else {
        srgb(84.0, 82.0, 80.0)
    } * var;
    if class == cls::TRACK {
        // two ruts, grass between
        let rut = boxcov(dd - 0.8, 0.22, fw);
        c = mixc(c, srgb(118.0, 100.0, 80.0), rut);
        c = mixc(c, srgb(104.0, 118.0, 70.0), boxcov(f.d, 0.35, fw) * 0.7);
    } else if paved {
        let white = srgb(232.0, 232.0, 226.0);
        let mut mk = boxcov(dd - (hw - 0.35), 0.075, fw);
        if class == cls::MOTORWAY {
            let med = sg.v[1] as f64;
            let lanes = ((fl >> flag::LANES_SHIFT) & 15).max(1);
            mk += boxcov(dd - (med + 0.45), 0.075, fw);
            for k in 1..lanes {
                mk += boxcov(dd - (med + 0.6 + 3.6 * k as f64), 0.075, fw) * dash(f.ar, 16.0, 6.0, fw);
            }
            // the median barrier
            let bar = boxcov(f.d, 0.3, fw);
            c = mixc(c, srgb(186.0, 184.0, 178.0), bar);
        } else if class != cls::RAMP && class != cls::LOCAL {
            mk += boxcov(f.d, 0.075, fw) * dash(f.ar, 12.0, 4.0, fw);
        }
        c = mixc(c, white, mk.min(1.0));
    }
    over(o, a, c, hs);
    o.cls = sg.class;
}

/// A road or railway segment.
fn road_px(sg: &FSeg, f: &Frame, e: &Env, class: usize) -> Px {
    let mut o = Px::none();
    let fl = sg.flags;
    if fl & flag::TUNNEL != 0 {
        return o;
    }
    let hw = sg.v[0] as f64;
    let vg = sg.v[2] as f64;
    let dd = f.d.abs();
    let fw = e.fw;
    let seed = sg.v[3] as u64;
    let has_deck = super::has_deck(sg.deck_a) && super::has_deck(sg.deck_b);
    let deck = if has_deck { lerp(sg.deck_a as f64, sg.deck_b as f64, f.t) } else { e.ground };
    let bridge = fl & flag::BRIDGE != 0 && has_deck;
    let hs = if !has_deck {
        e.ground
    } else if fl & flag::FILL != 0 || bridge {
        deck.max(e.ground)
    } else {
        deck
    };
    let raised = hs - e.ground;
    let urban = fl & flag::URBAN != 0;
    let edge = hw + if bridge { 0.6 } else { vg };
    // the shadow of a bridge deck on what lies below
    if bridge && raised > 1.0 {
        let l = raised / e.sun_tan.max(0.1);
        let ds = f.dl + f.sl * l;
        let as_ = f.ar + f.su * l;
        let s0 = sg.s0 as f64;
        if as_ >= s0 && as_ <= s0 + f.len {
            o.shade_low = 0.75 * boxcov(ds, hw + 0.6, fw) * (1.0 - boxcov(f.dl, hw + 0.6, fw));
        }
    }
    // embankment / cutting slopes (1:2) down / up to the ground
    // (each segment its own stretch, measured square to its line: no scalloped joints)
    let ddl = f.dl.abs();
    let own = f.ar >= sg.s0 as f64 - 0.5 * fw - 0.3 && f.ar <= sg.s0 as f64 + f.len + 0.5 * fw + 0.3;
    if has_deck && !bridge && own && ddl > edge - fw {
        let sl = (ddl - edge).max(0.0) * 0.5;
        if raised > 0.0 {
            let toe = edge + 2.0 * raised.min(15.0);
            over(&mut o, aa(toe - ddl, fw), mixc(e.below, srgb(112.0, 124.0, 74.0), 0.2), hs - sl);
        } else {
            // (cuttings are steep rock faces (2:1), in the colour of the ground: 1:2 slopes in a
            // pale tint drew 30 m wide ribbons where roads climb canyon walls)
            let slc = (ddl - edge).max(0.0) * 2.0;
            let toe = edge + 0.5 * (-raised).min(15.0);
            let strata = 0.9 + 0.1 * ((hs + slc) / 3.0).sin();
            over(&mut o, aa(toe - ddl, fw), mixc(e.below, srgb(140.0, 128.0, 100.0) * strata, 0.2), hs + slc);
        }
        o.z = hs - sl;
        o.clear = o.cov;
    }
    // verge: shoulder, gravel, grass (sidewalks in towns; a parapet on bridges)
    // (the verge, too, square to the line: parapets and verges of decks at slightly different
    // heights left rings at every joint)
    let av = if own { boxcov(f.dl, edge, fw) } else { 0.0 };
    if av > 0.0 {
        let c = if bridge {
            srgb(184.0, 182.0, 176.0)
        } else if urban {
            mixc(srgb(172.0, 170.0, 164.0), srgb(196.0, 194.0, 188.0), boxcov(ddl - hw - 0.15, 0.15, fw))
        } else if class == cls::TRACK || class == cls::RAIL {
            mixc(e.below, srgb(150.0, 140.0, 120.0), 0.5)
        } else {
            mixc(mixc(e.below, srgb(92.0, 108.0, 66.0), 0.55), srgb(140.0, 136.0, 126.0), boxcov(ddl - hw, 0.8, fw))
        };
        over(&mut o, av, c, hs + if bridge { 1.0 } else { 0.0 });
        o.z = hs;
        o.clear = o.clear.max(av);
    }
    // the carriageway
    let a = boxcov(f.d, hw, fw);
    if a > 0.0 {
        surface(sg, f, class, fw, &mut o, hs);
        o.road = a;
        o.z = hs + 0.5 + 0.01 * class as f64 + if fl & flag::OVER != 0 { 20.0 } else { 0.0 };
        if class == cls::RAIL {
            // catenary masts, a train now and then
            let tracks = ((fl >> flag::LANES_SHIFT) & 15).max(1) as f64;
            if fl & flag::ELECTRIC != 0 {
                let mk = (f.ar / 60.0).round();
                let mast = boxcov(f.ar - mk * 60.0, 0.25, fw) * boxcov(dd - hw + 0.3, 0.25, fw);
                o.col = mixc(o.col, srgb(110.0, 112.0, 116.0), mast);
            }
            if u01k(seed, 11) < 0.35 {
                let tl = 120.0 + 260.0 * u01k(seed, 14);
                let st = u01k(seed, 12) * (super::S_WRAP - tl);
                let tc = if tracks > 1.5 {
                    if u01k(seed, 13) < 0.5 {
                        2.25
                    } else {
                        -2.25
                    }
                } else {
                    0.0
                };
                let x = f.ar - st;
                if x > 0.0 && x < tl {
                    let body = boxcov(f.d - tc, 1.45, fw) * dash(x, 25.0, 24.2, fw);
                    let tk = (u01k(seed, 15) * 3.99) as u32;
                    let tcol = if tk == 0 {
                        srgb(170.0, 40.0, 36.0)
                    } else if tk == 1 {
                        srgb(220.0, 222.0, 224.0)
                    } else if tk == 2 {
                        srgb(40.0, 70.0, 140.0)
                    } else {
                        srgb(150.0, 152.0, 150.0)
                    };
                    o.col = mixc(o.col, tcol * (0.9 + 0.1 * boxcov(f.d - tc, 0.5, fw)), body);
                    o.h = lerp(o.h, hs + 4.0, body);
                }
            }
        } else if sg.ha > 0.0 {
            // cars in their lanes (none on tracks)
            let dens = sg.ha as f64;
            let med = sg.v[1] as f64;
            let lanes = ((fl >> flag::LANES_SHIFT) & 15).max(1) as f64;
            let (lw, l0) = if class == cls::MOTORWAY {
                (3.6, med + 0.6)
            } else if class == cls::RAMP {
                (2.0 * hw, -hw)
            } else {
                (hw, 0.0)
            };
            let x = if class == cls::RAMP { f.d - l0 } else { dd - l0 };
            let k = (x / lw).floor();
            if k >= 0.0 && k < if class == cls::MOTORWAY { lanes } else { 1.0 } {
                let lc = l0 + (k + 0.5) * lw;
                let ls = if class == cls::MOTORWAY || class == cls::RAMP { 24.0 } else { 30.0 };
                let lane_id = k as i64 + if f.d < 0.0 && class != cls::RAMP { 8 } else { 0 };
                let j = (f.ar / ls).floor();
                let hc = hash3(seed ^ 0xCA5, j as i64, lane_id, 0);
                let e_car = band(3.0, e.gsd);
                if u01k(hc, 1) < dens {
                    let truck = class <= cls::TRUNK && u01k(hc, 3) < 0.15;
                    let (cl, cw) = if truck { (7.0, 1.25) } else { (2.25, 0.92) };
                    let ac = (j + 0.5 + 0.4 * (u01k(hc, 4) - 0.5)) * ls;
                    let du = f.ar - ac;
                    let dv = (if class == cls::RAMP { f.d } else { dd }) - lc;
                    let car = boxcov(du, cl, fw) * boxcov(dv, cw, fw);
                    let ccol = if truck { mixc(srgb(200.0, 200.0, 196.0), car_col(hc), 0.4) } else { car_col(hc) };
                    o.col = mixc(o.col, ccol, car * e_car);
                    // its shadow on the road
                    let sh = 1.5 / e.sun_tan.max(0.1);
                    let sd = if class == cls::RAMP || f.d >= 0.0 { 1.0 } else { -1.0 };
                    let car_s = boxcov(du + f.su * sh, cl, fw) * boxcov(dv + sd * f.sl * sh, cw, fw);
                    o.shade = o.shade.max(0.6 * car_s * (1.0 - car) * e_car);
                }
                // the mean of unresolved cars
                let mean = dens * 4.5 * 1.85 / (ls * lw);
                o.col = mixc(o.col, srgb(110.0, 110.0, 114.0), mean * (1.0 - e_car));
            }
        }
    }
    // lamps
    if sg.hb > 0.0 {
        let median = class == cls::MOTORWAY;
        o.emit = lamps(f, sg.s0 as f64, sg.hb as f64, hw + 0.5, median, seed, fw) * if urban { 1.2 } else { 1.0 };
    }
    o
}

/// A power line: the cleared corridor, pylons (lattice towers with cross-arms), wires, and their
/// shadows.
fn power_px(sg: &FSeg, f: &Frame, e: &Env) -> Px {
    let mut o = Px::none();
    let hw = sg.v[0] as f64;
    let sp = sg.v[1] as f64;
    let hgt = sg.v[2] as f64;
    let fw = e.fw;
    o.clear = boxcov(f.dl, hw, fw);
    let k = (f.ar / sp).round();
    let sk = k * sp;
    let s0 = sg.s0 as f64;
    let arm = 0.3 * hw + 4.0;
    let tan = e.sun_tan.max(0.1);
    // wires: three conductors, sagging between the towers
    let x = f.ar - (f.ar / sp).floor() * sp;
    let sag = 1.0 - (2.0 * x / sp - 1.0).powi(2);
    let wh = 0.82 * hgt - 0.22 * hgt * sag;
    let lw = wh / tan;
    let dw = f.dl + f.sl * lw;
    let mut wire = 0.0;
    let mut wsh = 0.0;
    for c in [-1.0, 0.0, 1.0] {
        wire += boxcov(f.dl - c * arm, 0.02, fw);
        wsh += boxcov(dw - c * arm, 0.02, fw);
    }
    if sk >= s0 && sk < s0 + f.len {
        let a = f.ar - sk;
        let m = a.abs().max(f.dl.abs());
        // the footprint: legs, bracing; the cross-arm
        let foot = boxcov(a, 4.5, fw) * boxcov(f.dl, 4.5, fw);
        let frame_l = (boxcov(m - 4.2, 0.3, fw) + 0.5 * boxcov(a.abs() - f.dl.abs(), 0.25, fw)).min(1.0);
        let arm_c = boxcov(a, 0.7, fw) * boxcov(f.dl, arm + 1.0, fw);
        let e_t = band(4.0, e.gsd);
        let cov = (frame_l * foot * e_t + 0.45 * foot * (1.0 - e_t)).max(arm_c * 0.8);
        let h = if m < 4.5 { e.ground + hgt * (1.0 - m / 4.5) } else { e.ground + 0.85 * hgt };
        over(&mut o, cov, srgb(150.0, 152.0, 156.0), h.max(e.ground + 0.85 * hgt * arm_c.min(1.0)));
        o.z = e.ground;
    }
    // tower shadows (the nearest two towers): a tapering wedge and the arm
    let mut sh: f64 = 0.0;
    for kk in [k - 1.0, k, k + 1.0] {
        let skk = kk * sp;
        if skk < s0 || skk >= s0 + f.len {
            continue;
        }
        let q = DVec2::new(f.ar - skk, f.dl);
        let sv = DVec2::new(f.su, f.sl).normalize_or(DVec2::X);
        let r = -q.dot(sv);
        let wdt = (q - (-sv) * r).length();
        let ls = hgt / tan;
        if r > 0.0 && r < ls {
            let half = 4.5 * (1.0 - r / ls) + 0.3;
            sh = sh.max(0.55 * aa(half - wdt, fw));
            let ra = 0.85 * ls;
            // the arm: a bar across the line at the arm's height
            let qa = q + sv * ra;
            sh = sh.max(0.5 * boxcov(qa.x, 0.7, fw) * boxcov(qa.y, arm + 1.0, fw));
        }
    }
    o.shade = sh.max(0.5 * wsh.min(1.0));
    if wire > 0.0 {
        over(&mut o, 0.7 * wire.min(1.0), srgb(40.0, 40.0, 44.0), e.ground + wh);
    }
    o
}

/// A segment's look (by kind).
pub fn seg_px(sg: &FSeg, e: &Env) -> Px {
    let class = (sg.kind - cls::KIND0) as usize;
    let f = frame(sg, e.p, e.sun);
    if f.d.abs() > sg.v[0] as f64 + 40.0 + 2.0 * e.fw && class != cls::POWER || class == cls::POWER && f.d.abs() > sg.hw as f64 + 2.0 * e.fw {
        return Px::none();
    }
    if class == cls::POWER {
        power_px(sg, &f, e)
    } else {
        road_px(sg, &f, e, class)
    }
}

/// Slot 8: a road or railway deck over a river or standing water (`wl`: the water surface).
pub fn deck_px(sg: &FSeg, p: DVec3, ground: f64, wl: f64, fw: f64) -> Px {
    let mut o = Px::none();
    let class = (sg.kind - cls::KIND0) as usize;
    if class == cls::POWER || sg.flags & flag::TUNNEL != 0 {
        return o;
    }
    let f = frame(sg, p, DVec3::ZERO);
    let hw = sg.v[0] as f64;
    if f.d.abs() > hw + 0.6 + fw {
        return o;
    }
    let base = ground.max(wl) + 2.5;
    let hs = if super::has_deck(sg.deck_a) && super::has_deck(sg.deck_b) { lerp(sg.deck_a as f64, sg.deck_b as f64, f.t).max(base) } else { base };
    over(&mut o, boxcov(f.d, hw + 0.6, fw), srgb(184.0, 182.0, 176.0), hs + 1.0);
    surface(sg, &f, class, fw, &mut o, hs);
    o.cls = crate::landcover::BRIDGE;
    o.road = boxcov(f.d, hw, fw);
    o.z = hs;
    o
}

/// A stamp's look (portals, stations, yards, roundabouts).
pub fn stamp_px(st: &FStamp, e: &Env) -> Px {
    let mut o = Px::none();
    let d = e.p - st.center;
    let q = DVec2::new(d.dot(st.ex), d.dot(st.ey));
    let fw = e.fw;
    if q.x.abs() > st.half[0] as f64 + fw || q.y.abs() > st.half[1] as f64 + fw {
        return o;
    }
    let seed = st.v[2] as u64;
    match st.template {
        tpl::PORTAL => {
            // the headwall across the road at the tunnel mouth, the dark mouth before it
            let hw = st.v[0] as f64;
            let hd = if super::has_deck(st.h) { st.h as f64 } else { e.ground };
            let wall = boxcov(q.x - 1.25, 1.25, fw) * boxcov(q.y, hw + 4.0, fw);
            over(&mut o, wall, srgb(176.0, 173.0, 166.0), hd.max(e.ground) + 7.5);
            let mouth = boxcov(q.x + 1.5, 1.5, fw) * boxcov(q.y, hw, fw);
            over(&mut o, mouth, srgb(26.0, 26.0, 30.0), hd);
            o.shade = 0.8 * mouth;
            o.z = hd + 30.0;
            o.cls = crate::landcover::BUILDING;
        }
        tpl::STATION => {
            let tracks = st.v[0] as f64;
            let len = st.v[1] as f64;
            let side = st.v[3] as f64;
            let w0 = 0.5 * (tracks - 1.0) * 5.5;
            let inl = boxcov(q.x, 0.5 * len, fw);
            // tracks with platforms between pairs, canopies over the platforms
            let ballast = inl * boxcov(q.y, w0 + 2.6, fw);
            over(&mut o, ballast, srgb(124.0, 114.0, 102.0), e.ground);
            let kf = ((q.y + w0) / 5.5).round().clamp(0.0, tracks - 1.0);
            let tcy = kf * 5.5 - w0;
            let dt = q.y - tcy;
            let rail = boxcov(dt.abs() - 0.7175, 0.036, fw) * inl;
            let sleep = dash(q.x, 0.6, 0.26, fw) * boxcov(dt, 1.3, fw) * inl;
            o.col = mixc(mixc(o.col, srgb(150.0, 148.0, 142.0), sleep), srgb(178.0, 178.0, 184.0), rail);
            let kp = ((q.y + w0 - 2.75) / 11.0).round();
            let pcy = kp * 11.0 - w0 + 2.75;
            let platform = inl * boxcov(q.y - pcy, 1.6, fw) * (if pcy.abs() < w0 { 1.0 } else { 0.0 });
            over(&mut o, platform, srgb(176.0, 172.0, 164.0), e.ground + 0.9);
            let edge_l = platform * boxcov((q.y - pcy).abs() - 1.45, 0.08, fw);
            o.col = mixc(o.col, srgb(222.0, 196.0, 60.0), edge_l);
            let canopy = boxcov(q.x, 0.32 * len, fw) * boxcov(q.y - pcy, 2.4, fw) * (if pcy.abs() < w0 { 1.0 } else { 0.0 });
            let roof = if u01k(seed, 1) < 0.5 { srgb(150.0, 70.0, 56.0) } else { srgb(120.0, 124.0, 130.0) };
            over(&mut o, canopy * band(3.0, e.gsd).max(0.6), roof, e.ground + 5.5);
            // the station building and its forecourt on one side
            let yb = side * (w0 + 18.0);
            let bl = 0.12 * len + 14.0;
            let bld = boxcov(q.x, bl, fw) * boxcov(q.y - yb, 10.0, fw);
            over(&mut o, bld, roof * 0.9, e.ground + 11.0);
            let court = boxcov(q.x, bl + 12.0, fw) * boxcov(q.y - side * (w0 + 33.0), 6.0, fw);
            over(&mut o, court, srgb(168.0, 162.0, 152.0), e.ground);
            o.emit = DVec3::new(0.9, 0.92, 1.0) * (0.05 * (platform + court) * band(20.0, e.gsd) + 0.02 * ballast * (1.0 - band(20.0, e.gsd)));
            o.cls = if bld > 0.5 { crate::landcover::BUILDING } else { crate::landcover::RAILWAY };
            o.road = ballast.max(bld).max(court);
            o.clear = o.cov;
            o.z = e.ground + 40.0;
        }
        tpl::YARD => {
            let tracks = st.v[0] as f64;
            let len = st.v[1] as f64;
            let w0 = 0.5 * (tracks - 1.0) * 5.0;
            let kf = ((q.y + w0) / 5.0).round().clamp(0.0, tracks - 1.0);
            let ki = kf - 0.5 * (tracks - 1.0);
            // the ladder: outer tracks end earlier
            let tl = 0.5 * len - ki.abs() * 32.0;
            let area = boxcov(q.y, w0 + 3.0, fw) * aa(0.5 * len - q.x.abs() - (q.y.abs() - 3.0).max(0.0) * 6.4, fw);
            over(&mut o, area, srgb(118.0, 108.0, 96.0), e.ground);
            let dt = q.y - (kf * 5.0 - w0);
            let on = aa(tl - q.x.abs(), fw);
            let rail = boxcov(dt.abs() - 0.7175, 0.036, fw) * on;
            let sleep = dash(q.x, 0.6, 0.26, fw) * boxcov(dt, 1.3, fw) * on;
            o.col = mixc(mixc(o.col, srgb(92.0, 78.0, 64.0), sleep), srgb(170.0, 170.0, 176.0), rail);
            // wagons: strings of 15 m cars on most tracks
            let hk = hash2(seed, kf as i64, 0);
            if u01k(hk, 1) < 0.7 {
                let w_lo = -tl + 40.0 + u01k(hk, 2) * 0.3 * len;
                let w_hi = tl - 40.0 - u01k(hk, 3) * 0.3 * len;
                let x = q.x - w_lo;
                if q.x > w_lo && q.x < w_hi {
                    let wi = (x / 16.0).floor();
                    let hw2 = hash2(hk, wi as i64, 1);
                    let wk = (u01k(hw2, 1) * 5.99) as u32;
                    let wc = if wk == 0 {
                        srgb(120.0, 62.0, 44.0)
                    } else if wk == 1 {
                        srgb(70.0, 72.0, 78.0)
                    } else if wk == 2 {
                        srgb(40.0, 70.0, 130.0)
                    } else if wk == 3 {
                        srgb(40.0, 40.0, 42.0)
                    } else if wk == 4 {
                        srgb(200.0, 200.0, 196.0)
                    } else {
                        srgb(60.0, 110.0, 70.0)
                    };
                    let body = boxcov(dt, 1.45, fw) * dash(x, 16.0, 15.0, fw);
                    over(&mut o, body * area, wc, e.ground + 4.0);
                }
            }
            // floodlight masts along the middle
            let mk = (q.x / 150.0).round();
            let d2 = (q.x - mk * 150.0).powi(2) + q.y.powi(2);
            o.emit = DVec3::new(1.0, 0.6, 0.25) * point_light(d2, 4.0, 0.6, fw) + DVec3::new(1.0, 0.6, 0.25) * 0.03 * area;
            o.cls = crate::landcover::RAILWAY;
            o.road = area;
            o.clear = area;
            o.z = e.ground + 40.0;
        }
        tpl::ROUNDABOUT => {
            let r = st.v[0] as f64;
            let hw = st.v[1] as f64;
            let rr = q.length();
            let ring = boxcov(rr - (r - hw), hw, fw);
            let island = aa(r - 2.0 * hw - rr, fw);
            let mut c = srgb(70.0, 70.0, 72.0);
            c = mixc(c, srgb(232.0, 232.0, 226.0), boxcov(rr - (r - 0.35), 0.075, fw) + boxcov(rr - (r - 2.0 * hw + 0.35), 0.075, fw));
            over(&mut o, ring, c, e.ground);
            let curb = boxcov(rr - (r - 2.0 * hw - 0.5), 0.5, fw);
            let green = if u01k(seed, 1) < 0.7 { srgb(96.0, 128.0, 66.0) } else { srgb(170.0, 160.0, 140.0) };
            over(&mut o, island, green, e.ground + 0.3);
            over(&mut o, curb, srgb(190.0, 188.0, 182.0), e.ground + 0.2);
            // a tree or a monument in the middle
            let tr = (r - 2.0 * hw) * 0.35;
            let tree = aa(tr - rr, fw) * if u01k(seed, 2) < 0.6 { 1.0 } else { 0.0 };
            over(&mut o, tree * band(3.0, e.gsd).max(0.5), srgb(60.0, 86.0, 44.0), e.ground + 7.0);
            o.cls = if ring > 0.5 { cls::LANDCOVER[cls::SECONDARY] } else { crate::landcover::GRASS };
            o.road = ring;
            o.clear = o.cov;
            o.z = e.ground + 30.0;
        }
        _ => {}
    }
    o
}
