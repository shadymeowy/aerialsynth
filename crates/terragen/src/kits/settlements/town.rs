//! The settlement pixel model (`docs/design/terrain-next.md` §6.2): a place stamp's footprint,
//! zones, districts, street patterns, hashed-BSP lots, building typologies, special parcels,
//! village forms, lights. WGSL twin: `st_*` in `gpu/wgsl/kits/settlements.wgsl` (same names,
//! same order of operations).
//!
//! ```text
//!  place (stamp) ─► footprint  rel = |q| / (R (1 + noise)); zone by rel (CBD … peri-urban)
//!                ├► district   W2 in the place frame (cell by level); pattern by zone × culture,
//!                │             industrial wedges downwind; district edges are arterials
//!                ├► block      grid / radial / organic / curvilinear / superblock / informal /
//!                │             industrial / street village / farms / kraals
//!                ├► lot        hashed BSP of the block (depth by zone)
//!                └► building   typology by zone × culture × lot: house, row, perimeter block,
//!                              mid-rise, tower, warehouse, hut; special parcels per block
//! ```

use crate::noise::*;
use crate::surface::{band_cov, mixc, point_light, srgb};
use glam::{DVec2, DVec3};

use super::places::form;

/// Street patterns (and village layouts).
pub mod pat {
    pub const GRID: u32 = 0;
    pub const RADIAL: u32 = 1;
    pub const ORGANIC: u32 = 2;
    pub const CURVI: u32 = 3;
    pub const SUPER: u32 = 4;
    pub const INFORMAL: u32 = 5;
    pub const INDUSTRIAL: u32 = 6;
    pub const LINEAR: u32 = 7;
    pub const DISPERSED: u32 = 8;
    pub const KRAAL: u32 = 9;
}

/// Zones by relative distance from the centre.
pub mod zone {
    pub const CBD: u32 = 0;
    pub const INNER: u32 = 1;
    pub const OUTER: u32 = 2;
    pub const SUBURB: u32 = 3;
    pub const PERI: u32 = 4;
}

/// Building typologies.
pub mod typ {
    pub const HOUSE: u32 = 1;
    pub const ROW: u32 = 2;
    pub const PERIM: u32 = 3;
    pub const MIDRISE: u32 = 4;
    pub const TOWER: u32 = 5;
    pub const WAREHOUSE: u32 = 6;
}

/// Special parcels (whole blocks).
pub mod parcel {
    pub const NONE: u32 = 0;
    pub const PARK: u32 = 1;
    pub const PLAZA: u32 = 2;
    pub const PARKING: u32 = 3;
    pub const SPORTS: u32 = 4;
    pub const CEMETERY: u32 = 5;
    pub const STADIUM: u32 = 6;
    pub const SITE: u32 = 7;
}

/// A place stamp, decoded.
#[derive(Clone, Copy, Debug)]
pub struct Pl {
    pub c: DVec3,
    pub ex: DVec3,
    pub ey: DVec3,
    pub r: f64,
    pub level: u32,
    pub form: u32,
    pub sub: bool,
    pub seed: u64,
    pub arch: u32,
    pub dev: f64,
    pub elong: f64,
    /// downwind direction in the place frame (radians)
    pub wind: f64,
    pub pop: f64,
}

pub fn decode(st: &crate::features::FStamp) -> Pl {
    let lv = st.v[1] as u32;
    Pl {
        c: st.center,
        ex: st.ex,
        ey: st.ey,
        r: st.v[0] as f64,
        level: lv & 7,
        form: (lv >> 3) & 7,
        sub: (lv >> 6) & 1 == 1,
        seed: mix64((st.v[2] as u64) | ((st.v[3] as u64) << 24)),
        arch: st.v[4] as u32,
        dev: st.v[5] as f64,
        elong: st.v[6] as f64,
        wind: st.v[7] as f64,
        pop: st.h as f64,
    }
}

/// The place frame coordinates of `p` (m).
pub fn local(pl: &Pl, p: DVec3) -> DVec2 {
    let d = p - pl.c;
    DVec2::new(d.dot(pl.ex), d.dot(pl.ey))
}

/// Relative distance from the centre in the elongated frame, without the footprint noise.
pub fn rel0(pl: &Pl, q: DVec2) -> f64 {
    let se = pl.elong.sqrt();
    DVec2::new(q.x / se, q.y * se).length() / pl.r
}

/// The footprint noise at `p` (relative radius change).
pub fn foot_noise(pl: &Pl, p: DVec3, gsd: f64) -> f64 {
    let r = pl.r;
    0.32 * perlin3(pl.seed ^ 0x71, p / (0.9 * r)) + 0.18 * perlin3(pl.seed ^ 0x72, p / (0.35 * r)) * band(0.35 * r, gsd)
}

/// Built-up density at relative distance `rel` (1 in the core, 0 outside): villages are
/// compact, towns and cities thin out over their outer half.
pub fn density(pl: &Pl, rel: f64) -> f64 {
    if pl.form == form::CITY as u32 {
        1.0 - smoothstep(0.35, 1.0, rel)
    } else {
        1.0 - smoothstep(0.45, 1.0, rel)
    }
}

/// Is `q` inside the strips of a forest-strip village (in the place frame)?
pub fn in_strips(pl: &Pl, q: DVec2) -> bool {
    pl.form == form::WALDHUFEN as u32 && q.x.abs() < 0.9 * pl.r * pl.elong.sqrt() && q.y.abs() < super::places::WALDHUFEN_STRIP + 60.0
}

/// Zone of a relative distance.
pub fn zone_of(pl: &Pl, rel: f64) -> u32 {
    let cbd = if pl.level >= 3 {
        0.07 + 0.03 * (pl.level as f64 - 3.0) + 0.03 * u01k(pl.seed, 40)
    } else if pl.level == 2 {
        0.07
    } else {
        0.0
    };
    if rel < cbd {
        zone::CBD
    } else if rel < 0.35 {
        zone::INNER
    } else if rel < 0.6 {
        zone::OUTER
    } else if rel < 0.85 {
        zone::SUBURB
    } else {
        zone::PERI
    }
}

#[inline]
fn wrap_pi(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    a - t * ((a + std::f64::consts::PI) / t).floor()
}

#[inline]
fn rot2(v: DVec2, a: f64) -> DVec2 {
    let (s, c) = a.sin_cos();
    DVec2::new(c * v.x + s * v.y, -s * v.x + c * v.y)
}

/// The street pattern of a town (culture; industrial districts apart): one layout for the whole
/// town, its zones differ by their buildings (districts of different patterns and orientations
/// read as pieces of different towns glued together; v0.2.0 drew one grid per town); the
/// peri-urban ring of poor towns is informal.
pub fn pick_pattern(pl: &Pl, z: u32, did: u64, industrial: bool) -> u32 {
    use pat::*;
    if industrial {
        return INDUSTRIAL;
    }
    let u = u01k(did, 4);
    if z == zone::PERI && pl.dev < 0.45 && matches!(pl.arch, 0 | 1 | 2 | 3 | 8) && u < 0.65 {
        return INFORMAL;
    }
    match pl.arch {
        3 => ORGANIC,
        // (curvilinear suburbs in the big cities: one grid family to the horizon read as a
        // pattern, metro; the old cores of old cultures are their grid's, `old_core`)
        5 | 6 | 7 | 9 if (z == zone::SUBURB || z == zone::PERI) && pl.level >= 3 && u < 0.75 => CURVI,
        4 | 9 if u < 0.5 => SUPER,
        _ => GRID,
    }
}

/// Do a city's industrial estates continue its street grid (blocks of two or three of its
/// blocks: no estate of its own grid inside a polygon of arterials, metro)? Where the town's
/// pattern is the grid.
pub fn shared_industry(pl: &Pl) -> bool {
    pick_pattern(pl, zone::OUTER, mix64(pl.seed ^ 0x2A7E), false) == pat::GRID
}

/// The distance from the centre (m, in the elongated place frame) and the radius of the ring
/// boulevard around a big city's old core there (a wobbling ring a little outside the CBD), for
/// the cities of old cultures; None: none.
fn core_ring(pl: &Pl, q: DVec2) -> Option<(f64, f64)> {
    if pl.form != form::CITY as u32 || pl.level < 2 || !matches!(pl.arch, 5 | 6 | 8 | 11) {
        return None;
    }
    let se = pl.elong.sqrt();
    let qe = DVec2::new(q.x / se, q.y * se);
    let phi = qe.y.atan2(qe.x);
    let t = std::f64::consts::TAU;
    let s = pl.seed ^ 0x219;
    let cbd = if pl.level >= 3 { 0.07 + 0.03 * (pl.level as f64 - 3.0) + 0.03 * u01k(pl.seed, 40) } else { 0.07 };
    let rr = pl.r
        * (1.25 * cbd + 0.015)
        * (1.0 + 0.12 * (2.0 * phi + t * u01k(s, 1)).sin() + 0.07 * (3.0 * phi + t * u01k(s, 2)).sin() + 0.04 * (5.0 * phi + t * u01k(s, 3)).sin());
    Some((qe.length(), rr))
}

/// Distance (m, in the elongated place frame) to the ring boulevard around a big city's old core
/// (1e9: none).
pub fn ring_dist(pl: &Pl, q: DVec2) -> f64 {
    match core_ring(pl, q) {
        Some((d, rr)) if pl.level >= 3 => (d - rr).abs(),
        _ => 1e9,
    }
}

/// The old quarter of a city of an old culture: its centre (place frame, elongated) and radius
/// (m; the outline wobbles by a fifth). Towns: around the centre, 1.5 of the ring's radius; big
/// cities: beside the CBD, astride the ring boulevard (round 11's quarter inside the CBD read as a
/// pale low disc in the ring of towers, metro oblique).
fn old_quarter(pl: &Pl, q: DVec2) -> Option<(f64, f64)> {
    if pl.form != form::CITY as u32 || pl.level < 2 || !matches!(pl.arch, 5 | 6 | 8 | 11) {
        return None;
    }
    // (the ring's mean radius: its wobble moved the quarter's centre with the direction of `q`)
    let cbd = if pl.level >= 3 { 0.07 + 0.03 * (pl.level as f64 - 3.0) + 0.03 * u01k(pl.seed, 40) } else { 0.07 };
    let rr = pl.r * (1.25 * cbd + 0.015);
    let se = pl.elong.sqrt();
    let s = pl.seed ^ 0x219;
    let a = std::f64::consts::TAU * u01k(s, 4);
    let (c0, ro) = if pl.level >= 3 { (0.95 * rr, 0.62 * rr) } else { (0.2 * rr, 1.5 * rr) };
    let rel = DVec2::new(q.x / se, q.y * se) - DVec2::new(a.cos(), a.sin()) * c0;
    // (an irregular outline: a round stamp read as stamped in, town@type)
    let phi = rel.y.atan2(rel.x);
    let t = std::f64::consts::TAU;
    let wob = 1.0 + 0.16 * (3.0 * phi + t * u01k(s, 5)).sin() + 0.09 * (5.0 * phi + t * u01k(s, 6)).sin() + 0.05 * (8.0 * phi + t * u01k(s, 7)).sin();
    Some((rel.length() / wob, ro))
}

/// How much `q` lies in the old core of a city of an old culture (1 inside, 0 outside, fading over
/// the outer half of its radius): the town's grid there is bent, its streets narrower and its
/// blocks cut by lanes into small blocks of terraced houses (`old_sub`; round 11: round 10's core
/// of its own W2 street network read as a crisp patch of rubble against the grid, metro).
pub fn old_core(pl: &Pl, q: DVec2) -> f64 {
    match old_quarter(pl, q) {
        Some((d, ro)) => 1.0 - smoothstep(0.45 * ro, ro, d),
        None => 0.0,
    }
}

/// Can a small block of the old quarter reach `q` (within 70 m of its outline)?
fn old_near(pl: &Pl, q: DVec2) -> bool {
    matches!(old_quarter(pl, q), Some((d, ro)) if d < ro + 70.0)
}

/// A small block of an old quarter: the grid block cut by lanes (~30-44 m apart, the lanes across
/// staggered from column to column, 6 in 10 of them open: houses back to back where none) into
/// small blocks; its id, the distance to its edge (an open lane's half width off), its centre
/// (block coordinates), the side of its nearest edge (0-3) and the coordinate along it.
#[derive(Clone, Copy, Debug)]
pub struct OldSub {
    pub id: u64,
    pub d: f64,
    pub c: DVec2,
    pub side: i64,
    pub along: f64,
}

pub fn old_sub(blk: &Blk) -> OldSub {
    let lw = 30.0 + 14.0 * u01k(blk.id, 15);
    let lane_hw = 1.4 + 0.8 * u01k(blk.id, 78);
    let nu = (blk.bw / lw).round().max(1.0);
    let nv = (blk.bd / lw).round().max(1.0);
    let (su, sv) = (blk.bw / nu, blk.bd / nv);
    let line = |k: f64, n: f64, s: f64, len: f64, salt: i64| -> f64 {
        if k <= 0.0 {
            0.0
        } else if k >= n {
            len
        } else {
            (k + 0.44 * (u01(hash2(blk.id ^ 0x1A5, salt, k as i64)) - 0.5)) * s
        }
    };
    let mut iu = (blk.u / su).floor().clamp(0.0, nu - 1.0);
    if blk.u < line(iu, nu, su, blk.bw, -1) {
        iu -= 1.0;
    } else if blk.u >= line(iu + 1.0, nu, su, blk.bw, -1) {
        iu += 1.0;
    }
    let iu = iu.clamp(0.0, nu - 1.0);
    let salt = iu as i64;
    let mut iv = (blk.v / sv).floor().clamp(0.0, nv - 1.0);
    if blk.v < line(iv, nv, sv, blk.bd, salt) {
        iv -= 1.0;
    } else if blk.v >= line(iv + 1.0, nv, sv, blk.bd, salt) {
        iv += 1.0;
    }
    let iv = iv.clamp(0.0, nv - 1.0);
    let (u0, u1) = (line(iu, nu, su, blk.bw, -1), line(iu + 1.0, nu, su, blk.bw, -1));
    let (v0, v1) = (line(iv, nv, sv, blk.bd, salt), line(iv + 1.0, nv, sv, blk.bd, salt));
    let lane = |k: f64, n: f64, salt: i64| if k > 0.0 && k < n && u01k(hash2(blk.id ^ 0x1A6, salt, k as i64), 1) < 0.6 { lane_hw } else { 0.0 };
    let ds = [blk.u - u0 - lane(iu, nu, -1), u1 - blk.u - lane(iu + 1.0, nu, -1), blk.v - v0 - lane(iv, nv, salt), v1 - blk.v - lane(iv + 1.0, nv, salt)];
    let mut side = 0;
    for k in 1..4 {
        if ds[k] < ds[side] {
            side = k;
        }
    }
    OldSub {
        id: hash3(blk.id ^ 0x01D, iu as i64, iv as i64, 5),
        d: ds[side],
        c: DVec2::new(0.5 * (u0 + u1), 0.5 * (v0 + v1)),
        side: side as i64,
        along: if side < 2 { blk.v } else { blk.u },
    }
}

/// A sample's block: which block, where in it, and the nearest street.
#[derive(Clone, Copy, Debug, Default)]
pub struct Blk {
    pub id: u64,
    /// position in the block interior (m from its corner) and its size
    pub u: f64,
    pub v: f64,
    pub bw: f64,
    pub bd: f64,
    /// distance to the nearest street's centre line, its half width, the coordinate along it,
    /// its id, a main street
    pub ds: f64,
    pub sw: f64,
    pub along: f64,
    pub sid: u64,
    pub major: bool,
    /// main streets along the block's sides (bits: x0, x1, y0, y1; grids)
    pub mj: u32,
    /// grids: the nearest street's segment between two crossings (its hash; 0: none) and the
    /// offset (place frame) to its middle
    pub seg: u64,
    pub smid: DVec2,
}

/// What a position of a place shows (`place_eval`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ev {
    /// coverage (crisp: a street, a lot or nothing), colour, height above ground (m), class
    pub cov: f64,
    pub col: DVec3,
    pub h: f64,
    pub cls: u8,
    /// roof coverage (lamps light the ground only), the lit window rim, its colour
    pub roof: f64,
    pub rim: f64,
    pub win: DVec3,
    /// emission besides lamps and windows (floods, porch lights)
    pub em: DVec3,
    /// tree crowns (yards, parks, streets): coverage and height
    pub tree: f64,
    pub tree_h: f64,
    /// the nearest street (lamps)
    pub ds: f64,
    pub sw: f64,
    pub along: f64,
    pub sid: u64,
    /// the lot of the building at the position (0: no building): the instance id's hash
    pub bid: u64,
    pub major: bool,
    pub street: f64,
    pub zone: u32,
    pub pattern: u32,
    /// lamp spacing (m), 0: no lamps
    pub lamp_sp: f64,
}

/// Inputs of a place evaluation that do not depend on the position.
#[derive(Clone, Copy, Debug)]
pub struct Env {
    pub gsd: f64,
    pub fw: f64,
    /// footprint noise at the sample
    pub n: f64,
    pub moist: f64,
    pub detail: f64,
    /// the ground below the place (fields, grass): yards and gardens take after it
    pub ground: DVec3,
}

// ---- colours (sRGB)
fn c_asphalt() -> DVec3 {
    srgb(58.0, 58.0, 62.0)
}
fn c_concrete() -> DVec3 {
    srgb(158.0, 154.0, 146.0)
}
fn c_paving() -> DVec3 {
    srgb(176.0, 164.0, 146.0)
}

/// Yard ground: green where moist, dusty where dry.
fn c_yard(moist: f64, u: f64) -> DVec3 {
    mixc(srgb(150.0, 132.0, 102.0), srgb(86.0, 110.0, 54.0), smoothstep(0.22, 0.6, moist)) * (0.9 + 0.2 * u)
}

/// A car park's surface at (u, v) of its parcel: asphalt, stall lines in rows of 16 m, the
/// transport kit's parked cars.
fn car_park(h: u64, u: f64, v: f64, fw: f64, gsd: f64) -> DVec3 {
    let row = v / 16.0 - (v / 16.0).floor();
    let in_row = if (row - 0.5).abs() > 0.2 { 1.0 } else { 0.0 };
    let line = band_cov((u / 2.6 - (u / 2.6).round()) * 2.6, 0.08, fw) * in_row;
    let col = mixc(c_asphalt() * 1.45, srgb(210.0, 210.0, 204.0), line * band(1.0, gsd));
    // (the cars' mean below 2 m pixels: empty dark asphalt read as a hole, town@type)
    let col = mixc(col, srgb(128.0, 128.0, 130.0), 0.3 * (1.0 - band(2.0, gsd)));
    let (pc, pcol) = crate::kits::transport::px::parked(h, u, v, fw, gsd);
    mixc(col, pcol, pc)
}

/// Roof colour by culture (`arch`), hash `u`, flat roofs (`flat`).
pub fn roof_col(arch: u32, u: f64, flat: bool) -> DVec3 {
    let k = (u * 3.99) as u32;
    match arch {
        // tower glass, warehouse metal, informal sheets, white dome, temple tiles
        // (round 13: towers in six tints and tones, dark blue and green glass to pale concrete
        // and bronze: one blue-grey ramp drew a uniform grey skyline, gallery g30)
        20 => {
            let k6 = (u * 5.99) as u32;
            let f = u * 6.0 - k6 as f64;
            let c = match k6 {
                0 => srgb(52.0, 66.0, 86.0),
                1 => srgb(64.0, 88.0, 92.0),
                2 => srgb(96.0, 104.0, 114.0),
                3 => srgb(108.0, 94.0, 78.0),
                4 => srgb(150.0, 152.0, 150.0),
                _ => srgb(184.0, 180.0, 170.0),
            };
            return c * (0.92 + 0.16 * f);
        }
        21 => {
            // (matte metal and membranes: light greys, beige, a few white, weathered and blue-grey
            // ones; round 10's blue-greys over whole estates read as glass or water, town@type)
            return if u < 0.3 {
                srgb(168.0, 168.0, 164.0)
            } else if u < 0.5 {
                srgb(180.0, 172.0, 156.0)
            } else if u < 0.62 {
                srgb(200.0, 198.0, 192.0)
            } else if u < 0.8 {
                srgb(142.0, 142.0, 140.0)
            } else if u < 0.93 {
                srgb(150.0, 142.0, 128.0)
            } else {
                srgb(124.0, 130.0, 136.0)
            };
        }
        22 => {
            return match k {
                0 => srgb(140.0, 140.0, 138.0),
                1 => srgb(132.0, 92.0, 62.0),
                2 => srgb(70.0, 96.0, 150.0),
                _ => srgb(96.0, 84.0, 72.0),
            }
        }
        23 => return srgb(222.0, 218.0, 206.0),
        24 => return srgb(160.0, 70.0, 50.0),
        _ => {}
    }
    if flat {
        return match k {
            0 => srgb(128.0, 126.0, 122.0),
            1 => srgb(104.0, 102.0, 100.0),
            2 => srgb(158.0, 154.0, 146.0),
            _ => srgb(86.0, 86.0, 90.0),
        };
    }
    match arch {
        // terracotta
        5 | 11 => match k {
            0 => srgb(172.0, 94.0, 64.0),
            1 => srgb(188.0, 112.0, 76.0),
            2 => srgb(150.0, 82.0, 60.0),
            _ => srgb(200.0, 140.0, 104.0),
        },
        // whitewashed / adobe flat-ish
        3 => match k {
            0 => srgb(214.0, 206.0, 190.0),
            1 => srgb(196.0, 176.0, 146.0),
            2 => srgb(180.0, 150.0, 116.0),
            _ => srgb(226.0, 220.0, 208.0),
        },
        // slate, red tile, brown
        6 | 4 => match k {
            0 => srgb(88.0, 88.0, 94.0),
            1 => srgb(146.0, 84.0, 66.0),
            2 => srgb(118.0, 86.0, 72.0),
            _ => srgb(70.0, 72.0, 80.0),
        },
        // metal roofs: green, blue, grey, red
        9 | 10 => match k {
            0 => srgb(84.0, 112.0, 92.0),
            1 => srgb(82.0, 96.0, 122.0),
            2 => srgb(128.0, 128.0, 130.0),
            _ => srgb(150.0, 62.0, 52.0),
        },
        // asphalt shingles
        7 => match k {
            0 => srgb(112.0, 106.0, 100.0),
            1 => srgb(92.0, 82.0, 74.0),
            2 => srgb(134.0, 124.0, 112.0),
            _ => srgb(76.0, 76.0, 80.0),
        },
        // thatch and metal
        2 => match k {
            0 => srgb(150.0, 128.0, 92.0),
            1 => srgb(132.0, 112.0, 80.0),
            2 => srgb(142.0, 142.0, 140.0),
            _ => srgb(120.0, 100.0, 76.0),
        },
        // corrugated metal and tile (tropical, monsoon)
        _ => match k {
            0 => srgb(152.0, 152.0, 150.0),
            1 => srgb(132.0, 86.0, 62.0),
            2 => srgb(76.0, 104.0, 140.0),
            _ => srgb(166.0, 92.0, 72.0),
        },
    }
}

/// The hashed BSP of a block: the lot containing (u, v) — its hash and rectangle.
pub fn bsp(h0: u64, u: f64, v: f64, bw: f64, bd: f64, depth: u32, min_side: f64) -> (u64, [f64; 4]) {
    let (mut x0, mut y0, mut x1, mut y1) = (0.0, 0.0, bw, bd);
    let mut h = h0;
    for d in 0..4u32 {
        if d >= depth {
            break;
        }
        let (w, hgt) = (x1 - x0, y1 - y0);
        if w.max(hgt) < 2.0 * min_side {
            break;
        }
        let t = 0.35 + 0.3 * u01k(h, 1);
        if w >= hgt {
            let sx = x0 + t * w;
            if u >= sx {
                x0 = sx;
                h = mix64(h ^ 0xB5);
            } else {
                x1 = sx;
                h = mix64(h ^ 0xA3);
            }
        } else {
            let sy = y0 + t * hgt;
            if v >= sy {
                y0 = sy;
                h = mix64(h ^ 0xC7);
            } else {
                y1 = sy;
                h = mix64(h ^ 0xD9);
            }
        }
    }
    (h, [x0, y0, x1, y1])
}

/// Coverage of an axis-aligned box (centre, half sizes) at `r`, antialiased over `fw`; and the
/// distances to its edges.
#[inline]
fn boxd(r: DVec2, c: DVec2, half: DVec2) -> DVec2 {
    half - (r - c).abs()
}

#[inline]
fn aa(d: f64, fw: f64) -> f64 {
    (d / fw + 0.5).clamp(0.0, 1.0)
}

/// A sample's district: its pattern and parameters (towns and cities: a W2 cell in the place
/// frame; villages: their form), its zone, the distance to its edge (arterials), and its
/// density multiplier (some districts dense, some half fields).
#[derive(Clone, Copy, Debug)]
pub struct Dist {
    pub did: u64,
    pub zone: u32,
    pub pattern: u32,
    pub theta: f64,
    pub art: f64,
    pub dmul: f64,
    pub b: f64,
    pub asp: f64,
    pub sw: f64,
    pub jit: f64,
    pub warp: f64,
    pub w2cell: f64,
    /// industrial estates on the town's grid: blocks of `mg` grid blocks (0: an estate's own grid),
    /// and the distance to the estate's edge (its green buffer; 1e9: none)
    pub mg: i64,
    pub buf: f64,
    /// beside an estate on the town's grid: the distance to its edge (1e9: none)
    pub nbind: f64,
}

/// The pattern of the district of site `point` (place frame) and id `id`, the zone at its site
/// and whether it is industrial (wedges around the downwind direction). The pattern is drawn
/// per town (`pick_pattern`), not per district or zone.
pub fn district_pattern(pl: &Pl, point: DVec2, id: u64) -> (u32, u32, bool) {
    let rd = rel0(pl, point);
    let zd = zone_of(pl, rd);
    // industrial wedges: angular sectors around the downwind direction
    let ang = point.y.atan2(point.x);
    let mut ind = false;
    let nw = if pl.level >= 3 {
        2
    } else if pl.level == 2 {
        1
    } else {
        0
    };
    for k in 0..2u32 {
        if k < nw {
            let a0 = if k == 0 { pl.wind } else { pl.wind + std::f64::consts::PI * 1.4 * (u01k(pl.seed, 41) - 0.5) };
            let hw = 0.22 + 0.2 * u01k(pl.seed, 42 + k as u64);
            if wrap_pi(ang - a0).abs() < hw && rd > 0.25 && rd < 0.95 && u01k(id, 3) < 0.8 {
                ind = true;
            }
        }
    }
    let mut pattern = pick_pattern(pl, zd, mix64(pl.seed ^ 0x2A7E), ind);
    if pl.level < 2 {
        pattern = pat::GRID;
    }
    (pattern, zd, ind)
}

/// The distance from `q` to the edge of an industrial estate on the town's grid where `q` lies
/// beside one (1e9: none): `Dist::nbind` without the rest of the district (round 12: the whole
/// district evaluated again at each block's centre cost ~2 s of pass B's compile time).
pub fn nbind_at(pl: &Pl, q: DVec2) -> f64 {
    if pl.form != form::CITY as u32 || pl.level < 2 || !shared_industry(pl) {
        return 1e9;
    }
    let dcell = [400.0, 400.0, 600.0, 1000.0, 1400.0][pl.level as usize];
    let wl = 0.3 * dcell;
    let qd = q
        + DVec2::new((q.y / wl + 6.0 * u01k(pl.seed, 51)).sin(), (q.x / (1.17 * wl) + 6.0 * u01k(pl.seed, 52)).sin()) * (0.16 * dcell)
        + DVec2::new((q.y / (0.4 * wl) + 6.0 * u01k(pl.seed, 54)).sin(), (q.x / (0.43 * wl) + 6.0 * u01k(pl.seed, 55)).sin()) * (0.045 * dcell);
    let dc = worley2(pl.seed ^ 0xD15, qd, dcell, 0.85);
    if district_pattern(pl, dc.point, dc.id).0 == pat::GRID && district_pattern(pl, dc.point2, dc.id2).0 == pat::INDUSTRIAL {
        worley2_edge_dist(&dc, qd)
    } else {
        1e9
    }
}

/// The district at `q` (relative distance `rel_here` at the sample).
pub fn district(pl: &Pl, q: DVec2, rel_here: f64) -> Dist {
    let fm = pl.form;
    let pattern;
    let mut z = zone_of(pl, rel_here);
    let mut theta = 0.0;
    let mut did = pl.seed;
    let mut art = 1e9;
    let mut b = 110.0;
    let mut asp = 1.0;
    let mut sw = 4.0;
    let mut jit = 0.3;
    let mut warp = 0.0;
    let mut w2cell = 0.0;
    let mut mg = 0i64;
    let mut buf = 1e9;
    let mut nbind = 1e9;
    if fm == form::CITY as u32 {
        let dcell = [400.0, 400.0, 600.0, 1000.0, 1400.0][pl.level as usize];
        // (the district borders wander: the cells are looked up at a warped position, so the
        // districts are not hexagons; and only some of the edges are arterials, the rest are
        // ordinary streets)
        let wl = 0.3 * dcell;
        let qd = q
            + DVec2::new((q.y / wl + 6.0 * u01k(pl.seed, 51)).sin(), (q.x / (1.17 * wl) + 6.0 * u01k(pl.seed, 52)).sin()) * (0.16 * dcell)
            + DVec2::new((q.y / (0.4 * wl) + 6.0 * u01k(pl.seed, 54)).sin(), (q.x / (0.43 * wl) + 6.0 * u01k(pl.seed, 55)).sin()) * (0.045 * dcell);
        let dc = worley2(pl.seed ^ 0xD15, qd, dcell, 0.85);
        did = dc.id;
        let (pt, zd, ind) = district_pattern(pl, dc.point, dc.id);
        pattern = pt;
        // arterials only between districts of different layouts (a town's districts of one
        // pattern share one street layout: no seams through its blocks; estates on the town's
        // grid share it); a green buffer along the edge of an estate
        let shared = shared_industry(pl);
        let net = |p: u32| if shared && p == pat::INDUSTRIAL { pat::GRID } else { p };
        if pl.level >= 2 {
            let (pt2, _, _) = district_pattern(pl, dc.point2, dc.id2);
            if net(pt2) != net(pt) || (pt == pat::INDUSTRIAL && !shared) {
                art = worley2_edge_dist(&dc, qd);
            }
            if pt == pat::INDUSTRIAL && pt2 != pat::INDUSTRIAL {
                buf = worley2_edge_dist(&dc, qd);
            }
            if shared && pt == pat::GRID && pt2 == pat::INDUSTRIAL {
                nbind = worley2_edge_dist(&dc, qd);
            }
        }
        z = if ind { zone::OUTER } else { zd.max(z.min(zone::INNER)).min(zone::PERI) };
        // one layout per pattern and town (estates of industry: per district, or the town's grid)
        if pt != pat::INDUSTRIAL || shared {
            did = mix64(pl.seed ^ 0x1A70 ^ net(pt) as u64);
        }
        if pt == pat::INDUSTRIAL && shared {
            mg = 2 + (u01k(dc.id, 14) < 0.5) as i64;
        }
        let survey = pl.arch == 7;
        // (one street orientation per town: districts turned against each other read as pieces
        // of different towns)
        theta = if survey { 0.0 } else { (u01k(pl.seed ^ 0x7E7A, 5) - 0.5) * 0.3 };
        let bf = [1.0, 1.0, 0.9, 0.85, 1.0, 0.9, 1.05, 1.25, 0.8, 1.0, 0.9, 0.9][pl.arch as usize % 12];
        let u2 = u01k(did, 6);
        match pattern {
            pat::GRID => {
                b = (70.0 + 60.0 * u2) * bf;
                asp = if survey { 2.2 } else { 1.0 + 0.6 * u01k(did, 7) };
                sw = if z == zone::CBD { 7.0 } else { 4.5 + 2.5 * u01k(did, 8) };
            }
            pat::RADIAL => {
                b = 80.0 + 40.0 * u2;
                sw = 5.0;
            }
            pat::CURVI => {
                // (crescents and loops: long blocks, the streets bent by a warp of 190 m)
                b = (80.0 + 40.0 * u2) * bf;
                asp = 1.8 + 0.6 * u01k(did, 7);
                sw = 4.5;
                jit = 0.15;
                warp = 34.0;
            }
            pat::ORGANIC => {
                w2cell = if pl.arch == 3 { 40.0 } else { 55.0 + 35.0 * u2 };
                sw = if pl.arch == 3 { 1.7 } else { 2.8 + 1.4 * u01k(did, 8) };
            }
            pat::SUPER => {
                b = 280.0 + 110.0 * u2;
                sw = 9.0;
                jit = 0.1;
            }
            pat::INFORMAL => {
                b = 60.0 + 20.0 * u2;
                sw = 2.3;
                jit = 0.45;
                warp = 9.0;
                w2cell = 7.5;
            }
            _ if mg > 0 => {
                // (the town's grid: its block size and proportions, wider streets)
                b = (70.0 + 60.0 * u2) * bf;
                asp = if survey { 2.2 } else { 1.0 + 0.6 * u01k(did, 7) };
                sw = 1.3 * (4.5 + 2.5 * u01k(did, 8));
            }
            _ => {
                b = 230.0 + 110.0 * u2;
                asp = 0.7 + 0.4 * u01k(did, 7);
                sw = 8.0;
                jit = 0.1;
            }
        }
    } else {
        z = if rel_here < 0.35 { zone::INNER } else { zone::SUBURB };
        match fm {
            f if f == form::NUCLEATED as u32 => {
                pattern = pat::ORGANIC;
                w2cell = 52.0;
                sw = 2.6;
            }
            f if f == form::PERCHED as u32 => {
                pattern = pat::ORGANIC;
                w2cell = 34.0;
                sw = 1.8;
            }
            f if f == form::LINEAR as u32 || f == form::WALDHUFEN as u32 => {
                pattern = pat::LINEAR;
                b = 22.0;
                sw = 3.5;
                jit = 0.45;
            }
            f if f == form::DISPERSED as u32 => {
                pattern = pat::DISPERSED;
                w2cell = 170.0;
            }
            _ => {
                pattern = pat::KRAAL;
                w2cell = if pl.level == 0 { 1e5 } else { 120.0 };
            }
        }
    }
    let dmul = if fm == form::CITY as u32 && pl.level >= 2 { (0.8 + 0.4 * u01k(did, 13)).min(1.0) } else { 1.0 };
    Dist { did, zone: z, pattern, theta, art, dmul, b, asp, sw, jit, warp, w2cell, mg, buf, nbind }
}

/// The large-scale bend and stretch of a city's grid (WGSL `st_big_grid_warp`): ±105 m of bend at
/// 2.7-7.9 km wavelengths, block sizes ±12 % over 10 km.
fn big_grid_warp(qr: DVec2, seed: u64) -> DVec2 {
    let ph = |k: u64| std::f64::consts::TAU * u01k(seed ^ 0xB16, k);
    let b = DVec2::new((qr.y / 1100.0 + ph(1)).sin() + 0.5 * (qr.y / 430.0 + ph(2)).sin(), (qr.x / 1250.0 + ph(3)).sin() + 0.5 * (qr.x / 470.0 + ph(4)).sin())
        * 70.0;
    let st = DVec2::new((qr.x / 1600.0 + ph(5)).sin(), (qr.y / 1700.0 + ph(6)).sin()) * 190.0;
    qr + b + st
}

/// The evaluation of a place at position `q` (place frame; district `d` of the sample): block,
/// lot, building. Shadow steps use only its height.
pub fn place_eval(pl: &Pl, q: DVec2, d: &Dist, e: &Env) -> Ev {
    let mut o = Ev { lamp_sp: 0.0, ds: 1e9, ..Default::default() };
    let fw = e.fw;
    let rel_here = rel0(pl, q) / (1.0 + e.n);
    let dens_here = density(pl, rel_here) * d.dmul;
    let fm = pl.form;
    let Dist { did, zone: z, pattern, theta, art, dmul, b, asp, sw, jit, warp, w2cell, mg, buf, nbind } = *d;
    // (the old core of a city of an old culture: its grid bent, its streets narrower, lanes in
    // its blocks; fading into the grid over a band)
    let oldk = if pattern == pat::GRID { old_core(pl, q) } else { 0.0 };
    let oldg = pattern == pat::GRID && old_near(pl, q);
    let sw = sw * (1.0 - 0.4 * oldk);
    o.zone = z;
    o.pattern = pattern;
    // ---- the block
    let mut blk = Blk { id: did, ds: 1e9, sw, ..Default::default() };
    let gridlike = matches!(pattern, pat::GRID | pat::CURVI | pat::SUPER | pat::INFORMAL | pat::INDUSTRIAL);
    if gridlike {
        let mut qr = rot2(q, theta);
        if warp > 0.0 {
            let l = 230.0 * warp / 42.0;
            qr += DVec2::new((qr.y / l + 6.0 * u01k(did, 9)).sin(), (qr.x / (1.13 * l) + 6.0 * u01k(did, 10)).sin()) * warp;
        }
        if (pattern == pat::GRID || mg > 0) && pl.level >= 3 {
            // (a city's grid bends gently and its blocks grow and shrink over kilometres: one straight
            // grid of one block size to the horizon read as a pattern, metro)
            qr = big_grid_warp(qr, pl.seed);
        }
        if oldk > 0.0 {
            let s = did ^ 0x01D;
            qr += DVec2::new(
                (qr.y / 170.0 + 6.0 * u01k(s, 1)).sin() + 0.4 * (qr.y / 61.0 + 6.0 * u01k(s, 2)).sin(),
                (qr.x / 190.0 + 6.0 * u01k(s, 3)).sin() + 0.4 * (qr.x / 67.0 + 6.0 * u01k(s, 4)).sin(),
            ) * (16.0 * oldk);
        }
        let bx = b;
        let by = b * asp;
        let sd = did ^ 0x5EE7;
        let line = |axis: i64, i: i64, bb: f64| (i as f64 + jit * (u01(hash2(sd ^ axis as u64, axis, i)) - 0.5)) * bb;
        // (an estate on the town's grid: blocks of `mg` of its blocks)
        let m = mg.max(1);
        let cell = |axis: i64, x: f64, bb: f64| -> (i64, f64, f64) {
            let mut i = (x / bb).floor() as i64;
            if x < line(axis, i, bb) {
                i -= 1;
            } else if x >= line(axis, i + 1, bb) {
                i += 1;
            }
            let i = i.div_euclid(m) * m;
            let (a0, a1) = (line(axis, i, bb), line(axis, i + m, bb));
            (i, x - a0, a1 - a0)
        };
        let (bix, bqx, bsx) = cell(0, qr.x, bx);
        let (biy, bqy, bsy) = cell(1, qr.y, by);
        let near_x = bix + if bqx > bsx - bqx { m } else { 0 };
        let near_y = biy + if bqy > bsy - bqy { m } else { 0 };
        let dx = bqx.min(bsx - bqx);
        let dy = bqy.min(bsy - bqy);
        let main_x = near_x.rem_euclid(4) == 0 && pattern == pat::GRID;
        let main_y = near_y.rem_euclid(4) == 0 && pattern == pat::GRID;
        let (swx, swy) = (if main_x { sw * 1.5 } else { sw }, if main_y { sw * 1.5 } else { sw });
        // (the street's segment between two crossings: streets end at crossings, all or nothing;
        // decided per block they ended mid-block in empty stubs, or one half of their width, fp_town)
        let mid = if dx - swx < dy - swy {
            blk.ds = dx;
            blk.sw = swx;
            blk.along = qr.y;
            blk.sid = hash2(sd ^ 0x1A, 0, near_x);
            blk.major = main_x;
            blk.seg = hash2(blk.sid, 3, biy);
            DVec2::new(if near_x > bix { bsx - bqx } else { -bqx }, 0.5 * bsy - bqy)
        } else {
            blk.ds = dy;
            blk.sw = swy;
            blk.along = qr.x;
            blk.sid = hash2(sd ^ 0x1A, 1, near_y);
            blk.major = main_y;
            blk.seg = hash2(blk.sid, 4, bix);
            DVec2::new(0.5 * bsx - bqx, if near_y > biy { bsy - bqy } else { -bqy })
        };
        blk.smid = rot2(mid, -theta);
        blk.id = hash3(sd, bix, biy, 7);
        let mjl = |i: i64| (pattern == pat::GRID && i.rem_euclid(4) == 0) as u32;
        blk.mj = mjl(bix) | mjl(bix + m) << 1 | mjl(biy) << 2 | mjl(biy + m) << 3;
        blk.u = bqx - sw;
        blk.v = bqy - sw;
        blk.bw = bsx - 2.0 * sw;
        blk.bd = bsy - 2.0 * sw;
    } else if pattern == pat::RADIAL {
        let r = q.length();
        let phi = q.y.atan2(q.x) + std::f64::consts::PI;
        let dr = b;
        let i = (r / dr).floor();
        let v = r - i * dr;
        let nsp = (std::f64::consts::TAU * (i + 0.5) * dr / (1.6 * dr)).max(4.0).log2().ceil().exp2();
        let dphi = std::f64::consts::TAU / nsp;
        let j = (phi / dphi).floor();
        let arc = (phi - j * dphi) * r;
        let bwr = dphi * r;
        let (d_ring, d_rad) = (v.min(dr - v), arc.min(bwr - arc));
        if d_ring < d_rad {
            blk.ds = d_ring;
            blk.along = phi * r;
            blk.sid = hash2(pl.seed ^ 0x2A, i as i64 + (v > 0.5 * dr) as i64, 0);
            blk.major = (i as i64).rem_euclid(3) == 0;
        } else {
            blk.ds = d_rad;
            blk.along = r;
            blk.sid = hash2(pl.seed ^ 0x2B, i as i64, j as i64);
            blk.major = false;
        }
        blk.id = hash2(pl.seed ^ 0x2C, i as i64, j as i64);
        blk.u = arc - sw;
        blk.v = v - sw;
        blk.bw = bwr - 2.0 * sw;
        blk.bd = dr - 2.0 * sw;
    } else if pattern == pat::LINEAR {
        // the street along the place axis; farm lots along it on both sides
        let sd = pl.seed ^ 0x11E;
        let line = |i: i64| (i as f64 + jit * (u01(hash2(sd, 0, i)) - 0.5)) * b;
        let mut i = (q.x / b).floor() as i64;
        if q.x < line(i) {
            i -= 1;
        } else if q.x >= line(i + 1) {
            i += 1;
        }
        let side = if q.y < 0.0 { -1i64 } else { 1 };
        blk.ds = q.y.abs();
        blk.along = q.x;
        blk.sid = hash1(sd, 1);
        blk.major = true;
        blk.id = hash3(sd, i, side, 3);
        blk.u = q.x - line(i);
        blk.bw = line(i + 1) - line(i);
        blk.v = q.y.abs() - sw;
        blk.bd = 40.0 + 18.0 * u01k(blk.id, 2);
    }
    // ---- the cellular layouts (one W2 for all of them)
    let mut wc = Cell2 { id: 0, point: DVec2::ZERO, f1: 0.0, f2: 0.0, id2: 0, point2: DVec2::ZERO };
    let mut qw = q;
    if w2cell > 0.0 {
        if pattern == pat::ORGANIC {
            qw = q + DVec2::new((q.y / (1.7 * w2cell) + 6.0 * u01k(did, 11)).sin(), (q.x / (1.9 * w2cell) + 6.0 * u01k(did, 12)).sin()) * (0.22 * w2cell);
        }
        wc = worley2(did ^ 0x0A6, qw, w2cell, 0.9);
        if pattern == pat::ORGANIC {
            let ed = worley2_edge_dist(&wc, qw);
            let dv = (wc.point2 - wc.point).normalize_or_zero();
            blk.ds = ed;
            blk.along = qw.dot(DVec2::new(-dv.y, dv.x));
            blk.sid = wc.id ^ wc.id2;
            blk.id = wc.id;
            let rel = rot2(qw - wc.point, u01k(wc.id, 1) * std::f64::consts::FRAC_PI_2);
            let half = 0.45 * w2cell;
            blk.bw = 2.0 * (half - sw);
            blk.bd = blk.bw;
            blk.u = (rel.x + half - sw).clamp(0.0, blk.bw);
            blk.v = (rel.y + half - sw).clamp(0.0, blk.bd);
        }
    }
    o.ds = blk.ds;
    o.sw = blk.sw;
    o.along = blk.along;
    o.sid = blk.sid;
    o.major = blk.major;
    // ---- streets and arterials
    // (streets end where the town thins out: they ran on far into the fields, a grid of empty
    // streets around every town; arterials narrower: they drew a dark outline around every
    // district, the districts read as pieces of different towns)
    // (the threshold varies by block: a town ended in a staircase along its street grid)
    let edge_thr = 0.42 + 0.1 * (u01k(blk.id, 31) - 0.5);
    // (main streets run on beyond the town's edge with houses along them: ribbons)
    // (grids: by the street's segment, at its middle)
    let street_ok = if blk.seg != 0 && fm == form::CITY as u32 {
        let ds = density(pl, rel0(pl, q + blk.smid) / (1.0 + e.n)) * dmul;
        ds > 0.42 + 0.1 * (u01k(blk.seg, 31) - 0.5) - if blk.major { 0.04 } else { 0.0 }
    } else {
        dens_here > edge_thr - if blk.major { 0.18 } else { 0.0 }
    };
    let art_hw = [3.5, 3.5, 5.0, 6.0, 7.5][pl.level as usize];
    let has_streets = !matches!(pattern, pat::DISPERSED | pat::KRAAL);
    if art < art_hw + 2.0 * fw && dens_here > 0.35 {
        let c = band_cov(art, art_hw, fw);
        if c > 0.0 {
            o.ds = art;
            o.sw = art_hw;
            o.major = true;
            o.sid = did ^ 0xA87;
            o.street = c;
        }
    }
    if has_streets && street_ok {
        o.street = o.street.max(band_cov(blk.ds, blk.sw, fw));
    }
    // (the ring boulevard around a big city's old core)
    let rd = ring_dist(pl, q);
    let ring_hw = art_hw + 2.0;
    if rd < ring_hw + 2.0 * fw && dens_here > 0.35 {
        let c = band_cov(rd, ring_hw, fw);
        if c > o.street {
            o.ds = rd;
            o.sw = ring_hw;
            o.major = true;
            o.sid = pl.seed ^ 0x219;
            o.street = c;
        }
    }
    o.lamp_sp = if has_streets {
        if pattern == pat::LINEAR || fm != form::CITY as u32 {
            34.0
        } else {
            18.0 + 6.0 * (z as f64).min(3.0)
        }
    } else {
        0.0
    };
    let walk =
        if has_streets && street_ok && pattern != pat::INFORMAL && pattern != pat::LINEAR { band_cov(blk.ds, blk.sw + 0.2 * blk.sw + 0.6, fw) } else { 0.0 };
    // ---- block content
    let mut cov_lot: f64 = 0.0;
    let mut col = c_asphalt();
    let mut h = 0.0;
    let mut cls = crate::landcover::URBAN;
    // the building's lot (the instance; whole blocks where there is no finer lot)
    let mut lot_id = blk.id;
    let inside = blk.u >= 0.0 && blk.v >= 0.0 && blk.u < blk.bw && (blk.v < blk.bd || pattern == pat::LINEAR);
    let ublk = DVec2::new(blk.u, blk.v);
    // the lot density at the block centre / lot centre (not per pixel: lots stay whole)
    let rel_at = |dq: DVec2| rel0(pl, q + dq) / (1.0 + e.n);
    let dens_at = |dq: DVec2| density(pl, rel_at(dq)) * dmul;
    let mut roof = 0.0;
    let mut rim = 0.0;
    let mut win = DVec3::new(1.0, 0.72, 0.42);
    let mut tree = 0.0;
    let mut tree_h = 0.0;
    let mut em = DVec3::ZERO;
    // (yards and gardens: the land around, a little trodden; not a uniform lot colour)
    let yard = mixc(e.ground * (0.92 + 0.16 * u01k(blk.id, 19)), c_yard(e.moist, u01k(blk.id, 19)), 0.35);
    // the roof drawn at the sample (one colour evaluation: culture or code, hash, flat, tint),
    // its coverage and shading; the mean of a lot where its buildings are below the pixel
    // (ground colour, roof fraction, height)
    let mut rf = (pl.arch, 0.5, false, 1.0);
    let mut rf_ins = 0.0;
    let mut rf_shade = 1.0;
    let mut lot_mean: Option<(DVec3, f64, f64)> = None;
    let lot_scale = match pattern {
        pat::INDUSTRIAL => 40.0,
        pat::SUPER => 50.0,
        pat::INFORMAL | pat::KRAAL => 6.0,
        _ => 13.0,
    };
    // special parcels: whole blocks
    let mut special = parcel::NONE;
    if inside && fm == form::CITY as u32 && pattern != pat::INFORMAL {
        let us = u01k(blk.id, 20);
        let zi = z as usize;
        let p_park = [0.05, 0.07, 0.07, 0.05, 0.02][zi];
        let p_plaza = [0.06, 0.03, 0.0, 0.0, 0.0][zi];
        let p_park2 = p_park + p_plaza;
        let p_parking = p_park2 + [0.03, 0.02, 0.01, 0.0, 0.0][zi];
        let p_sports = p_parking + [0.0, 0.01, 0.03, 0.04, 0.02][zi];
        let p_cem = p_sports + [0.0, 0.01, 0.02, 0.02, 0.02][zi];
        let p_stad = p_cem + if pl.level >= 3 && blk.bw.min(blk.bd) > 130.0 && (z == zone::OUTER || z == zone::SUBURB) { 0.05 } else { 0.0 };
        let p_site = p_stad + 0.01;
        special = if pattern == pat::INDUSTRIAL {
            if us < 0.03 {
                parcel::PARKING
            } else {
                parcel::NONE
            }
        } else if us < p_park {
            parcel::PARK
        } else if us < p_park2 {
            parcel::PLAZA
        } else if us < p_parking {
            parcel::PARKING
        } else if us < p_sports {
            parcel::SPORTS
        } else if us < p_cem {
            parcel::CEMETERY
        } else if us < p_stad {
            parcel::STADIUM
        } else if us < p_site {
            parcel::SITE
        } else {
            parcel::NONE
        };
        // the block exists by the density at its centre
        let bc = DVec2::new(0.5 * blk.bw - blk.u, 0.5 * blk.bd - blk.v);
        let dbc = dens_at(rot2(bc, -theta));
        if dbc < 0.3 {
            special = parcel::NONE;
        }
    }
    if inside && special != parcel::NONE {
        cov_lot = 1.0;
        let (u, v) = (blk.u, blk.v);
        let c = DVec2::new(0.5 * blk.bw, 0.5 * blk.bd);
        let r = ublk - c;
        match special {
            parcel::PARK => {
                col = srgb(80.0, 116.0, 52.0) * (1.0 + 0.15 * e.detail);
                cls = crate::landcover::PARK;
                // diagonal paths, tree clumps, a pond
                let dpath = (r.x * blk.bd - r.y * blk.bw).abs() / blk.bw.hypot(blk.bd);
                let path = band_cov(dpath, 1.6, fw);
                let tc = (ublk / 13.0).floor();
                let th = hash2(blk.id ^ 0x7EE, tc.x as i64, tc.y as i64);
                let tp = (tc + 0.5 + 0.5 * (DVec2::new(u01k(th, 1), u01k(th, 2)) - 0.5)) * 13.0;
                let tr = 3.5 + 2.5 * u01k(th, 3);
                if u01k(th, 4) < 0.55 && path < 0.5 {
                    tree = aa(tr - (ublk - tp).length(), fw);
                    tree_h = 9.0 + 8.0 * u01k(th, 5);
                }
                if u01k(blk.id, 21) < 0.3 {
                    let pr = 0.22 * blk.bw.min(blk.bd);
                    let pond = aa(pr - (r - DVec2::new(0.1, -0.15) * blk.bw).length(), fw);
                    col = mixc(col, srgb(52.0, 78.0, 84.0), pond);
                    tree *= 1.0 - pond;
                }
                col = mixc(col, srgb(170.0, 156.0, 130.0), path);
            }
            parcel::PLAZA => {
                col = c_paving() * (0.9 + 0.1 * u01k(blk.id, 22));
                let grid = band_cov((u / 6.0 - (u / 6.0).round()) * 6.0, 0.15, fw) + band_cov((v / 6.0 - (v / 6.0).round()) * 6.0, 0.15, fw);
                col *= 1.0 - 0.08 * grid.min(1.0) * band(6.0, e.gsd);
                cls = crate::landcover::PARKING;
                em += DVec3::new(1.0, 0.85, 0.65) * 0.02 * band(20.0, e.gsd);
                // (rows of trees around the square: a bare paved block read as an empty hole)
                let tq = (DVec2::new(u, v) / 10.0).round() * 10.0;
                let de = tq.x.min(tq.y).min(blk.bw - tq.x).min(blk.bd - tq.y);
                if (4.0..=11.0).contains(&de) {
                    tree = aa(2.4 - (DVec2::new(u, v) - tq).length(), fw);
                    tree_h = 8.0 + 3.0 * u01k(hash2(blk.id ^ 0x9A, tq.x as i64, tq.y as i64), 1);
                }
            }
            parcel::PARKING => {
                // stall lines in rows, the transport kit's parked cars
                col = car_park(blk.id, u, v, fw, e.gsd);
                cls = crate::landcover::PARKING;
                em += DVec3::new(1.0, 0.6, 0.25) * 0.03 * band(30.0, e.gsd);
                // (trees between the rows of stalls: a bare asphalt block read as an empty hole)
                let tq = DVec2::new((u / 9.0).round() * 9.0, (v / 16.0).round() * 16.0);
                if u01k(hash2(blk.id ^ 0x9B, tq.x as i64, tq.y as i64), 1) < 0.7 {
                    tree = aa(2.6 - (DVec2::new(u, v) - tq).length(), fw);
                    tree_h = 7.0 + 3.0 * u01k(hash2(blk.id ^ 0x9B, tq.x as i64, tq.y as i64), 2);
                }
            }
            parcel::SPORTS => {
                // a pitch and an athletics track around it (school / club)
                let ph = DVec2::new(34.0, 52.0).min(DVec2::new(0.5 * blk.bw - 12.0, 0.5 * blk.bd - 12.0));
                let rr = if blk.bw > blk.bd { DVec2::new(r.y, r.x) } else { r };
                let dp = boxd(rr, DVec2::ZERO, ph);
                let pitch = aa(dp.min_element(), fw);
                let core = DVec2::new(0.0, (rr.y.abs() - (ph.y - ph.x)).max(0.0));
                let dtr = DVec2::new(rr.x, core.y).length() - ph.x - 3.0;
                let track = band_cov(dtr - 4.5, 4.5, fw);
                let stripe = ((rr.y / 6.0).floor() as i64).rem_euclid(2) as f64;
                col = mixc(srgb(118.0, 140.0, 84.0), srgb(84.0, 128.0, 58.0) * (0.95 + 0.07 * stripe * band(6.0, e.gsd)), pitch);
                col = mixc(col, srgb(170.0, 82.0, 62.0), track);
                cls = crate::landcover::SPORTS_FIELD;
            }
            parcel::CEMETERY => {
                col = srgb(100.0, 120.0, 76.0);
                cls = crate::landcover::CEMETERY;
                let gu = u / 2.4 - (u / 2.4).floor();
                let gv = v / 3.2 - (v / 3.2).floor();
                let stone = if (gu - 0.5).abs() < 0.22 && (gv - 0.5).abs() < 0.3 { 1.0 } else { 0.0 };
                let path = band_cov(r.x, 1.5, fw).max(band_cov(r.y, 1.5, fw));
                col = mixc(col, srgb(176.0, 172.0, 164.0), stone * band(2.4, e.gsd) * (1.0 - path) + 0.25 * (1.0 - band(2.4, e.gsd)));
                col = mixc(col, srgb(160.0, 150.0, 130.0), path);
                let tc = (ublk / 18.0).floor();
                let th = hash2(blk.id ^ 0xCE7, tc.x as i64, tc.y as i64);
                if u01k(th, 1) < 0.4 {
                    let tp = (tc + 0.5) * 18.0;
                    tree = aa(3.0 - (ublk - tp).length(), fw);
                    tree_h = 8.0;
                }
            }
            parcel::STADIUM => {
                let a = 0.45 * blk.bw.min(blk.bd).min(220.0);
                let ab = DVec2::new(a, 0.78 * a);
                let rr = if blk.bw > blk.bd { r } else { DVec2::new(r.y, r.x) };
                let rn = (rr / ab).length();
                col = c_concrete() * 0.95;
                cls = crate::landcover::SPORTS_FIELD;
                if rn < 1.0 {
                    if rn < 0.62 {
                        let stripe = ((rr.x / 5.0).floor() as i64).rem_euclid(2) as f64;
                        col = srgb(70.0, 124.0, 52.0) * (0.94 + 0.08 * stripe * band(5.0, e.gsd));
                        let pl2 = boxd(rr, DVec2::ZERO, DVec2::new(0.5 * a, 0.36 * a)).min_element();
                        col = mixc(col, srgb(220.0, 220.0, 220.0), band_cov(pl2, 0.12, fw) * band(1.0, e.gsd));
                    } else {
                        // stands, rising outwards
                        h = 8.0 + 22.0 * smoothstep(0.62, 0.98, rn);
                        col = mixc(
                            srgb(190.0, 190.0, 196.0),
                            srgb(150.0, 40.0, 40.0),
                            0.4 * (((rr.y.atan2(rr.x) * 12.0).floor() as i64).rem_euclid(2) as f64) * band(30.0, e.gsd),
                        );
                        roof = 1.0;
                        cls = crate::landcover::BUILDING;
                    }
                }
                // four floodlight masts
                let fl = rr.abs() - ab * 0.9;
                em += DVec3::new(0.95, 0.97, 1.0) * point_light(fl.length_squared(), 30.0, 1.2, fw);
            }
            _ => {
                // construction site
                col = srgb(150.0, 126.0, 98.0) * (0.9 + 0.2 * e.detail);
                cls = crate::landcover::BARE;
            }
        }
    } else if inside {
        match pattern {
            pat::SUPER => {
                // slabs in rows along the longer side, point towers between, green and parking
                let long_x = blk.bw >= blk.bd;
                let (a, c, _, lc) = if long_x { (blk.u, blk.v, blk.bw, blk.bd) } else { (blk.v, blk.u, blk.bd, blk.bw) };
                let pitch = 52.0 + 12.0 * u01k(blk.id, 23);
                let ri = (c / pitch).floor();
                let cr = c - (ri + 0.5) * pitch;
                let nrow = (lc / pitch).floor();
                let rh = hash2(blk.id ^ 0x5AB, ri as i64, 0);
                let seg = 70.0 + 50.0 * u01k(rh, 1);
                let si = (a / seg).floor();
                let ar = a - (si + 0.5) * seg;
                let sh = hash2(rh, si as i64, 1);
                lot_id = sh;
                let dbc = dens_at(DVec2::ZERO);
                cov_lot = 1.0;
                col = mixc(yard, c_asphalt() * 1.15, 0.3 * u01k(sh, 4));
                cls = crate::landcover::RESIDENTIAL;
                lot_mean = Some((col, 0.3, 6.0));
                rf = (pl.arch, u01k(sh, 6), true, 1.0);
                if ri < nrow && u01k(sh, 2) < 0.85 && dbc > 0.25 {
                    let point = u01k(sh, 3) < 0.25;
                    let half = if point { DVec2::new(11.0, 11.0) } else { DVec2::new(0.5 * seg - 6.0, 6.5) };
                    let dd = boxd(DVec2::new(ar, cr), DVec2::ZERO, half);
                    let ins = aa(dd.min_element(), fw);
                    if ins > 0.0 {
                        let floors = if point { 14.0 + 12.0 * u01k(sh, 5) } else { 5.0 + 7.0 * u01k(sh, 5) } + if pl.arch == 8 { 8.0 } else { 0.0 };
                        h = floors * 3.0 * ins;
                        rf_ins = ins;
                        roof = ins;
                        rim = (1.0 - smoothstep(0.0, 0.9, dd.min_element())) * ins;
                        if ins > 0.5 {
                            cls = if floors * 3.0 > 30.0 { crate::landcover::BUILDING_TALL } else { crate::landcover::BUILDING };
                        }
                    }
                } else if dbc <= 0.25 {
                    cov_lot = 0.0;
                }
                let tcell = (DVec2::new(a, c) / 15.0).floor();
                let th = hash2(blk.id ^ 0x5AC, tcell.x as i64, tcell.y as i64);
                if roof < 0.5 && u01k(th, 1) < 0.3 && cov_lot > 0.0 {
                    let tp = (tcell + 0.5) * 15.0;
                    tree = aa(3.5 - (DVec2::new(a, c) - tp).length(), fw) * (1.0 - roof);
                    tree_h = 10.0;
                }
            }
            pat::INFORMAL => {
                // dense tiny roofs in alleys, thinning out with the footprint (per roof cell)
                let dbc = dens_at(wc.point - q);
                if u01k(wc.id, 7) < dbc.powf(0.8) * 1.15 {
                    lot_id = wc.id;
                    cov_lot = 1.0;
                    col = srgb(128.0, 110.0, 90.0);
                    cls = crate::landcover::RESIDENTIAL;
                    lot_mean = Some((col, 0.7, 3.5 * (0.25 + 0.75 * dbc)));
                    rf = (22, u01k(wc.id, 4), false, 0.85 + 0.3 * u01k(wc.id, 5));
                    let ed = worley2_edge_dist(&wc, qw);
                    let ins = aa(ed - 0.6, fw);
                    if u01k(wc.id, 3) < 0.25 + 0.75 * dbc {
                        rf_ins = ins;
                        h = (3.0 + 3.5 * u01k(wc.id, 6)) * ins;
                        roof = ins;
                        if ins > 0.5 {
                            cls = crate::landcover::BUILDING;
                        }
                    }
                }
            }
            pat::LINEAR => {
                // a farm: house by the street, barn behind, yard; strips beyond (forest-strip)
                let lc = DVec2::new(0.5 * blk.bw - blk.u, -q.y);
                let dl = dens_at(lc);
                let farm = u01k(blk.id, 3) < dl.powf(0.6) * 1.1;
                if farm && blk.v < blk.bd {
                    cov_lot = 1.0;
                    col = mixc(yard, srgb(150.0, 138.0, 118.0), 0.4);
                    cls = crate::landcover::FARMYARD;
                    lot_mean = Some((col, 0.25, 1.6));
                    rf = (pl.arch, u01k(blk.id, 5), false, 1.0);
                    let hh = DVec2::new((0.5 * blk.bw - 2.5).min(6.5), 5.5);
                    let hc = DVec2::new(0.5 * blk.bw, 4.0 + 2.0 * u01k(blk.id, 4) + hh.y);
                    let bh = DVec2::new((0.5 * blk.bw - 2.0).min(8.0), 6.0);
                    let bc = DVec2::new(0.5 * blk.bw, blk.bd - 4.0 - bh.y);
                    let dh = boxd(ublk, hc, hh);
                    let db = boxd(ublk, bc, bh);
                    let ih = aa(dh.min_element(), fw);
                    let ib = aa(db.min_element(), fw);
                    if ih > 0.0 {
                        let ridge = (dh.y / hh.y).clamp(0.0, 1.0);
                        rf_ins = ih;
                        rf_shade = 0.9 + 0.2 * ridge;
                        h = (5.5 + 2.5 * ridge) * ih;
                        roof = ih;
                        rim = (1.0 - smoothstep(0.0, 0.9, dh.min_element())) * ih;
                        if ih > 0.5 {
                            cls = crate::landcover::BUILDING;
                        }
                    } else if ib > 0.0 {
                        let ridge = (db.y / bh.y).clamp(0.0, 1.0);
                        rf = (pl.arch, u01k(blk.id, 6), false, 0.85);
                        rf_ins = ib;
                        rf_shade = 0.9 + 0.15 * ridge;
                        h = (6.0 + 2.0 * ridge) * ib;
                        roof = ib;
                        if ib > 0.5 {
                            cls = crate::landcover::BUILDING;
                        }
                    }
                    if ih <= 0.0 && ib <= 0.0 && u01k(blk.id, 7) < 0.6 {
                        let tp = DVec2::new(0.25 * blk.bw, 0.55 * blk.bd);
                        tree = aa(3.5 - (ublk - tp).length(), fw);
                        tree_h = 9.0;
                    }
                    em += DVec3::new(1.0, 0.72, 0.42) * point_light((ublk - DVec2::new(hc.x, hc.y - hh.y - 1.0)).length_squared(), 2.0, 0.4, fw);
                } else if farm && fm == form::WALDHUFEN as u32 {
                    // the farm's strip up the valley side: fields with hedges, forest at its end
                    let len = 500.0 + 900.0 * u01k(blk.id, 8).powf(0.7);
                    let sv = blk.v - blk.bd;
                    if sv < len {
                        cov_lot = 1.0;
                        let hedge = band_cov(blk.u.min(blk.bw - blk.u), 1.6, fw);
                        let seg = (sv / (120.0 + 80.0 * u01k(blk.id, 9))).floor();
                        let fh = hash2(blk.id ^ 0xF1E, seg as i64, 0);
                        let crop = match (u01k(fh, 1) * 3.99) as u32 {
                            0 => srgb(168.0, 150.0, 92.0),
                            1 => srgb(110.0, 132.0, 66.0),
                            2 => srgb(132.0, 110.0, 80.0),
                            _ => srgb(96.0, 124.0, 60.0),
                        };
                        col = mixc(crop * (0.94 + 0.12 * e.detail), srgb(52.0, 74.0, 40.0), hedge);
                        cls = if hedge > 0.5 { crate::landcover::HEDGEROW } else { crate::landcover::CROP };
                        tree = hedge * smoothstep(0.3, 0.6, u01k(hash2(blk.id, (sv / 9.0).floor() as i64, 2), 1));
                        tree_h = 8.0;
                    }
                }
            }
            _ => {
                if matches!(pattern, pat::DISPERSED | pat::KRAAL) {
                    // handled below (W2 units)
                } else if let Some(os) = if oldg {
                    // (by small block, from the share at its centre: the quarter fades into the grid
                    // over a band, its outline irregular; whole blocks drew a hard star of
                    // W2 cells, town@type)
                    let os = old_sub(&blk);
                    (u01k(os.id, 77) < old_core(pl, q + rot2(os.c - ublk, -theta))).then_some(os)
                } else {
                    None
                } {
                    // the old core: the block cut by lanes into small blocks, terraced houses
                    // 9-13 m deep along the streets and lanes, courtyards inside, a small square
                    // here and there
                    let sh = os.id;
                    let dsub = os.d;
                    let depthb = 8.0 + 6.0 * u01k(sh, 2);
                    // (houses of 6-11 m frontage along the edge, each its own height and roof: rows
                    // of one roof read as stamped perimeter blocks)
                    let uw = 6.0 + 5.0 * u01k(sh, 12);
                    let hu = hash3(sh ^ 0x40F, os.side, (os.along / uw).floor() as i64, 1);
                    lot_id = hu;
                    cov_lot = 1.0;
                    // (courtyards of gardens and trees, not pale paving)
                    col = mixc(mixc(yard, c_paving(), 0.3), srgb(70.0, 96.0, 52.0), 0.35 + 0.3 * e.moist.clamp(0.0, 1.0));
                    cls = if z == zone::CBD { crate::landcover::COMMERCIAL } else { crate::landcover::RESIDENTIAL };
                    let square = u01k(sh, 1) < 0.07;
                    // (a big city's quarter 4-7 storeys: a low patch beside the towers read as a hole)
                    let hb = 3.0 * if pl.level >= 3 { 4.0 + 3.0 * u01k(sh, 3) } else { 3.0 + 2.0 * u01k(sh, 3) } * if z == zone::CBD { 1.25 } else { 1.0 };
                    let flat = u01k(sh, 6) < 0.06;
                    let frac = if square { 0.0 } else { 0.7 };
                    lot_mean = Some((col, frac, hb * frac));
                    let hb = hb + 3.0 * ((u01k(hu, 1) * 2.99).floor() - 1.0) * 0.8;
                    col = mixc(col, c_paving() * 0.9, aa(-dsub, fw).max(if square { 1.0 } else { 0.0 }));
                    // (the warm middle of the culture's pitched roofs: tiles, not the dark slates)
                    rf = (pl.arch, 0.2 + 0.6 * u01k(hu, 9), flat, 0.86 + 0.28 * u01k(hu, 11));
                    let dr = dsub.min(depthb - dsub);
                    let ins = if square { 0.0 } else { aa(dr, fw) };
                    if ins > 0.0 {
                        let hh = if flat { hb } else { hb + 0.5 * dr };
                        if flat {
                            rf_shade = 1.0 - 0.15 * band_cov(dr, 0.6, fw);
                        }
                        rf_ins = ins;
                        h = hh * ins;
                        roof = ins;
                        rim = (1.0 - smoothstep(0.0, 0.9, dr)) * ins;
                        if ins > 0.5 {
                            cls = crate::landcover::BUILDING;
                        }
                    } else if dsub > depthb || square {
                        // courtyard trees
                        let tc = (qw / 10.0).floor();
                        let th = hash2(sh ^ 0xC07, tc.x as i64, tc.y as i64);
                        if u01k(th, 1) < 0.35 {
                            let tp = (tc + 0.5 + 0.4 * (DVec2::new(u01k(th, 2), u01k(th, 3)) - 0.5)) * 10.0;
                            tree = aa(2.6 + 1.2 * u01k(th, 4) - (qw - tp).length(), fw);
                            tree_h = 8.0 + 5.0 * u01k(th, 5);
                        }
                    }
                } else {
                    // lots by BSP and buildings by typology
                    // (a town's centre: v0.2.0's dense small lots, 13-25 m wide)
                    let core = town_core(pl, z, pattern);
                    let depth = match z {
                        _ if core => 4,
                        zone::CBD => 1,
                        zone::INNER => 2,
                        _ => 3,
                    };
                    // (blocks of light industry beside an estate on the town's grid, fewer with the
                    // distance to it (by the district at the block's centre: whole blocks): the
                    // estates read as separate patches, town@type, fp_town)
                    let lind = pattern == pat::GRID && nbind < 260.0 && {
                        let qc = q + rot2(DVec2::new(0.5 * blk.bw - blk.u, 0.5 * blk.bd - blk.v), -theta);
                        u01k(blk.id, 79) < 0.6 * (1.0 - smoothstep(30.0, 230.0, nbind_at(pl, qc)))
                    };
                    let ind = pattern == pat::INDUSTRIAL || lind;
                    // (estates: lots of 30-150 m, halls, yards and car parks; lots of 45 m and up
                    // on blocks merged from the town's drew one sheet of roof, town@type)
                    let depth = if ind { 3 } else { depth };
                    let min_side = if lind {
                        22.0
                    } else if ind {
                        30.0
                    } else if z == zone::CBD {
                        22.0
                    } else {
                        9.0
                    };
                    let (lh, lr) = bsp(blk.id, blk.u, blk.v, blk.bw, blk.bd, depth, min_side);
                    lot_id = lh;
                    let lc = DVec2::new(0.5 * (lr[0] + lr[2]), 0.5 * (lr[1] + lr[3]));
                    let lot_dq = rot2(lc - ublk, -theta);
                    let dl = dens_at(lot_dq);
                    // (a town's lots end where its streets do (above their density 0.42): houses strewn over the fields
                    // beyond read as a broken town)
                    // (the threshold of each lot its own: lots ending along the blocks drew the town's edge
                    // as a staircase of its street grid, fp_town; now houses and gardens thin out
                    // beyond the last streets)
                    // (round 11: and lots along the main streets further out, and a few scattered houses
                    // beyond the streets: the edge still followed the grid, fp_town)
                    let on_major = (blk.mj & 1 != 0 && lr[0] <= 0.0)
                        || (blk.mj & 2 != 0 && lr[2] >= blk.bw)
                        || (blk.mj & 4 != 0 && lr[1] <= 0.0)
                        || (blk.mj & 8 != 0 && lr[3] >= blk.bd);
                    let thr = edge_thr - 0.16 * u01k(lh, 33) - if on_major { 0.16 } else { 0.0 };
                    let edge = if fm == form::CITY as u32 { smoothstep(thr, thr + 0.1, dl).max(0.22 * smoothstep(thr - 0.14, thr - 0.04, dl)) } else { 1.0 };
                    // (within the streets most lots are built: suburbs at the old 55-75 % read
                    // as sparse next to v0.2.0's dense blocks; beyond them houses stay few)
                    let occ = lerp(dl.powf(0.7) * 1.05, dl.powf(0.35) * 1.1, smoothstep(0.35, 0.5, dl));
                    // (round 12: the lots along the streets of the edge and along the ribbons mostly
                    // built: streets between empty lots read as stubs, fp_town)
                    let occ = if fm == form::CITY as u32 {
                        occ.max(0.88 * smoothstep(0.32, 0.42, dl)).max(if on_major { 0.85 * smoothstep(thr, thr + 0.06, dl) } else { 0.0 })
                    } else {
                        occ
                    };
                    let built = u01k(lh, 3) < occ * edge;
                    // typology
                    let a = pl.arch;
                    let ut = u01k(lh, 12);
                    let euro = matches!(a, 3 | 5 | 6 | 8 | 9 | 11 | 4);
                    let t = if ind {
                        typ::WAREHOUSE
                    } else if fm != form::CITY as u32 {
                        if z == zone::INNER && euro {
                            typ::PERIM
                        } else if a == 3 || (euro && ut < 0.6) {
                            typ::ROW
                        } else {
                            typ::HOUSE
                        }
                    } else {
                        // (the skyline thins and falls off from the centre, and towers stand beyond the
                        // CBD near it: a round cake of towers with a hard edge read as stamped on)
                        let fall = smoothstep(0.0, 0.14, rel_here);
                        match z {
                            zone::CBD => {
                                if pl.level >= 3 && ut < 0.75 * (1.0 - 0.5 * fall) {
                                    typ::TOWER
                                } else {
                                    typ::MIDRISE
                                }
                            }
                            zone::INNER if pl.level >= 3 && u01k(lh, 21) < 0.12 * (1.0 - smoothstep(0.1, 0.2, rel_here)) => typ::TOWER,
                            zone::INNER => {
                                // (towns: mostly detached blocks of flats on lawns, v0.2.0's
                                // boxes; perimeter blocks are the cities' (they filled every
                                // block of a town centre))
                                let (perim, mid) = if pl.level >= 3 { (0.8, 0.6) } else { (0.25, 0.75) };
                                if euro && pattern != pat::RADIAL && ut < perim || pattern == pat::ORGANIC {
                                    typ::PERIM
                                } else if ut < mid {
                                    typ::MIDRISE
                                } else {
                                    typ::ROW
                                }
                            }
                            zone::OUTER => {
                                if ut < 0.45 {
                                    typ::MIDRISE
                                } else if ut < 0.75 {
                                    typ::ROW
                                } else {
                                    typ::HOUSE
                                }
                            }
                            _ => {
                                if matches!(a, 0 | 1 | 8 | 3) && ut < 0.6 {
                                    typ::ROW
                                } else {
                                    typ::HOUSE
                                }
                            }
                        }
                    };
                    let town_box = t == typ::MIDRISE && core;
                    // an estate's lot: a hall, a car park, a storage yard or vacant
                    let wuse = if t == typ::WAREHOUSE {
                        let lu = u01k(lh, 40);
                        if lu < 0.6 {
                            0
                        } else if lu < 0.74 {
                            1
                        } else if lu < 0.88 {
                            2
                        } else {
                            3
                        }
                    } else {
                        0
                    };
                    // (a hall's loading apron: its side and depth, the other setbacks)
                    let wside = (u01k(lh, 41) * 3.99) as u32;
                    let w_so = 5.0 + 3.0 * u01k(lh, 42);
                    let w_sf = 14.0 + 12.0 * u01k(lh, 43);
                    if built || dl > 0.65 {
                        cov_lot = 1.0;
                        col = match t {
                            typ::WAREHOUSE => mixc(c_concrete(), c_asphalt(), 0.35 + 0.45 * u01k(lh, 10)),
                            // (a town's blocks of flats stand on watered lawns: v0.2.0's green lots)
                            typ::MIDRISE if town_box => mixc(mixc(yard, srgb(86.0, 112.0, 54.0), 0.7), c_paving(), 0.15),
                            typ::TOWER | typ::MIDRISE => mixc(c_paving(), yard, 0.25),
                            // (courtyards: gardens and trees, not a bare paved square)
                            typ::PERIM => mixc(mixc(yard, c_paving(), 0.35), srgb(84.0, 112.0, 60.0), 0.25 + 0.35 * e.moist.clamp(0.0, 1.0)),
                            _ => yard,
                        };
                        cls = match t {
                            typ::WAREHOUSE => crate::landcover::INDUSTRIAL,
                            typ::TOWER => crate::landcover::COMMERCIAL,
                            typ::MIDRISE if z == zone::CBD => crate::landcover::COMMERCIAL,
                            _ => crate::landcover::RESIDENTIAL,
                        };
                    }
                    let lw = lr[2] - lr[0];
                    let ld = lr[3] - lr[1];
                    if built && wuse == 0 {
                        let rl = ublk - lc;
                        // footprint: setbacks by typology (and its centre in the lot)
                        let mut cen = DVec2::ZERO;
                        let (half, hb, flat) = match t {
                            typ::TOWER => {
                                let s = (0.5 * lw.min(ld) - 4.0).clamp(6.0, 24.0);
                                let ps = smoothstep(1e5, 1.5e7, pl.pop);
                                let fall = smoothstep(0.0, 0.2, rel_here);
                                (DVec2::splat(s), (40.0 + 360.0 * ps * u01k(lh, 4).powi(3) + 40.0 * u01k(lh, 5)) * (1.0 - 0.6 * fall), true)
                            }
                            typ::MIDRISE => {
                                if town_box {
                                    // (a town's: v0.2.0's houses and blocks, the lot minus
                                    // 1.5-4 m setbacks, 3.5-8 m high and a few taller ones,
                                    // mostly pitched roofs)
                                    let sb = 1.5 + 2.5 * u01k(lh, 1);
                                    let u5 = u01k(lh, 5);
                                    (
                                        DVec2::new((0.5 * lw - sb).min(13.0), (0.5 * ld - sb).min(11.0)),
                                        3.5 + 4.5 * u01k(lh, 4) + 16.0 * u5 * u5 * u5 * u5,
                                        u01k(lh, 6) < 0.35,
                                    )
                                } else {
                                    let fill = if z <= zone::INNER { DVec2::new(60.0, 60.0) } else { DVec2::new(40.0, 14.0 + 10.0 * u01k(lh, 6)) };
                                    (
                                        DVec2::new(0.5 * lw - 2.5, 0.5 * ld - 2.5).min(fill),
                                        3.0 * (4.0 + 4.0 * u01k(lh, 4)) + if z == zone::CBD { 12.0 } else { 0.0 },
                                        true,
                                    )
                                }
                            }
                            typ::WAREHOUSE => {
                                // (the apron on one side; at most 160 x 100 m; half of them with a
                                // shallow gable)
                                let sb = [0u32, 1, 2, 3].map(|k| if k == wside { w_sf } else { w_so });
                                let hx = 0.5 * (lw - sb[0] - sb[1]);
                                let hy = 0.5 * (ld - sb[2] - sb[3]);
                                cen = DVec2::new(0.5 * (sb[0] - sb[1]), 0.5 * (sb[2] - sb[3]));
                                let half = if hx > hy { DVec2::new(hx.min(80.0), hy.min(50.0)) } else { DVec2::new(hx.min(50.0), hy.min(80.0)) };
                                (half, 8.0 + 6.0 * u01k(lh, 4), u01k(lh, 6) < 0.5)
                            }
                            typ::ROW => {
                                (DVec2::new(0.5 * lw - 0.4, (0.5 * ld - 2.0).min(7.5)), 3.0 * (2.0 + (1.6 * u01k(lh, 4)).floor()), u01k(lh, 6) < 0.3 || a == 3)
                            }
                            typ::PERIM => (DVec2::splat(1e4), 3.0 * (3.0 + 3.0 * u01k(lh, 4)) * if pl.level >= 3 { 1.0 } else { 0.7 }, u01k(lh, 6) < 0.35),
                            _ => {
                                // (v0.2.0's houses: 10-25 m footprints, 3.5-7.5 m high; the
                                // smaller, lower houses of 8-15 m read as a sparse town)
                                let sb = 2.0 + 2.5 * u01k(lh, 1);
                                (
                                    DVec2::new((0.5 * lw - sb).min(8.0 + 4.0 * u01k(lh, 7)), (0.5 * ld - sb).min(6.5 + 4.0 * u01k(lh, 8))),
                                    3.5 + 4.0 * u01k(lh, 4),
                                    a == 3 || u01k(lh, 6) < 0.12,
                                )
                            }
                        };
                        let mut podium = 0.0;
                        let (dd, mut ins) = if t == typ::PERIM {
                            // a band along the block's edge, the courtyard inside
                            let dblk = blk.u.min(blk.v).min(blk.bw - blk.u).min(blk.bd - blk.v);
                            let depthb = 11.0 + 3.0 * u01k(blk.id, 24);
                            let dd = DVec2::splat(depthb - dblk);
                            (dd, aa(dd.x, fw))
                        } else {
                            let dd = boxd(rl, cen, half);
                            if t == typ::TOWER {
                                // the podium around the tower
                                podium = aa(boxd(rl, DVec2::ZERO, DVec2::new(0.5 * lw - 3.0, 0.5 * ld - 3.0)).min_element(), fw);
                            }
                            (dd, aa(dd.min_element(), fw))
                        };
                        if half.min_element() < 2.0 {
                            ins = 0.0;
                        }
                        rf = (
                            if t == typ::TOWER {
                                20
                            } else if t == typ::WAREHOUSE {
                                21
                            } else {
                                a
                            },
                            u01k(lh, 9),
                            flat,
                            0.84 + 0.32 * u01k(lh, 11),
                        );
                        let frac = match t {
                            typ::HOUSE => 0.3,
                            _ if town_box => 0.45,
                            typ::TOWER => 0.75,
                            typ::WAREHOUSE => 0.6,
                            _ => 0.55,
                        };
                        lot_mean = Some((col, frac, hb * frac));
                        if ins > 0.0 {
                            let mut hh = hb;
                            if !flat {
                                // pitched roof along the longer axis
                                let dperp = if t == typ::PERIM {
                                    dd.x
                                } else if half.x > half.y {
                                    dd.y
                                } else {
                                    dd.x
                                };
                                let hw = if t == typ::PERIM { 0.5 * (11.0 + 3.0 * u01k(blk.id, 24)) } else { half.x.min(half.y) };
                                let pitch = if t == typ::WAREHOUSE { 0.1 } else { 0.35 };
                                hh = hb + pitch * hw * (dperp / hw).clamp(0.0, 1.0);
                            } else {
                                rf_shade = 1.0 - 0.15 * band_cov(dd.min_element(), 0.6, fw);
                            }
                            rf_ins = ins;
                            h = hh * ins;
                            roof = ins;
                            rim = (1.0 - smoothstep(0.0, 0.9, dd.min_element())) * ins;
                            if ins > 0.5 {
                                cls = if hh > 30.0 { crate::landcover::BUILDING_TALL } else { crate::landcover::BUILDING };
                            }
                            if t == typ::TOWER || (t == typ::MIDRISE && z == zone::CBD) {
                                win = DVec3::new(0.82, 0.90, 1.0);
                            }
                            if t == typ::WAREHOUSE {
                                // skylight strips across the hall, HVAC units (their mean below
                                // 4-5 m pixels)
                                let rb = rl - cen;
                                let a_l = if half.x > half.y { rb.x } else { rb.y };
                                let e_in = dd.min_element();
                                let mut det = 0.0;
                                if u01k(lh, 44) < 0.6 {
                                    let ps = 12.0 + 6.0 * u01k(lh, 45);
                                    let f = a_l / ps - (a_l / ps).round();
                                    det = 0.2 * band_cov(f * ps, 0.9, fw) * smoothstep(2.0, 3.0, e_in) * band(4.0, e.gsd);
                                }
                                let cu = (rb / 16.0).floor();
                                let ch = hash2(lh ^ 0xAC, cu.x as i64, cu.y as i64);
                                if u01k(ch, 1) < 0.35 {
                                    let cp = (cu + 0.5 + 0.5 * (DVec2::new(u01k(ch, 2), u01k(ch, 3)) - 0.5)) * 16.0;
                                    let unit = aa(boxd(rb, cp, DVec2::splat(1.2 + 1.3 * u01k(ch, 4))).min_element(), fw) * smoothstep(3.0, 4.0, e_in);
                                    det = det.max(0.3 * unit * band(5.0, e.gsd));
                                    h += 1.5 * unit * ins;
                                }
                                rf_shade *= 1.0 - det;
                            }
                        } else if podium > 0.0 {
                            rf = (a, u01k(lh, 19), true, 1.0);
                            rf_ins = podium;
                            h = (12.0 + 8.0 * u01k(lh, 20)) * podium;
                            roof = podium;
                            if podium > 0.5 {
                                cls = crate::landcover::BUILDING;
                            }
                        } else if t == typ::HOUSE {
                            // a pool behind some houses in rich suburbs, a yard tree
                            if pl.dev > 0.55 && matches!(a, 5 | 7 | 3 | 0 | 1) && u01k(lh, 13) < 0.35 {
                                let pc = DVec2::new(0.0, half.y + 0.5 * (0.5 * ld - half.y));
                                let dp = boxd(rl, pc, DVec2::new(2.2, 4.0));
                                let pool = aa(dp.min_element(), fw);
                                col = mixc(col, srgb(70.0, 170.0, 190.0), pool);
                            }
                            if u01k(lh, 14) < 0.2 {
                                let tp = DVec2::new(0.6 * (0.5 * lw) * (u01k(lh, 15) - 0.5) * 2.0, -(half.y + 0.5 * (0.5 * ld - half.y)));
                                tree = aa(3.0 + 1.5 * u01k(lh, 16) - (rl - tp).length(), fw);
                                tree_h = 8.0 + 5.0 * u01k(lh, 17);
                            }
                            em += DVec3::new(1.0, 0.72, 0.42)
                                * point_light((rl - DVec2::new(0.0, half.y + 1.0)).length_squared(), 2.0, 0.4, fw)
                                * (u01k(lh, 18) < 0.6) as u32 as f64;
                        } else if t == typ::PERIM {
                            // courtyard trees
                            let tc = (ublk / 11.0).floor();
                            let th = hash2(blk.id ^ 0xC07, tc.x as i64, tc.y as i64);
                            if u01k(th, 1) < 0.25 {
                                let tp = (tc + 0.5 + 0.4 * (DVec2::new(u01k(th, 2), u01k(th, 3)) - 0.5)) * 11.0;
                                tree = aa(2.8 + 1.5 * u01k(th, 4) - (ublk - tp).length(), fw);
                                tree_h = 9.0 + 6.0 * u01k(th, 5);
                            }
                        } else if t == typ::WAREHOUSE {
                            // trailers at the loading docks, a tank in the corner of the apron
                            let rb = rl - cen;
                            let (n, t2, hf, ht) = match wside {
                                0 => (-rb.x, rb.y, half.x, half.y),
                                1 => (rb.x, rb.y, half.x, half.y),
                                2 => (-rb.y, rb.x, half.y, half.x),
                                _ => (rb.y, rb.x, half.y, half.x),
                            };
                            let dn = n - hf;
                            if dn > 0.0 && dn < 16.0 && t2.abs() < ht - 3.0 {
                                let k = (t2 / 4.2).floor();
                                let th = hash2(lh ^ 0x7C, k as i64, 0);
                                if u01k(th, 1) < 0.45 {
                                    let c = (t2 - (k + 0.5) * 4.2).abs();
                                    let tr = aa(1.3 - c, fw) * aa(dn - 1.0, fw) * aa(15.0 - dn, fw) * band(3.0, e.gsd);
                                    col = mixc(col, if u01k(th, 2) < 0.6 { srgb(214.0, 214.0, 208.0) } else { srgb(150.0, 152.0, 156.0) }, tr);
                                    h = h.max(4.0 * tr);
                                }
                            }
                            if u01k(lh, 13) < 0.3 && (wside == 1 || wside == 3) {
                                let tr = (6.0 + 9.0 * u01k(lh, 14)).min(0.5 * w_sf - 2.5);
                                let tc = DVec2::new(0.5 * lw - tr - 3.0, 0.5 * ld - tr - 3.0);
                                let dt = (rl - tc).length();
                                let tank = aa(tr - dt, fw);
                                col = mixc(col, srgb(206.0, 206.0, 200.0) * (0.85 + 0.15 * (1.0 - dt / tr)), tank);
                                h = h.max(10.0 * tank);
                                roof = roof.max(tank);
                            }
                            em += DVec3::new(1.0, 0.5, 0.15) * 0.025 * band(30.0, e.gsd);
                        }
                    } else if wuse > 0 && cov_lot > 0.0 {
                        let (u, v) = (ublk.x - lr[0], ublk.y - lr[1]);
                        if wuse == 1 {
                            col = car_park(lh, u, v, fw, e.gsd);
                            cls = crate::landcover::PARKING;
                        } else if wuse == 2 {
                            // a storage yard: rows of containers
                            col = mixc(c_concrete(), srgb(150.0, 140.0, 124.0), u01k(lh, 46)) * (0.9 + 0.1 * e.detail);
                            let (a, c) = if lw > ld { (u, v) } else { (v, u) };
                            let (ri, ci) = ((c / 6.0).floor(), (a / 12.8).floor());
                            let ch = hash2(lh ^ 0xC0, ci as i64, ri as i64);
                            if u01k(ch, 1) < 0.55 {
                                let din = u.min(v).min(lw - u).min(ld - v);
                                let cont = aa(1.25 - (c - (ri + 0.5) * 6.0).abs(), fw)
                                    * aa(6.1 - (a - (ci + 0.5) * 12.8).abs(), fw)
                                    * aa(din - 4.0, fw)
                                    * band(3.0, e.gsd);
                                let cc = [
                                    srgb(150.0, 74.0, 52.0),
                                    srgb(62.0, 84.0, 124.0),
                                    srgb(78.0, 106.0, 74.0),
                                    srgb(204.0, 204.0, 198.0),
                                    srgb(124.0, 124.0, 122.0),
                                ];
                                col = mixc(col, cc[(u01k(ch, 2) * 4.99) as usize], cont);
                                h = h.max(2.6 * (1.0 + (2.0 * u01k(ch, 3)).floor()) * cont);
                            }
                        } else {
                            // vacant: rough grass
                            col = mixc(yard, srgb(118.0, 126.0, 78.0), 0.5) * (0.9 + 0.15 * e.detail);
                            cls = crate::landcover::GRASS;
                        }
                    } else if fm == form::CITY as u32 && dl > edge_thr - 0.06 && u01k(lh, 22) < 0.55 {
                        // (the town's fringe: gardens and allotments on the lots left unbuilt, not
                        // fields butting against the last blocks)
                        cov_lot = 1.0;
                        let sx = ((ublk.x - lr[0]) / 6.0).floor();
                        let gk = u01k(hash2(lh ^ 0x6A4D, sx as i64, 0), 1);
                        col = mixc(srgb(96.0, 118.0, 62.0), srgb(122.0, 104.0, 78.0), gk) * (0.92 + 0.1 * e.detail);
                        cls = crate::landcover::CROP;
                    }
                }
            }
        }
    }
    // ---- farms and kraals (cellular)
    if pattern == pat::DISPERSED || pattern == pat::KRAAL {
        let pc = wc.point;
        let dl = if pattern == pat::KRAAL && pl.level == 0 { 1.0 } else { dens_at(pc - q) };
        let unit_c = if pattern == pat::KRAAL && pl.level == 0 { DVec2::ZERO } else { pc };
        let ok = pattern == pat::KRAAL && pl.level == 0 || u01k(wc.id, 3) < dl * 0.9;
        let r = q - unit_c;
        let uh = if pattern == pat::KRAAL && pl.level == 0 { pl.seed } else { wc.id };
        if ok && pattern == pat::DISPERSED {
            let rr = rot2(r, (u01k(uh, 1) - 0.5) * 0.8);
            let yh = DVec2::new(20.0 + 8.0 * u01k(uh, 2), 14.0 + 6.0 * u01k(uh, 3));
            let dy = boxd(rr, DVec2::ZERO, yh);
            let iy = aa(dy.min_element(), fw);
            if iy > 0.0 {
                cov_lot = iy;
                col = mixc(yard, srgb(150.0, 138.0, 118.0), 0.5);
                cls = crate::landcover::FARMYARD;
                let hc = DVec2::new(-0.45 * yh.x, -0.3 * yh.y);
                let dh = boxd(rr, hc, DVec2::new(5.5, 4.5));
                let ih = aa(dh.min_element(), fw);
                let bc = DVec2::new(0.3 * yh.x, 0.35 * yh.y);
                let db = boxd(rr, bc, DVec2::new(10.0, 6.0));
                let ib = aa(db.min_element(), fw);
                if ih > 0.0 {
                    let ridge = (dh.y / 4.5).clamp(0.0, 1.0);
                    rf = (pl.arch, u01k(uh, 4), false, 1.0);
                    rf_ins = ih;
                    h = (5.0 + 2.0 * ridge) * ih;
                    roof = ih;
                    rim = (1.0 - smoothstep(0.0, 0.9, dh.min_element())) * ih;
                    if ih > 0.5 {
                        cls = crate::landcover::BUILDING;
                    }
                } else if ib > 0.0 {
                    let ridge = (db.y / 6.0).clamp(0.0, 1.0);
                    rf = (pl.arch, u01k(uh, 5), false, 0.85);
                    rf_ins = ib;
                    h = (6.0 + 1.5 * ridge) * ib;
                    roof = ib;
                    if ib > 0.5 {
                        cls = crate::landcover::BUILDING;
                    }
                }
                em += DVec3::new(1.0, 0.72, 0.42) * point_light((rr - hc - DVec2::new(0.0, 5.5)).length_squared(), 2.0, 0.4, fw);
            }
            // shelter trees around the yard
            let dyr = -dy.min_element();
            if dyr > 2.0 && dyr < 9.0 && u01k(hash2(uh, (rr.x / 8.0).floor() as i64, (rr.y / 8.0).floor() as i64), 1) < 0.45 {
                tree = 1.0;
                tree_h = 10.0;
            }
        } else if ok {
            // a ring of round huts around a cattle enclosure, a thorn fence outside
            let rk = 14.0 + 10.0 * u01k(uh, 2);
            let d = r.length();
            let ang = r.y.atan2(r.x);
            let nh = (6.0 + 8.0 * u01k(uh, 3)).floor();
            let k = (ang / std::f64::consts::TAU * nh).round();
            let a0 = k / nh * std::f64::consts::TAU + 0.3 * (u01k(hash1(uh, k as i64), 1) - 0.5);
            let hc = DVec2::new(a0.cos(), a0.sin()) * rk;
            let hr = 2.6 + 0.8 * u01k(hash1(uh, k as i64), 2);
            let dh = (r - hc).length();
            let ih = aa(hr - dh, fw);
            let enc = aa(0.55 * rk - d, fw);
            let fence = band_cov(d - (rk + 6.0), 0.8, fw).max(band_cov(d - 0.55 * rk, 0.6, fw));
            let yardk = aa(rk + 6.0 - d, fw);
            if yardk > 0.0 {
                cov_lot = yardk;
                col = srgb(156.0, 130.0, 98.0) * (0.92 + 0.15 * e.detail);
                cls = crate::landcover::FARMYARD;
                col = mixc(col, srgb(112.0, 92.0, 70.0), enc);
                col = mixc(col, srgb(84.0, 76.0, 58.0), fence * band(1.6, e.gsd) + 0.3 * fence * (1.0 - band(1.6, e.gsd)));
                if ih > 0.0 {
                    let cone = 1.0 - (dh / hr).clamp(0.0, 1.0);
                    rf = (2, u01k(hash1(uh, k as i64), 3), false, 1.0);
                    rf_ins = ih;
                    rf_shade = 0.85 + 0.25 * cone;
                    h = (2.2 + 1.6 * cone) * ih;
                    roof = ih;
                    if ih > 0.5 {
                        cls = crate::landcover::BUILDING;
                    }
                }
            }
        }
    }
    // ---- the village centre: a plaza and its landmark (church, mosque, temple)
    if fm != form::CITY as u32 && fm != form::DISPERSED as u32 && fm != form::KRAAL as u32 && pl.level >= 1 {
        let d = q.length();
        if d < 26.0 {
            let pz = aa(22.0 - d, fw);
            col = mixc(col, c_paving(), pz);
            cov_lot = cov_lot.max(pz);
            o.street = 0.0;
            let lm = match pl.arch {
                3 | 4 => 1,
                8 | 0 | 1 => 2,
                _ => 0,
            };
            let (ins, hh, rc) = if lm == 1 {
                // a dome and a minaret
                let dd = 8.0 - d;
                let mn = 1.6 - (q - DVec2::new(11.0, 0.0)).length();
                let i1 = aa(dd, fw).max(aa(mn, fw));
                let hh = if mn > dd { 30.0 } else { 9.0 + 6.0 * (dd / 8.0).clamp(0.0, 1.0).sqrt() };
                (i1, hh, 23u32)
            } else if lm == 2 {
                let dd = boxd(q, DVec2::ZERO, DVec2::new(8.0, 8.0));
                (aa(dd.min_element(), fw), 10.0 + 4.0 * (dd.min_element() / 8.0).clamp(0.0, 1.0), 24u32)
            } else {
                // nave and tower
                let dn = boxd(q, DVec2::new(0.0, 0.0), DVec2::new(14.0, 5.5));
                let dt = boxd(q, DVec2::new(-16.0, 0.0), DVec2::new(3.2, 3.2));
                let i1 = aa(dn.min_element(), fw).max(aa(dt.min_element(), fw));
                let hh = if dt.min_element() > dn.min_element() { 32.0 } else { 12.0 + 4.0 * (dn.y / 5.5).clamp(0.0, 1.0) };
                (i1, hh, pl.arch)
            };
            if ins > 0.0 {
                rf = (rc, 0.3, false, 0.9);
                rf_ins = ins;
                rf_shade = 1.0;
                h = hh * ins;
                roof = ins;
                if ins > 0.5 {
                    cls = crate::landcover::BUILDING;
                }
            }
            em += DVec3::new(1.0, 0.8, 0.55) * 0.05 * pz * band(20.0, e.gsd);
        }
    }
    // ---- the roof (one colour), lots below the pixel (their mean)
    let rc = roof_col(rf.0, rf.1, rf.2) * rf.3;
    if rf_ins > 0.0 {
        col = mixc(col, rc * rf_shade, rf_ins);
    }
    if let Some((g, frac, mh)) = lot_mean {
        let el = band(lot_scale, e.gsd);
        if el < 1.0 {
            col = mixc(mixc(g, rc, frac), col, el);
            h = mh + (h - mh) * el;
            roof *= el;
            rim *= el;
        }
    }
    // ---- the green buffer along an estate's edge: grass and clumps of trees over its outer lots
    // (estates of grey roofs to their polygon's edge read as stamped on, metro)
    if pattern == pat::INDUSTRIAL && buf < 30.0 && inside {
        let k = smoothstep(30.0, 22.0, buf);
        cov_lot = cov_lot.max(k);
        let g = mixc(yard, srgb(84.0, 112.0, 56.0), 0.5) * (0.95 + 0.1 * e.detail);
        col = mixc(col, g, k);
        h *= 1.0 - k;
        roof *= 1.0 - k;
        rim *= 1.0 - k;
        if k > 0.5 {
            cls = crate::landcover::PARK;
        }
        let tcl = (ublk / 10.0).floor();
        let th = hash2(blk.id ^ 0xB0F, tcl.x as i64, tcl.y as i64);
        let tp = (tcl + 0.5 + 0.5 * (DVec2::new(u01k(th, 1), u01k(th, 2)) - 0.5)) * 10.0;
        if u01k(th, 3) < 0.6 * k {
            let tc = aa(3.0 + 1.5 * u01k(th, 4) - (ublk - tp).length(), fw);
            if tc > tree {
                tree = tc;
                tree_h = 8.0 + 6.0 * u01k(th, 5);
            }
        }
    }
    // ---- street trees (residential streets of tree-lined cultures; some of the streets: a
    // crown every 11 m along every street drew a busy sawtooth around every block, v0.2.0's towns
    // have none)
    let treed = matches!(pl.arch, 5 | 6 | 7 | 9 | 4) && (z == zone::INNER || z == zone::OUTER || z == zone::SUBURB) && u01k(blk.sid, 31) < 0.4;
    if treed && has_streets && street_ok && pattern != pat::INFORMAL && pattern != pat::LINEAR {
        let off = blk.sw + 1.6;
        let dt = (blk.ds - off).abs();
        if dt < 4.5 {
            let k = (blk.along / 11.0).round();
            let th = hash2(blk.sid ^ 0x7EE5, k as i64, (blk.ds > off) as i64);
            if u01k(th, 1) < 0.8 {
                let da = blk.along - k * 11.0;
                let tc = aa(2.8 + 0.8 * u01k(th, 2) - (dt * dt + da * da).sqrt(), fw);
                if tc > tree {
                    tree = tc;
                    tree_h = 9.0 + 4.0 * u01k(th, 3);
                }
            }
        }
    }
    // ---- composite the parts: lot, sidewalk, street
    let street = if has_streets { o.street } else { 0.0 };
    let cov = cov_lot.max(walk).max(street);
    if cov > 0.0 {
        col = mixc(col, c_concrete() * 0.92, (walk - street).max(0.0) * (1.0 - cov_lot) / cov);
        let asph = if o.major {
            c_asphalt() * 0.92
        } else if fm != form::CITY as u32 || (pattern == pat::ORGANIC && pl.level < 3) || pattern == pat::INFORMAL {
            // lanes: stone paving, dirt where poor
            mixc(srgb(150.0, 128.0, 100.0), c_paving() * 0.9, smoothstep(0.35, 0.6, pl.dev))
        } else {
            // (the old core's streets: paved)
            mixc(srgb(76.0, 76.0, 80.0) * (1.0 + 0.05 * e.detail), c_paving() * 0.8, 0.6 * smoothstep(0.3, 0.8, oldk))
        };
        col = mixc(col, asph, street / cov);
        // lane markings on main streets
        if o.major && street > 0.5 {
            let dash = (o.along / 9.0 - (o.along / 9.0).floor() < 0.5) as u32 as f64;
            col = mixc(col, srgb(220.0, 220.0, 210.0), band_cov(o.ds, 0.15, fw) * dash * band(3.0, e.gsd));
        }
        if street > 0.5 {
            cls = crate::landcover::ROAD_MINOR;
            if o.major {
                cls = crate::landcover::ROAD_MAJOR;
            }
        }
        h *= 1.0 - street;
        roof *= 1.0 - street;
        rim *= 1.0 - street;
        tree *= 1.0 - street;
    }
    o.cov = cov;
    o.col = col;
    o.h = h;
    o.cls = cls;
    o.bid = if cls == crate::landcover::BUILDING || cls == crate::landcover::BUILDING_TALL { lot_id } else { 0 };
    o.roof = roof;
    o.rim = rim;
    o.win = win;
    o.em = em;
    o.tree = tree;
    o.tree_h = tree_h;
    o
}

/// Shares of green and paved ground of a zone (the rest is roofs).
pub fn zone_fracs(pl: &Pl, d: &Dist) -> (f64, f64) {
    if town_centre(pl, d) {
        // (a town's centre: v0.2.0's dense houses on small lots)
        return (0.3, 0.3);
    }
    let zi = d.zone as usize;
    ([0.05, 0.12, 0.25, 0.42, 0.5][zi], [0.5, 0.4, 0.32, 0.25, 0.18][zi])
}

/// Is the district a town's centre (v0.2.0's dense houses on small lots, not perimeter blocks)?
fn town_centre(pl: &Pl, d: &Dist) -> bool {
    town_core(pl, d.zone, d.pattern)
}

/// A town's inner and outer zones (towns, levels <= 2; not industrial estates): small lots
/// (BSP depth 4) of houses and blocks of 1-3 floors, v0.2.0's town centres.
fn town_core(pl: &Pl, z: u32, pattern: u32) -> bool {
    (z == zone::INNER || z == zone::OUTER) && pl.level <= 2 && pl.form == form::CITY as u32 && pattern != pat::INDUSTRIAL
}

/// The land-cover class of a sample where the district's lots are below the pixel: drawn from
/// the zone's shares of roofs, streets and lots by a hash of a 6 m cell (classes are
/// categorical: the parent's histogram matches its children's).
pub fn mean_class(pl: &Pl, d: &Dist, q: DVec2) -> u8 {
    let (green, paved) = zone_fracs(pl, d);
    let roofs = 1.0 - green - paved;
    let u = u01k(hash2(pl.seed ^ 0xC1A55, (q.x / 6.0).floor() as i64, (q.y / 6.0).floor() as i64), 1);
    if u < roofs {
        if d.zone == zone::CBD && pl.level >= 3 {
            crate::landcover::BUILDING_TALL
        } else {
            crate::landcover::BUILDING
        }
    } else if u < roofs + 0.55 * paved {
        crate::landcover::ROAD_MINOR
    } else if d.zone == zone::CBD {
        crate::landcover::COMMERCIAL
    } else if d.pattern == pat::INDUSTRIAL {
        crate::landcover::INDUSTRIAL
    } else {
        crate::landcover::RESIDENTIAL
    }
}

/// Mean look of a district where its lots are below the pixel: colour and building height.
pub fn zone_mean(pl: &Pl, d: &Dist, moist: f64) -> (DVec3, f64) {
    let z = d.zone;
    let zi = z as usize;
    let (green, paved) = zone_fracs(pl, d);
    let roofs = 1.0 - green - paved;
    let flat = z <= zone::OUTER && pl.form == form::CITY as u32;
    // (a town centre's roofs: a third flat)
    let rc = if town_centre(pl, d) { mixc(roof_col(pl.arch, 0.35, false), roof_col(pl.arch, 0.35, true), 0.35) } else { roof_col(pl.arch, 0.35, flat) };
    let rc = match d.pattern {
        pat::INDUSTRIAL => srgb(140.0, 140.0, 138.0),
        pat::INFORMAL => srgb(124.0, 104.0, 86.0),
        _ => rc,
    } * (0.85 + 0.3 * u01k(d.did, 14));
    let lawn = if town_centre(pl, d) { mixc(c_yard(moist, 0.5), srgb(86.0, 112.0, 54.0), 0.6) } else { c_yard(moist, 0.5) };
    let col = (rc * roofs + c_asphalt() * paved + lawn * green) * 0.88;
    let hm = [30.0 * smoothstep(1e5, 1.5e7, pl.pop) + 12.0, 12.0, 8.0, 4.0, 2.5][zi] * if pl.level >= 2 { 1.0 } else { 0.5 };
    (col, hm)
}

/// The street-lamp light of a place at a sample: resolved lamp heads and pools along the
/// nearest street (thinned with their energy kept where they would merge), or their mean.
#[allow(clippy::too_many_arguments)]
pub fn lamps(pl: &Pl, o: &Ev, dens: f64, gsd: f64, fw: f64, block: f64, resolved: f64) -> DVec3 {
    // (round 13: sodium and amber less saturated: a garish orange night, night@type's saturation
    // 0.59 against v0.2.0's 0.31; every lamp colour of next likewise: sodium (1, .68, .40), amber (1, .84, .64))
    const LAMPS: [DVec3; 3] = [DVec3::new(1.0, 0.68, 0.40), DVec3::new(1.0, 0.84, 0.64), DVec3::new(0.86, 0.92, 1.0)];
    // sodium share by development (and culture), white LED in CBDs and on main streets
    let sodium = 0.25 + 0.5 * (1.0 - pl.dev) * u01k(pl.seed, 20);
    let pick = |u: f64, major: bool| -> DVec3 {
        if o.zone == zone::CBD || (major && u < 0.45) {
            LAMPS[2]
        } else if o.pattern == pat::INDUSTRIAL || u < sodium {
            LAMPS[0]
        } else if u < 0.88 {
            LAMPS[1]
        } else {
            LAMPS[2]
        }
    };
    let sp = o.lamp_sp;
    if sp <= 0.0 {
        return DVec3::ZERO;
    }
    let lamp_res = band(block, gsd) * resolved;
    let fwl = fw.max(0.5 * gsd);
    let m = (2.0 * fwl / sp).max(1.0).log2().ceil().exp2();
    let sp_eff = sp * m;
    let present = 0.5 + 0.45 * pl.dev.min(1.0);
    let mut e = DVec3::ZERO;
    if lamp_res > 0.0 && o.ds < o.sw + 20.0 + 3.0 * fwl {
        let k = (o.along / sp_eff).round() * m;
        let side = if (k as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
        let _ = side;
        let dp = o.ds - o.sw * 0.85;
        let da = o.along - k * sp;
        let d2 = dp * dp + da * da;
        let lh = hash2(o.sid ^ 0x1A3B, k as i64, 0);
        if u01k(lh, 1) < present {
            let sa = 0.28 * sp;
            let pool = 0.03 * (-(dp * dp) / (2.0 * 3.5 * 3.5) - (da * da) / (2.0 * sa * sa)).exp();
            let core = point_light(d2, 6.0, 0.4, fwl);
            e = pick(u01k(hash1(o.sid, 3), 1), o.major) * ((pool + core) * m * (0.75 + 0.5 * u01k(lh, 2)));
        }
    }
    let lamp_e = present * (std::f64::consts::TAU * (6.0 * 0.16 + 0.03 * 3.5 * 0.28 * sp));
    let zf = [1.6, 1.2, 1.0, 0.75, 0.45][o.zone as usize];
    let lamp_mean = lamp_e * 2.0 / (block * sp) * zf;
    let mc = if o.zone == zone::CBD {
        LAMPS[2]
    } else if o.pattern == pat::INDUSTRIAL {
        LAMPS[0]
    } else {
        mixc(LAMPS[1], LAMPS[0], sodium)
    };
    e * lamp_res + mc * lamp_mean * (1.0 - lamp_res) * smoothstep(0.1, 0.35, dens)
}
