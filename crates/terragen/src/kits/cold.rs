//! The cold kit (`docs/design/terrain-next.md` §5.4, `docs/next/kits/cold.md`): glaciers and ice
//! caps, periglacial ground, the lakes of the permafrost and of glaciated lowlands, sea ice and
//! frozen lakes, seasonal snow. The WGSL twin is `gpu/wgsl/kits/cold.wgsl` (same names, same
//! order of operations).
//!
//! | slot | layer |
//! |---|---|
//! | 1 zonal | [`zonal`]: the perennial snow mask (`Masks::snow`) of the kit's snow model: snow line and glaciers by the summer temperature |
//! | 3 azonal | [`azonal`]: thermokarst and kettle lakes, drained lake basins, pingos |
//! | 8 water | [`water`]: sea ice (floes, leads, pressure ridges, fast ice), frozen lakes |
//! | 9 seasonal | [`seasonal`]: glaciers (firn, bare ice, crevasses, ogives, medial moraines), lateral and terminal moraines, seasonal snow |
//!
//! Kernel `cold_polygons` (YAML): ice-wedge polygons and sorted circles on a 3D Worley lattice
//! (seamless, no local frame).
//!
//! **Climate.** The kit's snow and ice follow the temperature of the warmest month (snow line,
//! glaciers) and the temperature at the ecoregion's date (seasonal snow, frozen water): the
//! sample's annual mean (lapse-corrected to the sub-sample's ground) ± half the annual range
//! ([`climate`]). The date of an ecoregion is its style's season (spring … autumn), or a winter
//! date for a share of the ecoregions (from their site climate).

use crate::instances::{Family, InstSite, Instance};
use crate::kernels::{band_cov, KIn, KOut, KParams, KernelSpec, C, P};
use crate::kits::{Kit, ReliefIn};
use crate::noise::*;
use crate::stack::{hmode, slot, Layer, Stack};
use crate::surface::{mixc, srgb};
use glam::DVec3;

pub static KIT: Kit = Kit {
    name: "cold",
    biomes_yaml: include_str!("../../biomes/cold.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/cold.wgsl"),
    layers: &[(slot::ZONAL, zonal), (slot::AZONAL, azonal), (slot::WATER, water), (slot::SEASONAL, seasonal)],
    kernels: &[KernelSpec {
        name: "cold_polygons",
        kind: 112,
        params: POLY_P,
        size_slot: 0,
        dyn_size: None,
        reach: None,
        default_hmode: "add",
        eval: polygons,
        wgsl: "cold_polygons",
    }],
    relief: Some(relief),
    relief_wgsl: include_str!("../gpu/wgsl/kits/cold_relief.wgsl"),
    families: &[Family { name: "cold_drumlins", cell: 45_000.0, jitter: 0.8, reach: 14_000.0, relief: true, exists: drumlins_exist }],
    ..Kit::EMPTY
};

/// Land-cover v2 class ids (`registry::classes`; checked by a test).
mod cls {
    pub const LAKE: u8 = 2;
    pub const RIVER: u8 = 3;
    pub const TUNDRA: u8 = 15;
    pub const SEA_ICE: u8 = 26;
    pub const GLACIER: u8 = 27;
    pub const FROZEN_WATER: u8 = 28;
    pub const GRAVEL: u8 = 33;
    pub const MORAINE: u8 = 36;
    pub const MARSH: u8 = 59;
    pub const SEASONAL_SNOW: u8 = 110;
}

// ============================================================================= climate

/// The cold kit's climate at a sample.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColdClimate {
    /// annual mean temperature at the sub-sample's ground (°C)
    pub t: f64,
    /// temperature of the warmest month (°C)
    pub summer: f64,
    /// temperature at the ecoregion's date (°C)
    pub date: f64,
}

/// The date of an ecoregion (fraction of the year from 1 January, northern-hemisphere
/// convention: the coldest day is at 0.04): its season (0 spring … 1 autumn) mapped to mid May …
/// late September, the season of the world's crops and canopies. (A share of the ecoregions in
/// winter painted whole regions white in summer datasets — snowy taiga, frozen grey lakes —
/// with hard borders to their summer neighbours.)
pub(crate) fn year_date(season: f64) -> f64 {
    0.37 + 0.37 * season
}

/// The climate at a sample.
pub(crate) fn climate(s: &Stack) -> ColdClimate {
    let t = s.t;
    let lapse = s.world.cfg.climate.lapse_rate_c_per_km;
    let ts = t.temp - lapse * (s.l.ground.max(0.0) - t.ground.max(0.0)) * 0.001;
    // (the atlas' warmest − coldest month, interpolated: continuous)
    let half = 0.5 * t.temp_range;
    let d = year_date(s.bio.style.season);
    // (the date's effect fades to the ecoregion border over the ecotone, where both sides meet at
    // the annual mean: no patches of a winter and a spring ecoregion in the border mosaic)
    let date = ts - half * (std::f64::consts::TAU * (d - 0.04)).cos() * smoothstep(0.0, 10.0, s.bio.edge_km);
    ColdClimate { t: ts, summer: ts + 0.95 * half, date }
}

/// Glaciation of the land around (0..1: lowlands shaped by the last-glacial ice sheets; the
/// atlas).
pub(crate) fn glaciation(s: &Stack) -> f64 {
    s.t.glaciation
}

/// How much the ground faces the equator (m/m: the slope's component towards the equator;
/// negative on shady slopes).
pub(crate) fn sunny(s: &Stack) -> f64 {
    // (the gradient points uphill: a slope facing south has a positive north component)
    s.l.grad[1] * s.ctx.lat.signum()
}

// ============================================================================= snow and ice

/// The kit's snow model at a sample.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Ice {
    /// perennial-snow index: < 0 above the snow line (°C)
    pub p: f64,
    /// glacier index: the snow index lowered in valleys (ice flows down them)
    pub g: f64,
    /// glacier ice cover 0..1
    pub ice: f64,
    /// perennial snow and ice cover 0..1
    pub snow: f64,
    /// the snow noise (~[-1.75, 1.75])
    pub n: f64,
}

/// Seeds of the kit's noises.
const SEED_SNOW: u64 = 0xC01D_5E00;
const SEED_ICE: u64 = 0xC01D_1CE0;

/// The perennial snow and the glaciers: the summer temperature against the equilibrium line
/// (−2 °C dry … +3 °C wet), lower in hollows and below it in valleys (glacier tongues); ice
/// does not hold on slopes steeper than ~40° (nunataks, rock walls), snow on ones steeper than ~50°.
pub(crate) fn ice(s: &Stack, c: &ColdClimate) -> Ice {
    let t = s.t;
    let l = s.l;
    let tela = -2.0 + 5.0 * smoothstep(0.15, 0.85, t.moist);
    let base = c.summer - tela;
    // (the terms below lower the index by at most 2 + 2.6 + 1.2 · 1.75 + the aspect's)
    if base > 8.5 {
        return Ice { p: base, g: base, ..Default::default() };
    }
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let hollow = smoothstep(0.1, 0.8, -t.gully);
    let spur = smoothstep(0.2, 0.8, t.gully);
    let mut n = s.pf.snow;
    let b60 = band(60.0, gsd);
    if b60 > 0.0 {
        n += 0.5 * perlin3(SEED_SNOW, p / 60.0) * b60;
        let b18 = band(18.0, gsd);
        if b18 > 0.0 {
            n += 0.25 * perlin3(SEED_SNOW + 1, p / 18.0) * b18;
        }
    }
    // (shady slopes keep their snow: cirques face the pole)
    let pi = base - 2.0 * hollow + 0.8 * spur + 1.2 * n + 1.5 * sunny(s).clamp(-1.0, 1.0);
    let rv = if l.river_hw > 0.0 { 1.0 - smoothstep(l.river_hw + 80.0, l.river_hw + 500.0, l.river_d.abs()) } else { 0.0 };
    let valley = rv.max(0.6 * hollow);
    let g = pi - 2.6 * valley;
    let steep = 1.0 - smoothstep(0.45, 0.75, l.slope + 0.1 * n + 0.15 * spur);
    let ice = smoothstep(0.15, -0.15, g + 0.4) * steep;
    // (snow holds on slopes to ~45°: the cut from 35° left half of the high ranges grey rock
    // where v0.2.0's alpine scenes are white)
    let snow = smoothstep(0.15, -0.15, pi) * (1.0 - 0.85 * smoothstep(1.0, 1.7, l.slope));
    Ice { p: pi, g, ice, snow: snow.max(ice), n }
}

/// Slot 1: the perennial snow mask of the kit's model replaces the core's (natural land use
/// ends at it; the core's snow layer paints it).
pub fn zonal(s: &mut Stack) {
    let c = climate(s);
    let g = ice(s, &c);
    s.m.snow = g.snow;
}

// ============================================================================= lakes

/// A lattice of analytic lakes: ellipsoids at the sites of a jittered 3D lattice, sliced by the
/// surface (a site's distance from the surface sets its lake's size there), elongated along a
/// prevailing wind (per site, from its own local frame: no direction field). Kinds: open lake,
/// drained basin (sedge meadow), drained basin with a pingo.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LakeLat {
    pub seed: u64,
    pub cell: f64,
    pub r0: f64,
    pub r1: f64,
    pub dens: f64,
    /// share of open lakes; then drained basins; the rest have a pingo
    pub p_lake: f64,
    pub p_basin: f64,
}

pub(crate) const LAKE_LATS: [LakeLat; 2] = [
    LakeLat { seed: 0xC01D_1A4E, cell: 900.0, r0: 100.0, r1: 400.0, dens: 0.85, p_lake: 0.72, p_basin: 0.22 },
    LakeLat { seed: 0xC01D_1A4F, cell: 260.0, r0: 25.0, r1: 115.0, dens: 0.9, p_lake: 0.66, p_basin: 0.28 },
];

/// Largest elongation and shore noise of a lake (its reach: `r1 · LAKE_EL · LAKE_WN`).
const LAKE_EL: f64 = 1.85;
const LAKE_WN: f64 = 1.26;

/// The lake of a lattice at a point.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LakeHit {
    /// coverage 0..1 (the shore antialiased)
    pub cov: f64,
    /// 0 at the centre … 1 at the shore
    pub rn: f64,
    /// 0 lake, 1 basin, 2 basin with a pingo
    pub kind: u32,
    pub h: u64,
    /// minor semi-axis of the lake's outline at the point (m)
    pub b: f64,
}

/// The lake of `lat` at `p` (lakes scaled by `sc`; `fw` the filter width).
pub(crate) fn lake_at(lat: &LakeLat, p: DVec3, sc: f64, fw: f64) -> LakeHit {
    let q = p / lat.cell;
    let qf = q.floor();
    let fq = q - qf;
    let (ix, iy, iz) = (qf.x as i64, qf.y as i64, qf.z as i64);
    let reach = (lat.r1 * LAKE_EL * LAKE_WN * sc + fw) / lat.cell;
    let mut best = LakeHit::default();
    for dz in -1..=1i64 {
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (gx, gy, gz) = (nb_gap(fq.x, dx, 0.4), nb_gap(fq.y, dy, 0.4), nb_gap(fq.z, dz, 0.4));
                if gx * gx + gy * gy + gz * gz > reach * reach {
                    continue;
                }
                let (cx, cy, cz) = (ix + dx, iy + dy, iz + dz);
                let h = hash3(lat.seed, cx, cy, cz);
                if u01k(h, 4) >= lat.dens {
                    continue;
                }
                let c = DVec3::new(
                    cx as f64 + 0.5 + 0.8 * (u01k(h, 1) - 0.5),
                    cy as f64 + 0.5 + 0.8 * (u01k(h, 2) - 0.5),
                    cz as f64 + 0.5 + 0.8 * (u01k(h, 3) - 0.5),
                ) * lat.cell;
                let rel = p - c;
                let r = lat.r0 + (lat.r1 - lat.r0) * u01k(h, 5);
                let el = 1.25 + 0.6 * u01k(h, 6);
                let lim = r * el * LAKE_WN * sc + fw;
                if rel.length_squared() > lim * lim {
                    continue;
                }
                // the site's frame: its up, east, north; the long axis ~25° north of east
                let up = c.normalize();
                let e = if up.z.abs() > 0.999 { DVec3::X } else { DVec3::Z.cross(up).normalize() };
                let nn = up.cross(e);
                let th = 0.45 + 0.35 * (u01k(h, 7) - 0.5);
                let (sn, cs) = th.sin_cos();
                let ax = e * cs + nn * sn;
                let ay = nn * cs - e * sn;
                let (a, b) = (r * el, r / el);
                let x = rel.dot(ax) / a;
                let y = rel.dot(ay) / b;
                let z = rel.dot(up) / r;
                let rho = (x * x + y * y + z * z).sqrt();
                if rho > LAKE_WN * sc + fw / b {
                    continue;
                }
                // ragged shore (bays and points): noise in the lake's own frame
                let se = sc * (1.0 + 0.24 * perlin3(h, DVec3::new(2.6 * x, 2.6 * y, 0.37)));
                // distance to the shore (m): the index over its gradient in the plane
                let gr = ((x / a) * (x / a) + (y / b) * (y / b)).sqrt() / rho.max(1e-6);
                let dist = (se - rho) / gr.max(1e-9);
                let cov = (dist / fw + 0.5).clamp(0.0, 1.0);
                if cov > best.cov {
                    let u = u01k(h, 8);
                    let kind = if u < lat.p_lake {
                        0
                    } else if u < lat.p_lake + lat.p_basin {
                        1
                    } else {
                        2
                    };
                    best = LakeHit { cov, rn: (rho / se.max(1e-6)).min(1.0), kind, h, b: b * se };
                }
            }
        }
    }
    best
}

/// The mean coverage of a lattice's lakes (scaled by `sc`): the expected slice area of its
/// ellipsoids per area, as the coverage of a Poisson field.
pub(crate) fn lake_mean_cov(lat: &LakeLat, sc: f64) -> f64 {
    let er3 = (lat.r1.powi(4) - lat.r0.powi(4)) / (4.0 * (lat.r1 - lat.r0));
    let lam = lat.dens * (4.0 / 3.0) * std::f64::consts::PI * er3 * sc * sc * sc / (lat.cell * lat.cell * lat.cell);
    1.0 - (-lam).exp()
}

/// Where lakes form (0..1): permafrost lowlands (thermokarst) and glaciated lowlands (kettles),
/// flat, low, not too dry; lake-rich and lake-poor regions.
pub(crate) fn lake_mask(s: &Stack, c: &ColdClimate) -> f64 {
    let t = s.t;
    let l = s.l;
    let thermo = smoothstep(-1.0, -4.0, c.t);
    let kettle = glaciation(s) * smoothstep(9.0, 4.0, c.t);
    let m = thermo.max(0.8 * kettle);
    if m <= 0.0 {
        return 0.0;
    }
    m * (1.0 - smoothstep(0.03, 0.07, l.slope))
        * (1.0 - smoothstep(350.0, 700.0, l.ground))
        * (1.0 - smoothstep(0.2, 0.5, t.mountain))
        * smoothstep(0.02, 0.12, t.moist)
        * (1.0 - smoothstep(0.1, 0.3, t.sand))
        * (0.3 + 0.7 * smoothstep(0.25, 0.65, t.style[0]))
}

/// Lake colours (linear): dark, tea-coloured, silty; the shallow margin; the drained basins;
/// lake ice.
fn lake_colour(h: u64, rn: f64) -> DVec3 {
    let u = u01k(h, 9);
    let deep = if u < 0.55 {
        srgb(16.0, 28.0, 42.0)
    } else if u < 0.8 {
        srgb(36.0, 38.0, 28.0)
    } else {
        srgb(58.0, 88.0, 94.0)
    };
    let shallow = mixc(deep, srgb(98.0, 104.0, 86.0), 0.5);
    mixc(deep, shallow, smoothstep(0.62, 0.97, rn))
}

/// The mean colour of open lakes and of the frozen ones, of basins (for coarse pixels).
fn lake_mean_albedo(lat: &LakeLat, frozen: bool) -> DVec3 {
    let lake = if frozen {
        mixc(srgb(150.0, 170.0, 184.0), srgb(228.0, 233.0, 239.0), 0.6)
    } else {
        let w = |c: DVec3| mixc(c, mixc(c, srgb(98.0, 104.0, 86.0), 0.5), 0.3);
        w(srgb(16.0, 28.0, 42.0)) * 0.55 + w(srgb(36.0, 38.0, 28.0)) * 0.25 + w(srgb(58.0, 88.0, 94.0)) * 0.2
    };
    let basin = mixc(srgb(104.0, 126.0, 60.0), srgb(82.0, 100.0, 58.0), 0.3);
    mixc(basin, lake, lat.p_lake)
}

/// Slot 3: the lakes of permafrost and glaciated lowlands, drained basins and pingos.
pub fn azonal(s: &mut Stack) {
    drumlin_look(s);
    let c = climate(s);
    let lm = lake_mask(s, &c);
    if lm < 0.03 {
        return;
    }
    let sc = lm.cbrt();
    let frozen = c.date < -1.5;
    for lat in &LAKE_LATS {
        lakes(s, lat, sc, frozen);
        if s.done {
            return;
        }
    }
}

/// One lattice's lakes at the sample: explicit where resolved, the mean colour where not.
fn lakes(s: &mut Stack, lat: &LakeLat, sc: f64, frozen: bool) {
    let gsd = s.ctx.gsd;
    let p = s.ctx.p;
    let fw = s.l.fw;
    let e = smoothstep(1.2 * gsd, 3.0 * gsd, 0.5 * (lat.r0 + lat.r1));
    let mc = lake_mean_cov(lat, sc);
    let hit = if e > 0.0 { lake_at(lat, p, sc, fw) } else { LakeHit::default() };
    let xc = e * hit.cov;
    let mw = (1.0 - e) * mc;
    let cov = xc + mw;
    if cov <= 0.0 {
        return;
    }
    let mut ly = Layer { cov, ..Default::default() };
    let mut xa = DVec3::ZERO;
    if xc > 0.0 {
        if hit.kind == 0 {
            if frozen {
                // snow-covered lake ice, clear dark ice where the wind swept it
                let sw = smoothstep(-0.15, 0.25, perlin3(hit.h ^ 0x1CE, p / 40.0) * band(40.0, gsd) + 0.2 - 0.5 * hit.rn);
                xa = mixc(srgb(228.0, 233.0, 239.0), srgb(150.0, 170.0, 184.0), sw);
                ly.cls = cls::FROZEN_WATER;
                ly.hmode = hmode::BLEND;
                ly.dh = -0.3;
            } else {
                xa = lake_colour(hit.h, hit.rn) * (1.0 + 0.05 * perlin3(hit.h ^ 0x5ED, p / 70.0) * band(70.0, gsd));
                ly.cls = cls::LAKE;
                ly.hmode = hmode::ABS;
                ly.dh = s.l.ground - 0.4;
                ly.relit = 1.0;
                ly.water = true;
            }
        } else {
            // drained basin: sedge meadow, wetter towards the centre
            xa = mixc(srgb(104.0, 126.0, 60.0), srgb(82.0, 100.0, 58.0), smoothstep(0.55, 0.0, hit.rn))
                * (1.0 + 0.08 * perlin3(hit.h ^ 0xBA5, p / 9.0) * band(9.0, gsd));
            ly.cls = cls::MARSH;
            ly.hmode = hmode::BLEND;
            ly.dh = -1.2;
            if hit.kind == 2 {
                // a pingo: an ice-cored hill in the basin's centre, some with a collapsed summit
                let rp = 0.28 * hit.b;
                let pr = hit.rn / 0.28;
                let pc = ((1.0 - pr) * rp / fw + 0.5).clamp(0.0, 1.0);
                if pc > 0.0 {
                    let hh = (0.35 * rp).clamp(4.0, 35.0);
                    let mut dh = hh * (1.0 - pr * pr).max(0.0).powf(0.7);
                    if u01k(hit.h, 10) < 0.35 && pr < 0.3 {
                        let xr = pr / 0.3;
                        dh -= 0.4 * hh * (1.0 - xr * xr);
                    }
                    let pa = mixc(srgb(122.0, 118.0, 78.0), srgb(150.0, 134.0, 104.0), smoothstep(0.45, 0.0, pr));
                    xa = mixc(xa, pa, pc);
                    ly.dh = lerp(ly.dh, dh, pc);
                    if pc > 0.5 {
                        ly.cls = cls::TUNDRA;
                    }
                }
            }
        }
    }
    // explicit and mean weighted by their coverage; class and water from a dominant explicit part
    ly.albedo = (xa * xc + lake_mean_albedo(lat, frozen) * mw) / cov;
    if xc < mw || xc < 0.5 {
        ly.cls = 0;
        ly.water = false;
        if ly.hmode == hmode::ABS {
            ly.hmode = hmode::NONE;
        }
    }
    ly.clear = 1.0;
    let water = ly.water;
    s.composite(ly);
    // an open lake ends the stack (no vegetation, rivers or snow on it)
    if water {
        s.done = true;
    }
}

// ============================================================================= drumlins

/// The index of the drumlin-field family (`FAM_COLD_DRUMLINS`).
fn fam_drumlins() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| crate::instances::families().position(|f| f.name == "cold_drumlins").unwrap())
}

/// A drumlin field on 70 % of the sites in glaciated lowlands (the atlas' last-glacial ice):
/// radius 9–14 km, ice-flow azimuth, drumlin height 8–22 m, spacing scale, esker phase.
fn drumlins_exist(w: &crate::world::World, s: &InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 1) >= 0.7 {
        return None;
    }
    let a = w.atlas().sample(s.center.normalize());
    if a.glaciation < 0.5 || a.coast_km < 5.0 || a.elevation_m > 700.0 {
        return None;
    }
    let f = |x: f64| x as f32;
    Some([
        f(9000.0 + 5000.0 * u01k(s.id, 2)),
        f(std::f64::consts::TAU * u01k(s.id, 3)),
        f(8.0 + 14.0 * u01k(s.id, 4)),
        f(0.7 + 0.6 * u01k(s.id, 5)),
        f(u01k(s.id, 6)),
        0.0,
        0.0,
        0.0,
    ])
}

/// Mean height of unresolved drumlins per unit height: density × mean area share × mean of the
/// profile (1 − r²)² (1/3), the stoss side shortened.
const DRUMLIN_MEAN: f64 = 0.109;

/// Height (m), drumlin crest (0..1) and esker (0..1) of the drumlin fields at `p`: streamlined
/// hills on a jittered lattice in the field's ice-flow frame (steep stoss side up-ice), a
/// sinuous, broken esker ridge along the flow; band-limited (unresolved drumlins: their mean).
pub(crate) fn drumlin_field(inst: &[Instance], p: DVec3, gsd: f64) -> (f64, f64, f64) {
    let (mut z, mut crest, mut esk) = (0.0, 0.0f64, 0.0f64);
    for it in inst {
        let v = it.v.map(|x| x as f64);
        let r = v[0];
        let rel = p - it.center;
        let d2 = rel.length_squared();
        if d2 >= r * r {
            continue;
        }
        let up = it.center.normalize();
        let e = if up.z.abs() > 0.999 { DVec3::X } else { DVec3::Z.cross(up).normalize() };
        let n = up.cross(e);
        let (sn, cs) = v[1].sin_cos();
        let u = rel.dot(e * cs + n * sn);
        let wq = rel.dot(n * cs - e * sn);
        let fade = smoothstep(r, 0.6 * r, d2.sqrt());
        let (cu, cw) = (900.0 * v[3], 320.0 * v[3]);
        let bnd = band(0.35 * cw, gsd);
        let mut zd: f64 = 0.0;
        if bnd > 0.0 {
            let (iu, iw) = ((u / cu).floor() as i64, (wq / cw).floor() as i64);
            for du in -1..=1i64 {
                for dw in -1..=1i64 {
                    let (gu, gw) = (iu + du, iw + dw);
                    let h = hash2(it.id, gu, gw);
                    if u01k(h, 1) >= 0.85 {
                        continue;
                    }
                    let c0 = (gu as f64 + 0.5 + 0.6 * (u01k(h, 2) - 0.5)) * cu;
                    let c1 = (gw as f64 + 0.5 + 0.6 * (u01k(h, 3) - 0.5)) * cw;
                    let a = cu * (0.33 + 0.12 * u01k(h, 4));
                    let b = cw * (0.3 + 0.1 * u01k(h, 5));
                    let mut x = (u - c0) / a;
                    if x < 0.0 {
                        x *= 1.25;
                    }
                    let y = (wq - c1) / b;
                    let r2 = x * x + y * y;
                    if r2 < 1.0 {
                        zd = zd.max((1.0 - r2) * (1.0 - r2) * (0.7 + 0.6 * u01k(h, 6)));
                    }
                }
            }
        }
        z += v[2] * (zd * bnd + DRUMLIN_MEAN * (1.0 - bnd)) * fade;
        crest = crest.max(zd * bnd * fade);
        let we = 40.0;
        let be = band(2.0 * we, gsd);
        if be > 0.0 {
            let ph = v[4];
            let cl = 0.5 * r * (ph - 0.5) + 600.0 * (u / 2300.0 + ph * 6.3).sin() + 250.0 * (u / 700.0 + ph * 17.0).sin();
            let dd = (wq - cl).abs();
            if dd < we {
                let on = smoothstep(-0.2, 0.0, (u / 3100.0 + ph * 11.0).sin());
                let x = dd / we;
                let ze = (1.0 - x * x) * on * be * fade;
                z += 12.0 * ze;
                esk = esk.max(ze);
            }
        }
    }
    (z, crest, esk)
}

/// Pass A: the drumlin fields (the drainage follows them).
fn relief(w: &crate::world::World, r: &ReliefIn, h: &mut f64) {
    let inst = r.instances(w, fam_drumlins());
    if inst.is_empty() {
        return;
    }
    *h += drumlin_field(&inst, r.ctx.p, r.ctx.gsd).0;
}

/// Pass B on drumlin fields: drier, paler crests; eskers of pale gravel.
fn drumlin_look(s: &mut Stack) {
    let inst = s.instances(fam_drumlins());
    if inst.is_empty() {
        return;
    }
    let (_, crest, esk) = drumlin_field(&inst, s.ctx.p, s.ctx.gsd);
    if crest > 0.05 {
        let c = s.col * DVec3::new(1.12, 1.08, 0.94);
        s.composite(Layer::paint(0.6 * smoothstep(0.05, 0.6, crest), c));
    }
    if esk > 0.0 {
        let g = mixc(s.pal(crate::registry::pal::SAND), srgb(150.0, 142.0, 124.0), 0.5);
        s.composite(Layer { cov: smoothstep(0.1, 0.4, esk), albedo: g, cls: cls::GRAVEL, ..Default::default() });
    }
}

// ============================================================================= water

const SEED_FLOE: u64 = 0xC01D_F10E;

/// Slot 8: sea ice on cold oceans (floes, leads, pressure ridges, fast ice along the coast) and
/// frozen lakes, by the temperature at the ecoregion's date.
pub fn water(s: &mut Stack) {
    let l = s.l;
    if l.water <= l.ground {
        rivers(s);
        return;
    }
    let c = climate(s);
    if c.date > -1.0 && c.summer > -1.0 {
        return;
    }
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let fw = l.fw;
    let depth = l.water - l.ground;
    if l.water_kind == crate::world::water::OCEAN {
        // concentration: forming below −2 °C, pack ice below ~−8 °C, perennial where the summer
        // stays below freezing
        let con = smoothstep(-1.5, -8.0, c.date).max(smoothstep(0.0, -2.0, c.summer));
        if con <= 0.02 {
            return;
        }
        let fast = smoothstep(14.0, 6.0, depth) * smoothstep(0.3, 0.6, con);
        let snowc = srgb(232.0, 237.0, 243.0);
        // coarse pixels: the mean
        let e = smoothstep(1.2 * gsd, 3.0 * gsd, 300.0);
        let mcov = con * (0.15 + 0.85 * con) * 0.93;
        let mcov = mcov + (1.0 - mcov) * fast;
        let mut cov = (1.0 - e) * mcov;
        let mut col = snowc * 0.97 * cov;
        let mut dh = 0.4;
        if e > 0.0 {
            // big floes (W3, ~1.6 km), their borders leads (open or refrozen) or pressure ridges
            let big = worley3(SEED_FLOE, p, 1600.0, 0.9);
            let eb = worley_edge_dist(&big, p);
            let pair = u01k(big.id ^ big.id2, 3);
            let lead = pair < 0.35 + 0.6 * (1.0 - con);
            let lead_w = if lead { 12.0 + 160.0 * (1.0 - con) * (1.0 - con) + 30.0 * u01k(big.id ^ big.id2, 4) } else { 0.0 };
            let mut ic = if u01k(big.id, 1) < 0.15 + 0.85 * con { ((eb - lead_w) / fw + 0.5).clamp(0.0, 1.0) } else { 0.0 };
            // the marginal ice zone: some floes broken into small ones
            let mz = smoothstep(0.9, 0.5, con);
            if mz > 0.0 && u01k(big.id, 2) < mz {
                let sm = worley3(SEED_FLOE + 1, p, 110.0, 0.9);
                let es = worley_edge_dist(&sm, p);
                let ws = 2.0 + 12.0 * (1.0 - con);
                ic *= if u01k(sm.id, 1) < con { ((es - ws) / fw + 0.5).clamp(0.0, 1.0) } else { 0.0 };
            }
            ic = ic + (1.0 - ic) * fast;
            // surface: wind-packed snow, pressure ridges (bright, raised) on closed borders
            let mut xa = snowc * (1.0 + 0.04 * perlin3(SEED_FLOE + 2, p / 30.0) * band(30.0, gsd));
            let mut xdh = 0.4;
            if !lead {
                let rg = band_cov(eb, 3.0, fw) * (1.0 - fast);
                xa = mixc(xa, srgb(246.0, 249.0, 252.0), rg);
                xdh += 2.0 * rg;
            }
            // refrozen leads: grey new ice in the deep cold
            let nilas = smoothstep(-6.0, -12.0, c.date) * (1.0 - ic);
            if nilas > 0.0 {
                xa = mixc(xa, srgb(92.0, 102.0, 110.0), nilas / (ic + nilas));
                ic += nilas;
            }
            let xc = e * ic;
            cov += xc;
            col += xa * xc;
            dh = xdh;
        }
        if cov <= 0.0 {
            return;
        }
        let albedo = col / cov;
        let mut ly = Layer { cov, albedo, dh: l.water + dh, hmode: hmode::ABS, relit: 1.0, ..Default::default() };
        if cov > 0.5 {
            ly.cls = cls::SEA_ICE;
            ly.solid = true;
        }
        s.composite(ly);
    } else if c.date < -1.5 {
        // frozen lakes: snow on the ice, cracks, darker clear ice where the wind swept it
        let mut col = srgb(226.0, 231.0, 237.0);
        let sw = smoothstep(0.1, 0.5, perlin3(SEED_ICE, p / 90.0) * band(90.0, gsd) + 0.25 * perlin3(SEED_ICE + 1, p / 25.0) * band(25.0, gsd));
        col = mixc(col, srgb(146.0, 166.0, 182.0), 0.8 * sw);
        let e = smoothstep(1.2 * gsd, 3.0 * gsd, 4.0);
        if e > 0.0 {
            let ck = worley3(SEED_ICE + 2, p, 70.0, 0.9);
            let cr = band_cov(worley_edge_dist(&ck, p), 0.35, fw) * e * (0.4 + 0.6 * sw);
            col = mixc(col, srgb(120.0, 138.0, 150.0), cr);
        }
        s.composite(Layer {
            cov: 1.0,
            albedo: col,
            dh: l.water + 0.2,
            hmode: hmode::ABS,
            cls: cls::FROZEN_WATER,
            relit: 1.0,
            solid: true,
            ..Default::default()
        });
    }
}

/// Rivers by the season rather than the annual mean (the core freezes them below about
/// −2.5 °C): ice at a winter date (frozen water, snow on it later), open water in summer.
fn rivers(s: &mut Stack) {
    let l = s.l;
    let t = s.t;
    if l.river_hw <= 0.0 || t.river_wet <= 0.5 {
        return;
    }
    let gsd = s.ctx.gsd;
    let cov = band_cov(l.river_d, l.river_hw, l.fw.max(gsd * 0.35));
    if cov <= 0.0 {
        return;
    }
    let c = climate(s);
    let detail = s.pf.detail;
    let core_ice = smoothstep(-1.5, -4.0, t.temp + 1.5 * s.pf.snow) > 0.5;
    if c.date < -1.5 {
        // snow-covered river ice with darker, wind-swept clear ice
        let col = mixc(s.sm.pal.snow * 0.92, srgb(150.0, 172.0, 186.0), 0.3 + 0.25 * detail);
        s.m.river_cov = 0.4 * cov;
        s.composite(Layer { cov, albedo: col, dh: l.river_level, hmode: hmode::ABS, cls: cls::FROZEN_WATER, relit: 1.0, solid: true, ..Default::default() });
    } else if core_ice && c.date > 0.5 && s.m.snow < 0.3 {
        // the summer thaw: open water (the core's colours without the ice)
        let pal = &s.sm.pal;
        let mut wcol = mixc(pal.river, pal.lake_deep, smoothstep(30.0, 200.0, l.river_hw * 2.0));
        wcol = mixc(wcol, srgb(96.0, 138.0, 140.0), 0.7 * t.mountain * smoothstep(8.0, 0.0, t.temp));
        s.m.river_cov = cov;
        s.composite(Layer { cov, albedo: wcol, dh: l.river_level, hmode: hmode::ABS, cls: cls::RIVER, relit: 1.0, water: true, ..Default::default() });
    }
}

// ============================================================================= seasonal

/// Slot 9: glaciers, their moraines, seasonal snow.
pub fn seasonal(s: &mut Stack) {
    let c = climate(s);
    let g = ice(s, &c);
    if g.ice > 0.0 {
        glacier(s, &c, &g);
    }
    if g.p > 0.0 && g.g < 0.6 {
        moraine(s, &g);
    }
    snow(s, &c, &g);
}

/// Glacier ice: firn above the equilibrium line, bare blue-grey ice below it with dirt bands
/// (ogives), crevasses across the flow (along the contours) where the ice is steep, medial
/// moraines along the valley's axis, blue ice on cold, wind-scoured ice caps.
fn glacier(s: &mut Stack, c: &ColdClimate, g: &Ice) {
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let l = s.l;
    let fw = l.fw;
    let slope = l.slope;
    let _ = c;
    let abl = smoothstep(-1.6, -0.4, g.p);
    // firn: wind ripples (sastrugi) and drifts
    let tex = 0.025 * perlin3(SEED_ICE + 3, p / 7.0) * band(7.0, gsd)
        + 0.02 * perlin3(SEED_ICE + 4, p / 2.5) * band(2.5, gsd)
        + 0.03 * perlin3(SEED_ICE + 5, p / 120.0) * band(120.0, gsd);
    let firn = srgb(236.0, 241.0, 247.0) * (1.0 + tex);
    // bare ice: blue-grey, dirtier towards the margin and the snout
    let dirt = smoothstep(-0.2, 0.5, perlin3(SEED_ICE + 6, p / 150.0) * band(150.0, gsd) + 0.6 * smoothstep(-1.4, -0.4, g.g));
    let bare = mixc(srgb(158.0, 190.0, 212.0), srgb(128.0, 128.0, 124.0), 0.6 * dirt) * (1.0 + 1.5 * tex);
    let mut col = mixc(firn, bare, abl);
    let mut dh = 0.0;
    // blue ice on cold flat ice caps
    if g.p < -4.0 && slope < 0.08 {
        let bi = smoothstep(0.32, 0.36, perlin3(SEED_ICE + 7, p / 900.0) + 0.3 * perlin3(SEED_ICE + 8, p / 300.0) * band(300.0, gsd))
            * smoothstep(-4.0, -6.0, g.p)
            * (1.0 - smoothstep(0.04, 0.08, slope));
        col = mixc(col, srgb(150.0, 186.0, 212.0), bi);
    }
    // ogives: arcs of darker ice below icefalls (bands along the contours, 35 m in height)
    let sl = slope.max(0.03);
    if abl > 0.0 {
        let og_w = 35.0 / sl;
        let og = (0.5 + 0.5 * (std::f64::consts::TAU * l.ground / 35.0).cos()) * band(og_w, gsd) * smoothstep(0.08, 0.2, slope);
        col *= 1.0 - 0.08 * og * abl;
    }
    // crevasses: across the flow, i.e. along the contours of the ground, every 6 m of height,
    // broken (en echelon), where the ice is steep
    let cs = smoothstep(0.12, 0.35, slope) * (0.45 + 0.55 * abl);
    if cs > 0.0 {
        let spacing = 6.0 / sl;
        let e = smoothstep(1.2 * gsd, 3.0 * gsd, spacing);
        let on = smoothstep(-0.05, 0.05, perlin3(SEED_ICE + 9, p / 20.0) + 0.55 * cs - 0.35);
        let mut cr = 0.0;
        if e > 0.0 {
            let x = l.ground / 6.0 + 0.6 * perlin3(SEED_ICE + 10, p / 45.0);
            let f = x - x.floor();
            let d = f.min(1.0 - f) * spacing;
            cr = band_cov(d, 1.2, fw) * on * e;
        }
        // the mean of unresolved crevasses: their area share
        cr += (1.0 - e) * on * (2.4 / spacing).min(1.0);
        let cr = cr * cs;
        col = mixc(col, srgb(34.0, 62.0, 88.0), cr);
        dh -= 4.0 * cr;
    }
    // medial moraines along the valley's axis (the drainage line under the ice) and beside it
    if abl > 0.2 && l.river_hw > 0.0 {
        let ad = l.river_d.abs();
        let w = 12.0 + 8.0 * perlin3(SEED_ICE + 11, p / 80.0) + 3.0 * perlin3(SEED_ICE + 14, p / 9.0) * band(9.0, gsd);
        let off = 110.0 + 0.6 * l.river_hw;
        let mm = band_cov(ad, w, fw.max(0.6 * w)).max(0.7 * band_cov(ad - off, 0.7 * w, fw.max(0.4 * w))) * smoothstep(0.2, 0.6, abl);
        col = mixc(col, srgb(104.0, 98.0, 92.0), 0.85 * mm);
        dh += 1.5 * mm;
    }
    let ly = Layer { cov: g.ice, albedo: col, dh: s.m.micro + dh, hmode: hmode::BLEND, cls: cls::GLACIER, relit: 1.0, clear: 1.0, ..Default::default() };
    s.composite(ly);
}

/// Lateral and terminal moraines: a ridge of grey-brown debris around the glacier tongues
/// (where the ice reaches below the snow line).
fn moraine(s: &mut Stack, g: &Ice) {
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let x = g.g + 0.4;
    let mo = smoothstep(-0.05, 0.05, x) * smoothstep(0.6, 0.5, x + 0.15 * g.n) * (1.0 - smoothstep(0.5, 0.75, s.l.slope));
    if mo <= 0.0 {
        return;
    }
    let boulders = 0.12 * perlin3(SEED_ICE + 12, p / 4.0) * band(4.0, gsd) + 0.08 * perlin3(SEED_ICE + 13, p / 1.3) * band(1.3, gsd);
    let rock = s.pal(crate::registry::pal::ROCK);
    let col = mixc(rock, srgb(120.0, 114.0, 106.0), 0.5) * (0.9 + boulders);
    let ridge = (std::f64::consts::PI * (x / 0.55).clamp(0.0, 1.0)).sin();
    s.composite(Layer { cov: mo, albedo: col, dh: s.m.micro + 3.0 * ridge, hmode: hmode::BLEND, cls: cls::MORAINE, clear: 1.0, ..Default::default() });
}

/// Seasonal snow at the ecoregion's date: ragged edges, lingering in hollows, shed by steep
/// slopes; tree crowns stay dark, roads are ploughed, towns grey.
fn snow(s: &mut Stack, c: &ColdClimate, g: &Ice) {
    if c.date > 2.5 || s.class == cls::FROZEN_WATER {
        return;
    }
    let t = s.t;
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let hollow = smoothstep(0.1, 0.8, -t.gully);
    let spur = smoothstep(0.2, 0.8, t.gully);
    let mut n = 0.6 * s.pf.snow - 1.6 * hollow + 0.5 * spur;
    let b45 = band(45.0, gsd);
    if b45 > 0.0 {
        n += 0.5 * perlin3(SEED_SNOW + 2, p / 45.0) * b45;
        let b12 = band(12.0, gsd);
        if b12 > 0.0 {
            n += 0.25 * perlin3(SEED_SNOW + 3, p / 12.0) * b12;
        }
    }
    // (slopes facing the equator lose their snow first)
    let mut x = c.date + n + 2.5 * sunny(s).clamp(-1.0, 1.0);
    // late melt: rims and centres of the frost polygons bare first, the troughs keep their snow
    // (a net of snow), the edges ragged with fine noise
    if x > -3.2 && x < 1.0 {
        let b4 = band(4.0, gsd);
        if b4 > 0.0 {
            x += 0.8 * perlin3(SEED_SNOW + 4, p / 4.0) * b4 + 0.4 * perlin3(SEED_SNOW + 5, p / 1.3) * band(1.3, gsd);
        }
        let b18 = band(18.0, gsd);
        if b18 > 0.0 {
            let wc = worley3(SEED_SNOW + 6, p, 18.0, 0.9);
            let ed = worley_edge_dist(&wc, p);
            x += 1.2 * (u01k(wc.id, 1) - 0.5) * b18 - 1.6 * (1.0 - smoothstep(0.0, 1.8, ed)) * b18;
        }
    }
    let mut cov = smoothstep(-0.55, -1.25, x) * (1.0 - smoothstep(0.75, 1.3, s.l.slope));
    if cov <= 0.0 {
        return;
    }
    // crowns (tall above the ground) keep a little snow; ploughed roads; grey towns
    let crown = smoothstep(1.5, 4.0, s.height - s.l.ground);
    cov *= 1.0 - 0.7 * crown;
    cov *= 1.0 - 0.85 * s.m.road_cov;
    cov *= 1.0 - 0.55 * s.m.town_cov;
    cov *= 1.0 - s.m.river_cov;
    cov *= 1.0 - g.snow;
    if cov <= 0.0 {
        return;
    }
    let col = s.sm.pal.snow * (1.0 + 0.03 * s.pf.detail) * DVec3::new(1.0 - 0.03 * hollow, 1.0 - 0.015 * hollow, 1.0);
    s.composite(Layer::paint_cls(cov, col, cls::SEASONAL_SNOW));
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ============================================================================= kernel

const POLY_P: &[P] = &[
    P::Num("cell", 0, 1.0, 500.0, 18.0),
    P::Num("jitter", 1, 0.0, 1.0, 0.9),
    P::Num("trough", 2, 0.0, 50.0, 1.6),
    P::Num("trough_depth", 3, -10.0, 10.0, 0.5),
    P::Num("rim", 4, 0.0, 50.0, 1.2),
    P::Num("rim_height", 5, -10.0, 10.0, 0.3),
    P::Num("pond", 6, 0.0, 1.0, 0.12),
    P::Num("high", 7, 0.0, 1.0, 0.3),
    P::Num("colour_var", 8, 0.0, 1.0, 0.25),
    P::Num("pond_size", 9, 0.05, 0.9, 0.45),
    P::Col("colour", 0, C::N("tundra")),
    P::Col("trough_colour", 1, C::Rgb([66.0, 76.0, 50.0])),
    P::Col("pond_colour", 2, C::Rgb([28.0, 42.0, 52.0])),
];

/// `cold_polygons`: ice-wedge polygons (or sorted circles) on a 3D Worley lattice at the ECEF
/// point: troughs along the cell borders, raised rims beside them, low-centred polygons with a
/// pond, high-centred ones domed and paler. Covers the ground (`amount`).
/// v: cell, jitter, trough, trough_depth, rim, rim_height, pond, high, colour_var, pond_size.
fn polygons(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let wc = worley3(k.seed, i.p, cell, v[1]);
    let ed = worley_edge_dist(&wc, i.p);
    let tint = 1.0 + v[8] * (u01k(wc.id, 2) - 0.5);
    let mut col = k.col[0] * tint;
    let mut dh = 0.0;
    let x = (wc.f1 / (0.5 * cell)).min(1.0);
    if u01k(wc.id, 3) < v[7] {
        // high-centred: domed, drier
        dh = 1.5 * v[5] * (1.0 - x * x);
        col *= 1.08;
    } else if u01k(wc.id, 1) < v[6] {
        let pr = v[9] * 0.5 * cell * (0.6 + 0.4 * u01k(wc.id, 4));
        let pc = band_cov(wc.f1, pr, i.fw);
        col = col + (k.col[2] - col) * pc;
        dh -= 0.2 * pc;
    }
    if v[4] > 0.0 {
        let rc = band_cov(ed - 0.5 * v[2] - 0.5 * v[4], 0.5 * v[4], i.fw);
        col = col + (col * 1.1 - col) * rc;
        dh = dh + (v[5] - dh) * rc;
    }
    let tc = band_cov(ed, 0.5 * v[2], i.fw);
    col = col + (k.col[1] - col) * tc;
    dh = dh + (-v[3] - dh) * tc;
    KOut { cov: i.amount.min(1.0), albedo: col, dh, emit: 0.0, id: wc.id }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;

    #[test]
    fn class_ids_are_the_v2_table() {
        let id = |n: &str| crate::registry::classes::id(n).unwrap();
        assert_eq!(cls::LAKE, id("lake"));
        assert_eq!(cls::RIVER, id("river"));
        assert_eq!(cls::TUNDRA, id("tundra"));
        assert_eq!(cls::SEA_ICE, id("sea_ice"));
        assert_eq!(cls::GLACIER, id("glacier"));
        assert_eq!(cls::FROZEN_WATER, id("frozen_water"));
        assert_eq!(cls::GRAVEL, id("gravel"));
        assert_eq!(cls::MORAINE, id("moraine"));
        assert_eq!(cls::MARSH, id("marsh"));
        assert_eq!(cls::SEASONAL_SNOW, id("seasonal_snow"));
    }

    #[test]
    fn registry_compiles_with_the_cold_biomes() {
        let r = crate::registry::Registry::builtin().unwrap();
        for id in ["polygon_tundra", "shrub_tundra", "polar_desert", "forest_tundra"] {
            assert!(r.biomes.iter().any(|b| b.name == id), "{id}");
        }
    }

    /// Lakes reach no farther than the nearest site of a cell two away can be (1.1 cells with a
    /// jitter of 0.8): the 3 × 3 × 3 cells cover them.
    #[test]
    fn lakes_stay_within_reach() {
        for lat in &LAKE_LATS {
            assert!(lat.r1 * LAKE_EL * LAKE_WN <= 1.1 * lat.cell, "{lat:?}");
        }
    }

    /// The lakes' mean coverage is what the explicit lakes average to, for several mask scales.
    #[test]
    fn lake_means_match_explicit_averages() {
        let base = DVec3::new(2_000_000.0, 1_500_000.0, 5_800_000.0).normalize() * 6_371_000.0;
        let up = base.normalize();
        let e = DVec3::Z.cross(up).normalize();
        let nn = up.cross(e);
        for lat in &LAKE_LATS {
            for sc in [1.0, 0.7] {
                let span = 40.0 * lat.cell;
                let n = 20000;
                let mut c = 0.0;
                for j in 0..n {
                    let q = DVec2::new(crate::kernels::halton(j + 1, 2) * span, crate::kernels::halton(j + 1, 3) * span);
                    let p = (base + e * q.x + nn * q.y).normalize() * 6_371_000.0;
                    c += lake_at(lat, p, sc, 0.5).cov;
                }
                let c = c / n as f64;
                let m = lake_mean_cov(lat, sc);
                assert!((c - m).abs() < 0.015 + 0.1 * m, "cell {}: sc {sc}: explicit {c:.4} mean {m:.4}", lat.cell);
            }
        }
    }

    #[test]
    fn dates() {
        // every ecoregion in the warm half of the year
        for k in 0..=10 {
            let d = year_date(0.1 * k as f64);
            assert!((std::f64::consts::TAU * (d - 0.04)).cos() < -0.25, "{d}");
        }
    }

    /// Places of the default world where the kit draws, by feature (for stills):
    /// `cargo test --release -p terragen find_cold_places -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_cold_places() {
        use rayon::prelude::*;
        let w = crate::world::World::new(crate::Config::default());
        let n = 150_000;
        let pts: Vec<(f64, f64, crate::world::Terrain)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let h = mix64(0xC01D ^ i as u64);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin().to_degrees();
                let lon = 360.0 * u01k(h, 2) - 180.0;
                (lat, lon, w.terrain(&crate::world::Ctx::new(lat.to_radians(), lon.to_radians(), 100.0, &w.ell)))
            })
            .collect();
        let summer = |_lat: f64, t: &crate::world::Terrain| t.temp + 0.95 * 0.5 * t.temp_range;
        let tela = |t: &crate::world::Terrain| -2.0 + 5.0 * smoothstep(0.15, 0.85, t.moist);
        let land = |t: &crate::world::Terrain| t.water_kind == 0 && t.ground > 2.0;
        #[allow(clippy::type_complexity)]
        let kinds: [(&str, Box<dyn Fn(f64, &crate::world::Terrain) -> bool + Sync>); 7] = [
            ("valley_glacier", Box::new(|la, t| land(t) && t.mountain > 0.5 && (-2.5..0.0).contains(&(summer(la, t) - tela(t))))),
            ("ice_cap", Box::new(|la, t| land(t) && summer(la, t) - tela(t) < -4.0)),
            ("polygon_lakes", Box::new(|_, t| land(t) && t.temp < -4.0 && t.ground < 250.0 && t.mountain < 0.1 && t.moist > 0.05)),
            ("kettle_lakes", Box::new(|_, t| land(t) && t.glaciation > 0.5 && (-2.0..6.0).contains(&t.temp) && t.ground < 300.0 && t.mountain < 0.1)),
            ("forest_tundra", Box::new(|_, t| land(t) && (-8.0..-2.0).contains(&t.temp) && t.ground < 600.0 && t.mountain < 0.2)),
            ("polar_desert", Box::new(|_, t| land(t) && t.temp < -10.0 && t.moist < 0.3)),
            ("sea_ice", Box::new(|la, t| t.water_kind == 1 && t.water - t.ground < 40.0 && summer(la, t) < 2.0)),
        ];
        for (name, f) in &kinds {
            let hits: Vec<String> =
                pts.iter().filter(|(la, _, t)| f(*la, t)).take(8).map(|(la, lo, t)| format!("({la:.3}, {lo:.3}) t {:.1} h {:.0}", t.temp, t.ground)).collect();
            let count = pts.iter().filter(|(la, _, t)| f(*la, t)).count();
            let moist: Vec<f64> = pts.iter().filter(|(la, _, t)| f(*la, t)).map(|p| p.2.moist).collect();
            let mm = moist.iter().sum::<f64>() / moist.len().max(1) as f64;
            eprintln!("{name} ({count}, mean moisture {mm:.2}): {}", hits.join("; "));
        }
        // drumlin fields (instances of the family) near glaciated lowland points
        let fields: Vec<String> = pts
            .iter()
            .filter(|(_, _, t)| land(t) && t.glaciation > 0.5 && t.ground < 400.0)
            .filter_map(|(la, lo, t)| {
                let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(la.to_radians(), lo.to_radians(), t.ground), &w.ell);
                let i = crate::instances::near(&w, fam_drumlins(), p, 0.0);
                (!i.is_empty()).then(|| format!("({la:.3}, {lo:.3})"))
            })
            .take(10)
            .collect();
        eprintln!("drumlin fields: {}", fields.join("; "));
    }

    /// Places of the default world where the kit draws: (name, lat, lon).
    pub(crate) const PLACES: &[(&str, f64, f64)] = &[
        ("valley_glaciers", -66.095, 123.288),
        ("tropical_glaciers", -8.298, -124.179),
        ("ice_field", -72.199, -89.333),
        ("polygon_lakes", -72.339, -102.499),
        ("kettle_lakes", 58.609, -92.893),
        ("forest_tundra", -71.075, -168.603),
        ("polar_desert", -78.591, -52.657),
        ("sea_ice", -71.842, -70.019),
        ("drumlins", 57.314, -26.177),
    ];

    fn tile_at(name: &str, z: u8) -> geodesy::tiles::TileId {
        let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == name).unwrap();
        geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)
    }

    /// Share of samples differing by more than `tol`, and the mean difference.
    fn off<T: Copy>(a: &[T], b: &[T], tol: f64, f: impl Fn(T) -> f64) -> (f64, f64) {
        let (mut bad, mut sum) = (0usize, 0.0);
        for (x, y) in a.iter().zip(b) {
            let e = (f(*x) - f(*y)).abs();
            sum += e;
            bad += (e > tol) as usize;
        }
        (bad as f64 / a.len() as f64, sum / a.len() as f64)
    }

    /// Parent ≈ mean of its children (elevation, albedo) where the kit draws: glaciers, lakes.
    #[test]
    fn lod_consistent_at_cold_places() {
        let g = crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [("valley_glaciers", 12u8), ("polygon_lakes", 13)] {
            let parent = tile_at(name, z);
            let p = g.tile(parent);
            let kids = parent.children().map(|c| g.tile(c));
            let (mut eh, mut ea, mut cnt) = (0.0, 0.0, 0.0);
            for (k, kid) in kids.iter().enumerate() {
                let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
                for j in (0..n).step_by(2) {
                    for i in (0..n).step_by(2) {
                        let ix = [j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1];
                        let m: f32 = ix.iter().map(|&q| kid.elevation[q]).sum::<f32>() / 4.0;
                        let pi = (oy + j / 2) * n + ox + i / 2;
                        eh += (m - p.elevation[pi]).abs() as f64;
                        for c in 0..3 {
                            let ma: f64 = ix.iter().map(|&q| kid.albedo[3 * q + c] as f64).sum::<f64>() / 4.0;
                            ea += (ma - p.albedo[3 * pi + c] as f64).abs();
                        }
                        cnt += 1.0;
                    }
                }
            }
            let (eh, ea) = (eh / cnt, ea / (3.0 * cnt));
            eprintln!("{name} z{z}: |parent − mean(children)| elevation {eh:.2} m, albedo {ea:.2} DN");
            assert!(eh < 6.0, "{name}: elevation {eh}");
            assert!(ea < 12.0, "{name}: albedo {ea}");
        }
    }

    /// No seam between neighbouring tiles where the kit draws.
    #[test]
    fn seamless_at_cold_places() {
        let g = crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [("valley_glaciers", 14u8), ("polygon_lakes", 15)] {
            let id = tile_at(name, z);
            let a = g.tile(id);
            let b = g.tile(geodesy::tiles::TileId::new(z, id.x + 1, id.y));
            let (mut across, mut inside) = (0.0, 0.0);
            for j in 0..n {
                for c in 0..3 {
                    across += (a.albedo[3 * (j * n + n - 1) + c] as f64 - b.albedo[3 * (j * n) + c] as f64).abs();
                    inside += (a.albedo[3 * (j * n + n - 2) + c] as f64 - a.albedo[3 * (j * n + n - 1) + c] as f64).abs();
                }
            }
            eprintln!("{name} z{z}: albedo across {across:.0} inside {inside:.0}");
            assert!(across < 2.0 * inside + 2.0 * n as f64, "{name}: across {across} inside {inside}");
        }
    }

    /// The GPU's tiles are the CPU's where the kit draws.
    #[cfg(feature = "gpu")]
    #[test]
    fn tiles_match_the_cpu() {
        let cfg = crate::Config::default();
        let gen = match crate::gpu::GpuGenerator::new(cfg.clone()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU generator ({e}): skipped");
                return;
            }
        };
        let cpu = crate::NextGenerator::new(cfg);
        for (name, z) in [
            ("valley_glaciers", 15u8),
            ("valley_glaciers", 12),
            ("ice_field", 14),
            ("polygon_lakes", 15),
            ("polygon_lakes", 17),
            ("kettle_lakes", 14),
            ("forest_tundra", 16),
            ("sea_ice", 13),
        ] {
            let id = tile_at(name, z);
            let g = gen.tiles(&[id]).unwrap().remove(0);
            let c = cpu.tile_cpu(id);
            let rgb = off(&g.rgb, &c.rgb, 2.0, |v| v as f64);
            let el = off(&g.elevation, &c.elevation, 0.05, |v| v as f64);
            let lc = off(&g.landcover, &c.landcover, 0.5, |v| v as f64);
            eprintln!("{name} z{z}: rgb off {:.3}% elevation off {:.3}% landcover off {:.3}%", rgb.0 * 100.0, el.0 * 100.0, lc.0 * 100.0);
            assert!(rgb.0 < 0.005 && el.0 < 0.005 && lc.0 < 0.005, "{name} z{z}");
        }
    }

    /// Every place generates on the GPU at all zooms (no CPU fallback: an overflowing family, a
    /// shader error): `Backend::Gpu` returns the error.
    #[cfg(feature = "gpu")]
    #[test]
    fn places_stay_on_the_gpu() {
        let Ok(g) = crate::gpu::shared() else { return };
        if g.check_generator().is_err() {
            return;
        }
        let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        for &(name, lat, lon) in PLACES {
            let ids: Vec<_> = [6u8, 8, 10, 12, 14, 16].iter().map(|&z| geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)).collect();
            if let Err(e) = gen.tiles(&ids) {
                panic!("{name}: GPU generation failed: {e:#}");
            }
        }
        assert!(gen.backend_name().starts_with("GPU"), "{}", gen.backend_name());
    }

    /// GPU tiles/s at the kit's places (8 x 8 tiles per place and zoom; performance budget):
    /// `cargo test -p terragen --release --lib kits::cold::tests::bench_places -- --ignored --nocapture`.
    /// For the baseline, run it with the kit's `KITS` entry commented out (the module stays).
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn bench_places() {
        let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        gen.tiles(&[geodesy::tiles::TileId::new(3, 4, 3)]).unwrap();
        for z in [8u8, 10, 12, 14, 16] {
            let mut ids = Vec::new();
            for &(_, lat, lon) in PLACES {
                let c = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
                let n = 1i64 << z;
                for dy in 0..8i64 {
                    for dx in 0..8i64 {
                        ids.push(geodesy::tiles::TileId::new(z, (c.x as i64 + dx - 4).rem_euclid(n) as u32, (c.y as i64 + dy - 4).clamp(0, n - 1) as u32));
                    }
                }
            }
            let t0 = std::time::Instant::now();
            for chunk in ids.chunks(16) {
                gen.tiles(chunk).unwrap();
            }
            let dt = t0.elapsed().as_secs_f64();
            println!("bench z{z}: {} tiles {dt:.2} s {:.1} tiles/s", ids.len(), ids.len() as f64 / dt);
        }
        println!("backend: {}", gen.backend_name());
    }

    fn params(y: &str) -> KParams {
        let m: std::collections::BTreeMap<String, serde_yaml::Value> = serde_yaml::from_str(y).unwrap();
        crate::kernels::compile_params(crate::kernels::spec("cold_polygons").unwrap(), &m, 0x1234, &crate::registry::BiomePal::default_pal()).unwrap()
    }

    /// The kernel's calibrated mean is its explicit average elsewhere (a different point set).
    #[test]
    fn polygon_mean_matches_explicit_average() {
        for y in ["{cell: 16}", "{cell: 4, trough: 0.8, rim: 0, pond: 0, high: 0}"] {
            let k = params(y);
            let m = crate::kernels::calibrate(&k);
            let sz = k.v[0];
            let n = 6000;
            let (mut a, mut h) = (DVec3::ZERO, 0.0);
            for j in 0..n {
                let q = DVec2::new(crate::kernels::halton(j + 7, 5) * 90.0 * sz + 1e4, crate::kernels::halton(j + 7, 7) * 90.0 * sz - 3e3);
                let i =
                    KIn { q, p: DVec3::new(6_371_000.0, q.x, q.y), east: DVec3::Y, north: DVec3::Z, gsd: sz / 50.0, fw: sz / 50.0, amount: 1.0, aux: [0.0; 4] };
                let o = polygons(&k, &i);
                a += o.albedo;
                h += o.dh;
            }
            let a = a / n as f64;
            assert!((a - m.albedo).abs().max_element() < 0.01, "{y}: albedo {a} vs mean {}", m.albedo);
            assert!((h / n as f64 - m.dh).abs() < 0.05, "{y}: height");
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn kernels_match_the_cpu() {
        let Some(r) =
            crate::gpu::tests::kernel_parity(&[("cold_polygons", "{cell: 16}"), ("cold_polygons", "{cell: 4, trough: 0.8, rim: 0.5, pond: 0.3, high: 0.2}")])
        else {
            return;
        };
        for (name, bad, mx) in r {
            eprintln!("  {name} off {:.3}% max {mx:.3}", bad * 100.0);
            assert!(bad < 0.005, "{name}");
        }
    }
}
