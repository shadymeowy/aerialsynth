//! A stand-in for the planetary atlas (`atlas.rs`, branch `next-atlas`) until it is merged:
//! the same types and field names ([`AtlasSample`], [`Lithology`], [`Archetype`], [`Koppen`]),
//! derived from today's analytic climate fields. Only ecoregion and culture sites read it
//! ([`crate::eco`]), on the host, so both backends see the same values.
//!
//! Switching to the real atlas: `use crate::atlas as atlas_src` in `eco.rs` and its handle in
//! place of [`StubAtlas`].

use crate::noise::*;
use crate::world::World;
use glam::DVec3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Lithology {
    #[default]
    Sedimentary = 0,
    Carbonate = 1,
    Crystalline = 2,
    Volcanic = 3,
    Unconsolidated = 4,
}

impl Lithology {
    pub const ALL: [Lithology; 5] = [Lithology::Sedimentary, Lithology::Carbonate, Lithology::Crystalline, Lithology::Volcanic, Lithology::Unconsolidated];
    pub fn name(self) -> &'static str {
        ["sedimentary", "carbonate", "crystalline", "volcanic", "unconsolidated"][self as usize]
    }
}

/// Culture archetype (per culture area, from the climate at its site).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Archetype {
    #[default]
    TropicalSmallholder = 0,
    TropicalPlantation = 1,
    SavannaPastoral = 2,
    DesertOasis = 3,
    SteppeNomadic = 4,
    Mediterranean = 5,
    TemperateVillage = 6,
    SurveyGrid = 7,
    MonsoonPaddy = 8,
    Boreal = 9,
    Arctic = 10,
    Highland = 11,
}

impl Archetype {
    pub const ALL: [Archetype; 12] = [
        Archetype::TropicalSmallholder,
        Archetype::TropicalPlantation,
        Archetype::SavannaPastoral,
        Archetype::DesertOasis,
        Archetype::SteppeNomadic,
        Archetype::Mediterranean,
        Archetype::TemperateVillage,
        Archetype::SurveyGrid,
        Archetype::MonsoonPaddy,
        Archetype::Boreal,
        Archetype::Arctic,
        Archetype::Highland,
    ];
    pub fn name(self) -> &'static str {
        [
            "tropical_smallholder",
            "tropical_plantation",
            "savanna_pastoral",
            "desert_oasis",
            "steppe_nomadic",
            "mediterranean",
            "temperate_village",
            "survey_grid",
            "monsoon_paddy",
            "boreal",
            "arctic",
            "highland",
        ][self as usize]
    }
}

/// Köppen–Geiger climate class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
#[rustfmt::skip]
pub enum Koppen {
    Af, Am, Aw,
    BWh, BWk, BSh, BSk,
    Csa, Csb, Csc, Cwa, Cwb, Cwc, Cfa, Cfb, Cfc,
    Dsa, Dsb, Dsc, Dsd, Dwa, Dwb, Dwc, Dwd, Dfa, Dfb, Dfc, Dfd,
    ET, EF,
}

impl Koppen {
    #[rustfmt::skip]
    pub const ALL: [Koppen; 30] = [
        Koppen::Af, Koppen::Am, Koppen::Aw,
        Koppen::BWh, Koppen::BWk, Koppen::BSh, Koppen::BSk,
        Koppen::Csa, Koppen::Csb, Koppen::Csc, Koppen::Cwa, Koppen::Cwb, Koppen::Cwc, Koppen::Cfa, Koppen::Cfb, Koppen::Cfc,
        Koppen::Dsa, Koppen::Dsb, Koppen::Dsc, Koppen::Dsd, Koppen::Dwa, Koppen::Dwb, Koppen::Dwc, Koppen::Dwd,
        Koppen::Dfa, Koppen::Dfb, Koppen::Dfc, Koppen::Dfd,
        Koppen::ET, Koppen::EF,
    ];
    pub fn code(self) -> &'static str {
        #[rustfmt::skip]
        const CODES: [&str; 30] = [
            "Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk",
            "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Cfa", "Cfb", "Cfc",
            "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", "Dfa", "Dfb", "Dfc", "Dfd",
            "ET", "EF",
        ];
        CODES[self as usize]
    }
    pub fn from_code(s: &str) -> Option<Koppen> {
        Self::ALL.iter().copied().find(|k| k.code() == s)
    }
}

/// The atlas fields at a point (the subset of the atlas' `AtlasSample` the generator reads).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtlasSample {
    /// smooth elevation (m)
    pub elevation_m: f64,
    /// signed distance to the coast (km): > 0 on land
    pub coast_km: f64,
    /// annual precipitation (mm/year)
    pub precip_mm: f64,
    /// annual mean temperature at sea level (°C)
    pub temp_c: f64,
    /// warmest minus coldest monthly mean (°C)
    pub temp_range_c: f64,
    /// −1 dry summers … +1 summer rains
    pub regime: f64,
    pub plate_dist_km: f64,
    pub uplift: f64,
    pub volcanism: f64,
    pub glaciation: f64,
    pub development: f64,
    pub population: f64,
    pub continental: bool,
    pub litho: Lithology,
    pub litho2: Lithology,
    pub litho2_frac: f64,
    pub culture: u32,
    pub archetype: Archetype,
}

/// The Köppen class at `elevation_m` (`atlas::koppen_with_lapse`).
pub fn koppen_with_lapse(s: &AtlasSample, elevation_m: f64, lapse_c_per_km: f64) -> Koppen {
    use Koppen::*;
    let t = s.temp_c - lapse_c_per_km * elevation_m.max(0.0) / 1000.0;
    let half = 0.5 * s.temp_range_c.max(0.0);
    let (hot, cold) = (t + half, t - half);
    let p = s.precip_mm.max(0.0);
    let pm = p / 12.0;
    let c = (s.regime * std::f64::consts::FRAC_PI_2).clamp(-1.0, 1.0);
    let (s_max, s_min, w_max, w_min) = if c >= 0.0 { (pm * (1.0 + c), pm, pm, pm * (1.0 - c)) } else { (pm, pm * (1.0 + c), pm * (1.0 - c), pm) };
    if hot < 10.0 {
        return if hot < 0.0 { EF } else { ET };
    }
    let summer_share = 0.5 * (1.0 + s.regime);
    let th = 20.0 * t + if summer_share >= 0.7 { 280.0 } else if summer_share >= 0.3 { 140.0 } else { 0.0 };
    if p < th {
        return match (p < 0.5 * th, t >= 18.0) {
            (true, true) => BWh,
            (true, false) => BWk,
            (false, true) => BSh,
            (false, false) => BSk,
        };
    }
    if cold >= 18.0 {
        let dry = s_min.min(w_min);
        return if dry >= 60.0 {
            Af
        } else if dry >= 100.0 - p / 25.0 {
            Am
        } else {
            Aw
        };
    }
    let n10 = if half <= 0.0 { 12.0 } else { 12.0 * ((10.0 - t) / half).clamp(-1.0, 1.0).acos() / std::f64::consts::PI };
    let dry_summer = s_min < 40.0 && s_min < w_max / 3.0;
    let dry_winter = w_min < s_max / 10.0;
    let third = if cold < -38.0 {
        3
    } else if hot >= 22.0 {
        0
    } else if n10 >= 4.0 {
        1
    } else {
        2
    };
    let pick = |s: [Koppen; 4], w: [Koppen; 4], f: [Koppen; 4]| if dry_summer { s[third] } else if dry_winter { w[third] } else { f[third] };
    if cold > 0.0 {
        pick([Csa, Csb, Csc, Csc], [Cwa, Cwb, Cwc, Cwc], [Cfa, Cfb, Cfc, Cfc])
    } else {
        pick([Dsa, Dsb, Dsc, Dsd], [Dwa, Dwb, Dwc, Dwd], [Dfa, Dfb, Dfc, Dfd])
    }
}

/// Months (0..12) with less than 60 mm of rain, from the annual sum and the regime.
pub fn dry_months(s: &AtlasSample) -> f64 {
    let pm = s.precip_mm.max(0.0) / 12.0;
    let a = s.regime.abs().min(1.0);
    // monthly rain p(m) = pm (1 + a cos θ): months below 60 mm
    if pm * (1.0 + a) < 60.0 {
        return 12.0;
    }
    if pm * (1.0 - a) >= 60.0 {
        return 0.0;
    }
    let c = ((60.0 / pm - 1.0) / a.max(1e-6)).clamp(-1.0, 1.0);
    12.0 * (1.0 - c.acos() / std::f64::consts::PI)
}

/// The atlas stand-in: smooth climate from the world's macro fields, lithology and cultures
/// from coarse lattices.
pub struct StubAtlas<'a> {
    pub world: &'a World,
}

const KM: f64 = 1000.0;

impl StubAtlas<'_> {
    /// The atlas at the surface point in direction `dir` (unit vector from the centre).
    pub fn sample(&self, dir: DVec3) -> AtlasSample {
        let w = self.world;
        let g = geodesy::ecef2geodetic(dir * w.ell.a, &w.ell);
        let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(g.lat, g.lon, 0.0), &w.ell);
        let m = w.macro_at(p, 20.0 * KM);
        let (t0, moist) = w.climate(&m, g.lat, 0.0);
        let elev = w.smooth_elevation(p).max(0.0);
        let latd = g.lat.abs().to_degrees();
        let seed = w.seed();
        // seasonality: latitude and continentality
        let cont = smoothstep(0.02, 0.45, m.cont);
        let temp_range_c = 3.0 + 26.0 * smoothstep(5.0, 65.0, latd) * (0.45 + 0.55 * cont) + 2.0 * perlin3(seed ^ 0xA7A1, p / (900.0 * KM));
        // regime: dry summers on the subtropical west sides (a noise stands in for the coasts),
        // summer rains (monsoon) in the outer tropics
        let rn = perlin3(seed ^ 0xA7A2, p / (1600.0 * KM));
        let med = (-((latd - 37.0) / 7.0).powi(2)).exp() * smoothstep(0.0, 0.35, rn);
        let mon = (-((latd - 15.0) / 9.0).powi(2)).exp() * smoothstep(0.0, 0.35, -rn);
        let regime = (0.8 * mon - 0.85 * med).clamp(-1.0, 1.0);
        let precip_mm = 40.0 + 2900.0 * moist.powf(1.7);
        // lithology: a coarse cell lattice (≈ 500 km provinces) with a secondary rock
        let lc = worley3(seed ^ 0xA7A3, p, 500.0 * KM, 0.9);
        let pick_litho = |h: u64| -> Lithology {
            let u = u01k(h, 7);
            // volcanic more likely in mountain belts (stand-in for arcs)
            let volc = 0.08 + 0.15 * smoothstep(0.6, 0.9, 1.0 - m.belt.abs());
            if u < volc {
                Lithology::Volcanic
            } else if u < volc + 0.2 {
                Lithology::Carbonate
            } else if u < volc + 0.42 {
                Lithology::Crystalline
            } else if u < volc + 0.62 {
                Lithology::Unconsolidated
            } else {
                Lithology::Sedimentary
            }
        };
        let litho = pick_litho(lc.id);
        let litho2 = pick_litho(lc.id2);
        let litho2_frac = 0.5 * (1.0 - smoothstep(0.0, 0.25, (lc.f2 - lc.f1) / (500.0 * KM)));
        // cultures: ≈ 1000 km areas, archetype by the climate at their site
        let cc = worley3(seed ^ 0xC017, p, 1000.0 * KM, 0.9);
        let culture = (cc.id >> 40) as u32 & 0xFFF;
        let archetype = self.archetype_at(cc.point, cc.id);
        let development = (0.15 + 0.85 * u01k(cc.id, 3)) * (0.85 + 0.15 * perlin3(seed ^ 0xA7A4, p / (300.0 * KM)));
        AtlasSample {
            elevation_m: elev,
            coast_km: m.cont / 0.3 * 600.0,
            precip_mm,
            temp_c: t0,
            temp_range_c: temp_range_c.max(1.0),
            regime,
            plate_dist_km: 1000.0,
            uplift: 0.0,
            volcanism: if litho == Lithology::Volcanic { 0.6 } else { 0.0 },
            glaciation: smoothstep(48.0, 66.0, latd) * smoothstep(0.0, 0.2, m.cont),
            development: development.clamp(0.0, 1.0),
            population: (smoothstep(0.2, 0.5, moist) * smoothstep(0.0, 12.0, t0) * (1.0 - smoothstep(24.0, 30.0, t0))).clamp(0.0, 1.0),
            continental: m.cont > 0.0,
            litho,
            litho2,
            litho2_frac,
            culture,
            archetype,
        }
    }

    /// The archetype of a culture site (by the climate there).
    fn archetype_at(&self, pt: DVec3, id: u64) -> Archetype {
        use Archetype::*;
        let w = self.world;
        let g = geodesy::ecef2geodetic(pt, &w.ell);
        let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(g.lat, g.lon, 0.0), &w.ell);
        let m = w.macro_at(p, 50.0 * KM);
        let (t, moist) = w.climate(&m, g.lat, w.smooth_elevation(p).max(0.0));
        let u = u01k(id, 9);
        let elev = w.smooth_elevation(p);
        if t < -4.0 {
            Arctic
        } else if t < 3.0 {
            Boreal
        } else if elev > 1800.0 && u < 0.6 {
            Highland
        } else if moist < 0.18 {
            DesertOasis
        } else if moist < 0.32 {
            if t > 19.0 {
                SavannaPastoral
            } else {
                SteppeNomadic
            }
        } else if t > 21.0 {
            if moist > 0.65 {
                if u < 0.45 {
                    MonsoonPaddy
                } else if u < 0.75 {
                    TropicalPlantation
                } else {
                    TropicalSmallholder
                }
            } else if u < 0.6 {
                TropicalSmallholder
            } else {
                SavannaPastoral
            }
        } else if t > 14.0 && moist < 0.5 && u < 0.55 {
            Mediterranean
        } else if u < 0.35 {
            SurveyGrid
        } else if u < 0.5 && t > 15.0 {
            MonsoonPaddy
        } else {
            TemperateVillage
        }
    }
}
