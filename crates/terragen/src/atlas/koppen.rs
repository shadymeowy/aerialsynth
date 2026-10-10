//! Köppen–Geiger climate classes from the atlas fields.
//!
//! The atlas keeps annual values and two seasonal shapes: the temperature range (warmest −
//! coldest month) and the precipitation regime (summer vs winter half-year). Monthly values are
//! modelled as sinusoids: T(m) = T̄ + range/2·cos θ, P(m) = P̄·(1 + c·cos θ) with c = regime·π/2
//! (the half-year contrast of a sinusoid), θ = 0 in mid-summer.

use super::AtlasSample;

/// Main group of a [`Koppen`] class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KoppenGroup {
    /// tropical
    A,
    /// arid
    B,
    /// temperate
    C,
    /// continental
    D,
    /// polar
    E,
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

    /// The class code, e.g. "Cfb".
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

    pub fn group(self) -> KoppenGroup {
        match self.code().as_bytes()[0] {
            b'A' => KoppenGroup::A,
            b'B' => KoppenGroup::B,
            b'C' => KoppenGroup::C,
            b'D' => KoppenGroup::D,
            _ => KoppenGroup::E,
        }
    }

    /// The usual map colour (sRGB).
    pub fn colour(self) -> [u8; 3] {
        #[rustfmt::skip]
        const RGB: [[u8; 3]; 30] = [
            [0, 0, 255], [0, 120, 255], [70, 170, 250],
            [255, 0, 0], [255, 150, 150], [245, 165, 0], [255, 220, 100],
            [255, 255, 0], [200, 200, 0], [150, 150, 0], [150, 255, 150], [100, 200, 100], [50, 150, 50],
            [200, 255, 80], [100, 255, 80], [50, 200, 0],
            [255, 0, 255], [200, 0, 200], [150, 50, 150], [150, 100, 150],
            [170, 175, 255], [90, 120, 220], [75, 80, 180], [50, 0, 135],
            [0, 255, 255], [55, 200, 255], [0, 125, 125], [0, 70, 95],
            [178, 178, 178], [102, 102, 102],
        ];
        RGB[self as usize]
    }
}

/// [`koppen_with_lapse`] with the default lapse rate (6 °C/km).
pub fn koppen(s: &AtlasSample, elevation_m: f64) -> Koppen {
    koppen_with_lapse(s, elevation_m, 6.0)
}

/// The Köppen class at `elevation_m` (the sea-level temperature lowered by `lapse_c_per_km`).
pub fn koppen_with_lapse(s: &AtlasSample, elevation_m: f64, lapse_c_per_km: f64) -> Koppen {
    use Koppen::*;
    let t = s.temp_c - lapse_c_per_km * elevation_m.max(0.0) / 1000.0;
    let half = 0.5 * s.temp_range_c.max(0.0);
    let (hot, cold) = (t + half, t - half);
    let p = s.precip_mm.max(0.0);
    let pm = p / 12.0;
    let c = (s.regime * std::f64::consts::FRAC_PI_2).clamp(-1.0, 1.0);
    // monthly extremes in summer and winter
    let (s_max, s_min, w_max, w_min) = if c >= 0.0 { (pm * (1.0 + c), pm, pm, pm * (1.0 - c)) } else { (pm, pm * (1.0 + c), pm * (1.0 - c), pm) };
    if hot < 10.0 {
        return if hot < 0.0 { EF } else { ET };
    }
    // arid: threshold by the share of the precipitation falling in summer
    let summer_share = 0.5 * (1.0 + s.regime);
    let th = 20.0 * t
        + if summer_share >= 0.7 {
            280.0
        } else if summer_share >= 0.3 {
            140.0
        } else {
            0.0
        };
    if p < th {
        let hotdry = t >= 18.0;
        return match (p < 0.5 * th, hotdry) {
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
    // months above 10 °C, from the sinusoid
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
    let pick = |s: [Koppen; 4], w: [Koppen; 4], f: [Koppen; 4]| {
        let row = if dry_summer {
            s
        } else if dry_winter {
            w
        } else {
            f
        };
        row[third]
    };
    if cold > 0.0 {
        // (C has no "d": the coldest month is above 0 °C)
        pick([Csa, Csb, Csc, Csc], [Cwa, Cwb, Cwc, Cwc], [Cfa, Cfb, Cfc, Cfc])
    } else {
        pick([Dsa, Dsb, Dsc, Dsd], [Dwa, Dwb, Dwc, Dwd], [Dfa, Dfb, Dfc, Dfd])
    }
}
