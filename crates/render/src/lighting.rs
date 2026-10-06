//! Sun / day-cycle lighting: fixed sun or clock-driven solar position (NOAA algorithm), with
//! irradiance and sky brightness through twilight, and the switch for artificial (town) lights.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SunMode {
    /// Use `sun_azimuth_deg` / `sun_elevation_deg` (in the local frame of the first pose).
    Fixed,
    /// Solar position from `date` + `time_utc` at the camera location; time advances with the
    /// trajectory time multiplied by `time_scale` (0 = static clock).
    Clock,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LightsMode {
    /// On when the sun is below `lights_on_below_deg`.
    Auto,
    On,
    Off,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LightingConfig {
    pub mode: SunMode,
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    /// YYYY-MM-DD
    pub date: String,
    /// HH:MM[:SS] UTC at trajectory time 0.
    pub time_utc: String,
    pub time_scale: f64,
    /// Direct sun irradiance scale (sunlit white Lambertian at normal incidence ≈ this).
    pub sun_intensity: f64,
    /// Sky (ambient) light scale.
    pub sky_intensity: f64,
    /// Terrain / tree / building cast shadows (relit shading).
    pub shadows: bool,
    pub lights: LightsMode,
    pub lights_on_below_deg: f64,
    /// Radiance scale of artificial lights.
    pub lights_intensity: f64,
}

impl Default for LightingConfig {
    fn default() -> Self {
        LightingConfig {
            mode: SunMode::Fixed,
            sun_azimuth_deg: 145.0,
            sun_elevation_deg: 52.0,
            date: "2026-06-21".into(),
            time_utc: "08:30:00".into(),
            time_scale: 1.0,
            sun_intensity: 1.0,
            sky_intensity: 1.0,
            shadows: true,
            lights: LightsMode::Auto,
            lights_on_below_deg: 1.0,
            lights_intensity: 1.0,
        }
    }
}

/// Days from 1970-01-01 for a civil date (proleptic Gregorian).
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parse date + time (UTC) to Unix seconds.
pub fn parse_utc(date: &str, time: &str) -> anyhow::Result<f64> {
    let d: Vec<i64> = date.split('-').map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    let t: Vec<f64> = time.split(':').map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    if d.len() != 3 || t.is_empty() {
        anyhow::bail!("bad date/time {date} {time}");
    }
    let secs = t[0] * 3600.0 + t.get(1).copied().unwrap_or(0.0) * 60.0 + t.get(2).copied().unwrap_or(0.0);
    Ok(days_from_civil(d[0], d[1], d[2]) as f64 * 86400.0 + secs)
}

/// Solar azimuth (rad, clockwise from north) and elevation (rad, with refraction) — NOAA.
pub fn solar_position(unix: f64, lat: f64, lon: f64) -> (f64, f64) {
    let jd = unix / 86400.0 + 2440587.5;
    let jc = (jd - 2451545.0) / 36525.0;
    let deg = std::f64::consts::PI / 180.0;
    let l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)).rem_euclid(360.0);
    let m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
    let e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);
    let c = (m * deg).sin() * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + (2.0 * m * deg).sin() * (0.019993 - 0.000101 * jc)
        + (3.0 * m * deg).sin() * 0.000289;
    let true_long = l0 + c;
    let omega = 125.04 - 1934.136 * jc;
    let app_long = true_long - 0.00569 - 0.00478 * (omega * deg).sin();
    let eps0 = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0;
    let eps = eps0 + 0.00256 * (omega * deg).cos();
    let decl = ((eps * deg).sin() * (app_long * deg).sin()).asin();
    let y = (eps * deg / 2.0).tan().powi(2);
    let eq_time = 4.0
        / deg
        * (y * (2.0 * l0 * deg).sin() - 2.0 * e * (m * deg).sin() + 4.0 * e * y * (m * deg).sin() * (2.0 * l0 * deg).cos()
            - 0.5 * y * y * (4.0 * l0 * deg).sin()
            - 1.25 * e * e * (2.0 * m * deg).sin());
    let minutes = (unix / 60.0).rem_euclid(1440.0);
    let tst = (minutes + eq_time + 4.0 * lon / deg).rem_euclid(1440.0);
    let ha = if tst / 4.0 < 0.0 { tst / 4.0 + 180.0 } else { tst / 4.0 - 180.0 };
    let cos_z = lat.sin() * decl.sin() + lat.cos() * decl.cos() * (ha * deg).cos();
    let zen = cos_z.clamp(-1.0, 1.0).acos();
    let el = std::f64::consts::FRAC_PI_2 - zen;
    // atmospheric refraction (approx, degrees)
    let eld = el / deg;
    let refr = if eld > 85.0 {
        0.0
    } else if eld > 5.0 {
        (58.1 / (eld * deg).tan() - 0.07 / (eld * deg).tan().powi(3) + 0.000086 / (eld * deg).tan().powi(5)) / 3600.0
    } else if eld > -0.575 {
        (1735.0 + eld * (-518.2 + eld * (103.4 + eld * (-12.79 + eld * 0.711)))) / 3600.0
    } else {
        -20.772 / (eld * deg).tan() / 3600.0
    };
    let el = el + refr * deg;
    let az = {
        let s = zen.sin();
        if s.abs() < 1e-9 {
            0.0
        } else {
            let ca = ((lat.sin() * zen.cos()) - decl.sin()) / (lat.cos() * s);
            let a = ca.clamp(-1.0, 1.0).acos() / deg;
            if ha > 0.0 { (a + 180.0).rem_euclid(360.0) } else { (540.0 - a).rem_euclid(360.0) }
        }
    };
    (az * deg, el)
}

/// Lighting state for one frame.
#[derive(Clone, Copy, Debug)]
pub struct SunState {
    pub azimuth: f64,
    pub elevation: f64,
    /// direct sun radiance multiplier (0 below the horizon, reduced at low sun by air mass)
    pub direct: f64,
    /// sky ambient multiplier (twilight → night)
    pub sky: f64,
    /// artificial lights multiplier (0..1)
    pub lights: f64,
}

impl LightingConfig {
    /// Sun state at trajectory time `t` (s) and location (rad).
    pub fn sun_at(&self, t: f64, lat: f64, lon: f64) -> SunState {
        let (az, el) = match self.mode {
            SunMode::Fixed => (self.sun_azimuth_deg.to_radians(), self.sun_elevation_deg.to_radians()),
            SunMode::Clock => {
                let t0 = parse_utc(&self.date, &self.time_utc).unwrap_or(0.0);
                solar_position(t0 + t * self.time_scale, lat, lon)
            }
        };
        let eld = el.to_degrees();
        // relative air mass (Kasten–Young) → direct transmittance
        let am = if eld > -1.0 { 1.0 / ((eld.max(0.0) * std::f64::consts::PI / 180.0).sin() + 0.50572 * (eld.max(0.0) + 6.07995).powf(-1.6364)) } else { 40.0 };
        let direct = if eld > -0.8 { (0.9f64).powf(am.min(40.0)) / 0.9 * smooth(-0.8, 1.0, eld) } else { 0.0 } * self.sun_intensity;
        let sky = (0.03 + 0.97 * smooth(-6.0, 12.0, eld)) * smooth(-18.0, -6.0, eld).max(0.002) * self.sky_intensity;
        let lights = match self.lights {
            LightsMode::On => 1.0,
            LightsMode::Off => 0.0,
            LightsMode::Auto => smooth(self.lights_on_below_deg, self.lights_on_below_deg - 4.0, eld),
        } * self.lights_intensity;
        SunState { azimuth: az, elevation: el, direct, sky, lights }
    }
}

fn smooth(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solar_noon_ankara_solstice() {
        // 2026-06-21, Ankara (39.92N, 32.85E): solar noon ≈ 09:47 UTC, elevation ≈ 73.6°
        let lat = 39.92f64.to_radians();
        let lon = 32.85f64.to_radians();
        let t = parse_utc("2026-06-21", "09:47:00").unwrap();
        let (az, el) = solar_position(t, lat, lon);
        assert!((el.to_degrees() - 73.6).abs() < 0.5, "el {}", el.to_degrees());
        assert!((az.to_degrees() - 180.0).abs() < 5.0, "az {}", az.to_degrees());
        // sunrise ≈ 02:17 UTC (05:17 local): elevation ~0
        let t = parse_utc("2026-06-21", "02:17:00").unwrap();
        let (az, el) = solar_position(t, lat, lon);
        assert!(el.to_degrees().abs() < 1.5, "{}", el.to_degrees());
        assert!(az.to_degrees() > 50.0 && az.to_degrees() < 70.0, "{}", az.to_degrees());
    }

    #[test]
    fn epoch() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
    }
}
