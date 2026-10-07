//! Compact JPL DE440 ephemeris (`scripts/build_planets.py`): Chebyshev segments of the Sun, the
//! planet barycentres, the Earth-Moon barycentre and the Moon, 1990–2060, embedded in the
//! binary (2.8 MB). Positions in km and velocities in km/day, ICRF axes, TDB (≈ TT).

use anyhow::{bail, Result};
use glam::DVec3;
use std::sync::OnceLock;

static BUILTIN: &[u8] = include_bytes!("../../data/planets.bin");

struct Segment {
    center: i32,
    target: i32,
    jd0: f64,
    len: f64,
    n: usize,
    nc: usize,
    /// per interval and axis: constant term
    c0: Vec<f64>,
    /// per interval and axis: terms 1..nc
    c: Vec<f32>,
}

impl Segment {
    fn posvel(&self, jd: f64) -> Option<(DVec3, DVec3)> {
        let x = (jd - self.jd0) / self.len;
        if x < 0.0 || x > self.n as f64 {
            return None;
        }
        let i = (x.floor() as usize).min(self.n - 1);
        let tau = 2.0 * (x - i as f64) - 1.0;
        let mut p = [0.0; 3];
        let mut v = [0.0; 3];
        for ax in 0..3 {
            let k = i * 3 + ax;
            let cs = &self.c[k * (self.nc - 1)..(k + 1) * (self.nc - 1)];
            // Chebyshev T_n and dT_n/dτ
            let (mut t0, mut t1) = (1.0, tau);
            let (mut d0, mut d1) = (0.0, 1.0);
            let mut s = self.c0[k] + cs[0] as f64 * t1;
            let mut ds = cs[0] as f64 * d1;
            for c in &cs[1..] {
                let t2 = 2.0 * tau * t1 - t0;
                let d2 = 2.0 * t1 + 2.0 * tau * d1 - d0;
                s += *c as f64 * t2;
                ds += *c as f64 * d2;
                (t0, t1, d0, d1) = (t1, t2, d1, d2);
            }
            p[ax] = s;
            v[ax] = ds * 2.0 / self.len;
        }
        Some((DVec3::from(p), DVec3::from(v)))
    }
}

pub struct Ephemeris {
    segs: Vec<Segment>,
    emrat: f64,
}

/// Bodies (NAIF ids).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Body {
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
}

impl Body {
    pub fn naif(self) -> u32 {
        match self {
            Body::Sun => 10,
            Body::Mercury => 199,
            Body::Venus => 299,
            Body::Earth => 399,
            Body::Moon => 301,
            Body::Mars => 499,
            Body::Jupiter => 599,
            Body::Saturn => 699,
            Body::Uranus => 799,
            Body::Neptune => 899,
        }
    }
}

impl Ephemeris {
    pub fn parse(b: &[u8]) -> Result<Ephemeris> {
        if b.len() < 20 || &b[..8] != b"PLANETS1" {
            bail!("not a planetary ephemeris (PLANETS1)");
        }
        let u32at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let f64at = |o: usize| f64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let nb = u32at(8) as usize;
        let emrat = f64at(12);
        let mut o = 20;
        let mut segs = vec![];
        for _ in 0..nb {
            let center = u32at(o) as i32;
            let target = u32at(o + 4) as i32;
            let (jd0, len) = (f64at(o + 8), f64at(o + 16));
            let (n, nc) = (u32at(o + 24) as usize, u32at(o + 28) as usize);
            o += 32;
            let mut c0 = Vec::with_capacity(n * 3);
            let mut c = Vec::with_capacity(n * 3 * (nc - 1));
            for _ in 0..n * 3 {
                c0.push(f64at(o));
                o += 8;
                for _ in 0..nc - 1 {
                    c.push(f32::from_le_bytes(b[o..o + 4].try_into().unwrap()));
                    o += 4;
                }
            }
            segs.push(Segment { center, target, jd0, len, n, nc, c0, c });
        }
        if o != b.len() {
            bail!("planetary ephemeris: trailing bytes");
        }
        Ok(Ephemeris { segs, emrat })
    }

    pub fn builtin() -> &'static Ephemeris {
        static E: OnceLock<Ephemeris> = OnceLock::new();
        E.get_or_init(|| Ephemeris::parse(BUILTIN).expect("embedded planetary ephemeris"))
    }

    fn seg(&self, center: i32, target: i32, jd: f64) -> Option<(DVec3, DVec3)> {
        self.segs.iter().find(|s| s.center == center && s.target == target)?.posvel(jd)
    }

    /// Barycentric position (km) and velocity (km/day) of `body` at `jd` (TDB); None outside
    /// the covered dates.
    pub fn barycentric(&self, body: Body, jd: f64) -> Option<(DVec3, DVec3)> {
        let bary = |t: i32| self.seg(0, t, jd);
        match body {
            Body::Sun => bary(10),
            Body::Mercury => bary(1),
            Body::Venus => bary(2),
            Body::Mars => bary(4),
            Body::Jupiter => bary(5),
            Body::Saturn => bary(6),
            Body::Uranus => bary(7),
            Body::Neptune => bary(8),
            Body::Earth | Body::Moon => {
                let (pe, ve) = bary(3)?;
                let (pm, vm) = self.seg(3, 301, jd)?;
                Some(if body == Body::Moon { (pe + pm, ve + vm) } else { (pe - pm / self.emrat, ve - vm / self.emrat) })
            }
        }
    }
}
