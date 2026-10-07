//! Star catalogue (`scripts/build_stars.py`): header `STARCAT1`, u32 count, f64 epoch (Julian
//! year, ICRS positions), then records sorted by V:
//! id u32 (HIP number; Tycho-2 as 1<<31 | TYC1<<17 | TYC2<<3 | TYC3), RA u32 and Dec i32
//! (units of 2π / 2³²), μα* and μδ f32 (mas/yr), parallax f32 (mas), V and B−V i16 (mmag).
//! The built-in catalogue (Hipparcos + Tycho-2, V ≤ 9, ~130k stars) is embedded in the binary.

use anyhow::{bail, Context, Result};
use glam::DVec3;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const REC: usize = 28;
const UNIT: f64 = std::f64::consts::TAU / 4_294_967_296.0;
const MAS: f64 = std::f64::consts::PI / 180.0 / 3600e3;

static BUILTIN: &[u8] = include_bytes!("../../data/stars_v9.bin");

#[derive(Clone, Copy, Debug)]
pub struct Star {
    pub id: u32,
    /// ICRS unit vector at the catalogue epoch
    pub dir: DVec3,
    /// proper motion as a tangent vector (rad / Julian year)
    pub pm: DVec3,
    /// parallax (rad)
    pub plx: f64,
    pub v: f32,
    pub bv: f32,
}

pub struct Catalog {
    pub stars: Vec<Star>,
    /// Julian year of the positions
    pub epoch: f64,
}

impl Catalog {
    pub fn parse(b: &[u8]) -> Result<Catalog> {
        if b.len() < 20 || &b[..8] != b"STARCAT1" {
            bail!("not a star catalogue (STARCAT1)");
        }
        let n = u32::from_le_bytes(b[8..12].try_into()?) as usize;
        let epoch = f64::from_le_bytes(b[12..20].try_into()?);
        if b.len() != 20 + n * REC {
            bail!("star catalogue: size mismatch ({} records, {} bytes)", n, b.len());
        }
        let f32at = |r: &[u8], o: usize| f32::from_le_bytes(r[o..o + 4].try_into().unwrap()) as f64;
        let stars = b[20..]
            .as_chunks::<REC>()
            .0
            .iter()
            .map(|r| {
                let id = u32::from_le_bytes(r[0..4].try_into().unwrap());
                let ra = u32::from_le_bytes(r[4..8].try_into().unwrap()) as f64 * UNIT;
                let de = i32::from_le_bytes(r[8..12].try_into().unwrap()) as f64 * UNIT;
                let (sa, ca) = ra.sin_cos();
                let (sd, cd) = de.sin_cos();
                let dir = DVec3::new(cd * ca, cd * sa, sd);
                let ea = DVec3::new(-sa, ca, 0.0);
                let ed = DVec3::new(-sd * ca, -sd * sa, cd);
                let pm = (ea * f32at(r, 12) + ed * f32at(r, 16)) * MAS;
                let plx = f32at(r, 20) * MAS;
                let v = i16::from_le_bytes(r[24..26].try_into().unwrap()) as f32 / 1000.0;
                let bv = i16::from_le_bytes(r[26..28].try_into().unwrap()) as f32 / 1000.0;
                Star { id, dir, pm, plx, v, bv }
            })
            .collect();
        Ok(Catalog { stars, epoch })
    }

    /// The built-in catalogue or a catalogue file (cached per process).
    pub fn load(path: Option<&str>) -> Result<Arc<Catalog>> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<Catalog>>>> = OnceLock::new();
        let key = path.unwrap_or("").to_string();
        let mut c = CACHE.get_or_init(Default::default).lock().unwrap();
        if let Some(cat) = c.get(&key) {
            return Ok(cat.clone());
        }
        let cat = Arc::new(match path {
            None => Catalog::parse(BUILTIN)?,
            Some(p) => Catalog::parse(&std::fs::read(p).with_context(|| format!("reading star catalogue {p}"))?)?,
        });
        c.insert(key, cat.clone());
        Ok(cat)
    }
}

/// Catalogue designation: "HIP 32349" or "TYC 4-1-2".
pub fn designation(id: u32) -> String {
    if id & (1 << 31) == 0 {
        format!("HIP {id}")
    } else {
        format!("TYC {}-{}-{}", (id >> 17) & 0x3FFF, (id >> 3) & 0x3FFF, id & 7)
    }
}
