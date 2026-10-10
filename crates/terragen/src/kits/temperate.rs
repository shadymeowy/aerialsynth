//! The temperate, boreal and wetland kit (`docs/design/terrain-next.md` §5.7,
//! `docs/kits/temperate.md`).
//!
//! * **Biomes** (`biomes/temperate.yaml`): broadleaf deciduous, mixed forest, taiga, lichen
//!   woodland, maquis, garrigue, chaparral, feather-grass steppe, heathland, moorland, and the
//!   zones above the tree line (subalpine krummholz, alpine meadow) of the temperate biomes.
//! * **Layer functions** (here; WGSL twins in `gpu/wgsl/kits/temperate.wgsl`), climate- and
//!   terrain-driven in every biome where they fit.

use crate::kits::Kit;
use crate::stack::slot;

pub static KIT: Kit = Kit {
    name: "temperate",
    biomes_yaml: include_str!("../../biomes/temperate.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/temperate.wgsl"),
    layers: &[],
    ..Kit::EMPTY
};

#[allow(dead_code)]
const _SLOTS: [usize; 1] = [slot::AZONAL];

#[cfg(test)]
mod tests {
    use crate::noise::*;
    use crate::registry::Registry;
    use crate::world::{Ctx, World};

    #[test]
    fn registry_compiles() {
        let r = Registry::builtin().unwrap();
        for id in ["broadleaf_deciduous", "mixed_forest", "taiga", "lichen_woodland", "maquis", "garrigue", "chaparral", "feather_grass_steppe", "heathland", "moorland", "subalpine_krummholz", "alpine_meadow"] {
            assert!(r.index(id).is_some(), "{id}");
        }
    }

    /// Places of the default world where the kit's biomes are (`cargo test --release -p terragen
    /// find_places -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn find_places() {
        use rayon::prelude::*;
        let w = World::new(crate::Config::default());
        let reg = Registry::builtin().unwrap();
        let n = 60_000usize;
        let hits: Vec<(f64, f64, String, String, f64, f64, f64)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = mix64(0xF1ED ^ i as u64);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin().to_degrees();
                let lon = 360.0 * u01k(h, 2) - 180.0;
                let ctx = Ctx::new(lat.to_radians(), lon.to_radians(), 100.0, &w.ell);
                let t = w.terrain(&ctx);
                if t.water_kind != 0 || t.ground < 2.0 {
                    return None;
                }
                let [(id, site), _] = crate::eco::sites(&w, ctx.p);
                let e = crate::eco::compute(&w, &reg, id, site);
                let mut b = e.biome;
                for &(below, zb) in &reg.biomes[e.biome as usize].zones {
                    if t.temp < below {
                        b = zb;
                    }
                }
                let k = crate::atlas_stub::Koppen::ALL[e.climate.koppen as usize].code().to_string();
                Some((lat, lon, reg.biomes[b as usize].name.clone(), k, t.temp, t.moist, t.ground))
            })
            .collect();
        let mut by: std::collections::BTreeMap<String, Vec<_>> = Default::default();
        for h in &hits {
            by.entry(h.2.clone()).or_default().push(h.clone());
        }
        let total = hits.len() as f64;
        for (b, v) in &by {
            let s: Vec<String> = v.iter().take(5).map(|h| format!("{:.3},{:.3} {} T{:.0} M{:.2} h{:.0}", h.0, h.1, h.3, h.4, h.5, h.6)).collect();
            println!("{b:22} {:5.2}%  {}", 100.0 * v.len() as f64 / total, s.join(" | "));
        }
        let mut kp: std::collections::BTreeMap<String, usize> = Default::default();
        for h in &hits {
            *kp.entry(h.3.clone()).or_default() += 1;
        }
        println!("koppen: {kp:?}");
    }
}
