//! Variety budget (`docs/design/terrain-next.md` §11): the world's landscapes do not collapse
//! into a few looks. Ecoregion biomes over random land points: every biome with a `min_share`
//! reaches it, none exceeds 45 % of the land, and every 1000 km circle holds several biomes.
use terragen::noise::{mix64, u01k};
use terragen::world::{Ctx, World};

fn land_point(w: &World, h: u64) -> Option<(f64, f64, terragen::world::Terrain)> {
    let lat = (2.0 * u01k(h, 1) - 1.0).asin();
    let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
    let t = w.terrain(&Ctx::new(lat, lon, 30.0, &w.ell));
    (t.water_kind == 0 && t.ground > 0.0).then_some((lat, lon, t))
}

#[test]
fn ecoregion_biomes_keep_their_shares() {
    let w = World::new(terragen::Config::default());
    let reg = terragen::registry::Registry::builtin().unwrap();
    let mut count = vec![0usize; reg.biomes.len()];
    let mut n = 0;
    for i in 0..2500u64 {
        let Some((_, _, t)) = land_point(&w, mix64(0x7A21 ^ i)) else { continue };
        let e = terragen::eco::compute(&w, &reg, t.eco.id, t.eco.center);
        count[e.biome as usize] += 1;
        n += 1;
    }
    assert!(n > 500, "{n} land points");
    let shares: Vec<(String, f64)> = reg.biomes.iter().zip(&count).map(|(b, &c)| (b.name.clone(), c as f64 / n as f64)).collect();
    eprintln!("biome shares: {shares:?}");
    for (b, (name, s)) in reg.biomes.iter().zip(&shares) {
        assert!(*s >= b.min_share, "biome {name}: share {s:.3} below its min_share {}", b.min_share);
        assert!(*s <= 0.45, "biome {name}: share {s:.3} above 45 %");
    }
    assert!(shares.iter().filter(|s| s.1 > 0.0).count() >= 8, "only {} biomes appear", shares.iter().filter(|s| s.1 > 0.0).count());
}

#[test]
fn every_region_holds_several_biomes() {
    let w = World::new(terragen::Config::default());
    let reg = terragen::registry::Registry::builtin().unwrap();
    let mut checked = 0;
    for c in 0..40u64 {
        let Some((lat0, lon0, _)) = land_point(&w, mix64(0xC1C ^ c)) else { continue };
        let mut biomes = std::collections::BTreeSet::new();
        let mut land = 0;
        for i in 0..160u64 {
            let h = mix64((c << 20) ^ i ^ 0x5EED);
            // within ~1000 km
            let (dlat, dlon) = ((u01k(h, 1) - 0.5) * 0.28, (u01k(h, 2) - 0.5) * 0.28 / lat0.cos().max(0.2));
            let t = w.terrain(&Ctx::new((lat0 + dlat).clamp(-1.5, 1.5), lon0 + dlon, 30.0, &w.ell));
            if t.water_kind != 0 || t.ground <= 0.0 {
                continue;
            }
            land += 1;
            biomes.insert(terragen::eco::compute(&w, &reg, t.eco.id, t.eco.center).biome);
        }
        if land >= 60 {
            checked += 1;
            assert!(biomes.len() >= 2, "only {} biome(s) within 1000 km of {:.1},{:.1}", biomes.len(), lat0.to_degrees(), lon0.to_degrees());
        }
        if checked >= 12 {
            break;
        }
    }
    assert!(checked >= 6, "{checked} regions checked");
}
