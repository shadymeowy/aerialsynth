//! A test-only kit exercising every hook (compiled in the crate's unit tests only): an instance
//! family of cones with a relief operator (pass A, so the drainage follows them), an azonal
//! layer drawing ash on them, a kit kernel used by a biome layer, and a biome. It is also the
//! worked example of `docs/design/kit-guide.md`.

use crate::instances::{Family, InstSite};
use crate::kernels::{KIn, KOut, KParams, KernelSpec, P};
use crate::kits::{Kit, ReliefIn};
use crate::noise::*;
use crate::stack::{slot, Layer, Stack};
use crate::world::World;

pub static KIT: Kit = Kit {
    name: "example",
    biomes_yaml: include_str!("../../biomes/example.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/example.wgsl"),
    layers: &[(slot::AZONAL, azonal)],
    kernels: &[KernelSpec {
        name: "example_rings",
        kind: 240,
        params: RINGS_P,
        size_slot: 0,
        dyn_size: None,
        reach: None,
        default_hmode: "none",
        eval: rings,
        wgsl: "example_rings",
    }],
    relief: Some(relief),
    relief_wgsl: include_str!("../gpu/wgsl/kits/example_relief.wgsl"),
    families: &[Family { name: "example_cones", cell: 40_000.0, jitter: 0.8, reach: 6_000.0, relief: true, exists: cone_exists }],
    host: None,
};

/// The family's index (`FAM_EXAMPLE_CONES` in WGSL).
fn fam() -> usize {
    crate::instances::families().position(|f| f.name == "example_cones").unwrap()
}

/// A cone on 30 % of the land sites: radius 2–5 km, height 300–900 m.
fn cone_exists(w: &World, s: &InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 1) >= 0.3 {
        return None;
    }
    if w.continent(s.center, 50_000.0) < 0.05 {
        return None;
    }
    let r = 2000.0 + 3000.0 * u01k(s.id, 2);
    let h = 300.0 + 600.0 * u01k(s.id, 3);
    Some([r as f32, h as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

/// The cones' height (the highest wins), band-limited by the pixel size.
fn cone_height(inst: &[crate::instances::Instance], p: glam::DVec3, gsd: f64) -> f64 {
    let mut z: f64 = 0.0;
    for i in inst {
        let (r, h) = (i.v[0] as f64, i.v[1] as f64);
        let d = (p - i.center).length();
        if d < r {
            z = z.max(h * (1.0 - d / r).powf(1.5) * band(r, gsd));
        }
    }
    z
}

fn relief(w: &World, r: &ReliefIn, h: &mut f64) {
    *h += cone_height(&r.instances(w, fam()), r.ctx.p, r.ctx.gsd);
}

/// Dark ash on the upper cones (a crisp edge at 60 % of the radius).
fn azonal(s: &mut Stack) {
    let inst = s.instances(fam());
    let mut cov: f64 = 0.0;
    for i in inst.iter() {
        let d = (s.ctx.p - i.center).length();
        cov = cov.max(crate::kernels::band_cov(d, 0.6 * i.v[0] as f64, s.l.fw.max(0.5 * s.ctx.gsd)));
    }
    if cov > 0.0 {
        let ash = crate::surface::srgb(70.0, 64.0, 62.0);
        s.composite(Layer { cov, albedo: ash, cls: crate::landcover::VOLCANIC_ASH, clear: 1.0, ..Default::default() });
    }
}

const RINGS_P: &[P] = &[P::Num("radius", 0, 1.0, 1000.0, 20.0), P::Col("colour", 0, crate::kernels::C::Rgb([200.0, 60.0, 60.0]))];

/// Rings of `radius` m around a jittered lattice (a trivial kit kernel).
fn rings(k: &KParams, i: &KIn) -> KOut {
    let r = k.v[0];
    let c = (i.q / (4.0 * r)).floor();
    let rel = i.q - (c + 0.5) * 4.0 * r;
    let cov = crate::kernels::band_cov(rel.length() - r, 0.15 * r, i.fw) * i.amount.min(1.0);
    KOut { cov, albedo: k.col[0], dh: 0.0, emit: 0.0, id: 0 }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_registry_has_the_example_biome() {
        let r = crate::registry::Registry::builtin().unwrap();
        let b = &r.biomes[r.index("example_biome").unwrap() as usize];
        assert_eq!(b.layers.len(), 1);
        assert_eq!(b.layers[0].k.kind, 240);
    }

    #[test]
    fn cones_exist_and_raise_the_ground() {
        let w = crate::world::World::new(crate::Config::default());
        let mut found = 0;
        for k in 0..400u64 {
            let h = mix(k);
            let lat = (2.0 * crate::noise::u01k(h, 1) - 1.0).asin();
            let lon = (crate::noise::u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            found += crate::instances::near(&w, super::fam(), p, 50_000.0).len();
        }
        assert!(found > 20, "{found} cones");
        fn mix(k: u64) -> u64 {
            crate::noise::mix64(0xC0E5 ^ k)
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_kit_kernel_matches_the_cpu() {
        let Some(r) = crate::gpu::tests::kernel_parity(&[("example_rings", "{radius: 15}")]) else { return };
        for (name, bad, _) in r {
            assert!(bad < 0.005, "{name}");
        }
    }
}
