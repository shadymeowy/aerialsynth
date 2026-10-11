//! Sparse instance lattices (`docs/design/terrain-next.md` §3.4): landforms and objects with a
//! bounded reach (volcanoes, atolls, inselbergs, kettle lakes, cinder cones, quarries) are the
//! sites of a 3D jittered lattice per family (seamless on the sphere), each existing by an
//! analytic test at its centre. Each 16 × 16-pixel block lists the ≤ [`K`] instances of a family
//! that can reach it; a sample evaluates only those. The WGSL twin is `gpu/wgsl/instances.wgsl`.

use crate::noise::*;
use crate::world::World;
use glam::DVec3;

/// At most this many instances of a family reach one block.
pub const K: usize = 8;

/// An instance site: its id and the surface point below its lattice site, with the local frame
/// there.
#[derive(Clone, Copy, Debug)]
pub struct InstSite {
    pub id: u64,
    pub center: DVec3,
    pub up: DVec3,
    pub east: DVec3,
    pub north: DVec3,
    pub lat: f64,
    pub lon: f64,
}

/// An existing instance: id, centre and the parameters its existence test returned.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub id: u64,
    pub center: DVec3,
    pub v: [f32; 8],
}

/// A family of instances (`Kit::families`).
pub struct Family {
    /// `<kit>_<what>`; WGSL: `fn <name>_exists(site: InstSite) -> InstOut`
    pub name: &'static str,
    /// lattice cell (m)
    pub cell: f64,
    /// site jitter (≤ 0.9)
    pub jitter: f64,
    /// how far an instance reaches from its centre (m; ≤ `cell`)
    pub reach: f64,
    /// pass A (relief: also shapes the drainage) or pass B only
    pub relief: bool,
    /// existence and parameters from the site (analytic fields only); None: no instance
    pub exists: fn(&World, &InstSite) -> Option<[f32; 8]>,
}

/// The families of all kits, in order (`FAM_<NAME>` in WGSL is the index).
pub fn families() -> impl Iterator<Item = &'static Family> {
    crate::kits::KITS.iter().flat_map(|k| k.families.iter())
}

pub fn family(i: usize) -> &'static Family {
    families().nth(i).expect("a family index")
}

/// The lattice key of a family (from its name).
pub fn key(f: &Family) -> u64 {
    f.name.bytes().fold(0xF4A1_17E5_u64, |h, b| mix64(h ^ b as u64))
}

/// The site of lattice cell `c` of family `f` (None: too far from the surface to be an instance
/// there: each surface point sees a site at most once). The centre is the geocentric projection
/// of the lattice site on the ellipsoid; latitude, longitude and frame are those of that surface
/// point (closed forms, the same in WGSL: `inst_site`).
pub fn site(w: &World, f: &Family, c: (i64, i64, i64)) -> Option<InstSite> {
    let (id, pt) = worley3_site(w.seed() ^ key(f), c, f.cell, f.jitter);
    let (a, b) = (w.ell.a, w.ell.b);
    let len = pt.length().max(1.0);
    let d = pt / len;
    let rho2 = d.x * d.x + d.y * d.y;
    let re = a * b / (b * b * rho2 + a * a * d.z * d.z).sqrt();
    if (len - re).abs() >= 0.5 * f.cell {
        return None;
    }
    let center = d * re;
    let e2 = 1.0 - (b * b) / (a * a);
    let wxy = (center.x * center.x + center.y * center.y).sqrt();
    let lat = center.z.atan2((1.0 - e2) * wxy);
    let lon = center.y.atan2(center.x);
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    let up = DVec3::new(cl * co, cl * so, sl);
    let east = DVec3::new(-so, co, 0.0);
    let north = DVec3::new(-sl * co, -sl * so, cl);
    Some(InstSite { id, center, up, east, north, lat, lon })
}

/// The instances of family `fi` that can reach the disc of radius `r` around `p`, by id.
pub fn near(w: &World, fi: usize, p: DVec3, r: f64) -> Vec<Instance> {
    let f = family(fi);
    let rr = r + f.reach;
    let lo = ((p - DVec3::splat(rr + 0.5 * f.cell)) / f.cell).floor();
    let hi = ((p + DVec3::splat(rr + 0.5 * f.cell)) / f.cell).floor();
    let mut out = Vec::new();
    for z in lo.z as i64..=hi.z as i64 {
        for y in lo.y as i64..=hi.y as i64 {
            for x in lo.x as i64..=hi.x as i64 {
                let Some(s) = cached_site(w, fi, f, (x, y, z)) else { continue };
                if (s.0.center - p).length() > rr {
                    continue;
                }
                if let Some(v) = s.1 {
                    out.push(Instance { id: s.0.id, center: s.0.center, v });
                }
            }
        }
    }
    out.sort_unstable_by_key(|i| i.id);
    // (as the GPU's lists: the first K by id; more is a family too dense for its cell)
    out.truncate(K);
    out
}

/// A family's site and existence (cached per thread: pure functions of the world and the cell).
fn cached_site(w: &World, fi: usize, f: &Family, c: (i64, i64, i64)) -> Option<(InstSite, Option<[f32; 8]>)> {
    type Key = (u64, usize, i64, i64, i64);
    type Site = Option<(InstSite, Option<[f32; 8]>)>;
    thread_local! {
        static CACHE: std::cell::RefCell<FxHashMap<Key, Site>> = Default::default();
    }
    let key = (w.cache_key, fi, c.0, c.1, c.2);
    if let Some(v) = CACHE.with(|m| m.borrow().get(&key).copied()) {
        return v;
    }
    let v = site(w, f, c).map(|s| (s, (f.exists)(w, &s)));
    CACHE.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() > 200_000 {
            m.clear();
        }
        m.insert(key, v);
    });
    v
}

/// The instances of every family that can reach the disc of radius `r` around `p` (only the
/// relief families with `relief_only`).
pub fn lists(w: &World, p: DVec3, r: f64, relief_only: bool) -> Vec<Vec<Instance>> {
    families().enumerate().map(|(i, f)| if relief_only && !f.relief { Vec::new() } else { near(w, i, p, r) }).collect()
}

/// The WGSL of the families (in every module): constants `FAM_<NAME>`, their lattice
/// parameters and the existence dispatch (`<name>_exists` lives in the kit's relief WGSL).
pub fn wgsl() -> String {
    let fams: Vec<&Family> = families().collect();
    let mut s = format!("const NFAM: u32 = {}u;\n", fams.len());
    for (i, f) in fams.iter().enumerate() {
        s.push_str(&format!("const FAM_{}: u32 = {i}u;\n", f.name.to_uppercase()));
    }
    let sw = |name: &str, ty: &str, def: &str, val: &dyn Fn(&Family) -> String| {
        let mut o = format!("fn {name}(f: u32) -> {ty} {{\n    switch f {{\n");
        for (i, f) in fams.iter().enumerate() {
            o.push_str(&format!("        case {i}u: {{ return {}; }}\n", val(f)));
        }
        o.push_str(&format!("        default: {{ return {def}; }}\n    }}\n}}\n"));
        o
    };
    s += &sw("fam_cell", "f64", "1.0lf", &|f| format!("{:?}lf", f.cell));
    s += &sw("fam_jitter", "f32", "0.0", &|f| format!("{:?}", f.jitter as f32));
    s += &sw("fam_reach", "f32", "0.0", &|f| format!("{:?}", f.reach as f32));
    s += &sw("fam_key", "u64", "0lu", &|f| format!("{:#x}lu", key(f)));
    s.push_str("fn family_exists(f: u32, site: InstSite) -> InstOut {\n    switch f {\n");
    for (i, f) in fams.iter().enumerate() {
        s.push_str(&format!("        case {i}u: {{ return {}_exists(site); }}\n", f.name));
    }
    s.push_str("        default: { var o: InstOut; o.ok = false; return o; }\n    }\n}\n");
    s
}
