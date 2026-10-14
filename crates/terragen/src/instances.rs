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
    if r == 0.0 {
        return near_point(w, fi, p);
    }
    near_scan(w, fi, p, r)
}

/// [`near`] by a scan of the lattice cells around the disc.
fn near_scan(w: &World, fi: usize, p: DVec3, r: f64) -> Vec<Instance> {
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

/// Side of the query cells of [`near_point`] (lattice cells of the family).
const QUERY_CELL: f64 = 0.5;

/// An existing instance and its lattice cell.
type CellInst = ((i64, i64, i64), Instance);

/// The instances a point query from the query cell `q` can find (`near_point`), by id.
type QueryKey = (u64, usize, i64, i64, i64);
static QUERY_LISTS: crate::memo::Memo<QueryKey, std::sync::Arc<[CellInst]>> = crate::memo::Memo::new(1 << 17);

/// [`near`] of a point (`r` = 0; host point evaluations, the kits' relief): the same instances,
/// selected from the list of the point's query cell (half a lattice cell; shared memo) by the
/// scan's rule.
fn near_point(w: &World, fi: usize, p: DVec3) -> Vec<Instance> {
    let f = family(fi);
    let qs = QUERY_CELL * f.cell;
    let q = (p / qs).floor();
    let list = QUERY_LISTS.get_or((w.cache_key, fi, q.x as i64, q.y as i64, q.z as i64), || {
        // every site the scan of a point of the query cell can visit and keep, by id
        let q0 = q * qs;
        let qm = q0 + DVec3::splat(0.5 * qs);
        let hd = 0.5 * 3f64.sqrt() * qs + 1.0;
        let rr = f.reach;
        let lo = ((q0 - DVec3::splat(rr + 0.5 * f.cell)) / f.cell).floor();
        let hi = ((q0 + DVec3::splat(qs + rr + 0.5 * f.cell)) / f.cell).floor();
        let mut v: Vec<CellInst> = Vec::new();
        for z in lo.z as i64..=hi.z as i64 {
            for y in lo.y as i64..=hi.y as i64 {
                for x in lo.x as i64..=hi.x as i64 {
                    let Some(s) = cached_site(w, fi, f, (x, y, z)) else { continue };
                    if (s.0.center - qm).length() > rr + hd {
                        continue;
                    }
                    if let Some(vv) = s.1 {
                        v.push(((x, y, z), Instance { id: s.0.id, center: s.0.center, v: vv }));
                    }
                }
            }
        }
        v.sort_by_key(|i| i.1.id);
        v.into()
    });
    let rr = f.reach;
    let lo = ((p - DVec3::splat(rr + 0.5 * f.cell)) / f.cell).floor();
    let hi = ((p + DVec3::splat(rr + 0.5 * f.cell)) / f.cell).floor();
    let (lo, hi) = ((lo.x as i64, lo.y as i64, lo.z as i64), (hi.x as i64, hi.y as i64, hi.z as i64));
    let mut out = Vec::new();
    for (c, i) in list.iter() {
        if c.0 < lo.0 || c.0 > hi.0 || c.1 < lo.1 || c.1 > hi.1 || c.2 < lo.2 || c.2 > hi.2 || (i.center - p).length() > rr {
            continue;
        }
        out.push(*i);
        if out.len() == K {
            break;
        }
    }
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

/// Whether family `f` is left out of the lists of a block whose finest pixel is `gsd` (m): its
/// instances reach at most one pixel there (planet-scale tiles, zooms ~0 to 5). Their relief is
/// below the pixel (a few metres of mean height) and their colour a few isolated pixels, while
/// the block's scan visits millions of lattice cells at zoom 0 (half a minute per tile) and
/// thousands of instances reach the block, of which [`K`] arbitrary ones would be kept. The
/// same `f32` comparison as the GPU's `instance_lists`.
pub fn sub_pixel(f: &Family, gsd: f32) -> bool {
    f.reach as f32 <= gsd
}

/// The finest pixel (m) of the 16-pixel block row `by` of a tile's pass-A grid (rows `oy - 2 +
/// j`, `j` < `na2`, of the global pixel grid at zoom `z`; as the GPU's `bin_gsd`, from the
/// tile's row table).
pub fn block_gsd(ell: &geodesy::Ellipsoid, z: u8, oy: f64, na2: usize, by: usize) -> f32 {
    let gsd = |j: usize| {
        let (lat, _) = geodesy::tiles::pixel_to_latlon(glam::DVec2::new(0.0, oy + j as f64 - 2.0 + 0.5), z, crate::TILE_SIZE as u32);
        geodesy::tiles::gsd_ew(lat, z, crate::TILE_SIZE as u32, ell) as f32
    };
    let j0 = by * 16;
    gsd(j0).min(gsd((j0 + 15).min(na2 - 1)))
}

/// The instances of every family that can reach the disc of radius `r` around `p` (only the
/// relief families with `relief_only`), in a block whose finest pixel is `gsd` (0: any; see
/// [`sub_pixel`]).
pub fn lists(w: &World, p: DVec3, r: f64, relief_only: bool, gsd: f32) -> Vec<Vec<Instance>> {
    families().enumerate().map(|(i, f)| if (relief_only && !f.relief) || sub_pixel(f, gsd) { Vec::new() } else { near(w, i, p, r) }).collect()
}

/// The WGSL of the families (in every module): constants `FAM_<NAME>`, their lattice
/// parameters and the existence dispatch (`<name>_exists` lives in the kit's relief WGSL).
pub fn wgsl() -> String {
    let fams: Vec<&Family> = families().collect();
    let mut s = format!("const NFAM: u32 = {}u;\n// (at least 1: array sizes)\nconst NFAM_1: u32 = {}u;\n", fams.len(), fams.len().max(1));
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
    s += &sw("fam_relief", "bool", "false", &|f| format!("{}", f.relief));
    s.push_str("fn family_exists(f: u32, site: InstSite) -> InstOut {\n    switch f {\n");
    let mut i = 0;
    for k in crate::kits::KITS.iter() {
        let n = k.families.len();
        if k.shared_exists && n > 0 {
            // (one case, one call: the kit's test is inlined once)
            let cases: Vec<String> = (i..i + n).map(|j| format!("{j}u")).collect();
            s.push_str(&format!("        case {}: {{ return {}_exists(f, site); }}\n", cases.join(", "), k.name));
        } else {
            for (j, f) in k.families.iter().enumerate() {
                s.push_str(&format!("        case {}u: {{ return {}_exists(site); }}\n", i + j, f.name));
            }
        }
        i += n;
    }
    s.push_str("        default: { var o: InstOut; o.ok = false; return o; }\n    }\n}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Point queries from the query cells' lists find the instances of the scan, in its order.
    #[test]
    fn point_lists_match_the_scan() {
        let w = World::new(crate::Config::default());
        let nf = families().count();
        let (mut n, mut found) = (0, 0);
        for i in 0..400u64 {
            let h = mix64(0x1A57 ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let p0 = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            for fi in 0..nf {
                // points near the instances of the area (and random ones)
                let f = family(fi);
                let mut at: Vec<DVec3> = near_scan(&w, fi, p0, 6.0 * f.cell).iter().take(3).map(|i| i.center).collect();
                at.push(p0);
                for (k, c) in at.iter().enumerate() {
                    let hk = mix64(h ^ k as u64 ^ ((fi as u64) << 8));
                    let p = *c + DVec3::new(u01k(hk, 1) - 0.5, u01k(hk, 2) - 0.5, u01k(hk, 3) - 0.5) * (1.6 * f.reach);
                    let a = near_point(&w, fi, p);
                    let b = near_scan(&w, fi, p, 0.0);
                    assert_eq!(format!("{a:?}"), format!("{b:?}"), "family {} at {p:?}", f.name);
                    n += 1;
                    found += a.len();
                }
            }
        }
        eprintln!("{n} point queries, {found} instances");
        assert!(nf == 0 || found > 100);
    }
}
