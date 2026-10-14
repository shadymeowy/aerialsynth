// ---------------------------------------------------------------- kit desert: helpers and relief
// The twin of `kits/desert.rs` (proxies, ridge fields, slab lattices, the relief operator), in
// the same order. Included in every module with pass A; the tile module's pass B uses the
// helpers too (`desert.wgsl`).

const DS_WIND2: u64 = 0xDE5E0002lu;
const DS_REG1: u64 = 0xDE5E0003lu;
const DS_REG2: u64 = 0xDE5E0004lu;
const DS_TRANS: u64 = 0xDE5E0010lu;
const DS_SEIF: u64 = 0xDE5E0011lu;
const DS_YARD: u64 = 0xDE5E0013lu;
const DS_YZONE: u64 = 0xDE5E0014lu;
const DS_STAR: u64 = 0xDE5E0020lu;
const DS_BAR: u64 = 0xDE5E0023lu;

const DS_TRANS_LAM: f32 = 600.0;
const DS_SEIF_LAM: f32 = 1700.0;
const DS_YARD_LAM: f32 = 180.0;
const DS_TRANS_EXT: vec3<f32> = vec3<f32>(1.3, 0.9, 0.9);
const DS_SEIF_EXT: vec3<f32> = vec3<f32>(1.5, 0.8, 0.8);
const DS_YARD_EXT: vec3<f32> = vec3<f32>(1.5, 0.6, 0.6);
const DS_TRANS_MEAN: f32 = 0.4821;
const DS_SEIF_MEAN: f32 = 0.2633;
const DS_YARD_MEAN: f32 = 0.2272;

const DS_STAR_CELL: f32 = 2600.0;
const DS_STAR_DENS: f32 = 1.0;
const DS_STAR_R0: f32 = 750.0;
const DS_STAR_R1: f32 = 1150.0;
const DS_STAR_CV: f32 = 0.3239;
const DS_ARM_MEAN: f32 = 0.375;
const DS_INS_CELL: f32 = 16000.0;
const DS_INS_REACH: f32 = 7200.0;
const DS_INS_DENS: f32 = 0.45;
const DS_INS_CV: f32 = 1.4049;
const DS_PLAYA_CELL: f32 = 36000.0;
const DS_PLAYA_DENS: f32 = 0.55;

/// `desert::aridity`: 0 moist .. 1 hyper-arid (proxy of the atlas precipitation).
fn desert_aridity(moist: f32) -> f32 {
    return 1.0 - smoothstep1(0.16, 0.30, moist);
}

/// `desert::wind`: the direction the wind blows to (east, north) and its variability.
struct DesertWind {
    d: vec2<f32>,
    rho: f32,
}

fn desert_wind(p: vec3<f64>, lat: f32, wa: vec2<f32>) -> DesertWind {
    let latd = abs(lat) * (180.0 / PI);
    var ns = 1.0;
    if (lat < 0.0) {
        ns = -1.0;
    }
    let trade = 1.0 - smoothstep1(25.0, 35.0, latd);
    let belt = vec2<f32>(-0.70710678, -0.70710678 * ns) * trade + vec2<f32>(0.966, 0.259 * ns) * (1.0 - trade);
    let v = wa + belt * 0.6;
    let th = atan2(v.y, v.x) + 0.3 * perlin3(cfg.seed ^ DS_WIND2, p / 260000.0lf);
    let speed = length(wa);
    var o: DesertWind;
    o.d = vec2<f32>(cos(th), sin(th));
    o.rho = saturate1(0.9 - 0.11 * speed + 0.6 * perlin3(cfg.seed ^ DS_REG1, p / 800000.0lf) + 0.25 * perlin3(cfg.seed ^ DS_REG2, p / 230000.0lf));
    return o;
}

/// `desert::regime`: weights of transverse, linear and star dunes.
fn desert_regime(rho: f32) -> vec3<f32> {
    let wt = 1.0 - smoothstep1(0.30, 0.42, rho);
    let ws = smoothstep1(0.62, 0.74, rho);
    return vec3<f32>(wt, 1.0 - wt - ws, ws);
}

/// `desert::profile`.
fn desert_profile(kind: u32, x: f32, lee: f32) -> f32 {
    if (kind == 0u) {
        if (x < 1.0 - lee) {
            let s = x / (1.0 - lee);
            return s * s * (3.0 - 2.0 * s);
        }
        let s = (1.0 - x) / lee;
        return s * s * (3.0 - 2.0 * s);
    }
    let t = 1.0 - abs(2.0 * x - 1.0);
    if (kind == 1u) {
        return pow(max((t - 0.35) / 0.65, 0.0), 1.3);
    }
    let y = max((t - 0.5) / 0.5, 0.0);
    return y * (2.0 - y);
}

/// `desert::ridges`: oriented ridges of a jittered 3D lattice with anisotropic kernels.
fn desert_ridges(seed: u64, q: vec3<f64>, ew: vec3<f32>, ec: vec3<f32>, up: vec3<f32>, k: f32, ext: vec3<f32>, kind: u32, lee: f32, warp: f32) -> f32 {
    let x0 = warp * perlin3(seed ^ 0x3A9lu, q * 1.3lf);
    let qf = floor(q);
    let ix = i64(qf.x);
    let iy = i64(qf.y);
    let iz = i64(qf.z);
    let f = vec3<f32>(q - qf);
    let rmax = max(ext.x, max(ext.y, ext.z));
    let r2 = rmax * rmax + 1e-6;
    let ia = 1.0 / (ext.x * ext.x);
    let ib = 1.0 / (ext.y * ext.y);
    let iu = 1.0 / (ext.z * ext.z);
    var v = 0.0;
    var wt = 0.0;
    for (var dz = -2; dz <= 2; dz++) {
        let gz = gully_gap(f.z, dz);
        if (gz > r2) {
            continue;
        }
        for (var dy = -2; dy <= 2; dy++) {
            let gzy = gz + gully_gap(f.y, dy);
            if (gzy > r2) {
                continue;
            }
            for (var dx = -2; dx <= 2; dx++) {
                if (gzy + gully_gap(f.x, dx) > r2) {
                    continue;
                }
                let h = hash3(seed, ix + i64(dx), iy + i64(dy), iz + i64(dz));
                let jit = vec3<f32>(u01k(h, 1lu), u01k(h, 2lu), u01k(h, 3lu)) * 0.5;
                let pp = f - vec3<f32>(f32(dx), f32(dy), f32(dz)) - jit;
                let a = dot(pp, ec);
                let b = dot(pp, ew);
                let u = dot(pp, up);
                let d2 = a * a * ia + b * b * ib + u * u * iu;
                if (d2 >= 1.0) {
                    continue;
                }
                let w = (1.0 - d2) * (1.0 - d2) * (1.0 - d2);
                var x = k * b + u01k(h, 4lu) + x0;
                x = x - floor(x);
                v += w * desert_profile(kind, x, lee);
                wt += w;
            }
        }
    }
    return v / max(wt, 0.3);
}

/// `desert::slab_sites`: ≤ 4 sites of a jittered 3D lattice within half a cell of the
/// ellipsoid and within `reach` of `p` (hash, offset east / north, surface point).
struct DesertSites {
    n: u32,
    id: array<u64, 4>,
    rel: array<vec2<f32>, 4>,
    c: array<vec3<f64>, 4>,
}

fn desert_slab_gap(f: f32, o: i32) -> f32 {
    let lo = f32(o) + 0.1;
    let hi = f32(o) + 0.9;
    let g = max(max(lo - f, f - hi), 0.0);
    return g * g;
}

fn desert_slab_sites(seed: u64, p: vec3<f64>, east: vec3<f32>, north: vec3<f32>, cell: f32, reach: f32) -> DesertSites {
    var s: DesertSites;
    s.n = 0u;
    let cd = f64(cell);
    let q = p / cd;
    let qf = floor(q);
    let ix = i64(qf.x);
    let iy = i64(qf.y);
    let iz = i64(qf.z);
    let f = vec3<f32>(q - qf);
    let r3 = (reach * reach + 0.25 * cell * cell) / (cell * cell);
    let ia2 = 1.0lf / (cfg.ell_a * cfg.ell_a);
    let ib2 = 1.0lf / (cfg.ell_b * cfg.ell_b);
    for (var dz = -1; dz <= 1; dz++) {
        let gz = desert_slab_gap(f.z, dz);
        for (var dy = -1; dy <= 1; dy++) {
            let gzy = gz + desert_slab_gap(f.y, dy);
            for (var dx = -1; dx <= 1; dx++) {
                if (gzy + desert_slab_gap(f.x, dx) > r3) {
                    continue;
                }
                let cx = ix + i64(dx);
                let cy = iy + i64(dy);
                let cz = iz + i64(dz);
                let h = hash3(seed, cx, cy, cz);
                let pt = vec3<f64>(
                    f64(cx) + 0.5lf + 0.8lf * (u01kd(h, 1lu) - 0.5lf),
                    f64(cy) + 0.5lf + 0.8lf * (u01kd(h, 2lu) - 0.5lf),
                    f64(cz) + 0.5lf + 0.8lf * (u01kd(h, 3lu) - 0.5lf),
                ) * cd;
                let s2 = (pt.x * pt.x + pt.y * pt.y) * ia2 + pt.z * pt.z * ib2;
                let sq = sqrt(s2);
                let hg = sqrt(dot(pt, pt)) * (1.0lf - 1.0lf / sq);
                if (abs(hg) >= 0.5lf * cd) {
                    continue;
                }
                let d = vec3<f32>(pt - p);
                let rel = vec2<f32>(dot(d, east), dot(d, north));
                if (dot(rel, rel) > reach * reach || s.n >= 4u) {
                    continue;
                }
                s.id[s.n] = h;
                s.rel[s.n] = rel;
                s.c[s.n] = pt / sq;
                s.n += 1u;
            }
        }
    }
    return s;
}

/// `desert::stars`: star dunes (m, for 35 m dunes).
fn desert_stars(c: Ctx, amount: f32) -> f32 {
    let gsd = c.gsd;
    let er3 = (DS_STAR_R0 + DS_STAR_R1) * (DS_STAR_R0 * DS_STAR_R0 + DS_STAR_R1 * DS_STAR_R1) / 4.0;
    let mean = amount * DS_STAR_DENS * DS_STAR_CV * 0.2 * er3 / (DS_STAR_CELL * DS_STAR_CELL);
    let e = smoothstep1(1.2 * gsd, 3.0 * gsd, DS_STAR_R0);
    if (e <= 0.02) {
        return mean;
    }
    let st = desert_slab_sites(cfg.seed ^ DS_STAR, c.p, c.east, c.north, DS_STAR_CELL, 0.45 * DS_STAR_CELL);
    var best = 0.0;
    for (var i = 0u; i < st.n; i++) {
        let h = st.id[i];
        if (u01k(h, 5lu) >= DS_STAR_DENS * amount) {
            continue;
        }
        let r = DS_STAR_R0 + (DS_STAR_R1 - DS_STAR_R0) * u01k(h, 6lu);
        let hh = r * (0.14 + 0.12 * u01k(h, 7lu));
        let v = -st.rel[i];
        let d = length(v);
        if (d >= r) {
            continue;
        }
        let n = 3.0 + floor(3.0 * u01k(h, 8lu));
        let th = atan2(v.y, v.x);
        let arm = 0.5 + 0.5 * cos(n * (th - TAU * u01k(h, 9lu)) + 0.7 * sin(th + TAU * u01k(h, 10lu)));
        let a2 = DS_ARM_MEAN + (arm * arm - DS_ARM_MEAN) * band(TAU * d / (3.0 * n), gsd);
        let x = d / (r * (0.42 + 0.58 * a2));
        if (x < 1.0) {
            best = max(best, hh * pow(1.0 - x, 1.5));
        }
    }
    return e * best + (1.0 - e) * mean;
}

/// `desert::site_fields`: moisture, mountain mask, hill amplitude (m), sand-sea and mesa masks,
/// continent value at a family site.
struct DesertSite {
    moist: f32,
    temp: f32,
    mountain: f32,
    hills: f32,
    sand: f32,
    mesa: f32,
    cont: f32,
}

fn desert_site_fields(c: vec3<f64>) -> DesertSite {
    let m = macro_at(c, 20000.0);
    var f: DesertSite;
    f.mountain = mountain_mask(m).x;
    f.hills = hill_amplitude(m);
    f.moist = clamp(m.moist, 0.0, 1.0);
    f.temp = m.temp;
    f.sand = (1.0 - smoothstep1(0.1, 0.28, f.moist)) * smoothstep1(-0.25, 0.15, m.sand);
    f.mesa = smoothstep1(0.1, 0.45, m.mesa);
    f.cont = m.cont;
    return f;
}

/// `desert::playa_exists`.
fn desert_playas_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 5lu) >= DS_PLAYA_DENS) {
        return o;
    }
    let a = atlas_sample(vec3<f32>(normalize(s.center)));
    if (moisture_index(a.precip_mm, a.temp_c) > 0.33 || a.temp_c < 2.0 || a.coast_km < 0.0) {
        return o;
    }
    let f = desert_site_fields(s.center);
    if (desert_aridity(f.moist) < 0.5 || f.temp < 4.0 || f.cont < 0.05 || f.mountain > 0.1 || f.hills > 130.0 || f.sand > 0.4 || f.mesa > 0.3) {
        return o;
    }
    o.ok = true;
    o.v = array<f32, 8>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    return o;
}

/// `desert::inselberg_exists`.
fn desert_inselbergs_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 5lu) >= DS_INS_DENS) {
        return o;
    }
    let a = atlas_sample(vec3<f32>(normalize(s.center)));
    if ((atlas_litho(a) != 2u && u01k(s.id, 13lu) >= 0.3) || moisture_index(a.precip_mm, a.temp_c) > 0.52 || a.coast_km < 0.0) {
        return o;
    }
    let f = desert_site_fields(s.center);
    if (f.moist > 0.42 || f.cont < 0.05 || f.mountain > 0.15 || f.sand > 0.6 || f.mesa > 0.3) {
        return o;
    }
    o.ok = true;
    o.v = array<f32, 8>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    return o;
}

/// `desert::offset`.
fn desert_offset(p: vec3<f64>, center: vec3<f64>, east: vec3<f32>, north: vec3<f32>) -> vec2<f32> {
    let d = vec3<f32>(p - center);
    return vec2<f32>(dot(d, east), dot(d, north));
}

/// `desert::ins_shape`: (normalized distance, radius, height).
fn desert_ins_shape(id: u64, v: vec2<f32>) -> vec3<f32> {
    let u6 = u01k(id, 6lu);
    let r = 500.0 + 2100.0 * u6 * u6;
    let hh = min(r * (0.18 + 0.22 * u01k(id, 7lu)), 600.0);
    let ex = 1.0 + 0.7 * u01k(id, 8lu);
    let al = TAU * u01k(id, 9lu);
    let sa = sin(al);
    let ca = cos(al);
    let vx = v.x * ca + v.y * sa;
    let vy = -v.x * sa + v.y * ca;
    let dn = sqrt(vx * vx / ex + vy * vy * ex);
    let th = atan2(vy, vx);
    let edge = 1.0 + 0.10 * sin(3.0 * th + TAU * u01k(id, 10lu)) + 0.06 * sin(5.0 * th + TAU * u01k(id, 11lu)) + 0.035 * sin(9.0 * th + TAU * u01k(id, 12lu));
    return vec3<f32>(dn / (r * edge), r, hh);
}

/// `desert::ins_profile`.
fn desert_ins_profile(x: f32) -> f32 {
    var dome = 0.0;
    if (x < 1.0) {
        dome = pow(max(1.0 - pow(x, 1.5), 0.0), 1.2);
    }
    return dome + 0.04 * (1.0 - smoothstep1(0.9, 1.6, x));
}

/// `desert::inselbergs` (m).
fn desert_inselbergs(l: InstList, c: Ctx) -> f32 {
    let gsd = c.gsd;
    var best = 0.0;
    for (var i = 0u; i < l.n; i++) {
        let v = desert_offset(c.p, l.items[i].center, c.east, c.north);
        let sh = desert_ins_shape(l.items[i].id, v);
        let x = sh.x;
        let r = sh.y;
        let hh = sh.z;
        let e = smoothstep1(1.2 * gsd, 3.0 * gsd, r);
        var hx = 0.0;
        if (e > 0.0 && x < 1.6) {
            hx = hh * desert_ins_profile(x);
        }
        let rb = min(0.95 * DS_INS_REACH, max(1.6 * r, 3.0 * gsd));
        let hp = 2.0 * hh * r * r * DS_INS_CV / (PI * rb * rb);
        let db = length(v) / rb;
        var blob = 0.0;
        if (db < 1.0) {
            blob = hp * (1.0 - db * db);
        }
        best = max(best, e * hx + (1.0 - e) * blob);
    }
    return best;
}

/// `desert::playa_shape`: (normalized distance, radius).
fn desert_playa_shape(id: u64, v: vec2<f32>) -> vec2<f32> {
    let r = 2000.0 + 4500.0 * u01k(id, 6lu);
    let ex = 1.0 + 0.8 * u01k(id, 7lu);
    let al = TAU * u01k(id, 8lu);
    let sa = sin(al);
    let ca = cos(al);
    let vx = v.x * ca + v.y * sa;
    let vy = -v.x * sa + v.y * ca;
    let dn = sqrt(vx * vx / ex + vy * vy * ex);
    let th = atan2(vy, vx);
    let edge = 1.0 + 0.14 * sin(2.0 * th + TAU * u01k(id, 9lu)) + 0.08 * sin(3.0 * th + TAU * u01k(id, 10lu)) + 0.04 * sin(7.0 * th + TAU * u01k(id, 11lu));
    return vec2<f32>(dn / (r * edge), r);
}

/// `desert::playa`: the nearest playa (normalized distance, radius, found) and its hash.
struct DesertPlaya {
    x: f32,
    r: f32,
    found: bool,
    id: u64,
}

fn desert_playa(l: InstList, p: vec3<f64>, east: vec3<f32>, north: vec3<f32>) -> DesertPlaya {
    var best: DesertPlaya;
    best.found = false;
    best.x = 0.0;
    best.r = 0.0;
    best.id = 0lu;
    for (var i = 0u; i < l.n; i++) {
        let xr = desert_playa_shape(l.items[i].id, desert_offset(p, l.items[i].center, east, north));
        if (!best.found || xr.x < best.x) {
            best.found = true;
            best.x = xr.x;
            best.r = xr.y;
            best.id = l.items[i].id;
        }
    }
    return best;
}

/// `desert::relief`: dunes by wind regime (replacing the core's), inselbergs, playa floors,
/// yardangs.
fn desert_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {
    let gsd = c.gsd;
    let dh = cfg.dune_height;
    if (r.sand > 1e-3 && dh > 0.0) {
        let core = dh * r.sand * dunes(c, m, gsd);
        let wd = desert_wind(c.p, c.lat, atlas_sample(vec3<f32>(normalize(c.p))).wind);
        let wr = desert_regime(wd.rho);
        let env = smoothstep1(0.35, 0.75, r.sand);
        let ew = c.east * wd.d.x + c.north * wd.d.y;
        let ec = cross(c.up, ew);
        var sum = 0.0;
        if (env > 0.0) {
            let wtr = wr.x + 0.5 * wr.z;
            if (wtr > 0.0) {
                let b = band(DS_TRANS_LAM, gsd);
                var v = DS_TRANS_MEAN;
                if (b > 0.0) {
                    let lee = clamp(2.5 * gsd / DS_TRANS_LAM, 0.14, 0.5);
                    v += (desert_ridges(cfg.seed ^ DS_TRANS, c.p / f64(2.0 * DS_TRANS_LAM), ew, ec, c.up, 2.0, DS_TRANS_EXT, 0u, lee, 0.18) - DS_TRANS_MEAN) * b;
                }
                sum += wtr * 1.4 * v;
            }
            if (wr.y > 0.0) {
                let b = band(DS_SEIF_LAM, gsd);
                var v = DS_SEIF_MEAN;
                if (b > 0.0) {
                    v += (desert_ridges(cfg.seed ^ DS_SEIF, c.p / f64(3.0 * DS_SEIF_LAM), ec, ew, c.up, 3.0, DS_SEIF_EXT, 1u, 0.5, 0.22) - DS_SEIF_MEAN) * b;
                }
                sum += wr.y * 1.9 * v;
            }
        }
        var star = 0.0;
        if (wr.z * env > 0.0) {
            star = desert_stars(c, wr.z * env) * (dh / 35.0);
        }
        *h += dh * r.sand * env * sum + star - 0.5 * core;
    }
    let ins = inst_list(FAM_DESERT_INSELBERGS, r.blk, c.p);
    if (ins.n > 0u) {
        *h += desert_inselbergs(ins, c);
    }
    let pl = desert_playa(inst_list(FAM_DESERT_PLAYAS, r.blk, c.p), c.p, c.east, c.north);
    if (pl.found && pl.x < 1.0) {
        let k = 1.0 - smoothstep1(0.8, 1.0, pl.x);
        *h += (r.smooth_h - 2.0 - *h) * k;
    }
    let hyper = (1.0 - smoothstep1(0.05, 0.12, r.moist)) * smoothstep1(-2.0, 2.0, r.temp);
    if (hyper > 0.0 && r.sand < 0.2) {
        let zone = hyper * smoothstep1(0.1, 0.3, perlin3(cfg.seed ^ DS_YZONE, c.p / 120000.0lf)) * (1.0 - smoothstep1(0.02, 0.2, r.sand)) * (1.0 - r.mountain);
        if (zone > 0.0) {
            let b = band(DS_YARD_LAM, gsd);
            var v = DS_YARD_MEAN;
            if (b > 0.0) {
                let wd = desert_wind(c.p, c.lat, atlas_sample(vec3<f32>(normalize(c.p))).wind);
                let ew = c.east * wd.d.x + c.north * wd.d.y;
                let ec = cross(c.up, ew);
                v += (desert_ridges(cfg.seed ^ DS_YARD, c.p / f64(2.0 * DS_YARD_LAM), ec, ew, c.up, 2.0, DS_YARD_EXT, 2u, 0.5, 0.1) - DS_YARD_MEAN) * b;
            }
            *h += 12.0 * zone * v;
        }
    }
}
