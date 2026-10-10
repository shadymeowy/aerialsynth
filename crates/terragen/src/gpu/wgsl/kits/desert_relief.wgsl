// ---------------------------------------------------------------- kit desert: helpers and relief
// The twin of `kits/desert.rs` (proxies, ridge fields, slab lattices, the relief operator), in
// the same order. Included in every module with pass A; the tile module's pass B uses the
// helpers too (`desert.wgsl`).

const DS_WIND1: u64 = 0xDE5E0001lu;
const DS_WIND2: u64 = 0xDE5E0002lu;
const DS_REG1: u64 = 0xDE5E0003lu;
const DS_REG2: u64 = 0xDE5E0004lu;
const DS_TRANS: u64 = 0xDE5E0010lu;
const DS_SEIF: u64 = 0xDE5E0011lu;
const DS_SMALL: u64 = 0xDE5E0012lu;
const DS_YARD: u64 = 0xDE5E0013lu;
const DS_YZONE: u64 = 0xDE5E0014lu;
const DS_STAR: u64 = 0xDE5E0020lu;
const DS_INS: u64 = 0xDE5E0021lu;
const DS_PLAYA: u64 = 0xDE5E0022lu;
const DS_BAR: u64 = 0xDE5E0023lu;

const DS_TRANS_LAM: f32 = 600.0;
const DS_SEIF_LAM: f32 = 1700.0;
const DS_SMALL_LAM: f32 = 240.0;
const DS_YARD_LAM: f32 = 180.0;
const DS_TRANS_EXT: vec3<f32> = vec3<f32>(1.2, 0.7, 0.7);
const DS_SEIF_EXT: vec3<f32> = vec3<f32>(1.5, 0.55, 0.55);
const DS_YARD_EXT: vec3<f32> = vec3<f32>(1.5, 0.45, 0.45);
const DS_TRANS_MEAN: f32 = 0.3711;
const DS_SEIF_MEAN: f32 = 0.1685;
const DS_YARD_MEAN: f32 = 0.1409;

const DS_STAR_CELL: f32 = 2200.0;
const DS_STAR_DENS: f32 = 0.8;
const DS_STAR_R0: f32 = 500.0;
const DS_STAR_R1: f32 = 950.0;
const DS_STAR_CV: f32 = 0.2383;
const DS_ARM_MEAN: f32 = 0.3125;
const DS_INS_CELL: f32 = 16000.0;
const DS_INS_DENS: f32 = 0.35;
const DS_INS_CV: f32 = 2.4046;
const DS_PLAYA_CELL: f32 = 36000.0;
const DS_PLAYA_DENS: f32 = 0.6;
const DS_PLAYA_FLAT: f32 = -0.25;

/// `desert::aridity`: 0 moist .. 1 hyper-arid (proxy of the atlas precipitation).
fn desert_aridity(moist: f32) -> f32 {
    return 1.0 - smoothstep1(0.16, 0.30, moist);
}

/// `desert::wind`: the direction the wind blows to (east, north) and its variability.
struct DesertWind {
    d: vec2<f32>,
    rho: f32,
}

fn desert_wind(p: vec3<f64>, lat: f32) -> DesertWind {
    let latd = abs(lat) * (180.0 / PI);
    var ns = 1.0;
    if (lat < 0.0) {
        ns = -1.0;
    }
    let trade = 1.0 - smoothstep1(25.0, 35.0, latd);
    let base = vec2<f32>(-0.7071, -0.7071 * ns) * trade + vec2<f32>(0.966, 0.259 * ns) * (1.0 - trade);
    let th = atan2(base.y, base.x) + 1.1 * perlin3(cfg.seed ^ DS_WIND1, p / 900000.0lf) + 0.45 * perlin3(cfg.seed ^ DS_WIND2, p / 260000.0lf);
    var o: DesertWind;
    o.d = vec2<f32>(cos(th), sin(th));
    o.rho = saturate1(0.5 + 0.9 * perlin3(cfg.seed ^ DS_REG1, p / 800000.0lf) + 0.3 * perlin3(cfg.seed ^ DS_REG2, p / 230000.0lf));
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
fn desert_ridges(seed: u64, q: vec3<f64>, ew: vec3<f32>, ec: vec3<f32>, up: vec3<f32>, k: f32, ext: vec3<f32>, kind: u32, lee: f32) -> f32 {
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
                var x = k * b + u01k(h, 4lu);
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
    let mean = amount * DS_STAR_DENS * DS_STAR_CV * 0.185 * er3 / (DS_STAR_CELL * DS_STAR_CELL);
    let e = smoothstep1(1.2 * gsd, 3.0 * gsd, DS_STAR_R0);
    if (e <= 0.0) {
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
        let hh = r * (0.12 + 0.13 * u01k(h, 7lu));
        let v = -st.rel[i];
        let d = length(v);
        if (d >= r) {
            continue;
        }
        let n = 3.0 + floor(3.0 * u01k(h, 8lu));
        let th = atan2(v.y, v.x);
        let arm = 0.5 + 0.5 * cos(n * (th - TAU * u01k(h, 9lu)) + 0.7 * sin(th + TAU * u01k(h, 10lu)));
        let a3 = DS_ARM_MEAN + (arm * arm * arm - DS_ARM_MEAN) * band(TAU * d / (3.0 * n), gsd);
        let x = d / (r * (0.3 + 0.7 * a3));
        if (x < 1.0) {
            best = max(best, hh * pow(1.0 - x, 1.5));
        }
    }
    return e * best + (1.0 - e) * mean;
}

/// `desert::inselbergs` (m).
fn desert_inselbergs(c: Ctx, m: f32) -> f32 {
    let gsd = c.gsd;
    let reach = 0.45 * DS_INS_CELL;
    let st = desert_slab_sites(cfg.seed ^ DS_INS, c.p, c.east, c.north, DS_INS_CELL, reach);
    var best = 0.0;
    for (var i = 0u; i < st.n; i++) {
        let h = st.id[i];
        if (u01k(h, 5lu) >= DS_INS_DENS) {
            continue;
        }
        let u6 = u01k(h, 6lu);
        let r = (500.0 + 2100.0 * u6 * u6) * m;
        if (r < 1.0) {
            continue;
        }
        let hh = min(r * (0.12 + 0.22 * u01k(h, 7lu)), 450.0);
        let ex = 1.0 + 0.7 * u01k(h, 8lu);
        let al = TAU * u01k(h, 9lu);
        let sa = sin(al);
        let ca = cos(al);
        let v = -st.rel[i];
        let vx = v.x * ca + v.y * sa;
        let vy = -v.x * sa + v.y * ca;
        let dn = sqrt(vx * vx / ex + vy * vy * ex);
        let th = atan2(vy, vx);
        let edge = 1.0 + 0.10 * sin(3.0 * th + TAU * u01k(h, 10lu)) + 0.06 * sin(5.0 * th + TAU * u01k(h, 11lu)) + 0.035 * sin(9.0 * th + TAU * u01k(h, 12lu));
        let x = dn / (r * edge);
        let e = smoothstep1(1.2 * gsd, 3.0 * gsd, r);
        var hx = 0.0;
        if (e > 0.0 && x < 1.6) {
            var dome = 0.0;
            if (x < 1.0) {
                dome = pow(max(1.0 - pow(x, 2.4), 0.0), 0.55);
            }
            let ped = 0.05 * (1.0 - smoothstep1(0.9, 1.6, x));
            hx = hh * (dome + ped);
        }
        let rb = min(0.95 * reach, max(1.6 * r, 3.0 * gsd));
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

/// `desert::playa_mask`.
fn desert_playa_mask(moist: f32, sand: f32, mountain: f32, mesa: f32) -> f32 {
    return desert_aridity(moist) * (1.0 - 0.85 * sand) * (1.0 - mountain) * (1.0 - smoothstep1(0.2, 0.5, mesa));
}

/// `desert::playa`: (normalized distance, radius, found 0/1) and its hash.
struct DesertPlaya {
    x: f32,
    r: f32,
    found: bool,
    id: u64,
}

fn desert_playa(p: vec3<f64>, east: vec3<f32>, north: vec3<f32>, m: f32) -> DesertPlaya {
    var best: DesertPlaya;
    best.found = false;
    best.x = 0.0;
    best.r = 0.0;
    best.id = 0lu;
    if (m <= 0.0) {
        return best;
    }
    let st = desert_slab_sites(cfg.seed ^ DS_PLAYA, p, east, north, DS_PLAYA_CELL, 0.45 * DS_PLAYA_CELL);
    for (var i = 0u; i < st.n; i++) {
        let h = st.id[i];
        if (u01k(h, 5lu) >= DS_PLAYA_DENS) {
            continue;
        }
        let r = (2000.0 + 5500.0 * u01k(h, 6lu)) * m;
        let ex = 1.0 + 0.8 * u01k(h, 7lu);
        let al = TAU * u01k(h, 8lu);
        let sa = sin(al);
        let ca = cos(al);
        let v = -st.rel[i];
        let vx = v.x * ca + v.y * sa;
        let vy = -v.x * sa + v.y * ca;
        let dn = sqrt(vx * vx / ex + vy * vy * ex);
        if (dn > 1.4 * 1.27 * r) {
            continue;
        }
        let ha = fbm(FBM_HILL_AMP, st.c[i], 20000.0) * fbm_norm(FBM_HILL_AMP) * 1.6;
        if (ha > DS_PLAYA_FLAT) {
            continue;
        }
        let th = atan2(vy, vx);
        let edge = 1.0 + 0.14 * sin(2.0 * th + TAU * u01k(h, 9lu)) + 0.08 * sin(3.0 * th + TAU * u01k(h, 10lu)) + 0.04 * sin(7.0 * th + TAU * u01k(h, 11lu));
        let x = dn / max(r * edge, 1e-3);
        if (!best.found || x < best.x) {
            best.found = true;
            best.x = x;
            best.r = r;
            best.id = h;
        }
    }
    return best;
}

/// `desert::relief`: dunes by wind regime (replacing the core's), inselbergs, playa floors,
/// yardangs.
fn desert_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {
    let gsd = c.gsd;
    let semi = 1.0 - smoothstep1(0.30, 0.45, r.moist);
    if (semi <= 0.0 && r.sand <= 0.0) {
        return;
    }
    let dh = cfg.dune_height;
    if (r.sand > 1e-3 && dh > 0.0) {
        let core = dh * r.sand * dunes(c, m, gsd);
        let wd = desert_wind(c.p, c.lat);
        let wr = desert_regime(wd.rho);
        let env = smoothstep1(0.35, 0.75, r.sand);
        let ew = c.east * wd.d.x + c.north * wd.d.y;
        let ec = cross(c.up, ew);
        var sum = 0.0;
        if (env > 0.0) {
            if (wr.x > 0.0) {
                let b = band(DS_TRANS_LAM, gsd);
                var v = DS_TRANS_MEAN;
                if (b > 0.0) {
                    let lee = clamp(2.5 * gsd / DS_TRANS_LAM, 0.14, 0.5);
                    v += (desert_ridges(cfg.seed ^ DS_TRANS, c.p / f64(2.0 * DS_TRANS_LAM), ew, ec, c.up, 2.0, DS_TRANS_EXT, 0u, lee) - DS_TRANS_MEAN) * b;
                }
                sum += wr.x * v;
            }
            if (wr.y > 0.0) {
                let b = band(DS_SEIF_LAM, gsd);
                var v = DS_SEIF_MEAN;
                if (b > 0.0) {
                    v += (desert_ridges(cfg.seed ^ DS_SEIF, c.p / f64(3.0 * DS_SEIF_LAM), ec, ew, c.up, 3.0, DS_SEIF_EXT, 1u, 0.5) - DS_SEIF_MEAN) * b;
                }
                sum += wr.y * 1.9 * v;
            }
            if (wr.y + wr.z > 0.0) {
                let b = band(DS_SMALL_LAM, gsd);
                var v = DS_TRANS_MEAN;
                if (b > 0.0) {
                    let lee = clamp(2.5 * gsd / DS_SMALL_LAM, 0.14, 0.5);
                    v += (desert_ridges(cfg.seed ^ DS_SMALL, c.p / f64(2.0 * DS_SMALL_LAM), ew, ec, c.up, 2.0, DS_TRANS_EXT, 0u, lee) - DS_TRANS_MEAN) * b;
                }
                sum += (wr.y + wr.z) * 0.14 * v;
            }
        }
        var star = 0.0;
        if (wr.z * env > 0.0) {
            star = desert_stars(c, wr.z * env) * (dh / 35.0);
        }
        *h += dh * r.sand * env * sum + star - core;
    }
    if (semi <= 0.0) {
        return;
    }
    let mi = semi * (1.0 - r.mountain) * (1.0 - 0.6 * r.sand) * (1.0 - smoothstep1(0.2, 0.5, r.mesa));
    if (mi > 0.0) {
        *h += desert_inselbergs(c, mi);
    }
    let mp = desert_playa_mask(r.moist, r.sand, r.mountain, r.mesa);
    let pl = desert_playa(c.p, c.east, c.north, mp);
    if (pl.found && pl.x < 1.0) {
        let k = 1.0 - smoothstep1(0.8, 1.0, pl.x);
        *h += (r.smooth_h - 2.0 - *h) * k;
    }
    let hyper = 1.0 - smoothstep1(0.05, 0.12, r.moist);
    if (hyper > 0.0 && r.sand < 0.2) {
        let zone = hyper * smoothstep1(0.1, 0.3, perlin3(cfg.seed ^ DS_YZONE, c.p / 120000.0lf)) * (1.0 - smoothstep1(0.02, 0.2, r.sand)) * (1.0 - r.mountain);
        if (zone > 0.0) {
            let b = band(DS_YARD_LAM, gsd);
            var v = DS_YARD_MEAN;
            if (b > 0.0) {
                let wd = desert_wind(c.p, c.lat);
                let ew = c.east * wd.d.x + c.north * wd.d.y;
                let ec = cross(c.up, ew);
                v += (desert_ridges(cfg.seed ^ DS_YARD, c.p / f64(2.0 * DS_YARD_LAM), ec, ew, c.up, 2.0, DS_YARD_EXT, 2u, 0.5) - DS_YARD_MEAN) * b;
            }
            *h += 12.0 * zone * v;
        }
    }
}
