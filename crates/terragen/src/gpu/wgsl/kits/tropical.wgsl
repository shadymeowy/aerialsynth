// The tropical and savanna kit (`kits/tropical.rs`): the WGSL twins of its kernels, in the same
// order with the same operations.

const TROPICAL_R0: f64 = 6371000.0lf;

fn tropical_logistic(n: f32, sigma: f32) -> f32 {
    return 1.0 / (1.0 + exp(-1.7 * n / sigma));
}

fn tropical_wrap(a: f32) -> f32 {
    return a - TAU * floor(a / TAU + 0.5);
}

/// The sample on the reference sphere and its tangent frame (`sphere_frame`).
struct TropicalFrame {
    pu: vec3<f64>,
    east: vec3<f32>,
    north: vec3<f32>,
}

fn tropical_sphere_frame(p: vec3<f64>) -> TropicalFrame {
    var f: TropicalFrame;
    let r = sqrt(dot(p, p));
    f.pu = p * (TROPICAL_R0 / r);
    let up = vec3<f32>(f.pu / TROPICAL_R0);
    let el = sqrt(up.x * up.x + up.y * up.y);
    f.east = vec3<f32>(1.0, 0.0, 0.0);
    if (el > 1e-6) {
        f.east = vec3<f32>(-up.y / el, up.x / el, 0.0);
    }
    f.north = cross(up, f.east);
    return f;
}

/// A site of the instance lattice (`sphere_site`): `ok` false outside the slab.
struct TropicalSite {
    ok: bool,
    h: u64,
    c: vec3<f64>,
}

fn tropical_sphere_site(seed: u64, c: vec3<i64>, cell: f32, jit: f32) -> TropicalSite {
    var s: TropicalSite;
    s.h = hash3(seed, c.x, c.y, c.z);
    let j = vec3<f32>(0.5) + jit * (vec3<f32>(u01k(s.h, 1lu), u01k(s.h, 2lu), u01k(s.h, 3lu)) - 0.5);
    let sp = (vec3<f64>(f64(c.x), f64(c.y), f64(c.z)) + vec3<f64>(j)) * f64(cell);
    let rs = sqrt(dot(sp, sp));
    s.ok = abs(rs - TROPICAL_R0) < 0.5lf * f64(cell);
    s.c = sp * (TROPICAL_R0 / rs);
    return s;
}

fn tropical_thin(u: f32, dens: f32) -> f32 {
    return sqrt(smoothstep1(0.0, 0.3, (dens - u) / max(dens, 1e-6)));
}

// ---------------------------------------------------------------- tropical_crowns

fn tropical_crowns(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    var out = kout_none();
    let cell = k.v[0];
    let dens = k.v[1] * i.amount;
    if (dens <= 0.0) {
        return out;
    }
    let sh = u32(k.v[2]);
    let jit = k.v[7];
    let cf = floor(i.q / cell);
    let ix = i64(cf.x);
    let iy = i64(cf.y);
    var best = -FMAX;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let h = hash2(k.seed, ix + i64(dx), iy + i64(dy));
            let u = u01k(h, 3lu);
            if (u >= dens) {
                continue;
            }
            let c = (cf + vec2<f32>(f32(dx), f32(dy)) + 0.5 + jit * (vec2<f32>(u01k(h, 1lu), u01k(h, 2lu)) - 0.5)) * cell;
            let f = tropical_thin(u, dens);
            let r = (k.v[3] + (k.v[4] - k.v[3]) * u01k(h, 4lu)) * f;
            let rel = kcell_rel(i.q, cell, cf, dx, dy, jit, h);
            let d = length(rel);
            if (d > r + i.fw) {
                continue;
            }
            if (k.v[8] > 0.0) {
                let cq = vec2<f64>(c) / f64(k.v[8]);
                let g = perlin3(k.seed ^ 0x6Clu, vec3<f64>(cq.x, cq.y, 0.37lf)) + 0.5 * perlin3(k.seed ^ 0x6Dlu, vec3<f64>(2.3lf * cq.x, 2.3lf * cq.y, 0.71lf));
                if (tropical_logistic(g, 0.3) >= k.v[9]) {
                    continue;
                }
            }
            var re = r;
            if (sh == 4u) {
                let a = atan2(rel.y, rel.x) + u01k(h, 9lu) * 6.3;
                re *= 0.8 + 0.2 * cos(7.0 * a);
            }
            let cov = kband_cov(d, re, i.fw) * select(0.0, 1.0, d <= re + i.fw);
            if (cov <= 0.0) {
                continue;
            }
            let x = min(d / max(re, 1e-6), 1.0);
            let hh = (k.v[5] + (k.v[6] - k.v[5]) * u01k(h, 5lu)) * (0.5 + 0.5 * f) * crown_profile(sh, x);
            if (hh > best) {
                best = hh;
                let uc = u01k(h, 6lu);
                var base = k.col[0];
                if (uc < k.v[11]) {
                    base = k.col[2];
                } else if (uc < k.v[11] + k.v[10]) {
                    base = k.col[1];
                }
                let hv = u01k(h, 8lu) - 0.5;
                let va = k.v[12];
                let hue = vec3<f32>(1.0 + 0.5 * va * hv, 1.0 + 0.15 * va * (u01k(h, 10lu) - 0.5), 1.0 - 0.45 * va * hv);
                let tint = 1.0 + va * (u01k(h, 7lu) - 0.5);
                let wl = band(1.2, i.gsd);
                var leaf = 1.0;
                if (wl > 0.0) {
                    leaf = 1.0 + k.v[13] * 0.3 * perlin3(h, i.p * (1.0lf / 1.2lf)) * wl;
                }
                out.albedo = base * hue * tint * leaf * (0.74 + 0.34 * (1.0 - x * x));
                out.dh = hh;
                out.id = h;
            }
            out.cov = max(out.cov, cov);
        }
    }
    return out;
}

// ---------------------------------------------------------------- tropical_fire

fn tropical_fire_colour(k: KP, age: f32) -> vec3<f32> {
    if (age < 0.5) {
        return k.col[0] + (k.col[1] - k.col[0]) * (2.0 * age);
    }
    return k.col[1] + (k.col[2] - k.col[1]) * (2.0 * age - 1.0);
}

fn tropical_fire(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    let sc = f64(k.v[0]);
    var n = 0.0;
    var a = 1.0;
    var norm = 0.0;
    var lam = sc;
    for (var o = 0u; o < 3u; o++) {
        n += a * perlin3(k.seed ^ (0xF1lu + u64(o)), i.p * (1.0lf / lam)) * band(f32(lam), i.gsd);
        norm += a * a;
        a *= 0.55;
        lam *= 0.43lf;
    }
    let u = tropical_logistic(n / sqrt(norm), 0.27);
    let frac = k.v[1] * i.amount;
    let w = clamp(2.0 * i.fw / k.v[0], 1e-3, 0.5);
    let m = frac - u;
    let cov = smoothstep1(-w, w, m);
    if (cov <= 0.0) {
        return kout_none();
    }
    let ua = tropical_logistic(perlin3(k.seed ^ 0xA6Elu, i.p * (1.0lf / (1.7lf * sc))), 0.27);
    let ages = max(round(k.v[2]), 1.0);
    let t = ua * ages;
    let bin = min(floor(t), ages - 1.0);
    let wa = clamp(3.0 * i.fw / k.v[0] * ages, 1e-3, 0.5);
    let fb = t - bin;
    var col = tropical_fire_colour(k, (bin + 0.5) / ages);
    if (bin + 1.0 < ages) {
        col = col + (tropical_fire_colour(k, (bin + 1.5) / ages) - col) * smoothstep1(1.0 - wa, 1.0, fb);
    }
    if (bin > 0.0) {
        col = col + (tropical_fire_colour(k, (bin - 0.5) / ages) - col) * (1.0 - smoothstep1(0.0, wa, fb));
    }
    let fresh = 1.0 - smoothstep1(0.0, 0.5, (bin + 0.5) / ages - 0.1);
    if (fresh > 0.0) {
        let wl = band(0.06 * k.v[0], i.gsd);
        if (wl > 0.0) {
            let s = perlin3(k.seed ^ 0xA54lu, i.p * (1.0lf / (0.06lf * sc)));
            col = col + (vec3<f32>(0.24, 0.23, 0.21) - col) * (k.v[3] * smoothstep1(0.15, 0.55, s) * wl * fresh);
        }
        col *= 1.0 - 0.35 * fresh * (1.0 - smoothstep1(0.0, 0.04, m)) * band(0.04 * k.v[0], i.gsd);
    }
    col *= 1.0 + k.v[4] * (2.0 * u - 1.0);
    let mw = band(0.2 * k.v[0], i.gsd);
    if (mw > 0.0) {
        col *= 1.0 + 0.22 * perlin3(k.seed ^ 0xB07lu, i.p * (1.0lf / (0.2lf * sc))) * mw;
    }
    var o = kout_none();
    o.cov = cov;
    o.albedo = col;
    o.id = u64(bin) + 1lu;
    return o;
}

// ---------------------------------------------------------------- tropical_swidden

fn tropical_swidden(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    let cell = k.v[0];
    let wq = vec2<f64>(i.q) / (0.4lf * f64(cell));
    let wn = vec2<f32>(perlin3(k.seed ^ 0x51lu, vec3<f64>(wq.x, wq.y, 0.5lf)), perlin3(k.seed ^ 0x52lu, vec3<f64>(wq.x, wq.y, 1.5lf)));
    let qw = i.q + wn * (k.v[4] * cell);
    let wc = worley2(k.seed, qw, cell, k.v[3]);
    if (u01k(wc.id, 1lu) >= k.v[1] * i.amount) {
        return kout_none();
    }
    let e = worley2_edge_dist(wc, qw);
    let cov = clamp((e - k.v[2] * cell) / max(i.fw, 1e-6) + 0.5, 0.0, 1.0);
    if (cov <= 0.0) {
        return kout_none();
    }
    let st = u01k(wc.id, 2lu);
    let tint = 1.0 + k.v[6] * (u01k(wc.id, 3lu) - 0.5);
    let w3 = band(3.0, i.gsd);
    var n3 = 0.0;
    if (w3 > 0.0) {
        n3 = perlin3(wc.id, vec3<f64>(f64(i.q.x) / 3.0lf, f64(i.q.y) / 3.0lf, 0.3lf)) * w3;
    }
    var o = kout_none();
    o.cov = cov;
    o.id = wc.id;
    if (st < 0.3) {
        o.albedo = k.col[0] * tint * (1.0 + 0.5 * n3);
    } else if (st < 0.65) {
        o.albedo = k.col[1] * tint * (1.0 + 0.3 * n3);
        o.dh = 0.4;
    } else {
        let age = u01k(wc.id, 4lu);
        o.albedo = k.col[2] * tint * (0.85 + 0.2 * age) * (1.0 + 0.35 * n3);
        o.dh = k.v[5] * (0.4 + 0.6 * age) * (0.75 + 0.25 * n3);
    }
    return o;
}

// ---------------------------------------------------------------- tropical_waterhole

fn tropical_waterhole(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    var out = kout_none();
    let cell = k.v[0];
    let dens = k.v[1] * i.amount;
    if (dens <= 0.0) {
        return out;
    }
    let ground = k.v[2] >= 0.5;
    var reach = k.v[4] * 1.3;
    if (ground) {
        reach = max(k.v[6], k.v[4] * k.v[8] * 1.3);
    }
    let fr = tropical_sphere_frame(i.p);
    let q = fr.pu / f64(cell);
    let qf = floor(q);
    let fq = vec3<f32>(q - qf);
    let half = 0.5 * k.v[9];
    let lr = (reach + i.fw) / cell;
    let lim = lr * lr + 0.25;
    var best = FMAX;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let g = vec3<f32>(nb_gap(fq.x, dx, half), nb_gap(fq.y, dy, half), nb_gap(fq.z, dz, half));
                if (dot(g, g) > lim) {
                    continue;
                }
                let si = tropical_sphere_site(k.seed, vec3<i64>(i64(qf.x) + i64(dx), i64(qf.y) + i64(dy), i64(qf.z) + i64(dz)), cell, k.v[9]);
                if (!si.ok) {
                    continue;
                }
                let h = si.h;
                let u = u01k(h, 4lu);
                if (u >= dens) {
                    continue;
                }
                let rel3 = vec3<f32>(fr.pu - si.c);
                let rel = vec2<f32>(dot(rel3, fr.east), dot(rel3, fr.north));
                let d = length(rel);
                let f = tropical_thin(u, dens);
                let r = (k.v[3] + (k.v[4] - k.v[3]) * u01k(h, 5lu)) * f;
                if (d > reach * f + i.fw) {
                    continue;
                }
                let th = atan2(rel.y, rel.x);
                let lobe = 1.0 + 0.12 * sin(3.0 * th + 6.3 * u01k(h, 6lu)) + 0.07 * sin(5.0 * th + 6.3 * u01k(h, 7lu));
                let rp = r * lobe;
                if (!ground) {
                    let cov = kband_cov(d, rp, i.fw);
                    if (cov > out.cov) {
                        let x = min(d / max(rp, 1e-6), 1.0);
                        out.cov = cov;
                        out.albedo = k.col[0] + (k.col[2] - k.col[0]) * smoothstep1(0.6, 1.0, x);
                        out.dh = 0.0;
                        out.id = h;
                    }
                    continue;
                }
                let ring = rp * k.v[8] * (1.0 + 0.18 * sin(4.0 * th + 6.3 * u01k(h, 8lu)));
                var cov = kband_cov(d, ring, i.fw);
                var col = k.col[0] * (0.9 + 0.2 * smoothstep1(rp, ring, d)) + (k.col[2] - k.col[0]) * (1.0 - smoothstep1(rp, 1.35 * rp, d));
                let n = round(k.v[5]);
                let len = k.v[6] * f;
                if (n >= 1.0 && d < len + i.fw && d > 0.5 * ring) {
                    let sector = TAU / n;
                    let j0 = round(tropical_wrap(th) / sector);
                    var tc = 0.0;
                    for (var dj = -1; dj <= 1; dj++) {
                        let j = j0 + f32(dj);
                        let jm = j - n * floor(j / n);
                        let ht = hash2(h, i64(jm), 7);
                        let lk = len * (0.45 + 0.55 * u01k(ht, 1lu));
                        if (d > lk + i.fw) {
                            continue;
                        }
                        let a = j * sector + 0.35 * sector * (u01k(ht, 2lu) - 0.5) + 0.25 * sin(d / (0.2 * max(len, 1.0)) + 6.3 * u01k(ht, 3lu)) * (d / max(len, 1.0));
                        let dp = d * min(abs(tropical_wrap(th - a)), 1.5);
                        let hw = 0.5 * k.v[7] * sqrt(1.0 - smoothstep1(0.5 * lk, lk, d));
                        tc = max(tc, kband_cov(dp, hw, i.fw));
                    }
                    if (tc > 0.0) {
                        if (cov > 0.0) {
                            col = col + (k.col[1] - col) * tc;
                        } else {
                            col = k.col[1];
                        }
                        cov = max(cov, tc);
                    }
                }
                if (cov > 0.0 && d < best) {
                    best = d;
                    out.cov = max(out.cov, cov);
                    out.albedo = col;
                    out.id = h;
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------- tropical_meander

fn tropical_meander(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    var out = kout_none();
    let cell = k.v[0];
    let dens = k.v[1] * i.amount;
    if (dens <= 0.0) {
        return out;
    }
    let scroll = k.v[2] >= 0.5;
    let reach = k.v[4] * 1.2 + k.v[5];
    let fr = tropical_sphere_frame(i.p);
    let q = fr.pu / f64(cell);
    let qf = floor(q);
    let fq = vec3<f32>(q - qf);
    let half_j = 0.5 * k.v[9];
    let lr = (reach + i.fw) / cell;
    let lim = lr * lr + 0.25;
    let hs = 0.5 * k.v[6];
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let g = vec3<f32>(nb_gap(fq.x, dx, half_j), nb_gap(fq.y, dy, half_j), nb_gap(fq.z, dz, half_j));
                if (dot(g, g) > lim) {
                    continue;
                }
                let si = tropical_sphere_site(k.seed, vec3<i64>(i64(qf.x) + i64(dx), i64(qf.y) + i64(dy), i64(qf.z) + i64(dz)), cell, k.v[9]);
                if (!si.ok) {
                    continue;
                }
                let h = si.h;
                let u = u01k(h, 4lu);
                if (u >= dens) {
                    continue;
                }
                let rel3 = vec3<f32>(fr.pu - si.c);
                let rel = vec2<f32>(dot(rel3, fr.east), dot(rel3, fr.north));
                let d = length(rel);
                let f = tropical_thin(u, dens);
                let rr = (k.v[3] + (k.v[4] - k.v[3]) * u01k(h, 5lu)) * f;
                if (d > rr * 1.2 + k.v[5] + i.fw) {
                    continue;
                }
                let a = tropical_wrap(atan2(rel.y, rel.x) - u01k(h, 6lu) * TAU);
                if (abs(a) > hs) {
                    continue;
                }
                let ra = rr * (1.0 + 0.16 * (u01k(h, 7lu) - 0.5) * cos(2.0 * a));
                let end = 1.0 - abs(a) / hs;
                let w = k.v[5] * f * sqrt(min(3.0 * end, 1.0));
                let dr = d - ra;
                var open_half = 0.0;
                if (u01k(h, 8lu) < k.v[7]) {
                    open_half = hs * (0.45 + 0.55 * u01k(h, 9lu));
                }
                let chan = kband_cov(dr, w, i.fw);
                if (!scroll) {
                    if (abs(a) < open_half && chan > out.cov) {
                        out.cov = chan;
                        out.albedo = k.col[0] + (k.col[1] - k.col[0]) * smoothstep1(0.4, 1.0, abs(dr) / max(w, 1e-6));
                        out.dh = 0.0;
                        out.id = h;
                    }
                    continue;
                }
                if (abs(a) >= open_half && chan > 0.0) {
                    out.cov = max(out.cov, chan);
                    out.albedo = k.col[2];
                    out.dh = 0.3 * k.v[10];
                    out.id = h;
                    continue;
                }
                let x = -dr - w;
                let xmax = 0.72 * ra * sqrt(max(1.0 - (a / hs) * (a / hs), 0.0));
                let inside = clamp(min(x, xmax - x) / max(i.fw, 1e-6) + 0.5, 0.0, 1.0);
                if (inside <= 0.0) {
                    continue;
                }
                let lam = k.v[8];
                let ph = x / lam;
                let s = ph - floor(ph);
                let wfr = max(i.fw / lam, 1e-4);
                let crisp = smoothstep1(0.0, wfr, s) * (1.0 - smoothstep1(0.45, 0.45 + wfr, s));
                let ridge = 0.45 + (crisp - 0.45) * band(lam, i.gsd);
                let wl = band(8.0, i.gsd);
                var tex = 1.0;
                if (wl > 0.0) {
                    tex = 1.0 + 0.18 * perlin3(h ^ 0x5Clu, i.p * (1.0lf / 8.0lf)) * wl;
                }
                let young = 1.0 - x / max(xmax, 1.0);
                out.cov = max(out.cov, inside);
                out.albedo = (k.col[1] + (k.col[0] - k.col[1]) * ridge) * tex * (0.92 + 0.12 * young);
                out.dh = k.v[10] * (0.7 + 0.3 * ridge) * (0.75 + 0.25 * (1.0 - young)) * tex;
                out.id = h;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------- tropical_mangrove

fn tropical_mangrove(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    let sc = f64(k.v[0]);
    if (i.amount <= 0.0) {
        return kout_none();
    }
    let ue = tropical_logistic(perlin3(k.seed ^ 0x3A1lu, i.p * (1.0lf / (0.9lf * sc))) + 0.5 * perlin3(k.seed ^ 0x3A2lu, i.p * (1.0lf / (0.33lf * sc))) * band(0.33 * k.v[0], i.gsd), 0.3);
    let w = clamp(2.0 * i.fw / k.v[0], 1e-3, 0.5);
    let ext = smoothstep1(-w, w, i.amount - 0.1 - 0.85 * ue);
    if (ext <= 0.0) {
        return kout_none();
    }
    let wl = 0.7lf * sc;
    let pw = i.p + vec3<f64>(vec3<f32>(perlin3(k.seed ^ 0x3B1lu, i.p * (1.0lf / wl)), perlin3(k.seed ^ 0x3B2lu, i.p * (1.0lf / wl)), perlin3(k.seed ^ 0x3B3lu, i.p * (1.0lf / wl))) * (0.35 * k.v[0]));
    let c1 = worley3(k.seed ^ 0x3C1lu, pw, sc, 0.9);
    let e1 = worley_edge_dist(c1, pw);
    let h1 = c1.id ^ c1.id2;
    let w1 = k.v[2] * (0.35 + 1.3 * u01k(h1, 1lu)) * select(0.0, 1.0, u01k(h1, 2lu) < 0.8);
    let c2 = worley3(k.seed ^ 0x3C2lu, pw, 0.36lf * sc, 0.9);
    let e2 = worley_edge_dist(c2, pw);
    let h2 = c2.id ^ c2.id2;
    let w2 = 0.4 * k.v[2] * (0.4 + 1.2 * u01k(h2, 1lu)) * select(0.0, 1.0, u01k(h2, 2lu) < 0.55);
    let chan = max(kband_cov(e1, w1, i.fw), kband_cov(e2, w2, i.fw));
    var o = kout_none();
    if (k.v[1] >= 0.5) {
        let cov = chan * ext;
        if (cov <= 0.0) {
            return o;
        }
        let wide = smoothstep1(0.0, 2.0 * k.v[2], w1 * kband_cov(e1, w1, i.fw));
        o.cov = cov;
        o.albedo = k.col[2] * (1.0 - 0.25 * wide);
        o.id = h1;
        return o;
    }
    let wc = worley2(k.seed ^ 0x3D1lu, i.q, k.v[4], 0.9);
    let x = min(wc.f1 / (0.62 * k.v[4]), 1.0);
    let bt = band(k.v[4], i.gsd);
    var base = k.col[0];
    if (u01k(wc.id, 1lu) < 0.3) {
        base = k.col[1];
    }
    let mean_c = k.col[0] * 0.7 + k.col[1] * 0.3;
    let tex = base * (1.0 + k.v[5] * (u01k(wc.id, 2lu) - 0.5)) * (0.76 + 0.34 * (1.0 - x * x));
    let col = mean_c * 0.93 + (tex - mean_c * 0.93) * bt;
    let near = min(smoothstep1(0.0, 3.0 * k.v[2] + 2.0, e1 - w1), smoothstep1(0.0, 1.5 * k.v[2] + 1.0, e2 - w2));
    let zone = 0.85 + 0.3 * (0.5 + 0.5 * perlin3(k.seed ^ 0x3D2lu, i.p * (1.0lf / (0.5lf * sc))));
    o.cov = ext * (1.0 - chan);
    o.albedo = col * (0.9 + 0.1 * near);
    o.dh = k.v[3] * (0.55 + 0.45 * near) * zone * (0.88 + 0.12 * (1.0 - x * x) * bt);
    o.id = wc.id;
    return o;
}

// ---------------------------------------------------------------- tropical_grass

fn tropical_grass(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    let sc = f64(k.v[0]);
    if (i.amount <= 0.0) {
        return kout_none();
    }
    let t = 0.5 + 0.5 * perlin3(k.seed ^ 0x6A1lu, i.p * (1.0lf / sc));
    let w = clamp(2.0 * i.fw / k.v[0], 0.15, 0.5);
    let cov = smoothstep1(-w, w, i.amount - 0.05 - 0.9 * (1.0 - t));
    if (cov <= 0.0) {
        return kout_none();
    }
    var col = k.col[0] + (k.col[1] - k.col[0]) * smoothstep1(0.3, 0.75, t);
    let wb = band(0.25 * k.v[0], i.gsd);
    var nb = perlin3(k.seed ^ 0x6A2lu, i.p * (1.0lf / (0.6lf * sc)));
    if (wb > 0.0) {
        nb += 0.5 * perlin3(k.seed ^ 0x6A3lu, i.p * (1.0lf / (0.25lf * sc))) * wb;
    }
    let ub = tropical_logistic(nb, 0.3);
    let bw = clamp(2.0 * i.fw / (0.6 * k.v[0]), 1e-3, 0.5);
    let bare = smoothstep1(-bw, bw, k.v[1] - ub);
    col = col + (k.col[2] - col) * bare;
    let w1 = band(1.6, i.gsd);
    var m = 0.0;
    if (w1 > 0.0) {
        m += 0.24 * perlin3(k.seed ^ 0x6A4lu, i.p * (1.0lf / 1.6lf)) * w1;
    }
    col *= (1.0 + k.v[2] * m * (1.0 - 0.6 * bare)) * (1.0 + k.v[3] * (t - 0.5));
    var o = kout_none();
    o.cov = cov;
    o.albedo = col;
    return o;
}

// ---------------------------------------------------------------- slot 8: river water

/// `kits::tropical::rivers`: white-water (café au lait) and black-water tropical rivers.
fn tropical_slot_water(s: ptr<function, Stack>) {
    let l = (*s).l;
    if (l.water > l.ground && l.water_kind != W_NONE) {
        return;
    }
    let t = (*s).t;
    if (l.river_hw <= 0.0 || t.river_wet < 0.5 || t.temp < 18.0 || l.ground > 900.0) {
        return;
    }
    let fwr = max(l.fw, (*s).c.gsd * 0.35);
    let cov = band_cov(l.river_d, l.river_hw, fwr);
    if (cov <= 0.0) {
        return;
    }
    let trop = smoothstep1(18.0, 21.0, t.temp) * (1.0 - smoothstep1(600.0, 900.0, l.ground)) * smoothstep1(0.3, 0.45, t.moist);
    let black = l.river_hw < 25.0 && t.moist > 0.62 && t.style.w > 0.55;
    var col = mixc(srgb(128.0, 100.0, 68.0), srgb(158.0, 128.0, 88.0), smoothstep1(10.0, 120.0, l.river_hw));
    if (black) {
        col = srgb(36.0, 28.0, 21.0);
    }
    composite(s, layer_paint(cov, mixc((*s).col, col, trop), 0u));
}

// ---------------------------------------------------------------- canopy slot: fishbone

/// `kits::tropical::fishbone`: colonization roads and lot clearings in the rainforest.
fn tropical_slot_canopy(s: ptr<function, Stack>) {
    let t = (*s).t;
    let l = (*s).l;
    if ((*s).is_water || t.temp < 21.0 || t.moist < 0.5 || l.ground > 700.0 || l.slope > 0.3 || t.development > 0.65 || t.agri > 0.55) {
        return;
    }
    let p = (*s).c.p;
    let c = floor(p * (1.0lf / 200000.0lf));
    let cc = (c + vec3<f64>(0.5lf)) * 200000.0lf;
    let up = normalize(vec3<f32>(cc));
    let o = vec3<f64>(up) * TROPICAL_R0;
    let el = sqrt(up.x * up.x + up.y * up.y);
    var east = vec3<f32>(1.0, 0.0, 0.0);
    if (el > 1e-6) {
        east = vec3<f32>(-up.y / el, up.x / el, 0.0);
    }
    let north = cross(up, east);
    let d = vec3<f32>(p - o);
    let q = vec2<f32>(dot(d, east), dot(d, north));
    let ah = hash3(0xF15Blu, i64(c.x), i64(c.y), i64(c.z));
    let bi = floor(q.x / 22000.0);
    let bj = floor(q.y / 22000.0);
    let bh = hash2(ah, i64(bi), i64(bj));
    if (u01k(bh, 1lu) >= 0.3) {
        return;
    }
    let gsd = (*s).c.gsd;
    let fw = max(l.fw, 0.5 * gsd);
    let y0 = (bj + 0.3 + 0.4 * u01k(bh, 2lu)) * 22000.0;
    let off = u01k(bh, 3lu) * 4400.0;
    let fi = round((q.x - off) / 4400.0);
    let x0 = fi * 4400.0 + off;
    let dx = q.x - x0;
    let dy = q.y - y0;
    let lj = floor(dy / 250.0);
    let hl = hash3(bh, i64(fi), i64(lj), select(0li, 1li, dx > 0.0));
    let age = 1.0 - min(abs(dy) / 11000.0, 1.0);
    let depth = 3000.0 * (0.1 + 0.9 * u01k(hl, 1lu)) * (0.25 + 0.75 * sqrt(age));
    let lk = floor(q.x / 250.0);
    let ht = hash3(bh ^ 0x7Blu, i64(lk), 0li, select(0li, 1li, dy > 0.0));
    let depth_t = 3000.0 * (0.2 + 0.6 * u01k(ht, 1lu));
    let n = 80.0 * perlin3(bh ^ 0x7Clu, p * (1.0lf / 350.0lf));
    let reach = abs(dy) < 11000.0 - 300.0;
    let feeder_lot = reach && abs(dx) < depth + n;
    let trunk_lot = abs(dy) < depth_t + n;
    let cleared = trunk_lot || feeder_lot;
    var h = 0lu;
    if (trunk_lot) {
        h = ht;
    } else if (feeder_lot) {
        h = hl;
    }
    let road = max(band_cov(dx, 3.0, fw) * select(0.0, 1.0, reach), band_cov(dy, 5.0, fw));
    if (!cleared && road <= 0.0) {
        return;
    }
    let laterite = spal(s, BP_LATERITE);
    var col = (*s).col;
    var cls = LC_TRACK;
    if (cleared) {
        let u = u01k(h, 2lu);
        var cc2 = srgb(68.0, 88.0, 58.0);
        cls = LC_CLEAR_CUT;
        if (u < 0.55) {
            cc2 = srgb(112.0, 116.0, 88.0);
            cls = LC_PASTURE;
        } else if (u < 0.62) {
            cc2 = srgb(62.0, 58.0, 52.0);
            cls = LC_BURN_SCAR;
        } else if (u < 0.7) {
            cc2 = mixc(laterite * 0.9, srgb(112.0, 100.0, 90.0), 0.4);
            cls = LC_BARE;
        } else if (u < 0.85) {
            cc2 = srgb(104.0, 112.0, 80.0);
            cls = LC_CROP;
        }
        col = cc2 * (0.9 + 0.2 * u01k(h, 3lu)) * (1.0 + 0.08 * perlin3(h, p * (1.0lf / 60.0lf)) * band(60.0, gsd));
    }
    col = mixc(col, laterite * 1.1, road);
    if (road > 0.5) {
        cls = LC_TRACK;
    }
    var ly = layer_paint(select(road, 1.0, cleared), col, cls);
    ly.hmode = HM_BLEND;
    ly.relit = 1.0;
    composite(s, ly);
}
