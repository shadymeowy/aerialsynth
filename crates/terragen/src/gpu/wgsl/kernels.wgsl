// The kernel library (`kernels.rs`): the same kernels, parameters and hashes. Biome layers and
// kits evaluate them by layer index (`kernel_eval`), or with their own parameters
// (`kernel_with`).

/// Inputs of a kernel at a sample (`KIn`).
struct KIn {
    q: vec2<f32>,
    p: vec3<f64>,
    east: vec3<f32>,
    north: vec3<f32>,
    gsd: f32,
    fw: f32,
    amount: f32,
    aux: vec4<f32>,
}

/// A kernel's result (`KOut`).
struct KOut {
    cov: f32,
    albedo: vec3<f32>,
    dh: f32,
    emit: f32,
    id: u64,
}

/// A kernel instance (`KParams`).
struct KP {
    kind: u32,
    seed: u64,
    v: array<f32, 16>,
    col: array<vec3<f32>, 3>,
}

fn kout_none() -> KOut {
    var o: KOut;
    o.cov = 0.0;
    o.albedo = vec3<f32>(0.0);
    o.dh = 0.0;
    o.emit = 0.0;
    o.id = 0lu;
    return o;
}

fn kp_load(li: u32) -> KP {
    let l = klayers[li];
    var k: KP;
    k.kind = l.head.y;
    k.seed = u64(l.head2.x) | (u64(l.head2.y) << 32u);
    for (var i = 0u; i < 16u; i++) {
        k.v[i] = l.v[i / 4u][i % 4u];
    }
    k.col[0] = l.col[0].xyz;
    k.col[1] = l.col[1].xyz;
    k.col[2] = l.col[2].xyz;
    return k;
}

fn kband_cov(d: f32, hw: f32, fw: f32) -> f32 {
    return clamp((hw - abs(d)) / max(fw, 1e-9) + 0.5, 0.0, 1.0);
}

fn krot(v: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec2<f32>(v.x * c + v.y * s, -v.x * s + v.y * c);
}

/// Height profile of a shape at x = d / r (`kernels::profile`).
fn crown_profile(sh: u32, x: f32) -> f32 {
    switch sh {
        case 0u, 5u: { return 1.0; }
        case 2u: { return 0.08 + 0.92 * pow(1.0 - x, 1.15); }
        case 3u: { return 0.72 + 0.28 * max(1.0 - pow(x, 6.0), 0.0); }
        case 4u: { return 0.6 + 0.4 * (1.0 - x); }
        case 6u: { return sqrt(max(1.0 - x * x, 0.0)); }
        case 7u: { return 1.0 - min(abs(2.0 * x - 1.0), 1.0) * 0.5; }
        default: { return 0.12 + 0.88 * sqrt(max(1.0 - x * x, 0.0)); }
    }
}

/// Inside test of a shape (`kernels::shape_at`): (coverage, x = d / r).
fn shape_at(sh: u32, rel: vec2<f32>, r: f32, aspect: f32, ring: f32, fw: f32, phase: f32) -> vec2<f32> {
    switch sh {
        case 4u: {
            let ang = atan2(rel.y, rel.x) + phase;
            let re = r * (0.82 + 0.18 * cos(8.0 * ang));
            let d = length(rel);
            return vec2<f32>(kband_cov(d, re, fw) * select(0.0, 1.0, d <= re + fw), min(d / max(re, 1e-6), 1.0));
        }
        case 5u: {
            let half = vec2<f32>(r, r * aspect);
            let e = half - abs(rel);
            let ax = abs(rel) / half;
            return vec2<f32>(clamp(min(e.x, e.y) / max(fw, 1e-9) + 0.5, 0.0, 1.0), min(max(ax.x, ax.y), 1.0));
        }
        case 6u: {
            let d = length(rel);
            let di = length(rel - vec2<f32>(0.55 * r, 0.0));
            return vec2<f32>(kband_cov(d, r, fw) * (1.0 - kband_cov(di, 0.85 * r, fw)), min(d / r, 1.0));
        }
        case 7u: {
            let d = length(rel);
            let w = max(ring * r, 0.05);
            return vec2<f32>(kband_cov(d - (r - 0.5 * w), 0.5 * w, fw), min(d / r, 1.0));
        }
        default: {
            let q = vec2<f32>(rel.x, rel.y / max(aspect, 0.1));
            let d = length(q);
            return vec2<f32>(kband_cov(d, r, fw) * select(0.0, 1.0, d <= r + fw), min(d / max(r, 1e-6), 1.0));
        }
    }
}

/// The site of lattice cell (ix + dx, iy + dy) in metres relative to q (exact small numbers in
/// cell units first).
fn kcell_rel(q: vec2<f32>, cell: f32, cf: vec2<f32>, dx: i32, dy: i32, jit: f32, h: u64) -> vec2<f32> {
    let fq = q / cell - cf;
    let rel = vec2<f32>(f32(dx), f32(dy)) + 0.5 + jit * (vec2<f32>(u01k(h, 1lu), u01k(h, 2lu)) - 0.5) - fq;
    return -rel * cell;
}

fn k_scatter(k: KP, i: KIn) -> KOut {
    var out = kout_none();
    let cell = k.v[0];
    let dens = k.v[1] * i.amount;
    let sh = u32(k.v[2]);
    let jit = k.v[7];
    if (dens <= 0.0) {
        return out;
    }
    let cf = floor(i.q / cell);
    let ix = i64(cf.x);
    let iy = i64(cf.y);
    var best = -FMAX;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let h = hash2(k.seed, ix + i64(dx), iy + i64(dy));
            if (u01k(h, 3lu) >= dens) {
                continue;
            }
            let r = k.v[3] + (k.v[4] - k.v[3]) * u01k(h, 4lu);
            var rel = kcell_rel(i.q, cell, cf, dx, dy, jit, h);
            let lim = r * max(k.v[10], 1.0) + i.fw;
            if (dot(rel, rel) > lim * lim) {
                continue;
            }
            var ang = u01k(h, 8lu) * TAU;
            if (k.v[11] >= 1.0) {
                ang = i.aux.x;
            }
            rel = krot(rel, ang);
            let sx = shape_at(sh, rel, r, k.v[10], k.v[12], i.fw, u01k(h, 9lu) * 6.3);
            if (sx.x <= 0.0) {
                continue;
            }
            let hh = (k.v[5] + (k.v[6] - k.v[5]) * u01k(h, 5lu)) * crown_profile(sh, sx.y);
            if (hh > best) {
                best = hh;
                var base = k.col[0];
                if (u01k(h, 6lu) < k.v[13]) {
                    base = k.col[1];
                }
                let tint = 1.0 + k.v[8] * (u01k(h, 7lu) - 0.5);
                out.albedo = base * tint * (1.0 - k.v[9] * 0.5 * sx.y);
                out.dh = hh;
                out.id = h;
            }
            out.cov = max(out.cov, sx.x);
        }
    }
    return out;
}

fn k_rows(k: KP, i: KIn) -> KOut {
    var ang = k.v[4];
    if (k.v[5] >= 1.0) {
        ang = i.aux.x;
    }
    let q = krot(i.q, ang);
    let sx = k.v[0];
    let sy = k.v[1];
    let row = round(q.y / sx);
    let dv = q.y - row * sx;
    var d = abs(dv);
    var j = 0.0;
    if (sy > 0.0) {
        j = round(q.x / sy);
        d = length(vec2<f32>(q.x - j * sy, dv));
    }
    let h = hash2(k.seed, i64(row), i64(j));
    if (u01k(h, 1lu) < k.v[6] || u01k(hash1(k.seed ^ 0x5Alu, i64(row)), 2lu) >= i.amount) {
        return kout_none();
    }
    let r = k.v[2];
    let cov = kband_cov(d, r, i.fw);
    if (cov <= 0.0) {
        return kout_none();
    }
    let x = min(d / r, 1.0);
    let tint = 1.0 + k.v[8] * (u01k(h, 3lu) - 0.5);
    var o = kout_none();
    o.cov = cov;
    o.albedo = k.col[0] * tint * (0.8 + 0.3 * (1.0 - x));
    o.dh = k.v[3] * sqrt(max(1.0 - x * x, 0.0));
    o.id = h;
    return o;
}

fn k_cells(k: KP, i: KIn) -> KOut {
    let asp = k.v[6];
    let qs = vec2<f32>(i.q.x / asp, i.q.y);
    let wc = worley2(k.seed, qs, k.v[0] / sqrt(asp), k.v[1]);
    let dn = wc.point2 - wc.point;
    let nrm = length(vec2<f32>(dn.x / asp, dn.y)) / max(length(dn), 1e-9);
    let edge = worley2_edge_dist(wc, qs) / max(nrm, 1e-6);
    let filled = u01k(wc.id, 1lu) < k.v[2] * i.amount;
    var ec = 0.0;
    if (k.v[3] > 0.0) {
        ec = kband_cov(edge, 0.5 * k.v[3], i.fw) * min(i.amount, 1.0);
    }
    var o = kout_none();
    o.id = wc.id;
    if (filled) {
        let t = u01k(wc.id, 2lu);
        o.cov = 1.0;
        o.albedo = (k.col[0] + (k.col[1] - k.col[0]) * t) * (1.0 + k.v[5] * (u01k(wc.id, 3lu) - 0.5));
        o.dh = k.v[7];
    }
    if (ec > 0.0) {
        let c0 = o.cov;
        if (c0 > 0.0) {
            o.albedo = o.albedo + (k.col[2] - o.albedo) * ec;
        } else {
            o.albedo = k.col[2];
        }
        o.dh = o.dh + (k.v[4] - o.dh) * ec;
        o.cov = max(c0, ec);
    }
    return o;
}

fn k_stripes(k: KP, i: KIn) -> KOut {
    var ang = k.v[1];
    if (k.v[2] >= 1.0) {
        ang = i.aux.x;
    }
    let dir = i.east * cos(ang) + i.north * sin(ang);
    let g = gully_octave(k.seed, i.p * (1.0lf / f64(k.v[0])), dir);
    let prof = pow(clamp(0.5 + 0.5 * g.x, 0.0, 1.0), k.v[3]);
    var cov = min(i.amount, 1.0);
    if (k.v[5] > 0.0) {
        let w = max(i.fw / k.v[0], 1e-3);
        cov = smoothstep1(k.v[5] - w, k.v[5] + w, prof) * min(i.amount, 1.0);
    }
    var o = kout_none();
    o.cov = cov;
    o.albedo = k.col[1] + (k.col[0] - k.col[1]) * prof;
    o.dh = k.v[4] * prof;
    return o;
}

fn k_contours(k: KP, i: KIn) -> KOut {
    let h = i.aux.x;
    let slope = max(abs(i.aux.y), 1e-3);
    let x = h / k.v[0] + k.v[3];
    let f = x - floor(x);
    var o = kout_none();
    if (k.v[2] >= 1.0) {
        let q = (floor(x) + smoothstep1(1.0 - k.v[4], 1.0, f) - x) * k.v[0];
        let riser = smoothstep1(1.0 - k.v[4], 1.0, f) * (1.0 - smoothstep1(1.0 - 0.3 * k.v[4], 1.0, f));
        o.cov = min(i.amount, 1.0);
        o.albedo = k.col[1] + (k.col[0] - k.col[1]) * riser;
        o.dh = q;
        return o;
    }
    let dist = min(f, 1.0 - f) * k.v[0] / slope;
    o.cov = kband_cov(dist, k.v[1], i.fw) * min(i.amount, 1.0);
    o.albedo = k.col[0];
    return o;
}

fn k_radial(k: KP, i: KIn) -> KOut {
    var out = kout_none();
    let cell = k.v[0];
    let cf = floor(i.q / cell);
    let ix = i64(cf.x);
    let iy = i64(cf.y);
    let dens = k.v[1] * i.amount;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let h = hash2(k.seed, ix + i64(dx), iy + i64(dy));
            if (u01k(h, 3lu) >= dens) {
                continue;
            }
            let rel = kcell_rel(i.q, cell, cf, dx, dy, k.v[11], h);
            let r0 = k.v[2] + (k.v[3] - k.v[2]) * u01k(h, 4lu);
            let d = length(rel);
            if (d > r0 * (1.0 + k.v[8]) + i.fw) {
                continue;
            }
            let ang = atan2(rel.y, rel.x);
            var arm = 0.0;
            if (k.v[7] >= 1.0) {
                arm = cos(round(k.v[7]) * ang + u01k(h, 6lu) * 6.3);
            }
            let re = r0 * (1.0 + k.v[8] * arm);
            let x = min(d / max(re, 1e-6), 1.0);
            let cov = kband_cov(d, re, i.fw);
            if (cov <= 0.0) {
                continue;
            }
            let hh = k.v[4] + (k.v[5] - k.v[4]) * u01k(h, 5lu);
            var z = hh * pow(1.0 - x, k.v[6]);
            if (k.v[9] > 0.0 && x < k.v[9]) {
                let xc = x / k.v[9];
                z -= k.v[10] * hh * (1.0 - xc * xc);
            }
            if (abs(z) > abs(out.dh) || out.cov <= 0.0) {
                out.dh = z;
                out.albedo = k.col[1] + (k.col[0] - k.col[1]) * (1.0 - x);
                out.id = h;
            }
            out.cov = max(out.cov, cov);
        }
    }
    return out;
}

fn k_crescent(k: KP, i: KIn) -> KOut {
    var out = kout_none();
    let cell = k.v[0];
    let cf = floor(i.q / cell);
    let ix = i64(cf.x);
    let iy = i64(cf.y);
    var ang = k.v[7];
    if (k.v[6] >= 1.0) {
        ang = i.aux.x;
    }
    let dens = k.v[1] * i.amount;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let h = hash2(k.seed, ix + i64(dx), iy + i64(dy));
            if (u01k(h, 3lu) >= dens) {
                continue;
            }
            let r = k.v[2] + (k.v[3] - k.v[2]) * u01k(h, 4lu);
            let rel = krot(kcell_rel(i.q, cell, cf, dx, dy, k.v[8], h), ang);
            let d = length(rel);
            if (d > r + i.fw) {
                continue;
            }
            let di = length(rel - vec2<f32>(k.v[5] * r, 0.0));
            let cov = kband_cov(d, r, i.fw) * (1.0 - kband_cov(di, (1.0 - 0.3 * k.v[5]) * r, i.fw));
            if (cov <= 0.0) {
                continue;
            }
            let x = min(d / r, 1.0);
            let z = k.v[4] * r * sqrt(max(1.0 - x * x, 0.0)) * smoothstep1(-1.0, 0.2, -rel.x / r + 0.3);
            if (z > out.dh) {
                out.dh = z;
                out.albedo = k.col[0] * (0.9 + 0.2 * (1.0 - x));
                out.id = h;
            }
            out.cov = max(out.cov, cov);
        }
    }
    return out;
}

fn k_lobes(k: KP, i: KIn) -> KOut {
    var out = kout_none();
    let cell = k.v[0];
    let cf = floor(i.q / cell);
    let ix = i64(cf.x);
    let iy = i64(cf.y);
    let dens = k.v[1] * i.amount;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let h = hash2(k.seed, ix + i64(dx), iy + i64(dy));
            if (u01k(h, 3lu) >= dens) {
                continue;
            }
            let r = k.v[2] + (k.v[3] - k.v[2]) * u01k(h, 4lu);
            let rel = kcell_rel(i.q, cell, cf, dx, dy, k.v[9], h);
            let d = length(rel);
            if (d > r * (1.0 + k.v[5]) + i.fw) {
                continue;
            }
            var dir = u01k(h, 6lu) * TAU;
            if (k.v[8] >= 1.0) {
                dir = i.aux.x;
            }
            var da = atan2(rel.y, rel.x) - dir;
            da = da + PI - TAU * floor((da + PI) / TAU) - PI;
            let half = 0.5 * k.v[4];
            if (abs(da) > half) {
                continue;
            }
            let ql = rel / (0.3 * r);
            let lobe = sin(da / half * 2.5 + u01k(h, 7lu) * 6.3) * 0.5 + 0.5 * perlin3(h, vec3<f64>(f64(ql.x), f64(ql.y), 0.5lf));
            let re = r * (1.0 - k.v[5] * 0.5 + k.v[5] * 0.5 * lobe) * (1.0 - pow(da / half, 4.0));
            let cov = kband_cov(d - 0.5 * re, 0.5 * re, i.fw);
            if (cov <= 0.0) {
                continue;
            }
            let x = min(d / max(re, 1e-6), 1.0);
            var ch = 0.0;
            if (k.v[6] > 0.0) {
                ch = smoothstep1(0.75, 0.95, cos(da / half * k.v[6] * PI)) * (1.0 - x);
            }
            out.albedo = k.col[0] + (k.col[1] - k.col[0]) * ch;
            out.dh = k.v[7] * (1.0 - x);
            out.cov = max(out.cov, cov);
            out.id = h;
        }
    }
    return out;
}

/// `patch_u`: a multi-scale noise mapped to ~uniform 0..1.
fn patch_u(seed: u64, p: vec3<f64>, scale: f32, oct: f32, rough: f32, gsd: f32) -> f32 {
    var n = 0.0;
    var a = 1.0;
    var norm = 0.0;
    // (wavelengths in f64: see `kernels::compile_params`)
    var lam = f64(scale);
    for (var o = 0u; o < 3u; o++) {
        if (f32(o) >= oct) {
            break;
        }
        n += a * perlin3(seed ^ (0x9A7lu + u64(o)), p * (1.0lf / lam)) * band(f32(lam), gsd);
        norm += a * a;
        a *= rough;
        lam *= 0.43lf;
    }
    let s = n / (0.27 * sqrt(norm));
    return 1.0 / (1.0 + exp(-1.7 * s));
}

fn k_patches(k: KP, i: KIn) -> KOut {
    let u = patch_u(k.seed, i.p, k.v[0], k.v[2], k.v[3], i.gsd);
    let frac = k.v[1] * i.amount;
    let w = clamp(2.0 * i.fw / k.v[0], 1e-3, 0.5);
    let cov = smoothstep1(-w, w, frac - u);
    if (cov <= 0.0) {
        return kout_none();
    }
    let t = 0.5 + 0.5 * perlin3(k.seed ^ 0x77lu, i.p * (1.0lf / (0.37lf * f64(k.v[0]))));
    var o = kout_none();
    o.cov = cov;
    o.albedo = (k.col[0] + (k.col[1] - k.col[0]) * t) * (1.0 + k.v[5] * (u - 0.5));
    o.dh = k.v[4];
    return o;
}

fn k_linear(k: KP, i: KIn) -> KOut {
    let d = i.aux.x;
    let s = i.aux.y;
    let hw = i.aux.z;
    let cov = kband_cov(d, hw + k.v[3], i.fw) * min(i.amount, 1.0);
    if (cov <= 0.0) {
        return kout_none();
    }
    let core = kband_cov(d, hw, i.fw);
    var col = k.col[1] + (k.col[0] - k.col[1]) * core;
    let x = min(abs(d) / max(hw, 1e-3), 1.0);
    var dh = k.v[4];
    switch u32(k.v[0]) {
        case 1u: { dh = k.v[4] * (1.0 - 0.3 * x * x); }
        case 2u: { dh = k.v[4] * sqrt(max(1.0 - x * x, 0.0)); }
        case 3u: { dh = -k.v[4] * max(1.0 - x * x, 0.0); }
        default: {}
    }
    if (k.v[5] > 0.0) {
        var on = 1.0;
        if (k.v[1] > 0.0) {
            let ph = s / k.v[1];
            on = select(0.0, 1.0, ph - floor(ph) < k.v[2]);
        }
        let m = kband_cov(d, 0.5 * k.v[5], i.fw) * on * band(max(k.v[1], 4.0 * k.v[5]), i.gsd);
        col = col + (k.col[2] - col) * m;
    }
    var o = kout_none();
    o.cov = cov;
    o.albedo = col;
    o.dh = dh;
    return o;
}

fn k_stamp(k: KP, i: KIn) -> KOut {
    let u = i.aux.x;
    let w = i.aux.y;
    let hu = i.aux.z;
    let hv = i.aux.w;
    let inside = clamp(min(hu - abs(u), hv - abs(w)) / max(i.fw, 1e-9) + 0.5, 0.0, 1.0) * min(i.amount, 1.0);
    if (inside <= 0.0) {
        return kout_none();
    }
    var col = k.col[0];
    var dh = 0.0;
    var emit = 0.0;
    switch u32(k.v[0]) {
        case 1u: {
            let ph = u / max(k.v[1], 1.0);
            let cl = kband_cov(w, 0.5, i.fw) * select(0.0, 1.0, ph - floor(ph) < 0.5);
            col = col + (k.col[1] - col) * 0.85 + (k.col[2] - k.col[1]) * cl;
            if (k.v[4] > 0.0) {
                let sp = 10.0 * max(k.v[2], 1.0);
                let du = u - round(u / sp) * sp;
                let dv = hv - abs(w);
                emit = point_light(du * du + dv * dv, k.v[4], 0.4, i.fw);
            }
        }
        case 2u: {
            let pitch = k.v[1] + k.v[2];
            let gu = u - round(u / pitch) * pitch;
            let gv = w - round(w / pitch) * pitch;
            let b = clamp(min(0.5 * k.v[1] - abs(gu), 0.5 * k.v[1] - abs(gv)) / max(i.fw, 1e-9) + 0.5, 0.0, 1.0);
            col = col + (k.col[1] - col) * b;
            dh = k.v[3] * b;
        }
        case 3u: {
            let pitch = k.v[1] + k.v[2];
            let g = w - round(w / pitch) * pitch;
            let b = kband_cov(g, 0.5 * k.v[1], i.fw);
            col = col + (k.col[1] - col) * b;
            dh = k.v[3] * b;
        }
        default: {}
    }
    var o = kout_none();
    o.cov = inside;
    o.albedo = col;
    o.dh = dh;
    o.emit = emit;
    return o;
}

fn k_canopy(k: KP, i: KIn) -> KOut {
    if (i.amount <= 0.0) {
        return kout_none();
    }
    let wc = worley2(k.seed ^ 0xC1lu, i.q, k.v[4], 0.9);
    let x = min(wc.f1 / (0.62 * k.v[4]), 1.0);
    let hue = u01k(wc.id, 1lu);
    var col = k.col[0] * (1.0 + k.v[5] * (hue - 0.5)) * (0.72 + 0.4 * (1.0 - x * x));
    col = col + (k.col[1] - col) * (0.5 * k.v[5] * smoothstep1(0.7, 1.0, u01k(wc.id, 2lu)));
    if (u01k(wc.id, 3lu) < k.v[6]) {
        col = k.col[2] * (0.85 + 0.3 * (1.0 - x));
    }
    var dh = k.v[2] * (0.88 + 0.12 * sqrt(max(1.0 - x * x, 0.0)));
    var ek: KP;
    ek.kind = KIND_SCATTER;
    ek.seed = k.seed ^ 0xE3lu;
    ek.v = array<f32, 16>(k.v[0], k.v[1], f32(SHAPE_DOME), 0.25 * k.v[0], 0.38 * k.v[0], k.v[3] * 0.85, k.v[3], 0.8, k.v[5], 0.4, 1.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    ek.col = array<vec3<f32>, 3>(k.col[0] * 1.08, k.col[1], k.col[2]);
    var ei = i;
    ei.amount = 1.0;
    let em = k_scatter(ek, ei);
    if (em.cov > 0.0 && em.dh > dh) {
        col = col + (em.albedo - col) * em.cov;
        dh = dh + (em.dh - dh) * em.cov;
    }
    var o = kout_none();
    o.cov = min(i.amount, 1.0);
    o.albedo = col;
    o.dh = dh;
    o.id = wc.id;
    return o;
}

fn k_water(k: KP, i: KIn) -> KOut {
    let depth = max(i.aux.x, 0.0);
    let t = 1.0 - exp(-depth / k.v[0]);
    var col = k.col[0] + (k.col[1] - k.col[0]) * t;
    col *= 1.0 + k.v[1] * perlin3(k.seed, i.p * (1.0lf / f64(k.v[2]))) * band(k.v[2], i.gsd);
    if (k.v[3] > 0.0) {
        let a = (depth - k.v[3]) / k.v[4];
        let foam = exp(-a * a) * band(3.0 * max(k.v[4], 1.0), i.gsd);
        col = col + (k.col[2] - col) * min(foam, 1.0);
    }
    var o = kout_none();
    o.cov = min(i.amount, 1.0);
    o.albedo = col;
    return o;
}

/// A core kernel's explicit evaluation (`kernels::explicit`); the kits' through `kits_kernel`.
fn kernel_explicit(k: KP, li: u32, i: KIn) -> KOut {
    switch k.kind {
        case 0u: { return k_scatter(k, i); }
        case 1u: { return k_rows(k, i); }
        case 2u: { return k_cells(k, i); }
        case 3u: { return k_stripes(k, i); }
        case 4u: { return k_contours(k, i); }
        case 5u: { return k_radial(k, i); }
        case 6u: { return k_crescent(k, i); }
        case 7u: { return k_lobes(k, i); }
        case 8u: { return k_patches(k, i); }
        case 9u: { return k_linear(k, i); }
        case 10u: { return k_stamp(k, i); }
        case 11u: { return k_canopy(k, i); }
        case 12u: { return k_water(k, i); }
        default: {
            if (k.kind >= KIT_BASE) {
                return kits_kernel(k.kind, li, i);
            }
            return kout_none();
        }
    }
}

/// The feature size at a sample (`kernels::size_at`).
fn kernel_size(k: KP, size: f32, i: KIn) -> f32 {
    switch k.kind {
        case 4u: { return k.v[0] / max(abs(i.aux.y), 1e-3); }
        case 9u: { return 2.0 * i.aux.z; }
        default: { return size; }
    }
}

/// Explicit where resolved, the calibrated mean where not (`kernels::eval`).
fn kernel_mix(k: KP, li: u32, size: f32, mean: vec4<f32>, mean_albedo: vec3<f32>, i: KIn) -> KOut {
    let e = smoothstep1(1.2 * i.gsd, 3.0 * i.gsd, kernel_size(k, size, i));
    let mc = mean.x * clamp(i.amount, 0.0, 1.0);
    var o = kout_none();
    if (e <= 0.0) {
        o.cov = mc;
        o.albedo = mean_albedo;
        o.dh = mean.y;
        o.emit = mean.z * i.amount;
        return o;
    }
    let x = kernel_explicit(k, li, i);
    if (e >= 1.0) {
        return x;
    }
    let cov = e * x.cov + (1.0 - e) * mc;
    o.emit = e * x.emit + (1.0 - e) * mean.z * i.amount;
    if (cov <= 0.0) {
        return o;
    }
    let we = e * x.cov / cov;
    let wm = (1.0 - e) * mc / cov;
    o.cov = cov;
    o.albedo = x.albedo * we + mean_albedo * wm;
    o.dh = x.dh * we + mean.y * wm;
    o.id = x.id;
    return o;
}

/// Kernel layer `li` of the registry at a sample.
fn kernel_eval(li: u32, i: KIn) -> KOut {
    let l = klayers[li];
    return kernel_mix(kp_load(li), li, l.sz.x, l.mean, l.mean_albedo.xyz, i);
}
