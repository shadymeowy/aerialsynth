// The coast kit's types, creeks and relief (`kits/coast.rs`; every module with pass A): the same
// functions in the same order of operations.

const CO_SEED: u64 = 0xC0A57000lu;
const CO_CREEK_MEAN: f32 = 0.055;

/// `coast::nz`.
fn co_nz(seed: u64, p: vec3<f64>, lam: f32, gsd: f32) -> f32 {
    let b = band(lam, gsd);
    if (b > 0.0) {
        return perlin3(seed, p * (1.0lf / f64(lam))) * b;
    }
    return 0.0;
}

fn co_cov(d: f32, hw: f32, fw: f32) -> f32 {
    return clamp((hw - abs(d)) / fw + 0.5, 0.0, 1.0);
}

struct CoTypes {
    cliff: f32,
    skerry: f32,
    flat: f32,
    barrier: f32,
    reef: f32,
    arch: f32,
    far: f32,
}

/// `coast::types`.
fn co_types(p: vec3<f64>, uplift: f32, mountain: f32, glac: f32, temp: f32) -> CoTypes {
    let u = perlin3(CO_SEED + 1lu, p * (1.0lf / 55000.0lf));
    let v = perlin3(CO_SEED + 2lu, p * (1.0lf / 38000.0lf));
    let w = perlin3(CO_SEED + 3lu, p * (1.0lf / 47000.0lf));
    let relp = max(smoothstep1(0.05, 0.4, uplift), mountain);
    let gl = smoothstep1(0.3, 0.6, glac);
    let cliff_s = relp * smoothstep1(-0.3, 0.3, u);
    let skerry_s = gl * (0.6 + 0.4 * smoothstep1(-0.4, 0.3, v));
    let flat_s = (1.0 - relp) * (1.0 - gl) * smoothstep1(0.05, 0.4, v);
    let barrier_s = (1.0 - relp) * (1.0 - gl) * smoothstep1(-0.05, -0.4, v);
    let mx = max(max(max(cliff_s, skerry_s), max(flat_s, barrier_s)), 0.0001);
    let gate = smoothstep1(0.0, 0.2, mx);
    var o: CoTypes;
    o.cliff = smoothstep1(0.7, 1.0, cliff_s / mx) * gate;
    o.skerry = smoothstep1(0.7, 1.0, skerry_s / mx) * gate;
    o.flat = smoothstep1(0.7, 1.0, flat_s / mx) * gate;
    o.barrier = smoothstep1(0.7, 1.0, barrier_s / mx) * gate;
    o.reef = smoothstep1(22.0, 25.0, temp) * smoothstep1(-0.5, 0.0, w);
    o.arch = smoothstep1(0.15, 0.5, uplift) * smoothstep1(-0.1, 0.3, -u) * (1.0 - o.cliff);
    o.far = smoothstep1(0.1, 0.3, perlin3(CO_SEED + 4lu, p * (1.0lf / 60000.0lf)));
    return o;
}

/// `coast::creek`.
fn co_creek(p: vec3<f64>, gsd: f32, fw: f32) -> f32 {
    let b1 = band(420.0, gsd);
    if (b1 <= 0.0) {
        return CO_CREEK_MEAN;
    }
    let n1 = perlin3(CO_SEED + 10lu, p * (1.0lf / 420.0lf));
    let d1 = abs(n1) * 420.0 / 1.3;
    var c = b1 * co_cov(d1, 7.0, fw) + (1.0 - b1) * CO_CREEK_MEAN;
    if (gsd < 37.0) {
        let q2 = p * (1.0lf / 150.0lf) + vec3<f64>(0.35lf, 0.21lf, -0.28lf) * f64(n1);
        let n2 = perlin3(CO_SEED + 11lu, q2);
        let d2 = abs(n2) * 150.0 / 1.3;
        c = max(c, band(150.0, gsd) * co_cov(d2, 3.0, fw) * (1.0 - smoothstep1(40.0, 130.0, d1)));
        if (gsd < 14.0) {
            let q3 = p * (1.0lf / 55.0lf) + vec3<f64>(0.3lf, -0.2lf, 0.33lf) * f64(n2);
            let n3 = perlin3(CO_SEED + 12lu, q3);
            let d3 = abs(n3) * 55.0 / 1.3;
            c = max(c, 0.9 * band(55.0, gsd) * co_cov(d3, 1.2, fw) * (1.0 - smoothstep1(20.0, 55.0, min(d1, d2))));
        }
    }
    return c;
}

/// `coast::coast_dh`.
fn co_dh(t: CoTypes, h0: f32, p: vec3<f64>, gsd: f32) -> f32 {
    var dh = 0.0;
    if (t.cliff > 0.0 && h0 > -12.0 && h0 < 500.0) {
        let n = perlin3(CO_SEED + 5lu, p * (1.0lf / 30000.0lf));
        let c = 12.0 + 100.0 * smoothstep1(-0.4, 0.6, n);
        let eps = max(2.0, 0.04 * gsd);
        let lift = c * smoothstep1(0.0, eps, h0) * (1.0 - smoothstep1(1.5 * c, 4.0 * c, h0));
        let u = smoothstep1(-10.0, -1.5, h0) * smoothstep1(0.3, -0.3, h0);
        let plat = (-1.3 + 0.1 * h0 - h0) * u;
        dh += t.cliff * (lift + plat);
    }
    if (t.skerry > 0.0 && h0 > -60.0 && h0 < 40.0) {
        let g = 1.0 - smoothstep1(18.0, 55.0, abs(h0 + 8.0));
        if (g > 0.0) {
            let n = 17.0 * co_nz(CO_SEED + 20lu, p, 700.0, gsd) + 8.0 * co_nz(CO_SEED + 21lu, p, 230.0, gsd) + 3.0 * co_nz(CO_SEED + 22lu, p, 75.0, gsd);
            dh += t.skerry * g * (n + 4.0);
        }
    }
    if (t.arch > 0.0 && h0 > -120.0 && h0 < 60.0) {
        let g = 1.0 - smoothstep1(30.0, 90.0, abs(h0 + 25.0));
        if (g > 0.0) {
            let n = 38.0 * co_nz(CO_SEED + 23lu, p, 2600.0, gsd) + 14.0 * co_nz(CO_SEED + 24lu, p, 900.0, gsd) + 5.0 * co_nz(CO_SEED + 25lu, p, 260.0, gsd);
            dh += t.arch * g * (n + 8.0);
        }
    }
    if (t.flat > 0.0 && h0 > -16.0 && h0 < 16.0) {
        let e = 1.0 - smoothstep1(6.0, 14.0, abs(h0 - 1.0));
        let h1 = lerp(h0, 0.2 * h0 + 0.4, e);
        let win = (1.0 - smoothstep1(1.6, 3.2, h1)) * smoothstep1(-5.0, -2.0, h1);
        var carve = 0.0;
        if (win > 0.0) {
            carve = 1.9 * co_creek(p, gsd, gsd) * win;
        }
        dh += t.flat * (h1 - h0 - carve);
    }
    if (t.barrier > 0.0 && h0 > -14.0 && h0 < 2.0) {
        let open = smoothstep1(-0.25, 0.1, co_nz(CO_SEED + 30lu, p, 3200.0, gsd));
        let x = (h0 + 6.0) / 2.8;
        dh += t.barrier * 8.7 * open * exp(-x * x);
    }
    if (t.reef > 0.0 && h0 > -50.0 && h0 < 2.0) {
        let hb = lerp(-7.0, -28.0, t.far);
        let wd = lerp(2.4, 9.0, t.far);
        let gap = smoothstep1(-0.35, 0.05, co_nz(CO_SEED + 31lu, p, lerp(900.0, 1800.0, t.far), gsd));
        let x = (h0 - hb) / wd;
        dh += t.reef * (-hb - 0.9) * gap * exp(-x * x);
    }
    return dh;
}

fn coast_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {
    let l = inst_list(FAM_COAST_ATOLLS, r.blk, c.p);
    for (var i = 0u; i < l.n; i++) {
        let a = co_atoll_at(l.items[i], c.p, c.gsd);
        if (a.ok) {
            let k = 1.0 - smoothstep1(5500.0, 7000.0, a.d - a.wre);
            *h += (max(*h, a.z) - *h) * k;
        }
    }
    if (m.coast_km > 60.0 || m.coast_km < -100.0) {
        return;
    }
    let h0 = *h;
    if (h0 < -120.0 || h0 > 500.0) {
        return;
    }
    let t = co_types(c.p, m.uplift, r.mountain, m.glaciation, m.temp);
    *h += co_dh(t, h0, c.p, c.gsd);
}

// ---------------------------------------------------------------- atolls

/// `coast::atoll_exists`.
fn coast_atolls_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 1lu) >= 0.25) {
        return o;
    }
    let a = atlas_sample(vec3<f32>(normalize(s.center)));
    if (a.elevation > -1500.0 || a.elevation < -5500.0 || a.coast_km > -150.0 || a.temp_c < 25.0) {
        return o;
    }
    let hot = atlas_hotspot(a) || a.volcanism > 0.12;
    var need = 0.35;
    if (hot) {
        need = 0.9;
    }
    if (u01k(s.id, 2lu) >= need) {
        return o;
    }
    o.ok = true;
    o.v = array<f32, 8>(2000.0 + 12000.0 * pow(u01k(s.id, 3lu), 1.5), 180.0 + 220.0 * u01k(s.id, 4lu), 12.0 + 18.0 * u01k(s.id, 5lu), 0.1 + 0.35 * u01k(s.id, 6lu), 0.0, 0.0, 0.0, 0.0);
    return o;
}

struct CoAt {
    ok: bool,
    z: f32,
    d: f32,
    wre: f32,
}

/// `coast::atoll_at`.
fn co_atoll_at(it: Inst, p: vec3<f64>, gsd: f32) -> CoAt {
    var o: CoAt;
    o.ok = false;
    let r0 = it.v[0];
    let wr = it.v[1];
    let l = it.v[2];
    let up = normalize(vec3<f32>(it.center));
    var e = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(up.z) <= 0.999) {
        e = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), up));
    }
    let n = cross(up, e);
    let rel = vec3<f32>(p - it.center);
    let x = dot(rel, e);
    let y = dot(rel, n);
    let r = length(vec2<f32>(x, y));
    if (r > 1.2 * r0 + 8000.0 + 3.0 * gsd) {
        return o;
    }
    let th = atan2(y, x);
    let edge = 1.0 + 0.10 * sin(3.0 * th + TAU * u01k(it.id, 10lu)) + 0.06 * sin(5.0 * th + TAU * u01k(it.id, 11lu)) + 0.04 * sin(8.0 * th + TAU * u01k(it.id, 12lu));
    let d = r / edge - r0;
    let wre = max(wr, 1.5 * gsd);
    if (d > wre + 7000.0) {
        return o;
    }
    let ad = abs(d);
    var z = 0.0;
    if (d < -wre) {
        let t = smoothstep1(0.0, 1.0, (-d - wre) / (350.0 + 0.04 * r0));
        z = -l + (l - 0.3) * (1.0 - t);
    } else if (d <= wre) {
        let thr = 0.45 - it.v[3];
        let motu = smoothstep1(thr, thr + 0.12, co_nz(CO_SEED ^ it.id ^ 0x80lu, p, 700.0, gsd));
        z = -0.3 + 2.2 * motu * (1.0 - smoothstep1(0.4 * wre, wre, ad));
    } else {
        z = -0.3 - 0.5 * (d - wre);
    }
    o.ok = true;
    o.z = z;
    o.d = d;
    o.wre = wre;
    return o;
}

// ---------------------------------------------------------------- deltas (`coast::delta`)

/// `delta::Shape`.
struct CoDelta {
    plain: f32,
    hd: f32,
    front: f32,
    chan: f32,
    u: f32,
    r0: f32,
}

/// `delta::finger`: angle, length (radii), angular half width.
fn co_finger(id: u64, k: u32) -> vec3<f32> {
    var side = 0.98;
    if (u01k(id, 19lu) < 0.5) {
        side = -0.98;
    }
    var base = array<f32, 4>(0.0, -0.52, 0.5, side);
    let a = base[k] + 0.16 * (u01k(id, 20lu + u64(k)) - 0.5);
    var l = 0.5 + 0.4 * u01k(id, 10lu + u64(k));
    if (k == 0u) {
        l = 0.8 + 0.2 * u01k(id, 10lu);
    }
    return vec3<f32>(a, l, 0.09 + 0.05 * u01k(id, 14lu + u64(k)));
}

/// `delta::shape`: the delta (centre `c`, `bearing`, `reach`, `id`, river half width `hw`) at
/// `p` for pixels of `gsd` m.
fn coast_delta_shape(c: vec3<f64>, bearing: f64, reach: f32, id: u64, hw: f32, p: vec3<f64>, gsd: f32) -> CoDelta {
    let f = astamp_frame(c, bearing);
    let up = f[2];
    let d = f[3];
    let left = cross(up, d);
    let dp = vec3<f32>(p - c);
    let u = dot(dp, d);
    let v = dot(dp, left);
    let r0 = reach / 1.4;
    let bird = u01k(id, 4lu) < 0.35;
    let x = u + 0.25 * r0;
    let r = sqrt(x * x + v * v);
    let phi = atan2(v, x);
    let g = max(1.0 - (phi / 1.25) * (phi / 1.25), 0.0);
    var rm = 0.0;
    if (bird) {
        rm = 0.4 * r0 * sqrt(g);
        for (var k = 0u; k < 4u; k++) {
            let fk = co_finger(id, k);
            let e = (phi - fk.x) / fk.z;
            rm = max(rm, fk.y * r0 * exp(-e * e));
        }
    } else {
        let lob = 0.5 * cos(5.0 * phi + TAU * u01k(id, 5lu)) + 0.3 * cos(9.0 * phi + TAU * u01k(id, 6lu)) + 0.2 * cos(14.0 * phi + TAU * u01k(id, 7lu));
        rm = r0 * sqrt(g) * (1.0 + 0.14 * lob);
    }
    let edge = max(0.05 * r0, gsd);
    var o: CoDelta;
    o.plain = 0.0;
    o.front = 0.0;
    if (rm > 0.0) {
        o.plain = smoothstep1(rm, rm - edge, r);
        o.front = smoothstep1(1.3 * rm + edge, rm, r) * (1.0 - o.plain);
    }
    o.hd = 0.5 + 1.5 * clamp(1.0 - r / r0, 0.0, 1.0);
    // distributaries from the apex: along the fingers (bird's-foot) or five forking ones (fan)
    let cw0 = max(0.5 * hw, 10.0);
    var chan = 0.0;
    var nk = 5u;
    if (bird) {
        nk = 4u;
    }
    for (var k = 0u; k < nk; k++) {
        var a = 0.0;
        if (bird) {
            a = co_finger(id, k).x;
        } else {
            a = (f32(k) - 2.0) * 0.42 + 0.14 * (u01k(id, 24lu + u64(k)) - 0.5);
        }
        a += 0.05 * sin(r / (0.21 * r0) + TAU * u01k(id, 30lu + u64(k)));
        var fork = 0.0;
        if (!bird) {
            fork = 0.11 * smoothstep1(0.5 * r0, 0.75 * r0, r);
        }
        for (var si = 0; si < 2; si++) {
            let s = f32(2 * si - 1);
            let dphi = abs(phi - a - s * fork);
            if (dphi <= 1.2) {
                let dist = r * sin(dphi);
                let cw = (1.0 - 0.45 * min(r / r0, 1.0)) * cw0;
                let cov = (min(dist + 0.5 * gsd, cw) - max(dist - 0.5 * gsd, -cw)) / gsd;
                let ext = smoothstep1(rm + 2.0 * cw + edge, rm, r);
                chan = max(chan, clamp(cov, 0.0, 1.0) * ext);
            }
            if (fork == 0.0) {
                break;
            }
        }
    }
    o.chan = chan;
    o.u = u;
    o.r0 = r0;
    return o;
}

/// `delta::relief` (`Kit::astamp`): the plain, the front and the distributaries.
fn coast_astamp(c: Ctx, st: Sink, ai: AStampIn, h: ptr<function, f32>) {
    if ((st.id & 0xFFlu) != 1lu) {
        return;
    }
    let sh = coast_delta_shape(st.c.xyz, st.c.w, -st.rad, st.id, st.level, c.p, c.gsd);
    if (sh.plain <= 0.0 && sh.front <= 0.0 && sh.chan <= 0.0) {
        return;
    }
    let h0 = *h;
    // the river's own channel upstream of the mouth stays
    var keep = 0.0;
    if (ai.river_hw > 0.0 && abs(ai.river_d) < 1.2 * ai.river_hw) {
        keep = 1.0 - smoothstep1(-0.05 * sh.r0, 0.1 * sh.r0, sh.u);
    }
    // lobes grow on the shelf only
    let shelf = smoothstep1(-60.0, -35.0, h0);
    var ht = h0;
    if (h0 < sh.hd) {
        ht = sh.hd;
    } else if (h0 < sh.hd + 4.0) {
        ht = sh.hd + (h0 - sh.hd) * (h0 - sh.hd) / 4.0;
    }
    var hn = h0 + (ht - h0) * sh.plain * shelf * (1.0 - keep);
    if (hn < -2.0) {
        hn += (-2.0 - hn) * sh.front * shelf;
    }
    hn += (min(hn, -1.5) - hn) * sh.chan * (1.0 - keep);
    *h = hn;
}
