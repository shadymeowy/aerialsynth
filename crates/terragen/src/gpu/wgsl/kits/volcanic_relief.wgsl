// The volcanic kit's families, relief and shared geometry (`kits/volcanic.rs`; every module with
// pass A): the same functions in the same order of operations.

const VC_SEED_ROUGH: u64 = 0x701CA002lu;
const VC_SEED_FLANK: u64 = 0x701CA001lu;
const VC_SEED_LAVA: u64 = 0x701CA003lu;
const VC_TWO_OVER_PI: f32 = 0.6366197723675814;
const VC_GULLY_AMP: f32 = 0.04;
const VC_K_CONE: f32 = 0.0085816;
const VC_K_MAAR: f32 = -0.0012251;
const VC_MEAN_CINDER: f32 = 0.1790;
const VC_MEAN_LAVA: f32 = 135.0;

// ---------------------------------------------------------------- existence

const VC_PRE_VOLCANISM: f32 = 0.08;

/// `volcanic::pre_volcanism`: the volcanism of the atlas texel nearest to a site.
fn volcanic_pre(center: vec3<f64>) -> f32 {
    let r = f32(atlas[2u]);
    let l = atlas_locate(normalize(vec3<f32>(center)), r);
    let n = min(vec2<u32>(floor(l.yz + 0.5)), vec2<u32>(atlas[2u] - 1u));
    let t = atlas_texel(u32(l.x), n.x + ATLAS_APRON, n.y + ATLAS_APRON);
    return unpack2x16float(atlas[t + 5u]).y;
}

/// `volcanic::volcano_exists`.
fn volcanic_volcanoes_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 1lu) >= 0.55 || volcanic_pre(s.center) < VC_PRE_VOLCANISM) {
        return o;
    }
    let a = atlas_sample(vec3<f32>(normalize(s.center)));
    let oceanic = a.elevation < -150.0;
    let vol = a.volcanism;
    var need = 0.2;
    if (oceanic) {
        need = 0.36;
    }
    if (vol < need) {
        return o;
    }
    if (oceanic && !(atlas_hotspot(a) || atlas_boundary(a) == 4u)) {
        return o;
    }
    if (u01k(s.id, 2lu) >= 0.15 + 0.85 * smoothstep1(need, need + 0.35, vol)) {
        return o;
    }
    var kind = 0.0;
    if (u01k(s.id, 3lu) < 0.2) {
        kind = 1.0;
    }
    let e = a.elevation;
    var summit = max(e, 0.0) + 2000.0 + 1700.0 * u01k(s.id, 4lu);
    if (oceanic) {
        summit = 250.0 + 1500.0 * u01k(s.id, 4lu);
    }
    let h = summit - e;
    let r = clamp(h * (3.6 + 0.9 * u01k(s.id, 5lu)), 6000.0, 25500.0);
    var rc = 220.0 + 500.0 * u01k(s.id, 6lu);
    var d = 70.0 + 150.0 * u01k(s.id, 7lu);
    if (kind > 0.5) {
        rc = 1600.0 + 2200.0 * u01k(s.id, 6lu);
        d = 350.0 + 450.0 * u01k(s.id, 7lu);
    }
    let zrim = e + h * pow(max(1.0 - rc / r, 0.0), 1.6);
    var lake = -1.0e6;
    if (kind > 0.5 && u01k(s.id, 8lu) < 0.8 && zrim - d > 30.0) {
        lake = zrim - 0.82 * d;
    } else if (kind < 0.5 && rc > 350.0 && u01k(s.id, 8lu) < 0.25 && zrim - d > 30.0) {
        lake = zrim - 0.65 * d;
    }
    o.ok = true;
    o.v = array<f32, 8>(r, e, h, rc, d, lake, u01k(s.id, 9lu), kind);
    return o;
}

/// `volcanic::shield_exists`.
fn volcanic_shields_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 1lu) >= 0.6 || volcanic_pre(s.center) < VC_PRE_VOLCANISM) {
        return o;
    }
    let a = atlas_sample(vec3<f32>(normalize(s.center)));
    let vol = a.volcanism;
    let rift = atlas_boundary(a) == 5u && atlas_continental(a);
    if (vol < 0.3 || !(atlas_hotspot(a) || (rift && u01k(s.id, 12lu) < 0.4))) {
        return o;
    }
    let oceanic = a.elevation < -150.0;
    let e = a.elevation;
    var summit = max(e, 0.0) + 700.0 + 900.0 * u01k(s.id, 4lu);
    if (oceanic) {
        summit = 400.0 + 2000.0 * u01k(s.id, 4lu);
    }
    let h = summit - e;
    let r = clamp(h * (7.0 + 3.0 * u01k(s.id, 5lu)), 22000.0, 45000.0);
    let rc = 900.0 + 1100.0 * u01k(s.id, 6lu);
    let d = 90.0 + 110.0 * u01k(s.id, 7lu);
    let zrim = e + h * pow(max(1.0 - rc / r, 0.0), 1.3);
    var lake = -1.0e6;
    if (u01k(s.id, 8lu) < 0.15 && zrim - d > 30.0) {
        lake = zrim - 0.7 * d;
    }
    o.ok = true;
    o.v = array<f32, 8>(r, e, h, rc, d, lake, u01k(s.id, 9lu), 2.0);
    return o;
}

/// `volcanic::field_exists`.
fn volcanic_fields_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 1lu) >= 0.7 || volcanic_pre(s.center) < VC_PRE_VOLCANISM) {
        return o;
    }
    let a = atlas_sample(vec3<f32>(normalize(s.center)));
    let vol = a.volcanism;
    if (a.elevation < -30.0 || vol < 0.18 || u01k(s.id, 2lu) >= 0.2 + 0.8 * smoothstep1(0.18, 0.45, vol)) {
        return o;
    }
    o.ok = true;
    o.v = array<f32, 8>(5500.0 + 3000.0 * u01k(s.id, 3lu), 550.0 + 350.0 * u01k(s.id, 4lu), 0.8 + 0.5 * u01k(s.id, 5lu), 0.2 + 0.3 * u01k(s.id, 6lu), u01k(s.id, 7lu), u01k(s.id, 8lu), 0.04 + 0.14 * u01k(s.id, 9lu), 0.0);
    return o;
}

// ---------------------------------------------------------------- geometry

struct VcFrame {
    e: vec3<f32>,
    n: vec3<f32>,
}

/// `volcanic::frame`.
fn vc_frame(center: vec3<f64>) -> VcFrame {
    let up = normalize(vec3<f32>(center));
    var f: VcFrame;
    f.e = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(up.z) <= 0.999) {
        f.e = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), up));
    }
    f.n = cross(up, f.e);
    return f;
}

fn vc_smax(a: f32, b: f32, k: f32) -> f32 {
    let t = max(k - abs(a - b), 0.0);
    return max(a, b) + t * t / (4.0 * k);
}

fn vc_gully_n(j: u32) -> f32 {
    if (j == 0u) {
        return 14.0;
    }
    if (j == 1u) {
        return 30.0;
    }
    return 62.0;
}
fn vc_gully_a(j: u32) -> f32 {
    if (j == 0u) {
        return 1.0;
    }
    if (j == 1u) {
        return 0.6;
    }
    return 0.35;
}
fn vc_rough_l(k: u32) -> f32 {
    if (k == 0u) {
        return 1600.0;
    }
    if (k == 1u) {
        return 520.0;
    }
    return 170.0;
}
fn vc_rough_a(k: u32) -> f32 {
    if (k == 0u) {
        return 0.02;
    }
    if (k == 1u) {
        return 0.01;
    }
    return 0.0045;
}

/// `volcanic::ravine_amp`.
fn vc_ravine_amp(id: u64, j: u32, phi: f32) -> f32 {
    let x = phi / PI;
    let k = floor(x);
    let n = i64(vc_gully_n(j));
    let k0 = i64(k);
    let f = x - k;
    let a0 = 0.55 + 0.9 * u01k(hash2(id, i64(j), ((k0 % n) + n) % n), 1lu);
    let a1 = 0.55 + 0.9 * u01k(hash2(id, i64(j), (((k0 + 1) % n) + n) % n), 1lu);
    return mix(a0, a1, smoothstep1(0.0, 1.0, f));
}

struct VcVol {
    id: u64,
    center: vec3<f64>,
    r: f32,
    e: f32,
    h: f32,
    rc: f32,
    d: f32,
    lake: f32,
    act: f32,
    kind: u32,
}

fn vc_vol(it: Inst) -> VcVol {
    var v: VcVol;
    v.id = it.id;
    v.center = it.center;
    v.r = it.v[0];
    v.e = it.v[1];
    v.h = it.v[2];
    v.rc = it.v[3];
    v.d = it.v[4];
    v.lake = it.v[5];
    v.act = it.v[6];
    v.kind = u32(it.v[7]);
    return v;
}

/// `Vol::pf`.
fn vc_pf(v: VcVol, q: f32) -> f32 {
    var ex = 1.6;
    if (v.kind == 2u) {
        ex = 1.3;
    }
    return pow(max(1.0 - q, 0.0), ex);
}

struct VcCone {
    ok: bool,
    x: f32,
    y: f32,
    rr: f32,
    u: f32,
    th: f32,
    rce: f32,
    z: f32,
    bowl: bool,
}

/// `volcanic::cone`.
fn vc_cone(v: VcVol, p: vec3<f64>, gsd: f32, full: bool) -> VcCone {
    var c: VcCone;
    c.ok = false;
    c.bowl = false;
    c.x = 0.0;
    c.y = 0.0;
    c.rr = 0.0;
    c.u = 0.0;
    c.th = 0.0;
    c.rce = 0.0;
    c.z = 0.0;
    let rw = max(v.r, 2.0 * gsd);
    let vh = v.h * (v.r / rw) * (v.r / rw);
    let fr = vc_frame(v.center);
    let rel = vec3<f32>(p - v.center);
    let x = dot(rel, fr.e);
    let y = dot(rel, fr.n);
    let r = sqrt(x * x + y * y);
    if (r >= 1.14 * rw) {
        return c;
    }
    let th = atan2(y, x);
    let edge = 1.0 + 0.07 * sin(3.0 * th + TAU * u01k(v.id, 10lu)) + 0.04 * sin(5.0 * th + TAU * u01k(v.id, 11lu)) + 0.025 * sin(9.0 * th + TAU * u01k(v.id, 12lu));
    let rr = r / edge;
    if (rr >= rw) {
        return c;
    }
    let u = rr / rw;
    let rce = max(v.rc, 1.5 * gsd);
    let de = v.d * (v.rc / rce) * (v.rc / rce);
    let zrim = v.e + vh * vc_pf(v, rce / rw);
    c.ok = true;
    c.x = x;
    c.y = y;
    c.rr = rr;
    c.u = u;
    c.th = th;
    c.rce = rce;
    if (rr < rce) {
        let q = rr / rce;
        if (v.kind == 1u) {
            c.z = zrim - de + de * smoothstep1(0.5, 1.0, q);
        } else {
            c.z = zrim - de * (1.0 - q * q);
        }
        c.bowl = true;
        return c;
    }
    var z = v.e + vh * vc_pf(v, u);
    if (full) {
        let fl = smoothstep1(1.0, 1.5, rr / rce);
        let f = fl * (1.0 - smoothstep1(0.7, 1.0, u));
        if (f > 0.0) {
            for (var k = 0u; k < 3u; k++) {
                let lam = vc_rough_l(k);
                let b = band(lam, gsd);
                if (b > 0.0) {
                    z += vc_rough_a(k) * vh * perlin3(VC_SEED_ROUGH ^ v.id ^ u64(k), vec3<f64>(f64(x / lam), f64(y / lam), 0.7lf)) * b * f;
                }
            }
        }
        if (v.kind != 2u) {
            let a = VC_GULLY_AMP * vh * smoothstep1(0.03, 0.16, u) * (1.0 - smoothstep1(0.5, 0.92, u)) * fl;
            if (a > 0.0) {
                let thw = th + 0.06 * perlin3(VC_SEED_FLANK ^ v.id, vec3<f64>(f64(x / 1800.0), f64(y / 1800.0), 0.37lf));
                for (var j = 0u; j < 3u; j++) {
                    let n = vc_gully_n(j);
                    let lam = TAU * rr / n;
                    let b = band(lam, gsd);
                    if (b > 0.0) {
                        let phi = 0.5 * n * thw + TAU * u01k(v.id, 13lu + u64(j));
                        z += a * vc_gully_a(j) * (abs(cos(phi)) - VC_TWO_OVER_PI) * vc_ravine_amp(v.id, j, phi) * b;
                    }
                }
            }
        }
    }
    c.z = z;
    return c;
}

/// `volcanic::apply`.
fn vc_apply(h: ptr<function, f32>, c: VcCone) {
    if (c.bowl) {
        *h = c.z;
        return;
    }
    let w = smoothstep1(1.0, 0.7, c.u);
    let m = vc_smax(*h, c.z, 40.0);
    *h += w * (m - *h);
}

/// `volcanic::cone_profile`.
fn vc_cone_profile(rho: f32) -> f32 {
    let g1 = pow(max(1.0 - max(rho, 0.22), 0.0), 1.35);
    let g0 = pow(0.78, 1.35);
    return g1 - 0.45 * g0 * max(1.0 - (rho / 0.22) * (rho / 0.22), 0.0);
}

struct VcField {
    z: f32,
    cinder: f32,
    chash: u64,
    crho: f32,
    ring: f32,
    lake: f32,
    lava: f32,
    age: f32,
    apron: f32,
    dens: f32,
}

/// `volcanic::ragged`.
fn vc_ragged(qx: f32, qy: f32, r: f32, h: u64) -> f32 {
    let th = atan2(qy, qx);
    return sqrt(qx * qx + qy * qy) / (r * (1.0 + 0.12 * sin(3.0 * th + TAU * u01k(h, 15lu)) + 0.08 * sin(5.0 * th + TAU * u01k(h, 16lu))));
}

fn vc_tongue_w(sg: f32, w0: f32) -> f32 {
    return w0 * pow(max(sin(PI * sg), 0.0), 0.6);
}

/// `volcanic::margin_noise`.
fn vc_margin_noise(id: u64, x: f32, y: f32, gsd: f32) -> f32 {
    let b = band(90.0, gsd);
    if (b > 0.0) {
        return perlin3(VC_SEED_LAVA ^ id, vec3<f64>(f64(x / 90.0), f64(y / 90.0), 0.37lf)) * b;
    }
    return 0.0;
}

/// `volcanic::tongue`: (cover, half width).
fn vc_tongue(x: f32, y: f32, dir: f32, s0: f32, len: f32, w0: f32, ph: f32, nzv: f32, fw: f32) -> vec2<f32> {
    let sn = sin(dir);
    let cs = cos(dir);
    let s = x * cs + y * sn;
    let t = -x * sn + y * cs;
    let sg = (s - s0) / (len - s0);
    if (sg <= 0.0 || sg >= 1.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let wander = 0.035 * len * sin(s / (0.22 * len) + ph);
    let w = vc_tongue_w(sg, w0) * (1.0 + 0.22 * nzv);
    return vec2<f32>(clamp((w - abs(t - wander)) / fw + 0.5, 0.0, 1.0), w);
}

fn vc_dens_at(dc: f32, rf: f32) -> f32 {
    return 0.8 * (1.0 - smoothstep1(0.35 * rf, rf, dc));
}

/// `volcanic::field`.
fn vc_field(it: Inst, p: vec3<f64>, gsd: f32, fw: f32, look: bool) -> VcField {
    var o: VcField;
    o.z = 0.0;
    o.cinder = 0.0;
    o.chash = 0lu;
    o.crho = 0.0;
    o.ring = 0.0;
    o.lake = 0.0;
    o.lava = 0.0;
    o.age = 0.0;
    o.apron = 0.0;
    o.dens = 0.0;
    let rf = it.v[0];
    let cs = it.v[1];
    let amp = it.v[2];
    let lshare = it.v[3];
    let lage = it.v[5];
    let mshare = it.v[6];
    let fr = vc_frame(it.center);
    let rel = vec3<f32>(p - it.center);
    let x = dot(rel, fr.e);
    let y = dot(rel, fr.n);
    let d = sqrt(x * x + y * y);
    if (d >= rf + 1.2 * cs) {
        return o;
    }
    let dm = vc_dens_at(d, rf);
    o.dens = dm;
    o.apron = 1.0 - smoothstep1(0.55 * rf, 1.1 * rf, d);
    let bnd = band(0.35 * cs, gsd);
    let mean_z = dm * cs * amp * ((1.0 - mshare) * VC_K_CONE + mshare * VC_K_MAAR);
    o.z = (1.0 - bnd) * mean_z;
    o.cinder = (1.0 - bnd) * dm * (1.0 - mshare) * VC_MEAN_CINDER;
    o.lava = (1.0 - bnd) * dm * (1.0 - mshare) * lshare * VC_MEAN_LAVA / cs;
    o.age = lage;
    if (bnd <= 0.0) {
        return o;
    }
    let ix = i64(floor(x / cs));
    let iy = i64(floor(y / cs));
    var zs = 0.0;
    var cinder = 0.0;
    var lava = 0.0;
    var age = lage;
    var nzv = 0.0;
    if (look) {
        nzv = vc_margin_noise(it.id, x, y, gsd);
    }
    let fx = x / cs - f32(ix);
    let fy = y / cs - f32(iy);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (!look && ((dx < 0 && fx > 0.35) || (dx > 0 && fx < 0.65) || (dy < 0 && fy > 0.35) || (dy > 0 && fy < 0.65))) {
                continue;
            }
            let gx = ix + i64(dx);
            let gy = iy + i64(dy);
            let h = hash2(it.id, gx, gy);
            let cx = (f32(gx) + 0.5 + 0.7 * (u01k(h, 2lu) - 0.5)) * cs;
            let cy = (f32(gy) + 0.5 + 0.7 * (u01k(h, 3lu) - 0.5)) * cs;
            if (u01k(h, 1lu) >= vc_dens_at(sqrt(cx * cx + cy * cy), rf)) {
                continue;
            }
            let qx = x - cx;
            let qy = y - cy;
            let dd = sqrt(qx * qx + qy * qy);
            if (u01k(h, 7lu) < mshare) {
                let rm = cs * (0.12 + 0.12 * u01k(h, 4lu));
                let dp = rm * (0.18 + 0.12 * u01k(h, 5lu)) * amp;
                let rho = vc_ragged(qx, qy, rm, h);
                if (rho < 1.45) {
                    if (rho < 1.0) {
                        zs -= dp * (1.0 - smoothstep1(0.55, 1.0, rho));
                    }
                    zs += 0.3 * dp * max(1.0 - ((rho - 1.0) / 0.45) * ((rho - 1.0) / 0.45), 0.0);
                    o.ring = max(o.ring, smoothstep1(0.7, 0.95, rho) * smoothstep1(1.45, 1.1, rho) * bnd);
                    o.lake = max(o.lake, smoothstep1(0.75, 0.62, rho) * bnd);
                }
            } else {
                let rc = cs * (0.16 + 0.2 * u01k(h, 4lu));
                let hc = 2.0 * rc * (0.2 + 0.16 * u01k(h, 5lu)) * amp;
                let rho = vc_ragged(qx, qy, rc, h);
                if (rho < 1.0) {
                    zs += hc * vc_cone_profile(rho);
                    if (smoothstep1(1.0, 0.8, rho) > cinder) {
                        cinder = smoothstep1(1.0, 0.8, rho);
                        o.chash = h;
                        o.crho = rho;
                    }
                }
                if (look && u01k(h, 8lu) < lshare) {
                    let dir = TAU * u01k(h, 9lu);
                    let len = cs * (0.55 + 0.45 * u01k(h, 10lu));
                    let w0 = 110.0 + 120.0 * u01k(h, 11lu);
                    let tg = vc_tongue(qx, qy, dir, 0.8 * rc, len, w0, TAU * u01k(h, 12lu), nzv, fw);
                    if (tg.x > lava) {
                        lava = tg.x;
                        age = clamp(lage + 0.5 * (u01k(h, 13lu) - 0.5), 0.0, 1.0);
                    }
                }
            }
        }
    }
    o.z += bnd * zs;
    o.cinder += bnd * cinder;
    if (lava > 0.0) {
        o.lava += bnd * lava;
        o.age = age;
    }
    return o;
}

// ---------------------------------------------------------------- relief

/// `volcanic::relief`.
fn volcanic_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {
    for (var f = 0u; f < 2u; f++) {
        let l = inst_list(FAM_VOLCANIC_VOLCANOES + f, r.blk, c.p);
        for (var i = 0u; i < l.n; i++) {
            let cn = vc_cone(vc_vol(l.items[i]), c.p, c.gsd, true);
            if (cn.ok) {
                vc_apply(h, cn);
            }
        }
    }
    let l2 = inst_list(FAM_VOLCANIC_FIELDS, r.blk, c.p);
    for (var i = 0u; i < l2.n; i++) {
        *h += vc_field(l2.items[i], c.p, c.gsd, c.gsd, false).z;
    }
}
