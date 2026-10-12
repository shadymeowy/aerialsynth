// The mountains kit's fields, zones, relief, carve hook and fan family (`kits/mountains.rs`; every
// module with pass A, and pass B's layer): the same functions in the same order of operations.

const MT_SEED: u64 = 0x4D545000lu;

const MT_SEDIMENTARY: u32 = 0u;
const MT_CARBONATE: u32 = 1u;
const MT_CRYSTALLINE: u32 = 2u;
const MT_VOLCANIC: u32 = 3u;
const MT_UNCONSOLIDATED: u32 = 4u;

const MT_WARP_LAM1: f64 = 14000.0lf;
const MT_WARP_LAM2: f64 = 11000.0lf;
const MT_WARP_A: f32 = 4000.0;

const MT_KARST_CELL: f32 = 420.0;
const MT_TOWER_DENS: f32 = 0.55;
const MT_TOWER_MEAN: f32 = 33.4;
const MT_COCKPIT_MEAN: f32 = 24.1;
const MT_RIDGE_CELL: f64 = 140000.0lf;
const MT_CREASE_MEAN: f32 = 0.4075;

/// `mountains::litho_at`: the atlas' dominant lithology at the texel nearest to `p`.
fn mt_litho(p: vec3<f64>) -> u32 {
    let r = f32(atlas[2u]);
    let l = atlas_locate(normalize(vec3<f32>(p)), r);
    let n = min(vec2<u32>(floor(l.yz + 0.5)), vec2<u32>(atlas[2u] - 1u));
    let t = atlas_texel(u32(l.x), n.x + ATLAS_APRON, n.y + ATLAS_APRON);
    return atlas[t + 8u] & 7u;
}

struct MtFields {
    lit: u32,
    n1: f32,
    n2: f32,
}

/// `mountains::fields`: the lithology at the warped position and the two patch noises.
fn mt_fields(c: Ctx) -> MtFields {
    var o: MtFields;
    o.n1 = perlin3(MT_SEED + 1lu, c.p / MT_WARP_LAM1);
    o.n2 = perlin3(MT_SEED + 2lu, c.p / MT_WARP_LAM2);
    let q = c.p + vec3<f64>(c.east * (MT_WARP_A * o.n1) + c.north * (MT_WARP_A * o.n2));
    o.lit = mt_litho(q);
    return o;
}

struct MtZIn {
    moist: f32,
    temp: f32,
    mountain: f32,
    uplift: f32,
    cont: f32,
    coast_km: f32,
    sand: f32,
}

struct MtZones {
    tower: f32,
    cockpit: f32,
    fold: f32,
    fault: f32,
    badland: f32,
    drown: f32,
}

/// `mountains::zones`.
fn mt_zones(z: MtZIn, lit: u32, n1: f32, n2: f32) -> MtZones {
    let carb = f32(lit == MT_CARBONATE);
    let soft = f32(lit == MT_SEDIMENTARY || lit == MT_UNCONSOLIDATED);
    let hard = f32(lit == MT_CRYSTALLINE || lit == MT_VOLCANIC);
    let land = smoothstep1(0.0, 0.08, z.cont);
    let shelf = smoothstep1(-0.25, -0.1, z.cont) * smoothstep1(-30.0, -12.0, z.coast_km);
    let patch_a = smoothstep1(-0.45, -0.05, n1);
    let patch_b = smoothstep1(-0.2, 0.2, n2);
    let humid = smoothstep1(0.42, 0.58, z.moist) * smoothstep1(15.0, 21.0, z.temp);
    var o: MtZones;
    o.tower = carb * humid * (1.0 - smoothstep1(0.35, 0.7, z.mountain)) * patch_a * shelf;
    o.cockpit = carb * smoothstep1(0.28, 0.42, z.moist) * smoothstep1(4.0, 12.0, z.temp) * (1.0 - humid) * (1.0 - smoothstep1(0.4, 0.7, z.mountain)) * patch_a * land;
    let u = max(z.uplift, 0.0);
    let old = smoothstep1(0.1, 0.4, z.mountain) * (1.0 - smoothstep1(0.2, 0.5, u));
    let fore = smoothstep1(0.04, 0.15, u) * (1.0 - smoothstep1(0.3, 0.5, u)) * (1.0 - 0.7 * z.mountain);
    o.fold = max(old, fore) * (1.0 - 0.5 * hard) * land * (1.0 - o.tower);
    o.fault = smoothstep1(-0.04, -0.18, z.uplift) * smoothstep1(0.08, 0.25, z.cont) * (1.0 - 0.6 * smoothstep1(0.3, 0.7, z.mountain));
    o.badland = soft * (1.0 - smoothstep1(0.14, 0.28, z.moist)) * smoothstep1(2.0, 10.0, z.temp) * (1.0 - smoothstep1(0.3, 0.6, z.mountain)) * (1.0 - smoothstep1(0.05, 0.3, z.sand)) * patch_b * land;
    o.drown = 1.0 - smoothstep1(-1.0, 10.0, z.coast_km);
    return o;
}

/// `mountains::tower_h`.
fn mt_tower_h(c: Cell3) -> f32 {
    if (u01k(c.id, 1lu) >= MT_TOWER_DENS) {
        return 0.0;
    }
    let hh = 70.0 + 170.0 * u01k(c.id, 2lu);
    let rad = 0.5 * MT_KARST_CELL * (0.55 + 0.4 * u01k(c.id, 3lu));
    let x = c.f1 / min(rad, 0.88 * c.f2);
    if (x >= 1.0) {
        return 0.0;
    }
    return hh * pow(1.0 - x, 0.4);
}

/// `mountains::cockpit_h`.
fn mt_cockpit_h(c: Cell3) -> f32 {
    let hc = 35.0 + 45.0 * u01k(c.id, 4lu);
    return hc * (1.0 - smoothstep1(0.0, 0.4, (c.f2 - c.f1) / MT_KARST_CELL));
}

/// `mountains::ridges`: parallel ridges of the 8 lattice corners around `p`.
fn mt_ridges(p: vec3<f64>, kind: u32, nw: f32, pl: f32) -> f32 {
    let q = p / MT_RIDGE_CELL;
    let i = floor(q);
    let f = vec3<f32>(q - i);
    let ix = i64(i.x);
    let iy = i64(i.y);
    let iz = i64(i.z);
    let sx = f.x * f.x * (3.0 - 2.0 * f.x);
    let sy = f.y * f.y * (3.0 - 2.0 * f.y);
    let sz = f.z * f.z * (3.0 - 2.0 * f.z);
    var lam0 = 9000.0;
    if (kind == 1u) {
        lam0 = 16000.0;
    }
    var v = 0.0;
    var sw = 0.0;
    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                var wx = 1.0 - sx;
                if (dx == 1) {
                    wx = sx;
                }
                var wy = 1.0 - sy;
                if (dy == 1) {
                    wy = sy;
                }
                var wz = 1.0 - sz;
                if (dz == 1) {
                    wz = sz;
                }
                let wt = wx * wy * wz;
                let w = wt * wt;
                let h = hash3(MT_SEED + 10lu, ix + i64(dx), iy + i64(dy), iz + i64(dz));
                let corner = vec3<f64>(f64(ix + i64(dx)), f64(iy + i64(dy)), f64(iz + i64(dz))) * MT_RIDGE_CELL;
                let d = normalize(vec3<f32>(corner));
                var e = cross(vec3<f32>(0.0, 0.0, 1.0), d);
                if (length(e) < 1e-6) {
                    e = vec3<f32>(1.0, 0.0, 0.0);
                } else {
                    e = normalize(e);
                }
                let n = cross(d, e);
                let th = TAU * u01k(h, 1lu);
                let nn = e * cos(th) + n * sin(th);
                let lam = lam0 * (0.7 + 0.6 * u01k(h, 2lu));
                var x = dot(vec3<f32>(p - corner), nn) / lam + u01k(h, 3lu) + 1.4 * nw;
                x = x - floor(x);
                var prof = 0.0;
                if (kind == 0u) {
                    let t = 1.0 - abs(2.0 * x - 1.0);
                    prof = pow(t, 1.6) - 1.0 / 2.6;
                } else if (x < 0.8) {
                    let s = x / 0.8;
                    prof = s * s * (3.0 - 2.0 * s) - 0.5;
                } else {
                    let s = (1.0 - x) / 0.2;
                    prof = s * s * (3.0 - 2.0 * s) - 0.5;
                }
                v += w * (0.7 + 0.6 * u01k(h, 4lu)) * prof;
                sw += w;
            }
        }
    }
    return v / sw * (0.25 + 0.75 * pl);
}

/// `mountains::badlands`.
fn mt_badlands(p: vec3<f64>, gsd: f32) -> f32 {
    let b1 = band(260.0, gsd);
    let b2 = band(90.0, gsd);
    var v = 0.0;
    if (b1 > 0.0) {
        v += (pow(abs(perlin3(MT_SEED + 20lu, p / 260.0lf)), 0.7) - MT_CREASE_MEAN) * b1;
    }
    if (b2 > 0.0) {
        v += 0.45 * (pow(abs(perlin3(MT_SEED + 21lu, p / 90.0lf)), 0.7) - MT_CREASE_MEAN) * b2;
    }
    return v;
}

struct MtFan {
    ok: bool,
    sec: f32,
    rn: f32,
    th: f32,
}

/// `mountains::fan_at`.
fn mt_fan_at(it: Inst, p: vec3<f64>, east: vec3<f32>, north: vec3<f32>) -> MtFan {
    var o: MtFan;
    o.ok = false;
    let rel = vec3<f32>(p - it.center);
    let x = dot(rel, east);
    let y = dot(rel, north);
    let r = sqrt(x * x + y * y);
    let rr = it.v[0];
    if (r >= rr) {
        return o;
    }
    o.ok = true;
    o.rn = r / rr;
    let cs = (x * it.v[1] + y * it.v[2]) / max(r, 1.0);
    let sector = lerp(1.0, smoothstep1(0.0, 0.5, cs), smoothstep1(0.0, 0.05, o.rn));
    o.sec = sector * smoothstep1(0.0, 0.1, 1.0 - o.rn);
    o.th = atan2(y, x);
    return o;
}

/// `mountains::site_fields`: (mountain mask, moisture, continent value) of the macro fields.
fn mt_site_fields(c: vec3<f64>) -> vec3<f32> {
    let m = macro_at(c, 20000.0);
    return vec3<f32>(mountain_mask(m).x, clamp(m.moist, 0.0, 1.0), m.cont);
}

/// `mountains::fans_exist`.
fn mountains_fans_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 1lu) >= 0.55) {
        return o;
    }
    let a = atlas_sample(vec3<f32>(normalize(s.center)));
    if (a.coast_km < 5.0 || moisture_index(a.precip_mm, a.temp_c) > 0.36 || a.temp_c < 0.0 || a.uplift < 0.03) {
        return o;
    }
    let f0 = mt_site_fields(s.center);
    if (f0.x < 0.2 || f0.x > 0.6 || f0.y > 0.34 || f0.z < 0.1) {
        return o;
    }
    let d = 8000.0;
    var g: array<f32, 4>;
    for (var k = 0u; k < 4u; k++) {
        var de = 0.0;
        var dn = 0.0;
        if (k == 0u) {
            de = d;
        } else if (k == 1u) {
            de = -d;
        } else if (k == 2u) {
            dn = d;
        } else {
            dn = -d;
        }
        g[k] = mt_site_fields(s.center + vec3<f64>(s.east * de + s.north * dn)).x;
    }
    let gx = g[0] - g[1];
    let gy = g[2] - g[3];
    let gl = sqrt(gx * gx + gy * gy);
    if (gl < 0.012) {
        return o;
    }
    let rr = 2600.0 + 3000.0 * u01k(s.id, 2lu);
    let hh = rr * (0.016 + 0.014 * u01k(s.id, 3lu));
    o.ok = true;
    o.v = array<f32, 8>(rr, -gx / gl, -gy / gl, hh, 0.0, 0.0, 0.0, 0.0);
    return o;
}

/// `mountains::relief`.
fn mountains_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {
    let gsd = c.gsd;
    if (m.cont < -0.3) {
        return;
    }
    let fl = mt_fields(c);
    var zi: MtZIn;
    zi.moist = r.moist;
    zi.temp = r.temp;
    zi.mountain = r.mountain;
    zi.uplift = m.uplift;
    zi.cont = m.cont;
    zi.coast_km = m.coast_km;
    zi.sand = r.sand;
    let z = mt_zones(zi, fl.lit, fl.n1, fl.n2);
    if (z.tower + z.cockpit > 0.0) {
        let plain = lerp(r.smooth_h, min(-11.0, r.smooth_h), z.drown);
        *h = lerp(*h, plain, 0.88 * z.tower);
        let e = smoothstep1(1.2 * gsd, 3.0 * gsd, 0.45 * MT_KARST_CELL);
        var tw = MT_TOWER_MEAN;
        var cp = MT_COCKPIT_MEAN;
        if (e > 0.0) {
            let wc = worley3(MT_SEED + 30lu, c.p, f64(MT_KARST_CELL), 0.85);
            tw = lerp(MT_TOWER_MEAN, mt_tower_h(wc), e);
            cp = lerp(MT_COCKPIT_MEAN, mt_cockpit_h(wc), e);
        }
        *h += z.tower * tw + z.cockpit * (cp - MT_COCKPIT_MEAN);
    }
    if (z.fold + z.fault > 0.0) {
        var kind = 0u;
        var zz = z.fold;
        var lam_min = 6300.0;
        var amp = 240.0;
        if (z.fault > z.fold) {
            kind = 1u;
            zz = z.fault;
            lam_min = 7700.0;
            amp = 520.0;
        }
        let b = band(lam_min, gsd);
        *h = lerp(*h, r.smooth_h, 0.6 * zz);
        if (b > 0.0) {
            let nw = perlin3(MT_SEED + 3lu, c.p / 38000.0lf);
            let pl = smoothstep1(-0.55, 0.3, perlin3(MT_SEED + 4lu, c.p / 26000.0lf));
            *h += zz * amp * mt_ridges(c.p, kind, nw, pl) * b;
        }
    }
    if (z.badland > 0.0) {
        *h += z.badland * (14.0 + 14.0 * smoothstep1(-0.3, 0.5, fl.n1)) * 2.0 * mt_badlands(c.p, gsd);
    }
    let fans = inst_list(FAM_MOUNTAINS_FANS, r.blk, c.p);
    for (var k = 0u; k < fans.n; k++) {
        let it = fans.items[k];
        let fa = mt_fan_at(it, c.p, c.east, c.north);
        if (fa.ok) {
            let cone = it.v[3] * pow(1.0 - fa.rn, 1.3);
            *h = lerp(*h, r.smooth_h + cone, 0.7 * fa.sec);
        }
    }
}

/// `mountains::carve`.
fn mountains_carve(c: Ctx, m: Macro, ci: CarveIn, cv: ptr<function, Carve>) {
    if (ci.level > 1u) {
        return;
    }
    let lvl1 = ci.level == 1u;
    let gsd = c.gsd;
    let glac = smoothstep1(0.3, 0.6, m.glaciation) * smoothstep1(0.25, 0.55, ci.mountain);
    if (glac > 0.0) {
        (*cv).u = max((*cv).u, glac);
        (*cv).wall_k *= lerp(1.0, 0.5, glac);
        (*cv).fp_w *= lerp(1.0, 1.5, glac);
        var dmul = 2.6;
        var over = 90.0;
        if (lvl1) {
            dmul = 8.0;
            over = 40.0;
        }
        (*cv).depth_cap *= lerp(1.0, dmul, glac);
        (*cv).floor = lerp((*cv).floor, min((*cv).floor, max((*cv).floor - over, 1.0)), glac);
        if (lvl1 && ci.spring) {
            let ch = glac * smoothstep1(0.35, 0.7, ci.mountain);
            let cz = 1.0 - smoothstep1(0.0, 0.14, ci.u);
            (*cv).fp_w = max((*cv).fp_w, 240.0 * ch * cz);
            (*cv).wall_k *= lerp(1.0, 0.8, ch * cz);
            let lr = 130.0 * ch * (1.0 - smoothstep1(0.02, 0.06, ci.u));
            if (lr > 0.4 * gsd) {
                (*cv).lake_r = lr;
                (*cv).lake_level = (*cv).floor + 1.4;
            }
        }
    }
    let ck = m.coast_km;
    let head = 45.0 + 25.0 * m.style.w;
    var lv = 1.0;
    if (lvl1) {
        lv = 0.45;
    }
    let fj = glac * smoothstep1(0.2, 0.45, ci.mountain) * (1.0 - smoothstep1(head - 6.0, head, ck)) * smoothstep1(-6.0, 0.0, ck) * lv;
    if (fj > 0.0) {
        let prof = smoothstep1(0.0, 12.0, ck) * (1.0 - smoothstep1(head - 30.0, head, ck));
        let dep = 70.0 + (260.0 + 200.0 * max(m.style.x, 0.0)) * prof;
        (*cv).floor = lerp((*cv).floor, -dep, fj);
        (*cv).depth_cap = lerp((*cv).depth_cap, 1500.0, fj);
        (*cv).fp_min = 0.5 - 2000.0 * fj;
        (*cv).fp_w *= 1.0 + 1.5 * fj;
        (*cv).wall_k *= 1.0 - 0.4 * fj;
        (*cv).u = max((*cv).u, fj);
    }
    let arid = 1.0 - smoothstep1(0.22, 0.36, ci.moist);
    if (arid > 0.0) {
        let lit = mt_litho(c.p);
        let rock = f32(lit == MT_SEDIMENTARY || lit == MT_CARBONATE);
        var l1 = 1.0;
        if (lvl1) {
            l1 = 0.35;
        }
        let cz = arid * rock * smoothstep1(150.0, 450.0, ci.h0) * (1.0 - smoothstep1(0.3, 0.6, ci.mountain)) * smoothstep1(-0.15, 0.25, ci.mesa) * l1;
        if (cz > 0.0) {
            let dm = lerp(250.0, 1000.0, saturate1(0.5 + 0.6 * m.style.y));
            let dc = max(min(dm, 0.85 * (ci.h0 - 40.0)), 0.0);
            (*cv).floor = lerp((*cv).floor, max((*cv).floor - dc, min((*cv).floor, 30.0)), cz);
            (*cv).depth_cap = lerp((*cv).depth_cap, 1500.0, cz);
            (*cv).wall_k *= lerp(1.0, 0.3, cz);
            (*cv).fp_w *= lerp(1.0, 0.6, cz);
            (*cv).steps = 55.0 + 25.0 * m.style.z;
            (*cv).step_mix = max((*cv).step_mix, cz * band(160.0, gsd));
        }
    }
}
