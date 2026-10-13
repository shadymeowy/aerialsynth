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
const MT_TOWER_DENS: f32 = 0.7;
const MT_TOWER_MEAN: f32 = 31.51;
const MT_COCKPIT_MEAN: f32 = 15.24;
const MT_RIDGE_CELL: f64 = 140000.0lf;
const MT_CREASE_MEAN: f32 = 0.4097;

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
    style: f32,
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
    o.fold = max(old, fore) * (1.0 - 0.5 * hard) * land * (1.0 - o.tower) * smoothstep1(-0.1, 0.3, z.style);
    o.fault = smoothstep1(-0.04, -0.18, z.uplift) * smoothstep1(0.08, 0.25, z.cont) * (1.0 - 0.6 * smoothstep1(0.3, 0.7, z.mountain));
    o.badland = soft * (1.0 - smoothstep1(0.14, 0.28, z.moist)) * smoothstep1(2.0, 10.0, z.temp) * (1.0 - smoothstep1(0.3, 0.6, z.mountain)) * (1.0 - smoothstep1(0.05, 0.3, z.sand)) * patch_b * land;
    o.drown = 1.0 - smoothstep1(-1.0, 10.0, z.coast_km);
    return o;
}

/// `mountains::cone_h`.
fn mt_cone_h(c: Cell3, p: vec3<f64>, east: vec3<f32>, north: vec3<f32>, kind: u32) -> f32 {
    let tower = kind == 0u;
    if (tower && u01k(c.id, 1lu) >= MT_TOWER_DENS) {
        return 0.0;
    }
    var hh = 60.0 + 100.0 * u01k(c.id, 4lu);
    var r0 = 1.1 + 0.4 * u01k(c.id, 3lu);
    if (tower) {
        hh = 110.0 + 190.0 * u01k(c.id, 2lu);
        r0 = 0.75 + 0.5 * u01k(c.id, 3lu);
    }
    let d = vec3<f32>(p - c.point);
    let dx = dot(d, east);
    let dy = dot(d, north);
    let r = max(sqrt(dx * dx + dy * dy), 1e-6);
    let co = dx / r;
    let si = dy / r;
    let c2 = co * co - si * si;
    let s2 = 2.0 * co * si;
    let c4 = c2 * c2 - s2 * s2;
    let s4 = 2.0 * c2 * s2;
    let sa = sin(TAU * u01k(c.id, 5lu));
    let ca = cos(TAU * u01k(c.id, 5lu));
    let sb = sin(TAU * u01k(c.id, 6lu));
    let cb = cos(TAU * u01k(c.id, 6lu));
    let rad = 0.5 * MT_KARST_CELL * r0 * (1.0 + 0.2 * (s2 * ca + c2 * sa) + 0.1 * (s4 * cb + c4 * sb));
    let x = c.f1 / min(rad, 0.9 * c.f2);
    if (x >= 1.0) {
        return 0.0;
    }
    if (tower) {
        return hh * sqrt(1.0 - x * x * sqrt(x)) * smoothstep1(1.0, 0.97, x);
    }
    return hh * (1.0 - x);
}

/// `mountains::ridges`.
fn mt_ridges(p: vec3<f64>, wf: f32, nw: f32, pl: f32) -> f32 {
    let q = p / MT_RIDGE_CELL;
    let i = floor(q);
    let f = vec3<f32>(q - i);
    let ix = i64(i.x);
    let iy = i64(i.y);
    let iz = i64(i.z);
    let sx = f.x * f.x * (3.0 - 2.0 * f.x);
    let sy = f.y * f.y * (3.0 - 2.0 * f.y);
    let sz = f.z * f.z * (3.0 - 2.0 * f.z);
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
                let hb = hash3(MT_SEED + 11lu, i64(floor(f64(ix + i64(dx)) / 4.0lf)), i64(floor(f64(iy + i64(dy)) / 4.0lf)), i64(floor(f64(iz + i64(dz)) / 4.0lf)));
                let th = TAU * (u01k(hb, 1lu) + 0.18 * (u01k(h, 1lu) - 0.5));
                let nn = e * cos(th) + n * sin(th);
                let lamk = 0.7 + 0.6 * u01k(h, 2lu);
                let along = dot(vec3<f32>(p - corner), nn);
                let ph = u01k(h, 3lu) + 1.4 * nw;
                var xf = along / (9000.0 * lamk) + ph;
                var xk = along / (16000.0 * lamk) + ph;
                xf = xf - floor(xf);
                xk = xk - floor(xk);
                let t = 1.0 - abs(2.0 * xf - 1.0);
                let pf = 700.0 * (pow(t, 1.6) - 1.0 / 2.6);
                var pk = 0.0;
                if (xk < 0.3) {
                    pk = 1150.0 * -0.35;
                } else if (xk < 0.8) {
                    let s = (xk - 0.3) / 0.5;
                    pk = 1150.0 * (s * s * (3.0 - 2.0 * s) - 0.35);
                } else {
                    let s = (1.0 - xk) / 0.2;
                    pk = 1150.0 * (s * s * (3.0 - 2.0 * s) - 0.35);
                }
                v += w * (0.7 + 0.6 * u01k(h, 4lu)) * lerp(pf, pk, wf);
                sw += w;
            }
        }
    }
    return v / sw * (0.25 + 0.75 * pl);
}

/// `mountains::crease`.
fn mt_crease(n: f32) -> f32 {
    return pow(n * n + 0.04, 0.35);
}

/// `mountains::badlands`.
fn mt_badlands(p: vec3<f64>, gsd: f32) -> f32 {
    var v = 0.0;
    let b0 = band(300.0, gsd);
    if (b0 > 0.0) {
        v += (mt_crease(perlin3(MT_SEED + 20lu, p / 300.0lf)) - MT_CREASE_MEAN) * b0;
    }
    let b1 = band(130.0, gsd);
    if (b1 > 0.0) {
        v += 0.55 * (mt_crease(perlin3(MT_SEED + 21lu, p / 130.0lf)) - MT_CREASE_MEAN) * b1;
    }
    let b2 = band(60.0, gsd);
    if (b2 > 0.0) {
        v += 0.25 * (mt_crease(perlin3(MT_SEED + 22lu, p / 60.0lf)) - MT_CREASE_MEAN) * b2;
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
    if (r >= 1.25 * rr) {
        return o;
    }
    let th = atan2(y, x);
    let co = x / max(r, 1e-6);
    let si = y / max(r, 1e-6);
    let c3 = 4.0 * co * co * co - 3.0 * co;
    let s3 = 3.0 * si - 4.0 * si * si * si;
    let c6 = c3 * c3 - s3 * s3;
    let s6 = 2.0 * s3 * c3;
    let sa = sin(TAU * u01k(it.id, 7lu));
    let ca = cos(TAU * u01k(it.id, 7lu));
    let sb = sin(TAU * u01k(it.id, 8lu));
    let cb = cos(TAU * u01k(it.id, 8lu));
    let k = 1.0 + 0.15 * (s3 * ca + c3 * sa) + 0.08 * (s6 * cb + c6 * sb);
    let rn = r / (rr * k);
    if (rn >= 1.0) {
        return o;
    }
    o.ok = true;
    o.rn = rn;
    let cs = (x * it.v[1] + y * it.v[2]) / max(r, 1.0);
    let sector = lerp(1.0, smoothstep1(0.0, 0.5, cs), smoothstep1(0.0, 0.05, rn));
    o.sec = sector * smoothstep1(0.0, 0.1, 1.0 - rn);
    o.th = th;
    return o;
}

/// `mountains::fans_exist`.
fn mountains_fans_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 1lu) >= 0.4) {
        return o;
    }
    let d = 12000.0;
    var up: array<f32, 5>;
    for (var k = 0u; k < 5u; k++) {
        var de = 0.0;
        var dn = 0.0;
        if (k == 1u) {
            de = d;
        } else if (k == 2u) {
            de = -d;
        } else if (k == 3u) {
            dn = d;
        } else if (k == 4u) {
            dn = -d;
        }
        let a = atlas_sample(vec3<f32>(normalize(s.center + vec3<f64>(s.east * de + s.north * dn))));
        if (k == 0u && (a.coast_km < 5.0 || moisture_index(a.precip_mm, a.temp_c) > 0.33 || a.temp_c < 0.0 || a.uplift < 0.12 || a.uplift > 0.55)) {
            return o;
        }
        up[k] = a.uplift;
    }
    let gx = up[1] - up[2];
    let gy = up[3] - up[4];
    let gl = sqrt(gx * gx + gy * gy);
    if (gl < 0.03) {
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
    zi.style = perlin3(MT_SEED + 5lu, c.p / 180000.0lf);
    let z = mt_zones(zi, fl.lit, fl.n1, fl.n2);
    let fans = inst_list(FAM_MOUNTAINS_FANS, r.blk, c.p);
    var ae = 0.0;
    if (z.tower + z.cockpit + z.fold + z.fault > 0.0 || fans.n > 0u) {
        ae = atlas_sample(vec3<f32>(normalize(c.p))).elevation;
    }
    if (z.tower + z.cockpit > 0.0) {
        let plain = lerp(ae, min(-11.0, ae), z.drown);
        *h = lerp(*h, plain, 0.88 * z.tower);
        let e = smoothstep1(1.2 * gsd, 3.0 * gsd, 0.45 * MT_KARST_CELL);
        var tw = MT_TOWER_MEAN;
        var cp = MT_COCKPIT_MEAN;
        if (e > 0.0) {
            let wc = worley3(MT_SEED + 30lu, c.p, f64(MT_KARST_CELL), 0.85);
            tw = lerp(MT_TOWER_MEAN, mt_cone_h(wc, c.p, c.east, c.north, 0u), e);
            cp = lerp(MT_COCKPIT_MEAN, mt_cone_h(wc, c.p, c.east, c.north, 1u), e);
        }
        *h += z.tower * tw + z.cockpit * (cp - MT_COCKPIT_MEAN);
    }
    let zt = min(z.fold + z.fault, 1.0);
    if (z.fold + z.fault > 0.0) {
        let wf = z.fault / (z.fold + z.fault);
        let b = lerp(band(6300.0, gsd), band(7700.0, gsd), wf);
        *h = lerp(*h, ae, 0.85 * zt);
        if (b > 0.0) {
            let nw = perlin3(MT_SEED + 3lu, c.p / 38000.0lf);
            let pl = smoothstep1(-0.55, 0.3, perlin3(MT_SEED + 4lu, c.p / 26000.0lf));
            *h += zt * mt_ridges(c.p, wf, nw, pl) * b;
        }
    }
    if (z.badland > 0.0) {
        *h += z.badland * (14.0 + 14.0 * smoothstep1(-0.3, 0.5, fl.n1)) * 2.4 * mt_badlands(c.p, gsd);
    }
    for (var k = 0u; k < fans.n; k++) {
        let it = fans.items[k];
        let fa = mt_fan_at(it, c.p, c.east, c.north);
        if (fa.ok) {
            let cone = it.v[3] * pow(1.0 - fa.rn, 1.3);
            *h = lerp(*h, ae + cone, 0.7 * fa.sec);
        }
    }
}

/// `mountains::fjord`: (weight, floor depth).
fn mt_fjord(m: Macro, mountain: f32, glac: f32) -> vec2<f32> {
    let ck = m.coast_km;
    let head = 30.0 + 40.0 * m.glaciation;
    let fj = glac * smoothstep1(0.2, 0.45, mountain) * (1.0 - smoothstep1(head - 6.0, head, ck)) * smoothstep1(-6.0, 0.0, ck);
    let prof = smoothstep1(0.0, 12.0, ck) * (1.0 - smoothstep1(head - 30.0, head, ck));
    return vec2<f32>(fj, 70.0 + (260.0 + 220.0 * mountain) * prof);
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
        (*cv).wall_k *= lerp(1.0, 0.75, glac);
        (*cv).fp_w *= lerp(1.0, 1.5, glac);
        var dmul = 2.0;
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
    let fjw = mt_fjord(m, ci.mountain, glac);
    var lv = 1.0;
    if (lvl1) {
        lv = 0.45;
    }
    let fj = fjw.x * lv;
    if (fj > 0.0) {
        (*cv).floor = lerp((*cv).floor, -fjw.y, fj);
        (*cv).depth_cap = lerp((*cv).depth_cap, 1500.0, fj);
        (*cv).fp_min = 0.5 - 2000.0 * fj;
        (*cv).fp_w = lerp((*cv).fp_w, min(0.5 * (*cv).fp_w, 450.0), fj);
        (*cv).wall_k *= 1.0 - 0.55 * fj;
        (*cv).narrow = fj;
        (*cv).u = max((*cv).u, fj);
    }
    let arid = 1.0 - smoothstep1(0.15, 0.45, ci.moist);
    if (arid > 0.0) {
        var l1 = 1.0;
        if (lvl1) {
            l1 = 0.35;
        }
        let cz = arid * smoothstep1(100.0, 700.0, ci.h0) * (1.0 - smoothstep1(0.3, 0.6, ci.mountain)) * smoothstep1(0.05, 0.3, ci.mesa) * smoothstep1(0.0, 0.15, m.cont) * l1;
        if (cz > 0.0) {
            let f0 = (*cv).floor;
            let prov = clamp(0.6 * smoothstep1(-0.05, 0.4, m.uplift) + 0.4 * smoothstep1(0.2, 0.8, m.plateau), 0.0, 1.0);
            (*cv).floor = lerp(f0, max(f0 - (400.0 + 1100.0 * prov), min(f0, 30.0)), cz);
            (*cv).depth_cap = lerp((*cv).depth_cap, 1500.0, cz);
            (*cv).wall_k *= lerp(1.0, 0.3, cz);
            (*cv).fp_w *= lerp(1.0, 0.6, cz);
            (*cv).steps = 50.0;
            (*cv).step_mix = max((*cv).step_mix, cz * band(160.0, gsd));
            (*cv).top = (*cv).floor + 500.0;
        }
    }
}
