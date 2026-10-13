// The volcanic kit's layer (`kits/volcanic.rs`): the same functions in the same order of
// operations. Geometry, families and relief: `volcanic_relief.wgsl`.

/// `volcanic::nz`.
fn vc_nz(seed: u64, p: vec3<f64>, lam: f32, gsd: f32) -> f32 {
    let b = band(lam, gsd);
    if (b > 0.0) {
        return perlin3(seed, p * (1.0lf / f64(lam))) * b;
    }
    return 0.0;
}

const VC_SEED_ASH: u64 = 0x701CA004lu;
const VC_SEED_HYD: u64 = 0x701CA005lu;

fn vc_ash_col(t: f32) -> vec3<f32> {
    return mixc(srgb(40.0, 38.0, 38.0), srgb(70.0, 63.0, 58.0), t);
}

struct VcLava {
    col: vec3<f32>,
    cls: u32,
    clear: f32,
}

/// `volcanic::lava_look`.
fn vc_lava_look(age: f32, p: vec3<f64>, gsd: f32) -> VcLava {
    let t1 = smoothstep1(0.22, 0.5, age);
    let t2 = 0.85 * smoothstep1(0.55, 0.95, age);
    let wf = (1.0 - t2) * (1.0 - t1);
    let wm = (1.0 - t2) * t1;
    let wo = t2;
    var c = vec3<f32>(0.0);
    if (wf > 0.0) {
        c += wf * srgb(24.0, 22.0, 24.0) * (1.0 + 0.2 * vc_nz(VC_SEED_LAVA, p, 14.0, gsd));
    }
    if (wm > 0.0) {
        c += wm * mixc(srgb(66.0, 58.0, 54.0), srgb(106.0, 70.0, 50.0), 0.5 + 0.5 * vc_nz(VC_SEED_LAVA + 1lu, p, 40.0, gsd));
    }
    if (wo > 0.0) {
        c += wo * mixc(srgb(88.0, 100.0, 66.0), srgb(124.0, 124.0, 98.0), 0.5 + 0.5 * vc_nz(VC_SEED_LAVA + 2lu, p, 25.0, gsd));
    }
    var o: VcLava;
    o.col = c;
    o.cls = 0u;
    if (age < 0.65) {
        o.cls = LC_LAVA;
    }
    o.clear = 1.0 - smoothstep1(0.5, 0.95, age);
    return o;
}

/// `volcanic::beach`.
fn vc_beach(s: ptr<function, Stack>) {
    let b = (*s).m.beach;
    if (b <= 0.0 || !((*s).t.volcanism > 0.3 || (*s).bio.st.litho == 3u)) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let dry = mixc(srgb(52.0, 50.0, 50.0), srgb(78.0, 72.0, 68.0), 0.5 + 0.5 * vc_nz(VC_SEED_ASH + 9lu, p, 9.0, gsd));
    let wet = srgb(26.0, 26.0, 28.0);
    let col = mixc(dry, wet, 1.0 - smoothstep1(0.05, 0.3, (*s).l.ground));
    composite(s, layer_paint(b, col, LC_VOLCANIC_ASH));
}

/// `volcanic::ash`.
fn vc_ash(s: ptr<function, Stack>, v: VcVol, k: VcCone) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let upper = (0.45 + 0.55 * v.act) * (1.0 - smoothstep1(0.1, 0.45, k.u));
    let apron = 0.9 * v.act * v.act * (1.0 - smoothstep1(0.2, 1.0, k.u));
    let m = max(upper, apron);
    if (m < 0.02) {
        return;
    }
    let n = 0.5 * sin(7.0 * k.th + TAU * u01k(v.id, 31lu)) * band(TAU * k.rr / 7.0, gsd)
        + 0.3 * sin(17.0 * k.th + TAU * u01k(v.id, 32lu)) * band(TAU * k.rr / 17.0, gsd)
        + 0.2 * sin(41.0 * k.th + TAU * u01k(v.id, 33lu)) * band(TAU * k.rr / 41.0, gsd)
        + 0.15 * vc_nz(VC_SEED_ASH + 1lu, p, 90.0, gsd);
    let wd = 0.07 + 0.3 * (1.0 - band(260.0, gsd));
    let cov = smoothstep1(-wd, wd, m - 0.35 + 0.5 * n);
    if (cov <= 0.0) {
        return;
    }
    var col = vc_ash_col(clamp(0.5 + 0.6 * n, 0.0, 1.0));
    col = mixc(col, srgb(104.0, 58.0, 42.0), 0.35 * smoothstep1(0.22, 0.06, k.u) * (0.3 + 0.7 * v.act));
    var ly = layer_paint(cov, col, LC_VOLCANIC_ASH);
    ly.clear = 0.95;
    composite(s, ly);
    (*s).m.natural_ok *= 1.0 - 0.8 * cov;
}

/// `volcanic::flows`.
fn vc_flows(s: ptr<function, Stack>, v: VcVol, k: VcCone) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let nf = 2.0 + 4.0 * v.act;
    let scale = clamp(v.r / 12000.0, 0.7, 1.6);
    let e = smoothstep1(1.2 * gsd, 3.0 * gsd, 200.0 * scale);
    var cov = 0.0;
    var age = 0.0;
    var mean = 0.0;
    var mage = 0.0;
    var nzv = 0.0;
    var have_nzv = false;
    for (var i = 0u; i < 6u; i++) {
        if (f32(i) >= nf) {
            break;
        }
        let hh = hash1(v.id, 50 + i64(i));
        let dir = TAU * u01k(hh, 1lu);
        var len = min(v.r * (0.3 + 0.55 * u01k(hh, 2lu)), 12000.0);
        if (v.kind == 2u) {
            len = min(v.r * (0.1 + 0.15 * u01k(hh, 2lu)), 7000.0);
        }
        let w0 = (90.0 + 200.0 * u01k(hh, 3lu)) * scale;
        let a = clamp((1.0 - v.act) * 0.8 + 0.6 * u01k(hh, 4lu) - 0.1, 0.0, 1.0);
        let s0 = 1.3 * k.rce;
        let ph = TAU * u01k(hh, 5lu);
        let d = rem_euclid(k.th - dir + PI, TAU) - PI;
        var tg = vec2<f32>(0.0, 0.0);
        if (e > 0.0 && abs(d) < 1.5 && k.rr * abs(sin(d)) < 1.3 * w0 + 0.04 * len + 2.0 * fw) {
            if (!have_nzv) {
                nzv = vc_margin_noise(v.id, k.x, k.y, gsd);
                have_nzv = true;
            }
            tg = vc_tongue(k.x, k.y, dir, s0, len, w0, ph, nzv, fw);
        }
        if (tg.x > cov) {
            cov = tg.x;
            age = a;
        }
        if (e < 1.0 && k.rr > s0 && k.rr < len) {
            let m = vc_tongue_w((k.rr - s0) / (len - s0), w0) / (PI * k.rr);
            if (m > mean) {
                mean = m;
                mage = a;
            }
        }
    }
    let c = e * cov + (1.0 - e) * mean;
    var a = mage;
    if (e * cov >= (1.0 - e) * mean) {
        a = age;
    }
    if (c <= 0.0) {
        return;
    }
    let lv = vc_lava_look(a, p, gsd);
    var ly = layer_paint(c, lv.col, lv.cls);
    ly.clear = lv.clear;
    composite(s, ly);
    (*s).m.natural_ok *= 1.0 - c * lv.clear;
}

/// `volcanic::hydrothermal`.
fn vc_hydro(s: ptr<function, Stack>, v: VcVol, k: VcCone) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let zone = 1.0 - smoothstep1(v.rc, v.rc * 2.6 + 200.0, k.rr);
    if (zone <= 0.0) {
        return;
    }
    let strength = 0.35 + 0.65 * v.act;
    let n = vc_nz(VC_SEED_HYD ^ v.id, p, 38.0, gsd) + 0.5 * vc_nz((VC_SEED_HYD ^ v.id) + 1lu, p, 11.0, gsd);
    let wd = 0.06 + 0.25 * (1.0 - band(38.0, gsd));
    let cov = smoothstep1(-wd, wd, n - (1.3 - zone * strength));
    if (cov <= 0.0) {
        return;
    }
    let pick = vc_nz(VC_SEED_HYD + 7lu, p, 90.0, gsd);
    let col = mixc(mixc(srgb(204.0, 178.0, 62.0), srgb(176.0, 92.0, 48.0), smoothstep1(-0.15, 0.2, pick)), srgb(176.0, 158.0, 118.0), smoothstep1(0.1, 0.5, n));
    var ly = layer_paint(cov, col, LC_BARE);
    ly.clear = 1.0;
    composite(s, ly);
    (*s).m.natural_ok *= 1.0 - cov;
}

/// `volcanic::crater_lake`.
fn vc_crater_lake(s: ptr<function, Stack>, v: VcVol, k: VcCone) {
    if (v.lake < -1.0e5 || k.rr > 1.02 * max(v.rc, k.rce)) {
        return;
    }
    let l = (*s).l;
    let depth = v.lake - l.ground;
    let cov = clamp(depth / max(l.slope * l.fw, 0.03) + 0.5, 0.0, 1.0);
    if (cov <= 0.0) {
        return;
    }
    let acid = u01k(v.id, 14lu) < 0.5 || v.kind == 0u;
    var deep = srgb(18.0, 56.0, 90.0);
    if (acid) {
        deep = srgb(54.0, 150.0, 138.0);
    }
    let col = mixc(deep, srgb(128.0, 176.0, 150.0), smoothstep1(25.0, 0.0, depth));
    var ly = layer_paint(cov, col, LC_LAKE);
    ly.hmode = HM_ABS;
    ly.dh = v.lake;
    ly.relit = 1.0;
    ly.water = true;
    ly.clear = 1.0;
    composite(s, ly);
    (*s).m.natural_ok *= 1.0 - cov;
}

/// `volcanic::field_look`.
fn vc_field_look(s: ptr<function, Stack>, it: Inst) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let f = vc_field(it, p, gsd, fw, true);
    if (f.dens <= 0.0 && f.apron <= 0.0) {
        return;
    }
    if (f.apron > 0.0) {
        let n = 0.5 + 0.5 * vc_nz(VC_SEED_ASH + 5lu, p, 120.0, gsd);
        let cov = 0.5 * f.apron * smoothstep1(0.2, 0.8, n);
        var ly = layer_paint(cov, vc_ash_col(n) * 0.9, LC_VOLCANIC_ASH);
        ly.clear = 0.8;
        composite(s, ly);
    }
    if (f.lava > 0.0) {
        let lv = vc_lava_look(f.age, p, gsd);
        var ly = layer_paint(f.lava, lv.col, lv.cls);
        ly.clear = lv.clear;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - f.lava * lv.clear;
    }
    if (f.cinder > 0.0) {
        let t = u01k(f.chash, 14lu);
        let rho = f.crho;
        var col = mixc(srgb(44.0, 38.0, 38.0), srgb(118.0, 62.0, 42.0), 0.65 * t * t);
        col *= mix(1.0, 0.72, smoothstep1(0.22, 0.12, rho)) * (1.0 + 0.15 * smoothstep1(0.15, 0.22, rho) * smoothstep1(0.4, 0.22, rho));
        col = mixc(col, vc_ash_col(u01k(f.chash, 17lu)) * 0.9, 0.5 * smoothstep1(0.55, 0.95, rho));
        col = mixc(col, (*s).col, 0.5 * smoothstep1(0.55, 1.0, rho));
        col = col * (1.0 + 0.18 * vc_nz(VC_SEED_ASH + 6lu, p, 9.0, gsd));
        var ly = layer_paint(f.cinder, col, LC_VOLCANIC_ASH);
        ly.clear = 1.0;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - f.cinder;
    }
    if (f.ring > 0.0) {
        var ly = layer_paint(f.ring, srgb(138.0, 118.0, 98.0), LC_VOLCANIC_ASH);
        ly.clear = 1.0;
        composite(s, ly);
    }
    if (f.lake > 0.0) {
        var ly = layer_paint(f.lake, srgb(30.0, 74.0, 80.0), LC_LAKE);
        ly.hmode = HM_BLEND;
        ly.dh = -0.2;
        ly.relit = 1.0;
        ly.water = true;
        ly.clear = 1.0;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - f.lake;
    }
}

fn volcanic_slot_azonal(s: ptr<function, Stack>) {
    vc_beach(s);
    let blk = (*s).l.blk;
    for (var f = 0u; f < 2u; f++) {
        let n = inst_count(FAM_VOLCANIC_VOLCANOES + f, blk);
        for (var i = 0u; i < n; i++) {
            let v = vc_vol(inst_get(FAM_VOLCANIC_VOLCANOES + f, blk, i));
            let k = vc_cone(v, (*s).c.p, (*s).c.gsd, false);
            if (k.ok) {
                if (k.u < 0.45 + 0.55 * v.act || k.rr < 1.6 * k.rce) {
                    (*s).no_town = true;
                }
                (*s).m.natural_ok *= smoothstep1(1.0, 1.6, k.rr / k.rce);
                vc_ash(s, v, k);
                vc_flows(s, v, k);
                vc_hydro(s, v, k);
                vc_crater_lake(s, v, k);
            }
        }
    }
    let n2 = inst_count(FAM_VOLCANIC_FIELDS, blk);
    for (var i = 0u; i < n2; i++) {
        vc_field_look(s, inst_get(FAM_VOLCANIC_FIELDS, blk, i));
    }
}
