// The transport kit (`kits/transport.rs`, `transport/px.rs`): roads, railways, power lines and
// their stamps (tunnel portals, stations, yards, roundabouts) in slot 7 (after the settlements'
// places), decks over water in slot 8. Same names (prefix `tr_`), same order of operations as the Rust.

const TR_KIND0: u32 = 208u;
const TR_MOTORWAY: u32 = 0u;
const TR_TRUNK: u32 = 1u;
const TR_SECONDARY: u32 = 2u;
const TR_LOCAL: u32 = 3u;
const TR_TRACK: u32 = 4u;
const TR_RAMP: u32 = 5u;
const TR_RAIL: u32 = 6u;
const TR_POWER: u32 = 7u;
const TR_N: u32 = 8u;

const TR_PORTAL: u32 = 216u;
const TR_STATION: u32 = 217u;
const TR_YARD: u32 = 218u;
const TR_ROUNDABOUT: u32 = 219u;

const TRF_BRIDGE: u32 = 1u;
const TRF_TUNNEL: u32 = 2u;
const TRF_URBAN: u32 = 4u;
const TRF_PAVED: u32 = 8u;
const TRF_ELECTRIC: u32 = 16u;
const TRF_OVER: u32 = 32u;
const TRF_FILL: u32 = 64u;
const TRF_LANES_SHIFT: u32 = 8u;

const TR_S_WRAP: f32 = 8400.0;

fn tr_has_deck(d: f32) -> bool {
    return d > -1.0e6;
}

struct TrEnv {
    p: vec3<f64>,
    ground: f32,
    gsd: f32,
    fw: f32,
    sun: vec3<f32>,
    sun_tan: f32,
    below: vec3<f32>,
}

/// `px::Px`.
struct TrPx {
    cov: f32,
    col: vec3<f32>,
    h: f32,
    cls: u32,
    emit: vec3<f32>,
    shade: f32,
    clear: f32,
    road: f32,
    z: f32,
}

fn tr_px_none() -> TrPx {
    var o: TrPx;
    o.cov = 0.0;
    o.col = vec3<f32>(0.0);
    o.h = 0.0;
    o.cls = 0u;
    o.emit = vec3<f32>(0.0);
    o.shade = 0.0;
    o.clear = 0.0;
    o.road = 0.0;
    o.z = -1e9;
    return o;
}

fn tr_over(o: ptr<function, TrPx>, a0: f32, c: vec3<f32>, h: f32) {
    let a = clamp(a0, 0.0, 1.0);
    if (a <= 0.0) {
        return;
    }
    let c0 = (*o).cov;
    let c1 = 1.0 - (1.0 - c0) * (1.0 - a);
    let k = a / c1;
    (*o).col = (*o).col + (c - (*o).col) * k;
    (*o).h = (*o).h + (h - (*o).h) * k;
    (*o).cov = c1;
}

fn tr_boxcov(d: f32, hw: f32, fw: f32) -> f32 {
    let lo = max(d - 0.5 * fw, -hw);
    let hi = min(d + 0.5 * fw, hw);
    return clamp((hi - lo) / fw, 0.0, 1.0);
}

fn tr_aa(sd: f32, fw: f32) -> f32 {
    return clamp(sd / fw + 0.5, 0.0, 1.0);
}

fn tr_dash_f(x: f32, period: f32, on: f32) -> f32 {
    let n = floor(x / period);
    return n * on + min(x - n * period, on);
}

fn tr_dash(s: f32, period: f32, on: f32, fw: f32) -> f32 {
    return clamp((tr_dash_f(s + 0.5 * fw, period, on) - tr_dash_f(s - 0.5 * fw, period, on)) / fw, 0.0, 1.0);
}

fn tr_norm2(v: vec2<f32>) -> vec2<f32> {
    let l = length(v);
    if (l < 1e-12) {
        return vec2<f32>(1.0, 0.0);
    }
    return v / l;
}

fn tr_car_col(h: u64) -> vec3<f32> {
    let k = u32(u01k(h, 2lu) * 5.99);
    if (k == 0u) {
        return srgb(230.0, 230.0, 228.0);
    } else if (k == 1u) {
        return srgb(30.0, 30.0, 34.0);
    } else if (k == 2u) {
        return srgb(150.0, 152.0, 156.0);
    } else if (k == 3u) {
        return srgb(140.0, 30.0, 30.0);
    } else if (k == 4u) {
        return srgb(40.0, 60.0, 120.0);
    }
    return srgb(90.0, 92.0, 96.0);
}

/// `px::parked`: (coverage, paint) of the cars in a parking lot (the settlements kit's).
fn tr_parked(h: u64, u: f32, v: f32, fw: f32, gsd: f32) -> vec4<f32> {
    let row = v / 16.0 - floor(v / 16.0);
    let stall = floor(u / 2.6);
    let lane = floor(v / 16.0);
    let ch = hash2(h ^ 0xCA2lu, i64(stall), i64(lane));
    if (abs(row - 0.5) <= 0.2 || u01k(ch, 1lu) >= 0.65) {
        return vec4<f32>(0.0);
    }
    let cu = u - (stall + 0.5) * 2.6;
    let cv = (row - select(0.15, 0.85, row > 0.5)) * 16.0;
    let car = tr_boxcov(cu, 0.9, fw) * tr_boxcov(cv, 2.2, fw);
    return vec4<f32>(car * band(2.0, gsd), tr_car_col(ch));
}

/// `px::Frame`.
struct TrFrame {
    d: f32,
    along: f32,
    t: f32,
    len: f32,
    su: f32,
    sl: f32,
    ar: f32,
    dl: f32,
}

fn tr_frame(sg: FSeg, p: vec3<f64>, sun: vec3<f32>) -> TrFrame {
    let ab = vec3<f32>(sg.b.xyz - sg.a.xyz);
    let ap = vec3<f32>(p - sg.a.xyz);
    let l2 = max(dot(ab, ab), 1e-9);
    let len = sqrt(l2);
    let tr = dot(ap, ab) / l2;
    let t = clamp(tr, 0.0, 1.0);
    let foot = sg.a.xyz + vec3<f64>(ab * t);
    let up = normalize(vec3<f32>(foot * (1.0lf / 6.4e6lf)));
    let u = ab / len;
    let left = normalize_or_zero(cross(up, ab));
    let fp = vec3<f32>(p - foot);
    let side = select(1.0, -1.0, dot(fp, left) < 0.0);
    var f: TrFrame;
    f.d = side * length(fp);
    f.along = sg.s0 + t * len;
    f.t = t;
    f.len = len;
    f.su = dot(sun, u);
    f.sl = dot(sun, left);
    f.ar = sg.s0 + tr * len;
    f.dl = dot(ap, left);
    return f;
}

fn tr_lamps(f: TrFrame, s0: f32, sp: f32, off: f32, median: bool, seed: u64, fw: f32) -> vec3<f32> {
    let m = exp2(ceil(log2(max(2.0 * fw / sp, 1.0))));
    let sp_eff = sp * m;
    let k = floor(f.ar / sp_eff);
    let sk = (k + 0.5) * sp_eff;
    if (sk < s0 || sk >= s0 + f.len) {
        return vec3<f32>(0.0);
    }
    let ki = i64(k);
    let even = ((ki % 2li) + 2li) % 2li == 0li;
    let side = select(-1.0, 1.0, median || even);
    let dp = f.dl - select(side * off, 0.0, median);
    let da = f.ar - sk;
    let d2 = dp * dp + da * da;
    let pool = 0.035 * exp(-(dp * dp) / (2.0 * 5.0 * 5.0) - (da * da) / (2.0 * 0.3 * sp * 0.3 * sp));
    let core = point_light(d2, 6.0, 0.4, fw);
    let c = select(vec3<f32>(1.0, 0.48, 0.12), vec3<f32>(0.86, 0.92, 1.0), u01k(seed, 7lu) < 0.5);
    return c * ((pool + core) * m);
}

/// `px::surface`: the carriageway or deck across its width.
fn tr_surface(sg: FSeg, f: TrFrame, kc: u32, fw: f32, o: ptr<function, TrPx>, hs: f32) {
    let hw = sg.v.x;
    let fl = sg.flags;
    let dd = abs(f.d);
    let seed = u64(u32(sg.v.w));
    let a = tr_boxcov(f.d, hw, fw);
    if (a <= 0.0) {
        return;
    }
    let vr = 0.94 + 0.12 * u01k(seed, 1lu);
    if (kc == TR_RAIL) {
        let tracks = f32(max((fl >> TRF_LANES_SHIFT) & 15u, 1u));
        var c = srgb(128.0, 118.0, 106.0) * vr;
        var tc = 0.0;
        if (tracks > 1.5) {
            tc = select(-2.25, 2.25, f.d > 0.0);
        }
        let dt = f.d - tc;
        let concrete = u01k(seed, 3lu) < 0.6;
        let sl = tr_dash(f.ar, 0.6, 0.26, fw) * tr_boxcov(dt, 1.3, fw);
        c = mixc(c, select(srgb(92.0, 78.0, 64.0), srgb(150.0, 148.0, 142.0), concrete), sl);
        let rail = tr_boxcov(abs(dt) - 0.7175, 0.036, fw);
        c = mixc(c, srgb(178.0, 178.0, 184.0), rail);
        tr_over(o, a, c, hs);
        (*o).cls = LC_RAILWAY;
        return;
    }
    let paved = (fl & TRF_PAVED) != 0u;
    var c: vec3<f32>;
    if (kc == TR_TRACK) {
        c = srgb(146.0, 126.0, 100.0);
    } else if (!paved) {
        c = srgb(160.0, 146.0, 122.0);
    } else if (kc <= TR_TRUNK || kc == TR_RAMP) {
        c = srgb(66.0, 66.0, 68.0);
    } else {
        c = srgb(84.0, 82.0, 80.0);
    }
    c = c * vr;
    if (kc == TR_TRACK) {
        let rut = tr_boxcov(dd - 0.8, 0.22, fw);
        c = mixc(c, srgb(118.0, 100.0, 80.0), rut);
        c = mixc(c, srgb(104.0, 118.0, 70.0), tr_boxcov(f.d, 0.35, fw) * 0.7);
    } else if (paved) {
        let white = srgb(232.0, 232.0, 226.0);
        var mk = tr_boxcov(dd - (hw - 0.35), 0.075, fw);
        if (kc == TR_MOTORWAY) {
            let med = sg.v.y;
            let lanes = max((fl >> TRF_LANES_SHIFT) & 15u, 1u);
            mk += tr_boxcov(dd - (med + 0.45), 0.075, fw);
            for (var k = 1u; k < lanes; k++) {
                mk += tr_boxcov(dd - (med + 0.6 + 3.6 * f32(k)), 0.075, fw) * tr_dash(f.ar, 16.0, 6.0, fw);
            }
            let bar = tr_boxcov(f.d, 0.3, fw);
            c = mixc(c, srgb(186.0, 184.0, 178.0), bar);
        } else if (kc != TR_RAMP && kc != TR_LOCAL) {
            mk += tr_boxcov(f.d, 0.075, fw) * tr_dash(f.ar, 12.0, 4.0, fw);
        }
        c = mixc(c, white, min(mk, 1.0));
    }
    tr_over(o, a, c, hs);
    (*o).cls = sg.cls;
}

/// `px::road_px`: a road or railway segment.
fn tr_road_px(sg: FSeg, f: TrFrame, e: TrEnv, kc: u32) -> TrPx {
    var o = tr_px_none();
    let fl = sg.flags;
    if ((fl & TRF_TUNNEL) != 0u) {
        return o;
    }
    let hw = sg.v.x;
    let vg = sg.v.z;
    let dd = abs(f.d);
    let fw = e.fw;
    let seed = u64(u32(sg.v.w));
    let has_deck = tr_has_deck(sg.deck_a) && tr_has_deck(sg.deck_b);
    var deck = e.ground;
    if (has_deck) {
        deck = lerp(sg.deck_a, sg.deck_b, f.t);
    }
    let bridge = (fl & TRF_BRIDGE) != 0u && has_deck;
    var hs = e.ground;
    if (has_deck) {
        if ((fl & TRF_FILL) != 0u || bridge) {
            hs = max(deck, e.ground);
        } else {
            hs = deck;
        }
    }
    let raised = hs - e.ground;
    let urban = (fl & TRF_URBAN) != 0u;
    let edge = hw + select(vg, 0.6, bridge);
    if (bridge && raised > 1.0) {
        let l = raised / max(e.sun_tan, 0.1);
        let ds = f.dl + f.sl * l;
        let as_ = f.ar + f.su * l;
        let s0 = sg.s0;
        if (as_ >= s0 && as_ <= s0 + f.len) {
            o.shade = 0.75 * tr_boxcov(ds, hw + 0.6, fw) * (1.0 - tr_boxcov(f.dl, hw + 0.6, fw));
        }
    }
    if (has_deck && !bridge && dd > edge - fw) {
        let sl = max(dd - edge, 0.0) * 0.5;
        if (raised > 0.0) {
            let toe = edge + 2.0 * min(raised, 15.0);
            tr_over(&o, tr_aa(toe - dd, fw), mixc(e.below, srgb(112.0, 124.0, 74.0), 0.35), hs - sl);
        } else {
            let toe = edge + 2.0 * min(-raised, 15.0);
            let strata = 0.9 + 0.1 * sin((hs + sl) / 3.0);
            tr_over(&o, tr_aa(toe - dd, fw), mixc(e.below, srgb(140.0, 128.0, 100.0) * strata, 0.4), hs + sl);
        }
        o.z = hs - sl;
        o.clear = o.cov;
    }
    let av = tr_boxcov(f.d, edge, fw);
    if (av > 0.0) {
        var c: vec3<f32>;
        if (bridge) {
            c = srgb(184.0, 182.0, 176.0);
        } else if (urban) {
            c = mixc(srgb(172.0, 170.0, 164.0), srgb(196.0, 194.0, 188.0), tr_boxcov(dd - hw - 0.15, 0.15, fw));
        } else if (kc == TR_TRACK || kc == TR_RAIL) {
            c = mixc(e.below, srgb(150.0, 140.0, 120.0), 0.5);
        } else {
            c = mixc(mixc(e.below, srgb(118.0, 128.0, 82.0), 0.45), srgb(150.0, 146.0, 136.0), tr_boxcov(dd - hw, 0.8, fw));
        }
        tr_over(&o, av, c, hs + select(0.0, 1.0, bridge));
        o.z = hs;
        o.clear = max(o.clear, av);
    }
    let a = tr_boxcov(f.d, hw, fw);
    if (a > 0.0) {
        tr_surface(sg, f, kc, fw, &o, hs);
        o.road = a;
        o.z = hs + 0.5 + 0.01 * f32(kc) + select(0.0, 20.0, (fl & TRF_OVER) != 0u);
        if (kc == TR_RAIL) {
            let tracks = f32(max((fl >> TRF_LANES_SHIFT) & 15u, 1u));
            if ((fl & TRF_ELECTRIC) != 0u) {
                let mk = round(f.ar / 60.0);
                let mast = tr_boxcov(f.ar - mk * 60.0, 0.25, fw) * tr_boxcov(dd - hw + 0.3, 0.25, fw);
                o.col = mixc(o.col, srgb(110.0, 112.0, 116.0), mast);
            }
            if (u01k(seed, 11lu) < 0.35) {
                let tl = 120.0 + 260.0 * u01k(seed, 14lu);
                let st = u01k(seed, 12lu) * (TR_S_WRAP - tl);
                var tc = 0.0;
                if (tracks > 1.5) {
                    tc = select(-2.25, 2.25, u01k(seed, 13lu) < 0.5);
                }
                let x = f.ar - st;
                if (x > 0.0 && x < tl) {
                    let body = tr_boxcov(f.d - tc, 1.45, fw) * tr_dash(x, 25.0, 24.2, fw);
                    let tk = u32(u01k(seed, 15lu) * 3.99);
                    var tcol = srgb(150.0, 152.0, 150.0);
                    if (tk == 0u) {
                        tcol = srgb(170.0, 40.0, 36.0);
                    } else if (tk == 1u) {
                        tcol = srgb(220.0, 222.0, 224.0);
                    } else if (tk == 2u) {
                        tcol = srgb(40.0, 70.0, 140.0);
                    }
                    o.col = mixc(o.col, tcol * (0.9 + 0.1 * tr_boxcov(f.d - tc, 0.5, fw)), body);
                    o.h = lerp(o.h, hs + 4.0, body);
                }
            }
        } else if (sg.ha > 0.0) {
            let dens = sg.ha;
            let med = sg.v.y;
            let lanes = f32(max((fl >> TRF_LANES_SHIFT) & 15u, 1u));
            var lw = hw;
            var l0 = 0.0;
            if (kc == TR_MOTORWAY) {
                lw = 3.6;
                l0 = med + 0.6;
            } else if (kc == TR_RAMP) {
                lw = 2.0 * hw;
                l0 = -hw;
            }
            let x = select(dd - l0, f.d - l0, kc == TR_RAMP);
            let k = floor(x / lw);
            if (k >= 0.0 && k < select(1.0, lanes, kc == TR_MOTORWAY)) {
                let lc = l0 + (k + 0.5) * lw;
                let ls = select(30.0, 24.0, kc == TR_MOTORWAY || kc == TR_RAMP);
                let lane_id = i64(k) + select(0li, 8li, f.d < 0.0 && kc != TR_RAMP);
                let j = floor(f.ar / ls);
                let hc = hash3(seed ^ 0xCA5lu, i64(j), lane_id, 0li);
                let e_car = band(3.0, e.gsd);
                if (u01k(hc, 1lu) < dens) {
                    let truck = kc <= TR_TRUNK && u01k(hc, 3lu) < 0.15;
                    let cl = select(2.25, 7.0, truck);
                    let cw = select(0.92, 1.25, truck);
                    let ac = (j + 0.5 + 0.4 * (u01k(hc, 4lu) - 0.5)) * ls;
                    let du = f.ar - ac;
                    let dv = select(dd, f.d, kc == TR_RAMP) - lc;
                    let car = tr_boxcov(du, cl, fw) * tr_boxcov(dv, cw, fw);
                    var ccol = tr_car_col(hc);
                    if (truck) {
                        ccol = mixc(srgb(200.0, 200.0, 196.0), ccol, 0.4);
                    }
                    o.col = mixc(o.col, ccol, car * e_car);
                    let sh = 1.5 / max(e.sun_tan, 0.1);
                    let sd = select(-1.0, 1.0, kc == TR_RAMP || f.d >= 0.0);
                    let car_s = tr_boxcov(du + f.su * sh, cl, fw) * tr_boxcov(dv + sd * f.sl * sh, cw, fw);
                    o.shade = max(o.shade, 0.6 * car_s * (1.0 - car) * e_car);
                }
                let mean = dens * 4.5 * 1.85 / (ls * lw);
                o.col = mixc(o.col, srgb(110.0, 110.0, 114.0), mean * (1.0 - e_car));
            }
        }
    }
    if (sg.hb > 0.0) {
        let median = kc == TR_MOTORWAY;
        o.emit = tr_lamps(f, sg.s0, sg.hb, hw + 0.5, median, seed, fw) * select(1.0, 1.2, urban);
    }
    return o;
}

/// `px::power_px`: the cleared corridor, pylons, wires and their shadows.
fn tr_power_px(sg: FSeg, f: TrFrame, e: TrEnv) -> TrPx {
    var o = tr_px_none();
    let hw = sg.v.x;
    let sp = sg.v.y;
    let hgt = sg.v.z;
    let fw = e.fw;
    o.clear = tr_boxcov(f.dl, hw, fw);
    let k = round(f.ar / sp);
    let sk = k * sp;
    let s0 = sg.s0;
    let arm = 0.3 * hw + 4.0;
    let tn = max(e.sun_tan, 0.1);
    let x = f.ar - floor(f.ar / sp) * sp;
    let xs = 2.0 * x / sp - 1.0;
    let sag = 1.0 - xs * xs;
    let wh = 0.82 * hgt - 0.22 * hgt * sag;
    let lw = wh / tn;
    let dw = f.dl + f.sl * lw;
    var wire = 0.0;
    var wsh = 0.0;
    for (var ci = 0u; ci < 3u; ci++) {
        let c = f32(ci) - 1.0;
        wire += tr_boxcov(f.dl - c * arm, 0.02, fw);
        wsh += tr_boxcov(dw - c * arm, 0.02, fw);
    }
    if (sk >= s0 && sk < s0 + f.len) {
        let a = f.ar - sk;
        let m = max(abs(a), abs(f.dl));
        let foot = tr_boxcov(a, 4.5, fw) * tr_boxcov(f.dl, 4.5, fw);
        let frame_l = min(tr_boxcov(m - 4.2, 0.3, fw) + 0.5 * tr_boxcov(abs(a) - abs(f.dl), 0.25, fw), 1.0);
        let arm_c = tr_boxcov(a, 0.7, fw) * tr_boxcov(f.dl, arm + 1.0, fw);
        let e_t = band(4.0, e.gsd);
        let cov = max(frame_l * foot * e_t + 0.45 * foot * (1.0 - e_t), arm_c * 0.8);
        var h = e.ground + 0.85 * hgt;
        if (m < 4.5) {
            h = e.ground + hgt * (1.0 - m / 4.5);
        }
        tr_over(&o, cov, srgb(150.0, 152.0, 156.0), max(h, e.ground + 0.85 * hgt * min(arm_c, 1.0)));
        o.z = e.ground;
    }
    var sh = 0.0;
    for (var kk = 0u; kk < 3u; kk++) {
        let skk = (k + f32(kk) - 1.0) * sp;
        if (skk < s0 || skk >= s0 + f.len) {
            continue;
        }
        let q = vec2<f32>(f.ar - skk, f.dl);
        let sv = tr_norm2(vec2<f32>(f.su, f.sl));
        let r = -dot(q, sv);
        let wdt = length(q - (-sv) * r);
        let ls = hgt / tn;
        if (r > 0.0 && r < ls) {
            let half = 4.5 * (1.0 - r / ls) + 0.3;
            sh = max(sh, 0.55 * tr_aa(half - wdt, fw));
            let ra = 0.85 * ls;
            let qa = q + sv * ra;
            sh = max(sh, 0.5 * tr_boxcov(qa.x, 0.7, fw) * tr_boxcov(qa.y, arm + 1.0, fw));
        }
    }
    o.shade = max(sh, 0.5 * min(wsh, 1.0));
    if (wire > 0.0) {
        tr_over(&o, 0.7 * min(wire, 1.0), srgb(40.0, 40.0, 44.0), e.ground + wh);
    }
    return o;
}

/// `px::seg_px`.
fn tr_seg_px(sg: FSeg, e: TrEnv) -> TrPx {
    let kc = sg.kind - TR_KIND0;
    let f = tr_frame(sg, e.p, e.sun);
    if ((abs(f.d) > sg.v.x + 40.0 + 2.0 * e.fw && kc != TR_POWER) || (kc == TR_POWER && abs(f.d) > sg.hw + 2.0 * e.fw)) {
        return tr_px_none();
    }
    if (kc == TR_POWER) {
        return tr_power_px(sg, f, e);
    }
    return tr_road_px(sg, f, e, kc);
}

/// `px::deck_px` (slot 8).
fn tr_deck_px(sg: FSeg, p: vec3<f64>, ground: f32, wl: f32, fw: f32) -> TrPx {
    var o = tr_px_none();
    let kc = sg.kind - TR_KIND0;
    if (kc == TR_POWER || (sg.flags & TRF_TUNNEL) != 0u) {
        return o;
    }
    let f = tr_frame(sg, p, vec3<f32>(0.0));
    let hw = sg.v.x;
    if (abs(f.d) > hw + 0.6 + fw) {
        return o;
    }
    let base = max(ground, wl) + 2.5;
    var hs = base;
    if (tr_has_deck(sg.deck_a) && tr_has_deck(sg.deck_b)) {
        hs = max(lerp(sg.deck_a, sg.deck_b, f.t), base);
    }
    tr_over(&o, tr_boxcov(f.d, hw + 0.6, fw), srgb(184.0, 182.0, 176.0), hs + 1.0);
    tr_surface(sg, f, kc, fw, &o, hs);
    o.cls = LC_BRIDGE;
    o.road = tr_boxcov(f.d, hw, fw);
    o.z = hs;
    return o;
}

/// `px::stamp_px`: tunnel portals, stations, yards, roundabouts.
fn tr_stamp_px(st: FStamp, e: TrEnv) -> TrPx {
    var o = tr_px_none();
    let q = stamp_uv(st, e.p);
    let fw = e.fw;
    if (abs(q.x) > st.half.x + fw || abs(q.y) > st.half.y + fw) {
        return o;
    }
    let seed = u64(u32(st.v0.z));
    if (st.tmpl == TR_PORTAL) {
        let hw = st.v0.x;
        var hd = e.ground;
        if (tr_has_deck(st.h)) {
            hd = st.h;
        }
        let wall = tr_boxcov(q.x - 1.25, 1.25, fw) * tr_boxcov(q.y, hw + 4.0, fw);
        tr_over(&o, wall, srgb(176.0, 173.0, 166.0), max(hd, e.ground) + 7.5);
        let mouth = tr_boxcov(q.x + 1.5, 1.5, fw) * tr_boxcov(q.y, hw, fw);
        tr_over(&o, mouth, srgb(26.0, 26.0, 30.0), hd);
        o.shade = 0.8 * mouth;
        o.z = hd + 30.0;
        o.cls = LC_BUILDING;
    } else if (st.tmpl == TR_STATION) {
        let tracks = st.v0.x;
        let len = st.v0.y;
        let side = st.v0.w;
        let w0 = 0.5 * (tracks - 1.0) * 5.5;
        let inl = tr_boxcov(q.x, 0.5 * len, fw);
        let ballast = inl * tr_boxcov(q.y, w0 + 2.6, fw);
        tr_over(&o, ballast, srgb(124.0, 114.0, 102.0), e.ground);
        let kf = clamp(round((q.y + w0) / 5.5), 0.0, tracks - 1.0);
        let tcy = kf * 5.5 - w0;
        let dt = q.y - tcy;
        let rail = tr_boxcov(abs(dt) - 0.7175, 0.036, fw) * inl;
        let sleep = tr_dash(q.x, 0.6, 0.26, fw) * tr_boxcov(dt, 1.3, fw) * inl;
        o.col = mixc(mixc(o.col, srgb(150.0, 148.0, 142.0), sleep), srgb(178.0, 178.0, 184.0), rail);
        let kp = round((q.y + w0 - 2.75) / 11.0);
        let pcy = kp * 11.0 - w0 + 2.75;
        let inside = select(0.0, 1.0, abs(pcy) < w0);
        let platform = inl * tr_boxcov(q.y - pcy, 1.6, fw) * inside;
        tr_over(&o, platform, srgb(176.0, 172.0, 164.0), e.ground + 0.9);
        let edge_l = platform * tr_boxcov(abs(q.y - pcy) - 1.45, 0.08, fw);
        o.col = mixc(o.col, srgb(222.0, 196.0, 60.0), edge_l);
        let canopy = tr_boxcov(q.x, 0.32 * len, fw) * tr_boxcov(q.y - pcy, 2.4, fw) * inside;
        let roof = select(srgb(120.0, 124.0, 130.0), srgb(150.0, 70.0, 56.0), u01k(seed, 1lu) < 0.5);
        tr_over(&o, canopy * max(band(3.0, e.gsd), 0.6), roof, e.ground + 5.5);
        let yb = side * (w0 + 18.0);
        let bl = 0.12 * len + 14.0;
        let bld = tr_boxcov(q.x, bl, fw) * tr_boxcov(q.y - yb, 10.0, fw);
        tr_over(&o, bld, roof * 0.9, e.ground + 11.0);
        let court = tr_boxcov(q.x, bl + 12.0, fw) * tr_boxcov(q.y - side * (w0 + 33.0), 6.0, fw);
        tr_over(&o, court, srgb(168.0, 162.0, 152.0), e.ground);
        o.emit = vec3<f32>(0.9, 0.92, 1.0) * (0.05 * (platform + court) * band(20.0, e.gsd) + 0.02 * ballast * (1.0 - band(20.0, e.gsd)));
        o.cls = select(LC_RAILWAY, LC_BUILDING, bld > 0.5);
        o.road = max(max(ballast, bld), court);
        o.clear = o.cov;
        o.z = e.ground + 40.0;
    } else if (st.tmpl == TR_YARD) {
        let tracks = st.v0.x;
        let len = st.v0.y;
        let w0 = 0.5 * (tracks - 1.0) * 5.0;
        let kf = clamp(round((q.y + w0) / 5.0), 0.0, tracks - 1.0);
        let ki = kf - 0.5 * (tracks - 1.0);
        let tl = 0.5 * len - abs(ki) * 32.0;
        let area = tr_boxcov(q.y, w0 + 3.0, fw) * tr_aa(0.5 * len - abs(q.x) - max(abs(q.y) - 3.0, 0.0) * 6.4, fw);
        tr_over(&o, area, srgb(118.0, 108.0, 96.0), e.ground);
        let dt = q.y - (kf * 5.0 - w0);
        let on = tr_aa(tl - abs(q.x), fw);
        let rail = tr_boxcov(abs(dt) - 0.7175, 0.036, fw) * on;
        let sleep = tr_dash(q.x, 0.6, 0.26, fw) * tr_boxcov(dt, 1.3, fw) * on;
        o.col = mixc(mixc(o.col, srgb(92.0, 78.0, 64.0), sleep), srgb(170.0, 170.0, 176.0), rail);
        let hk = hash2(seed, i64(kf), 0li);
        if (u01k(hk, 1lu) < 0.7) {
            let w_lo = -tl + 40.0 + u01k(hk, 2lu) * 0.3 * len;
            let w_hi = tl - 40.0 - u01k(hk, 3lu) * 0.3 * len;
            let x = q.x - w_lo;
            if (q.x > w_lo && q.x < w_hi) {
                let wi = floor(x / 16.0);
                let hw2 = hash2(hk, i64(wi), 1li);
                let wk = u32(u01k(hw2, 1lu) * 5.99);
                var wc = srgb(60.0, 110.0, 70.0);
                if (wk == 0u) {
                    wc = srgb(120.0, 62.0, 44.0);
                } else if (wk == 1u) {
                    wc = srgb(70.0, 72.0, 78.0);
                } else if (wk == 2u) {
                    wc = srgb(40.0, 70.0, 130.0);
                } else if (wk == 3u) {
                    wc = srgb(40.0, 40.0, 42.0);
                } else if (wk == 4u) {
                    wc = srgb(200.0, 200.0, 196.0);
                }
                let body = tr_boxcov(dt, 1.45, fw) * tr_dash(x, 16.0, 15.0, fw);
                tr_over(&o, body * area, wc, e.ground + 4.0);
            }
        }
        let mk = round(q.x / 150.0);
        let dx = q.x - mk * 150.0;
        let d2 = dx * dx + q.y * q.y;
        o.emit = vec3<f32>(1.0, 0.6, 0.25) * point_light(d2, 4.0, 0.6, fw) + vec3<f32>(1.0, 0.6, 0.25) * 0.03 * area;
        o.cls = LC_RAILWAY;
        o.road = area;
        o.clear = area;
        o.z = e.ground + 40.0;
    } else if (st.tmpl == TR_ROUNDABOUT) {
        let r = st.v0.x;
        let hw = st.v0.y;
        let rr = length(q);
        let ring = tr_boxcov(rr - (r - hw), hw, fw);
        let island = tr_aa(r - 2.0 * hw - rr, fw);
        var c = srgb(70.0, 70.0, 72.0);
        c = mixc(c, srgb(232.0, 232.0, 226.0), tr_boxcov(rr - (r - 0.35), 0.075, fw) + tr_boxcov(rr - (r - 2.0 * hw + 0.35), 0.075, fw));
        tr_over(&o, ring, c, e.ground);
        let curb = tr_boxcov(rr - (r - 2.0 * hw - 0.5), 0.5, fw);
        let green = select(srgb(170.0, 160.0, 140.0), srgb(96.0, 128.0, 66.0), u01k(seed, 1lu) < 0.7);
        tr_over(&o, island, green, e.ground + 0.3);
        tr_over(&o, curb, srgb(190.0, 188.0, 182.0), e.ground + 0.2);
        let tr = (r - 2.0 * hw) * 0.35;
        let tree = tr_aa(tr - rr, fw) * select(0.0, 1.0, u01k(seed, 2lu) < 0.6);
        tr_over(&o, tree * max(band(3.0, e.gsd), 0.5), srgb(60.0, 86.0, 44.0), e.ground + 7.0);
        o.cls = select(LC_GRASS, LC_ROAD_MAJOR, ring > 0.5);
        o.road = ring;
        o.clear = o.cov;
        o.z = e.ground + 30.0;
    }
    return o;
}

fn tr_composite(s: ptr<function, Stack>, o: TrPx) {
    var ly = layer_paint(o.cov, o.col, o.cls);
    ly.dh = o.h;
    ly.hmode = HM_ABS;
    ly.relit = 0.5;
    composite(s, ly);
}

/// Slot 7 (`transport::built`): after the settlements' places.
fn transport_slot_built(s: ptr<function, Stack>) {
    if (cfg.roads <= 0.0) {
        return;
    }
    let blk = (*s).l.blk;
    let nsg = feat_seg_n(blk);
    let nst = feat_stamp_n(blk);
    if (nsg == 0u && nst == 0u) {
        return;
    }
    var e: TrEnv;
    e.p = (*s).c.p;
    e.ground = (*s).l.ground;
    e.gsd = (*s).c.gsd;
    e.fw = (*s).l.fw;
    e.sun = (*s).c.east * cfg.sun_hx + (*s).c.north * cfg.sun_hy;
    e.sun_tan = cfg.sun_tan;
    e.below = (*s).col;
    var best = tr_px_none();
    var second = tr_px_none();
    var emit = vec3<f32>(0.0);
    var shade = 0.0;
    var clear = 0.0;
    var road = 0.0;
    var any = false;
    for (var k = 0u; k < nsg; k++) {
        let sg = feat_seg(blk, k);
        if (sg.kind < TR_KIND0 || sg.kind >= TR_KIND0 + TR_N) {
            continue;
        }
        any = true;
        let kc = sg.kind - TR_KIND0;
        if ((*s).no_town && !(kc == TR_MOTORWAY || kc == TR_TRUNK || kc == TR_RAMP || kc == TR_RAIL)) {
            continue;
        }
        let o = tr_seg_px(sg, e);
        emit += o.emit;
        shade = max(shade, o.shade);
        clear = max(clear, o.clear);
        road = max(road, o.road);
        if (o.cov > 0.0) {
            if (o.z > best.z) {
                second = best;
                best = o;
            } else if (o.z > second.z) {
                second = o;
            }
        }
    }
    if (second.cov > 0.0) {
        tr_composite(s, second);
    }
    if (best.cov > 0.0) {
        tr_composite(s, best);
    }
    for (var k = 0u; k < nst; k++) {
        let st = feat_stamp(blk, k);
        if (st.tmpl < TR_PORTAL || st.tmpl > TR_ROUNDABOUT) {
            continue;
        }
        any = true;
        let o = tr_stamp_px(st, e);
        emit += o.emit;
        shade = max(shade, o.shade);
        clear = max(clear, o.clear);
        road = max(road, o.road);
        if (o.cov > 0.0) {
            tr_composite(s, o);
        }
    }
    if (!any) {
        return;
    }
    var ls = layer_paint(select(0.0, 1e-4, shade > 0.0), (*s).col, 0u);
    ls.lit = 1.0 - shade;
    ls.emit = emit;
    composite(s, ls);
    (*s).m.veg *= 1.0 - clear;
    (*s).m.natural_ok *= 1.0 - road;
    (*s).m.road_cov = max((*s).m.road_cov, road);
}

/// Slot 8 (`transport::water`): decks over rivers, reservoirs, lakes and the sea.
fn transport_slot_water(s: ptr<function, Stack>) {
    if (cfg.roads <= 0.0) {
        return;
    }
    let l = (*s).l;
    let standing = l.water > l.ground && l.water_kind != W_NONE;
    let river = l.river_hw > 0.0 && abs(l.river_d) < l.river_hw + 0.5 * max(l.fw, 0.35 * (*s).c.gsd);
    if (!standing && !river && !(*s).is_water) {
        return;
    }
    let blk = l.blk;
    let nsg = feat_seg_n(blk);
    if (nsg == 0u) {
        return;
    }
    let p = (*s).c.p;
    let fw = l.fw;
    let wl = (*s).height;
    var best = tr_px_none();
    for (var k = 0u; k < nsg; k++) {
        let sg = feat_seg(blk, k);
        if (sg.kind < TR_KIND0 || sg.kind >= TR_KIND0 + TR_N) {
            continue;
        }
        let o = tr_deck_px(sg, p, l.ground, wl, fw);
        if (o.cov > 0.0 && o.z > best.z) {
            best = o;
        }
    }
    if (best.cov > 0.0) {
        var ly = layer_paint(best.cov, best.col, best.cls);
        ly.dh = best.h;
        ly.hmode = HM_ABS;
        ly.relit = 1.0;
        ly.solid = true;
        composite(s, ly);
        (*s).m.road_cov = max((*s).m.road_cov, best.road);
    }
}
