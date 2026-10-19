// The cold kit (`kits/cold.rs`): the same functions in the same order of operations.

const COLD_LC_LAKE: u32 = 2u;
const COLD_LC_TUNDRA: u32 = 15u;
const COLD_SEED_SNOW: u64 = 0xC01D5E00lu;
const COLD_SEED_ICE: u64 = 0xC01D1CE0lu;
const COLD_SEED_FLOE: u64 = 0xC01DF10Elu;
const COLD_LAKE_EL: f32 = 1.85;
const COLD_LAKE_WN: f32 = 1.26;
const COLD_FREEZE: f32 = -3.0;

// ---------------------------------------------------------------- climate

struct ColdClimate {
    t: f32,
    summer: f32,
    date: f32,
}

fn cold_year_date(season: f32) -> f32 {
    return 0.40 + 0.34 * season;
}

fn cold_climate(s: ptr<function, Stack>) -> ColdClimate {
    let t = (*s).t;
    let ts = t.temp - cfg.lapse * (max((*s).l.ground, 0.0) - max(t.ground, 0.0)) * 0.001;
    let half = 0.5 * t.temp_range;
    let d = cold_year_date((*s).bio.st.land.x);
    var c: ColdClimate;
    c.t = ts;
    c.summer = ts + 0.95 * half;
    let mid = cos(TAU * (cold_year_date(0.5) - 0.04));
    c.date = ts - half * lerp(mid, cos(TAU * (d - 0.04)), smoothstep1(0.0, 10.0, (*s).bio.edge_km));
    return c;
}

fn cold_glaciation(s: ptr<function, Stack>) -> f32 {
    return (*s).t.glaciation;
}

fn cold_sunny(s: ptr<function, Stack>) -> f32 {
    return (*s).l.grad.y * sign((*s).c.lat);
}

// ---------------------------------------------------------------- snow and ice

struct ColdIce {
    p: f32,
    g: f32,
    ice: f32,
    snow: f32,
    n: f32,
}

fn cold_ice(s: ptr<function, Stack>, c: ColdClimate) -> ColdIce {
    let t = (*s).t;
    let l = (*s).l;
    let tela = -2.0 + 5.0 * smoothstep1(0.15, 0.85, t.moist);
    let base = c.summer - tela;
    var o: ColdIce;
    o.p = base;
    o.g = base;
    o.ice = 0.0;
    o.snow = 0.0;
    o.n = 0.0;
    if (base > 8.5) {
        return o;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let hollow = smoothstep1(0.1, 0.8, -t.gully);
    let spur = smoothstep1(0.2, 0.8, t.gully);
    var n = (*s).pf.f[PF_SNOW];
    let b60 = band(60.0, gsd);
    if (b60 > 0.0) {
        n += 0.5 * perlin3(COLD_SEED_SNOW, p * (1.0lf / 60.0lf)) * b60;
        let b18 = band(18.0, gsd);
        if (b18 > 0.0) {
            n += 0.25 * perlin3(COLD_SEED_SNOW + 1lu, p * (1.0lf / 18.0lf)) * b18;
        }
    }
    let pi = base - 2.0 * hollow + 0.8 * spur + 1.2 * n + 1.5 * clamp(cold_sunny(s), -1.0, 1.0);
    var rv = 0.0;
    if (l.river_hw > 0.0) {
        rv = 1.0 - smoothstep1(l.river_hw + 80.0, l.river_hw + 500.0, abs(l.river_d));
    }
    let valley = max(rv, 0.6 * hollow) * smoothstep1(0.15, 0.4, t.mountain);
    let g = pi - 2.6 * valley;
    let steep = 1.0 - smoothstep1(0.45, 0.75, l.slope + 0.1 * n + 0.15 * spur);
    let ice = smoothstep1(0.15, -0.15, g + 0.4) * steep;
    let snow = smoothstep1(0.15, -0.15, pi) * (1.0 - 0.6 * smoothstep1(1.0, 1.8, l.slope));
    o.p = pi;
    o.g = g;
    o.ice = ice;
    o.snow = max(snow, ice);
    o.n = n;
    return o;
}

fn cold_slot_zonal(s: ptr<function, Stack>) {
    let c = cold_climate(s);
    let g = cold_ice(s, c);
    (*s).m.snow = g.snow;
}

// ---------------------------------------------------------------- lakes

struct ColdLakeLat {
    seed: u64,
    cell: f64,
    r0: f32,
    r1: f32,
    dens: f32,
    p_lake: f32,
    p_basin: f32,
}

fn cold_lake_lat(i: u32) -> ColdLakeLat {
    var l: ColdLakeLat;
    if (i == 0u) {
        l.seed = 0xC01D1A4Elu;
        l.cell = 900.0lf;
        l.r0 = 100.0;
        l.r1 = 400.0;
        l.dens = 0.85;
        l.p_lake = 0.72;
        l.p_basin = 0.22;
    } else {
        l.seed = 0xC01D1A4Flu;
        l.cell = 260.0lf;
        l.r0 = 25.0;
        l.r1 = 115.0;
        l.dens = 0.9;
        l.p_lake = 0.66;
        l.p_basin = 0.28;
    }
    return l;
}

struct ColdLake {
    cov: f32,
    rn: f32,
    kind: u32,
    h: u64,
    b: f32,
}

fn cold_lake_none() -> ColdLake {
    var l: ColdLake;
    l.cov = 0.0;
    l.rn = 0.0;
    l.kind = 0u;
    l.h = 0lu;
    l.b = 0.0;
    return l;
}

/// `cold::lake_site`: the lake of site `c` (hash `h`) at `p`.
fn cold_lake_site(lat: ColdLakeLat, c: vec3<f64>, h: u64, p: vec3<f64>, sc: f32, fw: f32) -> ColdLake {
    let rel = vec3<f32>(p - c);
    let r = lat.r0 + (lat.r1 - lat.r0) * u01k(h, 5lu);
    let el = 1.25 + 0.6 * u01k(h, 6lu);
    let lim = r * el * COLD_LAKE_WN * sc + fw;
    if (dot(rel, rel) > lim * lim) {
        return cold_lake_none();
    }
    let up = normalize(vec3<f32>(c));
    var e = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(up.z) <= 0.999) {
        e = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), up));
    }
    let nn = cross(up, e);
    let th = 0.45 + 0.35 * (u01k(h, 7lu) - 0.5);
    let sn = sin(th);
    let cs = cos(th);
    let ax = e * cs + nn * sn;
    let ay = nn * cs - e * sn;
    let a = r * el;
    let b = r / el;
    let x = dot(rel, ax) / a;
    let y = dot(rel, ay) / b;
    let z = dot(rel, up) / r;
    let rho = sqrt(x * x + y * y + z * z);
    if (rho > COLD_LAKE_WN * sc + fw / b) {
        return cold_lake_none();
    }
    let se = sc * (1.0 + 0.24 * perlin3(h, vec3<f64>(vec3<f32>(2.6 * x, 2.6 * y, 0.37))));
    let gr = sqrt((x / a) * (x / a) + (y / b) * (y / b)) / max(rho, 1e-6);
    let dist = (se - rho) / max(gr, 1e-9);
    let cov = clamp(dist / fw + 0.5, 0.0, 1.0);
    if (cov <= 0.0) {
        return cold_lake_none();
    }
    let u = u01k(h, 8lu);
    var kind = 2u;
    if (u < lat.p_lake) {
        kind = 0u;
    } else if (u < lat.p_lake + lat.p_basin) {
        kind = 1u;
    }
    var o: ColdLake;
    o.cov = cov;
    o.rn = min(rho / max(se, 1e-6), 1.0);
    o.kind = kind;
    o.h = h;
    o.b = b * se;
    return o;
}

/// `cold::stamped_lake`: the large lake of the host's stamps at the sample; its level in `.b`
/// of the second return value.
struct ColdStamped {
    hit: ColdLake,
    level: f32,
}

fn cold_stamped_lake(s: ptr<function, Stack>, li: u32, sc: f32, fw: f32) -> ColdStamped {
    let lat = cold_lake_lat(li);
    let p = (*s).c.p;
    let blk = (*s).l.blk;
    var o: ColdStamped;
    o.hit = cold_lake_none();
    o.level = 0.0;
    for (var k = 0u; k < feat_stamp_n(blk); k++) {
        let st = feat_stamp(blk, k);
        if (st.tmpl != 160u + li) {
            continue;
        }
        let h = u64(u32(st.v0.x)) | (u64(u32(st.v0.y)) << 24u) | (u64(u32(st.v0.z)) << 48u);
        let hit = cold_lake_site(lat, st.center.xyz, h, p, sc, fw);
        if (hit.cov > o.hit.cov) {
            o.hit = hit;
            o.level = st.h;
        }
    }
    return o;
}

fn cold_lake_mean_cov(lat: ColdLakeLat, sc: f32) -> f32 {
    let r0 = lat.r0;
    let r1 = lat.r1;
    let er3 = (r1 * r1 * r1 * r1 - r0 * r0 * r0 * r0) / (4.0 * (r1 - r0));
    let cell = f32(lat.cell);
    let lam = lat.dens * (4.0 / 3.0) * PI * er3 * sc * sc * sc / (cell * cell * cell);
    return 1.0 - exp(-lam);
}

fn cold_lake_mask(s: ptr<function, Stack>, c: ColdClimate) -> f32 {
    let t = (*s).t;
    let l = (*s).l;
    let thermo = smoothstep1(-1.0, -4.0, c.t);
    let kettle = cold_glaciation(s) * smoothstep1(9.0, 4.0, c.t);
    let m = max(thermo, 0.8 * kettle);
    if (m <= 0.0) {
        return 0.0;
    }
    return m * (1.0 - 0.7 * smoothstep1(0.15, 0.35, l.slope))
        * (1.0 - smoothstep1(350.0, 700.0, l.ground))
        * (1.0 - smoothstep1(0.2, 0.5, t.mountain))
        * smoothstep1(0.02, 0.12, t.moist)
        * (1.0 - smoothstep1(0.1, 0.3, t.sand))
        * (0.3 + 0.7 * smoothstep1(0.25, 0.65, t.style.x));
}

fn cold_lake_colour(h: u64, shore_m: f32) -> vec3<f32> {
    let u = u01k(h, 9lu);
    var deep = srgb(42.0, 58.0, 62.0);
    if (u < 0.55) {
        deep = srgb(28.0, 42.0, 56.0);
    } else if (u < 0.8) {
        deep = srgb(44.0, 48.0, 38.0);
    }
    let shallow = mixc(deep, srgb(98.0, 104.0, 86.0), 0.5);
    return mixc(deep, shallow, 0.7 * smoothstep1(4.0, 0.0, shore_m));
}

fn cold_lake_shallow_mix(c: vec3<f32>) -> vec3<f32> {
    return mixc(c, mixc(c, srgb(98.0, 104.0, 86.0), 0.5), 0.04);
}

fn cold_lake_mean_albedo(lat: ColdLakeLat, frozen: bool) -> vec3<f32> {
    var lake: vec3<f32>;
    if (frozen) {
        lake = mixc(srgb(150.0, 170.0, 184.0), srgb(228.0, 233.0, 239.0), 0.6);
    } else {
        lake = cold_lake_shallow_mix(srgb(28.0, 42.0, 56.0)) * 0.55 + cold_lake_shallow_mix(srgb(44.0, 48.0, 38.0)) * 0.25 + cold_lake_shallow_mix(srgb(42.0, 58.0, 62.0)) * 0.2;
    }
    let basin = mixc(srgb(72.0, 76.0, 52.0), srgb(60.0, 66.0, 46.0), 0.3);
    return mixc(basin, lake, lat.p_lake);
}

fn cold_drumlin_look(s: ptr<function, Stack>) {
    let l = inst_list_bin(FAM_COLD_DRUMLINS, (*s).l.blk);
    if (l.n == 0u) {
        return;
    }
    let f = cold_drumlin_field(l, (*s).c.p, (*s).c.gsd);
    let crest = f.y;
    let esk = f.z;
    if (crest > 0.05) {
        let c = (*s).col * vec3<f32>(1.12, 1.08, 0.94);
        composite(s, layer_paint(0.6 * smoothstep1(0.05, 0.6, crest), c, 0u));
    }
    if (esk > 0.0) {
        let g = mixc(spal(s, BP_SAND), srgb(150.0, 142.0, 124.0), 0.5);
        composite(s, layer_paint(smoothstep1(0.1, 0.4, esk), g, LC_GRAVEL));
    }
}

fn cold_slot_azonal(s: ptr<function, Stack>) {
    cold_drumlin_look(s);
    let c = cold_climate(s);
    let lm = cold_lake_mask(s, c) * (1.0 - (*s).m.snow);
    if (lm < 0.03) {
        return;
    }
    let sc = pow(lm, 1.0 / 3.0);
    let frozen = c.date < COLD_FREEZE;
    for (var i = 0u; i < 2u; i++) {
        cold_lakes(s, cold_lake_lat(i), i, sc, frozen);
        if ((*s).done) {
            return;
        }
    }
}

fn cold_lakes(s: ptr<function, Stack>, lat: ColdLakeLat, li: u32, sc: f32, frozen: bool) {
    let gsd = (*s).c.gsd;
    let p = (*s).c.p;
    let fw = (*s).l.fw;
    let e = smoothstep1(1.2 * gsd, 3.0 * gsd, 0.5 * (lat.r0 + lat.r1));
    let mc = cold_lake_mean_cov(lat, sc);
    var hit = cold_lake_none();
    var level = 0.0;
    if (e > 0.0) {
        let st = cold_stamped_lake(s, li, sc, fw);
        hit = st.hit;
        level = st.level;
    }
    let flat = (*s).l.ground - (*s).t.micro;
    if (hit.kind == 0u && hit.cov > 0.0) {
        hit.cov *= clamp((level - flat) / max((*s).l.slope, 0.01) / fw + 0.5, 0.0, 1.0);
    }
    let xc = e * hit.cov;
    let mw = (1.0 - e) * mc;
    let cov = xc + mw;
    if (cov <= 0.0) {
        return;
    }
    var ly = layer_none();
    ly.cov = cov;
    var xa = vec3<f32>(0.0);
    if (xc > 0.0) {
        if (hit.kind == 0u) {
            if (frozen) {
                let sw = smoothstep1(-0.15, 0.25, perlin3(hit.h ^ 0x1CElu, p * (1.0lf / 40.0lf)) * band(40.0, gsd) + 0.2 - 0.5 * hit.rn);
                xa = mixc(srgb(228.0, 233.0, 239.0), srgb(150.0, 170.0, 184.0), sw);
                ly.cls = LC_FROZEN_WATER;
                ly.hmode = HM_BLEND;
                ly.dh = -0.3;
            } else {
                xa = cold_lake_colour(hit.h, (1.0 - hit.rn) * hit.b) * (1.0 + 0.05 * perlin3(hit.h ^ 0x5EDlu, p * (1.0lf / 70.0lf)) * band(70.0, gsd));
                ly.cls = COLD_LC_LAKE;
                ly.hmode = HM_ABS;
                ly.dh = level;
                ly.relit = 1.0;
                ly.water = true;
            }
        } else {
            xa = mixc(srgb(72.0, 76.0, 52.0), srgb(60.0, 66.0, 46.0), smoothstep1(0.55, 0.0, hit.rn)) * (1.0 + 0.08 * perlin3(hit.h ^ 0xBA5lu, p * (1.0lf / 9.0lf)) * band(9.0, gsd) + 0.12 * perlin3(hit.h ^ 0xBA6lu, p * (1.0lf / 35.0lf)) * band(35.0, gsd));
            ly.cls = LC_MARSH;
            ly.hmode = HM_BLEND;
            ly.dh = -1.2;
            if (hit.kind == 2u) {
                let rp = 0.28 * hit.b;
                let pr = hit.rn / 0.28;
                let pc = clamp((1.0 - pr) * rp / fw + 0.5, 0.0, 1.0);
                if (pc > 0.0) {
                    let hh = clamp(0.35 * rp, 4.0, 35.0);
                    var dh = hh * pow(max(1.0 - pr * pr, 0.0), 0.7);
                    if (u01k(hit.h, 10lu) < 0.35 && pr < 0.3) {
                        let xr = pr / 0.3;
                        dh -= 0.4 * hh * (1.0 - xr * xr);
                    }
                    let pa = mixc(srgb(122.0, 118.0, 78.0), srgb(150.0, 134.0, 104.0), smoothstep1(0.45, 0.0, pr));
                    xa = mixc(xa, pa, pc);
                    ly.dh = lerp(ly.dh, dh, pc);
                    if (pc > 0.5) {
                        ly.cls = COLD_LC_TUNDRA;
                    }
                }
            }
        }
    }
    ly.albedo = (xa * xc + cold_lake_mean_albedo(lat, frozen) * mw) / cov;
    if (xc < mw || xc < 0.5) {
        ly.cls = 0u;
        ly.water = false;
        if (ly.hmode == HM_ABS) {
            ly.hmode = HM_NONE;
        }
    }
    ly.clear = 1.0;
    if (xc > 0.0 && hit.kind != 0u) {
        let n = perlin3(hit.h ^ 0xBA7lu, p * (1.0lf / 60.0lf)) * band(60.0, gsd);
        ly.clear = mix(1.0, 0.3 + 0.65 * smoothstep1(0.95, 0.4, hit.rn + 0.3 * n), xc / cov);
    }
    let water = ly.water;
    composite(s, ly);
    if (water) {
        (*s).done = true;
    }
}

// ---------------------------------------------------------------- water

fn cold_slot_water(s: ptr<function, Stack>) {
    let l = (*s).l;
    if (!(l.water > l.ground)) {
        cold_rivers(s);
        return;
    }
    let c = cold_climate(s);
    if (c.date > -1.0 && c.summer > -1.0) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = l.fw;
    let depth = l.water - l.ground;
    if (l.water_kind == W_OCEAN) {
        let con = max(smoothstep1(-1.5, -8.0, c.date), smoothstep1(0.0, -2.0, c.summer));
        if (con <= 0.02) {
            return;
        }
        let fast = smoothstep1(14.0, 6.0, depth) * smoothstep1(0.3, 0.6, con);
        let snowc = srgb(232.0, 237.0, 243.0);
        let e = smoothstep1(1.2 * gsd, 3.0 * gsd, 300.0);
        var mcov = con * (0.15 + 0.85 * con) * 0.93;
        mcov = mcov + (1.0 - mcov) * fast;
        var cov = (1.0 - e) * mcov;
        var col = snowc * 0.97 * cov;
        var dh = 0.4;
        if (e > 0.0) {
            let big = worley3(COLD_SEED_FLOE, p, 1600.0lf, 0.9);
            let eb = worley_edge_dist(big, p);
            let pair = u01k(big.id ^ big.id2, 3lu);
            let lead = pair < 0.35 + 0.6 * (1.0 - con);
            var lead_w = 0.0;
            if (lead) {
                lead_w = 12.0 + 160.0 * (1.0 - con) * (1.0 - con) + 30.0 * u01k(big.id ^ big.id2, 4lu);
            }
            var ic = 0.0;
            if (u01k(big.id, 1lu) < 0.15 + 0.85 * con) {
                ic = clamp((eb - lead_w) / fw + 0.5, 0.0, 1.0);
            }
            let mz = smoothstep1(0.9, 0.5, con);
            if (mz > 0.0 && u01k(big.id, 2lu) < mz) {
                let sm = worley3(COLD_SEED_FLOE + 1lu, p, 110.0lf, 0.9);
                let es = worley_edge_dist(sm, p);
                let ws = 2.0 + 12.0 * (1.0 - con);
                var k = 0.0;
                if (u01k(sm.id, 1lu) < con) {
                    k = clamp((es - ws) / fw + 0.5, 0.0, 1.0);
                }
                ic *= k;
            }
            ic = ic + (1.0 - ic) * fast;
            var xa = snowc * (1.0 + 0.04 * perlin3(COLD_SEED_FLOE + 2lu, p * (1.0lf / 30.0lf)) * band(30.0, gsd));
            var xdh = 0.4;
            if (!lead) {
                let rg = kband_cov(eb, 3.0, fw) * (1.0 - fast);
                xa = mixc(xa, srgb(246.0, 249.0, 252.0), rg);
                xdh += 2.0 * rg;
            }
            let nilas = smoothstep1(-6.0, -12.0, c.date) * (1.0 - ic);
            if (nilas > 0.0) {
                xa = mixc(xa, srgb(92.0, 102.0, 110.0), nilas / (ic + nilas));
                ic += nilas;
            }
            let xc = e * ic;
            cov += xc;
            col += xa * xc;
            dh = xdh;
        }
        if (cov <= 0.0) {
            return;
        }
        var ly = layer_none();
        ly.cov = cov;
        ly.albedo = col / cov;
        ly.dh = l.water + dh;
        ly.hmode = HM_ABS;
        ly.relit = 1.0;
        if (cov > 0.5) {
            ly.cls = LC_SEA_ICE;
            ly.solid = true;
        }
        composite(s, ly);
    } else if (c.date < COLD_FREEZE) {
        var col = srgb(226.0, 231.0, 237.0);
        let sw = smoothstep1(0.1, 0.5, perlin3(COLD_SEED_ICE, p * (1.0lf / 90.0lf)) * band(90.0, gsd) + 0.25 * perlin3(COLD_SEED_ICE + 1lu, p * (1.0lf / 25.0lf)) * band(25.0, gsd));
        col = mixc(col, srgb(146.0, 166.0, 182.0), 0.8 * sw);
        let e = smoothstep1(1.2 * gsd, 3.0 * gsd, 4.0);
        if (e > 0.0) {
            let ck = worley3(COLD_SEED_ICE + 2lu, p, 70.0lf, 0.9);
            let cr = kband_cov(worley_edge_dist(ck, p), 0.35, fw) * e * (0.4 + 0.6 * sw);
            col = mixc(col, srgb(120.0, 138.0, 150.0), cr);
        }
        var ly = layer_paint(1.0, col, LC_FROZEN_WATER);
        ly.dh = l.water + 0.2;
        ly.hmode = HM_ABS;
        ly.relit = 1.0;
        ly.solid = true;
        composite(s, ly);
    }
}

fn cold_rivers(s: ptr<function, Stack>) {
    let l = (*s).l;
    let t = (*s).t;
    if (l.river_hw <= 0.0 || t.river_wet <= 0.5) {
        return;
    }
    let gsd = (*s).c.gsd;
    let cov = band_cov(l.river_d, l.river_hw, max(l.fw, gsd * 0.35));
    if (cov <= 0.0) {
        return;
    }
    let c = cold_climate(s);
    let detail = (*s).pf.f[PF_DETAIL];
    let core_ice = smoothstep1(-1.5, -4.0, t.temp + 1.5 * (*s).pf.f[PF_SNOW]) > 0.5;
    if (c.date < COLD_FREEZE) {
        let col = mixc(pal3(PAL_SNOW) * 0.92, srgb(150.0, 172.0, 186.0), 0.3 + 0.25 * detail);
        (*s).m.river_cov = 0.4 * cov;
        var ly = layer_paint(cov, col, LC_FROZEN_WATER);
        ly.dh = l.river_level;
        ly.hmode = HM_ABS;
        ly.relit = 1.0;
        ly.solid = true;
        composite(s, ly);
    } else if (core_ice) {
        if ((*s).m.snow < 0.3) {
            var wcol = mixc(pal3(PAL_RIVER), pal3(PAL_LAKE_DEEP), smoothstep1(30.0, 200.0, l.river_hw * 2.0));
            wcol = mixc(wcol, srgb(96.0, 138.0, 140.0), 0.7 * t.mountain * smoothstep1(8.0, 0.0, t.temp));
            (*s).m.river_cov = cov;
            var ly = layer_paint(cov, wcol, LC_RIVER);
            ly.dh = l.river_level;
            ly.hmode = HM_ABS;
            ly.relit = 1.0;
            ly.water = true;
            composite(s, ly);
        }
    }
}

// ---------------------------------------------------------------- seasonal

fn cold_slot_seasonal(s: ptr<function, Stack>) {
    let c = cold_climate(s);
    let g = cold_ice(s, c);
    if (g.ice > 0.0) {
        cold_glacier(s, c, g);
    }
    if (g.p > 0.0 && g.g < 0.6) {
        cold_moraine(s, g);
    }
    cold_snow(s, c, g);
}

fn cold_glacier(s: ptr<function, Stack>, c: ColdClimate, g: ColdIce) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let l = (*s).l;
    let fw = l.fw;
    let slope = l.slope;
    let abl = smoothstep1(-1.6, -0.4, g.p);
    let n7 = perlin3(COLD_SEED_ICE + 3lu, p * (1.0lf / 7.0lf));
    let b7 = band(7.0, gsd);
    let tex = 0.025 * n7 * b7 + 0.02 * perlin3(COLD_SEED_ICE + 4lu, p * (1.0lf / 2.5lf)) * band(2.5, gsd) + 0.03 * perlin3(COLD_SEED_ICE + 5lu, p * (1.0lf / 120.0lf)) * band(120.0, gsd);
    let firn = srgb(236.0, 241.0, 247.0) * (1.0 + tex);
    let dirt = smoothstep1(-0.2, 0.5, perlin3(COLD_SEED_ICE + 6lu, p * (1.0lf / 150.0lf)) * band(150.0, gsd) + 0.6 * smoothstep1(-1.4, -0.4, g.g));
    let bare = mixc(srgb(172.0, 184.0, 192.0), srgb(128.0, 128.0, 124.0), 0.6 * dirt) * (1.0 + 1.5 * tex);
    var col = mixc(firn, bare, abl);
    var dh = 0.0;
    if (g.p < -4.0 && slope < 0.08) {
        let bi = smoothstep1(0.32, 0.36, perlin3(COLD_SEED_ICE + 7lu, p * (1.0lf / 900.0lf)) + 0.3 * perlin3(COLD_SEED_ICE + 8lu, p * (1.0lf / 300.0lf)) * band(300.0, gsd)) * smoothstep1(-4.0, -6.0, g.p) * (1.0 - smoothstep1(0.04, 0.08, slope));
        col = mixc(col, srgb(160.0, 180.0, 196.0), bi);
    }
    let sl = max(slope, 0.03);
    let cs = smoothstep1(0.18, 0.4, slope) * (0.45 + 0.55 * abl);
    if (cs > 0.0) {
        let spacing = 6.0 / sl;
        let e = smoothstep1(1.2 * gsd, 3.0 * gsd, spacing);
        let on = smoothstep1(-0.05, 0.05, perlin3(COLD_SEED_ICE + 9lu, p * (1.0lf / 20.0lf)) + 0.55 * cs - 0.35);
        var cr = 0.0;
        if (e > 0.0) {
            let x = l.ground / 6.0 + 0.6 * perlin3(COLD_SEED_ICE + 10lu, p * (1.0lf / 45.0lf));
            let f = x - floor(x);
            let d = min(f, 1.0 - f) * spacing;
            let dash = 0.4 + (smoothstep1(-0.05, 0.25, n7) - 0.4) * b7;
            cr = kband_cov(d, 1.0, fw) * on * e * dash;
        }
        cr += (1.0 - e) * on * min(0.8 / spacing, 1.0);
        cr = cr * cs;
        col = mixc(col, srgb(70.0, 82.0, 92.0), 0.4 * cr);
        dh -= 1.0 * cr;
    }
    var ly = layer_paint(g.ice, col, LC_GLACIER);
    ly.dh = (*s).m.micro + dh;
    ly.hmode = HM_BLEND;
    ly.relit = 1.0;
    ly.clear = 1.0;
    composite(s, ly);
}

fn cold_moraine(s: ptr<function, Stack>, g: ColdIce) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let x = g.g + 0.4;
    let mo = smoothstep1(-0.05, 0.05, x) * smoothstep1(0.6, 0.5, x + 0.15 * g.n) * (1.0 - smoothstep1(0.5, 0.75, (*s).l.slope));
    if (mo <= 0.0) {
        return;
    }
    let boulders = 0.12 * perlin3(COLD_SEED_ICE + 12lu, p * (1.0lf / 4.0lf)) * band(4.0, gsd) + 0.08 * perlin3(COLD_SEED_ICE + 13lu, p * (1.0lf / 1.3lf)) * band(1.3, gsd);
    let rock = spal(s, BP_ROCK);
    let col = mixc(rock, srgb(120.0, 114.0, 106.0), 0.5) * (0.9 + boulders);
    let ridge = sin(PI * clamp(x / 0.55, 0.0, 1.0));
    var ly = layer_paint(mo, col, LC_MORAINE);
    ly.dh = (*s).m.micro + 3.0 * ridge;
    ly.hmode = HM_BLEND;
    ly.clear = 1.0;
    composite(s, ly);
}

fn cold_snow(s: ptr<function, Stack>, c: ColdClimate, g: ColdIce) {
    let polar = smoothstep1(9.0, 3.0, c.summer);
    let date = c.date - 7.0 * polar;
    if (date > 2.5 || (*s).cls == LC_FROZEN_WATER) {
        return;
    }
    let t = (*s).t;
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let hollow = smoothstep1(0.1, 0.8, -t.gully);
    let spur = smoothstep1(0.2, 0.8, t.gully);
    var n = (0.6 + 1.2 * polar) * (*s).pf.f[PF_SNOW] - (1.6 + 1.4 * polar) * hollow + 0.5 * spur;
    let b45 = band(45.0, gsd);
    if (b45 > 0.0) {
        n += 0.5 * perlin3(COLD_SEED_SNOW + 2lu, p * (1.0lf / 45.0lf)) * b45;
        let b12 = band(12.0, gsd);
        if (b12 > 0.0) {
            n += 0.25 * perlin3(COLD_SEED_SNOW + 3lu, p * (1.0lf / 12.0lf)) * b12;
        }
    }
    var x = date + n + 2.5 * clamp(cold_sunny(s), -1.0, 1.0);
    if (x > -3.2 && x < 1.0) {
        let b4 = band(4.0, gsd);
        if (b4 > 0.0) {
            x += 0.8 * perlin3(COLD_SEED_SNOW + 4lu, p * (1.0lf / 4.0lf)) * b4 + 0.4 * perlin3(COLD_SEED_SNOW + 5lu, p * (1.0lf / 1.3lf)) * band(1.3, gsd);
        }
    }
    var cov = smoothstep1(-0.55, -1.25, x) * (1.0 - smoothstep1(0.95, 1.5, (*s).l.slope));
    if (cov <= 0.0) {
        return;
    }
    let crown = smoothstep1(1.5, 4.0, (*s).height - (*s).l.ground) * (1.0 - min(g.ice, 1.0));
    cov *= 1.0 - 0.7 * crown;
    cov *= 1.0 - 0.85 * (*s).m.road_cov;
    cov *= 1.0 - 0.55 * (*s).m.town_cov;
    cov *= 1.0 - (*s).m.river_cov;
    if ((*s).l.river_hw > 0.0) {
        cov *= 1.0 - polar * (1.0 - smoothstep1((*s).l.river_hw + 2.0, (*s).l.river_hw + 30.0, abs((*s).l.river_d)));
    }
    cov *= 1.0 - g.snow * (1.0 - polar);
    if (cov <= 0.0) {
        return;
    }
    let col = pal3(PAL_SNOW) * (1.0 + 0.03 * (*s).pf.f[PF_DETAIL]) * vec3<f32>(1.0 - 0.03 * hollow, 1.0 - 0.015 * hollow, 1.0);
    composite(s, layer_paint(cov, col, LC_SEASONAL_SNOW));
}

// ---------------------------------------------------------------- kernel

/// `cold_polygons` (`kits::cold::polygons`).
fn cold_polygons(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    let cell = k.v[0];
    let wc = worley3(k.seed, i.p, f64(cell), k.v[1]);
    let ed = worley_edge_dist(wc, i.p);
    let tint = 1.0 + k.v[8] * (u01k(wc.id, 2lu) - 0.5);
    var col = k.col[0] * tint;
    var dh = 0.0;
    let x = min(wc.f1 / (0.5 * cell), 1.0);
    if (u01k(wc.id, 3lu) < k.v[7]) {
        dh = 1.5 * k.v[5] * (1.0 - x * x);
        col *= 1.08;
    } else if (u01k(wc.id, 1lu) < k.v[6]) {
        let pr = k.v[9] * 0.5 * cell * (0.6 + 0.4 * u01k(wc.id, 4lu));
        let pc = kband_cov(wc.f1, pr, i.fw);
        col = col + (k.col[2] - col) * pc;
        dh -= 0.2 * pc;
    }
    if (k.v[4] > 0.0) {
        let rc = kband_cov(ed - 0.5 * k.v[2] - 0.5 * k.v[4], 0.5 * k.v[4], i.fw);
        col = col + (col * 1.1 - col) * rc;
        dh = dh + (k.v[5] - dh) * rc;
    }
    let tc = kband_cov(ed, 0.5 * k.v[2], i.fw);
    col = col + (k.col[1] - col) * tc;
    dh = dh + (-k.v[3] - dh) * tc;
    var o = kout_none();
    o.cov = min(i.amount, 1.0);
    o.albedo = col;
    o.dh = dh;
    o.id = wc.id;
    return o;
}
