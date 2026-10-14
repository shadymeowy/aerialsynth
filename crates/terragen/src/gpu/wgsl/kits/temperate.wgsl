// The temperate, boreal and wetland kit (`kits/temperate.rs`): the WGSL twins of its layers,
// same names and order of operations.

// ---------------------------------------------------------------- kernels

/// A kit kernel at the sample, explicit and mean parts apart (`kernel_parts`).
struct TemperateParts {
    e: f32,
    x: KOut,
    mc: f32,
}

fn temperate_parts(s: ptr<function, Stack>, k: KP, size: f32, mean_cov: f32, amount: f32, q: vec2<f32>, aux0: f32) -> TemperateParts {
    var kin = s_kin(s, k.kind, amount);
    kin.q = q;
    kin.aux.x = aux0;
    var r: TemperateParts;
    r.e = smoothstep1(1.2 * kin.gsd, 3.0 * kin.gsd, size);
    r.x = kout_none();
    if (r.e > 0.0) {
        r.x = k_scatter(k, kin);
    }
    r.mc = mean_cov * clamp(amount, 0.0, 1.0);
    return r;
}

/// The layer of a kit kernel's parts (`kernel_layer`).
fn temperate_klayer(r: TemperateParts, mean: vec4<f32>, malb: vec3<f32>, cx: vec3<f32>, cm: vec3<f32>, hk: f32) -> Layer {
    var ly = layer_none();
    let cov = r.e * r.x.cov + (1.0 - r.e) * r.mc;
    if (cov <= 0.0) {
        return ly;
    }
    let we = r.e * r.x.cov / cov;
    let wm = (1.0 - r.e) * r.mc / cov;
    ly.cov = cov;
    ly.albedo = r.x.albedo * cx * we + malb * cm * wm;
    ly.dh = (r.x.dh * we + mean.y * wm) * hk;
    ly.hmode = temperate_tree_hmode();
    return ly;
}

// ---------------------------------------------------------------- helpers

fn temperate_tree_hmode() -> u32 {
    if ((cfg.flags & CF_TREES_DSM) != 0u) {
        return HM_MAX;
    }
    return HM_NONE;
}

fn temperate_eco_u1(s: ptr<function, Stack>) -> f32 {
    return clamp(((*s).bio.st.veg.w - 0.6) / 0.9, 0.0, 1.0);
}

fn temperate_eco_u2(s: ptr<function, Stack>) -> f32 {
    return clamp(((*s).bio.st.veg.z - 0.6) / 0.8, 0.0, 1.0);
}

fn temperate_site_boreal(s: ptr<function, Stack>) -> f32 {
    return 1.0 - smoothstep1(1.0, 5.0, (*s).bio.site.x);
}

fn temperate_site_oceanic(s: ptr<function, Stack>) -> f32 {
    let c = (*s).bio.site;
    return smoothstep1(700.0, 1100.0, c.y) * (1.0 - smoothstep1(13.0, 20.0, (*s).t.temp_range)) * smoothstep1(0.0, 4.0, c.x) * (1.0 - smoothstep1(12.0, 16.0, c.x));
}

fn temperate_site_dry_summer(s: ptr<function, Stack>) -> f32 {
    let c = (*s).bio.site;
    return (1.0 - smoothstep1(-0.45, -0.15, (*s).t.regime)) * smoothstep1(9.0, 13.0, c.x) * (1.0 - smoothstep1(1100.0, 1600.0, c.y));
}

fn temperate_forestry(s: ptr<function, Stack>) -> f32 {
    let st = (*s).bio.st;
    var a = 0.1;
    switch st.arch {
        case 9u: { a = 1.0; }
        case 7u: { a = 0.75; }
        case 6u: { a = 0.55; }
        case 11u: { a = 0.35; }
        case 4u, 5u, 10u: { a = 0.2; }
        default: {}
    }
    let productive = smoothstep1(0.6, 0.75, sveg(s, VEG_TREES));
    return a * (0.35 + 0.9 * temperate_eco_u1(s)) * (0.6 + 0.6 * st.misc.y) * (1.0 - smoothstep1(14.0, 19.0, (*s).bio.site.x)) * productive;
}

fn temperate_forest_est(s: ptr<function, Stack>) -> f32 {
    let t = (*s).t;
    let td = cfg.tree_density;
    let trees_mult = sveg(s, VEG_TREES) * (*s).bio.st.veg.x;
    let cold = (1.0 - smoothstep1(0.0, 8.0, t.temp)) * (1.0 - smoothstep1(0.15, 0.4, t.mountain));
    let base_cover = smoothstep1(0.3 - 0.14 * cold, 0.68 - 0.24 * cold, t.moist) * smoothstep1(-6.0, 2.0, t.temp) * td * trees_mult;
    let fpu = 0.5 + 0.5 * (*s).pf.f[PF_FOREST];
    let forest = smoothstep1(-0.03, 0.03, base_cover - fpu);
    let clear = 1.0 - 0.85 * smoothstep1(0.05, 0.4, s_agri(s)) * (1.0 - smoothstep1(0.45, 0.7, (*s).l.slope));
    let line = 1.0 - smoothstep1(0.0, 0.6, t.mountain * smoothstep1(-2.0, -6.0, t.temp));
    let m = (*s).m;
    return forest * clear * line * m.natural_ok * m.veg * (1.0 - m.field_cov);
}

fn temperate_patch_u(seed: u64, p: vec3<f64>, scale: f32, rough: f32, gsd: f32) -> f32 {
    var n = 0.0;
    var a = 1.0;
    var norm = 0.0;
    var lam = f64(scale);
    for (var o = 0u; o < 3u; o++) {
        n += a * perlin3(seed ^ (0x9A7lu + u64(o)), p * (1.0lf / lam)) * band(f32(lam), gsd);
        norm += a * a;
        a *= rough;
        lam *= 0.43lf;
    }
    let sn = n / (0.27 * sqrt(norm));
    return 1.0 / (1.0 + exp(-1.7 * sn));
}

fn temperate_patch_cov(u: f32, amount: f32, scale: f32, gsd: f32, fw: f32) -> f32 {
    let w = clamp(2.0 * fw / scale, 1e-3, 0.5);
    let e = smoothstep1(1.2 * gsd, 3.0 * gsd, scale);
    return e * smoothstep1(-w, w, amount - u) + (1.0 - e) * amount;
}

fn temperate_river_dist(s: ptr<function, Stack>) -> f32 {
    if ((*s).l.river_hw > 0.0) {
        return abs((*s).l.river_d) - (*s).l.river_hw;
    }
    return 1e6;
}

fn temperate_layer(cov: f32, albedo: vec3<f32>, cls: u32, clear: f32) -> Layer {
    var ly = layer_paint(cov, albedo, cls);
    ly.clear = clear;
    return ly;
}

// ---------------------------------------------------------------- slot 3: wetlands

const T_SEED_MIRE: u64 = 0x7E3B0001lu;
const T_SEED_MARSH: u64 = 0x7E3B0002lu;
const T_SEED_BEAVER: u64 = 0x7E3B0003lu;

fn temperate_slot_azonal(s: ptr<function, Stack>) {
    let t = (*s).t;
    if (t.moist < 0.38 || t.temp < -7.0 || t.temp > 24.0 || (*s).l.slope > 0.14 || (*s).m.natural_ok <= 0.0) {
        return;
    }
    temperate_floodplain_meadow(s);
    temperate_mires(s);
    temperate_marsh(s);
    temperate_beaver_ponds(s);
}

fn temperate_floodplain_meadow(s: ptr<function, Stack>) {
    let t = (*s).t;
    if (t.floodplain < 0.35 || t.temp < 1.0 || t.temp > 21.0 || (*s).l.river_hw <= 0.0) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let cov = smoothstep1(0.42, 0.5, t.floodplain + 0.08 * (*s).pf.f[PF_PATCH]);
    if (cov <= 0.0) {
        return;
    }
    let ph = abs((*s).l.river_d) / 32.0 + 0.7 * perlin3(T_SEED_MIRE ^ 0x71lu, p * (1.0lf / 260.0lf));
    let f = ph - floor(ph);
    let swale = (smoothstep1(0.3, 0.42, f) - smoothstep1(0.58, 0.7, f)) * band(32.0, gsd);
    let lush = srgb(94.0, 126.0, 52.0) * (1.0 + 0.1 * (*s).pf.f[PF_DETAIL]);
    let col = mixc(lush, srgb(66.0, 96.0, 50.0), 0.75 * swale);
    composite(s, layer_paint(cov, col, 0u));
}

fn temperate_mires(s: ptr<function, Stack>) {
    let t = (*s).t;
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let wet = smoothstep1(0.5, 0.75, t.moist);
    let cool = 1.0 - smoothstep1(9.0, 14.0, t.temp);
    let flat = 1.0 - smoothstep1(0.02, 0.07, (*s).l.slope);
    let low = 1.0 - smoothstep1(0.3, 0.6, t.mountain);
    let region = (0.5 + 1.1 * temperate_site_boreal(s) + 0.7 * temperate_site_oceanic(s)) * (0.5 + 0.8 * temperate_eco_u2(s));
    let amount = min(0.4 * wet * cool * flat * low * region, 0.8);
    if (amount <= 0.01) {
        return;
    }
    let u = temperate_patch_u(T_SEED_MIRE, p, 1100.0, 0.5, gsd);
    let cov = temperate_patch_cov(u, amount, 1100.0, gsd, fw);
    if (cov <= 0.0) {
        return;
    }
    let dpt = clamp((amount - u) / max(amount, 0.05), 0.0, 1.0);
    let rd = temperate_river_dist(s);
    let tt = t.temp + 2.5 * perlin3(T_SEED_MIRE ^ 0x11lu, p * (1.0lf / 3000.0lf));
    let fen = max(1.0 - smoothstep1(120.0, 380.0, rd), smoothstep1(0.3, 0.6, -t.gully));
    let strings = (1.0 - smoothstep1(1.5, 3.5, tt)) * max(temperate_site_boreal(s), 0.3);
    let mot = perlin3(T_SEED_MIRE ^ 0x21lu, p * (1.0lf / 4.0lf)) * band(4.0, gsd) + 0.6 * perlin3(T_SEED_MIRE ^ 0x22lu, p * (1.0lf / 1.5lf)) * band(1.5, gsd);
    var col: vec3<f32>;
    var pool: f32;
    var clear: f32;
    if (fen > 0.5) {
        let g = smoothstep1(-0.3, 0.4, perlin3(T_SEED_MIRE ^ 0x31lu, p * (1.0lf / 30.0lf)) * band(30.0, gsd));
        col = mixc(srgb(110.0, 114.0, 72.0), srgb(134.0, 128.0, 90.0), g) * (1.0 + 0.16 * mot);
        let e = band(40.0, gsd);
        let pu = perlin3(T_SEED_MIRE ^ 0x32lu, p * (1.0lf / 22.0lf)) + 0.5 * perlin3(T_SEED_MIRE ^ 0x33lu, p * (1.0lf / 9.0lf)) * band(9.0, gsd);
        pool = e * smoothstep1(0.5, 0.56, pu) * smoothstep1(0.3, 0.5, dpt) + (1.0 - e) * 0.02 * smoothstep1(0.3, 0.5, dpt);
        clear = 0.55 + 0.3 * smoothstep1(0.1, 0.4, dpt);
    } else if (strings > 0.5) {
        let up = (*s).c.up;
        let d1 = perlin3_d(T_SEED_MIRE ^ 0x41lu, p * (1.0lf / 600.0lf));
        let d2 = perlin3_d(T_SEED_MIRE ^ 0x42lu, p * (1.0lf / 170.0lf));
        let fval = 600.0 * d1.x + 0.35 * 170.0 * d2.x;
        let g = d1.yzw + d2.yzw * 0.35;
        let gt = max(length(g - up * dot(g, up)), 0.05);
        let x = fval / 15.0;
        let f = x - floor(x);
        let spacing = 15.0 / gt;
        let e = smoothstep1(1.2 * gsd, 3.0 * gsd, spacing);
        let dist = min(f, 1.0 - f) * spacing;
        let string = e * band_cov(dist, 1.3, fw) + (1.0 - e) * min(2.6 / spacing, 0.5);
        let mid = smoothstep1(0.22, 0.3, f) * (1.0 - smoothstep1(0.7, 0.78, f));
        let seg = perlin3(T_SEED_MIRE ^ 0x43lu, p * (1.0lf / 28.0lf)) + 0.4 * perlin3(T_SEED_MIRE ^ 0x44lu, p * (1.0lf / 11.0lf)) * band(11.0, gsd);
        let wetf = smoothstep1(0.15, 0.5, dpt);
        pool = e * mid * smoothstep1(-0.12, -0.06, seg) * wetf + (1.0 - e) * 0.3 * wetf;
        let flark = mixc(srgb(156.0, 140.0, 84.0), srgb(130.0, 128.0, 72.0), smoothstep1(-0.2, 0.3, seg)) * (1.0 + 0.12 * mot);
        col = mixc(flark, srgb(92.0, 84.0, 52.0) * (1.0 + 0.2 * mot), string);
        clear = 0.85 + 0.12 * smoothstep1(0.1, 0.3, dpt);
    } else {
        let a = perlin3(T_SEED_MIRE ^ 0x51lu, p * (1.0lf / 45.0lf)) * band(45.0, gsd);
        let b = perlin3(T_SEED_MIRE ^ 0x52lu, p * (1.0lf / 110.0lf));
        let c0 = mixc(srgb(150.0, 132.0, 92.0), srgb(134.0, 98.0, 74.0), smoothstep1(-0.25, 0.35, a));
        let c1 = mixc(c0, srgb(112.0, 118.0, 72.0), 0.5 * smoothstep1(0.05, 0.45, b)) * (1.0 + 0.12 * mot);
        let lagg = 1.0 - smoothstep1(0.04, 0.12, dpt);
        col = mixc(c1, srgb(80.0, 88.0, 54.0), lagg);
        let ring = dpt * 7.0 + 0.25 * perlin3(T_SEED_MIRE ^ 0x53lu, p * (1.0lf / 70.0lf));
        let fr = ring - floor(ring);
        let e = band(70.0, gsd);
        let blob = perlin3(T_SEED_MIRE ^ 0x54lu, p * (1.0lf / 34.0lf)) + 0.4 * perlin3(T_SEED_MIRE ^ 0x55lu, p * (1.0lf / 13.0lf)) * band(13.0, gsd);
        let zone = smoothstep1(0.38, 0.48, dpt);
        pool = e * zone * (1.0 - smoothstep1(0.24, 0.32, fr)) * smoothstep1(-0.1, -0.02, blob) + (1.0 - e) * 0.15 * zone;
        clear = 0.5 + 0.42 * smoothstep1(0.06, 0.25, dpt);
    }
    (*s).m.natural_ok *= 1.0 - cov;
    composite(s, temperate_layer(cov, col, LC_BOG, clear));
    if (pool > 0.0) {
        let water = srgb(34.0, 38.0, 36.0) * (1.0 + 0.15 * perlin3(T_SEED_MIRE ^ 0x61lu, p * (1.0lf / 7.0lf)) * band(7.0, gsd));
        composite(s, temperate_layer(pool * cov, water, LC_LAKE, 1.0));
    }
}

fn temperate_marsh(s: ptr<function, Stack>) {
    let t = (*s).t;
    if (t.temp < 3.0 || t.moist < 0.45) {
        return;
    }
    let rd = temperate_river_dist(s);
    if (t.floodplain < 0.25 && rd > 120.0) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let near = (1.0 - smoothstep1(30.0, 120.0, rd)) * 0.6;
    let flat = 1.0 - smoothstep1(0.02, 0.06, (*s).l.slope);
    let amount = 0.5 * max(smoothstep1(0.25, 0.6, t.floodplain), near) * smoothstep1(0.45, 0.7, t.moist) * smoothstep1(3.0, 7.0, t.temp) * (1.0 - smoothstep1(16.0, 21.0, t.temp)) * flat * (1.0 - 0.8 * smoothstep1(0.3, 0.6, s_agri(s)));
    if (amount <= 0.01) {
        return;
    }
    let u = temperate_patch_u(T_SEED_MARSH, p, 380.0, 0.55, gsd);
    let cov = temperate_patch_cov(u, amount, 380.0, gsd, fw);
    if (cov <= 0.0) {
        return;
    }
    let dm = clamp((amount - u) / max(amount, 0.05), 0.0, 1.0);
    let clone = perlin3(T_SEED_MARSH ^ 0x11lu, p * (1.0lf / 22.0lf)) * band(22.0, gsd);
    let stand = perlin3(T_SEED_MARSH ^ 0x13lu, p * (1.0lf / 70.0lf)) * band(70.0, gsd);
    let fine = perlin3(T_SEED_MARSH ^ 0x12lu, p * (1.0lf / 2.5lf)) * band(2.5, gsd) + 0.6 * perlin3(T_SEED_MARSH ^ 0x14lu, p * (1.0lf / 0.9lf)) * band(0.9, gsd);
    let straw = max(smoothstep1(0.55, 0.85, (*s).bio.st.land.x), 1.0 - smoothstep1(4.0, 8.0, t.temp));
    let green = mixc(srgb(92.0, 108.0, 50.0), srgb(124.0, 132.0, 64.0), smoothstep1(-0.3, 0.3, stand));
    let c0 = mixc(green, srgb(158.0, 146.0, 96.0), 0.55 * smoothstep1(0.15, 0.45, clone));
    let col = mixc(c0, srgb(184.0, 162.0, 108.0), straw) * (1.0 + 0.16 * fine);
    (*s).m.natural_ok *= 1.0 - cov;
    composite(s, temperate_layer(cov, col, LC_MARSH, 0.6 + 0.4 * smoothstep1(0.04, 0.15, dm)));
    let core = temperate_patch_cov(u, 0.22 * amount, 380.0, gsd, fw);
    let ch = perlin3(T_SEED_MARSH ^ 0x15lu, p * (1.0lf / 140.0lf)) + 0.3 * perlin3(T_SEED_MARSH ^ 0x16lu, p * (1.0lf / 45.0lf)) * band(45.0, gsd);
    let ec = band(24.0, gsd);
    let creek = (ec * band_cov(ch / 0.0107, 3.0, fw) + (1.0 - ec) * 0.05) * smoothstep1(0.1, 0.3, dm);
    let water = max(core, creek) * cov;
    if (water > 0.0) {
        composite(s, temperate_layer(water, srgb(40.0, 54.0, 50.0), LC_LAKE, 1.0));
    }
}

fn temperate_beaver_ponds(s: ptr<function, Stack>) {
    let t = (*s).t;
    let hw = (*s).l.river_hw;
    if (hw < 0.6 || hw > 5.0 || t.river_wet < 0.5 || t.temp < -4.0 || t.temp > 12.0 || t.moist < 0.42 || (*s).l.slope > 0.08) {
        return;
    }
    let rd = temperate_river_dist(s);
    if (rd > 90.0) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let c = floor(p * (1.0lf / 450.0lf));
    let h = hash3(T_SEED_BEAVER, i64(c.x), i64(c.y), i64(c.z));
    let climate = smoothstep1(0.42, 0.6, t.moist) * (1.0 - smoothstep1(8.0, 12.0, t.temp)) * (0.4 + 0.6 * temperate_site_boreal(s));
    if (u01k(h, 4lu) >= 0.6 * climate) {
        return;
    }
    let site = (c + vec3<f64>(f64(0.25 + 0.5 * u01k(h, 1lu)), f64(0.25 + 0.5 * u01k(h, 2lu)), f64(0.25 + 0.5 * u01k(h, 3lu)))) * 450.0lf;
    let r = 70.0 + 40.0 * u01k(h, 5lu);
    let ds = dist64(p, site);
    if (ds >= r) {
        return;
    }
    let rho = ds / r;
    let wp = (10.0 + 26.0 * u01k(h, 6lu)) * sqrt(1.0 - rho * rho);
    if (2.0 * wp < 0.05 * gsd) {
        return;
    }
    let meadow = clamp((2.2 * wp + 20.0 - rd) / fw + 0.5, 0.0, 1.0);
    if (meadow <= 0.0) {
        return;
    }
    (*s).m.natural_ok *= 1.0 - meadow;
    let grass = srgb(114.0, 120.0, 76.0) * (1.0 + 0.12 * (*s).pf.f[PF_DETAIL]);
    composite(s, temperate_layer(meadow, grass, LC_GRASS, 1.0));
    let ring = clamp((1.5 * wp + 8.0 - rd) / fw + 0.5, 0.0, 1.0);
    if (ring > 0.0) {
        composite(s, layer_paint(ring, srgb(96.0, 104.0, 70.0), 0u));
        let kp = temperate_parts(s, temperate_kp_snags(), TEMPERATE_SNAGS_SIZE, TEMPERATE_SNAGS_MEAN.x, ring, (*s).q_loc, 0.0);
        composite(s, temperate_klayer(kp, TEMPERATE_SNAGS_MEAN, TEMPERATE_SNAGS_MALB, vec3<f32>(1.0), vec3<f32>(1.0), 1.0));
    }
    let pond = clamp((wp - rd) / fw + 0.5, 0.0, 1.0);
    if (pond > 0.0) {
        composite(s, temperate_layer(pond, srgb(38.0, 46.0, 38.0), LC_LAKE, 1.0));
    }
}

// ---------------------------------------------------------------- slot 4: forestry

const T_SEED_STAND: u64 = 0x7E3C0001lu;
const T_SEED_WIND: u64 = 0x7E3C0002lu;
const T_SEED_FIRE: u64 = 0x7E3C0003lu;

fn temperate_slot_disturbance(s: ptr<function, Stack>) {
    let t = (*s).t;
    if (t.moist < 0.2 || t.temp < -7.0 || !(*s).has_region) {
        return;
    }
    let fe = temperate_forest_est(s);
    let dry = temperate_site_dry_summer(s);
    if (fe < 0.05 && dry < 0.05) {
        return;
    }
    if (fe > 0.05) {
        temperate_stands(s, fe);
        temperate_windthrow(s, fe);
    }
    temperate_firebreaks(s, fe, dry);
}

fn temperate_stands(s: ptr<function, Stack>, fe: f32) {
    let fi = temperate_forestry(s);
    if (fi < 0.08) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let q = (*s).q_rot + vec2<f32>((*s).pf.f[PF_WARP2], (*s).pf.f[PF_PATCH]) * 24.0;
    let seed = mix64((*s).t.region_id ^ T_SEED_STAND);
    let wc = 150.0 + 150.0 * u01k(seed, 1lu);
    let xc = q.x / wc;
    let cx = floor(xc);
    let hc = hash1(seed, i64(cx));
    let hr = wc * (1.1 + 1.6 * u01k(hc, 1lu));
    let yr = (q.y + hr * u01k(hc, 2lu)) / hr;
    let cy = floor(yr);
    let fx = xc - cx;
    let fy = yr - cy;
    let bid = hash2(seed, i64(cx), i64(cy));
    let inside = smoothstep1(0.35, 0.65, fe);
    let pc = min(0.07 * fi, 0.09);
    let uc = u01k(bid, 3lu);
    if (uc < pc && inside > 0.0) {
        let age = 16.0 * pow(uc / pc, 0.6);
        let ex = min(fx, 1.0 - fx) * wc;
        let ey = min(fy, 1.0 - fy) * hr;
        let en = 18.0 * (*s).pf.f[PF_DETAIL];
        let inside = inside * clamp((min(ex, ey) - 8.0 + en) / fw + 0.5, 0.0, 1.0);
        let mot = perlin3(T_SEED_STAND ^ 0x11lu, p * (1.0lf / 6.0lf)) * band(6.0, gsd) + 0.5 * perlin3(T_SEED_STAND ^ 0x12lu, p * (1.0lf / 2.0lf)) * band(2.0, gsd);
        var col: vec3<f32>;
        var clear: f32;
        if (age < 3.0) {
            let c0 = mixc(srgb(92.0, 84.0, 70.0), srgb(106.0, 90.0, 72.0), smoothstep1(-0.3, 0.4, mot));
            let c1 = mixc(c0, spal(s, BP_SOIL), 0.15) * (1.0 + 0.22 * mot);
            let tr = q.x / 19.0 + u01k(bid, 4lu);
            let trail = band_cov((tr - floor(tr) - 0.5) * 19.0, 1.6, fw) * band(19.0, gsd) * smoothstep1(4.0, 14.0, ey);
            let c2 = mixc(c1, srgb(128.0, 116.0, 94.0), 0.7 * trail);
            let landing = band_cov(ey, 12.0, fw) * band_cov(ex - 0.5 * wc, 22.0, fw);
            col = mixc(c2, srgb(138.0, 126.0, 104.0), landing);
            clear = 0.93;
        } else {
            let g = smoothstep1(3.0, 7.0, age);
            let gt = mixc(srgb(92.0, 102.0, 64.0), srgb(78.0, 92.0, 60.0), u01k(bid, 6lu));
            let mot2 = perlin3(T_SEED_STAND ^ 0x14lu, p * (1.0lf / 40.0lf)) * band(40.0, gsd);
            let c0 = mixc(srgb(110.0, 104.0, 70.0), gt, g) * (1.0 + 0.16 * mot + 0.14 * mot2);
            let fire = smoothstep1(-0.1, 0.3, perlin3(T_SEED_STAND ^ 0x13lu, p * (1.0lf / 14.0lf)) * band(14.0, gsd)) * smoothstep1(2.5, 4.0, age) * (1.0 - smoothstep1(6.0, 9.0, age));
            col = mixc(c0, srgb(166.0, 92.0, 132.0), 0.55 * fire * (0.6 + 0.4 * smoothstep1(0.4, 0.7, (*s).bio.st.land.x)));
            clear = 1.0;
        }
        composite(s, temperate_layer(inside, col, LC_CLEAR_CUT, clear));
        if (age > 3.0) {
            let grow = clamp((age - 3.0) / 7.0, 0.0, 1.0);
            let kp = temperate_parts(s, temperate_kp_young(), TEMPERATE_YOUNG_SIZE, TEMPERATE_YOUNG_MEAN.x, inside * (0.25 + 0.75 * grow), q, 0.0);
            var tc = srgb(84.0, 112.0, 50.0);
            if (u01k(bid, 5lu) < 0.65) {
                tc = srgb(44.0, 70.0, 42.0);
            }
            composite(s, temperate_klayer(kp, TEMPERATE_YOUNG_MEAN, TEMPERATE_YOUNG_MALB, tc, tc, 0.6 * (age - 2.0)));
        }
    }
    var line = cx;
    if (fx >= 0.5) {
        line = cx + 1.0;
    }
    let dl = min(fx, 1.0 - fx) * wc;
    if (u01k(hash1(seed ^ 0x40ADlu, i64(line)), 1lu) < 0.5 * min(fi, 1.0) && 10.0 > 0.03 * gsd) {
        let verge = band_cov(dl, 5.0, fw) * inside;
        if (verge > 0.0) {
            composite(s, temperate_layer(verge, spal(s, BP_GRASS_WET) * 1.1, 0u, 1.0));
            let road = band_cov(dl, 1.8, fw) * inside;
            composite(s, temperate_layer(road, srgb(156.0, 142.0, 116.0) * (1.0 + 0.08 * (*s).pf.f[PF_DETAIL]), LC_TRACK, 1.0));
        }
    }
}

fn temperate_windthrow(s: ptr<function, Stack>, fe: f32) {
    let c = (*s).bio.site;
    let windy = smoothstep1(400.0, 700.0, c.y) * (1.0 - smoothstep1(10.0, 16.0, c.x));
    let amount = 0.05 * windy * (0.3 + 0.9 * temperate_eco_u2(s)) * smoothstep1(0.4, 0.7, fe);
    if (amount <= 0.002) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let u = temperate_patch_u(T_SEED_WIND, p, 260.0, 0.55, gsd);
    let cov = temperate_patch_cov(u, amount, 260.0, gsd, fw);
    if (cov <= 0.0) {
        return;
    }
    let mot = perlin3(T_SEED_WIND ^ 0x11lu, p * (1.0lf / 5.0lf)) * band(5.0, gsd);
    let col = mixc(srgb(124.0, 98.0, 70.0), srgb(98.0, 92.0, 66.0), smoothstep1(-0.2, 0.3, perlin3(T_SEED_WIND ^ 0x12lu, p * (1.0lf / 24.0lf)))) * (1.0 + 0.12 * mot);
    composite(s, temperate_layer(cov, col, LC_CLEAR_CUT, 0.82));
    let ang = TAU * u01k(mix64((*s).t.region_id ^ T_SEED_WIND), 1lu);
    let kp = temperate_parts(s, temperate_kp_logs(), TEMPERATE_LOGS_SIZE, TEMPERATE_LOGS_MEAN.x, cov, (*s).q_loc, ang);
    var ly = temperate_klayer(kp, TEMPERATE_LOGS_MEAN, TEMPERATE_LOGS_MALB, vec3<f32>(1.0), vec3<f32>(1.0), 1.0);
    ly.hmode = HM_ADD;
    composite(s, ly);
}

fn temperate_firebreaks(s: ptr<function, Stack>, fe: f32, dry: f32) {
    let t = (*s).t;
    let scrub = dry * smoothstep1(0.12, 0.2, t.moist) * (1.0 - smoothstep1(0.6, 0.7, t.moist)) * (*s).m.natural_ok * (*s).m.veg;
    let veg = max(fe, scrub);
    let inten = max(dry, 0.6 * temperate_site_boreal(s) * temperate_forestry(s));
    if (veg < 0.3 || inten < 0.1) {
        return;
    }
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let seed = mix64((*s).t.region_id ^ T_SEED_FIRE);
    let hw = 7.0 + 8.0 * dry;
    if (2.0 * hw < 0.03 * gsd) {
        return;
    }
    let q = (*s).q_rot;
    var d = 1e9;
    for (var axis = 0u; axis < 2u; axis++) {
        let sp = 1300.0 + 1200.0 * u01k(seed, 2lu + u64(axis));
        var qa = q.x;
        if (axis == 1u) {
            qa = q.y;
        }
        let x = qa / sp + u01k(seed, 4lu + u64(axis));
        let k = floor(x + 0.5);
        if (u01k(hash1(seed ^ (0xF1lu + u64(axis)), i64(k)), 1lu) < 0.6 * inten) {
            d = min(d, abs(x - k) * sp);
        }
    }
    let on = smoothstep1(0.3, 0.5, veg);
    let cov = band_cov(d, hw, fw) * on;
    if (cov <= 0.0) {
        return;
    }
    let ground = mixc(spal(s, BP_SOIL + 1u), spal(s, BP_GRASS_DRY), 0.45) * (1.0 + 0.12 * (*s).pf.f[PF_DETAIL]);
    composite(s, temperate_layer(cov, ground, LC_BARE, 1.0));
    let track = band_cov(d, 1.6, fw) * on;
    if (track > 0.0) {
        composite(s, layer_paint(track, srgb(176.0, 160.0, 130.0), LC_TRACK));
    }
}

// ---------------------------------------------------------------- canopy: autumn

fn temperate_autumn_colour(i: u32) -> vec3<f32> {
    switch i {
        case 0u: { return srgb(176.0, 150.0, 62.0); }
        case 1u: { return srgb(184.0, 124.0, 48.0); }
        case 2u: { return srgb(172.0, 92.0, 40.0); }
        case 3u: { return srgb(138.0, 52.0, 40.0); }
        default: { return srgb(112.0, 84.0, 54.0); }
    }
}

fn temperate_autumn_weight(i: u32, red: f32) -> f32 {
    switch i {
        case 0u: { return 0.3; }
        case 1u: { return 0.28; }
        case 2u: { return 0.2 - 0.5 * red; }
        case 3u: { return red; }
        default: { return 0.22 - 0.5 * red; }
    }
}

fn temperate_slot_canopy(s: ptr<function, Stack>) {
    let t = (*s).t;
    if (t.moist < 0.28 || t.temp < -6.0 || t.temp > 20.0) {
        return;
    }
    let aut = smoothstep1(0.55, 0.85, (*s).bio.st.land.x);
    let larch = 0.32 * (1.0 - smoothstep1(1.0, 7.0, t.temp)) * temperate_eco_u1(s) * max(temperate_site_boreal(s), 0.25);
    if (aut <= 0.0 && larch <= 0.01) {
        return;
    }
    let fe = temperate_forest_est(s);
    if (fe <= 0.05) {
        return;
    }
    let conifer = clamp((1.0 - smoothstep1(4.0, 13.0, t.temp)) + sveg(s, VEG_CONIFER) + (*s).bio.st.veg.y, 0.0, 1.0);
    let leaf = aut * fe * (1.0 - conifer) * 0.85;
    if (leaf > 0.01) {
        let kp = temperate_parts(s, temperate_kp_autumn(), TEMPERATE_AUTUMN_SIZE, TEMPERATE_AUTUMN_MEAN.x, leaf, (*s).q_loc, 0.0);
        let red = 0.06 + 0.3 * temperate_eco_u2(s);
        var mean_c = vec3<f32>(0.0);
        for (var i = 0u; i < 5u; i++) {
            mean_c += temperate_autumn_colour(i) * temperate_autumn_weight(i, red);
        }
        var col_x = mean_c;
        if (kp.x.cov > 0.0) {
            let u = u01k(kp.x.id, 0xA1lu);
            var acc = 0.0;
            for (var i = 0u; i < 5u; i++) {
                acc += temperate_autumn_weight(i, red);
                if (u < acc) {
                    col_x = temperate_autumn_colour(i);
                    break;
                }
            }
        }
        composite(s, temperate_klayer(kp, TEMPERATE_AUTUMN_MEAN, TEMPERATE_AUTUMN_MALB, col_x, mean_c, 1.0));
    }
    let la = larch * fe * max(conifer, 0.3);
    if (la > 0.01) {
        let kp = temperate_parts(s, temperate_kp_larch(), TEMPERATE_LARCH_SIZE, TEMPERATE_LARCH_MEAN.x, la, (*s).q_loc, 0.0);
        let c = mixc(srgb(98.0, 124.0, 62.0), srgb(204.0, 156.0, 56.0), aut);
        composite(s, temperate_klayer(kp, TEMPERATE_LARCH_MEAN, TEMPERATE_LARCH_MALB, c, c, 1.0));
    }
}

// ---- generated by kits::temperate::wgsl_consts (AERIALSYNTH_BLESS=1 cargo test -p terragen --lib temperate)
fn temperate_kp_autumn() -> KP {
    var k: KP;
    k.kind = 0u;
    k.seed = 0xa2dd4d4652a08696lu;
    k.v = array<f32, 16>(9.0, 1.0, 1.0, 3.0, 4.4, 13.0, 19.0, 0.85, 0.22, 0.55, 1.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    k.col = array<vec3<f32>, 3>(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.09758735, 0.116970666, 0.04231141), vec3<f32>(0.0, 0.0, 0.0));
    return k;
}
const TEMPERATE_AUTUMN_SIZE: f32 = 9.0;
const TEMPERATE_AUTUMN_MEAN: vec4<f32> = vec4<f32>(0.49636415, 11.480269, 0.0, 0.0);
const TEMPERATE_AUTUMN_MALB: vec3<f32> = vec3<f32>(0.8207362, 0.8207362, 0.8207362);
fn temperate_kp_larch() -> KP {
    var k: KP;
    k.kind = 0u;
    k.seed = 0x90cc8560d74460balu;
    k.v = array<f32, 16>(6.0, 1.0, 2.0, 1.5, 2.3, 14.0, 21.0, 0.85, 0.2, 0.5, 1.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    k.col = array<vec3<f32>, 3>(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.09758735, 0.116970666, 0.04231141), vec3<f32>(0.0, 0.0, 0.0));
    return k;
}
const TEMPERATE_LARCH_SIZE: f32 = 6.0;
const TEMPERATE_LARCH_MEAN: vec4<f32> = vec4<f32>(0.32059255, 6.234672, 0.0, 0.0);
const TEMPERATE_LARCH_MALB: vec3<f32> = vec3<f32>(0.83433354, 0.83433354, 0.83433354);
fn temperate_kp_young() -> KP {
    var k: KP;
    k.kind = 0u;
    k.seed = 0x9cc43f7835cc0e13lu;
    k.v = array<f32, 16>(2.6, 1.0, 2.0, 0.55, 1.0, 0.85, 1.15, 0.35, 0.45, 0.7, 1.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    k.col = array<vec3<f32>, 3>(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.09758735, 0.116970666, 0.04231141), vec3<f32>(0.0, 0.0, 0.0));
    return k;
}
const TEMPERATE_YOUNG_SIZE: f32 = 2.6;
const TEMPERATE_YOUNG_MEAN: vec4<f32> = vec4<f32>(0.29940763, 0.34119454, 0.0, 0.0);
const TEMPERATE_YOUNG_MALB: vec3<f32> = vec3<f32>(0.7615867, 0.7615867, 0.7615867);
fn temperate_kp_logs() -> KP {
    var k: KP;
    k.kind = 0u;
    k.seed = 0x59e1c2c3a91c9355lu;
    k.v = array<f32, 16>(7.0, 1.0, 5.0, 0.3, 0.45, 0.3, 0.6, 0.9, 0.2, 0.2, 10.0, 1.0, 0.3, 0.35, 0.0, 0.0);
    k.col = array<vec3<f32>, 3>(vec3<f32>(0.4452012, 0.39157248, 0.31398872), vec3<f32>(0.1746474, 0.107023105, 0.05780543), vec3<f32>(0.0, 0.0, 0.0));
    return k;
}
const TEMPERATE_LOGS_SIZE: f32 = 7.0;
const TEMPERATE_LOGS_MEAN: vec4<f32> = vec4<f32>(0.11681169, 0.4544741, 0.0, 0.0);
const TEMPERATE_LOGS_MALB: vec3<f32> = vec3<f32>(0.3231928, 0.2683953, 0.20567684);
fn temperate_kp_snags() -> KP {
    var k: KP;
    k.kind = 0u;
    k.seed = 0xe7bfebd8a962d865lu;
    k.v = array<f32, 16>(6.0, 1.0, 1.0, 0.35, 0.7, 4.0, 9.0, 0.9, 0.2, 0.4, 1.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    k.col = array<vec3<f32>, 3>(vec3<f32>(0.3231432, 0.3049873, 0.26225066), vec3<f32>(0.09758735, 0.116970666, 0.04231141), vec3<f32>(0.0, 0.0, 0.0));
    return k;
}
const TEMPERATE_SNAGS_SIZE: f32 = 6.0;
const TEMPERATE_SNAGS_MEAN: vec4<f32> = vec4<f32>(0.022928942, 4.300544, 0.0, 0.0);
const TEMPERATE_SNAGS_MALB: vec3<f32> = vec3<f32>(0.27784222, 0.26223156, 0.2254861);
// ---- end of generated
