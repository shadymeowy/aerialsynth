// ---------------------------------------------------------------- kit desert: pass B
// The twin of the surface half of `kits/desert.rs` (helpers and constants: desert_relief.wgsl).

const DS_CRACK: u64 = 0xDE5E0030lu;
const DS_BRINE: u64 = 0xDE5E0031lu;
const DS_VARN: u64 = 0xDE5E0032lu;
const DS_STREAK: u64 = 0xDE5E0033lu;
const DS_SANDTEX: u64 = 0xDE5E0034lu;
const DS_INSROCK: u64 = 0xDE5E0035lu;
const DS_WASH: u64 = 0xDE5E0040lu;
const DS_BRAID: u64 = 0xDE5E0041lu;
const DS_BRAID2: u64 = 0xDE5E0042lu;

const DS_LC_SAND: u32 = 5u;

const DS_BAR_CELL: f32 = 380.0;
const DS_BAR_DENS: f32 = 0.8;
const DS_BAR_R0: f32 = 35.0;
const DS_BAR_R1: f32 = 95.0;
const DS_BAR_CA: f32 = 1.4777;
const DS_BAR_CV: f32 = 0.2252;

/// `desert::barchans`: (coverage, height).
fn desert_barchans(p: vec3<f64>, east: vec3<f32>, north: vec3<f32>, wd: vec2<f32>, amount: f32, gsd: f32, fw: f32) -> vec2<f32> {
    let r0 = DS_BAR_R0;
    let r1 = DS_BAR_R1;
    let er2 = (r0 * r0 + r0 * r1 + r1 * r1) / 3.0;
    let er3 = (r0 + r1) * (r0 * r0 + r1 * r1) / 4.0;
    let mc = amount * DS_BAR_DENS * DS_BAR_CA * er2 / (DS_BAR_CELL * DS_BAR_CELL);
    let mh = DS_BAR_CV / DS_BAR_CA * 0.21 * er3 / er2;
    let e = smoothstep1(1.2 * gsd, 3.0 * gsd, r0);
    if (e <= 0.0) {
        return vec2<f32>(mc, mh);
    }
    let st = desert_slab_sites(cfg.seed ^ DS_BAR, p, east, north, DS_BAR_CELL, 0.45 * DS_BAR_CELL);
    var cov = 0.0;
    var dh = 0.0;
    for (var i = 0u; i < st.n; i++) {
        let h = st.id[i];
        if (u01k(h, 5lu) >= DS_BAR_DENS * amount) {
            continue;
        }
        let r = r0 + (r1 - r0) * u01k(h, 6lu);
        let hh = r * (0.16 + 0.10 * u01k(h, 7lu));
        let a = 0.5 * (u01k(h, 8lu) - 0.5);
        let sa = sin(a);
        let ca = cos(a);
        let dw = vec2<f32>(wd.x * ca - wd.y * sa, wd.x * sa + wd.y * ca);
        let v = -st.rel[i] / r;
        let u = dot(v, dw);
        let sv = v.x * dw.y - v.y * dw.x;
        let so = 1.0 - sqrt(u * u + sv * sv);
        let si = sqrt((u - 0.55) * (u - 0.55) + sv * sv) - 0.85;
        let ci = clamp(so * r / fw + 0.5, 0.0, 1.0) * clamp(si * r / fw + 0.5, 0.0, 1.0);
        if (ci <= 0.0) {
            continue;
        }
        let a1 = clamp(so / 0.7, 0.0, 1.0);
        let a2 = clamp(si / 0.3, 0.0, 1.0);
        let taper = pow(max(1.0 - sv * sv, 0.0), 1.5);
        let hx = hh * min(a1 * a1 * (3.0 - 2.0 * a1), a2 * a2 * (3.0 - 2.0 * a2)) * taper;
        cov = max(cov, ci);
        dh = max(dh, hx);
    }
    let c = e * cov + (1.0 - e) * mc;
    if (c <= 0.0) {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(c, (e * cov * dh + (1.0 - e) * mc * mh) / c);
}

/// `desert::sand_colour`.
fn desert_sand_colour(s: ptr<function, Stack>) -> vec3<f32> {
    let si = (*s).t.style.y * 2.0;
    let j = min(u32(floor(si)), 1u);
    let base = mixc(spal(s, BP_SAND + j), spal(s, BP_SAND + j + 1u), si - f32(j));
    let lum = dot(base, vec3<f32>(0.2126, 0.7152, 0.0722));
    switch (*s).bio.st.litho {
        case 1u: { return mixc(base, vec3<f32>(lum * 1.15), 0.35); }
        case 2u: { return base * vec3<f32>(1.10, 0.94, 0.84); }
        case 3u: { return mixc(base, vec3<f32>(0.045), 0.8); }
        case 4u: { return base * vec3<f32>(1.04, 1.0, 0.9); }
        default: { return base; }
    }
}

/// `desert::crust`: salt crust plates, cracks, a darker floor, brine pools.
fn desert_crust(s: ptr<function, Stack>, tone: f32, shade: f32, mid: f32, pools: f32, pink: f32) -> vec3<f32> {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let crust_w = vec3<f32>(0.80, 0.78, 0.72) * (0.92 + 0.08 * shade);
    var tint = vec3<f32>(1.0, 1.0, 1.0);
    let ti = u32(3.0 * tone) % 3u;
    if (ti == 1u) {
        tint = vec3<f32>(1.0, 0.95, 0.86);
    } else if (ti == 2u) {
        tint = vec3<f32>(1.0, 0.93, 0.92);
    }
    var col = crust_w * tint;
    let ce = smoothstep1(1.2 * gsd, 3.0 * gsd, 12.0);
    var crack = 0.06;
    var plate = 1.0;
    if (ce > 0.0) {
        let cl = worley3(cfg.seed ^ DS_CRACK, p, 12.0lf, 0.85);
        let ed = worley_edge_dist(cl, p);
        crack = ce * band_cov(ed, 0.35, fw) + (1.0 - ce) * 0.06;
        plate = 1.0 + ce * 0.06 * (u01k(cl.id, 1lu) - 0.5);
    }
    col = col * plate * (1.0 - 0.35 * crack);
    col = mixc(col, col * vec3<f32>(0.78, 0.76, 0.72), 0.5 * mid);
    if (pools < 0.5) {
        let pn = perlin3(cfg.seed ^ DS_BRINE ^ 7lu, p / 400.0lf) + 0.5 * perlin3(cfg.seed ^ DS_BRINE ^ 9lu, p / 130.0lf) * band(130.0, gsd);
        let pool = smoothstep1(0.25, 0.32, pn + 0.6 * mid - 0.4) * mid;
        var brine = srgb(120.0, 156.0, 120.0);
        if (pink < 0.5) {
            brine = srgb(214.0, 150.0, 156.0);
        }
        col = mixc(col, brine, 0.85 * pool);
    }
    return col;
}

/// `desert::dry_lake`: arid lakes as salt pans from their margins inwards.
fn desert_dry_lake(s: ptr<function, Stack>) {
    if ((*s).l.water_kind != 2u) {
        return;
    }
    let ar = desert_aridity((*s).t.moist) * smoothstep1(0.0, 4.0, (*s).t.temp);
    if (ar <= 0.0) {
        return;
    }
    let depth = (*s).l.water - (*s).l.ground;
    let dmax = 40.0 * pow(smoothstep1(0.4, 0.9, ar), 2.0);
    let cov = clamp((dmax - depth) / 0.4 + 0.5, 0.0, 1.0);
    if (cov <= 0.0) {
        return;
    }
    let st = (*s).t.style;
    let mid = smoothstep1(0.3, 1.0, depth / max(dmax, 1e-3));
    let col = desert_crust(s, st.w, st.x, mid, st.z, st.y);
    var ly = layer_paint(cov, col, LC_SALT_FLAT);
    ly.solid = true;
    composite(s, ly);
}

/// `desert::azonal`: playas, the sand seas' surface, their stony fringe and its barchans.
fn desert_slot_azonal(s: ptr<function, Stack>) {
    let t = (*s).t;
    let c = (*s).c;
    let p = c.p;
    let gsd = c.gsd;
    let fw = (*s).l.fw;
    let pl = desert_playa(inst_list(FAM_DESERT_PLAYAS, (*s).l.blk, p), p, c.east, c.north);
    if (pl.found && pl.x < 1.4) {
        let x = pl.x;
        let r = pl.r;
        let id = pl.id;
        let dm = (1.0 - x) * r;
        let cov = clamp(dm / max(fw, 1e-3) + 0.5, 0.0, 1.0);
        if (cov > 0.0) {
            let mid = 1.0 - smoothstep1(0.2, 0.6, x + 0.15 * perlin3(cfg.seed ^ DS_BRINE, p / 900.0lf));
            let col = desert_crust(s, u01k(id, 13lu), u01k(id, 12lu), mid, u01k(id, 14lu), u01k(id, 15lu));
            var ly = layer_paint(cov, col, LC_SALT_FLAT);
            ly.clear = 1.0;
            composite(s, ly);
        }
        if (x > 0.98) {
            var ring = 0.0;
            ring += band_cov((x - 1.05) * r, 0.006 * r, fw);
            ring += band_cov((x - 1.13) * r, 0.006 * r, fw);
            ring += band_cov((x - 1.24) * r, 0.006 * r, fw);
            ring += band_cov((x - 1.36) * r, 0.006 * r, fw);
            ring = min(ring, 1.0) * band(0.03 * r, gsd);
            if (ring > 0.0) {
                let crust_w = vec3<f32>(0.80, 0.78, 0.72) * (0.92 + 0.08 * u01k(id, 12lu));
                composite(s, layer_paint(0.45 * ring, mixc((*s).col, crust_w, 0.7), 0u));
            }
        }
    }
    let ins = inst_list(FAM_DESERT_INSELBERGS, (*s).l.blk, p);
    for (var i = 0u; i < ins.n; i++) {
        let sh = desert_ins_shape(ins.items[i].id, desert_offset(p, ins.items[i].center, c.east, c.north));
        let cov = clamp((0.9 - sh.x) * sh.y / max(fw, 1e-3) + 0.5, 0.0, 1.0);
        if (cov > 0.0) {
            let rock = spal(s, BP_ROCK + 1u) * vec3<f32>(1.18, 1.12, 1.08);
            let sn = perlin3(cfg.seed ^ DS_INSROCK, p / 60.0lf) * band(60.0, gsd) + 0.5 * perlin3(cfg.seed ^ DS_INSROCK ^ 1lu, p / 15.0lf) * band(15.0, gsd);
            let col = mixc(rock, rock * vec3<f32>(0.5, 0.46, 0.44), 0.3 + 0.3 * sn);
            var ly = layer_paint(cov, col, 6u);
            ly.clear = 0.9;
            composite(s, ly);
        }
    }
    if (t.sand <= 0.0) {
        return;
    }
    let wd = desert_wind(p, c.lat, t.wind);
    let wl = desert_regime(wd.rho).y;
    let sc = desert_sand_colour(s);
    let black = (*s).bio.st.litho == 3u;
    let pnoise = (*s).pf.f[PF_PATCH];
    let erg = smoothstep1(0.47, 0.53, t.sand + 0.1 * pnoise);
    let fringe = (1.0 - erg) * smoothstep1(0.0, 0.08, t.sand);
    if (fringe > 0.0) {
        let soil = spal(s, BP_SOIL + 1u);
        var pav = mixc(soil, spal(s, BP_ROCK), 0.3) * 0.82;
        let vn = perlin3(cfg.seed ^ DS_VARN, p / 420.0lf) + 0.5 * perlin3(cfg.seed ^ DS_VARN ^ 1lu, p / 130.0lf) * band(130.0, gsd);
        let ve = smoothstep1(1.2 * gsd, 3.0 * gsd, 130.0);
        let varn = ve * smoothstep1(0.2, 0.3, vn) + (1.0 - ve) * 0.3;
        pav = mixc(pav, pav * vec3<f32>(0.55, 0.5, 0.48), 0.8 * varn);
        let sb = band(45.0, gsd);
        var streak = 0.2;
        if (sb > 0.0) {
            let ec = cross(c.up, c.east * wd.d.x + c.north * wd.d.y);
            let g = gully_octave(cfg.seed ^ DS_STREAK, p / 45.0lf, ec).x;
            streak = 0.2 + (smoothstep1(0.2, 0.9, g) - 0.2) * sb;
        }
        pav = mixc(pav, sc, 0.35 * streak * min(2.0 * t.sand, 1.0));
        var ly = layer_paint(fringe, pav, LC_GRAVEL);
        ly.clear = 0.6;
        composite(s, ly);
    }
    var cls = DS_LC_SAND;
    if (black) {
        cls = LC_VOLCANIC_ASH;
    }
    if (erg > 0.0) {
        var col = sc * (1.0 + 0.05 * perlin3(cfg.seed ^ DS_SANDTEX, p / 35.0lf) * band(35.0, gsd) + 0.03 * perlin3(cfg.seed ^ DS_SANDTEX ^ 1lu, p / 9.0lf) * band(9.0, gsd));
        if (wl > 0.0) {
            let res = 1.0 - smoothstep1(10.0, 80.0, gsd);
            let flat = 1.0 - smoothstep1(0.02, 0.08, (*s).l.slope);
            let k = wl * (res * flat + (1.0 - res) * 0.45) * 0.5;
            col = mixc(col, mixc(spal(s, BP_SOIL + 1u), col, 0.5) * 0.92, k);
        }
        var ly = layer_paint(erg, col, cls);
        ly.clear = 1.0;
        composite(s, ly);
    }
}

/// `desert::barchan_field` (slot 4).
fn desert_slot_disturbance(s: ptr<function, Stack>) {
    let t = (*s).t;
    let z0 = smoothstep1(0.02, 0.1, t.sand) * (1.0 - smoothstep1(0.42, 0.58, t.sand));
    if (z0 <= 0.0) {
        return;
    }
    let c = (*s).c;
    let wd = desert_wind(c.p, c.lat, t.wind);
    let zb = z0 * (0.35 + 0.65 * (1.0 - smoothstep1(0.35, 0.6, wd.rho)));
    let b = desert_barchans(c.p, c.east, c.north, wd.d, zb, c.gsd, (*s).l.fw);
    if (b.x > 0.0) {
        var cls = DS_LC_SAND;
        if ((*s).bio.st.litho == 3u) {
            cls = LC_VOLCANIC_ASH;
        }
        var ly = layer_paint(b.x, desert_sand_colour(s) * 1.06, cls);
        ly.dh = b.y;
        ly.hmode = HM_ADD;
        ly.clear = 1.0;
        composite(s, ly);
    }
}

/// `desert::washes`: braided dry washes.
fn desert_slot_water(s: ptr<function, Stack>) {
    if ((*s).is_water) {
        desert_dry_lake(s);
        return;
    }
    let t = (*s).t;
    let rd = (*s).l.river_d;
    let hw = (*s).l.river_hw;
    if (hw <= 0.0 || t.river_wet >= 0.5 || t.sand > 0.45) {
        return;
    }
    let ar = (1.0 - smoothstep1(0.25, 0.42, t.moist)) * smoothstep1(-4.0, 0.0, t.temp) * (1.0 - smoothstep1(0.3, 0.45, t.sand));
    if (ar <= 0.0) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let half = hw * (1.0 + 1.4 * ar) * (1.0 + 0.25 * perlin3(cfg.seed ^ DS_WASH, p / 700.0lf));
    let fwr = max((*s).l.fw, gsd * 0.35);
    let cov = band_cov(rd, half, fwr);
    if (cov <= 0.0) {
        return;
    }
    let lam = max(0.25 * half, 3.0);
    let ph = rd / lam + 0.9 * perlin3(cfg.seed ^ DS_BRAID, p / 90.0lf) + 0.4 * perlin3(cfg.seed ^ DS_BRAID2, p / 31.0lf) * band(31.0, gsd);
    let th = 0.5 + 0.5 * cos(TAU * ph);
    let bw = band(lam, gsd);
    let bar = mixc(spal(s, BP_SAND + 2u), spal(s, BP_SOIL + 1u), 0.25) * 1.02;
    let chan = mixc(bar, spal(s, BP_ROCK), 0.25) * 0.9;
    let col = mixc(bar, chan, smoothstep1(0.7, 0.9, th) * bw + (1.0 - bw) * 0.15);
    var ly = layer_paint(cov, col, LC_DRY_RIVERBED);
    ly.clear = 0.8;
    composite(s, ly);
}
