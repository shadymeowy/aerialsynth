// The settlements kit (`kits/settlements.rs`, `settlements/town.rs`, `settlements/sites.rs`):
// places (zones, districts, street patterns, lots, buildings, village forms, lights) and sites
// (airports, ports, quarries, solar and wind farms, gas flares, reservoirs, dams). Same names
// (prefix `st_`), same order of operations as the Rust.

const ST_PLACE: u32 = 192u;
const ST_AIRPORT: u32 = 193u;
const ST_PORT: u32 = 194u;
const ST_QUARRY: u32 = 195u;
const ST_SOLAR: u32 = 196u;
const ST_WIND: u32 = 197u;
const ST_FLARE: u32 = 198u;
const ST_RESERVOIR: u32 = 200u;
const ST_DAM: u32 = 201u;

const SP_GRID: u32 = 0u;
const SP_RADIAL: u32 = 1u;
const SP_ORGANIC: u32 = 2u;
const SP_CURVI: u32 = 3u;
const SP_SUPER: u32 = 4u;
const SP_INFORMAL: u32 = 5u;
const SP_INDUSTRIAL: u32 = 6u;
const SP_LINEAR: u32 = 7u;
const SP_DISPERSED: u32 = 8u;
const SP_KRAAL: u32 = 9u;

const SZ_CBD: u32 = 0u;
const SZ_INNER: u32 = 1u;
const SZ_OUTER: u32 = 2u;
const SZ_SUBURB: u32 = 3u;
const SZ_PERI: u32 = 4u;

const STY_HOUSE: u32 = 1u;
const STY_ROW: u32 = 2u;
const STY_PERIM: u32 = 3u;
const STY_MIDRISE: u32 = 4u;
const STY_TOWER: u32 = 5u;
const STY_WAREHOUSE: u32 = 6u;

const SF_CITY: u32 = 0u;
const SF_NUCLEATED: u32 = 1u;
const SF_LINEAR: u32 = 2u;
const SF_WALDHUFEN: u32 = 3u;
const SF_DISPERSED: u32 = 4u;
const SF_KRAAL: u32 = 5u;
const SF_PERCHED: u32 = 6u;

const ST_STRIP: f32 = 1400.0;
const ST_TAU: f32 = 6.283185307179586;
const ST_PI: f32 = 3.141592653589793;

/// `town::Pl`.
struct StPl {
    c: vec3<f64>,
    ex: vec3<f32>,
    ey: vec3<f32>,
    r: f32,
    level: u32,
    form: u32,
    sub: bool,
    seed: u64,
    arch: u32,
    dev: f32,
    elong: f32,
    wind: f32,
    pop: f32,
}

fn st_seed(v2: f32, v3: f32) -> u64 {
    return mix64(u64(u32(v2)) | (u64(u32(v3)) << 24u));
}

fn st_decode(st: FStamp) -> StPl {
    var o: StPl;
    let lv = u32(st.v0.y);
    o.c = st.center.xyz;
    o.ex = st.ex.xyz;
    o.ey = st.ey.xyz;
    o.r = st.v0.x;
    o.level = lv & 7u;
    o.form = (lv >> 3u) & 7u;
    o.sub = ((lv >> 6u) & 1u) == 1u;
    o.seed = st_seed(st.v0.z, st.v0.w);
    o.arch = u32(st.v1.x);
    o.dev = st.v1.y;
    o.elong = st.v1.z;
    o.wind = st.v1.w;
    o.pop = st.h;
    return o;
}

fn st_local(pl: StPl, p: vec3<f64>) -> vec2<f32> {
    let d = vec3<f32>(p - pl.c);
    return vec2<f32>(dot(d, pl.ex), dot(d, pl.ey));
}

fn st_rel0(pl: StPl, q: vec2<f32>) -> f32 {
    let se = sqrt(pl.elong);
    return length(vec2<f32>(q.x / se, q.y * se)) / pl.r;
}

fn st_foot_noise(pl: StPl, p: vec3<f64>, gsd: f32) -> f32 {
    let r = pl.r;
    return 0.32 * perlin3(pl.seed ^ 0x71lu, p * (1.0lf / (0.9lf * f64(r)))) + 0.18 * perlin3(pl.seed ^ 0x72lu, p * (1.0lf / (0.35lf * f64(r)))) * band(0.35 * r, gsd);
}

fn st_density(pl: StPl, rel: f32) -> f32 {
    if (pl.form == SF_CITY) {
        return 1.0 - smoothstep1(0.35, 1.0, rel);
    }
    return 1.0 - smoothstep1(0.45, 1.0, rel);
}

fn st_in_strips(pl: StPl, q: vec2<f32>) -> bool {
    return pl.form == SF_WALDHUFEN && abs(q.x) < 0.9 * pl.r * sqrt(pl.elong) && abs(q.y) < ST_STRIP + 60.0;
}

fn st_zone_of(pl: StPl, rel: f32) -> u32 {
    var cbd = 0.0;
    if (pl.level >= 3u) {
        cbd = 0.07 + 0.03 * (f32(pl.level) - 3.0) + 0.03 * u01k(pl.seed, 40lu);
    } else if (pl.level == 2u) {
        cbd = 0.07;
    }
    if (rel < cbd) {
        return SZ_CBD;
    } else if (rel < 0.35) {
        return SZ_INNER;
    } else if (rel < 0.6) {
        return SZ_OUTER;
    } else if (rel < 0.85) {
        return SZ_SUBURB;
    }
    return SZ_PERI;
}

fn st_wrap_pi(a: f32) -> f32 {
    return a - ST_TAU * floor((a + ST_PI) / ST_TAU);
}

fn st_rot2(v: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec2<f32>(c * v.x + s * v.y, -s * v.x + c * v.y);
}

fn st_pick_pattern(pl: StPl, z: u32, did: u64, industrial: bool) -> u32 {
    if (industrial) {
        return SP_INDUSTRIAL;
    }
    let u = u01k(did, 4lu);
    let a = pl.arch;
    if (z == SZ_PERI && pl.dev < 0.45 && (a == 0u || a == 1u || a == 2u || a == 3u || a == 8u) && u < 0.65) {
        return SP_INFORMAL;
    }
    if (a == 3u) {
        return SP_ORGANIC;
    }
    if ((a == 5u || a == 6u || a == 7u || a == 9u) && (z == SZ_SUBURB || z == SZ_PERI) && pl.level >= 3u && u < 0.75) {
        return SP_CURVI;
    }
    if ((a == 4u || a == 9u) && u < 0.5) {
        return SP_SUPER;
    }
    return SP_GRID;
}

/// `town::shared_industry`.
fn st_shared_industry(pl: StPl) -> bool {
    return st_pick_pattern(pl, SZ_OUTER, mix64(pl.seed ^ 0x2A7Elu), false) == SP_GRID;
}

/// `town::core_ring` (x: distance, y: ring radius; y < 0: none).
fn st_core_ring(pl: StPl, q: vec2<f32>) -> vec2<f32> {
    let a = pl.arch;
    if (pl.form != SF_CITY || pl.level < 2u || !(a == 5u || a == 6u || a == 8u || a == 11u)) {
        return vec2<f32>(0.0, -1.0);
    }
    let se = sqrt(pl.elong);
    let qe = vec2<f32>(q.x / se, q.y * se);
    let phi = atan2(qe.y, qe.x);
    let s = pl.seed ^ 0x219lu;
    let cbd = select(0.07, 0.07 + 0.03 * (f32(pl.level) - 3.0) + 0.03 * u01k(pl.seed, 40lu), pl.level >= 3u);
    let rr = pl.r * (1.25 * cbd + 0.015) * (1.0 + 0.12 * sin(2.0 * phi + TAU * u01k(s, 1lu)) + 0.07 * sin(3.0 * phi + TAU * u01k(s, 2lu)) + 0.04 * sin(5.0 * phi + TAU * u01k(s, 3lu)));
    return vec2<f32>(length(qe), rr);
}

/// `town::ring_dist`.
fn st_ring_dist(pl: StPl, q: vec2<f32>) -> f32 {
    let c = st_core_ring(pl, q);
    if (c.y < 0.0 || pl.level < 3u) {
        return 1e9;
    }
    return abs(c.x - c.y);
}

/// `town::old_quarter` (x: distance, y: radius; y < 0: none).
fn st_old_quarter(pl: StPl, q: vec2<f32>) -> vec2<f32> {
    let ar = pl.arch;
    if (pl.form != SF_CITY || pl.level < 2u || !(ar == 5u || ar == 6u || ar == 8u || ar == 11u)) {
        return vec2<f32>(0.0, -1.0);
    }
    let cbd = select(0.07, 0.07 + 0.03 * (f32(pl.level) - 3.0) + 0.03 * u01k(pl.seed, 40lu), pl.level >= 3u);
    let rr = pl.r * (1.25 * cbd + 0.015);
    let se = sqrt(pl.elong);
    let s = pl.seed ^ 0x219lu;
    let a = TAU * u01k(s, 4lu);
    var c0 = 0.2 * rr;
    var ro = 1.5 * rr;
    if (pl.level >= 3u) {
        c0 = 0.95 * rr;
        ro = 0.62 * rr;
    }
    let rel = vec2<f32>(q.x / se, q.y * se) - vec2<f32>(cos(a), sin(a)) * c0;
    let phi = atan2(rel.y, rel.x);
    let wob = 1.0 + 0.16 * sin(3.0 * phi + TAU * u01k(s, 5lu)) + 0.09 * sin(5.0 * phi + TAU * u01k(s, 6lu)) + 0.05 * sin(8.0 * phi + TAU * u01k(s, 7lu));
    return vec2<f32>(length(rel) / wob, ro);
}

/// `town::old_core`.
fn st_old_core(pl: StPl, q: vec2<f32>) -> f32 {
    let c = st_old_quarter(pl, q);
    if (c.y < 0.0) {
        return 0.0;
    }
    return 1.0 - smoothstep1(0.45 * c.y, c.y, c.x);
}

/// `town::old_near`.
fn st_old_near(pl: StPl, q: vec2<f32>) -> bool {
    let c = st_old_quarter(pl, q);
    return c.y >= 0.0 && c.x < c.y + 70.0;
}

/// `town::Blk`.
struct StBlk {
    id: u64,
    u: f32,
    v: f32,
    bw: f32,
    bd: f32,
    ds: f32,
    sw: f32,
    along: f32,
    sid: u64,
    major: bool,
    mj: u32,
    seg: u64,
    smid: vec2<f32>,
}

/// `town::OldSub`.
struct StOldSub {
    id: u64,
    d: f32,
    c: vec2<f32>,
    side: i64,
    along: f32,
}

/// `town::old_sub`'s open lanes.
fn st_old_lane(id: u64, k: f32, n: f32, salt: i64, hw: f32) -> f32 {
    if (k > 0.0 && k < n && u01k(hash2(id ^ 0x1A6lu, salt, i64(k)), 1lu) < 0.6) {
        return hw;
    }
    return 0.0;
}

/// `town::old_sub`'s lane positions.
fn st_old_line(id: u64, k: f32, n: f32, s: f32, len: f32, salt: i64) -> f32 {
    if (k <= 0.0) {
        return 0.0;
    }
    if (k >= n) {
        return len;
    }
    return (k + 0.44 * (u01(hash2(id ^ 0x1A5lu, salt, i64(k))) - 0.5)) * s;
}

/// `town::old_sub`.
fn st_old_sub(blk: StBlk) -> StOldSub {
    let lw = 30.0 + 14.0 * u01k(blk.id, 15lu);
    let lane_hw = 1.4 + 0.8 * u01k(blk.id, 78lu);
    let nu = max(round(blk.bw / lw), 1.0);
    let nv = max(round(blk.bd / lw), 1.0);
    let su = blk.bw / nu;
    let sv = blk.bd / nv;
    var iu = clamp(floor(blk.u / su), 0.0, nu - 1.0);
    if (blk.u < st_old_line(blk.id, iu, nu, su, blk.bw, -1)) {
        iu -= 1.0;
    } else if (blk.u >= st_old_line(blk.id, iu + 1.0, nu, su, blk.bw, -1)) {
        iu += 1.0;
    }
    iu = clamp(iu, 0.0, nu - 1.0);
    let salt = i64(iu);
    var iv = clamp(floor(blk.v / sv), 0.0, nv - 1.0);
    if (blk.v < st_old_line(blk.id, iv, nv, sv, blk.bd, salt)) {
        iv -= 1.0;
    } else if (blk.v >= st_old_line(blk.id, iv + 1.0, nv, sv, blk.bd, salt)) {
        iv += 1.0;
    }
    iv = clamp(iv, 0.0, nv - 1.0);
    let u0 = st_old_line(blk.id, iu, nu, su, blk.bw, -1);
    let u1 = st_old_line(blk.id, iu + 1.0, nu, su, blk.bw, -1);
    let v0 = st_old_line(blk.id, iv, nv, sv, blk.bd, salt);
    let v1 = st_old_line(blk.id, iv + 1.0, nv, sv, blk.bd, salt);
    var ds = array<f32, 4>(
        blk.u - u0 - st_old_lane(blk.id, iu, nu, -1, lane_hw),
        u1 - blk.u - st_old_lane(blk.id, iu + 1.0, nu, -1, lane_hw),
        blk.v - v0 - st_old_lane(blk.id, iv, nv, salt, lane_hw),
        v1 - blk.v - st_old_lane(blk.id, iv + 1.0, nv, salt, lane_hw),
    );
    var side = 0u;
    for (var k = 1u; k < 4u; k++) {
        if (ds[k] < ds[side]) {
            side = k;
        }
    }
    var o: StOldSub;
    o.id = hash3(blk.id ^ 0x01Dlu, i64(iu), i64(iv), 5li);
    o.d = ds[side];
    o.c = vec2<f32>(0.5 * (u0 + u1), 0.5 * (v0 + v1));
    o.side = i64(side);
    o.along = select(blk.u, blk.v, side < 2u);
    return o;
}

/// `town::Ev`.
struct StEv {
    cov: f32,
    col: vec3<f32>,
    h: f32,
    cls: u32,
    roof: f32,
    rim: f32,
    win: vec3<f32>,
    em: vec3<f32>,
    tree: f32,
    tree_h: f32,
    ds: f32,
    sw: f32,
    along: f32,
    sid: u64,
    bid: u64,
    major: bool,
    street: f32,
    zone: u32,
    pattern: u32,
    lamp_sp: f32,
}

fn st_ev_none() -> StEv {
    var o: StEv;
    o.cov = 0.0;
    o.col = vec3<f32>(0.0);
    o.h = 0.0;
    o.cls = 0u;
    o.roof = 0.0;
    o.rim = 0.0;
    o.win = vec3<f32>(0.0);
    o.em = vec3<f32>(0.0);
    o.tree = 0.0;
    o.tree_h = 0.0;
    o.ds = 1e9;
    o.sw = 0.0;
    o.along = 0.0;
    o.sid = 0lu;
    o.bid = 0lu;
    o.major = false;
    o.street = 0.0;
    o.zone = 0u;
    o.pattern = 0u;
    o.lamp_sp = 0.0;
    return o;
}

/// `town::Env`.
struct StEnv {
    gsd: f32,
    fw: f32,
    n: f32,
    moist: f32,
    detail: f32,
    ground: vec3<f32>,
}

fn st_asphalt() -> vec3<f32> {
    return srgb(58.0, 58.0, 62.0);
}
fn st_concrete() -> vec3<f32> {
    return srgb(158.0, 154.0, 146.0);
}
fn st_paving() -> vec3<f32> {
    return srgb(176.0, 164.0, 146.0);
}

fn st_yard(moist: f32, u: f32) -> vec3<f32> {
    return mixc(srgb(150.0, 132.0, 102.0), srgb(86.0, 110.0, 54.0), smoothstep1(0.22, 0.6, moist)) * (0.9 + 0.2 * u);
}

/// `town::car_park`.
fn st_car_park(h: u64, u: f32, v: f32, fw: f32, gsd: f32) -> vec3<f32> {
    let row = v / 16.0 - floor(v / 16.0);
    let in_row = select(0.0, 1.0, abs(row - 0.5) > 0.2);
    let line = band_cov((u / 2.6 - round(u / 2.6)) * 2.6, 0.08, fw) * in_row;
    let col = mixc(mixc(st_asphalt() * 1.45, srgb(210.0, 210.0, 204.0), line * band(1.0, gsd)), srgb(128.0, 128.0, 130.0), 0.3 * (1.0 - band(2.0, gsd)));
    let pc = tr_parked(h, u, v, fw, gsd);
    return mixc(col, pc.yzw, pc.x);
}

/// `town::roof_col`.
fn st_roof_col(arch: u32, u: f32, flat: bool) -> vec3<f32> {
    let k = u32(u * 3.99);
    switch arch {
        case 20u: {
            let k6 = u32(u * 5.99);
            let f = u * 6.0 - f32(k6);
            var c = srgb(184.0, 180.0, 170.0);
            switch k6 {
                case 0u: { c = srgb(52.0, 66.0, 86.0); }
                case 1u: { c = srgb(64.0, 88.0, 92.0); }
                case 2u: { c = srgb(96.0, 104.0, 114.0); }
                case 3u: { c = srgb(108.0, 94.0, 78.0); }
                case 4u: { c = srgb(150.0, 152.0, 150.0); }
                default: {}
            }
            return c * (0.92 + 0.16 * f);
        }
        case 21u: {
            if (u < 0.3) {
                return srgb(168.0, 168.0, 164.0);
            } else if (u < 0.5) {
                return srgb(180.0, 172.0, 156.0);
            } else if (u < 0.62) {
                return srgb(200.0, 198.0, 192.0);
            } else if (u < 0.8) {
                return srgb(142.0, 142.0, 140.0);
            } else if (u < 0.93) {
                return srgb(150.0, 142.0, 128.0);
            }
            return srgb(124.0, 130.0, 136.0);
        }
        case 22u: {
            if (k == 0u) {
                return srgb(140.0, 140.0, 138.0);
            } else if (k == 1u) {
                return srgb(132.0, 92.0, 62.0);
            } else if (k == 2u) {
                return srgb(70.0, 96.0, 150.0);
            }
            return srgb(96.0, 84.0, 72.0);
        }
        case 23u: {
            return srgb(222.0, 218.0, 206.0);
        }
        case 24u: {
            return srgb(160.0, 70.0, 50.0);
        }
        default: {}
    }
    var c0: vec3<f32>;
    var c1: vec3<f32>;
    var c2: vec3<f32>;
    var c3: vec3<f32>;
    if (flat) {
        c0 = srgb(128.0, 126.0, 122.0);
        c1 = srgb(104.0, 102.0, 100.0);
        c2 = srgb(158.0, 154.0, 146.0);
        c3 = srgb(86.0, 86.0, 90.0);
    } else if (arch == 5u || arch == 11u) {
        c0 = srgb(172.0, 94.0, 64.0);
        c1 = srgb(188.0, 112.0, 76.0);
        c2 = srgb(150.0, 82.0, 60.0);
        c3 = srgb(200.0, 140.0, 104.0);
    } else if (arch == 3u) {
        c0 = srgb(214.0, 206.0, 190.0);
        c1 = srgb(196.0, 176.0, 146.0);
        c2 = srgb(180.0, 150.0, 116.0);
        c3 = srgb(226.0, 220.0, 208.0);
    } else if (arch == 6u || arch == 4u) {
        c0 = srgb(88.0, 88.0, 94.0);
        c1 = srgb(146.0, 84.0, 66.0);
        c2 = srgb(118.0, 86.0, 72.0);
        c3 = srgb(70.0, 72.0, 80.0);
    } else if (arch == 9u || arch == 10u) {
        c0 = srgb(84.0, 112.0, 92.0);
        c1 = srgb(82.0, 96.0, 122.0);
        c2 = srgb(128.0, 128.0, 130.0);
        c3 = srgb(150.0, 62.0, 52.0);
    } else if (arch == 7u) {
        c0 = srgb(112.0, 106.0, 100.0);
        c1 = srgb(92.0, 82.0, 74.0);
        c2 = srgb(134.0, 124.0, 112.0);
        c3 = srgb(76.0, 76.0, 80.0);
    } else if (arch == 2u) {
        c0 = srgb(150.0, 128.0, 92.0);
        c1 = srgb(132.0, 112.0, 80.0);
        c2 = srgb(142.0, 142.0, 140.0);
        c3 = srgb(120.0, 100.0, 76.0);
    } else {
        c0 = srgb(152.0, 152.0, 150.0);
        c1 = srgb(132.0, 86.0, 62.0);
        c2 = srgb(76.0, 104.0, 140.0);
        c3 = srgb(166.0, 92.0, 72.0);
    }
    if (k == 0u) {
        return c0;
    } else if (k == 1u) {
        return c1;
    } else if (k == 2u) {
        return c2;
    }
    return c3;
}

/// `town::bsp`: (lot hash, rectangle x0 y0 x1 y1).
struct StLot {
    h: u64,
    r: vec4<f32>,
}

fn st_bsp(h0: u64, u: f32, v: f32, bw: f32, bd: f32, depth: u32, min_side: f32) -> StLot {
    var x0 = 0.0;
    var y0 = 0.0;
    var x1 = bw;
    var y1 = bd;
    var h = h0;
    for (var d = 0u; d < 4u; d++) {
        if (d >= depth) {
            break;
        }
        let w = x1 - x0;
        let hg = y1 - y0;
        if (max(w, hg) < 2.0 * min_side) {
            break;
        }
        let t = 0.35 + 0.3 * u01k(h, 1lu);
        if (w >= hg) {
            let sx = x0 + t * w;
            if (u >= sx) {
                x0 = sx;
                h = mix64(h ^ 0xB5lu);
            } else {
                x1 = sx;
                h = mix64(h ^ 0xA3lu);
            }
        } else {
            let sy = y0 + t * hg;
            if (v >= sy) {
                y0 = sy;
                h = mix64(h ^ 0xC7lu);
            } else {
                y1 = sy;
                h = mix64(h ^ 0xD9lu);
            }
        }
    }
    var o: StLot;
    o.h = h;
    o.r = vec4<f32>(x0, y0, x1, y1);
    return o;
}

fn st_boxd(r: vec2<f32>, c: vec2<f32>, half: vec2<f32>) -> vec2<f32> {
    return half - abs(r - c);
}

fn st_aa(d: f32, fw: f32) -> f32 {
    return clamp(d / fw + 0.5, 0.0, 1.0);
}

fn st_min2(v: vec2<f32>) -> f32 {
    return min(v.x, v.y);
}

fn st_rem(i: i64, m: i64) -> i64 {
    return ((i % m) + m) % m;
}

fn st_normalize2(v: vec2<f32>) -> vec2<f32> {
    let l = length(v);
    if (l < 1e-12) {
        return vec2<f32>(0.0);
    }
    return v / l;
}

/// `town::Dist`.
struct StDist {
    did: u64,
    zone: u32,
    pattern: u32,
    theta: f32,
    art: f32,
    dmul: f32,
    b: f32,
    asp: f32,
    sw: f32,
    jit: f32,
    warp: f32,
    w2cell: f32,
    mg: i64,
    buf: f32,
    nbind: f32,
}

/// `town::district`.
/// `district_pattern`: (pattern, zone at the site, industrial).
fn st_district_pattern(pl: StPl, point: vec2<f32>, id: u64) -> vec3<u32> {
    let rd = st_rel0(pl, point);
    let zd = st_zone_of(pl, rd);
    let ang = atan2(point.y, point.x);
    var ind = false;
    var nw = 0u;
    if (pl.level >= 3u) {
        nw = 2u;
    } else if (pl.level == 2u) {
        nw = 1u;
    }
    for (var k = 0u; k < 2u; k++) {
        if (k < nw) {
            var a0 = pl.wind;
            if (k > 0u) {
                a0 = pl.wind + ST_PI * 1.4 * (u01k(pl.seed, 41lu) - 0.5);
            }
            let hw = 0.22 + 0.2 * u01k(pl.seed, 42lu + u64(k));
            if (abs(st_wrap_pi(ang - a0)) < hw && rd > 0.25 && rd < 0.95 && u01k(id, 3lu) < 0.8) {
                ind = true;
            }
        }
    }
    var pattern = st_pick_pattern(pl, zd, mix64(pl.seed ^ 0x2A7Elu), ind);
    if (pl.level < 2u) {
        pattern = SP_GRID;
    }
    return vec3<u32>(pattern, zd, select(0u, 1u, ind));
}

/// `town::nbind_at`.
fn st_nbind_at(pl: StPl, q: vec2<f32>) -> f32 {
    if (pl.form != SF_CITY || pl.level < 2u || !st_shared_industry(pl)) {
        return 1e9;
    }
    var dcell = 600.0;
    if (pl.level == 3u) {
        dcell = 1000.0;
    } else if (pl.level >= 4u) {
        dcell = 1400.0;
    }
    let wl = 0.3 * dcell;
    let qd = q
        + vec2<f32>(sin(q.y / wl + 6.0 * u01k(pl.seed, 51lu)), sin(q.x / (1.17 * wl) + 6.0 * u01k(pl.seed, 52lu))) * (0.16 * dcell)
        + vec2<f32>(sin(q.y / (0.4 * wl) + 6.0 * u01k(pl.seed, 54lu)), sin(q.x / (0.43 * wl) + 6.0 * u01k(pl.seed, 55lu))) * (0.045 * dcell);
    let dc = worley2(pl.seed ^ 0xD15lu, qd, dcell, 0.85);
    if (st_district_pattern(pl, dc.point, dc.id).x == SP_GRID && st_district_pattern(pl, dc.point2, dc.id2).x == SP_INDUSTRIAL) {
        return worley2_edge_dist(dc, qd);
    }
    return 1e9;
}

fn st_district(pl: StPl, q: vec2<f32>, rel_here: f32) -> StDist {
    let fm = pl.form;
    var pattern = SP_GRID;
    var z = st_zone_of(pl, rel_here);
    var theta = 0.0;
    var did = pl.seed;
    var art = 1e9;
    var b = 110.0;
    var asp = 1.0;
    var sw = 4.0;
    var jit = 0.3;
    var warp = 0.0;
    var w2cell = 0.0;
    var mg = 0li;
    var buf = 1e9;
    var nbind = 1e9;
    if (fm == SF_CITY) {
        var dcell = 400.0;
        if (pl.level == 2u) {
            dcell = 600.0;
        } else if (pl.level == 3u) {
            dcell = 1000.0;
        } else if (pl.level >= 4u) {
            dcell = 1400.0;
        }
        let wl = 0.3 * dcell;
        let qd = q
            + vec2<f32>(sin(q.y / wl + 6.0 * u01k(pl.seed, 51lu)), sin(q.x / (1.17 * wl) + 6.0 * u01k(pl.seed, 52lu))) * (0.16 * dcell)
            + vec2<f32>(sin(q.y / (0.4 * wl) + 6.0 * u01k(pl.seed, 54lu)), sin(q.x / (0.43 * wl) + 6.0 * u01k(pl.seed, 55lu))) * (0.045 * dcell);
        let dc = worley2(pl.seed ^ 0xD15lu, qd, dcell, 0.85);
        did = dc.id;
        let dp = st_district_pattern(pl, dc.point, dc.id);
        pattern = dp.x;
        let zd = dp.y;
        let shr = st_shared_industry(pl);
        let net1 = select(pattern, SP_GRID, shr && pattern == SP_INDUSTRIAL);
        if (pl.level >= 2u) {
            let pt2 = st_district_pattern(pl, dc.point2, dc.id2).x;
            let net2 = select(pt2, SP_GRID, shr && pt2 == SP_INDUSTRIAL);
            if (net2 != net1 || (pattern == SP_INDUSTRIAL && !shr)) {
                art = worley2_edge_dist(dc, qd);
            }
            if (pattern == SP_INDUSTRIAL && pt2 != SP_INDUSTRIAL) {
                buf = worley2_edge_dist(dc, qd);
            }
            if (shr && pattern == SP_GRID && pt2 == SP_INDUSTRIAL) {
                nbind = worley2_edge_dist(dc, qd);
            }
        }
        if (dp.z != 0u) {
            z = SZ_OUTER;
        } else {
            z = min(max(zd, min(z, SZ_INNER)), SZ_PERI);
        }
        if (pattern != SP_INDUSTRIAL || shr) {
            did = mix64(pl.seed ^ 0x1A70lu ^ u64(net1));
        }
        if (pattern == SP_INDUSTRIAL && shr) {
            mg = select(2li, 3li, u01k(dc.id, 14lu) < 0.5);
        }
        let survey = pl.arch == 7u;
        if (survey) {
            theta = 0.0;
        } else {
            theta = (u01k(pl.seed ^ 0x7E7Alu, 5lu) - 0.5) * 0.3;
        }
        var bf = 1.0;
        let ar = pl.arch % 12u;
        if (ar == 2u || ar == 5u || ar == 10u || ar == 11u) {
            bf = 0.9;
        } else if (ar == 3u) {
            bf = 0.85;
        } else if (ar == 6u) {
            bf = 1.05;
        } else if (ar == 7u) {
            bf = 1.25;
        } else if (ar == 8u) {
            bf = 0.8;
        }
        let u2 = u01k(did, 6lu);
        if (pattern == SP_GRID) {
            b = (70.0 + 60.0 * u2) * bf;
            asp = select(1.0 + 0.6 * u01k(did, 7lu), 2.2, survey);
            sw = select(4.5 + 2.5 * u01k(did, 8lu), 7.0, z == SZ_CBD);
        } else if (pattern == SP_RADIAL) {
            b = 80.0 + 40.0 * u2;
            sw = 5.0;
        } else if (pattern == SP_CURVI) {
            b = (80.0 + 40.0 * u2) * bf;
            asp = 1.8 + 0.6 * u01k(did, 7lu);
            sw = 4.5;
            jit = 0.15;
            warp = 34.0;
        } else if (pattern == SP_ORGANIC) {
            w2cell = select(55.0 + 35.0 * u2, 40.0, pl.arch == 3u);
            sw = select(2.8 + 1.4 * u01k(did, 8lu), 1.7, pl.arch == 3u);
        } else if (pattern == SP_SUPER) {
            b = 280.0 + 110.0 * u2;
            sw = 9.0;
            jit = 0.1;
        } else if (pattern == SP_INFORMAL) {
            b = 60.0 + 20.0 * u2;
            sw = 2.3;
            jit = 0.45;
            warp = 9.0;
            w2cell = 7.5;
        } else if (mg > 0li) {
            b = (70.0 + 60.0 * u2) * bf;
            asp = select(1.0 + 0.6 * u01k(did, 7lu), 2.2, survey);
            sw = 1.3 * (4.5 + 2.5 * u01k(did, 8lu));
        } else {
            b = 230.0 + 110.0 * u2;
            asp = 0.7 + 0.4 * u01k(did, 7lu);
            sw = 8.0;
            jit = 0.1;
        }
    } else {
        z = select(SZ_SUBURB, SZ_INNER, rel_here < 0.35);
        if (fm == SF_NUCLEATED) {
            pattern = SP_ORGANIC;
            w2cell = 52.0;
            sw = 2.6;
        } else if (fm == SF_PERCHED) {
            pattern = SP_ORGANIC;
            w2cell = 34.0;
            sw = 1.8;
        } else if (fm == SF_LINEAR || fm == SF_WALDHUFEN) {
            pattern = SP_LINEAR;
            b = 22.0;
            sw = 3.5;
            jit = 0.45;
        } else if (fm == SF_DISPERSED) {
            pattern = SP_DISPERSED;
            w2cell = 170.0;
        } else {
            pattern = SP_KRAAL;
            w2cell = select(120.0, 1e5, pl.level == 0u);
        }
    }
    var dmul = 1.0;
    if (fm == SF_CITY && pl.level >= 2u) {
        dmul = min(0.8 + 0.4 * u01k(did, 13lu), 1.0);
    }
    var o: StDist;
    o.did = did;
    o.zone = z;
    o.pattern = pattern;
    o.theta = theta;
    o.art = art;
    o.dmul = dmul;
    o.b = b;
    o.asp = asp;
    o.sw = sw;
    o.jit = jit;
    o.warp = warp;
    o.w2cell = w2cell;
    o.mg = mg;
    o.buf = buf;
    o.nbind = nbind;
    return o;
}

/// A jittered grid line (`line` closures of `place_eval`).
fn st_line(sd: u64, axis: i64, i: i64, bb: f32, jit: f32) -> f32 {
    return (f32(i) + jit * (u01(hash2(sd ^ u64(axis), axis, i)) - 0.5)) * bb;
}

/// A grid cell along one axis: (index, offset in it, its size).
struct StCell {
    i: i64,
    off: f32,
    size: f32,
}

fn st_cell(sd: u64, axis: i64, x: f32, bb: f32, jit: f32, m: i64) -> StCell {
    var i = i64(floor(x / bb));
    if (x < st_line(sd, axis, i, bb, jit)) {
        i -= 1li;
    } else if (x >= st_line(sd, axis, i + 1li, bb, jit)) {
        i += 1li;
    }
    i = (i - st_rem(i, m));
    let a0 = st_line(sd, axis, i, bb, jit);
    let a1 = st_line(sd, axis, i + m, bb, jit);
    var o: StCell;
    o.i = i;
    o.off = x - a0;
    o.size = a1 - a0;
    return o;
}

/// The lot density at offset `dq` from `q` (`dens_at` of `place_eval`).
fn st_dens_at(pl: StPl, q: vec2<f32>, dq: vec2<f32>, n: f32, dmul: f32) -> f32 {
    return st_density(pl, st_rel0(pl, q + dq) / (1.0 + n)) * dmul;
}

/// `town::place_eval`.
/// `town::big_grid_warp`.
fn st_big_grid_warp(qr: vec2<f32>, seed: u64) -> vec2<f32> {
    let s = seed ^ 0xB16lu;
    let b = vec2<f32>(
        sin(qr.y / 1100.0 + TAU * u01k(s, 1lu)) + 0.5 * sin(qr.y / 430.0 + TAU * u01k(s, 2lu)),
        sin(qr.x / 1250.0 + TAU * u01k(s, 3lu)) + 0.5 * sin(qr.x / 470.0 + TAU * u01k(s, 4lu)),
    ) * 70.0;
    let st = vec2<f32>(sin(qr.x / 1600.0 + TAU * u01k(s, 5lu)), sin(qr.y / 1700.0 + TAU * u01k(s, 6lu))) * 190.0;
    return qr + b + st;
}

fn st_place_eval(pl: StPl, q: vec2<f32>, d: StDist, e: StEnv) -> StEv {
    var o = st_ev_none();
    let fw = e.fw;
    let rel_here = st_rel0(pl, q) / (1.0 + e.n);
    let dens_here = st_density(pl, rel_here) * d.dmul;
    let fm = pl.form;
    let did = d.did;
    let z = d.zone;
    let pattern = d.pattern;
    let theta = d.theta;
    let art = d.art;
    let dmul = d.dmul;
    let b = d.b;
    let asp = d.asp;
    let oldk = select(0.0, st_old_core(pl, q), pattern == SP_GRID);
    let oldg = pattern == SP_GRID && st_old_near(pl, q);
    let sw = d.sw * (1.0 - 0.4 * oldk);
    let jit = d.jit;
    let warp = d.warp;
    let w2cell = d.w2cell;
    o.zone = z;
    o.pattern = pattern;
    // ---- the block
    var blk: StBlk;
    blk.id = did;
    blk.u = 0.0;
    blk.v = 0.0;
    blk.bw = 0.0;
    blk.bd = 0.0;
    blk.ds = 1e9;
    blk.sw = sw;
    blk.along = 0.0;
    blk.sid = 0lu;
    blk.major = false;
    blk.mj = 0u;
    blk.seg = 0lu;
    blk.smid = vec2<f32>(0.0);
    let gridlike = pattern == SP_GRID || pattern == SP_CURVI || pattern == SP_SUPER || pattern == SP_INFORMAL || pattern == SP_INDUSTRIAL;
    if (gridlike) {
        var qr = st_rot2(q, theta);
        if (warp > 0.0) {
            let l = 230.0 * warp / 42.0;
            qr += vec2<f32>(sin(qr.y / l + 6.0 * u01k(did, 9lu)), sin(qr.x / (1.13 * l) + 6.0 * u01k(did, 10lu))) * warp;
        }
        if ((pattern == SP_GRID || d.mg > 0li) && pl.level >= 3u) {
            qr = st_big_grid_warp(qr, pl.seed);
        }
        if (oldk > 0.0) {
            let s = did ^ 0x01Dlu;
            qr += vec2<f32>(
                sin(qr.y / 170.0 + 6.0 * u01k(s, 1lu)) + 0.4 * sin(qr.y / 61.0 + 6.0 * u01k(s, 2lu)),
                sin(qr.x / 190.0 + 6.0 * u01k(s, 3lu)) + 0.4 * sin(qr.x / 67.0 + 6.0 * u01k(s, 4lu)),
            ) * (16.0 * oldk);
        }
        let bx = b;
        let by = b * asp;
        let sd = did ^ 0x5EE7lu;
        let m = max(d.mg, 1li);
        let cx = st_cell(sd, 0li, qr.x, bx, jit, m);
        let cy = st_cell(sd, 1li, qr.y, by, jit, m);
        let near_x = cx.i + select(0li, m, cx.off > cx.size - cx.off);
        let near_y = cy.i + select(0li, m, cy.off > cy.size - cy.off);
        let dx = min(cx.off, cx.size - cx.off);
        let dy = min(cy.off, cy.size - cy.off);
        let main_x = st_rem(near_x, 4li) == 0li && pattern == SP_GRID;
        let main_y = st_rem(near_y, 4li) == 0li && pattern == SP_GRID;
        let swx = select(sw, sw * 1.5, main_x);
        let swy = select(sw, sw * 1.5, main_y);
        var mid = vec2<f32>(0.0);
        if (dx - swx < dy - swy) {
            blk.ds = dx;
            blk.sw = swx;
            blk.along = qr.y;
            blk.sid = hash2(sd ^ 0x1Alu, 0li, near_x);
            blk.major = main_x;
            blk.seg = hash2(blk.sid, 3li, cy.i);
            mid = vec2<f32>(select(-cx.off, cx.size - cx.off, near_x > cx.i), 0.5 * cy.size - cy.off);
        } else {
            blk.ds = dy;
            blk.sw = swy;
            blk.along = qr.x;
            blk.sid = hash2(sd ^ 0x1Alu, 1li, near_y);
            blk.major = main_y;
            blk.seg = hash2(blk.sid, 4li, cx.i);
            mid = vec2<f32>(0.5 * cx.size - cx.off, select(-cy.off, cy.size - cy.off, near_y > cy.i));
        }
        blk.smid = st_rot2(mid, -theta);
        blk.id = hash3(sd, cx.i, cy.i, 7li);
        if (pattern == SP_GRID) {
            blk.mj = select(0u, 1u, st_rem(cx.i, 4li) == 0li) | select(0u, 2u, st_rem(cx.i + m, 4li) == 0li) | select(0u, 4u, st_rem(cy.i, 4li) == 0li) | select(0u, 8u, st_rem(cy.i + m, 4li) == 0li);
        }
        blk.u = cx.off - sw;
        blk.v = cy.off - sw;
        blk.bw = cx.size - 2.0 * sw;
        blk.bd = cy.size - 2.0 * sw;
    } else if (pattern == SP_RADIAL) {
        let r = length(q);
        let phi = atan2(q.y, q.x) + ST_PI;
        let dr = b;
        let i = floor(r / dr);
        let v = r - i * dr;
        let nsp = exp2(ceil(log2(max(ST_TAU * (i + 0.5) * dr / (1.6 * dr), 4.0))));
        let dphi = ST_TAU / nsp;
        let j = floor(phi / dphi);
        let arc = (phi - j * dphi) * r;
        let bwr = dphi * r;
        let d_ring = min(v, dr - v);
        let d_rad = min(arc, bwr - arc);
        if (d_ring < d_rad) {
            blk.ds = d_ring;
            blk.along = phi * r;
            blk.sid = hash2(pl.seed ^ 0x2Alu, i64(i) + select(0li, 1li, v > 0.5 * dr), 0li);
            blk.major = st_rem(i64(i), 3li) == 0li;
        } else {
            blk.ds = d_rad;
            blk.along = r;
            blk.sid = hash2(pl.seed ^ 0x2Blu, i64(i), i64(j));
            blk.major = false;
        }
        blk.id = hash2(pl.seed ^ 0x2Clu, i64(i), i64(j));
        blk.u = arc - sw;
        blk.v = v - sw;
        blk.bw = bwr - 2.0 * sw;
        blk.bd = dr - 2.0 * sw;
    } else if (pattern == SP_LINEAR) {
        let sd = pl.seed ^ 0x11Elu;
        let cl = st_cell(sd, 0li, q.x, b, jit, 1li);
        let side = select(1li, -1li, q.y < 0.0);
        blk.ds = abs(q.y);
        blk.along = q.x;
        blk.sid = hash1(sd, 1li);
        blk.major = true;
        blk.id = hash3(sd, cl.i, side, 3li);
        blk.u = cl.off;
        blk.bw = cl.size;
        blk.v = abs(q.y) - sw;
        blk.bd = 40.0 + 18.0 * u01k(blk.id, 2lu);
    }
    // ---- the cellular layouts (one W2 for all of them)
    var wc: Cell2;
    wc.id = 0lu;
    wc.id2 = 0lu;
    wc.point = vec2<f32>(0.0);
    wc.point2 = vec2<f32>(0.0);
    wc.f1 = 0.0;
    wc.f2 = 0.0;
    var qw = q;
    if (w2cell > 0.0) {
        if (pattern == SP_ORGANIC) {
            qw = q + vec2<f32>(sin(q.y / (1.7 * w2cell) + 6.0 * u01k(did, 11lu)), sin(q.x / (1.9 * w2cell) + 6.0 * u01k(did, 12lu))) * (0.22 * w2cell);
        }
        wc = worley2(did ^ 0x0A6lu, qw, w2cell, 0.9);
        if (pattern == SP_ORGANIC) {
            let ed = worley2_edge_dist(wc, qw);
            let dv = st_normalize2(wc.point2 - wc.point);
            blk.ds = ed;
            blk.along = dot(qw, vec2<f32>(-dv.y, dv.x));
            blk.sid = wc.id ^ wc.id2;
            blk.id = wc.id;
            let rel = st_rot2(qw - wc.point, u01k(wc.id, 1lu) * (0.5 * ST_PI));
            let half = 0.45 * w2cell;
            blk.bw = 2.0 * (half - sw);
            blk.bd = blk.bw;
            blk.u = clamp(rel.x + half - sw, 0.0, blk.bw);
            blk.v = clamp(rel.y + half - sw, 0.0, blk.bd);
        }
    }
    o.ds = blk.ds;
    o.sw = blk.sw;
    o.along = blk.along;
    o.sid = blk.sid;
    o.major = blk.major;
    // ---- streets and arterials
    let edge_thr = 0.42 + 0.1 * (u01k(blk.id, 31lu) - 0.5);
    var street_ok = dens_here > edge_thr - select(0.0, 0.18, blk.major);
    if (blk.seg != 0lu && fm == SF_CITY) {
        let dsg = st_density(pl, st_rel0(pl, q + blk.smid) / (1.0 + e.n)) * dmul;
        street_ok = dsg > 0.42 + 0.1 * (u01k(blk.seg, 31lu) - 0.5) - select(0.0, 0.04, blk.major);
    }
    var art_hw = 3.5;
    if (pl.level == 2u) {
        art_hw = 5.0;
    } else if (pl.level == 3u) {
        art_hw = 6.0;
    } else if (pl.level >= 4u) {
        art_hw = 7.5;
    }
    let has_streets = !(pattern == SP_DISPERSED || pattern == SP_KRAAL);
    if (art < art_hw + 2.0 * fw && dens_here > 0.35) {
        let c = band_cov(art, art_hw, fw);
        if (c > 0.0) {
            o.ds = art;
            o.sw = art_hw;
            o.major = true;
            o.sid = did ^ 0xA87lu;
            o.street = c;
        }
    }
    if (has_streets && street_ok) {
        o.street = max(o.street, band_cov(blk.ds, blk.sw, fw));
    }
    let rd = st_ring_dist(pl, q);
    let ring_hw = art_hw + 2.0;
    if (rd < ring_hw + 2.0 * fw && dens_here > 0.35) {
        let c = band_cov(rd, ring_hw, fw);
        if (c > o.street) {
            o.ds = rd;
            o.sw = ring_hw;
            o.major = true;
            o.sid = pl.seed ^ 0x219lu;
            o.street = c;
        }
    }
    o.lamp_sp = 0.0;
    if (has_streets) {
        if (pattern == SP_LINEAR || fm != SF_CITY) {
            o.lamp_sp = 34.0;
        } else {
            o.lamp_sp = 18.0 + 6.0 * min(f32(z), 3.0);
        }
    }
    var walk = 0.0;
    if (has_streets && street_ok && pattern != SP_INFORMAL && pattern != SP_LINEAR) {
        walk = band_cov(blk.ds, blk.sw + 0.2 * blk.sw + 0.6, fw);
    }
    // ---- block content
    var cov_lot = 0.0;
    var col = st_asphalt();
    var h = 0.0;
    var cls = LC_URBAN;
    var lot_id = blk.id;
    let inside = blk.u >= 0.0 && blk.v >= 0.0 && blk.u < blk.bw && (blk.v < blk.bd || pattern == SP_LINEAR);
    let ublk = vec2<f32>(blk.u, blk.v);
    var roof = 0.0;
    var rim = 0.0;
    var win = vec3<f32>(1.0, 0.72, 0.42);
    var tree = 0.0;
    var tree_h = 0.0;
    var em = vec3<f32>(0.0);
    let yard = mixc(e.ground * (0.92 + 0.16 * u01k(blk.id, 19lu)), st_yard(e.moist, u01k(blk.id, 19lu)), 0.35);
    var rf_arch = pl.arch;
    var rf_u = 0.5;
    var rf_flat = false;
    var rf_mul = 1.0;
    var rf_ins = 0.0;
    var rf_shade = 1.0;
    var lm_ok = false;
    var lm_g = vec3<f32>(0.0);
    var lm_frac = 0.0;
    var lm_h = 0.0;
    var lot_scale = 13.0;
    if (pattern == SP_INDUSTRIAL) {
        lot_scale = 40.0;
    } else if (pattern == SP_SUPER) {
        lot_scale = 50.0;
    } else if (pattern == SP_INFORMAL || pattern == SP_KRAAL) {
        lot_scale = 6.0;
    }
    // special parcels: whole blocks
    var special = 0u;
    if (inside && fm == SF_CITY && pattern != SP_INFORMAL) {
        let us = u01k(blk.id, 20lu);
        var p_park = 0.05;
        var p_plaza = 0.06;
        var p_pk = 0.03;
        var p_sp = 0.0;
        var p_ce = 0.0;
        if (z == SZ_INNER) {
            p_park = 0.07;
            p_plaza = 0.03;
            p_pk = 0.02;
            p_sp = 0.01;
            p_ce = 0.01;
        } else if (z == SZ_OUTER) {
            p_park = 0.07;
            p_plaza = 0.0;
            p_pk = 0.01;
            p_sp = 0.03;
            p_ce = 0.02;
        } else if (z == SZ_SUBURB) {
            p_park = 0.05;
            p_plaza = 0.0;
            p_pk = 0.0;
            p_sp = 0.04;
            p_ce = 0.02;
        } else if (z == SZ_PERI) {
            p_park = 0.02;
            p_plaza = 0.0;
            p_pk = 0.0;
            p_sp = 0.02;
            p_ce = 0.02;
        }
        let p_park2 = p_park + p_plaza;
        let p_parking = p_park2 + p_pk;
        let p_sports = p_parking + p_sp;
        let p_cem = p_sports + p_ce;
        var p_stad = p_cem;
        if (pl.level >= 3u && min(blk.bw, blk.bd) > 130.0 && (z == SZ_OUTER || z == SZ_SUBURB)) {
            p_stad = p_cem + 0.05;
        }
        let p_site = p_stad + 0.01;
        if (pattern == SP_INDUSTRIAL) {
            special = select(0u, 3u, us < 0.03);
        } else if (us < p_park) {
            special = 1u;
        } else if (us < p_park2) {
            special = 2u;
        } else if (us < p_parking) {
            special = 3u;
        } else if (us < p_sports) {
            special = 4u;
        } else if (us < p_cem) {
            special = 5u;
        } else if (us < p_stad) {
            special = 6u;
        } else if (us < p_site) {
            special = 7u;
        }
        let bc = vec2<f32>(0.5 * blk.bw - blk.u, 0.5 * blk.bd - blk.v);
        let dbc = st_dens_at(pl, q, st_rot2(bc, -theta), e.n, dmul);
        if (dbc < 0.3) {
            special = 0u;
        }
    }
    // (the old quarter's small blocks, `town::place_eval`)
    var os: StOldSub;
    var old_in = false;
    if (oldg && inside && special == 0u) {
        os = st_old_sub(blk);
        old_in = u01k(os.id, 77lu) < st_old_core(pl, q + st_rot2(os.c - ublk, -theta));
    }
    if (inside && special != 0u) {
        cov_lot = 1.0;
        let u = blk.u;
        let v = blk.v;
        let cc = vec2<f32>(0.5 * blk.bw, 0.5 * blk.bd);
        let r = ublk - cc;
        if (special == 1u) {
            col = srgb(80.0, 116.0, 52.0) * (1.0 + 0.15 * e.detail);
            cls = LC_PARK;
            let dpath = abs(r.x * blk.bd - r.y * blk.bw) / length(vec2<f32>(blk.bw, blk.bd));
            let path = band_cov(dpath, 1.6, fw);
            let tc = floor(ublk / 13.0);
            let th = hash2(blk.id ^ 0x7EElu, i64(tc.x), i64(tc.y));
            let tp = (tc + 0.5 + 0.5 * (vec2<f32>(u01k(th, 1lu), u01k(th, 2lu)) - 0.5)) * 13.0;
            let tr = 3.5 + 2.5 * u01k(th, 3lu);
            if (u01k(th, 4lu) < 0.55 && path < 0.5) {
                tree = st_aa(tr - length(ublk - tp), fw);
                tree_h = 9.0 + 8.0 * u01k(th, 5lu);
            }
            if (u01k(blk.id, 21lu) < 0.3) {
                let pr = 0.22 * min(blk.bw, blk.bd);
                let pond = st_aa(pr - length(r - vec2<f32>(0.1, -0.15) * blk.bw), fw);
                col = mixc(col, srgb(52.0, 78.0, 84.0), pond);
                tree *= 1.0 - pond;
            }
            col = mixc(col, srgb(170.0, 156.0, 130.0), path);
        } else if (special == 2u) {
            col = st_paving() * (0.9 + 0.1 * u01k(blk.id, 22lu));
            let grid = band_cov((u / 6.0 - round(u / 6.0)) * 6.0, 0.15, fw) + band_cov((v / 6.0 - round(v / 6.0)) * 6.0, 0.15, fw);
            col *= 1.0 - 0.08 * min(grid, 1.0) * band(6.0, e.gsd);
            cls = LC_PARKING;
            em += vec3<f32>(1.0, 0.85, 0.65) * 0.02 * band(20.0, e.gsd);
            let tq = round(vec2<f32>(u, v) / 10.0) * 10.0;
            let de = min(min(tq.x, tq.y), min(blk.bw - tq.x, blk.bd - tq.y));
            if (de >= 4.0 && de <= 11.0) {
                tree = st_aa(2.4 - length(vec2<f32>(u, v) - tq), fw);
                tree_h = 8.0 + 3.0 * u01k(hash2(blk.id ^ 0x9Alu, i64(tq.x), i64(tq.y)), 1lu);
            }
        } else if (special == 3u) {
            col = st_car_park(blk.id, u, v, fw, e.gsd);
            cls = LC_PARKING;
            em += vec3<f32>(1.0, 0.6, 0.25) * 0.03 * band(30.0, e.gsd);
            let tq = vec2<f32>(round(u / 9.0) * 9.0, round(v / 16.0) * 16.0);
            let th = hash2(blk.id ^ 0x9Blu, i64(tq.x), i64(tq.y));
            if (u01k(th, 1lu) < 0.7) {
                tree = st_aa(2.6 - length(vec2<f32>(u, v) - tq), fw);
                tree_h = 7.0 + 3.0 * u01k(th, 2lu);
            }
        } else if (special == 4u) {
            let ph = min(vec2<f32>(34.0, 52.0), vec2<f32>(0.5 * blk.bw - 12.0, 0.5 * blk.bd - 12.0));
            let rr = select(r, vec2<f32>(r.y, r.x), blk.bw > blk.bd);
            let dp = st_boxd(rr, vec2<f32>(0.0), ph);
            let pitch = st_aa(st_min2(dp), fw);
            let core = vec2<f32>(0.0, max(abs(rr.y) - (ph.y - ph.x), 0.0));
            let dtr = length(vec2<f32>(rr.x, core.y)) - ph.x - 3.0;
            let track = band_cov(dtr - 4.5, 4.5, fw);
            let stripe = f32(st_rem(i64(floor(rr.y / 6.0)), 2li));
            col = mixc(srgb(118.0, 140.0, 84.0), srgb(84.0, 128.0, 58.0) * (0.95 + 0.07 * stripe * band(6.0, e.gsd)), pitch);
            col = mixc(col, srgb(170.0, 82.0, 62.0), track);
            cls = LC_SPORTS_FIELD;
        } else if (special == 5u) {
            col = srgb(100.0, 120.0, 76.0);
            cls = LC_CEMETERY;
            let gu = u / 2.4 - floor(u / 2.4);
            let gv = v / 3.2 - floor(v / 3.2);
            let stone = select(0.0, 1.0, abs(gu - 0.5) < 0.22 && abs(gv - 0.5) < 0.3);
            let path = max(band_cov(r.x, 1.5, fw), band_cov(r.y, 1.5, fw));
            col = mixc(col, srgb(176.0, 172.0, 164.0), stone * band(2.4, e.gsd) * (1.0 - path) + 0.25 * (1.0 - band(2.4, e.gsd)));
            col = mixc(col, srgb(160.0, 150.0, 130.0), path);
            let tc = floor(ublk / 18.0);
            let th = hash2(blk.id ^ 0xCE7lu, i64(tc.x), i64(tc.y));
            if (u01k(th, 1lu) < 0.4) {
                let tp = (tc + 0.5) * 18.0;
                tree = st_aa(3.0 - length(ublk - tp), fw);
                tree_h = 8.0;
            }
        } else if (special == 6u) {
            let a = 0.45 * min(min(blk.bw, blk.bd), 220.0);
            let ab = vec2<f32>(a, 0.78 * a);
            let rr = select(vec2<f32>(r.y, r.x), r, blk.bw > blk.bd);
            let rn = length(rr / ab);
            col = st_concrete() * 0.95;
            cls = LC_SPORTS_FIELD;
            if (rn < 1.0) {
                if (rn < 0.62) {
                    let stripe = f32(st_rem(i64(floor(rr.x / 5.0)), 2li));
                    col = srgb(70.0, 124.0, 52.0) * (0.94 + 0.08 * stripe * band(5.0, e.gsd));
                    let pl2 = st_min2(st_boxd(rr, vec2<f32>(0.0), vec2<f32>(0.5 * a, 0.36 * a)));
                    col = mixc(col, srgb(220.0, 220.0, 220.0), band_cov(pl2, 0.12, fw) * band(1.0, e.gsd));
                } else {
                    h = 8.0 + 22.0 * smoothstep1(0.62, 0.98, rn);
                    col = mixc(srgb(190.0, 190.0, 196.0), srgb(150.0, 40.0, 40.0), 0.4 * f32(st_rem(i64(floor(atan2(rr.y, rr.x) * 12.0)), 2li)) * band(30.0, e.gsd));
                    roof = 1.0;
                    cls = LC_BUILDING;
                }
            }
            let fl = abs(rr) - ab * 0.9;
            em += vec3<f32>(0.95, 0.97, 1.0) * point_light(dot(fl, fl), 30.0, 1.2, fw);
        } else {
            col = srgb(150.0, 126.0, 98.0) * (0.9 + 0.2 * e.detail);
            cls = LC_BARE;
        }
    } else if (inside) {
        if (pattern == SP_SUPER) {
            let long_x = blk.bw >= blk.bd;
            let a = select(blk.v, blk.u, long_x);
            let c = select(blk.u, blk.v, long_x);
            let lc = select(blk.bw, blk.bd, long_x);
            let pitch = 52.0 + 12.0 * u01k(blk.id, 23lu);
            let ri = floor(c / pitch);
            let cr = c - (ri + 0.5) * pitch;
            let nrow = floor(lc / pitch);
            let rh = hash2(blk.id ^ 0x5ABlu, i64(ri), 0li);
            let seg = 70.0 + 50.0 * u01k(rh, 1lu);
            let si = floor(a / seg);
            let ar = a - (si + 0.5) * seg;
            let sh = hash2(rh, i64(si), 1li);
            lot_id = sh;
            let dbc = st_dens_at(pl, q, vec2<f32>(0.0), e.n, dmul);
            cov_lot = 1.0;
            col = mixc(yard, st_asphalt() * 1.15, 0.3 * u01k(sh, 4lu));
            cls = LC_RESIDENTIAL;
            lm_ok = true;
            lm_g = col;
            lm_frac = 0.3;
            lm_h = 6.0;
            rf_arch = pl.arch;
            rf_u = u01k(sh, 6lu);
            rf_flat = true;
            rf_mul = 1.0;
            if (ri < nrow && u01k(sh, 2lu) < 0.85 && dbc > 0.25) {
                let point = u01k(sh, 3lu) < 0.25;
                let half = select(vec2<f32>(0.5 * seg - 6.0, 6.5), vec2<f32>(11.0, 11.0), point);
                let dd = st_boxd(vec2<f32>(ar, cr), vec2<f32>(0.0), half);
                let ins = st_aa(st_min2(dd), fw);
                if (ins > 0.0) {
                    let floors = select(5.0 + 7.0 * u01k(sh, 5lu), 14.0 + 12.0 * u01k(sh, 5lu), point) + select(0.0, 8.0, pl.arch == 8u);
                    h = floors * 3.0 * ins;
                    rf_ins = ins;
                    roof = ins;
                    rim = (1.0 - smoothstep1(0.0, 0.9, st_min2(dd))) * ins;
                    if (ins > 0.5) {
                        cls = select(LC_BUILDING, LC_BUILDING_TALL, floors * 3.0 > 30.0);
                    }
                }
            } else if (dbc <= 0.25) {
                cov_lot = 0.0;
            }
            let tcell = floor(vec2<f32>(a, c) / 15.0);
            let th = hash2(blk.id ^ 0x5AClu, i64(tcell.x), i64(tcell.y));
            if (roof < 0.5 && u01k(th, 1lu) < 0.3 && cov_lot > 0.0) {
                let tp = (tcell + 0.5) * 15.0;
                tree = st_aa(3.5 - length(vec2<f32>(a, c) - tp), fw) * (1.0 - roof);
                tree_h = 10.0;
            }
        } else if (pattern == SP_INFORMAL) {
            let dbc = st_dens_at(pl, q, wc.point - q, e.n, dmul);
            if (u01k(wc.id, 7lu) < pow(dbc, 0.8) * 1.15) {
                lot_id = wc.id;
                cov_lot = 1.0;
                col = srgb(128.0, 110.0, 90.0);
                cls = LC_RESIDENTIAL;
                lm_ok = true;
                lm_g = col;
                lm_frac = 0.7;
                lm_h = 3.5 * (0.25 + 0.75 * dbc);
                rf_arch = 22u;
                rf_u = u01k(wc.id, 4lu);
                rf_flat = false;
                rf_mul = 0.85 + 0.3 * u01k(wc.id, 5lu);
                let ed = worley2_edge_dist(wc, qw);
                let ins = st_aa(ed - 0.6, fw);
                if (u01k(wc.id, 3lu) < 0.25 + 0.75 * dbc) {
                    rf_ins = ins;
                    h = (3.0 + 3.5 * u01k(wc.id, 6lu)) * ins;
                    roof = ins;
                    if (ins > 0.5) {
                        cls = LC_BUILDING;
                    }
                }
            }
        } else if (pattern == SP_LINEAR) {
            let lc = vec2<f32>(0.5 * blk.bw - blk.u, -q.y);
            let dl = st_dens_at(pl, q, lc, e.n, dmul);
            let farm = u01k(blk.id, 3lu) < pow(dl, 0.6) * 1.1;
            if (farm && blk.v < blk.bd) {
                cov_lot = 1.0;
                col = mixc(yard, srgb(150.0, 138.0, 118.0), 0.4);
                cls = LC_FARMYARD;
                lm_ok = true;
                lm_g = col;
                lm_frac = 0.25;
                lm_h = 1.6;
                rf_arch = pl.arch;
                rf_u = u01k(blk.id, 5lu);
                rf_flat = false;
                rf_mul = 1.0;
                let hh = vec2<f32>(min(0.5 * blk.bw - 2.5, 6.5), 5.5);
                let hc = vec2<f32>(0.5 * blk.bw, 4.0 + 2.0 * u01k(blk.id, 4lu) + hh.y);
                let bh = vec2<f32>(min(0.5 * blk.bw - 2.0, 8.0), 6.0);
                let bc = vec2<f32>(0.5 * blk.bw, blk.bd - 4.0 - bh.y);
                let dh = st_boxd(ublk, hc, hh);
                let db = st_boxd(ublk, bc, bh);
                let ih = st_aa(st_min2(dh), fw);
                let ib = st_aa(st_min2(db), fw);
                if (ih > 0.0) {
                    let ridge = clamp(dh.y / hh.y, 0.0, 1.0);
                    rf_ins = ih;
                    rf_shade = 0.9 + 0.2 * ridge;
                    h = (5.5 + 2.5 * ridge) * ih;
                    roof = ih;
                    rim = (1.0 - smoothstep1(0.0, 0.9, st_min2(dh))) * ih;
                    if (ih > 0.5) {
                        cls = LC_BUILDING;
                    }
                } else if (ib > 0.0) {
                    let ridge = clamp(db.y / bh.y, 0.0, 1.0);
                    rf_u = u01k(blk.id, 6lu);
                    rf_mul = 0.85;
                    rf_ins = ib;
                    rf_shade = 0.9 + 0.15 * ridge;
                    h = (6.0 + 2.0 * ridge) * ib;
                    roof = ib;
                    if (ib > 0.5) {
                        cls = LC_BUILDING;
                    }
                }
                if (ih <= 0.0 && ib <= 0.0 && u01k(blk.id, 7lu) < 0.6) {
                    let tp = vec2<f32>(0.25 * blk.bw, 0.55 * blk.bd);
                    tree = st_aa(3.5 - length(ublk - tp), fw);
                    tree_h = 9.0;
                }
                let lp = ublk - vec2<f32>(hc.x, hc.y - hh.y - 1.0);
                em += vec3<f32>(1.0, 0.72, 0.42) * point_light(dot(lp, lp), 2.0, 0.4, fw);
            } else if (farm && fm == SF_WALDHUFEN) {
                let len = 500.0 + 900.0 * pow(u01k(blk.id, 8lu), 0.7);
                let sv = blk.v - blk.bd;
                if (sv < len) {
                    cov_lot = 1.0;
                    let hedge = band_cov(min(blk.u, blk.bw - blk.u), 1.6, fw);
                    let seg = floor(sv / (120.0 + 80.0 * u01k(blk.id, 9lu)));
                    let fh = hash2(blk.id ^ 0xF1Elu, i64(seg), 0li);
                    let ck = u32(u01k(fh, 1lu) * 3.99);
                    var crop = srgb(96.0, 124.0, 60.0);
                    if (ck == 0u) {
                        crop = srgb(168.0, 150.0, 92.0);
                    } else if (ck == 1u) {
                        crop = srgb(110.0, 132.0, 66.0);
                    } else if (ck == 2u) {
                        crop = srgb(132.0, 110.0, 80.0);
                    }
                    col = mixc(crop * (0.94 + 0.12 * e.detail), srgb(52.0, 74.0, 40.0), hedge);
                    cls = select(LC_CROP, LC_HEDGEROW, hedge > 0.5);
                    tree = hedge * smoothstep1(0.3, 0.6, u01k(hash2(blk.id, i64(floor(sv / 9.0)), 2li), 1lu));
                    tree_h = 8.0;
                }
            }
        } else if (pattern != SP_DISPERSED && pattern != SP_KRAAL && old_in) {
            // the old core (`town::place_eval`)
            let sh = os.id;
            let dsub = os.d;
            let depthb = 8.0 + 6.0 * u01k(sh, 2lu);
            let uw = 6.0 + 5.0 * u01k(sh, 12lu);
            let hu = hash3(sh ^ 0x40Flu, os.side, i64(floor(os.along / uw)), 1li);
            lot_id = hu;
            cov_lot = 1.0;
            col = mixc(mixc(yard, st_paving(), 0.3), srgb(70.0, 96.0, 52.0), 0.35 + 0.3 * clamp(e.moist, 0.0, 1.0));
            cls = select(LC_RESIDENTIAL, LC_COMMERCIAL, z == SZ_CBD);
            let square = u01k(sh, 1lu) < 0.07;
            let hb0 = 3.0 * select(3.0 + 2.0 * u01k(sh, 3lu), 4.0 + 3.0 * u01k(sh, 3lu), pl.level >= 3u) * select(1.0, 1.25, z == SZ_CBD);
            let flat = u01k(sh, 6lu) < 0.06;
            let frac = select(0.7, 0.0, square);
            lm_ok = true;
            lm_g = col;
            lm_frac = frac;
            lm_h = hb0 * frac;
            let hb = hb0 + 3.0 * (floor(u01k(hu, 1lu) * 2.99) - 1.0) * 0.8;
            col = mixc(col, st_paving() * 0.9, max(st_aa(-dsub, fw), select(0.0, 1.0, square)));
            rf_arch = pl.arch;
            rf_u = 0.2 + 0.6 * u01k(hu, 9lu);
            rf_flat = flat;
            rf_mul = 0.86 + 0.28 * u01k(hu, 11lu);
            let dr = min(dsub, depthb - dsub);
            let ins = select(st_aa(dr, fw), 0.0, square);
            if (ins > 0.0) {
                var hh = hb;
                if (flat) {
                    rf_shade = 1.0 - 0.15 * band_cov(dr, 0.6, fw);
                } else {
                    hh = hb + 0.5 * dr;
                }
                rf_ins = ins;
                h = hh * ins;
                roof = ins;
                rim = (1.0 - smoothstep1(0.0, 0.9, dr)) * ins;
                if (ins > 0.5) {
                    cls = LC_BUILDING;
                }
            } else if (dsub > depthb || square) {
                let tc = floor(qw / 10.0);
                let th = hash2(sh ^ 0xC07lu, i64(tc.x), i64(tc.y));
                if (u01k(th, 1lu) < 0.35) {
                    let tp = (tc + 0.5 + 0.4 * (vec2<f32>(u01k(th, 2lu), u01k(th, 3lu)) - 0.5)) * 10.0;
                    tree = st_aa(2.6 + 1.2 * u01k(th, 4lu) - length(qw - tp), fw);
                    tree_h = 8.0 + 5.0 * u01k(th, 5lu);
                }
            }
        } else if (pattern != SP_DISPERSED && pattern != SP_KRAAL) {
            // lots by BSP and buildings by typology
            let core = st_town_core(pl, z, pattern);
            var depth = 3u;
            if (core) {
                depth = 4u;
            } else if (z == SZ_CBD) {
                depth = 1u;
            } else if (z == SZ_INNER) {
                depth = 2u;
            }
            var lind = false;
            if (pattern == SP_GRID && d.nbind < 260.0) {
                let qc = q + st_rot2(vec2<f32>(0.5 * blk.bw - blk.u, 0.5 * blk.bd - blk.v), -theta);
                lind = u01k(blk.id, 79lu) < 0.6 * (1.0 - smoothstep1(30.0, 230.0, st_nbind_at(pl, qc)));
            }
            let ind = pattern == SP_INDUSTRIAL || lind;
            if (ind) {
                depth = 3u;
            }
            var min_side = 9.0;
            if (lind) {
                min_side = 22.0;
            } else if (ind) {
                min_side = 30.0;
            } else if (z == SZ_CBD) {
                min_side = 22.0;
            }
            let lot = st_bsp(blk.id, blk.u, blk.v, blk.bw, blk.bd, depth, min_side);
            let lh = lot.h;
            lot_id = lh;
            let lr = lot.r;
            let lc = vec2<f32>(0.5 * (lr.x + lr.z), 0.5 * (lr.y + lr.w));
            let lot_dq = st_rot2(lc - ublk, -theta);
            let dl = st_dens_at(pl, q, lot_dq, e.n, dmul);
            var edge = 1.0;
            var occ = mix(pow(dl, 0.7) * 1.05, pow(dl, 0.35) * 1.1, smoothstep1(0.35, 0.5, dl));
            if (fm == SF_CITY) {
                let on_major = ((blk.mj & 1u) != 0u && lr.x <= 0.0) || ((blk.mj & 2u) != 0u && lr.z >= blk.bw) || ((blk.mj & 4u) != 0u && lr.y <= 0.0) || ((blk.mj & 8u) != 0u && lr.w >= blk.bd);
                let thr = edge_thr - 0.16 * u01k(lh, 33lu) - select(0.0, 0.16, on_major);
                edge = max(smoothstep1(thr, thr + 0.1, dl), 0.22 * smoothstep1(thr - 0.14, thr - 0.04, dl));
                occ = max(occ, 0.88 * smoothstep1(0.32, 0.42, dl));
                if (on_major) {
                    occ = max(occ, 0.85 * smoothstep1(thr, thr + 0.06, dl));
                }
            }
            let built = u01k(lh, 3lu) < occ * edge;
            let a = pl.arch;
            let ut = u01k(lh, 12lu);
            let euro = a == 3u || a == 5u || a == 6u || a == 8u || a == 9u || a == 11u || a == 4u;
            var t = STY_HOUSE;
            if (ind) {
                t = STY_WAREHOUSE;
            } else if (fm != SF_CITY) {
                if (z == SZ_INNER && euro) {
                    t = STY_PERIM;
                } else if (a == 3u || (euro && ut < 0.6)) {
                    t = STY_ROW;
                }
            } else if (z == SZ_CBD) {
                t = select(STY_MIDRISE, STY_TOWER, pl.level >= 3u && ut < 0.75 * (1.0 - 0.5 * smoothstep1(0.0, 0.14, rel_here)));
            } else if (z == SZ_INNER && pl.level >= 3u && u01k(lh, 21lu) < 0.12 * (1.0 - smoothstep1(0.1, 0.2, rel_here))) {
                t = STY_TOWER;
            } else if (z == SZ_INNER) {
                let big = pl.level >= 3u;
                if ((euro && pattern != SP_RADIAL && ut < select(0.25, 0.8, big)) || pattern == SP_ORGANIC) {
                    t = STY_PERIM;
                } else if (ut < select(0.75, 0.6, big)) {
                    t = STY_MIDRISE;
                } else {
                    t = STY_ROW;
                }
            } else if (z == SZ_OUTER) {
                if (ut < 0.45) {
                    t = STY_MIDRISE;
                } else if (ut < 0.75) {
                    t = STY_ROW;
                }
            } else if ((a == 0u || a == 1u || a == 8u || a == 3u) && ut < 0.6) {
                t = STY_ROW;
            }
            let town_box = t == STY_MIDRISE && core;
            var wuse = 0u;
            if (t == STY_WAREHOUSE) {
                let lu = u01k(lh, 40lu);
                if (lu >= 0.88) {
                    wuse = 3u;
                } else if (lu >= 0.74) {
                    wuse = 2u;
                } else if (lu >= 0.6) {
                    wuse = 1u;
                }
            }
            let wside = u32(u01k(lh, 41lu) * 3.99);
            let w_so = 5.0 + 3.0 * u01k(lh, 42lu);
            let w_sf = 14.0 + 12.0 * u01k(lh, 43lu);
            if (built || dl > 0.65) {
                cov_lot = 1.0;
                if (t == STY_WAREHOUSE) {
                    col = mixc(st_concrete(), st_asphalt(), 0.35 + 0.45 * u01k(lh, 10lu));
                    cls = LC_INDUSTRIAL;
                } else if (t == STY_TOWER || t == STY_MIDRISE) {
                    col = mixc(st_paving(), yard, 0.25);
                    if (town_box) {
                        col = mixc(mixc(yard, srgb(86.0, 112.0, 54.0), 0.7), st_paving(), 0.15);
                    }
                    cls = select(LC_RESIDENTIAL, LC_COMMERCIAL, t == STY_TOWER || z == SZ_CBD);
                } else if (t == STY_PERIM) {
                    col = mixc(mixc(yard, st_paving(), 0.35), srgb(84.0, 112.0, 60.0), 0.25 + 0.35 * clamp(e.moist, 0.0, 1.0));
                    cls = LC_RESIDENTIAL;
                } else {
                    col = yard;
                    cls = LC_RESIDENTIAL;
                }
            }
            let lw = lr.z - lr.x;
            let ld = lr.w - lr.y;
            if (built && wuse == 0u) {
                let rl = ublk - lc;
                var cen = vec2<f32>(0.0);
                var half = vec2<f32>(0.0);
                var hb = 0.0;
                var flat = true;
                if (t == STY_TOWER) {
                    let s2 = clamp(0.5 * min(lw, ld) - 4.0, 6.0, 24.0);
                    let ps = smoothstep1(1e5, 1.5e7, pl.pop);
                    let u4 = u01k(lh, 4lu);
                    half = vec2<f32>(s2);
                    hb = (40.0 + 360.0 * ps * (u4 * u4 * u4) + 40.0 * u01k(lh, 5lu)) * (1.0 - 0.6 * smoothstep1(0.0, 0.2, rel_here));
                } else if (t == STY_MIDRISE && town_box) {
                    let sb = 1.5 + 2.5 * u01k(lh, 1lu);
                    let u5 = u01k(lh, 5lu);
                    half = vec2<f32>(min(0.5 * lw - sb, 13.0), min(0.5 * ld - sb, 11.0));
                    hb = 3.5 + 4.5 * u01k(lh, 4lu) + 16.0 * u5 * u5 * u5 * u5;
                    flat = u01k(lh, 6lu) < 0.35;
                } else if (t == STY_MIDRISE) {
                    var fill = vec2<f32>(40.0, 14.0 + 10.0 * u01k(lh, 6lu));
                    if (z <= SZ_INNER) {
                        fill = vec2<f32>(60.0, 60.0);
                    }
                    half = min(vec2<f32>(0.5 * lw - 2.5, 0.5 * ld - 2.5), fill);
                    hb = 3.0 * (4.0 + 4.0 * u01k(lh, 4lu)) + select(0.0, 12.0, z == SZ_CBD);
                } else if (t == STY_WAREHOUSE) {
                    let sb = vec4<f32>(select(w_so, w_sf, wside == 0u), select(w_so, w_sf, wside == 1u), select(w_so, w_sf, wside == 2u), select(w_so, w_sf, wside == 3u));
                    let hx = 0.5 * (lw - sb.x - sb.y);
                    let hy = 0.5 * (ld - sb.z - sb.w);
                    cen = vec2<f32>(0.5 * (sb.x - sb.y), 0.5 * (sb.z - sb.w));
                    half = select(vec2<f32>(min(hx, 50.0), min(hy, 80.0)), vec2<f32>(min(hx, 80.0), min(hy, 50.0)), hx > hy);
                    hb = 8.0 + 6.0 * u01k(lh, 4lu);
                    flat = u01k(lh, 6lu) < 0.5;
                } else if (t == STY_ROW) {
                    half = vec2<f32>(0.5 * lw - 0.4, min(0.5 * ld - 2.0, 7.5));
                    hb = 3.0 * (2.0 + floor(1.6 * u01k(lh, 4lu)));
                    flat = u01k(lh, 6lu) < 0.3 || a == 3u;
                } else if (t == STY_PERIM) {
                    half = vec2<f32>(1e4);
                    hb = 3.0 * (3.0 + 3.0 * u01k(lh, 4lu)) * select(0.7, 1.0, pl.level >= 3u);
                    flat = u01k(lh, 6lu) < 0.35;
                } else {
                    let sb = 2.0 + 2.5 * u01k(lh, 1lu);
                    half = vec2<f32>(min(0.5 * lw - sb, 8.0 + 4.0 * u01k(lh, 7lu)), min(0.5 * ld - sb, 6.5 + 4.0 * u01k(lh, 8lu)));
                    hb = 3.5 + 4.0 * u01k(lh, 4lu);
                    flat = a == 3u || u01k(lh, 6lu) < 0.12;
                }
                var podium = 0.0;
                var dd: vec2<f32>;
                var ins: f32;
                if (t == STY_PERIM) {
                    let dblk = min(min(blk.u, blk.v), min(blk.bw - blk.u, blk.bd - blk.v));
                    let depthb = 11.0 + 3.0 * u01k(blk.id, 24lu);
                    dd = vec2<f32>(depthb - dblk);
                    ins = st_aa(dd.x, fw);
                } else {
                    dd = st_boxd(rl, cen, half);
                    if (t == STY_TOWER) {
                        podium = st_aa(st_min2(st_boxd(rl, vec2<f32>(0.0), vec2<f32>(0.5 * lw - 3.0, 0.5 * ld - 3.0))), fw);
                    }
                    ins = st_aa(st_min2(dd), fw);
                }
                if (st_min2(half) < 2.0) {
                    ins = 0.0;
                }
                rf_arch = a;
                if (t == STY_TOWER) {
                    rf_arch = 20u;
                } else if (t == STY_WAREHOUSE) {
                    rf_arch = 21u;
                }
                rf_u = u01k(lh, 9lu);
                rf_flat = flat;
                rf_mul = 0.84 + 0.32 * u01k(lh, 11lu);
                var frac = 0.55;
                if (t == STY_HOUSE) {
                    frac = 0.3;
                } else if (town_box) {
                    frac = 0.45;
                } else if (t == STY_TOWER) {
                    frac = 0.75;
                } else if (t == STY_WAREHOUSE) {
                    frac = 0.6;
                }
                lm_ok = true;
                lm_g = col;
                lm_frac = frac;
                lm_h = hb * frac;
                if (ins > 0.0) {
                    var hh = hb;
                    if (!flat) {
                        var dperp = dd.x;
                        if (t != STY_PERIM && half.x > half.y) {
                            dperp = dd.y;
                        }
                        var hw = min(half.x, half.y);
                        if (t == STY_PERIM) {
                            hw = 0.5 * (11.0 + 3.0 * u01k(blk.id, 24lu));
                        }
                        hh = hb + select(0.35, 0.1, t == STY_WAREHOUSE) * hw * clamp(dperp / hw, 0.0, 1.0);
                    } else {
                        rf_shade = 1.0 - 0.15 * band_cov(st_min2(dd), 0.6, fw);
                    }
                    rf_ins = ins;
                    h = hh * ins;
                    roof = ins;
                    rim = (1.0 - smoothstep1(0.0, 0.9, st_min2(dd))) * ins;
                    if (ins > 0.5) {
                        cls = select(LC_BUILDING, LC_BUILDING_TALL, hh > 30.0);
                    }
                    if (t == STY_TOWER || (t == STY_MIDRISE && z == SZ_CBD)) {
                        win = vec3<f32>(0.82, 0.90, 1.0);
                    }
                    if (t == STY_WAREHOUSE) {
                        let rb = rl - cen;
                        let a_l = select(rb.y, rb.x, half.x > half.y);
                        let e_in = st_min2(dd);
                        var det = 0.0;
                        if (u01k(lh, 44lu) < 0.6) {
                            let ps = 12.0 + 6.0 * u01k(lh, 45lu);
                            let f = a_l / ps - round(a_l / ps);
                            det = 0.2 * band_cov(f * ps, 0.9, fw) * smoothstep1(2.0, 3.0, e_in) * band(4.0, e.gsd);
                        }
                        let cu = floor(rb / 16.0);
                        let ch = hash2(lh ^ 0xAClu, i64(cu.x), i64(cu.y));
                        if (u01k(ch, 1lu) < 0.35) {
                            let cp = (cu + 0.5 + 0.5 * (vec2<f32>(u01k(ch, 2lu), u01k(ch, 3lu)) - 0.5)) * 16.0;
                            let unit = st_aa(st_min2(st_boxd(rb, cp, vec2<f32>(1.2 + 1.3 * u01k(ch, 4lu)))), fw) * smoothstep1(3.0, 4.0, e_in);
                            det = max(det, 0.3 * unit * band(5.0, e.gsd));
                            h += 1.5 * unit * ins;
                        }
                        rf_shade *= 1.0 - det;
                    }
                } else if (podium > 0.0) {
                    rf_arch = a;
                    rf_u = u01k(lh, 19lu);
                    rf_flat = true;
                    rf_mul = 1.0;
                    rf_ins = podium;
                    h = (12.0 + 8.0 * u01k(lh, 20lu)) * podium;
                    roof = podium;
                    if (podium > 0.5) {
                        cls = LC_BUILDING;
                    }
                } else if (t == STY_HOUSE) {
                    if (pl.dev > 0.55 && (a == 5u || a == 7u || a == 3u || a == 0u || a == 1u) && u01k(lh, 13lu) < 0.35) {
                        let pc = vec2<f32>(0.0, half.y + 0.5 * (0.5 * ld - half.y));
                        let dp = st_boxd(rl, pc, vec2<f32>(2.2, 4.0));
                        let pool = st_aa(st_min2(dp), fw);
                        col = mixc(col, srgb(70.0, 170.0, 190.0), pool);
                    }
                    if (u01k(lh, 14lu) < 0.2) {
                        let tp = vec2<f32>(0.6 * (0.5 * lw) * (u01k(lh, 15lu) - 0.5) * 2.0, -(half.y + 0.5 * (0.5 * ld - half.y)));
                        tree = st_aa(3.0 + 1.5 * u01k(lh, 16lu) - length(rl - tp), fw);
                        tree_h = 8.0 + 5.0 * u01k(lh, 17lu);
                    }
                    let lp = rl - vec2<f32>(0.0, half.y + 1.0);
                    em += vec3<f32>(1.0, 0.72, 0.42) * point_light(dot(lp, lp), 2.0, 0.4, fw) * select(0.0, 1.0, u01k(lh, 18lu) < 0.6);
                } else if (t == STY_PERIM) {
                    let tc = floor(ublk / 11.0);
                    let th = hash2(blk.id ^ 0xC07lu, i64(tc.x), i64(tc.y));
                    if (u01k(th, 1lu) < 0.25) {
                        let tp = (tc + 0.5 + 0.4 * (vec2<f32>(u01k(th, 2lu), u01k(th, 3lu)) - 0.5)) * 11.0;
                        tree = st_aa(2.8 + 1.5 * u01k(th, 4lu) - length(ublk - tp), fw);
                        tree_h = 9.0 + 6.0 * u01k(th, 5lu);
                    }
                } else if (t == STY_WAREHOUSE) {
                    let rb = rl - cen;
                    var nn = rb.y;
                    var t2 = rb.x;
                    var hf = half.y;
                    var ht = half.x;
                    if (wside == 0u) {
                        nn = -rb.x;
                        t2 = rb.y;
                        hf = half.x;
                        ht = half.y;
                    } else if (wside == 1u) {
                        nn = rb.x;
                        t2 = rb.y;
                        hf = half.x;
                        ht = half.y;
                    } else if (wside == 2u) {
                        nn = -rb.y;
                    }
                    let dn = nn - hf;
                    if (dn > 0.0 && dn < 16.0 && abs(t2) < ht - 3.0) {
                        let k = floor(t2 / 4.2);
                        let th = hash2(lh ^ 0x7Clu, i64(k), 0li);
                        if (u01k(th, 1lu) < 0.45) {
                            let c = abs(t2 - (k + 0.5) * 4.2);
                            let tr = st_aa(1.3 - c, fw) * st_aa(dn - 1.0, fw) * st_aa(15.0 - dn, fw) * band(3.0, e.gsd);
                            col = mixc(col, select(srgb(150.0, 152.0, 156.0), srgb(214.0, 214.0, 208.0), u01k(th, 2lu) < 0.6), tr);
                            h = max(h, 4.0 * tr);
                        }
                    }
                    if (u01k(lh, 13lu) < 0.3 && (wside == 1u || wside == 3u)) {
                        let tr = min(6.0 + 9.0 * u01k(lh, 14lu), 0.5 * w_sf - 2.5);
                        let tc = vec2<f32>(0.5 * lw - tr - 3.0, 0.5 * ld - tr - 3.0);
                        let dt = length(rl - tc);
                        let tank = st_aa(tr - dt, fw);
                        col = mixc(col, srgb(206.0, 206.0, 200.0) * (0.85 + 0.15 * (1.0 - dt / tr)), tank);
                        h = max(h, 10.0 * tank);
                        roof = max(roof, tank);
                    }
                    em += vec3<f32>(1.0, 0.5, 0.15) * 0.025 * band(30.0, e.gsd);
                }
            } else if (wuse > 0u && cov_lot > 0.0) {
                let u = ublk.x - lr.x;
                let v = ublk.y - lr.y;
                if (wuse == 1u) {
                    col = st_car_park(lh, u, v, fw, e.gsd);
                    cls = LC_PARKING;
                } else if (wuse == 2u) {
                    col = mixc(st_concrete(), srgb(150.0, 140.0, 124.0), u01k(lh, 46lu)) * (0.9 + 0.1 * e.detail);
                    let a = select(v, u, lw > ld);
                    let c = select(u, v, lw > ld);
                    let ri = floor(c / 6.0);
                    let ci = floor(a / 12.8);
                    let ch = hash2(lh ^ 0xC0lu, i64(ci), i64(ri));
                    if (u01k(ch, 1lu) < 0.55) {
                        let din = min(min(u, v), min(lw - u, ld - v));
                        let cont = st_aa(1.25 - abs(c - (ri + 0.5) * 6.0), fw) * st_aa(6.1 - abs(a - (ci + 0.5) * 12.8), fw) * st_aa(din - 4.0, fw) * band(3.0, e.gsd);
                        let ck = u32(u01k(ch, 2lu) * 4.99);
                        var cc = srgb(124.0, 124.0, 122.0);
                        if (ck == 0u) {
                            cc = srgb(150.0, 74.0, 52.0);
                        } else if (ck == 1u) {
                            cc = srgb(62.0, 84.0, 124.0);
                        } else if (ck == 2u) {
                            cc = srgb(78.0, 106.0, 74.0);
                        } else if (ck == 3u) {
                            cc = srgb(204.0, 204.0, 198.0);
                        }
                        col = mixc(col, cc, cont);
                        h = max(h, 2.6 * (1.0 + floor(2.0 * u01k(ch, 3lu))) * cont);
                    }
                } else {
                    col = mixc(yard, srgb(118.0, 126.0, 78.0), 0.5) * (0.9 + 0.15 * e.detail);
                    cls = LC_GRASS;
                }
            } else if (fm == SF_CITY && dl > edge_thr - 0.06 && u01k(lh, 22lu) < 0.55) {
                cov_lot = 1.0;
                let sx = floor((ublk.x - lr.x) / 6.0);
                let gk = u01k(hash2(lh ^ 0x6A4Dlu, i64(sx), 0li), 1lu);
                col = mixc(srgb(96.0, 118.0, 62.0), srgb(122.0, 104.0, 78.0), gk) * (0.92 + 0.1 * e.detail);
                cls = LC_CROP;
            }
        }
    }
    // ---- farms and kraals (cellular)
    if (pattern == SP_DISPERSED || pattern == SP_KRAAL) {
        let pc = wc.point;
        let one = pattern == SP_KRAAL && pl.level == 0u;
        var dl = 1.0;
        if (!one) {
            dl = st_dens_at(pl, q, pc - q, e.n, dmul);
        }
        let unit_c = select(pc, vec2<f32>(0.0), one);
        let ok = one || u01k(wc.id, 3lu) < dl * 0.9;
        let r = q - unit_c;
        let uh = select(wc.id, pl.seed, one);
        if (ok && pattern == SP_DISPERSED) {
            let rr = st_rot2(r, (u01k(uh, 1lu) - 0.5) * 0.8);
            let yh = vec2<f32>(20.0 + 8.0 * u01k(uh, 2lu), 14.0 + 6.0 * u01k(uh, 3lu));
            let dy = st_boxd(rr, vec2<f32>(0.0), yh);
            let iy = st_aa(st_min2(dy), fw);
            if (iy > 0.0) {
                cov_lot = iy;
                col = mixc(yard, srgb(150.0, 138.0, 118.0), 0.5);
                cls = LC_FARMYARD;
                let hc = vec2<f32>(-0.45 * yh.x, -0.3 * yh.y);
                let dh = st_boxd(rr, hc, vec2<f32>(5.5, 4.5));
                let ih = st_aa(st_min2(dh), fw);
                let bc = vec2<f32>(0.3 * yh.x, 0.35 * yh.y);
                let db = st_boxd(rr, bc, vec2<f32>(10.0, 6.0));
                let ib = st_aa(st_min2(db), fw);
                if (ih > 0.0) {
                    let ridge = clamp(dh.y / 4.5, 0.0, 1.0);
                    rf_arch = pl.arch;
                    rf_u = u01k(uh, 4lu);
                    rf_flat = false;
                    rf_mul = 1.0;
                    rf_ins = ih;
                    h = (5.0 + 2.0 * ridge) * ih;
                    roof = ih;
                    rim = (1.0 - smoothstep1(0.0, 0.9, st_min2(dh))) * ih;
                    if (ih > 0.5) {
                        cls = LC_BUILDING;
                    }
                } else if (ib > 0.0) {
                    let ridge = clamp(db.y / 6.0, 0.0, 1.0);
                    rf_arch = pl.arch;
                    rf_u = u01k(uh, 5lu);
                    rf_flat = false;
                    rf_mul = 0.85;
                    rf_ins = ib;
                    h = (6.0 + 1.5 * ridge) * ib;
                    roof = ib;
                    if (ib > 0.5) {
                        cls = LC_BUILDING;
                    }
                }
                let lp = rr - hc - vec2<f32>(0.0, 5.5);
                em += vec3<f32>(1.0, 0.72, 0.42) * point_light(dot(lp, lp), 2.0, 0.4, fw);
            }
            let dyr = -st_min2(dy);
            if (dyr > 2.0 && dyr < 9.0 && u01k(hash2(uh, i64(floor(rr.x / 8.0)), i64(floor(rr.y / 8.0))), 1lu) < 0.45) {
                tree = 1.0;
                tree_h = 10.0;
            }
        } else if (ok) {
            let rk = 14.0 + 10.0 * u01k(uh, 2lu);
            let dk = length(r);
            let ang = atan2(r.y, r.x);
            let nh = floor(6.0 + 8.0 * u01k(uh, 3lu));
            let k = round(ang / ST_TAU * nh);
            let hk = hash1(uh, i64(k));
            let a0 = k / nh * ST_TAU + 0.3 * (u01k(hk, 1lu) - 0.5);
            let hc = vec2<f32>(cos(a0), sin(a0)) * rk;
            let hr = 2.6 + 0.8 * u01k(hk, 2lu);
            let dh = length(r - hc);
            let ih = st_aa(hr - dh, fw);
            let enc = st_aa(0.55 * rk - dk, fw);
            let fence = max(band_cov(dk - (rk + 6.0), 0.8, fw), band_cov(dk - 0.55 * rk, 0.6, fw));
            let yardk = st_aa(rk + 6.0 - dk, fw);
            if (yardk > 0.0) {
                cov_lot = yardk;
                col = srgb(156.0, 130.0, 98.0) * (0.92 + 0.15 * e.detail);
                cls = LC_FARMYARD;
                col = mixc(col, srgb(112.0, 92.0, 70.0), enc);
                col = mixc(col, srgb(84.0, 76.0, 58.0), fence * band(1.6, e.gsd) + 0.3 * fence * (1.0 - band(1.6, e.gsd)));
                if (ih > 0.0) {
                    let cone = 1.0 - clamp(dh / hr, 0.0, 1.0);
                    rf_arch = 2u;
                    rf_u = u01k(hk, 3lu);
                    rf_flat = false;
                    rf_mul = 1.0;
                    rf_ins = ih;
                    rf_shade = 0.85 + 0.25 * cone;
                    h = (2.2 + 1.6 * cone) * ih;
                    roof = ih;
                    if (ih > 0.5) {
                        cls = LC_BUILDING;
                    }
                }
            }
        }
    }
    // ---- the village centre: a plaza and its landmark
    if (fm != SF_CITY && fm != SF_DISPERSED && fm != SF_KRAAL && pl.level >= 1u) {
        let dq = length(q);
        if (dq < 26.0) {
            let pz = st_aa(22.0 - dq, fw);
            col = mixc(col, st_paving(), pz);
            cov_lot = max(cov_lot, pz);
            o.street = 0.0;
            var lmk = 0u;
            if (pl.arch == 3u || pl.arch == 4u) {
                lmk = 1u;
            } else if (pl.arch == 8u || pl.arch == 0u || pl.arch == 1u) {
                lmk = 2u;
            }
            var ins = 0.0;
            var hh = 0.0;
            var rcode = pl.arch;
            if (lmk == 1u) {
                let dd = 8.0 - dq;
                let mn = 1.6 - length(q - vec2<f32>(11.0, 0.0));
                ins = max(st_aa(dd, fw), st_aa(mn, fw));
                hh = select(9.0 + 6.0 * sqrt(clamp(dd / 8.0, 0.0, 1.0)), 30.0, mn > dd);
                rcode = 23u;
            } else if (lmk == 2u) {
                let dd = st_boxd(q, vec2<f32>(0.0), vec2<f32>(8.0, 8.0));
                ins = st_aa(st_min2(dd), fw);
                hh = 10.0 + 4.0 * clamp(st_min2(dd) / 8.0, 0.0, 1.0);
                rcode = 24u;
            } else {
                let dn = st_boxd(q, vec2<f32>(0.0, 0.0), vec2<f32>(14.0, 5.5));
                let dt = st_boxd(q, vec2<f32>(-16.0, 0.0), vec2<f32>(3.2, 3.2));
                ins = max(st_aa(st_min2(dn), fw), st_aa(st_min2(dt), fw));
                hh = select(12.0 + 4.0 * clamp(dn.y / 5.5, 0.0, 1.0), 32.0, st_min2(dt) > st_min2(dn));
                rcode = pl.arch;
            }
            if (ins > 0.0) {
                rf_arch = rcode;
                rf_u = 0.3;
                rf_flat = false;
                rf_mul = 0.9;
                rf_ins = ins;
                rf_shade = 1.0;
                h = hh * ins;
                roof = ins;
                if (ins > 0.5) {
                    cls = LC_BUILDING;
                }
            }
            em += vec3<f32>(1.0, 0.8, 0.55) * 0.05 * pz * band(20.0, e.gsd);
        }
    }
    // ---- the roof (one colour), lots below the pixel (their mean)
    let rc = st_roof_col(rf_arch, rf_u, rf_flat) * rf_mul;
    if (rf_ins > 0.0) {
        col = mixc(col, rc * rf_shade, rf_ins);
    }
    if (lm_ok) {
        let el = band(lot_scale, e.gsd);
        if (el < 1.0) {
            col = mixc(mixc(lm_g, rc, lm_frac), col, el);
            h = lm_h + (h - lm_h) * el;
            roof *= el;
            rim *= el;
        }
    }
    // ---- the green buffer along an estate's edge
    if (pattern == SP_INDUSTRIAL && d.buf < 30.0 && inside) {
        let k = smoothstep1(30.0, 22.0, d.buf);
        cov_lot = max(cov_lot, k);
        let g = mixc(yard, srgb(84.0, 112.0, 56.0), 0.5) * (0.95 + 0.1 * e.detail);
        col = mixc(col, g, k);
        h *= 1.0 - k;
        roof *= 1.0 - k;
        rim *= 1.0 - k;
        if (k > 0.5) {
            cls = LC_PARK;
        }
        let tcl = floor(ublk / 10.0);
        let th = hash2(blk.id ^ 0xB0Flu, i64(tcl.x), i64(tcl.y));
        let tp = (tcl + 0.5 + 0.5 * (vec2<f32>(u01k(th, 1lu), u01k(th, 2lu)) - 0.5)) * 10.0;
        if (u01k(th, 3lu) < 0.6 * k) {
            let tc = st_aa(3.0 + 1.5 * u01k(th, 4lu) - length(ublk - tp), fw);
            if (tc > tree) {
                tree = tc;
                tree_h = 8.0 + 6.0 * u01k(th, 5lu);
            }
        }
    }
    // ---- street trees
    let ar = pl.arch;
    let treed = (ar == 5u || ar == 6u || ar == 7u || ar == 9u || ar == 4u) && (z == SZ_INNER || z == SZ_OUTER || z == SZ_SUBURB) && u01k(blk.sid, 31lu) < 0.4;
    if (treed && has_streets && street_ok && pattern != SP_INFORMAL && pattern != SP_LINEAR) {
        let off = blk.sw + 1.6;
        let dt = abs(blk.ds - off);
        if (dt < 4.5) {
            let k = round(blk.along / 11.0);
            let th = hash2(blk.sid ^ 0x7EE5lu, i64(k), select(0li, 1li, blk.ds > off));
            if (u01k(th, 1lu) < 0.8) {
                let da = blk.along - k * 11.0;
                let tc = st_aa(2.8 + 0.8 * u01k(th, 2lu) - sqrt(dt * dt + da * da), fw);
                if (tc > tree) {
                    tree = tc;
                    tree_h = 9.0 + 4.0 * u01k(th, 3lu);
                }
            }
        }
    }
    // ---- composite the parts: lot, sidewalk, street
    var street = 0.0;
    if (has_streets) {
        street = o.street;
    }
    let cov = max(max(cov_lot, walk), street);
    if (cov > 0.0) {
        col = mixc(col, st_concrete() * 0.92, max(walk - street, 0.0) * (1.0 - cov_lot) / cov);
        var asph = srgb(76.0, 76.0, 80.0) * (1.0 + 0.05 * e.detail);
        if (o.major) {
            asph = st_asphalt() * 0.92;
        } else if (fm != SF_CITY || (pattern == SP_ORGANIC && pl.level < 3u) || pattern == SP_INFORMAL) {
            asph = mixc(srgb(150.0, 128.0, 100.0), st_paving() * 0.9, smoothstep1(0.35, 0.6, pl.dev));
        } else {
            asph = mixc(asph, st_paving() * 0.8, 0.6 * smoothstep1(0.3, 0.8, oldk));
        }
        col = mixc(col, asph, street / cov);
        if (o.major && street > 0.5) {
            let dash = select(0.0, 1.0, o.along / 9.0 - floor(o.along / 9.0) < 0.5);
            col = mixc(col, srgb(220.0, 220.0, 210.0), band_cov(o.ds, 0.15, fw) * dash * band(3.0, e.gsd));
        }
        if (street > 0.5) {
            cls = LC_ROAD_MINOR;
            if (o.major) {
                cls = LC_ROAD_MAJOR;
            }
        }
        h *= 1.0 - street;
        roof *= 1.0 - street;
        rim *= 1.0 - street;
        tree *= 1.0 - street;
    }
    o.cov = cov;
    o.col = col;
    o.h = h;
    o.cls = cls;
    if (cls == LC_BUILDING || cls == LC_BUILDING_TALL) {
        o.bid = lot_id;
    }
    o.roof = roof;
    o.rim = rim;
    o.win = win;
    o.em = em;
    o.tree = tree;
    o.tree_h = tree_h;
    return o;
}

/// `town::town_core`.
fn st_town_core(pl: StPl, z: u32, pattern: u32) -> bool {
    return (z == SZ_INNER || z == SZ_OUTER) && pl.level <= 2u && pl.form == SF_CITY && pattern != SP_INDUSTRIAL;
}

/// `town::town_centre`.
fn st_town_centre(pl: StPl, d: StDist) -> bool {
    return st_town_core(pl, d.zone, d.pattern);
}

/// `town::zone_fracs`: (green, paved).

fn st_zone_fracs(pl: StPl, d: StDist) -> vec2<f32> {
    let z = d.zone;
    if (st_town_centre(pl, d)) {
        return vec2<f32>(0.3, 0.3);
    }
    if (z == SZ_INNER) {
        return vec2<f32>(0.12, 0.4);
    } else if (z == SZ_OUTER) {
        return vec2<f32>(0.25, 0.32);
    } else if (z == SZ_SUBURB) {
        return vec2<f32>(0.42, 0.25);
    } else if (z == SZ_PERI) {
        return vec2<f32>(0.5, 0.18);
    }
    return vec2<f32>(0.05, 0.5);
}

/// `town::mean_class`.
fn st_mean_class(pl: StPl, d: StDist, q: vec2<f32>) -> u32 {
    let gp = st_zone_fracs(pl, d);
    let roofs = 1.0 - gp.x - gp.y;
    let u = u01k(hash2(pl.seed ^ 0xC1A55lu, i64(floor(q.x / 6.0)), i64(floor(q.y / 6.0))), 1lu);
    if (u < roofs) {
        return select(LC_BUILDING, LC_BUILDING_TALL, d.zone == SZ_CBD && pl.level >= 3u);
    } else if (u < roofs + 0.55 * gp.y) {
        return LC_ROAD_MINOR;
    } else if (d.zone == SZ_CBD) {
        return LC_COMMERCIAL;
    } else if (d.pattern == SP_INDUSTRIAL) {
        return LC_INDUSTRIAL;
    }
    return LC_RESIDENTIAL;
}

/// `town::zone_mean`: (colour, building height).
fn st_zone_mean(pl: StPl, d: StDist, moist: f32) -> vec4<f32> {
    let z = d.zone;
    let gp = st_zone_fracs(pl, d);
    let green = gp.x;
    let paved = gp.y;
    var hm = 30.0 * smoothstep1(1e5, 1.5e7, pl.pop) + 12.0;
    if (z == SZ_INNER) {
        hm = 12.0;
    } else if (z == SZ_OUTER) {
        hm = 8.0;
    } else if (z == SZ_SUBURB) {
        hm = 4.0;
    } else if (z == SZ_PERI) {
        hm = 2.5;
    }
    let roofs = 1.0 - green - paved;
    let flat = z <= SZ_OUTER && pl.form == SF_CITY;
    var rc = st_roof_col(pl.arch, 0.35, flat);
    if (st_town_centre(pl, d)) {
        rc = mixc(st_roof_col(pl.arch, 0.35, false), st_roof_col(pl.arch, 0.35, true), 0.35);
    }
    if (d.pattern == SP_INDUSTRIAL) {
        rc = srgb(140.0, 140.0, 138.0);
    } else if (d.pattern == SP_INFORMAL) {
        rc = srgb(124.0, 104.0, 86.0);
    }
    rc *= 0.85 + 0.3 * u01k(d.did, 14lu);
    var lawn = st_yard(moist, 0.5);
    if (st_town_centre(pl, d)) {
        lawn = mixc(lawn, srgb(86.0, 112.0, 54.0), 0.6);
    }
    let col = (rc * roofs + st_asphalt() * paved + lawn * green) * 0.88;
    return vec4<f32>(col, hm * select(0.5, 1.0, pl.level >= 2u));
}

/// `town::lamps`.
fn st_lamps(pl: StPl, o: StEv, dens: f32, gsd: f32, fw: f32, block: f32, resolved: f32) -> vec3<f32> {
    let l0 = vec3<f32>(1.0, 0.68, 0.40);
    let l1 = vec3<f32>(1.0, 0.84, 0.64);
    let l2 = vec3<f32>(0.86, 0.92, 1.0);
    let sodium = 0.25 + 0.5 * (1.0 - pl.dev) * u01k(pl.seed, 20lu);
    let sp = o.lamp_sp;
    if (sp <= 0.0) {
        return vec3<f32>(0.0);
    }
    let lamp_res = band(block, gsd) * resolved;
    let fwl = max(fw, 0.5 * gsd);
    let m = exp2(ceil(log2(max(2.0 * fwl / sp, 1.0))));
    let sp_eff = sp * m;
    let present = 0.5 + 0.45 * min(pl.dev, 1.0);
    var e = vec3<f32>(0.0);
    if (lamp_res > 0.0 && o.ds < o.sw + 20.0 + 3.0 * fwl) {
        let k = round(o.along / sp_eff) * m;
        let dp = o.ds - o.sw * 0.85;
        let da = o.along - k * sp;
        let d2 = dp * dp + da * da;
        let lh = hash2(o.sid ^ 0x1A3Blu, i64(k), 0li);
        if (u01k(lh, 1lu) < present) {
            let sa = 0.28 * sp;
            let pool = 0.03 * exp(-(dp * dp) / (2.0 * 3.5 * 3.5) - (da * da) / (2.0 * sa * sa));
            let core = point_light(d2, 6.0, 0.4, fwl);
            let u = u01k(hash1(o.sid, 3li), 1lu);
            var lc = l2;
            if (o.zone == SZ_CBD || (o.major && u < 0.45)) {
                lc = l2;
            } else if (o.pattern == SP_INDUSTRIAL || u < sodium) {
                lc = l0;
            } else if (u < 0.88) {
                lc = l1;
            }
            e = lc * ((pool + core) * m * (0.75 + 0.5 * u01k(lh, 2lu)));
        }
    }
    let lamp_e = present * (ST_TAU * (6.0 * 0.16 + 0.03 * 3.5 * 0.28 * sp));
    var zf = 0.45;
    if (o.zone == SZ_CBD) {
        zf = 1.6;
    } else if (o.zone == SZ_INNER) {
        zf = 1.2;
    } else if (o.zone == SZ_OUTER) {
        zf = 1.0;
    } else if (o.zone == SZ_SUBURB) {
        zf = 0.75;
    }
    let lamp_mean = lamp_e * 2.0 / (block * sp) * zf;
    var mc = mixc(l1, l0, sodium);
    if (o.zone == SZ_CBD) {
        mc = l2;
    } else if (o.pattern == SP_INDUSTRIAL) {
        mc = l0;
    }
    return e * lamp_res + mc * lamp_mean * (1.0 - lamp_res) * smoothstep1(0.1, 0.35, dens);
}

// ---------------------------------------------------------------- sites (`sites.rs`)

/// `sites::SitePx`.
struct StSite {
    cov: f32,
    col: vec3<f32>,
    dh: f32,
    abs: bool,
    cls: u32,
    emit: vec3<f32>,
    clear: f32,
    shade: f32,
}

fn st_site_none() -> StSite {
    var o: StSite;
    o.cov = 0.0;
    o.col = vec3<f32>(0.0);
    o.dh = 0.0;
    o.abs = false;
    o.cls = 0u;
    o.emit = vec3<f32>(0.0);
    o.clear = 0.0;
    o.shade = 0.0;
    return o;
}

fn st_uv(st: FStamp, p: vec3<f64>) -> vec2<f32> {
    let d = vec3<f32>(p - st.center.xyz);
    return vec2<f32>(dot(d, st.ex.xyz), dot(d, st.ey.xyz));
}

fn st_boxm(r: vec2<f32>, c: vec2<f32>, half: vec2<f32>) -> f32 {
    let d = half - abs(r - c);
    return min(d.x, d.y);
}

/// `sites::airport_px`.
fn st_airport_px(st: FStamp, p: vec3<f64>, ground: f32, gsd: f32, fw: f32) -> StSite {
    let q = st_uv(st, p);
    let hx = st.half.x;
    let hy = st.half.y;
    let len = st.v0.x;
    let lay = u32(st.v0.y);
    let seed = st_seed(st.v0.z, st.v0.w);
    let spacing = st.v1.x;
    let side = st.v1.z;
    let margin = 220.0;
    let din = min(hx - abs(q.x), hy - abs(q.y));
    var o = st_site_none();
    if (din <= 0.0) {
        return o;
    }
    var en = din + 55.0 * perlin3(seed ^ 0xED1lu, p * (1.0lf / 170.0lf));
    let b60 = band(60.0, gsd);
    if (b60 > 0.0) {
        en += 25.0 * perlin3(seed ^ 0xED2lu, p * (1.0lf / 60.0lf)) * b60;
    }
    let edge = smoothstep1(0.0, 0.8 * margin, en);
    o.cov = edge;
    o.abs = true;
    o.clear = edge;
    o.cls = LC_GRASS;
    let stripe = f32(st_rem(i64(floor(q.x / 18.0)), 2li));
    var tone = 1.0 + 0.10 * perlin3(seed ^ 0xED3lu, p * (1.0lf / 200.0lf)) * band(200.0, gsd);
    let b45 = band(45.0, gsd);
    if (b45 > 0.0) {
        tone += 0.06 * perlin3(seed ^ 0xED4lu, p * (1.0lf / 45.0lf)) * b45;
    }
    o.col = srgb(132.0, 146.0, 92.0) * (0.95 + 0.06 * stripe * band(18.0, gsd)) * tone;
    let wgt = smoothstep1(0.0, margin, din);
    o.dh = ground + (st.h - ground) * wgt;
    let wr = select(22.5, 30.0, lay == 2u);
    let yr0 = select(-side * 250.0, 0.5 * spacing, lay > 0u);
    var rw = 0.0;
    var rx = 0.0;
    var ry = 0.0;
    for (var k = 0u; k < 2u; k++) {
        if (k == 0u || lay > 0u) {
            let y = select(-0.5 * spacing, yr0, k == 0u);
            let c = st_aa(st_boxm(q, vec2<f32>(0.0, y), vec2<f32>(0.5 * len, wr)), fw);
            if (c > rw) {
                rw = c;
                rx = q.x;
                ry = q.y - y;
            }
        }
    }
    if (lay == 2u) {
        let a = st.v1.w;
        let s = sin(a);
        let c = cos(a);
        let cq = q - vec2<f32>(0.15 * len, 0.0);
        let r = vec2<f32>(c * cq.x + s * cq.y, -s * cq.x + c * cq.y);
        let lc = min(0.6 * len, 2.0 * hy - 300.0);
        let cc = st_aa(st_boxm(r, vec2<f32>(0.0), vec2<f32>(0.5 * lc, 22.5)), fw);
        if (cc > rw) {
            rw = cc;
            rx = r.x;
            ry = r.y;
        }
    }
    let ty = select(-side * 250.0 + side * 190.0, 0.5 * spacing - 190.0, lay > 0u);
    let ty2 = -0.5 * spacing + 190.0;
    var tw = st_aa(st_boxm(q, vec2<f32>(0.0, ty), vec2<f32>(0.5 * len - 100.0, 11.5)), fw);
    if (lay > 0u) {
        tw = max(tw, st_aa(st_boxm(q, vec2<f32>(0.0, ty2), vec2<f32>(0.5 * len - 100.0, 11.5)), fw));
    }
    let ck = round(q.x / 600.0);
    if (abs(ck) * 600.0 < 0.5 * len) {
        var y0 = min(yr0, ty);
        var y1 = max(yr0, ty);
        if (lay > 0u) {
            if (q.y > 0.0) {
                y0 = ty;
                y1 = yr0;
            } else {
                y0 = -0.5 * spacing;
                y1 = ty2;
            }
        }
        tw = max(tw, st_aa(st_boxm(q, vec2<f32>(ck * 600.0, 0.5 * (y0 + y1)), vec2<f32>(11.5, 0.5 * abs(y1 - y0))), fw));
    }
    var ay0 = ty2 + 60.0;
    var ay1 = ty - 60.0;
    if (lay == 0u) {
        if (side > 0.0) {
            ay0 = ty + 60.0;
            ay1 = ty + 360.0;
        } else {
            ay0 = ty - 360.0;
            ay1 = ty - 60.0;
        }
    }
    let ac = vec2<f32>(0.0, 0.5 * (ay0 + ay1));
    let ah = vec2<f32>(0.33 * len, 0.5 * abs(ay1 - ay0));
    let apron = st_aa(st_boxm(q, ac, ah), fw);
    let term_c = vec2<f32>(0.0, ac.y + 0.25 * ah.y * select(side, 0.0, lay > 0u));
    let term_h = vec2<f32>(0.22 * len, 32.0);
    let term = st_aa(st_boxm(q, term_c, term_h), fw);
    var hang = 0.0;
    for (var i = 0u; i < 3u; i++) {
        let hc = vec2<f32>(ac.x + ah.x - 60.0 - 110.0 * f32(i), ac.y - 0.5 * ah.y * max(side, 0.5));
        hang = max(hang, st_aa(st_boxm(q, hc, vec2<f32>(40.0, 30.0)), fw));
    }
    let park = st_aa(st_boxm(q, vec2<f32>(-0.6 * ah.x, term_c.y), vec2<f32>(0.25 * ah.x, 0.6 * ah.y)), fw) * (1.0 - term);
    if (apron > 0.0) {
        o.col = mixc(o.col, srgb(168.0, 166.0, 160.0), apron);
        o.cls = LC_TAXIWAY;
        let sk = round(q.x / 70.0);
        let sx = q.x - sk * 70.0;
        let sy = q.y - (term_c.y - side * (term_h.y + 45.0));
        let ah2 = hash2(seed ^ 0xA9lu, i64(sk), 0li);
        if ((u01k(ah2, 1lu) < 0.6 && lay == 0u) || (u01k(ah2, 1lu) < 0.7 && lay > 0u)) {
            let fus = st_aa(st_boxm(vec2<f32>(sx, sy), vec2<f32>(0.0), vec2<f32>(2.2, 22.0)), fw);
            let wing = st_aa(st_boxm(vec2<f32>(sx, sy - 2.0), vec2<f32>(0.0), vec2<f32>(18.0, 2.6)), fw);
            o.col = mixc(o.col, srgb(232.0, 234.0, 236.0), max(fus, wing) * band(4.0, gsd));
        }
    }
    if (park > 0.0) {
        o.col = mixc(o.col, srgb(70.0, 70.0, 74.0), park);
        let pc = tr_parked(seed ^ 0xA1Clu, q.x + 0.6 * ah.x, q.y - term_c.y, fw, gsd);
        o.col = mixc(o.col, pc.yzw, pc.x * park);
        o.cls = LC_PARKING;
    }
    if (tw > 0.0) {
        o.col = mixc(o.col, srgb(92.0, 92.0, 98.0), tw);
        o.cls = LC_TAXIWAY;
    }
    if (rw > 0.0) {
        var c = srgb(56.0, 56.0, 60.0) * (1.0 + 0.04 * (1.0 - smoothstep1(0.3 * len, 0.5 * len, abs(rx))));
        let dash = select(0.0, 1.0, rx / 50.0 - floor(rx / 50.0) < 0.6);
        let m1 = band_cov(ry, 0.45, fw) * dash;
        let m2 = band_cov(abs(ry) - (wr - 1.5), 0.45, fw);
        let thr = 0.5 * len - abs(rx);
        let keys = select(0.0, 1.0, thr > 6.0 && thr < 50.0 && abs(ry) < wr - 3.0 && fract(abs(ry) / 3.6) < 0.55);
        let td = select(0.0, 1.0, thr > 150.0 && thr < 600.0 && fract((thr - 150.0) / 150.0) < 0.2 && abs(abs(ry) - 0.45 * wr) < 0.2 * wr);
        let mk = min(m1 + m2 + keys + td, 1.0) * band(1.5, gsd);
        c = mixc(c, srgb(232.0, 232.0, 228.0), mk);
        o.col = mixc(o.col, c, rw);
        o.cls = LC_RUNWAY;
        let lk = round(rx / 60.0);
        let ex = rx - lk * 60.0;
        let ey = abs(ry) - wr - 1.0;
        o.emit += vec3<f32>(1.0, 0.95, 0.85) * point_light(ex * ex + ey * ey, 8.0, 0.35, fw);
    }
    if (abs(ry) < 15.0 && abs(rx) > 0.5 * len && abs(rx) < 0.5 * len + 420.0) {
        let ak = round((abs(rx) - 0.5 * len) / 30.0);
        let dx = abs(rx) - 0.5 * len - ak * 30.0;
        let bar = (1.0 - smoothstep1(6.0, 12.0, abs(ry))) * point_light(dx * dx, 6.0, 0.4, fw);
        o.emit += vec3<f32>(1.0, 0.92, 0.8) * bar;
    }
    if (tw > 0.0 && tw < 1.0) {
        o.emit += vec3<f32>(0.25, 0.45, 1.0) * 0.6 * band(30.0, gsd);
    }
    if (term > 0.0) {
        o.col = mixc(o.col, srgb(196.0, 198.0, 204.0) * (1.0 - 0.1 * band_cov(st_boxm(q, term_c, term_h), 1.2, fw)), term);
        o.dh += 18.0 * term * wgt;
        if (term > 0.5) {
            o.cls = LC_BUILDING;
        }
    }
    if (hang > 0.0 && term <= 0.0) {
        o.col = mixc(o.col, srgb(176.0, 182.0, 188.0), hang);
        o.dh += 16.0 * hang * wgt;
        if (hang > 0.5) {
            o.cls = LC_BUILDING;
        }
    }
    o.emit += vec3<f32>(1.0, 0.86, 0.62) * 0.12 * apron * (1.0 - term);
    o.col = mixc(o.col, srgb(90.0, 90.0, 92.0), band_cov(din - margin - 6.0, 3.5, fw));
    return o;
}

/// `sites::port_px` on land.
fn st_port_land_px(st: FStamp, p: vec3<f64>, gsd: f32, fw: f32) -> StSite {
    let q = st_uv(st, p);
    let len = st.v0.x;
    let land = st.v0.y;
    let seed = st_seed(st.v0.z, st.v0.w);
    var o = st_site_none();
    if (abs(q.x) > 0.5 * len + 250.0) {
        return o;
    }
    let along = abs(q.x) < 0.5 * len;
    let depth = -q.y;
    if (!along || depth < -60.0 || depth > land) {
        return o;
    }
    o.cov = 1.0;
    o.clear = 1.0;
    o.cls = LC_PORT;
    o.col = srgb(132.0, 130.0, 126.0);
    let quay = band_cov(depth - 8.0, 8.0, fw);
    let row = floor((depth - 40.0) / 16.0);
    let ry = depth - 40.0 - row * 16.0;
    let bay = floor(q.x / 13.0);
    let ch = hash2(seed ^ 0xC07lu, i64(bay), i64(row));
    let in_yard = depth > 40.0 && depth < 0.62 * land;
    let zn = u01k(hash1(seed, i64(floor(q.x / 200.0))), 1lu);
    if (in_yard && zn < 0.7) {
        let bx = st_aa(min(6.0 - abs(ry - 8.0), 6.0 - abs(q.x - (bay + 0.5) * 13.0)), fw);
        if (u01k(ch, 1lu) < 0.85) {
            let ck = u32(u01k(ch, 2lu) * 5.99);
            var cc = srgb(120.0, 70.0, 50.0);
            if (ck == 0u) {
                cc = srgb(170.0, 60.0, 44.0);
            } else if (ck == 1u) {
                cc = srgb(50.0, 90.0, 150.0);
            } else if (ck == 2u) {
                cc = srgb(200.0, 170.0, 70.0);
            } else if (ck == 3u) {
                cc = srgb(70.0, 120.0, 80.0);
            } else if (ck == 4u) {
                cc = srgb(200.0, 200.0, 196.0);
            }
            let e = band(6.0, gsd);
            o.col = mixc(o.col, cc, bx * e + 0.45 * (1.0 - e));
            o.dh = (2.6 * (1.0 + floor(u01k(ch, 3lu) * 3.99))) * bx;
        }
    } else if (depth > 0.62 * land || (in_yard && zn >= 0.7)) {
        let wk = floor(q.x / 120.0);
        let wh = hash1(seed ^ 0x3A2lu, i64(wk));
        let wcn = vec2<f32>((wk + 0.5) * 120.0, -(0.62 * land + 0.19 * land));
        let ins = st_aa(st_boxm(q, wcn, vec2<f32>(48.0, 0.15 * land)), fw);
        if (ins > 0.0 && u01k(wh, 1lu) < 0.8) {
            o.col = mixc(o.col, select(srgb(110.0, 130.0, 156.0), srgb(176.0, 178.0, 180.0), u01k(wh, 2lu) < 0.5), ins);
            o.dh = 12.0 * ins;
            if (ins > 0.5) {
                o.cls = LC_BUILDING;
            }
        }
    }
    let ck = round(q.x / 90.0);
    let cx = q.x - ck * 90.0;
    var crane = 0.0;
    if (u01k(hash1(seed ^ 0xC4Alu, i64(ck)), 1lu) < 0.7) {
        let legs = band_cov(abs(cx) - 9.0, 1.2, fw) * band_cov(depth - 20.0, 20.0, fw);
        let boom = band_cov(cx, 1.6, fw) * band_cov(depth - 10.0, 50.0, fw);
        crane = max(legs, boom);
    }
    o.col = mixc(o.col, srgb(186.0, 182.0, 176.0), quay);
    if (crane > 0.0) {
        o.col = mixc(o.col, select(srgb(60.0, 100.0, 150.0), srgb(160.0, 48.0, 40.0), u01k(seed, 5lu) < 0.5), crane);
        o.dh = max(o.dh, 45.0 * crane);
    }
    let fl = vec2<f32>(q.x - round(q.x / 80.0) * 80.0, depth - round(depth / 80.0) * 80.0);
    o.emit += vec3<f32>(1.0, 0.84, 0.6) * (point_light(dot(fl, fl), 14.0, 0.8, fw) + 0.025);
    return o;
}

/// `sites::port_px` on the water (level `wl`).
fn st_port_water_px(st: FStamp, p: vec3<f64>, wl: f32, gsd: f32, fw: f32) -> StSite {
    let q = st_uv(st, p);
    let len = st.v0.x;
    let seed = st_seed(st.v0.z, st.v0.w);
    let pier = st.v1.x;
    var o = st_site_none();
    if (abs(q.x) > 0.5 * len + 250.0) {
        return o;
    }
    let along = abs(q.x) < 0.5 * len;
    let k = round(q.x / 260.0);
    let px = q.x - k * 260.0;
    let pw = 26.0 + 10.0 * u01k(hash1(seed, i64(k)), 1lu);
    let ins = st_aa(st_boxm(vec2<f32>(px, q.y), vec2<f32>(0.0, 0.5 * pier - 40.0), vec2<f32>(pw, 0.5 * pier + 40.0)), fw);
    let on_pier = select(0.0, ins, along);
    let bw = band_cov(q.y - (pier + 160.0), 14.0, fw) * select(0.0, 1.0, abs(q.x) < 0.5 * len + 200.0);
    let sh = hash2(seed ^ 0x5A1lu, i64(k), select(0li, 1li, px > 0.0));
    let sx = abs(px) - pw - 18.0;
    let slen = 60.0 + 180.0 * u01k(sh, 1lu);
    var ship = 0.0;
    if (along && u01k(sh, 2lu) < 0.65) {
        ship = st_aa(st_boxm(vec2<f32>(sx, q.y), vec2<f32>(0.0, 0.5 * pier), vec2<f32>(14.0, 0.5 * slen)), fw);
    }
    if (on_pier > 0.0) {
        o.cov = on_pier;
        o.col = srgb(150.0, 148.0, 142.0) * (0.95 + 0.1 * u01k(hash2(seed, i64(floor(q.y / 12.0)), i64(k)), 1lu));
        o.dh = wl + 2.5;
        o.abs = true;
        o.cls = LC_PORT;
        let fl = vec2<f32>(px, round(q.y / 70.0) * 70.0 - q.y);
        o.emit += vec3<f32>(1.0, 0.82, 0.55) * (point_light(dot(fl, fl), 10.0, 0.6, fw) + 0.03 * on_pier);
    } else if (ship > 0.0) {
        o.cov = ship;
        let dk = u32(u01k(sh, 3lu) * 3.99);
        var deck = srgb(150.0, 150.0, 146.0);
        if (dk == 0u) {
            deck = srgb(120.0, 40.0, 36.0);
        } else if (dk == 1u) {
            deck = srgb(40.0, 60.0, 90.0);
        } else if (dk == 2u) {
            deck = srgb(60.0, 60.0, 64.0);
        }
        let cont = f32(st_rem(i64(floor(q.y / 6.5)) + i64(floor(sx / 2.6)), 3li));
        o.col = mixc(deck, srgb(170.0, 90.0, 60.0) * (0.8 + 0.2 * cont), 0.5 * band(6.0, gsd) * select(0.0, 1.0, u01k(sh, 4lu) < 0.5));
        o.dh = wl + 12.0;
        o.abs = true;
        o.cls = LC_PORT;
    } else if (bw > 0.0) {
        o.cov = bw;
        o.col = srgb(128.0, 124.0, 116.0) * (0.85 + 0.25 * u01k(hash1(seed, i64(floor(q.x / 4.0))), 1lu));
        o.dh = wl + 4.0;
        o.abs = true;
        o.cls = LC_ROCK;
    }
    return o;
}

/// `sites::quarry_px`.
fn st_quarry_px(st: FStamp, p: vec3<f64>, gsd: f32, fw: f32) -> StSite {
    let q = st_uv(st, p);
    let r = st.v0.x;
    let depth = st.v0.y;
    let seed = st_seed(st.v0.z, st.v0.w);
    let kind = u32(st.v1.x);
    let bh = st.v1.y;
    let ta = st.v1.z;
    let tr = st.v1.w;
    var o = st_site_none();
    o.clear = 1.0;
    o.cls = LC_QUARRY;
    let qa = vec2<f32>(q.x / 1.25, q.y * 1.1);
    let th0 = atan2(qa.y, qa.x);
    let wob = 1.0 + 0.13 * sin(3.0 * th0 + ST_TAU * u01k(seed, 8lu)) + 0.07 * sin(5.0 * th0 + ST_TAU * u01k(seed, 9lu));
    let rn = length(qa) / (r * wob);
    var rock = srgb(170.0, 150.0, 118.0);
    var bench = srgb(196.0, 180.0, 148.0);
    if (kind == 0u) {
        rock = srgb(196.0, 192.0, 180.0);
        bench = srgb(222.0, 218.0, 204.0);
    } else if (kind == 1u) {
        rock = srgb(150.0, 112.0, 82.0);
        bench = srgb(186.0, 150.0, 112.0);
    } else if (kind == 2u) {
        rock = srgb(150.0, 134.0, 128.0);
        bench = srgb(184.0, 170.0, 160.0);
    }
    if (rn < 1.0) {
        let floor_r = 0.25;
        let dn = clamp((1.0 - rn) / (1.0 - floor_r), 0.0, 1.0);
        let d = depth * dn;
        let res = band(bh * 1.2 / max(depth / ((1.0 - floor_r) * r), 0.05), gsd);
        let b = floor(d / bh);
        let f = d / bh - b;
        let step = b + smoothstep1(0.75, 1.0, f);
        let z = -bh * (step * res + (d / bh) * (1.0 - res));
        o.cov = 1.0;
        o.dh = z;
        let face = smoothstep1(0.7, 0.95, f) * res;
        o.col = mixc(bench, rock * 0.75, face) * (0.9 + 0.15 * u01k(hash1(seed, i64(b)), 1lu));
        let th = atan2(qa.y, qa.x);
        let turns = 1.5 + min(depth / 120.0, 2.5);
        let road_th = dn * turns * ST_TAU;
        let dth = (th - road_th) - ST_TAU * round((th - road_th) / ST_TAU);
        o.col = mixc(o.col, bench * 1.05, band_cov(dth * rn * r, 7.0, fw));
        if (kind == 1u && dn > 0.92) {
            o.col = srgb(70.0, 140.0, 130.0);
            o.cls = LC_LAKE;
        }
        return o;
    }
    let sc = vec2<f32>(cos(ta), sin(ta)) * (1.3 * r + 60.0);
    let qs = q - sc;
    let ths = atan2(qs.y, qs.x);
    let ds = length(qs) / ((0.35 * r + 40.0) * (1.0 + 0.25 * sin(4.0 * ths + ST_TAU * u01k(seed, 10lu))));
    if (ds < 1.0) {
        o.cov = 1.0;
        o.dh = min(0.12 * r, 35.0) * (1.0 - ds * ds);
        o.col = bench * (0.92 + 0.12 * (1.0 - ds));
        return o;
    }
    if (tr > 0.0) {
        let tc = vec2<f32>(cos(ta + 2.2), sin(ta + 2.2)) * (r * 1.35 + tr + 120.0);
        let dt = length(q - tc) - tr;
        if (dt < 25.0) {
            o.cov = 1.0;
            let beach = smoothstep1(-0.25 * tr, 0.0, dt);
            o.col = mixc(srgb(96.0, 150.0, 146.0), srgb(196.0, 184.0, 156.0), beach);
            o.col = mixc(o.col, srgb(150.0, 140.0, 120.0), band_cov(dt - 10.0, 9.0, fw));
            o.dh = 3.0 * band_cov(dt - 10.0, 9.0, fw);
            o.cls = select(LC_QUARRY, LC_AQUACULTURE, beach < 0.5);
            return o;
        }
    }
    let rim = band_cov(rn - 1.04, 0.04, fw / r);
    if (rim > 0.0) {
        o.cov = rim;
        o.col = bench * 0.95;
    }
    return o;
}

/// `sites::solar_px`.
fn st_solar_px(st: FStamp, p: vec3<f64>, gsd: f32, fw: f32) -> StSite {
    let q = st_uv(st, p);
    let hx = st.half.x;
    let hy = st.half.y;
    let pitch = st.v1.x;
    let pw = st.v1.y;
    let din = min(hx - abs(q.x), hy - abs(q.y));
    var o = st_site_none();
    if (din <= 0.0) {
        return o;
    }
    o.cov = 1.0;
    o.clear = 1.0;
    o.cls = LC_SOLAR_FARM;
    let gr = srgb(150.0, 140.0, 116.0);
    let panel = srgb(34.0, 42.0, 70.0);
    let e = band(pitch, gsd);
    let fy = q.y / pitch - floor(q.y / pitch);
    let blockx = (q.x + hx) / 130.0;
    let road = max(band_cov((blockx - round(blockx)) * 130.0, 3.0, fw), band_cov(q.y, 4.0, fw));
    let row = select(0.0, 1.0, fy * pitch < pw);
    let mean = pw / pitch;
    let cov_p = (row * e + mean * (1.0 - e)) * (1.0 - road) * smoothstep1(8.0, 12.0, din);
    o.col = mixc(gr, panel * (1.0 + 0.25 * min(fy * pitch / pw, 1.0) * e), cov_p);
    o.col = mixc(o.col, srgb(170.0, 160.0, 140.0), road * 0.7);
    o.col = mixc(o.col, srgb(80.0, 80.0, 82.0), band_cov(din - 4.0, 0.6, fw));
    o.dh = 1.6 * cov_p;
    return o;
}

/// `sites::wind_px`.
fn st_wind_px(st: FStamp, p: vec3<f64>, sun: vec2<f32>, sun_tan: f32, gsd: f32, fw: f32) -> StSite {
    let q = st_uv(st, p);
    let hx = st.half.x;
    let hy = st.half.y;
    let sp = st.v0.x;
    let hub = st.v0.y;
    let seed = st_seed(st.v0.z, st.v0.w);
    let rr = st.v1.x;
    let dens = st.v1.y;
    var o = st_site_none();
    if ((q.x / hx) * (q.x / hx) + (q.y / hy) * (q.y / hy) > 1.0) {
        return o;
    }
    let cell = vec2<f32>(sp, 1.6 * sp);
    let ci = round(q / cell);
    let th = hash2(seed, i64(ci.x), i64(ci.y));
    let tc = ci * cell + (vec2<f32>(u01k(th, 1lu), u01k(th, 2lu)) - 0.5) * 0.3 * cell;
    let tcn = vec2<f32>(tc.x / hx, tc.y / hy);
    if (u01k(th, 3lu) >= dens || length(tcn) > 0.95) {
        return o;
    }
    let r = q - tc;
    let fwt = max(fw, 0.3 * gsd);
    let track = band_cov(r.y, 2.5, fwt) * select(0.0, 1.0, abs(r.x) < 0.5 * cell.x);
    let pad = st_aa(18.0 - length(r), fwt);
    let tower = st_aa(2.4 - length(r), fwt);
    let a0 = u01k(th, 4lu) * ST_TAU;
    var blade = 0.0;
    for (var k = 0u; k < 3u; k++) {
        let a = a0 + f32(k) * ST_TAU / 3.0;
        let d = vec2<f32>(cos(a), sin(a));
        let t = dot(r, d);
        let n = length(r - d * t);
        if (t > 0.0 && t < rr) {
            blade = max(blade, band_cov(n, 1.3 * (1.0 - 0.6 * t / rr), fwt));
        }
    }
    let sl = hub / max(sun_tan, 0.1);
    let ts = dot(r, -sun);
    let tn = length(r + sun * ts);
    var shadow = 0.0;
    if (ts > 0.0 && ts < sl) {
        shadow = band_cov(tn, 1.8, fwt);
    }
    let gravel = max(track, pad);
    o.emit = vec3<f32>(1.0, 0.1, 0.05) * point_light(dot(r, r), 4.0, 0.5, fw);
    o.shade = 0.7 * shadow * (1.0 - max(blade, tower));
    let white = max(blade, tower);
    o.cov = max(gravel, white);
    if (o.cov <= 0.0) {
        return o;
    }
    o.col = mixc(srgb(160.0, 150.0, 132.0), srgb(236.0, 238.0, 240.0), white / o.cov);
    o.cls = select(0u, LC_TRACK, gravel > 0.5);
    o.clear = gravel;
    o.dh = hub * tower;
    return o;
}

/// `sites::flare_px`.
fn st_flare_px(st: FStamp, p: vec3<f64>, gsd: f32, fw: f32) -> StSite {
    let q = st_uv(st, p);
    let n = u32(st.v0.x);
    let seed = st_seed(st.v0.z, st.v0.w);
    var o = st_site_none();
    let fwt = max(fw, 0.4 * gsd);
    var best = 0.0;
    var track = 0.0;
    for (var k = 0u; k < 5u; k++) {
        if (k < n) {
            let h = hash1(seed, i64(k));
            var c = vec2<f32>(0.0);
            if (k > 0u) {
                c = (vec2<f32>(u01k(h, 1lu), u01k(h, 2lu)) - 0.5) * 2400.0;
            }
            let pad = st_aa(st_boxm(q, c, vec2<f32>(35.0 + 20.0 * u01k(h, 3lu))), fwt);
            best = max(best, pad);
            if (k > 0u) {
                let d = max(length(c), 1.0);
                let dir = c / d;
                let t = clamp(dot(q, dir), 0.0, d);
                track = max(track, band_cov(length(q - dir * t), 2.5, fwt));
            }
        }
    }
    let fc = vec2<f32>(20.0, 20.0);
    let dfl = length(q - fc);
    let scorch = st_aa(22.0 - dfl, fwt);
    o.emit = vec3<f32>(1.0, 0.55, 0.16) * point_light(dfl * dfl, 60.0, 3.0, fw);
    let cov = max(max(best, track), scorch);
    if (cov > 0.0) {
        o.cov = cov;
        o.col = mixc(srgb(176.0, 160.0, 130.0), srgb(46.0, 40.0, 36.0), scorch * (1.0 - smoothstep1(10.0, 22.0, dfl)));
        o.cls = LC_BARE;
        o.clear = cov;
    }
    return o;
}

/// `sites::reservoir_at`: (water coverage, bathtub ring coverage).
fn st_reservoir_at(sg: FSeg, p: vec3<f64>, ground: f32, fw: f32) -> vec2<f32> {
    let ab = vec3<f32>(sg.b.xyz - sg.a.xyz);
    let len = max(length(ab), 1.0);
    let dx = ab / len;
    let left = normalize(cross(normalize(vec3<f32>(sg.a.xyz * (1.0lf / 6.4e6lf))), dx));
    let ap = vec3<f32>(p - sg.a.xyz);
    let x = dot(ap, dx);
    let y = dot(ap, left);
    let t = clamp(x, 0.0, len);
    let dist = length(vec2<f32>(x - t, y));
    if (dist > sg.hw) {
        return vec2<f32>(0.0);
    }
    let rel = vec2<f32>(x - sg.v.x, y - sg.v.y);
    if (dot(rel, vec2<f32>(cos(sg.v.z), sin(sg.v.z))) > 0.0) {
        return vec2<f32>(0.0);
    }
    let level = sg.ha;
    let crest = sg.hb;
    let wet = st_aa(level - ground, fw * 0.05 + 0.3);
    var ring = 0.0;
    if (ground < crest - 1.0) {
        ring = 1.0 - wet;
    }
    return vec2<f32>(wet, ring);
}

/// `sites::dam_at`: (coverage, crest height, downstream face shade).
fn st_dam_at(sg: FSeg, p: vec3<f64>, ground: f32, fw: f32) -> vec3<f32> {
    let f = seg_frame(sg, p);
    let crest = sg.ha;
    if (f.z <= 0.0 || f.z >= 1.0 || ground > crest + 1.0) {
        return vec3<f32>(0.0, crest, 0.0);
    }
    return vec3<f32>(band_cov(f.x, sg.hw, fw), crest, smoothstep1(-sg.hw, sg.hw, f.x));
}

// ---------------------------------------------------------------- the slots (`settlements.rs`)

/// Slot 7: the land sites and the place at the sample (`settlements::built`).
/// `settlements::glow`: the expected built cover and street lights of the levels whose places
/// are not listed at this pixel size.
fn st_glow(s: ptr<function, Stack>) {
    let gsd = (*s).c.gsd;
    if (gsd < 150.0 || (*s).is_water || cfg.towns <= 0.0) {
        return;
    }
    let t = (*s).t;
    let habitable = (1.0 - smoothstep1(0.5, 0.6, t.glaciation)) * smoothstep1(-16.0, -12.0, t.temp);
    if (habitable <= 0.0) {
        return;
    }
    let pop = t.population;
    let dev = t.development;
    let arch = min((*s).bio.st.arch, 11u);
    var gmax = array<f32, 5>(150.0, 800.0, 3000.0, 12000.0, 50000.0);
    var cell = array<f32, 5>(1600.0, 6000.0, 25000.0, 90000.0, 300000.0);
    var base = array<f32, 5>(0.42, 0.62, 0.72, 0.72, 0.75);
    var pw0 = array<f32, 5>(-0.05, -0.02, 0.02, 0.06, 0.15);
    var pw1 = array<f32, 5>(0.3, 0.35, 0.45, 0.55, 0.75);
    var r0 = array<f32, 5>(45.0, 180.0, 1000.0, 3500.0, 13000.0);
    var r1 = array<f32, 5>(150.0, 800.0, 3600.0, 13000.0, 32000.0);
    var c1 = array<f32, 5>(1.6437, 1.9004, 1.707, 1.7338, 1.4268);
    var cal = array<f32, 5>(0.76, 1.19, 1.24, 1.28, 1.43);
    var kc = array<f32, 5>(0.0, 0.0, 0.00024, 0.00035, 0.00064);
    var kr = array<f32, 5>(0.03597, 0.03597, 0.01737, 0.01728, 0.01706);
    var kj = array<f32, 5>(1.8895, 1.8895, 1.5689, 1.5689, 1.5689);
    var aw_h = array<f32, 12>(1.5, 0.8, 1.3, 0.3, 0.7, 0.5, 0.8, 1.2, 0.6, 1.0, 0.3, 1.2);
    var aw_v = array<f32, 12>(1.0, 0.9, 0.7, 0.8, 0.5, 1.25, 1.15, 0.6, 1.4, 0.55, 0.25, 0.9);
    var aw_t = array<f32, 12>(1.0, 0.85, 0.7, 0.8, 0.6, 1.2, 1.05, 0.9, 1.25, 0.6, 0.25, 0.85);
    var spr = array<f32, 12>(0.95, 0.9, 0.9, 0.8, 0.95, 0.82, 0.92, 1.3, 0.85, 1.05, 0.85, 0.85);
    let sodium = 0.25 + 0.25 * (1.0 - dev);
    let lamp = mixc(vec3<f32>(1.0, 0.84, 0.64), vec3<f32>(1.0, 0.68, 0.40), sodium);
    let devm = (0.5 + 0.45 * min(dev, 1.0)) * (0.55 + 0.9 * dev);
    let sprawl2 = pow(spr[arch] * (0.85 + 0.3 * dev), 2.0) * (1.0 + 0.04 / 12.0);
    var cover = 0.0;
    var emit = vec3<f32>(0.0);
    for (var l = 0u; l < 5u; l++) {
        if (gsd < gmax[l]) {
            continue;
        }
        var cw = aw_t[arch];
        if (l == 0u) {
            cw = aw_h[arch];
        } else if (l == 1u) {
            cw = aw_v[arch];
        }
        let pe = min(base[l] * smoothstep1(pw0[l], pw1[l], pop) * cw * cfg.towns, 0.97);
        let n = pe / (cell[l] * cell[l]);
        let a = 2.0 * log(r1[l] / r0[l]);
        let sv = smoothstep1(pw0[l], 1.0, pop);
        let x = 0.18 * a * sv;
        var g = 1.0;
        if (x > 1e-6) {
            g = (exp(x) - 1.0) / x;
        }
        let r2 = r0[l] * r0[l] * c1[l] * exp(0.27 * a * sv) * g * sprawl2;
        let ar = n * r2 * cal[l] * habitable;
        cover += ar * kj[l];
        emit += (vec3<f32>(0.86, 0.92, 1.0) * kc[l] + lamp * kr[l]) * (ar * devm);
    }
    if (cover <= 0.0) {
        return;
    }
    let col = srgb(128.0, 122.0, 112.0) * (0.92 + 0.1 * (*s).pf.f[PF_DETAIL]);
    var ly = layer_paint(min(cover, 0.9), col, 0u);
    ly.emit = emit;
    ly.clear = 1.0;
    composite(s, ly);
}

fn settlements_slot_built(s: ptr<function, Stack>) {
    if ((*s).no_town) {
        return;
    }
    st_glow(s);
    let blk = (*s).l.blk;
    let ns = feat_stamp_n(blk);
    if (ns == 0u) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let ground = (*s).l.ground;
    let sun = (*s).c.east * cfg.sun_hx + (*s).c.north * cfg.sun_hy;
    var found = false;
    var best_pl: StPl;
    var best_score = 0.0;
    var best_rel = 0.0;
    var best_n = 0.0;
    var best_q = vec2<f32>(0.0);
    for (var k = 0u; k < ns; k++) {
        let st = feat_stamp(blk, k);
        var sp = st_site_none();
        if (st.tmpl == ST_PLACE) {
            let pl = st_decode(st);
            let q = st_local(pl, p);
            let r0 = st_rel0(pl, q);
            let strips = st_in_strips(pl, q);
            if (r0 < 1.62 || strips) {
                let n = st_foot_noise(pl, p, gsd);
                let rel = r0 / (1.0 + n);
                var score = -1.0;
                if (rel < 1.0) {
                    score = (1.0 - rel) * select(1.0, 1.15, pl.sub);
                } else if (strips) {
                    score = 0.001;
                }
                if (score > 0.0 && (!found || score > best_score)) {
                    found = true;
                    best_pl = pl;
                    best_score = score;
                    best_rel = rel;
                    best_n = n;
                    best_q = q;
                }
            }
            continue;
        } else if (st.tmpl == ST_AIRPORT) {
            sp = st_airport_px(st, p, ground, gsd, fw);
        } else if (st.tmpl == ST_PORT) {
            sp = st_port_land_px(st, p, gsd, fw);
        } else if (st.tmpl == ST_QUARRY) {
            sp = st_quarry_px(st, p, gsd, fw);
        } else if (st.tmpl == ST_SOLAR) {
            sp = st_solar_px(st, p, gsd, fw);
        } else if (st.tmpl == ST_WIND) {
            sp = st_wind_px(st, p, vec2<f32>(dot(sun, st.ex.xyz), dot(sun, st.ey.xyz)), cfg.sun_tan, gsd, fw);
        } else if (st.tmpl == ST_FLARE) {
            sp = st_flare_px(st, p, gsd, fw);
        } else {
            continue;
        }
        if (sp.cov > 0.0 || any(sp.emit != vec3<f32>(0.0)) || sp.shade > 0.0) {
            var ly = layer_paint(sp.cov, sp.col, sp.cls);
            ly.dh = sp.dh;
            ly.hmode = select(HM_BLEND, HM_ABS, sp.abs);
            ly.emit = sp.emit;
            ly.clear = sp.clear;
            ly.lit = 1.0 - sp.shade;
            composite(s, ly);
            (*s).m.natural_ok *= 1.0 - sp.cov * sp.clear;
        }
    }
    if (found) {
        st_place(s, best_pl, best_rel, best_n, best_q, sun);
    }
}

/// `settlements::place`.
fn st_place(s: ptr<function, Stack>, pl: StPl, rel: f32, n: f32, q: vec2<f32>, sun: vec3<f32>) {
    let t = (*s).t;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let town_slope = 0.12 + 0.75 * t.rock_expect;
    let slope_ok = 1.0 - smoothstep1(0.45, 0.8, town_slope);
    let dens0 = st_density(pl, rel) * slope_ok * (*s).m.river_clear;
    let strips = st_in_strips(pl, q);
    if (dens0 <= 0.0 && !strips) {
        return;
    }
    var env: StEnv;
    env.gsd = gsd;
    env.fw = fw;
    env.n = n;
    env.moist = t.moist;
    env.detail = (*s).pf.f[PF_DETAIL];
    env.ground = (*s).col;
    let city = pl.form == SF_CITY;
    let block = select(40.0, 90.0, city);
    let e_blk = band(block, gsd);
    let d = st_district(pl, q, rel);
    let z = d.zone;
    let dens = dens0 * d.dmul;
    let zm = st_zone_mean(pl, d, t.moist);
    let mcol = zm.xyz;
    let mh = zm.w;
    let mean_cov = pow(dens, 0.7);
    var ev = st_ev_none();
    ev.zone = z;
    ev.pattern = d.pattern;
    ev.lamp_sp = select(34.0, 24.0, city);
    var shade = 0.0;
    if (e_blk > 0.0) {
        let sq = st_normalize2(vec2<f32>(dot(sun, pl.ex), dot(sun, pl.ey)));
        let steps = select(1u, 4u, (cfg.flags & CF_SHADOWS) != 0u);
        let tan_el = cfg.sun_tan;
        for (var k = 0u; k < steps; k++) {
            let qk = q + sq * (3.5 * f32(k));
            let e = st_place_eval(pl, qk, d, env);
            if (k == 0u) {
                ev = e;
                if (e.h > 0.5 || e.cov <= 0.0 || e.street > 0.5) {
                    break;
                }
            } else if (e.h > 3.5 * f32(k) * tan_el) {
                shade = 0.75;
                break;
            }
        }
    }
    let cov = mean_cov + (ev.cov - mean_cov) * e_blk;
    (*s).m.veg *= 1.0 - smoothstep1(0.0, 0.08, dens) * select(0.6, 1.0, city);
    (*s).m.town_urban = dens;
    if (cov <= 0.0 && ev.tree <= 0.0) {
        return;
    }
    let col = mixc(mcol * (0.92 + 0.1 * (*s).pf.f[PF_DETAIL]), ev.col, e_blk);
    let h = mh * dens + (ev.h - mh * dens) * e_blk;
    var cls = ev.cls;
    if (e_blk < 0.5) {
        cls = st_mean_class(pl, d, q);
    }
    let dev = 0.55 + 0.9 * pl.dev;
    let lamp = st_lamps(pl, ev, dens, gsd, fw, block, e_blk);
    let win_lit = 0.1 + select(0.0, 0.25, z == SZ_CBD);
    let windows = ev.win * (0.55 * ev.rim * win_lit * band(2.6, gsd));
    let emit = (lamp * (1.0 - ev.roof) + windows + ev.em * e_blk) * dev;
    var ly = layer_paint(cov, col, cls);
    ly.dh = h;
    ly.hmode = select(HM_NONE, HM_BLEND, (cfg.flags & CF_BUILDINGS_DSM) != 0u);
    ly.emit = emit;
    ly.lit = 1.0 - shade * e_blk;
    ly.clear = 1.0;
    if (e_blk >= 0.5 && ev.bid != 0lu) {
        ly.inst = instance_id(ev.bid);
    }
    composite(s, ly);
    if (ev.tree > 0.0 && e_blk > 0.0) {
        let tc = spal(s, BP_CROWN_DECID) * 0.9 * (0.85 + 0.3 * u01k(ev.sid ^ 0x7EElu, 1lu));
        var lt = layer_paint(ev.tree * e_blk, tc, 0u);
        lt.dh = ev.tree_h;
        lt.hmode = HM_BLEND;
        composite(s, lt);
    }
    (*s).m.town_cov = cov;
    (*s).m.natural_ok *= 1.0 - cov;
}

/// Slot 8: reservoirs and dam walls on land, the water side of ports on standing water
/// (`settlements::water`).
fn settlements_slot_water(s: ptr<function, Stack>) {
    let blk = (*s).l.blk;
    let nst = feat_stamp_n(blk);
    let nsg = feat_seg_n(blk);
    if ((nst == 0u && nsg == 0u) || (*s).no_town) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let l = (*s).l;
    let standing = l.water > l.ground && l.water_kind != W_NONE;
    if (standing) {
        if (l.water_kind != W_OCEAN) {
            return;
        }
        for (var k = 0u; k < nst; k++) {
            let st = feat_stamp(blk, k);
            if (st.tmpl == ST_PORT) {
                let sp = st_port_water_px(st, p, l.water, gsd, fw);
                if (sp.cov > 0.0 || any(sp.emit != vec3<f32>(0.0))) {
                    var ly = layer_paint(sp.cov, sp.col, sp.cls);
                    ly.dh = sp.dh;
                    ly.hmode = HM_ABS;
                    ly.emit = sp.emit;
                    ly.solid = true;
                    ly.relit = 1.0;
                    composite(s, ly);
                }
            }
        }
        return;
    }
    let ground = l.ground;
    var wet = 0.0;
    var ring = 0.0;
    var level = 0.0;
    var dam = vec3<f32>(0.0);
    for (var k = 0u; k < nsg; k++) {
        let sg = feat_seg(blk, k);
        if (sg.kind == ST_RESERVOIR) {
            let wr = st_reservoir_at(sg, p, ground, fw);
            if (wr.x > wet) {
                wet = wr.x;
                level = sg.ha;
            }
            ring = max(ring, wr.y);
        } else if (sg.kind == ST_DAM) {
            let dm = st_dam_at(sg, p, ground, fw);
            if (dm.x > dam.x) {
                dam = dm;
            }
        }
    }
    if (wet > 0.0) {
        let depth = max(level - ground, 0.0);
        let col = mixc(srgb(70.0, 96.0, 84.0), srgb(28.0, 58.0, 76.0), smoothstep1(0.0, 25.0, depth));
        var ly = layer_paint(wet, col, LC_RESERVOIR);
        ly.dh = level;
        ly.hmode = HM_ABS;
        ly.water = true;
        ly.relit = 1.0;
        ly.clear = 1.0;
        composite(s, ly);
    }
    if (ring > 0.0) {
        var ly = layer_paint(ring * (1.0 - wet) * max(band(8.0, gsd), 0.5), srgb(178.0, 166.0, 140.0), LC_BARE);
        ly.clear = 1.0;
        composite(s, ly);
    }
    if (dam.x > 0.0) {
        let col = mixc(srgb(190.0, 188.0, 180.0), srgb(150.0, 148.0, 142.0), dam.z);
        var ly = layer_paint(dam.x, col, LC_DAM);
        ly.dh = dam.y;
        ly.hmode = HM_ABS;
        ly.solid = true;
        ly.relit = 1.0;
        composite(s, ly);
    }
}
