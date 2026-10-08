// Macro-scale world model ("pass A"): `world.rs` / `hydro.rs::river_query`. Continents,
// relief, drainage carving, lakes, climate, land-use sites and roads.

/// The world config (`Config`), with derived values.
struct Cfg {
    home_p: vec4<f64>,
    lvl_inv_mlam: vec4<f64>,
    lvl_inv_fplam: vec4<f64>,
    inv_gully_lam: f64,
    lake_cell: f64,
    region_cell: f64,
    town_cell: f64,
    ell_a: f64,
    ell_b: f64,
    seed: u64,
    nlevels: u32,
    flags: u32,
    cont_warp: f32,
    cont_threshold: f32,
    home_r: f32,
    home_st: f32,
    mtn_height: f32,
    hill_height: f32,
    micro_height: f32,
    mesas: f32,
    dune_height: f32,
    erosion: f32,
    gully_lam: f32,
    lake_density: f32,
    eq_temp: f32,
    pole_drop: f32,
    lapse: f32,
    moist_bias: f32,
    tree_density: f32,
    agriculture: f32,
    towns: f32,
    roads: f32,
    sun_hx: f32,
    sun_hy: f32,
    sun_tan: f32,
    ambient: f32,
    direct: f32,
    exposure: f32,
    haze: f32,
    l0: f32,
    sun_e: f32,
    sun_n: f32,
    sun_u: f32,
    saturation: f32,
    brightness: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
    /// per drainage level: cell (m), width min, width max, valley half width (m)
    lvl_a: array<vec4<f32>, 4>,
    /// per drainage level: wet moisture, meander, max depth (m), meander wavelength (m)
    lvl_b: array<vec4<f32>, 4>,
}

const CF_RIVERS: u32 = 1u;
const CF_TREES_DSM: u32 = 2u;
const CF_BUILDINGS_DSM: u32 = 4u;
const CF_SHADOWS: u32 = 8u;
const CF_ADAPTIVE: u32 = 16u;

@group(0) @binding(0) var<uniform> cfg: Cfg;

const KM: f32 = 1000.0;
const NONE_F: f32 = 3.0e38;

const W_NONE: u32 = 0u;
const PI_F64: f64 = 3.141592653589793lf;
const W_OCEAN: u32 = 1u;
const W_LAKE: u32 = 2u;
const W_RIVER: u32 = 3u;

const MODE_FULL: u32 = 0u;
const MODE_NOLAKES: u32 = 1u;
const MODE_RELIEF: u32 = 2u;

/// Sampling context of a point on the ellipsoid surface.
struct Ctx {
    p: vec3<f64>,
    up: vec3<f32>,
    east: vec3<f32>,
    north: vec3<f32>,
    lat: f32,
    gsd: f32,
}

/// The context from the trig of latitude / longitude and the ECEF point (h = 0).
fn ctx_make(p: vec3<f64>, sl: f32, cl: f32, so: f32, co: f32, lat: f32, gsd: f32) -> Ctx {
    var c: Ctx;
    c.p = p;
    c.up = vec3<f32>(cl * co, cl * so, sl);
    c.east = vec3<f32>(-so, co, 0.0);
    c.north = vec3<f32>(-sl * co, -sl * so, cl);
    c.lat = lat;
    c.gsd = gsd;
    return c;
}

fn ctx_offset(c: Ctx, de: f32, dn: f32) -> vec3<f64> {
    return c.p + vec3<f64>(c.east * de + c.north * dn);
}

fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l > 0.0) {
        return v / l;
    }
    return vec3<f32>(0.0);
}

fn dist64(a: vec3<f64>, b: vec3<f64>) -> f32 {
    return length(vec3<f32>(a - b));
}

/// Large-scale fields (`Macro`, without `pre`).
struct Macro {
    cont: f32,
    plateau: f32,
    belt: f32,
    belt2: f32,
    belt_var: f32,
    hill_amp: f32,
    rough: f32,
    temp: f32,
    moist: f32,
    mesa: f32,
    sand: f32,
    agri: f32,
    style: vec4<f32>,
    river_width: f32,
    mtn_warp: vec2<f32>,
}

/// Smooth inputs from the tile's coarse grid (`Pre`); `flags` marks the fields that are there.
struct Pre {
    flags: u32,
    gully: vec2<f32>,
    road_major: vec3<f32>,
    road_minor: vec3<f32>,
    relief: vec4<f32>,
    relief4: f32,
    relief_cut: vec2<f32>,
    river_warp01: vec4<f32>,
    river_warp23: vec4<f32>,
    region_warp: vec3<f32>,
    gully_oct: vec4<f32>,
    floodplain: vec4<f32>,
    site_lake: Sites,
    site_region: Sites,
    site_town: Sites,
}

const P_GULLY: u32 = 1u;
const P_ROADS: u32 = 2u;
const P_RELIEF: u32 = 4u;
const P_RIVER_WARP0: u32 = 8u; // .. 8 << 3 for level 3
const P_REGION_WARP: u32 = 128u;
const P_GULLY_OCT: u32 = 256u;
const P_FLOODPLAIN0: u32 = 512u; // .. 512 << 3
const P_SITE_LAKE: u32 = 8192u;
const P_SITE_REGION: u32 = 16384u;
const P_SITE_TOWN: u32 = 32768u;

fn pre_none() -> Pre {
    var p: Pre;
    p.flags = 0u;
    return p;
}

fn pre_river_warp(p: Pre, li: u32) -> vec2<f32> {
    switch li {
        case 0u: { return p.river_warp01.xy; }
        case 1u: { return p.river_warp01.zw; }
        case 2u: { return p.river_warp23.xy; }
        default: { return p.river_warp23.zw; }
    }
}

// ---------------------------------------------------------------- large-scale fields

fn continent(p: vec3<f64>, gsd: f32) -> f32 {
    let cw = fbm_wavelength(FBM_CONT);
    let warp = vec3<f32>(fbm(FBM_CONT_WARP0, p, gsd), fbm(FBM_CONT_WARP1, p, gsd), fbm(FBM_CONT_WARP2, p, gsd)) * (cfg.cont_warp * cw);
    let gl = max(gsd, 15.0 * KM);
    var c = fbm(FBM_CONT, p + vec3<f64>(warp), gl) * fbm_norm(FBM_CONT) * 1.6 - cfg.cont_threshold;
    if (cfg.home_p.w > 0.0lf) {
        let d = dist64(p, cfg.home_p.xyz);
        let w = 1.0 - smoothstep1(0.4 * cfg.home_r, cfg.home_r, d);
        c = lerp(c, max(c, 0.18 + 0.1 * c), w * cfg.home_st);
    }
    return c;
}

/// Octaves of the ridged mountains of wavelength >= `cut` (`low`) or < `cut` (not `low`,
/// continuing from `st`): sum, low-passed sum, weight (`World::ridged_part`).
fn ridged_part(p: vec3<f64>, gsd: f32, sharp: f32, cut: f32, low: bool, st: vec3<f32>) -> vec3<f32> {
    var sum = st.x;
    var low_sum = st.y;
    var weight = st.z;
    var lam = 14000.0lf;
    var inv = 1.0lf / 14000.0lf;
    var amp = 1.0;
    let fb = fbms[FBM_MTN];
    for (var i = 0u; i < fb.n; i++) {
        let lamf = f32(lam);
        if (low && lamf < cut) {
            break;
        }
        // ridges contain harmonics above the octave frequency: band-limit more strictly
        let wb = band(lamf, 1.6 * gsd);
        if (wb <= 0.0) {
            break;
        }
        if (low || lamf < cut) {
            let o = octs[fb.first + i];
            let s = p * inv;
            let q = o.ax.xyz * s.x + o.ay.xyz * s.y + o.az.xyz * s.z + o.off.xyz;
            let n = perlin3(o.seed, q);
            var r = max(1.0 - abs(n), 0.0);
            r = pow(r, sharp);
            r *= weight;
            weight = clamp(r * 1.8, 0.0, 1.0);
            sum += r * amp * wb;
            if (lamf > 6.0 * KM) {
                low_sum += r * amp * wb;
            }
        }
        lam *= 0.5lf;
        inv *= 2.0lf;
        amp *= select(0.42, 0.58, lam > 1000.0lf);
    }
    return vec3<f32>(sum, low_sum, weight);
}

fn ridged(p: vec3<f64>, gsd: f32, sharp: f32) -> vec2<f32> {
    let st = ridged_part(p, gsd, sharp, 0.0, true, vec3<f32>(0.0, 0.0, 1.0));
    return st.xy * 0.5;
}

/// Octaves of the hill fBm of wavelength >= `cut` (`low`) or < `cut`: sum, low-passed sum.
fn hills_part(p: vec3<f64>, gsd: f32, gain: f32, cut: f32, low: bool) -> vec2<f32> {
    var lam = 9000.0lf;
    var inv = 1.0lf / 9000.0lf;
    var amp = 1.0;
    var sum = 0.0;
    var low_sum = 0.0;
    let fb = fbms[FBM_HILLS];
    for (var i = 0u; i < fb.n; i++) {
        let lamf = f32(lam);
        if (low && lamf < cut) {
            break;
        }
        let wb = band(lamf, gsd);
        if (wb <= 0.0 || lamf < 20.0) {
            break;
        }
        if (low || lamf < cut) {
            let o = octs[fb.first + i];
            let s = p * inv;
            let q = o.ax.xyz * s.x + o.ay.xyz * s.y + o.az.xyz * s.z + o.off.xyz;
            let n = perlin3(o.seed, q);
            sum += n * amp * wb;
            if (lamf >= 4.0 * KM) {
                low_sum += n * amp * wb;
            }
        }
        lam *= 0.5lf;
        inv *= 2.0lf;
        amp *= gain;
    }
    return vec2<f32>(sum, low_sum);
}

fn hills(p: vec3<f64>, gsd: f32, gain: f32) -> vec2<f32> {
    return hills_part(p, gsd, gain, 0.0, true) * 0.7;
}

/// Octaves of the erosion gullies of wavelength >= `cut` (`low`) or < `cut` (continuing from
/// `st`): height and the accumulated derivative (`World::gullies_part`).
fn gullies_part(p: vec3<f64>, up: vec3<f32>, grad: vec3<f32>, gsd: f32, cut: f32, low: bool, st: vec4<f32>) -> vec4<f32> {
    let dir0 = normalize_or_zero(cross(up, grad));
    var a = 1.0;
    var lam = f64(cfg.gully_lam);
    var inv = cfg.inv_gully_lam;
    var h = st.x;
    var hd = st.yzw;
    for (var i = 0u; i < 5u; i++) {
        let lamf = cfg.gully_lam * exp2(-f32(i));
        if (low && lamf < cut) {
            break;
        }
        let wb = band(lamf, gsd);
        if (wb <= 0.0) {
            break;
        }
        if (low || lamf < cut) {
            // bending rotates the stripe direction; keep its length (= stripe frequency) at 1
            let dir = normalize_or_zero(dir0 + cross(up, hd) * 0.7);
            let vd = gully_octave(cfg.seed ^ (0xE205lu + u64(i) * 0x9E37lu), p * inv, dir);
            h += vd.x * a * wb;
            hd += vd.yzw * (a * wb);
        }
        a *= 0.45;
        inv *= 2.0lf;
    }
    return vec4<f32>(h, hd);
}

fn gullies(p: vec3<f64>, up: vec3<f32>, grad: vec3<f32>, gsd: f32) -> f32 {
    return gullies_part(p, up, grad, gsd, 0.0, true, vec4<f32>(0.0)).x;
}

/// Sand dunes (`World::dunes`).
fn dunes(c: Ctx, m: Macro, gsd: f32) -> f32 {
    let lam0 = 520.0;
    var sum = 0.0;
    if (band(lam0, gsd) > 0.0) {
        let bend = 0.7 * perlin3(cfg.seed ^ 0xB3Dlu, c.p / 2080.0lf) + 0.3 * perlin3(cfg.seed ^ 0xB3Elu, c.p / 780.0lf);
        let th = 2.5 * m.style.x + 0.8 * m.style.z + bend;
        let wind = c.east * cos(th) + c.north * sin(th);
        let v = gully_octave(cfg.seed ^ 0xD0Elu, c.p / 520.0lf, wind).x;
        sum += pow(max(0.5 + 0.5 * v, 0.0), 1.25) * band(lam0, gsd);
    }
    var lam = 520.0lf * 0.33lf;
    var amp = 0.3;
    let fb = fbms[FBM_DUNES];
    for (var i = 0u; i < 3u; i++) {
        let wb = band(f32(lam), 1.5 * gsd);
        if (wb <= 0.0) {
            break;
        }
        let o = octs[fb.first + i];
        let s = c.p / lam;
        let q = o.ax.xyz * s.x + o.ay.xyz * s.y + o.az.xyz * s.z + o.off.xyz;
        let r = 1.0 - abs(perlin3(o.seed, q));
        sum += r * r * amp * wb;
        amp *= 0.35;
        lam *= 0.4lf;
    }
    return sum;
}

fn mtn_warp_at(p: vec3<f64>, gsd: f32) -> vec2<f32> {
    return vec2<f32>(fbm(FBM_MTN_WARP0, p, gsd), fbm(FBM_MTN_WARP1, p, gsd));
}

/// All large-scale fields at a point (`World::macro_at`).
fn macro_at(p: vec3<f64>, gsd: f32) -> Macro {
    var m: Macro;
    m.cont = continent(p, gsd);
    m.plateau = fbm(FBM_PLATEAU, p, gsd) * fbm_norm(FBM_PLATEAU) * 1.8;
    m.belt = fbm(FBM_BELT, p, gsd) * fbm_norm(FBM_BELT) * 2.0;
    m.belt2 = fbm(FBM_BELT2, p, gsd) * fbm_norm(FBM_BELT2) * 2.0;
    m.belt_var = fbm(FBM_BELT_VAR, p, gsd) * fbm_norm(FBM_BELT_VAR) * 1.6;
    m.hill_amp = fbm(FBM_HILL_AMP, p, gsd) * fbm_norm(FBM_HILL_AMP) * 1.6;
    m.rough = fbm(FBM_ROUGH, p, gsd) * fbm_norm(FBM_ROUGH) * 1.6;
    m.temp = 5.0 * fbm(FBM_TEMP, p, 50.0 * KM);
    m.moist = fbm(FBM_MOIST, p, 20.0 * KM) * fbm_norm(FBM_MOIST) * 1.6;
    m.mesa = fbm(FBM_MESA, p, gsd) * fbm_norm(FBM_MESA) * 1.8;
    m.sand = fbm(FBM_SAND, p, gsd) * fbm_norm(FBM_SAND) * 1.8;
    m.agri = fbm(FBM_AGRI, p, gsd) * fbm_norm(FBM_AGRI) * 1.8;
    m.style = vec4<f32>(fbm(FBM_STYLE0, p, gsd), fbm(FBM_STYLE1, p, gsd), fbm(FBM_STYLE2, p, gsd), fbm(FBM_STYLE3, p, gsd));
    m.river_width = fbm(FBM_RIVER_WIDTH, p, gsd);
    m.mtn_warp = mtn_warp_at(p, gsd);
    return m;
}

// ---------------------------------------------------------------- drainage noise

fn level_key(lvl: u32) -> u64 {
    // seed.rotate_left(23) ^ (lvl + 1) * K1
    let s = cfg.seed;
    return ((s << 23u) | (s >> 41u)) ^ (u64(lvl + 1u) * 0x9E3779B97F4A7C15lu);
}

/// The meander warp (east, north; × 0.22 wavelength) of drainage level `li`.
fn meander_warp(li: u32, p: vec3<f64>) -> vec2<f32> {
    let inv = cfg.lvl_inv_mlam[li];
    let k = level_key(li);
    let w1 = perlin3(k ^ 1lu, p * inv) + 0.45 * perlin3(k ^ 2lu, p * (inv / 0.37lf));
    let w2 = perlin3(k ^ 3lu, p * inv) + 0.45 * perlin3(k ^ 4lu, p * (inv / 0.37lf));
    return vec2<f32>(w1, w2);
}

fn floodplain_wavelength(li: u32) -> f32 {
    let lc = cfg.lvl_a[li];
    return 400.0 + 0.75 * (lc.y + lc.z);
}

fn floodplain_noise(li: u32, p: vec3<f64>) -> f32 {
    return perlin3(0xF10Dlu ^ u64(li), p * cfg.lvl_inv_fplam[li]);
}

/// Warp (m) of the lookup in the region lattice.
fn region_warp(p: vec3<f64>) -> vec3<f32> {
    let inv = 1.0lf / (0.9lf * cfg.region_cell);
    let a = vec3<f32>(perlin3(cfg.seed ^ 0xA1lu, p * inv), perlin3(cfg.seed ^ 0xA2lu, p * inv), perlin3(cfg.seed ^ 0xA3lu, p * inv)) * (0.18 * f32(cfg.region_cell));
    let i2 = 1.0lf / 1500.0lf;
    let b = vec3<f32>(perlin3(cfg.seed ^ 0xA4lu, p * i2), perlin3(cfg.seed ^ 0xA5lu, p * i2), perlin3(cfg.seed ^ 0xA6lu, p * i2)) * 120.0;
    return a + b;
}

fn base_elevation(s: f32) -> f32 {
    if (s > 0.0) {
        return 20.0 + 900.0 * pow(s, 1.3);
    }
    return 20.0 - 120.0 * smoothstep1(0.0, 0.04, -s) - 3800.0 * smoothstep1(0.03, 0.35, -s);
}

/// (mountain mask, mountain amplitude)
fn mountain_mask(m: Macro) -> vec2<f32> {
    let b1 = 1.0 - abs(m.belt);
    let b2 = 1.0 - abs(m.belt2);
    let belt = max(b1 * 0.75 + b2 * 0.45 + 0.35 * m.belt_var, 0.0);
    let mountain = smoothstep1(0.62, 0.92, belt) * smoothstep1(-0.04, 0.08, m.cont);
    let amp_m = cfg.mtn_height * (0.55 + 0.45 * smoothstep1(-0.4, 0.6, m.belt_var)) * mountain;
    return vec2<f32>(mountain, amp_m);
}

fn hill_amplitude(m: Macro) -> f32 {
    let land = smoothstep1(-0.06, 0.05, m.cont);
    return cfg.hill_height * (0.15 + 0.85 * smoothstep1(-0.5, 0.6, m.hill_amp)) * (0.25 + 0.75 * land);
}

/// Climate (temperature °C, moisture 0..1) at a given elevation.
fn climate(m: Macro, lat: f32, elev: f32) -> vec2<f32> {
    let la = abs(lat) / (0.5 * PI);
    var t = cfg.eq_temp - cfg.pole_drop * pow(la, 1.6) + m.temp;
    t -= cfg.lapse * max(elev, 0.0) / KM;
    let latd = abs(lat) * (180.0 / PI);
    let x = (latd - 24.0) / 9.0;
    let hadley = exp(-(x * x));
    var w = 0.56 + 0.62 * m.moist;
    w -= 0.40 * hadley;
    w += 0.12 * (1.0 - smoothstep1(0.0, 0.25, m.cont));
    w -= 0.22 * smoothstep1(0.15, 0.55, m.cont);
    w -= 0.10 * smoothstep1(1500.0, 3500.0, elev);
    w += cfg.moist_bias;
    return vec2<f32>(t, clamp(w, 0.0, 1.0));
}

/// Low-passed relief at `q` (for the gully gradient).
fn low_relief(c: Ctx, m: Macro, q: vec3<f64>, mountain: f32, amp_m: f32, hill_amp: f32, gain: f32, gl: f32) -> f32 {
    var v = hill_amp * hills(q, gl, gain).x;
    if (mountain > 1e-3) {
        let wp = m.mtn_warp * 9.0 * KM;
        v += amp_m * ridged(q + vec3<f64>(c.east * wp.x + c.north * wp.y), gl, 1.6 + 0.8 * m.style.z).x;
    }
    return v;
}

/// Gradient (east, north, per metre) of the relief low-passed at half the gully wavelength.
fn low_relief_gradient(c: Ctx, m: Macro, mountain: f32, amp_m: f32, hill_amp: f32, gain: f32) -> vec2<f32> {
    let lam_e = cfg.gully_lam;
    let gl = lam_e * 0.5;
    let e = lam_e * 0.15;
    let h0 = low_relief(c, m, c.p, mountain, amp_m, hill_amp, gain, gl);
    let he = low_relief(c, m, ctx_offset(c, e, 0.0), mountain, amp_m, hill_amp, gain, gl);
    let hn = low_relief(c, m, ctx_offset(c, 0.0, e), mountain, amp_m, hill_amp, gain, gl);
    return vec2<f32>((he - h0) / e, (hn - h0) / e);
}

/// Shared domain warp of the road networks: values (w0, w1) and gradients.
struct NetWarp {
    w: vec2<f32>,
    g0: vec3<f32>,
    g1: vec3<f32>,
}

fn network_warp(c: Ctx) -> NetWarp {
    let a = fbm_d(FBM_RIVER_WARP0, c.p, c.gsd);
    let b = fbm_d(FBM_RIVER_WARP1, c.p, c.gsd);
    var n: NetWarp;
    n.w = vec2<f32>(a.x, b.x);
    n.g0 = a.yzw;
    n.g1 = b.yzw;
    return n;
}

/// A road network's noise value and gradient (east, north) after the warp.
fn network_field(f: u32, c: Ctx, nw: NetWarp, amp: f32) -> vec3<f32> {
    let qw = c.p + vec3<f64>((c.east * nw.w.x + c.north * nw.w.y) * amp);
    let ng = fbm_d(f, qw, max(c.gsd, 200.0));
    let g = ng.yzw;
    let grad = g + (nw.g0 * dot(c.east, g) + nw.g1 * dot(c.north, g)) * amp;
    return vec3<f32>(ng.x, dot(grad, c.east), dot(grad, c.north));
}

fn network_dist(v: vec3<f32>) -> f32 {
    return v.x / max(length(v.yz), 1e-30);
}

/// The smooth inputs at a coarse-grid node (`World::pre_at`).
fn pre_at(c: Ctx, m: Macro, gully: bool, roads: bool, has_cut: bool, cut: vec2<f32>) -> Pre {
    var pre = pre_none();
    if (has_cut) {
        let wp = m.mtn_warp * 9.0 * KM;
        let pw = c.p + vec3<f64>(c.east * wp.x + c.north * wp.y);
        let r = ridged_part(pw, c.gsd, 1.6 + 0.8 * m.style.z, cut.x, true, vec3<f32>(0.0, 0.0, 1.0));
        let hl = hills_part(c.p, c.gsd, 0.47 + 0.08 * m.rough, cut.y, true);
        pre.relief = vec4<f32>(r, hl.x);
        pre.relief4 = hl.y;
        pre.relief_cut = cut;
        pre.flags |= P_RELIEF;
        for (var li = 0u; li < min(cfg.nlevels, 4u); li++) {
            if (0.37 * cfg.lvl_b[li].w >= cut.y) {
                let w = meander_warp(li, c.p);
                switch li {
                    case 0u: { pre.river_warp01 = vec4<f32>(w, pre.river_warp01.zw); }
                    case 1u: { pre.river_warp01 = vec4<f32>(pre.river_warp01.xy, w); }
                    case 2u: { pre.river_warp23 = vec4<f32>(w, pre.river_warp23.zw); }
                    default: { pre.river_warp23 = vec4<f32>(pre.river_warp23.xy, w); }
                }
                pre.flags |= P_RIVER_WARP0 << li;
            }
            if (floodplain_wavelength(li) >= cut.y) {
                pre.floodplain[li] = floodplain_noise(li, c.p);
                pre.flags |= P_FLOODPLAIN0 << li;
            }
        }
        if (min(1500.0, 0.9 * f32(cfg.region_cell)) >= cut.y) {
            pre.region_warp = region_warp(c.p);
            pre.flags |= P_REGION_WARP;
        }
    }
    if (gully) {
        let mm = mountain_mask(m);
        let hill_amp = hill_amplitude(m);
        let gain = 0.47 + 0.08 * m.rough;
        pre.gully = low_relief_gradient(c, m, mm.x, mm.y, hill_amp, gain);
        pre.flags |= P_GULLY;
    }
    if (roads) {
        let nw = network_warp(c);
        pre.road_major = network_field(FBM_ROAD_MAJOR, c, nw, 2500.0);
        pre.road_minor = network_field(FBM_ROAD_MINOR, c, nw, 700.0);
        pre.flags |= P_ROADS;
    }
    if (has_cut) {
        pre.site_lake = worley3_sites(cfg.seed ^ 0x1A4Elu, c.p, 1.0lf / cfg.lake_cell, 0.85);
        let pw = c.p + vec3<f64>(region_warp(c.p));
        pre.site_region = worley3_sites(cfg.seed ^ 0x5E61lu, pw, 1.0lf / cfg.region_cell, 0.9);
        pre.site_town = worley3_sites(cfg.seed ^ 0x70E1lu, c.p, 1.0lf / cfg.town_cell, 0.8);
        pre.flags |= P_SITE_LAKE | P_SITE_REGION | P_SITE_TOWN;
    }
    if (has_cut && (pre.flags & P_GULLY) != 0u) {
        let grad = c.east * pre.gully.x + c.north * pre.gully.y;
        pre.gully_oct = gullies_part(c.p, c.up, grad, c.gsd, cut.y, true, vec4<f32>(0.0));
        pre.flags |= P_GULLY_OCT;
    }
    return pre;
}

// ---------------------------------------------------------------- pass A

/// Pass A up to the drainage: the relief (`terrain_impl` before the rivers).
struct Relief {
    h: f32,
    smooth_h: f32,
    temp0: f32,
    moist: f32,
    mountain: f32,
    ridged: f32,
    hill_amp: f32,
    micro: f32,
    mesa: f32,
    sand: f32,
    gully_n: f32,
}

fn relief(c: Ctx, m: Macro, pre: Pre) -> Relief {
    let p = c.p;
    let gsd = c.gsd;
    let s = m.cont;
    var o: Relief;

    let base = base_elevation(s);
    let plateau = smoothstep1(0.15, 0.55, m.plateau) * 900.0 * smoothstep1(0.02, 0.15, s);

    // mountain belts
    let mm = mountain_mask(m);
    let mountain = mm.x;
    let amp_m = mm.y;
    var rg = vec2<f32>(0.0);
    if (mountain > 1e-3) {
        let wp = m.mtn_warp * 9.0 * KM;
        let pw = p + vec3<f64>(c.east * wp.x + c.north * wp.y);
        let sharp = 1.6 + 0.8 * m.style.z;
        if ((pre.flags & P_RELIEF) != 0u) {
            // the long octaves from the tile's coarse grid
            let st = ridged_part(pw, gsd, sharp, pre.relief_cut.x, false, vec3<f32>(pre.relief.x, pre.relief.y, clamp(pre.relief.z, 0.0, 1.0)));
            rg = st.xy * 0.5;
        } else {
            rg = ridged(pw, gsd, sharp);
        }
    }
    let uplift = 0.22 * amp_m;
    let mtn = amp_m * rg.x;

    // hills
    let rough = m.rough;
    let hill_amp = hill_amplitude(m);
    let gain = 0.47 + 0.08 * rough;
    var hl = vec2<f32>(0.0);
    if ((pre.flags & P_RELIEF) != 0u) {
        let hs = hills_part(p, gsd, gain, pre.relief_cut.y, false);
        hl = vec2<f32>((pre.relief.w + hs.x) * 0.7, (pre.relief4 + hs.y) * 0.7);
    } else {
        hl = hills(p, gsd, gain);
    }
    let hills_h = hill_amp * hl.x;

    // erosion gullies on mountain and hill slopes
    let relief_amp = amp_m + 0.8 * hill_amp;
    let lam_e = cfg.gully_lam;
    var gully = 0.0;
    var gully_n = 0.0;
    if (cfg.erosion > 0.0 && relief_amp > 40.0 && gsd < lam_e * 0.5) {
        var gg = vec2<f32>(0.0);
        if ((pre.flags & P_GULLY) != 0u) {
            gg = pre.gully;
        } else {
            gg = low_relief_gradient(c, m, mountain, amp_m, hill_amp, gain);
        }
        let grad = c.east * gg.x + c.north * gg.y;
        let slope_l = length(gg);
        let mask = smoothstep1(0.03, 0.25, slope_l) * smoothstep1(40.0, 140.0, relief_amp);
        if (mask > 0.0) {
            var g = 0.0;
            if ((pre.flags & P_GULLY_OCT) != 0u) {
                g = gullies_part(p, c.up, grad, gsd, pre.relief_cut.y, false, pre.gully_oct).x;
            } else {
                g = gullies(p, c.up, grad, gsd);
            }
            gully_n = g * mask;
            gully = gully_n * cfg.erosion * (0.05 * amp_m + 0.12 * hill_amp);
        }
    }

    // micro relief
    var micro = 0.0;
    if (cfg.micro_height > 0.0) {
        micro = cfg.micro_height * (0.4 + 0.6 * smoothstep1(-0.3, 0.6, rough) + mountain) * fbm(FBM_MICRO, p, gsd);
    }

    var h = base + plateau + uplift + mtn + hills_h + micro + gully;
    let smooth_h = base + plateau + uplift + amp_m * rg.y * 0.6 + hill_amp * hl.y;

    // climate (from the smooth elevation, so it does not alias)
    let cl = climate(m, c.lat, max(smooth_h, 0.0));
    let temp0 = cl.x;
    let moist = cl.y;

    // mesas (arid terraces)
    let arid = 1.0 - smoothstep1(0.18, 0.42, moist);
    let mesa_noise = m.mesa;
    let mesa = arid * smoothstep1(0.1, 0.45, mesa_noise) * (1.0 - mountain) * smoothstep1(0.02, 0.1, s) * cfg.mesas;
    if (mesa > 1e-3) {
        let xb = mesa_noise * 7.0;
        let kb = floor(xb);
        let sh0 = 35.0 + 90.0 * u01(hash1(cfg.seed, i64(kb)));
        let sh1 = 35.0 + 90.0 * u01(hash1(cfg.seed, i64(kb) + 1li));
        let step = lerp(sh0, sh1, smoothstep1(0.0, 1.0, xb - kb));
        let x = h / step;
        let k = floor(x);
        let f = x - k;
        let ft = smoothstep1(0.72, 0.97, f);
        let ht = (k + ft) * step;
        h = lerp(h, ht, mesa * max(band(step, gsd), 0.35));
    }

    // sand seas with dunes
    let sand = (1.0 - smoothstep1(0.1, 0.28, moist)) * smoothstep1(6.0, 14.0, temp0) * smoothstep1(-0.25, 0.15, m.sand) * (1.0 - mountain) * smoothstep1(0.01, 0.06, s);
    if (sand > 1e-3 && cfg.dune_height > 0.0) {
        h += cfg.dune_height * sand * dunes(c, m, gsd);
    }

    o.h = h;
    o.smooth_h = smooth_h;
    o.temp0 = temp0;
    o.moist = moist;
    o.mountain = mountain;
    o.ridged = rg.x;
    o.hill_amp = hill_amp;
    o.micro = micro;
    o.mesa = mesa;
    o.sand = sand;
    o.gully_n = gully_n;
    return o;
}

/// Pass-A result at a pixel centre (`Terrain`; the land-use sites as ids and edge distance).
struct Terrain {
    ground: f32,
    water: f32,
    water_kind: u32,
    river_d: f32,
    river_hw: f32,
    river_level: f32,
    river_wet: f32,
    temp: f32,
    moist: f32,
    mountain: f32,
    rock_expect: f32,
    sand: f32,
    floodplain: f32,
    mesa: f32,
    cont: f32,
    agri: f32,
    habit: f32,
    gully: f32,
    road_major: f32,
    road_minor: f32,
    region_edge: f32,
    town: u32,
    style: vec4<f32>,
    region_id: u64,
    region_id2: u64,
}

// ---------------------------------------------------------------- drainage and lakes (batch data)

/// One drainage channel piece (`Seg`): ECEF end points, floor heights, widths.
struct Seg {
    a: vec4<f64>,
    b: vec4<f64>,
    ha: f32,
    hb: f32,
    hw: f32,
    valley: f32,
    hw_b: f32,
    level: u32,
    _p0: u32,
    _p1: u32,
}

/// A lake fed by channels ending in a closed basin: centre, radius, level (NONE_F: none).
struct Sink {
    c: vec4<f64>,
    id: u64,
    rad: f32,
    level: f32,
}

@group(1) @binding(0) var<storage, read> segs: array<Seg>;
@group(1) @binding(1) var<storage, read> seg_list: array<u32>;
@group(1) @binding(2) var<storage, read> sinks: array<Sink>;
/// lake levels by lake id (open addressing; key 0 = empty): level or NONE_F
@group(1) @binding(3) var<storage, read> lake_keys: array<u64>;
@group(1) @binding(4) var<storage, read> lake_vals: array<f32>;

/// The level of the lattice lake `id` (NONE_F: none, or not provided).
fn lake_level(id: u64) -> f32 {
    let n = arrayLength(&lake_vals);
    var k = u32(mix64(id) % u64(n));
    for (var i = 0u; i < n; i++) {
        let key = lake_keys[k];
        if (key == id) {
            return lake_vals[k];
        }
        if (key == 0lu) {
            return NONE_F;
        }
        k = (k + 1u) % n;
    }
    return NONE_F;
}

/// Is the level of lattice lake `id` known (possibly as "no lake")?
fn lake_known(id: u64) -> bool {
    let n = arrayLength(&lake_vals);
    var k = u32(mix64(id) % u64(n));
    for (var i = 0u; i < n; i++) {
        let key = lake_keys[k];
        if (key == id) {
            return true;
        }
        if (key == 0lu) {
            return false;
        }
        k = (k + 1u) % n;
    }
    return false;
}

/// Where a point's drainage data is: channel pieces `seg_list[seg0 .. seg0 + nseg]` (or
/// `segs[seg0 ..]` with DR_DIRECT), sink lakes `sinks[sink0 .. sink0 + nsink]`.
struct Drain {
    seg0: u32,
    nseg: u32,
    sink0: u32,
    nsink: u32,
    flags: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// the pieces are `segs[seg0 ..]` (no list)
const DR_DIRECT: u32 = 1u;
/// the pieces are those of a group of points: keep only those `river_segments` keeps for this
/// point (by distance)
const DR_KEEP: u32 = 2u;

/// Is the piece within `keep` of `c`?
fn seg_near(sg: Seg, c: vec3<f64>, keep: f32) -> bool {
    let ab = vec3<f32>(sg.b.xyz - sg.a.xyz);
    let ca = vec3<f32>(c - sg.a.xyz);
    let u = clamp(dot(ca, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
    return length(ca - ab * u) <= keep;
}

/// The distance within which `river_segments(p, 0, ..)` keeps a piece of level `lvl`.
fn keep_dist(lvl: u32) -> f32 {
    let cell = cfg.lvl_a[lvl].x;
    return 0.4 * cell + 1.4 * cfg.lvl_a[lvl].w + 0.35 * cfg.lvl_b[lvl].y * cell;
}

/// Piece `k` of a point's drainage data (LAT_NONE: not kept).
fn drain_seg(dr: Drain, k: u32, p: vec3<f64>) -> u32 {
    var si = dr.seg0 + k;
    if ((dr.flags & DR_DIRECT) == 0u) {
        si = seg_list[dr.seg0 + k];
    }
    if ((dr.flags & DR_KEEP) != 0u && !seg_near(segs[si], p, keep_dist(segs[si].level))) {
        return 0xffffffffu;
    }
    return si;
}

/// The warp of the region lattice lookup at `p` (from the grid when there).
fn region_warp_at(p: vec3<f64>, pre: Pre) -> vec3<f32> {
    if ((pre.flags & P_REGION_WARP) != 0u) {
        return pre.region_warp;
    }
    return region_warp(p);
}

/// The cell of the region lattice at the warped point `pw`.
fn region_cell(pw: vec3<f64>, pre: Pre) -> Cell3 {
    if ((pre.flags & P_SITE_REGION) != 0u) {
        return worley3_from(pw, cfg.region_cell, 1.0lf / cfg.region_cell, pre.site_region);
    }
    return worley3(cfg.seed ^ 0x5E61lu, pw, cfg.region_cell, 0.9);
}

/// The rest of pass A after the relief (`terrain_impl`): drainage carving, lakes, the sea,
/// climate at the final height, land use, sites and roads.
fn terrain_rest(c: Ctx, m: Macro, pre: Pre, r: Relief, mode: u32, dr: Drain) -> Terrain {
    let with_lakes = mode == MODE_FULL;
    let p = c.p;
    let gsd = c.gsd;
    let s = m.cont;
    let land = smoothstep1(-0.06, 0.05, s);
    var h = r.h;
    let mountain = r.mountain;
    let moist = r.moist;
    let micro = r.micro;

    var t: Terrain;
    t.river_d = NONE_F;
    t.river_hw = 0.0;
    t.river_level = 0.0;
    t.river_wet = 0.0;
    t.water = -NONE_F;
    t.water_kind = W_NONE;
    var floodplain = 0.0;

    // ---- rivers
    let land_fade = smoothstep1(0.0, 0.1, land);
    if (mode != MODE_RELIEF && (cfg.flags & CF_RIVERS) != 0u && land > 0.0) {
        let h0 = h;
        let wn = smoothstep1(-0.6, 0.6, m.river_width);
        var warped: array<vec3<f64>, 4>;
        var have_warp = 0u;
        var fp_noise = pre.floodplain;
        var fp_have = (pre.flags / P_FLOODPLAIN0) & 15u;
        // nearest channel: (|d| - hw), its piece and values
        var best = NONE_F;
        var best_d = 0.0;
        var best_hw = 0.0;
        var best_floor = 0.0;
        var best_lvl = 0u;
        for (var k = 0u; k < dr.nseg; k++) {
            let si = drain_seg(dr, k, p);
            if (si == 0xffffffffu) {
                continue;
            }
            let sg = segs[si];
            let li = sg.level;
            if ((have_warp & (1u << li)) == 0u) {
                var w = vec2<f32>(0.0);
                if ((pre.flags & (P_RIVER_WARP0 << li)) != 0u) {
                    w = pre_river_warp(pre, li);
                } else {
                    w = meander_warp(li, p);
                }
                let lam = cfg.lvl_b[li].w;
                warped[li] = p + vec3<f64>((c.east * w.x + c.north * w.y) * (0.22 * lam));
                have_warp |= 1u << li;
            }
            let pw = warped[li];
            let ab = vec3<f32>(sg.b.xyz - sg.a.xyz);
            let l2 = max(dot(ab, ab), 1e-9);
            let pa = vec3<f32>(pw - sg.a.xyz);
            let u = clamp(dot(pa, ab) / l2, 0.0, 1.0);
            let dv = pa - ab * u;
            let dist = length(dv);
            let floor_h = sg.ha + (sg.hb - sg.ha) * u;
            let reach = max(sg.valley * 1.5 + sg.hw + 200.0, 11.3 * sg.hw + 6.0 * (h0 - floor_h + 2.0 + 0.04 * sg.hw) + 50.0);
            if (dist > reach) {
                continue;
            }
            var sign = -1.0;
            if (dot(cross(ab, dv), c.up) >= 0.0) {
                sign = 1.0;
            }
            let rd = sign * dist;
            let hw = sg.hw + (sg.hw_b - sg.hw) * u;
            let valley_hit = sg.valley;
            // ---- carve (every channel from the uncarved height; the lowest result wins)
            let lv = cfg.lvl_a[li];
            let lvb = cfg.lvl_b[li];
            let ad = abs(rd);
            let width = 2.0 * hw;
            var fpn = 0.0;
            if ((fp_have & (1u << li)) != 0u) {
                fpn = fp_noise[li];
            } else {
                fpn = floodplain_noise(li, p);
                fp_noise[li] = fpn;
                fp_have |= 1u << li;
            }
            let fp_w = (hw + width * (1.0 + 3.0 * wn)) * (1.0 + 0.25 * fpn);
            let incision = 1.0 + 0.02 * width;
            let flo = max(floor_h, 1.0) - incision;
            let wall_k = lerp(6.0, 2.2, mountain);
            let valley = max(valley_hit, fp_w + wall_k * (h0 - flo));
            if (h0 > flo && h0 > -4.0 && ad < valley) {
                var wall = smoothstep1(fp_w, valley, ad);
                wall = wall * wall * (3.0 - 2.0 * wall);
                let fpl = max(flo + 0.8, 0.5) + 0.4 * abs(micro);
                var tgt = fpl;
                if (ad < hw) {
                    tgt = flo - 0.8 - 0.02 * width;
                }
                var carved = min(h0, lerp(tgt, h0, wall));
                carved = max(carved, h0 - lvb.z * (1.0 - 0.3 * wall));
                var fade = 1.0;
                if (width < 0.3 * gsd) {
                    fade = smoothstep1(0.25, 0.5, valley_hit / gsd);
                }
                let shelf = smoothstep1(-4.0, 0.5, h0);
                h = min(h, h0 + (carved - h0) * fade * land_fade * shelf);
                floodplain = max(floodplain, (1.0 - wall) * land_fade * smoothstep1(20.0, 120.0, width) * fade);
            }
            let key = ad - hw;
            if (key < best) {
                best = key;
                best_d = rd;
                best_hw = hw;
                best_floor = floor_h;
                best_lvl = li;
            }
        }
        // channel attributes from the nearest channel
        if (best < NONE_F) {
            let ad = abs(best_d);
            let flo = max(best_floor, 1.0) - (1.0 + 0.04 * best_hw);
            t.river_d = best_d;
            t.river_hw = best_hw;
            if (ad < best_hw) {
                t.river_level = max(min(flo, h + 1.0 + 0.06 * best_hw), h + 0.6);
            } else {
                t.river_level = h;
            }
            let wm = cfg.lvl_b[best_lvl].x;
            t.river_wet = smoothstep1(wm, wm + 0.12, moist);
        }
    }

    // ---- lakes (Worley cells; flat surface at the basin spill height)
    let lake_cell = f32(cfg.lake_cell);
    if (with_lakes && cfg.lake_density > 0.0 && land > 0.3 && lake_cell > gsd) {
        var wc: Cell3;
        if ((pre.flags & P_SITE_LAKE) != 0u) {
            wc = worley3_from(p, cfg.lake_cell, 1.0lf / cfg.lake_cell, pre.site_lake);
        } else {
            wc = worley3(cfg.seed ^ 0x1A4Elu, p, cfg.lake_cell, 0.85);
        }
        for (var k = 0; k < 2; k++) {
            var id = wc.id;
            var pt = wc.point;
            if (k == 1) {
                id = wc.id2;
                pt = wc.point2;
            }
            let prob = cfg.lake_density * (0.3 + 0.9 * moist) * (1.0 - 0.8 * mountain);
            if (u01k(id, 1lu) > prob) {
                continue;
            }
            let rad = min(300.0 * exp(u01k(id, 2lu) * 2.7), lake_cell * 0.3);
            // centre on the surface below the 3D feature point
            let plen = sqrt(dot(p, p));
            let pc = pt * (plen / sqrt(dot(pt, pt)));
            let d = dist64(p, pc);
            if (d > rad * 1.5) {
                continue;
            }
            let level = lake_level(id);
            if (level >= NONE_F) {
                continue;
            }
            let lw = rad * 0.6;
            let warpn = perlin3(id, p / f64(lw)) * 0.3 + perlin3(id ^ 7lu, p / f64(lw * 0.3)) * 0.12 * band(lw * 0.3, gsd);
            let de = d / rad * (1.0 + warpn);
            let depth = 3.0 + 0.01 * rad;
            if (de < 1.0) {
                let bowl = level - depth * (1.0 - de * de);
                h = min(h, lerp(h, bowl, smoothstep1(1.0, 0.7, de)));
            }
            if (de < 1.0 && h < level) {
                t.water = max(t.water, level);
                t.water_kind = W_LAKE;
            } else if (de < 1.3 && h < level + 0.8) {
                h = lerp(level + 0.8, h, smoothstep1(1.0, 1.3, de));
            }
        }
    }
    // lakes at the end of rivers that drain into a closed basin
    if (with_lakes && (cfg.flags & CF_RIVERS) != 0u && land > 0.0 && mode != MODE_RELIEF) {
        for (var k = 0u; k < dr.nsink; k++) {
            let sk = sinks[dr.sink0 + k];
            let rad = sk.rad;
            if (dist64(p, sk.c.xyz) >= 1.6 * rad) {
                continue;
            }
            let level = sk.level;
            if (level >= NONE_F) {
                continue;
            }
            let id = sk.id;
            let plen = sqrt(dot(p, p));
            let pc = sk.c.xyz * (plen / sqrt(dot(sk.c.xyz, sk.c.xyz)));
            let up = vec3<f32>(pc / plen);
            let ax = normalize_or_zero(cross(up, vec3<f32>(u01k(id, 2lu) - 0.5, u01k(id, 3lu) - 0.5, u01k(id, 4lu) - 0.5)));
            let el = 1.0 + 1.2 * u01k(id, 5lu);
            let dv = vec3<f32>(p - pc);
            let da = dot(dv, ax);
            let db = length(dv - ax * da);
            let d = sqrt(da * da / el + db * db * el);
            let lw = rad * 0.6;
            let warpn = perlin3(id, p / f64(1.4 * lw)) * 0.45
                + perlin3(id ^ 7lu, p / f64(lw * 0.45)) * 0.18 * band(lw * 0.45, gsd)
                + perlin3(id ^ 9lu, p / f64(lw * 0.12)) * 0.06 * band(lw * 0.12, gsd);
            let de = d / rad * (1.0 + warpn);
            let depth = 3.0 + 0.01 * rad;
            let near = 1.0 - smoothstep1(1.5, 4.0, h - level);
            if (de < 1.0 && near > 0.0) {
                let bowl = level - depth * (1.0 - de * de) - 0.3;
                h = min(h, lerp(h, bowl, near));
                if (h < level) {
                    t.water = max(t.water, level);
                    t.water_kind = W_LAKE;
                }
            } else if (de < 1.5 && h < level + 4.0) {
                let shore = level + 0.4 + 12.0 * (de - 1.0) * (de - 1.0);
                h = min(h, lerp(shore, h, smoothstep1(1.0, 1.5, de)));
            }
        }
    }

    // ---- ocean
    if (h < 0.0 && t.water_kind == W_NONE) {
        t.water = 0.0;
        t.water_kind = W_OCEAN;
    }

    // ---- climate at the actual elevation
    let temp = r.temp0 - cfg.lapse * (max(h, 0.0) - max(r.smooth_h, 0.0)) / KM;

    // ---- land-use suitability
    let an = m.agri;
    let climate_ok = smoothstep1(2.0, 8.0, temp) * (1.0 - smoothstep1(27.0, 31.0, temp));
    let wet_ok = smoothstep1(0.22, 0.42, moist);
    let irrig = (1.0 - wet_ok) * smoothstep1(0.15, 0.6, an) * smoothstep1(14.0, 20.0, temp);
    let agri = clamp(climate_ok * (wet_ok + 0.7 * irrig) * (0.35 + 0.65 * smoothstep1(-0.6, 0.2, an))
        * (1.0 - mountain * 0.9)
        * (1.0 - 0.75 * smoothstep1(120.0, 320.0, r.hill_amp * (0.6 + 0.8 * smoothstep1(-0.3, 0.6, m.rough))))
        * cfg.agriculture, 0.0, 1.0);
    let habit = climate_ok * (0.4 + 0.6 * wet_ok) * (1.0 - mountain) * land;
    let style = clamp(vec4<f32>(0.5) + 0.5 * m.style * 1.4, vec4<f32>(0.0), vec4<f32>(1.0));

    // ---- land-use sites
    let region_cell_m = f32(cfg.region_cell);
    t.region_id = 0lu;
    t.region_id2 = 0lu;
    t.region_edge = NONE_F;
    if (mode != MODE_RELIEF && gsd < region_cell_m * 0.5) {
        let wq = region_warp_at(p, pre);
        let pw = p + vec3<f64>(wq);
        let wc = region_cell(pw, pre);
        t.region_id = wc.id;
        t.region_id2 = wc.id2;
        t.region_edge = worley_edge_dist(wc, pw);
    }
    t.town = 0u;
    let town_cell = f32(cfg.town_cell);
    if (mode != MODE_RELIEF && gsd < town_cell * 0.25 && cfg.towns > 0.0) {
        var wc: Cell3;
        if ((pre.flags & P_SITE_TOWN) != 0u) {
            wc = worley3_from(p, cfg.town_cell, 1.0lf / cfg.town_cell, pre.site_town);
        } else {
            wc = worley3(cfg.seed ^ 0x70E1lu, p, cfg.town_cell, 0.8);
        }
        t.town = select(0u, 1u, wc.id != 0lu);
    }

    // ---- road networks, only where people live
    t.road_major = NONE_F;
    t.road_minor = NONE_F;
    if (mode != MODE_RELIEF && cfg.roads > 0.0 && habit > 0.02 && gsd < 400.0) {
        if ((pre.flags & P_ROADS) != 0u) {
            t.road_major = network_dist(pre.road_major);
            if (gsd < 200.0) {
                t.road_minor = network_dist(pre.road_minor);
            }
        } else {
            let nw = network_warp(c);
            t.road_major = network_dist(network_field(FBM_ROAD_MAJOR, c, nw, 2500.0));
            if (gsd < 200.0) {
                t.road_minor = network_dist(network_field(FBM_ROAD_MINOR, c, nw, 700.0));
            }
        }
    }

    t.ground = h;
    t.gully = r.gully_n;
    t.temp = temp;
    t.moist = moist;
    t.mountain = mountain;
    t.rock_expect = clamp(mountain * smoothstep1(0.15, 0.6, r.ridged) + 0.25 * r.mesa, 0.0, 1.0);
    t.sand = r.sand;
    t.floodplain = floodplain;
    t.mesa = r.mesa;
    t.cont = s;
    t.agri = agri;
    t.habit = habit;
    t.style = style;
    return t;
}
