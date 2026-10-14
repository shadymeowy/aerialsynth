// The agriculture kit (`kits/agriculture.rs`): field systems by culture × climate × terrain.
// Same functions, same order of operations as the Rust; the tables are generated from it.

struct AgSys {
    part: u32,
    size: vec2<f32>,
    kinds: vec4<u32>,
    kw: vec4<f32>,
    borders: vec2<u32>,
    bp: vec2<f32>,
    slope: vec2<f32>,
    terraces: f32,
    trees: f32,
}

struct AgMix {
    crops: vec4<u32>,
    w: vec4<f32>,
}

// BEGIN GENERATED (kits::agriculture::wgsl_tables; AERIALSYNTH_BLESS=1 cargo test -p terragen --lib agriculture)
fn agri_arch_w(a: u32) -> array<f32, 15> {
    switch a {
        case 0u: { return array<f32, 15>(0.08, 0.0, 0.0, 0.0, 0.0, 0.5, 0.12, 0.06, 0.0, 0.04, 0.08, 0.0, 0.12, 0.0, 0.02); }
        case 1u: { return array<f32, 15>(0.1, 0.0, 0.0, 0.0, 0.0, 0.15, 0.1, 0.04, 0.0, 0.02, 0.5, 0.0, 0.09, 0.0, 0.04); }
        case 2u: { return array<f32, 15>(0.08, 0.0, 0.0, 0.0, 0.0, 0.32, 0.0, 0.02, 0.0, 0.0, 0.02, 0.0, 0.5, 0.06, 0.15); }
        case 3u: { return array<f32, 15>(0.08, 0.05, 0.0, 0.0, 0.0, 0.1, 0.0, 0.04, 0.02, 0.22, 0.0, 0.09, 0.05, 0.35, 0.06); }
        case 4u: { return array<f32, 15>(0.12, 0.04, 0.42, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.07, 0.18); }
        case 5u: { return array<f32, 15>(0.12, 0.0, 0.0, 0.04, 0.0, 0.06, 0.0, 0.16, 0.3, 0.25, 0.0, 0.08, 0.06, 0.03, 0.05); }
        case 6u: { return array<f32, 15>(0.28, 0.0, 0.02, 0.2, 0.2, 0.0, 0.0, 0.02, 0.08, 0.06, 0.0, 0.03, 0.14, 0.0, 0.0); }
        case 7u: { return array<f32, 15>(0.1, 0.66, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.05, 0.0, 0.02, 0.09, 0.04, 0.16); }
        case 8u: { return array<f32, 15>(0.1, 0.0, 0.0, 0.0, 0.0, 0.15, 0.6, 0.04, 0.0, 0.02, 0.05, 0.02, 0.02, 0.0, 0.0); }
        case 9u: { return array<f32, 15>(0.4, 0.0, 0.0, 0.08, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.45, 0.0, 0.0); }
        case 10u: { return array<f32, 15>(0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0); }
        default: { return array<f32, 15>(0.06, 0.0, 0.0, 0.06, 0.0, 0.24, 0.08, 0.38, 0.0, 0.02, 0.02, 0.0, 0.14, 0.0, 0.04); }
    }
}
fn agri_sys(i: u32) -> AgSys {
    switch i {
        case 0u: { return AgSys(0u, vec2<f32>(200.0, 200.0), vec4<u32>(0u, 1u, 2u, 3u), vec4<f32>(0.7, 0.15, 0.1, 0.05), vec2<u32>(0u, 0u), vec2<f32>(0.0, 0.0), vec2<f32>(0.22, 0.32), 0.0, 0.0); }
        case 1u: { return AgSys(4u, vec2<f32>(402.336, 402.336), vec4<u32>(0u, 2u, 1u, 11u), vec4<f32>(0.84, 0.07, 0.07, 0.02), vec2<u32>(2u, 4u), vec2<f32>(0.06, 0.08), vec2<f32>(0.22, 0.32), 0.0, 0.0); }
        case 2u: { return AgSys(0u, vec2<f32>(1300.0, 900.0), vec4<u32>(0u, 1u, 2u, 3u), vec4<f32>(0.68, 0.22, 0.06, 0.04), vec2<u32>(2u, 5u), vec2<f32>(0.82, 0.1), vec2<f32>(0.22, 0.32), 0.0, 0.0); }
        case 3u: { return AgSys(2u, vec2<f32>(600.0, 34.0), vec4<u32>(0u, 1u, 3u, 2u), vec4<f32>(0.72, 0.12, 0.12, 0.04), vec2<u32>(0u, 0u), vec2<f32>(1.0, 0.0), vec2<f32>(0.22, 0.32), 0.0, 0.0); }
        case 4u: { return AgSys(1u, vec2<f32>(260.0, 1.5), vec4<u32>(2u, 0u, 3u, 1u), vec4<f32>(0.5, 0.3, 0.14, 0.06), vec2<u32>(1u, 3u), vec2<f32>(0.55, 0.06), vec2<f32>(0.3, 0.42), 0.0, 0.0); }
        case 5u: { return AgSys(1u, vec2<f32>(64.0, 1.8), vec4<u32>(0u, 1u, 2u, 12u), vec4<f32>(0.6, 0.24, 0.1, 0.06), vec2<u32>(0u, 1u), vec2<f32>(0.6, 0.08), vec2<f32>(0.3, 0.42), 0.0, 0.3); }
        case 6u: { return AgSys(0u, vec2<f32>(130.0, 90.0), vec4<u32>(9u, 11u, 0u, 12u), vec4<f32>(0.9, 0.025, 0.065, 0.01), vec2<u32>(6u, 7u), vec2<f32>(0.75, 0.25), vec2<f32>(0.7, 0.9), 1.0, 0.0); }
        case 7u: { return AgSys(1u, vec2<f32>(130.0, 1.3), vec4<u32>(0u, 5u, 1u, 4u), vec4<f32>(0.42, 0.28, 0.2, 0.1), vec2<u32>(3u, 0u), vec2<f32>(0.75, 0.25), vec2<f32>(0.7, 0.9), 1.0, 0.0); }
        case 8u: { return AgSys(0u, vec2<f32>(140.0, 70.0), vec4<u32>(6u, 5u, 4u, 1u), vec4<f32>(0.7, 0.1, 0.1, 0.1), vec2<u32>(5u, 0u), vec2<f32>(0.15, 0.85), vec2<f32>(0.6, 0.8), 0.0, 0.0); }
        case 9u: { return AgSys(1u, vec2<f32>(240.0, 1.4), vec4<u32>(5u, 4u, 0u, 1u), vec4<f32>(0.55, 0.2, 0.15, 0.1), vec2<u32>(3u, 0u), vec2<f32>(0.5, 0.5), vec2<f32>(0.6, 0.8), 0.0, 0.0); }
        case 10u: { return AgSys(0u, vec2<f32>(480.0, 360.0), vec4<u32>(7u, 13u, 8u, 0u), vec4<f32>(0.45, 0.22, 0.15, 0.18), vec2<u32>(5u, 0u), vec2<f32>(0.9, 0.1), vec2<f32>(0.6, 0.8), 0.0, 0.0); }
        case 11u: { return AgSys(0u, vec2<f32>(95.0, 60.0), vec4<u32>(10u, 0u, 1u, 11u), vec4<f32>(0.76, 0.13, 0.09, 0.02), vec2<u32>(5u, 0u), vec2<f32>(0.25, 0.75), vec2<f32>(0.22, 0.32), 0.0, 0.0); }
        case 12u: { return AgSys(0u, vec2<f32>(420.0, 300.0), vec4<u32>(2u, 3u, 0u, 11u), vec4<f32>(0.6, 0.22, 0.16, 0.02), vec2<u32>(0u, 1u), vec2<f32>(0.8, 0.2), vec2<f32>(0.38, 0.5), 0.0, 0.0); }
        case 13u: { return AgSys(3u, vec2<f32>(805.0, 805.0), vec4<u32>(0u, 1u, 0u, 0u), vec4<f32>(0.75, 0.25, 0.0, 0.0), vec2<u32>(6u, 6u), vec2<f32>(1.0, 0.0), vec2<f32>(0.22, 0.32), 0.0, 0.0); }
        default: { return AgSys(0u, vec2<f32>(2200.0, 1500.0), vec4<u32>(2u, 3u, 1u, 11u), vec4<f32>(0.86, 0.04, 0.08, 0.02), vec2<u32>(5u, 0u), vec2<f32>(0.4, 0.6), vec2<f32>(0.38, 0.5), 0.0, 0.0); }
    }
}
/// A crop: linear colours in spring, summer, autumn; (row spacing, contrast, height).
fn agri_crop(i: u32) -> array<vec3<f32>, 4> {
    switch i {
        case 0u: { return array<vec3<f32>, 4>(vec3<f32>(0.122138776, 0.19461784, 0.045186203), vec3<f32>(0.55201143, 0.4125426, 0.13843161), vec3<f32>(0.49102086, 0.40197778, 0.23074006), vec3<f32>(1.1, 0.1, 0.8)); }
        case 1u: { return array<vec3<f32>, 4>(vec3<f32>(0.16202937, 0.13286832, 0.07227185), vec3<f32>(0.09758735, 0.16202937, 0.048171826), vec3<f32>(0.4452012, 0.3515326, 0.15592647), vec3<f32>(0.76, 0.14, 2.2)); }
        case 2u: { return array<vec3<f32>, 4>(vec3<f32>(0.16202937, 0.13286832, 0.08021982), vec3<f32>(0.11193243, 0.18116425, 0.051269457), vec3<f32>(0.4125426, 0.26225066, 0.07618538), vec3<f32>(0.5, 0.1, 0.8)); }
        case 3u: { return array<vec3<f32>, 4>(vec3<f32>(0.67244315, 0.5647115, 0.064803265), vec3<f32>(0.14995979, 0.15592647, 0.054480277), vec3<f32>(0.26225066, 0.18782078, 0.10224173), vec3<f32>(0.9, 0.08, 1.2)); }
        case 4u: { return array<vec3<f32>, 4>(vec3<f32>(0.12743768, 0.1746474, 0.051269457), vec3<f32>(0.67244315, 0.45641103, 0.029556835), vec3<f32>(0.08865558, 0.061246052, 0.034339808), vec3<f32>(0.7, 0.1, 1.8)); }
        case 5u: { return array<vec3<f32>, 4>(vec3<f32>(0.20863687, 0.13843161, 0.08437621), vec3<f32>(0.08865558, 0.14412847, 0.048171826), vec3<f32>(0.2158605, 0.14412847, 0.08865558), vec3<f32>(0.75, 0.2, 0.6)); }
        case 6u: { return array<vec3<f32>, 4>(vec3<f32>(0.093058966, 0.19461784, 0.039546236), vec3<f32>(0.16202937, 0.24620132, 0.054480277), vec3<f32>(0.122138776, 0.19461784, 0.048171826), vec3<f32>(6.5, 0.1, 0.5)); }
        case 7u: { return array<vec3<f32>, 4>(vec3<f32>(0.26225066, 0.16202937, 0.093058966), vec3<f32>(0.10224173, 0.1746474, 0.048171826), vec3<f32>(0.4677838, 0.40197778, 0.2541521), vec3<f32>(1.0, 0.12, 1.0)); }
        case 8u: { return array<vec3<f32>, 4>(vec3<f32>(0.122138776, 0.23074006, 0.048171826), vec3<f32>(0.093058966, 0.2158605, 0.04231141), vec3<f32>(0.20155625, 0.24620132, 0.064803265), vec3<f32>(1.5, 0.08, 3.0)); }
        case 9u: { return array<vec3<f32>, 4>(vec3<f32>(0.16202937, 0.14995979, 0.116970666), vec3<f32>(0.18782078, 0.13843161, 0.2704978), vec3<f32>(0.13843161, 0.13843161, 0.107023105), vec3<f32>(1.6, 0.3, 0.6)); }
        case 10u: { return array<vec3<f32>, 4>(vec3<f32>(0.13843161, 0.18116425, 0.054480277), vec3<f32>(0.33245152, 0.13843161, 0.054480277), vec3<f32>(0.43415365, 0.3049873, 0.14995979), vec3<f32>(0.75, 0.1, 1.8)); }
        default: { return array<vec3<f32>, 4>(vec3<f32>(0.09758735, 0.18116425, 0.04231141), vec3<f32>(0.08437621, 0.1682694, 0.03688945), vec3<f32>(0.12743768, 0.1746474, 0.051269457), vec3<f32>(1.0, 0.15, 1.5)); }
    }
}
/// A crop mix: crops and shares.
fn agri_mix(i: u32) -> AgMix {
    switch i {
        case 0u: { return AgMix(vec4<u32>(0u, 1u, 3u, 5u), vec4<f32>(0.42, 0.24, 0.14, 0.2)); }
        case 1u: { return AgMix(vec4<u32>(1u, 2u, 0u, 6u), vec4<f32>(0.45, 0.35, 0.12, 0.08)); }
        case 2u: { return AgMix(vec4<u32>(0u, 4u, 6u, 3u), vec4<f32>(0.55, 0.3, 0.1, 0.05)); }
        case 3u: { return AgMix(vec4<u32>(0u, 4u, 9u, 5u), vec4<f32>(0.52, 0.22, 0.05, 0.21)); }
        case 4u: { return AgMix(vec4<u32>(1u, 11u, 10u, 8u), vec4<f32>(0.35, 0.3, 0.2, 0.15)); }
        default: { return AgMix(vec4<u32>(6u, 0u, 7u, 1u), vec4<f32>(0.4, 0.3, 0.2, 0.1)); }
    }
}
// END GENERATED

const AG_CORE: u32 = 0u;
const AG_SECTION: u32 = 1u;
const AG_SHELTER: u32 = 2u;
const AG_PADDY: u32 = 6u;
const AG_TERRACE: u32 = 7u;
const AG_GREENH: u32 = 11u;
const AG_PASTURE: u32 = 12u;
const AG_RANCH: u32 = 14u;
const AG_N: u32 = 15u;

const AK_CROP: u32 = 0u;
const AK_FALLOW: u32 = 1u;
const AK_PASTURE: u32 = 2u;
const AK_HAY: u32 = 3u;
const AK_ORCHARD: u32 = 4u;
const AK_OLIVE: u32 = 5u;
const AK_VINE: u32 = 6u;
const AK_PALM: u32 = 7u;
const AK_TEA: u32 = 8u;
const AK_PADDY: u32 = 9u;
const AK_GREENH: u32 = 10u;
const AK_POND: u32 = 11u;
const AK_YARD: u32 = 12u;
const AK_RUBBER: u32 = 13u;

const AP_GRID: u32 = 0u;
const AP_VOR: u32 = 1u;
const AP_STRIP: u32 = 2u;
const AP_PIVOT: u32 = 3u;
const AP_SECTION: u32 = 4u;

const AB_MARGIN: u32 = 0u;
const AB_HEDGE: u32 = 1u;
const AB_BELT: u32 = 2u;
const AB_WALL: u32 = 3u;
const AB_DITCH: u32 = 4u;
const AB_TRACK: u32 = 5u;
const AB_CANAL: u32 = 7u;

const AG_MILE: f32 = 1609.344;
const AF_MED: u32 = 1u;
const AF_DESERT: u32 = 2u;
const AF_TROPIC: u32 = 4u;

/// What the kit reads of an ecoregion (`EcoV`).
struct EcoV {
    id: u64,
    arch: u32,
    koppen: u32,
    temp: f32,
    precip: f32,
    dev: f32,
    scale: f32,
    hedges: f32,
}

struct Plan {
    ok: bool,
    sys: u32,
    e: EcoV,
    seed: u64,
    mix: u32,
    flags: u32,
    season: f32,
    scale: f32,
}

fn rotl64(x: u64, k: u32) -> u64 {
    return (x << k) | (x >> (64u - k));
}

/// `weights`.
fn agri_weights(e: EcoV) -> array<f32, 15> {
    var w = agri_arch_w(min(e.arch, 11u));
    let k = e.koppen;
    var m = array<f32, 15>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    if (k >= 7u && k <= 9u) {
        m[8] = 3.0;
        m[9] = 3.0;
        m[7] = 1.5;
        m[11] = 2.0;
        m[6] = 0.2;
        m[4] = 0.2;
        m[1] = 0.6;
        m[10] = 0.0;
        m[14] = 0.6;
    } else if (k == 14u || k == 15u) {
        m[4] = 2.5;
        m[12] = 1.6;
        m[8] = 0.4;
        m[9] = 0.5;
        m[6] = 0.1;
        m[13] = 0.0;
        m[14] = 0.0;
    } else if (k == 10u || k == 13u) {
        m[6] = 1.8;
        m[10] = 1.2;
        m[14] = 0.5;
    } else if (k <= 2u) {
        var wet = 1.0;
        if (k <= 1u) {
            wet = 1.5;
        }
        m[10] = wet;
        m[6] = 1.4;
        m[8] = 0.05;
        m[11] = 0.2;
        m[2] = 0.0;
        m[1] = 0.4;
        m[4] = 0.0;
        m[3] = 0.2;
        m[14] = select(1.0, 0.3, k <= 1u);
    } else if (k >= 3u && k <= 6u) {
        let desert = k <= 4u;
        m[13] = select(2.0, 4.0, desert);
        m[12] = 1.5;
        m[6] = 0.05;
        m[4] = 0.0;
        m[10] = 0.0;
        m[8] = 0.5;
        m[11] = select(1.0, 1.5, desert);
        m[14] = select(3.0, 0.3, desert);
    } else if (k >= 16u && k <= 27u) {
        m[2] = 1.6;
        m[6] = 0.15;
        m[10] = 0.0;
        m[9] = 0.4;
        m[8] = 0.4;
        m[4] = 0.6;
        m[14] = 0.7;
    } else if (k >= 28u) {
        for (var i = 0u; i < AG_N; i++) {
            if (i != AG_CORE && i != AG_PASTURE) {
                m[i] = 0.0;
            }
        }
    }
    let dev = e.dev;
    m[11] *= 0.3 + 1.4 * dev;
    m[5] *= 1.6 - 1.0 * dev;
    m[1] *= 0.4 + 1.0 * dev;
    m[13] *= (0.4 + 1.0 * dev) * (0.1 + 0.9 * (1.0 - smoothstep1(700.0, 1200.0, e.precip)));
    var total = 0.0;
    for (var i = 0u; i < AG_N; i++) {
        w[i] *= m[i] * (0.4 + 1.2 * u01k(e.id, 0xA600lu + u64(i)));
        total += w[i];
    }
    let v = u01k(e.id, 0xA6FFlu) * total;
    var acc = 0.0;
    for (var i = 0u; i < AG_N; i++) {
        acc += w[i];
        if (v < acc) {
            w[i] *= 3.0;
            break;
        }
    }
    return w;
}

/// `pick_system`.
fn agri_pick_system(e: EcoV, rid: u64) -> u32 {
    let w = agri_weights(e);
    var total = 0.0;
    for (var i = 0u; i < AG_N; i++) {
        total += w[i];
    }
    let v = u01k(mix64(rid ^ rotl64(e.id, 29u)), 0xA61lu) * total;
    var acc = 0.0;
    for (var i = 0u; i < AG_N; i++) {
        acc += w[i];
        if (v < acc) {
            return i;
        }
    }
    return AG_CORE;
}

fn agri_crop_mix(e: EcoV) -> u32 {
    if (e.temp > 20.0 && e.precip > 900.0) {
        return 4u;
    } else if (e.precip < 400.0) {
        return 5u;
    } else if (e.koppen >= 7u && e.koppen <= 9u) {
        return 3u;
    } else if (e.precip < 650.0 && e.temp < 12.0) {
        return 2u;
    } else if (e.arch == 7u) {
        return 1u;
    }
    return 0u;
}

/// `plan`.
fn agri_plan(s: ptr<function, Stack>) -> Plan {
    var pl: Plan;
    pl.ok = false;
    if (!(*s).has_region || (*s).t.region_id == 0lu || cfg.agriculture <= 0.0 || s_agri(s) <= 0.02) {
        return pl;
    }
    var id = (*s).t.eco_id;
    if ((*s).l.eco_edge < 0.0) {
        id = (*s).t.eco_id2;
    }
    if (id == 0lu) {
        return pl;
    }
    let ke = eco_find(id);
    if (ke < 0) {
        return pl;
    }
    let eo = eco_vals[ke];
    var e: EcoV;
    e.id = eo.id;
    e.arch = eo.arch;
    e.koppen = u32(eo.clim.z + 0.5);
    e.temp = eo.misc.z;
    e.precip = eo.misc.w;
    e.dev = eo.misc.y;
    e.scale = eo.land.w;
    e.hedges = eo.town.x;
    let sy = agri_pick_system(e, (*s).t.region_id);
    if (sy == AG_CORE) {
        return pl;
    }
    let r = (*s).r;
    pl.ok = true;
    pl.sys = sy;
    pl.e = e;
    pl.seed = mix64(r.split ^ 0xA6A6lu);
    var flags = 0u;
    if ((e.koppen >= 7u && e.koppen <= 9u) || (e.temp > 13.0 && e.precip < 900.0 && e.koppen > 6u)) {
        flags |= AF_MED;
    }
    if (e.temp > 18.0 && e.precip < 350.0) {
        flags |= AF_DESERT;
    }
    if (e.temp > 20.0 && e.precip > 900.0) {
        flags |= AF_TROPIC;
    }
    pl.mix = agri_crop_mix(e);
    pl.flags = flags;
    pl.season = r.season;
    pl.scale = sqrt(e.scale) * (0.75 + 0.5 * u01k(pl.seed, 3lu));
    return pl;
}

// ---------------------------------------------------------------- layers

fn agriculture_slot_disturbance(s: ptr<function, Stack>) {
    let pl = agri_plan(s);
    if (pl.ok) {
        (*s).m.flat_ok *= 0.25;
    }
}

fn agriculture_slot_agriculture(s: ptr<function, Stack>) {
    let pl = agri_plan(s);
    if (!pl.ok) {
        return;
    }
    (*s).m.flat_ok *= 4.0;
    agri_draw(s, pl);
}

fn agriculture_slot_linear(s: ptr<function, Stack>) {
    let pl = agri_plan(s);
    if (pl.ok && pl.sys == AG_RANCH) {
        agri_feedlot(s, pl);
        return;
    }
    if (!pl.ok || pl.sys != AG_SECTION) {
        return;
    }
    let q = (*s).q_loc;
    let gsd = (*s).c.gsd;
    let ix = round(q.x / AG_MILE);
    let iy = round(q.y / AG_MILE);
    let d = min(abs(q.x - ix * AG_MILE), abs(q.y - iy * AG_MILE));
    let steep = smoothstep1(-0.02, 0.02, 0.45 - (*s).l.slope);
    let cov = band_cov(d, 4.0, max((*s).l.fw, 0.5 * gsd)) * steep * (1.0 - (*s).m.snow);
    if (cov <= 0.0 || (*s).cls == LC_LAKE) {
        return;
    }
    let col = srgb(168.0, 150.0, 122.0) * (1.0 + 0.05 * (*s).pf.f[PF_DETAIL]);
    let cls = LC_TRACK;
    (*s).m.road_cov = max((*s).m.road_cov, cov);
    var ly = layer_paint(cov, col, cls);
    ly.hmode = HM_BLEND;
    ly.relit = 0.5;
    ly.clear = 1.0;
    composite(s, ly);
}

/// `feedlot`: pens, feed alleys and bunks, cattle, a feed mill, manure lagoons (ranches).
fn agri_feedlot(s: ptr<function, Stack>, pl: Plan) {
    let seed = pl.seed;
    if (u01k(seed, 0xFEE0lu) >= 0.2 + 0.45 * pl.e.dev) {
        return;
    }
    let c = vec2<f32>(1600.0 * (u01k(seed, 0xFEE1lu) - 0.5), 1000.0 * (u01k(seed, 0xFEE2lu) - 0.5));
    let hx = 220.0 + 260.0 * u01k(seed, 0xFEE3lu);
    let hy = 140.0 + 160.0 * u01k(seed, 0xFEE4lu);
    let lag = 130.0 + 90.0 * u01k(seed, 0xFEE5lu);
    let d = (*s).q_rot - c;
    if (d.x < -hx - 30.0 || d.x > hx + lag + 30.0 || abs(d.y) > hy + 30.0 || (*s).l.slope > 0.15 || (*s).cls == LC_LAKE) {
        return;
    }
    let gsd = (*s).c.gsd;
    let p = (*s).c.p;
    let fw = max((*s).l.fw, 0.5 * gsd);
    let soil = spal(s, BP_SOIL + 1u);
    var col: vec3<f32>;
    var dh = 0.0;
    var cls: u32;
    var cov: f32;
    if (d.x < hx + 20.0) {
        cov = band_cov(d.x, hx, fw) * band_cov(d.y, hy, fw);
        let pen_x = 60.0;
        let pen_y = 40.0;
        let alley = 12.0;
        let per = 2.0 * pen_y + alley;
        let yy = rem_euclid(d.y + hy, per);
        let pid = hash2(seed ^ 0xFEE6lu, i64(floor(d.x / pen_x)), i64(floor((d.y + hy) / per)) * 2li + select(0li, 1li, yy > pen_y));
        let manure = srgb(64.0, 52.0, 40.0) * (0.85 + 0.3 * u01k(pid, 1lu));
        var c0 = manure * (1.0 + 0.1 * perlin3(seed ^ 0xFEE7lu, p * (1.0lf / 9.0lf)) * band(9.0, gsd));
        let cell = 2.6;
        let ci = i64(floor(d.x / cell));
        let cj = i64(floor(d.y / cell));
        let hc = hash2(seed ^ 0xFEE8lu, ci, cj);
        if (u01k(hc, 1lu) < 0.22) {
            let kc = u32(u01k(hc, 2lu) * 2.99);
            var cc = srgb(30.0, 26.0, 22.0);
            if (kc == 1u) {
                cc = srgb(96.0, 60.0, 40.0);
            } else if (kc == 2u) {
                cc = srgb(196.0, 186.0, 170.0);
            }
            let off = vec2<f32>(d.x - (f32(ci) + 0.5) * cell, d.y - (f32(cj) + 0.5) * cell);
            c0 = mixc(c0, cc, band_cov(length(off), 0.8, fw) * band(cell, gsd));
        }
        let alley_c = srgb(168.0, 160.0, 146.0);
        let in_alley = band_cov(yy - (pen_y + 0.5 * alley), 0.5 * alley, fw);
        let bunk = min(band_cov(yy - (pen_y - 1.5), 1.5, fw) + band_cov(yy - (pen_y + alley + 1.5), 1.5, fw), 1.0);
        let fx = abs(d.x / pen_x - round(d.x / pen_x)) * pen_x;
        let fence = max(band_cov(fx, 0.25, fw), band_cov(min(yy, per - yy), 0.25, fw)) * band(1.0, gsd);
        col = mixc(c0, alley_c, in_alley * band(alley, gsd));
        col = mixc(col, srgb(196.0, 194.0, 186.0), bunk * band(3.0, gsd));
        col = mixc(col, srgb(150.0, 140.0, 126.0), fence * (1.0 - in_alley));
        cls = LC_FARMYARD;
        dh = 0.3 * bunk;
        if (d.x < -hx + 80.0) {
            let m = vec2<f32>(d.x + hx - 40.0, d.y - 0.45 * hy * (u01k(seed, 0xFEE9lu) - 0.5));
            let pad = band_cov(m.x, 40.0, fw) * band_cov(m.y, 60.0, fw);
            col = mixc(col, srgb(176.0, 172.0, 164.0), pad);
            for (var k = 0u; k < 3u; k++) {
                let b = band_cov(m.x - 10.0, 22.0, fw) * band_cov(m.y - 14.0 * f32(k) - 6.0, 5.0, fw);
                col = mixc(col, select(srgb(222.0, 222.0, 216.0), srgb(40.0, 40.0, 42.0), k == 1u), b);
                dh += 2.5 * b;
            }
            let shed = band_cov(m.x + 18.0, 14.0, fw) * band_cov(m.y + 30.0, 22.0, fw);
            col = mixc(col, srgb(200.0, 204.0, 206.0), shed);
            dh += 9.0 * shed;
            for (var k = 0u; k < 2u; k++) {
                let o = m - vec2<f32>(-24.0 + 12.0 * f32(k), -2.0);
                let silo = band_cov(length(o), 4.5, fw);
                col = mixc(col, srgb(216.0, 218.0, 220.0), silo);
                dh += 22.0 * silo;
            }
            if (pad > 0.5) {
                cls = LC_BUILDING;
            }
        }
    } else {
        let x0 = hx + 30.0 + 0.5 * (lag - 30.0);
        let yc = select(-0.5 * hy, 0.5 * hy, d.y > 0.0);
        let o = vec2<f32>(d.x - x0, d.y - yc);
        let ax = 0.5 * (lag - 30.0) - 6.0;
        let ay = 0.5 * hy - 14.0;
        let q = vec2<f32>(abs(o.x) - ax + 12.0, abs(o.y) - ay + 12.0);
        let dist = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - 12.0;
        let water = clamp(0.5 - dist / fw, 0.0, 1.0);
        let bank = max(clamp(0.5 - (dist - 7.0) / fw, 0.0, 1.0) - water, 0.0);
        cov = min(water + bank, 1.0);
        col = mixc(soil * 0.9, srgb(58.0, 60.0, 42.0) * (1.0 + 0.08 * perlin3(seed ^ 0xFEEAlu, p * (1.0lf / 30.0lf)) * band(30.0, gsd)), water / max(cov, 1e-6));
        dh = 1.2 * bank - 1.0 * water;
        cls = select(LC_BARE, LC_LAKE, water > 0.5);
    }
    if (cov <= 0.0) {
        return;
    }
    var ly = layer_paint(cov, col, cls);
    ly.dh = dh;
    ly.hmode = HM_BLEND;
    ly.clear = 1.0;
    composite(s, ly);
}

fn agri_kind_class(k: u32) -> u32 {
    switch k {
        case 0u: { return LC_CROP; }
        case 1u: { return LC_FALLOW; }
        case 2u, 3u: { return LC_PASTURE; }
        case 4u, 5u: { return LC_ORCHARD; }
        case 6u: { return LC_VINEYARD; }
        case 7u, 8u, 13u: { return LC_PLANTATION; }
        case 9u: { return LC_RICE_PADDY; }
        case 10u: { return LC_GREENHOUSE; }
        case 11u: { return LC_AQUACULTURE; }
        default: { return LC_FARMYARD; }
    }
}

struct AgPx {
    p: vec3<f64>,
    gsd: f32,
    fw: f32,
    season: f32,
    moist: f32,
    slope: f32,
    ground: f32,
    soil: vec3<f32>,
    grass: vec3<f32>,
    grass_dry: vec3<f32>,
    crown: vec3<f32>,
    flags: u32,
}

struct Parcel {
    id: u64,
    fx: f32,
    fy: f32,
    edge: f32,
    fc: vec2<f32>,
    size: f32,
    inside: f32,
    bedge: f32,
    circ: f32,
}

struct Shade {
    col: vec3<f32>,
    dh: f32,
    cls: u32,
}

fn agri_pivot_share(e: EcoV) -> f32 {
    return 0.08 + 0.72 * (1.0 - smoothstep1(450.0, 800.0, e.precip));
}

fn agri_draw(s: ptr<function, Stack>, pl: Plan) {
    let d = agri_sys(pl.sys);
    let m = (*s).m;
    let slope = (*s).l.slope;
    let slope_ok = 1.0 - smoothstep1(d.slope.x, d.slope.y, slope);
    var rip = m.riparian;
    if (pl.sys == AG_PADDY) {
        rip = 0.0;
    }
    let keep0 = m.natural_ok * slope_ok * (1.0 - rip) * (1.0 - m.woodlot) * m.shore_keep;
    let keep = smoothstep1(0.45, 0.55, keep0);
    if (keep <= 0.0) {
        return;
    }
    let r = (*s).r;
    let agri = s_agri(s);
    let cult = smoothstep1(0.02, 0.45, agri);
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let p = (*s).c.p;
    let fwe = max(fw, 0.6 * gsd);
    let loc = d.part == AP_SECTION || d.part == AP_PIVOT;
    var q = (*s).q_rot;
    var ax = r.ex.xyz;
    var ay = r.ey.xyz;
    if (loc) {
        q = (*s).q_loc;
        ax = r.east.xyz;
        ay = r.north.xyz;
    }
    var px: AgPx;
    px.p = p;
    px.gsd = gsd;
    px.fw = fwe;
    px.season = pl.season;
    px.moist = (*s).t.moist;
    px.slope = slope;
    px.ground = (*s).l.ground;
    let soil0 = mixc(mixc(spal(s, BP_SOIL), spal(s, BP_SOIL + 2u), 0.5), spal(s, BP_SOIL + 1u), 0.3);
    px.soil = mixc(mixc(soil0, vec3<f32>(dot(soil0, vec3<f32>(0.2126, 0.7152, 0.0722))), 0.3), vec3<f32>(0.2016, 0.15, 0.0976), 0.4);
    px.grass = spal(s, BP_GRASS_WET);
    px.grass_dry = spal(s, BP_GRASS_DRY);
    px.crown = spal(s, BP_CROWN_DECID);
    px.flags = pl.flags;
    let pc = agri_parcel(pl, d, q, p, fwe);
    var fs1 = d.size.y;
    if (d.part == AP_VOR) {
        fs1 = d.size.x;
    }
    var fsize = min(d.size.x, fs1);
    if (!loc) {
        fsize *= pl.scale;
    }
    let k = fsize / sqrt(fsize * fsize + 4.0 * gsd * gsd);
    let kc = fsize / sqrt(fsize * fsize + 10.0 * gsd * gsd);
    var col = vec3<f32>(0.0);
    var dh = 0.0;
    var cov = 0.0;
    var cls = agri_kind_class(d.kinds.x);
    if (pc.inside > 0.0) {
        let c3 = r.center.xyz + vec3<f64>(ax * pc.fc.x + ay * pc.fc.y);
        let mask = cultivated_mask(c3, gsd);
        if (mask < cult && u01k(pc.id, 5lu) >= 0.04) {
            let cluster = worley2(pl.seed ^ 0xC1Clu, pc.fc, 700.0, 1.0).id;
            let sh = agri_explicit(pl, d, pc, px, q, cluster);
            col = sh.col;
            dh = sh.dh;
            cls = sh.cls;
            cov = pc.inside;
        }
    }
    var inside_mean = 1.0;
    if (d.part == AP_PIVOT) {
        inside_mean = 0.82 * 0.69;
    } else if (d.part == AP_SECTION) {
        inside_mean = 1.0 - 0.21 * agri_pivot_share(pl.e);
    }
    let cov_m = lerp(cult * 0.9 * inside_mean, cov, k);
    if (cov_m <= 0.0) {
        return;
    }
    if (kc < 0.999) {
        let mean = agri_sys_mean(pl, d, px);
        if (cov > 0.0) {
            col = mixc(mean, col, kc);
        } else {
            col = mean;
        }
    }
    let fv = 1.0 + 0.09 * s_pf_lazy(s, PF_FIELD_VAR);
    var mottle = 0.0;
    if (cov > 0.0) {
        mottle = 0.11 * perlin3(pc.id ^ 0x3077lu, p * (1.0lf / 24.0lf)) * band(24.0, gsd);
    }
    let y = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    col = mixc(col, vec3<f32>(y), 0.25) * (fv + mottle);
    let a = cov_m * keep;
    (*s).m.field_cov = a;
    var ly = layer_paint(a, col, cls);
    ly.dh = dh * k - 0.6 * m.micro;
    ly.hmode = HM_ADD;
    if (cov > 0.0 && lc_group(cls) == LG_AGRICULTURE) {
        ly.inst = instance_id(pc.id);
    }
    // lit greenhouses
    if (pl.sys == AG_GREENH && gsd < AG_GH_GSD) {
        var lit = 0.0;
        if (cls == LC_GREENHOUSE && u01k(pc.id, 41lu) < AG_GH_LIT) {
            lit = cov;
        }
        ly.emit = AG_GH_GLOW * (lerp(cult * 0.9 * inside_mean * d.kw.x * AG_GH_LIT, lit, k) * keep * (1.0 - smoothstep1(0.5 * AG_GH_GSD, AG_GH_GSD, gsd)));
    }
    composite(s, ly);
}

const AG_GH_LIT: f32 = 0.2;
const AG_GH_GLOW: vec3<f32> = vec3<f32>(0.25, 0.14, 0.05);
const AG_GH_GSD: f32 = 300.0;

// ---------------------------------------------------------------- parcels

fn agri_parcel(pl: Plan, d: AgSys, q: vec2<f32>, p: vec3<f64>, fw: f32) -> Parcel {
    let seed = pl.seed;
    switch d.part {
        case 0u: {
            var sub = 1.0;
            if (pl.sys == AG_SHELTER || pl.sys == AG_PASTURE) {
                sub = 2.5;
            }
            return agri_part_grid(seed, q, p, d.size.x * pl.scale, d.size.y * pl.scale, sub);
        }
        case 1u: { return agri_part_cad(seed, q, d.size.x * pl.scale, d.size.y); }
        case 2u: { return agri_part_strip(seed, q, d.size.x * pl.scale, d.size.y * pl.scale); }
        case 3u: {
            var sp = 402.0;
            if (u01k(seed, 4lu) < 0.5) {
                sp = 805.0;
            }
            return agri_part_pivot(seed, q, sp, fw);
        }
        default: { return agri_part_section(seed, q, fw, agri_pivot_share(pl.e)); }
    }
}

fn agri_row_line(seed: u64, j: i64, h: f32) -> f32 {
    return (f32(j) + 0.32 * (u01(hash2(seed ^ 0x40lu, 7li, j)) - 0.5)) * h;
}

fn agri_col_line(seed: u64, j: i64, i: i64, w: f32) -> f32 {
    return (f32(i) + 0.45 * (u01(hash2(seed ^ 0x41lu, j, i)) - 0.5)) * w;
}

fn agri_part_grid(seed: u64, q0: vec2<f32>, p: vec3<f64>, w: f32, h: f32, sub: f32) -> Parcel {
    let inv900 = 1.0lf / 900.0lf;
    let wq = vec2<f32>(perlin3(seed, p * inv900), perlin3(seed ^ 1lu, p * inv900));
    let q = q0 + wq * (0.05 * min(w, h));
    var j = i64(floor(q.y / h));
    if (q.y < agri_row_line(seed, j, h)) {
        j -= 1li;
    } else if (q.y >= agri_row_line(seed, j + 1li, h)) {
        j += 1li;
    }
    let y0 = agri_row_line(seed, j, h);
    let y1 = agri_row_line(seed, j + 1li, h);
    let shift = u01(hash1(seed, j)) * w;
    let x = q.x + shift;
    var i = i64(floor(x / w));
    if (x < agri_col_line(seed, j, i, w)) {
        i -= 1li;
    } else if (x >= agri_col_line(seed, j, i + 1li, w)) {
        i += 1li;
    }
    let x0 = agri_col_line(seed, j, i, w);
    let x1 = agri_col_line(seed, j, i + 1li, w);
    let cw = x1 - x0;
    let ch = y1 - y0;
    let fx = x - x0;
    let fy0 = q.y - y0;
    let hc = hash2(seed ^ 0x55lu, i, j);
    let nstrip = 1 + i32(u01k(hc, 1lu) * sub);
    let sh = ch / f32(nstrip);
    let k = clamp(floor(fy0 / sh), 0.0, f32(nstrip - 1));
    let fy = fy0 - k * sh;
    var o: Parcel;
    o.id = mix64(hc ^ u64(k));
    o.fx = fx;
    o.fy = fy;
    o.edge = min(min(fx, cw - fx), min(fy, sh - fy));
    o.fc = vec2<f32>(x0 + 0.5 * cw - shift, y0 + (k + 0.5) * sh);
    o.size = min(cw, sh);
    o.inside = 1.0;
    o.bedge = min(min(fx, cw - fx), min(fy0, ch - fy0));
    o.circ = 0.0;
    return o;
}

struct CadBlock {
    id: u64,
    c: vec2<f32>,
    rel: vec2<f32>,
    size: vec2<f32>,
    edge: f32,
}

fn agri_cad_block(seed: u64, q: vec2<f32>, b: f32) -> CadBlock {
    var j = i64(floor(q.y / b));
    if (q.y < agri_row_line(seed, j, b)) {
        j -= 1li;
    } else if (q.y >= agri_row_line(seed, j + 1li, b)) {
        j += 1li;
    }
    let y0 = agri_row_line(seed, j, b);
    let y1 = agri_row_line(seed, j + 1li, b);
    let shift = u01(hash1(seed, j)) * b;
    let x = q.x + shift;
    var i = i64(floor(x / b));
    if (x < agri_col_line(seed, j, i, b)) {
        i -= 1li;
    } else if (x >= agri_col_line(seed, j, i + 1li, b)) {
        i += 1li;
    }
    let x0 = agri_col_line(seed, j, i, b);
    let x1 = agri_col_line(seed, j, i + 1li, b);
    var o: CadBlock;
    o.size = vec2<f32>(x1 - x0, y1 - y0);
    o.c = vec2<f32>(x0 + 0.5 * o.size.x - shift, y0 + 0.5 * o.size.y);
    o.rel = q - o.c;
    o.edge = min(0.5 * o.size.x - abs(o.rel.x), 0.5 * o.size.y - abs(o.rel.y));
    o.id = hash2(seed ^ 0xCADlu, i, j);
    return o;
}

fn agri_part_cad(seed: u64, q: vec2<f32>, cell: f32, aspect_max: f32) -> Parcel {
    let b = 2.6 * cell;
    let bl = agri_cad_block(seed, q, b);
    let ang = 0.45 * (u01k(bl.id, 1lu) - 0.5);
    let sa = sin(ang);
    let ca = cos(ang);
    let u = bl.rel.x * ca + bl.rel.y * sa;
    let v = -bl.rel.x * sa + bl.rel.y * ca;
    let hx = 0.5 * (bl.size.x * abs(ca) + bl.size.y * abs(sa));
    let hy = 0.5 * (bl.size.x * abs(sa) + bl.size.y * abs(ca));
    var ax0 = -hx;
    var ay0 = -hy;
    var ax1 = hx;
    var ay1 = hy;
    var id = bl.id;
    let strip_p = 0.6 * (aspect_max - 1.0);
    for (var it = 0; it < 5; it++) {
        let w = ax1 - ax0;
        let h = ay1 - ay0;
        let tgt = cell * (0.65 + 0.7 * u01k(id, 2lu));
        if (w * h < tgt * tgt) {
            break;
        }
        var split_x = w >= h;
        if (u01k(id, 4lu) < strip_p && min(w, h) > 0.9 * tgt) {
            split_x = !split_x;
        }
        let t = 0.3 + 0.4 * u01k(id, 3lu);
        if (split_x) {
            let xs = ax0 + t * w;
            if (u < xs) {
                ax1 = xs;
                id = mix64(id ^ 0x11lu);
            } else {
                ax0 = xs;
                id = mix64(id ^ 0x22lu);
            }
        } else {
            let ys = ay0 + t * h;
            if (v < ys) {
                ay1 = ys;
                id = mix64(id ^ 0x33lu);
            } else {
                ay0 = ys;
                id = mix64(id ^ 0x44lu);
            }
        }
    }
    let lc = vec2<f32>(0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1));
    var o: Parcel;
    o.id = id;
    o.fx = u - lc.x;
    o.fy = v - lc.y;
    o.edge = min(min(min(u - ax0, ax1 - u), min(v - ay0, ay1 - v)), bl.edge);
    o.fc = bl.c + vec2<f32>(lc.x * ca - lc.y * sa, lc.x * sa + lc.y * ca);
    o.size = min(ax1 - ax0, ay1 - ay0);
    o.inside = 1.0;
    o.bedge = bl.edge;
    o.circ = 0.0;
    return o;
}

fn agri_part_strip(seed: u64, q: vec2<f32>, w: f32, sw0: f32) -> Parcel {
    let bl = agri_cad_block(seed ^ 0x57lu, q, w);
    let fid = bl.id;
    let bedge = bl.edge;
    let rel = bl.rel;
    var ang = 0.0;
    if (u01k(fid, 1lu) >= 0.5) {
        ang = 0.5 * PI;
    }
    ang += 0.3 * (u01k(fid, 2lu) - 0.5);
    let sa = sin(ang);
    let ca = cos(ang);
    let a = rel.x * ca + rel.y * sa;
    let c = -rel.x * sa + rel.y * ca;
    var amp = 0.0;
    if (u01k(fid, 4lu) < 0.6) {
        amp = 0.03 * w;
    }
    let c2 = c + amp * sin(TAU * a / w);
    let sw = sw0 * (0.6 + 0.8 * u01k(fid, 3lu));
    let n = floor(c2 / sw);
    let fy = c2 - n * sw;
    var o: Parcel;
    o.id = hash2(fid ^ 0x5751lu, i64(n), 0li);
    o.fx = a;
    o.fy = fy;
    o.edge = min(min(fy, sw - fy), bedge);
    o.fc = bl.c + vec2<f32>(-sa, ca) * ((n + 0.5) * sw);
    o.size = sw;
    o.inside = 1.0;
    o.bedge = bedge;
    o.circ = 0.0;
    return o;
}

fn agri_part_pivot(seed: u64, q: vec2<f32>, sp: f32, fw: f32) -> Parcel {
    let cq = floor(q / sp);
    let c = (cq + 0.5) * sp;
    let rel = q - c;
    let d = length(rel);
    let rad = 0.47 * sp;
    let hc = hash2(seed ^ 0x9lu, i64(cq.x), i64(cq.y));
    var inside = 0.0;
    if (u01k(hc, 3lu) < 0.82) {
        inside = band_cov(d, rad, fw);
    }
    let nsec = 1 + i32(u01k(hc, 1lu) * 3.0);
    let ang = atan2(rel.y, rel.x) + PI + u01k(hc, 2lu) * 6.0;
    var sec = i32(floor(ang / TAU * f32(nsec))) % nsec;
    if (sec < 0) {
        sec += nsec;
    }
    var o: Parcel;
    o.id = mix64(hc ^ u64(sec));
    o.fx = rel.x;
    o.fy = rel.y;
    o.edge = abs(rad - d);
    o.fc = c;
    o.size = sp;
    o.inside = inside;
    o.bedge = o.edge;
    o.circ = 1.0;
    return o;
}

fn agri_part_section(seed: u64, q: vec2<f32>, fw: f32, pivot_p: f32) -> Parcel {
    let half = 0.5 * AG_MILE;
    let sc = floor(q / AG_MILE);
    let hs = hash2(seed ^ 0x5EClu, i64(sc.x), i64(sc.y));
    let l = q - sc * AG_MILE;
    let qi = clamp(floor(l / half), vec2<f32>(0.0), vec2<f32>(1.0));
    let hq = hash2(hs, i64(qi.x), i64(qi.y));
    let lq = l - qi * half;
    let o0 = sc * AG_MILE + qi * half;
    var o: Parcel;
    o.inside = 1.0;
    if (u01k(hq, 7lu) < pivot_p) {
        let rel = lq - vec2<f32>(0.5 * half);
        let d = length(rel);
        let rad = 0.48 * half;
        o.id = hq;
        o.fx = rel.x;
        o.fy = rel.y;
        o.edge = abs(rad - d);
        o.fc = o0 + vec2<f32>(0.5 * half);
        o.size = half;
        o.inside = band_cov(d, rad, fw);
        o.bedge = o.edge;
        o.circ = 1.0;
        return o;
    }
    let mode = u01k(hq, 1lu);
    var n = 4.0;
    if (mode < 0.25) {
        n = 1.0;
    } else if (mode < 0.6) {
        n = 2.0;
    }
    let along_x = u01k(hq, 2lu) < 0.5;
    var a = lq.y;
    var c = lq.x;
    if (along_x) {
        a = lq.x;
        c = lq.y;
    }
    let sw = half / n;
    let k = clamp(floor(c / sw), 0.0, n - 1.0);
    let fy = c - k * sw;
    let cc = (k + 0.5) * sw;
    var fco = vec2<f32>(cc, 0.5 * half);
    if (along_x) {
        fco = vec2<f32>(0.5 * half, cc);
    }
    o.id = mix64(hq ^ u64(k));
    o.fx = a;
    o.fy = fy;
    o.edge = min(min(a, half - a), min(fy, sw - fy));
    o.fc = o0 + fco;
    o.size = sw;
    o.bedge = min(min(lq.x, half - lq.x), min(lq.y, half - lq.y));
    o.circ = 0.0;
    return o;
}

// ---------------------------------------------------------------- shading

fn agri_pick_kind(d: AgSys, u: f32, season: f32) -> u32 {
    var w = d.kw;
    for (var i = 0u; i < 4u; i++) {
        if (d.kinds[i] == AK_FALLOW) {
            w[i] *= 0.5 + season;
        }
    }
    let total = w.x + w.y + w.z + w.w;
    let v = u * total;
    var acc = 0.0;
    for (var i = 0u; i < 4u; i++) {
        acc += w[i];
        if (v < acc) {
            return d.kinds[i];
        }
    }
    return d.kinds.x;
}

fn agri_pick_crop(mix: u32, u: f32) -> u32 {
    let mx = agri_mix(mix);
    var acc = 0.0;
    for (var i = 0u; i < 4u; i++) {
        acc += mx.w[i];
        if (u < acc) {
            return mx.crops[i];
        }
    }
    return mx.crops.w;
}

fn agri_stage(c: array<vec3<f32>, 4>, ph: f32) -> vec3<f32> {
    if (ph < 0.5) {
        return mixc(c[0], c[1], ph * 2.0);
    }
    return mixc(c[1], c[2], ph * 2.0 - 1.0);
}

fn agri_sys_mean(pl: Plan, d: AgSys, px: AgPx) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    var wt = 0.0;
    for (var i = 0u; i < 4u; i++) {
        var w = d.kw[i];
        if (d.kinds[i] == AK_FALLOW) {
            w *= 0.5 + px.season;
        }
        acc += agri_kind_mean(pl, d.kinds[i], px) * w;
        wt += w;
    }
    return acc / max(wt, 1e-6);
}

fn agri_pasture_col(px: AgPx) -> vec3<f32> {
    return mixc(px.grass_dry, px.grass, smoothstep1(0.15, 0.5, px.moist));
}

fn agri_vine_col(season: f32, u: f32) -> vec3<f32> {
    var autumn = srgb(132.0, 64.0, 48.0);
    if (u < 0.5) {
        autumn = srgb(156.0, 118.0, 52.0);
    }
    if (season < 0.5) {
        return mixc(srgb(104.0, 124.0, 60.0), srgb(70.0, 98.0, 46.0), season * 2.0);
    }
    return mixc(srgb(70.0, 98.0, 46.0), autumn, season * 2.0 - 1.0);
}

fn agri_paddy_col(st: u32) -> vec3<f32> {
    switch st {
        case 0u: { return srgb(64.0, 84.0, 94.0); }
        case 1u: { return srgb(76.0, 100.0, 86.0); }
        case 2u: { return srgb(80.0, 116.0, 50.0); }
        case 3u: { return srgb(188.0, 172.0, 92.0); }
        default: { return srgb(150.0, 128.0, 88.0); }
    }
}

fn agri_paddy_state(ph: f32) -> u32 {
    if (ph < 0.18) {
        return 0u;
    } else if (ph < 0.32) {
        return 1u;
    } else if (ph < 0.66) {
        return 2u;
    } else if (ph < 0.84) {
        return 3u;
    }
    return 4u;
}

fn agri_paddy_mean(season: f32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    for (var i = 0; i < 5; i++) {
        acc += agri_paddy_col(agri_paddy_state(clamp(season + 0.075 * (f32(i) - 2.0), 0.0, 1.0)));
    }
    return acc / 5.0;
}

fn agri_kind_mean(pl: Plan, k: u32, px: AgPx) -> vec3<f32> {
    switch k {
        case 0u: {
            let mx = agri_mix(pl.mix);
            var acc = vec3<f32>(0.0);
            for (var i = 0u; i < 4u; i++) {
                acc += agri_stage(agri_crop(mx.crops[i]), px.season) * mx.w[i];
            }
            return acc;
        }
        case 1u: { return mixc(px.soil * 0.8, mixc(px.soil, px.grass_dry, 0.5), 0.5 - 0.4 * px.season); }
        case 2u: { return agri_pasture_col(px) * 0.97; }
        case 3u: { return agri_pasture_col(px) * 1.05; }
        case 4u: { return mixc(mixc(px.grass, px.soil, 0.3), px.crown * 1.1, 0.4); }
        case 5u: { return mixc(px.soil * 1.15, srgb(104.0, 112.0, 84.0), 0.3); }
        case 6u: { return mixc(px.soil * 1.2, agri_vine_col(px.season, 0.5), 0.35); }
        case 7u: { return mixc(srgb(92.0, 100.0, 60.0), srgb(58.0, 92.0, 44.0), 0.55); }
        case 8u: { return srgb(118.0, 156.0, 58.0) * 0.92; }
        case 9u: { return agri_paddy_mean(px.season); }
        case 10u: { return srgb(214.0, 216.0, 212.0) * 0.9; }
        case 11u: { return srgb(52.0, 74.0, 76.0); }
        case 12u: { return srgb(172.0, 140.0, 108.0); }
        default: { return srgb(40.0, 70.0, 38.0) * 0.9; }
    }
}

fn agri_tree_grid(u: f32, v: f32, sx: f32, sy: f32, r: f32, gsd: f32, fw: f32) -> f32 {
    let gx = round(u / sx) * sx;
    let gy = round(v / sy) * sy;
    let d = length(vec2<f32>(u - gx, v - gy));
    let ex = band(min(sx, sy), gsd);
    let mean = min(PI * r * r / (sx * sy), 1.0);
    return band_cov(d, r, fw) * ex + (1.0 - ex) * mean;
}

fn agri_scatter_tree(seed: u64, q: vec2<f32>, cell: f32, dens: f32, gsd: f32, fw: f32) -> vec2<f32> {
    let ci = floor(q / cell);
    let h = hash2(seed ^ 0x7EElu, i64(ci.x), i64(ci.y));
    let ex = band(cell, gsd);
    let mean = dens * PI * 0.0147;
    var cv = 0.0;
    var th = 0.0;
    if (u01k(h, 1lu) < dens && ex > 0.0) {
        let c = (ci + vec2<f32>(0.25 + 0.5 * u01k(h, 2lu), 0.25 + 0.5 * u01k(h, 3lu))) * cell;
        let r = (0.09 + 0.06 * u01k(h, 4lu)) * cell;
        let d = length(q - c);
        cv = band_cov(d, r, fw);
        let dr = min(d / r, 1.0);
        th = (5.0 + 5.0 * u01k(h, 5lu)) * sqrt(1.0 - dr * dr);
    }
    return vec2<f32>(cv * ex + (1.0 - ex) * mean, th * ex + (1.0 - ex) * 7.5);
}

fn agri_explicit(pl: Plan, d: AgSys, pc: Parcel, px: AgPx, q: vec2<f32>, cluster: u64) -> Shade {
    let p = px.p;
    let gsd = px.gsd;
    let fw = px.fw;
    var id = pc.id;
    var k = agri_pick_kind(d, u01k(id, 40lu), px.season);
    if (k == AK_OLIVE && (px.flags & AF_MED) == 0u) {
        if ((px.flags & AF_DESERT) != 0u) {
            k = AK_PALM;
        } else {
            k = AK_ORCHARD;
        }
    }
    var col = px.soil;
    var dh = 0.0;
    var cls = agri_kind_class(k);
    var lip = 1e9;
    var t_dh = 0.0;
    if (d.terraces > 0.0 && px.slope > 0.07) {
        var dhs = 1.8 + 1.6 * u01k(pl.seed, 40lu);
        if (pl.sys == AG_PADDY) {
            dhs = 1.2 + 1.0 * u01k(pl.seed, 40lu);
        }
        let t = px.ground / dhs + u01k(pl.seed, 41lu);
        let ti = floor(t);
        let f = t - ti;
        let wb = dhs / max(px.slope, 0.02);
        let ex = smoothstep1(1.2 * gsd, 3.0 * gsd, wb);
        lip = f * wb / max(ex, 1e-3) + (1.0 - ex) * 1e9;
        t_dh = (0.5 - f) * dhs * ex;
        id = mix64(id ^ bitcast<u64>(i64(ti)));
    }
    let rot = u01k(id, 9lu) < 0.3;
    var ru = pc.fx;
    var rv = pc.fy;
    if (rot) {
        ru = pc.fy;
        rv = pc.fx;
    }
    switch k {
        case 0u, 3u: {
            var u_crop = u01k(id, 6lu);
            if (u01k(id, 16lu) < 0.7) {
                u_crop = u01k(cluster, 6lu);
            }
            var c = 6u;
            if (k != AK_HAY) {
                c = agri_pick_crop(pl.mix, u_crop);
            }
            let cc = agri_crop(c);
            let row = cc[3];
            let ph = clamp(px.season + 0.3 * (u01k(id, 20lu) - 0.5), 0.0, 1.0);
            col = agri_stage(cc, ph) * (0.95 + 0.10 * u01k(id, 8lu));
            col *= 1.0 + 0.2 * perlin3(id, p * (1.0lf / 35.0lf)) * band(35.0, gsd) + 0.11 * perlin3(id ^ 1lu, p * (1.0lf / 120.0lf)) * band(120.0, gsd);
            col *= 1.0 + 0.15 * perlin3(id ^ 0x7E1lu, p * (1.0lf / 7.0lf)) * band(7.0, gsd) + 0.10 * perlin3(id ^ 0x7E2lu, p * (1.0lf / 2.5lf)) * band(2.5, gsd);
            let gz = perlin3(id ^ 0x6A0lu, p * (1.0lf / 55.0lf)) * band(55.0, gsd);
            col = mixc(col, col * vec3<f32>(1.12, 1.04, 0.82), 0.55 * smoothstep1(0.0, 0.6, gz));
            col = mixc(col, col * vec3<f32>(0.84, 0.92, 0.86), 0.5 * smoothstep1(0.0, 0.6, -gz));
            let hw = 8.0 + 10.0 * u01k(id, 11lu);
            if (pc.edge < hw) {
                col *= 0.96;
            }
            let rc = sqrt(pc.fx * pc.fx + pc.fy * pc.fy);
            var u = ru;
            if (pc.circ > 0.0) {
                u = rc;
            }
            col *= 1.0 + row.y * sin(TAU * u / row.x) * band(row.x, gsd);
            if (pc.edge >= hw && (c == 0u || c == 2u || c == 3u)) {
                let sp = 18.0 + 6.0 * floor(u01k(id, 15lu) * 3.0);
                let mt = rem_euclid(u, sp);
                let tl = band_cov(mt - 0.9, 0.22, fw) + band_cov(mt - 2.7, 0.22, fw);
                let vis = max(band(1.2, gsd), 0.35 * band(sp, gsd));
                col = mixc(col, col * 0.78, tl * vis);
            }
            dh = row.z * max(1.0 - abs(2.0 * ph - 1.0), 0.25);
            if (pc.circ > 0.0) {
                let sp = 48.0 + 8.0 * u01k(id, 17lu);
                let mt = rem_euclid(rc + 0.5 * sp, sp) - 0.5 * sp;
                let vis = max(band(1.5, gsd), 0.3 * band(sp, gsd));
                col = mixc(col, px.soil * 0.9, band_cov(mt, 0.5, fw) * vis * smoothstep1(20.0, 30.0, rc));
            }
            if (k == AK_HAY) {
                cls = LC_PASTURE;
            }
        }
        case 1u: {
            if (u01k(id, 21lu) < 0.5 + 0.4 * px.season) {
                col = px.soil * (0.74 + 0.12 * u01k(id, 22lu));
                let sp = 0.35 + 0.15 * u01k(id, 23lu);
                let fur = sin(TAU * ru / sp) * band(sp, gsd);
                col *= 1.0 + 0.16 * fur + 0.10 * perlin3(id ^ 0xC10lu, p * (1.0lf / 2.5lf)) * band(2.5, gsd);
                let land = 22.0 + 18.0 * u01k(id, 24lu);
                col *= 1.0 + 0.05 * sign(sin(TAU * ru / (2.0 * land))) * band(land, gsd);
                dh = 0.06 * fur;
            } else {
                let n = perlin3(id ^ 0xF0Alu, p * (1.0lf / 18.0lf)) * band(18.0, gsd);
                col = mixc(px.soil, px.grass_dry, 0.5 + 0.3 * n) * (1.0 + 0.08 * perlin3(id ^ 0xF0Blu, p * (1.0lf / 4.0lf)) * band(4.0, gsd));
                dh = 0.2;
            }
        }
        case 2u: {
            let sh = agri_pasture(pl, pc, px, id, q);
            col = sh.col;
            dh = sh.dh;
            cls = sh.cls;
        }
        case 4u, 5u: {
            let olive = k == AK_OLIVE;
            var sx = 5.0 + 2.0 * u01k(id, 10lu);
            var sy = 3.6 + 1.0 * u01k(id, 12lu);
            var r = 1.6 + 0.5 * u01k(id, 13lu);
            var ground = mixc(px.grass, px.soil, 0.3);
            var crown_c = px.crown * 1.1;
            var hh = 4.0;
            if (olive) {
                let s0 = 7.0 + 3.0 * u01k(id, 10lu);
                sx = s0;
                sy = s0;
                r = 2.4 + 0.8 * u01k(id, 12lu);
                ground = px.soil * (1.12 + 0.08 * sin(TAU * ru / 0.5) * band(0.5, gsd));
                crown_c = srgb(104.0, 112.0, 84.0);
                hh = 5.0;
            }
            let cv = agri_tree_grid(ru, rv, sx, sy, r, gsd, fw);
            let tex = 1.0 + 0.12 * perlin3(id ^ 0x0C1lu, p * (1.0lf / 1.3lf)) * band(1.3, gsd);
            col = mixc(ground, crown_c * tex, cv);
            dh = hh * cv;
        }
        case 6u: {
            let sp = 2.2 + 0.6 * u01k(id, 10lu);
            let hw = 0.45 + 0.15 * u01k(id, 12lu);
            var v = ru / sp;
            var sp_act = sp;
            if (px.slope > 0.12 && u01k(id, 13lu) < 0.5) {
                let dz = sp * 0.25;
                v = px.ground / dz;
                sp_act = dz / max(px.slope, 0.05);
            }
            let mrow = abs(v - floor(v) - 0.5) * sp_act;
            let ex = band(sp_act, gsd);
            let cv = band_cov(mrow, hw, fw) * ex + (1.0 - ex) * min(2.0 * hw / sp_act, 1.0);
            var ground = px.soil * 1.25;
            if (u01k(id, 14lu) < 0.4) {
                ground = mixc(px.grass, px.grass_dry, 0.5);
            }
            col = mixc(ground, agri_vine_col(px.season, u01k(id, 15lu)), cv);
            dh = 1.6 * cv;
        }
        case 7u: {
            let fine = 1.0 + 0.08 * perlin3(id ^ 0x7A1lu, p * (1.0lf / 6.0lf)) * band(6.0, gsd);
            if ((px.flags & AF_DESERT) != 0u) {
                let cv = agri_tree_grid(ru, rv, 8.0, 8.0, 3.4, gsd, fw);
                col = mixc(srgb(186.0, 164.0, 124.0) * fine, srgb(98.0, 110.0, 66.0), cv);
                dh = 9.0 * cv;
                cls = LC_ORCHARD;
            } else {
                let age = 0.35 + 0.65 * u01k(id, 34lu);
                let sx = 9.0;
                let sy = 7.8;
                let j = round(rv / sy);
                var off = 0.5 * sx;
                if (((i32(j) % 2) + 2) % 2 == 0) {
                    off = 0.0;
                }
                let i = round((ru - off) / sx);
                let rel = vec2<f32>(ru - off - i * sx, rv - j * sy);
                let ang = atan2(rel.y, rel.x) + u01k(mix64(id ^ bitcast<u64>(i64(i)) ^ (bitcast<u64>(i64(j)) << 20u)), 1lu) * 6.3;
                let r_eff = 4.4 * age * (0.82 + 0.18 * cos(8.0 * ang));
                let dd = length(rel);
                let ex = band(sx, gsd);
                let crown = band_cov(dd, r_eff, fw) * ex + (1.0 - ex) * (0.75 * age);
                let shade = 0.85 + 0.25 * (1.0 - min(dd / max(r_eff, 0.1), 1.0)) * ex;
                col = mixc(srgb(92.0, 100.0, 60.0) * fine, srgb(58.0, 92.0, 44.0) * shade * fine, crown);
                dh = (3.0 + 9.0 * age) * crown;
            }
        }
        case 13u: {
            let sp = 7.0;
            let mr = abs(rv / sp - floor(rv / sp) - 0.5) * sp;
            let ex = band(sp, gsd);
            let cv = band_cov(mr, 2.9, fw) * ex + (1.0 - ex) * (5.8 / sp);
            let tex = 1.0 + 0.14 * perlin3(id ^ 0x2BBlu, p * (1.0lf / 3.0lf)) * band(3.0, gsd);
            col = mixc(srgb(70.0, 66.0, 50.0), srgb(40.0, 70.0, 38.0) * tex, cv);
            dh = 16.0 * cv;
        }
        case 8u: {
            var v = rv / 1.5;
            var sp_act = 1.5;
            if (px.slope > 0.06) {
                let dz = 1.5 * 0.2;
                v = px.ground / dz;
                sp_act = dz / max(px.slope, 0.05);
            }
            let mr = abs(v - floor(v) - 0.5) * sp_act;
            let ex = band(sp_act, gsd);
            let cv = band_cov(mr, 0.42 * sp_act, fw) * ex + (1.0 - ex) * 0.84;
            let tex = 1.0 + 0.08 * perlin3(id ^ 0x7EAlu, p * (1.0lf / 2.0lf)) * band(2.0, gsd);
            col = mixc(srgb(96.0, 82.0, 62.0), srgb(118.0, 156.0, 58.0) * tex, cv);
            dh = 1.0 * cv;
        }
        case 9u: {
            let ph = clamp(px.season + 0.3 * (u01k(id, 31lu) - 0.5), 0.0, 1.0);
            let st = agri_paddy_state(ph);
            col = agri_paddy_col(st) * (0.94 + 0.12 * u01k(id, 32lu));
            let fine = perlin3(id ^ 0x9ADlu, p * (1.0lf / 7.0lf)) * band(7.0, gsd);
            col *= 1.0 + 0.06 * fine;
            if (st == 1u) {
                let rr = sin(TAU * ru / 0.3) * band(0.3, gsd);
                col = mixc(col, agri_paddy_col(2u), 0.25 + 0.2 * rr);
            }
            var bd = min(pc.edge, lip);
            if (lip > 1e8) {
                let sx = 18.0 + 22.0 * u01k(id, 33lu);
                let sy = sx * (1.2 + 0.6 * u01k(id, 35lu));
                let mx = rem_euclid(pc.fx + 1000.0, sx);
                let my = rem_euclid(pc.fy + 1000.0, sy);
                bd = min(bd, min(min(mx, sx - mx), min(my, sy - my)) / max(band(sx, gsd), 1e-3));
            }
            let bund = band_cov(bd, 0.4, fw);
            col = mixc(col, srgb(122.0, 116.0, 82.0), bund);
            dh = 0.3 * bund - 0.05;
        }
        case 10u: {
            let inside = clamp((pc.edge - 1.8) / fw + 0.5, 0.0, 1.0);
            let bw = 8.0 + 1.6 * u01k(id, 10lu);
            let mbay = ru / bw - floor(ru / bw);
            let ex = band(bw, gsd);
            let roof = 1.0 + 0.10 * cos(TAU * mbay) * ex;
            let gutter = band_cov(min(mbay, 1.0 - mbay) * bw, 0.35, fw) * ex;
            let u = u01k(id, 11lu);
            var base = srgb(240.0, 238.0, 226.0);
            if (u < 0.15) {
                base = srgb(150.0, 156.0, 150.0);
            } else if (u < 0.45) {
                base = srgb(232.0, 234.0, 230.0);
            } else if (u < 0.8) {
                base = srgb(212.0, 216.0, 214.0);
            }
            let dirt = 1.0 + 0.06 * perlin3(id ^ 0x6E1lu, p * (1.0lf / 10.0lf)) * band(10.0, gsd);
            let gh = mixc(base * roof * dirt, srgb(118.0, 122.0, 124.0), gutter);
            col = mixc(px.soil * 1.1, gh, inside);
            dh = 4.0 * inside;
            if (inside < 0.5) {
                cls = LC_CROP;
            }
        }
        case 11u: {
            let m_in = 3.0 + 3.0 * u01k(id, 10lu);
            let rp = 18.0 + 30.0 * u01k(id, 12lu);
            let dc = abs(q - pc.fc) - vec2<f32>(rp);
            let w = clamp((pc.edge - m_in) / fw + 0.5, 0.0, 1.0) * clamp(-max(dc.x, dc.y) / fw + 0.5, 0.0, 1.0);
            let aq = pl.sys == AG_PADDY || pl.sys == AG_GREENH;
            var water = srgb(48.0, 70.0, 74.0);
            if (aq) {
                water = srgb(62.0, 90.0, 72.0);
            }
            col = mixc(srgb(140.0, 128.0, 100.0), water * (0.9 + 0.2 * u01k(id, 11lu)), w);
            dh = -0.6 * w;
            if (w > 0.5) {
                cls = select(LC_RESERVOIR, LC_AQUACULTURE, aq);
            } else {
                cls = LC_FALLOW;
            }
        }
        case 12u: {
            let sh = agri_yard(pc, px, id, q);
            col = sh.col;
            dh = sh.dh;
            cls = sh.cls;
        }
        default: {
            col = px.soil;
        }
    }
    if (lip < 1e8) {
        if (pl.sys == AG_TERRACE) {
            let w = band_cov(lip, 0.5, fw);
            col = mixc(col, srgb(150.0, 144.0, 132.0) * (0.9 + 0.2 * perlin3(id ^ 0x5701lu, p * (1.0lf / 1.5lf)) * band(1.5, gsd)), w);
        }
        dh += t_dh;
    }
    var trees = d.trees;
    var tcell = 30.0;
    if ((pl.sys == AG_PASTURE || pl.sys == AG_RANCH) && pl.e.temp > 18.0) {
        trees = 0.12;
        tcell = 60.0;
    }
    if (trees > 0.0) {
        let tr = agri_scatter_tree(pl.seed, q, tcell, trees, gsd, fw);
        if (tr.x > 0.0) {
            col = mixc(col, px.crown * (0.85 + 0.25 * u01k(pc.id, 50lu)) * (0.72 + 0.4 * min(tr.y / 10.0, 1.0)), tr.x);
            dh = lerp(dh, tr.y, tr.x);
        }
    }
    let bsel = u01k(pc.id, 0xB0lu);
    let hedge_m = 0.3 + 0.7 * pl.e.hedges;
    var p0 = d.bp.x;
    if (d.borders.x == AB_HEDGE || d.borders.x == AB_BELT) {
        p0 *= hedge_m;
    }
    var b = AB_MARGIN;
    if (bsel < p0) {
        b = d.borders.x;
    } else if (bsel < p0 + d.bp.y) {
        b = d.borders.y;
    }
    var edge = pc.edge;
    if (b == AB_CANAL) {
        edge = pc.bedge;
    }
    var o: Shade;
    o.col = col;
    o.dh = dh;
    o.cls = cls;
    return agri_border(b, edge, o, pc, px);
}

fn agri_pasture(pl: Plan, pc: Parcel, px: AgPx, id: u64, q: vec2<f32>) -> Shade {
    let p = px.p;
    let gsd = px.gsd;
    let fw = px.fw;
    var col = agri_pasture_col(px) * (0.86 + 0.28 * u01k(id, 8lu));
    col *= 1.0 + 0.08 * perlin3(id ^ 0x9A5lu, p * (1.0lf / 14.0lf)) * band(14.0, gsd) + 0.05 * perlin3(id ^ 0x9A6lu, p * (1.0lf / 4.0lf)) * band(4.0, gsd);
    let tus = perlin3(id ^ 0x9A7lu, p * (1.0lf / 1.7lf)) * band(1.7, gsd);
    col *= 1.0 + 0.10 * tus + 0.12 * perlin3(id ^ 0x9A8lu, p * (1.0lf / 40.0lf)) * band(40.0, gsd);
    var cls = LC_PASTURE;
    var dh = 0.1;
    if (pl.sys == AG_PASTURE || pl.sys == AG_SHELTER || pl.sys == AG_SECTION || pl.sys == AG_RANCH) {
        let half = 0.5 * pc.size;
        let t = vec2<f32>(0.4 * half * (u01k(id, 60lu) - 0.5), 0.4 * half * (u01k(id, 61lu) - 0.5));
        let rel = q - pc.fc - t;
        let r = length(rel);
        let n = 5.0 + floor(u01k(id, 62lu) * 4.0);
        let th = atan2(rel.y, rel.x) / TAU * n + n;
        var dmin = 1e9;
        for (var o = 0; o < 2; o++) {
            let kk = floor(th) + f32(o);
            let kh = mix64(id ^ u64(rem_euclid(kk, n)) ^ 0x7A11lu);
            let a = kk + 0.3 * (u01k(kh, 1lu) - 0.5) + 0.12 * sin(r / 45.0 + 6.0 * u01k(kh, 2lu));
            let len = (0.3 + 0.7 * u01k(kh, 3lu)) * half;
            let dd = abs(th - a) / n * TAU * r;
            if (r < len) {
                dmin = min(dmin, dd);
            }
        }
        let vis = max(band(1.5, gsd), 0.3 * band(30.0, gsd));
        let trail = band_cov(dmin, 0.45, fw) * vis * smoothstep1(8.0, 20.0, r);
        let bare = px.soil * 0.95;
        col = mixc(col, bare, trail * 0.8);
        let tramp = (1.0 - smoothstep1(5.0, 13.0 + 5.0 * perlin3(id ^ 0x7A12lu, p * (1.0lf / 9.0lf)), r)) * select(0.0, 1.0, u01k(id, 65lu) < 0.35);
        col = mixc(col, bare * 0.9, 0.5 * tramp);
        let trough = band_cov(r, 1.4, fw);
        col = mixc(col, srgb(112.0, 118.0, 122.0), trough);
        dh += 0.8 * trough;
        if (u01k(id, 63lu) < 0.15) {
            let dc = rel - vec2<f32>(30.0, 0.0);
            let rd = 12.0 + 14.0 * u01k(id, 64lu);
            let w = band_cov(length(dc), rd, fw);
            let ring = band_cov(length(dc), rd + 3.0, fw) - w;
            col = mixc(col, bare * 0.85, max(ring, 0.0));
            col = mixc(col, srgb(56.0, 74.0, 70.0), w);
            if (w > 0.5) {
                cls = LC_RESERVOIR;
            }
            dh -= 0.6 * w;
        }
    }
    var o: Shade;
    o.col = col;
    o.dh = dh;
    o.cls = cls;
    return o;
}

fn agri_yard(pc: Parcel, px: AgPx, id: u64, q: vec2<f32>) -> Shade {
    let p = px.p;
    let gsd = px.gsd;
    let fw = px.fw;
    let rel = q - pc.fc;
    let r = length(rel);
    let ry = 14.0 + 10.0 * u01k(id, 70lu);
    var o: Shade;
    o.col = px.soil * 1.15 * (1.0 + 0.05 * perlin3(id ^ 0x7A8lu, p * (1.0lf / 3.0lf)) * band(3.0, gsd));
    o.dh = 0.0;
    o.cls = LC_FARMYARD;
    if (r > ry) {
        o.col = mixc(px.soil, srgb(82.0, 114.0, 54.0), 0.5 + 0.3 * perlin3(id ^ 0x7A9lu, p * (1.0lf / 5.0lf)) * band(5.0, gsd));
        o.dh = 0.6;
        o.cls = LC_CROP;
        return o;
    }
    let fence = band_cov(r - ry, 0.8, fw);
    o.col = mixc(o.col, srgb(92.0, 90.0, 60.0), fence);
    o.dh += 1.5 * fence;
    let nh = 2.0 + floor(u01k(id, 71lu) * 4.0);
    for (var i = 0u; i < 5u; i++) {
        if (f32(i) < nh) {
            let a = TAU * (f32(i) / nh + u01k(id, 72lu));
            let c = vec2<f32>(cos(a), sin(a)) * (0.5 * ry);
            let hr = 2.2 + 1.3 * u01k(id, 73lu + u64(i));
            let dd = length(rel - c);
            let hut = band_cov(dd, hr, fw);
            if (hut > 0.0) {
                let tin = u01k(id, 80lu + u64(i)) < 0.35;
                var roof = srgb(146.0, 122.0, 86.0) * (0.9 + 0.2 * (1.0 - dd / hr));
                if (tin) {
                    roof = srgb(176.0, 180.0, 182.0);
                }
                o.col = mixc(o.col, roof, hut);
                o.dh = lerp(o.dh, 2.0 + 1.5 * (1.0 - dd / hr), hut);
                if (hut > 0.5) {
                    o.cls = LC_BUILDING;
                }
            }
        }
    }
    return o;
}

fn agri_border(b: u32, edge: f32, i: Shade, pc: Parcel, px: AgPx) -> Shade {
    let p = px.p;
    let gsd = px.gsd;
    let fwb = max(px.fw, 0.4 * gsd);
    let hb = mix64(pc.id ^ 0xED6Elu);
    var o = i;
    switch b {
        case 0u: {
            let c = band_cov(edge, 0.9, fwb);
            o.col = mixc(i.col, mixc(px.grass_dry, px.grass, px.moist), 0.45 * c);
        }
        case 1u, 2u: {
            let belt = b == AB_BELT;
            var hw0 = 1.0 + 1.0 * u01k(hb, 4lu);
            var lam = 5.0;
            var dark = 0.95;
            var h = 3.0 + 4.0 * u01k(hb, 2lu);
            if (belt) {
                hw0 = 4.0 + 3.0 * u01k(hb, 4lu);
                lam = 7.0;
                dark = 0.82;
                h = 9.0 + 6.0 * u01k(hb, 2lu);
            }
            if (abs(edge) > 4.5 * hw0 + fwb) {
                return o;
            }
            let clump = perlin3(hb, p * (1.0lf / f64(lam)));
            let b11 = band(14.0, gsd);
            var tb = 0.0;
            var gap = 1.0;
            if (!belt) {
                let tn = perlin3(hb ^ 3lu, p * (1.0lf / 14.0lf));
                tb = smoothstep1(0.3, 0.6, tn) * b11 + (1.0 - b11) * 0.08;
                gap = smoothstep1(-0.5, -0.2, tn) * b11 + (1.0 - b11) * 0.85;
            }
            // crowns of different size along the hedge: its width varies (the mean below the pixel)
            let hw = hw0 * (0.875 + 0.4 * clump * band(lam, gsd)) * (1.0 + 2.0 * tb);
            let c = gap * max(min(edge + 0.5 * fwb, hw) - max(edge - 0.5 * fwb, -hw), 0.0) / fwb;
            if (c <= 0.0) {
                return o;
            }
            let tex = 1.0 + 0.18 * perlin3(hb ^ 1lu, p * (1.0lf / 2.2lf)) * band(2.2, gsd);
            let ea = abs(edge) / hw;
            let across = sqrt(max(1.0 - ea * ea, 0.0));
            let hc = px.crown * dark * (0.85 + 0.3 * clump) * tex * (0.82 + 0.3 * across) * (1.0 - 0.1 * tb);
            let along = 0.55 + 0.45 * (0.5 + 0.5 * clump);
            o.col = mixc(i.col, hc, c);
            o.dh = lerp(i.dh, (h + 7.0 * tb) * across * along, c);
            if (c > 0.5) {
                o.cls = LC_HEDGEROW;
            }
        }
        case 3u: {
            let c = band_cov(edge, 0.45, fwb);
            let sc = srgb(150.0, 146.0, 136.0) * (0.88 + 0.24 * (0.5 + 0.5 * perlin3(hb, p * (1.0lf / 1.5lf))));
            o.col = mixc(i.col, sc, c);
            o.dh = lerp(i.dh, 1.0, c);
        }
        case 4u: {
            let c = band_cov(edge, 1.0, fwb);
            o.col = mixc(i.col, srgb(66.0, 76.0, 58.0), c);
            o.dh = i.dh - 0.5 * c;
        }
        case 5u: {
            let c = band_cov(edge, 2.5, fwb);
            o.col = mixc(i.col, srgb(162.0, 146.0, 120.0) * (1.0 + 0.06 * perlin3(hb, p * (1.0lf / 3.0lf)) * band(3.0, gsd)), c);
            o.dh = lerp(i.dh, 0.0, c);
            if (c > 0.5) {
                o.cls = LC_TRACK;
            }
        }
        case 7u: {
            let bank = band_cov(edge, 3.0, fwb);
            let w = band_cov(edge, 1.6, fwb);
            let c1 = mixc(i.col, srgb(128.0, 120.0, 92.0), bank);
            o.col = mixc(c1, srgb(50.0, 74.0, 78.0), w);
            o.dh = i.dh + 0.3 * (bank - w) - 0.6 * w;
            if (w > 0.5) {
                o.cls = LC_CANAL;
            }
        }
        default: {}
    }
    return o;
}
