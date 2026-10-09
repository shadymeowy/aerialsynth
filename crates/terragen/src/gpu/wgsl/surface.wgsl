// Fine-scale surface ("pass B", `surface.rs`): pixel fields, and the surface at one sub-sample.

// ---------------------------------------------------------------- pixel fields

const PF_N: u32 = 16u;
const PF_DETAIL: u32 = 0u;
const PF_PATCH: u32 = 1u;
const PF_LAND: u32 = 2u;
const PF_STRATA: u32 = 3u;
const PF_STRATA2: u32 = 4u;
const PF_SNOW: u32 = 5u;
const PF_FOREST: u32 = 6u;
const PF_STAND: u32 = 7u;
const PF_FIELD_VAR: u32 = 8u;
const PF_FIELD_VAR2: u32 = 9u;
const PF_FIELD_VAR3: u32 = 10u;
const PF_WARP2: u32 = 11u;
const PF_WATER: u32 = 12u;
/// the fields only some surfaces need, completed on first use
const PF_LAZY: u32 = (1u << 3u) | (1u << 4u) | (1u << 7u) | (1u << 8u) | (1u << 9u) | (1u << 10u) | (1u << 12u);

/// all octaves / the octaves >= cut / the octaves < cut
const SPLIT_ALL: u32 = 0u;
const SPLIT_LOW: u32 = 1u;
const SPLIT_HIGH: u32 = 2u;

/// fBm `f` evaluated at p·k (a field at p·k sees sample spacing gsd·k in its own domain).
fn pf_fbm(f: u32, k: f64, p: vec3<f64>, gsd: f32, split: u32, cut: f32) -> f32 {
    let kf = f32(k);
    if (split == SPLIT_ALL) {
        return fbm(f, p * k, gsd * kf);
    }
    return fbm_part(f, p * k, gsd * kf, cut * kf, split == SPLIT_LOW);
}

/// A single-octave term of wavelength `lam` belongs to the part?
fn pf_single(lam: f32, split: u32, cut: f32) -> bool {
    if (split == SPLIT_ALL) {
        return true;
    }
    return (lam >= cut) == (split == SPLIT_LOW);
}

/// Pixel field `i` (or a part of it, `SurfaceModel::pixel_field`).
fn pixel_field(i: u32, p: vec3<f64>, gsd: f32, split: u32, cut: f32) -> f32 {
    if (i >= 13u) {
        if (pf_single(180.0, split, cut)) {
            return perlin3(0x57A1lu + u64(i - 13u), p / 180.0lf);
        }
        return 0.0;
    }
    // the fBm and its scale per field, then one call (the driver inlines every call site of
    // the fBm: one per field made pass B take minutes to compile)
    var f = FBM_DETAIL;
    var k = 1.0lf;
    switch i {
        case 1u: { f = FBM_PATCH; }
        case 2u: { f = FBM_LAND; }
        case 3u: { f = FBM_STRATA; }
        case 4u: { f = FBM_STRATA; k = 1.7lf; }
        case 5u: { f = FBM_SNOW; }
        case 6u: { f = FBM_FOREST; }
        case 7u: { f = FBM_PATCH; k = 0.3lf; }
        case 8u: { f = FBM_FIELD_VAR; }
        case 9u: { f = FBM_FIELD_VAR; k = 1.7lf; }
        case 10u: { f = FBM_FIELD_VAR; k = 3.0lf; }
        case 11u: { f = FBM_WARP2; }
        case 12u: { f = FBM_PATCH; k = 0.37lf; }
        default: {}
    }
    var v = pf_fbm(f, k, p, gsd, split, cut);
    if (i == 2u || i == 6u) {
        v = v * fbm_norm(f) * 1.8;
    }
    if (i == 7u && pf_single(1200.0, split, cut)) {
        v += 0.5 * perlin3(0x57Alu, p / 1200.0lf) * band(1200.0, gsd);
    }
    return v;
}

/// Forest stand at `p` given the stand warp (`SurfaceModel::stand_id`).
fn stand_id(p: vec3<f64>, warp: vec3<f32>) -> u64 {
    let sp = p + vec3<f64>(warp * 70.0);
    return worley3(0x57A4lu, sp, 240.0lf, 0.9).id;
}

fn stand_warp_at(p: vec3<f64>) -> vec3<f32> {
    return vec3<f32>(perlin3(0x57A1lu, p / 180.0lf), perlin3(0x57A2lu, p / 180.0lf), perlin3(0x57A3lu, p / 180.0lf));
}

/// Grid nodes: the long octaves of the pixel fields and the forest stand.
fn grid_nodes_surface(ti: TileInfo, k: u32, ctx: Ctx, nb: u32, ib: u32) {
    for (var i = 0u; i < PF_N; i++) {
        node_f[nb + NODE_PF + i] = pixel_field(i, ctx.p, ctx.gsd, SPLIT_LOW, ti.pf_cut);
    }
    node_ids[ib + 6u] = stand_id(ctx.p, stand_warp_at(ctx.p));
}

// ---------------------------------------------------------------- surface tables (batch data)

@group(0) @binding(4) var<storage, read> pal: array<vec4<f32>>;

// (the class ids LC_* come from classes.wgsl, generated from landcover.rs)

/// Field system of a land-use region (`RegionInfo`).
struct Region {
    center: vec4<f64>,
    ex: vec4<f32>,
    ey: vec4<f32>,
    east: vec4<f32>,
    north: vec4<f32>,
    /// bits of `split` (the seed of the region's field lattices)
    split: u64,
    style: u32,
    _p: u32,
    fw: f32,
    fh: f32,
    hedge: f32,
    track: f32,
    border_w: f32,
    palette: f32,
    agri: f32,
    season: f32,
}

/// An existing town (`TownInfo`), with the reciprocal wavelengths of its footprint noise.
struct Town {
    center: vec4<f64>,
    inv_r09: f64,
    inv_r035: f64,
    seed: u64,
    _p: u64,
    ex: vec4<f32>,
    ey: vec4<f32>,
    sun: vec4<f32>,
    radius: f32,
    block: f32,
    street: f32,
    organic: f32,
    roof_style: f32,
    height: f32,
    lot: f32,
    elong: f32,
}

/// regions by id (open addressing, key 0 = empty) -> index into `regions`
@group(1) @binding(5) var<storage, read> region_keys: array<u64>;
@group(1) @binding(6) var<storage, read> region_idx: array<u32>;
@group(1) @binding(7) var<storage, read> regions: array<Region>;
/// town lattice cells (key: `town_cell_key`) -> (first, count) of their candidates in `town_list`
@group(1) @binding(8) var<storage, read> town_keys: array<u64>;
@group(1) @binding(9) var<storage, read> town_cells: array<vec2<u32>>;
@group(1) @binding(10) var<storage, read> town_list: array<u32>;
@group(1) @binding(11) var<storage, read> towns: array<Town>;

fn region_find(id: u64) -> i32 {
    let n = arrayLength(&region_idx);
    var k = u32(mix64(id) % u64(n));
    for (var i = 0u; i < n; i++) {
        let key = region_keys[k];
        if (key == id) {
            return i32(region_idx[k]);
        }
        if (key == 0lu) {
            return -1;
        }
        k = (k + 1u) % n;
    }
    return -1;
}

fn town_cell_of(p: vec3<f64>) -> vec3<i64> {
    let q = floor(p * (1.0lf / cfg.town_cell));
    return vec3<i64>(i64(q.x), i64(q.y), i64(q.z));
}

fn town_cell_key(c: vec3<i64>) -> u64 {
    return hash3(0x7C311lu, c.x, c.y, c.z) | 1lu;
}

/// (first, count) of the candidate towns of a town lattice cell; count 0xffffffff: unknown.
fn town_cell_find(c: vec3<i64>) -> vec2<u32> {
    let key = town_cell_key(c);
    let n = arrayLength(&town_cells);
    var k = u32(mix64(key) % u64(n));
    for (var i = 0u; i < n; i++) {
        let kk = town_keys[k];
        if (kk == key) {
            return town_cells[k];
        }
        if (kk == 0lu) {
            return vec2<u32>(0u, 0xffffffffu);
        }
        k = (k + 1u) % n;
    }
    return vec2<u32>(0u, 0xffffffffu);
}

// ---------------------------------------------------------------- helpers

fn mixc(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * clamp(t, 0.0, 1.0);
}

fn s2l(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn l2s(c0: f32) -> f32 {
    let c = clamp(c0, 0.0, 1.0);
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn srgb(r: f32, g: f32, b: f32) -> vec3<f32> {
    return vec3<f32>(s2l(r / 255.0), s2l(g / 255.0), s2l(b / 255.0));
}

fn pal3(i: u32) -> vec3<f32> {
    return pal[i].xyz;
}

fn max3(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

/// A small light source of peak `amp` and radius `sigma` at squared distance `d2`, widened to
/// the sample spacing `fw` with its energy kept.
fn point_light(d2: f32, amp: f32, sigma: f32, fw: f32) -> f32 {
    let s = max(sigma, 0.6 * fw);
    return amp * (sigma * sigma) / (s * s) * exp(-d2 / (2.0 * s * s));
}

/// Coverage of a band of half-width `hw` at distance `d`, box-filtered with width `fw`.
fn band_cov(d: f32, hw: f32, fw: f32) -> f32 {
    return clamp((hw - abs(d)) / fw + 0.5, 0.0, 1.0);
}

fn rem_euclid(x: f32, m: f32) -> f32 {
    return x - m * floor(x / m);
}

/// Crop kind for a field from the region's season and a random draw (`crop_kind`).
fn crop_kind(season: f32, u: f32, dry: bool) -> u32 {
    var spring = array<f32, 9>(0.35, 0.25, 0.0, 0.0, 0.15, 0.04, 0.13, 0.02, 0.06);
    var summer = array<f32, 9>(0.14, 0.10, 0.30, 0.20, 0.08, 0.03, 0.10, 0.0, 0.05);
    var autumn = array<f32, 9>(0.10, 0.03, 0.02, 0.20, 0.33, 0.12, 0.12, 0.0, 0.08);
    var acc = 0.0;
    var kind = 8u;
    for (var k = 0u; k < 9u; k++) {
        var a = 0.0;
        var b = 0.0;
        var f = 0.0;
        if (season < 0.5) {
            a = spring[k];
            b = summer[k];
            f = season * 2.0;
        } else {
            a = summer[k];
            b = autumn[k];
            f = season * 2.0 - 1.0;
        }
        acc += a + (b - a) * f;
        if (u < acc) {
            kind = k;
            break;
        }
    }
    if (dry && kind <= 1u) {
        kind = select(3u, 5u, u < 0.5);
    }
    return kind;
}

// ---------------------------------------------------------------- pixel fields of a pixel

/// The pixel fields shared by a pixel's sub-samples; the lazy ones are completed on first use
/// (`PixFields`).
struct PixFields {
    f: array<f32, 16>,
    pending: u32,
    p: vec3<f64>,
    gsd: f32,
    cut: f32,
    split: u32,
    has_stand: u32,
    stand: u64,
}

fn pf_lazy(pf: ptr<function, PixFields>, i: u32) -> f32 {
    if (((*pf).pending & (1u << i)) != 0u) {
        (*pf).f[i] += pixel_field(i, (*pf).p, (*pf).gsd, (*pf).split, (*pf).cut);
        (*pf).pending &= ~(1u << i);
    }
    return (*pf).f[i];
}

/// Inputs to pass B at one sub-sample (`Local`), besides the nearest pixel's `Terrain`.
struct Local {
    ground: f32,
    water: f32,
    water_kind: u32,
    river_d: f32,
    river_hw: f32,
    river_level: f32,
    road_major: f32,
    road_minor: f32,
    slope: f32,
    fw: f32,
}

struct Surface {
    albedo: vec3<f32>,
    height: f32,
    cls: u32,
    lit: f32,
    emission: vec3<f32>,
}

// ---------------------------------------------------------------- trees

struct TreeLayer {
    color: vec3<f32>,
    cell: f32,
    tone: vec3<f32>,
    density: f32,
    seed: u64,
    closure: f32,
    height: f32,
    conifer: f32,
    scale: f32,
}

/// (colour, canopy height, coverage)
struct Trees {
    col: vec3<f32>,
    h: f32,
    cov: f32,
}

/// Tree crowns of the layers in `mask` at local position `q` (`SurfaceModel::trees`).
fn trees(layers: array<TreeLayer, 4>, mask: u32, q: vec2<f32>, gsd: f32, fw: f32, p: vec3<f64>) -> Trees {
    var best_h = 0.0;
    var best_col = vec3<f32>(0.0);
    var cov_total = 0.0;
    var mean_col = vec3<f32>(0.0);
    var mean_w = 0.0;
    var mean_h = 0.0;
    for (var li = 0u; li < 4u; li++) {
        if ((mask & (1u << li)) == 0u) {
            continue;
        }
        let layer = layers[li];
        if (layer.density <= 1e-4) {
            continue;
        }
        let expl = smoothstep1(1.2 * gsd, 3.0 * gsd, layer.cell);
        // expected coverage when unresolved
        let mean_r = layer.cell * 0.48;
        let cov_mean = min(layer.density * PI * mean_r * mean_r / (layer.cell * layer.cell), 1.0);
        if (expl < 1.0) {
            let w = cov_mean * (1.0 - expl);
            let l3 = layer.cell * 3.0;
            let l9 = layer.cell * 9.0;
            var tex = 0.0;
            let b3 = band(l3, gsd);
            if (b3 > 0.0) {
                tex += perlin3(layer.seed, p / f64(l3)) * b3;
            }
            let b9 = band(l9, gsd);
            if (b9 > 0.0) {
                tex += 0.6 * perlin3(layer.seed ^ 5lu, p / f64(l9)) * b9;
            }
            mean_col += layer.color * layer.tone * (0.85 + 0.3 * tex) * w;
            mean_w += w;
            mean_h += layer.height * 0.6 * cov_mean * (1.0 - expl);
        }
        if (expl <= 0.0) {
            continue;
        }
        let cq = q / layer.cell;
        let cf = floor(cq);
        let ix = i64(cf.x);
        let iy = i64(cf.y);
        let fq = cq - cf;
        let reach = (0.6 * 1.55 * max(layer.scale, 0.0) + fw / layer.cell) * (1.0 + 1e-6) + 1e-6;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let gx = nb_gap(fq.x, dx, 0.4);
                let gy = nb_gap(fq.y, dy, 0.4);
                if (gx * gx + gy * gy > reach * reach) {
                    continue;
                }
                let h = hash2(layer.seed, ix + i64(dx), iy + i64(dy));
                let u = u01k(h, 3lu);
                if (u >= layer.density) {
                    continue;
                }
                let fade = smoothstep1(0.0, 0.3, (layer.density - u) / max(layer.density, 1e-6));
                // crown centre relative to q (lattice units: exact small numbers)
                let rel = vec2<f32>(f32(dx), f32(dy)) + 0.5 + 0.8 * (vec2<f32>(u01k(h, 1lu), u01k(h, 2lu)) - 0.5) - fq;
                let r = layer.cell * (0.36 + 0.24 * u01k(h, 4lu)) * (1.0 + 0.55 * smoothstep1(0.35, 0.9, layer.closure)) * layer.scale;
                let d = length(rel) * layer.cell;
                if (d > r + fw) {
                    continue;
                }
                let cov = clamp((r - d) / fw + 0.5, 0.0, 1.0) * expl * fade;
                let x = min(d / r, 1.0);
                let hh = layer.height * (0.55 + 0.9 * u01k(h, 5lu)) * (0.45 + 0.55 * layer.scale);
                var prof = 0.0;
                if (layer.conifer > 0.5) {
                    prof = 0.08 + 0.92 * pow(1.0 - x, 1.15);
                } else {
                    prof = 0.12 + 0.88 * sqrt(1.0 - x * x);
                }
                var clump = 0.0;
                let b16 = band(1.6, gsd);
                if (b16 > 0.0) {
                    clump = perlin3(h ^ 0xC1lu, p * (1.0lf / 1.6lf)) * b16;
                    let b08 = band(0.8, gsd);
                    if (b08 > 0.0) {
                        clump += 0.5 * perlin3(h ^ 0xC2lu, p * (1.0lf / 0.8lf)) * b08;
                    }
                }
                let th = hh * prof * (1.0 + 0.10 * clump * (0.3 + 0.7 * (1.0 - x))) * fade;
                if (th > best_h) {
                    best_h = th;
                    let tint = 0.86 + 0.24 * u01k(h, 6lu);
                    let hv = u01k(h, 7lu) - 0.5;
                    let hue = vec3<f32>(1.0 + 0.16 * hv, 1.0 + 0.04 * (u01k(h, 9lu) - 0.5), 1.0 - 0.12 * hv);
                    var leaf = 0.82;
                    let b11 = band(1.1, gsd);
                    if (b11 > 0.0) {
                        leaf += 0.36 * (0.5 + 0.5 * perlin3(h, p * (1.0lf / 1.1lf))) * b11;
                    }
                    best_col = layer.color * layer.tone * tint * hue * leaf * (0.75 + 0.35 * (1.0 - x));
                }
                cov_total = max(cov_total, cov);
            }
        }
    }
    var o: Trees;
    let mean_cov = min(mean_w, 1.0);
    var mc = vec3<f32>(0.0);
    if (mean_w > 0.0) {
        mc = mean_col / mean_w;
    }
    let cov = min(cov_total + mean_cov * (1.0 - cov_total), 1.0);
    if (cov <= 0.0) {
        o.col = vec3<f32>(0.0);
        o.h = 0.0;
        o.cov = 0.0;
        return o;
    }
    if (cov_total > 0.0) {
        o.col = mixc(mc, best_col, cov_total / cov);
    } else {
        o.col = mc;
    }
    o.h = max(best_h, mean_h);
    o.cov = cov;
    return o;
}

// ---------------------------------------------------------------- farmsteads

/// (colour, height, coverage, cls, emission); ok = false: none
struct Built {
    ok: bool,
    col: vec3<f32>,
    h: f32,
    cov: f32,
    cls: u32,
    em: vec3<f32>,
}

fn boxc(rel: vec2<f32>, c: vec2<f32>, half: vec2<f32>, fwe: f32) -> f32 {
    let d4 = half - abs(rel - c);
    return clamp(min(d4.x, d4.y) / fwe + 0.5, 0.0, 1.0);
}

/// Farmstead: house, barn, gravel yard and yard lamp on a sparse lattice (`farmstead`).
fn farmstead(r: Region, t: Terrain, q: vec2<f32>, gsd: f32, fw: f32) -> Built {
    var o: Built;
    o.ok = false;
    let wc = worley2(r.split ^ 0xFA4lu, q, 650.0, 0.8);
    let fid = wc.id;
    if (u01k(fid, 1lu) > 0.55 * smoothstep1(0.05, 0.4, t.agri)) {
        return o;
    }
    let rel0 = q - wc.point;
    if (length(rel0) > 60.0) {
        return o;
    }
    let ang = (u01k(fid, 2lu) - 0.5) * 0.4;
    let sa = sin(ang);
    let ca = cos(ang);
    let rel = vec2<f32>(rel0.x * ca + rel0.y * sa, -rel0.x * sa + rel0.y * ca);
    let fwe = max(fw, 0.3 * gsd);
    let yard_half = vec2<f32>(20.0 + 10.0 * u01k(fid, 3lu), 14.0 + 8.0 * u01k(fid, 4lu));
    let yard = boxc(rel, vec2<f32>(0.0), yard_half, fwe);
    let lamp = rel - vec2<f32>(2.0, 6.0);
    let d2 = dot(lamp, lamp);
    var lamp_col = vec3<f32>(0.86, 0.92, 1.0);
    if (u01k(fid, 9lu) < 0.6) {
        lamp_col = vec3<f32>(1.0, 0.72, 0.38);
    }
    let res = band(10.0, gsd);
    let emission = lamp_col * (0.04 * exp(-d2 / (2.0 * 7.0 * 7.0)) + point_light(d2, 5.0, 0.4, fw)) * res
        + lamp_col * 0.02 * (1.0 - res) * (1.0 - smoothstep1(20.0, 60.0, length(rel)));
    if (yard <= 0.0) {
        if (max3(emission) > 1e-4) {
            o.ok = true;
            o.col = vec3<f32>(0.0);
            o.h = 0.0;
            o.cov = 0.0;
            o.cls = LC_CROP;
            o.em = emission;
        }
        return o;
    }
    var col = mixc(pal3(PAL_GRAVEL), pal3(PAL_CONCRETE), 0.3 * u01k(fid, 5lu)) * (0.9 + 0.2 * u01k(fid, 6lu));
    var h = 0.0;
    var cls = LC_URBAN;
    let hc = vec2<f32>(-yard_half.x * 0.45, -yard_half.y * 0.3);
    let hh = vec2<f32>(5.5, 4.5);
    let house = boxc(rel, hc, hh, fwe);
    if (house > 0.0) {
        let roof = pal3(PAL_ROOFS + u32(u01k(fid, 7lu) * 2.99)) * (0.9 + 0.2 * u01k(fid, 8lu));
        let ridge = clamp((hh.y - abs(rel.y - hc.y)) / hh.y, 0.0, 1.0);
        col = mixc(col, roof, house);
        h = (5.5 + 2.0 * ridge) * house;
        cls = LC_BUILDING;
    }
    let bc = vec2<f32>(yard_half.x * 0.3, yard_half.y * 0.35);
    let bh = vec2<f32>(11.0 + 5.0 * u01k(fid, 10lu), 6.0);
    let barn = boxc(rel, bc, bh, fwe);
    if (barn > 0.0) {
        let roof = mixc(pal3(PAL_ROOFS + 4u), pal3(PAL_ROOFS + 2u), u01k(fid, 11lu));
        let ridge = clamp((bh.y - abs(rel.y - bc.y)) / bh.y, 0.0, 1.0);
        col = mixc(col, roof, barn);
        h = max(h, (6.0 + 1.5 * ridge) * barn);
        cls = LC_BUILDING;
    }
    o.ok = true;
    o.col = col;
    o.h = h;
    o.cov = yard * max(smoothstep1(1.0 * gsd, 3.0 * gsd, 30.0), 0.25);
    o.cls = cls;
    o.em = emission;
    return o;
}

// ---------------------------------------------------------------- fields

/// (colour, extra height, coverage, kind: 0 crop, 1 hedge, 2 track); ok = false: none
struct FieldOut {
    ok: bool,
    col: vec3<f32>,
    h: f32,
    cov: f32,
    kind: u32,
}

/// Contiguous cultivation zones: a smooth mask (compared with the cultivated fraction).
fn cultivated_mask(c3: vec3<f64>, gsd: f32) -> f32 {
    return 0.5 + 0.5 * fbm(FBM_CULT, c3, min(gsd, 100.0)) * fbm_norm(FBM_CULT) * 2.2;
}

fn field_explicit(r: Region, t: Terrain, q: vec2<f32>, p: vec3<f64>, gsd: f32, fw: f32, cult: f32, tint: vec3<f32>, pf: ptr<function, PixFields>) -> FieldOut {
    var o: FieldOut;
    o.ok = false;
    let seed = r.split;
    var id = 0lu;
    var fx = 0.0;
    var fy = 0.0;
    var edge = 0.0;
    var inside = 1.0;
    var fc = vec2<f32>(0.0);
    if (r.style == 0u || r.style == 3u) {
        let inv900 = 1.0lf / 900.0lf;
        let wq = vec2<f32>(perlin3(seed, p * inv900), perlin3(seed ^ 1lu, p * inv900));
        let q2 = q + wq * (0.08 * r.fw);
        let w = r.fw;
        let h = max(r.fh, r.fw);
        // rows with jittered heights; within a row, jittered field boundaries
        var j = i64(floor(q2.y / h));
        if (q2.y < row_line(seed, j, h)) {
            j -= 1li;
        } else if (q2.y >= row_line(seed, j + 1li, h)) {
            j += 1li;
        }
        let y0 = row_line(seed, j, h);
        let y1 = row_line(seed, j + 1li, h);
        let shift = u01(hash1(seed, j)) * w;
        let x = q2.x + shift;
        var i = i64(floor(x / w));
        if (x < col_line(seed, j, i, w)) {
            i -= 1li;
        } else if (x >= col_line(seed, j, i + 1li, w)) {
            i += 1li;
        }
        let x0 = col_line(seed, j, i, w);
        let x1 = col_line(seed, j, i + 1li, w);
        let cw = x1 - x0;
        let ch = y1 - y0;
        fx = x - x0;
        let fy0 = q2.y - y0;
        // split some cells into strips
        let hc = hash2(seed ^ 0x55lu, i, j);
        var smax = 3.5;
        if (r.style == 3u) {
            smax = 1.0;
        }
        let nstrip = 1 + i32(u01k(hc, 1lu) * smax);
        let sh = ch / f32(nstrip);
        let k = min(floor(fy0 / sh), f32(nstrip - 1));
        fy = fy0 - k * sh;
        edge = min(min(fx, cw - fx), min(fy, sh - fy));
        fc = vec2<f32>(x0 + 0.5 * cw - shift, y0 + k * sh + 0.5 * sh);
        id = mix64(hc ^ u64(max(k, 0.0)));
    } else if (r.style == 1u) {
        // irregular fields: Voronoi cells in a stretched frame, cell size varying over the region
        let aspect = 1.0 + 2.2 * u01k(seed, 21lu);
        let qs = vec2<f32>(q.x / aspect, q.y);
        let scale = 0.7 + 0.6 * (0.5 + 0.5 * perlin3(seed ^ 0x5CA1lu, p * (1.0lf / 2500.0lf)));
        let cell = r.fw / sqrt(aspect) * scale;
        let wc = worley2(seed, qs, cell, 0.85);
        let dn = wc.point2 - wc.point;
        let nrm = length(vec2<f32>(dn.x / aspect, dn.y)) / max(length(dn), 1e-9);
        edge = worley2_edge_dist(wc, qs) / max(nrm, 1e-6);
        let rel = q - vec2<f32>(wc.point.x * aspect, wc.point.y);
        id = wc.id;
        fx = rel.x;
        fy = rel.y;
        fc = vec2<f32>(wc.point.x * aspect, wc.point.y);
    } else {
        // centre pivots
        let s = r.fw;
        let cq = floor(q / s);
        let c = (cq + 0.5) * s;
        let rel = q - c;
        let d = length(rel);
        let rad = s * 0.48;
        let hc = hash2(seed, i64(cq.x), i64(cq.y));
        inside = band_cov(d, rad, fw);
        let nsec = 1 + i32(u01k(hc, 1lu) * 3.0);
        let ang = atan2(rel.y, rel.x) + PI + u01k(hc, 2lu) * 6.0;
        var sec = i32(floor(ang / TAU * f32(nsec))) % nsec;
        if (sec < 0) {
            sec += nsec;
        }
        edge = abs(rad - d);
        id = mix64(hc ^ u64(sec));
        fx = rel.x;
        fy = rel.y;
        fc = c;
    }
    if (inside <= 0.0) {
        return o;
    }
    // is this field cultivated? (both depend on the field only)
    let c3 = r.center.xyz + vec3<f64>(r.ex.xyz * fc.x + r.ey.xyz * fc.y);
    if (u01k(id, 5lu) < 0.06) {
        return o;
    }
    let mask = cultivated_mask(c3, gsd);
    if (mask >= cult) {
        return o;
    }
    let cluster = worley2(seed ^ 0xC1Clu, fc, 700.0, 1.0).id;
    var u_crop = u01k(id, 6lu);
    if (u01k(id, 16lu) < 0.5) {
        u_crop = u01k(cluster, 6lu);
    }
    let kind = crop_kind(r.season, u_crop, t.moist < 0.33 && r.style != 2u);
    let tropic = smoothstep1(19.0, 25.0, t.temp) * smoothstep1(0.5, 0.7, t.moist);
    var col = vec3<f32>(0.0);
    var extra_h = 0.0;
    if (u01k(id, 30lu) < tropic) {
        let tk = u_crop;
        var fine = 1.0;
        let b6 = band(6.0, gsd);
        if (b6 > 0.0) {
            fine += 0.08 * perlin3(id ^ 0x7A1lu, p * (1.0lf / 6.0lf)) * b6;
            let b15 = band(1.5, gsd);
            if (b15 > 0.0) {
                fine += 0.06 * perlin3(id ^ 0x7A2lu, p * (1.0lf / 1.5lf)) * b15;
            }
        }
        if (tk < 0.35) {
            // rice paddies cut into small plots by earth bunds
            col = mixc(srgb(74.0, 106.0, 62.0), srgb(58.0, 82.0, 74.0), u01k(id, 31lu)) * fine;
            let sx = 18.0 + 22.0 * u01k(id, 32lu);
            let sy = sx * (1.2 + 0.8 * u01k(id, 33lu));
            let mx = rem_euclid(fx, sx);
            let my = rem_euclid(fy, sy);
            let d = min(min(mx, sx - mx), min(my, sy - my));
            let bund = band_cov(d, 0.6, fw) * band(sx, gsd);
            col = mixc(col, srgb(108.0, 106.0, 74.0), bund);
            extra_h = 0.3 * bund;
        } else if (tk < 0.75) {
            // oil palm: star-shaped crowns on a triangular grid
            let age = 0.35 + 0.65 * u01k(id, 34lu);
            let sx = 9.0;
            let sy = 7.8;
            let j = round(fy / sy);
            var off = 0.5 * sx;
            if ((i32(j) % 2 + 2) % 2 == 0) {
                off = 0.0;
            }
            let i = round((fx - off) / sx);
            let rel = vec2<f32>(fx - off - i * sx, fy - j * sy);
            let hh = mix64(id ^ bitcast<u64>(i64(i)) ^ (bitcast<u64>(i64(j)) << 20u));
            let ang = atan2(rel.y, rel.x) + u01k(hh, 1lu) * 6.3;
            let r_eff = 4.4 * age * (0.82 + 0.18 * cos(8.0 * ang));
            let d = length(rel);
            let expl = band(sx, gsd);
            let crown = band_cov(d, r_eff, fw) * expl + (1.0 - expl) * (0.75 * age);
            let ground = srgb(92.0, 100.0, 60.0) * fine;
            let shade = 0.85 + 0.25 * (1.0 - min(d / max(r_eff, 0.1), 1.0));
            col = mixc(ground, srgb(58.0, 92.0, 44.0) * shade * fine, crown);
            extra_h = (3.0 + 9.0 * age) * crown;
        } else if (tk < 0.92) {
            // sugarcane / banana in rows
            col = srgb(86.0, 116.0, 60.0) * fine;
            col *= 1.0 + 0.08 * sin(fx * TAU / 1.5) * band(1.5, gsd);
            extra_h = 2.5;
        } else {
            // freshly ploughed red laterite
            col = srgb(152.0, 90.0, 62.0) * fine;
            col *= 1.0 + 0.1 * sin(fx * TAU / 0.9) * band(0.9, gsd);
        }
        col *= tint;
    } else {
        col = pal3(PAL_CROP + kind);
        col *= 0.94 + 0.12 * u01k(id, 8lu);
        col *= tint;
        // within-field variation at several scales
        var var1 = 0.10 * pf_lazy(pf, PF_FIELD_VAR) + 0.08 * perlin3(id ^ 1lu, p / f64(0.8 * r.fw));
        let b35 = band(35.0, gsd);
        if (b35 > 0.0) {
            var1 += 0.12 * perlin3(id, p * (1.0lf / 35.0lf)) * b35;
        }
        col *= 1.0 + var1;
        // growth zones
        let b55 = band(55.0, gsd);
        if (b55 > 0.0) {
            let gz = perlin3(id ^ 0x6A0lu, p * (1.0lf / 55.0lf)) * b55;
            col = mixc(col, col * vec3<f32>(1.12, 1.04, 0.82), 0.5 * smoothstep1(0.0, 0.6, gz));
            col = mixc(col, col * vec3<f32>(0.86, 0.93, 0.88), 0.5 * smoothstep1(0.0, 0.6, -gz));
        }
        // management direction: rows / tramlines along one field axis, along the edge in the
        // headland
        var row_ang = 0.0;
        if (u01k(id, 9lu) >= 0.7) {
            row_ang = 0.5 * PI;
        }
        let headland_w = 8.0 + 10.0 * u01k(id, 11lu);
        let in_headland = edge < headland_w && kind <= 4u;
        if (in_headland) {
            row_ang += 0.5 * PI;
        }
        let along = fx * cos(row_ang) + fy * sin(row_ang);
        // soil / growth texture
        var tex = 0.0;
        let b90 = band(90.0, gsd);
        if (b90 > 0.0) {
            tex += 0.10 * perlin3(id ^ 0x7E3lu, p * (1.0lf / 90.0lf)) * b90;
            let b22 = band(22.0, gsd);
            if (b22 > 0.0) {
                tex += 0.12 * perlin3(id ^ 0x7E4lu, p * (1.0lf / 22.0lf)) * b22;
                let b7 = band(7.0, gsd);
                if (b7 > 0.0) {
                    tex += 0.18 * perlin3(id ^ 0x7E1lu, p * (1.0lf / 7.0lf)) * b7;
                    let b25 = band(2.5, gsd);
                    if (b25 > 0.0) {
                        tex += 0.13 * perlin3(id ^ 0x7E2lu, p * (1.0lf / 2.5lf)) * b25;
                        let b09 = band(0.9, gsd);
                        if (b09 > 0.0) {
                            tex += 0.07 * perlin3(id ^ 0x7E5lu, p * (1.0lf / 0.9lf)) * b09;
                        }
                    }
                }
            }
        }
        col *= 1.0 + tex;
        // wet hollows / bare patches inside some fields
        if (u01k(id, 12lu) < 0.35) {
            let wp = perlin3(id ^ 0x5A7lu, p / f64(40.0 + 60.0 * u01k(id, 13lu)));
            let m = smoothstep1(0.25, 0.45, wp) * max(band(30.0, gsd), 0.3);
            col = mixc(col, col * vec3<f32>(0.82, 0.86, 0.80), m);
        }
        if (in_headland) {
            col *= 0.96 + 0.03 * u01k(id, 14lu);
        }
        if (kind == 0u || kind == 1u) {
            let sp = 0.8 + 0.8 * u01k(id, 10lu);
            col *= 1.0 + 0.12 * sin(along * TAU / sp) * band(sp, gsd);
        } else if (kind == 3u) {
            let sp = 6.0 + 4.0 * u01k(id, 10lu);
            var sg = 1.0;
            if (sin(along / sp * TAU) < 0.0) {
                sg = -1.0;
            }
            col *= 1.0 + 0.07 * sg * band(sp, gsd);
        } else if (kind == 4u) {
            let sp = 0.45;
            col *= 1.0 + 0.15 * sin(along * TAU / sp) * band(sp, gsd);
            col *= 1.0 - 0.15 * smoothstep1(0.2, 0.6, pf_lazy(pf, PF_FIELD_VAR2));
        } else if (kind == 8u) {
            // orchard: rows of small trees
            let sx = 5.0 + 2.0 * u01k(id, 10lu);
            let sy = 4.0;
            let gx = round(fx / sx) * sx;
            let gy = round(fy / sy) * sy;
            let d = length(vec2<f32>(fx - gx, fy - gy));
            let expl = band(sx, gsd);
            let cov = band_cov(d, 1.7, fw) * expl + (1.0 - expl) * 0.4;
            col = mixc(col, pal3(PAL_CROWN_DECID) * 1.1, cov);
            extra_h = 4.0 * cov;
        }
        // tramlines in cereals / green crops / stubble
        if ((kind <= 3u || kind == 7u) && !in_headland) {
            let sp = 18.0 + 18.0 * floor(u01k(id, 15lu) * 2.0) / 2.0;
            let m = rem_euclid(along, sp);
            let tl = band_cov(m - 0.9, 0.22, fw) + band_cov(m - 2.7, 0.22, fw);
            let vis = max(band(1.2, gsd), 0.35 * band(sp, gsd));
            col = mixc(col, col * vec3<f32>(0.78, 0.76, 0.74), tl * vis);
        }
    }
    // field borders: hedges, tracks or a thin margin
    let bw = r.border_w;
    let bcov = band_cov(edge, bw, max(fw, gsd * 0.4));
    var kind_out = 0u;
    if (bcov > 0.0) {
        let hb = mix64(id ^ 0xED6Elu);
        if (u01k(hb, 1lu) < r.hedge) {
            let hcol = pal3(PAL_CROWN_DECID) * (0.8 + 0.3 * pf_lazy(pf, PF_FIELD_VAR3));
            col = mixc(col, hcol, bcov);
            let e = abs(edge) / max(bw, 0.1);
            let across = sqrt(max(1.0 - e * e, 0.0));
            let along = 0.45 + 0.55 * (0.5 + 0.5 * perlin3(hb, p * (1.0lf / 6.0lf)));
            extra_h = lerp(extra_h, (2.2 + 2.3 * u01k(hb, 2lu)) * across * along, bcov);
            if (bcov > 0.5) {
                kind_out = 1u;
            }
        } else if (u01k(hb, 3lu) < r.track * 0.4) {
            col = mixc(col, pal3(PAL_GRAVEL), bcov);
            if (bcov > 0.5) {
                kind_out = 2u;
            }
        } else {
            col = mixc(col, mixc(pal3(PAL_GRASS_DRY), pal3(PAL_GRASS_WET), t.moist), bcov * 0.7);
        }
    }
    o.ok = true;
    o.col = col;
    o.h = extra_h;
    o.cov = inside;
    o.kind = kind_out;
    return o;
}

fn row_line(seed: u64, j: i64, h: f32) -> f32 {
    return (f32(j) + 0.32 * (u01(hash2(seed ^ 0x40lu, 7li, j)) - 0.5)) * h;
}

fn col_line(seed: u64, j: i64, i: i64, w: f32) -> f32 {
    return (f32(i) + 0.45 * (u01(hash2(seed ^ 0x41lu, j, i)) - 0.5)) * w;
}

/// Agricultural field at rotated local coords (`SurfaceModel::field`).
fn field(r: Region, t: Terrain, q: vec2<f32>, p: vec3<f64>, gsd: f32, fw: f32, pf: ptr<function, PixFields>) -> FieldOut {
    let cult = smoothstep1(0.02, 0.45, t.agri);
    let tint = vec3<f32>(1.0 + 0.08 * (r.palette - 0.5), 1.0, 1.0 - 0.06 * (r.palette - 0.5));
    // fields near the pixel size: contrast reduced towards the mean
    var fsize = r.fw;
    if (r.fh > 0.0) {
        fsize = min(r.fw, r.fh);
    }
    let k = fsize / sqrt(fsize * fsize + 4.0 * gsd * gsd);
    let tropic = smoothstep1(19.0, 25.0, t.temp) * smoothstep1(0.5, 0.7, t.moist);
    let mean = mixc(pal3(PAL_CROP_MEAN), srgb(78.0, 100.0, 58.0), tropic) * tint * (1.0 + 0.10 * pf_lazy(pf, PF_FIELD_VAR));
    let fwe = max(fw, 0.6 * gsd);
    var fe = field_explicit(r, t, q, p, gsd, fwe, cult, tint, pf);
    if (!fe.ok) {
        fe.col = mean;
        fe.h = 0.0;
        fe.cov = 0.0;
        fe.kind = 0u;
    }
    let cov_m = lerp(cult * 0.9, fe.cov, k);
    var o: FieldOut;
    o.ok = cov_m > 0.0;
    if (fe.cov > 0.0) {
        o.col = mixc(mean, fe.col, k);
    } else {
        o.col = mean;
    }
    o.h = fe.h * k;
    o.cov = cov_m;
    o.kind = fe.kind;
    return o;
}

// ---------------------------------------------------------------- towns

/// How built-up a town is at `p` (1 in the centre, 0 outside its footprint) and the relative
/// distance from its centre (`town_urban`).
fn town_urban(town: Town, p: vec3<f64>, gsd: f32, slope: f32, clear: f32, warp2: f32) -> vec2<f32> {
    let d = vec3<f32>(p - town.center.xyz);
    let q0 = vec2<f32>(dot(d, town.ex.xyz), dot(d, town.ey.xyz));
    let r = town.radius;
    let e = town.elong;
    let q2 = dot(q0, q0);
    let r2 = 4.0 * r * r * (1.0 + 1e-6);
    if ((e >= 1.0 && q2 > r2 * e) || (e < 1.0 && e * q2 > r2)) {
        return vec2<f32>(0.0, 2.0);
    }
    let se = sqrt(town.elong);
    let qa = vec2<f32>(q0.x / se, q0.y * se);
    if (length(qa) > r * 2.0) {
        return vec2<f32>(0.0, 2.0);
    }
    var n1 = 0.32 * perlin3(town.seed ^ 0x71lu, p * town.inv_r09) + 0.08 * warp2;
    let b = band(0.35 * r, gsd);
    if (b > 0.0) {
        n1 += 0.18 * perlin3(town.seed ^ 0x72lu, p * town.inv_r035) * b;
    }
    let rel = length(qa) / max(r * (1.0 + n1), 1.0);
    return vec2<f32>((1.0 - smoothstep1(0.3, 1.0, rel)) * (1.0 - smoothstep1(0.45, 0.8, slope)) * clear, rel);
}

fn street_line(town: Town, axis: i64, i: i64) -> f32 {
    return (f32(i) + 0.36 * (u01(hash2(town.seed ^ 0x5EE7lu, axis, i)) - 0.5)) * town.block;
}

/// (cell index, position in the cell, cell size) along one axis of the street grid
struct GridCell {
    i: i64,
    q: f32,
    s: f32,
}

fn street_cell(town: Town, axis: i64, x: f32) -> GridCell {
    var i = i64(floor(x / town.block));
    if (x < street_line(town, axis, i)) {
        i -= 1li;
    } else if (x >= street_line(town, axis, i + 1li)) {
        i += 1li;
    }
    let a0 = street_line(town, axis, i);
    let a1 = street_line(town, axis, i + 1li);
    var g: GridCell;
    g.i = i;
    g.q = x - a0;
    g.s = a1 - a0;
    return g;
}

fn seg_here(h: u64, u: f32) -> f32 {
    if (u > 0.15 + 0.12 * u01k(h, 4lu) && !(u < 0.45 && u01k(h, 3lu) < 0.4)) {
        return 1.0;
    }
    return 0.0;
}

fn lamp_kind(k: u32) -> vec3<f32> {
    switch k {
        case 0u: { return vec3<f32>(1.0, 0.48, 0.12); }
        case 1u: { return vec3<f32>(1.0, 0.72, 0.38); }
        default: { return vec3<f32>(0.86, 0.92, 1.0); }
    }
}

/// A town at a point (`SurfaceModel::town` without its cast shadows): colour, height above
/// ground, coverage, cls, emission; `shadow_ok`: the point may lie in a building's shadow,
/// `street`, `resolved`: for the shadow.
struct TownPx {
    ok: bool,
    col: vec3<f32>,
    h: f32,
    cov: f32,
    cls: u32,
    em: vec3<f32>,
    shadow_ok: bool,
    street: f32,
    resolved: f32,
}

fn town_core(town: Town, p: vec3<f64>, gsd: f32, fw: f32, slope: f32, clear: f32, detail: f32, warp2: f32) -> TownPx {
    var o: TownPx;
    o.ok = false;
    let dd = vec3<f32>(p - town.center.xyz);
    let q0 = vec2<f32>(dot(dd, town.ex.xyz), dot(dd, town.ey.xyz));
    let ur = town_urban(town, p, gsd, slope, clear, warp2);
    let urban = ur.x;
    let rel = ur.y;
    if (urban <= 0.0) {
        return o;
    }
    // organic (curved) streets in old towns
    let inv350 = 1.0lf / 350.0lf;
    let warp = vec2<f32>(perlin3(town.seed, p * inv350), perlin3(town.seed ^ 9lu, p * inv350)) * (35.0 * town.organic);
    let q = q0 + warp;
    let cx = street_cell(town, 0li, q.x);
    let cy = street_cell(town, 1li, q.y);
    let bix = cx.i;
    let bqx = cx.q;
    let bsx = cx.s;
    let biy = cy.i;
    let bqy = cy.q;
    let bsy = cy.s;
    let sw = town.street * 0.7;
    var near_x = bix;
    if (bqx > bsx - bqx) {
        near_x = bix + 1li;
    }
    var near_y = biy;
    if (bqy > bsy - bqy) {
        near_y = biy + 1li;
    }
    // each street segment exists or not as a whole (town density at its midpoint)
    let ymid = 0.5 * (street_line(town, 1li, biy) + street_line(town, 1li, biy + 1li));
    let xmid = 0.5 * (street_line(town, 0li, bix) + street_line(town, 0li, bix + 1li));
    let dq0 = vec2<f32>(street_line(town, 0li, bix), ymid) - q;
    let dq1 = vec2<f32>(street_line(town, 0li, bix + 1li), ymid) - q;
    let dq2 = vec2<f32>(xmid, street_line(town, 1li, biy)) - q;
    let dq3 = vec2<f32>(xmid, street_line(town, 1li, biy + 1li)) - q;
    let x0 = seg_here(hash2(town.seed ^ 0x57lu, bix, biy), town_urban(town, p + vec3<f64>(town.ex.xyz * dq0.x + town.ey.xyz * dq0.y), gsd, slope, clear, warp2).x);
    let x1 = seg_here(hash2(town.seed ^ 0x57lu, bix + 1li, biy), town_urban(town, p + vec3<f64>(town.ex.xyz * dq1.x + town.ey.xyz * dq1.y), gsd, slope, clear, warp2).x);
    let y0 = seg_here(hash2(town.seed ^ 0x58lu, biy, bix), town_urban(town, p + vec3<f64>(town.ex.xyz * dq2.x + town.ey.xyz * dq2.y), gsd, slope, clear, warp2).x);
    let y1 = seg_here(hash2(town.seed ^ 0x58lu, biy + 1li, bix), town_urban(town, p + vec3<f64>(town.ex.xyz * dq3.x + town.ey.xyz * dq3.y), gsd, slope, clear, warp2).x);
    var here_x = x1;
    if (near_x == bix) {
        here_x = x0;
    }
    var here_y = y1;
    if (near_y == biy) {
        here_y = y0;
    }
    // a block is built on only if a street runs along one of its sides
    let access = max(max(x0, x1), max(y0, y1));
    var sw_x = town.street * 0.5;
    if (((near_x % 4li) + 4li) % 4li == 0li) {
        sw_x = town.street * 1.4 * 0.5;
    }
    var sw_y = town.street * 0.5;
    if (((near_y % 4li) + 4li) % 4li == 0li) {
        sw_y = town.street * 1.4 * 0.5;
    }
    let fws = max(fw, gsd * 0.5);
    let ex_ = min(bqx, bsx - bqx);
    let ey_ = min(bqy, bsy - bqy);
    let street = max(band_cov(ex_, sw_x, fws) * here_x, band_cov(ey_, sw_y, fws) * here_y);
    let walk_w = 0.2 * town.street + 0.5;
    let walk = max(band_cov(ex_, sw_x + walk_w, fws) * here_x, band_cov(ey_, sw_y + walk_w, fws) * here_y);
    let bh = hash2(town.seed ^ 0xB10Clu, bix, biy);
    let block_kind = u01k(bh, 1lu);
    var col = pal3(PAL_ASPHALT);
    var height = 0.0;
    var cls = LC_URBAN;
    var cov_lot = 0.0;
    var porch = 0.0;
    var roof_frac = 0.0;
    var win_col = vec3<f32>(1.0, 0.74, 0.44);
    var windows = 0.0;
    let inner = vec2<f32>(bqx - sw, bqy - sw);
    let bw = bsx - 2.0 * sw;
    let bd = bsy - 2.0 * sw;
    let central = max(1.0 - rel, 0.0);
    let buildings_resolved = band(town.lot, gsd);
    var shadow_ok = false;
    if (urban > 0.5 && block_kind < 0.07) {
        // park
        col = pal3(PAL_GRASS_WET) * (1.0 + 0.15 * detail);
        cls = LC_GRASS;
        cov_lot = 1.0;
    } else if (urban > 0.45 && block_kind < 0.12) {
        col = pal3(PAL_CONCRETE) * (0.82 + 0.1 * u01k(bh, 4lu));
        cov_lot = 1.0;
    } else if (inner.x >= 0.0 && inner.y >= 0.0 && inner.x < bw && inner.y < bd) {
        let industrial = block_kind > 0.92 && central < 0.5;
        var lot_w = town.lot * (0.7 + 0.6 * u01k(bh, 5lu));
        if (industrial) {
            lot_w = bw;
        }
        var rows = 2.0;
        if (industrial || bd < 2.6 * town.lot) {
            rows = 1.0;
        }
        let li = floor(inner.x / lot_w);
        let lj = floor(inner.y / (bd / rows));
        let lx = inner.x - li * lot_w;
        let ly = inner.y - lj * (bd / rows);
        let lh = hash2(bh, i64(li), i64(lj));
        // decided with the town density at the lot's centre
        let lot_c = vec2<f32>(li * lot_w + 0.5 * lot_w - inner.x, (lj + 0.5) * (bd / rows) - inner.y);
        let urban_lot = town_urban(town, p + vec3<f64>(town.ex.xyz * lot_c.x + town.ey.xyz * lot_c.y), gsd, slope, clear, warp2).x;
        let built = u01k(lh, 3lu) < pow(urban_lot, 0.7) * 1.05 && access > 0.5;
        let lj_even = ((i32(lj) % 2) + 2) % 2 == 0;
        if (built || urban_lot > 0.65) {
            cov_lot = 1.0;
            let yard = mixc(mixc(pal3(PAL_GRASS_WET), pal3(PAL_SOIL), 0.3 + 0.4 * u01k(lh, 10lu)), pal3(PAL_CONCRETE), 0.25 * central) * (1.0 + 0.2 * detail);
            col = yard;
            if (built) {
                var setb = 1.5 + 3.5 * u01k(lh, 1lu);
                if (industrial) {
                    setb = 6.0;
                }
                let fwid = lot_w - 2.0 * min(setb, lot_w * 0.3);
                let fdep = max(bd / rows - setb - 2.0 - 5.0 * u01k(lh, 2lu), 0.0);
                if (fwid > 3.0 && fdep > 3.0) {
                    let bcx = lx - lot_w * 0.5;
                    var bcy = ly - (bd / rows - setb - fdep * 0.5);
                    if (lj_even) {
                        bcy = ly - setb - fdep * 0.5;
                    }
                    let ex = fwid * 0.5 - abs(bcx);
                    let ey = fdep * 0.5 - abs(bcy);
                    var inside = clamp(min(ex, ey) / fw + 0.5, 0.0, 1.0);
                    if (!industrial && u01k(lh, 11lu) < 0.3) {
                        // L-shape: remove a corner quadrant
                        var qx = -bcx;
                        if (u01k(lh, 12lu) < 0.5) {
                            qx = bcx;
                        }
                        var qy = -bcy;
                        if (u01k(lh, 13lu) < 0.5) {
                            qy = bcy;
                        }
                        let cut = min(qx - fwid * 0.1, qy - fdep * 0.1);
                        inside *= 1.0 - clamp(cut / fw + 0.5, 0.0, 1.0);
                    }
                    let tall = central * central * town.height;
                    var hb = 3.5 + 4.0 * u01k(lh, 4lu) + 40.0 * tall * u01k(lh, 5lu);
                    if (industrial) {
                        hb = 7.0 + 7.0 * u01k(lh, 4lu);
                    }
                    let flat_roof = industrial || hb > 12.0 || u01k(lh, 6lu) < 0.2 + 0.3 * central;
                    let ri = u32(town.roof_style * 3.0 + u01k(lh, 7lu) * 4.0) % 7u;
                    var roof = pal3(PAL_ROOFS + ri);
                    if (industrial) {
                        roof = mixc(pal3(PAL_ROOFS + 4u), pal3(PAL_ROOFS + 2u), u01k(lh, 8lu));
                    }
                    roof *= 0.82 + 0.36 * u01k(lh, 9lu);
                    var h_here = hb;
                    if (!flat_roof) {
                        // pitched roof along the longer axis
                        var half = fwid * 0.5;
                        var dperp = ex;
                        if (fwid > fdep) {
                            half = fdep * 0.5;
                            dperp = ey;
                        }
                        h_here = hb + 0.35 * half * clamp(dperp / half, 0.0, 1.0);
                    } else {
                        roof *= 1.0 - 0.15 * band_cov(min(ex, ey), 0.6, fw);
                    }
                    if (inside > 0.0) {
                        col = mixc(col, roof, inside);
                        height = h_here * inside;
                        roof_frac = inside * buildings_resolved;
                        // lit windows along the walls
                        let rim = (1.0 - smoothstep1(0.0, 0.9, min(ex, ey))) * inside;
                        if (rim > 0.0) {
                            var along = bcx;
                            if (ex < ey) {
                                along = bcy;
                            }
                            let wsp = 2.6;
                            let wi = floor(along / wsp);
                            let wf = along / wsp - wi;
                            let lit_frac = 0.08 + 0.2 * central + 0.12 * u01k(lh, 15lu);
                            let wc = u01k(lh, 16lu);
                            if (wc < 0.6) {
                                win_col = vec3<f32>(1.0, 0.70, 0.40);
                            } else if (wc < 0.85) {
                                win_col = vec3<f32>(1.0, 0.86, 0.66);
                            } else {
                                win_col = vec3<f32>(0.82, 0.90, 1.0);
                            }
                            let lit = u01k(hash2(lh ^ 0x3D0lu, i64(wi), select(0li, 1li, ex < ey)), 1lu) < lit_frac;
                            let expl = band(wsp, gsd);
                            var pane = 0.0;
                            if (lit && wf >= 0.3 && wf < 0.65) {
                                pane = 1.0;
                            }
                            windows = rim * (pane * expl + 0.55 * lit_frac * (1.0 - expl));
                        }
                        if (inside > 0.5) {
                            cls = LC_BUILDING;
                        }
                    }
                    // porch / yard light in front of some houses
                    if (!industrial && u01k(lh, 14lu) < 0.6) {
                        var front_y = bd / rows - setb * 0.5;
                        if (lj_even) {
                            front_y = setb * 0.5;
                        }
                        let ax = lx - lot_w * 0.5;
                        let ay = ly - front_y;
                        porch = point_light(ax * ax + ay * ay, 2.0, 0.4, fw);
                    }
                }
            }
        }
        // mean appearance when lots are unresolved
        if (buildings_resolved < 1.0) {
            let mean_roof = mixc(pal3(PAL_ROOFS + u32(town.roof_style * 6.99)), pal3(PAL_CONCRETE), 0.3);
            let mean = mixc(mixc(pal3(PAL_GRASS_WET), pal3(PAL_SOIL), 0.4), mean_roof, 0.5);
            col = mixc(mean, col, buildings_resolved);
            height = lerp(3.0 * urban, height, buildings_resolved);
            cov_lot = lerp(pow(urban, 0.7), cov_lot, buildings_resolved);
        }
        shadow_ok = buildings_resolved > 0.0 && height < 0.5 && cov_lot > 0.0;
    }
    let cov = max(cov_lot, walk);
    if (cov <= 0.0) {
        return o;
    }
    col = mixc(col, pal3(PAL_CONCRETE) * 0.9, max(walk - street, 0.0) * (1.0 - cov_lot) / cov);
    col = mixc(col, pal3(PAL_ASPHALT) * (1.0 + 0.05 * detail), street / cov);
    if (street > 0.5) {
        cls = LC_ROAD;
    }
    height *= 1.0 - street;

    // ---- night lights
    let lamp_sp = 18.0 + 8.0 * town.organic;
    let mix_sodium = 0.25 + 0.4 * u01k(town.seed, 20lu);
    var dominant = 1u;
    if (mix_sodium > 0.45) {
        dominant = 0u;
    }
    let lamp_col = lamp_kind(dominant);
    let lamp_res = band(town.block, gsd);
    let fwl = max(fw, 0.5 * gsd);
    let m = exp2(ceil(log2(max(2.0 * fwl / lamp_sp, 1.0))));
    let sp_eff = lamp_sp * m;
    var emission = vec3<f32>(0.0);
    if (lamp_res > 0.0 && max(here_x, here_y) > 0.0) {
        for (var axis = 0u; axis < 2u; axis++) {
            var bi = bix;
            var bq = bqx;
            var bs = bsx;
            var along = q.y;
            var here = here_x;
            var swa = sw_x;
            if (axis == 1u) {
                bi = biy;
                bq = bqy;
                bs = bsy;
                along = q.x;
                here = here_y;
                swa = sw_y;
            }
            var off = bq;
            var li = bi;
            if (!(bq < bs - bq)) {
                off = -(bs - bq);
                li = bi + 1li;
            }
            if (here <= 0.0 || abs(off) > swa + 22.0 + 3.0 * fwl) {
                continue;
            }
            let k = round(along / sp_eff) * m;
            var side = -1.0;
            if (((i64(k) % 2li) + 2li) % 2li == 0li) {
                side = 1.0;
            }
            let lamp_off = side * swa * 0.85;
            let dp = off - lamp_off;
            let da = along - k * lamp_sp;
            let d2 = dp * dp + da * da;
            let lh = hash2(town.seed ^ 0x1A3Blu ^ (u64(axis) << 40u), li, i64(k));
            if (u01k(lh, 1lu) < 0.93) {
                let sa = 0.28 * lamp_sp;
                let pool = 0.03 * exp(-(dp * dp) / (2.0 * 3.5 * 3.5) - (da * da) / (2.0 * sa * sa));
                let core = point_light(d2, 6.0, 0.4, max(fw, 0.5 * gsd));
                // lamp type of the street
                let u = u01k(hash2(town.seed ^ 0x1A3Clu ^ (u64(axis) << 40u), li, 0li), 1lu);
                let main = ((li % 4li) + 4li) % 4li == 0li;
                var lt = 2u;
                if (main && u < 0.45) {
                    lt = 2u;
                } else if (u < mix_sodium) {
                    lt = 0u;
                } else if (u < 0.85) {
                    lt = 1u;
                }
                emission += lamp_kind(lt) * ((pool + core) * m) * (0.75 + 0.5 * u01k(lh, 2lu));
            }
        }
    }
    // the lamps light the ground, not the roofs
    let ground_lit = 1.0 - roof_frac * (1.0 - street);
    emission *= ground_lit;
    porch *= ground_lit;
    let lamp_e = 0.93 * (2.0 * PI * (6.0 * 0.16 + 0.03 * 3.5 * 0.28 * lamp_sp));
    let lamp_mean = lamp_e * 2.0 / (town.block * lamp_sp);
    emission = emission * lamp_res + lamp_col * lamp_mean * (1.0 - lamp_res) * smoothstep1(0.15, 0.4, urban);
    emission += win_col * (0.55 * windows * buildings_resolved);
    if (block_kind > 0.92 && central < 0.5) {
        emission += vec3<f32>(1.0, 0.88, 0.7) * 0.02 * band(lamp_sp, gsd);
    }
    emission += vec3<f32>(1.0, 0.72, 0.42) * porch;
    o.ok = true;
    o.col = col;
    o.h = height;
    o.cov = cov;
    o.cls = cls;
    o.em = emission / max(cov, 0.05);
    o.shadow_ok = shadow_ok;
    o.street = street;
    o.resolved = buildings_resolved;
    return o;
}

/// A town at a point with the cast shadows of its buildings (`SurfaceModel::town`); the shadow
/// in `o.street`'s place is returned in `shadow`.
struct TownOut {
    px: TownPx,
    shadow: f32,
}

fn town_eval(town: Town, p: vec3<f64>, gsd: f32, fw: f32, slope: f32, clear: f32, detail: f32, warp2: f32) -> TownOut {
    var o: TownOut;
    o.shadow = 0.0;
    var shadow = 0.0;
    // k = 0: the pixel; k = 1..4: the march toward the sun (one call site of `town_core`: the
    // driver inlines each)
    for (var k = 0; k <= 4; k++) {
        let dist_s = f32(k) * 3.5;
        var sp = p;
        var fw_k = fw;
        var slope_k = slope;
        var clear_k = clear;
        if (k > 0) {
            sp = p + vec3<f64>(town.sun.xyz * dist_s);
            fw_k = 0.01;
            slope_k = 0.0;
            clear_k = 1.0;
        }
        let b = town_core(town, sp, gsd, fw_k, slope_k, clear_k, detail, warp2);
        if (k == 0) {
            o.px = b;
            if (!b.ok) {
                return o;
            }
            if (!((cfg.flags & CF_SHADOWS) != 0u && b.shadow_ok)) {
                break;
            }
            continue;
        }
        let need = dist_s * cfg.sun_tan;
        var hh = 0.0;
        if (b.ok) {
            hh = b.h * b.cov;
        }
        if (hh > need) {
            shadow = 0.75 * o.px.resolved;
            break;
        }
    }
    o.shadow = shadow * (1.0 - o.px.street * 0.5);
    return o;
}

/// The most built-up town at `p` among the candidates of its lattice cell (`select_town`):
/// index into `towns` (-1: none) and how built-up; -2: the cell's candidates are unknown.
fn select_town(p: vec3<f64>, gsd: f32, slope: f32, clear: f32, warp2: f32) -> vec2<f32> {
    let cell = town_cell_find(town_cell_of(p));
    if (cell.y == 0xffffffffu) {
        return vec2<f32>(-2.0, 0.0);
    }
    var best = -1;
    var bu = 0.0;
    for (var k = 0u; k < cell.y; k++) {
        let ti = town_list[cell.x + k];
        let u = town_urban(towns[ti], p, gsd, slope, clear, warp2).x;
        if (u > bu) {
            bu = u;
            best = i32(ti);
        }
    }
    return vec2<f32>(f32(best), bu);
}

// ---------------------------------------------------------------- the surface at a sub-sample

/// Flags of a sample that pass B could not evaluate exactly (the host is to provide data).
var<private> surface_missing: u32;
const MISS_REGION: u32 = 1u;
const MISS_TOWN: u32 = 2u;

/// Evaluate the surface at one sub-sample (`SurfaceModel::eval`).
fn surface_eval(c: Ctx, t: Terrain, l: Local, pf: ptr<function, PixFields>) -> Surface {
    let p = c.p;
    let gsd = c.gsd;
    let fw = l.fw;
    var o: Surface;
    o.emission = vec3<f32>(0.0);

    // ---- standing water
    if (l.water > l.ground && l.water_kind != W_NONE) {
        let depth = l.water - l.ground;
        var col = vec3<f32>(0.0);
        if (l.water_kind == W_OCEAN) {
            col = mixc(pal3(PAL_OCEAN_SHALLOW), pal3(PAL_OCEAN_DEEP), smoothstep1(0.0, 28.0, depth));
        } else {
            col = mixc(mixc(pal3(PAL_RIVER), pal3(PAL_OCEAN_SHALLOW), 0.25), pal3(PAL_LAKE_DEEP), smoothstep1(0.0, 5.0, depth));
        }
        col *= 1.0 + 0.06 * pf_lazy(pf, PF_WATER);
        if (l.water_kind == W_OCEAN && depth < 3.0) {
            // the sandy bottom shows through clear shallow water
            let sh = 1.0 - smoothstep1(0.0, 3.0, depth);
            col = mixc(col, mixc(pal3(PAL_OCEAN_SHALLOW), pal3(PAL_BEACH), 0.55) * 1.05, 0.75 * sh * sh);
            // surf: thin broken lines of foam along the shore
            let b3 = band(3.0, gsd);
            if (b3 > 0.0) {
                let wave = depth + 0.18 * perlin3(0x5F1lu, p * (1.0lf / 40.0lf));
                let broken = smoothstep1(-0.25, 0.35, perlin3(0x5F2lu, p * (1.0lf / 22.0lf)) + 0.5 * perlin3(0x5F3lu, p * (1.0lf / 7.0lf)));
                let a = (wave - 0.06) / 0.05;
                let b = (wave - 0.45) / 0.05;
                let cc = (wave - 1.0) / 0.06;
                let foam = (0.85 * exp(-a * a) + 0.6 * broken * exp(-b * b) + 0.4 * broken * exp(-cc * cc)) * (0.75 + 0.25 * perlin3(7lu, p * (1.0lf / 3.0lf))) * b3;
                col = mixc(col, srgb(225.0, 232.0, 230.0), min(foam, 1.0));
            }
        }
        o.albedo = col;
        o.height = l.water;
        o.cls = select(LC_LAKE, LC_OCEAN, l.water_kind == W_OCEAN);
        o.lit = 1.0;
        return o;
    }

    let slope = l.slope;
    let st = t.style;
    // ---- natural ground
    let detail = (*pf).f[PF_DETAIL];
    let patchv = (*pf).f[PF_PATCH];
    let wet = t.moist;
    let temp = t.temp;
    let soil_i = st.x * 3.0;
    let i0 = min(u32(floor(soil_i)), 2u);
    var soil = mixc(pal3(PAL_SOIL + i0), pal3(PAL_SOIL + i0 + 1u), soil_i - f32(i0));
    // red laterite soils in hot, wet climates
    soil = mixc(soil, srgb(146.0, 82.0, 54.0), 0.75 * smoothstep1(19.0, 25.0, temp) * smoothstep1(0.45, 0.7, wet));
    let grass_green = mixc(pal3(PAL_GRASS_DRY), pal3(PAL_GRASS_WET), smoothstep1(0.25, 0.75, wet + 0.15 * patchv));
    var grass = mixc(pal3(PAL_GRASS_COLD), grass_green, smoothstep1(-2.0, 8.0, temp));
    grass = grass * vec3<f32>(1.0 + 0.10 * (st.y - 0.5), 1.0 + 0.06 * (st.w - 0.5), 1.0 - 0.08 * (st.y - 0.5));
    let land_n = (*pf).f[PF_LAND];
    let cover = smoothstep1(0.08, 0.45, wet + 0.25 * patchv + 0.2 * land_n) * smoothstep1(-9.0, -1.0, temp);
    var col = mixc(soil, grass, cover) * (1.0 + 0.16 * land_n);
    var cls = select(LC_BARE, LC_GRASS, cover > 0.5);
    if (temp < 0.0 && cover > 0.3) {
        col = mixc(col, pal3(PAL_TUNDRA), smoothstep1(0.0, -6.0, temp));
        cls = LC_TUNDRA;
    }
    if (t.floodplain > 0.3 && wet > 0.55) {
        let m = smoothstep1(0.3, 0.9, t.floodplain) * smoothstep1(0.55, 0.8, wet) * smoothstep1(-0.1, 0.3, patchv);
        col = mixc(col, pal3(PAL_MARSH), m);
        if (m > 0.5) {
            cls = LC_WETLAND;
        }
    }
    col *= 1.0 + 0.22 * detail;
    // meadow texture
    {
        let b30 = band(30.0, gsd);
        if (b30 > 0.0 && cover > 0.0) {
            let dry_p = smoothstep1(0.05, 0.55, perlin3(0x3EADlu, p * (1.0lf / 60.0lf)) + 0.5 * perlin3(0x3EAElu, p * (1.0lf / 22.0lf))) * b30 * cover;
            col = mixc(col, col * vec3<f32>(1.16, 1.06, 0.80), 0.45 * dry_p);
        }
        let b12 = band(12.0, gsd);
        if (b12 > 0.0 && cover > 0.0) {
            var m = 0.10 * perlin3(0x3EB1lu, p * (1.0lf / 12.0lf)) * b12;
            let b4 = band(4.0, gsd);
            if (b4 > 0.0) {
                m += 0.08 * perlin3(0x3EB2lu, p * (1.0lf / 4.0lf)) * b4;
                let b13 = band(1.3, gsd);
                if (b13 > 0.0) {
                    m += 0.06 * perlin3(0x3EB3lu, p * (1.0lf / 1.3lf)) * b13;
                }
            }
            col *= 1.0 + m * cover;
        }
    }
    // drainage lines
    if (t.gully != 0.0) {
        let ch = smoothstep1(0.1, 0.8, -t.gully);
        col = mixc(col, mixc(col * 0.8, pal3(PAL_GRASS_WET) * 0.85, 0.5 * smoothstep1(-6.0, 4.0, temp)), 0.6 * ch);
        col *= 1.0 + 0.06 * smoothstep1(0.2, 1.0, t.gully);
    }

    // ---- rock (slope + expected)
    let resolve = 1.0 - smoothstep1(8.0, 80.0, gsd);
    let exp_slope = 0.12 + 0.75 * t.rock_expect;
    let slope_eff = lerp(exp_slope, max(slope, exp_slope * 0.6), resolve);
    let rock_n = 0.7 * patchv + 0.3 * land_n;
    let rock = smoothstep1(0.55, 0.85, slope_eff + 0.25 * detail + 0.25 * rock_n + 0.2 * (t.rock_expect - 0.4) + 0.15 * t.gully)
        * (1.0 - 0.5 * smoothstep1(0.3, 0.8, cover) * (1.0 - t.mountain));
    if (rock > 0.0) {
        let ri = st.z * 2.0;
        let j = min(u32(floor(ri)), 1u);
        var rc = mixc(pal3(PAL_ROCK + j), pal3(PAL_ROCK + j + 1u), ri - f32(j));
        let strata_h = 6.0 + 10.0 * st.w;
        let strata = sin(l.ground / strata_h + 3.0 * pf_lazy(pf, PF_STRATA));
        let strata_w = TAU * strata_h / max(slope, 0.05);
        rc *= 1.0 + 0.06 * strata * band(strata_w, 1.5 * gsd) + 0.25 * detail + 0.12 * pf_lazy(pf, PF_STRATA2);
        col = mixc(col, rc, rock);
        if (rock > 0.5) {
            cls = LC_ROCK;
        }
    }

    // ---- sand seas, beaches
    if (t.sand > 0.0) {
        let si = st.y * 2.0;
        let j = min(u32(floor(si)), 1u);
        let sc = mixc(pal3(PAL_SAND + j), pal3(PAL_SAND + j + 1u), si - f32(j)) * (1.0 + 0.06 * detail);
        let s = smoothstep1(0.2, 0.6, t.sand + 0.2 * patchv);
        col = mixc(col, sc, s);
        if (s > 0.5) {
            cls = LC_SAND;
        }
    }
    let coastal = 1.0 - smoothstep1(0.3, 0.7, t.floodplain);
    var beach = 0.0;
    var shore_keep = 1.0;
    if (coastal > 0.0) {
        shore_keep = 1.0 - coastal * (1.0 - smoothstep1(4.0, 8.0, l.ground + 2.0 * patchv));
    }
    if (l.ground < 6.0 && coastal > 0.0 && slope < 0.3) {
        let b = (1.0 - smoothstep1(2.6, 4.2, l.ground + 1.0 * detail + 0.8 * patchv)) * coastal * (1.0 - smoothstep1(0.15, 0.3, slope));
        beach = b;
        var bm = 1.0;
        let b9 = band(9.0, gsd);
        if (b9 > 0.0) {
            bm += 0.05 * perlin3(0xBE1lu, p * (1.0lf / 9.0lf)) * b9;
            let b25 = band(2.5, gsd);
            if (b25 > 0.0) {
                bm += 0.04 * perlin3(0xBE2lu, p * (1.0lf / 2.5lf)) * b25;
            }
        }
        let bc0 = pal3(PAL_BEACH) * bm;
        let bc = mixc(bc0, mixc(pal3(PAL_BEACH), pal3(PAL_WET_SAND), 0.7), 1.0 - smoothstep1(0.05, 0.3, l.ground + 0.08 * perlin3(0xBE3lu, p * (1.0lf / 15.0lf))));
        col = mixc(col, bc, b);
        if (b > 0.5) {
            cls = LC_BEACH;
        }
    }

    // ---- snow
    let snow_n = (*pf).f[PF_SNOW];
    let snow_base = temp + 1.0 * snow_n - 1.6 * smoothstep1(0.1, 0.8, -t.gully) + 0.6 * smoothstep1(0.2, 0.8, t.gully);
    var snow = 0.0;
    // (the noise terms add at most ~1.3 · 0.75)
    if (snow_base - 1.5 < -2.6) {
        var snow_t = snow_base;
        let b60 = band(60.0, gsd);
        if (b60 > 0.0) {
            snow_t += 0.5 * perlin3(0x5E0lu, p * (1.0lf / 60.0lf)) * b60;
            let b18 = band(18.0, gsd);
            if (b18 > 0.0) {
                snow_t += 0.25 * perlin3(0x5E1lu, p * (1.0lf / 18.0lf)) * b18;
            }
        }
        snow = smoothstep1(-2.6, -2.8, snow_t) * (1.0 - 0.75 * smoothstep1(0.9, 1.6, slope));
    }
    if (snow > 0.0) {
        col = mixc(col, pal3(PAL_SNOW) * (1.0 + 0.03 * detail), snow);
        if (snow > 0.5) {
            cls = LC_SNOW;
        }
    }

    // micro-relief of the ground
    var micro_relief = 0.0;
    let b9m = band(9.0, gsd);
    if (b9m > 0.0) {
        micro_relief = 0.30 * perlin3(0x9A01lu, p * (1.0lf / 9.0lf)) * b9m;
        let b32 = band(3.2, gsd);
        if (b32 > 0.0) {
            micro_relief += 0.14 * perlin3(0x9A02lu, p * (1.0lf / 3.2lf)) * b32;
            let b11 = band(1.1, gsd);
            if (b11 > 0.0) {
                micro_relief += 0.06 * perlin3(0x9A03lu, p * (1.0lf / 1.1lf)) * b11;
            }
        }
        micro_relief *= (1.0 - 0.5 * rock) * (1.0 - 0.7 * beach);
    }
    var height = l.ground + micro_relief;
    var lit = 1.0;
    var emission = vec3<f32>(0.0);
    let natural_ok = (1.0 - rock) * (1.0 - snow) * (1.0 - t.sand) * (1.0 - beach);

    // local planar frame of the land-use region
    var r: Region;
    var have_region = false;
    var q_loc = vec2<f32>(0.0);
    var q_rot = vec2<f32>(0.0);
    if (t.region_id != 0lu) {
        let ri = region_find(t.region_id);
        if (ri >= 0) {
            r = regions[ri];
            have_region = true;
            let d = vec3<f32>(p - r.center.xyz);
            q_loc = vec2<f32>(dot(d, r.east.xyz), dot(d, r.north.xyz));
            q_rot = vec2<f32>(dot(d, r.ex.xyz), dot(d, r.ey.xyz));
        } else {
            surface_missing |= MISS_REGION;
        }
    }

    // riparian belt along rivers
    var riparian = 0.0;
    if (l.river_hw > 0.0 && t.river_wet > 0.3) {
        let ad = abs(l.river_d);
        let belt = 4.0 + 0.6 * min(l.river_hw, 60.0);
        riparian = (1.0 - smoothstep1(l.river_hw + 0.3 * belt, l.river_hw + belt, ad)) * smoothstep1(0.35, 0.6, t.moist)
            * smoothstep1(0.3, 0.7, t.river_wet) * natural_ok * (0.55 + 0.45 * smoothstep1(-0.3, 0.3, patchv));
    }

    // woodlots
    let fpu = 0.5 + 0.5 * (*pf).f[PF_FOREST];
    let wl_cover = smoothstep1(0.12, 0.45, max(t.moist, 0.0)) * smoothstep1(-6.0, 2.0, t.temp) * cfg.tree_density;
    let woodlot = smoothstep1(-0.02, 0.02, 0.34 * wl_cover - fpu) * natural_ok;

    // ---- fields
    let flat_ok = 1.0 - smoothstep1(0.22, 0.32, slope);
    var field_cov = 0.0;
    if (have_region && t.agri > 0.02 && natural_ok * flat_ok > 0.3 && cfg.agriculture > 0.0) {
        let fr = field(r, t, q_rot, p, gsd, fw, pf);
        if (fr.ok) {
            let keep = natural_ok * flat_ok * (1.0 - riparian) * (1.0 - woodlot) * shore_keep;
            let a = fr.cov * smoothstep1(0.45, 0.55, keep);
            col = mixc(col, fr.col, a);
            height += fr.h * a - 0.6 * micro_relief * a;
            field_cov = a;
            if (a > 0.5) {
                if (fr.kind == 1u) {
                    cls = LC_FOREST;
                } else if (fr.kind == 2u) {
                    cls = LC_ROAD;
                } else {
                    cls = LC_CROP;
                }
            }
        }
    }

    // ---- towns (evaluated here: their lots and streets mask the trees)
    var river_clear = 1.0;
    if (l.river_hw > 0.0) {
        let bank = 2.0 + 0.1 * l.river_hw;
        river_clear = 1.0 - (1.0 - smoothstep1(l.river_hw + bank, l.river_hw + 2.0 * bank + 2.0, abs(l.river_d))) * smoothstep1(0.3, 0.6, t.river_wet);
    }
    let town_slope = 0.12 + 0.75 * t.rock_expect;
    var town_i = -1;
    var town_urban_v = 0.0;
    if (t.town != 0u && cfg.towns > 0.0) {
        let sel = select_town(p, gsd, town_slope, river_clear, (*pf).f[PF_WARP2]);
        if (sel.x == -2.0) {
            surface_missing |= MISS_TOWN;
        } else {
            town_i = i32(sel.x);
            town_urban_v = sel.y;
        }
    }
    var town_px: TownOut;
    town_px.px.ok = false;
    var town_cov = 0.0;
    if (town_i >= 0) {
        town_px = town_eval(towns[town_i], p, gsd, fw, town_slope, river_clear, detail, (*pf).f[PF_WARP2]);
        if (town_px.px.ok) {
            town_cov = town_px.px.cov;
        }
    }
    let not_urban = (1.0 - smoothstep1(0.0, 0.08, town_urban_v)) * (1.0 - town_cov);

    // ---- trees
    if (cfg.tree_density > 0.0 && natural_ok * not_urban > 0.05) {
        let base_cover = smoothstep1(0.3, 0.68, wet) * smoothstep1(-6.0, 2.0, temp) * cfg.tree_density;
        let edge = 0.03;
        let forest = smoothstep1(-edge, edge, base_cover - fpu);
        let savanna = smoothstep1(0.18, 0.35, wet) * (1.0 - smoothstep1(0.55, 0.7, wet)) * smoothstep1(12.0, 20.0, temp) * 0.12;
        let groves = 0.04 * smoothstep1(0.15, 0.3, wet);
        let clear = 1.0 - 0.85 * smoothstep1(0.05, 0.4, t.agri) * (1.0 - smoothstep1(0.45, 0.7, slope)) * (1.0 - woodlot);
        var dens = (forest * 0.9 * clear + savanna + groves) * natural_ok * (1.0 - field_cov) * cfg.tree_density;
        dens = max(dens, 0.8 * riparian * cfg.tree_density);
        let gully_scrub = 0.55 * smoothstep1(0.2, 0.9, -t.gully) * smoothstep1(0.2, 0.5, wet) * natural_ok * (1.0 - field_cov);
        dens *= 1.0 - smoothstep1(0.9, 1.4, slope);
        dens *= 1.0 - smoothstep1(0.0, 0.6, t.mountain * smoothstep1(-2.0, -6.0, temp));
        dens *= 1.0 - smoothstep1(-1.2, -2.4, temp + 1.0 * snow_n);
        dens *= not_urban;
        let shrub_clim = smoothstep1(0.15, 0.3, wet) * (1.0 - smoothstep1(0.6, 0.8, wet)) * smoothstep1(2.0, 10.0, temp);
        let shrub_patch = smoothstep1(-0.2, 0.5, patchv + 0.4 * land_n);
        let shrub = clamp((0.45 * shrub_clim * shrub_patch * (1.0 - forest) * max(natural_ok, 0.4 * rock) * (1.0 - field_cov) + gully_scrub) * cfg.tree_density * not_urban, 0.0, 0.7);
        // forest stands of their own age, tone and species mix
        var stand = (*pf).stand;
        if ((*pf).has_stand == 0u) {
            stand = stand_id(p, vec3<f32>((*pf).f[13], (*pf).f[14], (*pf).f[15]));
        }
        let age = u01k(stand, 1lu);
        let tone_u = u01k(stand, 2lu);
        let stand_tone = mixc(vec3<f32>(0.86, 0.93, 0.92), vec3<f32>(1.12, 1.08, 0.88), tone_u) * (0.92 + 0.12 * age);
        let gap_w = smoothstep1(0.3, 0.7, dens);
        if (gap_w > 0.0) {
            let gap = smoothstep1(0.3, 0.6, perlin3(0x6A9lu, p * (1.0lf / 30.0lf)) + 0.5 * perlin3(0x6AAlu, p * (1.0lf / 11.0lf))) * (0.2 + 0.8 * u01k(stand, 4lu));
            dens *= 1.0 - 0.9 * gap * gap_w;
        }
        if (dens > 0.0 || shrub > 0.01) {
            let stand_d = smoothstep1(0.3, 0.75, dens);
            var conifer = (1.0 - smoothstep1(4.0, 13.0, temp)) * max(stand_d, 1.0 - smoothstep1(-5.0, 1.0, temp));
            conifer = clamp(conifer + 0.9 * (u01k(stand, 3lu) - 0.5) * (1.0 - abs(2.0 * conifer - 1.0)), 0.0, 1.0);
            let scale = 0.7 + 0.55 * age;
            let tropic = smoothstep1(19.0, 25.0, temp) * smoothstep1(0.55, 0.75, wet);
            let dry = 1.0 - smoothstep1(0.3, 0.5, wet);
            let tall = 0.5 + 0.5 * st.w;
            var layers: array<TreeLayer, 4>;
            layers[0] = TreeLayer(pal3(PAL_CROWN_CONIFER), 5.5, stand_tone, dens * conifer, 0x7EE1lu, dens, 14.0 + 10.0 * tall, 1.0, scale);
            layers[1] = TreeLayer(mixc(pal3(PAL_CROWN_DECID), pal3(PAL_CROWN_DRY), dry), 8.5, stand_tone, dens * (1.0 - conifer) * (1.0 - tropic), 0x7EE2lu, dens, (10.0 + 10.0 * tall) * (0.65 + 0.35 * stand_d), 0.0, scale);
            layers[2] = TreeLayer(pal3(PAL_CROWN_TROPIC), 13.0, stand_tone, dens * tropic, 0x7EE3lu, dens, 22.0 + 14.0 * tall, 0.0, scale);
            layers[3] = TreeLayer(pal3(PAL_SHRUB), 3.2, vec3<f32>(1.0), shrub, 0x7EE4lu, 0.0, 1.6, 0.0, 1.0);
            // forest floor: shaded litter and understory between the crowns
            let floor_ = smoothstep1(0.25, 0.8, dens);
            col = mixc(col, mixc(pal3(PAL_CROWN_CONIFER), pal3(PAL_SOIL), 0.45) * 0.7, floor_ * 0.85);
            let tr = trees(layers, 15u, q_loc, gsd, fw, p);
            if (tr.cov > 0.0) {
                let stv = pf_lazy(pf, PF_STAND);
                let tc = tr.col * vec3<f32>(1.0 + 0.10 * stv, 1.0 + 0.14 * stv, 1.0 + 0.05 * stv);
                col = mixc(col, tc, tr.cov);
                if ((cfg.flags & CF_TREES_DSM) != 0u) {
                    height = max(height, l.ground + tr.h);
                }
                if (tr.cov > 0.5) {
                    cls = LC_FOREST;
                }
            }
            // cast shadows from neighbouring trees
            if ((cfg.flags & CF_SHADOWS) != 0u && tr.cov < 0.99) {
                var shadow = 0.0;
                for (var li = 0u; li < 4u; li++) {
                    if (layers[li].density <= 0.0 || layers[li].cell < 2.0 * gsd) {
                        continue;
                    }
                    let off = vec2<f32>(cfg.sun_hx, cfg.sun_hy) * (0.6 * layers[li].height / cfg.sun_tan);
                    let sc = trees(layers, 1u << li, q_loc + off, gsd, fw, p).cov;
                    shadow = max(shadow, sc);
                }
                lit = min(lit, 1.0 - 0.9 * shadow * (1.0 - tr.cov));
            }
            // dark understory under sparse prefiltered forest
            if (tr.cov < 0.01 && dens > 0.0) {
                col *= 1.0 - 0.15 * dens;
            }
        }
    }

    // ---- roads
    var road_major_cov = 0.0;
    if (cfg.roads > 0.0) {
        let steep = smoothstep1(-0.02, 0.02, 0.5 - slope);
        let habit = smoothstep1(-0.005, 0.005, t.habit - 0.03) * steep * (1.0 - snow) * (1.0 - t.sand * 0.7);
        var road_cov = 0.0;
        var road_col = pal3(PAL_ASPHALT);
        if (habit > 0.0) {
            let w_major = 12.0;
            let c1 = band_cov(l.road_major, w_major * 0.5, max(fw, gsd * 0.5));
            if (c1 > 0.0) {
                road_cov = c1 * habit;
                road_major_cov = road_cov;
                let sh = band_cov(l.road_major, w_major * 0.5 + 1.5, fw) - band_cov(l.road_major, w_major * 0.5, fw);
                road_col = mixc(pal3(PAL_ASPHALT), pal3(PAL_CONCRETE), max(sh, 0.0) * 0.6);
            }
            let w_minor = 6.0;
            let c2 = band_cov(l.road_minor, w_minor * 0.5, max(fw, gsd * 0.5)) * habit;
            if (c2 > road_cov) {
                road_cov = c2;
                road_col = mixc(pal3(PAL_ASPHALT), pal3(PAL_GRAVEL), smoothstep1(0.4, 0.7, st.x));
            }
        }
        // farm tracks along land-use region borders
        if (have_region && r.agri > 0.1 && t.agri > 0.05) {
            let pair = t.region_id ^ t.region_id2;
            if (u01k(pair, 3lu) < 0.7) {
                let c3 = band_cov(t.region_edge, 3.0, max(fw, gsd * 0.5)) * steep;
                if (c3 > road_cov) {
                    road_cov = c3;
                    road_col = select(pal3(PAL_ASPHALT), pal3(PAL_GRAVEL), u01k(pair, 4lu) < 0.5);
                }
            }
        }
        if (road_cov > 0.0) {
            let rc = road_col * (1.0 + 0.05 * detail);
            col = mixc(col, rc, road_cov);
            height = lerp(height, l.ground, road_cov);
            lit = lerp(lit, 1.0, road_cov * 0.5);
            if (road_cov > 0.5) {
                cls = LC_ROAD;
            }
        }
    }

    // ---- farmsteads
    if (have_region && t.agri > 0.06 && flat_ok > 0.3 && natural_ok > 0.3 && cfg.towns > 0.0) {
        let fs = farmstead(r, t, q_rot, gsd, fw);
        if (fs.ok) {
            col = mixc(col, fs.col, fs.cov);
            if ((cfg.flags & CF_BUILDINGS_DSM) != 0u) {
                height = lerp(height, l.ground + fs.h, fs.cov);
            }
            emission += fs.em;
            if (fs.cov > 0.5) {
                cls = fs.cls;
            }
        }
    }

    // ---- lit main roads near towns
    if (road_major_cov > 0.0 && town_i >= 0) {
        let town = towns[town_i];
        let dist = dist64(p, town.center.xyz);
        let near = 1.0 - smoothstep1(1.2 * town.radius, 2.2 * town.radius, dist);
        let sp = 38.0;
        let res = band(sp, gsd);
        if (near > 0.0) {
            let kq = round(q_loc / sp);
            let dq = q_loc - kq * sp;
            let d2 = dot(dq, dq);
            let lh = hash2(town.seed ^ 0x40ADlu, i64(kq.x), i64(kq.y));
            var lamp_col = vec3<f32>(0.86, 0.92, 1.0);
            if (u01k(lh, 1lu) < 0.5) {
                lamp_col = vec3<f32>(1.0, 0.48, 0.12);
            }
            let pool = (0.04 * exp(-d2 / (2.0 * 6.0 * 6.0)) + point_light(d2, 6.0, 0.4, fw)) * res + 0.04 * (1.0 - res);
            emission += lamp_col * pool * near * road_major_cov;
        }
    }

    // ---- towns: embankment lamps, then the town itself
    if (town_urban_v > 0.25 && l.river_hw > 0.0 && t.river_wet > 0.5) {
        let bank = 2.0 + 0.1 * l.river_hw;
        let dl = abs(l.river_d) - (l.river_hw + 0.5 * bank);
        let b4 = band(4.0, gsd);
        var dots = 0.3 * (1.0 - b4);
        if (b4 > 0.0) {
            dots += smoothstep1(0.35, 0.6, perlin3(0xE3Blu, p * (1.0lf / 4.0lf))) * b4;
        }
        emission += vec3<f32>(1.0, 0.80, 0.55) * (4.0 * exp(-(dl * dl) / (2.0 * 0.5 * 0.5)) * dots * smoothstep1(0.25, 0.45, town_urban_v));
    }
    if (town_i >= 0 && town_px.px.ok) {
        let tp = town_px.px;
        emission += tp.em * tp.cov;
        col = mixc(col, tp.col, tp.cov);
        if ((cfg.flags & CF_BUILDINGS_DSM) != 0u) {
            height = lerp(height, l.ground + tp.h, tp.cov);
        }
        lit = min(lit, 1.0 - town_px.shadow);
        if (tp.cov > 0.5) {
            cls = tp.cls;
        }
    }

    // ---- rivers (on top)
    if (l.river_hw > 0.0) {
        let fwr = max(fw, gsd * 0.35);
        let cov = band_cov(l.river_d, l.river_hw, fwr);
        if (cov > 0.0) {
            let wet_r = t.river_wet;
            var wcol = mixc(pal3(PAL_RIVER), pal3(PAL_LAKE_DEEP), smoothstep1(30.0, 200.0, l.river_hw * 2.0));
            // glacial flour in mountain rivers
            wcol = mixc(wcol, srgb(96.0, 138.0, 140.0), 0.7 * t.mountain * smoothstep1(8.0, 0.0, temp));
            // frozen and snowed on in the cold
            let ice = smoothstep1(-1.5, -4.0, temp + 1.5 * snow_n);
            wcol = mixc(wcol, mixc(pal3(PAL_SNOW) * 0.9, srgb(170.0, 190.0, 200.0), 0.35 * (0.5 + 0.5 * detail)), ice);
            // dry beds: a subtle pale line
            let dry_col = mixc(col, mixc(pal3(PAL_GRAVEL), pal3(PAL_SAND + 2u), 0.5) * (1.0 + 0.1 * detail), 0.55);
            let rc = mixc(dry_col, wcol, wet_r);
            col = mixc(col, rc, cov);
            height = lerp(height, l.river_level, cov);
            lit = lerp(lit, 1.0, cov);
            if (cov > 0.5 && ice > 0.5 && wet_r > 0.5) {
                cls = LC_SNOW;
            } else if (cov > 0.5) {
                if (wet_r > 0.5) {
                    o.albedo = col;
                    o.height = height;
                    o.cls = LC_RIVER;
                    o.lit = lit;
                    o.emission = emission;
                    return o;
                }
                cls = LC_SAND;
            }
        }
    }

    o.albedo = max(col, vec3<f32>(0.0));
    o.height = height;
    o.cls = cls;
    o.lit = lit;
    o.emission = emission;
    return o;
}
