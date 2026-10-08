// Tile kernels, pass A (`tile.rs`): coarse grid nodes, the relief per pixel centre (with the
// per-bin maximum height), the drainage pieces per bin, and the rest of pass A.

/// Latitude row of a tile (Mercator: one latitude per pixel row): N·cos(lat), the ECEF z of the
/// surface point, trig, latitude and the pixel sizes east-west / north-south.
struct Row {
    ncl: f64,
    z: f64,
    sl: f32,
    cl: f32,
    lat: f32,
    gsd: f32,
    gsd_ns: f32,
    _p: f32,
}

/// Longitude column: cos / sin of the longitude.
struct Col {
    co: f64,
    so: f64,
}

/// Per tile of a batch.
struct TileInfo {
    z: u32,
    /// TF_* flags
    flags: u32,
    /// coarse grid: nodes per side, the `Pre` fields its nodes hold (P_* flags)
    ng: u32,
    pre_flags: u32,
    pf_cut: f32,
    relief_cut_r: f32,
    relief_cut_h: f32,
    /// grid coordinate (node units) of pass-A pixel (0, 0): (2 + 0.5 - 2·16 ...) see host
    gu0: f32,
    /// first row / column of the tile's tables: pass-A centres (n + 4), grid nodes (ng),
    /// pass-B sub-samples ((n + 2)·ss)
    row_a: u32,
    col_a: u32,
    row_n: u32,
    col_n: u32,
    row_b: u32,
    col_b: u32,
    /// first grid node, first pass-A pixel, first bin
    node0: u32,
    pix0: u32,
    bin0: u32,
    /// the tile's channel pieces `segs[seg0 .. seg0 + nseg]` and sink lakes
    seg0: u32,
    nseg: u32,
    sink0: u32,
    nsink: u32,
    ss: u32,
}

const TF_GRID: u32 = 1u;
const TF_WARP_GRID: u32 = 2u;
const TF_GULLY_GRID: u32 = 4u;
const TF_ROADS_GRID: u32 = 8u;

/// pass-A grid: tile pixels plus a 2-pixel apron
const NA2: u32 = 260u;
/// bins of 16 x 16 pass-A pixels
const NBIN: u32 = 17u;
/// floats per grid node: Macro (19), the interpolated `Pre` inputs (34), pixel fields (16)
const NODE_F: u32 = 72u;
const NODE_MAC: u32 = 0u;
const NODE_PRE: u32 = 19u;
const NODE_PF: u32 = 53u;
/// floats per pass-A pixel: Macro (19), Relief (11)
const PIX_F: u32 = 32u;
/// the `Pre` fields a node holds (P_* flags, as f32 bits)
const NODE_FLAGS: u32 = 70u;

@group(2) @binding(0) var<storage, read> tiles: array<TileInfo>;
@group(2) @binding(1) var<storage, read> rows: array<Row>;
@group(2) @binding(2) var<storage, read> cols: array<Col>;
@group(2) @binding(3) var<storage, read_write> node_f: array<f32>;
/// per node: the lake, region and town sites (2 each), the forest stand
@group(2) @binding(4) var<storage, read_write> node_ids: array<u64>;
@group(2) @binding(5) var<storage, read_write> node_pts: array<vec4<f64>>;
@group(2) @binding(6) var<storage, read_write> pix_a: array<f32>;
@group(2) @binding(7) var<storage, read_write> terr: array<Terrain>;
/// per bin: first entry in `bin_list`, count, maximum relief height (f32 bits)
@group(2) @binding(8) var<storage, read_write> bins: array<vec4<u32>>;
@group(2) @binding(9) var<storage, read_write> bin_list: array<u32>;
/// [0]: entries used in `bin_list`, [1]: overflow. (Pass A2 reads the bin lists as `seg_list`.)
@group(2) @binding(10) var<storage, read_write> counters: array<atomic<u32>>;

fn row_col_ctx(r: Row, c: Col, gsd: f32) -> Ctx {
    let p = vec3<f64>(r.ncl * c.co, r.ncl * c.so, r.z);
    return ctx_make(p, r.sl, r.cl, f32(c.so), f32(c.co), r.lat, gsd);
}

fn macro_put(base: u32, m: Macro) {
    node_f[base + 0u] = m.cont;
    node_f[base + 1u] = m.plateau;
    node_f[base + 2u] = m.belt;
    node_f[base + 3u] = m.belt2;
    node_f[base + 4u] = m.belt_var;
    node_f[base + 5u] = m.hill_amp;
    node_f[base + 6u] = m.rough;
    node_f[base + 7u] = m.temp;
    node_f[base + 8u] = m.moist;
    node_f[base + 9u] = m.mesa;
    node_f[base + 10u] = m.sand;
    node_f[base + 11u] = m.agri;
    node_f[base + 12u] = m.style.x;
    node_f[base + 13u] = m.style.y;
    node_f[base + 14u] = m.style.z;
    node_f[base + 15u] = m.style.w;
    node_f[base + 16u] = m.river_width;
    node_f[base + 17u] = m.mtn_warp.x;
    node_f[base + 18u] = m.mtn_warp.y;
}

fn macro_from(a: array<f32, 19>) -> Macro {
    var m: Macro;
    m.cont = a[0];
    m.plateau = a[1];
    m.belt = a[2];
    m.belt2 = a[3];
    m.belt_var = a[4];
    m.hill_amp = a[5];
    m.rough = a[6];
    m.temp = a[7];
    m.moist = a[8];
    m.mesa = a[9];
    m.sand = a[10];
    m.agri = a[11];
    m.style = vec4<f32>(a[12], a[13], a[14], a[15]);
    m.river_width = a[16];
    m.mtn_warp = vec2<f32>(a[17], a[18]);
    return m;
}

/// `pack_pre` of `tile.rs` (with the mountain warp in front).
fn pre_put(base: u32, mtn_warp: vec2<f32>, p: Pre) {
    node_f[base + 0u] = mtn_warp.x;
    node_f[base + 1u] = mtn_warp.y;
    node_f[base + 2u] = p.gully.x;
    node_f[base + 3u] = p.gully.y;
    node_f[base + 4u] = p.road_major.x;
    node_f[base + 5u] = p.road_major.y;
    node_f[base + 6u] = p.road_major.z;
    node_f[base + 7u] = p.road_minor.x;
    node_f[base + 8u] = p.road_minor.y;
    node_f[base + 9u] = p.road_minor.z;
    node_f[base + 10u] = p.relief.x;
    node_f[base + 11u] = p.relief.y;
    node_f[base + 12u] = p.relief.z;
    node_f[base + 13u] = p.relief.w;
    node_f[base + 14u] = p.relief4;
    node_f[base + 15u] = p.river_warp01.x;
    node_f[base + 16u] = p.river_warp01.y;
    node_f[base + 17u] = p.river_warp01.z;
    node_f[base + 18u] = p.river_warp01.w;
    node_f[base + 19u] = p.river_warp23.x;
    node_f[base + 20u] = p.river_warp23.y;
    node_f[base + 21u] = p.river_warp23.z;
    node_f[base + 22u] = p.river_warp23.w;
    node_f[base + 23u] = p.region_warp.x;
    node_f[base + 24u] = p.region_warp.y;
    node_f[base + 25u] = p.region_warp.z;
    node_f[base + 26u] = p.gully_oct.x;
    node_f[base + 27u] = p.gully_oct.y;
    node_f[base + 28u] = p.gully_oct.z;
    node_f[base + 29u] = p.gully_oct.w;
    node_f[base + 30u] = p.floodplain.x;
    node_f[base + 31u] = p.floodplain.y;
    node_f[base + 32u] = p.floodplain.z;
    node_f[base + 33u] = p.floodplain.w;
}

fn pre_from(f: array<f32, 34>, flags: u32, cut: vec2<f32>) -> Pre {
    var p = pre_none();
    p.flags = flags;
    p.gully = vec2<f32>(f[2], f[3]);
    p.road_major = vec3<f32>(f[4], f[5], f[6]);
    p.road_minor = vec3<f32>(f[7], f[8], f[9]);
    p.relief = vec4<f32>(f[10], f[11], f[12], f[13]);
    p.relief4 = f[14];
    p.relief_cut = cut;
    p.river_warp01 = vec4<f32>(f[15], f[16], f[17], f[18]);
    p.river_warp23 = vec4<f32>(f[19], f[20], f[21], f[22]);
    p.region_warp = vec3<f32>(f[23], f[24], f[25]);
    p.gully_oct = vec4<f32>(f[26], f[27], f[28], f[29]);
    p.floodplain = vec4<f32>(f[30], f[31], f[32], f[33]);
    return p;
}

fn sites_put(k: u32, s: Sites) {
    node_ids[k] = s.id0;
    node_ids[k + 1u] = s.id1;
    node_pts[k] = vec4<f64>(s.p0, 0.0lf);
    node_pts[k + 1u] = vec4<f64>(s.p1, 0.0lf);
}

fn sites_get(k: u32) -> Sites {
    var s: Sites;
    s.id0 = node_ids[k];
    s.id1 = node_ids[k + 1u];
    s.p0 = node_pts[k].xyz;
    s.p1 = node_pts[k + 1u].xyz;
    return s;
}

/// Node ids per node: 3 sites x 2, the forest stand.
const NODE_IDS: u32 = 8u;

/// Coarse grid nodes: macro fields and the smooth inputs (`tile.rs`, `nodes`).
@compute @workgroup_size(64)
fn grid_nodes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = tiles[gid.y];
    let k = gid.x;
    if ((ti.flags & TF_GRID) == 0u || k >= ti.ng * ti.ng) {
        return;
    }
    let r = rows[ti.row_n + k / ti.ng];
    let c = cols[ti.col_n + k % ti.ng];
    let ctx = row_col_ctx(r, c, r.gsd);
    let m = macro_at(ctx.p, ctx.gsd);
    let pre = pre_at(ctx, m, (ti.flags & TF_GULLY_GRID) != 0u, (ti.flags & TF_ROADS_GRID) != 0u, true, vec2<f32>(ti.relief_cut_r, ti.relief_cut_h));
    let nb = (ti.node0 + k) * NODE_F;
    macro_put(nb + NODE_MAC, m);
    pre_put(nb + NODE_PRE, m.mtn_warp, pre);
    node_f[nb + NODE_FLAGS] = bitcast<f32>(pre.flags);
    let ib = (ti.node0 + k) * NODE_IDS;
    sites_put(ib + 0u, pre.site_lake);
    sites_put(ib + 2u, pre.site_region);
    sites_put(ib + 4u, pre.site_town);
    grid_nodes_surface(ti, k, ctx, nb, ib);
}

fn catmull_rom_weights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2));
}

/// Position of a point in the tile's coarse grid: lower node (i0, j0) of the cell and fraction.
struct GridPos {
    i0: u32,
    j0: u32,
    fx: f32,
    fy: f32,
}

/// `u`, `v`: grid coordinates (node units) relative to the tile's first node.
fn grid_pos(ti: TileInfo, u: f32, v: f32) -> GridPos {
    var g: GridPos;
    g.i0 = u32(clamp(i32(floor(u)), 1, i32(ti.ng) - 3));
    g.j0 = u32(clamp(i32(floor(v)), 1, i32(ti.ng) - 3));
    g.fx = u - f32(g.i0);
    g.fy = v - f32(g.j0);
    return g;
}

fn node_index(ti: TileInfo, i: u32, j: u32) -> u32 {
    return ti.node0 + j * ti.ng + i;
}

/// The macro fields at a pixel: bilinear between the nodes (`Macro::bilerp`); the mountain
/// warp from the Catmull-Rom interpolation when on the grid, else exact.
fn grid_macro(ti: TileInfo, g: GridPos, ctx: Ctx) -> Macro {
    var a: array<f32, 19>;
    let n00 = node_index(ti, g.i0, g.j0) * NODE_F;
    let n10 = node_index(ti, g.i0 + 1u, g.j0) * NODE_F;
    let n01 = node_index(ti, g.i0, g.j0 + 1u) * NODE_F;
    let n11 = node_index(ti, g.i0 + 1u, g.j0 + 1u) * NODE_F;
    for (var k = 0u; k < 19u; k++) {
        let x = node_f[n00 + k];
        let y = node_f[n10 + k];
        let z = node_f[n01 + k];
        let w = node_f[n11 + k];
        let t = x + (y - x) * g.fx;
        let u = z + (w - z) * g.fx;
        a[k] = t + (u - t) * g.fy;
    }
    var m = macro_from(a);
    if ((ti.flags & TF_WARP_GRID) != 0u) {
        let f = grid_pre_flat(ti, g);
        m.mtn_warp = vec2<f32>(f[0], f[1]);
    } else {
        m.mtn_warp = mtn_warp_at(ctx.p, ctx.gsd);
    }
    return m;
}

/// The Catmull-Rom interpolated `Pre` inputs (flat) at a pixel.
fn grid_pre_flat(ti: TileInfo, g: GridPos) -> array<f32, 34> {
    let wx = catmull_rom_weights(g.fx);
    let wy = catmull_rom_weights(g.fy);
    var f: array<f32, 34>;
    for (var b = 0u; b < 4u; b++) {
        for (var a = 0u; a < 4u; a++) {
            let w = wx[a] * wy[b];
            let nb = node_index(ti, g.i0 + a - 1u, g.j0 + b - 1u) * NODE_F + NODE_PRE;
            for (var k = 0u; k < 34u; k++) {
                f[k] += w * node_f[nb + k];
            }
        }
    }
    return f;
}

fn same_pair(a: Sites, b: Sites) -> bool {
    return (a.id0 == b.id0 && a.id1 == b.id1) || (a.id0 == b.id1 && a.id1 == b.id0);
}

/// The `Pre` inputs at a pixel: interpolated, with the lattice sites where the four nodes
/// around agree.
fn grid_pre(ti: TileInfo, g: GridPos) -> Pre {
    // the fields every node holds (those of node (i0, j0))
    let flags = bitcast<u32>(node_f[node_index(ti, g.i0, g.j0) * NODE_F + NODE_FLAGS]);
    var pre = pre_from(grid_pre_flat(ti, g), flags & ~(P_SITE_LAKE | P_SITE_REGION | P_SITE_TOWN), vec2<f32>(ti.relief_cut_r, ti.relief_cut_h));
    let i00 = node_index(ti, g.i0, g.j0) * NODE_IDS;
    let i10 = node_index(ti, g.i0 + 1u, g.j0) * NODE_IDS;
    let i01 = node_index(ti, g.i0, g.j0 + 1u) * NODE_IDS;
    let i11 = node_index(ti, g.i0 + 1u, g.j0 + 1u) * NODE_IDS;
    for (var k = 0u; k < 3u; k++) {
        let s0 = sites_get(i00 + 2u * k);
        if (same_pair(sites_get(i10 + 2u * k), s0) && same_pair(sites_get(i01 + 2u * k), s0) && same_pair(sites_get(i11 + 2u * k), s0)) {
            switch k {
                case 0u: {
                    pre.site_lake = s0;
                    pre.flags |= P_SITE_LAKE;
                }
                case 1u: {
                    pre.site_region = s0;
                    pre.flags |= P_SITE_REGION;
                }
                default: {
                    pre.site_town = s0;
                    pre.flags |= P_SITE_TOWN;
                }
            }
        }
    }
    return pre;
}

/// Grid coordinates of pass-A pixel (i, j).
fn grid_uv_a(ti: TileInfo, i: u32, j: u32) -> vec2<f32> {
    return vec2<f32>(ti.gu0 + f32(i) / 16.0, ti.gu0 + f32(j) / 16.0);
}

fn orderable(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    if ((b & 0x80000000u) != 0u) {
        return ~b;
    }
    return b | 0x80000000u;
}

fn from_orderable(u: u32) -> f32 {
    if ((u & 0x80000000u) != 0u) {
        return bitcast<f32>(u & 0x7fffffffu);
    }
    return bitcast<f32>(~u);
}

var<workgroup> wg_hmax: atomic<u32>;

/// Pass A up to the drainage, per pixel centre with the 2-pixel apron; the maximum height per
/// bin (= workgroup).
@compute @workgroup_size(16, 16)
fn pass_a1(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let ti = tiles[gid.z];
    let i = gid.x;
    let j = gid.y;
    if (li == 0u) {
        atomicStore(&wg_hmax, 0u);
    }
    workgroupBarrier();
    if (i < NA2 && j < NA2) {
        let r = rows[ti.row_a + j];
        let c = cols[ti.col_a + i];
        let ctx = row_col_ctx(r, c, r.gsd);
        var m: Macro;
        var pre = pre_none();
        if ((ti.flags & TF_GRID) != 0u) {
            let uv = grid_uv_a(ti, i, j);
            let g = grid_pos(ti, uv.x, uv.y);
            m = grid_macro(ti, g, ctx);
            pre = grid_pre(ti, g);
        } else {
            m = macro_at(ctx.p, ctx.gsd);
        }
        let rl = relief(ctx, m, pre);
        let b = (ti.pix0 + j * NA2 + i) * PIX_F;
        pix_a[b + 0u] = m.cont;
        pix_a[b + 1u] = m.plateau;
        pix_a[b + 2u] = m.belt;
        pix_a[b + 3u] = m.belt2;
        pix_a[b + 4u] = m.belt_var;
        pix_a[b + 5u] = m.hill_amp;
        pix_a[b + 6u] = m.rough;
        pix_a[b + 7u] = m.temp;
        pix_a[b + 8u] = m.moist;
        pix_a[b + 9u] = m.mesa;
        pix_a[b + 10u] = m.sand;
        pix_a[b + 11u] = m.agri;
        pix_a[b + 12u] = m.style.x;
        pix_a[b + 13u] = m.style.y;
        pix_a[b + 14u] = m.style.z;
        pix_a[b + 15u] = m.style.w;
        pix_a[b + 16u] = m.river_width;
        pix_a[b + 17u] = m.mtn_warp.x;
        pix_a[b + 18u] = m.mtn_warp.y;
        pix_a[b + 19u] = rl.h;
        pix_a[b + 20u] = rl.smooth_h;
        pix_a[b + 21u] = rl.temp0;
        pix_a[b + 22u] = rl.moist;
        pix_a[b + 23u] = rl.mountain;
        pix_a[b + 24u] = rl.ridged;
        pix_a[b + 25u] = rl.hill_amp;
        pix_a[b + 26u] = rl.micro;
        pix_a[b + 27u] = rl.mesa;
        pix_a[b + 28u] = rl.sand;
        pix_a[b + 29u] = rl.gully_n;
        atomicMax(&wg_hmax, orderable(rl.h));
        lake_requests(ctx, m, pre, rl, li);
    } else {
        wg_lake_id[2u * li] = 0lu;
        wg_lake_id[2u * li + 1u] = 0lu;
    }
    workgroupBarrier();
    if (li == 0u) {
        bins[ti.bin0 + wg.y * NBIN + wg.x].z = atomicLoad(&wg_hmax);
        flush_lake_requests();
    }
}

/// A lattice lake whose level the host is to provide.
struct LakeReq {
    pt: vec4<f64>,
    id: u64,
    _p: u64,
}

@group(2) @binding(11) var<storage, read_write> lake_req: array<LakeReq>;

var<workgroup> wg_lake_id: array<u64, 512>;
var<workgroup> wg_lake_pt: array<vec4<f64>, 512>;

/// The lattice lakes a pixel looks up whose levels are not known yet (the tests of
/// `terrain_rest`, which follows with the same macro fields and relief).
fn lake_requests(c: Ctx, m: Macro, pre: Pre, r: Relief, li: u32) {
    wg_lake_id[2u * li] = 0lu;
    wg_lake_id[2u * li + 1u] = 0lu;
    let land = smoothstep1(-0.06, 0.05, m.cont);
    let lake_cell = f32(cfg.lake_cell);
    if (!(cfg.lake_density > 0.0 && land > 0.3 && lake_cell > c.gsd)) {
        return;
    }
    var wc: Cell3;
    if ((pre.flags & P_SITE_LAKE) != 0u) {
        wc = worley3_from(c.p, cfg.lake_cell, 1.0lf / cfg.lake_cell, pre.site_lake);
    } else {
        wc = worley3(cfg.seed ^ 0x1A4Elu, c.p, cfg.lake_cell, 0.85);
    }
    for (var k = 0u; k < 2u; k++) {
        var id = wc.id;
        var pt = wc.point;
        if (k == 1u) {
            id = wc.id2;
            pt = wc.point2;
        }
        let prob = cfg.lake_density * (0.3 + 0.9 * r.moist) * (1.0 - 0.8 * r.mountain);
        if (u01k(id, 1lu) > prob * 1.0001 + 1e-6) {
            continue;
        }
        let rad = min(300.0 * exp(u01k(id, 2lu) * 2.7), lake_cell * 0.3);
        let plen = sqrt(dot(c.p, c.p));
        let pc = pt * (plen / sqrt(dot(pt, pt)));
        if (dist64(c.p, pc) > rad * 1.5 * 1.0001 + 1.0) {
            continue;
        }
        if (!lake_known(id)) {
            wg_lake_id[2u * li + k] = id;
            wg_lake_pt[2u * li + k] = vec4<f64>(pt, 0.0lf);
        }
    }
}

/// Append the workgroup's distinct lake requests.
fn flush_lake_requests() {
    var seen: array<u64, 8>;
    var n = 0u;
    for (var k = 0u; k < 512u; k++) {
        let id = wg_lake_id[k];
        if (id == 0lu) {
            continue;
        }
        var dup = false;
        for (var s = 0u; s < n; s++) {
            if (seen[s] == id) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }
        if (n < 8u) {
            seen[n] = id;
            n += 1u;
        }
        let slot = atomicAdd(&counters[2], 1u);
        if (slot < arrayLength(&lake_req)) {
            lake_req[slot].id = id;
            lake_req[slot].pt = wg_lake_pt[k];
        }
    }
}

fn pix_macro(b: u32) -> Macro {
    var a: array<f32, 19>;
    for (var k = 0u; k < 19u; k++) {
        a[k] = pix_a[b + k];
    }
    return macro_from(a);
}

fn pix_relief(b: u32) -> Relief {
    var r: Relief;
    r.h = pix_a[b + 19u];
    r.smooth_h = pix_a[b + 20u];
    r.temp0 = pix_a[b + 21u];
    r.moist = pix_a[b + 22u];
    r.mountain = pix_a[b + 23u];
    r.ridged = pix_a[b + 24u];
    r.hill_amp = pix_a[b + 25u];
    r.micro = pix_a[b + 26u];
    r.mesa = pix_a[b + 27u];
    r.sand = pix_a[b + 28u];
    r.gully_n = pix_a[b + 29u];
    return r;
}

var<workgroup> wg_flags: array<u32, 256>;
var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_base: u32;

/// Can channel piece `s` reach a disc of `radius` around `c` where the relief is at most
/// `h_max`? (`World::local_segments`)
fn seg_may_reach(s: Seg, c: vec3<f64>, radius: f32, h_max: f32) -> bool {
    let la = cfg.lvl_a[s.level];
    let lb = cfg.lvl_b[s.level];
    let warp = 0.9 * max(lb.y, 1e-3) * la.x;
    let floor_min = min(s.ha, s.hb);
    let reach = max(s.valley * 1.5 + s.hw + 200.0, 11.3 * s.hw + 6.0 * (h_max - floor_min + 2.0 + 0.04 * s.hw) + 50.0);
    let ab = vec3<f32>(s.b.xyz - s.a.xyz);
    let ca = vec3<f32>(c - s.a.xyz);
    let u = clamp(dot(ca, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
    return length(ca - ab * u) - radius - warp <= reach + 1.0 + 1e-3 * (radius + reach);
}

/// The channel pieces per bin, in the order of the tile's list (one workgroup per bin).
@compute @workgroup_size(256)
fn bin_segments(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let ti = tiles[wg.y];
    let bx = wg.x % NBIN;
    let by = wg.x / NBIN;
    let bi = ti.bin0 + wg.x;
    // the bin's centre and radius (pass-A pixel centres it covers)
    let i0 = bx * 16u;
    let j0 = by * 16u;
    let i1 = min(i0 + 15u, NA2 - 1u);
    let j1 = min(j0 + 15u, NA2 - 1u);
    let im = (i0 + i1) / 2u;
    let jm = (j0 + j1) / 2u;
    let rm = rows[ti.row_a + jm];
    let ctr = row_col_ctx(rm, cols[ti.col_a + im], rm.gsd).p;
    var radius = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let ii = select(i0, i1, (k & 1u) != 0u);
        let jj = select(j0, j1, (k & 2u) != 0u);
        let rr = rows[ti.row_a + jj];
        radius = max(radius, dist64(row_col_ctx(rr, cols[ti.col_a + ii], rr.gsd).p, ctr));
    }
    radius += max(rows[ti.row_a + j0].gsd, rows[ti.row_a + j1].gsd);
    let h_max = from_orderable(bins[bi].z);
    if (li == 0u) {
        atomicStore(&wg_count, 0u);
    }
    workgroupBarrier();
    // count
    for (var s0 = 0u; s0 < ti.nseg; s0 += 256u) {
        let s = s0 + li;
        if (s < ti.nseg && seg_may_reach(segs[ti.seg0 + s], ctr, radius, h_max)) {
            atomicAdd(&wg_count, 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        let n = atomicLoad(&wg_count);
        let base = atomicAdd(&counters[0], n);
        if (base + n > arrayLength(&bin_list)) {
            atomicStore(&counters[1], 1u);
            wg_base = 0xffffffffu;
        } else {
            wg_base = base;
        }
        bins[bi].x = base;
        bins[bi].y = n;
    }
    workgroupBarrier();
    let base = wg_base;
    if (base == 0xffffffffu) {
        return;
    }
    // write in order: a prefix sum over each chunk of 256 pieces
    var off = 0u;
    for (var s0 = 0u; s0 < ti.nseg; s0 += 256u) {
        let s = s0 + li;
        let f = select(0u, 1u, s < ti.nseg && seg_may_reach(segs[ti.seg0 + s], ctr, radius, h_max));
        wg_flags[li] = f;
        workgroupBarrier();
        // Hillis-Steele inclusive scan
        for (var d = 1u; d < 256u; d *= 2u) {
            var v = wg_flags[li];
            if (li >= d) {
                v += wg_flags[li - d];
            }
            workgroupBarrier();
            wg_flags[li] = v;
            workgroupBarrier();
        }
        if (f == 1u) {
            bin_list[base + off + wg_flags[li] - 1u] = ti.seg0 + s;
        }
        let total = wg_flags[255];
        workgroupBarrier();
        off += total;
    }
}

/// The rest of pass A per pixel centre.
@compute @workgroup_size(16, 16)
fn pass_a2(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let ti = tiles[gid.z];
    let i = gid.x;
    let j = gid.y;
    if (i >= NA2 || j >= NA2) {
        return;
    }
    let r = rows[ti.row_a + j];
    let c = cols[ti.col_a + i];
    let ctx = row_col_ctx(r, c, r.gsd);
    let b = (ti.pix0 + j * NA2 + i) * PIX_F;
    let m = pix_macro(b);
    let rl = pix_relief(b);
    var pre = pre_none();
    if ((ti.flags & TF_GRID) != 0u) {
        let uv = grid_uv_a(ti, i, j);
        pre = grid_pre(ti, grid_pos(ti, uv.x, uv.y));
    }
    let bin = bins[ti.bin0 + wg.y * NBIN + wg.x];
    var dr: Drain;
    dr.seg0 = bin.x;
    dr.nseg = bin.y;
    dr.sink0 = ti.sink0;
    dr.nsink = ti.nsink;
    terr[ti.pix0 + j * NA2 + i] = terrain_rest(ctx, m, pre, rl, MODE_FULL, dr);
}
