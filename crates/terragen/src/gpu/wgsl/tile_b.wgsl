// Tile kernels, pass B (`tile.rs`): the land-use sites the host is to provide, the supersampled
// surface per pixel (tile + 1-pixel apron), the canopy opening and the output layers.

/// pass-B grid: tile pixels plus a 1-pixel apron
const NA: u32 = 258u;
const N: u32 = 256u;
/// floats per pass-B pixel: emission (3), albedo (3), height, ground, lit, cls, canopy, -
const PB_F: u32 = 12u;
/// u32 per output pixel: rgb, albedo, emission, normal, elevation (f32 bits), land cover
const OUT_U: u32 = 6u;

/// A site whose data the host is to provide: point, id.
struct SiteReq {
    pt: vec4<f64>,
    id: u64,
    _p0: u64,
    _p1: u64,
    _p2: u64,
}

@group(2) @binding(12) var<storage, read_write> pixb: array<f32>;
@group(2) @binding(13) var<storage, read_write> scratch_a: array<f32>;
@group(2) @binding(14) var<storage, read_write> scratch_b: array<f32>;
@group(2) @binding(15) var<storage, read_write> tile_out: array<u32>;
/// per tile: max of orderable(min elevation) inverted, max of orderable(max elevation)
@group(2) @binding(16) var<storage, read_write> ranges: array<atomic<u32>>;
@group(2) @binding(17) var<storage, read_write> region_req: array<SiteReq>;
@group(2) @binding(18) var<storage, read_write> town_req: array<vec4<i32>>;
@group(2) @binding(19) var<storage, read_write> eco_req: array<SiteReq>;

const C_REGION_REQ: u32 = 3u;
const C_TOWN_REQ: u32 = 4u;
const C_MISSING: u32 = 5u;
const C_ECO_REQ: u32 = 6u;

var<workgroup> wg_eco_id: array<u64, 512>;

var<workgroup> wg_site_id: array<u64, 256>;
var<workgroup> wg_site_pt: array<vec4<f64>, 256>;

/// Is pass-B pixel (i, j) standing water at every sub-sample? (all its pass-A neighbours are
/// water above their ground: the sample's water level, their maximum, then lies above its
/// interpolated ground; pass B returns before the land use there)
fn all_water(ti: TileInfo, i: u32, j: u32) -> bool {
    for (var dj = 0u; dj < 3u; dj++) {
        for (var di = 0u; di < 3u; di++) {
            let t = terr[ti.pix0 + (j + dj) * NA2 + (i + di)];
            if (t.water_kind == W_NONE || !(t.water > t.ground)) {
                return false;
            }
        }
    }
    return true;
}

/// The region of every pass-B pixel on land (its id and site), deduplicated per workgroup.
@compute @workgroup_size(16, 16)
fn region_requests(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let ti = tiles[gid.z];
    let i = gid.x;
    let j = gid.y;
    wg_site_id[li] = 0lu;
    wg_eco_id[2u * li] = 0lu;
    wg_eco_id[2u * li + 1u] = 0lu;
    if (i < NA && j < NA) {
        // the ecoregions of the pixel (water too: its surface layers)
        let te = terr[ti.pix0 + (j + 1u) * NA2 + (i + 1u)];
        wg_eco_id[2u * li] = te.eco_id;
        wg_eco_id[2u * li + 1u] = te.eco_id2;
    }
    if (i < NA && j < NA && !all_water(ti, i, j)) {
        // the pass-A pixel at the pass-B pixel's centre
        let ia = i + 1u;
        let ja = j + 1u;
        let t = terr[ti.pix0 + ja * NA2 + ia];
        if (t.region_id != 0lu) {
            let r = rows[ti.row_a + ja];
            let c = cols[ti.col_a + ia];
            let ctx = row_col_ctx(r, c, r.gsd);
            var pre = pre_none();
            if ((ti.flags & TF_GRID) != 0u) {
                let uv = grid_uv_a(ti, ia, ja);
                pre = grid_pre(ti, grid_pos(ti, uv.x, uv.y));
            }
            let pw = ctx.p + vec3<f64>(region_warp_at(ctx.p, pre));
            let wc = region_cell(pw, pre);
            wg_site_id[li] = wc.id;
            wg_site_pt[li] = vec4<f64>(wc.point, 0.0lf);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        var seen: array<u64, 64>;
        var n = 0u;
        for (var k = 0u; k < 256u; k++) {
            let id = wg_site_id[k];
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
            if (n < 64u) {
                seen[n] = id;
                n += 1u;
            }
            let slot = atomicAdd(&counters[C_REGION_REQ], 1u);
            if (slot < arrayLength(&region_req)) {
                region_req[slot].id = id;
                region_req[slot].pt = wg_site_pt[k];
            }
        }
        // the ecoregions (a handful per workgroup): ids only, the host finds their sites
        var eseen: array<u64, 16>;
        var ne = 0u;
        for (var k = 0u; k < 512u; k++) {
            let id = wg_eco_id[k];
            if (id == 0lu) {
                continue;
            }
            var dup = false;
            for (var e = 0u; e < ne; e++) {
                if (eseen[e] == id) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            if (ne < 16u) {
                eseen[ne] = id;
                ne += 1u;
            }
            let slot = atomicAdd(&counters[C_ECO_REQ], 1u);
            if (slot < arrayLength(&eco_req)) {
                eco_req[slot].id = id;
            }
        }
    }
}

var<workgroup> wg_cell: array<vec4<i32>, 1024>;

/// Position of pass-B sub-sample (sx, sy) of pixel (i, j) in the pass-A grid's neighbours.
fn sub_offset(s: u32, ss: u32) -> f32 {
    return (f32(s) + 0.5) / f32(ss) - 0.5;
}

/// The town lattice cells of every pass-B sub-sample that looks for a town.
@compute @workgroup_size(16, 16)
fn town_requests(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let ti = tiles[gid.z];
    let i = gid.x;
    let j = gid.y;
    let ss = ti.ss;
    for (var k = 0u; k < 4u; k++) {
        wg_cell[li * 4u + k] = vec4<i32>(0, 0, 0, 0);
    }
    if (i < NA && j < NA && cfg.towns > 0.0 && !all_water(ti, i, j)) {
        let t = terr[ti.pix0 + (j + 1u) * NA2 + (i + 1u)];
        if (t.town != 0u) {
            var k = 0u;
            for (var sy = 0u; sy < ss; sy++) {
                for (var sx = 0u; sx < ss; sx++) {
                    let r = rows[ti.row_b + j * ss + sy];
                    let c = cols[ti.col_b + i * ss + sx];
                    let p = vec3<f64>(r.ncl * c.co, r.ncl * c.so, r.z);
                    let cell = town_cell_of(p);
                    let v = vec4<i32>(i32(cell.x), i32(cell.y), i32(cell.z), 1);
                    // (only distinct cells of the pixel; at most 4 slots)
                    var dup = false;
                    for (var q = 0u; q < k; q++) {
                        if (all(wg_cell[li * 4u + q] == v)) {
                            dup = true;
                        }
                    }
                    if (!dup && k < 4u) {
                        wg_cell[li * 4u + k] = v;
                        k += 1u;
                    }
                }
            }
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        var seen: array<vec4<i32>, 64>;
        var n = 0u;
        for (var k = 0u; k < 1024u; k++) {
            let v = wg_cell[k];
            if (v.w == 0) {
                continue;
            }
            var dup = false;
            for (var s = 0u; s < n; s++) {
                if (all(seen[s] == v)) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            if (n < 64u) {
                seen[n] = v;
                n += 1u;
            }
            let slot = atomicAdd(&counters[C_TOWN_REQ], 1u);
            if (slot < arrayLength(&town_req)) {
                town_req[slot] = v;
            }
        }
    }
}

/// pass-A terrain at pass-B grid coordinates (i, j) in [-1, NA] (clamped).
fn at(ti: TileInfo, i: i32, j: i32) -> Terrain {
    let ii = u32(clamp(i + 1, 0, i32(NA2) - 1));
    let jj = u32(clamp(j + 1, 0, i32(NA2) - 1));
    return terr[ti.pix0 + jj * NA2 + ii];
}

fn bilerp4(v: vec4<f32>, fx: f32, fy: f32) -> f32 {
    let a = v.x + (v.y - v.x) * fx;
    let b = v.z + (v.w - v.z) * fx;
    return a + (b - a) * fy;
}

/// The pixel fields of pass-B pixel (i, j): per pixel and from the grid, with the forest stand
/// where the four grid nodes around agree.
fn pixel_fields_b(ti: TileInfo, i: u32, j: u32, p: vec3<f64>, gsd: f32) -> PixFields {
    var pf: PixFields;
    pf.p = p;
    pf.gsd = gsd;
    pf.pending = PF_LAZY;
    pf.has_stand = 0u;
    if ((ti.flags & TF_GRID) != 0u) {
        pf.cut = ti.pf_cut;
        pf.split = SPLIT_HIGH;
        for (var k = 0u; k < PF_N; k++) {
            if ((PF_LAZY & (1u << k)) == 0u) {
                pf.f[k] = pixel_field(k, p, gsd, SPLIT_HIGH, ti.pf_cut);
            } else {
                pf.f[k] = 0.0;
            }
        }
        // the grid coordinates of the pass-B pixel centre (= pass-A pixel (i + 1, j + 1))
        let uv = grid_uv_a(ti, i + 1u, j + 1u);
        let g = grid_pos(ti, uv.x, uv.y);
        let wx = catmull_rom_weights(g.fx);
        let wy = catmull_rom_weights(g.fy);
        for (var b = 0u; b < 4u; b++) {
            for (var a = 0u; a < 4u; a++) {
                let w = wx[a] * wy[b];
                let nb = node_index(ti, g.i0 + a - 1u, g.j0 + b - 1u) * NODE_F + NODE_PF;
                for (var k = 0u; k < PF_N; k++) {
                    pf.f[k] += w * node_f[nb + k];
                }
            }
        }
        let s00 = node_ids[node_index(ti, g.i0, g.j0) * NODE_IDS + 6u];
        let s10 = node_ids[node_index(ti, g.i0 + 1u, g.j0) * NODE_IDS + 6u];
        let s01 = node_ids[node_index(ti, g.i0, g.j0 + 1u) * NODE_IDS + 6u];
        let s11 = node_ids[node_index(ti, g.i0 + 1u, g.j0 + 1u) * NODE_IDS + 6u];
        if (s10 == s00 && s01 == s00 && s11 == s00) {
            pf.stand = s00;
            pf.has_stand = 1u;
        }
    } else {
        pf.cut = 0.0;
        pf.split = SPLIT_ALL;
        for (var k = 0u; k < PF_N; k++) {
            if ((PF_LAZY & (1u << k)) == 0u) {
                pf.f[k] = pixel_field(k, p, gsd, SPLIT_ALL, 0.0);
            } else {
                pf.f[k] = 0.0;
            }
        }
    }
    return pf;
}

/// The inputs of sub-sample (sx, sy) of pass-B pixel (i, j): its frame, terrain, local values
/// and bare ground.
struct SampleIn {
    c: Ctx,
    t: Terrain,
    l: Local,
    ground: f32,
}

fn sample_in(ti: TileInfo, i: u32, j: u32, sx: u32, sy: u32, slope: f32, grad: vec2<f32>, gsd: f32) -> SampleIn {
    let ss = ti.ss;
    let fxo = sub_offset(sx, ss);
    let fyo = sub_offset(sy, ss);
    let ii = i32(i);
    let jj = i32(j);
    var i0 = ii;
    var j0 = jj;
    var fx = fxo;
    var fy = fyo;
    if (fxo < 0.0) {
        i0 = ii - 1;
        fx = 1.0 + fxo;
    }
    if (fyo < 0.0) {
        j0 = jj - 1;
        fy = 1.0 + fyo;
    }
    let n00 = at(ti, i0, j0);
    let n10 = at(ti, i0 + 1, j0);
    let n01 = at(ti, i0, j0 + 1);
    let n11 = at(ti, i0 + 1, j0 + 1);
    var t = at(ti, ii, jj);
    let ground = bilerp4(vec4<f32>(n00.ground, n10.ground, n01.ground, n11.ground), fx, fy);
    var wl = -NONE_F;
    var wk = W_NONE;
    if (n00.water_kind != W_NONE && n00.water > wl) {
        wl = n00.water;
        wk = n00.water_kind;
    }
    if (n10.water_kind != W_NONE && n10.water > wl) {
        wl = n10.water;
        wk = n10.water_kind;
    }
    if (n01.water_kind != W_NONE && n01.water > wl) {
        wl = n01.water;
        wk = n01.water_kind;
    }
    if (n11.water_kind != W_NONE && n11.water > wl) {
        wl = n11.water;
        wk = n11.water_kind;
    }
    var l: Local;
    l.ground = ground;
    l.water = wl;
    l.water_kind = wk;
    l.river_d = bilerp4(clamp(vec4<f32>(n00.river_d, n10.river_d, n01.river_d, n11.river_d), vec4<f32>(-1e6), vec4<f32>(1e6)), fx, fy);
    l.river_hw = max(max(n00.river_hw, n10.river_hw), max(n01.river_hw, n11.river_hw));
    let rl = vec4<f32>(
        select(ground, n00.river_level, n00.river_hw > 0.0),
        select(ground, n10.river_level, n10.river_hw > 0.0),
        select(ground, n01.river_level, n01.river_hw > 0.0),
        select(ground, n11.river_level, n11.river_hw > 0.0),
    );
    l.river_level = bilerp4(rl, fx, fy);
    l.road_major = bilerp4(clamp(vec4<f32>(n00.road_major, n10.road_major, n01.road_major, n11.road_major), vec4<f32>(-1e6), vec4<f32>(1e6)), fx, fy);
    l.road_minor = bilerp4(clamp(vec4<f32>(n00.road_minor, n10.road_minor, n01.road_minor, n11.road_minor), vec4<f32>(-1e6), vec4<f32>(1e6)), fx, fy);
    t.region_edge = bilerp4(min(vec4<f32>(n00.region_edge, n10.region_edge, n01.region_edge, n11.region_edge), vec4<f32>(1e6)), fx, fy);
    // the ecoregion border distance, signed by the side (`tile.rs`)
    let ee = min(vec4<f32>(n00.eco_edge, n10.eco_edge, n01.eco_edge, n11.eco_edge), vec4<f32>(1e7));
    let es = vec4<f32>(select(-1.0, 1.0, n00.eco_id == t.eco_id), select(-1.0, 1.0, n10.eco_id == t.eco_id), select(-1.0, 1.0, n01.eco_id == t.eco_id), select(-1.0, 1.0, n11.eco_id == t.eco_id));
    l.eco_edge = bilerp4(ee * es, fx, fy);
    l.slope = slope;
    l.grad = grad;
    l.fw = gsd / f32(ss);
    l.blk = ti.bin0 + ((j + 1u) / 16u) * NBIN + (i + 1u) / 16u;
    let r = rows[ti.row_b + j * ss + sy];
    let c = cols[ti.col_b + i * ss + sx];
    var o: SampleIn;
    o.c = row_col_ctx(r, c, gsd);
    o.t = t;
    o.l = l;
    o.ground = ground;
    return o;
}

// ---- pass B in stages
//
// The stack runs in `PB_STAGES` pipelines (`pass_b0`, `pass_b1`, ..: each a range of the
// stack's steps, `gpu::stages_wgsl`), so that no pipeline holds the whole surface model (the
// drivers inline every call into one function: one pipeline with every kit took minutes to
// compile, and its compile time grows faster than its size). Between stages the stack's
// mutable state goes through `pbw`. The sub-samples run in rounds of `PB_RS` (adaptive: the
// diagonal pair, then the other two only where it disagrees); the last stage accumulates the
// pixel's sums in `pixb` and finishes it.

/// The pass-B work buffer: header (round, first tile of the chunk, an opaque 0, chunk tiles),
/// then per chunk pixel: the stage state of each sample of the round, its pixel fields, its
/// samples' classes and flags.
/// (`PBW_HDR`, `PB_RS` samples per round, `ST_F` words of stage state per sample: masks (19),
/// colour (3), height, emission (3), lit, class and flags; `PFW` words of pixel fields: f (16),
/// pending, has_stand, stand (2); `ACCW` words of classes (8 bits each, up to 16) and flags
/// (1: finished): constants from `gpu::stages_wgsl`)
@group(2) @binding(22) var<storage, read_write> pbw: array<u32>;

fn st_save(sp: ptr<function, Stack>, b: u32) {
    let m = (*sp).m;
    pbw[b + 0u] = bitcast<u32>(m.cover);
    pbw[b + 1u] = bitcast<u32>(m.rock);
    pbw[b + 2u] = bitcast<u32>(m.sand);
    pbw[b + 3u] = bitcast<u32>(m.beach);
    pbw[b + 4u] = bitcast<u32>(m.snow);
    pbw[b + 5u] = bitcast<u32>(m.shore_keep);
    pbw[b + 6u] = bitcast<u32>(m.natural_ok);
    pbw[b + 7u] = bitcast<u32>(m.flat_ok);
    pbw[b + 8u] = bitcast<u32>(m.micro);
    pbw[b + 9u] = bitcast<u32>(m.riparian);
    pbw[b + 10u] = bitcast<u32>(m.woodlot);
    pbw[b + 11u] = bitcast<u32>(m.field_cov);
    pbw[b + 12u] = bitcast<u32>(m.town_urban);
    pbw[b + 13u] = bitcast<u32>(m.town_cov);
    pbw[b + 14u] = bitcast<u32>(m.river_clear);
    pbw[b + 15u] = bitcast<u32>(m.road_major_cov);
    pbw[b + 16u] = bitcast<u32>(m.road_cov);
    pbw[b + 17u] = bitcast<u32>(m.river_cov);
    pbw[b + 18u] = bitcast<u32>(m.veg);
    pbw[b + 19u] = bitcast<u32>((*sp).col.x);
    pbw[b + 20u] = bitcast<u32>((*sp).col.y);
    pbw[b + 21u] = bitcast<u32>((*sp).col.z);
    pbw[b + 22u] = bitcast<u32>((*sp).height);
    pbw[b + 23u] = bitcast<u32>((*sp).emission.x);
    pbw[b + 24u] = bitcast<u32>((*sp).emission.y);
    pbw[b + 25u] = bitcast<u32>((*sp).emission.z);
    pbw[b + 26u] = bitcast<u32>((*sp).lit);
    pbw[b + 27u] = ((*sp).cls & 0xffffu) | select(0u, 0x10000u, (*sp).is_water) | select(0u, 0x20000u, (*sp).done);
}

fn st_load(sp: ptr<function, Stack>, b: u32) {
    var m: Masks;
    m.cover = bitcast<f32>(pbw[b + 0u]);
    m.rock = bitcast<f32>(pbw[b + 1u]);
    m.sand = bitcast<f32>(pbw[b + 2u]);
    m.beach = bitcast<f32>(pbw[b + 3u]);
    m.snow = bitcast<f32>(pbw[b + 4u]);
    m.shore_keep = bitcast<f32>(pbw[b + 5u]);
    m.natural_ok = bitcast<f32>(pbw[b + 6u]);
    m.flat_ok = bitcast<f32>(pbw[b + 7u]);
    m.micro = bitcast<f32>(pbw[b + 8u]);
    m.riparian = bitcast<f32>(pbw[b + 9u]);
    m.woodlot = bitcast<f32>(pbw[b + 10u]);
    m.field_cov = bitcast<f32>(pbw[b + 11u]);
    m.town_urban = bitcast<f32>(pbw[b + 12u]);
    m.town_cov = bitcast<f32>(pbw[b + 13u]);
    m.river_clear = bitcast<f32>(pbw[b + 14u]);
    m.road_major_cov = bitcast<f32>(pbw[b + 15u]);
    m.road_cov = bitcast<f32>(pbw[b + 16u]);
    m.river_cov = bitcast<f32>(pbw[b + 17u]);
    m.veg = bitcast<f32>(pbw[b + 18u]);
    (*sp).m = m;
    (*sp).col = vec3<f32>(bitcast<f32>(pbw[b + 19u]), bitcast<f32>(pbw[b + 20u]), bitcast<f32>(pbw[b + 21u]));
    (*sp).height = bitcast<f32>(pbw[b + 22u]);
    (*sp).emission = vec3<f32>(bitcast<f32>(pbw[b + 23u]), bitcast<f32>(pbw[b + 24u]), bitcast<f32>(pbw[b + 25u]));
    (*sp).lit = bitcast<f32>(pbw[b + 26u]);
    let w = pbw[b + 27u];
    (*sp).cls = w & 0xffffu;
    (*sp).is_water = (w & 0x10000u) != 0u;
    (*sp).done = (w & 0x20000u) != 0u;
}

fn pf_save(b: u32, pf: PixFields) {
    for (var k = 0u; k < PF_N; k++) {
        pbw[b + k] = bitcast<u32>(pf.f[k]);
    }
    pbw[b + 16u] = pf.pending;
    pbw[b + 17u] = pf.has_stand;
    pbw[b + 18u] = u32(pf.stand & 0xfffffffflu);
    pbw[b + 19u] = u32(pf.stand >> 32u);
}

/// The pixel fields `pixel_fields_b` made (with the lazy ones evaluated since).
fn pf_load(ti: TileInfo, b: u32, p: vec3<f64>, gsd: f32) -> PixFields {
    var pf: PixFields;
    pf.p = p;
    pf.gsd = gsd;
    if ((ti.flags & TF_GRID) != 0u) {
        pf.cut = ti.pf_cut;
        pf.split = SPLIT_HIGH;
    } else {
        pf.cut = 0.0;
        pf.split = SPLIT_ALL;
    }
    for (var k = 0u; k < PF_N; k++) {
        pf.f[k] = bitcast<f32>(pbw[b + k]);
    }
    pf.pending = pbw[b + 16u];
    pf.has_stand = pbw[b + 17u];
    pf.stand = u64(pbw[b + 18u]) | (u64(pbw[b + 19u]) << 32u);
    return pf;
}

/// Pass B, stage `stage` (a constant per entry point), round `pbw[0]`, per pixel of the tile
/// and its 1-pixel apron.
fn pass_b_stage(gid: vec3<u32>, stage: u32) {
    let i = gid.x;
    let j = gid.y;
    if (i >= NA || j >= NA) {
        return;
    }
    let round = pbw[0];
    let ch = pbw[3];
    let ti = tiles[pbw[1] + gid.z];
    let last = stage + 1u == PB_STAGES;
    // chunk pixel; its state, fields and classes
    let lp = (gid.z * NA + j) * NA + i;
    let pfb = PBW_HDR + ch * NA * NA * PB_RS * ST_F + lp * PFW;
    let ab = PBW_HDR + ch * NA * NA * (PB_RS * ST_F + PFW) + lp * ACCW;
    if (round > 0u && (pbw[ab + 4u] & 1u) != 0u) {
        return;
    }
    if (round == 0u && stage == 0u) {
        pbw[ab + 4u] = 0u;
    }
    surface_missing = 0u;
    let ss = ti.ss;
    let n = ss * ss;
    // the pixel centre's row (= pass-A row j + 1)
    let rc = rows[ti.row_a + j + 1u];
    let gsd = rc.gsd;
    let ctx = row_col_ctx(rc, cols[ti.col_a + i + 1u], gsd);
    var pf: PixFields;
    if (round == 0u && stage == 0u) {
        pf = pixel_fields_b(ti, i, j, ctx.p, gsd);
    } else {
        pf = pf_load(ti, pfb, ctx.p, gsd);
    }
    // slope of the bare ground at pixel scale
    let ii = i32(i);
    let jj = i32(j);
    let dx = (at(ti, ii + 1, jj).ground - at(ti, ii - 1, jj).ground) / (2.0 * gsd);
    let dy = (at(ti, ii, jj + 1).ground - at(ti, ii, jj - 1).ground) / (2.0 * rc.gsd_ns);
    let slope = sqrt(dx * dx + dy * dy);
    let adaptive = (cfg.flags & CF_ADAPTIVE) != 0u;
    let pb = (ti.pix0 + j * NA2 + i) * PB_F;
    let k0 = round * PB_RS;
    let k1 = min(k0 + PB_RS, n);
    var first_s: Surface;
    // One call site of the stage (the driver inlines the surface model at every call).
    for (var k = k0; k < k1; k++) {
        var sx = k % ss;
        var sy = k / ss;
        if (adaptive) {
            // (0, 0), (1, 1), (1, 0), (0, 1)
            sx = select(k & 1u, 1u - (k & 1u), k >= 2u);
            sy = k & 1u;
        }
        let si = sample_in(ti, i, j, sx, sy, slope, vec2<f32>(dx, -dy), gsd);
        let water = stack_water(si.l);
        // (standing water: the last stage's step 10 only)
        if (water && !last) {
            continue;
        }
        var s = stack_new(si.c, si.t, si.l, pf);
        let sb = PBW_HDR + (lp * PB_RS + k - k0) * ST_F;
        if (stage > 0u && !water) {
            s_frame(&s);
            st_load(&s, sb);
        }
        stack_stage(&s, stage);
        pf = s.pf;
        if (!last) {
            st_save(&s, sb);
            continue;
        }
        // the last stage: the pixel's sums (in `pixb` until it is finished), the samples' classes
        let o = stack_surface(&s);
        if (k == 0u) {
            pixb[pb + 0u] = o.emission.x;
            pixb[pb + 1u] = o.emission.y;
            pixb[pb + 2u] = o.emission.z;
            pixb[pb + 3u] = o.albedo.x;
            pixb[pb + 4u] = o.albedo.y;
            pixb[pb + 5u] = o.albedo.z;
            pixb[pb + 6u] = o.height;
            pixb[pb + 7u] = si.ground;
            pixb[pb + 8u] = o.lit;
        } else {
            pixb[pb + 0u] += o.emission.x;
            pixb[pb + 1u] += o.emission.y;
            pixb[pb + 2u] += o.emission.z;
            pixb[pb + 3u] += o.albedo.x;
            pixb[pb + 4u] += o.albedo.y;
            pixb[pb + 5u] += o.albedo.z;
            pixb[pb + 6u] += o.height;
            pixb[pb + 7u] += si.ground;
            pixb[pb + 8u] += o.lit;
        }
        let cw = ab + k / 4u;
        let csh = (k & 3u) * 8u;
        let cv = min(o.cls, LC_MAX_CLASSES - 1u) << csh;
        if ((k & 3u) == 0u) {
            pbw[cw] = cv;
        } else {
            pbw[cw] |= cv;
        }
        var taken = 0u;
        if (adaptive && k == 0u) {
            first_s = o;
        }
        if (adaptive && k == 1u) {
            let da = abs(first_s.albedo - o.albedo);
            let de = abs(first_s.emission - o.emission);
            let similar = first_s.cls == o.cls && max3(da) < 0.012 && max3(de) < 0.02 && abs(first_s.height - o.height) < 0.15 && abs(first_s.lit - o.lit) < 0.05;
            if (similar) {
                taken = 2u;
            }
        }
        if (k + 1u == n) {
            taken = n;
        }
        if (taken > 0u) {
            pass_b_finish(pb, ab, taken);
            break;
        }
    }
    if (!(last && k1 == n)) {
        pf_save(pfb, pf);
    }
    if (surface_missing != 0u) {
        atomicOr(&counters[C_MISSING], surface_missing);
    }
}

/// A finished pixel: the means of its `taken` samples and their most frequent class (of equals
/// the highest id, as `max_by_key` over the counts).
fn pass_b_finish(pb: u32, ab: u32, taken: u32) {
    pbw[ab + 4u] = 1u;
    var cls = 0u;
    var best = 0u;
    for (var a = 0u; a < taken; a++) {
        let ca = (pbw[ab + a / 4u] >> ((a & 3u) * 8u)) & 0xFFu;
        var cnt = 0u;
        for (var b = 0u; b < taken; b++) {
            if (((pbw[ab + b / 4u] >> ((b & 3u) * 8u)) & 0xFFu) == ca) {
                cnt += 1u;
            }
        }
        if (cnt > best || (cnt == best && ca > cls)) {
            best = cnt;
            cls = ca;
        }
    }
    let inv = 1.0 / f32(taken);
    let e = vec3<f32>(pixb[pb + 0u], pixb[pb + 1u], pixb[pb + 2u]) * inv;
    let a = vec3<f32>(pixb[pb + 3u], pixb[pb + 4u], pixb[pb + 5u]) * inv;
    let h = pixb[pb + 6u] * inv;
    let g = pixb[pb + 7u] * inv;
    pixb[pb + 0u] = e.x;
    pixb[pb + 1u] = e.y;
    pixb[pb + 2u] = e.z;
    pixb[pb + 3u] = a.x;
    pixb[pb + 4u] = a.y;
    pixb[pb + 5u] = a.z;
    pixb[pb + 6u] = h;
    pixb[pb + 7u] = g;
    pixb[pb + 8u] = pixb[pb + 8u] * inv;
    pixb[pb + 9u] = f32(cls);
    pixb[pb + 10u] = max(h - g, 0.0);
}

// ---- canopy clean-up: morphological opening of the height above ground (~1 m radius)

/// The structuring element radius (pixels) of a tile, 0: none.
fn open_radius(ti: TileInfo) -> i32 {
    let g = rows[ti.row_a + 1u + NA / 2u].gsd;
    let r = i32(floor(0.9 / g));
    return min(r, 4);
}

fn pb_index(ti: TileInfo, i: u32, j: u32) -> u32 {
    return ti.pix0 + j * NA2 + i;
}

/// min (`mode` 0) or max (1) along x (`axis` 0) or y (1), from `src` (0: canopy, 1: scratch a,
/// 2: scratch b) into scratch a (`dst` 0) or b (1).
fn open_pass(gid: vec3<u32>, mode: u32, axis: u32, src: u32, dst: u32) {
    let ti = tiles[gid.z];
    let i = gid.x;
    let j = gid.y;
    let r = open_radius(ti);
    if (i >= NA || j >= NA || r < 1) {
        return;
    }
    var v = select(FMAX, -FMAX, mode == 1u);
    for (var k = -r; k <= r; k++) {
        var ii = i32(i);
        var jj = i32(j);
        if (axis == 0u) {
            ii = clamp(ii + k, 0, i32(NA) - 1);
        } else {
            jj = clamp(jj + k, 0, i32(NA) - 1);
        }
        let ix = pb_index(ti, u32(ii), u32(jj));
        var x = 0.0;
        if (src == 0u) {
            x = pixb[ix * PB_F + 10u];
        } else if (src == 1u) {
            x = scratch_a[ix];
        } else {
            x = scratch_b[ix];
        }
        if (mode == 0u) {
            v = min(v, x);
        } else {
            v = max(v, x);
        }
    }
    let o = pb_index(ti, i, j);
    if (dst == 0u) {
        scratch_a[o] = v;
    } else {
        scratch_b[o] = v;
    }
}

@compute @workgroup_size(16, 16)
fn open_min_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    open_pass(gid, 0u, 0u, 0u, 0u);
}
@compute @workgroup_size(16, 16)
fn open_min_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    open_pass(gid, 0u, 1u, 1u, 1u);
}
@compute @workgroup_size(16, 16)
fn open_max_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    open_pass(gid, 1u, 0u, 2u, 0u);
}
@compute @workgroup_size(16, 16)
fn open_max_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    open_pass(gid, 1u, 1u, 1u, 1u);
}

/// Crowns clipped into slivers (narrower than the element) are cut back to the opened canopy.
@compute @workgroup_size(16, 16)
fn open_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = tiles[gid.z];
    let i = gid.x;
    let j = gid.y;
    if (i >= NA || j >= NA || open_radius(ti) < 1) {
        return;
    }
    let ix = pb_index(ti, i, j);
    let opened = scratch_b[ix];
    let canopy = pixb[ix * PB_F + 10u];
    if (opened < canopy - 0.5) {
        pixb[ix * PB_F + 6u] = pixb[ix * PB_F + 7u] + opened;
    }
}

// ---- outputs

fn pack_rgb(c: vec3<f32>) -> u32 {
    let r = u32(round(l2s(c.x) * 255.0));
    let g = u32(round(l2s(c.y) * 255.0));
    let b = u32(round(l2s(c.z) * 255.0));
    return r | (g << 8u) | (b << 16u);
}

fn pack_snorm(v: vec3<f32>) -> u32 {
    let x = u32(i32(round(v.x * 127.0)) & 0xff);
    let y = u32(i32(round(v.y * 127.0)) & 0xff);
    let z = u32(i32(round(v.z * 127.0)) & 0xff);
    return x | (y << 8u) | (z << 16u);
}

/// The output layers per tile pixel: normals from the heights, the satellite look, albedo,
/// night lights, elevation, land cover.
@compute @workgroup_size(16, 16)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = tiles[gid.z];
    let i = gid.x;
    let j = gid.y;
    if (i >= N || j >= N) {
        return;
    }
    let ia = i + 1u;
    let ja = j + 1u;
    let rr = rows[ti.row_a + ja + 1u];
    let g = rr.gsd;
    let gn = rr.gsd_ns;
    let b = pb_index(ti, ia, ja) * PB_F;
    let h = pixb[b + 6u];
    let dhdx = (pixb[pb_index(ti, ia + 1u, ja) * PB_F + 6u] - pixb[pb_index(ti, ia - 1u, ja) * PB_F + 6u]) / (2.0 * g);
    let dhdn = -(pixb[pb_index(ti, ia, ja + 1u) * PB_F + 6u] - pixb[pb_index(ti, ia, ja - 1u) * PB_F + 6u]) / (2.0 * gn);
    let nrm = normalize(vec3<f32>(-dhdx, -dhdn, 1.0));
    let sun = vec3<f32>(cfg.sun_e, cfg.sun_n, cfg.sun_u);
    let lit = pixb[b + 8u];
    let light = cfg.ambient * (0.55 + 0.45 * nrm.z) + cfg.direct * max(dot(nrm, sun), 0.0) * lit;
    let a = vec3<f32>(pixb[b + 3u], pixb[b + 4u], pixb[b + 5u]);
    let lum = 0.2126 * a.x + 0.7152 * a.y + 0.0722 * a.z;
    let alb = max((vec3<f32>(lum) + (a - vec3<f32>(lum)) * cfg.saturation) * cfg.brightness, vec3<f32>(0.0));
    var c = alb * (light / cfg.l0) * cfg.exposure;
    c = c * (1.0 - cfg.haze) + vec3<f32>(0.50, 0.58, 0.70) * cfg.haze;
    let e = vec3<f32>(pixb[b + 0u], pixb[b + 1u], pixb[b + 2u]);
    let ee = pow(clamp(e / 16.0, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 3.0));
    let o = ((gid.z * N + j) * N + i) * OUT_U;
    tile_out[o + 0u] = pack_rgb(c);
    tile_out[o + 1u] = pack_rgb(alb);
    tile_out[o + 2u] = u32(round(ee.x * 255.0)) | (u32(round(ee.y * 255.0)) << 8u) | (u32(round(ee.z * 255.0)) << 16u);
    tile_out[o + 3u] = pack_snorm(nrm);
    tile_out[o + 4u] = bitcast<u32>(h);
    tile_out[o + 5u] = u32(pixb[b + 9u]);
    atomicMax(&ranges[gid.z * 2u], ~orderable(h));
    atomicMax(&ranges[gid.z * 2u + 1u], orderable(h));
}
