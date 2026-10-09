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

const C_REGION_REQ: u32 = 3u;
const C_TOWN_REQ: u32 = 4u;
const C_MISSING: u32 = 5u;

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

/// One sub-sample (sx, sy) of pass-B pixel (i, j): its surface and bare ground.
struct Sample {
    s: Surface,
    ground: f32,
}

fn sample_b(ti: TileInfo, i: u32, j: u32, sx: u32, sy: u32, slope: f32, gsd: f32, pf: ptr<function, PixFields>) -> Sample {
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
    l.slope = slope;
    l.fw = gsd / f32(ss);
    let r = rows[ti.row_b + j * ss + sy];
    let c = cols[ti.col_b + i * ss + sx];
    let ctx = row_col_ctx(r, c, gsd);
    var o: Sample;
    o.s = surface_eval(ctx, t, l, pf);
    o.ground = ground;
    return o;
}

/// Pass B per pixel of the tile and its 1-pixel apron: the supersampled surface.
@compute @workgroup_size(16, 16)
fn pass_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = tiles[gid.z];
    let i = gid.x;
    let j = gid.y;
    if (i >= NA || j >= NA) {
        return;
    }
    surface_missing = 0u;
    let ss = ti.ss;
    // the pixel centre's row (= pass-A row j + 1)
    let rc = rows[ti.row_a + j + 1u];
    let gsd = rc.gsd;
    let ctx = row_col_ctx(rc, cols[ti.col_a + i + 1u], gsd);
    var pf = pixel_fields_b(ti, i, j, ctx.p, gsd);
    // slope of the bare ground at pixel scale
    let ii = i32(i);
    let jj = i32(j);
    let dx = (at(ti, ii + 1, jj).ground - at(ti, ii - 1, jj).ground) / (2.0 * gsd);
    let dy = (at(ti, ii, jj + 1).ground - at(ti, ii, jj - 1).ground) / (2.0 * rc.gsd_ns);
    let slope = sqrt(dx * dx + dy * dy);
    var acc_a = vec3<f32>(0.0);
    var acc_e = vec3<f32>(0.0);
    var acc_h = 0.0;
    var acc_l = 0.0;
    var acc_g = 0.0;
    var counts: array<u32, 32>;
    var taken = ss * ss;
    // One call site of `sample_b` (the driver inlines the whole surface model at every call:
    // five of them took minutes to compile). Adaptive (ss = 2): the diagonal pair first, the
    // other two only where it disagrees.
    let adaptive = (cfg.flags & CF_ADAPTIVE) != 0u;
    var first_s: Surface;
    for (var k = 0u; k < ss * ss; k++) {
        var sx = k % ss;
        var sy = k / ss;
        if (adaptive) {
            // (0, 0), (1, 1), (1, 0), (0, 1)
            sx = select(k & 1u, 1u - (k & 1u), k >= 2u);
            sy = k & 1u;
        }
        let s = sample_b(ti, i, j, sx, sy, slope, gsd, &pf);
        acc_a += s.s.albedo;
        acc_e += s.s.emission;
        acc_h += s.s.height;
        acc_l += s.s.lit;
        acc_g += s.ground;
        counts[min(s.s.cls, 31u)] += 1u;
        if (adaptive && k == 0u) {
            first_s = s.s;
        }
        if (adaptive && k == 1u) {
            let da = abs(first_s.albedo - s.s.albedo);
            let de = abs(first_s.emission - s.s.emission);
            let similar = first_s.cls == s.s.cls && max3(da) < 0.012 && max3(de) < 0.02 && abs(first_s.height - s.s.height) < 0.15 && abs(first_s.lit - s.s.lit) < 0.05;
            if (similar) {
                taken = 2u;
                break;
            }
        }
    }
    let inv = 1.0 / f32(taken);
    // the most frequent cls (the last of equals, as `max_by_key`)
    var cls = 0u;
    var best = 0u;
    for (var k = 0u; k < 32u; k++) {
        if (counts[k] >= best && counts[k] > 0u) {
            best = counts[k];
            cls = k;
        }
    }
    let b = (ti.pix0 + j * NA2 + i) * PB_F;
    let e = acc_e * inv;
    let a = acc_a * inv;
    let h = acc_h * inv;
    let g = acc_g * inv;
    pixb[b + 0u] = e.x;
    pixb[b + 1u] = e.y;
    pixb[b + 2u] = e.z;
    pixb[b + 3u] = a.x;
    pixb[b + 4u] = a.y;
    pixb[b + 5u] = a.z;
    pixb[b + 6u] = h;
    pixb[b + 7u] = g;
    pixb[b + 8u] = acc_l * inv;
    pixb[b + 9u] = f32(cls);
    pixb[b + 10u] = max(h - g, 0.0);
    if (surface_missing != 0u) {
        atomicOr(&counters[C_MISSING], surface_missing);
    }
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
