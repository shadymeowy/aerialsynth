// The drainage network on the GPU (`hydro.rs`): the jittered lattice points of every level in a
// hash table that persists across batches (key: level and cell), their heights (relief mode),
// flow targets (steepest descent), sources, and the channel pieces a query (a disc around a
// point) can reach, in the CPU's order (cells sorted by z, y, x).

const LAT_EMPTY: u64 = 0lu;
const LAT_NONE: u32 = 0xffffffffu;
const LAT_OFF: i32 = 524288;

const LF_ACTIVE: u32 = 1u;
const LF_HEIGHT: u32 = 2u;
const LF_TGT: u32 = 4u;
const LF_SRC: u32 = 8u;
const LF_IS_SRC: u32 = 16u;

/// A box of lattice cells of one level for one query: what the enumeration or the gather
/// scans, in chunks of `CHUNK` cells (linear index over the box in z, y, x order), with the
/// shell of possibly active cells (`World::shell_cells`).
struct DBox {
    /// query centre
    center: vec4<f64>,
    /// squared radii of the shell, cell size, reach (gather)
    r_lo2: f64,
    r_hi2: f64,
    cell: f64,
    reach: f64,
    lo: vec4<i32>,
    dims: vec4<u32>,
    level: u32,
    /// dominant axis of the centre, its sign (1 / -1)
    ax: u32,
    sign: i32,
    query: u32,
    /// first chunk of the box in the chunk list, radius of the query (keep), -
    chunk0: u32,
    radius: f32,
    _p0: u32,
    _p1: u32,
}

const CHUNK: u32 = 4096u;

@group(2) @binding(0) var<storage, read_write> lat_keys: array<atomic<u64>>;
@group(2) @binding(1) var<storage, read_write> lat_s: array<vec4<f64>>;
@group(2) @binding(2) var<storage, read_write> lat_h: array<f32>;
@group(2) @binding(3) var<storage, read_write> lat_flags: array<u32>;
@group(2) @binding(4) var<storage, read_write> lat_tgt: array<u32>;
@group(2) @binding(5) var<storage, read> boxes: array<DBox>;
/// per chunk: its box
@group(2) @binding(6) var<storage, read> chunk_box: array<u32>;
@group(2) @binding(7) var<storage, read_write> new_list: array<u32>;
/// [0] new points, [1] probes overflowed, [2] incomplete gathers, [3] pieces written, [4] sink
/// pieces
@group(2) @binding(8) var<storage, read_write> dcount: array<atomic<u32>>;
@group(2) @binding(9) var<storage, read_write> out_segs: array<Seg>;
/// per chunk: pieces kept (count pass), then its first output index
@group(2) @binding(10) var<storage, read_write> chunk_n: array<u32>;
/// per query: first piece, count
@group(2) @binding(11) var<storage, read_write> q_range: array<vec2<u32>>;
/// sink pieces: (query, output index), piece
@group(2) @binding(12) var<storage, read_write> sink_out: array<SinkPiece>;
/// per query: its boxes [first, count] (gather)
@group(2) @binding(13) var<storage, read> q_boxes: array<vec2<u32>>;
/// a copy of the keys after the enumeration (plain loads for the lookups)
@group(2) @binding(14) var<storage, read> lat_keys_ro: array<u64>;
/// per slot: queued for a target (1), a source (2) in this batch
@group(2) @binding(15) var<storage, read_write> lat_mark: array<atomic<u32>>;
/// the points to compute targets / sources for ([5] / [6] of `dcount` entries)
@group(2) @binding(16) var<storage, read_write> work_t: array<u32>;
@group(2) @binding(17) var<storage, read_write> work_s: array<u32>;

struct SinkPiece {
    b: vec4<f64>,
    hw: f32,
    query: u32,
    index: u32,
    _p: u32,
}

fn lat_key(lvl: u32, c: vec3<i32>) -> u64 {
    return ((u64(lvl) + 1lu) << 60u) | (u64(u32(c.x + LAT_OFF)) << 40u) | (u64(u32(c.y + LAT_OFF)) << 20u) | u64(u32(c.z + LAT_OFF));
}

fn lat_cell(key: u64) -> vec3<i32> {
    return vec3<i32>(i32(u32((key >> 40u) & 0xFFFFFlu)) - LAT_OFF, i32(u32((key >> 20u) & 0xFFFFFlu)) - LAT_OFF, i32(u32(key & 0xFFFFFlu)) - LAT_OFF);
}

fn lat_level(key: u64) -> u32 {
    return u32(key >> 60u) - 1u;
}

/// Slot of a lattice point in the table, LAT_NONE if absent.
fn lat_find(key: u64) -> u32 {
    let mask = arrayLength(&lat_flags) - 1u;
    var k = u32(mix64(key)) & mask;
    for (var i = 0u; i < 64u; i++) {
        let v = lat_keys_ro[k];
        if (v == key) {
            return k;
        }
        if (v == LAT_EMPTY) {
            return LAT_NONE;
        }
        k = (k + 1u) & mask;
    }
    return LAT_NONE;
}

fn lvl_cell(lvl: u32) -> f64 {
    return f64(cfg.lvl_a[lvl].x);
}

/// The jittered lattice point of `c` (m).
fn lat_point(lvl: u32, c: vec3<i32>) -> vec3<f64> {
    let hh = hash3(level_key(lvl), i64(c.x), i64(c.y), i64(c.z));
    let j = vec3<f64>(u01kd(hh, 1lu), u01kd(hh, 2lu), u01kd(hh, 3lu));
    return (vec3<f64>(f64(c.x), f64(c.y), f64(c.z)) + 0.5lf + 0.8lf * (j - 0.5lf)) * lvl_cell(lvl);
}

/// False only for lattice points certainly not active (`World::maybe_active`).
fn lat_maybe_active(lvl: u32, c: vec3<i32>) -> bool {
    let p = lat_point(lvl, c);
    let r = sqrt(dot(p, p));
    let a = cfg.ell_a;
    let b = cfg.ell_b;
    let sz = p.z / max(r, 1.0lf);
    let cz2 = 1.0lf - sz * sz;
    let r_gc = a * b / sqrt(b * b * cz2 + a * a * sz * sz);
    return abs(r - r_gc) < 0.5lf * lvl_cell(lvl) + 200.0lf;
}

/// Is the cell in the box's shell? (the cells `World::shell_cells` lists)
fn in_shell(bx: DBox, c: vec3<i32>) -> bool {
    var ci = array<i32, 3>(c.x, c.y, c.z);
    var lo = array<i32, 3>(bx.lo.x, bx.lo.y, bx.lo.z);
    var dims = array<u32, 3>(bx.dims.x, bx.dims.y, bx.dims.z);
    let ax = bx.ax;
    let ua = (ax + 1u) % 3u;
    let va = (ax + 2u) % 3u;
    let cell = bx.cell;
    let iu = f64(ci[ua]);
    let iv = f64(ci[va]);
    let u0 = iu * cell;
    let u1 = (iu + 1.0lf) * cell;
    let v0 = iv * cell;
    let v1 = (iv + 1.0lf) * cell;
    let u2hi = max(u0 * u0, u1 * u1);
    var u2lo = min(u0 * u0, u1 * u1);
    if (u0 <= 0.0lf && u1 >= 0.0lf) {
        u2lo = 0.0lf;
    }
    let v2hi = max(v0 * v0, v1 * v1);
    var v2lo = min(v0 * v0, v1 * v1);
    if (v0 <= 0.0lf && v1 >= 0.0lf) {
        v2lo = 0.0lf;
    }
    let dmax2 = bx.r_hi2 - u2lo - v2lo;
    if (dmax2 <= 0.0lf) {
        return false;
    }
    let dmin = sqrt(max(bx.r_lo2 - u2hi - v2hi, 0.0lf));
    let dmax = sqrt(dmax2);
    var a0 = dmin;
    var a1 = dmax;
    if (bx.sign < 0) {
        a0 = -dmax;
        a1 = -dmin;
    }
    let k0 = max(i32(floor(a0 / cell)), lo[ax]);
    let k1 = min(i32(floor(a1 / cell)), lo[ax] + i32(dims[ax]) - 1);
    return ci[ax] >= k0 && ci[ax] <= k1;
}

/// Can the cell hold a lattice point near the surface? (a conservative f32 test on the cell
/// centre's distance from the Earth's centre: a filter only, `lat_maybe_active` decides)
fn near_shell(bx: DBox, c: vec3<i32>) -> bool {
    let cell = f32(bx.cell);
    let m = (vec3<f32>(c) + 0.5) * cell;
    let r = length(m);
    let slack = 0.87 * cell + 1000.0;
    return r >= sqrt(f32(bx.r_lo2)) - slack && r <= sqrt(f32(bx.r_hi2)) + slack;
}

/// Cell `i` (linear, z, y, x order) of the box.
fn box_cell(bx: DBox, i: u32) -> vec3<i32> {
    let x = i % bx.dims.x;
    let y = (i / bx.dims.x) % bx.dims.y;
    let z = i / (bx.dims.x * bx.dims.y);
    return bx.lo.xyz + vec3<i32>(i32(x), i32(y), i32(z));
}

fn box_size(bx: DBox) -> u32 {
    return bx.dims.x * bx.dims.y * bx.dims.z;
}

// ---------------------------------------------------------------- enumeration

/// Insert the possibly active lattice points of the boxes (one workgroup per chunk).
@compute @workgroup_size(256)
fn lat_enum(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    let chunk = wg.x + wg.y * nwg.x;
    if (chunk >= arrayLength(&chunk_box)) {
        return;
    }
    let bx = boxes[chunk_box[chunk]];
    let start = (chunk - bx.chunk0) * CHUNK;
    let n = box_size(bx);
    let mask = arrayLength(&lat_flags) - 1u;
    for (var k = li; k < CHUNK; k += 256u) {
        let i = start + k;
        if (i >= n) {
            break;
        }
        let c = box_cell(bx, i);
        if (!near_shell(bx, c) || surely_inactive(bx.level, c) || !lat_maybe_active(bx.level, c)) {
            continue;
        }
        let key = lat_key(bx.level, c);
        var s = u32(mix64(key)) & mask;
        var done = false;
        for (var probe = 0u; probe < 64u; probe++) {
            let r = atomicCompareExchangeWeak(&lat_keys[s], LAT_EMPTY, key);
            if (r.exchanged) {
                let j = atomicAdd(&dcount[0], 1u);
                if (j < arrayLength(&new_list)) {
                    new_list[j] = s;
                }
                done = true;
                break;
            }
            if (r.old_value == key) {
                done = true;
                break;
            }
            if (r.old_value == LAT_EMPTY) {
                // spurious failure: try the slot again
                probe -= 1u;
                continue;
            }
            s = (s + 1u) & mask;
        }
        if (!done) {
            atomicStore(&dcount[1], 1u);
        }
    }
}

// ---------------------------------------------------------------- heights

/// Geodetic latitude (sin, cos), longitude (sin, cos) and height of an ECEF point, by
/// fixed-point iteration (f64 without trig).
struct Geo {
    sl: f64,
    cl: f64,
    so: f64,
    co: f64,
    h: f64,
}

fn ecef_geo(p: vec3<f64>) -> Geo {
    let a = cfg.ell_a;
    let b = cfg.ell_b;
    let e2 = 1.0lf - (b * b) / (a * a);
    let pxy = sqrt(p.x * p.x + p.y * p.y);
    var g: Geo;
    if (pxy < 1e-3lf) {
        g.so = 0.0lf;
        g.co = 1.0lf;
        g.cl = 0.0lf;
        g.sl = select(-1.0lf, 1.0lf, p.z >= 0.0lf);
        g.h = abs(p.z) - b;
        return g;
    }
    g.co = p.x / pxy;
    g.so = p.y / pxy;
    var t = p.z / (pxy * (1.0lf - e2));
    for (var i = 0; i < 6; i++) {
        let s = t / sqrt(1.0lf + t * t);
        let n = a / sqrt(1.0lf - e2 * s * s);
        t = (p.z + e2 * n * s) / pxy;
    }
    g.cl = 1.0lf / sqrt(1.0lf + t * t);
    g.sl = t * g.cl;
    let n = a / sqrt(1.0lf - e2 * g.sl * g.sl);
    if (g.cl > 0.1lf) {
        g.h = pxy / g.cl - n;
    } else {
        g.h = p.z / g.sl - n * (1.0lf - e2);
    }
    return g;
}

/// Geometry and height (relief mode) of the new lattice points.
@compute @workgroup_size(64)
fn lat_heights(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= min(atomicLoad(&dcount[0]), arrayLength(&new_list))) {
        return;
    }
    let s = new_list[i];
    let key = lat_keys_ro[s];
    let lvl = lat_level(key);
    let c = lat_cell(key);
    let p = lat_point(lvl, c);
    let g = ecef_geo(p);
    let cell = lvl_cell(lvl);
    if (!(abs(g.h) < 0.5lf * cell)) {
        lat_flags[s] = LF_HEIGHT;
        return;
    }
    // the surface point below / above (h = 0)
    let a = cfg.ell_a;
    let b = cfg.ell_b;
    let e2 = 1.0lf - (b * b) / (a * a);
    let n = a / sqrt(1.0lf - e2 * g.sl * g.sl);
    let b2a2 = (b / a) * (b / a);
    let p0 = vec3<f64>(n * g.cl * g.co, n * g.cl * g.so, (n * b2a2) * g.sl);
    let lat = atan2(f32(g.sl), f32(g.cl));
    let ctx = ctx_make(p0, f32(g.sl), f32(g.cl), f32(g.so), f32(g.co), lat, f32(cell / 4.0lf));
    let m = macro_at(ctx.p, ctx.gsd);
    let rl = relief(ctx, m, pre_none());
    lat_s[s] = vec4<f64>(p0, 0.0lf);
    lat_h[s] = rl.h;
    lat_flags[s] = LF_HEIGHT | LF_ACTIVE;
}

// ---------------------------------------------------------------- targets, sources

/// Is no point of the cell anywhere near the surface? (a cheap f32 test with a wide margin:
/// the cell's point lies within 0.9 cells of its centre per axis)
fn surely_inactive(lvl: u32, c: vec3<i32>) -> bool {
    let cell = cfg.lvl_a[lvl].x;
    let m = (vec3<f32>(c) + 0.5) * cell;
    let r = length(m);
    let a = f32(cfg.ell_a);
    let b = f32(cfg.ell_b);
    let sz = m.z / max(r, 1.0);
    let r_gc = a * b / sqrt(b * b * (1.0 - sz * sz) + a * a * sz * sz);
    return abs(r - r_gc) > 2.1 * cell + 300.0;
}

/// A neighbour: 0 inactive, 1 active (slot in `slot`), 2 unknown (not evaluated yet).
fn neighbour(lvl: u32, c: vec3<i32>, slot: ptr<function, u32>) -> u32 {
    if (surely_inactive(lvl, c)) {
        return 0u;
    }
    let s = lat_find(lat_key(lvl, c));
    if (s == LAT_NONE) {
        if (lat_maybe_active(lvl, c)) {
            return 2u;
        }
        return 0u;
    }
    let f = lat_flags[s];
    if ((f & LF_HEIGHT) == 0u) {
        return 2u;
    }
    *slot = s;
    return select(0u, 1u, (f & LF_ACTIVE) != 0u);
}

/// The slot of the active lattice point at chunk cell `k` (LAT_NONE: none).
fn chunk_slot(gid: vec3<u32>, wg: vec3<u32>, li: u32, nwg: vec3<u32>, k: u32, extra: f32) -> u32 {
    let chunk = wg.x + wg.y * nwg.x;
    if (chunk >= arrayLength(&chunk_box)) {
        return LAT_NONE;
    }
    let bx = boxes[chunk_box[chunk]];
    let i = (chunk - bx.chunk0) * CHUNK + k;
    if (i >= box_size(bx)) {
        return LAT_NONE;
    }
    let c = box_cell(bx, i);
    if (!near_shell(bx, c) || surely_inactive(bx.level, c) || !lat_maybe_active(bx.level, c)) {
        return LAT_NONE;
    }
    let s = lat_find(lat_key(bx.level, c));
    if (s == LAT_NONE || (lat_flags[s] & LF_ACTIVE) == 0u) {
        return LAT_NONE;
    }
    // (only the points within reach of the query, plus `extra` cells)
    if (f64(dist64(lat_s[s].xyz, bx.center.xyz)) > bx.reach + f64(extra) * bx.cell) {
        return LAT_NONE;
    }
    return s;
}

/// Queue the active points of the boxes that lack a downstream neighbour (each point once).
@compute @workgroup_size(256)
fn mark_targets(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = li; k < CHUNK; k += 256u) {
        // (the nodes' 2-cell neighbourhoods: within 2·√3 + 2·0.8·√3 cells)
        let s = chunk_slot(gid, wg, li, nwg, k, 6.2);
        if (s != LAT_NONE && (lat_flags[s] & LF_TGT) == 0u && (atomicOr(&lat_mark[s], 1u) & 1u) == 0u) {
            let j = atomicAdd(&dcount[5], 1u);
            if (j < arrayLength(&work_t)) {
                work_t[j] = s;
            }
        }
    }
}

/// Downstream neighbours (steepest descent) of the queued points (`World::flow_target`).
@compute @workgroup_size(64)
fn lat_targets(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i < min(atomicLoad(&dcount[5]), arrayLength(&work_t))) {
        lat_target(work_t[i]);
    }
}

fn lat_target(s: u32) {
    let f = lat_flags[s];
    let key = lat_keys_ro[s];
    let lvl = lat_level(key);
    let c = lat_cell(key);
    let cell = f32(lvl_cell(lvl));
    let me_s = lat_s[s].xyz;
    let me_h = lat_h[s];
    var best = LAT_NONE;
    if (me_h > -150.0) {
        var best_slope = 0.0;
        for (var dz = -1; dz <= 1; dz++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    var o = 0u;
                    let st = neighbour(lvl, c + vec3<i32>(dx, dy, dz), &o);
                    if (st == 2u) {
                        return;
                    }
                    if (st == 0u) {
                        continue;
                    }
                    let d = dist64(lat_s[o].xyz, me_s);
                    if (d < 0.2 * cell || d > 1.8 * cell) {
                        continue;
                    }
                    let slope = (lat_h[o] - me_h) / d;
                    if (slope < best_slope) {
                        best_slope = slope;
                        best = o;
                    }
                }
            }
        }
        // no lower neighbour: look a little farther for an outlet
        if (best == LAT_NONE) {
            for (var dz = -2; dz <= 2; dz++) {
                for (var dy = -2; dy <= 2; dy++) {
                    for (var dx = -2; dx <= 2; dx++) {
                        if (max(abs(dx), max(abs(dy), abs(dz))) < 2) {
                            continue;
                        }
                        var o = 0u;
                        let st = neighbour(lvl, c + vec3<i32>(dx, dy, dz), &o);
                        if (st == 2u) {
                            return;
                        }
                        if (st == 0u) {
                            continue;
                        }
                        let d = dist64(lat_s[o].xyz, me_s);
                        if (d > 3.0 * cell) {
                            continue;
                        }
                        let slope = (lat_h[o] - me_h) / d;
                        if (slope < best_slope) {
                            best_slope = slope;
                            best = o;
                        }
                    }
                }
            }
        }
    }
    lat_tgt[s] = best;
    lat_flags[s] = f | LF_TGT;
}


/// Queue the active points of the boxes that are not known to be sources or not (each once).
@compute @workgroup_size(256)
fn mark_sources(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var k = li; k < CHUNK; k += 256u) {
        let s = chunk_slot(gid, wg, li, nwg, k, 0.0);
        if (s != LAT_NONE && (lat_flags[s] & LF_SRC) == 0u && (atomicOr(&lat_mark[s], 2u) & 2u) == 0u) {
            let j = atomicAdd(&dcount[6], 1u);
            if (j < arrayLength(&work_s)) {
                work_s[j] = s;
            }
        }
    }
}

/// Are the queued points sources (nothing drains into them)? (`World::is_source`)
@compute @workgroup_size(64)
fn lat_sources(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i < min(atomicLoad(&dcount[6]), arrayLength(&work_s))) {
        let s = work_s[i];
        if ((lat_flags[s] & (LF_TGT | LF_SRC)) == LF_TGT) {
            lat_source(s);
        }
    }
}

fn lat_source(s: u32) {
    let f = lat_flags[s];
    let key = lat_keys_ro[s];
    let lvl = lat_level(key);
    let c = lat_cell(key);
    for (var dz = -2; dz <= 2; dz++) {
        for (var dy = -2; dy <= 2; dy++) {
            for (var dx = -2; dx <= 2; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                var o = 0u;
                let st = neighbour(lvl, c + vec3<i32>(dx, dy, dz), &o);
                if (st == 2u) {
                    return;
                }
                if (st == 0u) {
                    continue;
                }
                let fo = lat_flags[o];
                if ((fo & LF_TGT) == 0u) {
                    return;
                }
                if (lat_tgt[o] == s) {
                    lat_flags[s] = f | LF_SRC;
                    return;
                }
            }
        }
    }
    lat_flags[s] = f | LF_SRC | LF_IS_SRC;
}

// ---------------------------------------------------------------- gather

fn seg_make(a: vec3<f64>, b: vec3<f64>, ha: f32, hb: f32, hw: f32, hw_b: f32, valley: f32, lvl: u32, sink: bool) -> Seg {
    var sg: Seg;
    sg.a = vec4<f64>(a, 0.0lf);
    sg.b = vec4<f64>(b, 0.0lf);
    sg.ha = ha;
    sg.hb = hb;
    sg.hw = hw;
    sg.valley = valley;
    sg.hw_b = hw_b;
    sg.level = lvl;
    sg._p0 = select(0u, 1u, sink);
    sg._p1 = 0u;
    return sg;
}

/// (half width, valley half width) of the channel leaving node `c`.
fn lat_width(lvl: u32, c: vec3<i32>) -> vec2<f32> {
    let lc = cfg.lvl_a[lvl];
    let hh = hash3(level_key(lvl) ^ 0x51DElu, i64(c.x), i64(c.y), i64(c.z));
    return vec2<f32>(0.5 * (lc.y + (lc.z - lc.y) * u01k(hh, 1lu)), lc.w * (0.6 + 0.8 * u01k(hh, 2lu)));
}

/// The pieces of the node at box cell `i` that the query keeps: count, and (with `write`)
/// written from `base`. Sets the incomplete counter when data is missing.
fn node_pieces(bx: DBox, i: u32, write: bool, base: u32) -> u32 {
    let c = box_cell(bx, i);
    let lvl = bx.level;
    if (!near_shell(bx, c) || surely_inactive(lvl, c) || !in_shell(bx, c) || !lat_maybe_active(lvl, c)) {
        return 0u;
    }
    let s = lat_find(lat_key(lvl, c));
    if (s == LAT_NONE) {
        atomicAdd(&dcount[2], 1u);
        return 0u;
    }
    let f = lat_flags[s];
    if ((f & LF_HEIGHT) == 0u) {
        atomicAdd(&dcount[2], 1u);
        return 0u;
    }
    if ((f & LF_ACTIVE) == 0u) {
        return 0u;
    }
    let fs = lat_s[s].xyz;
    if (f64(dist64(fs, bx.center.xyz)) > bx.reach) {
        return 0u;
    }
    if ((f & (LF_TGT | LF_SRC)) != (LF_TGT | LF_SRC)) {
        atomicAdd(&dcount[2], 1u);
        return 0u;
    }
    let t = lat_tgt[s];
    if (t == LAT_NONE) {
        return 0u;
    }
    let ft = lat_flags[t];
    if ((ft & LF_TGT) == 0u) {
        atomicAdd(&dcount[2], 1u);
        return 0u;
    }
    let lc = cfg.lvl_a[lvl];
    let lb = cfg.lvl_b[lvl];
    let cellf = f32(bx.cell);
    let keep = bx.radius + 0.4 * cellf + 1.4 * lc.w + 0.35 * lb.y * cellf;
    let wv = lat_width(lvl, c);
    let hw = wv.x;
    let valley = wv.y;
    let fh = lat_h[s];
    let ts = lat_s[t].xyz;
    let th = lat_h[t];
    let mid = 0.5lf * (fs + ts);
    let hmid = 0.5 * (fh + th);
    let ctr = bx.center.xyz;
    var n = 0u;
    if ((f & LF_IS_SRC) != 0u) {
        let sg = seg_make(fs, mid, fh, hmid, 0.08 * hw, hw, valley, lvl, false);
        if (seg_near(sg, ctr, keep)) {
            if (write) {
                out_segs[base + n] = sg;
            }
            n += 1u;
        }
    }
    let tt = lat_tgt[t];
    if (tt != LAT_NONE) {
        let qs = lat_s[tt].xyz;
        let qh = lat_h[tt];
        let hw2 = lat_width(lvl, lat_cell(lat_keys_ro[t])).x;
        let mid2 = 0.5lf * (ts + qs);
        let hmid2 = 0.5 * (th + qh);
        for (var k = 0u; k < 6u; k++) {
            let t0 = f64(k) / 6.0lf;
            let t1 = f64(k + 1u) / 6.0lf;
            let pa = mid * ((1.0lf - t0) * (1.0lf - t0)) + ts * (2.0lf * t0 * (1.0lf - t0)) + mid2 * (t0 * t0);
            let pb = mid * ((1.0lf - t1) * (1.0lf - t1)) + ts * (2.0lf * t1 * (1.0lf - t1)) + mid2 * (t1 * t1);
            let f0 = f32(t0);
            let f1 = f32(t1);
            let ha = hmid * (1.0 - f0) * (1.0 - f0) + th * 2.0 * f0 * (1.0 - f0) + hmid2 * f0 * f0;
            let hb = hmid * (1.0 - f1) * (1.0 - f1) + th * 2.0 * f1 * (1.0 - f1) + hmid2 * f1 * f1;
            let sg = seg_make(pa, pb, ha, hb, hw + (hw2 - hw) * f0, hw + (hw2 - hw) * f1, valley, lvl, false);
            if (seg_near(sg, ctr, keep)) {
                if (write) {
                    out_segs[base + n] = sg;
                }
                n += 1u;
            }
        }
    } else {
        let sg = seg_make(mid, ts, hmid, th, hw, hw, valley, lvl, th > 0.0);
        if (seg_near(sg, ctr, keep)) {
            if (write) {
                out_segs[base + n] = sg;
                if (th > 0.0) {
                    let j = atomicAdd(&dcount[4], 1u);
                    if (j < arrayLength(&sink_out)) {
                        sink_out[j].b = vec4<f64>(ts, 0.0lf);
                        sink_out[j].hw = hw;
                        sink_out[j].query = bx.query;
                        sink_out[j].index = base + n;
                    }
                }
            }
            n += 1u;
        }
    }
    return n;
}

var<workgroup> wg_n: array<u32, 256>;
var<workgroup> wg_total: u32;

/// Count the pieces each chunk keeps.
@compute @workgroup_size(256)
fn gather_count(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    let chunk = wg.x + wg.y * nwg.x;
    let valid = chunk < arrayLength(&chunk_box);
    var n = 0u;
    if (valid) {
        let bx = boxes[chunk_box[chunk]];
        let start = (chunk - bx.chunk0) * CHUNK;
        let size = box_size(bx);
        for (var k = li; k < CHUNK; k += 256u) {
            let i = start + k;
            if (i >= size) {
                break;
            }
            n += node_pieces(bx, i, false, 0u);
        }
    }
    wg_n[li] = n;
    workgroupBarrier();
    if (li == 0u && valid) {
        var t = 0u;
        for (var k = 0u; k < 256u; k++) {
            t += wg_n[k];
        }
        chunk_n[chunk] = t;
    }
}

/// Per query: the output range, and each of its chunks' first output index.
@compute @workgroup_size(64)
fn gather_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= arrayLength(&q_boxes)) {
        return;
    }
    let qb = q_boxes[q];
    var total = 0u;
    for (var b = qb.x; b < qb.x + qb.y; b++) {
        let bx = boxes[b];
        let nch = (box_size(bx) + CHUNK - 1u) / CHUNK;
        for (var c = bx.chunk0; c < bx.chunk0 + nch; c++) {
            total += chunk_n[c];
        }
    }
    let base = atomicAdd(&dcount[3], total);
    q_range[q] = vec2<u32>(base, total);
    var off = base;
    for (var b = qb.x; b < qb.x + qb.y; b++) {
        let bx = boxes[b];
        let nch = (box_size(bx) + CHUNK - 1u) / CHUNK;
        for (var c = bx.chunk0; c < bx.chunk0 + nch; c++) {
            let n = chunk_n[c];
            chunk_n[c] = off;
            off += n;
        }
    }
}

/// Write the pieces each chunk keeps, in order.
@compute @workgroup_size(256)
fn gather_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    let chunk = wg.x + wg.y * nwg.x;
    if (chunk >= arrayLength(&chunk_box)) {
        return;
    }
    let bx = boxes[chunk_box[chunk]];
    let start = (chunk - bx.chunk0) * CHUNK;
    let size = box_size(bx);
    var off = chunk_n[chunk];
    let cap = arrayLength(&out_segs);
    for (var k0 = 0u; k0 < CHUNK; k0 += 256u) {
        let i = start + k0 + li;
        var n = 0u;
        if (k0 + li < CHUNK && i < size) {
            n = node_pieces(bx, i, false, 0u);
        }
        wg_n[li] = n;
        workgroupBarrier();
        // exclusive prefix sum (Hillis-Steele, inclusive then shift)
        for (var d = 1u; d < 256u; d *= 2u) {
            var v = wg_n[li];
            if (li >= d) {
                v += wg_n[li - d];
            }
            workgroupBarrier();
            wg_n[li] = v;
            workgroupBarrier();
        }
        let incl = wg_n[li];
        if (n > 0u && off + incl <= cap) {
            node_pieces(bx, i, true, off + incl - n);
        }
        let total = wg_n[255];
        workgroupBarrier();
        off += total;
        if (start + k0 + 256u >= size) {
            break;
        }
    }
}
