// Pass A at single points (exact macro fields): drainage lattice heights (relief mode), lake
// basins (no lakes) and the sites of regions and towns (full).

struct PointIn {
    p: vec4<f64>,
    sl: f32,
    cl: f32,
    so: f32,
    co: f32,
    lat: f32,
    gsd: f32,
    mode: u32,
    _p: u32,
    dr: Drain,
    _q: vec4<u32>,
}

/// mode: report the sink pieces the point keeps (for its sink lakes), no evaluation
const MODE_REPORT: u32 = 3u;

/// A sink piece a point keeps: end point, half width; point, order.
struct SinkRep {
    b: vec4<f64>,
    hw: f32,
    point: u32,
    order: u32,
    _p: u32,
}

@group(2) @binding(0) var<storage, read> pts: array<PointIn>;
@group(2) @binding(1) var<storage, read_write> pout: array<Terrain>;
@group(2) @binding(2) var<storage, read_write> reps: array<SinkRep>;
@group(2) @binding(3) var<storage, read_write> rep_n: array<atomic<u32>>;

/// A point's macro fields and relief, from `eval_points` to `eval_points_kits` and
/// `eval_points_rest` (three pipelines: the core relief, the kits' relief operators and the rest
/// of pass A each compile in a fraction of the time of all three).
struct PointMid {
    m: Macro,
    r: Relief,
}
@group(2) @binding(4) var<storage, read_write> pmid: array<PointMid>;

fn point_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 64u;
}

fn point_ctx(q: PointIn) -> Ctx {
    return ctx_make(q.p.xyz, q.sl, q.cl, q.so, q.co, q.lat, q.gsd);
}

/// The macro fields and the relief of a point before the kits' operators.
@compute @workgroup_size(64)
fn eval_points(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = point_index(gid, nwg);
    if (i >= arrayLength(&pts)) {
        return;
    }
    let q = pts[i];
    if (q.mode == MODE_REPORT) {
        return;
    }
    let ctx = point_ctx(q);
    let m = macro_at(ctx.p, ctx.gsd);
    var o: PointMid;
    o.m = m;
    o.r = relief_core(ctx, m, pre_none());
    pmid[i] = o;
}

/// The kits' relief operators at a point (after `eval_points`).
@compute @workgroup_size(64)
fn eval_points_kits(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = point_index(gid, nwg);
    if (i >= arrayLength(&pts)) {
        return;
    }
    let q = pts[i];
    if (q.mode == MODE_REPORT) {
        return;
    }
    let mid = pmid[i];
    pmid[i].r = relief_kits(point_ctx(q), mid.m, mid.r, 0xffffffffu);
}

/// The rest of pass A at a point (after `eval_points`), or its sink pieces (`MODE_REPORT`).
@compute @workgroup_size(64)
fn eval_points_rest(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = point_index(gid, nwg);
    if (i >= arrayLength(&pts)) {
        return;
    }
    let q = pts[i];
    if (q.mode == MODE_REPORT) {
        var order = 0u;
        for (var k = 0u; k < q.dr.nseg; k++) {
            let si = drain_seg(q.dr, k, q.p.xyz);
            if (si == 0xffffffffu || segs[si]._p0 == 0u) {
                continue;
            }
            let j = atomicAdd(&rep_n[0], 1u);
            if (j < arrayLength(&reps)) {
                reps[j].b = segs[si].b;
                reps[j].hw = segs[si].hw;
                reps[j].point = i;
                reps[j].order = order;
            }
            order += 1u;
        }
        return;
    }
    let mid = pmid[i];
    pout[i] = terrain_rest(point_ctx(q), mid.m, pre_none(), mid.r, q.mode, q.dr);
}
