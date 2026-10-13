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

@compute @workgroup_size(64)
fn eval_points(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
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
    let ctx = ctx_make(q.p.xyz, q.sl, q.cl, q.so, q.co, q.lat, q.gsd);
    let m = macro_at(ctx.p, ctx.gsd);
    let pre = pre_none();
    let rl = relief(ctx, m, pre);
    pout[i] = terrain_rest(ctx, m, pre, rl, q.mode, q.dr);
}
