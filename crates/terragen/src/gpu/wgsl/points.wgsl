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

@group(2) @binding(0) var<storage, read> pts: array<PointIn>;
@group(2) @binding(1) var<storage, read_write> pout: array<Terrain>;

@compute @workgroup_size(64)
fn eval_points(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&pts)) {
        return;
    }
    let q = pts[i];
    let ctx = ctx_make(q.p.xyz, q.sl, q.cl, q.so, q.co, q.lat, q.gsd);
    let m = macro_at(ctx.p, ctx.gsd);
    let pre = pre_none();
    let rl = relief(ctx, m, pre);
    pout[i] = terrain_rest(ctx, m, pre, rl, q.mode, q.dr);
}
