// Noise primitives: the same functions as `noise.rs`, with the same hashes (u64) and lattice
// coordinates in f64 (ECEF metres over wavelengths down to decimetres); everything inside a
// lattice cell is f32.

const K1: u64 = 0x9E3779B97F4A7C15lu;
const K2: u64 = 0xC2B2AE3D27D4EB4Flu;
const K3: u64 = 0x165667B19E3779F9lu;
const KU: u64 = 0xD6E8FEB86659FD93lu;
const TAU: f32 = 6.283185307179586;
const PI: f32 = 3.141592653589793;
const FMAX: f32 = 3.0e38;

/// One octave of a noise: random rotation (columns), offset, seed, wavelength and amplitude.
struct Oct {
    ax: vec4<f64>,
    ay: vec4<f64>,
    az: vec4<f64>,
    off: vec4<f64>,
    lam: f64,
    inv_lam: f64,
    seed: u64,
    amp: f32,
    _p: f32,
}

/// An fBm: its octaves `octs[first .. first + n]`.
struct FbmD {
    first: u32,
    n: u32,
    norm: f32,
    wavelength: f32,
}

@group(0) @binding(1) var<storage, read> grads: array<vec4<f32>, 256>;
@group(0) @binding(2) var<storage, read> octs: array<Oct>;
@group(0) @binding(3) var<storage, read> fbms: array<FbmD>;

fn mix64(h0: u64) -> u64 {
    var h = h0;
    h = h ^ (h >> 33u);
    h = h * 0xff51afd7ed558ccdlu;
    h = h ^ (h >> 33u);
    h = h * 0xc4ceb9fe1a85ec53lu;
    h = h ^ (h >> 33u);
    return h;
}

fn hash1(seed: u64, a: i64) -> u64 {
    return mix64(seed ^ (bitcast<u64>(a) * K1));
}
fn hash2(seed: u64, a: i64, b: i64) -> u64 {
    return mix64(hash1(seed, a) ^ (bitcast<u64>(b) * K2));
}
fn hash3(seed: u64, a: i64, b: i64, c: i64) -> u64 {
    return mix64(hash2(seed, a, b) ^ (bitcast<u64>(c) * K3));
}

/// Uniform in [0, 1) from a hash (the top 24 of the CPU's 53 bits).
fn u01(h: u64) -> f32 {
    return f32(u32(h >> 40u)) * (1.0 / 16777216.0);
}
/// The same with all 53 bits.
fn u01d(h: u64) -> f64 {
    return f64(h >> 11u) * (1.0lf / 9007199254740992.0lf);
}
fn u01k(h: u64, k: u64) -> f32 {
    return u01(mix64(h ^ (k * KU)));
}
fn u01kd(h: u64, k: u64) -> f64 {
    return u01d(mix64(h ^ (k * KU)));
}

fn smoothstep1(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}
fn saturate1(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}
/// Band-limiting weight of a feature of wavelength `lam` sampled with pixel size `gsd`.
fn band(lam: f32, gsd: f32) -> f32 {
    return smoothstep1(2.0 * gsd, 4.0 * gsd, lam);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
fn dfade(t: f32) -> f32 {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

fn grad_at(h: u64) -> vec3<f32> {
    return grads[u32(mix64(h) >> 56u)].xyz;
}

/// 3D gradient noise (value), roughly in [-1, 1].
fn perlin3(seed: u64, p: vec3<f64>) -> f32 {
    let pf = floor(p);
    let ix = i64(pf.x);
    let iy = i64(pf.y);
    let iz = i64(pf.z);
    let f = vec3<f32>(p - pf);
    let u = fade(f.x);
    let v = fade(f.y);
    let w = fade(f.z);
    let hx0 = hash1(seed, ix);
    let hx1 = hash1(seed, ix + 1li);
    let hy0 = bitcast<u64>(iy) * K2;
    let hy1 = bitcast<u64>(iy + 1li) * K2;
    let hz0 = bitcast<u64>(iz) * K3;
    let hz1 = bitcast<u64>(iz + 1li) * K3;
    let h00 = mix64(hx0 ^ hy0);
    let h10 = mix64(hx1 ^ hy0);
    let h01 = mix64(hx0 ^ hy1);
    let h11 = mix64(hx1 ^ hy1);
    let g = f - vec3<f32>(1.0);
    let a = dot(grad_at(h00 ^ hz0), vec3<f32>(f.x, f.y, f.z));
    let b = dot(grad_at(h10 ^ hz0), vec3<f32>(g.x, f.y, f.z));
    let c = dot(grad_at(h01 ^ hz0), vec3<f32>(f.x, g.y, f.z));
    let d = dot(grad_at(h11 ^ hz0), vec3<f32>(g.x, g.y, f.z));
    let e = dot(grad_at(h00 ^ hz1), vec3<f32>(f.x, f.y, g.z));
    let ff = dot(grad_at(h10 ^ hz1), vec3<f32>(g.x, f.y, g.z));
    let gg = dot(grad_at(h01 ^ hz1), vec3<f32>(f.x, g.y, g.z));
    let h = dot(grad_at(h11 ^ hz1), vec3<f32>(g.x, g.y, g.z));
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + gg;
    let k6 = a - b - e + ff;
    let k7 = -a + b + c - d + e - ff - gg + h;
    let n = a + k1 * u + k2 * v + k3 * w + k4 * u * v + k5 * v * w + k6 * w * u + k7 * u * v * w;
    return n * 1.1;
}

/// 3D gradient noise with its analytic gradient: (value, d/dx, d/dy, d/dz).
fn perlin3_d(seed: u64, p: vec3<f64>) -> vec4<f32> {
    let pf = floor(p);
    let ix = i64(pf.x);
    let iy = i64(pf.y);
    let iz = i64(pf.z);
    let f = vec3<f32>(p - pf);
    let u = fade(f.x);
    let v = fade(f.y);
    let w = fade(f.z);
    let du = dfade(f.x);
    let dv = dfade(f.y);
    let dw = dfade(f.z);
    let hx0 = hash1(seed, ix);
    let hx1 = hash1(seed, ix + 1li);
    let hy0 = bitcast<u64>(iy) * K2;
    let hy1 = bitcast<u64>(iy + 1li) * K2;
    let hz0 = bitcast<u64>(iz) * K3;
    let hz1 = bitcast<u64>(iz + 1li) * K3;
    let h00 = mix64(hx0 ^ hy0);
    let h10 = mix64(hx1 ^ hy0);
    let h01 = mix64(hx0 ^ hy1);
    let h11 = mix64(hx1 ^ hy1);
    let g = f - vec3<f32>(1.0);
    let ga = grad_at(h00 ^ hz0);
    let gb = grad_at(h10 ^ hz0);
    let gc = grad_at(h01 ^ hz0);
    let gd = grad_at(h11 ^ hz0);
    let ge = grad_at(h00 ^ hz1);
    let gf = grad_at(h10 ^ hz1);
    let gg = grad_at(h01 ^ hz1);
    let gh = grad_at(h11 ^ hz1);
    let a = dot(ga, vec3<f32>(f.x, f.y, f.z));
    let b = dot(gb, vec3<f32>(g.x, f.y, f.z));
    let c = dot(gc, vec3<f32>(f.x, g.y, f.z));
    let d = dot(gd, vec3<f32>(g.x, g.y, f.z));
    let e = dot(ge, vec3<f32>(f.x, f.y, g.z));
    let ff = dot(gf, vec3<f32>(g.x, f.y, g.z));
    let ggv = dot(gg, vec3<f32>(f.x, g.y, g.z));
    let h = dot(gh, vec3<f32>(g.x, g.y, g.z));
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + ggv;
    let k6 = a - b - e + ff;
    let k7 = -a + b + c - d + e - ff - ggv + h;
    let n = a + k1 * u + k2 * v + k3 * w + k4 * u * v + k5 * v * w + k6 * w * u + k7 * u * v * w;
    let gk1 = gb - ga;
    let gk2 = gc - ga;
    let gk3 = ge - ga;
    let gk4 = ga - gb - gc + gd;
    let gk5 = ga - gc - ge + gg;
    let gk6 = ga - gb - ge + gf;
    let gk7 = -ga + gb + gc - gd + ge - gf - gg + gh;
    let gi = ga + gk1 * u + gk2 * v + gk3 * w + gk4 * (u * v) + gk5 * (v * w) + gk6 * (w * u) + gk7 * (u * v * w);
    let dn = vec3<f32>(
        du * (k1 + k4 * v + k6 * w + k7 * v * w),
        dv * (k2 + k5 * w + k4 * u + k7 * w * u),
        dw * (k3 + k6 * u + k5 * v + k7 * u * v),
    );
    return vec4<f32>(n, gi + dn) * 1.1;
}

/// Lattice coordinate of `p` (m) in an octave's frame.
fn oct_q(o: Oct, p: vec3<f64>) -> vec3<f64> {
    let s = p * o.inv_lam;
    return o.ax.xyz * s.x + o.ay.xyz * s.y + o.az.xyz * s.z + o.off.xyz;
}
/// The octave's rotation applied transposed to a gradient (back to world axes).
fn oct_rt(o: Oct, g: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(vec3<f32>(o.ax.xyz), g), dot(vec3<f32>(o.ay.xyz), g), dot(vec3<f32>(o.az.xyz), g));
}

fn fbm_norm(f: u32) -> f32 {
    return fbms[f].norm;
}
fn fbm_wavelength(f: u32) -> f32 {
    return fbms[f].wavelength;
}

/// fBm value, band-limited by `gsd` (`Fbm::eval`).
fn fbm(f: u32, p: vec3<f64>, gsd: f32) -> f32 {
    let d = fbms[f];
    var sum = 0.0;
    for (var i = 0u; i < d.n; i++) {
        let o = octs[d.first + i];
        let w = band(f32(o.lam), gsd);
        if (w <= 0.0) {
            break;
        }
        sum += o.amp * w * perlin3(o.seed, oct_q(o, p));
    }
    return sum;
}

/// The octaves of wavelength >= `cut` (`low`) or < `cut` (not `low`) (`Fbm::eval_part`).
fn fbm_part(f: u32, p: vec3<f64>, gsd: f32, cut: f32, low: bool) -> f32 {
    let d = fbms[f];
    var sum = 0.0;
    for (var i = 0u; i < d.n; i++) {
        let o = octs[d.first + i];
        let lam = f32(o.lam);
        if (low && lam < cut) {
            break;
        }
        let w = band(lam, gsd);
        if (w <= 0.0) {
            break;
        }
        if (low || lam < cut) {
            sum += o.amp * w * perlin3(o.seed, oct_q(o, p));
        }
    }
    return sum;
}

/// fBm with its gradient (per metre): (value, gradient) (`Fbm::eval_d`, all octaves).
fn fbm_d(f: u32, p: vec3<f64>, gsd: f32) -> vec4<f32> {
    let d = fbms[f];
    var sum = 0.0;
    var grad = vec3<f32>(0.0);
    for (var i = 0u; i < d.n; i++) {
        let o = octs[d.first + i];
        let lam = f32(o.lam);
        let w = band(lam, gsd);
        if (w <= 0.0) {
            break;
        }
        let ng = perlin3_d(o.seed, oct_q(o, p));
        sum += o.amp * w * ng.x;
        grad += oct_rt(o, ng.yzw) * (o.amp * w / lam);
    }
    return vec4<f32>(sum, grad);
}

/// Octaves of wavelength >= `min_lambda` only (`Fbm::eval_lowpass`).
fn fbm_lowpass(f: u32, p: vec3<f64>, gsd: f32, min_lambda: f32) -> f32 {
    let d = fbms[f];
    var sum = 0.0;
    for (var i = 0u; i < d.n; i++) {
        let o = octs[d.first + i];
        let lam = f32(o.lam);
        let w = band(lam, gsd) * smoothstep1(min_lambda * 0.5, min_lambda, lam);
        if (w <= 0.0) {
            break;
        }
        sum += o.amp * w * perlin3(o.seed, oct_q(o, p));
    }
    return sum;
}

// ---------------------------------------------------------------- cellular noise

/// The two nearest feature points of a jittered 3D lattice: ids and points (lattice units).
struct Sites {
    id0: u64,
    id1: u64,
    p0: vec3<f64>,
    p1: vec3<f64>,
}

/// Result of a 3D cellular query: nearest / second nearest point (m) and distances (m).
struct Cell3 {
    id: u64,
    id2: u64,
    point: vec3<f64>,
    point2: vec3<f64>,
    f1: f32,
    f2: f32,
}

fn nb_gap(f: f32, d: i32, half: f32) -> f32 {
    if (d == -1) {
        return max(f + 0.5 - half, 0.0);
    }
    if (d == 1) {
        return max(1.5 - half - f, 0.0);
    }
    return 0.0;
}

/// Feature point of lattice cell `c` (lattice units), its id in `id`.
fn worley3_site(seed: u64, c: vec3<i64>, jitter: f32) -> Sites {
    let h = hash3(seed, c.x, c.y, c.z);
    var s: Sites;
    s.id0 = h;
    s.p0 = vec3<f64>(f64(c.x), f64(c.y), f64(c.z)) + vec3<f64>(vec3<f32>(0.5) + jitter * (vec3<f32>(u01k(h, 1lu), u01k(h, 2lu), u01k(h, 3lu)) - 0.5));
    return s;
}

/// The two nearest sites of the jittered 3D lattice of cell size 1/`inv_cell` (`worley3_sites`).
fn worley3_sites(seed: u64, p: vec3<f64>, inv_cell: f64, jitter: f32) -> Sites {
    let q = p * inv_cell;
    let qf = floor(q);
    let iq = vec3<i64>(i64(qf.x), i64(qf.y), i64(qf.z));
    let fq = vec3<f32>(q - qf);
    let half = 0.5 * abs(jitter);
    var best: Sites;
    var f1 = FMAX;
    var f2 = FMAX;
    // the centre cell, then faces, edges and corners: the nearest candidates first, so the
    // distance bound skips most of the far ones
    for (var ring = 0; ring <= 3; ring++) {
        for (var dz = -1; dz <= 1; dz++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    if (i32(dx != 0) + i32(dy != 0) + i32(dz != 0) != ring) {
                        continue;
                    }
                    let gx = nb_gap(fq.x, dx, half);
                    let gy = nb_gap(fq.y, dy, half);
                    let gz = nb_gap(fq.z, dz, half);
                    if (gx * gx + gy * gy + gz * gz > f2 + 1e-6) {
                        continue;
                    }
                    let c = iq + vec3<i64>(i64(dx), i64(dy), i64(dz));
                    let h = hash3(seed, c.x, c.y, c.z);
                    let jit = jitter * (vec3<f32>(u01k(h, 1lu), u01k(h, 2lu), u01k(h, 3lu)) - 0.5);
                    // relative to the query's cell: exact small numbers
                    let rel = vec3<f32>(f32(dx), f32(dy), f32(dz)) + 0.5 + jit - fq;
                    let d = dot(rel, rel);
                    let fp = vec3<f64>(f64(c.x), f64(c.y), f64(c.z)) + vec3<f64>(vec3<f32>(0.5) + jit);
                    if (d < f1) {
                        f2 = f1;
                        best.id1 = best.id0;
                        best.p1 = best.p0;
                        f1 = d;
                        best.id0 = h;
                        best.p0 = fp;
                    } else if (d < f2) {
                        f2 = d;
                        best.id1 = h;
                        best.p1 = fp;
                    }
                }
            }
        }
    }
    return best;
}

/// The cell at `p` given its two nearest sites (either order) (`worley3_from`).
fn worley3_from(p: vec3<f64>, cell: f64, inv_cell: f64, s: Sites) -> Cell3 {
    let q = p * inv_cell;
    let da = vec3<f32>(s.p0 - q);
    let db = vec3<f32>(s.p1 - q);
    let fa = dot(da, da);
    let fb = dot(db, db);
    var c: Cell3;
    if (fb < fa) {
        c.id = s.id1;
        c.id2 = s.id0;
        c.point = s.p1 * cell;
        c.point2 = s.p0 * cell;
        c.f1 = sqrt(fb) * f32(cell);
        c.f2 = sqrt(fa) * f32(cell);
    } else {
        c.id = s.id0;
        c.id2 = s.id1;
        c.point = s.p0 * cell;
        c.point2 = s.p1 * cell;
        c.f1 = sqrt(fa) * f32(cell);
        c.f2 = sqrt(fb) * f32(cell);
    }
    return c;
}

fn worley3(seed: u64, p: vec3<f64>, cell: f64, jitter: f32) -> Cell3 {
    let inv = 1.0lf / cell;
    return worley3_from(p, cell, inv, worley3_sites(seed, p, inv, jitter));
}

/// Distance (m) from `q` to the bisector between the two nearest points.
fn worley_edge_dist(c: Cell3, q: vec3<f64>) -> f32 {
    let d = vec3<f32>(c.point2 - c.point);
    let len = length(d);
    if (len < 1e-6) {
        return FMAX;
    }
    let mid = vec3<f32>((c.point + c.point2) * 0.5lf - q);
    return abs(dot(mid, d / len));
}

/// 2D cellular query in a local planar frame (m).
struct Cell2 {
    id: u64,
    id2: u64,
    point: vec2<f32>,
    point2: vec2<f32>,
    f1: f32,
    f2: f32,
}

fn worley2(seed: u64, p: vec2<f32>, cell: f32, jitter: f32) -> Cell2 {
    let q = p / cell;
    let qf = floor(q);
    let ix = i64(qf.x);
    let iy = i64(qf.y);
    let fq = q - qf;
    let half = 0.5 * abs(jitter);
    var best: Cell2;
    best.f1 = FMAX;
    best.f2 = FMAX;
    for (var ring = 0; ring <= 2; ring++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (i32(dx != 0) + i32(dy != 0) != ring) {
                    continue;
                }
                let gx = nb_gap(fq.x, dx, half);
                let gy = nb_gap(fq.y, dy, half);
                if (gx * gx + gy * gy > best.f2 + 1e-6) {
                    continue;
                }
                let h = hash2(seed, ix + i64(dx), iy + i64(dy));
                let rel = vec2<f32>(f32(dx), f32(dy)) + 0.5 + jitter * (vec2<f32>(u01k(h, 1lu), u01k(h, 2lu)) - 0.5);
                let fp = qf + rel;
                let r = rel - fq;
                let d = dot(r, r);
                if (d < best.f1) {
                    best.f2 = best.f1;
                    best.id2 = best.id;
                    best.point2 = best.point;
                    best.f1 = d;
                    best.id = h;
                    best.point = fp;
                } else if (d < best.f2) {
                    best.f2 = d;
                    best.id2 = h;
                    best.point2 = fp;
                }
            }
        }
    }
    best.f1 = sqrt(best.f1) * cell;
    best.f2 = sqrt(best.f2) * cell;
    best.point *= cell;
    best.point2 *= cell;
    return best;
}

fn worley2_edge_dist(c: Cell2, q: vec2<f32>) -> f32 {
    let d = c.point2 - c.point;
    let len = length(d);
    if (len < 1e-6) {
        return FMAX;
    }
    let mid = (c.point + c.point2) * 0.5;
    return abs(dot(mid - q, d / len));
}

/// One octave of gradient-aligned gully noise in a jittered 3D lattice (`World::gully_octave`):
/// (stripe value, derivative in lattice units).
fn gully_octave(seed: u64, q: vec3<f64>, dir: vec3<f32>) -> vec4<f32> {
    let qf = floor(q);
    let ix = i64(qf.x);
    let iy = i64(qf.y);
    let iz = i64(qf.z);
    let f = vec3<f32>(q - qf);
    var v = 0.0;
    var d = vec3<f32>(0.0);
    var wt = 0.0;
    let R2 = 2.25 + 1e-6;
    for (var dz = -2; dz <= 2; dz++) {
        let gz = gully_gap(f.z, dz);
        if (gz > R2) {
            continue;
        }
        for (var dy = -2; dy <= 2; dy++) {
            let gzy = gz + gully_gap(f.y, dy);
            if (gzy > R2) {
                continue;
            }
            for (var dx = -2; dx <= 2; dx++) {
                if (gzy + gully_gap(f.x, dx) > R2) {
                    continue;
                }
                let h = hash3(seed, ix + i64(dx), iy + i64(dy), iz + i64(dz));
                let jit = vec3<f32>(u01k(h, 1lu), u01k(h, 2lu), u01k(h, 3lu)) * 0.5;
                let pp = f - vec3<f32>(f32(dx), f32(dy), f32(dz)) - jit;
                let d2 = dot(pp, pp);
                if (d2 >= 2.25) {
                    continue;
                }
                let k = 1.0 - d2 / 2.25;
                let w = k * k * k;
                wt += w;
                let mag = dot(pp, dir) * TAU;
                v += cos(mag) * w;
                d -= dir * (sin(mag) * w);
            }
        }
    }
    return vec4<f32>(v, d) / wt;
}

fn gully_gap(f: f32, o: i32) -> f32 {
    let t = f - f32(o);
    if (t < 0.0) {
        return t * t;
    }
    if (t > 0.5) {
        return (t - 0.5) * (t - 0.5);
    }
    return 0.0;
}
