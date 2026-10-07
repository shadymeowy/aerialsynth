// Event sensor pixel pass (port of EventSensor::step in events.rs): one invocation per pixel and
// sensor step. The radiance of the step is interpolated between the two keyframes resident on the
// GPU (each: radiance, flicker cos, flicker sin; 3 floats per pixel each), as in
// FrameOut::radiance_at. The pixel state stays on the GPU; events are appended to `ev` as
// (pixel | polarity << 31, crossing time as a fraction of the step, step index in the batch).
// Event times are kept relative to the step start, so f32 is exact enough; the time since the
// pixel's last event (`ago`, for the refractory period) is relative to the step start too.
// Random numbers come from a counter-based hash per (step, pixel): statistically equivalent to
// the CPU's per-row streams, not bit-identical.

struct P {
    n: u32,
    mode: u32,      // 0: initialise the state from this image, 1: step
    step: u32,      // index in the batch
    seed: u32,
    seed2: u32,
    split0: u32,    // key has the flicker split
    split1: u32,
    _p: u32,
    a: f32,         // interpolation weight of key 1
    c: f32,         // flicker cos / sin at the step time
    s: f32,
    dt: f32,
    gain: f32,
    eps: f32,
    cut_hz: f32,
    cut_half: f32,
    cut_min: f32,
    a_hpf: f32,
    refr: f32,
    shot_hz: f32,
    noise_ref: f32,
    dark_lo: f32,
    dark_hi: f32,
    _q: f32,
};

struct St {
    lp: f32,
    lref: f32,
    ago: f32,
    _p: f32,
};

struct Ev {
    count: atomic<u32>,
    d: array<u32>,
};

@group(0) @binding(0) var<uniform> pr: P;
@group(0) @binding(1) var<storage, read_write> st: array<St>;
// per pixel: contrast ON, contrast OFF, leak rate (Hz)
@group(0) @binding(2) var<storage, read> cst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> k0: array<f32>;
@group(0) @binding(4) var<storage, read> k1: array<f32>;
@group(0) @binding(5) var<storage, read_write> ev: Ev;

fn hash(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

var<private> rs: u32;

fn uniform01() -> f32 {
    rs = hash(rs + 0x9e3779b9u);
    return f32(rs >> 8u) * (1.0 / 16777216.0);
}

fn rad(k: u32, i: u32) -> vec3<f32> {
    let n3 = pr.n * 3u;
    let j = i * 3u;
    if k == 0u {
        var r = vec3<f32>(k0[j], k0[j + 1u], k0[j + 2u]);
        if pr.split0 != 0u {
            r += pr.c * vec3<f32>(k0[n3 + j], k0[n3 + j + 1u], k0[n3 + j + 2u]) + pr.s * vec3<f32>(k0[2u * n3 + j], k0[2u * n3 + j + 1u], k0[2u * n3 + j + 2u]);
        }
        return r;
    }
    var r = vec3<f32>(k1[j], k1[j + 1u], k1[j + 2u]);
    if pr.split1 != 0u {
        r += pr.c * vec3<f32>(k1[n3 + j], k1[n3 + j + 1u], k1[n3 + j + 2u]) + pr.s * vec3<f32>(k1[2u * n3 + j], k1[2u * n3 + j + 1u], k1[2u * n3 + j + 2u]);
    }
    return r;
}

// 1 - exp(-x) without cancellation for small x
fn one_minus_exp(x: f32) -> f32 {
    if x < 1e-3 {
        return x * (1.0 - 0.5 * x);
    }
    return 1.0 - exp(-x);
}

fn emit(i: u32, up: bool, a: f32) {
    let k = atomicAdd(&ev.count, 1u);
    if (k + 1u) * 3u <= arrayLength(&ev.d) {
        ev.d[k * 3u] = i | select(0u, 0x80000000u, up);
        ev.d[k * 3u + 1u] = bitcast<u32>(a);
        ev.d[k * 3u + 2u] = pr.step;
    }
}

const TAU: f32 = 6.2831853;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if i >= pr.n {
        return;
    }
    var r = rad(0u, i);
    if pr.a != 0.0 {
        r += pr.a * (rad(1u, i) - r);
    }
    let lin = log(pr.gain * (0.2126 * r.x + 0.7152 * r.y + 0.0722 * r.z) + pr.eps);
    var p = st[i];
    if pr.mode == 0u {
        st[i] = St(lin, lin, 3.0e38, 0.0);
        return;
    }
    rs = hash(pr.seed ^ hash(i ^ hash(pr.seed2)));
    let c = cst[i];
    let dt = pr.dt;
    // photoreceptor low-pass: cutoff grows with the photocurrent (luminance)
    let lum = max((exp(lin) - pr.eps) / pr.gain, 0.0);
    let fc = max(pr.cut_hz * lum / (lum + pr.cut_half), pr.cut_min);
    let alpha = one_minus_exp(TAU * fc * dt);
    let lp0 = p.lp;
    let lp1 = lp0 + alpha * (lin - lp0);
    p.lp = lp1;
    // high-pass: reference relaxes towards the signal; leak: reference drifts down
    if pr.a_hpf > 0.0 {
        p.lref += pr.a_hpf * (lp0 - p.lref);
    }
    p.lref -= c.z * dt * c.x;
    let dl = lp1 - lp0;
    for (var it = 0u; it < 4096u; it++) {
        let up = lp1 - p.lref >= c.x;
        let down = p.lref - lp1 >= c.y;
        if !up && !down {
            break;
        }
        let tgt = select(p.lref - c.y, p.lref + c.x, up);
        var a: f32;
        if abs(dl) > 1e-9 {
            a = clamp((tgt - lp0) / dl, 0.0, 1.0);
        } else {
            a = uniform01();
        }
        let te = a * dt;
        p.lref = tgt;
        if te + p.ago >= pr.refr {
            p.ago = -te;
            emit(i, up, a);
        }
    }
    // shot-noise background activity, stronger in the dark
    if pr.shot_hz > 0.0 {
        let dark = clamp(sqrt(pr.noise_ref / (lum + 1e-6)), pr.dark_lo, pr.dark_hi);
        if uniform01() < pr.shot_hz * dark * dt {
            let a = uniform01();
            let te = a * dt;
            if te + p.ago >= pr.refr {
                p.ago = -te;
                emit(i, (hash(rs) & 1u) == 1u, a);
            }
        }
    }
    p.ago = min(p.ago + dt, 3.0e38);
    st[i] = p;
}
