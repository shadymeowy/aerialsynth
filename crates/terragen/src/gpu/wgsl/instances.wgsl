// Instance families (`instances.rs`): sites of a 3D jittered lattice per family, each existing
// by an analytic test at its centre. In every module (relief operators use them).

const INST_K: u32 = 8u;

/// An instance site (`InstSite`).
struct InstSite {
    id: u64,
    center: vec3<f64>,
    up: vec3<f32>,
    east: vec3<f32>,
    north: vec3<f32>,
    lat: f32,
    lon: f32,
}

/// A family's existence test result: parameters when the instance exists.
struct InstOut {
    ok: bool,
    v: array<f32, 8>,
}

struct Inst {
    id: u64,
    center: vec3<f64>,
    v: array<f32, 8>,
}

/// The instances of a family near a point, by id (≤ INST_K).
struct InstList {
    n: u32,
    items: array<Inst, 8>,
}

/// The site of lattice cell `c` (`instances::site`); id 0: none there.
fn inst_site(key: u64, c: vec3<i64>, cell: f64, jitter: f32) -> InstSite {
    var s: InstSite;
    let h = hash3(cfg.seed ^ key, c.x, c.y, c.z);
    let j = f64(jitter);
    let pt = (vec3<f64>(f64(c.x), f64(c.y), f64(c.z)) + 0.5lf + j * (vec3<f64>(u01kd(h, 1lu), u01kd(h, 2lu), u01kd(h, 3lu)) - 0.5lf)) * cell;
    let a = cfg.ell_a;
    let b = cfg.ell_b;
    let len = max(sqrt(dot(pt, pt)), 1.0lf);
    let d = pt / len;
    let rho2 = d.x * d.x + d.y * d.y;
    let re = a * b / sqrt(b * b * rho2 + a * a * d.z * d.z);
    s.id = 0lu;
    if (abs(len - re) >= 0.5lf * cell) {
        return s;
    }
    s.id = h;
    let center = d * re;
    s.center = center;
    let e2 = 1.0lf - (b * b) / (a * a);
    let wxy = sqrt(center.x * center.x + center.y * center.y);
    let lat = atan2(f32(center.z), f32((1.0lf - e2) * wxy));
    let lon = atan2(f32(center.y), f32(center.x));
    s.lat = lat;
    s.lon = lon;
    let sl = sin(lat);
    let cl = cos(lat);
    let so = sin(lon);
    let co = cos(lon);
    s.up = vec3<f32>(cl * co, cl * so, sl);
    s.east = vec3<f32>(-so, co, 0.0);
    s.north = vec3<f32>(-sl * co, -sl * so, cl);
    return s;
}

/// The instances of family `f` that can reach the disc of radius `r` around `p`
/// (`instances::near`), the first INST_K by id; `*over`: more were there.
fn inst_scan_n(f: u32, p: vec3<f64>, r: f32, over: ptr<function, bool>) -> InstList {
    var l: InstList;
    l.n = 0u;
    let cell = fam_cell(f);
    let rr = f64(r + fam_reach(f));
    let ext = rr + 0.5lf * cell;
    let lo = floor((p - vec3<f64>(ext)) / cell);
    let hi = floor((p + vec3<f64>(ext)) / cell);
    let key = fam_key(f);
    let jit = fam_jitter(f);
    for (var z = i64(lo.z); z <= i64(hi.z); z++) {
        for (var y = i64(lo.y); y <= i64(hi.y); y++) {
            for (var x = i64(lo.x); x <= i64(hi.x); x++) {
                let s = inst_site(key, vec3<i64>(x, y, z), cell, jit);
                if (s.id == 0lu || dist64(s.center, p) > f32(rr)) {
                    continue;
                }
                let e = family_exists(f, s);
                if (!e.ok) {
                    continue;
                }
                // insert by id (the list stays sorted; the largest falls off when full)
                var k = l.n;
                if (k == INST_K) {
                    *over = true;
                    if (s.id > l.items[INST_K - 1u].id) {
                        continue;
                    }
                    k = INST_K - 1u;
                } else {
                    l.n += 1u;
                }
                while (k > 0u && l.items[k - 1u].id > s.id) {
                    l.items[k] = l.items[k - 1u];
                    k -= 1u;
                }
                var it: Inst;
                it.id = s.id;
                it.center = s.center;
                it.v = e.v;
                l.items[k] = it;
            }
        }
    }
    return l;
}

fn inst_scan(f: u32, p: vec3<f64>, r: f32) -> InstList {
    var over = false;
    return inst_scan_n(f, p, r, &over);
}
