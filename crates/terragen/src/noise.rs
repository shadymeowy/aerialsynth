//! Deterministic f64 noise primitives.
//!
//! Everything is a pure function of (seed, position). Lattices are hashed (no permutation tables),
//! so the domain is effectively unbounded — we evaluate at ECEF meters (|p| ~ 6.4e6) divided by a
//! wavelength, which keeps full f64 precision down to sub-millimetre features.

use glam::{DMat3, DVec2, DVec3};

#[inline(always)]
pub fn mix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

#[inline(always)]
pub fn hash1(seed: u64, a: i64) -> u64 {
    mix64(seed ^ (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}
#[inline(always)]
pub fn hash2(seed: u64, a: i64, b: i64) -> u64 {
    mix64(hash1(seed, a) ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
}
#[inline(always)]
pub fn hash3(seed: u64, a: i64, b: i64, c: i64) -> u64 {
    mix64(hash2(seed, a, b) ^ (c as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
}

/// Uniform in [0,1) from a hash.
#[inline(always)]
pub fn u01(h: u64) -> f64 {
    (h >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}
/// Uniform in [0,1) from a hash, sub-stream `k` (for drawing several numbers from one cell).
#[inline(always)]
pub fn u01k(h: u64, k: u64) -> f64 {
    u01(mix64(h ^ k.wrapping_mul(0xD6E8_FEB8_6659_FD93)))
}

/// 256 pseudo-random unit gradients (fibonacci sphere, shuffled), shared by all noise instances.
static GRADS: once_cell_grads::Grads = once_cell_grads::Grads::new();

mod once_cell_grads {
    use std::sync::OnceLock;
    pub struct Grads(OnceLock<[[f64; 3]; 256]>);
    impl Grads {
        pub const fn new() -> Self {
            Grads(OnceLock::new())
        }
        #[inline(always)]
        pub fn get(&self) -> &[[f64; 3]; 256] {
            self.0.get_or_init(|| {
                let mut g = [[0.0; 3]; 256];
                let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
                for (i, gi) in g.iter_mut().enumerate() {
                    let y = 1.0 - (i as f64 + 0.5) / 128.0;
                    let r = (1.0 - y * y).sqrt();
                    let th = golden * i as f64;
                    *gi = [r * th.cos(), y, r * th.sin()];
                }
                // deterministic shuffle so neighbouring indices are uncorrelated
                let mut s = 0x1234_5678_9abc_def0u64;
                for i in (1..256).rev() {
                    s = super::mix64(s.wrapping_add(i as u64));
                    let j = (s % (i as u64 + 1)) as usize;
                    g.swap(i, j);
                }
                g
            })
        }
    }
}

#[inline(always)]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
#[inline(always)]
fn dfade(t: f64) -> f64 {
    30.0 * t * t * (t * (t - 2.0) + 1.0)
}

/// 3D gradient (Perlin) noise with analytic gradient. Output roughly in [-1, 1].
#[inline]
pub fn perlin3_d(seed: u64, p: DVec3) -> (f64, DVec3) {
    let g = GRADS.get();
    let pf = p.floor();
    let (ix, iy, iz) = (pf.x as i64, pf.y as i64, pf.z as i64);
    let f = p - pf;
    let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
    let (du, dv, dw) = (dfade(f.x), dfade(f.y), dfade(f.z));

    let hx0 = hash1(seed, ix);
    let hx1 = hash1(seed, ix + 1);
    let mut val = [0.0f64; 8];
    let mut grd = [[0.0f64; 3]; 8];
    let mut k = 0;
    for dz in 0..2i64 {
        for dy in 0..2i64 {
            for (dx, hx) in [(0i64, hx0), (1, hx1)] {
                let h = mix64(mix64(hx ^ ((iy + dy) as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
                    ^ ((iz + dz) as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
                let gv = g[(h >> 56) as usize];
                let d = [f.x - dx as f64, f.y - dy as f64, f.z - dz as f64];
                val[k] = gv[0] * d[0] + gv[1] * d[1] + gv[2] * d[2];
                grd[k] = gv;
                k += 1;
            }
        }
    }
    // corners order: (x,y,z): 000 100 010 110 001 101 011 111
    let (a, b, c, d, e, ff, gg, h) = (val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
    let k0 = a;
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + gg;
    let k6 = a - b - e + ff;
    let k7 = -a + b + c - d + e - ff - gg + h;
    let n = k0 + k1 * u + k2 * v + k3 * w + k4 * u * v + k5 * v * w + k6 * w * u + k7 * u * v * w;

    // gradient: interpolated gradients + derivative of the interpolation weights
    let lerp_g = |i: usize| DVec3::new(grd[i][0], grd[i][1], grd[i][2]);
    let ga = lerp_g(0);
    let gk1 = lerp_g(1) - ga;
    let gk2 = lerp_g(2) - ga;
    let gk3 = lerp_g(4) - ga;
    let gk4 = ga - lerp_g(1) - lerp_g(2) + lerp_g(3);
    let gk5 = ga - lerp_g(2) - lerp_g(4) + lerp_g(6);
    let gk6 = ga - lerp_g(1) - lerp_g(4) + lerp_g(5);
    let gk7 = -ga + lerp_g(1) + lerp_g(2) - lerp_g(3) + lerp_g(4) - lerp_g(5) - lerp_g(6) + lerp_g(7);
    let gi = ga + gk1 * u + gk2 * v + gk3 * w + gk4 * (u * v) + gk5 * (v * w) + gk6 * (w * u) + gk7 * (u * v * w);
    let dn = DVec3::new(
        du * (k1 + k4 * v + k6 * w + k7 * v * w),
        dv * (k2 + k5 * w + k4 * u + k7 * w * u),
        dw * (k3 + k6 * u + k5 * v + k7 * u * v),
    );
    (n * 1.1, (gi + dn) * 1.1)
}

#[inline]
pub fn perlin3(seed: u64, p: DVec3) -> f64 {
    perlin3_d(seed, p).0
}

/// 2D gradient noise (used in local tangent frames).
#[inline]
pub fn perlin2(seed: u64, p: DVec2) -> f64 {
    let g = GRADS.get();
    let pf = p.floor();
    let (ix, iy) = (pf.x as i64, pf.y as i64);
    let f = p - pf;
    let (u, v) = (fade(f.x), fade(f.y));
    let corner = |dx: i64, dy: i64| {
        let h = hash2(seed, ix + dx, iy + dy);
        let gv = g[(h >> 56) as usize];
        // project 3D gradient to 2D (still well distributed in direction, length varies -> fine)
        gv[0] * (f.x - dx as f64) + gv[2] * (f.y - dy as f64)
    };
    let n00 = corner(0, 0);
    let n10 = corner(1, 0);
    let n01 = corner(0, 1);
    let n11 = corner(1, 1);
    let nx0 = n00 + u * (n10 - n00);
    let nx1 = n01 + u * (n11 - n01);
    (nx0 + v * (nx1 - nx0)) * 1.4
}

/// A random rotation + offset per octave, to kill lattice alignment between octaves.
#[derive(Clone, Debug)]
pub struct OctaveFrames {
    pub rot: Vec<DMat3>,
    pub off: Vec<DVec3>,
    pub seeds: Vec<u64>,
}

impl OctaveFrames {
    pub fn new(seed: u64, n: usize) -> Self {
        let mut rot = Vec::with_capacity(n);
        let mut off = Vec::with_capacity(n);
        let mut seeds = Vec::with_capacity(n);
        for i in 0..n {
            let h = hash1(seed, i as i64 + 1000);
            rot.push(random_rotation(h));
            off.push(DVec3::new(u01k(h, 1), u01k(h, 2), u01k(h, 3)) * 1000.0);
            seeds.push(mix64(h ^ 0xABCD));
        }
        OctaveFrames { rot, off, seeds }
    }
}

pub fn random_rotation(h: u64) -> DMat3 {
    // uniform random quaternion (Shoemake)
    let u1 = u01k(h, 11);
    let u2 = u01k(h, 12) * std::f64::consts::TAU;
    let u3 = u01k(h, 13) * std::f64::consts::TAU;
    let a = (1.0 - u1).sqrt();
    let b = u1.sqrt();
    let q = glam::DQuat::from_xyzw(a * u2.sin(), a * u2.cos(), b * u3.sin(), b * u3.cos()).normalize();
    DMat3::from_quat(q)
}

/// Band-limiting weight for a feature of wavelength `lambda` sampled with pixel size `gsd`:
/// 1 when well resolved (lambda >= 4 gsd), 0 when below Nyquist (lambda <= 2 gsd).
#[inline(always)]
pub fn band(lambda: f64, gsd: f64) -> f64 {
    smoothstep(2.0 * gsd, 4.0 * gsd, lambda)
}

#[inline(always)]
pub fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
#[inline(always)]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}
#[inline(always)]
pub fn saturate(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// Fractal Brownian motion in 3D (point in meters), band-limited by `gsd`.
/// Returns value (sum of octaves, amplitude 1 at the first octave) and the gradient (per meter).
#[derive(Clone, Debug)]
pub struct Fbm {
    pub frames: OctaveFrames,
    pub wavelength: f64,
    pub octaves: usize,
    pub lacunarity: f64,
    pub gain: f64,
}

impl Fbm {
    pub fn new(seed: u64, wavelength: f64, octaves: usize, lacunarity: f64, gain: f64) -> Self {
        Fbm { frames: OctaveFrames::new(seed, octaves), wavelength, octaves, lacunarity, gain }
    }

    /// Normalization so the output stays roughly within [-1,1].
    pub fn norm(&self) -> f64 {
        let mut s = 0.0;
        let mut a = 1.0;
        for _ in 0..self.octaves {
            s += a * a;
            a *= self.gain;
        }
        1.0 / s.sqrt().max(1e-9) * 0.75
    }

    #[inline]
    pub fn eval(&self, p: DVec3, gsd: f64) -> f64 {
        self.eval_d(p, gsd, self.octaves).0
    }

    /// Evaluate with at most `max_oct` octaves; `gsd` fades out unresolvable octaves.
    pub fn eval_d(&self, p: DVec3, gsd: f64, max_oct: usize) -> (f64, DVec3) {
        let mut lam = self.wavelength;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut grad = DVec3::ZERO;
        for i in 0..self.octaves.min(max_oct) {
            let w = band(lam, gsd);
            if w <= 0.0 {
                break;
            }
            let q = self.frames.rot[i] * (p / lam) + self.frames.off[i];
            let (n, g) = perlin3_d(self.frames.seeds[i], q);
            sum += amp * w * n;
            grad += (self.frames.rot[i].transpose() * g) * (amp * w / lam);
            lam /= self.lacunarity;
            amp *= self.gain;
        }
        (sum, grad)
    }

    /// Evaluate only octaves with wavelength >= `min_lambda` (low-pass), plus gsd band-limit.
    pub fn eval_lowpass(&self, p: DVec3, gsd: f64, min_lambda: f64) -> f64 {
        let mut lam = self.wavelength;
        let mut amp = 1.0;
        let mut sum = 0.0;
        for i in 0..self.octaves {
            let w = band(lam, gsd) * smoothstep(min_lambda * 0.5, min_lambda, lam);
            if w <= 0.0 {
                break;
            }
            let q = self.frames.rot[i] * (p / lam) + self.frames.off[i];
            sum += amp * w * perlin3(self.frames.seeds[i], q);
            lam /= self.lacunarity;
            amp *= self.gain;
        }
        sum
    }
}

/// Result of a 3D cellular (Worley) query.
#[derive(Clone, Copy, Debug)]
pub struct Cell3 {
    /// hash identifying the nearest feature point's cell
    pub id: u64,
    /// nearest feature point (in the scaled domain, i.e. meters if `cell` was in meters)
    pub point: DVec3,
    pub f1: f64,
    pub f2: f64,
    pub id2: u64,
    pub point2: DVec3,
}

/// Id and feature point (meters) of lattice cell `c` of the jittered 3D lattice of [`worley3`].
pub fn worley3_site(seed: u64, c: (i64, i64, i64), cell: f64, jitter: f64) -> (u64, DVec3) {
    let h = hash3(seed, c.0, c.1, c.2);
    let fp = DVec3::new(
        c.0 as f64 + 0.5 + jitter * (u01k(h, 1) - 0.5),
        c.1 as f64 + 0.5 + jitter * (u01k(h, 2) - 0.5),
        c.2 as f64 + 0.5 + jitter * (u01k(h, 3) - 0.5),
    );
    (h, fp * cell)
}

/// Nearest two feature points of a jittered 3D lattice with cell size `cell` (meters).
pub fn worley3(seed: u64, p: DVec3, cell: f64, jitter: f64) -> Cell3 {
    let q = p / cell;
    let qf = q.floor();
    let (ix, iy, iz) = (qf.x as i64, qf.y as i64, qf.z as i64);
    let mut best = Cell3 { id: 0, point: DVec3::ZERO, f1: f64::MAX, f2: f64::MAX, id2: 0, point2: DVec3::ZERO };
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy, cz) = (ix + dx, iy + dy, iz + dz);
                let h = hash3(seed, cx, cy, cz);
                let fp = DVec3::new(
                    cx as f64 + 0.5 + jitter * (u01k(h, 1) - 0.5),
                    cy as f64 + 0.5 + jitter * (u01k(h, 2) - 0.5),
                    cz as f64 + 0.5 + jitter * (u01k(h, 3) - 0.5),
                );
                let d = (fp - q).length_squared();
                if d < best.f1 {
                    best.f2 = best.f1;
                    best.id2 = best.id;
                    best.point2 = best.point;
                    best.f1 = d;
                    best.id = h;
                    best.point = fp;
                } else if d < best.f2 {
                    best.f2 = d;
                    best.id2 = h;
                    best.point2 = fp;
                }
            }
        }
    }
    best.f1 = best.f1.sqrt() * cell;
    best.f2 = best.f2.sqrt() * cell;
    best.point *= cell;
    best.point2 *= cell;
    best
}

/// Approximate distance (meters) from `q` to the bisector between the two nearest Worley points.
#[inline]
pub fn worley_edge_dist(c: &Cell3, q: DVec3) -> f64 {
    let d = c.point2 - c.point;
    let len = d.length();
    if len < 1e-12 {
        return f64::MAX;
    }
    let mid = (c.point + c.point2) * 0.5;
    ((mid - q).dot(d / len)).abs()
}

/// 2D cellular query in a local planar frame. Returns (id, feature point, f1, f2, id2, point2).
#[derive(Clone, Copy, Debug)]
pub struct Cell2 {
    pub id: u64,
    pub point: DVec2,
    pub f1: f64,
    pub f2: f64,
    pub id2: u64,
    pub point2: DVec2,
}

pub fn worley2(seed: u64, p: DVec2, cell: f64, jitter: f64) -> Cell2 {
    let q = p / cell;
    let qf = q.floor();
    let (ix, iy) = (qf.x as i64, qf.y as i64);
    let mut best = Cell2 { id: 0, point: DVec2::ZERO, f1: f64::MAX, f2: f64::MAX, id2: 0, point2: DVec2::ZERO };
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (cx, cy) = (ix + dx, iy + dy);
            let h = hash2(seed, cx, cy);
            let fp = DVec2::new(
                cx as f64 + 0.5 + jitter * (u01k(h, 1) - 0.5),
                cy as f64 + 0.5 + jitter * (u01k(h, 2) - 0.5),
            );
            let d = (fp - q).length_squared();
            if d < best.f1 {
                best.f2 = best.f1;
                best.id2 = best.id;
                best.point2 = best.point;
                best.f1 = d;
                best.id = h;
                best.point = fp;
            } else if d < best.f2 {
                best.f2 = d;
                best.id2 = h;
                best.point2 = fp;
            }
        }
    }
    best.f1 = best.f1.sqrt() * cell;
    best.f2 = best.f2.sqrt() * cell;
    best.point *= cell;
    best.point2 *= cell;
    best
}

#[inline]
pub fn worley2_edge_dist(c: &Cell2, q: DVec2) -> f64 {
    let d = c.point2 - c.point;
    let len = d.length();
    if len < 1e-12 {
        return f64::MAX;
    }
    let mid = (c.point + c.point2) * 0.5;
    ((mid - q).dot(d / len)).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin_gradient_matches_finite_difference() {
        let p = DVec3::new(12.3, -4.56, 7.89);
        let (_, g) = perlin3_d(7, p);
        let e = 1e-6;
        let fd = DVec3::new(
            (perlin3(7, p + DVec3::X * e) - perlin3(7, p - DVec3::X * e)) / (2.0 * e),
            (perlin3(7, p + DVec3::Y * e) - perlin3(7, p - DVec3::Y * e)) / (2.0 * e),
            (perlin3(7, p + DVec3::Z * e) - perlin3(7, p - DVec3::Z * e)) / (2.0 * e),
        );
        assert!((g - fd).length() < 1e-5, "{g:?} vs {fd:?}");
    }

    #[test]
    fn perlin_range_and_determinism() {
        let mut mn = f64::MAX;
        let mut mx = f64::MIN;
        for i in 0..20000 {
            let p = DVec3::new(i as f64 * 0.37, (i as f64 * 0.11).sin() * 50.0, i as f64 * 0.013);
            let v = perlin3(3, p);
            assert_eq!(v, perlin3(3, p));
            mn = mn.min(v);
            mx = mx.max(v);
        }
        assert!(mn > -1.3 && mx < 1.3 && mn < -0.5 && mx > 0.5, "{mn} {mx}");
    }

    #[test]
    fn fbm_gradient_matches() {
        let f = Fbm::new(5, 1000.0, 5, 2.0, 0.5);
        let p = DVec3::new(6.4e6, 1.2e5, 3.3e5);
        let (_, g) = f.eval_d(p, 1.0, 99);
        let e = 1e-3;
        let fd = (f.eval(p + DVec3::X * e, 1.0) - f.eval(p - DVec3::X * e, 1.0)) / (2.0 * e);
        assert!((g.x - fd).abs() < 1e-6 + 1e-3 * fd.abs(), "{} {}", g.x, fd);
    }
}
