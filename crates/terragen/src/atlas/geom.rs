//! Cube-map geometry of the atlas: faces, the tangent warp, texel centres and lookups.
//!
//! A direction belongs to the face of its largest |component| (x before y before z on ties).
//! On a face, the gnomonic coordinates (a, b) ∈ [-1, 1]² are warped to s = atan(a) / (π/4), which
//! makes the texels nearly uniform in size (±30 % instead of ±160 %). Texel `i` of a face of
//! resolution `r` has its centre at s = (i + 0.5)·2/r − 1; continuous texel coordinates
//! x = (s + 1)·r/2 − 0.5 put the centres at integers.

use glam::DVec3;
use std::f64::consts::FRAC_PI_4;

/// Face frames: outward normal, u axis, v axis (u × v = normal).
pub const FACES: [[DVec3; 3]; 6] = [
    [DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 1.0)],
    [DVec3::new(-1.0, 0.0, 0.0), DVec3::new(0.0, -1.0, 0.0), DVec3::new(0.0, 0.0, 1.0)],
    [DVec3::new(0.0, 1.0, 0.0), DVec3::new(-1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 1.0)],
    [DVec3::new(0.0, -1.0, 0.0), DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 1.0)],
    [DVec3::new(0.0, 0.0, 1.0), DVec3::new(0.0, 1.0, 0.0), DVec3::new(-1.0, 0.0, 0.0)],
    [DVec3::new(0.0, 0.0, -1.0), DVec3::new(0.0, 1.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
];

/// Largest |s| of an extended (off-face) texel coordinate: tan(π/4·s) diverges at |s| = 2.
const S_EXT_MAX: f64 = 1.9;

/// The face of a direction: 2·axis + (component < 0), the axis of the largest |component|.
#[inline]
pub fn face_of(d: DVec3) -> usize {
    let a = d.abs();
    if a.x >= a.y && a.x >= a.z {
        (d.x < 0.0) as usize
    } else if a.y >= a.z {
        2 + (d.y < 0.0) as usize
    } else {
        4 + (d.z < 0.0) as usize
    }
}

/// Face and warped coordinates (s_u, s_v) of a direction (not necessarily unit).
#[inline]
pub fn warped(d: DVec3) -> (usize, f64, f64) {
    let f = face_of(d);
    let [n, u, v] = FACES[f];
    let m = d.dot(n);
    (f, (d.dot(u) / m).atan() / FRAC_PI_4, (d.dot(v) / m).atan() / FRAC_PI_4)
}

/// Unit direction at warped coordinates (s_u, s_v) of face `f` (also beyond the face, |s| < 2).
#[inline]
pub fn dir_at(f: usize, su: f64, sv: f64) -> DVec3 {
    let [n, u, v] = FACES[f];
    (n + u * (su * FRAC_PI_4).tan() + v * (sv * FRAC_PI_4).tan()).normalize()
}

/// The texels of one face resolution: 6 faces of `r` × `r`, index `(f·r + j)·r + i`.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub r: usize,
}

impl Grid {
    pub fn new(r: usize) -> Self {
        Grid { r }
    }
    /// Number of texels.
    #[inline]
    pub fn n(&self) -> usize {
        6 * self.r * self.r
    }
    #[inline]
    pub fn idx(&self, f: usize, i: usize, j: usize) -> usize {
        (f * self.r + j) * self.r + i
    }
    #[inline]
    pub fn fij(&self, k: usize) -> (usize, usize, usize) {
        let r = self.r;
        (k / (r * r), k % r, (k / r) % r)
    }
    /// Warped coordinate of (possibly extended) texel index `i`.
    #[inline]
    pub fn s_of(&self, i: f64) -> f64 {
        (i + 0.5) * 2.0 / self.r as f64 - 1.0
    }
    /// Continuous texel coordinate of warped coordinate `s`.
    #[inline]
    pub fn x_of(&self, s: f64) -> f64 {
        (s + 1.0) * 0.5 * self.r as f64 - 0.5
    }
    /// Centre direction of texel `k`.
    pub fn dir(&self, k: usize) -> DVec3 {
        let (f, i, j) = self.fij(k);
        dir_at(f, self.s_of(i as f64), self.s_of(j as f64))
    }
    /// Centre direction of a texel of face `f` at (possibly extended) indices, or None when too
    /// far off the face.
    pub fn dir_ext(&self, f: usize, i: i64, j: i64) -> Option<DVec3> {
        let (su, sv) = (self.s_of(i as f64), self.s_of(j as f64));
        (su.abs() < S_EXT_MAX && sv.abs() < S_EXT_MAX).then(|| dir_at(f, su, sv))
    }
    /// Face and continuous texel coordinates (centres at integers) of a direction.
    #[inline]
    pub fn locate(&self, d: DVec3) -> (usize, f64, f64) {
        let (f, su, sv) = warped(d);
        (f, self.x_of(su), self.x_of(sv))
    }
    /// The texel containing a direction.
    pub fn nearest(&self, d: DVec3) -> usize {
        let (f, x, y) = self.locate(d);
        let c = |x: f64| ((x + 0.5).floor().max(0.0) as usize).min(self.r - 1);
        self.idx(f, c(x), c(y))
    }
    /// Texel (i, j) of face `f`, or for indices off the face the texel of the neighbouring face
    /// containing that (extended) texel centre; None when too far off the face.
    #[inline]
    pub fn at(&self, f: usize, i: i64, j: i64) -> Option<usize> {
        let r = self.r as i64;
        if (0..r).contains(&i) && (0..r).contains(&j) {
            Some(self.idx(f, i as usize, j as usize))
        } else {
            self.dir_ext(f, i, j).map(|d| self.nearest(d))
        }
    }
    /// Bilinear taps (texel, weight) at a direction; taps off the face come from the
    /// neighbouring faces (their nearest texels).
    pub fn bilinear(&self, d: DVec3) -> [(usize, f64); 4] {
        let (f, x, y) = self.locate(d);
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let (i, j) = (x0 as i64, y0 as i64);
        let k = |di: i64, dj: i64| self.at(f, i + di, j + dj).unwrap_or_else(|| self.nearest(d));
        [(k(0, 0), (1.0 - tx) * (1.0 - ty)), (k(1, 0), tx * (1.0 - ty)), (k(0, 1), (1.0 - tx) * ty), (k(1, 1), tx * ty)]
    }
    /// Nominal texel spacing (radians).
    pub fn spacing(&self) -> f64 {
        std::f64::consts::FRAC_PI_2 / self.r as f64
    }
    /// The 4 edge neighbours of texel `k` (+i, −i, +j, −j), across faces where needed.
    pub fn neighbours4(&self, k: usize) -> [usize; 4] {
        let (f, i, j) = self.fij(k);
        let (i, j) = (i as i64, j as i64);
        let g = |di: i64, dj: i64| self.at(f, i + di, j + dj).unwrap_or(k);
        [g(1, 0), g(-1, 0), g(0, 1), g(0, -1)]
    }
}

/// [`Grid::at`] for all indices of the extended faces (|s| < 1.9), precomputed: jump flooding
/// and splatting look up texels far beyond the face edges.
pub struct ExtGrid {
    pub g: Grid,
    lo: i64,
    side: usize,
    tab: Vec<u32>,
}

impl ExtGrid {
    pub fn new(g: Grid) -> Self {
        use rayon::prelude::*;
        let r = g.r as f64;
        let lo = (-0.5 * (S_EXT_MAX - 1.0) * r - 0.5).ceil() as i64;
        let hi = (0.5 * (S_EXT_MAX + 1.0) * r - 0.5).floor() as i64;
        let side = (hi - lo + 1) as usize;
        let tab = (0..6 * side * side)
            .into_par_iter()
            .map(|t| {
                let (f, j, i) = (t / (side * side), (t / side) % side, t % side);
                g.at(f, i as i64 + lo, j as i64 + lo).map_or(u32::MAX, |k| k as u32)
            })
            .collect();
        ExtGrid { g, lo, side, tab }
    }

    /// The same as [`Grid::at`].
    #[inline]
    pub fn at(&self, f: usize, i: i64, j: i64) -> Option<usize> {
        let r = self.g.r as i64;
        if (0..r).contains(&i) && (0..r).contains(&j) {
            return Some(self.g.idx(f, i as usize, j as usize));
        }
        let (a, b) = (i - self.lo, j - self.lo);
        if a < 0 || b < 0 || a >= self.side as i64 || b >= self.side as i64 {
            return None;
        }
        let v = self.tab[(f * self.side + b as usize) * self.side + a as usize];
        (v != u32::MAX).then_some(v as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texel_round_trip_and_neighbours() {
        let g = Grid::new(16);
        for k in 0..g.n() {
            let d = g.dir(k);
            assert!((d.length() - 1.0).abs() < 1e-12);
            assert_eq!(g.nearest(d), k);
            let (f, i, j) = g.fij(k);
            assert_eq!(g.idx(f, i, j), k);
            // edge neighbours are close (also across faces): < 1.7 nominal spacings
            for n in g.neighbours4(k) {
                let a = g.dir(n).angle_between(d);
                assert!(a > 0.3 * g.spacing() && a < 1.7 * g.spacing(), "{k} -> {n}: {a}");
            }
        }
        let e = ExtGrid::new(g);
        for f in 0..6 {
            for j in -20..36 {
                for i in -20..36 {
                    assert_eq!(e.at(f, i, j), g.at(f, i, j), "{f} {i} {j}");
                }
            }
        }
    }
}
