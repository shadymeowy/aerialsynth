//! Attitude helpers.
//!
//! Conventions:
//!
//! * Body frame is **FRD**: x forward, y right, z down.
//! * Quaternions are Hamilton ([`DQuat`], stored `x, y, z, w`) and act as *active*
//!   rotations of vectors: `q_a_b` (named `q_<to>_<from>`) maps a vector expressed in frame
//!   `b` into frame `a`, `v_a = q_a_b * v_b`, and composes as `q_a_c = q_a_b * q_b_c`.
//! * Euler angles are aerospace yaw-pitch-roll (3-2-1, intrinsic Z-Y'-X''):
//!   `R_ned_body = Rz(yaw) · Ry(pitch) · Rx(roll)`. Yaw is the heading clockwise from north
//!   (seen from above), pitch is nose-up positive, roll is right-wing-down positive.

use glam::{DMat3, DQuat};

use crate::frames::rot_ecef2ned;

/// Aerospace 3-2-1 Euler angles (rad) → `q_ned_body` (`v_ned = q * v_body`).
pub fn euler_zyx_to_quat(yaw: f64, pitch: f64, roll: f64) -> DQuat {
    DQuat::from_rotation_z(yaw) * DQuat::from_rotation_y(pitch) * DQuat::from_rotation_x(roll)
}

/// `q_ned_body` → aerospace 3-2-1 Euler angles `(yaw, pitch, roll)` (rad), with
/// `yaw, roll ∈ (-π, π]` and `pitch ∈ [-π/2, π/2]`. The quaternion is normalized first.
/// At gimbal lock (`|pitch| = π/2`) yaw and roll are not separable; the split returned
/// is whatever the matrix elements give.
pub fn quat_to_euler_zyx(q: DQuat) -> (f64, f64, f64) {
    let q = q.normalize();
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    // Elements of R = Rz Ry Rx.
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r10 = 2.0 * (x * y + w * z);
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    let yaw = r10.atan2(r00);
    let pitch = (-r20).atan2(r21.hypot(r22));
    let roll = r21.atan2(r22);
    (yaw, pitch, roll)
}

/// `q_ecef_ned`: rotation mapping NED vectors at `lat`, `lon` (rad) into ECEF.
pub fn quat_ecef_from_ned(lat: f64, lon: f64) -> DQuat {
    DQuat::from_mat3(&rot_ecef2ned(lat, lon).transpose())
}

/// Convert a body→NED attitude (`q_ned_body`) at geodetic `lat`, `lon` (rad) into a
/// body→ECEF attitude: `q_ecef_body = q_ecef_ned * q_ned_body`.
pub fn body2ned_to_body2ecef(q_ned_body: DQuat, lat: f64, lon: f64) -> DQuat {
    quat_ecef_from_ned(lat, lon) * q_ned_body
}

/// Convert a body→ECEF attitude (`q_ecef_body`) into body→NED at `lat`, `lon` (rad):
/// `q_ned_body = q_ecef_nedᵀ * q_ecef_body`.
pub fn body2ecef_to_body2ned(q_ecef_body: DQuat, lat: f64, lon: f64) -> DQuat {
    quat_ecef_from_ned(lat, lon).conjugate() * q_ecef_body
}

/// Rotation matrix → quaternion (thin wrapper over [`DQuat::from_mat3`]; `m` must be a
/// proper rotation).
#[inline]
pub fn dmat3_to_quat(m: &DMat3) -> DQuat {
    DQuat::from_mat3(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::tests::Rng;
    use crate::frames::{ecef2nedv, ned2ecefv};
    use glam::DVec3;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn rx(a: f64) -> DMat3 {
        let (s, c) = a.sin_cos();
        DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c])
    }
    fn ry(a: f64) -> DMat3 {
        let (s, c) = a.sin_cos();
        DMat3::from_cols_array(&[c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c])
    }
    fn rz(a: f64) -> DMat3 {
        let (s, c) = a.sin_cos();
        DMat3::from_cols_array(&[c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0])
    }

    #[test]
    fn euler_matches_matrix_product() {
        let mut rng = Rng::new(21);
        for _ in 0..2000 {
            let (yaw, pitch, roll) = (
                rng.uniform(-PI, PI),
                rng.uniform(-1.5, 1.5),
                rng.uniform(-PI, PI),
            );
            let q = euler_zyx_to_quat(yaw, pitch, roll);
            let m = rz(yaw) * ry(pitch) * rx(roll);
            assert!(DMat3::from_quat(q).abs_diff_eq(m, 1e-14));
            let (y2, p2, r2) = quat_to_euler_zyx(q);
            assert!(
                (y2 - yaw).abs() < 1e-12 && (p2 - pitch).abs() < 1e-12 && (r2 - roll).abs() < 1e-12
            );
            assert!(
                dmat3_to_quat(&m).abs_diff_eq(q, 1e-14) || dmat3_to_quat(&m).abs_diff_eq(-q, 1e-14)
            );
        }
    }

    #[test]
    fn euler_semantics() {
        // Yaw 90°: body forward points east.
        let q = euler_zyx_to_quat(FRAC_PI_2, 0.0, 0.0);
        assert!((q * DVec3::X).abs_diff_eq(DVec3::Y, 1e-15));
        // Pitch +30°: nose up -> forward has negative down component.
        let q = euler_zyx_to_quat(0.0, 30f64.to_radians(), 0.0);
        let f = q * DVec3::X;
        assert!((f.z + 0.5).abs() < 1e-15 && f.x > 0.0);
        // Roll +90°: right wing points down.
        let q = euler_zyx_to_quat(0.0, 0.0, FRAC_PI_2);
        assert!((q * DVec3::Y).abs_diff_eq(DVec3::Z, 1e-15));
        // Near gimbal lock pitch stays accurate.
        let (_, p, _) = quat_to_euler_zyx(euler_zyx_to_quat(0.3, FRAC_PI_2 - 1e-9, -0.2));
        assert!((p - (FRAC_PI_2 - 1e-9)).abs() < 1e-12);
    }

    #[test]
    fn ned_ecef_attitude() {
        let mut rng = Rng::new(8);
        for _ in 0..1000 {
            let (lat, lon) = (rng.uniform(-FRAC_PI_2, FRAC_PI_2), rng.uniform(-PI, PI));
            let q_nb = euler_zyx_to_quat(
                rng.uniform(-PI, PI),
                rng.uniform(-1.5, 1.5),
                rng.uniform(-PI, PI),
            );
            let q_eb = body2ned_to_body2ecef(q_nb, lat, lon);
            let v_b = DVec3::new(
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            );
            // Consistent with the vector rotation helpers.
            let v_e = ned2ecefv(q_nb * v_b, lat, lon);
            assert!((q_eb * v_b).abs_diff_eq(v_e, 1e-14));
            assert!(ecef2nedv(q_eb * v_b, lat, lon).abs_diff_eq(q_nb * v_b, 1e-14));
            let back = body2ecef_to_body2ned(q_eb, lat, lon);
            assert!(back.abs_diff_eq(q_nb, 1e-14) || back.abs_diff_eq(-q_nb, 1e-14));
        }
        // At lat=0, lon=0: NED north = ECEF +z, east = +y, down = -x.
        let q = quat_ecef_from_ned(0.0, 0.0);
        assert!((q * DVec3::X).abs_diff_eq(DVec3::Z, 1e-15));
        assert!((q * DVec3::Y).abs_diff_eq(DVec3::Y, 1e-15));
        assert!((q * DVec3::Z).abs_diff_eq(-DVec3::X, 1e-15));
    }
}
