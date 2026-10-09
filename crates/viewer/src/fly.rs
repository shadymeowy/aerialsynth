//! Flight camera: a position (ECEF) and heading / pitch / roll relative to the local horizon
//! (east, north, up at the current position), so "level" stays level anywhere on the planet.
//!
//! * Free flight: WASD moves along the view, Space / C up and down, mouse drag looks around,
//!   scroll changes the speed, Shift boosts ×5.
//! * Plane: always flying forward at the airspeed; W/S pitch, A/D roll (banked turns follow
//!   from the roll), Q/E rudder, Shift / Ctrl throttle.

use crate::globe::CamFrame;
use geodesy::{ecef2geodetic, geodetic2ecef, Ellipsoid, Geodetic};
use glam::{DMat4, DVec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlyMode {
    Free,
    Plane,
}

/// The keys and mouse motion of one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct FlyInput {
    pub forward: f64,
    pub right: f64,
    pub up: f64,
    pub yaw: f64,
    pub boost: bool,
    pub slow: bool,
    /// mouse look (rad)
    pub look_yaw: f64,
    pub look_pitch: f64,
    /// speed change, in wheel steps
    pub speed_steps: f64,
}

#[derive(Clone, Debug)]
pub struct FlyCam {
    pub pos: DVec3,
    /// rad: heading clockwise from north, pitch up, roll right wing down
    pub heading: f64,
    pub pitch: f64,
    pub roll: f64,
    /// m/s (free flight: at the slowest; plane: airspeed)
    pub speed: f64,
    pub fov_y: f64,
    pub mode: FlyMode,
}

fn enu(lat: f64, lon: f64) -> (DVec3, DVec3, DVec3) {
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    (DVec3::new(-so, co, 0.0), DVec3::new(-sl * co, -sl * so, cl), DVec3::new(cl * co, cl * so, sl))
}

impl FlyCam {
    /// Start at `eye` looking along `dir` (ECEF).
    pub fn at(eye: DVec3, dir: DVec3, ell: &Ellipsoid, fov_y: f64, mode: FlyMode) -> FlyCam {
        let g = ecef2geodetic(eye, ell);
        let (e, n, u) = enu(g.lat, g.lon);
        let (de, dn, du) = (dir.dot(e), dir.dot(n), dir.dot(u));
        let heading = de.atan2(dn);
        let pitch = du.clamp(-1.0, 1.0).asin();
        let speed = (g.h.abs() * 0.3).clamp(20.0, 2.0e6);
        FlyCam { pos: eye, heading, pitch, roll: 0.0, speed, fov_y, mode }
    }

    pub fn geodetic(&self, ell: &Ellipsoid) -> Geodetic {
        ecef2geodetic(self.pos, ell)
    }

    /// Forward, right and up of the camera (ECEF).
    fn axes(&self, ell: &Ellipsoid) -> (DVec3, DVec3, DVec3) {
        let g = self.geodetic(ell);
        let (e, n, u) = enu(g.lat, g.lon);
        let (sh, ch) = self.heading.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let (sr, cr) = self.roll.sin_cos();
        let fwd = (e * sh + n * ch) * cp + u * sp;
        let right0 = e * ch - n * sh;
        let up0 = right0.cross(fwd);
        (fwd, right0 * cr - up0 * sr, up0 * cr + right0 * sr)
    }

    /// Advance by `dt` seconds. `ground` is the terrain height (m above the ellipsoid,
    /// exaggerated as drawn) under the camera; the camera stays above it.
    pub fn update(&mut self, dt: f64, inp: &FlyInput, ell: &Ellipsoid, ground: f64) {
        let dt = dt.clamp(0.0, 0.1);
        self.speed = (self.speed * 1.25f64.powf(inp.speed_steps)).clamp(1.0, 5.0e6);
        match self.mode {
            FlyMode::Free => {
                self.heading += inp.look_yaw;
                self.pitch = (self.pitch + inp.look_pitch).clamp(-1.55, 1.55);
                self.roll *= (-6.0 * dt).exp();
                let (fwd, right, _) = self.axes(ell);
                let up = ecef2geodetic(self.pos, ell);
                let (_, _, u) = enu(up.lat, up.lon);
                let mut v = fwd * inp.forward + right * inp.right + u * inp.up;
                if v.length_squared() > 1.0 {
                    v = v.normalize();
                }
                let k = if inp.boost {
                    5.0
                } else if inp.slow {
                    0.2
                } else {
                    1.0
                };
                self.pos += v * (self.speed * k * dt);
            }
            FlyMode::Plane => {
                // stick: roll and pitch rates; rudder: yaw rate; throttle: airspeed
                self.roll = (self.roll + inp.right * 1.6 * dt).clamp(-1.3, 1.3);
                if inp.right == 0.0 {
                    self.roll *= (-0.8 * dt).exp(); // a stable plane levels its wings
                }
                self.pitch = (self.pitch + inp.forward * -0.9 * dt * self.roll.cos()).clamp(-1.4, 1.4);
                // a coordinated turn: the bank turns the heading (g tan φ / V)
                let turn = 9.81 * self.roll.tan() / self.speed.max(10.0) + inp.yaw * 0.4;
                self.heading += turn * dt + inp.look_yaw;
                self.pitch = (self.pitch + inp.look_pitch).clamp(-1.4, 1.4);
                if inp.boost {
                    self.speed *= (0.7 * dt).exp();
                }
                if inp.slow {
                    self.speed *= (-0.7 * dt).exp();
                }
                let (fwd, _, _) = self.axes(ell);
                self.pos += fwd * (self.speed * dt);
            }
        }
        self.heading = self.heading.rem_euclid(std::f64::consts::TAU);
        // stay above the ground (a plane pulls up when it touches it)
        let g = self.geodetic(ell);
        let floor = ground + 2.0;
        if g.h < floor {
            self.pos = geodetic2ecef(Geodetic { h: floor, ..g }, ell);
            if self.mode == FlyMode::Plane && self.pitch < 0.05 {
                self.pitch = 0.05;
            }
        }
    }

    /// The camera frame; `ground` as in [`FlyCam::update`] (sets the near plane).
    pub fn frame(&self, ell: &Ellipsoid, aspect: f64, ground: f64) -> CamFrame {
        let (fwd, _, up) = self.axes(ell);
        let agl = (self.geodetic(ell).h - ground).max(0.5);
        let near = (agl * 0.3).clamp(0.1, 50_000.0);
        let view = DMat4::look_to_rh(DVec3::ZERO, fwd, up);
        let proj = DMat4::perspective_infinite_reverse_rh(self.fov_y, aspect, near);
        CamFrame { eye: self.pos, view_proj: proj * view, dir: fwd, cam_up: up, fov_y: self.fov_y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start(mode: FlyMode, h: f64) -> (FlyCam, Ellipsoid) {
        let ell = Ellipsoid::WGS84;
        let p = geodetic2ecef(Geodetic { lat: 0.7, lon: 0.6, h }, &ell);
        (FlyCam::at(p, DVec3::X, &ell, 0.8, mode), ell)
    }

    #[test]
    fn frame_looks_along_the_start_direction() {
        let ell = Ellipsoid::WGS84;
        let p = geodetic2ecef(Geodetic { lat: 0.7, lon: 0.6, h: 500.0 }, &ell);
        let dir = DVec3::new(0.3, -0.8, 0.2).normalize();
        let f = FlyCam::at(p, dir, &ell, 0.8, FlyMode::Free);
        assert!((f.frame(&ell, 1.5, 0.0).dir - dir).length() < 1e-9);
    }

    #[test]
    fn level_flight_keeps_heading_and_height() {
        let (mut f, ell) = start(FlyMode::Free, 1000.0);
        (f.heading, f.pitch, f.speed) = (1.0, 0.0, 50.0);
        let inp = FlyInput { forward: 1.0, ..Default::default() };
        for _ in 0..600 {
            f.update(1.0 / 60.0, &inp, &ell, 0.0);
        }
        // 500 m along a straight line: the ground curves away by d² / 2R ≈ 2 cm
        let h = f.geodetic(&ell).h;
        assert!((h - 1000.0).abs() < 0.1, "{h}");
        assert!((f.heading - 1.0).abs() < 1e-9);
        assert!((f.pos - start(FlyMode::Free, 1000.0).0.pos).length() > 499.0);
    }

    #[test]
    fn a_banked_plane_turns_at_g_tan_bank_over_speed() {
        let (mut f, ell) = start(FlyMode::Plane, 2000.0);
        (f.heading, f.pitch, f.roll, f.speed) = (0.0, 0.0, 0.5, 60.0);
        let dt = 1e-3;
        f.update(dt, &FlyInput::default(), &ell, 0.0);
        let rate = f.heading / dt;
        let want = 9.81 * 0.5f64.tan() / 60.0;
        assert!((rate - want).abs() < 0.01 * want, "{rate} vs {want}");
    }

    #[test]
    fn the_ground_stops_the_camera() {
        let (mut f, ell) = start(FlyMode::Free, 100.0);
        (f.pitch, f.speed) = (-1.5, 200.0);
        let inp = FlyInput { forward: 1.0, ..Default::default() };
        for _ in 0..120 {
            f.update(1.0 / 60.0, &inp, &ell, 80.0);
        }
        let h = f.geodetic(&ell).h;
        assert!((h - 82.0).abs() < 1e-6, "{h}");
    }
}
