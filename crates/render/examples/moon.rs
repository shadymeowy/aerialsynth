fn main() {
    let l = render::lighting::LightingConfig { mode: render::lighting::SunMode::Clock, date: "2026-03-03".into(), time_utc: "21:00:00".into(), ..Default::default() };
    let s = l.sun_at(0.0, 40.03f64.to_radians(), 32.9f64.to_radians());
    println!("sun el {:.1} moon az {:.1} el {:.1} phase {:.3} direct {:.2e} sky {:.1e}", s.elevation.to_degrees(), s.moon_azimuth.to_degrees(), s.moon_elevation.to_degrees(), s.moon_phase, s.moon_direct, s.sky);
}
