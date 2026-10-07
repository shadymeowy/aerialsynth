//! Micro-benchmark of the noise primitives: `cargo run --release -p terragen --example noise_bench`
use glam::DVec3;
use std::time::Instant;
use terragen::noise::{perlin3, worley3};

fn main() {
    let n = 20_000_000usize;
    let pts: Vec<DVec3> = (0..1024).map(|i| DVec3::new(4.1e6 + i as f64 * 0.731, 3.3e5 + i as f64 * 0.377, 4.7e6 - i as f64 * 0.119) / 37.0).collect();
    let mut acc = 0.0;
    let t = Instant::now();
    for i in 0..n {
        acc += perlin3(i as u64 & 7, pts[i & 1023] + DVec3::splat((i >> 10) as f64 * 0.013));
    }
    let dt = t.elapsed().as_secs_f64();
    println!("perlin3: {:.2} ns/call ({acc:.3})", dt / n as f64 * 1e9);
    let m = n / 10;
    let t = Instant::now();
    let mut s = 0u64;
    for i in 0..m {
        s = s.wrapping_add(worley3(3, pts[i & 1023] * 37.0 + DVec3::splat((i >> 10) as f64 * 0.7), 240.0, 0.9).id);
    }
    let dt = t.elapsed().as_secs_f64();
    println!("worley3: {:.2} ns/call ({s})", dt / m as f64 * 1e9);
}
