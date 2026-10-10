//! A process that used the shared GPU device exits cleanly: the device lives until the process
//! ends, and the NVIDIA driver's exit handlers used to unload its core library under its own
//! worker threads (a segfault after all the work was done, about one exit in six).
use geodesy::tiles::tile_for_latlon;
use terragen::{Config, Generator};

const RUNS: usize = 8;

/// The child process: GPU work from several threads (two generators set up at once and
/// generating, as the bindings' `World`s in threads do), then exit.
#[test]
#[ignore = "run by `exits_cleanly` in a child process"]
fn gpu_exit_child() {
    let threads: Vec<_> = (0..2)
        .map(|_| {
            std::thread::spawn(move || {
                let g = Generator::new(Config { tile_supersample: 1, ..Config::default() });
                let t = g.tile(tile_for_latlon(39.9f64.to_radians(), 32.8f64.to_radians(), 12));
                assert!(t.elevation.iter().all(|e| e.is_finite()));
                g.backend_name()
            })
        })
        .collect();
    for t in threads {
        println!("generated on {}", t.join().unwrap());
    }
}

#[test]
fn exits_cleanly() {
    if terragen::gpu::shared().is_err_and(|e| {
        eprintln!("no GPU ({e:#}): skipped");
        true
    }) {
        return;
    }
    let exe = std::env::current_exe().unwrap();
    for run in 0..RUNS {
        let out = std::process::Command::new(&exe).args(["gpu_exit_child", "--exact", "--ignored", "--nocapture", "--test-threads=1"]).output().unwrap();
        assert!(
            out.status.success(),
            "run {run}: the child exited with {} after:\n{}\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
