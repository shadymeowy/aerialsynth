//! Compile examples/tile.c and examples/render.c with the system C compiler (`cc`, or `$CC`)
//! against the library built for this test (target/<profile>/libaerialsynth.so) and run them on
//! a temporary tile store. Skipped when there is no C compiler.

use std::path::{Path, PathBuf};
use std::process::Command;

fn cc() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    Command::new(&cc).arg("--version").output().ok().filter(|o| o.status.success()).map(|_| cc)
}

/// Compile `examples/<example>.c` into `dir/<example>` (None without a C compiler).
fn compile(example: &str, dir: &Path) -> Option<PathBuf> {
    let Some(cc) = cc() else {
        eprintln!("{example}.c skipped: no C compiler (cc / $CC)");
        return None;
    };
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    // this test is target/<profile>/deps/c_example-<hash>; the library is in target/<profile>
    // (after `cargo build`), or only in target/<profile>/deps (`cargo test` does not copy it up)
    let exe = std::env::current_exe().unwrap();
    let deps = exe.parent().unwrap();
    let name = if cfg!(target_os = "macos") { "libaerialsynth.dylib" } else { "libaerialsynth.so" };
    let lib_dir = [deps.parent().unwrap(), deps].into_iter().find(|d| d.join(name).exists()).unwrap_or_else(|| panic!("{name} not built")).to_path_buf();
    let bin = dir.join(example);
    let out = Command::new(&cc)
        .args(["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic"])
        .arg(crate_dir.join(format!("examples/{example}.c")))
        .arg("-I")
        .arg(crate_dir.join("include"))
        .arg("-L")
        .arg(&lib_dir)
        .arg("-laerialsynth")
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("running the C compiler");
    assert!(out.status.success(), "compiling examples/{example}.c failed:\n{}", String::from_utf8_lossy(&out.stderr));
    Some(bin)
}

/// A fresh temporary directory with a cheap world config.
fn temp_dir(name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("aerialsynth-c-example-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("world.yaml");
    std::fs::write(&config, "world: { seed: 4, tile_supersample: 1 }\ntiles: { max_zoom: 6 }\n").unwrap();
    (dir, config)
}

#[test]
#[cfg(unix)]
fn c_example_runs() {
    let (dir, config) = temp_dir("tile");
    let Some(bin) = compile("tile", &dir) else { return };
    let store: PathBuf = dir.join("store/world.h5");
    let run = || {
        let out = Command::new(&bin).arg(&store).arg(&config).args(["3", "5", "3"]).output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "tile failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
        stdout
    };
    let first = run(); // generates the tile
    assert!(first.contains("tile 3/5/3 elevation: min") && first.contains("rgb: 3 channels, 196608 bytes"), "{first}");
    assert!(first.contains("zoom 7: refused"), "{first}");
    assert_eq!(run(), first, "the stored tile differs from the generated one");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[cfg(unix)]
fn c_render_example_runs() {
    let (dir, config) = temp_dir("render");
    let Some(bin) = compile("render", &dir) else { return };
    let ppm = dir.join("out.ppm");
    let out = Command::new(&bin).arg(dir.join("world.h5")).arg(&config).args(["45", "10", "1500"]).arg(&ppm).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "render failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("camera 160x120 on the") && stdout.contains("latitude 91: refused"), "{stdout}");
    let img = std::fs::read(&ppm).unwrap();
    let header = b"P6\n160 120\n255\n";
    assert!(img.starts_with(header) && img.len() == header.len() + 160 * 120 * 3, "{} bytes", img.len());
    let _ = std::fs::remove_dir_all(&dir);
}
