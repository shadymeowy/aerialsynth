//! Compile examples/tile.c with the system C compiler (`cc`, or `$CC`) against the library built
//! for this test (target/<profile>/libaerialsynth.so) and run it on a temporary tile store.
//! Skipped when there is no C compiler.

use std::path::{Path, PathBuf};
use std::process::Command;

fn cc() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    Command::new(&cc).arg("--version").output().ok().filter(|o| o.status.success()).map(|_| cc)
}

#[test]
#[cfg(unix)]
fn c_example_runs() {
    let Some(cc) = cc() else {
        eprintln!("c_example_runs skipped: no C compiler (cc / $CC)");
        return;
    };
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    // this test is target/<profile>/deps/c_example-<hash>; the library is in target/<profile>
    let exe = std::env::current_exe().unwrap();
    let lib_dir = exe.parent().and_then(Path::parent).unwrap().to_path_buf();
    let lib = lib_dir.join(if cfg!(target_os = "macos") { "libaerialsynth.dylib" } else { "libaerialsynth.so" });
    assert!(lib.exists(), "{} not built", lib.display());

    let dir = std::env::temp_dir().join(format!("aerialsynth-c-example-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("tile");
    let out = Command::new(&cc)
        .args(["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic"])
        .arg(crate_dir.join("examples/tile.c"))
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
    assert!(out.status.success(), "compiling examples/tile.c failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let config = dir.join("world.yaml");
    std::fs::write(&config, "world: { seed: 4, tile_supersample: 1 }\ntiles: { max_zoom: 6 }\n").unwrap();
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
