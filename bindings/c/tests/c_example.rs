//! Compile examples/tile.c with the system C compiler against the shared library built for this
//! test (target/<profile>/libaerialsynth.so / .dylib, or aerialsynth.dll on Windows) and run it on
//! a temporary tile store. The compiler: `cc` (or `$CC`) on Unix, MSVC's `cl.exe` (found by the
//! `cc` crate) on Windows. Skipped when there is none.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The command compiling `src` into the executable `bin`, linked against the library in
/// `lib_dir`; `None` without a C compiler.
#[cfg(not(windows))]
fn compile_cmd(src: &Path, include: &Path, lib_dir: &Path, bin: &Path) -> Option<Command> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    Command::new(&cc).arg("--version").output().ok().filter(|o| o.status.success())?;
    let lib = lib_dir.join(if cfg!(target_os = "macos") { "libaerialsynth.dylib" } else { "libaerialsynth.so" });
    assert!(lib.exists(), "{} not built", lib.display());
    let mut cmd = Command::new(&cc);
    cmd.args(["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic"])
        .arg(src)
        .arg("-I")
        .arg(include)
        .arg("-L")
        .arg(lib_dir)
        .arg("-laerialsynth")
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-o")
        .arg(bin);
    Some(cmd)
}

#[cfg(windows)]
fn compile_cmd(src: &Path, include: &Path, lib_dir: &Path, bin: &Path) -> Option<Command> {
    let target = format!("{}-pc-windows-msvc", std::env::consts::ARCH);
    let cl = cc::windows_registry::find_tool(&target, "cl.exe")?;
    // the import library of aerialsynth.dll (aerialsynth.lib is the static library)
    let lib = lib_dir.join("aerialsynth.dll.lib");
    assert!(lib.exists(), "{} not built", lib.display());
    let mut cmd = cl.to_command(); // cl.exe with the INCLUDE / LIB / PATH of its toolchain
    let out_dir = bin.parent().unwrap();
    cmd.args(["/nologo", "/std:c11", "/W4", "/WX", "/D_CRT_SECURE_NO_WARNINGS"])
        .arg(src)
        .arg(format!("/I{}", include.display()))
        .arg(format!("/Fo{}\\", out_dir.display()))
        .arg(format!("/Fe{}", bin.display()))
        .arg("/link")
        .arg(&lib);
    Some(cmd)
}

#[test]
fn c_example_runs() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    // this test is target/<profile>/deps/c_example-<hash>; the library is in target/<profile>
    let exe = std::env::current_exe().unwrap();
    let lib_dir = exe.parent().and_then(Path::parent).unwrap().to_path_buf();

    let dir = std::env::temp_dir().join(format!("aerialsynth-c-example-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join(if cfg!(windows) { "tile.exe" } else { "tile" });
    let Some(mut compile) = compile_cmd(&crate_dir.join("examples/tile.c"), &crate_dir.join("include"), &lib_dir, &bin) else {
        eprintln!("c_example_runs skipped: no C compiler (cc / $CC, or MSVC on Windows)");
        return;
    };
    let out = compile.output().expect("running the C compiler");
    assert!(out.status.success(), "compiling examples/tile.c failed:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));

    let config = dir.join("world.yaml");
    std::fs::write(&config, "world: { seed: 4, tile_supersample: 1 }\ntiles: { max_zoom: 6 }\n").unwrap();
    let store: PathBuf = dir.join("store").join("world.h5");
    let run = || {
        let mut cmd = Command::new(&bin);
        if cfg!(windows) {
            // the DLL is found on the PATH (Unix: the rpath)
            let path = std::env::var_os("PATH").unwrap_or_default();
            let paths = std::iter::once(lib_dir.clone()).chain(std::env::split_paths(&path));
            cmd.env("PATH", std::env::join_paths(paths).unwrap());
        }
        let out = cmd.arg(&store).arg(&config).args(["3", "5", "3"]).output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "tile failed ({}):\n{stdout}\n{}", out.status, String::from_utf8_lossy(&out.stderr));
        stdout
    };
    let first = run(); // generates the tile
    assert!(first.contains("tile 3/5/3 elevation: min") && first.contains("rgb: 3 channels, 196608 bytes"), "{first}");
    assert!(first.contains("zoom 7: refused"), "{first}");
    assert_eq!(run(), first, "the stored tile differs from the generated one");
    let _ = std::fs::remove_dir_all(&dir);
}
