//! Compile examples/tile.c and examples/render.c with the system C compiler against the shared
//! library built for this test (libaerialsynth.so / .dylib, or aerialsynth.dll on Windows) and run
//! them on a temporary tile store. The compiler: `cc` (or `$CC`) on Unix, MSVC's `cl.exe` (found
//! by the `cc` crate) on Windows. Skipped when there is none.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The file the linker is given: the shared library, or its import library on Windows.
const LINK_FILE: &str = if cfg!(windows) {
    "aerialsynth.dll.lib"
} else if cfg!(target_os = "macos") {
    "libaerialsynth.dylib"
} else {
    "libaerialsynth.so"
};

/// The directory of the library: this test is target/<profile>/deps/c_example-<hash>; the library
/// is in target/<profile> (after `cargo build`), or only in target/<profile>/deps (`cargo test`
/// does not copy it up).
fn lib_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let deps = exe.parent().unwrap();
    let dir = [deps.parent().unwrap(), deps].into_iter().find(|d| d.join(LINK_FILE).exists()).unwrap_or_else(|| panic!("{LINK_FILE} not built")).to_path_buf();
    dir
}

/// The command compiling `src` into the executable `bin`, linked against the library in
/// `lib_dir`; `None` without a C compiler.
#[cfg(not(windows))]
fn compile_cmd(src: &Path, include: &Path, lib_dir: &Path, bin: &Path) -> Option<Command> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    Command::new(&cc).arg("--version").output().ok().filter(|o| o.status.success())?;
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
    let mut cmd = cl.to_command(); // cl.exe with the INCLUDE / LIB / PATH of its toolchain
    let out_dir = bin.parent().unwrap();
    cmd.args(["/nologo", "/std:c11", "/W4", "/WX", "/D_CRT_SECURE_NO_WARNINGS"])
        .arg(src)
        .arg(format!("/I{}", include.display()))
        .arg(format!("/Fo{}\\", out_dir.display()))
        .arg(format!("/Fe{}", bin.display()))
        .arg("/link")
        .arg(lib_dir.join(LINK_FILE));
    Some(cmd)
}

/// Compile `examples/<example>.c` into `dir/<example>` (None without a C compiler).
fn compile(example: &str, dir: &Path) -> Option<PathBuf> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bin = dir.join(if cfg!(windows) { format!("{example}.exe") } else { example.to_string() });
    let src = crate_dir.join("examples").join(format!("{example}.c"));
    let Some(mut cmd) = compile_cmd(&src, &crate_dir.join("include"), &lib_dir(), &bin) else {
        eprintln!("{example}.c skipped: no C compiler (cc / $CC, or MSVC on Windows)");
        return None;
    };
    let out = cmd.output().expect("running the C compiler");
    assert!(out.status.success(), "compiling examples/{example}.c failed:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    Some(bin)
}

/// The command running a compiled example (Windows: the DLL found on the PATH; Unix: the rpath).
fn example_cmd(bin: &Path) -> Command {
    let mut cmd = Command::new(bin);
    if cfg!(windows) {
        let path = std::env::var_os("PATH").unwrap_or_default();
        let paths = std::iter::once(lib_dir()).chain(std::env::split_paths(&path));
        cmd.env("PATH", std::env::join_paths(paths).unwrap());
    }
    cmd
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
fn c_example_runs() {
    let (dir, config) = temp_dir("tile");
    let Some(bin) = compile("tile", &dir) else { return };
    let store: PathBuf = dir.join("store").join("world.h5");
    let run = || {
        let out = example_cmd(&bin).arg(&store).arg(&config).args(["3", "5", "3"]).output().unwrap();
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

#[test]
fn c_render_example_runs() {
    let (dir, config) = temp_dir("render");
    let Some(bin) = compile("render", &dir) else { return };
    let ppm = dir.join("out.ppm");
    let out = example_cmd(&bin).arg(dir.join("world.h5")).arg(&config).args(["45", "10", "1500"]).arg(&ppm).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "render failed ({}):\n{stdout}\n{}", out.status, String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("camera 160x120 on the") && stdout.contains("latitude 91: refused"), "{stdout}");
    let img = std::fs::read(&ppm).unwrap();
    let header = b"P6\n160 120\n255\n";
    assert!(img.starts_with(header) && img.len() == header.len() + 160 * 120 * 3, "{} bytes", img.len());
    let _ = std::fs::remove_dir_all(&dir);
}
