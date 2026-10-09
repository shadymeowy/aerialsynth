//! macOS: the install name of libaerialsynth.dylib is `@rpath/libaerialsynth.dylib` (the linker's
//! default is its build path), so a program linked against it finds it through its rpath
//! (`-Wl,-rpath,<dir>`), wherever the library is installed.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libaerialsynth.dylib");
    }
}
