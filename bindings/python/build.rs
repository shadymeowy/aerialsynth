//! macOS: an extension module leaves the Python symbols undefined (the interpreter provides them
//! when it loads the module), which Apple's linker refuses unless told so. maturin passes the flag
//! itself; this makes a plain `cargo build --workspace` link the module too (as
//! `pyo3_build_config::add_extension_module_link_args` does).
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
