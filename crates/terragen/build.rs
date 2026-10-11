//! Records the wgpu version of the workspace's Cargo.lock (part of the pipeline cache key).
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut version = String::from("unknown");
    let mut dir = Some(Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).to_path_buf());
    while let Some(d) = dir {
        let lock = d.join("Cargo.lock");
        if let Ok(text) = std::fs::read_to_string(&lock) {
            println!("cargo:rerun-if-changed={}", lock.display());
            let mut lines = text.lines();
            while let Some(l) = lines.next() {
                if l == "name = \"wgpu\"" {
                    if let Some(v) = lines.next().and_then(|v| v.strip_prefix("version = \"")) {
                        version = v.trim_end_matches('"').to_string();
                    }
                    break;
                }
            }
            break;
        }
        dir = d.parent().map(|p| p.to_path_buf());
    }
    println!("cargo:rustc-env=TERRAGEN_WGPU_VERSION={version}");
}
