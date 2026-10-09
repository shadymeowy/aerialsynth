//! include/aerialsynth.h must be what cbindgen generates from src/lib.rs (with cbindgen.toml).
//! `AERIALSYNTH_BLESS=1 cargo test -p aerialsynth-capi --test header` rewrites it.

use std::path::Path;

#[test]
fn header_is_up_to_date() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = cbindgen::Config::from_file(dir.join("cbindgen.toml")).expect("cbindgen.toml");
    let bindings = cbindgen::Builder::new().with_config(config).with_src(dir.join("src/lib.rs")).generate().expect("cbindgen");
    let mut out = Vec::new();
    bindings.write(&mut out);
    let generated = String::from_utf8(out).unwrap();
    let path = dir.join("include/aerialsynth.h");
    if std::env::var_os("AERIALSYNTH_BLESS").is_some() {
        std::fs::write(&path, &generated).unwrap();
        return;
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == generated,
        "{} is out of date with src/lib.rs: regenerate it with AERIALSYNTH_BLESS=1 cargo test -p aerialsynth-capi --test header",
        path.display()
    );
}
