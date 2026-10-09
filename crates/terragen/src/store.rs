//! The world a tile store holds: tiles of different worlds (seed, config) or generator versions
//! must never be mixed in one store.

use crate::{Config, Generator};
use anyhow::{bail, Result};
use serde_yaml::Value;
use tilestore::{Layer, StoreMeta, TileStore};

/// Version of the generator's output. Bump it whenever a change alters generated tiles (stores
/// remember it, and tiles of another version are not appended to them).
/// 1: until 2026-10-08; 2: fast generation (grid-interpolated long octaves, adaptive
/// supersampling); 3: the grid / exact choices depend on the zoom only (version 2 decided them
/// per tile from its centre latitude: north-south seams at the switch latitudes; tiles centred
/// between 35.2° and 49.1° are unchanged).
pub const GENERATOR_VERSION: u32 = 3;

impl Generator {
    /// Metadata of a new store for this world.
    pub fn store_meta(&self) -> StoreMeta {
        StoreMeta {
            ellipsoid_a: self.world.ell.a,
            ellipsoid_b: self.world.ell.b,
            generator_config: self.config().to_yaml(),
            seed: self.config().seed,
            generator_version: GENERATOR_VERSION,
            layers: Layer::ALL.to_vec(),
        }
    }

    /// Open `path` for adding tiles of this world, creating it if missing. An existing store
    /// must hold this world (an empty one is re-stamped).
    pub fn open_store_rw(&self, path: &std::path::Path) -> Result<TileStore> {
        if !path.exists() {
            return TileStore::create(path, self.store_meta());
        }
        let mut st = TileStore::open_rw(path)?;
        if st.is_empty() {
            st.set_generator(&self.store_meta())?;
        } else {
            self.check_store(&st, true)?;
        }
        Ok(st)
    }

    /// Open `path` read-only; it must hold this world (another generator version is only a
    /// warning when reading).
    pub fn open_store_ro(&self, path: &std::path::Path) -> Result<TileStore> {
        let st = TileStore::open(path)?;
        if !st.is_empty() {
            self.check_store(&st, false)?;
        }
        Ok(st)
    }

    /// Does `store` hold tiles of this world? With `writing`, its generator version must match
    /// too (new tiles would not match its old ones).
    pub fn check_store(&self, store: &TileStore, writing: bool) -> Result<()> {
        let m = store.meta();
        let name = store.path().display();
        if m.generator_config.trim().is_empty() {
            bail!("{name} records no world config: use a new tiles file");
        }
        let stored = match Config::from_yaml_str(&m.generator_config) {
            Ok(c) => c,
            Err(e) => bail!("{name}: its world config does not parse with this version ({e:#}); it was written by an older generator: use a new tiles file"),
        };
        let (a, b) = (to_value(&stored), to_value(self.config()));
        if a != b {
            let mut diffs = Vec::new();
            diff(&a, &b, "world", &mut diffs);
            let shown: Vec<String> = diffs.iter().take(6).cloned().collect();
            bail!(
                "{name} holds another world ({} setting{} differ{}: {}{}): use another tiles file, or the scenario's world",
                diffs.len(),
                if diffs.len() == 1 { "" } else { "s" },
                if diffs.len() == 1 { "s" } else { "" },
                shown.join(", "),
                if diffs.len() > shown.len() { ", …" } else { "" }
            );
        }
        let (ea, eb) = (self.world.ell.a, self.world.ell.b);
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * y.abs().max(1.0);
        if !close(m.ellipsoid_a, ea) || !close(m.ellipsoid_b, eb) {
            bail!(
                "{name} holds tiles on another ellipsoid (a = {} m, b = {} m; the world's: a = {ea} m, b = {eb} m): use another tiles file",
                m.ellipsoid_a,
                m.ellipsoid_b
            );
        }
        if m.generator_version != GENERATOR_VERSION {
            let what = if m.generator_version == 0 {
                "an older generator (version not recorded)".to_string()
            } else {
                format!("generator version {}", m.generator_version)
            };
            if writing {
                bail!("{name} was written by {what}, this is version {GENERATOR_VERSION}: new tiles would not match its old ones; use a new tiles file");
            }
            // (once per store and process: commands open a store several times)
            static WARNED: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());
            let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
            if !warned.iter().any(|p| p == store.path()) {
                warned.push(store.path().to_path_buf());
                eprintln!("warning: {name} was written by {what} (this is version {GENERATOR_VERSION}); reading only");
            }
        }
        Ok(())
    }
}

fn to_value(c: &Config) -> Value {
    serde_yaml::to_value(c).unwrap_or(Value::Null)
}

/// Paths (with "store → scenario" values) where two config trees differ.
fn diff(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Mapping(ma), Value::Mapping(mb)) => {
            for (k, va) in ma {
                let key = k.as_str().unwrap_or("?");
                let p = format!("{path}.{key}");
                match mb.get(k) {
                    Some(vb) => diff(va, vb, &p, out),
                    None => out.push(p),
                }
            }
            for k in mb.keys() {
                if !ma.contains_key(k) {
                    out.push(format!("{path}.{}", k.as_str().unwrap_or("?")));
                }
            }
        }
        _ if a != b => {
            let short = |v: &Value| {
                let s = serde_yaml::to_string(v).unwrap_or_default().trim().replace('\n', " ");
                if s.chars().count() > 24 {
                    format!("{}…", s.chars().take(24).collect::<String>())
                } else {
                    s
                }
            };
            out.push(format!("{path} ({} → {})", short(a), short(b)));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_store_keeps_its_world() {
        let dir = std::env::temp_dir().join(format!("terragen-store-test-{}", std::process::id()));
        let path = dir.join("w.h5");
        let _ = std::fs::remove_dir_all(&dir);
        let g1 = Generator::new(Config { tile_supersample: 1, ..Config::default() });
        let st = g1.open_store_rw(&path).unwrap();
        st.write_tiles(&[g1.tile(geodesy::tiles::TileId::new(2, 1, 1))]).unwrap();
        drop(st);
        // same world: fine; another seed or setting: refused, naming what differs
        assert!(g1.open_store_rw(&path).is_ok());
        let g7 = Generator::new(Config { seed: 7, tile_supersample: 1, ..Config::default() });
        let e = match g7.open_store_rw(&path) {
            Ok(_) => panic!("a store of seed 1 accepted seed 7"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("world.seed (1 → 7)"), "{e}");
        assert!(g7.open_store_ro(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_store_takes_the_new_worlds_ellipsoid() {
        let dir = std::env::temp_dir().join(format!("terragen-store-ell-{}", std::process::id()));
        let path = dir.join("w.h5");
        let _ = std::fs::remove_dir_all(&dir);
        let mut sphere = Config { tile_supersample: 1, ..Config::default() };
        sphere.planet = crate::config::Planet { a: 6371000.0, inv_f: 0.0 };
        let gs = Generator::new(sphere);
        drop(gs.open_store_rw(&path).unwrap());
        assert_eq!(TileStore::open(&path).unwrap().meta().ellipsoid_b, 6371000.0);
        // re-used (empty) for the default world: re-stamped with its ellipsoid too
        let g = Generator::new(Config { tile_supersample: 1, ..Config::default() });
        drop(g.open_store_rw(&path).unwrap());
        let st = TileStore::open(&path).unwrap();
        let e = g.world.ell;
        assert_eq!((st.meta().ellipsoid_a, st.meta().ellipsoid_b), (e.a, e.b));
        g.check_store(&st, true).unwrap();
        drop(st);
        // a store whose ellipsoid disagrees with its config (as such re-use left them) is refused
        let bad = dir.join("bad.h5");
        drop(TileStore::create(&bad, StoreMeta { ellipsoid_a: 6371000.0, ellipsoid_b: 6371000.0, ..g.store_meta() }).unwrap());
        let e = g.check_store(&TileStore::open(&bad).unwrap(), false).unwrap_err().to_string();
        assert!(e.contains("another ellipsoid"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
