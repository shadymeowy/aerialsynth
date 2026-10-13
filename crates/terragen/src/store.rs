//! The world a tile store holds: tiles of different worlds (seed, config), generators or
//! generator versions must never be mixed in one store.
//!
//! A store records the world config (YAML, with `generator: classic | next`) and the version of
//! the generator's output. Stores written before the generator switch have no `generator` key:
//! version 3 or older is the classic generator (v0.2.0 stores open with it), version 4 the
//! next one.

use crate::config::GeneratorKind;
use crate::{Config, Generator};
use anyhow::{bail, Result};
use serde_yaml::Value;
use tilestore::{Layer, StoreMeta, TileStore};

/// Version of the next generator's output (`world.generator: next`). Bump it whenever a change
/// alters generated tiles (stores remember it, and tiles of another version are not appended to
/// them).
/// 1: until 2026-10-08; 2: fast generation (grid-interpolated long octaves, adaptive
/// supersampling); 3: the grid / exact choices depend on the zoom only (version 2 decided them
/// per tile from its centre latitude: north-south seams at the switch latitudes; tiles centred
/// between 35.2° and 49.1° are unchanged); 4: the composite stack over a biome registry,
/// ecoregions and cultures (`docs/design/terrain-next.md`).
pub const GENERATOR_VERSION: u32 = 4;

/// Version of the classic generator's output (`world.generator: classic`): v0.2.0's version 3,
/// frozen (the classic generator is v0.2.0's, unchanged).
pub const CLASSIC_GENERATOR_VERSION: u32 = 3;

/// The settings that make the world of `c`, as a YAML tree (what a store records and compares):
/// for the classic generator those of v0.2.0 and the generator (the next generator's own
/// settings do not change a classic world).
pub fn world_value(c: &Config) -> Value {
    let mut v = serde_yaml::to_value(c).unwrap_or(Value::Null);
    if c.generator == GeneratorKind::Classic {
        if let Value::Mapping(m) = &mut v {
            for k in NEXT_ONLY {
                m.remove(k);
            }
            if let Some(Value::Mapping(r)) = m.get_mut("relief") {
                for k in NEXT_ONLY_RELIEF {
                    r.remove(k);
                }
            }
        }
    }
    v
}

/// Settings of the next generator only (not read by the classic one).
const NEXT_ONLY: [&str; 3] = ["ecoregions", "biomes", "atlas"];
const NEXT_ONLY_RELIEF: [&str; 2] = ["tectonic_mountains", "belt_mountains"];

/// The generator of a stored world config: its `generator` key, else by the version (stores
/// written before the switch: up to 3 classic, v0.2.0's, 4 and later next).
fn stored_generator(raw: &Value, version: u32) -> Result<GeneratorKind> {
    match raw.get("generator") {
        Some(g) => Ok(serde_yaml::from_value(g.clone())?),
        // (0: not recorded, the oldest stores)
        None if version <= CLASSIC_GENERATOR_VERSION => Ok(GeneratorKind::Classic),
        None => Ok(GeneratorKind::Next),
    }
}

/// The generator whose tiles a store holds (`None`: its world config does not say, or does
/// not parse).
pub fn generator_of(m: &StoreMeta) -> Option<GeneratorKind> {
    if m.generator_config.trim().is_empty() {
        return None;
    }
    let raw: Value = serde_yaml::from_str(&m.generator_config).ok()?;
    stored_generator(&raw, m.generator_version).ok()
}

impl Generator {
    /// Metadata of a new store for this world.
    pub fn store_meta(&self) -> StoreMeta {
        let ell = self.ell();
        StoreMeta {
            ellipsoid_a: ell.a,
            ellipsoid_b: ell.b,
            generator_config: serde_yaml::to_string(&world_value(self.config())).unwrap_or_default(),
            seed: self.config().seed,
            generator_version: self.version(),
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
        let parsed = serde_yaml::from_str::<Value>(&m.generator_config)
            .map_err(anyhow::Error::from)
            .and_then(|raw| Ok((stored_generator(&raw, m.generator_version)?, Config::from_yaml_str(&m.generator_config)?)));
        let (kind, stored) = match parsed {
            Ok(p) => p,
            Err(e) => bail!("{name}: its world config does not parse with this version ({e:#}); it was written by an older generator: use a new tiles file"),
        };
        if kind != self.kind() {
            bail!(
                "{name} holds tiles of the {kind} generator (world.generator: {kind}), this world uses the {} generator: use another tiles file, or set world.generator: {kind}",
                self.kind()
            );
        }
        let stored = Config { generator: kind, ..stored };
        let (a, b) = (world_value(&stored), world_value(self.config()));
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
        let (ea, eb) = (self.ell().a, self.ell().b);
        let version = self.version();
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * y.abs().max(1.0);
        if !close(m.ellipsoid_a, ea) || !close(m.ellipsoid_b, eb) {
            bail!(
                "{name} holds tiles on another ellipsoid (a = {} m, b = {} m; the world's: a = {ea} m, b = {eb} m): use another tiles file",
                m.ellipsoid_a,
                m.ellipsoid_b
            );
        }
        if m.generator_version != version {
            let what = if m.generator_version == 0 {
                "an older generator (version not recorded)".to_string()
            } else {
                format!("generator version {}", m.generator_version)
            };
            if writing {
                bail!("{name} was written by {what}, this is version {version} ({kind}): new tiles would not match its old ones; use a new tiles file");
            }
            // (once per store and process: commands open a store several times)
            static WARNED: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());
            let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
            if !warned.iter().any(|p| p == store.path()) {
                warned.push(store.path().to_path_buf());
                eprintln!("warning: {name} was written by {what} (this is version {version}, {kind}); reading only");
            }
        }
        Ok(())
    }
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
    use crate::Backend;
    use tilestore::TileData;

    fn cfg(kind: GeneratorKind) -> Config {
        Config { generator: kind, tile_supersample: 1, ..Config::default() }
    }

    /// A tile with every layer (any content: the stores' checks look at their metadata).
    fn dummy_tile(id: geodesy::tiles::TileId) -> TileData {
        let n = crate::TILE_SIZE * crate::TILE_SIZE;
        TileData {
            id,
            rgb: vec![0; 3 * n],
            albedo: vec![0; 3 * n],
            elevation: vec![0.0; n],
            normal: vec![0; 3 * n],
            landcover: vec![1; n],
            emission: vec![0; 3 * n],
            elev_min: 0.0,
            elev_max: 0.0,
        }
    }

    fn temp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("terragen-store-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_store_keeps_its_world() {
        for kind in GeneratorKind::ALL {
            let dir = temp(&format!("world-{kind}"));
            let path = dir.join("w.h5");
            let g1 = Generator::with_backend(cfg(kind), Backend::Cpu);
            let st = g1.open_store_rw(&path).unwrap();
            assert_eq!(st.meta().generator_version, kind.version());
            st.write_tiles(&[g1.tile(geodesy::tiles::TileId::new(2, 1, 1))]).unwrap();
            drop(st);
            // same world: fine; another seed or setting: refused, naming what differs
            assert!(g1.open_store_rw(&path).is_ok());
            let g7 = Generator::with_backend(Config { seed: 7, ..cfg(kind) }, Backend::Cpu);
            let e = match g7.open_store_rw(&path) {
                Ok(_) => panic!("a store of seed 1 accepted seed 7"),
                Err(e) => e.to_string(),
            };
            assert!(e.contains("world.seed (1 → 7)"), "{e}");
            assert!(g7.open_store_ro(&path).is_err());
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A store of one generator refuses the other, saying which generator it holds.
    #[test]
    fn a_store_keeps_its_generator() {
        let dir = temp("kind");
        let id = geodesy::tiles::TileId::new(3, 2, 2);
        for kind in GeneratorKind::ALL {
            let path = dir.join(format!("{kind}.h5"));
            let g = Generator::with_backend(cfg(kind), Backend::Cpu);
            let st = g.open_store_rw(&path).unwrap();
            st.write_tiles(&[dummy_tile(id)]).unwrap();
            drop(st);
            assert!(TileStore::open(&path).unwrap().meta().generator_config.contains(&format!("generator: {kind}")));
            let other = GeneratorKind::ALL.into_iter().find(|&k| k != kind).unwrap();
            let go = Generator::with_backend(cfg(other), Backend::Cpu);
            for e in [go.open_store_rw(&path).err(), go.open_store_ro(&path).err()] {
                let e = e.expect("a store of the other generator opened").to_string();
                assert!(e.contains(&format!("holds tiles of the {kind} generator")) && e.contains(&format!("world.generator: {kind}")), "{e}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Stores written before the switch (no `generator` key): v0.2.0's (version 3) hold classic
    /// worlds, open with the default config and take more tiles; version 4 ones hold next worlds.
    #[test]
    fn stores_without_a_generator_key() {
        let dir = temp("legacy");
        let id = geodesy::tiles::TileId::new(3, 2, 2);
        let classic = Generator::with_backend(cfg(GeneratorKind::Classic), Backend::Cpu);
        let next = Generator::with_backend(cfg(GeneratorKind::Next), Backend::Cpu);
        let strip = |g: &Generator| {
            let mut v = world_value(g.config());
            if let Value::Mapping(m) = &mut v {
                m.remove("generator");
            }
            serde_yaml::to_string(&v).unwrap()
        };
        // v0.2.0: version 3, its config (the settings v0.2.0 had)
        let v020 = dir.join("v020.h5");
        let st = TileStore::create(&v020, StoreMeta { generator_config: strip(&classic), generator_version: 3, ..classic.store_meta() }).unwrap();
        st.write_tiles(&[dummy_tile(id)]).unwrap();
        drop(st);
        assert!(!TileStore::open(&v020).unwrap().meta().generator_config.contains("atlas"));
        classic.open_store_rw(&v020).unwrap();
        let e = next.open_store_ro(&v020).err().expect("a v0.2.0 store opened with next").to_string();
        assert!(e.contains("holds tiles of the classic generator"), "{e}");
        // master before the switch: version 4
        let v4 = dir.join("v4.h5");
        let st = TileStore::create(&v4, StoreMeta { generator_config: strip(&next), generator_version: 4, ..next.store_meta() }).unwrap();
        st.write_tiles(&[dummy_tile(id)]).unwrap();
        drop(st);
        next.open_store_rw(&v4).unwrap();
        let e = classic.open_store_ro(&v4).err().expect("a version-4 store opened with classic").to_string();
        assert!(e.contains("holds tiles of the next generator"), "{e}");
        // the next generator's own settings are not part of a classic world
        let mut c = cfg(GeneratorKind::Classic);
        c.atlas.plates = 9;
        c.relief.belt_mountains = 0.1;
        Generator::with_backend(c, Backend::Cpu).open_store_rw(&v020).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_store_takes_the_new_worlds_ellipsoid() {
        let dir = temp("ell");
        let path = dir.join("w.h5");
        let mut sphere = cfg(GeneratorKind::Classic);
        sphere.planet = crate::config::Planet { a: 6371000.0, inv_f: 0.0 };
        let gs = Generator::with_backend(sphere, Backend::Cpu);
        drop(gs.open_store_rw(&path).unwrap());
        assert_eq!(TileStore::open(&path).unwrap().meta().ellipsoid_b, 6371000.0);
        // re-used (empty) for the default world: re-stamped with its ellipsoid too
        let g = Generator::with_backend(cfg(GeneratorKind::Classic), Backend::Cpu);
        drop(g.open_store_rw(&path).unwrap());
        let st = TileStore::open(&path).unwrap();
        let e = g.ell();
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
