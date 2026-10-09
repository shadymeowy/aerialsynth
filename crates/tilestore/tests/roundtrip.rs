use tilestore::{Layer, StoreMeta, TileData, TileId, TileStore, TILE_SIZE};

fn tile(id: TileId, k: u8) -> TileData {
    let n = TILE_SIZE * TILE_SIZE;
    TileData {
        id,
        rgb: (0..n * 3).map(|i| (i as u8).wrapping_mul(k)).collect(),
        albedo: (0..n * 3).map(|i| (i as u8).wrapping_add(k)).collect(),
        elevation: (0..n).map(|i| i as f32 * 0.25 - 100.0 + k as f32).collect(),
        normal: (0..n * 3).map(|i| (i % 255) as i8).collect(),
        landcover: (0..n).map(|i| (i % 17) as u8).collect(),
        emission: (0..n * 3).map(|i| (i % 7) as u8).collect(),
        elev_min: -100.0,
        elev_max: 1000.0 + k as f32,
    }
}

#[test]
fn write_reopen_append_read() {
    let dir = std::env::temp_dir().join(format!("tilestore-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.h5");
    {
        let s = TileStore::create(&path, StoreMeta::default()).unwrap();
        s.write_tiles(&[tile(TileId::new(3, 1, 2), 1), tile(TileId::new(3, 2, 2), 2), tile(TileId::new(5, 9, 9), 3)]).unwrap();
        assert_eq!(s.len(), 3);
    }
    {
        let s = TileStore::open_rw(&path).unwrap();
        assert_eq!(s.len(), 3);
        // overwrite one, append one
        s.write_tiles(&[tile(TileId::new(3, 1, 2), 7), tile(TileId::new(3, 0, 0), 4)]).unwrap();
        assert_eq!(s.len(), 4);
    }
    let s = TileStore::open(&path).unwrap();
    assert_eq!(s.zooms(), vec![3, 5]);
    let t = s.read_tile(TileId::new(3, 1, 2), &Layer::ALL).unwrap().unwrap();
    let e = tile(TileId::new(3, 1, 2), 7);
    assert_eq!(t.rgb, e.rgb);
    assert_eq!(t.albedo, e.albedo);
    assert_eq!(t.elevation, e.elevation);
    assert_eq!(t.normal, e.normal);
    assert_eq!(t.landcover, e.landcover);
    assert_eq!(s.elev_range(TileId::new(3, 1, 2)), Some((-100.0, 1007.0)));
    let t = s.read_tile(TileId::new(5, 9, 9), &[Layer::Elevation]).unwrap().unwrap();
    assert!(t.rgb.is_empty());
    assert_eq!(t.elevation, tile(TileId::new(5, 9, 9), 3).elevation);
    assert!(s.read_tile(TileId::new(4, 0, 0), &Layer::ALL).unwrap().is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_invalid_tiles() {
    let dir = std::env::temp_dir().join(format!("tilestore-invalid-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let s = TileStore::create(dir.join("t.h5"), StoreMeta::default()).unwrap();
    // short layer buffer, missing layer, out-of-range tile id
    let mut short = tile(TileId::new(3, 1, 1), 1);
    short.elevation.truncate(10);
    assert!(s.write_tiles(&[short]).is_err());
    let mut missing = tile(TileId::new(3, 1, 1), 1);
    missing.rgb.clear();
    assert!(s.write_tiles(&[missing]).is_err());
    assert!(s.write_tiles(&[tile(TileId::new(2, 100, 100), 1)]).is_err());
    // nothing was stored, and valid writes still work
    assert_eq!(s.len(), 0);
    s.write_tiles(&[tile(TileId::new(3, 1, 1), 1)]).unwrap();
    assert!(s.read_tile(TileId::new(3, 1, 1), &Layer::ALL).unwrap().is_some());
}

fn test_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tilestore-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_new_store_opens_before_any_write() {
    let dir = test_dir("fresh");
    let path = dir.join("t.h5");
    drop(TileStore::create(&path, StoreMeta::default()).unwrap());
    // (built as t.h5.tmp and renamed: no temporary file is left)
    assert!(!dir.join("t.h5.tmp").exists());
    let s = TileStore::open(&path).unwrap();
    assert!(s.is_empty());
    assert_eq!(s.meta().layers, Layer::ALL.to_vec());
    drop(s);
    let s = TileStore::open_rw(&path).unwrap();
    s.write_tile(&tile(TileId::new(2, 1, 1), 1)).unwrap();
    assert_eq!(s.len(), 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_truncated_store_is_reported_as_incomplete() {
    let dir = test_dir("truncated");
    let path = dir.join("t.h5");
    drop(TileStore::create(&path, StoreMeta::default()).unwrap());
    // what a process killed during creation left behind (HDF5 superblock only)
    let head = std::fs::read(&path).unwrap()[..48].to_vec();
    let cut = dir.join("cut.h5");
    std::fs::write(&cut, head).unwrap();
    for e in [TileStore::open(&cut).err(), TileStore::open_rw(&cut).err()] {
        let e = format!("{:#}", e.expect("a truncated store opened"));
        assert!(e.contains("looks incomplete") && e.contains("can be deleted"), "{e}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_partial_level_is_skipped_and_completed() {
    let dir = test_dir("partial");
    let path = dir.join("t.h5");
    {
        let s = TileStore::create(&path, StoreMeta::default()).unwrap();
        s.write_tile(&tile(TileId::new(3, 1, 2), 1)).unwrap();
        // a level whose creation was interrupted: an index and one layer, no other layers
        let g = s.file().ensure_group("levels/5").unwrap();
        g.new_dataset::<i32>().shape(&[0, 2]).max_shape(&[None, Some(2)]).chunk(&[1024, 2]).create("index").unwrap();
        g.new_dataset::<f32>().shape(&[0, 2]).max_shape(&[None, Some(2)]).chunk(&[1024, 2]).create("elev_range").unwrap();
        g.new_dataset::<u8>().shape(&[0, 256, 256, 3]).max_shape(&[None, Some(256), Some(256), Some(3)]).chunk(&[1, 256, 256, 3]).create("rgb").unwrap();
        s.flush().unwrap();
    }
    let s = TileStore::open(&path).unwrap();
    assert_eq!(s.zooms(), vec![3]);
    assert!(s.read_tile(TileId::new(5, 1, 1), &Layer::ALL).unwrap().is_none());
    drop(s);
    // writing to that zoom completes the level (no panic on its missing layers)
    let s = TileStore::open_rw(&path).unwrap();
    s.write_tile(&tile(TileId::new(5, 1, 1), 5)).unwrap();
    drop(s);
    let s = TileStore::open(&path).unwrap();
    assert_eq!(s.zooms(), vec![3, 5]);
    let t = s.read_tile(TileId::new(5, 1, 1), &Layer::ALL).unwrap().unwrap();
    assert_eq!(t.elevation, tile(TileId::new(5, 1, 1), 5).elevation);
    assert_eq!(t.emission, tile(TileId::new(5, 1, 1), 5).emission);
    std::fs::remove_dir_all(&dir).ok();
}
