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
