//! Generator invariants: determinism, seamless tile borders, LOD consistency.
use geodesy::tiles::{tile_for_latlon, TileId};
use terragen::{Config, Generator, TILE_SIZE};

fn gen() -> Generator {
    Generator::new(Config { supersample: 1, ..Config::default() })
}

#[test]
fn deterministic() {
    let g = gen();
    let id = tile_for_latlon(39.9f64.to_radians(), 32.8f64.to_radians(), 14);
    let a = g.tile(id);
    let b = g.tile(id);
    assert_eq!(a.rgb, b.rgb);
    assert_eq!(a.elevation, b.elevation);
    // a fresh generator (empty caches) gives the same result
    let c = gen().tile(id);
    assert_eq!(a.elevation, c.elevation);
    assert_eq!(a.albedo, c.albedo);
}

#[test]
fn seamless_borders() {
    let g = gen();
    for z in [11u8, 15] {
        let id = tile_for_latlon(39.95f64.to_radians(), 32.85f64.to_radians(), z);
        let right = TileId::new(z, id.x + 1, id.y);
        let a = g.tile(id);
        let b = g.tile(right);
        // last column of `a` vs first column of `b`: neighbouring pixels → small differences
        // relative to the typical pixel-to-pixel difference inside the tiles
        let n = TILE_SIZE;
        let mut across = 0.0;
        let mut inside = 0.0;
        for j in 0..n {
            across += (a.elevation[j * n + n - 1] - b.elevation[j * n]).abs() as f64;
            inside += (a.elevation[j * n + n - 2] - a.elevation[j * n + n - 1]).abs() as f64;
        }
        assert!(across < 3.0 * inside + 1.0 * n as f64, "z{z}: across {across} inside {inside}");
    }
}

#[test]
fn lod_consistency() {
    let g = gen();
    let parent = tile_for_latlon(39.9f64.to_radians(), 32.8f64.to_radians(), 12);
    let p = g.tile(parent);
    let kids = parent.children().map(|c| g.tile(c));
    let n = TILE_SIZE;
    let mut err = 0.0;
    let mut cnt = 0.0;
    for (k, kid) in kids.iter().enumerate() {
        let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
        for j in (0..n).step_by(2) {
            for i in (0..n).step_by(2) {
                let m = (kid.elevation[j * n + i] + kid.elevation[j * n + i + 1] + kid.elevation[(j + 1) * n + i] + kid.elevation[(j + 1) * n + i + 1]) / 4.0;
                let pv = p.elevation[(oy + j / 2) * n + ox + i / 2];
                err += (m - pv).abs() as f64;
                cnt += 1.0;
            }
        }
    }
    let mean_err = err / cnt;
    assert!(mean_err < 6.0, "mean |parent - avg(children)| = {mean_err} m");
}
