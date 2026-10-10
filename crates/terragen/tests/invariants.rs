//! Generator invariants: determinism, seamless tile borders, LOD consistency.
use geodesy::tiles::{tile_for_latlon, TileId};
use terragen::{Config, Generator, TILE_SIZE};

fn gen() -> Generator {
    Generator::new(Config { tile_supersample: 1, ..Config::default() })
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

/// The normals of a tile's bottom row (from pass B's apron) agree with the pixels of its
/// southern neighbour, i.e. both tiles produce the same terrain there. Rows 5609 / 5610 at z14
/// straddle 49.2°N, where the switch between grid-interpolated and exact gully inputs used to be
/// decided per tile from its centre latitude (mountains near 178°W: steps of up to 17 m).
#[test]
fn seamless_north_south() {
    use geodesy::tiles::{gsd_ew, gsd_ns, pixel_to_latlon};
    use glam::{DVec2, DVec3};
    let g = gen();
    let ell = g.world.ell;
    let n = TILE_SIZE;
    let (z, x, y) = (14u8, 91u32, 5609u32);
    let a = g.tile(TileId::new(z, x, y));
    let b = g.tile(TileId::new(z, x, y + 1));
    let h = |t: &terragen::TileData, i: usize, j: usize| t.elevation[j * n + i] as f64;
    let j = n - 1;
    let (lat, _) = pixel_to_latlon(DVec2::new(0.0, y as f64 * n as f64 + j as f64 + 0.5), z, n as u32);
    let (gx, gy) = (gsd_ew(lat, z, n as u32, &ell), gsd_ns(lat, z, n as u32, &ell));
    let mut bad = 0;
    for i in 1..n - 1 {
        // the normal as the tile computes it (ENU, central differences), with the neighbour's
        // first row in place of the apron
        let v = DVec3::new(-(h(&a, i + 1, j) - h(&a, i - 1, j)) / (2.0 * gx), (h(&b, i, 0) - h(&a, i, j - 1)) / (2.0 * gy), 1.0).normalize();
        let k = 3 * (j * n + i);
        let d = (0..3).map(|c| ((v[c] * 127.0).round() as i32 - a.normal[k + c] as i32).abs()).max().unwrap();
        if d > 1 {
            bad += 1;
        }
    }
    assert!(bad <= 2, "{bad} of {} bottom-edge normals disagree with the southern neighbour", n - 2);
}

/// Parent ≈ mean of children for colour and land cover: the mean albedo of each 2 × 2 block of
/// children vs the parent pixel, and the land-cover group shares of the whole tile (places
/// with farmland, forest and towns; a coarse zoom where ecoregions blend).
#[test]
fn lod_consistency_of_colour_and_classes() {
    let g = gen();
    for &(lat, lon, z) in &[(39.9f64, 32.8f64, 12u8), (43.6, 116.5, 13), (6.95, -71.65, 9), (40.9, 33.9, 6)] {
        let parent = tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
        let p = g.tile(parent);
        let kids = parent.children().map(|c| g.tile(c));
        let n = TILE_SIZE;
        let lin = |v: u8| ((v as f64 / 255.0 + 0.055) / 1.055).powf(2.4);
        // block means (8 parent pixels = 16 child pixels): the bias of the coarse look, not
        // the sub-pixel texture
        let (mut err, mut cnt) = (0.0, 0.0);
        let mut groups_p = [0.0f64; 11];
        let mut groups_k = [0.0f64; 11];
        let gi = |c: u8| terragen::landcover::group(c) as usize;
        const B: usize = 8;
        for (k, kid) in kids.iter().enumerate() {
            let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
            for bj in 0..n / (2 * B) {
                for bi in 0..n / (2 * B) {
                    for c in 0..3 {
                        let (mut mk, mut mp) = (0.0, 0.0);
                        for j in 0..2 * B {
                            for i in 0..2 * B {
                                mk += lin(kid.albedo[3 * ((bj * 2 * B + j) * n + bi * 2 * B + i) + c]);
                            }
                        }
                        for j in 0..B {
                            for i in 0..B {
                                mp += lin(p.albedo[3 * ((oy + bj * B + j) * n + ox + bi * B + i) + c]);
                            }
                        }
                        err += (mk / (4 * B * B) as f64 - mp / (B * B) as f64).abs();
                        cnt += 1.0;
                    }
                }
            }
            for c in &kid.landcover {
                groups_k[gi(*c)] += 0.25 / (n * n) as f64;
            }
        }
        for c in &p.landcover {
            groups_p[gi(*c)] += 1.0 / (n * n) as f64;
        }
        let mean_err = err / cnt;
        assert!(mean_err < 0.006, "z{z} at {lat},{lon}: mean |parent − avg(children)| albedo over 8-pixel blocks {mean_err:.4}");
        for (k, (a, b)) in groups_p.iter().zip(&groups_k).enumerate() {
            assert!((a - b).abs() < 0.08, "z{z} at {lat},{lon}: land-cover group {k} share {a:.3} (parent) vs {b:.3} (children)");
        }
    }
}
