//! Point queries (`Generator::terrain_points`, used by the `ground` example and `showcase/route.py`)
//! describe the same world as the tiles of the configured generator: for both generators the
//! surface at tile pixels matches the tile's elevation, and the two worlds differ (so querying one
//! generator for the other's tiles would fail this test).
use geodesy::tiles::{pixel_to_latlon, tile_for_latlon, TileId};
use glam::DVec2;
use terragen::{Backend, Config, Generator, GeneratorKind, TILE_SIZE};

const PLACES: [(f64, f64); 6] = [(46.5, 8.0), (39.9, 32.8), (30.0, 90.0), (-20.0, 140.0), (40.0, -105.0), (5.0, 20.0)];
const Z: u8 = 12;

fn generator(kind: GeneratorKind) -> Generator {
    Generator::with_backend(Config { generator: kind, tile_supersample: 1, ..Config::default() }, Backend::Cpu)
}

/// Tile pixels (a 4 x 4 grid) at each place: (tile, pixel index, lat, lon, pixel size in m).
fn samples() -> Vec<(TileId, usize, f64, f64, f64)> {
    let n = TILE_SIZE;
    let mut out = vec![];
    for (lat, lon) in PLACES {
        let id = tile_for_latlon(lat.to_radians(), lon.to_radians(), Z);
        for j in 0..4 {
            for i in 0..4 {
                let (px, py) = (n / 8 + i * n / 4, n / 8 + j * n / 4);
                let g = DVec2::new((id.x as usize * n + px) as f64 + 0.5, (id.y as usize * n + py) as f64 + 0.5);
                let (la, lo) = pixel_to_latlon(g, Z, TILE_SIZE as u32);
                let gsd = 2.0 * std::f64::consts::PI * 6_378_137.0 * la.cos() / ((1u64 << Z) as f64 * n as f64);
                out.push((id, py * n + px, la, lo, gsd));
            }
        }
    }
    out
}

/// (median, mean) |tile elevation − point surface| of `tiles_of`'s tiles against `points_of`'s
/// point queries.
fn gaps(tiles_of: &Generator, points_of: &Generator) -> (f64, f64) {
    let s = samples();
    let pts: Vec<(f64, f64, f64)> = s.iter().map(|&(_, _, la, lo, gsd)| (la, lo, gsd)).collect();
    let surf = points_of.terrain_points(&pts).unwrap();
    let mut tiles = std::collections::HashMap::new();
    let mut g: Vec<f64> = s
        .iter()
        .zip(&surf)
        .map(|(&(id, k, ..), p)| {
            let t = tiles.entry(id).or_insert_with(|| tiles_of.tile_cpu(id));
            (t.elevation[k] as f64 - p.surface()).abs()
        })
        .collect();
    g.sort_by(f64::total_cmp);
    (g[g.len() / 2], g.iter().sum::<f64>() / g.len() as f64)
}

#[test]
fn point_queries_match_the_configured_generators_tiles() {
    let classic = generator(GeneratorKind::Classic);
    let next = generator(GeneratorKind::Next);
    // the default world (what `terrain run` and the `ground` example use) is classic
    assert_eq!(Config::default().generator, GeneratorKind::Classic);
    // own points vs own tiles: within what stands on the ground (trees, buildings)
    let (cc, nn) = (gaps(&classic, &classic), gaps(&next, &next));
    assert!(cc.0 < 30.0, "classic points vs classic tiles: median gap {:.1} m", cc.0);
    assert!(nn.0 < 30.0, "next points vs next tiles: median gap {:.1} m", nn.0);
    // the worlds differ (on land; the sea is at 0 m in both): the other generator's points
    // don't match the tiles
    let cn = gaps(&classic, &next).1;
    let own = cc.1.max(nn.1).max(10.0);
    assert!(cn > 4.0 * own, "classic tiles vs next points: mean gap {cn:.1} m (own: {:.1}, {:.1} m)", cc.1, nn.1);
}
