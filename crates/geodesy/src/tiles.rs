//! Web-Mercator (EPSG:3857) XYZ "slippy map" tile math (Google / OSM scheme).
//!
//! Conventions:
//!
//! * The *spherical* Mercator formulas are applied directly to **WGS84 geodetic**
//!   latitude/longitude (this is what EPSG:3857 does). Angles are radians.
//! * Normalized Mercator coordinates `(u, v)`: `u ∈ [0, 1]` grows **east** from the
//!   antimeridian (`lon = -π`), `v ∈ [0, 1]` grows **south** from `lat = +MAX_MERCATOR_LAT`.
//! * Zoom `z` has `2^z × 2^z` tiles; tile `(x, y)` covers `u ∈ [x, x+1) / 2^z`,
//!   `v ∈ [y, y+1) / 2^z` (x east, y south, `(0, 0)` is the north-west corner).
//! * Global pixel coordinates at zoom `z` for tile size `ts` are `(u, v) · ts · 2^z`; pixel
//!   `(i, j)` covers `[i, i+1) × [j, j+1)`, so its centre is at `(i + 0.5, j + 0.5)`.

use std::f64::consts::{PI, TAU};
use std::fmt;

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::ellipsoid::Ellipsoid;

/// Latitude limit of Web Mercator in **degrees**: `atan(sinh(π))`, where the square world
/// map ends (`v = 0` / `v = 1`).
pub const MAX_MERCATOR_LAT: f64 = 85.051_128_779_806_59;
/// Latitude limit of Web Mercator in **radians**.
pub const MAX_MERCATOR_LAT_RAD: f64 = MAX_MERCATOR_LAT * (PI / 180.0);
/// Sphere radius used by EPSG:3857 (m) (the WGS84 semi-major axis).
pub const WEB_MERCATOR_R: f64 = 6_378_137.0;
/// Largest supported zoom level (keeps all tile/pixel arithmetic exact in `u32`/`f64`).
pub const MAX_ZOOM: u8 = 30;

#[inline]
fn tiles_per_side(z: u8) -> u64 {
    1u64 << z
}

/// An XYZ tile address.
#[derive(Clone, Default, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TileId {
    /// Zoom level (`0..=MAX_ZOOM`).
    pub z: u8,
    /// Column, `0..2^z`, growing east.
    pub x: u32,
    /// Row, `0..2^z`, growing south.
    pub y: u32,
}

impl TileId {
    /// New tile id (not validated; see [`TileId::is_valid`]).
    #[inline]
    pub fn new(z: u8, x: u32, y: u32) -> Self {
        Self { z, x, y }
    }

    /// `true` if `z <= MAX_ZOOM` and `x, y < 2^z`.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.z <= MAX_ZOOM && (self.x as u64) < tiles_per_side(self.z) && (self.y as u64) < tiles_per_side(self.z)
    }

    /// Parent tile at `z - 1`, or `None` at zoom 0.
    #[inline]
    pub fn parent(&self) -> Option<TileId> {
        (self.z > 0).then(|| TileId::new(self.z - 1, self.x >> 1, self.y >> 1))
    }

    /// The four children at `z + 1`, ordered `(2x,2y), (2x+1,2y), (2x,2y+1), (2x+1,2y+1)`
    /// (NW, NE, SW, SE).
    ///
    /// # Panics
    /// If `self.z >= MAX_ZOOM`.
    pub fn children(&self) -> [TileId; 4] {
        assert!(self.z < MAX_ZOOM, "children of zoom {} exceed MAX_ZOOM", self.z);
        let (z, x, y) = (self.z + 1, 2 * self.x, 2 * self.y);
        [TileId::new(z, x, y), TileId::new(z, x + 1, y), TileId::new(z, x, y + 1), TileId::new(z, x + 1, y + 1)]
    }

    /// Ancestor at zoom `z` (the tile itself if `z == self.z`).
    ///
    /// # Panics
    /// If `z > self.z`.
    pub fn ancestor(&self, z: u8) -> TileId {
        assert!(z <= self.z, "ancestor zoom {z} > tile zoom {}", self.z);
        let s = self.z - z;
        TileId::new(z, self.x >> s, self.y >> s)
    }

    /// Geographic bounds of the tile (radians).
    pub fn bounds(&self) -> LatLonBounds {
        let n = tiles_per_side(self.z) as f64;
        let (lat_max, lon_min) = uv_to_latlon(DVec2::new(self.x as f64 / n, self.y as f64 / n));
        let (lat_min, lon_max) = uv_to_latlon(DVec2::new((self.x as f64 + 1.0) / n, (self.y as f64 + 1.0) / n));
        LatLonBounds { lat_min, lat_max, lon_min, lon_max }
    }

    /// `(lat, lon)` in radians of the tile centre **in Mercator space** (the image centre).
    /// Its latitude is slightly poleward of the mean of the latitude bounds.
    pub fn center(&self) -> (f64, f64) {
        let n = tiles_per_side(self.z) as f64;
        uv_to_latlon(DVec2::new((self.x as f64 + 0.5) / n, (self.y as f64 + 0.5) / n))
    }

    /// Tile offset by `(dx, dy)` at the same zoom. `x` wraps around the antimeridian; returns
    /// `None` if `y + dy` leaves `[0, 2^z)` or `self` is invalid.
    pub fn neighbor(&self, dx: i32, dy: i32) -> Option<TileId> {
        if !self.is_valid() {
            return None;
        }
        let n = tiles_per_side(self.z) as i64;
        let y = self.y as i64 + dy as i64;
        if !(0..n).contains(&y) {
            return None;
        }
        let x = (self.x as i64 + dx as i64).rem_euclid(n);
        Some(TileId::new(self.z, x as u32, y as u32))
    }

    /// Bing-maps quadkey (`""` at zoom 0). Digit = `xbit + 2·ybit`, most significant first.
    pub fn quadkey(&self) -> String {
        (1..=self.z)
            .rev()
            .map(|i| {
                let mask = 1u32 << (i - 1);
                let d = u8::from(self.x & mask != 0) + 2 * u8::from(self.y & mask != 0);
                char::from(b'0' + d)
            })
            .collect()
    }
}

impl fmt::Display for TileId {
    /// Formats as `z/x/y`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.z, self.x, self.y)
    }
}

/// A latitude/longitude box in radians. `lon_min > lon_max` denotes a box crossing the
/// antimeridian.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatLonBounds {
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
}

/// Geodetic `(lat, lon)` (rad) → normalized Mercator `(u, v)`.
///
/// `u = (lon + π) / 2π`, `v = 1/2 − asinh(tan lat) / 2π`. No clamping or wrapping is done:
/// inputs with `|lat| ≤ MAX_MERCATOR_LAT_RAD`, `lon ∈ [-π, π]` map into `[0, 1]²`; other
/// inputs are extrapolated (`lat = ±π/2` gives `v = ∓∞`).
#[inline]
pub fn latlon_to_uv(lat: f64, lon: f64) -> DVec2 {
    DVec2::new((lon + PI) / TAU, 0.5 - lat.tan().asinh() / TAU)
}

/// Normalized Mercator `(u, v)` → geodetic `(lat, lon)` (rad). Inverse of [`latlon_to_uv`].
#[inline]
pub fn uv_to_latlon(uv: DVec2) -> (f64, f64) {
    let lon = uv.x * TAU - PI;
    let lat = (PI * (1.0 - 2.0 * uv.y)).sinh().atan();
    (lat, lon)
}

/// Geodetic `(lat, lon)` (rad) → global pixel coordinates at zoom `z` for tiles of
/// `tile_size` pixels: `(u, v) · tile_size · 2^z`.
#[inline]
pub fn latlon_to_pixel(lat: f64, lon: f64, z: u8, tile_size: u32) -> DVec2 {
    latlon_to_uv(lat, lon) * world_pixels(z, tile_size)
}

/// Global pixel coordinates → geodetic `(lat, lon)` (rad). Inverse of [`latlon_to_pixel`].
#[inline]
pub fn pixel_to_latlon(px: DVec2, z: u8, tile_size: u32) -> (f64, f64) {
    uv_to_latlon(px / world_pixels(z, tile_size))
}

#[inline]
fn world_pixels(z: u8, tile_size: u32) -> f64 {
    tile_size as f64 * tiles_per_side(z) as f64
}

/// The tile at zoom `z` containing `(lat, lon)` (rad). Latitude is clamped to the Mercator
/// range and longitude is wrapped, so the result is always valid.
///
/// # Panics
/// If `z > MAX_ZOOM`.
pub fn tile_for_latlon(lat: f64, lon: f64, z: u8) -> TileId {
    assert!(z <= MAX_ZOOM, "zoom {z} > MAX_ZOOM");
    let lat = lat.clamp(-MAX_MERCATOR_LAT_RAD, MAX_MERCATOR_LAT_RAD);
    let uv = latlon_to_uv(lat, lon);
    let n = tiles_per_side(z);
    let idx = |t: f64| ((t * n as f64).floor().max(0.0) as u64).min(n - 1) as u32;
    TileId::new(z, idx(uv.x.rem_euclid(1.0)), idx(uv.y))
}

/// Geodetic `(lat, lon)` (rad) → EPSG:3857 `(x, y)` in meters
/// (`x = R·lon`, `y = R·asinh(tan lat)` with `R = WEB_MERCATOR_R`).
#[inline]
pub fn mercator_xy(lat: f64, lon: f64) -> DVec2 {
    DVec2::new(WEB_MERCATOR_R * lon, WEB_MERCATOR_R * lat.tan().asinh())
}

/// EPSG:3857 `(x, y)` meters → geodetic `(lat, lon)` (rad).
#[inline]
pub fn mercator_xy_to_latlon(xy: DVec2) -> (f64, f64) {
    ((xy.y / WEB_MERCATOR_R).sinh().atan(), xy.x / WEB_MERCATOR_R)
}

/// True east-west ground size (m) of one pixel at latitude `lat` (rad), zoom `z`, tile size
/// `ts`, on ellipsoid `ell`: `2π · N(lat) · cos(lat) / (ts · 2^z)` (the parallel's
/// circumference divided by the number of pixels around the world).
pub fn gsd_ew(lat: f64, z: u8, ts: u32, ell: &Ellipsoid) -> f64 {
    TAU * ell.prime_vertical_radius(lat) * lat.cos() / world_pixels(z, ts)
}

/// True north-south ground size (m) of one pixel at latitude `lat` (rad) (meridian arc):
/// `M(lat) · |dlat/dv| / (ts · 2^z)` with `|dlat/dv| = 2π cos(lat)` for Web Mercator, i.e.
/// `2π · M(lat) · cos(lat) / (ts · 2^z)`. On a sphere this equals [`gsd_ew`] (conformal);
/// on WGS84 it is smaller by the factor `M/N`.
pub fn gsd_ns(lat: f64, z: u8, ts: u32, ell: &Ellipsoid) -> f64 {
    TAU * ell.meridian_radius(lat) * lat.cos() / world_pixels(z, ts)
}

/// All tiles at zoom `z` intersecting `b` (radians). Latitudes are clamped to the Mercator
/// range; `lon_min > lon_max` is treated as a box crossing the antimeridian. Boxes that
/// merely touch a tile edge do not include that tile. Output is row-major (by `y`, then `x`
/// going east from `lon_min`).
///
/// # Panics
/// If `z > MAX_ZOOM`.
pub fn tiles_in_bounds(b: &LatLonBounds, z: u8) -> Vec<TileId> {
    assert!(z <= MAX_ZOOM, "zoom {z} > MAX_ZOOM");
    let n = tiles_per_side(z);
    let nf = n as f64;
    // Tolerance (in tiles) absorbing round-off of bounds that coincide with tile edges.
    const EPS: f64 = 1e-9;
    let lo_idx = |t: f64| ((t * nf + EPS).floor().max(0.0) as u64).min(n - 1);
    let hi_idx = |t: f64, lo: u64| (((t * nf - EPS).ceil() - 1.0).max(0.0) as u64).clamp(lo, n - 1);
    let range = |t0: f64, t1: f64| {
        let lo = lo_idx(t0);
        lo..=hi_idx(t1, lo)
    };

    let lat_max = b.lat_max.clamp(-MAX_MERCATOR_LAT_RAD, MAX_MERCATOR_LAT_RAD);
    let lat_min = b.lat_min.clamp(-MAX_MERCATOR_LAT_RAD, MAX_MERCATOR_LAT_RAD);
    if lat_min > lat_max {
        return Vec::new();
    }
    let ys = range(latlon_to_uv(lat_max, 0.0).y, latlon_to_uv(lat_min, 0.0).y);

    let u = |lon: f64| latlon_to_uv(0.0, lon.clamp(-PI, PI)).x;
    // longitudes outside [-π, π] are wrapped (a box from 179.9° to 180.1° crosses the
    // antimeridian); a span of 2π or more is the whole world
    let wrap = |l: f64| (l + PI).rem_euclid(TAU) - PI;
    let (lon_min, lon_max) = if b.lon_min <= b.lon_max && b.lon_max - b.lon_min >= TAU {
        (-PI, PI)
    } else if b.lon_min <= b.lon_max && b.lon_min >= -PI && b.lon_max <= PI {
        (b.lon_min, b.lon_max)
    } else {
        let hi = wrap(b.lon_max);
        (wrap(b.lon_min), if hi == -PI { PI } else { hi })
    };
    let mut xs: Vec<u64> = Vec::new();
    if lon_min <= lon_max {
        xs.extend(range(u(lon_min), u(lon_max)));
    } else {
        xs.extend(range(u(lon_min), 1.0));
        for x in range(0.0, u(lon_max)) {
            if !xs.contains(&x) {
                xs.push(x);
            }
        }
    }

    ys.flat_map(|y| xs.iter().map(move |&x| TileId::new(z, x as u32, y as u32))).collect()
}

/// Fractional zoom at which [`gsd_ew`] at `lat` equals `gsd_m`:
/// `log2(gsd_ew(lat, 0) / gsd_m)` (may be negative or exceed `MAX_ZOOM`).
pub fn zoom_for_gsd_f(lat: f64, gsd_m: f64, ts: u32, ell: &Ellipsoid) -> f64 {
    (gsd_ew(lat, 0, ts, ell) / gsd_m).log2()
}

/// Smallest integer zoom whose [`gsd_ew`] at `lat` is `<= gsd_m`, clamped to `[0, max_z]`
/// (returns `max_z` if even that zoom is too coarse).
pub fn zoom_for_gsd(lat: f64, gsd_m: f64, ts: u32, ell: &Ellipsoid, max_z: u8) -> u8 {
    let max_z = max_z.min(MAX_ZOOM);
    (0..=max_z).find(|&z| gsd_ew(lat, z, ts, ell) <= gsd_m).unwrap_or(max_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::tests::Rng;

    const WGS84: Ellipsoid = Ellipsoid::WGS84;

    /// Standard OSM wiki formula (`deg2num`).
    fn deg2num(lat_deg: f64, lon_deg: f64, z: u8) -> (u32, u32) {
        let n = 2f64.powi(z as i32);
        let lr = lat_deg.to_radians();
        let x = ((lon_deg + 180.0) / 360.0 * n).floor() as u32;
        let y = ((1.0 - lr.tan().asinh() / PI) / 2.0 * n).floor() as u32;
        (x, y)
    }

    #[test]
    fn constants() {
        assert!((MAX_MERCATOR_LAT_RAD - PI.sinh().atan()).abs() < 1e-15);
        assert!((MAX_MERCATOR_LAT - 85.0511287798066).abs() < 1e-12);
        let uv = latlon_to_uv(MAX_MERCATOR_LAT_RAD, -PI);
        assert!(uv.x.abs() < 1e-15 && uv.y.abs() < 1e-14, "{uv:?}");
        let uv = latlon_to_uv(-MAX_MERCATOR_LAT_RAD, PI);
        assert!((uv.x - 1.0).abs() < 1e-15 && (uv.y - 1.0).abs() < 1e-14, "{uv:?}");
    }

    #[test]
    fn root_tile_bounds() {
        let b = TileId::new(0, 0, 0).bounds();
        assert!((b.lat_max - MAX_MERCATOR_LAT_RAD).abs() < 1e-15);
        assert!((b.lat_min + MAX_MERCATOR_LAT_RAD).abs() < 1e-15);
        assert!((b.lon_min + PI).abs() < 1e-15);
        assert!((b.lon_max - PI).abs() < 1e-15);
        let (lat, lon) = TileId::new(0, 0, 0).center();
        assert!(lat.abs() < 1e-15 && lon.abs() < 1e-15);
        // z=1: the four quadrants meet at (0, 0).
        let b = TileId::new(1, 1, 0).bounds();
        assert!(b.lat_min.abs() < 1e-15 && b.lon_min.abs() < 1e-15);
    }

    #[test]
    fn known_tiles() {
        // Ankara (39.92N, 32.85E): via the OSM formula in Python (deg2num).
        let (lat, lon) = (39.92f64.to_radians(), 32.85f64.to_radians());
        assert_eq!(tile_for_latlon(lat, lon, 10), TileId::new(10, 605, 387));
        assert_eq!(tile_for_latlon(lat, lon, 15), TileId::new(15, 19374, 12414));
        // Sydney.
        let t = tile_for_latlon((-33.8688f64).to_radians(), 151.2093f64.to_radians(), 12);
        assert_eq!(t, TileId::new(12, 3768, 2457));
        // Bing documentation example: tile (3, 5) at level 3 has quadkey "213".
        assert_eq!(TileId::new(3, 3, 5).quadkey(), "213");
        assert_eq!(TileId::new(0, 0, 0).quadkey(), "");
        assert_eq!(TileId::new(10, 605, 387).to_string(), "10/605/387");
    }

    #[test]
    fn tile_for_latlon_matches_formula_and_bounds() {
        let mut rng = Rng::new(99);
        for _ in 0..20_000 {
            let lat_deg = rng.uniform(-85.0, 85.0);
            let lon_deg = rng.uniform(-180.0, 180.0);
            let z = (rng.next_u64() % 21) as u8;
            let t = tile_for_latlon(lat_deg.to_radians(), lon_deg.to_radians(), z);
            assert_eq!((t.x, t.y), deg2num(lat_deg, lon_deg, z));
            assert!(t.is_valid());
            let b = t.bounds();
            let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
            assert!(b.lat_min <= lat && lat <= b.lat_max && b.lon_min <= lon && lon <= b.lon_max);
        }
        // Clamping / wrapping.
        assert_eq!(tile_for_latlon(PI / 2.0, 0.0, 4), TileId::new(4, 8, 0));
        assert_eq!(tile_for_latlon(-PI / 2.0, 0.0, 4), TileId::new(4, 8, 15));
        assert_eq!(tile_for_latlon(0.1, PI, 4).x, 0);
        assert_eq!(tile_for_latlon(0.1, -PI, 4).x, 0);
        assert_eq!(tile_for_latlon(0.1, 3.0 * PI - 1e-9, 4).x, 15);
    }

    #[test]
    fn hierarchy() {
        let t = TileId::new(10, 605, 387);
        let c = t.children();
        assert_eq!(c, [TileId::new(11, 1210, 774), TileId::new(11, 1211, 774), TileId::new(11, 1210, 775), TileId::new(11, 1211, 775)]);
        for ch in c {
            assert_eq!(ch.parent(), Some(t));
            assert_eq!(ch.quadkey()[..10], t.quadkey());
        }
        assert_eq!(TileId::new(0, 0, 0).parent(), None);
        assert_eq!(t.ancestor(10), t);
        assert_eq!(t.ancestor(0), TileId::new(0, 0, 0));
        assert_eq!(t.ancestor(8), TileId::new(8, 151, 96));
        // Children tile the parent exactly.
        let b = t.bounds();
        let (nw, se) = (c[0].bounds(), c[3].bounds());
        assert_eq!((nw.lat_max, nw.lon_min), (b.lat_max, b.lon_min));
        assert_eq!((se.lat_min, se.lon_max), (b.lat_min, b.lon_max));
        assert!((nw.lat_min - se.lat_max).abs() < 1e-15);
        assert!(TileId::new(3, 7, 7).is_valid() && !TileId::new(3, 8, 0).is_valid());
        assert!(!TileId::new(MAX_ZOOM + 1, 0, 0).is_valid());
    }

    #[test]
    fn neighbors() {
        let t = TileId::new(3, 0, 0);
        assert_eq!(t.neighbor(-1, 0), Some(TileId::new(3, 7, 0)));
        assert_eq!(t.neighbor(1, 1), Some(TileId::new(3, 1, 1)));
        assert_eq!(t.neighbor(0, -1), None);
        assert_eq!(TileId::new(3, 7, 7).neighbor(1, 0), Some(TileId::new(3, 0, 7)));
        assert_eq!(TileId::new(3, 7, 7).neighbor(0, 1), None);
        assert_eq!(TileId::new(3, 2, 2).neighbor(-17, 3), Some(TileId::new(3, 1, 5)));
        assert_eq!(TileId::new(0, 0, 0).neighbor(5, 0), Some(TileId::new(0, 0, 0)));
        assert_eq!(TileId::new(3, 9, 0).neighbor(0, 0), None);
    }

    #[test]
    fn uv_pixel_mercator_round_trips() {
        let mut rng = Rng::new(1);
        for _ in 0..10_000 {
            let lat = rng.uniform(-MAX_MERCATOR_LAT_RAD, MAX_MERCATOR_LAT_RAD);
            let lon = rng.uniform(-PI, PI);
            let (la, lo) = uv_to_latlon(latlon_to_uv(lat, lon));
            assert!((la - lat).abs() < 1e-14 && (lo - lon).abs() < 1e-15);
            let px = latlon_to_pixel(lat, lon, 18, 256);
            let (la, lo) = pixel_to_latlon(px, 18, 256);
            assert!((la - lat).abs() < 1e-14 && (lo - lon).abs() < 1e-15);
            let (la, lo) = mercator_xy_to_latlon(mercator_xy(lat, lon));
            assert!((la - lat).abs() < 1e-14 && (lo - lon).abs() < 1e-15);
            // The pixel lies inside the tile reported by tile_for_latlon.
            let t = tile_for_latlon(lat, lon, 18);
            let tx = (px / 256.0).floor();
            assert_eq!((tx.x as u32, tx.y as u32), (t.x, t.y));
        }
        // Pixel centres: pixel (0,0) of tile 0/0/0 covers [0,1)².
        let px = latlon_to_pixel(0.0, 0.0, 0, 256);
        assert_eq!(px, DVec2::new(128.0, 128.0));
        // EPSG:3857 reference (Python): Ankara.
        let xy = mercator_xy(39.92f64.to_radians(), 32.85f64.to_radians());
        assert!((xy.x - 3656845.2725590374).abs() < 1e-6);
        assert!((xy.y - 4854323.698611295).abs() < 1e-6);
        // World extent: ±π R.
        let xy = mercator_xy(MAX_MERCATOR_LAT_RAD, PI);
        assert!((xy.x - PI * WEB_MERCATOR_R).abs() < 1e-6 && (xy.y - PI * WEB_MERCATOR_R).abs() < 1e-6);
    }

    #[test]
    fn gsd() {
        let sphere = Ellipsoid::sphere(WEB_MERCATOR_R);
        let g0 = gsd_ew(0.0, 0, 256, &sphere);
        assert!((g0 - 156543.03392804097).abs() < 1e-8);
        // Same on WGS84 at the equator (N(0) = a).
        assert!((gsd_ew(0.0, 0, 256, &WGS84) - 156543.03392804097).abs() < 1e-8);
        // Halves per zoom level; scales with tile size.
        assert!((gsd_ew(0.0, 10, 256, &sphere) - g0 / 1024.0).abs() < 1e-12);
        assert!((gsd_ew(0.0, 0, 512, &sphere) - g0 / 2.0).abs() < 1e-10);
        let lat = 39.92f64.to_radians();
        // Sphere: conformal -> square pixels on the ground, and = R cos(lat) scale.
        assert!((gsd_ew(lat, 12, 256, &sphere) - gsd_ns(lat, 12, 256, &sphere)).abs() < 1e-12);
        assert!((gsd_ew(lat, 0, 256, &sphere) - g0 * lat.cos()).abs() < 1e-8);
        // Ellipsoid: ratio is M/N.
        let r = gsd_ns(lat, 12, 256, &WGS84) / gsd_ew(lat, 12, 256, &WGS84);
        assert!((r - WGS84.meridian_radius(lat) / WGS84.prime_vertical_radius(lat)).abs() < 1e-15);

        // gsd_ns matches the actual meridian length of a pixel (finite difference).
        let z = 14;
        let (i, j) = (9000.0, 6200.0);
        let (la0, _) = pixel_to_latlon(DVec2::new(i, j), z, 256);
        let (la1, _) = pixel_to_latlon(DVec2::new(i, j + 1.0), z, 256);
        let mid = 0.5 * (la0 + la1);
        let arc = WGS84.meridian_radius(mid) * (la0 - la1);
        assert!((arc - gsd_ns(mid, z, 256, &WGS84)).abs() / arc < 1e-8);
        // gsd_ew matches the length of a pixel along the parallel.
        let (la, lo0) = pixel_to_latlon(DVec2::new(i, j), z, 256);
        let (_, lo1) = pixel_to_latlon(DVec2::new(i + 1.0, j), z, 256);
        let arc = WGS84.prime_vertical_radius(la) * la.cos() * (lo1 - lo0);
        assert!((arc - gsd_ew(la, z, 256, &WGS84)).abs() / arc < 1e-9);
    }

    #[test]
    fn zoom_selection() {
        let lat = 39.92f64.to_radians();
        for target in [0.5, 1.0, 2.0, 7.3, 100.0, 1.0e5] {
            let z = zoom_for_gsd(lat, target, 256, &WGS84, 22);
            assert!(gsd_ew(lat, z, 256, &WGS84) <= target);
            if z > 0 {
                assert!(gsd_ew(lat, z - 1, 256, &WGS84) > target);
            }
            let zf = zoom_for_gsd_f(lat, target, 256, &WGS84);
            assert_eq!(zf.ceil() as u8, z, "{zf}");
            assert!((gsd_ew(lat, 0, 256, &WGS84) / 2f64.powf(zf) - target).abs() < 1e-9 * target);
        }
        assert_eq!(zoom_for_gsd(lat, 1e-3, 256, &WGS84, 18), 18);
        assert_eq!(zoom_for_gsd(lat, 1e9, 256, &WGS84, 18), 0);
        // Exact hit.
        let g = gsd_ew(0.3, 13, 256, &WGS84);
        assert_eq!(zoom_for_gsd(0.3, g, 256, &WGS84, 20), 13);
    }

    #[test]
    fn tiles_in_bounds_cases() {
        // A tile's own bounds give exactly that tile.
        for t in [TileId::new(10, 605, 387), TileId::new(0, 0, 0), TileId::new(5, 31, 0), TileId::new(17, 0, 131071)] {
            assert_eq!(tiles_in_bounds(&t.bounds(), t.z), vec![t]);
            // ...and its 4 children one level down.
            let mut v = tiles_in_bounds(&t.bounds(), t.z + 1);
            v.sort();
            let mut c = t.children().to_vec();
            c.sort();
            assert_eq!(v, c);
        }
        // Whole world.
        let world = LatLonBounds { lat_min: -PI / 2.0, lat_max: PI / 2.0, lon_min: -PI, lon_max: PI };
        assert_eq!(tiles_in_bounds(&world, 3).len(), 64);
        // Small box around Ankara at z=10.
        let d = 0.01f64.to_radians();
        let (lat, lon) = (39.92f64.to_radians(), 32.85f64.to_radians());
        let b = LatLonBounds { lat_min: lat - d, lat_max: lat + d, lon_min: lon - d, lon_max: lon + d };
        assert_eq!(tiles_in_bounds(&b, 10), vec![TileId::new(10, 605, 387)]);
        // Antimeridian crossing: 179E .. 179W at z=3 -> x = 7 then 0.
        let b = LatLonBounds { lat_min: 0.1, lat_max: 0.2, lon_min: 179f64.to_radians(), lon_max: (-179f64).to_radians() };
        assert_eq!(tiles_in_bounds(&b, 3), vec![TileId::new(3, 7, 3), TileId::new(3, 0, 3)]);
        // Every reported tile intersects the box; tiles of a random box are complete.
        let mut rng = Rng::new(17);
        for _ in 0..200 {
            let la0 = rng.uniform(-1.4, 1.3);
            let lo0 = rng.uniform(-PI, 3.0);
            let b = LatLonBounds { lat_min: la0, lat_max: la0 + 0.05, lon_min: lo0, lon_max: lo0 + 0.1 };
            let z = 9;
            let v = tiles_in_bounds(&b, z);
            for _ in 0..50 {
                let la = rng.uniform(b.lat_min, b.lat_max);
                let lo = rng.uniform(b.lon_min, b.lon_max.min(PI));
                assert!(v.contains(&tile_for_latlon(la, lo, z)));
            }
            for t in &v {
                let tb = t.bounds();
                assert!(tb.lat_max > b.lat_min && tb.lat_min < b.lat_max);
                assert!(tb.lon_max > b.lon_min && tb.lon_min < b.lon_max);
            }
        }
    }

    #[test]
    fn tiles_in_bounds_wraps_longitudes() {
        let b = |a: f64, c: f64| LatLonBounds { lat_min: 0.1, lat_max: 0.2, lon_min: a.to_radians(), lon_max: c.to_radians() };
        let xs = |v: Vec<TileId>| {
            let mut x: Vec<u32> = v.iter().map(|t| t.x).collect();
            x.sort();
            x.dedup();
            x
        };
        assert_eq!(xs(tiles_in_bounds(&b(179.9, 180.1), 8)), vec![0, 255]);
        assert_eq!(xs(tiles_in_bounds(&b(-180.1, -179.9), 8)), vec![0, 255]);
        assert_eq!(xs(tiles_in_bounds(&b(-200.0, 200.0), 3)).len(), 8);
        assert_eq!(xs(tiles_in_bounds(&b(10.0, 20.0), 8)), xs(tiles_in_bounds(&b(370.0, 380.0), 8)));
    }
}
