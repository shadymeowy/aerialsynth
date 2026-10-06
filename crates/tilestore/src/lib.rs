//! XYZ tile pyramid stored in a single HDF5 file.
//!
//! Layout (format version 1):
//! ```text
//! /                      attrs: format="terrain-tiles", format_version, tile_size, scheme="xyz",
//!                               projection="EPSG:3857", ellipsoid_a, ellipsoid_b,
//!                               vertical_datum="ellipsoid", generator_config (yaml), seed
//! /levels/<z>/index      i32 [N,2]   (x, y) of row n
//! /levels/<z>/elev_range f32 [N,2]   (min, max) of the elevation layer of row n
//! /levels/<z>/<layer>    [N,256,256(,C)] chunk = one tile, shuffle + deflate
//!     rgb        u8  x3  satellite look (baked lighting), sRGB
//!     albedo     u8  x3  surface albedo, sRGB encoded
//!     elevation  f32     DSM, meters above the ellipsoid, at pixel centres
//!     normal     i8  x3  unit normal (east, north, up) * 127
//!     landcover  u8      class id (see terragen::landcover)
//!     emission   u8  x3  night-time artificial light, linear radiance = 4 * (v/255)^2.2
//! ```
//! Rows are appended in any order, so the file can be grown lazily (generate only the tiles a
//! trajectory needs, add more later). Tile pixels are pixel-centre registered: pixel (i, j) of
//! tile (z, x, y) covers global pixel (256 x + i, 256 y + j) at zoom z, row 0 = north.

mod codec;
mod store;

pub use geodesy::tiles::TileId;
pub use store::{StoreMeta, TileStore};

pub const TILE_SIZE: usize = 256;
pub const FORMAT: &str = "terrain-tiles";
pub const FORMAT_VERSION: i32 = 1;

/// A data layer of a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Layer {
    Rgb,
    Albedo,
    Elevation,
    Normal,
    Landcover,
    /// Night-time artificial light emission, u8 x3: linear radiance = 4 * (v/255)^2.2
    Emission,
}

impl Layer {
    pub const ALL: [Layer; 6] = [Layer::Rgb, Layer::Albedo, Layer::Elevation, Layer::Normal, Layer::Landcover, Layer::Emission];

    pub fn name(self) -> &'static str {
        match self {
            Layer::Rgb => "rgb",
            Layer::Albedo => "albedo",
            Layer::Elevation => "elevation",
            Layer::Normal => "normal",
            Layer::Landcover => "landcover",
            Layer::Emission => "emission",
        }
    }
    pub fn from_name(s: &str) -> Option<Layer> {
        Layer::ALL.into_iter().find(|l| l.name() == s)
    }
    pub fn channels(self) -> usize {
        match self {
            Layer::Rgb | Layer::Albedo | Layer::Normal | Layer::Emission => 3,
            _ => 1,
        }
    }
    /// Bytes per element (per channel).
    pub fn elem_size(self) -> usize {
        match self {
            Layer::Elevation => 4,
            _ => 1,
        }
    }
    pub fn tile_bytes(self) -> usize {
        TILE_SIZE * TILE_SIZE * self.channels() * self.elem_size()
    }
}

/// All layers of one tile (row-major, row 0 = north edge). A layer that was not loaded is empty.
#[derive(Clone, Debug, Default)]
pub struct TileData {
    pub id: TileId,
    pub rgb: Vec<u8>,
    pub albedo: Vec<u8>,
    pub elevation: Vec<f32>,
    pub normal: Vec<i8>,
    pub landcover: Vec<u8>,
    pub emission: Vec<u8>,
    pub elev_min: f32,
    pub elev_max: f32,
}

impl TileData {
    /// Drop the layers not in `keep` (to save memory in caches).
    pub fn retain_layers(&mut self, keep: &[Layer]) {
        for l in Layer::ALL {
            if !keep.contains(&l) {
                match l {
                    Layer::Rgb => self.rgb = vec![],
                    Layer::Albedo => self.albedo = vec![],
                    Layer::Elevation => self.elevation = vec![],
                    Layer::Normal => self.normal = vec![],
                    Layer::Landcover => self.landcover = vec![],
                    Layer::Emission => self.emission = vec![],
                }
            }
        }
    }

    pub fn has(&self, l: Layer) -> bool {
        match l {
            Layer::Rgb => !self.rgb.is_empty(),
            Layer::Albedo => !self.albedo.is_empty(),
            Layer::Elevation => !self.elevation.is_empty(),
            Layer::Normal => !self.normal.is_empty(),
            Layer::Landcover => !self.landcover.is_empty(),
            Layer::Emission => !self.emission.is_empty(),
        }
    }

    /// Raw little-endian bytes of a layer.
    pub fn layer_bytes(&self, l: Layer) -> &[u8] {
        fn cast<T>(v: &[T]) -> &[u8] {
            // SAFETY: plain-old-data numeric slices reinterpreted as bytes.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
        }
        match l {
            Layer::Rgb => &self.rgb,
            Layer::Albedo => &self.albedo,
            Layer::Elevation => cast(&self.elevation),
            Layer::Normal => cast(&self.normal),
            Layer::Landcover => &self.landcover,
            Layer::Emission => &self.emission,
        }
    }

    pub fn set_layer_bytes(&mut self, l: Layer, b: Vec<u8>) {
        match l {
            Layer::Rgb => self.rgb = b,
            Layer::Albedo => self.albedo = b,
            Layer::Landcover => self.landcover = b,
            Layer::Emission => self.emission = b,
            Layer::Normal => self.normal = b.into_iter().map(|x| x as i8).collect(),
            Layer::Elevation => {
                self.elevation = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            }
        }
    }

    /// Elevation at pixel (i, j), clamped to the tile.
    #[inline]
    pub fn elev(&self, i: isize, j: isize) -> f32 {
        let i = i.clamp(0, TILE_SIZE as isize - 1) as usize;
        let j = j.clamp(0, TILE_SIZE as isize - 1) as usize;
        self.elevation[j * TILE_SIZE + i]
    }
}
