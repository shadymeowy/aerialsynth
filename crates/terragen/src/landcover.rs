//! Land-cover class ids stored in the `landcover` layer.

pub const UNKNOWN: u8 = 0;
pub const OCEAN: u8 = 1;
pub const LAKE: u8 = 2;
pub const RIVER: u8 = 3;
pub const BEACH: u8 = 4;
pub const SAND: u8 = 5;
pub const ROCK: u8 = 6;
pub const SNOW: u8 = 7;
pub const GRASS: u8 = 8;
pub const SHRUB: u8 = 9;
pub const FOREST: u8 = 10;
pub const CROP: u8 = 11;
pub const BUILDING: u8 = 12;
pub const ROAD: u8 = 13;
pub const WETLAND: u8 = 14;
pub const TUNDRA: u8 = 15;
pub const BARE: u8 = 16;
pub const URBAN: u8 = 17;

pub const NAMES: [&str; 18] = [
    "unknown", "ocean", "lake", "river", "beach", "sand", "rock", "snow", "grass", "shrub", "forest", "crop", "building", "road", "wetland", "tundra", "bare",
    "urban",
];

/// True if the class is a water surface (useful for renderers: specular, flat).
pub fn is_water(c: u8) -> bool {
    matches!(c, OCEAN | LAKE | RIVER)
}

/// A display palette (sRGB) for landcover previews.
pub fn palette(c: u8) -> [u8; 3] {
    match c {
        OCEAN => [20, 50, 110],
        LAKE => [40, 90, 160],
        RIVER => [60, 130, 200],
        BEACH => [240, 220, 160],
        SAND => [220, 190, 120],
        ROCK => [130, 120, 110],
        SNOW => [250, 250, 255],
        GRASS => [140, 190, 80],
        SHRUB => [150, 150, 70],
        FOREST => [30, 100, 40],
        CROP => [230, 200, 60],
        BUILDING => [200, 60, 60],
        ROAD => [60, 60, 60],
        WETLAND => [70, 140, 130],
        TUNDRA => [160, 160, 130],
        BARE => [160, 120, 90],
        URBAN => [180, 150, 150],
        _ => [255, 0, 255],
    }
}
