//! Land-cover classes (v2): the class ids stored in the `landcover` layer, as one table of data
//! ([`CLASSES`]): name, group, legacy id, display colour and material of every class.
//!
//! * **Ids are stable forever.** 0–17 are the classes of the first release ([`NAMES`]); new
//!   classes are appended in per-group id ranges (20–37 water, ice and bare ground, 40–45 forest,
//!   52–61 other vegetation and disturbed land, 70–78 agriculture, 80–90 settlements, 100–110
//!   transport and seasonal snow) and never renumbered. Ids are below [`MAX_CLASSES`] (128);
//!   255 is reserved for the sky in sequence outputs.
//! * **Every class has a [`Group`]** (11 coarse groups) and a **legacy id** (one of 0–17), so
//!   consumers of the first classes keep working ([`Mapping`], the sequence option
//!   `output.landcover`).
//! * **Material** per class ([`Material`]): what the renderers' shading needs (glint, specular,
//!   roughness, self-emission) instead of a water test.
//!
//! The shaders get the same table as WGSL (`gpu/wgsl/classes.wgsl`, [`WGSL`]): `LC_*` / `LG_*`
//! constants and `lc_group`, `lc_material`, `lc_palette`. That file is generated from this one
//! ([`wgsl`]); a test checks that it is up to date (`AERIALSYNTH_BLESS=1 cargo test -p terragen
//! --lib landcover` rewrites it).
//!
//! The generator emits classes 0–17 for now; the others carry their data for the coming kits.

/// Coarse class groups (stable ids 0–10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Group {
    Unknown = 0,
    Water = 1,
    Wetland = 2,
    Bare = 3,
    SnowIce = 4,
    Vegetation = 5,
    Forest = 6,
    Agriculture = 7,
    Disturbed = 8,
    Built = 9,
    Transport = 10,
}

impl Group {
    pub const ALL: [Group; 11] = [
        Group::Unknown,
        Group::Water,
        Group::Wetland,
        Group::Bare,
        Group::SnowIce,
        Group::Vegetation,
        Group::Forest,
        Group::Agriculture,
        Group::Disturbed,
        Group::Built,
        Group::Transport,
    ];

    pub const fn name(self) -> &'static str {
        GROUP_NAMES[self as usize]
    }

    /// Display colour (sRGB) of the group (previews of `output.landcover: group`).
    pub const fn rgb(self) -> [u8; 3] {
        match self {
            Group::Unknown => [255, 0, 255],
            Group::Water => [30, 80, 160],
            Group::Wetland => [70, 140, 130],
            Group::Bare => [190, 160, 120],
            Group::SnowIce => [240, 245, 255],
            Group::Vegetation => [140, 190, 80],
            Group::Forest => [30, 100, 40],
            Group::Agriculture => [230, 200, 60],
            Group::Disturbed => [90, 60, 50],
            Group::Built => [200, 60, 60],
            Group::Transport => [60, 60, 60],
        }
    }
}

/// Names of the groups, indexed by `Group as u8`.
pub const GROUP_NAMES: [&str; 11] =
    ["unknown", "water", "wetland", "bare", "snow-ice", "vegetation", "forest", "agriculture", "disturbed", "built", "transport"];

/// Shading of a class, used by both renderers (`render::raster`, `shade.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Surface roughness 0..1 of the glint lobe (Blinn-Phong exponent [`Material::shininess`]).
    pub roughness: f32,
    /// Specular reflectance at normal incidence (Fresnel F0) of the glint.
    pub specular: f32,
    /// Weight 0..1 of the mirror-like reflection (sky reflection + sun glint, Schlick Fresnel).
    /// 1 on open water; 0 for matte classes (no specular term at all, as before v2).
    pub glint: f32,
    /// Self-emission at night (lava, lit greenhouses): added to the illumination in units of the
    /// artificial-light level, so it shows with the lights.
    pub emissive: f32,
}

impl Material {
    /// Matte (Lambertian only): every class without a glint.
    pub const MATTE: Material = Material { roughness: 0.9, specular: 0.04, glint: 0.0, emissive: 0.0 };
    /// Open water: the sky reflection with Fresnel from 2 % and a sun glint of exponent 300.
    pub const WATER: Material = Material { roughness: 0.081_378_85, specular: 0.02, glint: 1.0, emissive: 0.0 };

    /// Blinn-Phong exponent of the glint lobe: 2 / roughness² − 2, rounded (water: 300).
    pub fn shininess(&self) -> f32 {
        let r = (self.roughness as f64).max(1e-3);
        (2.0 / (r * r) - 2.0).round().clamp(1.0, 1e5) as f32
    }
}

const fn mat(roughness: f32, specular: f32, glint: f32, emissive: f32) -> Material {
    Material { roughness, specular, glint, emissive }
}

/// One land-cover class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Class {
    pub id: u8,
    /// Machine name (lower case, `_`-separated; the `class_names` attribute).
    pub name: &'static str,
    /// Human-readable name.
    pub label: &'static str,
    pub group: Group,
    /// The class among 0–17 that stands for it (`output.landcover: legacy`).
    pub legacy: u8,
    /// Display colour (sRGB) for previews.
    pub rgb: [u8; 3],
    pub material: Material,
}

/// Class ids are below this (the majority vote of the tile generator counts this many).
pub const MAX_CLASSES: usize = 128;

macro_rules! classes {
    ($( $id:literal $konst:ident $name:literal $label:literal $group:ident $legacy:ident [$r:literal, $g:literal, $b:literal] $mat:expr; )*) => {
        $( #[doc = $label] pub const $konst: u8 = $id; )*
        /// Every class, by increasing id.
        pub const CLASSES: &[Class] = &[ $( Class { id: $id, name: $name, label: $label, group: Group::$group, legacy: $legacy, rgb: [$r, $g, $b], material: $mat } ),* ];
        #[cfg(test)]
        const CONST_NAMES: &[&str] = &[ $( stringify!($konst) ),* ];
    };
}

const M: Material = Material::MATTE;
const W: Material = Material::WATER;

classes! {
    0 UNKNOWN "unknown" "unknown" Unknown UNKNOWN [255, 0, 255] M;
    1 OCEAN "ocean" "ocean" Water OCEAN [20, 50, 110] W;
    2 LAKE "lake" "lake" Water LAKE [40, 90, 160] W;
    3 RIVER "river" "river" Water RIVER [60, 130, 200] W;
    4 BEACH "beach" "beach" Bare BEACH [240, 220, 160] M;
    5 SAND "sand" "sand" Bare SAND [220, 190, 120] M;
    6 ROCK "rock" "rock" Bare ROCK [130, 120, 110] M;
    7 SNOW "snow" "snow" SnowIce SNOW [250, 250, 255] M;
    8 GRASS "grass" "grass" Vegetation GRASS [140, 190, 80] M;
    9 SHRUB "shrub" "shrub" Vegetation SHRUB [150, 150, 70] M;
    10 FOREST "forest" "forest" Forest FOREST [30, 100, 40] M;
    11 CROP "crop" "crop" Agriculture CROP [230, 200, 60] M;
    12 BUILDING "building" "building" Built BUILDING [200, 60, 60] M;
    13 ROAD "road" "road" Transport ROAD [60, 60, 60] M;
    14 WETLAND "wetland" "wetland" Wetland WETLAND [70, 140, 130] M;
    15 TUNDRA "tundra" "tundra" Vegetation TUNDRA [160, 160, 130] M;
    16 BARE "bare" "bare" Bare BARE [160, 120, 90] M;
    17 URBAN "urban" "urban" Built URBAN [180, 150, 150] M;
    20 RESERVOIR "reservoir" "reservoir" Water LAKE [50, 100, 170] W;
    21 LAGOON "lagoon" "lagoon" Water OCEAN [40, 120, 150] W;
    22 CANAL "canal" "canal" Water RIVER [70, 140, 210] W;
    23 AQUACULTURE "aquaculture" "aquaculture / salt pond" Water LAKE [90, 150, 170] W;
    24 TIDAL_FLAT "tidal_flat" "tidal flat" Wetland WETLAND [150, 140, 110] mat(0.15, 0.02, 0.5, 0.0);
    25 CORAL_REEF "coral_reef" "coral reef (shallow)" Water OCEAN [60, 190, 190] W;
    26 SEA_ICE "sea_ice" "sea ice" SnowIce SNOW [220, 235, 245] mat(0.2, 0.03, 0.4, 0.0);
    27 GLACIER "glacier" "glacier" SnowIce SNOW [200, 230, 250] mat(0.3, 0.03, 0.25, 0.0);
    28 FROZEN_WATER "frozen_water" "frozen water" SnowIce SNOW [180, 210, 235] mat(0.12, 0.03, 0.6, 0.0);
    29 DRY_RIVERBED "dry_riverbed" "dry riverbed / wash" Bare SAND [200, 180, 140] M;
    30 SALT_FLAT "salt_flat" "salt flat / playa" Bare BARE [245, 240, 225] M;
    31 LAVA "lava" "lava" Bare ROCK [40, 30, 30] mat(0.9, 0.04, 0.0, 1.0);
    32 VOLCANIC_ASH "volcanic_ash" "volcanic ash / black sand" Bare SAND [70, 65, 65] M;
    33 GRAVEL "gravel" "gravel / alluvial fan" Bare BARE [175, 165, 145] M;
    34 BADLANDS "badlands" "badlands" Bare BARE [190, 120, 80] M;
    35 SCREE "scree" "scree / talus" Bare ROCK [150, 145, 140] M;
    36 MORAINE "moraine" "moraine" Bare BARE [165, 160, 150] M;
    37 CLIFF "cliff" "cliff" Bare ROCK [100, 90, 85] M;
    40 TROPICAL_RAINFOREST "tropical_rainforest" "tropical rainforest" Forest FOREST [10, 90, 30] M;
    41 MANGROVE "mangrove" "mangrove" Forest FOREST [30, 110, 80] M;
    42 BROADLEAF_FOREST "broadleaf_forest" "broadleaf forest" Forest FOREST [40, 120, 40] M;
    43 NEEDLELEAF_FOREST "needleleaf_forest" "needleleaf forest" Forest FOREST [20, 80, 50] M;
    44 MIXED_FOREST "mixed_forest" "mixed forest" Forest FOREST [35, 100, 45] M;
    45 WOODLAND "woodland" "woodland (open trees)" Forest FOREST [90, 140, 60] M;
    52 SAVANNA "savanna" "savanna (grass + trees)" Vegetation SHRUB [190, 180, 90] M;
    53 STEPPE "steppe" "steppe grassland" Vegetation GRASS [190, 190, 120] M;
    54 DESERT_SCRUB "desert_scrub" "desert scrub" Vegetation SHRUB [180, 160, 110] M;
    55 MAQUIS "maquis" "maquis / chaparral" Vegetation SHRUB [120, 130, 60] M;
    56 ALPINE_MEADOW "alpine_meadow" "alpine meadow" Vegetation GRASS [130, 180, 110] M;
    57 POLYGON_TUNDRA "polygon_tundra" "polygon tundra" Vegetation TUNDRA [150, 155, 125] M;
    58 BOG "bog" "bog / peatland" Wetland WETLAND [110, 110, 80] M;
    59 MARSH "marsh" "marsh / reed" Wetland WETLAND [90, 150, 110] M;
    60 BURN_SCAR "burn_scar" "burn scar" Disturbed BARE [50, 40, 35] M;
    61 CLEAR_CUT "clear_cut" "clear-cut / regrowth" Disturbed SHRUB [170, 150, 100] M;
    70 RICE_PADDY "rice_paddy" "rice paddy" Agriculture CROP [120, 200, 170] M;
    71 ORCHARD "orchard" "orchard / grove" Agriculture CROP [110, 160, 60] M;
    72 VINEYARD "vineyard" "vineyard" Agriculture CROP [140, 90, 140] M;
    73 PLANTATION "plantation" "plantation" Agriculture CROP [70, 130, 50] M;
    74 PASTURE "pasture" "pasture" Agriculture GRASS [170, 210, 100] M;
    75 GREENHOUSE "greenhouse" "greenhouse" Agriculture BUILDING [220, 230, 240] mat(0.12, 0.08, 0.6, 0.3);
    76 FALLOW "fallow" "fallow / ploughed" Agriculture CROP [150, 110, 70] M;
    77 HEDGEROW "hedgerow" "hedgerow / shelterbelt" Forest FOREST [60, 120, 50] M;
    78 FARMYARD "farmyard" "farmyard" Built URBAN [190, 140, 110] M;
    80 RESIDENTIAL "residential" "residential" Built URBAN [220, 130, 110] M;
    81 COMMERCIAL "commercial" "commercial / CBD" Built URBAN [230, 80, 90] M;
    82 INDUSTRIAL "industrial" "industrial" Built URBAN [170, 130, 170] M;
    83 BUILDING_TALL "building_tall" "building, tall (> 30 m)" Built BUILDING [150, 30, 40] mat(0.15, 0.06, 0.3, 0.0);
    84 PARK "park" "park / urban green" Vegetation GRASS [100, 200, 100] M;
    85 SPORTS_FIELD "sports_field" "sports field / stadium" Built URBAN [160, 210, 140] M;
    86 PARKING "parking" "parking / paved" Built URBAN [120, 120, 120] M;
    87 SOLAR_FARM "solar_farm" "solar farm" Built URBAN [40, 50, 90] mat(0.1, 0.05, 0.7, 0.0);
    88 PORT "port" "port / dock" Built URBAN [100, 110, 140] M;
    89 CEMETERY "cemetery" "cemetery" Built URBAN [120, 150, 110] M;
    90 QUARRY "quarry" "quarry / mine" Bare BARE [200, 170, 150] M;
    100 MOTORWAY "motorway" "motorway" Transport ROAD [230, 120, 40] M;
    101 ROAD_MAJOR "road_major" "road, major" Transport ROAD [240, 180, 80] M;
    102 ROAD_MINOR "road_minor" "road, minor / street" Transport ROAD [90, 90, 90] M;
    103 TRACK "track" "track (unpaved)" Transport ROAD [150, 120, 90] M;
    104 RAILWAY "railway" "railway" Transport ROAD [110, 60, 110] M;
    105 RUNWAY "runway" "runway" Transport ROAD [40, 40, 50] M;
    106 TAXIWAY "taxiway" "taxiway / apron" Transport ROAD [80, 80, 95] M;
    107 BRIDGE "bridge" "bridge" Transport ROAD [200, 200, 60] M;
    108 DAM "dam" "dam" Built BUILDING [160, 160, 180] M;
    110 SEASONAL_SNOW "seasonal_snow" "seasonal snow" SnowIce SNOW [235, 240, 250] M;
}

/// Names of the legacy classes 0–17 (the first release's `class_names`).
pub const NAMES: [&str; 18] = [
    "unknown", "ocean", "lake", "river", "beach", "sand", "rock", "snow", "grass", "shrub", "forest", "crop", "building", "road", "wetland", "tundra", "bare",
    "urban",
];

/// Position in [`CLASSES`] of each id (255: no such class).
const INDEX: [u8; 256] = {
    let mut t = [255u8; 256];
    let mut i = 0;
    while i < CLASSES.len() {
        t[CLASSES[i].id as usize] = i as u8;
        i += 1;
    }
    t
};

/// The class with id `c` (None: no class has that id).
pub fn class(c: u8) -> Option<&'static Class> {
    CLASSES.get(INDEX[c as usize] as usize)
}

/// Machine name of class `c` ("" if there is none).
pub fn name(c: u8) -> &'static str {
    class(c).map_or("", |k| k.name)
}

/// Group of class `c` (unknown ids: [`Group::Unknown`]).
pub fn group(c: u8) -> Group {
    class(c).map_or(Group::Unknown, |k| k.group)
}

/// Legacy class (0–17) of class `c` (unknown ids: 0).
pub fn legacy(c: u8) -> u8 {
    class(c).map_or(UNKNOWN, |k| k.legacy)
}

/// Material of class `c` (unknown ids: matte).
pub fn material(c: u8) -> Material {
    class(c).map_or(Material::MATTE, |k| k.material)
}

/// True if the class is a water surface (group water: flat, specular).
pub fn is_water(c: u8) -> bool {
    group(c) == Group::Water
}

/// A display palette (sRGB) for landcover previews (magenta: no such class).
pub fn palette(c: u8) -> [u8; 3] {
    class(c).map_or([255, 0, 255], |k| k.rgb)
}

/// How sequence outputs write land cover (`output.landcover`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mapping {
    /// The class ids of this table (0–127).
    #[default]
    V2,
    /// The legacy classes 0–17 (each class's `legacy`).
    Legacy,
    /// The group ids 0–10 ([`Group`]).
    Group,
}

impl Mapping {
    /// The written value of class `c` (255, the sky, stays 255).
    pub fn map(self, c: u8) -> u8 {
        match (self, c) {
            (_, 255) | (Mapping::V2, _) => c,
            (Mapping::Legacy, _) => legacy(c),
            (Mapping::Group, _) => group(c) as u8,
        }
    }

    /// Lookup table of [`Mapping::map`] for all 256 values.
    pub fn lut(self) -> [u8; 256] {
        std::array::from_fn(|c| self.map(c as u8))
    }

    /// Names of the written values, indexed by value ("" where no class has that id): the
    /// `class_names` attribute.
    pub fn value_names(self) -> Vec<&'static str> {
        match self {
            Mapping::V2 => dense(|k| k.name),
            Mapping::Legacy => NAMES.to_vec(),
            Mapping::Group => GROUP_NAMES.to_vec(),
        }
    }

    /// Group name of each written value (`class_groups`).
    pub fn value_groups(self) -> Vec<&'static str> {
        match self {
            Mapping::V2 => dense(|k| k.group.name()),
            Mapping::Legacy => (0..NAMES.len() as u8).map(|c| group(c).name()).collect(),
            Mapping::Group => GROUP_NAMES.to_vec(),
        }
    }

    /// Legacy id of each written value (`class_legacy`; None for groups, which have none).
    pub fn value_legacy(self) -> Option<Vec<u8>> {
        match self {
            Mapping::V2 => {
                let mut v = vec![0u8; max_id() as usize + 1];
                for k in CLASSES {
                    v[k.id as usize] = k.legacy;
                }
                Some(v)
            }
            Mapping::Legacy => Some((0..NAMES.len() as u8).collect()),
            Mapping::Group => None,
        }
    }
}

/// The largest class id.
pub fn max_id() -> u8 {
    CLASSES.iter().map(|k| k.id).max().unwrap_or(0)
}

/// `f` of each class, indexed by id up to the largest ("" in the gaps).
fn dense(f: impl Fn(&Class) -> &'static str) -> Vec<&'static str> {
    let mut v = vec![""; max_id() as usize + 1];
    for k in CLASSES {
        v[k.id as usize] = f(k);
    }
    v
}

/// The WGSL form of the table (`gpu/wgsl/classes.wgsl`, generated by [`wgsl`]), prepended to
/// the generator's and the renderer's shaders.
pub const WGSL: &str = include_str!("gpu/wgsl/classes.wgsl");

/// WGSL float literal.
fn wf(x: f32) -> String {
    let s = format!("{x:?}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// The body of a `fn(c: u32) -> ty`: `var r: ty = D; switch c { case a, b: { r = X; } … }
/// return r;`, the classes grouped by equal values (in order of first appearance), the
/// default's classes left out.
fn wgsl_switch(out: &mut String, ty: &str, ret: impl Fn(&Class) -> String, default: &str) {
    let mut arms: Vec<(String, Vec<u8>)> = Vec::new();
    for k in CLASSES {
        let r = ret(k);
        if r == default {
            continue;
        }
        match arms.iter_mut().find(|(v, _)| *v == r) {
            Some((_, ids)) => ids.push(k.id),
            None => arms.push((r, vec![k.id])),
        }
    }
    out.push_str(&format!("    var r: {ty} = {default};\n    switch c {{\n"));
    for (r, ids) in arms {
        let ids: Vec<String> = ids.iter().map(|i| format!("{i}u")).collect();
        out.push_str(&format!("        case {}: {{ r = {r}; }}\n", ids.join(", ")));
    }
    out.push_str("        default: {}\n    }\n    return r;\n");
}

/// The WGSL source of the class table (what `gpu/wgsl/classes.wgsl` must hold).
pub fn wgsl() -> String {
    let mut s = String::new();
    s.push_str("// Land-cover classes: GENERATED from crates/terragen/src/landcover.rs (`landcover::wgsl()`); do not\n");
    s.push_str("// edit. `AERIALSYNTH_BLESS=1 cargo test -p terragen --lib landcover` rewrites it.\n\n");
    s.push_str(&format!("const LC_MAX_CLASSES: u32 = {MAX_CLASSES}u;\n"));
    for k in CLASSES {
        s.push_str(&format!("const LC_{}: u32 = {}u;\n", k.name.to_uppercase(), k.id));
    }
    s.push('\n');
    for g in Group::ALL {
        s.push_str(&format!("const LG_{}: u32 = {}u;\n", g.name().to_uppercase().replace('-', "_"), g as u8));
    }
    s.push_str("\n// group (LG_*) of class c\nfn lc_group(c: u32) -> u32 {\n");
    wgsl_switch(&mut s, "u32", |k| format!("LG_{}", k.group.name().to_uppercase().replace('-', "_")), "LG_UNKNOWN");
    s.push_str("}\n\n// material of class c: x = glint weight, y = specular F0, z = glint exponent (Blinn-Phong), w = emissive\n");
    s.push_str("fn lc_material(c: u32) -> vec4<f32> {\n");
    let m4 = |m: &Material| format!("vec4<f32>({}, {}, {}, {})", wf(m.glint), wf(m.specular), wf(m.shininess()), wf(m.emissive));
    wgsl_switch(&mut s, "vec4<f32>", |k| m4(&k.material), &m4(&Material::MATTE));
    s.push_str("}\n\n// display colour of class c (sRGB, 0..1; magenta: no such class)\nfn lc_palette(c: u32) -> vec3<f32> {\n");
    let rgb = |c: [u8; 3]| format!("vec3<f32>({}.0, {}.0, {}.0) / 255.0", c[0], c[1], c[2]);
    wgsl_switch(&mut s, "vec3<f32>", |k| rgb(k.rgb), &rgb([255, 0, 255]));
    s.push_str("}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_are_unique_sorted_and_bounded() {
        let mut seen = HashSet::new();
        for w in CLASSES.windows(2) {
            assert!(w[0].id < w[1].id, "{} before {}", w[0].name, w[1].name);
        }
        for k in CLASSES {
            assert!((k.id as usize) < MAX_CLASSES, "{}", k.name);
            assert!(seen.insert(k.id));
            assert_eq!(class(k.id), Some(k));
        }
        assert!(CLASSES.len() < 255);
        assert_eq!(class(19), None);
        assert_eq!(class(255), None);
        assert_eq!(name(19), "");
    }

    #[test]
    fn the_first_classes_keep_their_ids() {
        for (i, n) in NAMES.iter().enumerate() {
            let k = class(i as u8).unwrap();
            assert_eq!(k.name, *n);
            assert_eq!(k.legacy, i as u8, "{n} is its own legacy class");
        }
        assert_eq!((OCEAN, LAKE, RIVER, URBAN), (1, 2, 3, 17));
        // the palette of the first classes is the one of the previews before v2
        assert_eq!(palette(OCEAN), [20, 50, 110]);
        assert_eq!(palette(URBAN), [180, 150, 150]);
        assert_eq!(palette(UNKNOWN), [255, 0, 255]);
        assert_eq!(palette(200), [255, 0, 255]);
    }

    #[test]
    fn names_constants_groups_and_legacy_are_consistent() {
        let mut names = HashSet::new();
        for (k, konst) in CLASSES.iter().zip(CONST_NAMES) {
            assert_eq!(k.name.to_uppercase(), *konst);
            assert!(k.name.chars().all(|c| c.is_ascii_lowercase() || c == '_'), "{}", k.name);
            assert!(names.insert(k.name), "duplicate name {}", k.name);
            assert!((k.legacy as usize) < NAMES.len(), "{}: legacy {}", k.name, k.legacy);
            // a class and its legacy class are in the same group, except where the design
            // regroups (tidal flats are wetland, burn scars and clear-cuts are disturbed, ...)
            assert!(k.group != Group::Unknown || k.id == UNKNOWN, "{} has no group", k.name);
        }
        assert_eq!(CLASSES.len(), CONST_NAMES.len());
        // every group has classes
        for g in Group::ALL {
            assert!(CLASSES.iter().any(|k| k.group == g), "{g:?} is empty");
            assert_eq!(Group::ALL[g as usize], g);
        }
        assert_eq!(group(RESERVOIR), Group::Water);
        assert_eq!(legacy(RESERVOIR), LAKE);
        assert_eq!(legacy(PASTURE), GRASS);
        assert_eq!(legacy(GREENHOUSE), BUILDING);
        assert_eq!(group(BURN_SCAR), Group::Disturbed);
    }

    #[test]
    fn palette_colours_are_distinct() {
        let mut seen = HashSet::new();
        for k in CLASSES {
            assert!(seen.insert(k.rgb), "{} shares its colour", k.name);
        }
    }

    #[test]
    fn materials() {
        for k in CLASSES {
            let m = k.material;
            assert!((0.0..=1.0).contains(&m.glint) && (0.0..=1.0).contains(&m.specular) && m.roughness > 0.0 && m.roughness <= 1.0, "{}", k.name);
            assert!(m.emissive >= 0.0);
            // every open water class glints
            if k.group == Group::Water {
                assert_eq!(m, Material::WATER, "{}", k.name);
            }
        }
        assert_eq!(Material::WATER.shininess(), 300.0);
        assert!(is_water(OCEAN) && is_water(LAKE) && is_water(RIVER) && is_water(CANAL));
        assert!(!is_water(WETLAND) && !is_water(SNOW) && !is_water(255));
        // before v2 only water glinted: the classes the generator emits keep that look
        for c in 0..=URBAN {
            assert_eq!(material(c).glint > 0.0, is_water(c), "{}", name(c));
        }
        assert!(material(SOLAR_FARM).glint > 0.0 && material(LAVA).emissive > 0.0);
    }

    #[test]
    fn mappings() {
        assert_eq!(Mapping::V2.map(RESERVOIR), RESERVOIR);
        assert_eq!(Mapping::Legacy.map(RESERVOIR), LAKE);
        assert_eq!(Mapping::Group.map(RESERVOIR), Group::Water as u8);
        assert_eq!(Mapping::Group.map(MOTORWAY), Group::Transport as u8);
        for m in [Mapping::V2, Mapping::Legacy, Mapping::Group] {
            assert_eq!(m.map(255), 255, "the sky stays");
            let lut = m.lut();
            let names = m.value_names();
            let groups = m.value_groups();
            assert_eq!(names.len(), groups.len());
            for k in CLASSES {
                let v = lut[k.id as usize];
                assert!(!names[v as usize].is_empty(), "{m:?}: {} → {v}", k.name);
                assert_eq!(groups[v as usize], if m == Mapping::Legacy { group(k.legacy).name() } else { k.group.name() });
                if let Some(l) = m.value_legacy() {
                    assert_eq!(l[v as usize], k.legacy, "{m:?}: {}", k.name);
                }
            }
        }
        assert_eq!(Mapping::V2.value_names()[OCEAN as usize], "ocean");
        assert_eq!(Mapping::V2.value_names()[19], "");
        assert_eq!(Mapping::Legacy.value_names().len(), 18);
        assert_eq!(Mapping::Group.value_names().len(), 11);
        assert!(Mapping::Group.value_legacy().is_none());
        let m: Mapping = serde_yaml::from_str("legacy").unwrap();
        assert_eq!(m, Mapping::Legacy);
        assert_eq!(serde_yaml::from_str::<Mapping>("v2").unwrap(), Mapping::V2);
        assert!(serde_yaml::from_str::<Mapping>("v3").is_err());
    }

    /// docs/formats.md lists every class (id, name, label).
    #[test]
    fn the_docs_list_every_class() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/formats.md");
        let Ok(doc) = std::fs::read_to_string(&path) else { return }; // (a crate packaged without the docs)
        for k in CLASSES {
            assert!(doc.contains(&format!("| {} | `{}` | {} |", k.id, k.name, k.label)), "docs/formats.md lacks class {} {}", k.id, k.name);
        }
    }

    /// `gpu/wgsl/classes.wgsl` is what `wgsl()` generates (`AERIALSYNTH_BLESS=1` rewrites it).
    #[test]
    fn wgsl_is_up_to_date() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/wgsl/classes.wgsl");
        let generated = wgsl();
        if std::env::var_os("AERIALSYNTH_BLESS").is_some() {
            std::fs::write(&path, &generated).unwrap();
            return;
        }
        assert!(
            WGSL == generated,
            "{} is out of date with landcover.rs: regenerate it with AERIALSYNTH_BLESS=1 cargo test -p terragen --lib landcover",
            path.display()
        );
    }

    /// The generated WGSL parses and validates (naga).
    #[test]
    fn wgsl_validates() {
        let m = naga::front::wgsl::parse_str(&wgsl()).expect("classes.wgsl parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&m).expect("classes.wgsl validates");
    }
}
