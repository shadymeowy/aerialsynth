// Land-cover classes: GENERATED from crates/terragen/src/landcover.rs (`landcover::wgsl()`); do not
// edit. `AERIALSYNTH_BLESS=1 cargo test -p terragen --lib landcover` rewrites it.

const LC_MAX_CLASSES: u32 = 128u;
const LC_UNKNOWN: u32 = 0u;
const LC_OCEAN: u32 = 1u;
const LC_LAKE: u32 = 2u;
const LC_RIVER: u32 = 3u;
const LC_BEACH: u32 = 4u;
const LC_SAND: u32 = 5u;
const LC_ROCK: u32 = 6u;
const LC_SNOW: u32 = 7u;
const LC_GRASS: u32 = 8u;
const LC_SHRUB: u32 = 9u;
const LC_FOREST: u32 = 10u;
const LC_CROP: u32 = 11u;
const LC_BUILDING: u32 = 12u;
const LC_ROAD: u32 = 13u;
const LC_WETLAND: u32 = 14u;
const LC_TUNDRA: u32 = 15u;
const LC_BARE: u32 = 16u;
const LC_URBAN: u32 = 17u;
const LC_RESERVOIR: u32 = 20u;
const LC_LAGOON: u32 = 21u;
const LC_CANAL: u32 = 22u;
const LC_AQUACULTURE: u32 = 23u;
const LC_TIDAL_FLAT: u32 = 24u;
const LC_CORAL_REEF: u32 = 25u;
const LC_SEA_ICE: u32 = 26u;
const LC_GLACIER: u32 = 27u;
const LC_FROZEN_WATER: u32 = 28u;
const LC_DRY_RIVERBED: u32 = 29u;
const LC_SALT_FLAT: u32 = 30u;
const LC_LAVA: u32 = 31u;
const LC_VOLCANIC_ASH: u32 = 32u;
const LC_GRAVEL: u32 = 33u;
const LC_BADLANDS: u32 = 34u;
const LC_SCREE: u32 = 35u;
const LC_MORAINE: u32 = 36u;
const LC_CLIFF: u32 = 37u;
const LC_TROPICAL_RAINFOREST: u32 = 40u;
const LC_MANGROVE: u32 = 41u;
const LC_BROADLEAF_FOREST: u32 = 42u;
const LC_NEEDLELEAF_FOREST: u32 = 43u;
const LC_MIXED_FOREST: u32 = 44u;
const LC_WOODLAND: u32 = 45u;
const LC_SAVANNA: u32 = 52u;
const LC_STEPPE: u32 = 53u;
const LC_DESERT_SCRUB: u32 = 54u;
const LC_MAQUIS: u32 = 55u;
const LC_ALPINE_MEADOW: u32 = 56u;
const LC_POLYGON_TUNDRA: u32 = 57u;
const LC_BOG: u32 = 58u;
const LC_MARSH: u32 = 59u;
const LC_BURN_SCAR: u32 = 60u;
const LC_CLEAR_CUT: u32 = 61u;
const LC_RICE_PADDY: u32 = 70u;
const LC_ORCHARD: u32 = 71u;
const LC_VINEYARD: u32 = 72u;
const LC_PLANTATION: u32 = 73u;
const LC_PASTURE: u32 = 74u;
const LC_GREENHOUSE: u32 = 75u;
const LC_FALLOW: u32 = 76u;
const LC_HEDGEROW: u32 = 77u;
const LC_FARMYARD: u32 = 78u;
const LC_RESIDENTIAL: u32 = 80u;
const LC_COMMERCIAL: u32 = 81u;
const LC_INDUSTRIAL: u32 = 82u;
const LC_BUILDING_TALL: u32 = 83u;
const LC_PARK: u32 = 84u;
const LC_SPORTS_FIELD: u32 = 85u;
const LC_PARKING: u32 = 86u;
const LC_SOLAR_FARM: u32 = 87u;
const LC_PORT: u32 = 88u;
const LC_CEMETERY: u32 = 89u;
const LC_QUARRY: u32 = 90u;
const LC_MOTORWAY: u32 = 100u;
const LC_ROAD_MAJOR: u32 = 101u;
const LC_ROAD_MINOR: u32 = 102u;
const LC_TRACK: u32 = 103u;
const LC_RAILWAY: u32 = 104u;
const LC_RUNWAY: u32 = 105u;
const LC_TAXIWAY: u32 = 106u;
const LC_BRIDGE: u32 = 107u;
const LC_DAM: u32 = 108u;
const LC_SEASONAL_SNOW: u32 = 110u;

const LG_UNKNOWN: u32 = 0u;
const LG_WATER: u32 = 1u;
const LG_WETLAND: u32 = 2u;
const LG_BARE: u32 = 3u;
const LG_SNOW_ICE: u32 = 4u;
const LG_VEGETATION: u32 = 5u;
const LG_FOREST: u32 = 6u;
const LG_AGRICULTURE: u32 = 7u;
const LG_DISTURBED: u32 = 8u;
const LG_BUILT: u32 = 9u;
const LG_TRANSPORT: u32 = 10u;

// group (LG_*) of class c
fn lc_group(c: u32) -> u32 {
    var r: u32 = LG_UNKNOWN;
    switch c {
        case 1u, 2u, 3u, 20u, 21u, 22u, 23u, 25u: { r = LG_WATER; }
        case 4u, 5u, 6u, 16u, 29u, 30u, 31u, 32u, 33u, 34u, 35u, 36u, 37u, 90u: { r = LG_BARE; }
        case 7u, 26u, 27u, 28u, 110u: { r = LG_SNOW_ICE; }
        case 8u, 9u, 15u, 52u, 53u, 54u, 55u, 56u, 57u, 84u: { r = LG_VEGETATION; }
        case 10u, 40u, 41u, 42u, 43u, 44u, 45u, 77u: { r = LG_FOREST; }
        case 11u, 70u, 71u, 72u, 73u, 74u, 75u, 76u: { r = LG_AGRICULTURE; }
        case 12u, 17u, 78u, 80u, 81u, 82u, 83u, 85u, 86u, 87u, 88u, 89u, 108u: { r = LG_BUILT; }
        case 13u, 100u, 101u, 102u, 103u, 104u, 105u, 106u, 107u: { r = LG_TRANSPORT; }
        case 14u, 24u, 58u, 59u: { r = LG_WETLAND; }
        case 60u, 61u: { r = LG_DISTURBED; }
        default: {}
    }
    return r;
}

// material of class c: x = glint weight, y = specular F0, z = glint exponent (Blinn-Phong), w = emissive
fn lc_material(c: u32) -> vec4<f32> {
    var r: vec4<f32> = vec4<f32>(0.0, 0.04, 1.0, 0.0);
    switch c {
        case 1u, 2u, 3u, 20u, 21u, 22u, 23u, 25u: { r = vec4<f32>(1.0, 0.02, 300.0, 0.0); }
        case 24u: { r = vec4<f32>(0.5, 0.02, 87.0, 0.0); }
        case 26u: { r = vec4<f32>(0.4, 0.03, 48.0, 0.0); }
        case 27u: { r = vec4<f32>(0.25, 0.03, 20.0, 0.0); }
        case 28u: { r = vec4<f32>(0.6, 0.03, 137.0, 0.0); }
        case 31u: { r = vec4<f32>(0.0, 0.04, 1.0, 1.0); }
        case 75u: { r = vec4<f32>(0.6, 0.08, 137.0, 0.3); }
        case 83u: { r = vec4<f32>(0.3, 0.06, 87.0, 0.0); }
        case 87u: { r = vec4<f32>(0.7, 0.05, 198.0, 0.0); }
        default: {}
    }
    return r;
}

// display colour of class c (sRGB, 0..1; magenta: no such class)
fn lc_palette(c: u32) -> vec3<f32> {
    var r: vec3<f32> = vec3<f32>(255.0, 0.0, 255.0) / 255.0;
    switch c {
        case 1u: { r = vec3<f32>(20.0, 50.0, 110.0) / 255.0; }
        case 2u: { r = vec3<f32>(40.0, 90.0, 160.0) / 255.0; }
        case 3u: { r = vec3<f32>(60.0, 130.0, 200.0) / 255.0; }
        case 4u: { r = vec3<f32>(240.0, 220.0, 160.0) / 255.0; }
        case 5u: { r = vec3<f32>(220.0, 190.0, 120.0) / 255.0; }
        case 6u: { r = vec3<f32>(130.0, 120.0, 110.0) / 255.0; }
        case 7u: { r = vec3<f32>(250.0, 250.0, 255.0) / 255.0; }
        case 8u: { r = vec3<f32>(140.0, 190.0, 80.0) / 255.0; }
        case 9u: { r = vec3<f32>(150.0, 150.0, 70.0) / 255.0; }
        case 10u: { r = vec3<f32>(30.0, 100.0, 40.0) / 255.0; }
        case 11u: { r = vec3<f32>(230.0, 200.0, 60.0) / 255.0; }
        case 12u: { r = vec3<f32>(200.0, 60.0, 60.0) / 255.0; }
        case 13u: { r = vec3<f32>(60.0, 60.0, 60.0) / 255.0; }
        case 14u: { r = vec3<f32>(70.0, 140.0, 130.0) / 255.0; }
        case 15u: { r = vec3<f32>(160.0, 160.0, 130.0) / 255.0; }
        case 16u: { r = vec3<f32>(160.0, 120.0, 90.0) / 255.0; }
        case 17u: { r = vec3<f32>(180.0, 150.0, 150.0) / 255.0; }
        case 20u: { r = vec3<f32>(50.0, 100.0, 170.0) / 255.0; }
        case 21u: { r = vec3<f32>(40.0, 120.0, 150.0) / 255.0; }
        case 22u: { r = vec3<f32>(70.0, 140.0, 210.0) / 255.0; }
        case 23u: { r = vec3<f32>(90.0, 150.0, 170.0) / 255.0; }
        case 24u: { r = vec3<f32>(150.0, 140.0, 110.0) / 255.0; }
        case 25u: { r = vec3<f32>(60.0, 190.0, 190.0) / 255.0; }
        case 26u: { r = vec3<f32>(220.0, 235.0, 245.0) / 255.0; }
        case 27u: { r = vec3<f32>(200.0, 230.0, 250.0) / 255.0; }
        case 28u: { r = vec3<f32>(180.0, 210.0, 235.0) / 255.0; }
        case 29u: { r = vec3<f32>(200.0, 180.0, 140.0) / 255.0; }
        case 30u: { r = vec3<f32>(245.0, 240.0, 225.0) / 255.0; }
        case 31u: { r = vec3<f32>(40.0, 30.0, 30.0) / 255.0; }
        case 32u: { r = vec3<f32>(70.0, 65.0, 65.0) / 255.0; }
        case 33u: { r = vec3<f32>(175.0, 165.0, 145.0) / 255.0; }
        case 34u: { r = vec3<f32>(190.0, 120.0, 80.0) / 255.0; }
        case 35u: { r = vec3<f32>(150.0, 145.0, 140.0) / 255.0; }
        case 36u: { r = vec3<f32>(165.0, 160.0, 150.0) / 255.0; }
        case 37u: { r = vec3<f32>(100.0, 90.0, 85.0) / 255.0; }
        case 40u: { r = vec3<f32>(10.0, 90.0, 30.0) / 255.0; }
        case 41u: { r = vec3<f32>(30.0, 110.0, 80.0) / 255.0; }
        case 42u: { r = vec3<f32>(40.0, 120.0, 40.0) / 255.0; }
        case 43u: { r = vec3<f32>(20.0, 80.0, 50.0) / 255.0; }
        case 44u: { r = vec3<f32>(35.0, 100.0, 45.0) / 255.0; }
        case 45u: { r = vec3<f32>(90.0, 140.0, 60.0) / 255.0; }
        case 52u: { r = vec3<f32>(190.0, 180.0, 90.0) / 255.0; }
        case 53u: { r = vec3<f32>(190.0, 190.0, 120.0) / 255.0; }
        case 54u: { r = vec3<f32>(180.0, 160.0, 110.0) / 255.0; }
        case 55u: { r = vec3<f32>(120.0, 130.0, 60.0) / 255.0; }
        case 56u: { r = vec3<f32>(130.0, 180.0, 110.0) / 255.0; }
        case 57u: { r = vec3<f32>(150.0, 155.0, 125.0) / 255.0; }
        case 58u: { r = vec3<f32>(110.0, 110.0, 80.0) / 255.0; }
        case 59u: { r = vec3<f32>(90.0, 150.0, 110.0) / 255.0; }
        case 60u: { r = vec3<f32>(50.0, 40.0, 35.0) / 255.0; }
        case 61u: { r = vec3<f32>(170.0, 150.0, 100.0) / 255.0; }
        case 70u: { r = vec3<f32>(120.0, 200.0, 170.0) / 255.0; }
        case 71u: { r = vec3<f32>(110.0, 160.0, 60.0) / 255.0; }
        case 72u: { r = vec3<f32>(140.0, 90.0, 140.0) / 255.0; }
        case 73u: { r = vec3<f32>(70.0, 130.0, 50.0) / 255.0; }
        case 74u: { r = vec3<f32>(170.0, 210.0, 100.0) / 255.0; }
        case 75u: { r = vec3<f32>(220.0, 230.0, 240.0) / 255.0; }
        case 76u: { r = vec3<f32>(150.0, 110.0, 70.0) / 255.0; }
        case 77u: { r = vec3<f32>(60.0, 120.0, 50.0) / 255.0; }
        case 78u: { r = vec3<f32>(190.0, 140.0, 110.0) / 255.0; }
        case 80u: { r = vec3<f32>(220.0, 130.0, 110.0) / 255.0; }
        case 81u: { r = vec3<f32>(230.0, 80.0, 90.0) / 255.0; }
        case 82u: { r = vec3<f32>(170.0, 130.0, 170.0) / 255.0; }
        case 83u: { r = vec3<f32>(150.0, 30.0, 40.0) / 255.0; }
        case 84u: { r = vec3<f32>(100.0, 200.0, 100.0) / 255.0; }
        case 85u: { r = vec3<f32>(160.0, 210.0, 140.0) / 255.0; }
        case 86u: { r = vec3<f32>(120.0, 120.0, 120.0) / 255.0; }
        case 87u: { r = vec3<f32>(40.0, 50.0, 90.0) / 255.0; }
        case 88u: { r = vec3<f32>(100.0, 110.0, 140.0) / 255.0; }
        case 89u: { r = vec3<f32>(120.0, 150.0, 110.0) / 255.0; }
        case 90u: { r = vec3<f32>(200.0, 170.0, 150.0) / 255.0; }
        case 100u: { r = vec3<f32>(230.0, 120.0, 40.0) / 255.0; }
        case 101u: { r = vec3<f32>(240.0, 180.0, 80.0) / 255.0; }
        case 102u: { r = vec3<f32>(90.0, 90.0, 90.0) / 255.0; }
        case 103u: { r = vec3<f32>(150.0, 120.0, 90.0) / 255.0; }
        case 104u: { r = vec3<f32>(110.0, 60.0, 110.0) / 255.0; }
        case 105u: { r = vec3<f32>(40.0, 40.0, 50.0) / 255.0; }
        case 106u: { r = vec3<f32>(80.0, 80.0, 95.0) / 255.0; }
        case 107u: { r = vec3<f32>(200.0, 200.0, 60.0) / 255.0; }
        case 108u: { r = vec3<f32>(160.0, 160.0, 180.0) / 255.0; }
        case 110u: { r = vec3<f32>(235.0, 240.0, 250.0) / 255.0; }
        default: {}
    }
    return r;
}
