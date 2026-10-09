// Biome registry (`registry.rs`) and ecoregions (`eco.rs`): the tables and a sample's biomes.

/// A biome (`GBiome`): natural palette, vegetation parameters, ranges into the other tables.
struct Biome {
    pal: array<vec4<f32>, 24>,
    /// trees, conifer, shrubs, savanna
    veg0: vec4<f32>,
    /// groves, woodlots, tall, laterite
    veg1: vec4<f32>,
    /// meadow, tropic, crown_scale, gallery
    veg2: vec4<f32>,
    /// agriculture, towns
    land: vec4<f32>,
    /// first crown, crowns, first zone, zones
    r0: vec4<u32>,
    /// first layer, layers, first band range
    r1: vec4<u32>,
}

/// A crown layer of the canopy (`GCrown`).
struct Crown {
    share: u32,
    shape: u32,
    seed_lo: u32,
    seed_hi: u32,
    cell: f32,
    h0: f32,
    h_tall: f32,
    open_height: f32,
    density: f32,
    /// bit 0: closure, bit 1: stand, bit 2: dry colour
    flags: u32,
    _p0: u32,
    _p1: u32,
    colour: vec4<f32>,
    colour_dry: vec4<f32>,
}

struct Zone {
    below: f32,
    biome: u32,
}

/// A kernel layer of a biome (`GLayer`): kernel, mask, parameters, calibrated mean.
struct KLayer {
    /// slot, kind, class, hmode | mat << 8
    head: vec4<u32>,
    /// seed lo, seed hi, windows
    head2: vec4<u32>,
    /// size, clear
    sz: vec4<f32>,
    wf: vec4<u32>,
    win: array<vec4<f32>, 4>,
    v: array<vec4<f32>, 4>,
    col: array<vec4<f32>, 3>,
    /// coverage, height, emission
    mean: vec4<f32>,
    mean_albedo: vec4<f32>,
}

/// An ecoregion's resolved parameters (`GEco`, host-computed).
struct Eco {
    id: u64,
    biome: u32,
    arch: u32,
    litho: u32,
    culture: u32,
    _p0: u32,
    _p1: u32,
    soil: vec4<f32>,
    rock: vec4<f32>,
    grass: vec4<f32>,
    crown: vec4<f32>,
    /// trees, conifer, shrubs, woodlots
    veg: vec4<f32>,
    /// season, agri, towns, field scale
    land: vec4<f32>,
    fields: vec4<f32>,
    /// hedges, roof, height, block
    town: vec4<f32>,
    /// sodium, development, site temperature, site precipitation
    misc: vec4<f32>,
    /// temperature range, dry months, Köppen
    clim: vec4<f32>,
}

@group(0) @binding(5) var<storage, read> biomes: array<Biome>;
@group(0) @binding(6) var<storage, read> crowns: array<Crown>;
@group(0) @binding(7) var<storage, read> zones: array<Zone>;
@group(0) @binding(8) var<storage, read> klayers: array<KLayer>;
@group(0) @binding(9) var<storage, read> band_ranges: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read> band_idx: array<u32>;
/// ecoregions by id (open addressing, key 0 = empty)
@group(1) @binding(12) var<storage, read> eco_keys: array<u64>;
@group(1) @binding(13) var<storage, read> eco_vals: array<Eco>;

fn eco_find(id: u64) -> i32 {
    let n = arrayLength(&eco_keys);
    var k = u32(mix64(id) % u64(n));
    for (var i = 0u; i < n; i++) {
        let key = eco_keys[k];
        if (key == id) {
            return i32(k);
        }
        if (key == 0lu) {
            return -1;
        }
        k = (k + 1u) % n;
    }
    return -1;
}

/// The style of an ecoregion (`EcoStyle`), blendable.
struct EcoStyle {
    soil: vec4<f32>,
    rock: vec4<f32>,
    grass: vec4<f32>,
    crown: vec4<f32>,
    veg: vec4<f32>,
    land: vec4<f32>,
    fields: vec4<f32>,
    town: vec4<f32>,
    misc: vec4<f32>,
    arch: u32,
    litho: u32,
}

fn eco_style(e: Eco) -> EcoStyle {
    var s: EcoStyle;
    s.soil = e.soil;
    s.rock = e.rock;
    s.grass = e.grass;
    s.crown = e.crown;
    s.veg = e.veg;
    s.land = e.land;
    s.fields = e.fields;
    s.town = e.town;
    s.misc = e.misc;
    s.arch = e.arch;
    s.litho = e.litho;
    return s;
}

/// `EcoStyle::mix`: continuous parameters blended, discrete ones from `a`.
fn style_mix(a: EcoStyle, b: EcoStyle, w: f32) -> EcoStyle {
    var s = a;
    s.soil = mix(a.soil, b.soil, w);
    s.rock = mix(a.rock, b.rock, w);
    s.grass = mix(a.grass, b.grass, w);
    s.crown = mix(a.crown, b.crown, w);
    s.veg = mix(a.veg, b.veg, w);
    s.land = mix(a.land, b.land, w);
    s.fields = mix(a.fields, b.fields, w);
    s.town = mix(a.town, b.town, w);
    s.misc = mix(a.misc, b.misc, w);
    return s;
}

/// The biome(s) of a sample (`BioSample`).
struct Bio {
    a: u32,
    b: u32,
    w: f32,
    edge_km: f32,
    st: EcoStyle,
    /// site temperature, precipitation, temperature range, dry months
    site: vec4<f32>,
}

/// `eco::ecotone_pick`: weight of the ecoregion across the border at `edge` m.
fn ecotone_pick(p: vec3<f64>, gsd: f32, edge: f32) -> f32 {
    let wd = cfg.ecotone;
    let wd_eff = max(wd, 1.5 * gsd);
    if (edge >= wd_eff || wd_eff <= 0.0) {
        return 0.0;
    }
    let pb = 0.5 * (1.0 - smoothstep1(0.0, wd_eff, edge));
    let ex = band(220.0, gsd);
    if (ex <= 0.0) {
        return pb;
    }
    let n = 0.75 * perlin3(cfg.seed ^ 0xEC7lu, p * (1.0lf / 220.0lf)) + 0.35 * perlin3(cfg.seed ^ 0xEC8lu, p * (1.0lf / 80.0lf)) * band(80.0, gsd);
    let u = 1.0 / (1.0 + exp(-1.7 * n / 0.29));
    let crisp = select(0.0, 1.0, u < pb);
    return pb + (crisp - pb) * ex;
}

/// `registry::zone`: (zone biome, partner, partner weight).
struct ZoneSel {
    z: u32,
    partner: u32,
    w: f32,
}

fn zone_of(b: u32, temp: f32, p: vec3<f64>, gsd: f32) -> ZoneSel {
    var o: ZoneSel;
    o.z = b;
    o.partner = b;
    o.w = 0.0;
    let bz = biomes[b];
    if (bz.r0.w == 0u) {
        return o;
    }
    let n = perlin3(0x20E5lu, p * (1.0lf / 260.0lf)) * 0.8 + 0.4 * perlin3(0x20E6lu, p * (1.0lf / 90.0lf));
    let ex = band(260.0, gsd);
    let tt = temp + 1.2 * n * ex;
    var pw = 0.0;
    for (var k = 0u; k < bz.r0.w; k++) {
        let zn = zones[bz.r0.z + k];
        let wz = (1.0 - ex) * (1.0 - smoothstep1(zn.below - 1.0, zn.below + 1.0, temp));
        if (tt < zn.below) {
            o.z = zn.biome;
        } else if (wz > pw) {
            pw = wz;
            o.partner = zn.biome;
        }
    }
    if (o.z == b) {
        o.w = pw;
    }
    return o;
}

/// `BioSample::at`: the ecoregion of the sample (dithered within the ecotone, blended where the
/// dither is below the pixel), then the zone of its biome.
fn bio_at(t: Terrain, eco_edge: f32, c: Ctx) -> Bio {
    var ia = t.eco_id;
    var ib = t.eco_id2;
    var edge = eco_edge;
    if (edge < 0.0) {
        ia = t.eco_id2;
        ib = t.eco_id;
        edge = -edge;
    }
    var o: Bio;
    let ka = eco_find(ia);
    if (ka < 0) {
        surface_missing |= MISS_ECO;
        o.a = 0u;
        o.b = 0u;
        o.w = 0.0;
        o.edge_km = edge / 1000.0;
        o.st.grass = vec4<f32>(1.0);
        o.st.crown = vec4<f32>(1.0);
        o.st.veg = vec4<f32>(1.0, 0.0, 1.0, 1.0);
        o.st.land = vec4<f32>(0.5, 1.0, 1.0, 1.0);
        return o;
    }
    let ea = eco_vals[ka];
    var wb = 0.0;
    if (ib != 0lu && ib != ia) {
        wb = ecotone_pick(c.p, c.gsd, edge);
    }
    var eb = ea;
    if (wb > 0.0) {
        let kb = eco_find(ib);
        if (kb >= 0) {
            eb = eco_vals[kb];
        } else {
            surface_missing |= MISS_ECO;
        }
    }
    var e1 = ea;
    var e2 = eb;
    var w = wb;
    if (wb > 0.5) {
        e1 = eb;
        e2 = ea;
        w = 1.0 - wb;
    }
    let z1 = zone_of(e1.biome, t.temp, c.p, c.gsd);
    if (w > 0.0) {
        let z2 = zone_of(e2.biome, t.temp, c.p, c.gsd);
        o.a = z1.z;
        o.b = z2.z;
        o.w = w;
        o.st = style_mix(eco_style(e1), eco_style(e2), w);
    } else {
        o.a = z1.z;
        o.b = z1.partner;
        o.w = z1.w;
        o.st = eco_style(e1);
    }
    if (o.a == o.b) {
        o.w = 0.0;
    }
    o.edge_km = edge / 1000.0;
    o.site = vec4<f32>(e1.misc.z, e1.misc.w, e1.clim.x, e1.clim.y);
    return o;
}

/// A palette colour of the sample's biome(s).
fn bio_pal(b: Bio, i: u32) -> vec3<f32> {
    let a = biomes[b.a].pal[i].xyz;
    if (b.w <= 0.0) {
        return a;
    }
    return a + (biomes[b.b].pal[i].xyz - a) * b.w;
}

/// Vegetation parameter `k` (0..11: trees, conifer, shrubs, savanna, groves, woodlots, tall,
/// laterite, meadow, tropic, crown_scale, gallery) of the sample's biome(s).
fn veg_of(bi: u32, k: u32) -> f32 {
    let bm = biomes[bi];
    switch k / 4u {
        case 0u: { return bm.veg0[k % 4u]; }
        case 1u: { return bm.veg1[k % 4u]; }
        default: { return bm.veg2[k % 4u]; }
    }
}

fn bio_veg(b: Bio, k: u32) -> f32 {
    let a = veg_of(b.a, k);
    if (b.w <= 0.0) {
        return a;
    }
    return a + (veg_of(b.b, k) - a) * b.w;
}

const VEG_TREES: u32 = 0u;
const VEG_CONIFER: u32 = 1u;
const VEG_SHRUBS: u32 = 2u;
const VEG_SAVANNA: u32 = 3u;
const VEG_GROVES: u32 = 4u;
const VEG_WOODLOTS: u32 = 5u;
const VEG_TALL: u32 = 6u;
const VEG_LATERITE: u32 = 7u;
const VEG_MEADOW: u32 = 8u;
const VEG_TROPIC: u32 = 9u;
const VEG_CROWN_SCALE: u32 = 10u;
const VEG_GALLERY: u32 = 11u;

/// A mask window (`Window::eval`).
fn window_eval(w: vec4<f32>, x: f32) -> f32 {
    var up = 0.0;
    if (w.y > w.x) {
        up = smoothstep1(w.x, w.y, x);
    } else if (x >= w.x) {
        up = 1.0;
    }
    var down = 0.0;
    if (w.w > w.z) {
        down = 1.0 - smoothstep1(w.z, w.w, x);
    } else if (x <= w.z) {
        down = 1.0;
    }
    return up * down;
}
