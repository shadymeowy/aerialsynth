// The composite stack of pass B (`stack.rs`, `layers/*.rs`): the same layers in the same order.

/// One layer at a sample (`Layer`).
struct Layer {
    cov: f32,
    albedo: vec3<f32>,
    dh: f32,
    hmode: u32,
    cls: u32,
    emit: vec3<f32>,
    lit: f32,
    relit: f32,
    mat: u32,
    clear: f32,
    water: bool,
    /// a solid surface over water (ice): no longer water
    solid: bool,
}

fn layer_none() -> Layer {
    var l: Layer;
    l.cov = 0.0;
    l.albedo = vec3<f32>(0.0);
    l.dh = 0.0;
    l.hmode = HM_NONE;
    l.cls = 0u;
    l.emit = vec3<f32>(0.0);
    l.lit = 1.0;
    l.relit = 0.0;
    l.mat = 0u;
    l.clear = 0.0;
    l.water = false;
    l.solid = false;
    return l;
}

fn layer_paint(cov: f32, albedo: vec3<f32>, cls: u32) -> Layer {
    var l = layer_none();
    l.cov = cov;
    l.albedo = albedo;
    l.cls = cls;
    return l;
}

fn layer_emit(e: vec3<f32>) -> Layer {
    var l = layer_none();
    l.emit = e;
    return l;
}

/// What the layers publish for the layers above (`Masks`).
struct Masks {
    cover: f32,
    rock: f32,
    sand: f32,
    beach: f32,
    snow: f32,
    shore_keep: f32,
    natural_ok: f32,
    flat_ok: f32,
    micro: f32,
    riparian: f32,
    woodlot: f32,
    field_cov: f32,
    town_urban: f32,
    town_cov: f32,
    river_clear: f32,
    road_major_cov: f32,
    road_cov: f32,
    river_cov: f32,
    veg: f32,
}

/// The composite of layers 6–7 as a function of what lies below (`Deferred`).
struct Deferred {
    k: f32,
    c: vec3<f32>,
    kh: f32,
    ch: f32,
    la: f32,
    lb: f32,
    lmin: f32,
    cls: u32,
}

fn deferred_none() -> Deferred {
    var d: Deferred;
    d.k = 1.0;
    d.c = vec3<f32>(0.0);
    d.kh = 1.0;
    d.ch = 0.0;
    d.la = 1.0;
    d.lb = 0.0;
    d.lmin = 1.0;
    d.cls = 0u;
    return d;
}

/// One sub-sample going through the stack (`Stack`).
struct Stack {
    c: Ctx,
    t: Terrain,
    l: Local,
    pf: PixFields,
    bio: Bio,
    m: Masks,
    has_region: bool,
    r: Region,
    q_loc: vec2<f32>,
    q_rot: vec2<f32>,
    town_i: i32,
    town: TownOut,
    col: vec3<f32>,
    height: f32,
    cls: u32,
    lit: f32,
    emission: vec3<f32>,
    is_water: bool,
    done: bool,
    deferring: bool,
    def: Deferred,
}

fn st_apply(s: ptr<function, Stack>, ly: Layer) {
    let a = clamp(ly.cov, 0.0, 1.0);
    (*s).emission += ly.emit;
    if (a <= 0.0) {
        return;
    }
    (*s).col = (*s).col + (ly.albedo - (*s).col) * a;
    switch ly.hmode {
        case 1u: { (*s).height = lerp((*s).height, (*s).l.ground + ly.dh, a); }
        case 2u: { (*s).height = max((*s).height, (*s).l.ground + ly.dh); }
        case 3u: { (*s).height += ly.dh * a; }
        case 4u: { (*s).height = lerp((*s).height, ly.dh, a); }
        default: {}
    }
    if (ly.relit > 0.0) {
        (*s).lit = lerp((*s).lit, 1.0, ly.relit * a);
    }
    (*s).lit = min((*s).lit, ly.lit);
    if (a > 0.5 && ly.cls != 0u) {
        (*s).cls = ly.cls;
    }
    if (a > 0.5 && ly.water) {
        (*s).is_water = true;
    }
    if (a > 0.5 && ly.solid) {
        (*s).is_water = false;
    }
}

fn st_defer(s: ptr<function, Stack>, ly: Layer) {
    (*s).emission += ly.emit;
    let a = clamp(ly.cov, 0.0, 1.0);
    if (a <= 0.0) {
        return;
    }
    var d = (*s).def;
    d.k *= 1.0 - a;
    d.c = d.c * (1.0 - a) + ly.albedo * a;
    if (ly.hmode == HM_BLEND || ly.hmode == HM_MAX || ly.hmode == HM_ABS) {
        var tgt = (*s).l.ground + ly.dh;
        if (ly.hmode == HM_ABS) {
            tgt = ly.dh;
        }
        d.kh *= 1.0 - a;
        d.ch = d.ch * (1.0 - a) + tgt * a;
    } else if (ly.hmode == HM_ADD) {
        d.ch += ly.dh * a;
    }
    if (ly.relit > 0.0) {
        let w = ly.relit * a;
        d.la *= 1.0 - w;
        d.lb = d.lb * (1.0 - w) + w;
        d.lmin = d.lmin * (1.0 - w) + w;
    }
    d.lmin = min(d.lmin, ly.lit);
    if (a > 0.5 && ly.cls != 0u) {
        d.cls = ly.cls;
    }
    (*s).def = d;
}

/// Composite `ly` over the stack (`Stack::composite`).
fn composite(s: ptr<function, Stack>, ly: Layer) {
    if (ly.clear > 0.0) {
        (*s).m.veg *= 1.0 - ly.clear * clamp(ly.cov, 0.0, 1.0);
    }
    if ((*s).deferring) {
        st_defer(s, ly);
        return;
    }
    st_apply(s, ly);
}

fn st_flush(s: ptr<function, Stack>) {
    let d = (*s).def;
    (*s).col = (*s).col * d.k + d.c;
    (*s).height = (*s).height * d.kh + d.ch;
    (*s).lit = min((*s).lit * d.la + d.lb, d.lmin);
    if (d.cls != 0u) {
        (*s).cls = d.cls;
    }
    (*s).def = deferred_none();
}

/// A lazy pixel field of the sample (`pf_lazy` on the stack's pixel fields).
fn s_pf_lazy(s: ptr<function, Stack>, i: u32) -> f32 {
    if (((*s).pf.pending & (1u << i)) != 0u) {
        (*s).pf.f[i] += pixel_field(i, (*s).pf.p, (*s).pf.gsd, (*s).pf.split, (*s).pf.cut);
        (*s).pf.pending &= ~(1u << i);
    }
    return (*s).pf.f[i];
}

/// A colour of the sample's natural palette with the ecoregion's style (`Stack::pal`).
fn spal(s: ptr<function, Stack>, i: u32) -> vec3<f32> {
    let c = bio_pal((*s).bio, i);
    let st = (*s).bio.st;
    if (i <= 3u) {
        return c + (st.soil.xyz - c) * min(st.soil.w * clamp(0.25 + 1.3 * (*s).t.style.z, 0.0, 1.2), 1.0);
    }
    if (i >= BP_ROCK && i <= BP_ROCK + 2u) {
        return c + (st.rock.xyz - c) * min(st.rock.w * clamp(0.4 + 1.2 * (*s).t.style.y, 0.0, 1.2), 1.0);
    }
    if (i == BP_GRASS_WET || i == BP_GRASS_DRY || i == BP_GRASS_COLD || i == BP_TUNDRA || i == BP_MARSH) {
        return c * st.grass.xyz;
    }
    if (i >= BP_CROWN_CONIFER && i <= BP_SHRUB) {
        return c * st.crown.xyz;
    }
    return c;
}

fn sveg(s: ptr<function, Stack>, k: u32) -> f32 {
    return bio_veg((*s).bio, k);
}

/// The sample's agricultural intensity (`Stack::agri`).
fn s_agri(s: ptr<function, Stack>) -> f32 {
    var b = biomes[(*s).bio.a].land.x;
    if ((*s).bio.w > 0.0) {
        b = b + (biomes[(*s).bio.b].land.x - b) * (*s).bio.w;
    }
    return clamp((*s).t.agri * b * (*s).bio.st.land.y, 0.0, 1.0);
}

/// A named field at the sample (`Stack::field`).
fn s_field(s: ptr<function, Stack>, f: u32) -> f32 {
    let t = (*s).t;
    switch f {
        case 0u: { return t.temp; }
        case 1u: { return t.moist; }
        case 2u: { return (*s).l.slope; }
        case 3u: { return t.gully; }
        case 4u: { return (*s).l.ground; }
        case 5u: {
            if ((*s).l.river_hw > 0.0) {
                return abs((*s).l.river_d) - (*s).l.river_hw;
            }
            return 1e6;
        }
        case 6u: { return (*s).pf.f[PF_PATCH]; }
        case 7u: { return (*s).pf.f[PF_DETAIL]; }
        case 8u: { return (*s).pf.f[PF_LAND]; }
        case 9u: { return (*s).pf.f[PF_FOREST]; }
        case 10u: { return (*s).pf.f[PF_SNOW]; }
        case 11u: { return t.mountain; }
        case 12u: { return t.floodplain; }
        case 13u: { return t.agri; }
        case 14u: { return t.habit; }
        case 15u: { return t.sand; }
        case 16u: { return t.rock_expect; }
        case 17u: { return t.mesa; }
        case 18u: { return t.cont; }
        case 19u: { return abs((*s).c.lat) * (180.0 / PI); }
        case 20u: { return (*s).m.town_urban; }
        case 21u: { return (*s).m.field_cov; }
        case 22u: { return (*s).m.natural_ok; }
        case 23u: { return (*s).m.veg; }
        case 24u: { return t.style.x; }
        case 25u: { return t.style.y; }
        case 26u: { return t.style.z; }
        case 27u: { return t.style.w; }
        case 28u: { return (*s).bio.edge_km; }
        case 29u: { return (*s).bio.site.y; }
        case 30u: { return t.temp_range; }
        case 31u: { return (*s).bio.site.w; }
        case 32u: { return t.coast_km; }
        case 33u: { return t.wind.x; }
        case 34u: { return t.wind.y; }
        case 35u: { return t.volcanism; }
        case 36u: { return t.glaciation; }
        case 37u: { return t.population; }
        case 38u: { return t.development; }
        case 39u: { return t.uplift; }
        case 40u: { return t.regime; }
        default: { return 0.0; }
    }
}

// ---------------------------------------------------------------- frame, biome layers

/// The local frame (`layers::frame`).
fn s_frame(s: ptr<function, Stack>) {
    let p = (*s).c.p;
    (*s).has_region = false;
    if ((*s).t.region_id != 0lu) {
        let ri = region_find((*s).t.region_id);
        if (ri >= 0) {
            let r = regions[ri];
            (*s).r = r;
            (*s).has_region = true;
            let d = vec3<f32>(p - r.center.xyz);
            (*s).q_loc = vec2<f32>(dot(d, r.east.xyz), dot(d, r.north.xyz));
            (*s).q_rot = vec2<f32>(dot(d, r.ex.xyz), dot(d, r.ey.xyz));
            return;
        }
        surface_missing |= MISS_REGION;
    }
    (*s).q_loc = vec2<f32>(f32(dot(p, vec3<f64>((*s).c.east))), f32(dot(p, vec3<f64>((*s).c.north))));
}

/// Kernel inputs at the sample (`layers::kin`).
fn s_kin(s: ptr<function, Stack>, kind: u32, amount: f32) -> KIn {
    var k: KIn;
    k.q = (*s).q_loc;
    k.p = (*s).c.p;
    if ((*s).has_region) {
        k.east = (*s).r.east.xyz;
        k.north = (*s).r.north.xyz;
    } else {
        k.east = (*s).c.east;
        k.north = (*s).c.north;
    }
    k.gsd = (*s).c.gsd;
    k.fw = (*s).l.fw;
    k.amount = amount;
    k.aux = vec4<f32>(0.0);
    if (kind == KIND_CONTOURS) {
        k.aux = vec4<f32>((*s).l.ground, (*s).l.slope, 0.0, 0.0);
    } else if (kind == KIND_WATER) {
        k.aux = vec4<f32>(max((*s).l.water - (*s).l.ground, 0.0), 0.0, 0.0, 0.0);
    }
    return k;
}

fn band_index(gsd: f32) -> u32 {
    var b = 0u;
    var g = 1.0;
    while (b < NBANDS - 1u && gsd >= g) {
        b += 1u;
        g *= 4.0;
    }
    return b;
}

/// The kernel layers of the sample's biome(s) at `slot` (`layers::biome_layers`).
fn biome_layers(s: ptr<function, Stack>, slot: u32) {
    let band = band_index((*s).c.gsd);
    for (var k = 0u; k < 2u; k++) {
        var bi = (*s).bio.a;
        var wt = 1.0 - (*s).bio.w;
        if (k == 1u) {
            bi = (*s).bio.b;
            wt = (*s).bio.w;
            if (bi == (*s).bio.a) {
                continue;
            }
        } else if ((*s).bio.b == bi) {
            wt = 1.0;
        }
        if (wt <= 0.0) {
            continue;
        }
        let bm = biomes[bi];
        if (bm.r1.y == 0u) {
            continue;
        }
        let br = band_ranges[bm.r1.z + band];
        for (var j = 0u; j < br.y; j++) {
            let li = band_idx[br.x + j];
            let l = klayers[li];
            if (l.head.x != slot || (*s).done) {
                continue;
            }
            var m = 1.0;
            for (var w = 0u; w < l.head2.z; w++) {
                m *= window_eval(l.win[w], s_field(s, l.wf[w]));
            }
            if (m <= 0.0) {
                continue;
            }
            let o = kernel_eval(li, s_kin(s, l.head.y, m * wt));
            if (o.cov <= 0.0 && o.emit <= 0.0) {
                continue;
            }
            var ly = layer_none();
            ly.cov = o.cov;
            ly.albedo = o.albedo;
            ly.dh = o.dh;
            ly.hmode = l.head.w & 0xffu;
            ly.cls = l.head.z;
            ly.emit = l.col[2].xyz * o.emit;
            ly.clear = l.sz.y;
            ly.mat = l.head.w >> 8u;
            composite(s, ly);
        }
    }
}

// ---------------------------------------------------------------- the core layers

/// Slot 1 (`layers::ground::zonal`).
fn layer_zonal(s: ptr<function, Stack>) {
    let t = (*s).t;
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let st = t.style;
    let detail = (*s).pf.f[PF_DETAIL];
    let patchv = (*s).pf.f[PF_PATCH];
    let wet = t.moist;
    let temp = t.temp;
    let soil_i = st.x * 3.0;
    let i0 = min(u32(floor(soil_i)), 2u);
    var soil = mixc(spal(s, BP_SOIL + i0), spal(s, BP_SOIL + i0 + 1u), soil_i - f32(i0));
    let lat_w = min(0.75 * sveg(s, VEG_LATERITE), 1.0);
    soil = mixc(soil, spal(s, BP_LATERITE), lat_w * smoothstep1(19.0, 25.0, temp) * smoothstep1(0.45, 0.7, wet));
    let grass_green = mixc(spal(s, BP_GRASS_DRY), spal(s, BP_GRASS_WET), smoothstep1(0.25, 0.75, wet + 0.15 * patchv));
    var grass = mixc(spal(s, BP_GRASS_COLD), grass_green, smoothstep1(-2.0, 8.0, temp));
    grass = grass * vec3<f32>(1.0 + 0.10 * (st.y - 0.5), 1.0 + 0.06 * (st.w - 0.5), 1.0 - 0.08 * (st.y - 0.5));
    let land_n = (*s).pf.f[PF_LAND];
    let cover = smoothstep1(0.08, 0.45, wet + 0.25 * patchv + 0.2 * land_n) * smoothstep1(-9.0, -1.0, temp);
    var col = mixc(soil, grass, cover) * (1.0 + 0.16 * land_n);
    var cls = select(LC_BARE, LC_GRASS, cover > 0.5);
    if (temp < 0.0 && cover > 0.3) {
        col = mixc(col, spal(s, BP_TUNDRA), smoothstep1(0.0, -6.0, temp));
        cls = LC_TUNDRA;
    }
    if (t.floodplain > 0.3 && wet > 0.55) {
        let m = smoothstep1(0.3, 0.9, t.floodplain) * smoothstep1(0.55, 0.8, wet) * smoothstep1(-0.1, 0.3, patchv);
        col = mixc(col, spal(s, BP_MARSH), m);
        if (m > 0.5) {
            cls = LC_WETLAND;
        }
    }
    col *= 1.0 + 0.22 * detail;
    let meadow = sveg(s, VEG_MEADOW);
    {
        let b30 = band(30.0, gsd);
        if (b30 > 0.0 && cover > 0.0) {
            let dry_p = smoothstep1(0.05, 0.55, perlin3(0x3EADlu, p * (1.0lf / 60.0lf)) + 0.5 * perlin3(0x3EAElu, p * (1.0lf / 22.0lf))) * b30 * cover;
            col = mixc(col, col * vec3<f32>(1.16, 1.06, 0.80), 0.45 * dry_p * meadow);
        }
        let b12 = band(12.0, gsd);
        if (b12 > 0.0 && cover > 0.0) {
            var m = 0.10 * perlin3(0x3EB1lu, p * (1.0lf / 12.0lf)) * b12;
            let b4 = band(4.0, gsd);
            if (b4 > 0.0) {
                m += 0.08 * perlin3(0x3EB2lu, p * (1.0lf / 4.0lf)) * b4;
                let b13 = band(1.3, gsd);
                if (b13 > 0.0) {
                    m += 0.06 * perlin3(0x3EB3lu, p * (1.0lf / 1.3lf)) * b13;
                }
            }
            col *= 1.0 + m * cover * meadow;
        }
    }
    if (t.gully != 0.0) {
        let ch = smoothstep1(0.1, 0.8, -t.gully);
        col = mixc(col, mixc(col * 0.8, spal(s, BP_GRASS_WET) * 0.85, 0.5 * smoothstep1(-6.0, 4.0, temp)), 0.6 * ch);
        col *= 1.0 + 0.06 * smoothstep1(0.2, 1.0, t.gully);
    }
    (*s).m.cover = cover;
    composite(s, layer_paint(1.0, col, cls));
}

/// Slot 3 (`layers::azonal::layer`).
fn layer_azonal(s: ptr<function, Stack>) {
    let t = (*s).t;
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let st = t.style;
    let detail = (*s).pf.f[PF_DETAIL];
    let patchv = (*s).pf.f[PF_PATCH];
    let slope = (*s).l.slope;
    let resolve = 1.0 - smoothstep1(8.0, 80.0, gsd);
    let exp_slope = 0.12 + 0.75 * t.rock_expect;
    let slope_eff = lerp(exp_slope, max(slope, exp_slope * 0.6), resolve);
    let rock_n = 0.7 * patchv + 0.3 * (*s).pf.f[PF_LAND];
    let rock = smoothstep1(0.55, 0.85, slope_eff + 0.25 * detail + 0.25 * rock_n + 0.2 * (t.rock_expect - 0.4) + 0.15 * t.gully)
        * (1.0 - 0.5 * smoothstep1(0.3, 0.8, (*s).m.cover) * (1.0 - t.mountain));
    (*s).m.rock = rock;
    if (rock > 0.0) {
        let ri = st.z * 2.0;
        let j = min(u32(floor(ri)), 1u);
        var rc = mixc(spal(s, BP_ROCK + j), spal(s, BP_ROCK + j + 1u), ri - f32(j));
        let strata_h = 6.0 + 10.0 * st.w;
        let strata = sin((*s).l.ground / strata_h + 3.0 * s_pf_lazy(s, PF_STRATA));
        let strata_w = TAU * strata_h / max(slope, 0.05);
        rc *= 1.0 + 0.06 * strata * band(strata_w, 1.5 * gsd) + 0.25 * detail + 0.12 * s_pf_lazy(s, PF_STRATA2);
        composite(s, layer_paint(rock, rc, LC_ROCK));
    }
    (*s).m.sand = t.sand;
    if (t.sand > 0.0) {
        let si = st.y * 2.0;
        let j = min(u32(floor(si)), 1u);
        let sc = mixc(spal(s, BP_SAND + j), spal(s, BP_SAND + j + 1u), si - f32(j)) * (1.0 + 0.06 * detail);
        let a = smoothstep1(0.2, 0.6, t.sand + 0.2 * patchv);
        composite(s, layer_paint(a, sc, LC_SAND));
    }
    let ground = (*s).l.ground;
    let coastal = 1.0 - smoothstep1(0.3, 0.7, t.floodplain);
    if (coastal > 0.0) {
        (*s).m.shore_keep = 1.0 - coastal * (1.0 - smoothstep1(4.0, 8.0, ground + 2.0 * patchv));
    }
    if (ground < 6.0 && coastal > 0.0 && slope < 0.3) {
        let b = (1.0 - smoothstep1(2.6, 4.2, ground + 1.0 * detail + 0.8 * patchv)) * coastal * (1.0 - smoothstep1(0.15, 0.3, slope));
        (*s).m.beach = b;
        let beach = spal(s, BP_BEACH);
        var bm = 1.0;
        let b9 = band(9.0, gsd);
        if (b9 > 0.0) {
            bm += 0.05 * perlin3(0xBE1lu, p * (1.0lf / 9.0lf)) * b9;
            let b25 = band(2.5, gsd);
            if (b25 > 0.0) {
                bm += 0.04 * perlin3(0xBE2lu, p * (1.0lf / 2.5lf)) * b25;
            }
        }
        let bc0 = beach * bm;
        let bc = mixc(bc0, mixc(beach, spal(s, BP_WET_SAND), 0.7), 1.0 - smoothstep1(0.05, 0.3, ground + 0.08 * perlin3(0xBE3lu, p * (1.0lf / 15.0lf))));
        composite(s, layer_paint(b, bc, LC_BEACH));
    }
}

/// After the azonal layers (`layers::azonal::finish_ground`).
fn finish_ground(s: ptr<function, Stack>) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    var micro = 0.0;
    let b9m = band(9.0, gsd);
    if (b9m > 0.0) {
        micro = 0.30 * perlin3(0x9A01lu, p * (1.0lf / 9.0lf)) * b9m;
        let b32 = band(3.2, gsd);
        if (b32 > 0.0) {
            micro += 0.14 * perlin3(0x9A02lu, p * (1.0lf / 3.2lf)) * b32;
            let b11 = band(1.1, gsd);
            if (b11 > 0.0) {
                micro += 0.06 * perlin3(0x9A03lu, p * (1.0lf / 1.1lf)) * b11;
            }
        }
        micro *= (1.0 - 0.5 * (*s).m.rock) * (1.0 - 0.7 * (*s).m.beach);
    }
    (*s).m.micro = micro;
    (*s).height = (*s).l.ground + micro;
    (*s).m.natural_ok *= (1.0 - (*s).m.rock) * (1.0 - (*s).m.snow) * (1.0 - (*s).m.sand) * (1.0 - (*s).m.beach);
    (*s).m.flat_ok = 1.0 - smoothstep1(0.22, 0.32, (*s).l.slope);
    let l = (*s).l;
    let t = (*s).t;
    if (l.river_hw > 0.0 && t.river_wet > 0.3) {
        let ad = abs(l.river_d);
        let belt = 4.0 + 0.6 * min(l.river_hw, 60.0);
        (*s).m.riparian = (1.0 - smoothstep1(l.river_hw + 0.3 * belt, l.river_hw + belt, ad)) * smoothstep1(0.35, 0.6, t.moist)
            * smoothstep1(0.3, 0.7, t.river_wet) * (*s).m.natural_ok * (0.55 + 0.45 * smoothstep1(-0.3, 0.3, (*s).pf.f[PF_PATCH]));
    }
}

/// Slot 5 (`layers::agriculture::layer`).
fn layer_agriculture(s: ptr<function, Stack>) {
    let t = (*s).t;
    let fpu = 0.5 + 0.5 * (*s).pf.f[PF_FOREST];
    let cover = smoothstep1(0.12, 0.45, max(t.moist, 0.0)) * smoothstep1(-6.0, 2.0, t.temp) * cfg.tree_density;
    let share = min(0.34 * sveg(s, VEG_WOODLOTS) * (*s).bio.st.veg.w, 0.95);
    (*s).m.woodlot = smoothstep1(-0.02, 0.02, share * cover - fpu) * (*s).m.natural_ok;
    let agri = s_agri(s);
    let m = (*s).m;
    if (!(*s).has_region || !(agri > 0.02 && m.natural_ok * m.flat_ok > 0.3 && cfg.agriculture > 0.0)) {
        return;
    }
    // (the pixel fields through a local copy: no pointer into the stack is passed on)
    var pfl = (*s).pf;
    let fr = field((*s).r, t, agri, (*s).q_rot, (*s).c.p, (*s).c.gsd, (*s).l.fw, &pfl);
    (*s).pf = pfl;
    if (!fr.ok) {
        return;
    }
    let keep = m.natural_ok * m.flat_ok * (1.0 - m.riparian) * (1.0 - m.woodlot) * m.shore_keep;
    let a = fr.cov * smoothstep1(0.45, 0.55, keep);
    (*s).m.field_cov = a;
    var cls = LC_CROP;
    if (fr.kind == 1u) {
        cls = LC_HEDGEROW;
    } else if (fr.kind == 2u) {
        cls = LC_TRACK;
    }
    var ly = layer_paint(a, fr.col, cls);
    ly.dh = fr.h - 0.6 * m.micro;
    ly.hmode = HM_ADD;
    composite(s, ly);
}

/// Before the canopy: the town at the sample (`layers::built::select_town`).
fn select_town_layer(s: ptr<function, Stack>) {
    let l = (*s).l;
    let t = (*s).t;
    var river_clear = 1.0;
    if (l.river_hw > 0.0) {
        let bank = 2.0 + 0.1 * l.river_hw;
        river_clear = 1.0 - (1.0 - smoothstep1(l.river_hw + bank, l.river_hw + 2.0 * bank + 2.0, abs(l.river_d))) * smoothstep1(0.3, 0.6, t.river_wet);
    }
    (*s).m.river_clear = river_clear;
    let town_slope = 0.12 + 0.75 * t.rock_expect;
    (*s).town_i = -1;
    (*s).town.px.ok = false;
    var urban = 0.0;
    if (t.town != 0u && cfg.towns > 0.0) {
        let sel = select_town((*s).c.p, (*s).c.gsd, town_slope, river_clear, (*s).pf.f[PF_WARP2]);
        if (sel.x == -2.0) {
            surface_missing |= MISS_TOWN;
        } else {
            (*s).town_i = i32(sel.x);
            urban = sel.y;
        }
    }
    var cov = 0.0;
    if ((*s).town_i >= 0) {
        (*s).town = town_eval(towns[(*s).town_i], (*s).c.p, (*s).c.gsd, l.fw, town_slope, river_clear, (*s).pf.f[PF_DETAIL], (*s).pf.f[PF_WARP2]);
        if ((*s).town.px.ok) {
            cov = (*s).town.px.cov;
        }
    }
    (*s).m.town_urban = urban;
    (*s).m.town_cov = cov;
    (*s).m.veg *= (1.0 - smoothstep1(0.0, 0.08, urban)) * (1.0 - cov);
}

/// Slot 6 (`layers::linear::layer`).
fn layer_linear(s: ptr<function, Stack>) {
    if (cfg.roads <= 0.0) {
        return;
    }
    let t = (*s).t;
    let l = (*s).l;
    let gsd = (*s).c.gsd;
    let fw = l.fw;
    let slope = l.slope;
    let steep = smoothstep1(-0.02, 0.02, 0.5 - slope);
    let habit = smoothstep1(-0.005, 0.005, t.habit - 0.03 - 0.05 * (1.0 - t.development)) * steep * (1.0 - (*s).m.snow) * (1.0 - t.sand * 0.7);
    var road_cov = 0.0;
    var road_col = pal3(PAL_ASPHALT);
    if (habit > 0.0) {
        let w_major = 12.0;
        let c1 = band_cov(l.road_major, w_major * 0.5, max(fw, gsd * 0.5));
        if (c1 > 0.0) {
            road_cov = c1 * habit;
            (*s).m.road_major_cov = road_cov;
            let sh = band_cov(l.road_major, w_major * 0.5 + 1.5, fw) - band_cov(l.road_major, w_major * 0.5, fw);
            road_col = mixc(pal3(PAL_ASPHALT), pal3(PAL_CONCRETE), max(sh, 0.0) * 0.6);
        }
        let w_minor = 6.0;
        let c2 = band_cov(l.road_minor, w_minor * 0.5, max(fw, gsd * 0.5)) * habit;
        if (c2 > road_cov) {
            road_cov = c2;
            road_col = mixc(pal3(PAL_ASPHALT), pal3(PAL_GRAVEL), smoothstep1(0.4, 0.7, t.style.x));
        }
    }
    if ((*s).has_region && (*s).r.agri > 0.1 && s_agri(s) > 0.05) {
        let pair = t.region_id ^ t.region_id2;
        if (u01k(pair, 3lu) < 0.7) {
            let c3 = band_cov(t.region_edge, 3.0, max(fw, gsd * 0.5)) * steep;
            if (c3 > road_cov) {
                road_cov = c3;
                road_col = select(pal3(PAL_ASPHALT), pal3(PAL_GRAVEL), u01k(pair, 4lu) < 0.5);
            }
        }
    }
    if (road_cov > 0.0) {
        (*s).m.road_cov = road_cov;
        var ly = layer_paint(road_cov, road_col * (1.0 + 0.05 * (*s).pf.f[PF_DETAIL]), LC_ROAD);
        ly.hmode = HM_BLEND;
        ly.relit = 0.5;
        composite(s, ly);
    }
}

/// Slot 7 (`layers::built::layer`).
fn layer_built(s: ptr<function, Stack>) {
    let t = (*s).t;
    let l = (*s).l;
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = l.fw;
    let m = (*s).m;
    var in_dsm = HM_NONE;
    if ((cfg.flags & CF_BUILDINGS_DSM) != 0u) {
        in_dsm = HM_BLEND;
    }
    let agri = s_agri(s);
    if ((*s).has_region && agri > 0.06 && m.flat_ok > 0.3 && m.natural_ok > 0.3 && cfg.towns > 0.0) {
        let fs = farmstead((*s).r, agri, (*s).q_rot, gsd, fw);
        if (fs.ok) {
            var ly = layer_paint(fs.cov, fs.col, fs.cls);
            ly.dh = fs.h;
            ly.hmode = in_dsm;
            ly.emit = fs.em;
            composite(s, ly);
        }
    }
    if (m.road_major_cov > 0.0 && (*s).town_i >= 0) {
        let town = towns[(*s).town_i];
        let dist = dist64(p, town.center.xyz);
        let near = 1.0 - smoothstep1(1.2 * town.radius, 2.2 * town.radius, dist);
        let sp = 38.0;
        let res = band(sp, gsd);
        if (near > 0.0) {
            let q_loc = (*s).q_loc;
            let kq = round(q_loc / sp);
            let dq = q_loc - kq * sp;
            let d2 = dot(dq, dq);
            let lh = hash2(town.seed ^ 0x40ADlu, i64(kq.x), i64(kq.y));
            var lamp_col = vec3<f32>(0.86, 0.92, 1.0);
            if (u01k(lh, 1lu) < 0.5) {
                lamp_col = vec3<f32>(1.0, 0.48, 0.12);
            }
            let pool = (0.04 * exp(-d2 / (2.0 * 6.0 * 6.0)) + point_light(d2, 6.0, 0.4, fw)) * res + 0.04 * (1.0 - res);
            composite(s, layer_emit(lamp_col * pool * near * m.road_major_cov));
        }
    }
    if (m.town_urban > 0.25 && l.river_hw > 0.0 && t.river_wet > 0.5) {
        let bank = 2.0 + 0.1 * l.river_hw;
        let dl = abs(l.river_d) - (l.river_hw + 0.5 * bank);
        let b4 = band(4.0, gsd);
        var dots = 0.3 * (1.0 - b4);
        if (b4 > 0.0) {
            dots += smoothstep1(0.35, 0.6, perlin3(0xE3Blu, p * (1.0lf / 4.0lf))) * b4;
        }
        composite(s, layer_emit(vec3<f32>(1.0, 0.80, 0.55) * (4.0 * exp(-(dl * dl) / (2.0 * 0.5 * 0.5)) * dots * smoothstep1(0.25, 0.45, m.town_urban))));
    }
    if ((*s).town_i >= 0 && (*s).town.px.ok) {
        let tp = (*s).town.px;
        var ly = layer_paint(tp.cov, tp.col, tp.cls);
        ly.dh = tp.h;
        ly.hmode = in_dsm;
        ly.emit = tp.em * (tp.cov * (0.55 + 0.9 * t.development));
        ly.lit = 1.0 - (*s).town.shadow;
        composite(s, ly);
    }
}

/// The canopy (`layers::canopy::layer`).
fn layer_canopy(s: ptr<function, Stack>) {
    let m = (*s).m;
    if (!(cfg.tree_density > 0.0 && m.natural_ok * m.veg > 0.05)) {
        return;
    }
    let t = (*s).t;
    let l = (*s).l;
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = l.fw;
    let wet = t.moist;
    let temp = t.temp;
    let slope = l.slope;
    let st = t.style;
    let style = (*s).bio.st;
    let td = cfg.tree_density;
    let trees_mult = sveg(s, VEG_TREES) * style.veg.x;
    let base_cover = smoothstep1(0.3, 0.68, wet) * smoothstep1(-6.0, 2.0, temp) * td * trees_mult;
    let fpu = 0.5 + 0.5 * (*s).pf.f[PF_FOREST];
    let forest = smoothstep1(-0.03, 0.03, base_cover - fpu);
    let savanna = smoothstep1(0.18, 0.35, wet) * (1.0 - smoothstep1(0.55, 0.7, wet)) * smoothstep1(12.0, 20.0, temp) * 0.12 * sveg(s, VEG_SAVANNA);
    let groves = 0.04 * smoothstep1(0.15, 0.3, wet) * sveg(s, VEG_GROVES);
    let clear = 1.0 - 0.85 * smoothstep1(0.05, 0.4, s_agri(s)) * (1.0 - smoothstep1(0.45, 0.7, slope)) * (1.0 - m.woodlot);
    let base = m.natural_ok * (1.0 - m.field_cov) * td;
    var dens = (forest * 0.9 * clear + savanna + groves) * base;
    var dens_sav = savanna * base;
    dens = max(dens, 0.8 * m.riparian * td);
    let gully_scrub = 0.55 * smoothstep1(0.2, 0.9, -t.gully) * smoothstep1(0.2, 0.5, wet) * m.natural_ok * (1.0 - m.field_cov);
    let snow_n = (*s).pf.f[PF_SNOW];
    let limit = (1.0 - smoothstep1(0.9, 1.4, slope)) * (1.0 - smoothstep1(0.0, 0.6, t.mountain * smoothstep1(-2.0, -6.0, temp)))
        * (1.0 - smoothstep1(-1.2, -2.4, temp + 1.0 * snow_n)) * m.veg;
    dens *= limit;
    dens_sav *= limit;
    let shrub_clim = smoothstep1(0.15, 0.3, wet) * (1.0 - smoothstep1(0.6, 0.8, wet)) * smoothstep1(2.0, 10.0, temp);
    let shrub_patch = smoothstep1(-0.2, 0.5, (*s).pf.f[PF_PATCH] + 0.4 * (*s).pf.f[PF_LAND]);
    let shrub = clamp((0.45 * shrub_clim * shrub_patch * (1.0 - forest) * max(m.natural_ok, 0.4 * m.rock) * (1.0 - m.field_cov) * sveg(s, VEG_SHRUBS) * style.veg.z + gully_scrub)
        * td * m.veg, 0.0, 0.7);
    var stand = (*s).pf.stand;
    if ((*s).pf.has_stand == 0u) {
        stand = stand_id(p, vec3<f32>((*s).pf.f[13], (*s).pf.f[14], (*s).pf.f[15]));
    }
    let age = u01k(stand, 1lu);
    let tone_u = u01k(stand, 2lu);
    let stand_tone = mixc(vec3<f32>(0.86, 0.93, 0.92), vec3<f32>(1.12, 1.08, 0.88), tone_u) * (0.92 + 0.12 * age);
    let gap_w = smoothstep1(0.3, 0.7, dens);
    if (gap_w > 0.0) {
        let gap = smoothstep1(0.3, 0.6, perlin3(0x6A9lu, p * (1.0lf / 30.0lf)) + 0.5 * perlin3(0x6AAlu, p * (1.0lf / 11.0lf))) * (0.2 + 0.8 * u01k(stand, 4lu));
        dens *= 1.0 - 0.9 * gap * gap_w;
    }
    if (!(dens > 0.0 || shrub > 0.01)) {
        return;
    }
    let stand_d = smoothstep1(0.3, 0.75, dens);
    var conifer = (1.0 - smoothstep1(4.0, 13.0, temp)) * max(stand_d, 1.0 - smoothstep1(-5.0, 1.0, temp));
    conifer = clamp(conifer + 0.9 * (u01k(stand, 3lu) - 0.5) * (1.0 - abs(2.0 * conifer - 1.0)) + sveg(s, VEG_CONIFER) + style.veg.y, 0.0, 1.0);
    let scale = (0.7 + 0.55 * age) * sveg(s, VEG_CROWN_SCALE);
    let tropic = clamp(smoothstep1(19.0, 25.0, temp) * smoothstep1(0.55, 0.75, wet) + sveg(s, VEG_TROPIC), 0.0, 1.0);
    let dry = 1.0 - smoothstep1(0.3, 0.5, wet);
    let tall = 0.5 + 0.5 * st.w + sveg(s, VEG_TALL);
    let bm = biomes[(*s).bio.a];
    let n = min(bm.r0.y, MAX_CROWNS);
    var layers: array<TreeLayer, 6>;
    for (var k = 0u; k < n; k++) {
        let cr = crowns[bm.r0.x + k];
        var share = dens;
        switch cr.share {
            case 0u: { share = dens * conifer; }
            case 1u: { share = dens * (1.0 - conifer) * (1.0 - tropic); }
            case 2u: { share = dens * tropic; }
            case 3u: { share = shrub; }
            case 4u: { share = dens_sav; }
            default: {}
        }
        var colour = cr.colour.xyz;
        if ((cr.flags & 4u) != 0u) {
            colour = mixc(cr.colour.xyz, cr.colour_dry.xyz, dry);
        }
        let stand_ok = (cr.flags & 2u) != 0u;
        layers[k] = TreeLayer(
            colour * style.crown.xyz,
            cr.cell,
            select(vec3<f32>(1.0), stand_tone, stand_ok),
            share * cr.density,
            u64(cr.seed_lo) | (u64(cr.seed_hi) << 32u),
            select(0.0, dens, (cr.flags & 1u) != 0u),
            (cr.h0 + cr.h_tall * tall) * (cr.open_height + (1.0 - cr.open_height) * stand_d),
            cr.shape,
            select(1.0, scale, stand_ok),
        );
    }
    let all = (1u << n) - 1u;
    let floor_ = smoothstep1(0.25, 0.8, dens);
    composite(s, layer_paint(floor_ * 0.85, spal(s, BP_FLOOR) * 0.7, 0u));
    let tr = trees(layers, all, (*s).q_loc, gsd, fw, p);
    if (tr.cov > 0.0) {
        let stv = s_pf_lazy(s, PF_STAND);
        let tc = tr.col * vec3<f32>(1.0 + 0.10 * stv, 1.0 + 0.14 * stv, 1.0 + 0.05 * stv);
        var cls = LC_MIXED_FOREST;
        if (dens < 0.05 && shrub > dens) {
            cls = LC_SHRUB;
        } else if (tropic > 0.5 && wet > 0.6) {
            cls = LC_TROPICAL_RAINFOREST;
        } else if (conifer > 0.7) {
            cls = LC_NEEDLELEAF_FOREST;
        } else if (conifer < 0.3) {
            cls = LC_BROADLEAF_FOREST;
        }
        var ly = layer_paint(tr.cov, tc, cls);
        ly.dh = tr.h;
        if ((cfg.flags & CF_TREES_DSM) != 0u) {
            ly.hmode = HM_MAX;
        }
        composite(s, ly);
    }
    if ((cfg.flags & CF_SHADOWS) != 0u && tr.cov < 0.99) {
        var shadow = 0.0;
        for (var li = 0u; li < n; li++) {
            if (layers[li].density <= 0.0 || layers[li].cell < 2.0 * gsd) {
                continue;
            }
            let off = vec2<f32>(cfg.sun_hx, cfg.sun_hy) * (0.6 * layers[li].height / cfg.sun_tan);
            let sc = trees(layers, 1u << li, (*s).q_loc + off, gsd, fw, p).cov;
            shadow = max(shadow, sc);
        }
        (*s).lit = min((*s).lit, 1.0 - 0.9 * shadow * (1.0 - tr.cov));
    }
    if (tr.cov < 0.01 && dens > 0.0) {
        (*s).col *= 1.0 - 0.15 * dens;
    }
}

/// Standing water (`layers::water::standing`).
fn layer_standing(s: ptr<function, Stack>) {
    let l = (*s).l;
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let depth = l.water - l.ground;
    var col = vec3<f32>(0.0);
    if (l.water_kind == W_OCEAN) {
        col = mixc(pal3(PAL_OCEAN_SHALLOW), pal3(PAL_OCEAN_DEEP), smoothstep1(0.0, 28.0, depth));
    } else {
        col = mixc(mixc(pal3(PAL_RIVER), pal3(PAL_OCEAN_SHALLOW), 0.25), pal3(PAL_LAKE_DEEP), smoothstep1(0.0, 5.0, depth));
    }
    col *= 1.0 + 0.06 * s_pf_lazy(s, PF_WATER);
    if (l.water_kind == W_OCEAN && depth < 3.0) {
        let sh = 1.0 - smoothstep1(0.0, 3.0, depth);
        col = mixc(col, mixc(pal3(PAL_OCEAN_SHALLOW), pal3(PAL_BEACH), 0.55) * 1.05, 0.75 * sh * sh);
        let b3 = band(3.0, gsd);
        if (b3 > 0.0) {
            let wave = depth + 0.18 * perlin3(0x5F1lu, p * (1.0lf / 40.0lf));
            let broken = smoothstep1(-0.25, 0.35, perlin3(0x5F2lu, p * (1.0lf / 22.0lf)) + 0.5 * perlin3(0x5F3lu, p * (1.0lf / 7.0lf)));
            let a = (wave - 0.06) / 0.05;
            let b = (wave - 0.45) / 0.05;
            let cc = (wave - 1.0) / 0.06;
            let foam = (0.85 * exp(-a * a) + 0.6 * broken * exp(-b * b) + 0.4 * broken * exp(-cc * cc)) * (0.75 + 0.25 * perlin3(7lu, p * (1.0lf / 3.0lf))) * b3;
            col = mixc(col, srgb(225.0, 232.0, 230.0), min(foam, 1.0));
        }
    }
    var ly = layer_paint(1.0, col, select(LC_LAKE, LC_OCEAN, l.water_kind == W_OCEAN));
    ly.dh = l.water;
    ly.hmode = HM_ABS;
    ly.relit = 1.0;
    ly.water = true;
    composite(s, ly);
}

/// Rivers (`layers::water::rivers`).
fn layer_rivers(s: ptr<function, Stack>) {
    let l = (*s).l;
    let t = (*s).t;
    if (l.river_hw <= 0.0) {
        return;
    }
    let gsd = (*s).c.gsd;
    let fwr = max(l.fw, gsd * 0.35);
    let cov = band_cov(l.river_d, l.river_hw, fwr);
    if (cov <= 0.0) {
        return;
    }
    let temp = t.temp;
    let detail = (*s).pf.f[PF_DETAIL];
    let wet_r = t.river_wet;
    var wcol = mixc(pal3(PAL_RIVER), pal3(PAL_LAKE_DEEP), smoothstep1(30.0, 200.0, l.river_hw * 2.0));
    wcol = mixc(wcol, srgb(96.0, 138.0, 140.0), 0.7 * t.mountain * smoothstep1(8.0, 0.0, temp));
    let ice = smoothstep1(-1.5, -4.0, temp + 1.5 * (*s).pf.f[PF_SNOW]);
    wcol = mixc(wcol, mixc(pal3(PAL_SNOW) * 0.9, srgb(170.0, 190.0, 200.0), 0.35 * (0.5 + 0.5 * detail)), ice);
    let dry_col = mixc((*s).col, mixc(pal3(PAL_GRAVEL), pal3(PAL_SAND + 2u), 0.5) * (1.0 + 0.1 * detail), 0.55);
    let rc = mixc(dry_col, wcol, wet_r);
    let frozen = ice > 0.5 && wet_r > 0.5;
    var cls = LC_SAND;
    if (frozen) {
        cls = LC_SNOW;
    } else if (wet_r > 0.5) {
        cls = LC_RIVER;
    }
    // (the snow cover leaves open water free, and frozen rivers a faint line)
    (*s).m.river_cov = cov * select(select(0.0, 1.0, wet_r > 0.5), 0.4, frozen);
    var ly = layer_paint(cov, rc, cls);
    ly.dh = l.river_level;
    ly.hmode = HM_ABS;
    ly.relit = 1.0;
    ly.water = !frozen && wet_r > 0.5;
    composite(s, ly);
}

/// The snow mask (`layers::snow::prepare`).
fn snow_prepare(s: ptr<function, Stack>) {
    let t = (*s).t;
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let snow_base = t.temp + 1.0 * (*s).pf.f[PF_SNOW] - 1.6 * smoothstep1(0.1, 0.8, -t.gully) + 0.6 * smoothstep1(0.2, 0.8, t.gully);
    var snow = 0.0;
    // (the noise terms add at most ~1.3 · 0.75)
    if (snow_base - 1.5 < -2.6) {
        var snow_t = snow_base;
        let b60 = band(60.0, gsd);
        if (b60 > 0.0) {
            snow_t += 0.5 * perlin3(0x5E0lu, p * (1.0lf / 60.0lf)) * b60;
            let b18 = band(18.0, gsd);
            if (b18 > 0.0) {
                snow_t += 0.25 * perlin3(0x5E1lu, p * (1.0lf / 18.0lf)) * b18;
            }
        }
        snow = smoothstep1(-2.6, -2.8, snow_t) * (1.0 - 0.75 * smoothstep1(0.9, 1.6, (*s).l.slope));
    }
    (*s).m.snow = snow;
}

/// Slot 9 (`layers::snow::layer`).
fn layer_snow(s: ptr<function, Stack>) {
    let snow = (*s).m.snow * (1.0 - (*s).m.river_cov) * (1.0 - (*s).m.road_cov);
    if (snow > 0.0) {
        composite(s, layer_paint(snow, pal3(PAL_SNOW) * (1.0 + 0.03 * (*s).pf.f[PF_DETAIL]), LC_SNOW));
    }
}

// ---------------------------------------------------------------- the stack

/// Flags of a sample that pass B could not evaluate exactly (the host is to provide data).
var<private> surface_missing: u32;
const MISS_REGION: u32 = 1u;
const MISS_TOWN: u32 = 2u;
const MISS_ECO: u32 = 4u;

fn masks_none() -> Masks {
    var m: Masks;
    m.cover = 0.0;
    m.rock = 0.0;
    m.sand = 0.0;
    m.beach = 0.0;
    m.snow = 0.0;
    m.shore_keep = 1.0;
    m.natural_ok = 1.0;
    m.flat_ok = 1.0;
    m.micro = 0.0;
    m.riparian = 0.0;
    m.woodlot = 0.0;
    m.field_cov = 0.0;
    m.town_urban = 0.0;
    m.town_cov = 0.0;
    m.river_clear = 1.0;
    m.road_major_cov = 0.0;
    m.road_cov = 0.0;
    m.river_cov = 0.0;
    m.veg = 1.0;
    return m;
}

/// The surface at one sub-sample (`stack::eval`).
fn surface_eval(c: Ctx, t: Terrain, l: Local, pf: ptr<function, PixFields>) -> Surface {
    var s: Stack;
    s.c = c;
    s.t = t;
    s.l = l;
    s.pf = *pf;
    s.bio = bio_at(t, l.eco_edge, c);
    s.m = masks_none();
    s.has_region = false;
    s.q_loc = vec2<f32>(0.0);
    s.q_rot = vec2<f32>(0.0);
    s.town_i = -1;
    s.town.px.ok = false;
    s.town.shadow = 0.0;
    s.col = vec3<f32>(0.0);
    s.height = l.ground;
    s.cls = LC_UNKNOWN;
    s.lit = 1.0;
    s.emission = vec3<f32>(0.0);
    s.is_water = false;
    s.done = false;
    s.deferring = false;
    s.def = deferred_none();
    let sp = &s;
    if (l.water > l.ground && l.water_kind != W_NONE) {
        layer_standing(sp);
        biome_layers(sp, SLOT_WATER);
        kits_slot_water(sp);
    } else {
        snow_prepare(sp);
        s_frame(sp);
        layer_zonal(sp);
        biome_layers(sp, SLOT_ZONAL);
        kits_slot_zonal(sp);
        biome_layers(sp, SLOT_ALTITUDINAL);
        kits_slot_altitudinal(sp);
        layer_azonal(sp);
        biome_layers(sp, SLOT_AZONAL);
        kits_slot_azonal(sp);
        finish_ground(sp);
        biome_layers(sp, SLOT_DISTURBANCE);
        kits_slot_disturbance(sp);
        layer_agriculture(sp);
        biome_layers(sp, SLOT_AGRICULTURE);
        kits_slot_agriculture(sp);
        s.deferring = true;
        select_town_layer(sp);
        layer_linear(sp);
        biome_layers(sp, SLOT_LINEAR);
        kits_slot_linear(sp);
        layer_built(sp);
        biome_layers(sp, SLOT_BUILT);
        kits_slot_built(sp);
        s.deferring = false;
        layer_canopy(sp);
        biome_layers(sp, SLOT_CANOPY);
        kits_slot_canopy(sp);
        st_flush(sp);
        layer_rivers(sp);
        if (!s.done) {
            biome_layers(sp, SLOT_WATER);
            kits_slot_water(sp);
        }
        if (!s.done && !s.is_water) {
            layer_snow(sp);
            biome_layers(sp, SLOT_SEASONAL);
            kits_slot_seasonal(sp);
        }
    }
    *pf = s.pf;
    var o: Surface;
    if (s.is_water) {
        o.albedo = s.col;
    } else {
        o.albedo = max(s.col, vec3<f32>(0.0));
    }
    o.height = s.height;
    o.cls = s.cls;
    o.lit = s.lit;
    o.emission = s.emission;
    return o;
}
