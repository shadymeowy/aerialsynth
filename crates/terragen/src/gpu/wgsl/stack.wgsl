// The composite stack of pass B (`stack.rs`, `layers/*.rs`): the same layers in the same order.

/// One layer at a sample (`Layer`).
struct Layer {
    cov: f32,
    albedo: vec3<f32>,
    dh: f32,
    hmode: u32,
    cls: u32,
    /// instance id taken with `cls` (`Layer::inst`)
    inst: u32,
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
    l.inst = 0u;
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

/// An instance id from a hash (`stack::instance_id`): 30 bits with bit 23 set.
fn instance_id(h: u64) -> u32 {
    let v = u32(h ^ (h >> 32u));
    return (v & 0x3F7FFFFFu) | 0x00800000u;
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
    inst: u32,
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
    d.inst = 0u;
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
    /// the region's index in `regions` (-1: none; pass B's stages keep it: `s_frame_at`)
    ri: i32,
    r: Region,
    q_loc: vec2<f32>,
    q_rot: vec2<f32>,
    town_i: i32,
    town: TownOut,
    col: vec3<f32>,
    height: f32,
    cls: u32,
    inst: u32,
    lit: f32,
    emission: vec3<f32>,
    is_water: bool,
    done: bool,
    no_town: bool,
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
        (*s).inst = ly.inst;
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
        d.inst = ly.inst;
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
        (*s).inst = d.inst;
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
    var ri = -1;
    if ((*s).t.region_id != 0lu) {
        ri = region_find((*s).t.region_id);
        if (ri < 0) {
            surface_missing |= MISS_REGION;
        }
    }
    s_frame_at(s, ri);
}

/// The local frame in region `ri` (`regions` index; -1: none).
fn s_frame_at(s: ptr<function, Stack>, ri: i32) {
    let p = (*s).c.p;
    (*s).ri = ri;
    (*s).has_region = false;
    if (ri >= 0) {
        let r = regions[ri];
        (*s).r = r;
        (*s).has_region = true;
        let d = vec3<f32>(p - r.center.xyz);
        (*s).q_loc = vec2<f32>(dot(d, r.east.xyz), dot(d, r.north.xyz));
        (*s).q_rot = vec2<f32>(dot(d, r.ex.xyz), dot(d, r.ey.xyz));
        return;
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

/// The layer range of biome `bi` (weight `wt`) at band `band`: (first, count) in `band_idx`.
fn biome_band_range(bi: u32, wt: f32, band: u32) -> vec2<u32> {
    if (wt <= 0.0) {
        return vec2<u32>(0u);
    }
    let bm = biomes[bi];
    if (bm.r1.y == 0u) {
        return vec2<u32>(0u);
    }
    return band_ranges[bm.r1.z + band];
}

/// The kernel layers of the sample's biome(s) at a slot (`layers::biome_layers`): those of
/// biome a, then of biome b, in one loop. Pass B's stages run the loop (`biome_layers_st<n>`,
/// generated by `gpu::stages_wgsl`) with their own subset of the kernels (`kernel_eval_st<n>`):
/// `bl_begin`, then `bl_next` until false, evaluating and compositing each layer it returns
/// (`bl_apply`).
struct BLIter {
    ra: vec2<u32>,
    rb: vec2<u32>,
    rc: vec2<u32>,
    wa: f32,
    wb: f32,
    wc: f32,
    q: u32,
    li: u32,
    kin: KIn,
}

fn bl_begin(s: ptr<function, Stack>) -> BLIter {
    var it: BLIter;
    let band = band_index((*s).c.gsd);
    let ba = (*s).bio.a;
    let bb = (*s).bio.b;
    let bc = (*s).bio.c;
    it.wc = select(0.0, (*s).bio.wst, bc != ba);
    it.wa = select(1.0 - (*s).bio.w, 1.0, bb == ba);
    it.wb = (*s).bio.w;
    it.ra = biome_band_range(ba, it.wa, band);
    it.rb = vec2<u32>(0u);
    if (bb != ba) {
        it.rb = biome_band_range(bb, it.wb, band);
    }
    it.rc = biome_band_range(bc, it.wc, band);
    it.q = 0u;
    return it;
}

/// The next layer of the slot to evaluate (`it.li`, its inputs `it.kin`); false at the end.
fn bl_next(s: ptr<function, Stack>, slot: u32, it: ptr<function, BLIter>) -> bool {
    let nab = (*it).ra.y + (*it).rb.y;
    let n = nab + (*it).rc.y;
    while ((*it).q < n) {
        let q = (*it).q;
        (*it).q = q + 1u;
        var li = 0u;
        var wt = (*it).wa;
        if (q < (*it).ra.y) {
            li = band_idx[(*it).ra.x + q];
        } else if (q < nab) {
            li = band_idx[(*it).rb.x + q - (*it).ra.y];
            wt = (*it).wb;
        } else {
            li = band_idx[(*it).rc.x + q - nab];
            wt = (*it).wc;
        }
        let l = klayers[li];
        if (l.head.x != slot || (*s).done) {
            continue;
        }
        // (`layers::biome_layers`: the other ecoregion's flat layers blend in by `wc`; layers
        // with a height stay the mosaic's)
        let flat = (l.head.w & 0xffu) == HM_NONE || slot == 10u;
        if (q >= nab) {
            if (!flat) {
                continue;
            }
        } else if (flat) {
            wt *= 1.0 - (*it).wc;
        }
        var m = 1.0;
        for (var w = 0u; w < l.head2.z; w++) {
            m *= window_eval(l.win[w], s_field(s, l.wf[w]));
        }
        if (m <= 0.0) {
            continue;
        }
        (*it).li = li;
        (*it).kin = s_kin(s, l.head.y, m * wt);
        return true;
    }
    return false;
}

/// Composite layer `li`'s kernel output `o`.
fn bl_apply(s: ptr<function, Stack>, li: u32, o: KOut) {
    if (o.cov <= 0.0 && o.emit <= 0.0) {
        return;
    }
    let l = klayers[li];
    var ly = layer_none();
    ly.cov = o.cov;
    ly.albedo = o.albedo;
    ly.dh = o.dh;
    ly.hmode = l.head.w & 0xffu;
    if (ly.hmode == HM_MAX && l.head.y == 0u) {
        ly.dh = o.dh * smoothstep1(0.0, 0.5, o.cov);
    }
    ly.cls = l.head.z;
    ly.emit = l.col[2].xyz * o.emit;
    ly.clear = l.sz.y;
    ly.mat = l.head.w >> 8u;
    composite(s, ly);
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
    let grass_green = mixc(spal(s, BP_GRASS_DRY), spal(s, BP_GRASS_WET), smoothstep1(0.25, 0.75, wet + 0.15 * patchv + 0.18 * (1.0 - smoothstep1(6.0, 14.0, (*s).bio.site.x)) * (1.0 - smoothstep1(0.55, 0.8, t.mountain))));
    var grass = mixc(spal(s, BP_GRASS_COLD), grass_green, smoothstep1(-2.0, 8.0, temp));
    grass = grass * vec3<f32>(1.0 + 0.10 * (st.y - 0.5), 1.0 + 0.06 * (st.w - 0.5), 1.0 - 0.08 * (st.y - 0.5));
    let land_n = (*s).pf.f[PF_LAND];
    let cover = smoothstep1(0.08, 0.45, wet + 0.25 * patchv + 0.2 * land_n) * smoothstep1(-9.0, -1.0, temp);
    let semi = smoothstep1(0.02, 0.12, wet) * (1.0 - smoothstep1(0.3, 0.45, wet)) * smoothstep1(-1.0, 4.0, temp);
    let b5 = band(8.0, gsd);
    var tm = 0.17;
    if (semi > 0.0) {
        tm = 0.17 * (1.0 + 0.9 * semi_clumps(p, gsd));
    }
    var cover_px = cover + tm * semi * (1.0 - cover);
    if (semi > 0.0 && b5 > 0.0) {
        let n = perlin3(0x7C55lu, p * (1.0lf / 8.0lf)) + 0.5 * perlin3(0x7C56lu, p * (1.0lf / 3.0lf)) * band(3.0, gsd);
        cover_px = clamp(cover + semi * (2.4 * cover * (1.0 - cover) * n + 0.6 * smoothstep1(-0.1, 0.6, n) * (1.0 - cover)) * b5 + tm * semi * (1.0 - cover) * (1.0 - b5), 0.0, 1.0);
    }
    var col = mixc(soil, grass, cover_px) * (1.0 + 0.16 * land_n);
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
            let dry_p = smoothstep1(0.05, 0.55, perlin3(0x3EADlu, p * (1.0lf / 60.0lf)) + 0.5 * perlin3(0x3EAElu, p * (1.0lf / 22.0lf))) * b30 * cover * (1.0 - smoothstep1(0.45, 0.65, wet));
            col = mixc(col, col * vec3<f32>(1.16, 1.06, 0.80), 0.45 * dry_p * meadow);
        }
        let b12 = band(12.0, gsd);
        if (b12 > 0.0) {
            var m = 0.10 * perlin3(0x3EB1lu, p * (1.0lf / 12.0lf)) * b12;
            let b4 = band(4.0, gsd);
            if (b4 > 0.0) {
                m += 0.08 * perlin3(0x3EB2lu, p * (1.0lf / 4.0lf)) * b4;
                let b13 = band(1.3, gsd);
                if (b13 > 0.0) {
                    m += 0.06 * perlin3(0x3EB3lu, p * (1.0lf / 1.3lf)) * b13;
                }
            }
            col *= 1.0 + m * (cover * meadow + 0.6 * (1.0 - cover));
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
        * (1.0 - 0.5 * smoothstep1(0.3, 0.8, (*s).m.cover) * (1.0 - t.mountain))
        * (1.0 - 0.85 * smoothstep1(0.5, 0.65, t.moist) * smoothstep1(16.0, 21.0, t.temp) * (1.0 - smoothstep1(0.9, 1.5, slope_eff)));
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
    let rwet = max(t.river_wet, 0.85 * smoothstep1(0.26, 0.4, t.moist));
    if (l.river_hw > 0.0 && rwet > 0.3) {
        let ad = abs(l.river_d);
        let belt = 7.0 + 0.6 * min(l.river_hw, 60.0);
        let fwr = max(l.fw, 0.35 * (*s).c.gsd);
        let area = max(min(ad + 0.5 * fwr, l.river_hw + belt) - max(ad - 0.5 * fwr, 0.0), 0.0) / fwr;
        (*s).m.riparian = min(area, 1.0) * smoothstep1(0.18, 0.42, t.moist)
            * smoothstep1(0.3, 0.7, rwet) * (*s).m.natural_ok * (0.55 + 0.45 * smoothstep1(-0.3, 0.3, (*s).pf.f[PF_PATCH]));
    }
}

/// Slot 5 (`layers::agriculture::layer`).
fn layer_agriculture(s: ptr<function, Stack>) {
    let t = (*s).t;
    let fpu = 0.5 + 0.5 * (*s).pf.f[PF_FOREST];
    let cover = smoothstep1(0.12, 0.45, max(t.moist, 0.0)) * smoothstep1(-6.0, 2.0, t.temp) * cfg.tree_density;
    let share = min(0.42 * sveg(s, VEG_WOODLOTS) * (*s).bio.st.veg.w, 0.95);
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
    if (cls == LC_CROP && fr.id != 0lu) {
        ly.inst = instance_id(fr.id);
    }
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
    if (KITS_LEGACY_TOWNS && t.town != 0u && !(*s).no_town && cfg.towns > 0.0) {
        let sel = select_town((*s).c.p, (*s).c.gsd, town_slope, river_clear, (*s).pf.f[PF_WARP2]);
        if (sel.x == -2.0) {
            surface_missing |= MISS_TOWN;
        } else {
            (*s).town_i = i32(sel.x);
            urban = sel.y;
        }
    }
    var cov = 0.0;
    if (KITS_LEGACY_TOWNS && (*s).town_i >= 0) {
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
    if (habit > 0.0 && KITS_LEGACY_ROADS) {
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
    if (KITS_LEGACY_TOWNS && m.road_major_cov > 0.0 && (*s).town_i >= 0) {
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
    if (KITS_LEGACY_TOWNS && (*s).town_i >= 0 && (*s).town.px.ok) {
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
    let cold = (1.0 - smoothstep1(0.0, 8.0, temp)) * (1.0 - smoothstep1(0.35, 0.7, t.mountain));
    let base_cover = smoothstep1(0.3 - 0.2 * cold, 0.68 - 0.36 * cold, wet) * smoothstep1(-6.0 - 3.0 * cold, 2.0 - 2.0 * cold, temp) * td * trees_mult;
    let fpu = 0.5 + 0.5 * (*s).pf.f[PF_FOREST];
    let forest = max(smoothstep1(-0.09, 0.09, base_cover - fpu), m.woodlot * smoothstep1(0.05, 0.4, s_agri(s)));
    let savanna = smoothstep1(0.18, 0.35, wet) * (1.0 - smoothstep1(0.55, 0.7, wet)) * smoothstep1(12.0, 20.0, temp) * 0.12 * sveg(s, VEG_SAVANNA);
    let groves = 0.04 * smoothstep1(0.15, 0.3, wet) * sveg(s, VEG_GROVES);
    let base = m.natural_ok * (1.0 - m.field_cov) * td;
    let clear = 1.0 - 0.85 * smoothstep1(0.05, 0.4, s_agri(s)) * (1.0 - smoothstep1(0.45, 0.7, slope)) * (1.0 - m.woodlot);
    var dens = (forest * 0.9 * clear + savanna + groves) * base;
    var dens_sav = savanna * base;
    dens = max(dens, m.riparian * td);
    let gully_scrub = 0.55 * smoothstep1(0.2, 0.9, -t.gully) * smoothstep1(0.2, 0.5, wet) * m.natural_ok * (1.0 - m.field_cov);
    let snow_n = (*s).pf.f[PF_SNOW];
    let limit = (1.0 - smoothstep1(0.9, 1.4, slope)) * (1.0 - smoothstep1(0.0, 0.6, t.mountain * smoothstep1(-2.0, -6.0, temp)))
        * (1.0 - smoothstep1(-1.2, -2.4, temp + 1.0 * snow_n)) * m.veg;
    dens *= limit;
    dens_sav *= limit;
    let shrub_clim = smoothstep1(0.04, 0.26, wet) * (1.0 - smoothstep1(0.6, 0.8, wet)) * smoothstep1(2.0, 10.0, temp);
    let shrub_patch = max(smoothstep1(-0.2, 0.5, (*s).pf.f[PF_PATCH] + 0.4 * (*s).pf.f[PF_LAND]), 0.6 * (1.0 - smoothstep1(0.2, 0.35, wet)));
    let shrub = clamp((0.45 * shrub_clim * shrub_patch * (1.0 - forest) * max(m.natural_ok, 0.4 * m.rock) * (1.0 - m.field_cov) * sveg(s, VEG_SHRUBS) * style.veg.z + gully_scrub)
        * td * m.veg * (1.0 + 0.9 * (1.0 - smoothstep1(0.3, 0.45, wet)) * semi_clumps(p, gsd)), 0.0, 0.7);
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
        if (cr.share == 3u) {
            colour = mixc(colour, (*s).col, 0.35);
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
    if (m.riparian > 0.0) {
        composite(s, layer_paint(0.85 * m.riparian * limit, mixc(spal(s, BP_GRASS_WET), spal(s, BP_FLOOR), 0.4), 0u));
    }
    let floor_ = smoothstep1(0.25, 0.8, dens);
    composite(s, layer_paint(floor_ * 0.85, spal(s, BP_FLOOR) * 0.7, 0u));
    // the crowns (`it` 0), then each layer's crowns shifted towards the sun (their shadows): one
    // call site of `trees`
    let shadows = (cfg.flags & CF_SHADOWS) != 0u;
    var tr: Trees;
    var shadow = 0.0;
    for (var it = 0u; it <= n; it++) {
        var mask = all;
        var off = vec2<f32>(0.0);
        if (it > 0u) {
            let li = it - 1u;
            if (!shadows || tr.cov >= 0.99 || layers[li].density <= 0.0 || layers[li].cell < 2.0 * gsd) {
                continue;
            }
            mask = 1u << li;
            off = vec2<f32>(cfg.sun_hx, cfg.sun_hy) * (0.6 * layers[li].height / cfg.sun_tan);
        }
        let t = trees(layers, mask, (*s).q_loc + off, gsd, fw, p);
        if (it == 0u) {
            tr = t;
        } else {
            shadow = max(shadow, t.cov);
        }
    }
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
        if (cls != LC_SHRUB) {
            ly.inst = instance_id(stand);
        }
        if ((cfg.flags & CF_TREES_DSM) != 0u) {
            ly.hmode = HM_MAX;
        }
        composite(s, ly);
    }
    if (shadows && tr.cov < 0.99) {
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
    let wn = s_pf_lazy(s, PF_WATER);
    var col = vec3<f32>(0.0);
    if (l.water_kind == W_OCEAN) {
        col = mixc(pal3(PAL_OCEAN_SHALLOW), pal3(PAL_OCEAN_DEEP), smoothstep1(0.0, 28.0, depth));
    } else {
        let sh = 1.0 - smoothstep1(0.0, 1.8, depth * max(1.0 + 0.5 * wn, 0.2));
        col = mixc(mixc(pal3(PAL_LAKE_DEEP), srgb(26.0, 32.0, 26.0), 0.35), mixc(pal3(PAL_RIVER), srgb(78.0, 76.0, 56.0), 0.6), 0.65 * sh * sh);
    }
    col *= 1.0 + 0.06 * wn;
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
    wcol = mixc(wcol, srgb(84.0, 84.0, 58.0), 0.85 * (1.0 - smoothstep1(0.2, 0.4, t.moist)));
    wcol = mixc(wcol, srgb(96.0, 138.0, 140.0), 0.7 * t.mountain * smoothstep1(8.0, 0.0, temp));
    let ice = smoothstep1(-1.5, -4.0, temp + 1.5 * (*s).pf.f[PF_SNOW]);
    wcol = mixc(wcol, mixc(pal3(PAL_SNOW) * 0.9, srgb(170.0, 190.0, 200.0), 0.35 * (0.5 + 0.5 * detail)), ice);
    var plateau = 1.0;
    if (l.river_hw < 15.0) {
        plateau = 1.0 - smoothstep1(0.05, 0.3, t.mesa) * (1.0 - smoothstep1(0.2, 0.4, t.moist));
    }
    let dry_k = 0.4 * (1.0 - 0.7 * (*s).m.riparian) * (1.0 - 0.55 * (1.0 - smoothstep1(0.2, 0.4, t.moist))) * (1.0 - smoothstep1(0.2, 0.45, l.slope)) * plateau;
    let dry_col = mixc((*s).col, mixc(pal3(PAL_GRAVEL), pal3(PAL_SAND + 2u), 0.5) * (1.0 + 0.1 * detail), dry_k);
    let tw = l.river_hw * smoothstep1(0.35, 1.0, wet_r) * (1.0 - 0.25 * (1.0 - smoothstep1(0.9, 1.0, wet_r)) * (1.0 - perlin3(0x51DE7E02lu, (*s).c.p / 260.0lf)));
    let off = (l.river_hw - tw) * clamp(1.8 * perlin3(0x51DE7E01lu, (*s).c.p / 420.0lf), -1.0, 1.0);
    let adt = abs(l.river_d - off);
    let adb = abs(l.river_d);
    let ovt = max(min(tw, adt + 0.5 * fwr) - max(-tw, adt - 0.5 * fwr), 0.0);
    let ovb = max(min(l.river_hw, adb + 0.5 * fwr) - max(-l.river_hw, adb - 0.5 * fwr), 1e-9);
    let wt = min(ovt / ovb, 1.0);
    let rc = mixc(mixc(dry_col, wcol, 0.6 * wet_r), wcol, wt);
    let wet = wt > 0.5;
    let frozen = ice > 0.5 && wet;
    var cls = LC_SAND;
    if (frozen) {
        cls = LC_SNOW;
    } else if (wet) {
        cls = LC_RIVER;
    }
    // (the snow cover leaves open water free, and frozen rivers a faint line)
    (*s).m.river_cov = cov * select(select(0.0, 1.0, wet), 0.4, frozen);
    var ly = layer_paint(cov, rc, cls);
    ly.dh = l.river_level;
    ly.hmode = HM_ABS;
    if (tw <= 0.0) {
        ly.dh = -0.3;
        ly.hmode = HM_BLEND;
    }
    ly.relit = 1.0;
    ly.water = !frozen && wet;
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

/// A sub-sample's stack before its first layer (`stack::eval`), with its biomes `bio`
/// (`bio_at(t, l.eco_edge, c)`).
fn stack_new(c: Ctx, t: Terrain, l: Local, pf: PixFields, bio: Bio) -> Stack {
    var s: Stack;
    s.c = c;
    s.t = t;
    s.l = l;
    s.pf = pf;
    s.bio = bio;
    s.ri = -1;
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
    s.inst = 0u;
    s.lit = 1.0;
    s.emission = vec3<f32>(0.0);
    s.is_water = false;
    s.done = false;
    s.no_town = false;
    s.deferring = false;
    s.def = deferred_none();
    return s;
}

/// Is the sample standing water (step 10 only)?
fn stack_water(l: Local) -> bool {
    return l.water > l.ground && l.water_kind != W_NONE;
}

// The steps of the stack: 0..9 on land, 10 on standing water; each runs its core layer and
// returns its slot (`SLOT_SKIP`: the seasonal step skipped, the stack is done). One function
// per step: pass B runs them in stages (`stack_stage_<n>`, generated by `gpu::stages_wgsl`): each stage one pipeline,
// with ONE call site of `biome_layers` and of the kits' slot dispatcher (the drivers inline
// every call). `stack.rs::eval` is the CPU's sequence.
const SLOT_SKIP: u32 = 0xffffffffu;

fn stack_step_0(sp: ptr<function, Stack>) -> u32 {
    snow_prepare(sp);
    s_frame(sp);
    layer_zonal(sp);
    return SLOT_ZONAL;
}

fn stack_step_1(sp: ptr<function, Stack>) -> u32 {
    return SLOT_ALTITUDINAL;
}

fn stack_step_2(sp: ptr<function, Stack>) -> u32 {
    layer_azonal(sp);
    return SLOT_AZONAL;
}

fn stack_step_3(sp: ptr<function, Stack>) -> u32 {
    finish_ground(sp);
    return SLOT_DISTURBANCE;
}

fn stack_step_4(sp: ptr<function, Stack>) -> u32 {
    layer_agriculture(sp);
    return SLOT_AGRICULTURE;
}

fn stack_step_5(sp: ptr<function, Stack>) -> u32 {
    (*sp).deferring = true;
    select_town_layer(sp);
    layer_linear(sp);
    return SLOT_LINEAR;
}

fn stack_step_6(sp: ptr<function, Stack>) -> u32 {
    layer_built(sp);
    return SLOT_BUILT;
}

fn stack_step_7(sp: ptr<function, Stack>) -> u32 {
    (*sp).deferring = false;
    layer_canopy(sp);
    return SLOT_CANOPY;
}

fn stack_step_8(sp: ptr<function, Stack>) -> u32 {
    st_flush(sp);
    layer_rivers(sp);
    return SLOT_WATER;
}

fn stack_step_9(sp: ptr<function, Stack>) -> u32 {
    // (the seasonal layers skip water: ice over it clears `is_water`)
    if ((*sp).done || (*sp).is_water) {
        return SLOT_SKIP;
    }
    layer_snow(sp);
    return SLOT_SEASONAL;
}

fn stack_step_10(sp: ptr<function, Stack>) -> u32 {
    layer_standing(sp);
    return SLOT_WATER;
}

/// The surface of a finished stack.
fn stack_surface(sp: ptr<function, Stack>) -> Surface {
    var o: Surface;
    if ((*sp).is_water) {
        o.albedo = (*sp).col;
    } else {
        o.albedo = max((*sp).col, vec3<f32>(0.0));
    }
    o.height = (*sp).height;
    o.cls = (*sp).cls;
    o.inst = (*sp).inst;
    o.lit = (*sp).lit;
    o.emission = (*sp).emission;
    return o;
}

/// `ground::semi_clumps`.
/// Clumped density (`noise::clumped`).
fn clumped(d: f32, c0: f32, p: vec3<f64>, gsd: f32) -> f32 {
    let b1 = band(160.0, gsd);
    let b2 = band(34.0, 2.0 * gsd);
    let c = c0;
    if (c <= 0.0 || d <= 0.0 || d >= 1.0 || b1 <= 0.0) {
        return d;
    }
    let a2 = 0.8;
    var r = perlin3(0xC1A7lu, p * (1.0lf / 160.0lf)) * b1;
    if (b2 * a2 > 0.0) {
        r += a2 * perlin3(0xC1A8lu, p * (1.0lf / 34.0lf)) * b2;
    }
    let z = r / (0.211 * sqrt(1.0 + a2 * a2));
    let u2 = ((1.0 - b1 * b1) + a2 * a2 * (1.0 - b2 * b2)) / (1.0 + a2 * a2);
    let w2 = 0.22 * 0.22;
    let q = log(d / (1.0 - d)) / 1.702 * sqrt(1.0 + w2);
    let sv0 = 1.0 / (1.0 + exp(-1.702 * (q - z) / sqrt(u2 + w2)));
    let fl = 1.0 - smoothstep1(0.0, 0.4, u2);
    let sv = sv0 * (1.0 - fl + fl * smoothstep1(0.03, 0.15, sv0));
    return d + (sv - d) * c;
}

fn semi_clumps(p: vec3<f64>, gsd: f32) -> f32 {
    var v = 1.1 * perlin3(0x5B3lu, p * (1.0lf / 250.0lf)) * band(250.0, gsd) + 0.8 * perlin3(0x5B1lu, p * (1.0lf / 90.0lf)) * band(90.0, gsd);
    let b = band(34.0, gsd);
    if (b > 0.0) {
        v += 0.45 * perlin3(0x5B2lu, p * (1.0lf / 34.0lf)) * b;
    }
    return clamp(v, -1.0, 1.0);
}
