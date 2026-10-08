// Fine-scale surface ("pass B", `surface.rs`): pixel fields, and the surface at one sub-sample.

// ---------------------------------------------------------------- pixel fields

const PF_N: u32 = 16u;
const PF_DETAIL: u32 = 0u;
const PF_PATCH: u32 = 1u;
const PF_LAND: u32 = 2u;
const PF_STRATA: u32 = 3u;
const PF_STRATA2: u32 = 4u;
const PF_SNOW: u32 = 5u;
const PF_FOREST: u32 = 6u;
const PF_STAND: u32 = 7u;
const PF_FIELD_VAR: u32 = 8u;
const PF_FIELD_VAR2: u32 = 9u;
const PF_FIELD_VAR3: u32 = 10u;
const PF_WARP2: u32 = 11u;
const PF_WATER: u32 = 12u;
/// the fields only some surfaces need, completed on first use
const PF_LAZY: u32 = (1u << 3u) | (1u << 4u) | (1u << 7u) | (1u << 8u) | (1u << 9u) | (1u << 10u) | (1u << 12u);

/// all octaves / the octaves >= cut / the octaves < cut
const SPLIT_ALL: u32 = 0u;
const SPLIT_LOW: u32 = 1u;
const SPLIT_HIGH: u32 = 2u;

/// fBm `f` evaluated at p·k (a field at p·k sees sample spacing gsd·k in its own domain).
fn pf_fbm(f: u32, k: f64, p: vec3<f64>, gsd: f32, split: u32, cut: f32) -> f32 {
    let kf = f32(k);
    if (split == SPLIT_ALL) {
        return fbm(f, p * k, gsd * kf);
    }
    return fbm_part(f, p * k, gsd * kf, cut * kf, split == SPLIT_LOW);
}

/// A single-octave term of wavelength `lam` belongs to the part?
fn pf_single(lam: f32, split: u32, cut: f32) -> bool {
    if (split == SPLIT_ALL) {
        return true;
    }
    return (lam >= cut) == (split == SPLIT_LOW);
}

/// Pixel field `i` (or a part of it, `SurfaceModel::pixel_field`).
fn pixel_field(i: u32, p: vec3<f64>, gsd: f32, split: u32, cut: f32) -> f32 {
    switch i {
        case 0u: { return pf_fbm(FBM_DETAIL, 1.0lf, p, gsd, split, cut); }
        case 1u: { return pf_fbm(FBM_PATCH, 1.0lf, p, gsd, split, cut); }
        case 2u: { return pf_fbm(FBM_LAND, 1.0lf, p, gsd, split, cut) * fbm_norm(FBM_LAND) * 1.8; }
        case 3u: { return pf_fbm(FBM_STRATA, 1.0lf, p, gsd, split, cut); }
        case 4u: { return pf_fbm(FBM_STRATA, 1.7lf, p, gsd, split, cut); }
        case 5u: { return pf_fbm(FBM_SNOW, 1.0lf, p, gsd, split, cut); }
        case 6u: { return pf_fbm(FBM_FOREST, 1.0lf, p, gsd, split, cut) * fbm_norm(FBM_FOREST) * 1.8; }
        case 7u: {
            var v = pf_fbm(FBM_PATCH, 0.3lf, p, gsd, split, cut);
            if (pf_single(1200.0, split, cut)) {
                v += 0.5 * perlin3(0x57Alu, p / 1200.0lf) * band(1200.0, gsd);
            }
            return v;
        }
        case 8u: { return pf_fbm(FBM_FIELD_VAR, 1.0lf, p, gsd, split, cut); }
        case 9u: { return pf_fbm(FBM_FIELD_VAR, 1.7lf, p, gsd, split, cut); }
        case 10u: { return pf_fbm(FBM_FIELD_VAR, 3.0lf, p, gsd, split, cut); }
        case 11u: { return pf_fbm(FBM_WARP2, 1.0lf, p, gsd, split, cut); }
        case 12u: { return pf_fbm(FBM_PATCH, 0.37lf, p, gsd, split, cut); }
        default: {
            if (pf_single(180.0, split, cut)) {
                return perlin3(0x57A1lu + u64(i - 13u), p / 180.0lf);
            }
            return 0.0;
        }
    }
}

/// Forest stand at `p` given the stand warp (`SurfaceModel::stand_id`).
fn stand_id(p: vec3<f64>, warp: vec3<f32>) -> u64 {
    let sp = p + vec3<f64>(warp * 70.0);
    return worley3(0x57A4lu, sp, 240.0lf, 0.9).id;
}

fn stand_warp_at(p: vec3<f64>) -> vec3<f32> {
    return vec3<f32>(perlin3(0x57A1lu, p / 180.0lf), perlin3(0x57A2lu, p / 180.0lf), perlin3(0x57A3lu, p / 180.0lf));
}

/// Grid nodes: the long octaves of the pixel fields and the forest stand.
fn grid_nodes_surface(ti: TileInfo, k: u32, ctx: Ctx, nb: u32, ib: u32) {
    for (var i = 0u; i < PF_N; i++) {
        node_f[nb + NODE_PF + i] = pixel_field(i, ctx.p, ctx.gsd, SPLIT_LOW, ti.pf_cut);
    }
    node_ids[ib + 6u] = stand_id(ctx.p, stand_warp_at(ctx.p));
}
