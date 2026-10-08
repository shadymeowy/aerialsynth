//! Host mirrors of the WGSL structs (layouts as WGSL computes them; see the size tests).

use bytemuck::{Pod, Zeroable};

/// `Cfg` of world.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GCfg {
    pub home_p: [f64; 4],
    pub lvl_inv_mlam: [f64; 4],
    pub lvl_inv_fplam: [f64; 4],
    pub inv_gully_lam: f64,
    pub lake_cell: f64,
    pub region_cell: f64,
    pub town_cell: f64,
    pub seed: u64,
    pub nlevels: u32,
    pub flags: u32,
    pub cont_warp: f32,
    pub cont_threshold: f32,
    pub home_r: f32,
    pub home_st: f32,
    pub mtn_height: f32,
    pub hill_height: f32,
    pub micro_height: f32,
    pub mesas: f32,
    pub dune_height: f32,
    pub erosion: f32,
    pub gully_lam: f32,
    pub lake_density: f32,
    pub eq_temp: f32,
    pub pole_drop: f32,
    pub lapse: f32,
    pub moist_bias: f32,
    pub tree_density: f32,
    pub agriculture: f32,
    pub towns: f32,
    pub roads: f32,
    pub sun_hx: f32,
    pub sun_hy: f32,
    pub sun_tan: f32,
    pub ambient: f32,
    pub direct: f32,
    pub exposure: f32,
    pub haze: f32,
    pub l0: f32,
    pub sun_e: f32,
    pub sun_n: f32,
    pub sun_u: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub _p: [f32; 3],
    pub lvl_a: [[f32; 4]; 4],
    pub lvl_b: [[f32; 4]; 4],
}

pub(crate) const CF_RIVERS: u32 = 1;
pub(crate) const CF_TREES_DSM: u32 = 2;
pub(crate) const CF_BUILDINGS_DSM: u32 = 4;
pub(crate) const CF_SHADOWS: u32 = 8;
pub(crate) const CF_ADAPTIVE: u32 = 16;

/// `Terrain` of world.wgsl: the pass-A result.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct GTerrain {
    pub ground: f32,
    pub water: f32,
    pub water_kind: u32,
    pub river_d: f32,
    pub river_hw: f32,
    pub river_level: f32,
    pub river_wet: f32,
    pub temp: f32,
    pub moist: f32,
    pub mountain: f32,
    pub rock_expect: f32,
    pub sand: f32,
    pub floodplain: f32,
    pub mesa: f32,
    pub cont: f32,
    pub agri: f32,
    pub habit: f32,
    pub gully: f32,
    pub road_major: f32,
    pub road_minor: f32,
    pub region_edge: f32,
    pub town: u32,
    pub _p: [u32; 2],
    pub style: [f32; 4],
    pub region_id: u64,
    pub region_id2: u64,
}

/// `Seg` of world.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GSeg {
    pub a: [f64; 4],
    pub b: [f64; 4],
    pub ha: f32,
    pub hb: f32,
    pub hw: f32,
    pub valley: f32,
    pub hw_b: f32,
    pub level: u32,
    pub _p: [u32; 2],
}

impl GSeg {
    pub fn from(s: &crate::world::Seg) -> GSeg {
        GSeg {
            a: [s.a.x, s.a.y, s.a.z, 0.0],
            b: [s.b.x, s.b.y, s.b.z, 0.0],
            ha: s.ha as f32,
            hb: s.hb as f32,
            hw: s.hw as f32,
            valley: s.valley as f32,
            hw_b: s.hw_b as f32,
            level: s.level as u32,
            _p: [0; 2],
        }
    }
}

/// `Sink` of world.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GSink {
    pub c: [f64; 4],
    pub id: u64,
    pub rad: f32,
    pub level: f32,
    pub _p: [u64; 2],
}

/// `Drain` of world.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GDrain {
    pub seg0: u32,
    pub nseg: u32,
    pub sink0: u32,
    pub nsink: u32,
}

/// `PointIn` of points.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GPointIn {
    pub p: [f64; 4],
    pub sl: f32,
    pub cl: f32,
    pub so: f32,
    pub co: f32,
    pub lat: f32,
    pub gsd: f32,
    pub mode: u32,
    pub _p: u32,
    pub dr: GDrain,
    pub _q: [u32; 4],
}

/// `Row` of tile_a.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GRow {
    pub ncl: f64,
    pub z: f64,
    pub sl: f32,
    pub cl: f32,
    pub lat: f32,
    pub gsd: f32,
    pub gsd_ns: f32,
    pub _p: f32,
}

/// `Col` of tile_a.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GCol {
    pub co: f64,
    pub so: f64,
}

/// `TileInfo` of tile_a.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GTileInfo {
    pub z: u32,
    pub flags: u32,
    pub ng: u32,
    pub pre_flags: u32,
    pub pf_cut: f32,
    pub relief_cut_r: f32,
    pub relief_cut_h: f32,
    pub gu0: f32,
    pub row_a: u32,
    pub col_a: u32,
    pub row_n: u32,
    pub col_n: u32,
    pub row_b: u32,
    pub col_b: u32,
    pub node0: u32,
    pub pix0: u32,
    pub bin0: u32,
    pub seg0: u32,
    pub nseg: u32,
    pub sink0: u32,
    pub nsink: u32,
    pub ss: u32,
}

/// `Region` of surface.wgsl (`RegionInfo`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GRegion {
    pub center: [f64; 4],
    pub ex: [f32; 4],
    pub ey: [f32; 4],
    pub east: [f32; 4],
    pub north: [f32; 4],
    pub split: u64,
    pub style: u32,
    pub _p: u32,
    pub fw: f32,
    pub fh: f32,
    pub hedge: f32,
    pub track: f32,
    pub border_w: f32,
    pub palette: f32,
    pub agri: f32,
    pub season: f32,
    pub _q: [f32; 4],
}

/// `Town` of surface.wgsl (`TownInfo` of an existing town).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GTown {
    pub center: [f64; 4],
    pub inv_r09: f64,
    pub inv_r035: f64,
    pub seed: u64,
    pub _p: u64,
    pub ex: [f32; 4],
    pub ey: [f32; 4],
    pub sun: [f32; 4],
    pub radius: f32,
    pub block: f32,
    pub street: f32,
    pub organic: f32,
    pub roof_style: f32,
    pub height: f32,
    pub lot: f32,
    pub elong: f32,
    pub _q: [f32; 4],
}

/// `SiteReq` / `LakeReq` of the tile kernels: a site whose data the host is to provide.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GSiteReq {
    pub pt: [f64; 4],
    pub id: u64,
    pub _p: [u64; 3],
}

pub(crate) const TF_GRID: u32 = 1;
pub(crate) const TF_WARP_GRID: u32 = 2;
pub(crate) const TF_GULLY_GRID: u32 = 4;
pub(crate) const TF_ROADS_GRID: u32 = 8;

pub(crate) const MODE_FULL: u32 = 0;
pub(crate) const MODE_NOLAKES: u32 = 1;
pub(crate) const MODE_RELIEF: u32 = 2;

pub(crate) const W_NONE: u32 = 0;

/// "no value" of the GPU structs (`NONE_F`).
pub(crate) const NONE_F: f32 = 3.0e38;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_match_wgsl() {
        assert_eq!(std::mem::size_of::<GCfg>(), 416);
        assert_eq!(std::mem::size_of::<GTerrain>(), 128);
        assert_eq!(std::mem::size_of::<GSeg>(), 96);
        assert_eq!(std::mem::size_of::<GSink>(), 64);
        assert_eq!(std::mem::size_of::<GPointIn>(), 96);
        assert_eq!(std::mem::size_of::<GRow>(), 40);
        assert_eq!(std::mem::size_of::<GTileInfo>(), 88);
        assert_eq!(std::mem::size_of::<GRegion>(), 160);
        assert_eq!(std::mem::size_of::<GTown>(), 160);
        assert_eq!(std::mem::size_of::<GSiteReq>(), 64);
    }
}
