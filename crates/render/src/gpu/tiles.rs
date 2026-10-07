//! GPU-resident tile cache: a fixed pool of slots in texture arrays (one layer per tile and
//! layer kind), least-recently-used eviction, uploads only for tiles that are not resident.
//! A per-frame lookup table (open addressing, in a storage buffer) maps the frame's tile ids to
//! their slots, so shaders can sample neighbours and ancestors like the CPU `TileView`.

use crate::raster::Shading;
use geodesy::tiles::TileId;
use std::collections::HashMap;
use tilestore::TileData;

const N: u32 = 256;

/// Layer flags in the lookup table (which layers a tile has).
pub const HAS_COLOR: u32 = 1;
pub const HAS_NORMAL: u32 = 2;
pub const HAS_EMISSION: u32 = 4;
pub const HAS_ELEV: u32 = 8;
pub const HAS_LC: u32 = 16;

pub struct TilePool {
    pub slots: u32,
    shading: Shading,
    pub color: wgpu::Texture,
    pub normal: wgpu::Texture,
    pub emission: wgpu::Texture,
    pub elev: wgpu::Texture,
    pub lc: wgpu::Texture,
    pub blockmax: wgpu::Texture,
    /// tile → (slot, layer flags, last use)
    resident: HashMap<TileId, (u32, u32, u64)>,
    free: Vec<u32>,
    tick: u64,
    /// tiles uploaded so far (statistics)
    pub uploads: u64,
}

fn tex(device: &wgpu::Device, label: &str, size: u32, layers: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: layers },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn write(queue: &wgpu::Queue, t: &wgpu::Texture, slot: u32, size: u32, bytes_per_px: u32, data: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: t, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: slot }, aspect: wgpu::TextureAspect::All },
        data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * bytes_per_px), rows_per_image: Some(size) },
        wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
    );
}

impl TilePool {
    pub fn new(device: &wgpu::Device, slots: u32, shading: Shading) -> Self {
        TilePool {
            slots,
            shading,
            color: tex(device, "tile color", N, slots, wgpu::TextureFormat::Rgba8UnormSrgb),
            normal: tex(device, "tile normal", N, slots, wgpu::TextureFormat::Rgba8Snorm),
            emission: tex(device, "tile emission", N, slots, wgpu::TextureFormat::Rgba8Unorm),
            elev: tex(device, "tile elevation", N, slots, wgpu::TextureFormat::R32Float),
            lc: tex(device, "tile landcover", N, slots, wgpu::TextureFormat::R8Uint),
            blockmax: tex(device, "tile blockmax", 16, slots, wgpu::TextureFormat::R32Float),
            resident: HashMap::new(),
            free: (0..slots).rev().collect(),
            tick: 0,
            uploads: 0,
        }
    }

    /// Make `tiles` resident (uploading those that are not) and return (id, slot, flags) per tile.
    /// Tiles of this call are never evicted by it; at most `slots` tiles per frame.
    pub fn ensure(&mut self, queue: &wgpu::Queue, tiles: &[(TileId, &TileData)]) -> Vec<(TileId, u32, u32)> {
        self.tick += 1;
        let tick = self.tick;
        let mut out = Vec::with_capacity(tiles.len());
        for (id, t) in tiles.iter().take(self.slots as usize) {
            if let Some(e) = self.resident.get_mut(id) {
                e.2 = tick;
                out.push((*id, e.0, e.1));
                continue;
            }
            let slot = match self.free.pop() {
                Some(s) => s,
                None => {
                    // evict the least recently used tile not used in this frame
                    let (&victim, _) = self.resident.iter().filter(|(_, e)| e.2 != tick).min_by_key(|(_, e)| e.2).expect("tile pool too small for one frame");
                    self.resident.remove(&victim).unwrap().0
                }
            };
            let flags = self.upload(queue, slot, t);
            self.resident.insert(*id, (slot, flags, tick));
            out.push((*id, slot, flags));
        }
        out
    }

    fn upload(&mut self, queue: &wgpu::Queue, slot: u32, t: &TileData) -> u32 {
        self.uploads += 1;
        let px = (N * N) as usize;
        let mut flags = 0;
        let color = match self.shading {
            Shading::Relit => &t.albedo,
            Shading::Satellite => &t.rgb,
        };
        if color.len() == 3 * px {
            let mut b = vec![255u8; 4 * px];
            for k in 0..px {
                b[4 * k..4 * k + 3].copy_from_slice(&color[3 * k..3 * k + 3]);
            }
            write(queue, &self.color, slot, N, 4, &b);
            flags |= HAS_COLOR;
        }
        if t.normal.len() == 3 * px {
            let mut b = vec![0u8; 4 * px];
            for k in 0..px {
                for c in 0..3 {
                    b[4 * k + c] = t.normal[3 * k + c] as u8;
                }
            }
            write(queue, &self.normal, slot, N, 4, &b);
            flags |= HAS_NORMAL;
        }
        if t.emission.len() == 3 * px {
            let mut b = vec![0u8; 4 * px];
            for k in 0..px {
                b[4 * k..4 * k + 3].copy_from_slice(&t.emission[3 * k..3 * k + 3]);
            }
            write(queue, &self.emission, slot, N, 4, &b);
            flags |= HAS_EMISSION;
        }
        if t.elevation.len() == px {
            write(queue, &self.elev, slot, N, 4, bytemuck::cast_slice(&t.elevation));
            let mut bm = [f32::MIN; 256];
            for j in 0..256 {
                for i in 0..256 {
                    let k = (j / 16) * 16 + i / 16;
                    bm[k] = bm[k].max(t.elevation[j * 256 + i]);
                }
            }
            write(queue, &self.blockmax, slot, 16, 4, bytemuck::cast_slice(&bm));
            flags |= HAS_ELEV;
        }
        if t.landcover.len() == px {
            write(queue, &self.lc, slot, N, 1, &t.landcover);
            flags |= HAS_LC;
        }
        flags
    }
}

/// Hash of a tile id for the lookup table (must match `tile_hash` in shade.wgsl).
pub fn tile_hash(z: u32, x: u32, y: u32) -> u32 {
    let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ z.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h
}

pub const EMPTY: u32 = 0xFFFF_FFFF;

/// Open-addressing table: entries (z, x, y, slot | flags << 16); `EMPTY` in w = free.
/// Returns (entries, mask).
pub fn lookup_table(tiles: &[(TileId, u32, u32)]) -> (Vec<[u32; 4]>, u32) {
    let cap = (tiles.len() * 2).next_power_of_two().max(64);
    let mask = (cap - 1) as u32;
    let mut t = vec![[0, 0, 0, EMPTY]; cap];
    for (id, slot, flags) in tiles {
        let mut i = tile_hash(id.z as u32, id.x, id.y) & mask;
        while t[i as usize][3] != EMPTY {
            i = (i + 1) & mask;
        }
        t[i as usize] = [id.z as u32, id.x, id.y, slot | (flags << 16)];
    }
    (t, mask)
}
