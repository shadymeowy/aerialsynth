//! Tile generation on the GPU (wgpu compute): the generator of `tile.rs` / `world.rs` /
//! `surface.rs` in WGSL, with the same hashes and noise frames, so it builds the same world.

pub mod device;
pub(crate) mod tables;

pub use device::{shared, Gpu};

use wgpu::util::DeviceExt;

/// WGSL sources, in dependency order.
pub(crate) const NOISE_WGSL: &str = include_str!("wgsl/noise.wgsl");

/// A storage buffer holding `data`.
pub(crate) fn storage<T: bytemuck::Pod>(d: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    if bytes.is_empty() {
        return d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    }
    d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST })
}

/// An output storage buffer of `size` bytes (copyable to a read-back buffer).
pub(crate) fn output(d: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Copy `buf` back to the host (waits for the GPU).
pub(crate) fn read_back<T: bytemuck::Pod>(g: &Gpu, buf: &wgpu::Buffer, n: usize) -> anyhow::Result<Vec<T>> {
    let bytes = (n * std::mem::size_of::<T>()) as u64;
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let rb = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("read-back"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &rb, 0, bytes);
    g.queue.submit([enc.finish()]);
    rb.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("GPU read-back"));
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let out = bytemuck::cast_slice::<u8, T>(&rb.slice(..).get_mapped_range()?).to_vec();
    rb.unmap();
    Ok(out)
}

/// A compute pipeline of `src` with its layout derived from the shader.
pub(crate) fn pipeline(d: &wgpu::Device, label: &str, src: &str, entry: &str) -> wgpu::ComputePipeline {
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
    d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: None, module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
}

#[cfg(test)]
mod tests;
