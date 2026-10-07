//! Headless wgpu device (no surface, no window).

use anyhow::{anyhow, Result};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

impl Gpu {
    /// The first high-performance adapter (Vulkan / Metal / DX12), headless.
    pub fn new() -> Result<Gpu> {
        pollster::block_on(async {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = wgpu::Backends::PRIMARY;
            let instance = wgpu::Instance::new(desc);
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
                .await
                .map_err(|e| anyhow!("no GPU adapter: {e}"))?;
            let info = adapter.get_info();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor { label: Some("render"), required_limits: adapter.limits(), ..Default::default() })
                .await?;
            Ok(Gpu { device, queue, info })
        })
    }
}
