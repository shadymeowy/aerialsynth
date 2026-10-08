//! Headless wgpu device (no surface, no window), shared by the tile generator and the renderer.

use anyhow::{anyhow, bail, Result};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

/// Shader features the tile generator needs: f64 for the noise lattice coordinates (ECEF metres
/// over wavelengths down to decimetres), u64 for the hashes (the same as the CPU's) and 64-bit
/// atomics for the drainage lattice's hash table.
pub const GEN_FEATURES: wgpu::Features = wgpu::Features::SHADER_F64.union(wgpu::Features::SHADER_INT64).union(wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS);

impl Gpu {
    /// The first high-performance adapter (Vulkan / Metal / DX12), headless, with the
    /// generator's features where the adapter has them.
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
            let features = adapter.features() & (GEN_FEATURES | wgpu::Features::PIPELINE_CACHE);
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor { label: Some("terrain"), required_features: features, required_limits: adapter.limits(), ..Default::default() })
                .await?;
            Ok(Gpu { device, queue, info })
        })
    }

    /// Fails unless the device can run the tile generator.
    pub fn check_generator(&self) -> Result<()> {
        let missing = GEN_FEATURES - self.device.features();
        if !missing.is_empty() {
            bail!("the GPU ({}) lacks shader features the tile generator needs: {missing:?}", self.info.name);
        }
        Ok(())
    }
}

static SHARED: std::sync::OnceLock<Result<std::sync::Arc<Gpu>, String>> = std::sync::OnceLock::new();

/// The process-wide device (generator, renderer and event sensor share it).
pub fn shared() -> Result<std::sync::Arc<Gpu>> {
    SHARED.get_or_init(|| Gpu::new().map(std::sync::Arc::new).map_err(|e| e.to_string())).clone().map_err(|e| anyhow!("{e}"))
}
