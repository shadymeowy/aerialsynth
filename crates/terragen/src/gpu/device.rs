//! Headless wgpu device (no surface, no window), shared by the tile generator and the renderer.

use anyhow::{anyhow, bail, Result};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// The first error the device reported outside an error scope (validation, out of memory),
    /// or its loss. Sticky: every later [`Gpu::check`] fails.
    failed: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

/// Record `e` as the device's failure unless one is recorded already.
fn record(slot: &std::sync::Mutex<Option<String>>, e: String) {
    let mut g = slot.lock().unwrap_or_else(|p| p.into_inner());
    if g.is_none() {
        *g = Some(e);
    }
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
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("terrain"),
                    required_features: features,
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await?;
            // errors outside an error scope are recorded (wgpu's default handler panics) and make
            // the next GPU call of the generator or renderer fail
            let failed = std::sync::Arc::new(std::sync::Mutex::new(None));
            let f = failed.clone();
            device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| record(&f, e.to_string())));
            let f = failed.clone();
            device.set_device_lost_callback(move |reason, msg| record(&f, format!("device lost ({reason:?}): {msg}")));
            Ok(Gpu { device, queue, info, failed })
        })
    }

    /// Fails once the device has reported an error outside an error scope, or was lost.
    pub fn check(&self) -> Result<()> {
        match self.failed.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            Some(e) => bail!("the GPU ({}) failed: {e}", self.info.name),
            None => Ok(()),
        }
    }

    /// Map `slices` for reading and wait for them (errors instead of panics when the read-back
    /// fails: device lost, out of memory).
    pub fn map_read(&self, slices: &[wgpu::BufferSlice<'_>]) -> Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        for s in slices {
            let tx = tx.clone();
            s.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        }
        drop(tx);
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("GPU poll: {e}"))?;
        self.check()?;
        for _ in slices {
            // (the work is done, but another thread's poll may still be running the callback)
            let mut waited = 0;
            let r = loop {
                match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(r) => break r,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) if waited < 1200 => {
                        waited += 1;
                        self.device.poll(wgpu::PollType::Poll).map_err(|e| anyhow!("GPU poll: {e}"))?;
                    }
                    Err(_) => bail!("GPU read-back did not complete"),
                }
            };
            r.map_err(|e| anyhow!("GPU read-back: {e}"))?;
        }
        Ok(())
    }

    /// Run `f` (GPU work of this thread) in error scopes: an out-of-memory, validation or
    /// internal error it causes is returned as an error (and does not mark the device failed).
    pub fn scoped<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let val = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let int = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let r = f();
        let e_int = pollster::block_on(int.pop());
        let e_val = pollster::block_on(val.pop());
        let e_oom = pollster::block_on(oom.pop());
        if let Some(e) = e_oom.or(e_val).or(e_int) {
            bail!("GPU error: {e}");
        }
        self.check()?;
        r
    }

    /// Fails unless the device has the shader features of the generator's WGSL (f64, i64).
    pub fn check_shaders(&self) -> Result<()> {
        let missing = GEN_FEATURES - self.device.features();
        if !missing.is_empty() {
            bail!("the GPU ({}) lacks shader features the tile generator needs: {missing:?}", self.info.name);
        }
        Ok(())
    }

    /// Fails unless the device can run the tile generator (its shader features and buffer sizes).
    pub fn check_generator(&self) -> Result<()> {
        self.check_shaders()?;
        // the largest buffer the generator binds: the drainage lattice's points
        let need = (super::LAT_CAP * 32) as u64;
        let l = self.device.limits();
        if l.max_storage_buffer_binding_size < need || l.max_buffer_size < need {
            bail!(
                "the GPU ({}) binds at most {} MiB per storage buffer; the tile generator needs {} MiB",
                self.info.name,
                l.max_storage_buffer_binding_size.min(l.max_buffer_size) >> 20,
                need >> 20
            );
        }
        Ok(())
    }
}

static SHARED: std::sync::OnceLock<Result<std::sync::Arc<Gpu>, String>> = std::sync::OnceLock::new();

/// The process-wide device (generator, renderer and event sensor share it).
pub fn shared() -> Result<std::sync::Arc<Gpu>> {
    SHARED.get_or_init(|| Gpu::new().map(std::sync::Arc::new).map_err(|e| e.to_string())).clone().map_err(|e| anyhow!("{e}"))
}
