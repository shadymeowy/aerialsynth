//! Headless wgpu device (no surface, no window), shared by the tile generator and the renderer.

use anyhow::{anyhow, bail, Result};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// The first error the device reported outside an error scope (validation, out of memory),
    /// or its loss. Sticky: every later [`Gpu::check`] fails.
    failed: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Held by the one thread at a time that blocks in `poll` waiting for the GPU (see
    /// [`Gpu::map_read`]); no callback that `poll` runs takes it.
    waiter: std::sync::Mutex<()>,
    /// Read-backs completed (by any thread): the GPU's progress, for [`STALL`].
    done: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Held while pipelines are created (see [`Gpu::compiling`]).
    compile: std::sync::Mutex<()>,
}

/// How long one blocking wait for the GPU lasts before the waiter looks at its read-backs again.
const WAIT_SLICE: std::time::Duration = std::time::Duration::from_millis(100);

/// A read-back fails, and the device is marked failed, when no read-back of any thread has
/// completed for this long: the GPU stopped executing the queue (it never recovers), and
/// waiting longer would hang the caller instead of letting it fall back to the CPU.
const STALL: std::time::Duration = std::time::Duration::from_secs(120);

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

/// The environment variable choosing the GPU (see [`pick_adapter`]).
pub const GPU_ENV: &str = "AERIALSYNTH_GPU";

/// The adapter `AERIALSYNTH_GPU` asks for among `adapters`: an index into the list (as
/// `adapter_list` prints it), a PCI bus id (`0000:83:00.0`), or a case-insensitive part of
/// the adapter name (`6000`, `p4`); `Ok(None)` when the variable is unset or empty. A
/// selection that matches nothing is an error, never a silent fallback to another GPU.
/// `none` is no GPU: an error, so that generation and rendering run on the CPU.
pub fn pick_adapter(adapters: &[wgpu::Adapter]) -> Result<Option<wgpu::Adapter>> {
    let Ok(sel) = std::env::var(GPU_ENV) else { return Ok(None) };
    let sel = sel.trim().to_lowercase();
    if sel.is_empty() {
        return Ok(None);
    }
    if sel == "none" {
        bail!("{GPU_ENV}=none: no GPU");
    }
    let infos: Vec<wgpu::AdapterInfo> = adapters.iter().map(|a| a.get_info()).collect();
    let bus = |i: &wgpu::AdapterInfo| i.device_pci_bus_id.to_lowercase();
    let found = if let Ok(k) = sel.parse::<usize>() {
        (k < adapters.len()).then_some(k)
    } else {
        // a PCI bus id, with or without the domain (`83:00.0`), else a part of the name
        let pci = infos.iter().position(|i| {
            !bus(i).is_empty() && (bus(i) == sel || bus(i).ends_with(&format!(":{sel}")) || bus(i).trim_start_matches('0') == sel.trim_start_matches('0'))
        });
        pci.or_else(|| {
            let hits: Vec<usize> = (0..infos.len()).filter(|&k| infos[k].name.to_lowercase().contains(&sel)).collect();
            (hits.len() == 1).then(|| hits[0])
        })
    };
    match found {
        Some(k) => Ok(Some(adapters[k].clone())),
        None => bail!("{GPU_ENV}={sel} matches no single GPU; the adapters are:\n{}", adapter_list(adapters)),
    }
}

/// One line per adapter: index, name, backend, type, PCI bus id.
pub fn adapter_list(adapters: &[wgpu::Adapter]) -> String {
    adapters
        .iter()
        .enumerate()
        .map(|(k, a)| {
            let i = a.get_info();
            format!(
                "  {k}: {} ({:?}, {:?}{})",
                i.name,
                i.backend,
                i.device_type,
                if i.device_pci_bus_id.is_empty() { String::new() } else { format!(", PCI {}", i.device_pci_bus_id) }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The headless adapter: the one `AERIALSYNTH_GPU` selects (see [`pick_adapter`]), else the
/// first high-performance Vulkan / Metal / DX12 adapter.
pub async fn select_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter> {
    if std::env::var(GPU_ENV).is_ok_and(|s| !s.trim().is_empty()) {
        let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY).await;
        if let Some(a) = pick_adapter(&adapters)? {
            return Ok(a);
        }
    }
    instance
        .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
        .await
        .map_err(|e| anyhow!("no GPU adapter: {e}"))
}

/// The shared libraries loaded in the process (their paths).
#[cfg(target_os = "linux")]
fn loaded_libraries() -> Vec<std::ffi::CString> {
    unsafe extern "C" fn each(info: *mut libc::dl_phdr_info, _size: libc::size_t, out: *mut libc::c_void) -> libc::c_int {
        // SAFETY: `dl_iterate_phdr` passes a valid entry, and `out` is the `Vec` below
        unsafe {
            let name = (*info).dlpi_name;
            if !name.is_null() && *name != 0 {
                (*(out as *mut Vec<std::ffi::CString>)).push(std::ffi::CStr::from_ptr(name).to_owned());
            }
        }
        0
    }
    let mut out: Vec<std::ffi::CString> = Vec::new();
    // SAFETY: `each` only reads the entry and pushes to `out`, which outlives the call
    unsafe { libc::dl_iterate_phdr(Some(each), &mut out as *mut _ as *mut libc::c_void) };
    out
}

#[cfg(not(target_os = "linux"))]
fn loaded_libraries() -> Vec<std::ffi::CString> {
    Vec::new()
}

/// Keep the libraries the GPU driver loaded (those not in `before`) mapped until the process
/// ends. The device lives as long as the process (it is never destroyed), and with it the
/// driver's worker threads; at exit, the NVIDIA driver's EGL library `dlclose`s the core
/// library those threads run in (an exit handler) while they still run: a segfault after all
/// the work is done. Pinned (`RTLD_NODELETE`), the code stays until the process is gone.
#[cfg(target_os = "linux")]
fn pin_driver(before: &[std::ffi::CString]) {
    for lib in loaded_libraries() {
        if !before.contains(&lib) {
            // SAFETY: `RTLD_NOLOAD` only takes another reference to a library already loaded
            // (no initializers run); the handle is never closed
            unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD | libc::RTLD_NODELETE) };
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_driver(_before: &[std::ffi::CString]) {}

impl Gpu {
    /// The adapter [`select_adapter`] picks (`AERIALSYNTH_GPU`, else the first high-performance
    /// Vulkan / Metal / DX12 adapter), headless, with the generator's features where it has them.
    pub fn new() -> Result<Gpu> {
        let loaded_before = loaded_libraries();
        pollster::block_on(async {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = wgpu::Backends::PRIMARY;
            let instance = wgpu::Instance::new(desc);
            let adapter = select_adapter(&instance).await?;
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
            pin_driver(&loaded_before);
            Ok(Gpu { device, queue, info, failed, waiter: std::sync::Mutex::new(()), done: Default::default(), compile: std::sync::Mutex::new(()) })
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
    /// fails: device lost, out of memory, the GPU stalled).
    ///
    /// Any number of threads may read back at once. One of them at a time (the holder of
    /// `waiter`) blocks in `poll`, in bounded slices, and that poll runs the completed mappings'
    /// callbacks of every thread; the others wait on their own callbacks and take over when it
    /// leaves. No lock is held across `poll` that a callback needs, and no thread waits for a
    /// callback without someone polling.
    pub fn map_read(&self, slices: &[wgpu::BufferSlice<'_>]) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let (tx, rx) = std::sync::mpsc::channel();
        for s in slices {
            let (tx, done) = (tx.clone(), self.done.clone());
            s.map_async(wgpu::MapMode::Read, move |r| {
                done.fetch_add(1, Relaxed);
                let _ = tx.send(r);
            });
        }
        drop(tx);
        let mut left = slices.len();
        let got = |r: Result<(), wgpu::BufferAsyncError>, left: &mut usize| -> Result<()> {
            r.map_err(|e| anyhow!("GPU read-back: {e}"))?;
            *left -= 1;
            Ok(())
        };
        let mut progress = (self.done.load(Relaxed), std::time::Instant::now());
        loop {
            // the callbacks that ran (in this thread's poll or in another's)
            while let Ok(r) = rx.try_recv() {
                got(r, &mut left)?;
            }
            if left == 0 {
                return Ok(());
            }
            self.check()?;
            let n = self.done.load(Relaxed);
            if n != progress.0 {
                progress = (n, std::time::Instant::now());
            } else if progress.1.elapsed() > STALL {
                let e = format!("the GPU stalled (no read-back completed in {} s)", STALL.as_secs());
                record(&self.failed, e.clone());
                bail!("{e}");
            }
            match self.waiter.try_lock() {
                Ok(_waiting) => self.wait_slice()?,
                Err(std::sync::TryLockError::Poisoned(p)) => {
                    let _waiting = p.into_inner();
                    self.wait_slice()?;
                }
                // another thread waits for the GPU; its poll runs these callbacks too
                Err(std::sync::TryLockError::WouldBlock) => match rx.recv_timeout(std::time::Duration::from_millis(1)) {
                    Ok(r) => got(r, &mut left)?,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => bail!("GPU read-back did not complete"),
                },
            }
        }
    }

    /// Wait for the GPU to finish the work submitted so far (all threads'), running the
    /// callbacks of what completed.
    pub fn wait_idle(&self) -> Result<()> {
        let _waiting = self.waiter.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            match self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(WAIT_SLICE) }) {
                Ok(_) => break,
                Err(wgpu::PollError::Timeout) => self.check()?,
                Err(e) => bail!("GPU poll: {e}"),
            }
        }
        self.check()
    }

    /// Hold the returned guard while creating pipelines: one thread at a time compiles on the
    /// device. Generators of several worlds set up at once (threads of the bindings, the tests)
    /// compiled the same pipelines concurrently, and every so often the NVIDIA driver then
    /// stopped executing the queue (a submission never completes, the GPU idles): every later
    /// read-back of every thread waited forever. A generator still compiles its own pipelines in
    /// parallel, under one guard.
    pub fn compiling(&self) -> std::sync::MutexGuard<'_, ()> {
        self.compile.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// One bounded blocking wait for the work submitted so far (by the holder of `waiter`).
    fn wait_slice(&self) -> Result<()> {
        match self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(WAIT_SLICE) }) {
            Ok(_) | Err(wgpu::PollError::Timeout) => Ok(()),
            Err(e) => bail!("GPU poll: {e}"),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    /// Copy `data` back from the GPU in `parts` read-back buffers, mapped together.
    fn round_trip(g: &Gpu, data: &[u32], parts: usize) -> Result<Vec<u32>> {
        let bytes = (data.len() * 4) as u64;
        let src = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stress source"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // `parts` ranges, aligned to the copy alignment
        let step = (bytes / parts as u64).div_ceil(8) * 8;
        let bounds: Vec<(u64, u64)> = (0..parts as u64).map(|k| (k * step, ((k + 1) * step).min(bytes))).filter(|(a, b)| a < b).collect();
        let mut enc = g.device.create_command_encoder(&Default::default());
        let rbs: Vec<wgpu::Buffer> = bounds
            .iter()
            .map(|&(a, b)| {
                let rb = g.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("stress read-back"),
                    size: b - a,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                enc.copy_buffer_to_buffer(&src, a, &rb, 0, b - a);
                rb
            })
            .collect();
        g.queue.submit([enc.finish()]);
        let slices: Vec<wgpu::BufferSlice<'_>> = rbs.iter().map(|rb| rb.slice(..)).collect();
        g.map_read(&slices)?;
        let mut out = Vec::with_capacity(data.len());
        for s in &slices {
            out.extend_from_slice(bytemuck::cast_slice(&s.get_mapped_range()?));
        }
        drop(slices);
        rbs.iter().for_each(|rb| rb.unmap());
        Ok(out)
    }

    /// Many threads reading back from the shared device at once (the generator, renderer and
    /// event sensor of several cameras / worlds do): every read-back completes, with its data.
    /// A deadlock fails the test after a bound instead of hanging it.
    #[test]
    fn concurrent_read_backs() {
        let Ok(g) = shared() else {
            eprintln!("no GPU: skipped");
            return;
        };
        const THREADS: usize = 12;
        const ROUNDS: usize = 150;
        let (tx, rx) = std::sync::mpsc::channel();
        for t in 0..THREADS {
            let (g, tx) = (g.clone(), tx.clone());
            std::thread::spawn(move || {
                let r = (|| -> Result<()> {
                    for k in 0..ROUNDS {
                        let n = 64 + (t * 7919 + k * 104729) % 50_000;
                        let data: Vec<u32> = (0..n as u32).map(|i| i ^ ((t as u32) << 24) ^ ((k as u32) << 12)).collect();
                        let back = round_trip(&g, &data, 1 + (t + k) % 3)?;
                        anyhow::ensure!(back == data, "thread {t} round {k}: wrong data read back");
                    }
                    Ok(())
                })();
                let _ = tx.send(r);
            });
        }
        drop(tx);
        // rayon workers reading back too (the renderer's and generator's parallel code)
        let pool = std::thread::spawn({
            let g = g.clone();
            move || {
                use rayon::prelude::*;
                (0..200u32).into_par_iter().try_for_each(|k| {
                    let data: Vec<u32> = (0..1000 + k * 13).map(|i| i.wrapping_mul(2654435761) ^ k).collect();
                    let back = round_trip(&g, &data, 1)?;
                    anyhow::ensure!(back == data, "rayon read-back {k}: wrong data");
                    Ok(())
                })
            }
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        for _ in 0..THREADS {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            match rx.recv_timeout(left) {
                Ok(r) => r.unwrap(),
                Err(_) => panic!("GPU read-backs deadlocked (threads still waiting after 240 s)"),
            }
        }
        while !pool.is_finished() {
            assert!(std::time::Instant::now() < deadline, "rayon GPU read-backs deadlocked");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        pool.join().unwrap().unwrap();
        g.check().unwrap();
    }
}
