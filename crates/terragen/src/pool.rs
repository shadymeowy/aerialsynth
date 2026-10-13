//! Thread pool for the host work of the GPU generator (see [`host_pool`]).

/// The generator's thread pool for parallel host work during a GPU batch (the kits' features,
/// pass-A stamps, ecoregions), apart from rayon's global pool: a caller may generate tiles from the global
/// pool's threads (`par_iter` over tiles), which then wait for the batch in progress — host work
/// queued there could wait for them in turn (a deadlock: the batch holds the generator's cache).
/// Host code run here never calls the generator. Enter it from a plain thread ([`host_run`]):
/// a rayon worker waiting for another pool runs its own pool's jobs meanwhile — another tile of
/// the caller, which would wait for the cache the worker holds.
pub(crate) fn host_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads().max(2))
            .thread_name(|i| format!("terragen-host-{i}"))
            .build()
            .expect("terragen: host thread pool")
    })
}

/// Run `f` on the host pool such that the calling thread just blocks (it may hold the
/// generator's cache, see [`host_pool`]): directly on a host pool thread, by `install` from a
/// thread of no pool, else from a plain thread.
pub(crate) fn host_run<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    let pool = host_pool();
    if pool.current_thread_index().is_some() {
        return f();
    }
    if rayon::current_thread_index().is_none() {
        return pool.install(f);
    }
    std::thread::scope(|s| s.spawn(|| pool.install(f)).join()).unwrap_or_else(|e| std::panic::resume_unwind(e))
}

