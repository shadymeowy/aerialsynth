//! Progress of the generator's shader compilation (first run on a GPU, driver or version takes
//! about half a minute): an optional observer, and a status that front ends can poll.
//! The library itself is silent; [`stderr_observer`] is the message of the CLI and bindings.

use std::sync::{Mutex, RwLock};
use std::time::Instant;

/// What the generator is doing while it sets up its compute pipelines.
#[derive(Clone, Debug, PartialEq)]
pub enum CompileEvent {
    /// Compiling starts. `cold`: the pipeline cache has no entry for this GPU, driver and
    /// shader version (or the backend has no pipeline cache), so it will probably take long.
    Started { gpu_name: String, cold: bool },
    /// The pipelines are ready after `secs` (`cold` as in `Started`).
    Finished { secs: f64, cold: bool },
}

pub type Observer = Box<dyn Fn(&CompileEvent) + Send + Sync>;

static OBSERVER: RwLock<Option<Observer>> = RwLock::new(None);
static STATUS: Mutex<Option<(Instant, bool)>> = Mutex::new(None);

/// Install the observer of compile events (replaces the previous one; `None`: silent, the default).
/// It is called on the thread that creates the generator.
pub fn set_compile_observer(f: Option<Observer>) {
    *OBSERVER.write().unwrap_or_else(|p| p.into_inner()) = f;
}

/// The running compile, if any: when it started and whether it is expected to be cold.
pub fn compiling_since() -> Option<(Instant, bool)> {
    *STATUS.lock().unwrap_or_else(|p| p.into_inner())
}

pub(crate) fn emit(e: CompileEvent) {
    *STATUS.lock().unwrap_or_else(|p| p.into_inner()) = match &e {
        CompileEvent::Started { cold, .. } => Some((Instant::now(), *cold)),
        CompileEvent::Finished { .. } => None,
    };
    if let Some(f) = OBSERVER.read().unwrap_or_else(|p| p.into_inner()).as_ref() {
        f(&e);
    }
}

/// The messages of the CLI and the bindings: on a cold compile a notice and the time it took; on
/// a warm cache nothing, unless it unexpectedly takes more than 5 s.
pub fn stderr_observer() -> Observer {
    Box::new(|e| {
        if let Some(s) = message(e) {
            eprintln!("{s}");
        }
    })
}

pub(crate) fn message(e: &CompileEvent) -> Option<String> {
    match e {
        CompileEvent::Started { gpu_name, cold: true } => {
            Some(format!("compiling GPU shaders for {gpu_name} (first run for this GPU, driver or version; this can take a minute, cached afterwards)…"))
        }
        CompileEvent::Finished { secs, cold } if *cold || *secs > 5.0 => Some(format!("GPU shaders ready in {secs:.1} s")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_silent_and_messages_follow_cold() {
        assert!(OBSERVER.read().unwrap().is_none());
        assert_eq!(message(&CompileEvent::Started { gpu_name: "X".into(), cold: false }), None);
        assert_eq!(message(&CompileEvent::Finished { secs: 1.4, cold: false }), None);
        assert!(message(&CompileEvent::Finished { secs: 6.0, cold: false }).unwrap().contains("6.0 s"));
        assert!(message(&CompileEvent::Started { gpu_name: "X".into(), cold: true }).unwrap().contains("compiling GPU shaders for X"));
        assert!(message(&CompileEvent::Finished { secs: 27.44, cold: true }).unwrap().contains("27.4 s"));
    }
}
