//! The directory of the generators' disk caches (pipeline cache, atlas cache).

use std::ffi::OsString;
use std::path::PathBuf;

/// `$XDG_CACHE_HOME/terrain`, else `$HOME/.cache/terrain`, else `%LOCALAPPDATA%\terrain`
/// (Windows, where `HOME` is usually unset). Empty variables are unset (an empty
/// `XDG_CACHE_HOME` put the cache in the working directory).
pub(crate) fn cache_dir() -> Option<PathBuf> {
    cache_dir_in(|k| std::env::var_os(k))
}

/// [`cache_dir`] with the environment `var`.
pub(crate) fn cache_dir_in(var: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    let set = |k: &str| var(k).filter(|v| !v.is_empty());
    set("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| set("HOME").map(|h| std::path::Path::new(&h).join(".cache")))
        .or_else(|| set("LOCALAPPDATA").map(PathBuf::from))
        .map(|d| d.join("terrain"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_directory() {
        let env = |vars: &'static [(&'static str, &'static str)]| move |k: &str| vars.iter().find(|v| v.0 == k).map(|v| OsString::from(v.1));
        assert_eq!(cache_dir_in(env(&[("XDG_CACHE_HOME", "/x"), ("HOME", "/h")])), Some(PathBuf::from("/x/terrain")));
        assert_eq!(cache_dir_in(env(&[("XDG_CACHE_HOME", ""), ("HOME", "/h")])), Some(PathBuf::from("/h/.cache/terrain")));
        // Windows: no HOME
        assert_eq!(cache_dir_in(env(&[("LOCALAPPDATA", "C:/Users/u/AppData/Local")])), Some(PathBuf::from("C:/Users/u/AppData/Local/terrain")));
        assert_eq!(cache_dir_in(env(&[("HOME", "/h"), ("LOCALAPPDATA", "C:/l")])), Some(PathBuf::from("/h/.cache/terrain")));
        assert_eq!(cache_dir_in(env(&[])), None);
    }
}
