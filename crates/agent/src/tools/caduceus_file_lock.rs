//! In-process advisory file locking for `.caduceus/` shared files.
//!
//! Serialises cross-tool access to the same path (e.g. `kanban.json`
//! accessed by both the kanban tool and a sub-agent concurrently).

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

static LOCKED_FILES: LazyLock<Mutex<HashSet<PathBuf>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// RAII guard — automatically releases the lock when dropped.
pub struct FileLockGuard {
    path: PathBuf,
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        if let Ok(mut locks) = LOCKED_FILES.lock() {
            locks.remove(&self.path);
        }
    }
}

/// Acquire an advisory in-process lock on `path`.
///
/// Returns `Err` if the path is already locked by another call site.
/// The lock is released when the returned [`FileLockGuard`] is dropped.
pub fn acquire_file_lock(path: &std::path::Path) -> Result<FileLockGuard, String> {
    let path = path.to_path_buf();
    let mut locks = LOCKED_FILES
        .lock()
        .map_err(|e| format!("Lock poisoned: {e}"))?;
    if locks.contains(&path) {
        return Err(format!("File is locked: {}", path.display()));
    }
    locks.insert(path.clone());
    Ok(FileLockGuard { path })
}

/// List currently locked file paths.
pub fn list_locked_files() -> Vec<PathBuf> {
    LOCKED_FILES
        .lock()
        .map(|locks| locks.iter().cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_and_release() {
        let p = PathBuf::from("/fake/.caduceus/test.json");
        let guard = acquire_file_lock(&p).unwrap();
        assert!(acquire_file_lock(&p).is_err());
        drop(guard);
        assert!(acquire_file_lock(&p).is_ok());
    }

    #[test]
    fn different_paths_independent() {
        let a = PathBuf::from("/fake/.caduceus/a.json");
        let b = PathBuf::from("/fake/.caduceus/b.json");
        let _ga = acquire_file_lock(&a).unwrap();
        let _gb = acquire_file_lock(&b).unwrap();
    }
}
