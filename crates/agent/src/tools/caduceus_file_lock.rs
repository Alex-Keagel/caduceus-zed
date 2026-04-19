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
#[derive(Debug)]
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
/// The path is canonicalized before locking so that `./a.json`, `a.json`, and
/// any symlink that resolves to the same target all map to a single lock —
/// preventing concurrent writers from believing they hold independent locks
/// on what is in fact the same file. Falls back to the raw path if the file
/// does not yet exist (canonicalize requires the target to exist).
///
/// Returns `Err` if the path is already locked by another call site.
/// The lock is released when the returned [`FileLockGuard`] is dropped.
pub fn acquire_file_lock(path: &std::path::Path) -> Result<FileLockGuard, String> {
    let key = path
        .canonicalize()
        .or_else(|_| {
            // Fall back to canonicalizing the parent + appending the basename
            // so that a non-existent file still gets a stable key.
            if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
                parent.canonicalize().map(|p| p.join(name))
            } else {
                Ok(path.to_path_buf())
            }
        })
        .unwrap_or_else(|_| path.to_path_buf());
    let mut locks = LOCKED_FILES
        .lock()
        .map_err(|e| format!("Lock poisoned: {e}"))?;
    if locks.contains(&key) {
        return Err(format!("File is locked: {}", key.display()));
    }
    locks.insert(key.clone());
    Ok(FileLockGuard { path: key })
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

    /// Regression for bug #18: locks were keyed by the raw `PathBuf`, so the
    /// same file reachable via a symlink and via its canonical path would
    /// hand out two independent guards — defeating the purpose of the lock.
    /// The fix canonicalizes the key before insertion.
    #[test]
    fn symlink_and_canonical_share_one_lock() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("real.json");
        std::fs::write(&real, b"x").unwrap();
        let link = tmp.path().join("link.json");
        symlink(&real, &link).unwrap();

        let _g_real = acquire_file_lock(&real).expect("first lock should succeed");
        // Acquiring via the symlink must collide with the canonicalized
        // lock on the real path.
        let err = acquire_file_lock(&link);
        assert!(
            err.is_err(),
            "symlink path must canonicalize to same key as real path; got: {err:?}"
        );
    }

    /// Regression for bug #18: relative paths must canonicalize to the same
    /// key as their absolute equivalent so two callers using different forms
    /// of the same path do not both believe they hold the lock.
    #[test]
    fn relative_and_absolute_share_one_lock() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let abs = tmp.path().join("data.json");
        std::fs::write(&abs, b"x").unwrap();

        // Construct a path with a `.` segment that canonicalize() will collapse.
        let dotted = tmp.path().join(".").join("data.json");

        let _g = acquire_file_lock(&abs).expect("first lock");
        assert!(
            acquire_file_lock(&dotted).is_err(),
            "dotted path should canonicalize to the same key as abs"
        );
    }

    /// Regression for bug #18: nonexistent paths must still produce a stable
    /// key (we fall back to canonicalizing the parent and appending the
    /// basename) so locks for files that haven't been created yet are still
    /// honored on subsequent calls.
    #[test]
    fn nonexistent_path_still_locks() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let p = tmp.path().join("does-not-exist-yet.json");
        let _g = acquire_file_lock(&p).expect("lock for nonexistent file");
        assert!(
            acquire_file_lock(&p).is_err(),
            "second acquire on same nonexistent path must collide"
        );
    }
}
