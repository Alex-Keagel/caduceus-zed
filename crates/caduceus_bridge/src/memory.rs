//! Memory persistence — stores key-value pairs in .caduceus/memory.json
//!
//! Security: values are sanitized (newlines/control chars stripped, size capped)
//! to prevent prompt injection when memory content is injected into system prompts.

use std::collections::BTreeMap;
use std::path::Path;

/// Maximum length of a single memory value (prevents unbounded growth)
const MAX_VALUE_LEN: usize = 4096;
/// Maximum number of memory entries per project
const MAX_ENTRIES: usize = 100;

// Well-known system memory keys (use with store_system)
pub const KEY_PROJECT_OVERVIEW: &str = "wiki:project-overview";
pub const KEY_README: &str = "wiki:readme";
pub const KEY_GIT_BRANCH: &str = "wiki:git-branch";
pub const KEY_RECENT_COMMITS: &str = "wiki:recent-commits";

/// Sanitize a value: strip control characters, cap length
fn sanitize_value(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_control() || *c == ' ')
        .take(MAX_VALUE_LEN)
        .collect()
}

/// Validate a key: no control chars, not empty, reasonable length
fn validate_key(key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err("Memory key must not be empty".to_string());
    }
    if key.len() > 128 {
        return Err("Memory key must be ≤128 characters".to_string());
    }
    if key.chars().any(|c| c.is_control()) {
        return Err("Memory key must not contain control characters".to_string());
    }
    Ok(())
}

/// Reserved key prefixes that only the engine can write (not tool-accessible)
const RESERVED_PREFIXES: &[&str] = &["wiki:", "system:", "caduceus:", "auto:"];

fn read_store(project_root: &Path) -> BTreeMap<String, String> {
    let json_path = project_root.join(".caduceus/memory.json");
    let md_path = project_root.join(".caduceus/memory.md");

    // Try JSON first
    if let Ok(json) = std::fs::read_to_string(&json_path) {
        if let Ok(map) = serde_json::from_str(&json) {
            return map;
        }
    }

    // Legacy migration: parse .md format if .json doesn't exist
    if let Ok(content) = std::fs::read_to_string(&md_path) {
        let map: BTreeMap<String, String> = content
            .lines()
            .filter(|l| l.contains(": ") && !l.starts_with('#'))
            .filter_map(|l| {
                let (k, v) = l.split_once(": ")?;
                Some((k.to_string(), sanitize_value(v)))
            })
            .collect();
        if !map.is_empty() {
            // Write migrated JSON and remove legacy file
            let _ = write_store(project_root, &map);
            let _ = std::fs::remove_file(&md_path);
            log::info!("[caduceus] Migrated {} memory entries from memory.md → memory.json", map.len());
            return map;
        }
    }

    BTreeMap::new()
}

fn write_store(project_root: &Path, store: &BTreeMap<String, String>) -> Result<(), String> {
    let dir = project_root.join(".caduceus");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("memory.json");
    let json = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())
}

pub fn store(project_root: &Path, key: &str, value: &str) -> Result<(), String> {
    validate_key(key)?;
    // Block reserved key prefixes from tool/user writes
    if RESERVED_PREFIXES.iter().any(|p| key.starts_with(p)) {
        return Err(format!("Key '{}' uses a reserved prefix ({}). Use store_system() for engine-managed keys.",
            key, RESERVED_PREFIXES.join(", ")));
    }
    let sanitized = sanitize_value(value);
    let mut map = read_store(project_root);
    if map.len() >= MAX_ENTRIES && !map.contains_key(key) {
        return Err(format!("Memory limit reached ({} entries). Delete old entries first.", MAX_ENTRIES));
    }
    map.insert(key.to_string(), sanitized);
    write_store(project_root, &map)
}

/// Store a value under a system-reserved key (engine-only, no tool access).
pub fn store_system(project_root: &Path, key: &str, value: &str) -> Result<(), String> {
    validate_key(key)?;
    let sanitized = sanitize_value(value);
    let mut map = read_store(project_root);
    if map.len() >= MAX_ENTRIES && !map.contains_key(key) {
        return Err(format!("Memory limit reached ({} entries).", MAX_ENTRIES));
    }
    map.insert(key.to_string(), sanitized);
    write_store(project_root, &map)
}

pub fn get(project_root: &Path, key: &str) -> Option<String> {
    read_store(project_root).get(key).cloned()
}

pub fn list(project_root: &Path) -> Vec<(String, String)> {
    read_store(project_root).into_iter().collect()
}

pub fn delete(project_root: &Path, key: &str) -> Result<(), String> {
    // Block deletion of reserved keys from tool/user writes
    if RESERVED_PREFIXES.iter().any(|p| key.starts_with(p)) {
        return Err(format!("Cannot delete reserved key '{}' — managed by engine", key));
    }
    let mut map = read_store(project_root);
    if map.remove(key).is_none() {
        return Err(format!("Key '{}' not found", key));
    }
    write_store(project_root, &map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_and_get() {
        let dir = tempfile::tempdir().unwrap();
        store(dir.path(), "greeting", "hello").unwrap();
        assert_eq!(get(dir.path(), "greeting"), Some("hello".to_string()));
    }

    #[test]
    fn memory_update_existing() {
        let dir = tempfile::tempdir().unwrap();
        store(dir.path(), "key", "v1").unwrap();
        store(dir.path(), "key", "v2").unwrap();
        assert_eq!(get(dir.path(), "key"), Some("v2".to_string()));
    }

    #[test]
    fn memory_list_all() {
        let dir = tempfile::tempdir().unwrap();
        store(dir.path(), "a", "1").unwrap();
        store(dir.path(), "b", "2").unwrap();
        let entries = list(dir.path());
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn memory_get_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(get(dir.path(), "missing"), None);
    }

    #[test]
    fn memory_delete() {
        let dir = tempfile::tempdir().unwrap();
        store(dir.path(), "del_me", "value").unwrap();
        delete(dir.path(), "del_me").unwrap();
        assert_eq!(get(dir.path(), "del_me"), None);
    }

    #[test]
    fn memory_sanitizes_newlines_in_values() {
        let dir = tempfile::tempdir().unwrap();
        store(dir.path(), "key", "line1\nfake_key: injected\nline3").unwrap();
        let val = get(dir.path(), "key").unwrap();
        assert!(!val.contains('\n'), "Newlines must be stripped from values");
        // Without newlines, the value is a single string — no key injection possible
        // Verify only one entry exists (not 3 forged keys)
        let entries = list(dir.path());
        assert_eq!(entries.len(), 1, "Only 1 entry should exist, not 3 forged ones");
        assert_eq!(entries[0].0, "key");
    }

    #[test]
    fn memory_rejects_empty_key() {
        let dir = tempfile::tempdir().unwrap();
        assert!(store(dir.path(), "", "value").is_err());
    }

    #[test]
    fn memory_rejects_control_chars_in_key() {
        let dir = tempfile::tempdir().unwrap();
        assert!(store(dir.path(), "key\x00evil", "value").is_err());
    }

    #[test]
    fn memory_caps_value_length() {
        let dir = tempfile::tempdir().unwrap();
        let long_value = "x".repeat(10000);
        store(dir.path(), "big", &long_value).unwrap();
        let stored = get(dir.path(), "big").unwrap();
        assert!(stored.len() <= MAX_VALUE_LEN, "Value should be capped at {}", MAX_VALUE_LEN);
    }

    #[test]
    fn memory_enforces_entry_limit() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..MAX_ENTRIES {
            store(dir.path(), &format!("k{}", i), "v").unwrap();
        }
        let result = store(dir.path(), "overflow", "v");
        assert!(result.is_err(), "Should reject entries beyond limit");
        assert!(result.unwrap_err().contains("limit"));
    }

    #[test]
    fn memory_blocks_reserved_key_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        assert!(store(dir.path(), "wiki:evil", "injection").is_err());
        assert!(store(dir.path(), "system:config", "hack").is_err());
        assert!(store(dir.path(), "caduceus:internal", "bad").is_err());
        // But store_system can write reserved keys
        assert!(store_system(dir.path(), "wiki:overview", "legit").is_ok());
        assert_eq!(get(dir.path(), "wiki:overview"), Some("legit".to_string()));
    }

    #[test]
    fn memory_migrates_legacy_md() {
        let dir = tempfile::tempdir().unwrap();
        let caduceus_dir = dir.path().join(".caduceus");
        std::fs::create_dir_all(&caduceus_dir).unwrap();
        // Write legacy .md format
        std::fs::write(
            caduceus_dir.join("memory.md"),
            "# Caduceus Memory\ngreeting: hello\nproject: test",
        ).unwrap();
        // Read should auto-migrate
        let entries = list(dir.path());
        assert_eq!(entries.len(), 2);
        // .json should now exist
        assert!(caduceus_dir.join("memory.json").exists());
        // .md should be removed
        assert!(!caduceus_dir.join("memory.md").exists());
    }
}
