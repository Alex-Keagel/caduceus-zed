//! Memory persistence — stores key-value pairs in .caduceus/memory.md

use std::path::Path;

pub fn store(project_root: &Path, key: &str, value: &str) -> Result<(), String> {
    if key.contains('\n') || key.contains('\r') || key.is_empty() {
        return Err("Invalid memory key: must not contain newlines or be empty".to_string());
    }
    let dir = project_root.join(".caduceus");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("memory.md");
    let mut content = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| "# Caduceus Memory\n".to_string());

    // Check if key already exists — update it
    let key_prefix = format!("{}: ", key);
    let lines: Vec<&str> = content.lines().collect();
    let mut found = false;
    let new_content: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.starts_with(&key_prefix) {
                found = true;
                format!("{}: {}", key, value)
            } else {
                line.to_string()
            }
        })
        .collect();

    if found {
        content = new_content.join("\n");
    } else {
        content.push_str(&format!("\n{}: {}", key, value));
    }

    std::fs::write(&path, content).map_err(|e| e.to_string())
}

pub fn get(project_root: &Path, key: &str) -> Option<String> {
    let path = project_root.join(".caduceus/memory.md");
    let content = std::fs::read_to_string(&path).ok()?;
    let prefix = format!("{}: ", key);
    content
        .lines()
        .find(|l| l.starts_with(&prefix))
        .map(|l| l[prefix.len()..].to_string())
}

pub fn list(project_root: &Path) -> Vec<(String, String)> {
    let path = project_root.join(".caduceus/memory.md");
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    content
        .lines()
        .filter(|l| l.contains(": ") && !l.starts_with('#'))
        .filter_map(|l| {
            let (k, v) = l.split_once(": ")?;
            Some((k.to_string(), v.to_string()))
        })
        .collect()
}

pub fn delete(project_root: &Path, key: &str) -> Result<(), String> {
    let path = project_root.join(".caduceus/memory.md");
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let prefix = format!("{}: ", key);
    let new_content: Vec<&str> = content
        .lines()
        .filter(|l| !l.starts_with(&prefix))
        .collect();
    std::fs::write(&path, new_content.join("\n")).map_err(|e| e.to_string())
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
}
