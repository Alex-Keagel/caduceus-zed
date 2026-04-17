//! CRDT bridge — conflict-free replicated data types for collaborative editing.

use caduceus_crdt::{Anchor, Buffer, FragmentId, Operation, ReplicaId, VersionVector};

/// Wrapper around the CRDT Buffer for collaborative editing.
pub struct CrdtBridge {
    buffers: std::collections::HashMap<String, Buffer>,
}

impl CrdtBridge {
    pub fn new() -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
        }
    }

    /// Open or create a CRDT buffer for a file.
    pub fn open_buffer(&mut self, file_path: &str, initial_text: Option<&str>) -> BufferInfo {
        // Validate path — reject traversal attempts entirely
        if file_path.contains("..") || std::path::Path::new(file_path).is_absolute() {
            log::warn!("[caduceus] CRDT open_buffer: rejected unsafe path '{}'", file_path);
            return BufferInfo {
                file_path: file_path.to_string(),
                length: 0,
                fragment_count: 0,
            };
        }
        if !self.buffers.contains_key(file_path) {
            let buffer = match initial_text {
                Some(text) => Buffer::with_text(text),
                None => match std::fs::read_to_string(file_path) {
                    Ok(content) => Buffer::with_text(content),
                    Err(_) => Buffer::new(),
                },
            };
            self.buffers.insert(file_path.to_string(), buffer);
        }
        let buf = self.buffers.get(file_path).unwrap();
        BufferInfo {
            file_path: file_path.to_string(),
            length: buf.len(),
            fragment_count: buf.fragments().len(),
        }
    }

    /// Get the current text of a buffer.
    pub fn get_text(&self, file_path: &str) -> Option<String> {
        self.buffers.get(file_path).map(|b| b.text())
    }

    /// Insert text at a position.
    pub fn insert(
        &mut self,
        file_path: &str,
        offset: usize,
        text: &str,
        replica_id: u16,
    ) -> Result<Operation, String> {
        let buf = self
            .buffers
            .get_mut(file_path)
            .ok_or_else(|| format!("No buffer for {file_path}"))?;
        Ok(buf.insert(offset, text, ReplicaId(replica_id)))
    }

    /// Delete text at a position.
    pub fn delete(
        &mut self,
        file_path: &str,
        offset: usize,
        length: usize,
        replica_id: u16,
    ) -> Result<Operation, String> {
        let buf = self
            .buffers
            .get_mut(file_path)
            .ok_or_else(|| format!("No buffer for {file_path}"))?;
        Ok(buf.delete(offset, length, ReplicaId(replica_id)))
    }

    /// Apply a remote operation.
    pub fn apply_remote(
        &mut self,
        file_path: &str,
        op: Operation,
    ) -> Result<(), String> {
        let buf = self
            .buffers
            .get_mut(file_path)
            .ok_or_else(|| format!("No buffer for {file_path}"))?;
        buf.apply_remote(op);
        Ok(())
    }

    /// Get buffer length.
    pub fn buffer_len(&self, file_path: &str) -> Option<usize> {
        self.buffers.get(file_path).map(|b| b.len())
    }

    /// List all open buffers.
    pub fn list_buffers(&self) -> Vec<BufferInfo> {
        self.buffers
            .iter()
            .map(|(path, buf)| BufferInfo {
                file_path: path.clone(),
                length: buf.len(),
                fragment_count: buf.fragments().len(),
            })
            .collect()
    }

    /// Close a buffer.
    pub fn close_buffer(&mut self, file_path: &str) -> bool {
        self.buffers.remove(file_path).is_some()
    }

    // ── New CRDT methods ─────────────────────────────────────────────────

    /// Undo the last operation by its FragmentId.
    pub fn undo(
        &mut self,
        file_path: &str,
        op_id: FragmentId,
        replica_id: u16,
    ) -> Result<Operation, String> {
        let buf = self
            .buffers
            .get_mut(file_path)
            .ok_or_else(|| format!("No buffer for {file_path}"))?;
        Ok(buf.undo(op_id, ReplicaId(replica_id)))
    }

    /// Get an anchor at a given character offset.
    pub fn anchor_at(&self, file_path: &str, offset: usize) -> Result<Anchor, String> {
        let buf = self
            .buffers
            .get(file_path)
            .ok_or_else(|| format!("No buffer for {file_path}"))?;
        Ok(buf.anchor_at(offset))
    }

    /// Get the character offset for an anchor.
    pub fn offset_of(&self, file_path: &str, anchor: &Anchor) -> Result<usize, String> {
        let buf = self
            .buffers
            .get(file_path)
            .ok_or_else(|| format!("No buffer for {file_path}"))?;
        Ok(buf.offset_of(anchor))
    }

    /// Get the version vector for a buffer.
    pub fn version(&self, file_path: &str) -> Result<VersionVector, String> {
        let buf = self
            .buffers
            .get(file_path)
            .ok_or_else(|| format!("No buffer for {file_path}"))?;
        Ok(buf.version().clone())
    }
}

impl Default for CrdtBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub file_path: String,
    pub length: usize,
    pub fragment_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crdt_open_empty_buffer() {
        let mut crdt = CrdtBridge::new();
        let info = crdt.open_buffer("test.txt", None);
        assert_eq!(info.length, 0);
    }

    #[test]
    fn crdt_open_with_text() {
        let mut crdt = CrdtBridge::new();
        let info = crdt.open_buffer("test.txt", Some("hello world"));
        assert_eq!(info.length, 11);
    }

    #[test]
    fn crdt_insert_and_get() {
        let mut crdt = CrdtBridge::new();
        crdt.open_buffer("test.txt", Some("hello"));
        crdt.insert("test.txt", 5, " world", 1).unwrap();
        assert_eq!(crdt.get_text("test.txt"), Some("hello world".to_string()));
    }

    #[test]
    fn crdt_delete() {
        let mut crdt = CrdtBridge::new();
        crdt.open_buffer("test.txt", Some("hello world"));
        crdt.delete("test.txt", 5, 6, 1).unwrap();
        assert_eq!(crdt.get_text("test.txt"), Some("hello".to_string()));
    }

    #[test]
    fn crdt_apply_remote() {
        let mut crdt1 = CrdtBridge::new();
        let mut crdt2 = CrdtBridge::new();
        crdt1.open_buffer("f.txt", Some("hello"));
        crdt2.open_buffer("f.txt", Some("hello"));

        let op = crdt1.insert("f.txt", 5, " world", 1).unwrap();
        crdt2.apply_remote("f.txt", op).unwrap();

        assert_eq!(crdt1.get_text("f.txt"), crdt2.get_text("f.txt"));
    }

    #[test]
    fn crdt_list_buffers() {
        let mut crdt = CrdtBridge::new();
        crdt.open_buffer("a.txt", Some("aaa"));
        crdt.open_buffer("b.txt", Some("bbb"));
        assert_eq!(crdt.list_buffers().len(), 2);
    }

    #[test]
    fn crdt_close_buffer() {
        let mut crdt = CrdtBridge::new();
        crdt.open_buffer("x.txt", Some("data"));
        assert!(crdt.close_buffer("x.txt"));
        assert!(crdt.get_text("x.txt").is_none());
    }

    #[test]
    fn crdt_insert_missing_buffer_errors() {
        let mut crdt = CrdtBridge::new();
        let result = crdt.insert("nonexistent.txt", 0, "text", 1);
        assert!(result.is_err());
    }

    #[test]
    fn crdt_anchor_roundtrip() {
        let mut crdt = CrdtBridge::new();
        crdt.open_buffer("test.txt", Some("hello world"));
        let anchor = crdt.anchor_at("test.txt", 5).unwrap();
        let offset = crdt.offset_of("test.txt", &anchor).unwrap();
        assert_eq!(offset, 5);
    }

    #[test]
    fn crdt_anchor_start() {
        let mut crdt = CrdtBridge::new();
        crdt.open_buffer("test.txt", Some("abc"));
        let anchor = crdt.anchor_at("test.txt", 0).unwrap();
        let offset = crdt.offset_of("test.txt", &anchor).unwrap();
        assert_eq!(offset, 0);
    }

    #[test]
    fn crdt_version_vector() {
        let mut crdt = CrdtBridge::new();
        crdt.open_buffer("test.txt", Some("hello"));
        let v1 = crdt.version("test.txt").unwrap();
        crdt.insert("test.txt", 5, " world", 1).unwrap();
        let v2 = crdt.version("test.txt").unwrap();
        // After an insert, the version vector should have observed the new operation
        assert_ne!(format!("{:?}", v1), format!("{:?}", v2));
    }

    #[test]
    fn crdt_undo_insert() {
        let mut crdt = CrdtBridge::new();
        crdt.open_buffer("test.txt", Some("hello"));
        let op = crdt.insert("test.txt", 5, " world", 1).unwrap();
        assert_eq!(crdt.get_text("test.txt"), Some("hello world".to_string()));
        let op_id = op.id().clone();
        crdt.undo("test.txt", op_id, 1).unwrap();
        assert_eq!(crdt.get_text("test.txt"), Some("hello".to_string()));
    }

    #[test]
    fn crdt_version_missing_buffer() {
        let crdt = CrdtBridge::new();
        assert!(crdt.version("no.txt").is_err());
    }
}
