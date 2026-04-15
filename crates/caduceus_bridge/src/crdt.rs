//! CRDT bridge — conflict-free replicated data types for collaborative editing.

use caduceus_crdt::{Buffer, Operation, ReplicaId};

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
}
