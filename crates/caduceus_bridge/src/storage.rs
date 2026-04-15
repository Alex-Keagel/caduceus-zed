//! Storage bridge — session persistence, audit logs, cost tracking, memory CRUD.

use caduceus_core::{AuditEntry, SessionId};
use caduceus_storage::{MemoryRecord, SqliteStorage, StoredToolCall};
use std::path::Path;

/// Wrapper around SqliteStorage for the bridge.
pub struct StorageBridge {
    pub storage: SqliteStorage,
}

impl StorageBridge {
    pub fn open(db_path: &Path) -> Result<Self, String> {
        SqliteStorage::open(db_path)
            .map(|storage| Self { storage })
            .map_err(|e| e.to_string())
    }

    pub fn open_default() -> Result<Self, String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let db_path = format!("{home}/.caduceus/db.sqlite");
        let _ = std::fs::create_dir_all(format!("{home}/.caduceus"));
        Self::open(Path::new(&db_path))
    }

    /// Get raw storage reference for direct access.
    pub fn raw(&self) -> &SqliteStorage {
        &self.storage
    }

    pub async fn save_message(
        &self,
        session_id: &SessionId,
        role: &str,
        content: &str,
    ) -> Result<i64, String> {
        let msg = if role == "user" {
            caduceus_core::LlmMessage::user(content)
        } else {
            caduceus_core::LlmMessage::assistant(content)
        };
        self.storage
            .save_message(session_id, &msg, None)
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn list_messages(&self, session_id: &SessionId) -> Result<Vec<StoredMessage>, String> {
        self.storage
            .list_messages(session_id)
            .await
            .map(|msgs| {
                msgs.iter()
                    .map(|m| StoredMessage {
                        role: format!("{:?}", m.message.role),
                        content: m
                            .message
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                caduceus_core::ContentBlock::Text(t) => Some(t.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                        tokens: m.tokens,
                    })
                    .collect()
            })
            .map_err(|e| e.to_string())
    }

    pub async fn record_cost(
        &self,
        session_id: &SessionId,
        provider: &str,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
        cost_usd: f64,
    ) -> Result<i64, String> {
        self.storage
            .record_cost(
                session_id,
                &caduceus_core::ProviderId::new(provider),
                &caduceus_core::ModelId::new(model),
                input_tokens,
                output_tokens,
                cost_usd,
            )
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn total_cost(&self, session_id: &SessionId) -> Result<f64, String> {
        self.storage.total_cost(session_id).await.map_err(|e| e.to_string())
    }

    pub async fn export_transcript(&self, session_id: &SessionId, output_path: &Path) -> Result<(), String> {
        self.storage
            .export_transcript(session_id, output_path)
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn fork_session(&self, session_id: &SessionId) -> Result<SessionId, String> {
        self.storage
            .fork_session(session_id)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Tool calls ───────────────────────────────────────────────────────

    /// List all tool calls for a session.
    pub async fn list_tool_calls(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<StoredToolCall>, String> {
        self.storage
            .list_tool_calls(session_id)
            .await
            .map_err(|e| e.to_string())
    }

    /// Record a tool call.
    pub async fn record_tool_call(&self, call: &StoredToolCall) -> Result<(), String> {
        self.storage
            .record_tool_call(call)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Audit log ────────────────────────────────────────────────────────

    /// List audit log entries for a session.
    pub async fn list_audit(&self, session_id: &SessionId) -> Result<Vec<AuditEntry>, String> {
        self.storage
            .list_audit(session_id)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Session resume / recovery ────────────────────────────────────────

    /// Resume a session with full state + messages.
    pub async fn resume_session(
        &self,
        session_id: &SessionId,
    ) -> Result<caduceus_storage::ResumedSession, String> {
        self.storage
            .resume_session(session_id)
            .await
            .map_err(|e| e.to_string())
    }

    /// Find sessions that were left in 'running' phase (crashed) and mark them.
    pub fn recover_crashed_sessions(&self) -> Result<Vec<SessionId>, String> {
        self.storage
            .recover_crashed_sessions()
            .map_err(|e| e.to_string())
    }

    // ── Telemetry snapshots ──────────────────────────────────────────────

    /// Persist a telemetry snapshot (context health, attention, degradation stage, etc.)
    pub async fn save_telemetry_snapshot(
        &self,
        session_id: &SessionId,
        snapshot_type: &str,
        data: &serde_json::Value,
    ) -> Result<(), String> {
        self.storage
            .save_telemetry_snapshot(session_id, snapshot_type, data)
            .await
            .map_err(|e| e.to_string())
    }

    /// Load telemetry snapshots for a session.
    pub async fn load_telemetry_snapshots(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<(String, serde_json::Value, String)>, String> {
        self.storage
            .load_telemetry_snapshots(session_id, None)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Memory CRUD ──────────────────────────────────────────────────────

    /// Set (upsert) a memory key-value in a scope.
    pub async fn set_memory(
        &self,
        scope: &str,
        key: &str,
        value: &str,
    ) -> Result<(), String> {
        self.storage
            .set_memory(scope, key, value, "bridge")
            .await
            .map_err(|e| e.to_string())
    }

    /// Get a single memory record.
    pub async fn get_memory(
        &self,
        scope: &str,
        key: &str,
    ) -> Result<Option<MemoryRecord>, String> {
        self.storage
            .get_memory(scope, key)
            .await
            .map_err(|e| e.to_string())
    }

    /// List all memories in a scope (None = all scopes).
    pub async fn list_memories(
        &self,
        scope: &str,
    ) -> Result<Vec<MemoryRecord>, String> {
        self.storage
            .list_memories(Some(scope))
            .await
            .map_err(|e| e.to_string())
    }

    /// Delete a memory record.
    pub async fn delete_memory(&self, scope: &str, key: &str) -> Result<(), String> {
        self.storage
            .delete_memory(scope, key)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Structured data store ────────────────────────────────────────────

    /// Get a structured (JSON) value by key.
    pub async fn get_structured(
        &self,
        key: &str,
    ) -> Result<Option<serde_json::Value>, String> {
        self.storage
            .get_structured(key)
            .await
            .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    pub tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn storage_open() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path);
        assert!(storage.is_ok(), "Should open SQLite");
    }

    #[tokio::test]
    async fn storage_total_cost_empty_session() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();
        let sid = SessionId::new();
        // No session row — total_cost returns 0 (no FK needed for reads)
        let cost = storage.total_cost(&sid).await;
        assert!(cost.is_ok());
        assert_eq!(cost.unwrap(), 0.0);
    }

    #[tokio::test]
    async fn storage_list_empty_session() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();
        let sid = SessionId::new();
        let msgs = storage.list_messages(&sid).await.unwrap();
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn storage_list_tool_calls_empty() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();
        let sid = SessionId::new();
        let calls = storage.list_tool_calls(&sid).await.unwrap();
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn storage_list_audit_empty() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();
        let sid = SessionId::new();
        let entries = storage.list_audit(&sid).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn storage_recover_crashed_sessions_none() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();
        let recovered = storage.recover_crashed_sessions().unwrap();
        assert!(recovered.is_empty());
    }

    #[tokio::test]
    async fn storage_telemetry_snapshots_empty() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();
        let sid = SessionId::new();
        let snaps = storage.load_telemetry_snapshots(&sid).await.unwrap();
        assert!(snaps.is_empty());
    }

    #[tokio::test]
    async fn storage_telemetry_snapshot_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();
        let sid = SessionId::new();
        let data = serde_json::json!({"rot_score": 0.3, "drift": 0.1});
        storage
            .save_telemetry_snapshot(&sid, "context_health", &data)
            .await
            .unwrap();
        let snaps = storage.load_telemetry_snapshots(&sid).await.unwrap();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].0, "context_health");
        assert_eq!(snaps[0].1["rot_score"], 0.3);
    }

    #[tokio::test]
    async fn storage_memory_crud() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();

        // Set
        storage.set_memory("project", "lang", "rust").await.unwrap();

        // Get
        let rec = storage.get_memory("project", "lang").await.unwrap();
        assert!(rec.is_some());
        assert_eq!(rec.unwrap().value, "rust");

        // List
        let list = storage.list_memories("project").await.unwrap();
        assert_eq!(list.len(), 1);

        // Delete
        storage.delete_memory("project", "lang").await.unwrap();
        let rec = storage.get_memory("project", "lang").await.unwrap();
        assert!(rec.is_none());
    }

    #[tokio::test]
    async fn storage_get_structured_empty() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = StorageBridge::open(&db_path).unwrap();
        let val = storage.get_structured("nonexistent").await.unwrap();
        assert!(val.is_none());
    }
}
