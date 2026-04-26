//! Storage bridge — session persistence, audit logs, cost tracking, memory CRUD,
//! task management, wiki operations, trajectory import/export, and trace events.

use caduceus_core::{AuditEntry, SessionId};
use caduceus_storage::{
    GitTrackableStore, MemoryRecord, SqliteStorage, StoredCost, StoredToolCall, TraceEvent,
    TraceEventType, TrajectoryRecorder, WikiEngine, WikiIndex, WikiLinter, WikiPage,
};
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

    pub fn open_in_memory() -> Result<Self, String> {
        SqliteStorage::open_in_memory()
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

    pub async fn list_messages(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<StoredMessage>, String> {
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
        self.storage
            .total_cost(session_id)
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn export_transcript(
        &self,
        session_id: &SessionId,
        output_path: &Path,
    ) -> Result<(), String> {
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
    pub async fn set_memory(&self, scope: &str, key: &str, value: &str) -> Result<(), String> {
        self.storage
            .set_memory(scope, key, value, "bridge")
            .await
            .map_err(|e| e.to_string())
    }

    /// Get a single memory record.
    pub async fn get_memory(&self, scope: &str, key: &str) -> Result<Option<MemoryRecord>, String> {
        self.storage
            .get_memory(scope, key)
            .await
            .map_err(|e| e.to_string())
    }

    /// List all memories in a scope (None = all scopes).
    pub async fn list_memories(&self, scope: &str) -> Result<Vec<MemoryRecord>, String> {
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
    pub async fn get_structured(&self, key: &str) -> Result<Option<serde_json::Value>, String> {
        self.storage
            .get_structured(key)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Cost records ─────────────────────────────────────────────────────

    /// List cost records for a session.
    pub async fn list_costs(&self, session_id: &SessionId) -> Result<Vec<StoredCost>, String> {
        self.storage
            .list_costs(session_id)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Trace events ─────────────────────────────────────────────────────

    /// List trace events for a session.
    pub async fn list_trace_events(&self, session_id: &str) -> Result<Vec<TraceEvent>, String> {
        self.storage
            .list_trace_events(session_id)
            .await
            .map_err(|e| e.to_string())
    }

    /// Record a trace event.
    pub async fn record_trace_event(
        &self,
        session_id: &str,
        event_type: TraceEventType,
        data: serde_json::Value,
    ) -> Result<i64, String> {
        let event = TraceEvent {
            session_id: session_id.to_string(),
            event_type,
            event_data: data,
            duration_ms: None,
            timestamp: chrono::Utc::now(),
        };
        self.storage
            .record_trace_event(&event)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Audit log (append) ───────────────────────────────────────────────

    /// Append an audit log entry.
    pub async fn append_audit(&self, entry: &AuditEntry) -> Result<i64, String> {
        self.storage
            .append_audit(entry)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Task CRUD (Git-trackable) ────────────────────────────────────────

    /// List tasks from the git-trackable store.
    pub fn list_tasks(&self, project_root: &Path) -> Result<Vec<serde_json::Value>, String> {
        let store = GitTrackableStore::new(project_root);
        store.list_tasks()
    }

    /// Load a single task by ID.
    pub fn load_task(&self, project_root: &Path, id: &str) -> Result<serde_json::Value, String> {
        let store = GitTrackableStore::new(project_root);
        store.load_task(id)
    }

    /// Save (upsert) a task.
    pub fn save_task(&self, project_root: &Path, task: &serde_json::Value) -> Result<(), String> {
        let store = GitTrackableStore::new(project_root);
        store.save_task(task)
    }

    /// Delete a task by ID.
    pub fn delete_task(&self, project_root: &Path, id: &str) -> Result<(), String> {
        let store = GitTrackableStore::new(project_root);
        store.delete_task(id)
    }

    // ── Wiki operations ──────────────────────────────────────────────────

    /// List all wiki pages.
    pub fn list_pages(&self, project_root: &Path) -> Result<Vec<WikiPage>, String> {
        let wiki = WikiEngine::new(project_root);
        wiki.init().map_err(|e| e.to_string())?;
        wiki.list_pages().map_err(|e| e.to_string())
    }

    /// Read a wiki page by slug.
    pub fn read_page(&self, project_root: &Path, slug: &str) -> Result<String, String> {
        let wiki = WikiEngine::new(project_root);
        wiki.read_page(slug).map_err(|e| e.to_string())
    }

    /// Write (upsert) a wiki page.
    pub fn write_page(&self, project_root: &Path, slug: &str, content: &str) -> Result<(), String> {
        let wiki = WikiEngine::new(project_root);
        wiki.init().map_err(|e| e.to_string())?;
        wiki.write_page(slug, content).map_err(|e| e.to_string())
    }

    /// Delete a wiki page.
    pub fn delete_page(&self, project_root: &Path, slug: &str) -> Result<(), String> {
        let wiki = WikiEngine::new(project_root);
        wiki.delete_page(slug).map_err(|e| e.to_string())
    }

    /// Search wiki pages by query.
    pub fn search_pages(&self, project_root: &Path, query: &str) -> Result<Vec<WikiPage>, String> {
        let wiki = WikiEngine::new(project_root);
        wiki.init().map_err(|e| e.to_string())?;
        wiki.search_pages(query).map_err(|e| e.to_string())
    }

    /// Find orphan pages (no inbound links).
    pub fn find_orphans(&self, project_root: &Path) -> Result<Vec<String>, String> {
        let wiki = WikiEngine::new(project_root);
        wiki.init().map_err(|e| e.to_string())?;
        let pages = wiki.list_pages().map_err(|e| e.to_string())?;
        let index = WikiIndex::new();
        let findings = WikiLinter::find_orphans(&pages, &index);
        Ok(findings.into_iter().map(|f| f.page).collect())
    }

    /// Find empty wiki pages (zero bytes).
    pub fn find_empty_pages(&self, project_root: &Path) -> Result<Vec<String>, String> {
        let wiki = WikiEngine::new(project_root);
        wiki.init().map_err(|e| e.to_string())?;
        let pages = wiki.list_pages().map_err(|e| e.to_string())?;
        let findings = WikiLinter::find_empty_pages(&pages);
        Ok(findings.into_iter().map(|f| f.page).collect())
    }

    /// Find stale wiki pages (not updated in N days).
    pub fn find_stale_pages(
        &self,
        project_root: &Path,
        max_age_days: u32,
    ) -> Result<Vec<String>, String> {
        let wiki = WikiEngine::new(project_root);
        wiki.init().map_err(|e| e.to_string())?;
        let pages = wiki.list_pages().map_err(|e| e.to_string())?;
        let findings = WikiLinter::find_stale_pages(&pages, max_age_days);
        Ok(findings.into_iter().map(|f| f.page).collect())
    }

    /// Check whether a wiki page exists.
    pub fn page_exists(&self, project_root: &Path, slug: &str) -> bool {
        let wiki = WikiEngine::new(project_root);
        wiki.page_exists(slug)
    }

    // ── Trajectory import/export ─────────────────────────────────────────

    /// Import a trajectory from JSON.
    pub fn import_trajectory(&self, data: &str) -> Result<String, String> {
        let recorder = TrajectoryRecorder::import_trajectory(data).map_err(|e| e.to_string())?;
        Ok(recorder.export_trajectory())
    }

    /// Export a trajectory for a session (creates a new recorder with no events).
    pub fn export_trajectory(&self, session_id: &str) -> String {
        let recorder = TrajectoryRecorder::new(session_id);
        recorder.export_trajectory()
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
    async fn storage_open_in_memory() {
        let storage = StorageBridge::open_in_memory();
        assert!(storage.is_ok(), "Should open in-memory SQLite");
    }

    #[tokio::test]
    async fn storage_total_cost_empty_session() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        let cost = storage.total_cost(&sid).await;
        assert!(cost.is_ok());
        assert_eq!(cost.unwrap(), 0.0);
    }

    #[tokio::test]
    async fn storage_list_empty_session() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        let msgs = storage.list_messages(&sid).await.unwrap();
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn storage_list_tool_calls_empty() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        let calls = storage.list_tool_calls(&sid).await.unwrap();
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn storage_list_audit_empty() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        let entries = storage.list_audit(&sid).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn storage_recover_crashed_sessions_none() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let recovered = storage.recover_crashed_sessions().unwrap();
        assert!(recovered.is_empty());
    }

    #[tokio::test]
    async fn storage_telemetry_snapshots_empty() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        let snaps = storage.load_telemetry_snapshots(&sid).await.unwrap();
        assert!(snaps.is_empty());
    }

    #[tokio::test]
    async fn storage_telemetry_snapshot_roundtrip() {
        let storage = StorageBridge::open_in_memory().unwrap();
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
        let storage = StorageBridge::open_in_memory().unwrap();

        storage.set_memory("project", "lang", "rust").await.unwrap();
        let rec = storage.get_memory("project", "lang").await.unwrap();
        assert!(rec.is_some());
        assert_eq!(rec.unwrap().value, "rust");

        let list = storage.list_memories("project").await.unwrap();
        assert_eq!(list.len(), 1);

        storage.delete_memory("project", "lang").await.unwrap();
        let rec = storage.get_memory("project", "lang").await.unwrap();
        assert!(rec.is_none());
    }

    #[tokio::test]
    async fn storage_get_structured_empty() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let val = storage.get_structured("nonexistent").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn storage_list_costs_empty() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        let costs = storage.list_costs(&sid).await.unwrap();
        assert!(costs.is_empty());
    }

    #[tokio::test]
    async fn storage_record_cost_requires_session() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        // record_cost on non-existent session returns FK error
        let result = storage
            .record_cost(&sid, "anthropic", "claude", 100, 50, 0.005)
            .await;
        assert!(result.is_err(), "Should fail FK: no session row");
    }

    #[tokio::test]
    async fn storage_trace_event_roundtrip() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        let id = storage
            .record_trace_event(
                &sid.to_string(),
                TraceEventType::ToolExec,
                serde_json::json!({"tool": "bash"}),
            )
            .await
            .unwrap();
        assert!(id > 0);
        let events = storage.list_trace_events(&sid.to_string()).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_data["tool"], "bash");
    }

    #[tokio::test]
    async fn storage_list_trace_events_empty() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let events = storage.list_trace_events("no-such-session").await.unwrap();
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn storage_append_audit_requires_session() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let sid = SessionId::new();
        let entry = AuditEntry {
            session_id: sid.clone(),
            capability: "fs:read".to_string(),
            tool_name: "read_file".to_string(),
            args_redacted: "/foo.rs".to_string(),
            decision: caduceus_core::AuditDecision::Allowed,
            timestamp: chrono::Utc::now(),
        };
        // append_audit on non-existent session returns FK error
        let result = storage.append_audit(&entry).await;
        assert!(result.is_err(), "Should fail FK: no session row");
    }

    #[test]
    fn storage_task_crud() {
        let dir = tempfile::tempdir().unwrap();
        let storage = StorageBridge::open_in_memory().unwrap();
        let root = dir.path();

        let task = serde_json::json!({"id": "t1", "title": "Test task"});
        storage.save_task(root, &task).unwrap();

        let loaded = storage.load_task(root, "t1").unwrap();
        assert_eq!(loaded["title"], "Test task");

        let tasks = storage.list_tasks(root).unwrap();
        assert_eq!(tasks.len(), 1);

        storage.delete_task(root, "t1").unwrap();
        let tasks = storage.list_tasks(root).unwrap();
        assert!(tasks.is_empty());
    }

    #[test]
    fn storage_wiki_crud() {
        let dir = tempfile::tempdir().unwrap();
        let storage = StorageBridge::open_in_memory().unwrap();
        let root = dir.path();

        // Write
        storage
            .write_page(root, "test-page", "# Test\nHello wiki")
            .unwrap();
        assert!(storage.page_exists(root, "test-page"));

        // Read
        let content = storage.read_page(root, "test-page").unwrap();
        assert!(content.contains("Hello wiki"));

        // List
        let pages = storage.list_pages(root).unwrap();
        assert!(pages.iter().any(|p| p.slug == "test-page"));

        // Search
        let results = storage.search_pages(root, "Hello").unwrap();
        assert!(!results.is_empty());

        // Delete
        storage.delete_page(root, "test-page").unwrap();
        assert!(!storage.page_exists(root, "test-page"));
    }

    #[test]
    fn storage_find_empty_pages() {
        let dir = tempfile::tempdir().unwrap();
        let storage = StorageBridge::open_in_memory().unwrap();
        let root = dir.path();
        storage.write_page(root, "empty-page", "").unwrap();
        let empty = storage.find_empty_pages(root).unwrap();
        assert!(empty.contains(&"empty-page".to_string()));
    }

    #[test]
    fn storage_find_orphans() {
        let dir = tempfile::tempdir().unwrap();
        let storage = StorageBridge::open_in_memory().unwrap();
        let root = dir.path();
        storage.write_page(root, "page-a", "# A\nContent").unwrap();
        storage.write_page(root, "page-b", "# B\nContent").unwrap();
        let orphans = storage.find_orphans(root).unwrap();
        // Both pages are orphans since neither links to the other
        assert!(orphans.len() >= 2);
    }

    #[test]
    fn storage_trajectory_import_export() {
        let storage = StorageBridge::open_in_memory().unwrap();

        // Export creates an empty trajectory
        let json = storage.export_trajectory("test-session");
        assert!(json.contains("test-session"));

        // Round-trip import
        let reimported = storage.import_trajectory(&json).unwrap();
        assert!(reimported.contains("test-session"));
    }

    #[test]
    fn storage_trajectory_import_invalid() {
        let storage = StorageBridge::open_in_memory().unwrap();
        let result = storage.import_trajectory("not json");
        assert!(result.is_err());
    }
}
