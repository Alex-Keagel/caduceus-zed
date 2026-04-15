//! Storage bridge — session persistence, audit logs, cost tracking.

use caduceus_core::SessionId;
use caduceus_storage::SqliteStorage;
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
}
