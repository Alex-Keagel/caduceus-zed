//! Runtime bridge — file operations, path validation, snapshots, E2b volumes & templates.

use caduceus_runtime::{
    E2bSnapshotManager, E2bTemplateManager, E2bVolumeManager, FileOps, PocRunner,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// Unified bridge into caduceus-runtime subsystems.
pub struct RuntimeBridge {
    workspace_root: PathBuf,
    pub file_ops: FileOps,
    pub poc_runner: PocRunner,
    pub snapshots: E2bSnapshotManager,
    pub volumes: E2bVolumeManager,
    pub templates: E2bTemplateManager,
}

impl RuntimeBridge {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        let root: PathBuf = workspace_root.into();
        Self {
            file_ops: FileOps::new(&root),
            poc_runner: PocRunner::new(),
            snapshots: E2bSnapshotManager::new(),
            volumes: E2bVolumeManager::new(),
            templates: E2bTemplateManager::new(),
            workspace_root: root,
        }
    }

    // ── Path validation ──────────────────────────────────────────────────

    /// Validate that a file path is safe (no traversal, allowed extension).
    pub fn validate_path(&self, path: &str) -> Result<String, String> {
        self.poc_runner
            .validate_path(path, &self.workspace_root.to_string_lossy())
    }

    // ── File operations ──────────────────────────────────────────────────

    /// Read a file within the workspace.
    pub async fn read_file(&self, path: &str) -> Result<String, String> {
        self.file_ops.read(path).await.map_err(|e| e.to_string())
    }

    /// Write content to a file within the workspace.
    pub async fn write_file(&self, path: &str, content: &str) -> Result<(), String> {
        self.file_ops
            .write(path, content)
            .await
            .map_err(|e| e.to_string())
    }

    /// Check whether a file or directory exists within the workspace.
    pub async fn exists(&self, path: &str) -> Result<bool, String> {
        self.file_ops.exists(path).await.map_err(|e| e.to_string())
    }

    /// List directory entries (non-recursive).
    pub async fn list_dir(&self, path: &str) -> Result<Vec<String>, String> {
        self.file_ops
            .list_dir(path)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Search ───────────────────────────────────────────────────────────

    /// Glob search relative to workspace root.
    pub async fn glob_search(&self, pattern: &str) -> Result<Vec<String>, String> {
        self.file_ops
            .glob_search(pattern)
            .await
            .map_err(|e| e.to_string())
    }

    /// Grep search across workspace files.
    pub async fn grep_search(
        &self,
        pattern: &str,
        file_glob: Option<&str>,
        max_results: usize,
    ) -> Result<Vec<String>, String> {
        self.file_ops
            .grep_search(pattern, file_glob, max_results)
            .await
            .map_err(|e| e.to_string())
    }

    // ── Snapshots ────────────────────────────────────────────────────────

    /// Create an E2b snapshot, returning its ID.
    pub fn create_snapshot(
        &mut self,
        instance_id: &str,
        metadata: HashMap<String, String>,
    ) -> String {
        self.snapshots.create_snapshot(instance_id, metadata)
    }

    /// Restore an E2b snapshot, returning the instance ID.
    pub fn restore_snapshot(&self, snapshot_id: &str) -> Result<String, String> {
        self.snapshots
            .restore_snapshot(snapshot_id)
            .map_err(|e| e.to_string())
    }

    /// List all snapshots.
    pub fn list_snapshots(&self) -> Vec<crate::search::SnapshotInfo> {
        self.snapshots
            .list_snapshots()
            .iter()
            .map(|s| crate::search::SnapshotInfo {
                id: s.id.clone(),
                instance_id: s.instance_id.clone(),
                timestamp: s.timestamp,
            })
            .collect()
    }

    // ── E2b volumes ──────────────────────────────────────────────────────

    /// Create an E2b volume, returning its ID.
    pub fn create_volume(&mut self, name: &str, size_mb: u64, mount_path: &str) -> String {
        self.volumes.create_volume(name, size_mb, mount_path)
    }

    // ── E2b templates ────────────────────────────────────────────────────

    /// List registered E2b templates.
    pub fn list_templates(&self) -> Vec<&caduceus_runtime::E2bTemplate> {
        self.templates.list_templates()
    }

    /// Instantiate an E2b template, returning the instance.
    pub fn instantiate(
        &self,
        template_id: &str,
    ) -> Result<caduceus_runtime::E2bInstance, String> {
        self.templates
            .instantiate(template_id)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_validate_path_safe() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = RuntimeBridge::new(dir.path());
        // Create a file to validate
        std::fs::write(dir.path().join("test.py"), "print('hi')").unwrap();
        let result = bridge.validate_path("test.py");
        assert!(result.is_ok(), "Safe path should validate: {:?}", result);
    }

    #[test]
    fn runtime_validate_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = RuntimeBridge::new(dir.path());
        let result = bridge.validate_path("../../etc/passwd");
        assert!(result.is_err(), "Traversal should be rejected");
    }

    #[tokio::test]
    async fn runtime_read_write_exists() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = RuntimeBridge::new(dir.path());

        bridge.write_file("hello.txt", "world").await.unwrap();
        assert!(bridge.exists("hello.txt").await.unwrap());
        let content = bridge.read_file("hello.txt").await.unwrap();
        assert_eq!(content, "world");
    }

    #[tokio::test]
    async fn runtime_list_dir() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = RuntimeBridge::new(dir.path());
        bridge.write_file("a.txt", "a").await.unwrap();
        bridge.write_file("b.txt", "b").await.unwrap();
        let entries = bridge.list_dir(".").await.unwrap();
        assert!(entries.len() >= 2);
    }

    #[test]
    fn runtime_snapshot_create_restore_list() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = RuntimeBridge::new(dir.path());
        let snap_id = bridge.create_snapshot("inst-1", HashMap::new());
        assert!(!snap_id.is_empty());

        let snaps = bridge.list_snapshots();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].instance_id, "inst-1");

        let restored = bridge.restore_snapshot(&snap_id).unwrap();
        assert_eq!(restored, "inst-1");
    }

    #[test]
    fn runtime_volume_create() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = RuntimeBridge::new(dir.path());
        let vol_id = bridge.create_volume("data", 100, "/mnt/data");
        assert!(vol_id.contains("data"));
    }

    #[test]
    fn runtime_template_list_empty() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = RuntimeBridge::new(dir.path());
        assert!(bridge.list_templates().is_empty());
    }

    #[test]
    fn runtime_instantiate_missing_template() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = RuntimeBridge::new(dir.path());
        let result = bridge.instantiate("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn runtime_instantiate_registered_template() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = RuntimeBridge::new(dir.path());
        bridge.templates.register_template(caduceus_runtime::E2bTemplate {
            id: "tpl-1".into(),
            name: "python-sandbox".into(),
            dockerfile: None,
            start_command: None,
            env_vars: HashMap::new(),
        });
        let instance = bridge.instantiate("tpl-1").unwrap();
        assert_eq!(instance.template_id, "tpl-1");
        assert_eq!(instance.status, "running");
    }
}
