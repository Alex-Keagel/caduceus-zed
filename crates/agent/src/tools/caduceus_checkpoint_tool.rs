use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Create, list, restore, and diff file checkpoints for safe rollback.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusCheckpointToolInput {
    /// The checkpoint operation to perform.
    pub operation: CheckpointOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointOperation {
    /// Snapshot all modified files (per `git diff --name-only`) into a checkpoint directory.
    Create {
        /// A short human-readable label for this checkpoint.
        label: String,
    },
    /// List all existing checkpoints.
    List,
    /// Restore files from a checkpoint back to the working tree.
    Restore {
        /// The checkpoint directory name (e.g. `20250101-120000-my-label`).
        checkpoint_id: String,
    },
    /// Show a summary of what changed since a given checkpoint.
    Diff {
        /// The checkpoint directory name to compare against.
        checkpoint_id: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusCheckpointToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusCheckpointToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusCheckpointToolOutput) -> Self {
        match output {
            CaduceusCheckpointToolOutput::Text { text } => text.into(),
            CaduceusCheckpointToolOutput::Error { error } => {
                format!("Checkpoint error: {error}").into()
            }
        }
    }
}

pub struct CaduceusCheckpointTool {
    project_root: PathBuf,
}

impl CaduceusCheckpointTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }
}

/// Manifest stored alongside each checkpoint.
#[derive(Debug, Serialize, Deserialize)]
struct CheckpointManifest {
    label: String,
    timestamp: String,
    files: Vec<String>,
}

impl AgentTool for CaduceusCheckpointTool {
    type Input = CaduceusCheckpointToolInput;
    type Output = CaduceusCheckpointToolOutput;

    const NAME: &'static str = "caduceus_checkpoint";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Execute
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            let op = match &input.operation {
                CheckpointOperation::Create { label } => format!("create '{label}'"),
                CheckpointOperation::List => "list".to_string(),
                CheckpointOperation::Restore { checkpoint_id } => {
                    format!("restore '{checkpoint_id}'")
                }
                CheckpointOperation::Diff { checkpoint_id } => {
                    format!("diff '{checkpoint_id}'")
                }
            };
            format!("Checkpoint {op}").into()
        } else {
            "Checkpoint".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project_root = self.project_root.clone();
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusCheckpointToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let result: Result<String, String> = match input.operation {
                CheckpointOperation::Create { label } => create_checkpoint(&project_root, &label),
                CheckpointOperation::List => list_checkpoints(&project_root),
                CheckpointOperation::Restore { checkpoint_id } => {
                    restore_checkpoint(&project_root, &checkpoint_id)
                }
                CheckpointOperation::Diff { checkpoint_id } => {
                    diff_checkpoint(&project_root, &checkpoint_id)
                }
            };

            match result {
                Ok(text) => Ok(CaduceusCheckpointToolOutput::Text { text }),
                Err(e) => Err(CaduceusCheckpointToolOutput::Error { error: e }),
            }
        })
    }
}

fn checkpoints_dir(project_root: &std::path::Path) -> PathBuf {
    project_root.join(".caduceus").join("checkpoints")
}

fn create_checkpoint(project_root: &std::path::Path, label: &str) -> Result<String, String> {
    // Get modified files via `git diff --name-only`
    let output = std::process::Command::new("git")
        .args(["diff", "--name-only"])
        .current_dir(project_root)
        .output()
        .map_err(|e| format!("Failed to run git diff: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let files: Vec<String> = stdout
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();

    if files.is_empty() {
        return Ok("No modified files to checkpoint.".to_string());
    }

    let now = chrono::Utc::now();
    let timestamp = now.format("%Y%m%d-%H%M%S").to_string();
    let safe_label: String = label
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let checkpoint_name = format!("{timestamp}-{safe_label}");
    let checkpoint_dir = checkpoints_dir(project_root).join(&checkpoint_name);

    std::fs::create_dir_all(&checkpoint_dir)
        .map_err(|e| format!("Failed to create checkpoint dir: {e}"))?;

    for file in &files {
        let src = project_root.join(file);
        if src.exists() {
            let dest = checkpoint_dir.join(file);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create dir for {file}: {e}"))?;
            }
            std::fs::copy(&src, &dest).map_err(|e| format!("Failed to copy {file}: {e}"))?;
        }
    }

    let manifest = CheckpointManifest {
        label: label.to_string(),
        timestamp: now.to_rfc3339(),
        files: files.clone(),
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("Failed to serialize manifest: {e}"))?;
    std::fs::write(checkpoint_dir.join("manifest.json"), manifest_json)
        .map_err(|e| format!("Failed to write manifest: {e}"))?;

    Ok(format!(
        "Checkpoint '{checkpoint_name}' created with {} file(s):\n{}",
        files.len(),
        files
            .iter()
            .map(|f| format!("  - {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    ))
}

fn list_checkpoints(project_root: &std::path::Path) -> Result<String, String> {
    let dir = checkpoints_dir(project_root);
    if !dir.exists() {
        return Ok("No checkpoints found.".to_string());
    }

    let mut entries: Vec<(String, CheckpointManifest)> = Vec::new();
    let read_dir =
        std::fs::read_dir(&dir).map_err(|e| format!("Failed to read checkpoints dir: {e}"))?;

    for entry in read_dir {
        let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
        let manifest_path = entry.path().join("manifest.json");
        if manifest_path.exists() {
            let data = std::fs::read_to_string(&manifest_path)
                .map_err(|e| format!("Failed to read manifest: {e}"))?;
            if let Ok(manifest) = serde_json::from_str::<CheckpointManifest>(&data) {
                let name = entry.file_name().to_string_lossy().to_string();
                entries.push((name, manifest));
            }
        }
    }

    if entries.is_empty() {
        return Ok("No checkpoints found.".to_string());
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let lines: Vec<String> = entries
        .iter()
        .map(|(name, m)| {
            format!(
                "- **{name}** — \"{}\" ({} files, {})",
                m.label,
                m.files.len(),
                m.timestamp
            )
        })
        .collect();

    Ok(format!(
        "{} checkpoint(s):\n{}",
        entries.len(),
        lines.join("\n")
    ))
}

fn validate_checkpoint_id(checkpoint_id: &str) -> Result<(), String> {
    if checkpoint_id.is_empty() {
        return Err("Checkpoint id cannot be empty".to_string());
    }
    let p = std::path::Path::new(checkpoint_id);
    if p.is_absolute() {
        return Err("Checkpoint id must not be absolute".to_string());
    }
    let mut components = 0usize;
    for c in p.components() {
        match c {
            std::path::Component::Normal(_) => components += 1,
            _ => {
                return Err("Checkpoint id must be a single non-traversing path segment".to_string());
            }
        }
    }
    if components != 1 {
        return Err("Checkpoint id must be a single path segment".to_string());
    }
    Ok(())
}

fn restore_checkpoint(
    project_root: &std::path::Path,
    checkpoint_id: &str,
) -> Result<String, String> {
    validate_checkpoint_id(checkpoint_id)?;
    let checkpoint_dir = checkpoints_dir(project_root).join(checkpoint_id);

    // Prevent path traversal — canonicalize REQUIRES existence; if the dir
    // does not exist, the id is invalid (we already rejected traversal above).
    let base = checkpoints_dir(project_root);
    let _ = std::fs::create_dir_all(&base);
    let resolved = checkpoint_dir
        .canonicalize()
        .map_err(|_| format!("Checkpoint '{checkpoint_id}' not found."))?;
    let base_resolved = base.canonicalize().unwrap_or(base.clone());
    if !resolved.starts_with(&base_resolved) {
        return Err("Invalid checkpoint id: path traversal detected".to_string());
    }

    let manifest_path = checkpoint_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(format!("Checkpoint '{checkpoint_id}' not found."));
    }

    let data = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Failed to read manifest: {e}"))?;
    let manifest: CheckpointManifest =
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse manifest: {e}"))?;

    let project_canonical = project_root
        .canonicalize()
        .unwrap_or(project_root.to_path_buf());

    let mut restored = Vec::new();
    for file in &manifest.files {
        // Prevent restoring files outside project root
        let dest_normalized = {
            let mut p = project_root.to_path_buf();
            for component in std::path::Path::new(file).components() {
                match component {
                    std::path::Component::ParentDir => {
                        p.pop();
                    }
                    std::path::Component::Normal(c) => p.push(c),
                    _ => {}
                }
            }
            p
        };
        if !dest_normalized.starts_with(&project_canonical) {
            return Err(format!(
                "Refusing to restore '{}': path escapes project root",
                file
            ));
        }

        let src = checkpoint_dir.join(file);
        let dest = project_root.join(file);
        if src.exists() {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create dir for {file}: {e}"))?;
            }
            std::fs::copy(&src, &dest).map_err(|e| format!("Failed to restore {file}: {e}"))?;
            restored.push(file.clone());
        }
    }

    Ok(format!(
        "Restored {} file(s) from checkpoint '{checkpoint_id}':\n{}",
        restored.len(),
        restored
            .iter()
            .map(|f| format!("  - {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    ))
}

fn diff_checkpoint(project_root: &std::path::Path, checkpoint_id: &str) -> Result<String, String> {
    validate_checkpoint_id(checkpoint_id)?;
    let checkpoint_dir = checkpoints_dir(project_root).join(checkpoint_id);

    // Prevent path traversal — fail if the dir doesn't exist rather than
    // silently falling back to the un-canonicalized path.
    let base = checkpoints_dir(project_root);
    let _ = std::fs::create_dir_all(&base);
    let resolved = checkpoint_dir
        .canonicalize()
        .map_err(|_| format!("Checkpoint '{checkpoint_id}' not found."))?;
    let base_resolved = base.canonicalize().unwrap_or(base.clone());
    if !resolved.starts_with(&base_resolved) {
        return Err("Invalid checkpoint id: path traversal detected".to_string());
    }

    let manifest_path = checkpoint_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(format!("Checkpoint '{checkpoint_id}' not found."));
    }

    let data = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Failed to read manifest: {e}"))?;
    let manifest: CheckpointManifest =
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse manifest: {e}"))?;

    let mut diffs = Vec::new();
    for file in &manifest.files {
        let checkpoint_file = checkpoint_dir.join(file);
        let current_file = project_root.join(file);

        let cp_exists = checkpoint_file.exists();
        let cur_exists = current_file.exists();

        let status = match (cp_exists, cur_exists) {
            (true, true) => {
                let cp_content = std::fs::read(&checkpoint_file).unwrap_or_default();
                let cur_content = std::fs::read(&current_file).unwrap_or_default();
                if cp_content == cur_content {
                    "unchanged"
                } else {
                    "modified"
                }
            }
            (true, false) => "deleted (current file missing)",
            (false, true) => "new (not in checkpoint)",
            (false, false) => "missing from both",
        };

        diffs.push(format!("  - {file}: {status}"));
    }

    Ok(format!(
        "Diff against checkpoint '{checkpoint_id}':\n{}",
        diffs.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for bug C10b: checkpoint ids were used as path segments
    /// without validation, so `..`, absolute paths, multi-segment ids, or
    /// embedded path separators all let a caller break out of the
    /// `.caduceus/checkpoints/` jail. `validate_checkpoint_id` must reject
    /// every one of these adversarial inputs.
    #[test]
    fn rejects_path_traversal() {
        assert!(validate_checkpoint_id("..").is_err());
        assert!(validate_checkpoint_id("../escape").is_err());
        assert!(validate_checkpoint_id("foo/../bar").is_err());
    }

    #[test]
    fn rejects_absolute_paths() {
        assert!(validate_checkpoint_id("/etc/passwd").is_err());
        // Windows-style absolute roots also count as components other than Normal
        // on platforms that recognize them.
    }

    #[test]
    fn rejects_multi_segment_ids() {
        assert!(validate_checkpoint_id("foo/bar").is_err());
        assert!(validate_checkpoint_id("a/b/c").is_err());
    }

    #[test]
    fn rejects_empty_id() {
        assert!(validate_checkpoint_id("").is_err());
    }

    #[test]
    fn rejects_current_dir() {
        assert!(validate_checkpoint_id(".").is_err());
    }

    #[test]
    fn accepts_normal_single_segment() {
        assert!(validate_checkpoint_id("checkpoint-2024-01-01").is_ok());
        assert!(validate_checkpoint_id("abc123").is_ok());
        assert!(validate_checkpoint_id("snapshot_v1").is_ok());
    }
}
