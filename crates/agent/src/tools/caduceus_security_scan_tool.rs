use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Scans code for security vulnerabilities, exposed secrets, and OWASP issues.
/// Use this to audit files before committing or to check for potential security problems.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusSecurityScanToolInput {
    /// The file path to scan.
    pub path: String,

    /// The file content to scan. If not provided, the tool reads from the path.
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityFinding {
    pub severity: String,
    pub category: String,
    pub message: String,
    pub line: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusSecurityScanToolOutput {
    Success {
        findings: Vec<SecurityFinding>,
        secrets_found: Vec<String>,
        owasp_issues: Vec<String>,
    },
    Error {
        error: String,
    },
}

impl From<CaduceusSecurityScanToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusSecurityScanToolOutput) -> Self {
        match output {
            CaduceusSecurityScanToolOutput::Success {
                findings,
                secrets_found,
                owasp_issues,
            } => {
                if findings.is_empty() && secrets_found.is_empty() && owasp_issues.is_empty() {
                    return "No security issues found. Code looks clean.".into();
                }
                let mut text = String::new();
                if !findings.is_empty() {
                    text.push_str(&format!("## {} Security Findings\n", findings.len()));
                    for f in &findings {
                        text.push_str(&format!(
                            "- [{}] {}: {}{}\n",
                            f.severity,
                            f.category,
                            f.message,
                            f.line.map(|l| format!(" (line {})", l)).unwrap_or_default()
                        ));
                    }
                }
                if !secrets_found.is_empty() {
                    text.push_str(&format!("\n## {} Exposed Secrets\n", secrets_found.len()));
                    for s in &secrets_found {
                        // Strip the redacted preview from the output: even an 8-char
                        // prefix can leak short secrets (passwords, short tokens).
                        // The engine returns "[KIND] at position X-Y: PREVIEW" — keep
                        // only the kind+position so the model knows what/where but
                        // never sees secret material echoed back into context.
                        let location_only = redact_secret_preview(s);
                        text.push_str(&format!("- {location_only}\n"));
                    }
                }
                if !owasp_issues.is_empty() {
                    text.push_str(&format!("\n## {} OWASP Issues\n", owasp_issues.len()));
                    for o in &owasp_issues {
                        text.push_str(&format!("- {o}\n"));
                    }
                }
                text.into()
            }
            CaduceusSecurityScanToolOutput::Error { error } => {
                format!("Security scan error: {error}").into()
            }
        }
    }
}

pub struct CaduceusSecurityScanTool {
    engine: Arc<caduceus_bridge::engine::CaduceusEngine>,
}

impl CaduceusSecurityScanTool {
    pub fn new(engine: Arc<caduceus_bridge::engine::CaduceusEngine>) -> Self {
        Self { engine }
    }
}

impl AgentTool for CaduceusSecurityScanTool {
    type Input = CaduceusSecurityScanToolInput;
    type Output = CaduceusSecurityScanToolOutput;

    const NAME: &'static str = "caduceus_security_scan";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            format!("Security scan: {}", input.path).into()
        } else {
            "Security scan".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let engine = self.engine.clone();
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusSecurityScanToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let content = match &input.content {
                Some(c) => c.clone(),
                None => {
                    // SEC-17: validate path — reject traversal and absolute paths,
                    // resolve relative to engine project root, then re-check the
                    // canonical target so symlinks can't escape the project or
                    // bypass `is_sensitive_file`.
                    if input.path.contains("..") {
                        return Err(CaduceusSecurityScanToolOutput::Error {
                            error: format!("Path traversal not allowed: {}", input.path),
                        });
                    }
                    let raw = std::path::Path::new(&input.path);
                    if raw.is_absolute() {
                        return Err(CaduceusSecurityScanToolOutput::Error {
                            error:
                                "Absolute paths not allowed — use relative paths from project root"
                                    .to_string(),
                        });
                    }
                    if crate::tools::is_sensitive_file(&input.path) {
                        return Err(CaduceusSecurityScanToolOutput::Error {
                            error: format!("Cannot scan sensitive file: {}", input.path),
                        });
                    }
                    let project_root = engine.project_root.clone();
                    let project_canonical =
                        project_root.canonicalize().unwrap_or(project_root.clone());
                    let resolved = project_root.join(&input.path);
                    let canonical = resolved.canonicalize().map_err(|e| {
                        CaduceusSecurityScanToolOutput::Error {
                            error: format!("Cannot resolve path {}: {e}", input.path),
                        }
                    })?;
                    if !canonical.starts_with(&project_canonical) {
                        return Err(CaduceusSecurityScanToolOutput::Error {
                            error: format!(
                                "Path '{}' resolves outside the project root",
                                input.path
                            ),
                        });
                    }
                    if crate::tools::is_sensitive_file(&canonical.to_string_lossy()) {
                        return Err(CaduceusSecurityScanToolOutput::Error {
                            error: format!("Resolved path is sensitive: {}", canonical.display()),
                        });
                    }
                    std::fs::read_to_string(&canonical).map_err(|e| {
                        CaduceusSecurityScanToolOutput::Error {
                            error: format!("Failed to read file {}: {e}", input.path),
                        }
                    })?
                }
            };

            let findings: Vec<SecurityFinding> = engine
                .security_scan_file(&input.path, &content)
                .into_iter()
                .map(|f| SecurityFinding {
                    severity: f.severity,
                    category: f.rule_id,
                    message: f.description,
                    line: Some(f.line),
                })
                .collect();

            let secrets_found = engine.scan_secrets(&content);
            let owasp_issues = engine.owasp_check(&content);

            Ok(CaduceusSecurityScanToolOutput::Success {
                findings,
                secrets_found,
                owasp_issues,
            })
        })
    }
}

/// Strips the secret preview from an engine-formatted finding string.
///
/// The engine returns `"[KIND] at position X-Y: PREVIEW"`. Even a short
/// preview can leak the entire value when the underlying secret is small
/// (passwords, 6-digit OTP, narrow API tokens). The model only needs the
/// kind and location to decide where to look — it must never see the value.
fn redact_secret_preview(raw: &str) -> &str {
    raw.split(": ").next().unwrap_or(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for bug C8: the security scan output used to include the
    /// raw secret preview from the engine, leaking the value back into the
    /// LLM context. The redaction must drop everything after the first
    /// ": " separator.
    #[test]
    fn redacts_preview_after_colon_space() {
        let raw = "[AWS_KEY] at position 12-32: AKIAIOSFODNN7EXAMPLE";
        assert_eq!(redact_secret_preview(raw), "[AWS_KEY] at position 12-32");
    }

    #[test]
    fn redacts_short_secret_completely() {
        // Short tokens are the most dangerous case — must not appear at all.
        let raw = "[OTP] at position 0-6: 123456";
        let redacted = redact_secret_preview(raw);
        assert!(!redacted.contains("123456"));
        assert!(!redacted.contains("12"));
        assert_eq!(redacted, "[OTP] at position 0-6");
    }

    #[test]
    fn passes_through_when_no_separator() {
        let raw = "[GENERIC] no preview";
        assert_eq!(redact_secret_preview(raw), "[GENERIC] no preview");
    }

    /// Regression for bug C8: ensure that even a string that contains
    /// multiple `": "` sequences only keeps the kind+location prefix.
    #[test]
    fn redacts_only_to_first_separator() {
        let raw = "[JWT] at position 0-200: eyJhbG: ciOiJIUzI1NiJ9";
        let redacted = redact_secret_preview(raw);
        assert_eq!(redacted, "[JWT] at position 0-200");
        assert!(!redacted.contains("eyJ"));
    }
}
