//! Tool wiring — re-exports all Caduceus tool types for Zed integration,
//! plus bridge wrappers for dependency scanning, LSP detection, snippet
//! finding, scaffold templates, notebook parsing, and test running.

pub use caduceus_core::{ToolResult, ToolSpec};
pub use caduceus_tools::Tool;
pub use caduceus_tools::ToolRegistry;

// Re-export tool subsystem types for consumers.
pub use caduceus_tools::{
    CellType, DepScanner, DepVulnerability, LineNumberFinder, LockFileType, LspBridgeTool,
    NotebookCell, NotebookCellTool, ScaffoldRegistry, ScaffoldType, SelfVerifier, SnippetLocation,
    VerificationResult,
};

use std::path::PathBuf;

/// Thin wrapper exposing tool-subsystem helpers that don't require engine state.
pub struct ToolsBridge;

impl ToolsBridge {
    /// Deep-copy a tool registry.
    pub fn clone_registry(registry: &ToolRegistry) -> ToolRegistry {
        registry.clone_registry()
    }

    /// List all tool specifications from a registry.
    pub fn list_tool_specs(registry: &ToolRegistry) -> Vec<ToolSpec> {
        registry.specs()
    }

    // ── Dependency scanning ──────────────────────────────────────────────

    /// Detect lock files in a directory.
    pub fn detect_lock_files(dir: &str) -> Vec<(String, LockFileType)> {
        DepScanner::detect_lock_files(dir)
    }

    /// Parse OSV-scanner JSON output into vulnerability records.
    pub fn parse_osv_output(json: &str) -> Vec<DepVulnerability> {
        DepScanner::parse_osv_output(json)
    }

    // ── LSP detection ────────────────────────────────────────────────────

    /// Return the recommended LSP server command for a language.
    pub fn detect_lsp_server(language: &str) -> Option<&'static str> {
        LspBridgeTool::detect_lsp_server(language)
    }

    // ── Snippet finding ──────────────────────────────────────────────────

    /// Find all occurrences of a snippet in content.
    /// Returns `(start_line, end_line, start_col)` tuples.
    pub fn find_all_snippets(content: &str, snippet: &str) -> Vec<(usize, usize, usize)> {
        LineNumberFinder::find_all_snippets(content, snippet)
            .into_iter()
            .map(|s| (s.start_line, s.end_line, s.start_col))
            .collect()
    }

    // ── Scaffold templates ───────────────────────────────────────────────

    /// List available scaffold template names for a given type.
    pub fn list_scaffold_templates(scaffold_type: ScaffoldType) -> Vec<&'static str> {
        ScaffoldRegistry::list_templates(scaffold_type)
    }

    // ── Notebook parsing ─────────────────────────────────────────────────

    /// Parse a Jupyter notebook JSON string into cells.
    pub fn parse_notebook(json: &str) -> Result<Vec<NotebookCell>, String> {
        NotebookCellTool::parse_notebook(json)
    }

    // ── Test runner ──────────────────────────────────────────────────────

    /// Run a test command in the given workspace directory.
    pub async fn run_tests(workspace: &str, command: &str) -> Result<VerificationResult, String> {
        let mut verifier = SelfVerifier::new(PathBuf::from(workspace));
        verifier
            .run_tests(command)
            .await
            .map_err(|e| e.to_string())
    }

    // NOTE: scan_diff is already exposed as CaduceusEngine::security_scan_diff.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_clone_registry() {
        let reg = ToolRegistry::new();
        let cloned = ToolsBridge::clone_registry(&reg);
        assert_eq!(cloned.specs().len(), reg.specs().len());
    }

    #[test]
    fn tools_list_tool_specs_empty() {
        let reg = ToolRegistry::new();
        assert!(ToolsBridge::list_tool_specs(&reg).is_empty());
    }

    #[test]
    fn tools_detect_lock_files_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let found = ToolsBridge::detect_lock_files(dir.path().to_str().unwrap());
        assert!(found.is_empty());
    }

    #[test]
    fn tools_detect_lock_files_cargo() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.lock"), "# lock").unwrap();
        let found = ToolsBridge::detect_lock_files(dir.path().to_str().unwrap());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1, LockFileType::CargoLock);
    }

    #[test]
    fn tools_parse_osv_output_empty() {
        let vulns = ToolsBridge::parse_osv_output("{}");
        assert!(vulns.is_empty());
    }

    #[test]
    fn tools_parse_osv_output_invalid_json() {
        let vulns = ToolsBridge::parse_osv_output("not json");
        assert!(vulns.is_empty());
    }

    #[test]
    fn tools_detect_lsp_server_rust() {
        assert_eq!(
            ToolsBridge::detect_lsp_server("rust"),
            Some("rust-analyzer")
        );
    }

    #[test]
    fn tools_detect_lsp_server_python() {
        assert_eq!(
            ToolsBridge::detect_lsp_server("python"),
            Some("pyright-langserver --stdio")
        );
    }

    #[test]
    fn tools_detect_lsp_server_unknown() {
        assert!(ToolsBridge::detect_lsp_server("brainfuck").is_none());
    }

    #[test]
    fn tools_find_all_snippets_single() {
        let content = "line1\nline2\nline3\n";
        let results = ToolsBridge::find_all_snippets(content, "line2");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2); // start_line
        assert_eq!(results[0].1, 2); // end_line
    }

    #[test]
    fn tools_find_all_snippets_multiple() {
        let content = "aaa\nbbb\naaa\n";
        let results = ToolsBridge::find_all_snippets(content, "aaa");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn tools_find_all_snippets_not_found() {
        let content = "hello world";
        let results = ToolsBridge::find_all_snippets(content, "missing");
        assert!(results.is_empty());
    }

    #[test]
    fn tools_list_scaffold_templates_skill() {
        let templates = ToolsBridge::list_scaffold_templates(ScaffoldType::Skill);
        assert!(!templates.is_empty());
    }

    #[test]
    fn tools_list_scaffold_templates_agent() {
        let templates = ToolsBridge::list_scaffold_templates(ScaffoldType::Agent);
        assert!(!templates.is_empty());
    }

    #[test]
    fn tools_parse_notebook_valid() {
        let json = r#"{"cells":[{"cell_type":"code","source":"print(1)","outputs":[],"execution_count":1}]}"#;
        let cells = ToolsBridge::parse_notebook(json).unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].cell_type, CellType::Code);
        assert!(cells[0].source.contains("print"));
    }

    #[test]
    fn tools_parse_notebook_markdown_cell() {
        let json = r##"{"cells":[{"cell_type":"markdown","source":"# Hello","outputs":[]}]}"##;
        let cells = ToolsBridge::parse_notebook(json).unwrap();
        assert_eq!(cells[0].cell_type, CellType::Markdown);
    }

    #[test]
    fn tools_parse_notebook_invalid() {
        let result = ToolsBridge::parse_notebook("not json");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn tools_run_tests_echo() {
        let dir = tempfile::tempdir().unwrap();
        let result = ToolsBridge::run_tests(dir.path().to_str().unwrap(), "echo ok").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn tools_run_tests_empty_command() {
        let dir = tempfile::tempdir().unwrap();
        let result = ToolsBridge::run_tests(dir.path().to_str().unwrap(), "").await;
        assert!(result.is_err());
    }
}
