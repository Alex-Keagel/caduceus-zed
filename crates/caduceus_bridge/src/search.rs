//! Search types — semantic search + code graph + error analysis + project info.

#[derive(Debug, Clone)]
pub struct GraphNodeInfo {
    pub id: String,
    pub label: String,
    pub file: String,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct ErrorAnalysisResult {
    pub category: String,
    pub root_cause: String,
    pub severity: String,
    pub affected_files: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProjectInfo {
    pub languages: Vec<String>,
    pub file_count: u32,
    pub root: String,
}
