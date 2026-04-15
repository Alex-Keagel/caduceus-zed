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

#[derive(Debug, Clone)]
pub struct ChunkInfo {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub content_length: usize,
}

#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct AlternativeInfo {
    pub description: String,
    pub confidence: f64,
    pub estimated_effort: String,
}

#[derive(Debug, Clone)]
pub struct GraphStatsInfo {
    pub node_count: usize,
    pub edge_count: usize,
    pub components: usize,
}

#[derive(Debug, Clone)]
pub struct FederatedSearchResult {
    pub project: String,
    pub symbols: Vec<SymbolInfo>,
}

#[derive(Debug, Clone)]
pub struct SpacePointInfo {
    pub id: String,
    pub label: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub file: String,
    pub relevance: f64,
}

#[derive(Debug, Clone)]
pub struct HighlightInfo {
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub start_col: usize,
    pub end_col: usize,
    pub highlight_type: String,
    pub label: String,
    pub tooltip: String,
}

#[derive(Debug, Clone)]
pub struct SnapshotInfo {
    pub id: String,
    pub instance_id: String,
    pub timestamp: u64,
}
