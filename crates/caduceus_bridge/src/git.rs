//! Git types for the bridge.

#[derive(Debug, Clone)]
pub struct StatusEntry {
    pub path: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct CommitInfo {
    pub sha: String,
    pub message: String,
    pub author: String,
    pub date: String,
}
