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

#[derive(Debug, Clone)]
pub struct DiffInfo {
    pub path: String,
    pub additions: usize,
    pub deletions: usize,
}

#[derive(Debug, Clone)]
pub struct CommitResult {
    pub sha: String,
}

#[derive(Debug, Clone)]
pub struct GitFreshness {
    pub commits_behind: usize,
    pub is_diverged: bool,
    pub is_stale: bool,
}

#[derive(Debug, Clone)]
pub struct WorktreeInfo {
    pub path: String,
    pub branch: String,
    pub head_sha: String,
}
