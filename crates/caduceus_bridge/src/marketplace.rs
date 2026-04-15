//! Marketplace bridge — skill evolution, quality scoring, versioning, registry browsing.

use caduceus_marketplace::{
    EvolvedSkill, EvolverConfig, GarbageCandidate, McpRegistryBrowser, McpRegistryEntry,
    MemoryGarbageCollector, PatternAggregator, PatternEntry, SessionSummary, SkillAutoGenerator,
    SkillEvolver, SkillQualityScorer, SkillScoreInput, SkillVersionManager,
};

/// Unified bridge into caduceus-marketplace subsystems.
pub struct MarketplaceBridge {
    pub evolver: SkillEvolver,
    pub aggregator: PatternAggregator,
    pub scorer: SkillQualityScorer,
    pub versions: SkillVersionManager,
    pub gc: MemoryGarbageCollector,
    pub browser: McpRegistryBrowser,
}

impl MarketplaceBridge {
    pub fn new() -> Self {
        Self {
            evolver: SkillEvolver::new(EvolverConfig {
                min_sessions_before_evolve: 3,
                evolution_interval_secs: 3600,
                quality_threshold: 0.3,
            }),
            aggregator: PatternAggregator::new(2),
            scorer: SkillQualityScorer::new(),
            versions: SkillVersionManager::new(),
            gc: MemoryGarbageCollector::new(90, 1000),
            browser: McpRegistryBrowser::new(),
        }
    }

    // ── Registry browser ─────────────────────────────────────────────────

    /// Search marketplace entries by query.
    pub fn search(&self, query: &str) -> Vec<&McpRegistryEntry> {
        self.browser.search(query)
    }

    /// Add an entry to the registry browser.
    pub fn add_entry(&mut self, entry: McpRegistryEntry) {
        self.browser.add_entry(entry);
    }

    /// Filter entries by tag.
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&McpRegistryEntry> {
        self.browser.filter_by_tag(tag)
    }

    /// Top N entries by download count.
    pub fn top_downloaded(&self, n: usize) -> Vec<&McpRegistryEntry> {
        self.browser.top_downloaded(n)
    }

    // ── Quality scoring ──────────────────────────────────────────────────

    /// Score a skill's quality (0.0 – 1.0).
    pub fn score(&self, input: &SkillScoreInput) -> f64 {
        self.scorer.score(input)
    }

    /// Whether a skill should be promoted (score ≥ 0.7).
    pub fn should_promote(&self, score: f64) -> bool {
        self.scorer.should_promote(score)
    }

    /// Whether a skill should be deprecated (score < 0.3).
    pub fn should_deprecate(&self, score: f64) -> bool {
        self.scorer.should_deprecate(score)
    }

    // ── Skill auto-generation ────────────────────────────────────────────

    /// Generate a SKILL.md file from name, description, patterns and tools.
    pub fn generate_skill_md(
        name: &str,
        description: &str,
        patterns: &[String],
        tools: &[String],
    ) -> String {
        SkillAutoGenerator::generate_skill_md(name, description, patterns, tools)
    }

    /// Suggest a kebab-case skill name from a set of patterns.
    pub fn suggest_skill_name(patterns: &[String]) -> String {
        SkillAutoGenerator::suggest_skill_name(patterns)
    }

    // ── Versioning ───────────────────────────────────────────────────────

    /// Record a new version of a skill.
    pub fn record_version(&mut self, name: &str, content: &str, summary: &str) {
        self.versions.record_version(name, content, summary);
    }

    /// Diff two versions of a skill.
    pub fn diff_versions(&self, name: &str, v1: u32, v2: u32) -> Option<String> {
        self.versions.diff_versions(name, v1, v2)
    }

    /// Version history for a skill.
    pub fn history(&self, name: &str) -> Option<&[caduceus_marketplace::SkillVersion]> {
        self.versions.history(name)
    }

    // ── Pattern aggregation ──────────────────────────────────────────────

    /// Ingest session patterns into the aggregator.
    pub fn ingest_session(&mut self, session_id: &str, patterns: &[String]) {
        self.aggregator.ingest_session(session_id, patterns);
    }

    /// Top N recurring patterns.
    pub fn top_patterns(&self, n: usize) -> Vec<&PatternEntry> {
        self.aggregator.top_patterns(n)
    }

    // ── Garbage collection ───────────────────────────────────────────────

    /// Identify low-quality / stale memory entries for cleanup.
    pub fn identify_garbage(&self, items: &[GarbageCandidate]) -> Vec<String> {
        self.gc.identify_garbage(items)
    }

    // ── Skill evolution ──────────────────────────────────────────────────

    /// Evolve new skills from session summaries.
    pub fn evolve_from_summaries(&mut self, summaries: &[SessionSummary]) -> Vec<EvolvedSkill> {
        self.evolver.evolve_from_summaries(summaries)
    }
}

impl Default for MarketplaceBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn marketplace_search() {
        let mut bridge = MarketplaceBridge::new();
        bridge.add_entry(McpRegistryEntry {
            name: "sql-helper".into(),
            description: "Run SQL queries".into(),
            author: "test".into(),
            downloads: 100,
            version: "1.0.0".into(),
            tags: vec!["database".into()],
            verified: true,
        });
        bridge.add_entry(McpRegistryEntry {
            name: "git-ops".into(),
            description: "Git operations".into(),
            author: "test".into(),
            downloads: 200,
            version: "2.0.0".into(),
            tags: vec!["vcs".into()],
            verified: false,
        });
        let results = bridge.search("sql");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "sql-helper");
    }

    #[test]
    fn marketplace_score_and_lifecycle() {
        let bridge = MarketplaceBridge::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let input = SkillScoreInput {
            total_uses: 100,
            successful_uses: 90,
            user_ratings: vec![0.9, 0.8, 0.95],
            last_used_epoch: now,
            days_since_creation: 30,
        };
        let score = bridge.score(&input);
        assert!(score > 0.5, "High-quality skill should score well: {score}");
        assert!(bridge.should_promote(score));
        assert!(!bridge.should_deprecate(score));

        // Low-quality
        let low = SkillScoreInput {
            total_uses: 10,
            successful_uses: 1,
            user_ratings: vec![0.1],
            last_used_epoch: now - 365 * 86400,
            days_since_creation: 365,
        };
        let low_score = bridge.score(&low);
        assert!(bridge.should_deprecate(low_score));
    }

    #[test]
    fn marketplace_generate_skill_md() {
        let md = MarketplaceBridge::generate_skill_md(
            "my-skill",
            "Does things",
            &["pattern-a".into()],
            &["bash".into()],
        );
        assert!(md.contains("my-skill"));
        assert!(md.contains("Does things"));
        assert!(md.contains("pattern-a"));
    }

    #[test]
    fn marketplace_suggest_name() {
        let name =
            MarketplaceBridge::suggest_skill_name(&["Code Review Helper".into()]);
        assert_eq!(name, "code-review-helper");
    }

    #[test]
    fn marketplace_versioning_and_diff() {
        let mut bridge = MarketplaceBridge::new();
        bridge.record_version("my-skill", "v1 content", "initial");
        bridge.record_version("my-skill", "v2 content longer", "update");
        let hist = bridge.history("my-skill").unwrap();
        assert_eq!(hist.len(), 2);
        let diff = bridge.diff_versions("my-skill", 1, 2).unwrap();
        assert!(diff.contains("content differs"));
    }

    #[test]
    fn marketplace_top_downloaded_and_filter() {
        let mut bridge = MarketplaceBridge::new();
        bridge.add_entry(McpRegistryEntry {
            name: "a".into(),
            description: "".into(),
            author: "".into(),
            downloads: 50,
            version: "1".into(),
            tags: vec!["db".into()],
            verified: false,
        });
        bridge.add_entry(McpRegistryEntry {
            name: "b".into(),
            description: "".into(),
            author: "".into(),
            downloads: 200,
            version: "1".into(),
            tags: vec!["ai".into()],
            verified: true,
        });
        let top = bridge.top_downloaded(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].name, "b");

        let filtered = bridge.filter_by_tag("db");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "a");
    }

    #[test]
    fn marketplace_identify_garbage() {
        let bridge = MarketplaceBridge::new();
        let items = vec![
            GarbageCandidate {
                id: "old".into(),
                last_accessed_days_ago: 200,
                access_count: 1,
                size_bytes: 100,
            },
            GarbageCandidate {
                id: "recent".into(),
                last_accessed_days_ago: 5,
                access_count: 50,
                size_bytes: 100,
            },
        ];
        let garbage = bridge.identify_garbage(&items);
        assert!(garbage.contains(&"old".to_string()));
        assert!(!garbage.contains(&"recent".to_string()));
    }

    #[test]
    fn marketplace_evolve_from_summaries() {
        let mut bridge = MarketplaceBridge::new();
        let summaries: Vec<SessionSummary> = (0..5)
            .map(|i| SessionSummary {
                session_id: format!("s{i}"),
                patterns: vec!["rust-refactor".into()],
                tools_used: vec!["bash".into()],
                success: true,
            })
            .collect();
        let evolved = bridge.evolve_from_summaries(&summaries);
        assert!(!evolved.is_empty(), "Should evolve a skill from 5 sessions");
        assert_eq!(evolved[0].name, "rust-refactor");
    }

    #[test]
    fn marketplace_top_patterns() {
        let mut bridge = MarketplaceBridge::new();
        bridge.ingest_session("s1", &["debug".into(), "test".into()]);
        bridge.ingest_session("s2", &["debug".into()]);
        bridge.ingest_session("s3", &["debug".into(), "deploy".into()]);
        let top = bridge.top_patterns(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].pattern, "debug");
    }
}
