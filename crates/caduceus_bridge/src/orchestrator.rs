//! Orchestrator bridge — agent harness, conversation history, instructions,
//! PRD parsing, task management, progress inference, scaffolding, and tree visualization.

use caduceus_orchestrator::{
    AgentHarness, ConversationHistory, execute_tool_calls,
    instructions::{self, InstructionLoader, InstructionSet},
    PrdParser, PrdTask, TaskRecommender, TaskRecommendation,
    ProgressInferrer, InferredProgress,
    AgentScaffolder, SkillScaffolder,
    ExecutionTreeViz,
    TimeTracker,
    TaskTree, HierarchicalTask,
};
use caduceus_core::{ModelId, ProviderId, SessionState};
use caduceus_providers::LlmAdapter;
use caduceus_tools::ToolRegistry;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Re-export orchestrator types for consumers.
pub use caduceus_orchestrator::{
    PrdTask as BridgePrdTask,
    TaskRecommendation as BridgeTaskRecommendation,
    InferredProgress as BridgeInferredProgress,
    ExecutionTreeViz as BridgeExecutionTreeViz,
    VizTreeNode as BridgeVizTreeNode,
    TimeTracker as BridgeTimeTracker,
    TimeEntry as BridgeTimeEntry,
    TaskTree as BridgeTaskTree,
    HierarchicalTask as BridgeHierarchicalTask,
    kanban::{KanbanBoard, KanbanCard, KanbanColumn, CardStatus},
    automations::{Automation, AutomationTrigger, AutomationAgentConfig, AutomationRegistry},
    context::{self, ContextZone, ContextSource, ContextAssembler, AssembledContext, estimate_tokens, ContextManager, PinnedContext},
    compaction::{self, CompactMessage, CompactionPipeline, ContextStats, CompactionTrigger},
};

// Re-export types needed by tools.
pub use caduceus_orchestrator::modes::AgentMode as BridgeAgentMode;
pub use caduceus_core::ModelId as BridgeModelId;

/// Wrapper around the AgentHarness for the bridge.
pub struct OrchestratorBridge {
    project_root: PathBuf,
}

impl OrchestratorBridge {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    /// Load workspace instructions from .caduceus/ hierarchy.
    pub fn load_instructions(&self) -> Result<InstructionSet, String> {
        let loader = InstructionLoader::new(&self.project_root);
        loader.load().map_err(|e| e.to_string())
    }

    /// Create a new conversation history.
    pub fn new_history() -> ConversationHistory {
        ConversationHistory::new()
    }

    /// Create a new session state.
    pub fn new_session(&self, provider: &str, model: &str) -> SessionState {
        SessionState::new(
            &self.project_root,
            ProviderId::new(provider),
            ModelId::new(model),
        )
    }

    /// Extract learnable memories from a conversation exchange.
    pub fn extract_memories(user_input: &str, assistant_response: &str) -> Vec<String> {
        caduceus_orchestrator::extract_memories(user_input, assistant_response)
    }

    // ── Parallel tool execution ──────────────────────────────────────────

    /// Execute multiple tool calls in parallel.
    pub async fn execute_tool_calls(
        registry: &ToolRegistry,
        calls: &[(String, String, serde_json::Value)],
    ) -> Vec<(String, String, bool)> {
        execute_tool_calls(registry, calls).await
    }

    // ── Conversation history persistence ─────────────────────────────────

    /// Serialize conversation history to JSON.
    pub fn conversation_serialize(history: &ConversationHistory) -> Result<String, String> {
        history.serialize().map_err(|e| e.to_string())
    }

    /// Deserialize conversation history from JSON.
    pub fn conversation_deserialize(json: &str) -> Result<ConversationHistory, String> {
        ConversationHistory::deserialize(json).map_err(|e| e.to_string())
    }

    /// Truncate conversation history, keeping at most `max` messages.
    pub fn conversation_truncate(history: &mut ConversationHistory, max: usize) {
        history.truncate_oldest(max);
    }

    /// Clear all messages from conversation history.
    pub fn conversation_clear(history: &mut ConversationHistory) {
        history.clear();
    }

    // ── Instruction compaction ───────────────────────────────────────────

    /// Compact instructions to fit within a character budget.
    pub fn compact_instructions(content: &str, max_chars: usize) -> String {
        instructions::compact_instructions(content, max_chars)
    }

    // NOTE: semantic_match_score is private in caduceus-orchestrator::instructions — skipped.

    /// Build a full agent harness with tools and instructions.
    pub fn build_harness(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
    ) -> AgentHarness {
        AgentHarness::new(provider, tools, 200_000, system_prompt)
            .with_tool_timeout(std::time::Duration::from_secs(120))
            .with_instructions(&self.project_root)
    }

    /// Get the project root.
    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    // ── Agent turn execution ─────────────────────────────────────────────

    /// Run a single agent turn (non-streaming).
    pub async fn run_turn(
        harness: &AgentHarness,
        state: &mut SessionState,
        user_input: &str,
    ) -> Result<String, String> {
        harness.run_turn(state, user_input).await.map_err(|e| e.to_string())
    }

    /// Run a single agent turn with streaming.
    pub async fn stream_turn(
        harness: &AgentHarness,
        state: &mut SessionState,
        user_input: &str,
    ) -> Result<String, String> {
        harness.stream_turn(state, user_input).await.map_err(|e| e.to_string())
    }

    // ── PRD parsing ──────────────────────────────────────────────────────

    /// Extract `(heading, content)` pairs from markdown text.
    pub fn extract_sections(text: &str) -> Vec<(String, String)> {
        PrdParser::extract_sections(text)
    }

    /// Parse a markdown PRD document into a list of tasks.
    pub fn parse_prd(text: &str) -> Vec<PrdTask> {
        PrdParser::parse(text)
    }

    /// Infer dependency edges between tasks from keyword references.
    /// Returns `(dependent_id, dependency_id)` pairs.
    pub fn infer_dependencies(tasks: &[PrdTask]) -> Vec<(usize, usize)> {
        PrdParser::infer_dependencies(tasks)
    }

    // ── Task recommendation ──────────────────────────────────────────────

    /// Rank incomplete tasks by readiness, priority, and complexity.
    pub fn recommend_next(tasks: &[PrdTask], completed: &[usize]) -> Vec<TaskRecommendation> {
        TaskRecommender::recommend_next(tasks, completed)
    }

    // ── Progress inference ───────────────────────────────────────────────

    /// Estimate progress from git commit messages referencing a task title.
    pub fn infer_from_commits(task_title: &str, commit_messages: &[String]) -> InferredProgress {
        ProgressInferrer::infer_from_commits(task_title, commit_messages)
    }

    /// Progress from test suite pass rate (0–100).
    pub fn infer_from_tests(total: usize, passing: usize) -> f64 {
        ProgressInferrer::infer_from_tests(total, passing)
    }

    /// Progress from file creation ratio (0–100).
    pub fn infer_from_files(files_planned: usize, files_created: usize) -> f64 {
        ProgressInferrer::infer_from_files(files_planned, files_created)
    }

    /// Weighted average of commit/test/file progress (40/40/20).
    pub fn combined_progress(commits: f64, tests: f64, files: f64) -> f64 {
        ProgressInferrer::combined(commits, tests, files)
    }

    // ── Agent scaffolding ────────────────────────────────────────────────

    /// Suggest trigger phrases for an agent based on its description.
    pub fn suggest_triggers(description: &str) -> Vec<String> {
        AgentScaffolder::suggest_triggers(description)
    }

    /// List available tool set presets for agent scaffolding.
    pub fn available_tool_sets() -> Vec<(&'static str, &'static [&'static str])> {
        AgentScaffolder::available_tool_sets()
    }

    // ── Skill scaffolding ────────────────────────────────────────────────

    /// Extract a skill definition from a conversation history.
    pub fn from_conversation(messages: &[String]) -> String {
        SkillScaffolder::from_conversation(messages)
    }

    // ── Execution tree visualization ─────────────────────────────────────

    /// Render an execution tree as a Mermaid `graph TD` flowchart.
    pub fn to_mermaid(tree: &ExecutionTreeViz) -> String {
        tree.to_mermaid()
    }

    /// Render an execution tree as React Flow nodes + edges JSON.
    pub fn to_react_flow_json(tree: &ExecutionTreeViz) -> serde_json::Value {
        tree.to_react_flow_json()
    }

    /// Map a node status string to a CSS color.
    pub fn node_color(status: &str) -> &'static str {
        ExecutionTreeViz::node_color(status)
    }

    // ── Time tracking ────────────────────────────────────────────────────

    /// Create a new empty time tracker.
    pub fn new_time_tracker() -> TimeTracker {
        TimeTracker::new()
    }

    /// Start tracking a task with an estimated duration.
    pub fn start_task(tracker: &mut TimeTracker, task_id: usize, estimated: f64) {
        tracker.start_task(task_id, estimated);
    }

    /// Mark a task as completed with the actual hours spent.
    pub fn complete_task(tracker: &mut TimeTracker, task_id: usize, actual: f64) {
        tracker.complete_task(task_id, actual);
    }

    /// Velocity ratio (estimated / actual) for completed tasks.
    pub fn velocity(tracker: &TimeTracker) -> f64 {
        tracker.velocity()
    }

    /// Sum of estimated hours across all tracked tasks.
    pub fn total_estimated(tracker: &TimeTracker) -> f64 {
        tracker.total_estimated()
    }

    /// Sum of actual hours across all tracked tasks.
    pub fn total_actual(tracker: &TimeTracker) -> f64 {
        tracker.total_actual()
    }

    /// Task IDs that are still running and have exceeded their estimate.
    pub fn overdue_tasks(tracker: &TimeTracker) -> Vec<usize> {
        tracker.overdue_tasks()
    }

    // ── Hierarchical task tree ───────────────────────────────────────────

    /// Create a new empty task tree.
    pub fn new_task_tree() -> TaskTree {
        TaskTree::new()
    }

    /// Add a task to the tree, optionally under a parent. Returns the new task ID.
    pub fn add_task(tree: &mut TaskTree, title: &str, parent: Option<usize>) -> usize {
        tree.add_task(title, parent)
    }

    /// Get immediate children of a task.
    pub fn children(tree: &TaskTree, id: usize) -> Vec<&HierarchicalTask> {
        tree.children(id)
    }

    /// Get all descendants of a task (depth-first).
    pub fn subtree(tree: &TaskTree, id: usize) -> Vec<&HierarchicalTask> {
        tree.subtree(id)
    }

    /// Render the entire task tree as an indented string.
    pub fn to_tree_string(tree: &TaskTree) -> String {
        tree.to_tree_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caduceus_orchestrator::VizTreeNode;

    #[test]
    fn orchestrator_new_history() {
        let history = OrchestratorBridge::new_history();
        assert!(history.is_empty());
    }

    #[test]
    fn orchestrator_new_session() {
        let bridge = OrchestratorBridge::new(".");
        let state = bridge.new_session("anthropic", "claude-sonnet");
        assert_eq!(state.turn_count, 0);
    }

    #[test]
    fn orchestrator_load_instructions_empty() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let result = bridge.load_instructions();
        assert!(result.is_ok());
        let set = result.unwrap();
        assert!(set.system_prompt.is_empty());
    }

    #[test]
    fn orchestrator_load_instructions_with_caduceus_md() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("CADUCEUS.md"), "# Project\nUse Rust.").unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let set = bridge.load_instructions().unwrap();
        assert!(set.system_prompt.contains("Use Rust"));
    }

    #[test]
    fn orchestrator_extract_memories_preference() {
        let memories = OrchestratorBridge::extract_memories(
            "I prefer async/await over raw futures",
            "Got it, I'll use async/await.",
        );
        assert!(!memories.is_empty(), "Should extract preference");
    }

    #[test]
    fn orchestrator_extract_memories_no_signal() {
        let memories = OrchestratorBridge::extract_memories(
            "What is 2+2?",
            "4.",
        );
        assert!(memories.is_empty(), "No preference signal");
    }

    #[test]
    fn orchestrator_conversation_serialize_deserialize() {
        let history = OrchestratorBridge::new_history();
        let json = OrchestratorBridge::conversation_serialize(&history).unwrap();
        let restored = OrchestratorBridge::conversation_deserialize(&json).unwrap();
        assert_eq!(restored.len(), 0);
    }

    #[test]
    fn orchestrator_conversation_truncate() {
        let mut history = OrchestratorBridge::new_history();
        for _ in 0..10 {
            history.append(caduceus_providers::Message {
                role: "user".to_string(),
                content: String::new(),
                content_blocks: None,
                tool_calls: vec![],
                tool_result: None,
            });
        }
        assert_eq!(history.len(), 10);
        OrchestratorBridge::conversation_truncate(&mut history, 5);
        assert!(history.len() <= 5);
    }

    #[test]
    fn orchestrator_conversation_clear() {
        let mut history = OrchestratorBridge::new_history();
        history.append(caduceus_providers::Message {
            role: "user".to_string(),
            content: String::new(),
            content_blocks: None,
            tool_calls: vec![],
            tool_result: None,
        });
        assert!(!history.is_empty());
        OrchestratorBridge::conversation_clear(&mut history);
        assert!(history.is_empty());
    }

    #[test]
    fn orchestrator_compact_instructions_passthrough() {
        let short = "Use Rust.";
        let result = OrchestratorBridge::compact_instructions(short, 1000);
        assert_eq!(result, short);
    }

    #[test]
    fn orchestrator_compact_instructions_truncates() {
        let long = "# Rules\n- MUST use Rust\n- NEVER use C++\n\n```\nsome code block\n```\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(20);
        let result = OrchestratorBridge::compact_instructions(&long, 200);
        assert!(result.len() < long.len(), "Should be shorter than original");
    }

    #[test]
    fn orchestrator_conversation_deserialize_invalid() {
        let result = OrchestratorBridge::conversation_deserialize("not json");
        assert!(result.is_err());
    }

    // ── PRD parsing tests ────────────────────────────────────────────────

    #[test]
    fn orchestrator_extract_sections() {
        let md = "# Auth\nBuild login.\n## OAuth\nUse OAuth2.";
        let sections = OrchestratorBridge::extract_sections(md);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].0, "Auth");
        assert!(sections[0].1.contains("Build login"));
    }

    #[test]
    fn orchestrator_parse_prd() {
        let prd = "# Task A\nDo A.\n# Task B\nDo B.";
        let tasks = OrchestratorBridge::parse_prd(prd);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].title, "Task A");
        assert_eq!(tasks[1].title, "Task B");
    }

    #[test]
    fn orchestrator_infer_dependencies() {
        let tasks = OrchestratorBridge::parse_prd(
            "# Auth\nBuild auth.\n# API\nRequires Auth module.",
        );
        let deps = OrchestratorBridge::infer_dependencies(&tasks);
        // "API" description mentions "Auth", so (1, 0) expected
        assert!(!deps.is_empty());
        assert!(deps.contains(&(1, 0)));
    }

    // ── Task recommendation tests ────────────────────────────────────────

    #[test]
    fn orchestrator_recommend_next() {
        let tasks = OrchestratorBridge::parse_prd(
            "# Setup\nInit project.\n# Build\nBuild features.",
        );
        let recs = OrchestratorBridge::recommend_next(&tasks, &[]);
        assert_eq!(recs.len(), 2);
        assert!(recs[0].score >= recs[1].score);
    }

    #[test]
    fn orchestrator_recommend_next_with_completed() {
        let tasks = OrchestratorBridge::parse_prd("# A\nDo A.\n# B\nDo B.");
        let recs = OrchestratorBridge::recommend_next(&tasks, &[0]);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].task_id, 1);
    }

    // ── Progress inference tests ─────────────────────────────────────────

    #[test]
    fn orchestrator_infer_from_commits() {
        let commits = vec![
            "implement auth login".to_string(),
            "fix auth bug".to_string(),
        ];
        let progress = OrchestratorBridge::infer_from_commits("auth", &commits);
        assert!(progress.confidence > 0.0);
        assert!(!progress.evidence.is_empty());
    }

    #[test]
    fn orchestrator_infer_from_commits_empty() {
        let progress = OrchestratorBridge::infer_from_commits("auth", &[]);
        assert_eq!(progress.percentage, 0.0);
        assert_eq!(progress.confidence, 0.0);
    }

    #[test]
    fn orchestrator_infer_from_tests() {
        assert_eq!(OrchestratorBridge::infer_from_tests(10, 10), 100.0);
        assert_eq!(OrchestratorBridge::infer_from_tests(10, 5), 50.0);
        assert_eq!(OrchestratorBridge::infer_from_tests(0, 0), 0.0);
    }

    #[test]
    fn orchestrator_infer_from_files() {
        assert_eq!(OrchestratorBridge::infer_from_files(4, 2), 50.0);
        assert_eq!(OrchestratorBridge::infer_from_files(4, 4), 100.0);
        assert_eq!(OrchestratorBridge::infer_from_files(0, 5), 0.0);
    }

    #[test]
    fn orchestrator_combined_progress() {
        let result = OrchestratorBridge::combined_progress(100.0, 100.0, 100.0);
        assert_eq!(result, 100.0);
        let partial = OrchestratorBridge::combined_progress(50.0, 50.0, 50.0);
        assert!((partial - 50.0).abs() < 0.01);
    }

    // ── Agent scaffolding tests ──────────────────────────────────────────

    #[test]
    fn orchestrator_suggest_triggers() {
        let triggers = OrchestratorBridge::suggest_triggers("code review automation");
        assert!(!triggers.is_empty());
        assert!(triggers.iter().any(|t| t.contains("review")));
    }

    #[test]
    fn orchestrator_suggest_triggers_fallback() {
        let triggers = OrchestratorBridge::suggest_triggers("something unique");
        assert!(!triggers.is_empty());
    }

    #[test]
    fn orchestrator_available_tool_sets() {
        let sets = OrchestratorBridge::available_tool_sets();
        assert!(sets.len() >= 3);
        let names: Vec<&str> = sets.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"standard"));
        assert!(names.contains(&"read-only"));
    }

    // ── Skill scaffolding tests ──────────────────────────────────────────

    #[test]
    fn orchestrator_from_conversation() {
        let messages = vec![
            "First, run the linter.".to_string(),
            "Then fix the errors.".to_string(),
            "Finally, commit the changes.".to_string(),
        ];
        let skill = OrchestratorBridge::from_conversation(&messages);
        assert!(!skill.is_empty());
    }

    // ── Execution tree visualization tests ───────────────────────────────

    #[test]
    fn orchestrator_to_mermaid() {
        let mut tree = ExecutionTreeViz::new();
        tree.add_node(VizTreeNode {
            id: "root".to_string(),
            label: "Root Task".to_string(),
            status: "succeeded".to_string(),
            parent: None,
            error: None,
            depth: 0,
        });
        tree.add_node(VizTreeNode {
            id: "child".to_string(),
            label: "Child Task".to_string(),
            status: "active".to_string(),
            parent: Some("root".to_string()),
            error: None,
            depth: 1,
        });
        let mermaid = OrchestratorBridge::to_mermaid(&tree);
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("root"));
        assert!(mermaid.contains("child"));
    }

    #[test]
    fn orchestrator_to_react_flow_json() {
        let mut tree = ExecutionTreeViz::new();
        tree.add_node(VizTreeNode {
            id: "n1".to_string(),
            label: "Node 1".to_string(),
            status: "succeeded".to_string(),
            parent: None,
            error: None,
            depth: 0,
        });
        let json = OrchestratorBridge::to_react_flow_json(&tree);
        assert!(json.get("nodes").is_some());
        assert!(json.get("edges").is_some());
    }

    #[test]
    fn orchestrator_node_color() {
        assert_eq!(OrchestratorBridge::node_color("succeeded"), "#10b981");
        assert_eq!(OrchestratorBridge::node_color("failed"), "#ef4444");
        assert_eq!(OrchestratorBridge::node_color("active"), "#f59e0b");
        assert_eq!(OrchestratorBridge::node_color("pruned"), "#6b7280");
        assert_eq!(OrchestratorBridge::node_color("unknown"), "#6b7280");
    }

    // ── Time tracker tests ───────────────────────────────────────────────

    #[test]
    fn orchestrator_time_tracker_lifecycle() {
        let mut tracker = OrchestratorBridge::new_time_tracker();

        OrchestratorBridge::start_task(&mut tracker, 1, 2.0);
        OrchestratorBridge::start_task(&mut tracker, 2, 3.0);

        assert_eq!(OrchestratorBridge::total_estimated(&tracker), 5.0);
        assert_eq!(OrchestratorBridge::total_actual(&tracker), 0.0);

        OrchestratorBridge::complete_task(&mut tracker, 1, 1.5);
        assert!(OrchestratorBridge::total_actual(&tracker) > 0.0);

        let vel = OrchestratorBridge::velocity(&tracker);
        assert!(vel > 0.0);
    }

    #[test]
    fn orchestrator_time_tracker_velocity_default() {
        let tracker = OrchestratorBridge::new_time_tracker();
        assert_eq!(OrchestratorBridge::velocity(&tracker), 1.0);
    }

    #[test]
    fn orchestrator_overdue_tasks_empty() {
        let tracker = OrchestratorBridge::new_time_tracker();
        assert!(OrchestratorBridge::overdue_tasks(&tracker).is_empty());
    }

    // ── Hierarchical task tree tests ─────────────────────────────────────

    #[test]
    fn orchestrator_task_tree_add_and_children() {
        let mut tree = OrchestratorBridge::new_task_tree();
        let root = OrchestratorBridge::add_task(&mut tree, "Root", None);
        let child1 = OrchestratorBridge::add_task(&mut tree, "Child 1", Some(root));
        let child2 = OrchestratorBridge::add_task(&mut tree, "Child 2", Some(root));

        let children = OrchestratorBridge::children(&tree, root);
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].id, child1);
        assert_eq!(children[1].id, child2);
    }

    #[test]
    fn orchestrator_task_tree_subtree() {
        let mut tree = OrchestratorBridge::new_task_tree();
        let root = OrchestratorBridge::add_task(&mut tree, "Root", None);
        let child = OrchestratorBridge::add_task(&mut tree, "Child", Some(root));
        let _grandchild = OrchestratorBridge::add_task(&mut tree, "Grandchild", Some(child));

        let sub = OrchestratorBridge::subtree(&tree, root);
        assert_eq!(sub.len(), 2); // child + grandchild
    }

    #[test]
    fn orchestrator_task_tree_to_string() {
        let mut tree = OrchestratorBridge::new_task_tree();
        let root = OrchestratorBridge::add_task(&mut tree, "Root", None);
        OrchestratorBridge::add_task(&mut tree, "Child", Some(root));

        let output = OrchestratorBridge::to_tree_string(&tree);
        assert!(output.contains("Root"));
        assert!(output.contains("Child"));
    }
}
