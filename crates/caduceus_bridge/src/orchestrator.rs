//! Orchestrator bridge — agent harness, conversation history, instructions.

use caduceus_orchestrator::{
    AgentHarness, ConversationHistory, execute_tool_calls,
    instructions::{self, InstructionLoader, InstructionSet},
};
use caduceus_core::{ModelId, ProviderId, SessionState};
use caduceus_providers::LlmAdapter;
use caduceus_tools::ToolRegistry;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
