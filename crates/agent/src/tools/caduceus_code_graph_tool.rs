use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Explores the semantic code graph: find neighbors, affected files, and subgraphs
/// for a given code symbol. Use this to understand code relationships and impact analysis.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusCodeGraphToolInput {
    /// The code operation to perform.
    pub operation: CodeGraphOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CodeGraphOperation {
    /// Find direct neighbors of a code node (functions that call/are called by it)
    Neighbors { node_id: String },
    /// Find all code affected by changes to this node
    AffectedBy { node_id: String },
    /// Get the subgraph around a node (transitive dependencies)
    Subgraph { node_id: String },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusCodeGraphToolOutput {
    Success { nodes: Vec<GraphNode> },
    Error { error: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: String,
    pub file: String,
}

impl From<CaduceusCodeGraphToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusCodeGraphToolOutput) -> Self {
        match output {
            CaduceusCodeGraphToolOutput::Success { nodes } => {
                if nodes.is_empty() {
                    "No nodes found in the code graph for this identifier.".into()
                } else {
                    let mut text = format!("Found {} related code nodes:\n", nodes.len());
                    for n in &nodes {
                        text.push_str(&format!("- {} ({}) in {}\n", n.id, n.kind, n.file));
                    }
                    text.into()
                }
            }
            CaduceusCodeGraphToolOutput::Error { error } => {
                format!("Code graph error: {error}").into()
            }
        }
    }
}

pub struct CaduceusCodeGraphTool {
    engine: Arc<caduceus_bridge::engine::CaduceusEngine>,
}

impl CaduceusCodeGraphTool {
    pub fn new(engine: Arc<caduceus_bridge::engine::CaduceusEngine>) -> Self {
        Self { engine }
    }
}

impl AgentTool for CaduceusCodeGraphTool {
    type Input = CaduceusCodeGraphToolInput;
    type Output = CaduceusCodeGraphToolOutput;

    const NAME: &'static str = "caduceus_code_graph";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Search
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            match &input.operation {
                CodeGraphOperation::Neighbors { node_id } => format!("Graph neighbors: {node_id}").into(),
                CodeGraphOperation::AffectedBy { node_id } => format!("Affected by: {node_id}").into(),
                CodeGraphOperation::Subgraph { node_id } => format!("Subgraph: {node_id}").into(),
            }
        } else {
            "Code graph".into()
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
            let input = input.recv().await.map_err(|e| {
                CaduceusCodeGraphToolOutput::Error { error: format!("Failed to receive input: {e}") }
            })?;

            let nodes = match input.operation {
                CodeGraphOperation::Neighbors { node_id } => engine.code_neighbors(&node_id),
                CodeGraphOperation::AffectedBy { node_id } => engine.code_affected_by(&node_id),
                CodeGraphOperation::Subgraph { node_id } => engine.code_subgraph(&node_id),
            };

            let result: Vec<GraphNode> = nodes
                .into_iter()
                .map(|n| GraphNode {
                    id: n.id,
                    kind: n.label,
                    file: n.file,
                })
                .collect();
            Ok(CaduceusCodeGraphToolOutput::Success { nodes: result })
        })
    }
}
