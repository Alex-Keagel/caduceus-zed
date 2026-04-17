use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Decompose a complex task into a DAG of subtasks with dependencies.
/// Uses the engine's TaskDAG for parallel execution planning.
/// Returns a structured plan that can drive sub-agent spawning.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusTaskDecomposeToolInput {
    /// The operation to perform on the task DAG.
    pub operation: TaskDecomposeOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskDecomposeOperation {
    /// Create a new task DAG from a list of tasks with dependencies.
    Create {
        /// List of tasks to add.
        tasks: Vec<TaskDef>,
    },
    /// Get the list of tasks that are ready to execute (all deps satisfied).
    Ready,
    /// Mark a task as completed with a result.
    Complete {
        /// Task ID to mark complete.
        task_id: String,
        /// Result or summary of the completed task.
        result: String,
    },
    /// Mark a task as failed.
    Fail {
        /// Task ID to mark failed.
        task_id: String,
        /// Error description.
        error: String,
    },
    /// Show the current status of all tasks.
    Status,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TaskDef {
    /// Unique task identifier.
    pub id: String,
    /// Human-readable task title.
    pub title: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// IDs of tasks that must complete before this one.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Optional assignee agent name.
    #[serde(default)]
    pub assignee: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusTaskDecomposeToolOutput {
    Success { result: String },
    Error { error: String },
}

impl From<CaduceusTaskDecomposeToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusTaskDecomposeToolOutput) -> Self {
        match output {
            CaduceusTaskDecomposeToolOutput::Success { result } => result.into(),
            CaduceusTaskDecomposeToolOutput::Error { error } => {
                format!("❌ Task decomposition error: {error}").into()
            }
        }
    }
}

use std::sync::Mutex;
use caduceus_bridge::orchestrator::TaskDAG;

/// Maximum number of tasks per DAG
const MAX_DAG_TASKS: usize = 50;
const MAX_TASK_ID_LEN: usize = 64;
const MAX_TASK_TITLE_LEN: usize = 256;

pub struct CaduceusTaskDecomposeTool {
    dag: Arc<Mutex<TaskDAG>>,
}

impl CaduceusTaskDecomposeTool {
    pub fn new(dag: Arc<Mutex<TaskDAG>>) -> Self {
        Self { dag }
    }
}

impl AgentTool for CaduceusTaskDecomposeTool {
    type Input = CaduceusTaskDecomposeToolInput;
    type Output = CaduceusTaskDecomposeToolOutput;

    const NAME: &'static str = "caduceus_task_decompose";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Execute
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            let op = match &input.operation {
                TaskDecomposeOperation::Create { tasks } => format!("create {} tasks", tasks.len()),
                TaskDecomposeOperation::Ready => "get ready tasks".to_string(),
                TaskDecomposeOperation::Complete { task_id, .. } => format!("complete {task_id}"),
                TaskDecomposeOperation::Fail { task_id, .. } => format!("fail {task_id}"),
                TaskDecomposeOperation::Status => "status".to_string(),
            };
            format!("Task DAG: {op}").into()
        } else {
            "Task DAG".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(|e| {
                CaduceusTaskDecomposeToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let mut dag = self.dag.lock().unwrap_or_else(|e| e.into_inner());

            let result = match input.operation {
                TaskDecomposeOperation::Create { tasks } => {
                    if tasks.len() > MAX_DAG_TASKS {
                        return Err(CaduceusTaskDecomposeToolOutput::Error {
                            error: format!("Too many tasks ({}, max {})", tasks.len(), MAX_DAG_TASKS),
                        });
                    }
                    let prev_incomplete = dag.tasks().values().any(|t| !t.status.is_terminal());
                    *dag = TaskDAG::new();
                    let mut errors = Vec::new();
                    for task in &tasks {
                        if task.id.len() > MAX_TASK_ID_LEN || task.title.len() > MAX_TASK_TITLE_LEN {
                            errors.push(format!("{}: id or title too long", task.id));
                            continue;
                        }
                        let mut td = caduceus_bridge::orchestrator::TaskDefinition::new(
                            &task.id, &task.title,
                        );
                        if let Some(desc) = &task.description {
                            td = td.with_description(desc);
                        }
                        if let Some(assignee) = &task.assignee {
                            td = td.with_assignee(assignee);
                        }
                        if !task.depends_on.is_empty() {
                            td = td.depends_on(task.depends_on.clone());
                        }
                        if let Err(e) = dag.add_task(td) {
                            errors.push(format!("{}: {}", task.id, e));
                        }
                    }
                    if errors.is_empty() {
                        let prefix = if prev_incomplete { "⚠️ Previous DAG had incomplete tasks. " } else { "" };
                        format!("{prefix}✅ Created DAG with {} tasks. Use `ready` to see executable tasks.", tasks.len())
                    } else {
                        format!("⚠️ Created DAG with {} tasks, {} errors:\n{}", 
                            tasks.len(), errors.len(), errors.join("\n"))
                    }
                }
                TaskDecomposeOperation::Ready => {
                    let ready = dag.ready_tasks();
                    if ready.is_empty() {
                        if dag.is_complete() {
                            "✅ All tasks complete!".to_string()
                        } else {
                            "⏳ No tasks ready — waiting for dependencies.".to_string()
                        }
                    } else {
                        let mut out = format!("{} tasks ready to execute:\n", ready.len());
                        for t in &ready {
                            let assignee = t.assignee.as_deref().unwrap_or("unassigned");
                            out.push_str(&format!("  - **{}**: {} [{}]\n", t.id, t.title, assignee));
                        }
                        out
                    }
                }
                TaskDecomposeOperation::Complete { task_id, result } => {
                    match dag.complete_task(&task_id, result) {
                        Ok(()) => format!("✅ Task '{}' completed.", task_id),
                        Err(e) => format!("❌ {}", e),
                    }
                }
                TaskDecomposeOperation::Fail { task_id, error } => {
                    match dag.fail_task(&task_id, error) {
                        Ok(()) => format!("❌ Task '{}' marked as failed.", task_id),
                        Err(e) => format!("❌ {}", e),
                    }
                }
                TaskDecomposeOperation::Status => {
                    let tasks = dag.tasks();
                    if tasks.is_empty() {
                        "No task DAG created. Use `create` to define tasks.".to_string()
                    } else {
                        let mut out = format!("{} tasks in DAG:\n", tasks.len());
                        let mut sorted: Vec<_> = tasks.iter().collect();
                        sorted.sort_by_key(|(id, _)| id.clone());
                        for (id, t) in &sorted {
                            out.push_str(&format!("  [{:?}] {} — {}\n", t.status, id, t.title));
                        }
                        if dag.is_complete() {
                            out.push_str("\n✅ All tasks complete!");
                        }
                        out
                    }
                }
            };

            Ok(CaduceusTaskDecomposeToolOutput::Success { result })
        })
    }
}
