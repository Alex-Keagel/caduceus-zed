use std::sync::{Arc, Mutex};

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Track time spent on tasks: start, complete, velocity, and overdue detection.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusTimeTrackingToolInput {
    /// The time tracking operation to perform.
    pub operation: TimeTrackingOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TimeTrackingOperation {
    /// Start tracking a task with an estimated duration in minutes.
    StartTask {
        /// The task ID (numeric).
        task_id: usize,
        /// Estimated time in minutes.
        estimated_minutes: f64,
    },
    /// Mark a task as completed with actual time spent in minutes.
    CompleteTask {
        /// The task ID to complete.
        task_id: usize,
        /// Actual time spent in minutes.
        actual_minutes: f64,
    },
    /// Get the velocity ratio (estimated / actual) for completed tasks.
    Velocity,
    /// List task IDs that have exceeded their estimated time.
    OverdueTasks,
    /// Get total estimated hours across all tracked tasks.
    TotalEstimated,
    /// Get total actual hours across all tracked tasks.
    TotalActual,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusTimeTrackingToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusTimeTrackingToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusTimeTrackingToolOutput) -> Self {
        match output {
            CaduceusTimeTrackingToolOutput::Text { text } => text.into(),
            CaduceusTimeTrackingToolOutput::Error { error } => {
                format!("Time tracking error: {error}").into()
            }
        }
    }
}

pub struct CaduceusTimeTrackingTool {
    tracker: Arc<Mutex<caduceus_bridge::orchestrator::TimeTracker>>,
}

impl CaduceusTimeTrackingTool {
    pub fn new() -> Self {
        Self {
            tracker: Arc::new(Mutex::new(
                caduceus_bridge::orchestrator::OrchestratorBridge::new_time_tracker(),
            )),
        }
    }
}

impl AgentTool for CaduceusTimeTrackingTool {
    type Input = CaduceusTimeTrackingToolInput;
    type Output = CaduceusTimeTrackingToolOutput;

    const NAME: &'static str = "caduceus_time_tracking";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            let op = match &input.operation {
                TimeTrackingOperation::StartTask { .. } => "start task",
                TimeTrackingOperation::CompleteTask { .. } => "complete task",
                TimeTrackingOperation::Velocity => "velocity",
                TimeTrackingOperation::OverdueTasks => "overdue tasks",
                TimeTrackingOperation::TotalEstimated => "total estimated",
                TimeTrackingOperation::TotalActual => "total actual",
            };
            format!("Time {op}").into()
        } else {
            "Time tracking".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let tracker = self.tracker.clone();
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusTimeTrackingToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let mut guard = tracker
                .lock()
                .map_err(|e| CaduceusTimeTrackingToolOutput::Error {
                    error: format!("Lock poisoned: {e}"),
                })?;

            let text = match input.operation {
                TimeTrackingOperation::StartTask {
                    task_id,
                    estimated_minutes,
                } => {
                    let estimated_hours = estimated_minutes / 60.0;
                    caduceus_bridge::orchestrator::OrchestratorBridge::start_task(
                        &mut guard,
                        task_id,
                        estimated_hours,
                    );
                    format!(
                        "Started tracking task {task_id} (estimated: {estimated_minutes:.0} min)"
                    )
                }
                TimeTrackingOperation::CompleteTask {
                    task_id,
                    actual_minutes,
                } => {
                    let actual_hours = actual_minutes / 60.0;
                    caduceus_bridge::orchestrator::OrchestratorBridge::complete_task(
                        &mut guard,
                        task_id,
                        actual_hours,
                    );
                    format!("Completed task {task_id} (actual: {actual_minutes:.0} min)")
                }
                TimeTrackingOperation::Velocity => {
                    let v = caduceus_bridge::orchestrator::OrchestratorBridge::velocity(&guard);
                    format!("Velocity: {v:.2}")
                }
                TimeTrackingOperation::OverdueTasks => {
                    let overdue =
                        caduceus_bridge::orchestrator::OrchestratorBridge::overdue_tasks(&guard);
                    if overdue.is_empty() {
                        "No overdue tasks".to_string()
                    } else {
                        format!(
                            "Overdue tasks: {}",
                            overdue
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                }
                TimeTrackingOperation::TotalEstimated => {
                    let total =
                        caduceus_bridge::orchestrator::OrchestratorBridge::total_estimated(&guard);
                    format!(
                        "Total estimated: {:.1} hours ({:.0} min)",
                        total,
                        total * 60.0
                    )
                }
                TimeTrackingOperation::TotalActual => {
                    let total =
                        caduceus_bridge::orchestrator::OrchestratorBridge::total_actual(&guard);
                    format!("Total actual: {:.1} hours ({:.0} min)", total, total * 60.0)
                }
            };

            Ok(CaduceusTimeTrackingToolOutput::Text { text })
        })
    }
}
