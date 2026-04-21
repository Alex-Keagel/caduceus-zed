use std::sync::{Arc, Mutex};

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Query telemetry data: token usage, costs, budget, drift detection, and full reports.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusTelemetryToolInput {
    /// The telemetry operation to perform.
    pub operation: TelemetryOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryOperation {
    /// Get current session token usage (input/output/cached tokens).
    SessionUsage,
    /// Get total token usage across all sessions.
    TotalUsage,
    /// Get total cost across all logged events.
    TotalCost,
    /// Get remaining budget in USD.
    BudgetRemaining,
    /// Check if behavioral drift exceeds a threshold (default 0.5).
    IsDrifting,
    /// Generate a full telemetry report with usage, budget, SLOs, drift, and degradation.
    ExportReport,
    /// Configure OTLP endpoint and trigger a batch export.
    ExportOtlp {
        /// The OTLP HTTP endpoint, e.g. "http://localhost:4318"
        endpoint: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusTelemetryToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusTelemetryToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusTelemetryToolOutput) -> Self {
        match output {
            CaduceusTelemetryToolOutput::Text { text } => text.into(),
            CaduceusTelemetryToolOutput::Error { error } => {
                format!("Telemetry error: {error}").into()
            }
        }
    }
}

pub struct CaduceusTelemetryTool {
    bridge: Arc<Mutex<caduceus_bridge::telemetry::TelemetryBridge>>,
}

impl CaduceusTelemetryTool {
    pub fn new() -> Self {
        Self {
            bridge: Arc::new(Mutex::new(
                caduceus_bridge::telemetry::TelemetryBridge::new(),
            )),
        }
    }
}

impl AgentTool for CaduceusTelemetryTool {
    type Input = CaduceusTelemetryToolInput;
    type Output = CaduceusTelemetryToolOutput;

    const NAME: &'static str = "caduceus_telemetry";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            let op = match &input.operation {
                TelemetryOperation::SessionUsage => "session usage",
                TelemetryOperation::TotalUsage => "total usage",
                TelemetryOperation::TotalCost => "total cost",
                TelemetryOperation::BudgetRemaining => "budget remaining",
                TelemetryOperation::IsDrifting => "drift check",
                TelemetryOperation::ExportReport => "report",
                TelemetryOperation::ExportOtlp { .. } => "OTLP export",
            };
            format!("Telemetry {op}").into()
        } else {
            "Telemetry query".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let bridge = self.bridge.clone();
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusTelemetryToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let guard = bridge
                .lock()
                .map_err(|e| CaduceusTelemetryToolOutput::Error {
                    error: format!("Lock poisoned: {e}"),
                })?;

            let text = match input.operation {
                TelemetryOperation::SessionUsage => {
                    let u = guard.session_usage();
                    format!(
                        "Input tokens: {}, Output tokens: {}, Cached tokens: {}",
                        u.input_tokens, u.output_tokens, u.cached_tokens
                    )
                }
                TelemetryOperation::TotalUsage => {
                    let u = guard.total_usage();
                    format!(
                        "Input tokens: {}, Output tokens: {}, Cached tokens: {}",
                        u.input_tokens, u.output_tokens, u.cached_tokens
                    )
                }
                TelemetryOperation::TotalCost => {
                    format!("${:.6}", guard.total_cost())
                }
                TelemetryOperation::BudgetRemaining => {
                    format!("${:.4}", guard.budget_remaining())
                }
                TelemetryOperation::IsDrifting => {
                    let drifting = guard.is_drifting(0.5);
                    let score = guard.drift_score();
                    format!("Drifting: {drifting} (score: {score:.4})")
                }
                TelemetryOperation::ExportReport => guard.generate_report(),
                TelemetryOperation::ExportOtlp { endpoint } => {
                    drop(guard);
                    let mut guard =
                        bridge
                            .lock()
                            .map_err(|e| CaduceusTelemetryToolOutput::Error {
                                error: format!("Lock poisoned: {e}"),
                            })?;
                    guard.configure_otlp_endpoint(&endpoint);
                    format!("OTLP exporter configured for {endpoint}")
                }
            };

            Ok(CaduceusTelemetryToolOutput::Text { text })
        })
    }
}
