//! Telemetry bridge — token counting, cost calculation, usage tracking,
//! SLO monitoring, budget enforcement, drift detection, and degradation breaker.

use caduceus_core::{ModelId, ProviderId};
use caduceus_telemetry::{
    BehavioralDriftDetector, BudgetEnforcer, CognitiveDegradationBreaker, CostCalculator,
    CostLogger, CostRecord, DegradationIndicators, DegradationStage, ModelPricing, OtelExporter,
    OtelExporterConfig, Slo, SloMetric, SloMonitor, SloStatus, TelemetryEvent, TokenCounter,
};

/// Re-export the telemetry TokenUsage (different from core::TokenUsage)
pub use caduceus_telemetry::TokenUsage as TelemetryTokenUsage;

/// Wrapper around telemetry subsystems.
pub struct TelemetryBridge {
    pub counter: TokenCounter,
    pub calculator: CostCalculator,
    pub logger: CostLogger,
    pub slo_monitor: SloMonitor,
    pub budget: BudgetEnforcer,
    pub drift_detector: BehavioralDriftDetector,
    pub degradation_breaker: CognitiveDegradationBreaker,
    pub exporter: OtelExporter,
}

impl TelemetryBridge {
    pub fn new() -> Self {
        Self {
            counter: TokenCounter::new(),
            calculator: CostCalculator::new(),
            logger: CostLogger::new(),
            slo_monitor: SloMonitor::new(),
            budget: BudgetEnforcer::default(),
            drift_detector: BehavioralDriftDetector::new(),
            degradation_breaker: CognitiveDegradationBreaker::new(),
            exporter: OtelExporter::from_env(),
        }
    }

    /// Record token usage for a model.
    pub fn record_usage(&mut self, model: &str, usage: &TelemetryTokenUsage) {
        self.counter.record_for_model(&ModelId::new(model), usage);
    }

    /// Get session token usage.
    pub fn session_usage(&self) -> &TelemetryTokenUsage {
        self.counter.session_usage()
    }

    /// Get total token usage across all sessions.
    pub fn total_usage(&self) -> &TelemetryTokenUsage {
        self.counter.total_usage()
    }

    /// Get usage breakdown by model.
    pub fn model_usage(&self, model: &str) -> Option<&TelemetryTokenUsage> {
        self.counter.model_usage(model)
    }

    /// Get all model usage.
    pub fn all_model_usage(&self) -> &std::collections::HashMap<String, TelemetryTokenUsage> {
        self.counter.all_model_usage()
    }

    /// Calculate cost for a specific request.
    pub fn calculate_cost(&self, provider: &str, model: &str, usage: &TelemetryTokenUsage) -> f64 {
        self.calculator
            .calculate(&ProviderId::new(provider), &ModelId::new(model), usage)
    }

    /// Log a cost event.
    pub fn log_cost(&mut self, provider: &str, model: &str, usage: TelemetryTokenUsage) {
        self.logger
            .log(&ProviderId::new(provider), &ModelId::new(model), usage);
    }

    /// Get total cost across all logged events.
    pub fn total_cost(&self) -> f64 {
        self.logger.total_cost()
    }

    /// Get cost record count.
    pub fn cost_record_count(&self) -> usize {
        self.logger.records().len()
    }

    /// Reset session counters.
    pub fn reset_session(&mut self) {
        self.counter.reset_session();
    }

    // ── New methods ──────────────────────────────────────────────────────

    /// Add custom model pricing to the calculator.
    pub fn add_pricing(&mut self, pricing: ModelPricing) {
        self.calculator.add_pricing(pricing);
    }

    /// Cost breakdown for a specific model (sum of all records for that model).
    pub fn cost_for_model(&self, model: &str) -> f64 {
        self.logger
            .records_for_model(model)
            .iter()
            .map(|r| r.cost_usd)
            .sum()
    }

    /// All cost records filtered to a single model.
    pub fn records_for_model(&self, model: &str) -> Vec<&CostRecord> {
        self.logger.records_for_model(model)
    }

    /// Export full telemetry state as JSON.
    pub fn export_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self.logger.records()).map_err(|e| e.to_string())
    }

    // ── SLO tracking ─────────────────────────────────────────────────────

    /// Add a Service Level Objective.
    pub fn add_slo(&mut self, name: &str, target: f64) {
        self.slo_monitor.add_slo(Slo {
            name: name.to_string(),
            description: format!("SLO: {name}"),
            target,
            window_secs: 3600,
            metric: SloMetric::SuccessRate,
            measurements: Vec::new(),
        });
    }

    /// Check an SLO by name.
    pub fn check_slo(&self, name: &str) -> Option<SloStatus> {
        self.slo_monitor.check_slo(name)
    }

    /// Get all SLO statuses.
    pub fn all_slo_statuses(&self) -> Vec<SloStatus> {
        self.slo_monitor.all_statuses()
    }

    /// Record an SLO measurement.
    pub fn record_slo_measurement(&mut self, name: &str, value: f64) {
        self.slo_monitor.record_measurement(name, value);
    }

    // ── Budget enforcement ───────────────────────────────────────────────

    /// Check cost against budget and record if within limit.
    pub fn check_and_record(&mut self, cost: f64) -> Result<(), String> {
        self.budget.check_and_record(cost).map_err(|e| {
            format!(
                "Budget exceeded: spent ${:.4}, limit ${:.4}",
                e.spent_usd, e.limit_usd
            )
        })
    }

    /// Set budget limit.
    pub fn set_budget_limit(&mut self, max_usd: f64) {
        self.budget.set_limit(max_usd);
    }

    /// Get remaining budget.
    pub fn budget_remaining(&self) -> f64 {
        self.budget.remaining()
    }

    // ── Telemetry report ─────────────────────────────────────────────────

    /// Generate a full telemetry report string.
    pub fn generate_report(&self) -> String {
        let mut report = String::from("# Telemetry Report\n\n");
        report.push_str("## Token Usage\n");
        report.push_str(&format!(
            "- Input tokens:  {}\n",
            self.counter.total_usage().input_tokens
        ));
        report.push_str(&format!(
            "- Output tokens: {}\n",
            self.counter.total_usage().output_tokens
        ));
        report.push_str(&format!(
            "- Total cost:    ${:.6}\n\n",
            self.logger.total_cost()
        ));
        report.push_str("## Budget\n");
        report.push_str(&format!("- Limit:     ${:.4}\n", self.budget.limit()));
        report.push_str(&format!("- Spent:     ${:.4}\n", self.budget.spent()));
        report.push_str(&format!("- Remaining: ${:.4}\n\n", self.budget.remaining()));
        report.push_str("## SLOs\n");
        for status in self.slo_monitor.all_statuses() {
            report.push_str(&format!(
                "- {} — target: {:.2}, current: {:.2}, met: {}\n",
                status.name, status.target, status.current, status.is_met
            ));
        }
        report.push_str(&format!(
            "\n## Drift Score: {:.4}\n",
            self.drift_detector.drift_score()
        ));
        report.push_str(&format!(
            "## Degradation Stage: {:?}\n",
            self.degradation_breaker.current_stage()
        ));
        report
    }

    // ── Batch export ─────────────────────────────────────────────────────

    /// Export a batch of telemetry events.
    pub async fn export_batch(&self, events: Vec<TelemetryEvent>) -> Result<(), String> {
        self.exporter
            .export_batch(events)
            .await
            .map_err(|e| e.to_string())
    }

    /// Reconfigure the OTLP exporter to point at a specific endpoint.
    pub fn configure_otlp_endpoint(&mut self, endpoint: &str) {
        self.exporter = OtelExporter::new(OtelExporterConfig {
            endpoint: endpoint.to_string(),
            enabled: true,
            ..OtelExporterConfig::default()
        });
    }

    /// Returns the current OTLP endpoint URL.
    pub fn otlp_endpoint(&self) -> &str {
        &self.exporter.endpoint
    }

    // ── Turn recording ───────────────────────────────────────────────────

    /// Record a turn with input/output token counts.
    pub fn record_turn(&mut self, input_tokens: u32, output_tokens: u32) {
        let usage = TelemetryTokenUsage {
            input_tokens,
            output_tokens,
            cached_tokens: 0,
        };
        self.counter.record(&usage);
    }

    // ── Drift detection ──────────────────────────────────────────────────

    /// Get the current behavioral drift score (0.0–1.0).
    pub fn drift_score(&self) -> f64 {
        self.drift_detector.drift_score()
    }

    /// Check if drift exceeds a threshold.
    pub fn is_drifting(&self, threshold: f64) -> bool {
        self.drift_detector.is_drifting(threshold)
    }

    /// Record a behavior for drift tracking.
    pub fn record_behavior(&mut self, turn: usize, behavior: &str) {
        self.drift_detector.record_behavior(turn, behavior);
    }

    /// Set the baseline patterns for drift detection.
    pub fn set_drift_baseline(&mut self, patterns: Vec<String>) {
        self.drift_detector.set_baseline(patterns);
    }

    // ── Degradation breaker ──────────────────────────────────────────────

    /// Update the degradation stage based on current indicators.
    pub fn update_stage(&mut self, indicators: &DegradationIndicators) {
        self.degradation_breaker.update_stage(indicators);
    }

    /// Get the current degradation stage.
    pub fn current_stage(&self) -> &DegradationStage {
        self.degradation_breaker.current_stage()
    }

    /// Check if the degradation breaker recommends a reset.
    pub fn should_reset(&self) -> bool {
        self.degradation_breaker.should_reset()
    }
}

impl Default for TelemetryBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_new() {
        let t = TelemetryBridge::new();
        assert_eq!(t.session_usage().input_tokens, 0);
        assert_eq!(t.total_cost(), 0.0);
    }

    #[test]
    fn telemetry_record_usage() {
        let mut t = TelemetryBridge::new();
        let usage = TelemetryTokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cached_tokens: 0,
        };
        t.record_usage("claude-sonnet", &usage);
        let model = t.model_usage("claude-sonnet");
        assert!(model.is_some());
        assert_eq!(model.unwrap().input_tokens, 1000);
    }

    #[test]
    fn telemetry_log_cost() {
        let mut t = TelemetryBridge::new();
        let usage = TelemetryTokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cached_tokens: 0,
        };
        t.log_cost("anthropic", "claude-sonnet", usage);
        assert_eq!(t.cost_record_count(), 1);
    }

    #[test]
    fn telemetry_reset_session() {
        let mut t = TelemetryBridge::new();
        let usage = TelemetryTokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cached_tokens: 0,
        };
        t.record_usage("test", &usage);
        t.reset_session();
        assert_eq!(t.session_usage().input_tokens, 0);
    }

    #[test]
    fn telemetry_all_model_usage() {
        let mut t = TelemetryBridge::new();
        let u1 = TelemetryTokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cached_tokens: 0,
        };
        let u2 = TelemetryTokenUsage {
            input_tokens: 200,
            output_tokens: 100,
            cached_tokens: 0,
        };
        t.record_usage("model-a", &u1);
        t.record_usage("model-b", &u2);
        let all = t.all_model_usage();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn telemetry_add_pricing() {
        let mut t = TelemetryBridge::new();
        t.add_pricing(ModelPricing {
            provider_id: ProviderId::new("custom"),
            model_id: ModelId::new("custom-model"),
            input_per_million: 5.0,
            output_per_million: 15.0,
        });
        let usage = TelemetryTokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            cached_tokens: 0,
        };
        let cost = t.calculate_cost("custom", "custom-model", &usage);
        assert!(cost > 0.0, "Custom pricing should produce cost: {cost}");
    }

    #[test]
    fn telemetry_cost_for_model() {
        let mut t = TelemetryBridge::new();
        let usage = TelemetryTokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cached_tokens: 0,
        };
        t.log_cost("anthropic", "claude-sonnet", usage.clone());
        t.log_cost("anthropic", "claude-sonnet", usage);
        let cost = t.cost_for_model("claude-sonnet");
        let records = t.records_for_model("claude-sonnet");
        assert_eq!(records.len(), 2);
        assert!(cost >= 0.0);
    }

    #[test]
    fn telemetry_export_json() {
        let mut t = TelemetryBridge::new();
        let usage = TelemetryTokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cached_tokens: 0,
        };
        t.log_cost("anthropic", "claude-sonnet", usage);
        let json = t.export_json().unwrap();
        assert!(json.contains("claude-sonnet"));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn telemetry_slo_tracking() {
        let mut t = TelemetryBridge::new();
        t.add_slo("availability", 0.99);
        t.record_slo_measurement("availability", 1.0);
        t.record_slo_measurement("availability", 1.0);
        t.record_slo_measurement("availability", 0.0);
        let status = t.check_slo("availability").unwrap();
        assert_eq!(status.name, "availability");
        assert!(status.target > 0.0);
    }

    #[test]
    fn telemetry_check_slo_missing() {
        let t = TelemetryBridge::new();
        assert!(t.check_slo("nonexistent").is_none());
    }

    #[test]
    fn telemetry_budget_enforcement() {
        let mut t = TelemetryBridge::new();
        t.set_budget_limit(1.0);
        assert!(t.check_and_record(0.5).is_ok());
        assert!(t.budget_remaining() < 1.0);
        assert!(t.check_and_record(0.6).is_err());
    }

    #[test]
    fn telemetry_generate_report() {
        let mut t = TelemetryBridge::new();
        t.record_turn(100, 50);
        t.add_slo("latency", 0.95);
        let report = t.generate_report();
        assert!(report.contains("Telemetry Report"));
        assert!(report.contains("Token Usage"));
        assert!(report.contains("Budget"));
    }

    #[test]
    fn telemetry_record_turn() {
        let mut t = TelemetryBridge::new();
        t.record_turn(500, 200);
        assert_eq!(t.total_usage().input_tokens, 500);
        assert_eq!(t.total_usage().output_tokens, 200);
    }

    #[test]
    fn telemetry_drift_score_zero_initially() {
        let t = TelemetryBridge::new();
        assert_eq!(t.drift_score(), 0.0);
        assert!(!t.is_drifting(0.5));
    }

    #[test]
    fn telemetry_degradation_stage_healthy() {
        let t = TelemetryBridge::new();
        assert_eq!(*t.current_stage(), DegradationStage::Healthy);
        assert!(!t.should_reset());
    }

    #[test]
    fn telemetry_update_degradation_stage() {
        let mut t = TelemetryBridge::new();
        let indicators = DegradationIndicators {
            context_utilization: 0.99,
            error_rate: 0.7,
            repetition_rate: 0.8,
            drift_score: 0.9,
        };
        t.update_stage(&indicators);
        assert_ne!(*t.current_stage(), DegradationStage::Healthy);
    }
}
