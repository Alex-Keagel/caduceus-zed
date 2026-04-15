//! Telemetry bridge — token counting, cost calculation, usage tracking.

use caduceus_telemetry::{CostCalculator, CostLogger, CostRecord, ModelPricing, TokenCounter};
use caduceus_core::{ModelId, ProviderId};

/// Re-export the telemetry TokenUsage (different from core::TokenUsage)
pub use caduceus_telemetry::TokenUsage as TelemetryTokenUsage;

/// Wrapper around telemetry subsystems.
pub struct TelemetryBridge {
    pub counter: TokenCounter,
    pub calculator: CostCalculator,
    pub logger: CostLogger,
}

impl TelemetryBridge {
    pub fn new() -> Self {
        Self {
            counter: TokenCounter::new(),
            calculator: CostCalculator::new(),
            logger: CostLogger::new(),
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
        self.calculator.calculate(
            &ProviderId::new(provider),
            &ModelId::new(model),
            usage,
        )
    }

    /// Log a cost event.
    pub fn log_cost(&mut self, provider: &str, model: &str, usage: TelemetryTokenUsage) {
        self.logger.log(
            &ProviderId::new(provider),
            &ModelId::new(model),
            usage,
        );
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
        let u1 = TelemetryTokenUsage { input_tokens: 100, output_tokens: 50, cached_tokens: 0 };
        let u2 = TelemetryTokenUsage { input_tokens: 200, output_tokens: 100, cached_tokens: 0 };
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
        let usage = TelemetryTokenUsage { input_tokens: 1000, output_tokens: 500, cached_tokens: 0 };
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
        let usage = TelemetryTokenUsage { input_tokens: 100, output_tokens: 50, cached_tokens: 0 };
        t.log_cost("anthropic", "claude-sonnet", usage);
        let json = t.export_json().unwrap();
        assert!(json.contains("claude-sonnet"));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
    }
}
