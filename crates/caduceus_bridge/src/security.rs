//! Security types and permissions bridge.

use caduceus_core::SessionId;
use caduceus_permissions::{
    AuditEntry, AuditLog, Capability, OwaspChecker, OwaspComplianceStatus, PermissionEnforcer,
    PolicyAction, PolicyEngine, PolicyEvalContext, PolicyRule, PrivilegeManager, PrivilegeRing,
    TrustScorer,
};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Finding {
    pub rule_id: String,
    pub file: String,
    pub line: usize,
    pub severity: String,
    pub description: String,
    pub remediation: String,
}

#[derive(Debug, Clone)]
pub struct PromptSafety {
    pub injection_risk: bool,
    pub unsafe_output: bool,
    pub secrets_in_prompt: bool,
}

impl PromptSafety {
    pub fn is_safe(&self) -> bool {
        !self.injection_risk && !self.unsafe_output && !self.secrets_in_prompt
    }
}

/// Bridge for permission enforcement, policy engine, trust scoring, and OWASP compliance.
pub struct PermissionsBridge {
    pub enforcer: PermissionEnforcer,
    pub policy_engine: PolicyEngine,
    pub audit_log: AuditLog,
    pub privilege_manager: PrivilegeManager,
    pub trust_scorer: TrustScorer,
    pub owasp_checker: OwaspChecker,
}

impl PermissionsBridge {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            enforcer: PermissionEnforcer::new(workspace_root),
            policy_engine: PolicyEngine::new(),
            audit_log: AuditLog::new(),
            privilege_manager: PrivilegeManager::new(PrivilegeRing::Workspace),
            trust_scorer: TrustScorer::new(),
            owasp_checker: OwaspChecker::new(),
        }
    }

    // ── Capability management ────────────────────────────────────────────

    /// Grant a capability to the enforcer.
    pub fn grant_capability(&mut self, cap: Capability) {
        self.enforcer.grant_capability(cap);
    }

    /// Revoke a capability from the enforcer.
    pub fn revoke_capability(&mut self, cap: &Capability) {
        self.enforcer.revoke_capability(cap);
    }

    // ── Policy engine ────────────────────────────────────────────────────

    /// Evaluate a policy context and return the action.
    pub fn evaluate(&self, context: &PolicyEvalContext) -> PolicyAction {
        self.policy_engine.evaluate(context)
    }

    /// Add a policy rule.
    pub fn add_rule(&mut self, rule: PolicyRule) {
        self.policy_engine.add_rule(rule);
    }

    // ── Privilege ring permission check ──────────────────────────────────

    /// Check if a tool is allowed under the current privilege ring.
    pub fn check_permission(&self, tool: &str) -> Result<(), String> {
        self.privilege_manager.check_permission(tool)
    }

    // ── Audit log (in-memory) ────────────────────────────────────────────

    /// Get audit entries for a session from the in-memory log.
    pub fn entries_for_session(&self, session_id: &SessionId) -> Vec<&AuditEntry> {
        self.audit_log.entries_for_session(session_id)
    }

    /// Record an audit entry in the in-memory log.
    pub fn record_audit(&mut self, entry: AuditEntry) {
        self.audit_log.record(entry);
    }

    // ── OWASP compliance ─────────────────────────────────────────────────

    /// Generate the OWASP compliance report.
    pub fn generate_security_report(&self) -> String {
        self.owasp_checker.generate_report()
    }

    /// Get the OWASP compliance score (0.0–1.0).
    pub fn compliance_score(&self) -> f64 {
        self.owasp_checker.compliance_score()
    }

    /// Update the compliance status of an OWASP check.
    pub fn update_owasp_status(&mut self, id: &str, status: OwaspComplianceStatus) {
        self.owasp_checker.update_status(id, status);
    }

    // ── Trust scoring (record_violation) ─────────────────────────────────

    /// Record a policy violation for an agent.
    pub fn record_violation(&mut self, agent_id: &str) {
        self.trust_scorer.record_violation(agent_id);
    }

    /// Record a success for an agent.
    pub fn record_success(&mut self, agent_id: &str) {
        self.trust_scorer.record_success(agent_id);
    }

    /// Get trust score for an agent (0–1000).
    pub fn get_trust_score(&self, agent_id: &str) -> u32 {
        self.trust_scorer.get_score(agent_id)
    }

    /// Check if an agent is trusted above a threshold.
    pub fn is_trusted(&self, agent_id: &str, threshold: u32) -> bool {
        self.trust_scorer.is_trusted(agent_id, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caduceus_permissions::{PolicyCondition, PermissionDecision};

    #[test]
    fn permissions_grant_revoke_capability() {
        let mut bridge = PermissionsBridge::new("/tmp/test");
        bridge.grant_capability(Capability::FsRead);
        bridge.revoke_capability(&Capability::FsRead);
    }

    #[test]
    fn permissions_policy_evaluate_default_allow() {
        let bridge = PermissionsBridge::new("/tmp/test");
        let ctx = PolicyEvalContext {
            tool_name: "read".to_string(),
            args: serde_json::json!({}),
            estimated_cost: None,
            current_hour: 12,
        };
        let action = bridge.evaluate(&ctx);
        assert_eq!(action, PolicyAction::Allow);
    }

    #[test]
    fn permissions_policy_add_rule_deny() {
        let mut bridge = PermissionsBridge::new("/tmp/test");
        bridge.add_rule(PolicyRule {
            name: "deny-bash".to_string(),
            description: "Deny bash tool".to_string(),
            condition: PolicyCondition::ToolName("bash".to_string()),
            action: PolicyAction::Deny("Not allowed".to_string()),
            priority: 10,
        });
        let ctx = PolicyEvalContext {
            tool_name: "bash".to_string(),
            args: serde_json::json!({}),
            estimated_cost: None,
            current_hour: 12,
        };
        let action = bridge.evaluate(&ctx);
        assert!(matches!(action, PolicyAction::Deny(_)));
    }

    #[test]
    fn permissions_check_permission_read_ok() {
        let bridge = PermissionsBridge::new("/tmp/test");
        assert!(bridge.check_permission("read").is_ok());
    }

    #[test]
    fn permissions_check_permission_unrestricted_denied() {
        let bridge = PermissionsBridge::new("/tmp/test");
        assert!(bridge.check_permission("unsafe_shell").is_err());
    }

    #[test]
    fn permissions_audit_entries_for_session() {
        let mut bridge = PermissionsBridge::new("/tmp/test");
        let sid = SessionId::new();
        let entry = AuditEntry {
            session_id: sid.clone(),
            capability: Capability::FsRead,
            resource: "/foo.rs".to_string(),
            decision: PermissionDecision::Allowed,
            timestamp: chrono::Utc::now(),
        };
        bridge.record_audit(entry);
        let entries = bridge.entries_for_session(&sid);
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn permissions_owasp_report() {
        let bridge = PermissionsBridge::new("/tmp/test");
        let report = bridge.generate_security_report();
        assert!(report.contains("OWASP"));
        let score = bridge.compliance_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn permissions_trust_scoring() {
        let mut bridge = PermissionsBridge::new("/tmp/test");
        bridge.record_success("agent-1");
        bridge.record_success("agent-1");
        bridge.record_violation("agent-1");
        let score = bridge.get_trust_score("agent-1");
        assert!(score > 0);
        assert!(bridge.is_trusted("agent-1", 400));
    }

    #[test]
    fn permissions_record_violation_lowers_score() {
        let mut bridge = PermissionsBridge::new("/tmp/test");
        let base = bridge.get_trust_score("agent-x");
        bridge.record_violation("agent-x");
        let after = bridge.get_trust_score("agent-x");
        assert!(after < base);
    }
}
