//! Security types for the bridge.

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
