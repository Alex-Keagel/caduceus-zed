//! Vendor-diverse model selection for DAG fan-out.
//!
//! Given the registry of authenticated language models, returns up to `n`
//! IDs spread across distinct vendor families (claude / gpt / gemini / grok /
//! …). Used by the `suggest_models` tool so that a master agent decomposing
//! a plan into parallel sub-agents can run each branch on a different
//! vendor's best-available model — diversity-by-default.

use gpui::App;
use language_model::LanguageModelRegistry;
use serde::{Deserialize, Serialize};

/// One vendor-diverse model recommendation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiverseModel {
    pub model_id: String,
    pub provider_id: String,
    /// Family bucket key inferred from the model id prefix
    /// (e.g. `claude`, `gpt`, `gemini`, `grok`, `o`, `mistral`, `qwen`).
    pub family: String,
}

/// Group an arbitrary model id into a coarse vendor-family bucket. The bucket
/// is purely a string prefix heuristic — good enough to drive round-robin
/// selection across vendors without needing each provider to expose a
/// dedicated `family()` getter.
pub fn family_of(model_id: &str) -> String {
    let lower = model_id.to_ascii_lowercase();
    // Order matters: more specific prefixes first.
    for (prefix, family) in [
        ("claude", "claude"),
        ("anthropic", "claude"),
        ("gpt-", "gpt"),
        ("openai", "gpt"),
        ("o1", "o"),
        ("o3", "o"),
        ("o4", "o"),
        ("gemini", "gemini"),
        ("google", "gemini"),
        ("grok", "grok"),
        ("xai", "grok"),
        ("mistral", "mistral"),
        ("codestral", "mistral"),
        ("qwen", "qwen"),
        ("llama", "llama"),
        ("deepseek", "deepseek"),
    ] {
        if lower.starts_with(prefix) || lower.contains(&format!("/{}", prefix)) {
            return family.to_string();
        }
    }
    // Fall back to the first dash-separated segment so unknown providers
    // still bucket coherently rather than collapsing into one giant "other"
    // family.
    lower
        .split(['-', '/', ':'])
        .next()
        .unwrap_or("other")
        .to_string()
}

/// Return up to `n` model IDs spread across distinct vendor families.
///
/// Algorithm:
///   1. Enumerate every authenticated model from the registry.
///   2. Bucket by [`family_of`].
///   3. Sort families by bucket size (descending) so the most populated
///      vendors are visited first; within a bucket, prefer the alphabetically
///      latest id (rough proxy for "newest").
///   4. Round-robin across families: pick one per family, then loop, until
///      we have `n` picks or no families remain.
///
/// `exclude` is matched as a literal model id and removed from the candidate
/// pool — typically the master's own model so that the suggested fan-out is
/// disjoint from the master.
pub fn assign_diverse_models(
    n: usize,
    exclude: Option<&str>,
    cx: &App,
) -> Vec<DiverseModel> {
    if n == 0 {
        return Vec::new();
    }

    let registry = LanguageModelRegistry::read_global(cx);
    let mut by_family: std::collections::BTreeMap<String, Vec<DiverseModel>> =
        std::collections::BTreeMap::new();

    for model in registry.available_models(cx) {
        let id = model.id().0.to_string();
        if Some(id.as_str()) == exclude {
            continue;
        }
        let provider_id = model.provider_id().0.to_string();
        let family = family_of(&id);
        by_family.entry(family.clone()).or_default().push(DiverseModel {
            model_id: id,
            provider_id,
            family,
        });
    }

    // Within each family, sort by id descending so newer-looking ids win
    // when only one is needed.
    for bucket in by_family.values_mut() {
        bucket.sort_by(|a, b| b.model_id.cmp(&a.model_id));
    }

    // Order families by bucket size descending, then name ascending for
    // determinism on ties.
    let mut families: Vec<(String, Vec<DiverseModel>)> = by_family.into_iter().collect();
    families.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));

    let mut out: Vec<DiverseModel> = Vec::with_capacity(n);
    let mut idx_per_family: Vec<usize> = vec![0; families.len()];
    while out.len() < n {
        let mut picked_this_round = false;
        for (fam_idx, (_, bucket)) in families.iter().enumerate() {
            let i = idx_per_family[fam_idx];
            if i < bucket.len() {
                out.push(bucket[i].clone());
                idx_per_family[fam_idx] = i + 1;
                picked_this_round = true;
                if out.len() == n {
                    break;
                }
            }
        }
        if !picked_this_round {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_buckets() {
        assert_eq!(family_of("claude-opus-4.7"), "claude");
        assert_eq!(family_of("Claude-Sonnet-4.6"), "claude");
        assert_eq!(family_of("gpt-5.4"), "gpt");
        assert_eq!(family_of("gpt-5.3-codex"), "gpt");
        assert_eq!(family_of("gemini-2.5-pro"), "gemini");
        assert_eq!(family_of("grok-4"), "grok");
        assert_eq!(family_of("o3-mini"), "o");
        assert_eq!(family_of("mistral-large"), "mistral");
        assert_eq!(family_of("codestral-2501"), "mistral");
        // Unknown providers fall back to first segment, not a giant "other".
        assert_eq!(family_of("custom-model-v2"), "custom");
    }
}
