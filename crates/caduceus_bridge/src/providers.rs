//! Provider bridge — registry, model resolution, image encoding.

use caduceus_providers::ProviderRegistry;
use std::path::Path;

// Re-export the registry type for consumers.
pub use caduceus_providers::ProviderRegistry as BridgeProviderRegistry;

/// Thin wrapper around provider operations.
pub struct ProvidersBridge;

impl ProvidersBridge {
    /// Create a new empty provider registry.
    pub fn new_registry() -> ProviderRegistry {
        ProviderRegistry::new()
    }

    /// List registered provider names as plain strings.
    pub fn list_providers(registry: &ProviderRegistry) -> Vec<String> {
        registry
            .list_providers()
            .iter()
            .map(|p| p.0.clone())
            .collect()
    }

    /// Resolve a `"provider:model"` string into `(provider, model)` string pair.
    pub fn resolve_model(
        registry: &ProviderRegistry,
        model_string: &str,
    ) -> Option<(String, String)> {
        registry
            .resolve_model(model_string)
            .map(|(p, m)| (p.0, m.0))
    }

    /// Base64-encode an image file for inclusion in LLM API requests.
    pub fn encode_image_file(path: &Path) -> Result<caduceus_core::ImageContent, String> {
        caduceus_providers::encode_image_file(path).map_err(|e| e.to_string())
    }

    // NOTE: validate_key lives on ProviderConnector<S, P> which requires
    // application-specific AuthStore + ApiKeyPrompter trait implementations.
    // It cannot be cleanly bridged without concrete trait objects, so it is
    // intentionally omitted here.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn providers_new_registry() {
        let registry = ProvidersBridge::new_registry();
        assert!(ProvidersBridge::list_providers(&registry).is_empty());
    }

    #[test]
    fn providers_list_empty() {
        let registry = ProvidersBridge::new_registry();
        let names = ProvidersBridge::list_providers(&registry);
        assert_eq!(names.len(), 0);
    }

    #[test]
    fn providers_resolve_model_no_match() {
        let registry = ProvidersBridge::new_registry();
        assert!(ProvidersBridge::resolve_model(&registry, "anthropic:claude-sonnet").is_none());
    }

    #[test]
    fn providers_resolve_model_bad_format() {
        let registry = ProvidersBridge::new_registry();
        assert!(ProvidersBridge::resolve_model(&registry, "no-colon").is_none());
    }

    #[test]
    fn providers_encode_image_missing_file() {
        let result = ProvidersBridge::encode_image_file(Path::new("/nonexistent/img.png"));
        assert!(result.is_err());
    }

    #[test]
    fn providers_encode_image_bad_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, b"not an image").unwrap();
        let result = ProvidersBridge::encode_image_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn providers_encode_image_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.png");
        // Minimal PNG-like bytes (content doesn't need to be valid PNG for encoding)
        std::fs::write(&path, b"\x89PNG\r\n\x1a\nhello").unwrap();
        let result = ProvidersBridge::encode_image_file(&path);
        assert!(result.is_ok());
    }
}
