use std::sync::Arc;

use client::{Client, UserStore};
use credentials_provider::CredentialsProvider;
use gpui::{App, Context, Entity};
use language_model::LanguageModelRegistry;

pub mod extension;
pub mod provider;
mod settings;

pub use crate::extension::init_proxy as init_extension_proxy;

use crate::provider::copilot_chat::CopilotChatLanguageModelProvider;
pub use crate::settings::*;

pub fn init(user_store: Entity<UserStore>, client: Arc<Client>, cx: &mut App) {
    let registry = LanguageModelRegistry::global(cx);
    registry.update(cx, |registry, cx| {
        register_language_model_providers(
            registry,
            user_store,
            client.clone(),
            client.credentials_provider(),
            cx,
        );
    });

    // Caduceus: disabled extension-based and OpenAI-compatible providers.
    // Only GitHub Copilot Chat is supported.
}

fn register_language_model_providers(
    registry: &mut LanguageModelRegistry,
    _user_store: Entity<UserStore>,
    _client: Arc<Client>,
    _credentials_provider: Arc<dyn CredentialsProvider>,
    cx: &mut Context<LanguageModelRegistry>,
) {
    // Caduceus: only register GitHub Copilot Chat as the LLM provider.
    // All other providers (Anthropic, OpenAI, Google, etc.) are disabled.
    // Authentication and model selection happen through GitHub.
    registry.register_provider(Arc::new(CopilotChatLanguageModelProvider::new(cx)), cx);
}
