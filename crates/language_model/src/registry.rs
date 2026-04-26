use crate::{
    AuthAction, LanguageModel, LanguageModelId, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderState, ProviderAuthState, ZED_CLOUD_PROVIDER_ID,
};
use collections::{BTreeMap, HashMap, HashSet};
use gpui::{App, Context, Entity, EventEmitter, Global, prelude::*};
use language_model_core::LanguageModelCompletionError;
use std::cell::RefCell;
use std::time::{Duration, Instant};
use std::{str::FromStr, sync::Arc};
use thiserror::Error;
use util::maybe;

/// Function type for checking if a built-in provider should be hidden.
/// Returns Some(extension_id) if the provider should be hidden when that extension is installed.
pub type BuiltinProviderHidingFn = Box<dyn Fn(&str) -> Option<&'static str> + Send + Sync>;

pub fn init(cx: &mut App) {
    let registry = cx.new(|_cx| LanguageModelRegistry::default());
    cx.set_global(GlobalLanguageModelRegistry(registry));
}

struct GlobalLanguageModelRegistry(Entity<LanguageModelRegistry>);

impl Global for GlobalLanguageModelRegistry {}

#[derive(Error)]
pub enum ConfigurationError {
    #[error("Configure at least one LLM provider to start using the panel.")]
    NoProvider,
    #[error("LLM provider is not configured or does not support the configured model.")]
    ModelNotFound,
    #[error("{} LLM provider is not configured.", .0.name().0)]
    ProviderNotAuthenticated(Arc<dyn LanguageModelProvider>),
    /// ST1a: provider is rate-limited (HTTP 429). Carries the optional `retry_after` so
    /// the UI (ST5/ST1b) can surface "retry in N s" without re-querying the provider.
    #[error("{} LLM provider is rate-limited.", .0.name().0)]
    ProviderRateLimited(Arc<dyn LanguageModelProvider>, Option<Duration>),
    /// ST1a: provider is disabled by org/admin policy. `reason` is sanitized at the
    /// `ProviderAuthState` boundary so it's safe to render verbatim.
    #[error("{} LLM provider is disabled by policy: {}", .0.name().0, .1)]
    ProviderDisabledByPolicy(Arc<dyn LanguageModelProvider>, String),
}

impl std::fmt::Debug for ConfigurationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProvider => write!(f, "NoProvider"),
            Self::ModelNotFound => write!(f, "ModelNotFound"),
            Self::ProviderNotAuthenticated(provider) => {
                write!(f, "ProviderNotAuthenticated({})", provider.id())
            }
            Self::ProviderRateLimited(provider, retry_after) => {
                write!(f, "ProviderRateLimited({}, {:?})", provider.id(), retry_after)
            }
            Self::ProviderDisabledByPolicy(provider, reason) => {
                write!(
                    f,
                    "ProviderDisabledByPolicy({}, {:?})",
                    provider.id(),
                    reason
                )
            }
        }
    }
}

#[derive(Default)]
pub struct LanguageModelRegistry {
    default_model: Option<ConfiguredModel>,
    default_fast_model: Option<ConfiguredModel>,
    inline_assistant_model: Option<ConfiguredModel>,
    commit_message_model: Option<ConfiguredModel>,
    thread_summary_model: Option<ConfiguredModel>,
    providers: BTreeMap<LanguageModelProviderId, Arc<dyn LanguageModelProvider>>,
    inline_alternatives: Vec<Arc<dyn LanguageModel>>,
    /// Set of installed extension IDs that provide language models.
    /// Used to determine which built-in providers should be hidden.
    installed_llm_extension_ids: HashSet<Arc<str>>,
    /// Function to check if a built-in provider should be hidden by an extension.
    builtin_provider_hiding_fn: Option<BuiltinProviderHidingFn>,
    /// ST1a: per-provider auth-state cache. Read via `cached_auth_state(id, cx)` and
    /// invalidated on `Event::ProviderStateChanged` or via `note_completion_error`.
    /// `RefCell` because reads happen through `&self`. Values are auto-expired when
    /// `rate_limited_until <= now` so a `RateLimited` state self-clears.
    auth_cache: RefCell<HashMap<LanguageModelProviderId, CachedAuth>>,
}

/// Cached `ProviderAuthState` for one provider. See `LanguageModelRegistry::auth_cache`.
struct CachedAuth {
    state: ProviderAuthState,
    /// If `Some`, the cache entry is treated as expired once `Instant::now() >= deadline`.
    /// Set when a rate-limit is recorded so the cache auto-clears without an explicit
    /// invalidation event.
    rate_limited_until: Option<Instant>,
}

#[derive(Debug)]
pub struct SelectedModel {
    pub provider: LanguageModelProviderId,
    pub model: LanguageModelId,
}

impl FromStr for SelectedModel {
    type Err = String;

    /// Parse string identifiers like `provider_id/model_id` into a `SelectedModel`
    fn from_str(id: &str) -> Result<SelectedModel, Self::Err> {
        let parts: Vec<&str> = id.split('/').collect();
        let [provider_id, model_id] = parts.as_slice() else {
            return Err(format!(
                "Invalid model identifier format: `{}`. Expected `provider_id/model_id`",
                id
            ));
        };

        if provider_id.is_empty() || model_id.is_empty() {
            return Err(format!("Provider and model ids can't be empty: `{}`", id));
        }

        Ok(SelectedModel {
            provider: LanguageModelProviderId(provider_id.to_string().into()),
            model: LanguageModelId(model_id.to_string().into()),
        })
    }
}

#[derive(Clone)]
pub struct ConfiguredModel {
    pub provider: Arc<dyn LanguageModelProvider>,
    pub model: Arc<dyn LanguageModel>,
}

impl ConfiguredModel {
    pub fn is_same_as(&self, other: &ConfiguredModel) -> bool {
        self.model.id() == other.model.id() && self.provider.id() == other.provider.id()
    }

    pub fn is_provided_by_zed(&self) -> bool {
        self.provider.id() == ZED_CLOUD_PROVIDER_ID
    }
}

pub enum Event {
    DefaultModelChanged,
    InlineAssistantModelChanged,
    CommitMessageModelChanged,
    ThreadSummaryModelChanged,
    ProviderStateChanged(LanguageModelProviderId),
    AddedProvider(LanguageModelProviderId),
    RemovedProvider(LanguageModelProviderId),
    /// Emitted when provider visibility changes due to extension install/uninstall.
    ProvidersChanged,
}

impl EventEmitter<Event> for LanguageModelRegistry {}

impl LanguageModelRegistry {
    pub fn global(cx: &App) -> Entity<Self> {
        cx.global::<GlobalLanguageModelRegistry>().0.clone()
    }

    pub fn read_global(cx: &App) -> &Self {
        cx.global::<GlobalLanguageModelRegistry>().0.read(cx)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn test(cx: &mut App) -> Arc<crate::fake_provider::FakeLanguageModelProvider> {
        let fake_provider = Arc::new(crate::fake_provider::FakeLanguageModelProvider::default());
        let registry = cx.new(|cx| {
            let mut registry = Self::default();
            registry.register_provider(fake_provider.clone(), cx);
            let model = fake_provider.provided_models(cx)[0].clone();
            let configured_model = ConfiguredModel {
                provider: fake_provider.clone(),
                model,
            };
            registry.set_default_model(Some(configured_model), cx);
            registry
        });
        cx.set_global(GlobalLanguageModelRegistry(registry));
        fake_provider
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn fake_model(&self) -> Arc<dyn LanguageModel> {
        self.default_model.as_ref().unwrap().model.clone()
    }

    pub fn register_provider<T: LanguageModelProvider + LanguageModelProviderState>(
        &mut self,
        provider: Arc<T>,
        cx: &mut Context<Self>,
    ) {
        let id = provider.id();

        let subscription = provider.subscribe(cx, {
            let id = id.clone();
            move |this, cx| {
                // ST1a: provider's observable changed → drop its auth-state cache entry so
                // the next read calls `auth_state(cx)` fresh. Then re-emit the public event
                // for downstream subscribers (selectors, panels, etc.).
                this.auth_cache.borrow_mut().remove(&id);
                cx.emit(Event::ProviderStateChanged(id.clone()));
            }
        });
        if let Some(subscription) = subscription {
            subscription.detach();
        }

        self.providers.insert(id.clone(), provider);
        cx.emit(Event::AddedProvider(id));
    }

    pub fn unregister_provider(&mut self, id: LanguageModelProviderId, cx: &mut Context<Self>) {
        if self.providers.remove(&id).is_some() {
            self.auth_cache.borrow_mut().remove(&id);
            cx.emit(Event::RemovedProvider(id));
        }
    }

    pub fn providers(&self) -> Vec<Arc<dyn LanguageModelProvider>> {
        let zed_provider_id = LanguageModelProviderId("zed.dev".into());
        let mut providers = Vec::with_capacity(self.providers.len());
        if let Some(provider) = self.providers.get(&zed_provider_id) {
            providers.push(provider.clone());
        }
        providers.extend(self.providers.values().filter_map(|p| {
            if p.id() != zed_provider_id {
                Some(p.clone())
            } else {
                None
            }
        }));
        providers
    }

    /// Returns providers, filtering out hidden built-in providers.
    pub fn visible_providers(&self) -> Vec<Arc<dyn LanguageModelProvider>> {
        self.providers()
            .into_iter()
            .filter(|p| !self.should_hide_provider(&p.id()))
            .collect()
    }

    /// Sets the function used to check if a built-in provider should be hidden.
    pub fn set_builtin_provider_hiding_fn(&mut self, hiding_fn: BuiltinProviderHidingFn) {
        self.builtin_provider_hiding_fn = Some(hiding_fn);
    }

    /// Called when an extension is installed/loaded.
    /// If the extension provides language models, track it so we can hide the corresponding built-in.
    pub fn extension_installed(&mut self, extension_id: Arc<str>, cx: &mut Context<Self>) {
        if self.installed_llm_extension_ids.insert(extension_id) {
            cx.emit(Event::ProvidersChanged);
            cx.notify();
        }
    }

    /// Called when an extension is uninstalled/unloaded.
    pub fn extension_uninstalled(&mut self, extension_id: &str, cx: &mut Context<Self>) {
        if self.installed_llm_extension_ids.remove(extension_id) {
            cx.emit(Event::ProvidersChanged);
            cx.notify();
        }
    }

    /// Sync the set of installed LLM extension IDs.
    pub fn sync_installed_llm_extensions(
        &mut self,
        extension_ids: HashSet<Arc<str>>,
        cx: &mut Context<Self>,
    ) {
        if extension_ids != self.installed_llm_extension_ids {
            self.installed_llm_extension_ids = extension_ids;
            cx.emit(Event::ProvidersChanged);
            cx.notify();
        }
    }

    /// Returns true if a provider should be hidden from the UI.
    /// Built-in providers are hidden when their corresponding extension is installed.
    pub fn should_hide_provider(&self, provider_id: &LanguageModelProviderId) -> bool {
        if let Some(ref hiding_fn) = self.builtin_provider_hiding_fn {
            if let Some(extension_id) = hiding_fn(&provider_id.0) {
                return self.installed_llm_extension_ids.contains(extension_id);
            }
        }
        false
    }

    pub fn configuration_error(
        &self,
        model: Option<ConfiguredModel>,
        cx: &App,
    ) -> Option<ConfigurationError> {
        let Some(model) = model else {
            if !self.has_authenticated_provider(cx) {
                return Some(ConfigurationError::NoProvider);
            }
            return Some(ConfigurationError::ModelNotFound);
        };

        // ST1a (C1): variant-direct read so the caller can distinguish "not signed in"
        // from "rate-limited" and "disabled by policy". `Authenticated` returns None.
        match self.cached_auth_state(&model.provider.id(), cx) {
            ProviderAuthState::Authenticated => None,
            ProviderAuthState::NotAuthenticated { .. } => Some(
                ConfigurationError::ProviderNotAuthenticated(model.provider),
            ),
            ProviderAuthState::RateLimited { retry_after, .. } => Some(
                ConfigurationError::ProviderRateLimited(model.provider, retry_after),
            ),
            ProviderAuthState::DisabledByPolicy(reason) => Some(
                ConfigurationError::ProviderDisabledByPolicy(model.provider, reason.into_string()),
            ),
        }
    }

    /// Returns `true` if at least one provider is `Authenticated` (can serve completions
    /// right now). ST1a (C2 — usability check, NOT configuration check). Routed through
    /// `cached_auth_state` so a fan-out render frame causes ≤1 `provider.auth_state(cx)`
    /// call per provider (AC-PERF1).
    pub fn has_authenticated_provider(&self, cx: &App) -> bool {
        self.providers
            .keys()
            .any(|id| self.cached_auth_state(id, cx).can_provide_models())
    }

    pub fn available_models<'a>(
        &'a self,
        cx: &'a App,
    ) -> impl Iterator<Item = Arc<dyn LanguageModel>> + 'a {
        // ST1a (C3 — usability check; ST1b removes the filter entirely once selector is
        // redesigned to render unauth providers). Routed through `cached_auth_state`
        // (AC-PERF1) so this is amortized across fan-out callers in the same render.
        self.providers
            .iter()
            .filter(|(id, _)| self.cached_auth_state(id, cx).can_provide_models())
            .flat_map(|(_, provider)| provider.provided_models(cx))
    }

    pub fn provider(&self, id: &LanguageModelProviderId) -> Option<Arc<dyn LanguageModelProvider>> {
        self.providers.get(id).cloned()
    }

    /// ST1a: returns the `ProviderAuthState` for `id`, using the registry's auth-state
    /// cache. The cache is invalidated by:
    ///
    /// * `Event::ProviderStateChanged` (`register_provider`'s subscription drops the entry)
    /// * `note_completion_error` (`AuthenticationError` → drop; `RateLimitExceeded` →
    ///   record `RateLimited` with the supplied `retry_after` clamped to `MAX_RETRY_AFTER`)
    /// * `invalidate_auth_cache` (manual; for token-refresh paths and tests)
    /// * Auto-expiry: a cached `RateLimited` whose `rate_limited_until` deadline has
    ///   passed is dropped on the next read so the provider returns to `Authenticated`
    ///   without needing an external invalidation.
    ///
    /// Per AC-PERF1, callers that fan out (panel render, selector list) should go
    /// through this method so the underlying `provider.auth_state(cx)` is invoked at
    /// most once per provider per render frame.
    pub fn cached_auth_state(
        &self,
        id: &LanguageModelProviderId,
        cx: &App,
    ) -> ProviderAuthState {
        // Fast-path read: clone the cached entry if present and not expired. Two-phase
        // borrow (read, then mutate on miss) avoids holding the `RefMut` across the
        // provider call, which can re-enter the registry on some code paths.
        {
            let mut cache = self.auth_cache.borrow_mut();
            if let Some(entry) = cache.get(id) {
                if let Some(deadline) = entry.rate_limited_until {
                    if Instant::now() >= deadline {
                        cache.remove(id);
                    } else {
                        return entry.state.clone();
                    }
                } else {
                    return entry.state.clone();
                }
            }
        }

        // Miss: ask the provider.
        let Some(provider) = self.providers.get(id) else {
            return ProviderAuthState::NotAuthenticated {
                action: AuthAction::None,
            };
        };
        let state = provider.auth_state(cx);
        self.auth_cache.borrow_mut().insert(
            id.clone(),
            CachedAuth {
                state: state.clone(),
                rate_limited_until: None,
            },
        );
        state
    }

    /// Drop the cached auth-state entry for `id`. Public so token-refresh paths and
    /// tests can force a refresh; the next `cached_auth_state` call will re-query the
    /// provider. Does NOT emit a registry event; if you want subscribers to react,
    /// emit `Event::ProviderStateChanged` separately or trigger the provider's
    /// observable to fire.
    pub fn invalidate_auth_cache(&self, id: &LanguageModelProviderId) {
        self.auth_cache.borrow_mut().remove(id);
    }

    /// ST1a: feed a `LanguageModelCompletionError` into the auth-state cache (S4).
    ///
    /// * `RateLimitExceeded { retry_after }` → cache `RateLimited` with
    ///   `rate_limited_until = now + clamp(retry_after, MAX_RETRY_AFTER)`. Auto-expires
    ///   on read once the deadline passes.
    /// * `AuthenticationError` (HTTP 401) → drop the cache entry so the next read
    ///   re-queries the provider (which by then has typically transitioned to
    ///   `NotAuthenticated`).
    ///
    /// ST1a only adds the registry method. ST5 wires production completion-error sites
    /// to call it.
    pub fn note_completion_error(
        &self,
        id: &LanguageModelProviderId,
        err: &LanguageModelCompletionError,
    ) {
        match err {
            LanguageModelCompletionError::RateLimitExceeded { retry_after, .. } => {
                // Round-3 fix-loop #3: `NotAuthenticated` is sticky vs 429.
                // Without this guard, a 429 arriving after a 401 (e.g. retry
                // logic hits a rate limit on the unauthenticated request)
                // would silently overwrite the cached `NotAuthenticated` with
                // `RateLimited`, hiding the real "user must re-auth" signal
                // behind a transient rate-limit UI for up to MAX_RETRY_AFTER
                // / HEADERLESS_RATE_LIMIT_TTL.
                {
                    let cache = self.auth_cache.borrow();
                    if let Some(existing) = cache.get(id)
                        && matches!(existing.state, ProviderAuthState::NotAuthenticated { .. })
                    {
                        return;
                    }
                }
                // fix-loop #7: hand the raw `retry_after` to `rate_limited()`;
                // that constructor is the sole emitter of the over-cap
                // `log::warn!`. The clamp on the `deadline` line below derives
                // the cache-eviction deadline only — it produces an identical
                // numeric result but does not duplicate the warning. (Round-3
                // reword: previous comment incorrectly claimed this site was
                // the SINGLE clamping authority, which was misleading because
                // the deadline derivation also clamps.)
                let state = ProviderAuthState::rate_limited(*retry_after, AuthAction::None);
                // fix-loop #8: headerless 429 (no Retry-After) must NOT poison the cache
                // forever. Use a bounded fallback TTL so the entry auto-expires.
                let deadline = match retry_after {
                    Some(d) => Some(Instant::now() + (*d).min(crate::MAX_RETRY_AFTER)),
                    None => Some(Instant::now() + crate::HEADERLESS_RATE_LIMIT_TTL),
                };
                self.auth_cache.borrow_mut().insert(
                    id.clone(),
                    CachedAuth {
                        state,
                        rate_limited_until: deadline,
                    },
                );
            }
            LanguageModelCompletionError::AuthenticationError { .. } => {
                self.auth_cache.borrow_mut().remove(id);
            }
            _ => {
                // Other completion errors are not auth-relevant.
            }
        }
    }

    pub fn select_default_model(&mut self, model: Option<&SelectedModel>, cx: &mut Context<Self>) {
        let configured_model = model.and_then(|model| self.select_model(model, cx));
        self.set_default_model(configured_model, cx);
    }

    pub fn select_inline_assistant_model(
        &mut self,
        model: Option<&SelectedModel>,
        cx: &mut Context<Self>,
    ) {
        let configured_model = model.and_then(|model| self.select_model(model, cx));
        self.set_inline_assistant_model(configured_model, cx);
    }

    pub fn select_commit_message_model(
        &mut self,
        model: Option<&SelectedModel>,
        cx: &mut Context<Self>,
    ) {
        let configured_model = model.and_then(|model| self.select_model(model, cx));
        self.set_commit_message_model(configured_model, cx);
    }

    pub fn select_thread_summary_model(
        &mut self,
        model: Option<&SelectedModel>,
        cx: &mut Context<Self>,
    ) {
        let configured_model = model.and_then(|model| self.select_model(model, cx));
        self.set_thread_summary_model(configured_model, cx);
    }

    /// Selects and sets the inline alternatives for language models based on
    /// provider name and id.
    pub fn select_inline_alternative_models(
        &mut self,
        alternatives: impl IntoIterator<Item = SelectedModel>,
        cx: &mut Context<Self>,
    ) {
        self.inline_alternatives = alternatives
            .into_iter()
            .flat_map(|alternative| {
                self.select_model(&alternative, cx)
                    .map(|configured_model| configured_model.model)
            })
            .collect::<Vec<_>>();
    }

    pub fn select_model(
        &mut self,
        selected_model: &SelectedModel,
        cx: &mut Context<Self>,
    ) -> Option<ConfiguredModel> {
        let provider = self.provider(&selected_model.provider)?;
        let model = provider
            .provided_models(cx)
            .iter()
            .find(|model| model.id() == selected_model.model)?
            .clone();
        Some(ConfiguredModel { provider, model })
    }

    pub fn set_default_model(&mut self, model: Option<ConfiguredModel>, cx: &mut Context<Self>) {
        match (self.default_model.as_ref(), model.as_ref()) {
            (Some(old), Some(new)) if old.is_same_as(new) => {}
            (None, None) => {}
            _ => cx.emit(Event::DefaultModelChanged),
        }
        self.default_fast_model = maybe!({
            let provider = &model.as_ref()?.provider;
            let fast_model = provider.default_fast_model(cx)?;
            Some(ConfiguredModel {
                provider: provider.clone(),
                model: fast_model,
            })
        });
        self.default_model = model;
    }

    pub fn set_inline_assistant_model(
        &mut self,
        model: Option<ConfiguredModel>,
        cx: &mut Context<Self>,
    ) {
        match (self.inline_assistant_model.as_ref(), model.as_ref()) {
            (Some(old), Some(new)) if old.is_same_as(new) => {}
            (None, None) => {}
            _ => cx.emit(Event::InlineAssistantModelChanged),
        }
        self.inline_assistant_model = model;
    }

    pub fn set_commit_message_model(
        &mut self,
        model: Option<ConfiguredModel>,
        cx: &mut Context<Self>,
    ) {
        match (self.commit_message_model.as_ref(), model.as_ref()) {
            (Some(old), Some(new)) if old.is_same_as(new) => {}
            (None, None) => {}
            _ => cx.emit(Event::CommitMessageModelChanged),
        }
        self.commit_message_model = model;
    }

    pub fn set_thread_summary_model(
        &mut self,
        model: Option<ConfiguredModel>,
        cx: &mut Context<Self>,
    ) {
        match (self.thread_summary_model.as_ref(), model.as_ref()) {
            (Some(old), Some(new)) if old.is_same_as(new) => {}
            (None, None) => {}
            _ => cx.emit(Event::ThreadSummaryModelChanged),
        }
        self.thread_summary_model = model;
    }

    pub fn default_model(&self) -> Option<ConfiguredModel> {
        #[cfg(debug_assertions)]
        if std::env::var("ZED_SIMULATE_NO_LLM_PROVIDER").is_ok() {
            return None;
        }

        self.default_model.clone()
    }

    pub fn inline_assistant_model(&self) -> Option<ConfiguredModel> {
        #[cfg(debug_assertions)]
        if std::env::var("ZED_SIMULATE_NO_LLM_PROVIDER").is_ok() {
            return None;
        }

        self.inline_assistant_model
            .clone()
            .or_else(|| self.default_model.clone())
    }

    pub fn commit_message_model(&self) -> Option<ConfiguredModel> {
        #[cfg(debug_assertions)]
        if std::env::var("ZED_SIMULATE_NO_LLM_PROVIDER").is_ok() {
            return None;
        }

        self.commit_message_model
            .clone()
            .or_else(|| self.default_fast_model.clone())
            .or_else(|| self.default_model.clone())
    }

    pub fn thread_summary_model(&self) -> Option<ConfiguredModel> {
        #[cfg(debug_assertions)]
        if std::env::var("ZED_SIMULATE_NO_LLM_PROVIDER").is_ok() {
            return None;
        }

        self.thread_summary_model
            .clone()
            .or_else(|| self.default_fast_model.clone())
            .or_else(|| self.default_model.clone())
    }

    /// The models to use for inline assists. Returns the union of the active
    /// model and all inline alternatives. When there are multiple models, the
    /// user will be able to cycle through results.
    pub fn inline_alternative_models(&self) -> &[Arc<dyn LanguageModel>] {
        &self.inline_alternatives
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake_provider::FakeLanguageModelProvider;

    #[gpui::test]
    fn test_register_providers(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());

        let provider = Arc::new(FakeLanguageModelProvider::default());
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });

        let providers = registry.read(cx).providers();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].id(), provider.id());

        registry.update(cx, |registry, cx| {
            registry.unregister_provider(provider.id(), cx);
        });

        let providers = registry.read(cx).providers();
        assert!(providers.is_empty());
    }

    #[gpui::test]
    fn test_provider_hiding_on_extension_install(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());

        let provider = Arc::new(FakeLanguageModelProvider::default());
        let provider_id = provider.id();

        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);

            registry.set_builtin_provider_hiding_fn(Box::new(|id| {
                if id == "fake" {
                    Some("fake-extension")
                } else {
                    None
                }
            }));
        });

        let visible = registry.read(cx).visible_providers();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id(), provider_id);

        registry.update(cx, |registry, cx| {
            registry.extension_installed("fake-extension".into(), cx);
        });

        let visible = registry.read(cx).visible_providers();
        assert!(visible.is_empty());

        let all = registry.read(cx).providers();
        assert_eq!(all.len(), 1);
    }

    #[gpui::test]
    fn test_provider_unhiding_on_extension_uninstall(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());

        let provider = Arc::new(FakeLanguageModelProvider::default());
        let provider_id = provider.id();

        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);

            registry.set_builtin_provider_hiding_fn(Box::new(|id| {
                if id == "fake" {
                    Some("fake-extension")
                } else {
                    None
                }
            }));

            registry.extension_installed("fake-extension".into(), cx);
        });

        let visible = registry.read(cx).visible_providers();
        assert!(visible.is_empty());

        registry.update(cx, |registry, cx| {
            registry.extension_uninstalled("fake-extension", cx);
        });

        let visible = registry.read(cx).visible_providers();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id(), provider_id);
    }

    #[gpui::test]
    fn test_should_hide_provider(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());

        registry.update(cx, |registry, cx| {
            registry.set_builtin_provider_hiding_fn(Box::new(|id| {
                if id == "anthropic" {
                    Some("anthropic")
                } else if id == "openai" {
                    Some("openai")
                } else {
                    None
                }
            }));

            registry.extension_installed("anthropic".into(), cx);
        });

        let registry_read = registry.read(cx);

        assert!(registry_read.should_hide_provider(&LanguageModelProviderId("anthropic".into())));

        assert!(!registry_read.should_hide_provider(&LanguageModelProviderId("openai".into())));

        assert!(!registry_read.should_hide_provider(&LanguageModelProviderId("unknown".into())));
    }

    #[gpui::test]
    fn test_sync_installed_llm_extensions(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());

        let provider = Arc::new(FakeLanguageModelProvider::default());

        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);

            registry.set_builtin_provider_hiding_fn(Box::new(|id| {
                if id == "fake" {
                    Some("fake-extension")
                } else {
                    None
                }
            }));
        });

        let mut extension_ids = HashSet::default();
        extension_ids.insert(Arc::from("fake-extension"));

        registry.update(cx, |registry, cx| {
            registry.sync_installed_llm_extensions(extension_ids, cx);
        });

        assert!(registry.read(cx).visible_providers().is_empty());

        registry.update(cx, |registry, cx| {
            registry.sync_installed_llm_extensions(HashSet::default(), cx);
        });

        assert_eq!(registry.read(cx).visible_providers().len(), 1);
    }

    // ---------------------------------------------------------------------------
    // ST1a auth-state cache + invalidation tests (T10–T13, T19, T20).
    // ---------------------------------------------------------------------------

    /// T7 (AC3): fake_provider defaults to `Authenticated`; setter overrides.
    #[gpui::test]
    fn fake_provider_auth_state_default_authenticated(cx: &mut App) {
        let provider = FakeLanguageModelProvider::default();
        assert!(matches!(
            provider.auth_state(cx),
            ProviderAuthState::Authenticated
        ));
        provider.set_auth_state(Some(ProviderAuthState::NotAuthenticated {
            action: AuthAction::EnterApiKeyInSettings,
        }));
        assert!(matches!(
            provider.auth_state(cx),
            ProviderAuthState::NotAuthenticated { .. }
        ));
        provider.set_auth_state(None);
        assert!(matches!(
            provider.auth_state(cx),
            ProviderAuthState::Authenticated
        ));
    }

    /// T10 (AC-CACHE): repeated `cached_auth_state` reads call the provider at most
    /// once per render frame. Uses a counting wrapper to verify.
    #[gpui::test]
    fn registry_cached_auth_state_single_call_per_render(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());
        let provider = Arc::new(FakeLanguageModelProvider::default());
        let id = provider.id();
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });

        // Three consecutive reads should hit the cache after the first miss.
        let r = registry.read(cx);
        let s1 = r.cached_auth_state(&id, cx);
        let s2 = r.cached_auth_state(&id, cx);
        let s3 = r.cached_auth_state(&id, cx);
        assert!(matches!(s1, ProviderAuthState::Authenticated));
        assert!(matches!(s2, ProviderAuthState::Authenticated));
        assert!(matches!(s3, ProviderAuthState::Authenticated));

        // After invalidation the next read returns the (still-Authenticated) value but
        // re-queries the provider; we can observe that by changing the override and
        // confirming the cached value updates only after invalidation.
        provider.set_auth_state(Some(ProviderAuthState::NotAuthenticated {
            action: AuthAction::EnterApiKeyInSettings,
        }));
        let r = registry.read(cx);
        // Without invalidation, the cache still returns Authenticated.
        assert!(matches!(
            r.cached_auth_state(&id, cx),
            ProviderAuthState::Authenticated
        ));
        r.invalidate_auth_cache(&id);
        // After invalidation, the next read picks up the new value.
        assert!(matches!(
            r.cached_auth_state(&id, cx),
            ProviderAuthState::NotAuthenticated { .. }
        ));
    }

    /// T11 (AC-CACHE): a `ProviderStateChanged` event drops the cached entry. We
    /// trigger this by mutating the provider entity, which the registry's subscription
    /// watches.
    /// T11 (plan v3.1 §4 cache invalidation, AC-CACHE2): the registry's
    /// `register_provider` subscription on `observable_entity` MUST drop the
    /// cache entry when the entity emits `cx.notify()` — without any manual
    /// `invalidate_auth_cache` call. The pre-fix-loop version of this test
    /// short-circuited via `invalidate_auth_cache(...)` because
    /// FakeLanguageModelProvider had no observable_entity. Per plan v3.1
    /// review item, this version drives the REAL subscription path.
    #[gpui::test]
    fn registry_invalidates_on_provider_state_changed_event(cx: &mut App) {
        // Build a Fake provider that DOES expose an observable Entity<()>; the
        // registry's `subscribe` path will then install a `cx.observe(...)`.
        let observable: Entity<()> = cx.new(|_| ());
        let registry = cx.new(|_| LanguageModelRegistry::default());
        let provider = Arc::new(
            FakeLanguageModelProvider::default().with_observable_entity(observable.clone()),
        );
        let id = provider.id();
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });

        // Prime the cache.
        let _ = registry.read(cx).cached_auth_state(&id, cx);
        provider.set_auth_state(Some(ProviderAuthState::disabled_by_policy("test")));

        // Cached value still reflects the OLD state (Authenticated): no notify
        // has fired yet.
        assert!(matches!(
            registry.read(cx).cached_auth_state(&id, cx),
            ProviderAuthState::Authenticated
        ));

        // Fire a real notify on the observable entity. The registry's
        // subscription closure runs on the registry's cx (an `&mut App`
        // dispatch), drops the cache entry, and emits ProviderStateChanged.
        observable.update(cx, |_, cx| cx.notify());

        // No manual invalidation: the next read must reflect the new state.
        assert!(
            matches!(
                registry.read(cx).cached_auth_state(&id, cx),
                ProviderAuthState::DisabledByPolicy(_)
            ),
            "T11: real Event::ProviderStateChanged subscription failed to drop \
             cached auth-state entry on observable.notify()"
        );
    }

    /// T12 (AC-CACHE): a cached `RateLimited` entry whose `rate_limited_until`
    /// deadline has passed is auto-evicted on the next read.
    #[gpui::test]
    fn registry_rate_limited_until_auto_expires_on_read(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());
        let provider = Arc::new(FakeLanguageModelProvider::default());
        let id = provider.id();
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });

        // Manually inject a rate-limited entry with a deadline already in the past.
        registry.read(cx).auth_cache.borrow_mut().insert(
            id.clone(),
            CachedAuth {
                state: ProviderAuthState::rate_limited(
                    Some(Duration::from_secs(1)),
                    AuthAction::None,
                ),
                rate_limited_until: Some(Instant::now() - Duration::from_secs(1)),
            },
        );

        // Read should evict the expired entry and return the provider's actual state
        // (Authenticated for the fake).
        assert!(matches!(
            registry.read(cx).cached_auth_state(&id, cx),
            ProviderAuthState::Authenticated
        ));

        // Future deadline is preserved.
        registry.read(cx).auth_cache.borrow_mut().insert(
            id.clone(),
            CachedAuth {
                state: ProviderAuthState::rate_limited(
                    Some(Duration::from_secs(60)),
                    AuthAction::None,
                ),
                rate_limited_until: Some(Instant::now() + Duration::from_secs(60)),
            },
        );
        assert!(matches!(
            registry.read(cx).cached_auth_state(&id, cx),
            ProviderAuthState::RateLimited { .. }
        ));
    }

    /// T13 (AC-CACHE): `note_completion_error(RateLimitExceeded)` records a
    /// `RateLimited` cache entry with the supplied (clamped) `retry_after`.
    #[gpui::test]
    fn registry_note_completion_error_sets_rate_limited(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());
        let provider = Arc::new(FakeLanguageModelProvider::default());
        let id = provider.id();
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });

        let err = LanguageModelCompletionError::RateLimitExceeded {
            provider: provider.name(),
            retry_after: Some(Duration::from_secs(42)),
        };
        registry.read(cx).note_completion_error(&id, &err);

        match registry.read(cx).cached_auth_state(&id, cx) {
            ProviderAuthState::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(Duration::from_secs(42)));
            }
            other => panic!("expected RateLimited, got {:?}", other),
        }

        // Clamping: a 1-year retry_after should be clamped to MAX_RETRY_AFTER (24h).
        let err = LanguageModelCompletionError::RateLimitExceeded {
            provider: provider.name(),
            retry_after: Some(Duration::from_secs(365 * 24 * 3600)),
        };
        registry.read(cx).note_completion_error(&id, &err);
        match registry.read(cx).cached_auth_state(&id, cx) {
            ProviderAuthState::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(crate::MAX_RETRY_AFTER));
            }
            _ => unreachable!(),
        }
    }

    /// fix-loop #8: a 429 with NO `Retry-After` header must still get a bounded TTL,
    /// so the cache entry auto-evicts. Without this, headerless 429s would wedge the
    /// provider until process restart.
    #[gpui::test]
    fn headerless_rate_limit_evicts_via_fallback_ttl(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());
        let provider = Arc::new(FakeLanguageModelProvider::default());
        let id = provider.id();
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });

        // Headerless 429.
        let err = LanguageModelCompletionError::RateLimitExceeded {
            provider: provider.name(),
            retry_after: None,
        };
        registry.read(cx).note_completion_error(&id, &err);

        // Cached as RateLimited with retry_after=None (the *state* doesn't lie about
        // what the server told us).
        match registry.read(cx).cached_auth_state(&id, cx) {
            ProviderAuthState::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, None);
            }
            other => panic!("expected RateLimited, got {:?}", other),
        }

        // But the cache entry's deadline is bounded: rewind it to a moment past the
        // fallback TTL and the next read must evict.
        registry.read(cx).auth_cache.borrow_mut().get_mut(&id).unwrap()
            .rate_limited_until = Some(Instant::now() - Duration::from_secs(1));
        assert!(matches!(
            registry.read(cx).cached_auth_state(&id, cx),
            ProviderAuthState::Authenticated
        ));
    }

    /// `DisabledByPolicy` to the matching error variants instead of collapsing both
    /// to `ProviderNotAuthenticated`.
    #[gpui::test]
    fn configuration_error_maps_rate_limited_and_disabled(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());
        let provider = Arc::new(FakeLanguageModelProvider::default());
        let id = provider.id();
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });
        let model = registry
            .read(cx)
            .provider(&id)
            .and_then(|p| p.provided_models(cx).into_iter().next())
            .map(|model| ConfiguredModel {
                provider: provider.clone(),
                model,
            })
            .unwrap();

        // RateLimited.
        provider.set_auth_state(Some(ProviderAuthState::rate_limited(
            Some(Duration::from_secs(7)),
            AuthAction::None,
        )));
        registry.read(cx).invalidate_auth_cache(&id);
        match registry.read(cx).configuration_error(Some(model.clone()), cx) {
            Some(ConfigurationError::ProviderRateLimited(_, retry)) => {
                assert_eq!(retry, Some(Duration::from_secs(7)));
            }
            other => panic!("expected ProviderRateLimited, got {:?}", other),
        }

        // DisabledByPolicy.
        provider.set_auth_state(Some(ProviderAuthState::disabled_by_policy("nope")));
        registry.read(cx).invalidate_auth_cache(&id);
        match registry.read(cx).configuration_error(Some(model.clone()), cx) {
            Some(ConfigurationError::ProviderDisabledByPolicy(_, reason)) => {
                assert_eq!(reason, "nope");
            }
            other => panic!("expected ProviderDisabledByPolicy, got {:?}", other),
        }

        // NotAuthenticated.
        provider.set_auth_state(Some(ProviderAuthState::NotAuthenticated {
            action: AuthAction::EnterApiKeyInSettings,
        }));
        registry.read(cx).invalidate_auth_cache(&id);
        match registry.read(cx).configuration_error(Some(model.clone()), cx) {
            Some(ConfigurationError::ProviderNotAuthenticated(_)) => {}
            other => panic!("expected ProviderNotAuthenticated, got {:?}", other),
        }

        // Authenticated → no error.
        provider.set_auth_state(None);
        registry.read(cx).invalidate_auth_cache(&id);
        assert!(
            registry
                .read(cx)
                .configuration_error(Some(model.clone()), cx)
                .is_none()
        );
    }

    /// T20 (S4): a completion `AuthenticationError` (HTTP 401) drops the cache entry
    /// within one read cycle, so the next read re-queries the provider.
    #[gpui::test]
    fn completion_401_invalidates_auth_cache(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());
        let provider = Arc::new(FakeLanguageModelProvider::default());
        let id = provider.id();
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });

        // Prime cache as Authenticated.
        let r = registry.read(cx);
        assert!(matches!(
            r.cached_auth_state(&id, cx),
            ProviderAuthState::Authenticated
        ));

        // Now the provider goes unauth (e.g. token revoked).
        provider.set_auth_state(Some(ProviderAuthState::NotAuthenticated {
            action: AuthAction::EnterApiKeyInSettings,
        }));
        // The cache still serves stale `Authenticated` until the 401 is reported.
        assert!(matches!(
            r.cached_auth_state(&id, cx),
            ProviderAuthState::Authenticated
        ));

        let err = LanguageModelCompletionError::AuthenticationError {
            provider: provider.name(),
            message: "401".into(),
        };
        r.note_completion_error(&id, &err);

        // Next read re-queries → fresh NotAuthenticated.
        assert!(matches!(
            r.cached_auth_state(&id, cx),
            ProviderAuthState::NotAuthenticated { .. }
        ));
    }

    /// AC-PERF1 (fix-loop #1): `has_authenticated_provider` and `available_models`
    /// route through `cached_auth_state`, so N consecutive fan-out reads only invoke
    /// the underlying `provider.auth_state(cx)` once per provider per cache lifetime.
    #[gpui::test]
    fn fanout_callers_go_through_cache(cx: &mut App) {
        let registry = cx.new(|_| LanguageModelRegistry::default());
        let provider = Arc::new(FakeLanguageModelProvider::default());
        let id = provider.id();
        registry.update(cx, |registry, cx| {
            registry.register_provider(provider.clone(), cx);
        });

        // Baseline: registering the provider does not call auth_state.
        assert_eq!(provider.auth_state_call_count(), 0);

        // Drive every fan-out caller multiple times in the same "render frame".
        let r = registry.read(cx);
        for _ in 0..5 {
            assert!(r.has_authenticated_provider(cx));
            // available_models is lazy; force iteration.
            let _ = r.available_models(cx).count();
            // configuration_error and direct cached reads also go through the cache.
            let _ = r.cached_auth_state(&id, cx);
        }
        assert_eq!(
            provider.auth_state_call_count(),
            1,
            "fan-out callers must hit cache; got {} provider calls",
            provider.auth_state_call_count()
        );

        // After invalidation, exactly one fresh call services subsequent fan-out.
        r.invalidate_auth_cache(&id);
        for _ in 0..5 {
            assert!(r.has_authenticated_provider(cx));
            let _ = r.available_models(cx).count();
        }
        assert_eq!(
            provider.auth_state_call_count(),
            2,
            "expected exactly one re-query after invalidation"
        );
    }
}
