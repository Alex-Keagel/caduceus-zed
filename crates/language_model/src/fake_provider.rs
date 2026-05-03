use crate::{
    AuthenticateError, ConfigurationViewTargetAgent, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, LanguageModelToolChoice, ProviderAuthState,
};
use anyhow::anyhow;
use futures::{FutureExt, channel::mpsc, future::BoxFuture, stream::BoxStream, stream::StreamExt};
use gpui::{AnyView, App, AsyncApp, Entity, Task, Window};
use http_client::Result;
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering::SeqCst},
};

#[derive(Clone)]
pub struct FakeLanguageModelProvider {
    id: LanguageModelProviderId,
    name: LanguageModelProviderName,
    models: Vec<Arc<dyn LanguageModel>>,
    /// Override `auth_state()` for tests that need a non-`Authenticated` value. Wrapped in
    /// `Arc<Mutex<>>` so `&self`-only `auth_state(cx)` can read it and tests can mutate via
    /// `set_auth_state`.
    auth_state_override: Arc<Mutex<Option<ProviderAuthState>>>,
    /// Counts every call to `auth_state(cx)`. Used by registry cache tests
    /// (AC-PERF1) to prove a fan-out render frame causes ≤1 underlying call.
    auth_state_call_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Optional real `Entity<()>` so `observable_entity()` can return `Some(_)`.
    /// Tests that need to drive the registry's `Event::ProviderStateChanged`
    /// subscription path (T11) install one via `with_observable_entity`.
    observable: Option<Entity<()>>,
}

impl Default for FakeLanguageModelProvider {
    fn default() -> Self {
        Self {
            id: LanguageModelProviderId::from("fake".to_string()),
            name: LanguageModelProviderName::from("Fake".to_string()),
            models: vec![Arc::new(FakeLanguageModel::default())],
            auth_state_override: Arc::new(Mutex::new(None)),
            auth_state_call_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            observable: None,
        }
    }
}

impl LanguageModelProviderState for FakeLanguageModelProvider {
    type ObservableEntity = ();

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        self.observable.clone()
    }
}

impl LanguageModelProvider for FakeLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelProviderName {
        self.name.clone()
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.models.first().cloned()
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.models.first().cloned()
    }

    fn provided_models(&self, _: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.models.clone()
    }

    fn auth_state(&self, _: &App) -> ProviderAuthState {
        self.auth_state_call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.auth_state_override
            .lock()
            .clone()
            .unwrap_or(ProviderAuthState::Authenticated)
    }

    fn authenticate(&self, _: &mut App) -> Task<Result<(), AuthenticateError>> {
        Task::ready(Ok(()))
    }

    fn configuration_view(
        &self,
        _target_agent: ConfigurationViewTargetAgent,
        _window: &mut Window,
        _: &mut App,
    ) -> AnyView {
        unimplemented!()
    }

    fn reset_credentials(&self, _: &mut App) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }
}

impl FakeLanguageModelProvider {
    pub fn new(id: LanguageModelProviderId, name: LanguageModelProviderName) -> Self {
        Self {
            id,
            name,
            models: vec![Arc::new(FakeLanguageModel::default())],
            auth_state_override: Arc::new(Mutex::new(None)),
            auth_state_call_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            observable: None,
        }
    }

    /// Install an observable entity so the registry's `subscribe` path can fire
    /// when the entity emits `cx.notify()`. Used by T11 to exercise the real
    /// `Event::ProviderStateChanged` path (rather than calling
    /// `invalidate_auth_cache` manually).
    pub fn with_observable_entity(mut self, entity: Entity<()>) -> Self {
        self.observable = Some(entity);
        self
    }

    pub fn with_models(mut self, models: Vec<Arc<dyn LanguageModel>>) -> Self {
        self.models = models;
        self
    }

    /// Override the value returned by `auth_state(cx)`. Pass `None` to revert to the
    /// `Authenticated` default. Used by tests that exercise the non-authenticated paths.
    pub fn set_auth_state(&self, state: Option<ProviderAuthState>) {
        *self.auth_state_override.lock() = state;
    }

    /// Number of times `auth_state(cx)` has been called. Used by the registry's
    /// AC-PERF1 cache test to verify fan-out callers go through `cached_auth_state`.
    pub fn auth_state_call_count(&self) -> usize {
        self.auth_state_call_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn test_model(&self) -> FakeLanguageModel {
        FakeLanguageModel::default()
    }
}

#[derive(Debug, PartialEq)]
pub struct ToolUseRequest {
    pub request: LanguageModelRequest,
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
}

pub struct FakeLanguageModel {
    id: LanguageModelId,
    name: LanguageModelName,
    provider_id: LanguageModelProviderId,
    provider_name: LanguageModelProviderName,
    current_completion_txs: Mutex<
        Vec<(
            LanguageModelRequest,
            mpsc::UnboundedSender<
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
        )>,
    >,
    forbid_requests: AtomicBool,
    supports_thinking: AtomicBool,
    supports_streaming_tools: AtomicBool,
}

impl Default for FakeLanguageModel {
    fn default() -> Self {
        Self {
            id: LanguageModelId::from("fake".to_string()),
            name: LanguageModelName::from("Fake".to_string()),
            provider_id: LanguageModelProviderId::from("fake".to_string()),
            provider_name: LanguageModelProviderName::from("Fake".to_string()),
            current_completion_txs: Mutex::new(Vec::new()),
            forbid_requests: AtomicBool::new(false),
            supports_thinking: AtomicBool::new(false),
            supports_streaming_tools: AtomicBool::new(false),
        }
    }
}

impl FakeLanguageModel {
    pub fn with_id_and_thinking(
        provider_id: &str,
        id: &str,
        name: &str,
        supports_thinking: bool,
    ) -> Self {
        Self {
            id: LanguageModelId::from(id.to_string()),
            name: LanguageModelName::from(name.to_string()),
            provider_id: LanguageModelProviderId::from(provider_id.to_string()),
            supports_thinking: AtomicBool::new(supports_thinking),
            ..Default::default()
        }
    }

    pub fn allow_requests(&self) {
        self.forbid_requests.store(false, SeqCst);
    }

    pub fn forbid_requests(&self) {
        self.forbid_requests.store(true, SeqCst);
    }

    pub fn set_supports_thinking(&self, supports: bool) {
        self.supports_thinking.store(supports, SeqCst);
    }

    pub fn set_supports_streaming_tools(&self, supports: bool) {
        self.supports_streaming_tools.store(supports, SeqCst);
    }

    pub fn pending_completions(&self) -> Vec<LanguageModelRequest> {
        self.current_completion_txs
            .lock()
            .iter()
            .map(|(request, _)| request.clone())
            .collect()
    }

    pub fn completion_count(&self) -> usize {
        self.current_completion_txs.lock().len()
    }

    pub fn send_completion_stream_text_chunk(
        &self,
        request: &LanguageModelRequest,
        chunk: impl Into<String>,
    ) {
        self.send_completion_stream_event(
            request,
            LanguageModelCompletionEvent::Text(chunk.into()),
        );
    }

    pub fn send_completion_stream_event(
        &self,
        request: &LanguageModelRequest,
        event: impl Into<LanguageModelCompletionEvent>,
    ) {
        let current_completion_txs = self.current_completion_txs.lock();
        let tx = current_completion_txs
            .iter()
            .find(|(req, _)| req == request)
            .map(|(_, tx)| tx)
            .unwrap();
        tx.unbounded_send(Ok(event.into())).unwrap();
    }

    pub fn send_completion_stream_error(
        &self,
        request: &LanguageModelRequest,
        error: impl Into<LanguageModelCompletionError>,
    ) {
        let current_completion_txs = self.current_completion_txs.lock();
        let tx = current_completion_txs
            .iter()
            .find(|(req, _)| req == request)
            .map(|(_, tx)| tx)
            .unwrap();
        tx.unbounded_send(Err(error.into())).unwrap();
    }

    pub fn end_completion_stream(&self, request: &LanguageModelRequest) {
        self.current_completion_txs
            .lock()
            .retain(|(req, _)| req != request);
    }

    pub fn send_last_completion_stream_text_chunk(&self, chunk: impl Into<String>) {
        self.send_completion_stream_text_chunk(self.pending_completions().last().unwrap(), chunk);
    }

    pub fn send_last_completion_stream_event(
        &self,
        event: impl Into<LanguageModelCompletionEvent>,
    ) {
        self.send_completion_stream_event(self.pending_completions().last().unwrap(), event);
    }

    pub fn send_last_completion_stream_error(
        &self,
        error: impl Into<LanguageModelCompletionError>,
    ) {
        self.send_completion_stream_error(self.pending_completions().last().unwrap(), error);
    }

    pub fn end_last_completion_stream(&self) {
        self.end_completion_stream(self.pending_completions().last().unwrap());
    }
}

impl LanguageModel for FakeLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        self.name.clone()
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        self.provider_id.clone()
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        self.provider_name.clone()
    }

    fn supports_tools(&self) -> bool {
        false
    }

    fn supports_tool_choice(&self, _choice: LanguageModelToolChoice) -> bool {
        false
    }

    fn supports_images(&self) -> bool {
        false
    }

    fn supports_thinking(&self) -> bool {
        self.supports_thinking.load(SeqCst)
    }

    fn supports_streaming_tools(&self) -> bool {
        self.supports_streaming_tools.load(SeqCst)
    }

    fn telemetry_id(&self) -> String {
        "fake".to_string()
    }

    fn max_token_count(&self) -> u64 {
        1000000
    }

    fn count_tokens(&self, _: LanguageModelRequest, _: &App) -> BoxFuture<'static, Result<u64>> {
        futures::future::ready(Ok(0)).boxed()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        _: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        if self.forbid_requests.load(SeqCst) {
            async move {
                Err(LanguageModelCompletionError::Other(anyhow!(
                    "requests are forbidden"
                )))
            }
            .boxed()
        } else {
            let (tx, rx) = mpsc::unbounded();
            self.current_completion_txs.lock().push((request, tx));
            async move { Ok(rx.boxed()) }.boxed()
        }
    }

    fn as_fake(&self) -> &Self {
        self
    }
}
