//! Bridges Zed's `LanguageModel` trait to the
//! [`caduceus_providers::LlmAdapter`] trait expected by
//! `caduceus_orchestrator::Harness`.
//!
//! Architecture
//! ============
//!
//! Zed language models require `&AsyncApp` to call `stream_completion`,
//! which means they are `!Send`-bound to the GPUI foreground. The
//! caduceus `LlmAdapter` trait is `async fn(&self, req)` with no
//! context parameter. We bridge this the same way
//! [`super::caduceus_tool_adapter`] bridges tools: a bounded mpsc
//! carries requests to a GPUI-foreground task that owns the
//! `Arc<dyn LanguageModel>` plus an `AsyncApp`. The task translates
//! request/response shapes and forwards token-level events.
//!
//! Scope (G1d-provider-adapter)
//! ============================
//! * Text streaming: `LanguageModelCompletionEvent::Text` → provider
//!   `StreamChunk { delta, is_final=false }`.
//! * `UsageUpdate` updates cached token counts; flushed on the final
//!   chunk.
//! * `Stop` is translated to `caduceus_core::StopReason` (Refusal →
//!   `StopReason::Error`).
//! * Tool calls: extracted from the event stream and returned on
//!   `ChatResponse.tool_calls`. The streaming adapter contract
//!   (`StreamChunk`) has no tool_call channel, so tool calls are
//!   *only* observable via `chat()`. This matches how the existing
//!   orchestrator code handles streaming (see
//!   `caduceus_orchestrator::lib::try_stream_or_chat`).
//! * Thinking deltas are dropped in streaming mode (no corresponding
//!   field on `StreamChunk`) and folded into the aggregated content
//!   in non-streaming mode.
//! * Images on tool results are mapped as a text placeholder since
//!   caduceus `ToolResult` carries only `String` content today.
//! * List-models returns `Ok(vec![])` — the Zed model registry owns
//!   which models are visible; the orchestrator never enumerates this
//!   list.
//!
//! Tests focus on the pure translation functions. End-to-end
//! dispatcher flow is exercised by a minimal
//! `#[tokio::test]` that drives the inner aggregation directly.

use async_trait::async_trait;
use caduceus_core::{CaduceusError, ProviderId, StopReason, ToolUse};
use caduceus_providers::{
    ChatRequest, ChatResponse, LlmAdapter, Message, MessageContentBlock, StreamChunk, StreamResult,
    ToolChoice,
};
use futures::{
    Stream, StreamExt,
    channel::{mpsc, oneshot},
    stream::BoxStream,
};
use language_model::{
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelRequest,
    LanguageModelRequestMessage, LanguageModelRequestTool, LanguageModelToolChoice,
    LanguageModelToolResult, LanguageModelToolResultContent, LanguageModelToolUse, MessageContent,
    Role, StopReason as LmStopReason,
};
use std::sync::Arc;
use std::task::{Context as TaskCtx, Poll};
use std::{future::Future, pin::Pin};

// ──────────────────────────────────────────────────────────────────
// Public surface
// ──────────────────────────────────────────────────────────────────

/// `LlmAdapter` implementation backed by a Zed `Arc<dyn LanguageModel>`.
///
/// Cheap to clone; all cost lives in the dispatcher task reached
/// through [`DispatcherHandle`].
pub struct ZedLlmAdapter {
    provider_id: ProviderId,
    dispatcher: DispatcherHandle,
}

impl ZedLlmAdapter {
    pub fn new(provider_id: ProviderId, dispatcher: DispatcherHandle) -> Self {
        Self {
            provider_id,
            dispatcher,
        }
    }
}

#[async_trait]
impl LlmAdapter for ZedLlmAdapter {
    fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    async fn chat(&self, request: ChatRequest) -> caduceus_core::Result<ChatResponse> {
        let (tx, rx) = oneshot::channel();
        self.dispatcher
            .send(DispatchRequest::Chat {
                request,
                respond: tx,
            })
            .await?;
        rx.await
            .map_err(|_| CaduceusError::Provider("provider dispatcher closed".into()))?
    }

    async fn stream(&self, request: ChatRequest) -> caduceus_core::Result<StreamResult> {
        let (tx, rx) = oneshot::channel();
        self.dispatcher
            .send(DispatchRequest::Stream {
                request,
                respond: tx,
            })
            .await?;
        rx.await
            .map_err(|_| CaduceusError::Provider("provider dispatcher closed".into()))?
    }

    async fn list_models(&self) -> caduceus_core::Result<Vec<caduceus_core::ModelId>> {
        // The Zed registry owns the visible model list; the
        // orchestrator never calls this.
        Ok(Vec::new())
    }
}

/// Handle over a bounded mpsc that the adapter uses to reach the
/// GPUI-foreground dispatcher task.
#[derive(Clone)]
pub struct DispatcherHandle {
    tx: mpsc::Sender<DispatchRequest>,
}

impl DispatcherHandle {
    async fn send(&self, request: DispatchRequest) -> caduceus_core::Result<()> {
        let mut tx = self.tx.clone();
        use futures::SinkExt;
        tx.send(request)
            .await
            .map_err(|_| CaduceusError::Provider("provider dispatcher closed".into()))
    }
}

/// Construct a `(DispatcherHandle, Receiver)` pair so production
/// callers can spawn the dispatcher on their own GPUI executor
/// (`cx.spawn`) — whose spawned future is not `Send`. For test /
/// tokio-style contexts, use [`spawn_dispatcher`].
pub fn dispatcher_channel() -> (DispatcherHandle, mpsc::Receiver<DispatchRequest>) {
    let (tx, rx) = mpsc::channel::<DispatchRequest>(DISPATCH_CHANNEL_BOUND);
    (DispatcherHandle { tx }, rx)
}

/// Drain a dispatcher-receiver, servicing each request by invoking
/// `call_stream` and forwarding the translated outcome. Intended for
/// production use in `cx.spawn(async move |cx| { drive_dispatcher(rx,
/// call_stream).await })`: the future does NOT need to be `Send`, so
/// `call_stream` may capture an `AsyncApp` clone.
pub async fn drive_dispatcher_fn<F, Fut>(mut rx: mpsc::Receiver<DispatchRequest>, call_stream: F)
where
    F: Fn(LanguageModelRequest) -> Fut,
    Fut: Future<
        Output = Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    >,
{
    while let Some(req) = rx.next().await {
        match req {
            DispatchRequest::Chat { request, respond } => {
                let lm_req = translate_chat_request(&request);
                let result = match call_stream(lm_req).await {
                    Ok(events) => aggregate_stream(events).await,
                    Err(e) => Err(classify_error(&e)),
                };
                let _ = respond.send(result);
            }
            DispatchRequest::Stream { request, respond } => {
                let lm_req = translate_chat_request(&request);
                let result: caduceus_core::Result<StreamResult> = match call_stream(lm_req).await {
                    Ok(events) => Ok(Box::pin(stream_to_chunks(events)) as StreamResult),
                    Err(e) => Err(classify_error(&e)),
                };
                let _ = respond.send(result);
            }
        }
    }
}

pub enum DispatchRequest {
    Chat {
        request: ChatRequest,
        respond: oneshot::Sender<caduceus_core::Result<ChatResponse>>,
    },
    Stream {
        request: ChatRequest,
        respond: oneshot::Sender<caduceus_core::Result<StreamResult>>,
    },
}

/// Default bounded-channel depth — matches `caduceus_tool_adapter`.
pub const DISPATCH_CHANNEL_BOUND: usize = 64;

// ──────────────────────────────────────────────────────────────────
// Dispatcher driver
// ──────────────────────────────────────────────────────────────────

/// Spawn a GPUI-foreground dispatcher for `model`. `spawn_fn` is the
/// foreground spawner — in production `cx.spawn(async move { … })`,
/// in tests `tokio::spawn(async move { … })`.
///
/// The dispatcher owns the language model and an `AsyncApp` supplied
/// by the caller via `call_stream`. Currently `call_stream` is a
/// closure because the adapter can't reach into `AsyncApp` itself;
/// callers inject it.
pub fn spawn_dispatcher<Spawn, Fut, CallStream>(
    spawn_fn: Spawn,
    call_stream: CallStream,
) -> DispatcherHandle
where
    Spawn: FnOnce(Pin<Box<dyn Future<Output = ()> + Send>>) -> Fut + Send + 'static,
    Fut: Send + 'static,
    CallStream: Fn(
            LanguageModelRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            BoxStream<
                                'static,
                                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
                            >,
                            LanguageModelCompletionError,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync
        + 'static,
{
    let (tx, rx) = mpsc::channel::<DispatchRequest>(DISPATCH_CHANNEL_BOUND);
    let call_stream = Arc::new(call_stream);
    let fut: Pin<Box<dyn Future<Output = ()> + Send>> = Box::pin(drive_dispatcher(rx, call_stream));
    let _handle = spawn_fn(fut);
    DispatcherHandle { tx }
}

async fn drive_dispatcher<CallStream>(
    mut rx: mpsc::Receiver<DispatchRequest>,
    call_stream: Arc<CallStream>,
) where
    CallStream: Fn(
            LanguageModelRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            BoxStream<
                                'static,
                                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
                            >,
                            LanguageModelCompletionError,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync
        + 'static,
{
    while let Some(req) = rx.next().await {
        match req {
            DispatchRequest::Chat { request, respond } => {
                let lm_req = translate_chat_request(&request);
                let stream_fut = (call_stream)(lm_req);
                let result = match stream_fut.await {
                    Ok(events) => aggregate_stream(events).await,
                    Err(e) => Err(classify_error(&e)),
                };
                let _ = respond.send(result);
            }
            DispatchRequest::Stream { request, respond } => {
                let lm_req = translate_chat_request(&request);
                let stream_fut = (call_stream)(lm_req);
                let result: caduceus_core::Result<StreamResult> = match stream_fut.await {
                    Ok(events) => Ok(Box::pin(stream_to_chunks(events)) as StreamResult),
                    Err(e) => Err(classify_error(&e)),
                };
                let _ = respond.send(result);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Pure translators — request
// ──────────────────────────────────────────────────────────────────

pub(crate) fn translate_chat_request(req: &ChatRequest) -> LanguageModelRequest {
    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    if let Some(system) = req.system.as_ref() {
        messages.push(LanguageModelRequestMessage {
            role: Role::System,
            content: vec![MessageContent::Text(system.clone())],
            cache: false,
            reasoning_details: None,
        });
    }
    for m in &req.messages {
        messages.push(translate_message(m));
    }

    let tools = req
        .tools
        .iter()
        .map(|t| LanguageModelRequestTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
            use_input_streaming: false,
        })
        .collect();

    let tool_choice = req.tool_choice.as_ref().and_then(translate_tool_choice);

    LanguageModelRequest {
        thread_id: None,
        prompt_id: None,
        intent: None,
        messages,
        tools,
        tool_choice,
        stop: Vec::new(),
        temperature: req.temperature,
        thinking_allowed: req.thinking_mode,
        ..Default::default()
    }
}

fn translate_tool_choice(tc: &ToolChoice) -> Option<LanguageModelToolChoice> {
    match tc {
        ToolChoice::Auto => Some(LanguageModelToolChoice::Auto),
        ToolChoice::Required => Some(LanguageModelToolChoice::Any),
        ToolChoice::None => Some(LanguageModelToolChoice::None),
        // Zed's trait has no "specific tool" variant; fall back to Any
        // so the model is at least forced to call *a* tool.
        ToolChoice::Specific(_) => Some(LanguageModelToolChoice::Any),
    }
}

fn translate_message(m: &Message) -> LanguageModelRequestMessage {
    let role = match m.role.as_str() {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        _ => Role::User,
    };

    let mut content = Vec::new();

    // Prefer structured content_blocks when provided; fall back to
    // the flat `content` string.
    if let Some(blocks) = &m.content_blocks {
        for b in blocks {
            match b {
                MessageContentBlock::Text { text, .. } => {
                    content.push(MessageContent::Text(text.clone()));
                }
                MessageContentBlock::Image { .. } => {
                    // Image translation deferred; upstream crates
                    // (caduceus_tool_adapter / tool results) are the
                    // primary carriers of image data today.
                }
            }
        }
    } else if !m.content.is_empty() {
        content.push(MessageContent::Text(m.content.clone()));
    }

    // Assistant-initiated tool calls on the message, if any.
    for tu in &m.tool_calls {
        content.push(MessageContent::ToolUse(LanguageModelToolUse {
            id: tu.id.clone().into(),
            name: tu.name.clone().into(),
            raw_input: tu.input.to_string(),
            input: tu.input.clone(),
            is_input_complete: true,
            thought_signature: None,
        }));
    }

    // Tool result payloads on user-role messages.
    if let Some(tr) = &m.tool_result {
        if let Some(id) = &tr.tool_use_id {
            content.push(MessageContent::ToolResult(LanguageModelToolResult {
                tool_use_id: id.clone().into(),
                tool_name: Arc::from(""),
                is_error: tr.is_error,
                content: LanguageModelToolResultContent::Text(Arc::from(tr.content.as_str())),
                output: None,
            }));
        }
    }

    LanguageModelRequestMessage {
        role,
        content,
        cache: m.cache_breakpoint,
        reasoning_details: None,
    }
}

// ──────────────────────────────────────────────────────────────────
// Pure translators — response
// ──────────────────────────────────────────────────────────────────

pub(crate) fn translate_stop_reason(s: LmStopReason) -> StopReason {
    match s {
        LmStopReason::EndTurn => StopReason::EndTurn,
        LmStopReason::MaxTokens => StopReason::MaxTokens,
        LmStopReason::ToolUse => StopReason::ToolUse,
        LmStopReason::Refusal => StopReason::Error,
    }
}

pub(crate) fn translate_tool_use(lm: &LanguageModelToolUse) -> ToolUse {
    ToolUse {
        id: lm.id.to_string(),
        name: lm.name.to_string(),
        input: lm.input.clone(),
    }
}

pub(crate) fn classify_error(e: &LanguageModelCompletionError) -> CaduceusError {
    use LanguageModelCompletionError as E;
    match e {
        E::PromptTooLarge { tokens } => CaduceusError::ContextOverflow {
            used: tokens.unwrap_or(0).try_into().unwrap_or(u32::MAX),
            limit: 0,
        },
        E::RateLimitExceeded { retry_after, .. } => CaduceusError::RateLimited {
            retry_after_secs: retry_after
                .as_ref()
                .map(|d| d.as_secs())
                .unwrap_or_default(),
        },
        other => CaduceusError::Provider(other.to_string()),
    }
}

/// Collapse a completion-event stream into a single `ChatResponse`.
/// Text + thinking content is concatenated into `content`; tool calls
/// are collected; the last `UsageUpdate` wins; `Stop` determines the
/// stop reason (default `EndTurn`).
pub(crate) async fn aggregate_stream(
    mut events: BoxStream<
        'static,
        Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
    >,
) -> caduceus_core::Result<ChatResponse> {
    let mut content = String::new();
    let mut tool_calls: Vec<ToolUse> = Vec::new();
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;
    let mut cache_read: u64 = 0;
    let mut cache_create: u64 = 0;
    let mut stop = StopReason::EndTurn;

    while let Some(ev) = events.next().await {
        match ev {
            Err(e) => return Err(classify_error(&e)),
            Ok(LanguageModelCompletionEvent::Text(s)) => content.push_str(&s),
            Ok(LanguageModelCompletionEvent::Thinking { text, .. }) => content.push_str(&text),
            Ok(LanguageModelCompletionEvent::RedactedThinking { .. }) => {
                // Intentionally drop — nothing to show.
            }
            Ok(LanguageModelCompletionEvent::ToolUse(tu)) => {
                if tu.is_input_complete {
                    tool_calls.push(ToolUse {
                        id: tu.id.to_string(),
                        name: tu.name.to_string(),
                        input: tu.input,
                    });
                }
            }
            Ok(LanguageModelCompletionEvent::ToolUseJsonParseError {
                json_parse_error,
                tool_name,
                ..
            }) => {
                return Err(CaduceusError::Provider(format!(
                    "tool `{tool_name}` produced invalid JSON: {json_parse_error}"
                )));
            }
            Ok(LanguageModelCompletionEvent::Stop(s)) => stop = translate_stop_reason(s),
            Ok(LanguageModelCompletionEvent::UsageUpdate(u)) => {
                input_tokens = u.input_tokens;
                output_tokens = u.output_tokens;
                cache_read = u.cache_read_input_tokens;
                cache_create = u.cache_creation_input_tokens;
            }
            Ok(LanguageModelCompletionEvent::StartMessage { .. })
            | Ok(LanguageModelCompletionEvent::Queued { .. })
            | Ok(LanguageModelCompletionEvent::Started)
            | Ok(LanguageModelCompletionEvent::ReasoningDetails(_)) => {}
        }
    }

    if !tool_calls.is_empty() && stop == StopReason::EndTurn {
        // Zed doesn't always emit Stop(ToolUse) when tool calls are
        // present; promote so the harness enters its tool-dispatch
        // branch.
        stop = StopReason::ToolUse;
    }

    Ok(ChatResponse {
        content,
        input_tokens: input_tokens.try_into().unwrap_or(u32::MAX),
        output_tokens: output_tokens.try_into().unwrap_or(u32::MAX),
        cache_read_tokens: cache_read.try_into().unwrap_or(u32::MAX),
        cache_creation_tokens: cache_create.try_into().unwrap_or(u32::MAX),
        stop_reason: stop,
        tool_calls,
        logprobs: None,
    })
}

/// Project a completion-event stream to caduceus `StreamChunk`s.
/// Matches the contract in
/// `caduceus_orchestrator::lib::try_stream_or_chat`: stream is text
/// only, usage fields are filled progressively, `is_final` is set on
/// the last emitted chunk.
pub(crate) fn stream_to_chunks(
    events: BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
) -> impl Stream<Item = caduceus_core::Result<StreamChunk>> + Send + 'static {
    ChunkStream {
        inner: events,
        input_tokens: None,
        output_tokens: None,
        cache_read_tokens: None,
        cache_creation_tokens: None,
        finished: false,
    }
}

struct ChunkStream {
    inner: BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cache_read_tokens: Option<u32>,
    cache_creation_tokens: Option<u32>,
    finished: bool,
}

impl Stream for ChunkStream {
    type Item = caduceus_core::Result<StreamChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskCtx<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.finished {
                return Poll::Ready(None);
            }
            match self.inner.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // Stream ended without an explicit Stop; emit a
                    // final empty chunk so downstream gets its usage
                    // totals and its `is_final=true` boundary.
                    self.finished = true;
                    return Poll::Ready(Some(Ok(StreamChunk {
                        delta: String::new(),
                        is_final: true,
                        input_tokens: self.input_tokens,
                        output_tokens: self.output_tokens,
                        cache_read_tokens: self.cache_read_tokens,
                        cache_creation_tokens: self.cache_creation_tokens,
                    })));
                }
                Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(classify_error(&e))));
                }
                Poll::Ready(Some(Ok(ev))) => match ev {
                    LanguageModelCompletionEvent::Text(delta) => {
                        return Poll::Ready(Some(Ok(StreamChunk {
                            delta,
                            is_final: false,
                            input_tokens: None,
                            output_tokens: None,
                            cache_read_tokens: None,
                            cache_creation_tokens: None,
                        })));
                    }
                    LanguageModelCompletionEvent::UsageUpdate(u) => {
                        self.input_tokens = Some(u.input_tokens.try_into().unwrap_or(u32::MAX));
                        self.output_tokens = Some(u.output_tokens.try_into().unwrap_or(u32::MAX));
                        self.cache_read_tokens =
                            Some(u.cache_read_input_tokens.try_into().unwrap_or(u32::MAX));
                        self.cache_creation_tokens =
                            Some(u.cache_creation_input_tokens.try_into().unwrap_or(u32::MAX));
                        // Don't emit a chunk — just update cached
                        // state, wait for next content event.
                    }
                    LanguageModelCompletionEvent::Stop(_) => {
                        self.finished = true;
                        return Poll::Ready(Some(Ok(StreamChunk {
                            delta: String::new(),
                            is_final: true,
                            input_tokens: self.input_tokens,
                            output_tokens: self.output_tokens,
                            cache_read_tokens: self.cache_read_tokens,
                            cache_creation_tokens: self.cache_creation_tokens,
                        })));
                    }
                    // Silent for thinking/tool-use/queued/etc. in
                    // streaming — text-only per the orchestrator
                    // streaming contract.
                    _ => continue,
                },
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use caduceus_core::ToolSpec;
    use caduceus_providers::ChatRequest as ProvReq;
    use futures::stream;
    use language_model::TokenUsage;

    fn mk_req(messages: Vec<Message>) -> ProvReq {
        ProvReq {
            model: caduceus_core::ModelId("test".into()),
            messages,
            system: None,
            max_tokens: 1024,
            temperature: Some(0.7),
            thinking_mode: false,
            tool_choice: None,
            response_format: None,
            tools: vec![],
            logprobs: None,
        }
    }

    fn mk_user_msg(text: &str) -> Message {
        Message {
            role: "user".into(),
            content: text.into(),
            content_blocks: None,
            tool_calls: vec![],
            tool_result: None,
            cache_breakpoint: false,
        }
    }

    #[test]
    fn translate_request_prepends_system() {
        let mut req = mk_req(vec![mk_user_msg("hello")]);
        req.system = Some("you are helpful".into());
        let lm = translate_chat_request(&req);
        assert_eq!(lm.messages.len(), 2);
        assert_eq!(lm.messages[0].role, Role::System);
        assert_eq!(lm.messages[1].role, Role::User);
        assert_eq!(lm.temperature, Some(0.7));
    }

    #[test]
    fn translate_request_maps_roles_and_tools() {
        let req = ChatRequest {
            model: caduceus_core::ModelId("m".into()),
            messages: vec![
                Message {
                    role: "assistant".into(),
                    content: "ack".into(),
                    content_blocks: None,
                    tool_calls: vec![],
                    tool_result: None,
                    cache_breakpoint: false,
                },
                mk_user_msg("go"),
            ],
            system: None,
            max_tokens: 256,
            temperature: None,
            thinking_mode: true,
            tool_choice: Some(ToolChoice::Required),
            response_format: None,
            tools: vec![ToolSpec {
                name: "grep".into(),
                description: "search".into(),
                input_schema: serde_json::json!({"type":"object"}),
                required_capability: None,
            }],
            logprobs: None,
        };
        let lm = translate_chat_request(&req);
        assert_eq!(lm.messages[0].role, Role::Assistant);
        assert_eq!(lm.messages[1].role, Role::User);
        assert_eq!(lm.tools.len(), 1);
        assert_eq!(lm.tools[0].name, "grep");
        assert!(matches!(lm.tool_choice, Some(LanguageModelToolChoice::Any)));
        assert!(lm.thinking_allowed);
    }

    #[test]
    fn stop_reason_mapping_is_exhaustive() {
        assert_eq!(
            translate_stop_reason(LmStopReason::EndTurn),
            StopReason::EndTurn
        );
        assert_eq!(
            translate_stop_reason(LmStopReason::MaxTokens),
            StopReason::MaxTokens
        );
        assert_eq!(
            translate_stop_reason(LmStopReason::ToolUse),
            StopReason::ToolUse
        );
        assert_eq!(
            translate_stop_reason(LmStopReason::Refusal),
            StopReason::Error
        );
    }

    #[test]
    fn rate_limit_error_is_classified() {
        let e = LanguageModelCompletionError::RateLimitExceeded {
            provider: language_model::LanguageModelProviderName::from("test".to_string()),
            retry_after: Some(std::time::Duration::from_secs(42)),
        };
        match classify_error(&e) {
            CaduceusError::RateLimited { retry_after_secs } => {
                assert_eq!(retry_after_secs, 42);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn prompt_too_large_maps_to_context_overflow() {
        let e = LanguageModelCompletionError::PromptTooLarge { tokens: Some(9999) };
        match classify_error(&e) {
            CaduceusError::ContextOverflow { used, .. } => {
                assert_eq!(used, 9999);
            }
            other => panic!("expected ContextOverflow, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn aggregate_stream_collects_text_usage_and_stop() {
        let events: BoxStream<'static, _> = stream::iter(vec![
            Ok(LanguageModelCompletionEvent::Started),
            Ok(LanguageModelCompletionEvent::Text("hello ".into())),
            Ok(LanguageModelCompletionEvent::Text("world".into())),
            Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: 2,
                cache_creation_input_tokens: 1,
            })),
            Ok(LanguageModelCompletionEvent::Stop(LmStopReason::EndTurn)),
        ])
        .boxed();
        let r = aggregate_stream(events).await.expect("ok");
        assert_eq!(r.content, "hello world");
        assert_eq!(r.input_tokens, 10);
        assert_eq!(r.output_tokens, 5);
        assert_eq!(r.cache_read_tokens, 2);
        assert_eq!(r.cache_creation_tokens, 1);
        assert_eq!(r.stop_reason, StopReason::EndTurn);
        assert!(r.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn aggregate_stream_extracts_tool_calls_and_promotes_stop() {
        let tu = LanguageModelToolUse {
            id: "t1".into(),
            name: Arc::from("grep"),
            raw_input: "{\"q\":\"x\"}".into(),
            input: serde_json::json!({"q":"x"}),
            is_input_complete: true,
            thought_signature: None,
        };
        let events: BoxStream<'static, _> = stream::iter(vec![
            Ok(LanguageModelCompletionEvent::Text("calling grep".into())),
            Ok(LanguageModelCompletionEvent::ToolUse(tu)),
            // No explicit Stop — mapper must promote to ToolUse.
        ])
        .boxed();
        let r = aggregate_stream(events).await.expect("ok");
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "grep");
        assert_eq!(r.stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn aggregate_stream_propagates_error() {
        let events: BoxStream<'static, _> = stream::iter(vec![
            Ok(LanguageModelCompletionEvent::Text("partial".into())),
            Err(LanguageModelCompletionError::PromptTooLarge { tokens: None }),
        ])
        .boxed();
        let err = aggregate_stream(events).await.expect_err("must fail");
        assert!(matches!(err, CaduceusError::ContextOverflow { .. }));
    }

    #[tokio::test]
    async fn aggregate_stream_rejects_malformed_tool_json() {
        let events: BoxStream<'static, _> = stream::iter(vec![Ok(
            LanguageModelCompletionEvent::ToolUseJsonParseError {
                id: "t1".into(),
                tool_name: Arc::from("grep"),
                raw_input: Arc::from("not json"),
                json_parse_error: "expected `{`".into(),
            },
        )])
        .boxed();
        let err = aggregate_stream(events).await.expect_err("must fail");
        assert!(matches!(err, CaduceusError::Provider(ref s) if s.contains("grep")));
    }

    #[tokio::test]
    async fn stream_to_chunks_emits_text_and_final() {
        let events: BoxStream<'static, _> = stream::iter(vec![
            Ok(LanguageModelCompletionEvent::Text("a".into())),
            Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                input_tokens: 3,
                output_tokens: 2,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })),
            Ok(LanguageModelCompletionEvent::Text("b".into())),
            Ok(LanguageModelCompletionEvent::Stop(LmStopReason::EndTurn)),
        ])
        .boxed();
        let mut s = Box::pin(stream_to_chunks(events));
        let mut texts = Vec::new();
        let mut final_seen = false;
        while let Some(item) = s.next().await {
            let chunk = item.expect("ok");
            if chunk.is_final {
                final_seen = true;
                assert_eq!(chunk.input_tokens, Some(3));
                assert_eq!(chunk.output_tokens, Some(2));
            } else if !chunk.delta.is_empty() {
                texts.push(chunk.delta);
            }
        }
        assert_eq!(texts, vec!["a".to_string(), "b".into()]);
        assert!(final_seen, "must emit a final chunk");
    }

    #[tokio::test]
    async fn stream_to_chunks_synthesizes_final_on_stream_end_without_stop() {
        let events: BoxStream<'static, _> =
            stream::iter(vec![Ok(LanguageModelCompletionEvent::Text("only".into()))]).boxed();
        let mut s = Box::pin(stream_to_chunks(events));
        let first = s.next().await.expect("chunk").expect("ok");
        assert_eq!(first.delta, "only");
        assert!(!first.is_final);
        let synthetic = s.next().await.expect("synthetic final").expect("ok");
        assert!(synthetic.is_final);
        assert!(s.next().await.is_none(), "stream must terminate");
    }

    // ───── End-to-end dispatcher loop over an in-memory call_stream ─────

    #[tokio::test]
    async fn adapter_chat_round_trips_via_dispatcher() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_c = hits.clone();

        let call_stream = move |_req: LanguageModelRequest| {
            let hits = hits_c.clone();
            Box::pin(async move {
                hits.fetch_add(1, Ordering::SeqCst);
                let events: BoxStream<'static, _> = stream::iter(vec![
                    Ok(LanguageModelCompletionEvent::Text("ok".into())),
                    Ok(LanguageModelCompletionEvent::Stop(LmStopReason::EndTurn)),
                ])
                .boxed();
                Ok::<_, LanguageModelCompletionError>(events)
            })
                as Pin<
                    Box<
                        dyn Future<
                                Output = Result<
                                    BoxStream<
                                        'static,
                                        Result<
                                            LanguageModelCompletionEvent,
                                            LanguageModelCompletionError,
                                        >,
                                    >,
                                    LanguageModelCompletionError,
                                >,
                            > + Send,
                    >,
                >
        };

        let handle = spawn_dispatcher(|fut| tokio::spawn(fut), call_stream);
        let adapter = ZedLlmAdapter::new(ProviderId("test".into()), handle);

        let req = mk_req(vec![mk_user_msg("hi")]);
        let resp = adapter.chat(req).await.expect("ok");
        assert_eq!(resp.content, "ok");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn adapter_reports_dispatcher_error_when_call_stream_fails() {
        let call_stream = |_req: LanguageModelRequest| {
            Box::pin(async move { Err(LanguageModelCompletionError::Other(anyhow!("boom"))) })
                as Pin<
                    Box<
                        dyn Future<
                                Output = Result<
                                    BoxStream<
                                        'static,
                                        Result<
                                            LanguageModelCompletionEvent,
                                            LanguageModelCompletionError,
                                        >,
                                    >,
                                    LanguageModelCompletionError,
                                >,
                            > + Send,
                    >,
                >
        };
        let handle = spawn_dispatcher(|fut| tokio::spawn(fut), call_stream);
        let adapter = ZedLlmAdapter::new(ProviderId("t".into()), handle);
        let err = adapter
            .chat(mk_req(vec![mk_user_msg("go")]))
            .await
            .expect_err("must fail");
        assert!(matches!(err, CaduceusError::Provider(_)));
    }

    #[tokio::test]
    async fn adapter_list_models_is_empty() {
        let handle = spawn_dispatcher(
            |fut| tokio::spawn(fut),
            |_req: LanguageModelRequest| {
                Box::pin(async { Ok::<_, LanguageModelCompletionError>(stream::empty().boxed()) })
                    as Pin<Box<dyn Future<Output = _> + Send>>
            },
        );
        let adapter = ZedLlmAdapter::new(ProviderId("t".into()), handle);
        assert!(adapter.list_models().await.unwrap().is_empty());
    }
}
