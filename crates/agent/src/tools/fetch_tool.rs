use std::rc::Rc;
use std::sync::Arc;
use std::{borrow::Cow, cell::RefCell};

use agent_client_protocol as acp;
use agent_settings::AgentSettings;
use anyhow::{Context as _, Result, bail};
use futures::{AsyncReadExt as _, FutureExt as _};
use gpui::{App, AppContext as _, Task};
use html_to_markdown::{TagHandler, convert_html_to_markdown, markdown};
use http_client::{AsyncBody, HttpClientWithUrl};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::Settings;
use ui::SharedString;
use util::markdown::{MarkdownEscaped, MarkdownInlineCode};

use crate::{
    AgentTool, ToolCallEventStream, ToolInput, ToolPermissionDecision,
    decide_permission_from_settings,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
enum ContentType {
    Html,
    Plaintext,
    Json,
}

/// Fetches a URL and returns the content as Markdown.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct FetchToolInput {
    /// The URL to fetch.
    url: String,
}

pub struct FetchTool {
    http_client: Arc<HttpClientWithUrl>,
}

impl FetchTool {
    pub fn new(http_client: Arc<HttpClientWithUrl>) -> Self {
        Self { http_client }
    }

    async fn build_message(http_client: Arc<HttpClientWithUrl>, url: &str) -> Result<String> {
        let url = if !url.starts_with("https://") && !url.starts_with("http://") {
            Cow::Owned(format!("https://{url}"))
        } else {
            Cow::Borrowed(url)
        };

        let mut response = http_client.get(&url, AsyncBody::default(), true).await?;

        let mut body = Vec::new();
        response
            .body_mut()
            .read_to_end(&mut body)
            .await
            .context("error reading response body")?;

        if response.status().is_client_error() {
            let text = String::from_utf8_lossy(body.as_slice());
            bail!(
                "status error {}, response: {text:?}",
                response.status().as_u16()
            );
        }

        let Some(content_type) = response.headers().get("content-type") else {
            bail!("missing Content-Type header");
        };
        let content_type = content_type
            .to_str()
            .context("invalid Content-Type header")?;

        let content_type = if content_type.starts_with("text/plain") {
            ContentType::Plaintext
        } else if content_type.starts_with("application/json") {
            ContentType::Json
        } else {
            ContentType::Html
        };

        match content_type {
            ContentType::Html => {
                let mut handlers: Vec<TagHandler> = vec![
                    Rc::new(RefCell::new(markdown::WebpageChromeRemover)),
                    Rc::new(RefCell::new(markdown::ParagraphHandler)),
                    Rc::new(RefCell::new(markdown::HeadingHandler)),
                    Rc::new(RefCell::new(markdown::ListHandler)),
                    Rc::new(RefCell::new(markdown::TableHandler::new())),
                    Rc::new(RefCell::new(markdown::StyledTextHandler)),
                ];
                if url.contains("wikipedia.org") {
                    use html_to_markdown::structure::wikipedia;

                    handlers.push(Rc::new(RefCell::new(wikipedia::WikipediaChromeRemover)));
                    handlers.push(Rc::new(RefCell::new(wikipedia::WikipediaInfoboxHandler)));
                    handlers.push(Rc::new(
                        RefCell::new(wikipedia::WikipediaCodeHandler::new()),
                    ));
                } else {
                    handlers.push(Rc::new(RefCell::new(markdown::CodeHandler)));
                }

                convert_html_to_markdown(&body[..], &mut handlers)
            }
            ContentType::Plaintext => Ok(std::str::from_utf8(&body)?.to_owned()),
            ContentType::Json => {
                let json: serde_json::Value = serde_json::from_slice(&body)?;

                Ok(format!(
                    "```json\n{}\n```",
                    serde_json::to_string_pretty(&json)?
                ))
            }
        }
    }
}

impl AgentTool for FetchTool {
    type Input = FetchToolInput;
    type Output = String;

    const NAME: &'static str = "fetch";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Fetch
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => format!("Fetch {}", MarkdownEscaped(&input.url)).into(),
            Err(_) => "Fetch URL".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let http_client = self.http_client.clone();
        cx.spawn(async move |cx| {
            let input: FetchToolInput = input
                .recv()
                .await
                .map_err(|e| format!("Failed to receive tool input: {e}"))?;

            let decision = cx.update(|cx| {
                decide_permission_from_settings(
                    Self::NAME,
                    std::slice::from_ref(&input.url),
                    AgentSettings::get_global(cx),
                )
            });

            let authorize = match decision {
                ToolPermissionDecision::Allow => None,
                ToolPermissionDecision::Deny(reason) => {
                    return Err(reason);
                }
                ToolPermissionDecision::Confirm => Some(cx.update(|cx| {
                    let context =
                        crate::ToolPermissionContext::new(Self::NAME, vec![input.url.clone()]);
                    event_stream.authorize(
                        format!("Fetch {}", MarkdownInlineCode(&input.url)),
                        context,
                        cx,
                    )
                })),
            };

            let fetch_task = cx.background_spawn({
                let http_client = http_client.clone();
                let url = input.url.clone();
                async move {
                    if let Some(authorize) = authorize {
                        authorize.await?;
                    }
                    Self::build_message(http_client, &url).await
                }
            });

            let text = futures::select! {
                result = fetch_task.fuse() => result.map_err(|e| e.to_string())?,
                _ = event_stream.cancelled_by_user().fuse() => {
                    return Err("Fetch cancelled by user".to_string());
                }
            };
            if text.trim().is_empty() {
                return Err("no textual content found".to_string());
            }
            // Caduceus fetch-noise: strip common chrome (cookie banners,
            // skip-to-content, repeated nav) and cap size. Closes the
            // 60–70% token-waste path seen in the nanoreason thread review
            // where arXiv/Google fetches dumped hundreds of lines of
            // navigation, footers, and tooling bars per call.
            Ok(clean_fetch_output(&text))
        })
    }
}

/// Caduceus fetch-noise: post-process a fetch result to strip well-known
/// page chrome that the markdown converter leaves in (cookie banners,
/// "Skip to content", arXiv tool drawers, GitHub/footer boilerplate),
/// collapse long blank-line runs, and cap the output at
/// [`FETCH_OUTPUT_HARD_CAP`] bytes so a single fetch can't blow the
/// agent's context window. Conservative — only removes lines whose
/// shape clearly identifies them as chrome, never hand-tuned per site.
const FETCH_OUTPUT_HARD_CAP: usize = 50 * 1024;

fn clean_fetch_output(input: &str) -> String {
    // 1. Line-level filter for known chrome patterns.
    let noise_substrings: &[&str] = &[
        "Skip to content",
        "Skip to main content",
        "Sign in to GitHub",
        "Toggle navigation",
        "Appearance settings",
        "Search or jump to",
        "We read every piece of feedback",
        "We use cookies",
        "Accept all cookies",
        "Manage cookies",
        "Cookie settings",
        "© 20", // covers "© 2024", "© 2025", "© 2026 GitHub, Inc."
        "All rights reserved",
        "You signed in with another tab or window",
        "You switched accounts on another tab",
        "Reload to refresh your session",
        "Dismiss alert",
        "Resetting focus",
        "Loading...",
        "Disable MathJax",
        "arXivLabs is a framework",
        "arXiv Operational Status",
        "Bibliographic Tools",
        "Code, Data, Media",
        "Recommenders and Search Tools",
        "Influence Flower",
        "CORE Recommender",
        "Connected Papers",
        "Litmaps",
        "scite Smart Citations",
        "BibTeX formatted citation",
        "Saved searches",
        "Use saved searches",
        "Provide feedback",
    ];

    // 2. Walk lines, drop chrome and collapse 3+ blank lines to 2.
    let mut out = String::with_capacity(input.len());
    let mut blank_run = 0usize;
    for line in input.lines() {
        let trimmed = line.trim();
        if noise_substrings.iter().any(|n| trimmed.contains(n)) {
            continue;
        }
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run > 2 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line);
        out.push('\n');
    }

    // 3. Hard cap. If the fetch is enormous (e.g. a wiki dump), truncate
    // and tell the model where the cut happened so it can refine the URL.
    if out.len() > FETCH_OUTPUT_HARD_CAP {
        out.truncate(FETCH_OUTPUT_HARD_CAP);
        // Drop a possibly-mid-utf8-sequence trailing partial char.
        while !out.is_char_boundary(out.len()) {
            out.pop();
        }
        out.push_str(
            "\n\n…[fetch output truncated at 50 KB; refine the URL or fetch a sub-page if you need more]",
        );
    }

    out
}

#[cfg(test)]
mod fetch_noise_tests {
    use super::clean_fetch_output;

    #[test]
    fn strips_github_chrome() {
        let raw = "Skip to content\n\nReal content here\n\n© 2026 GitHub, Inc.\nMore content\n";
        let out = clean_fetch_output(raw);
        assert!(!out.contains("Skip to content"));
        assert!(!out.contains("© 2026"));
        assert!(out.contains("Real content here"));
        assert!(out.contains("More content"));
    }

    #[test]
    fn strips_arxiv_chrome() {
        let raw = "# Title\nArtifact text\n\nBibliographic Tools\n\narXivLabs is a framework that\nDisable MathJax (What is MathJax?)\nReal abstract paragraph.\n";
        let out = clean_fetch_output(raw);
        assert!(out.contains("Artifact text"));
        assert!(out.contains("Real abstract paragraph"));
        assert!(!out.contains("Bibliographic Tools"));
        assert!(!out.contains("Disable MathJax"));
        assert!(!out.contains("arXivLabs is a framework"));
    }

    #[test]
    fn collapses_blank_runs() {
        let raw = "a\n\n\n\n\n\nb\n";
        let out = clean_fetch_output(raw);
        // Max 2 blank lines between content.
        assert_eq!(out, "a\n\n\nb\n");
    }

    #[test]
    fn caps_at_50kb() {
        let big = "x".repeat(200_000);
        let out = clean_fetch_output(&big);
        assert!(out.len() <= 51 * 1024);
        assert!(out.contains("truncated"));
    }
}
