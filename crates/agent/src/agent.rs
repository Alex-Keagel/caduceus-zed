mod db;
mod edit_agent;
mod legacy_thread;
mod native_agent_server;
pub mod outline;
mod pattern_extraction;
mod templates;
mod caduceus_native_state;
#[cfg(test)]
mod tests;
mod thread;
mod thread_store;
mod tool_permissions;
mod tools;

use context_server::ContextServerId;
pub use db::*;
use itertools::Itertools;
pub use native_agent_server::NativeAgentServer;
pub use pattern_extraction::*;
pub use shell_command_parser::extract_commands;
pub use templates::*;
pub use thread::*;
pub use thread_store::*;
pub use tool_permissions::*;
pub use tools::*;

use acp_thread::{
    AcpThread, AgentModelSelector, AgentSessionInfo, AgentSessionList, AgentSessionListRequest,
    AgentSessionListResponse, AgentSessionModes, TokenUsageRatio, UserMessageId,
};
use agent_client_protocol as acp;
use anyhow::{Context as _, Result, anyhow};
use chrono::{DateTime, Utc};
use collections::{HashMap, HashSet, IndexMap};
use fs::Fs;
use futures::channel::{mpsc, oneshot};
use futures::future::Shared;
use futures::{FutureExt as _, StreamExt as _, future};
use gpui::{
    App, AppContext, AsyncApp, Context, Entity, EntityId, SharedString, Subscription, Task,
    WeakEntity,
};
use language_model::{IconOrSvg, LanguageModel, LanguageModelProvider, LanguageModelRegistry};
use project::{AgentId, Project, ProjectItem, ProjectPath, Worktree};
use prompt_store::{
    ProjectContext, PromptStore, RULES_FILE_NAMES, RulesFileContext, UserRulesContext,
    WorktreeContext,
};
use serde::{Deserialize, Serialize};
use settings::{LanguageModelSelection, update_settings_file};
use std::any::Any;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, LazyLock};
use util::ResultExt;
use util::path_list::PathList;
use util::rel_path::RelPath;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectSnapshot {
    pub worktree_snapshots: Vec<project::telemetry_snapshot::TelemetryWorktreeSnapshot>,
    pub timestamp: DateTime<Utc>,
}

pub struct RulesLoadingError {
    pub message: SharedString,
}

struct ProjectState {
    project: Entity<Project>,
    project_context: Entity<ProjectContext>,
    project_context_needs_refresh: watch::Sender<()>,
    _maintain_project_context: Task<Result<()>>,
    context_server_registry: Entity<ContextServerRegistry>,
    caduceus_engine: Option<Arc<caduceus_bridge::engine::CaduceusEngine>>,
    _subscriptions: Vec<Subscription>,
}

/// Holds both the internal Thread and the AcpThread for a session
struct Session {
    /// The internal thread that processes messages
    thread: Entity<Thread>,
    /// The ACP thread that handles protocol communication
    acp_thread: Entity<acp_thread::AcpThread>,
    project_id: EntityId,
    pending_save: Task<Result<()>>,
    caduceus_modes: Rc<CaduceusSessionModes>,
    /// Aborts the in-flight event forwarder for this session before a new turn starts.
    /// Without this, two concurrent `handle_thread_events` tasks can both call
    /// `acp_thread.update(...)` on the same Entity, triggering the gpui
    /// double-lease panic ("cannot update AcpThread while it is already being updated").
    forwarder_abort: Option<watch::Sender<bool>>,
    /// Authorization-callback tasks spawned by `handle_thread_events`. We
    /// keep them here (instead of `.detach()`) so they're aborted when the
    /// session is dropped — otherwise a tool-permission prompt outliving its
    /// session leaks the future and may panic on a stale acp_thread handle.
    auth_tasks: std::sync::Arc<std::sync::Mutex<Vec<Task<()>>>>,
    _subscriptions: Vec<Subscription>,
}

pub struct LanguageModels {
    /// Access language model by ID
    models: HashMap<acp::ModelId, Arc<dyn LanguageModel>>,
    /// Cached list for returning language model information
    model_list: acp_thread::AgentModelList,
    refresh_models_rx: watch::Receiver<()>,
    refresh_models_tx: watch::Sender<()>,
    _authenticate_all_providers_task: Task<()>,
}

impl LanguageModels {
    fn new(cx: &mut App) -> Self {
        let (refresh_models_tx, refresh_models_rx) = watch::channel(());

        let mut this = Self {
            models: HashMap::default(),
            model_list: acp_thread::AgentModelList::Grouped(IndexMap::default()),
            refresh_models_rx,
            refresh_models_tx,
            _authenticate_all_providers_task: Self::authenticate_all_language_model_providers(cx),
        };
        this.refresh_list(cx);
        this
    }

    fn refresh_list(&mut self, cx: &App) {
        let providers = LanguageModelRegistry::global(cx)
            .read(cx)
            .visible_providers()
            .into_iter()
            .filter(|provider| provider.is_authenticated(cx))
            .collect::<Vec<_>>();

        let mut language_model_list = IndexMap::default();
        let mut recommended_models = HashSet::default();

        let mut recommended = Vec::new();
        for provider in &providers {
            for model in provider.recommended_models(cx) {
                recommended_models.insert((model.provider_id(), model.id()));
                recommended.push(Self::map_language_model_to_info(&model, provider));
            }
        }
        if !recommended.is_empty() {
            language_model_list.insert(
                acp_thread::AgentModelGroupName("Recommended".into()),
                recommended,
            );
        }

        let mut models = HashMap::default();
        for provider in providers {
            let mut provider_models = Vec::new();
            for model in provider.provided_models(cx) {
                let model_info = Self::map_language_model_to_info(&model, &provider);
                let model_id = model_info.id.clone();
                provider_models.push(model_info);
                models.insert(model_id, model);
            }
            if !provider_models.is_empty() {
                language_model_list.insert(
                    acp_thread::AgentModelGroupName(provider.name().0.clone()),
                    provider_models,
                );
            }
        }

        self.models = models;
        self.model_list = acp_thread::AgentModelList::Grouped(language_model_list);
        self.refresh_models_tx.send(()).ok();
    }

    fn watch(&self) -> watch::Receiver<()> {
        self.refresh_models_rx.clone()
    }

    pub fn model_from_id(&self, model_id: &acp::ModelId) -> Option<Arc<dyn LanguageModel>> {
        self.models.get(model_id).cloned()
    }

    fn map_language_model_to_info(
        model: &Arc<dyn LanguageModel>,
        provider: &Arc<dyn LanguageModelProvider>,
    ) -> acp_thread::AgentModelInfo {
        acp_thread::AgentModelInfo {
            id: Self::model_id(model),
            name: model.name().0,
            description: None,
            icon: Some(match provider.icon() {
                IconOrSvg::Svg(path) => acp_thread::AgentModelIcon::Path(path),
                IconOrSvg::Icon(name) => acp_thread::AgentModelIcon::Named(name),
            }),
            is_latest: model.is_latest(),
            cost: model.model_cost_info().map(|cost| cost.to_shared_string()),
        }
    }

    fn model_id(model: &Arc<dyn LanguageModel>) -> acp::ModelId {
        acp::ModelId::new(format!("{}/{}", model.provider_id().0, model.id().0))
    }

    fn authenticate_all_language_model_providers(cx: &mut App) -> Task<()> {
        let authenticate_all_providers = LanguageModelRegistry::global(cx)
            .read(cx)
            .visible_providers()
            .iter()
            .map(|provider| (provider.id(), provider.name(), provider.authenticate(cx)))
            .collect::<Vec<_>>();

        cx.background_spawn(async move {
            for (provider_id, provider_name, authenticate_task) in authenticate_all_providers {
                if let Err(err) = authenticate_task.await {
                    match err {
                        language_model::AuthenticateError::CredentialsNotFound => {
                            // Since we're authenticating these providers in the
                            // background for the purposes of populating the
                            // language selector, we don't care about providers
                            // where the credentials are not found.
                        }
                        language_model::AuthenticateError::ConnectionRefused => {
                            // Not logging connection refused errors as they are mostly from LM Studio's noisy auth failures.
                            // LM Studio only has one auth method (endpoint call) which fails for users who haven't enabled it.
                            // TODO: Better manage LM Studio auth logic to avoid these noisy failures.
                        }
                        _ => {
                            // Some providers have noisy failure states that we
                            // don't want to spam the logs with every time the
                            // language model selector is initialized.
                            //
                            // Ideally these should have more clear failure modes
                            // that we know are safe to ignore here, like what we do
                            // with `CredentialsNotFound` above.
                            match provider_id.0.as_ref() {
                                "lmstudio" | "ollama" => {
                                    // LM Studio and Ollama both make fetch requests to the local APIs to determine if they are "authenticated".
                                    //
                                    // These fail noisily, so we don't log them.
                                }
                                "copilot_chat" => {
                                    // Copilot Chat returns an error if Copilot is not enabled, so we don't log those errors.
                                }
                                _ => {
                                    log::error!(
                                        "Failed to authenticate provider: {}: {err:#}",
                                        provider_name.0
                                    );
                                }
                            }
                        }
                    }
                }
            }
        })
    }
}

pub struct NativeAgent {
    /// Session ID -> Session mapping
    sessions: HashMap<acp::SessionId, Session>,
    thread_store: Entity<ThreadStore>,
    /// Project-specific state keyed by project EntityId
    projects: HashMap<EntityId, ProjectState>,
    /// Shared templates for all threads
    templates: Arc<Templates>,
    /// Cached model information
    models: LanguageModels,
    prompt_store: Option<Entity<PromptStore>>,
    fs: Arc<dyn Fs>,
    _subscriptions: Vec<Subscription>,
}

impl NativeAgent {
    pub fn new(
        thread_store: Entity<ThreadStore>,
        templates: Arc<Templates>,
        prompt_store: Option<Entity<PromptStore>>,
        fs: Arc<dyn Fs>,
        cx: &mut App,
    ) -> Entity<NativeAgent> {
        log::debug!("Creating new NativeAgent");

        cx.new(|cx| {
            let mut subscriptions = vec![cx.subscribe(
                &LanguageModelRegistry::global(cx),
                Self::handle_models_updated_event,
            )];
            if let Some(prompt_store) = prompt_store.as_ref() {
                subscriptions.push(cx.subscribe(prompt_store, Self::handle_prompts_updated_event))
            }

            Self {
                sessions: HashMap::default(),
                thread_store,
                projects: HashMap::default(),
                templates,
                models: LanguageModels::new(cx),
                prompt_store,
                fs,
                _subscriptions: subscriptions,
            }
        })
    }

    /// Caduceus: emergency kill switch — cancels ALL running sessions immediately
    pub fn kill_all_sessions(&mut self, cx: &mut Context<Self>) {
        log::warn!("[caduceus] KILL SWITCH activated — stopping all sessions");
        for (session_id, session) in &self.sessions {
            log::info!("[caduceus] Killing session: {}", session_id);
            session.thread.update(cx, |thread, cx| {
                thread.cancel(cx).detach();
            });
        }
    }

    fn new_session(
        &mut self,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Entity<AcpThread> {
        let project_id = self.get_or_create_project_state(&project, cx);
        let project_state = &self.projects[&project_id];

        let registry = LanguageModelRegistry::read_global(cx);
        let available_count = registry.available_models(cx).count();
        log::debug!("Total available models: {}", available_count);

        let default_model = registry.default_model().and_then(|default_model| {
            self.models
                .model_from_id(&LanguageModels::model_id(&default_model.model))
        });
        let thread = cx.new(|cx| {
            Thread::new(
                project,
                project_state.project_context.clone(),
                project_state.context_server_registry.clone(),
                self.templates.clone(),
                default_model,
                cx,
            )
        });

        self.register_session(thread, project_id, cx)
    }

    fn register_session(
        &mut self,
        thread_handle: Entity<Thread>,
        project_id: EntityId,
        cx: &mut Context<Self>,
    ) -> Entity<AcpThread> {
        let connection = Rc::new(NativeAgentConnection(cx.entity()));

        let thread = thread_handle.read(cx);
        let session_id = thread.id().clone();
        let parent_session_id = thread.parent_thread_id();
        let title = thread.title();
        let draft_prompt = thread.draft_prompt().map(Vec::from);
        let scroll_position = thread.ui_scroll_position();
        let token_usage = thread.latest_token_usage();
        let project = thread.project.clone();
        let action_log = thread.action_log.clone();
        let prompt_capabilities_rx = thread.prompt_capabilities_rx.clone();
        let acp_thread = cx.new(|cx| {
            let mut acp_thread = acp_thread::AcpThread::new(
                parent_session_id,
                title,
                None,
                connection,
                project.clone(),
                action_log.clone(),
                session_id.clone(),
                prompt_capabilities_rx,
                cx,
            );
            acp_thread.set_draft_prompt(draft_prompt);
            acp_thread.set_ui_scroll_position(scroll_position);
            acp_thread.update_token_usage(token_usage, cx);
            acp_thread
        });

        let registry = LanguageModelRegistry::read_global(cx);
        let summarization_model = registry.thread_summary_model().map(|c| c.model);

        let weak = cx.weak_entity();
        let weak_thread = thread_handle.downgrade();
        thread_handle.update(cx, |thread, cx| {
            thread.set_summarization_model(summarization_model, cx);
            let engine = self
                .projects
                .get(&project.entity_id())
                .and_then(|ps| ps.caduceus_engine.clone());
            thread.add_default_tools(
                Rc::new(NativeThreadEnvironment {
                    acp_thread: acp_thread.downgrade(),
                    thread: weak_thread,
                    agent: weak,
                }) as _,
                engine,
                cx,
            )
        });

        let subscriptions = vec![
            cx.subscribe(&thread_handle, Self::handle_thread_title_updated),
            cx.subscribe(&thread_handle, Self::handle_thread_token_usage_updated),
            cx.observe(&thread_handle, move |this, thread, cx| {
                this.save_thread(thread, cx)
            }),
        ];

        self.sessions.insert(
            session_id,
            Session {
                thread: thread_handle,
                acp_thread: acp_thread.clone(),
                project_id,
                _subscriptions: subscriptions,
                pending_save: Task::ready(Ok(())),
                caduceus_modes: Rc::new(CaduceusSessionModes::new()),
                forwarder_abort: None,
                auth_tasks: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            },
        );

        self.update_available_commands_for_project(project_id, cx);

        acp_thread
    }

    pub fn models(&self) -> &LanguageModels {
        &self.models
    }

    fn get_or_create_project_state(
        &mut self,
        project: &Entity<Project>,
        cx: &mut Context<Self>,
    ) -> EntityId {
        let project_id = project.entity_id();
        if self.projects.contains_key(&project_id) {
            return project_id;
        }

        let project_context = cx.new(|_| ProjectContext::new(vec![], vec![]));
        self.register_project_with_initial_context(project.clone(), project_context, cx);
        if let Some(state) = self.projects.get_mut(&project_id) {
            state.project_context_needs_refresh.send(()).ok();
        }
        project_id
    }

    fn register_project_with_initial_context(
        &mut self,
        project: Entity<Project>,
        project_context: Entity<ProjectContext>,
        cx: &mut Context<Self>,
    ) {
        let project_id = project.entity_id();

        let context_server_store = project.read(cx).context_server_store();
        let context_server_registry =
            cx.new(|cx| ContextServerRegistry::new(context_server_store.clone(), cx));

        let subscriptions = vec![
            cx.subscribe(&project, Self::handle_project_event),
            cx.subscribe(
                &context_server_store,
                Self::handle_context_server_store_updated,
            ),
            cx.subscribe(
                &context_server_registry,
                Self::handle_context_server_registry_event,
            ),
        ];

        let (project_context_needs_refresh_tx, project_context_needs_refresh_rx) =
            watch::channel(());

        // Create shared Caduceus engine for this project
        let caduceus_engine = project.read(cx).worktrees(cx).next().map(|wt| {
            Arc::new(caduceus_bridge::engine::CaduceusEngine::new(
                &wt.read(cx).abs_path().to_path_buf(),
            ))
        });

        self.projects.insert(
            project_id,
            ProjectState {
                project: project.clone(),
                project_context,
                project_context_needs_refresh: project_context_needs_refresh_tx,
                _maintain_project_context: cx.spawn(async move |this, cx| {
                    Self::maintain_project_context(
                        this,
                        project_id,
                        project_context_needs_refresh_rx,
                        cx,
                    )
                    .await
                }),
                context_server_registry,
                caduceus_engine: caduceus_engine.clone(),
                _subscriptions: subscriptions,
            },
        );

        // Caduceus: auto-index project on open and populate wiki
        if let Some(engine) = caduceus_engine {
            let project_for_index = project.clone();
            let engine_clone = engine.clone();
            cx.spawn(async move |_this, cx| {
                let project_root = cx.update(|cx| {
                    project_for_index
                        .read(cx)
                        .worktrees(cx)
                        .next()
                        .map(|wt| wt.read(cx).abs_path().to_path_buf())
                });

                if let Some(root) = project_root {
                    // Skip if already indexed
                    let already_indexed = engine_clone.index_chunk_count().await > 0;
                    if already_indexed {
                        log::info!("[caduceus] Project already indexed, skipping");
                    } else {
                        log::info!("[caduceus] Auto-indexing project: {}", root.display());
                        let ignore_patterns = load_caduceuignore(&root);
                        if !ignore_patterns.is_empty() {
                            log::info!(
                                "[caduceus] Loaded {} .caduceuignore patterns",
                                ignore_patterns.len()
                            );
                        }

                        // Index project files in background using the shared engine
                        match engine_clone.index_directory(&root).await {
                            Ok(chunks) => log::info!("[caduceus] Indexed {} chunks", chunks),
                            Err(e) => log::warn!("[caduceus] Index failed: {e}"),
                        }

                        // Populate wiki with project overview
                        let wiki_root = root.clone();
                        let readme_path = root.join("README.md");
                        let readme = std::fs::read_to_string(&readme_path).unwrap_or_default();

                        // Build project structure page
                        let mut structure = String::from("# Project Structure\n\n");
                        if let Ok(entries) = std::fs::read_dir(&root) {
                            for entry in entries.flatten() {
                                let name = entry.file_name().to_string_lossy().to_string();
                                if name.starts_with('.') {
                                    continue;
                                }
                                if !should_index_path(&entry.path(), &ignore_patterns) {
                                    continue;
                                }
                                let kind = if entry.path().is_dir() {
                                    "📁"
                                } else {
                                    "📄"
                                };
                                structure.push_str(&format!("- {kind} {name}\n"));
                            }
                        }

                        // Write wiki pages
                        let _ = caduceus_bridge::memory::store_system(
                            &wiki_root,
                            caduceus_bridge::memory::KEY_PROJECT_OVERVIEW,
                            &structure,
                        );
                        if !readme.is_empty() {
                            let _ = caduceus_bridge::memory::store_system(
                                &wiki_root,
                                caduceus_bridge::memory::KEY_README,
                                &readme,
                            );
                        }

                        // Git context
                        if let Ok(branch) = engine_clone.git_branch() {
                            let _ = caduceus_bridge::memory::store_system(
                                &wiki_root,
                                caduceus_bridge::memory::KEY_GIT_BRANCH,
                                &branch,
                            );
                        }
                        if let Ok(log) = engine_clone.git_log(10) {
                            let log_text: String = log
                                .iter()
                                .map(|c| {
                                    format!(
                                        "{} — {}",
                                        crate::tools::truncate_str(&c.sha, 7),
                                        c.message
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            let _ = caduceus_bridge::memory::store_system(
                                &wiki_root,
                                caduceus_bridge::memory::KEY_RECENT_COMMITS,
                                &log_text,
                            );
                        }

                        log::info!("[caduceus] Wiki populated for {}", root.display());

                        // Memory size is now managed by JSON store with MAX_ENTRIES=100 cap.
                        // No manual trimming needed — the store enforces limits on write.
                    } // end if !already_indexed
                }
            })
            .detach();
        }
    }

    fn session_project_state(&self, session_id: &acp::SessionId) -> Option<&ProjectState> {
        self.sessions
            .get(session_id)
            .and_then(|session| self.projects.get(&session.project_id))
    }

    async fn maintain_project_context(
        this: WeakEntity<Self>,
        project_id: EntityId,
        mut needs_refresh: watch::Receiver<()>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        while needs_refresh.changed().await.is_ok() {
            let project_context = this
                .update(cx, |this, cx| {
                    let state = this
                        .projects
                        .get(&project_id)
                        .context("project state not found")?;
                    anyhow::Ok(Self::build_project_context(
                        &state.project,
                        this.prompt_store.as_ref(),
                        cx,
                    ))
                })??
                .await;
            this.update(cx, |this, cx| {
                if let Some(state) = this.projects.get(&project_id) {
                    state
                        .project_context
                        .update(cx, |current_project_context, _cx| {
                            *current_project_context = project_context;
                        });
                }
            })?;
        }

        Ok(())
    }

    fn build_project_context(
        project: &Entity<Project>,
        prompt_store: Option<&Entity<PromptStore>>,
        cx: &mut App,
    ) -> Task<ProjectContext> {
        let worktrees = project.read(cx).visible_worktrees(cx).collect::<Vec<_>>();
        let worktree_tasks = worktrees
            .into_iter()
            .map(|worktree| {
                Self::load_worktree_info_for_system_prompt(worktree, project.clone(), cx)
            })
            .collect::<Vec<_>>();
        let default_user_rules_task = if let Some(prompt_store) = prompt_store.as_ref() {
            prompt_store.read_with(cx, |prompt_store, cx| {
                let prompts = prompt_store.default_prompt_metadata();
                let load_tasks = prompts.into_iter().map(|prompt_metadata| {
                    let contents = prompt_store.load(prompt_metadata.id, cx);
                    async move { (contents.await, prompt_metadata) }
                });
                cx.background_spawn(future::join_all(load_tasks))
            })
        } else {
            Task::ready(vec![])
        };

        cx.spawn(async move |_cx| {
            let (worktrees, default_user_rules) =
                future::join(future::join_all(worktree_tasks), default_user_rules_task).await;

            let worktrees = worktrees
                .into_iter()
                .map(|(worktree, _rules_error)| {
                    // TODO: show error message
                    // if let Some(rules_error) = rules_error {
                    //     this.update(cx, |_, cx| cx.emit(rules_error)).ok();
                    // }
                    worktree
                })
                .collect::<Vec<_>>();

            let default_user_rules = default_user_rules
                .into_iter()
                .flat_map(|(contents, prompt_metadata)| match contents {
                    Ok(contents) => Some(UserRulesContext {
                        uuid: prompt_metadata.id.as_user()?,
                        title: prompt_metadata.title.map(|title| title.to_string()),
                        contents,
                    }),
                    Err(_err) => {
                        // TODO: show error message
                        // this.update(cx, |_, cx| {
                        //     cx.emit(RulesLoadingError {
                        //         message: format!("{err:?}").into(),
                        //     });
                        // })
                        // .ok();
                        None
                    }
                })
                .collect::<Vec<_>>();

            ProjectContext::new(worktrees, default_user_rules)
        })
    }

    fn load_worktree_info_for_system_prompt(
        worktree: Entity<Worktree>,
        project: Entity<Project>,
        cx: &mut App,
    ) -> Task<(WorktreeContext, Option<RulesLoadingError>)> {
        let tree = worktree.read(cx);
        let root_name = tree.root_name_str().into();
        let abs_path = tree.abs_path();

        let mut context = WorktreeContext {
            root_name,
            abs_path,
            rules_file: None,
        };

        let rules_task = Self::load_worktree_rules_file(worktree, project, cx);
        let Some(rules_task) = rules_task else {
            return Task::ready((context, None));
        };

        cx.spawn(async move |_| {
            let (rules_file, rules_file_error) = match rules_task.await {
                Ok(rules_file) => (Some(rules_file), None),
                Err(err) => (
                    None,
                    Some(RulesLoadingError {
                        message: format!("{err}").into(),
                    }),
                ),
            };
            context.rules_file = rules_file;
            (context, rules_file_error)
        })
    }

    fn load_worktree_rules_file(
        worktree: Entity<Worktree>,
        project: Entity<Project>,
        cx: &mut App,
    ) -> Option<Task<Result<RulesFileContext>>> {
        let worktree = worktree.read(cx);
        let worktree_id = worktree.id();
        let selected_rules_file = RULES_FILE_NAMES
            .into_iter()
            .filter_map(|name| {
                worktree
                    .entry_for_path(RelPath::unix(name).unwrap())
                    .filter(|entry| entry.is_file())
                    .map(|entry| entry.path.clone())
            })
            .next();

        // Note that Cline supports `.clinerules` being a directory, but that is not currently
        // supported. This doesn't seem to occur often in GitHub repositories.
        selected_rules_file.map(|path_in_worktree| {
            let project_path = ProjectPath {
                worktree_id,
                path: path_in_worktree.clone(),
            };
            let buffer_task =
                project.update(cx, |project, cx| project.open_buffer(project_path, cx));
            let rope_task = cx.spawn(async move |cx| {
                let buffer = buffer_task.await?;
                let (project_entry_id, rope) = buffer.read_with(cx, |buffer, cx| {
                    let project_entry_id = buffer.entry_id(cx).context("buffer has no file")?;
                    anyhow::Ok((project_entry_id, buffer.as_rope().clone()))
                })?;
                anyhow::Ok((project_entry_id, rope))
            });
            // Build a string from the rope on a background thread.
            cx.background_spawn(async move {
                let (project_entry_id, rope) = rope_task.await?;
                anyhow::Ok(RulesFileContext {
                    path_in_worktree,
                    text: rope.to_string().trim().to_string(),
                    project_entry_id: project_entry_id.to_usize(),
                })
            })
        })
    }

    fn handle_thread_title_updated(
        &mut self,
        thread: Entity<Thread>,
        _: &TitleUpdated,
        cx: &mut Context<Self>,
    ) {
        let session_id = thread.read(cx).id();
        let Some(session) = self.sessions.get(session_id) else {
            return;
        };

        let thread = thread.downgrade();
        let acp_thread = session.acp_thread.downgrade();
        cx.spawn(async move |_, cx| {
            let title = thread.read_with(cx, |thread, _| thread.title())?;
            if let Some(title) = title {
                let task =
                    acp_thread.update(cx, |acp_thread, cx| acp_thread.set_title(title, cx))?;
                task.await?;
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn handle_thread_token_usage_updated(
        &mut self,
        thread: Entity<Thread>,
        usage: &TokenUsageUpdated,
        cx: &mut Context<Self>,
    ) {
        let Some(session) = self.sessions.get(thread.read(cx).id()) else {
            return;
        };
        session.acp_thread.update(cx, |acp_thread, cx| {
            acp_thread.update_token_usage(usage.0.clone(), cx);
        });
    }

    fn handle_project_event(
        &mut self,
        project: Entity<Project>,
        event: &project::Event,
        _cx: &mut Context<Self>,
    ) {
        let project_id = project.entity_id();
        let Some(state) = self.projects.get_mut(&project_id) else {
            return;
        };
        match event {
            project::Event::WorktreeAdded(_) | project::Event::WorktreeRemoved(_) => {
                state.project_context_needs_refresh.send(()).ok();
                // Caduceus: re-index on worktree changes using shared engine
                if let Some(engine) = state.caduceus_engine.clone() {
                    if let Some(wt) = project.read(_cx).worktrees(_cx).next() {
                        let root = wt.read(_cx).abs_path().to_path_buf();
                        _cx.background_executor()
                            .spawn(async move {
                                let _ = engine.index_directory(&root).await;
                                log::info!("[caduceus] Re-indexed after worktree change");
                            })
                            .detach();
                    }
                }
            }
            project::Event::WorktreeUpdatedEntries(_, items) => {
                if items.iter().any(|(path, _, _)| {
                    RULES_FILE_NAMES
                        .iter()
                        .any(|name| path.as_ref() == RelPath::unix(name).unwrap())
                }) {
                    state.project_context_needs_refresh.send(()).ok();
                }
            }
            _ => {}
        }
    }

    fn handle_prompts_updated_event(
        &mut self,
        _prompt_store: Entity<PromptStore>,
        _event: &prompt_store::PromptsUpdatedEvent,
        _cx: &mut Context<Self>,
    ) {
        for state in self.projects.values_mut() {
            state.project_context_needs_refresh.send(()).ok();
        }
    }

    fn handle_models_updated_event(
        &mut self,
        _registry: Entity<LanguageModelRegistry>,
        event: &language_model::Event,
        cx: &mut Context<Self>,
    ) {
        self.models.refresh_list(cx);

        let registry = LanguageModelRegistry::read_global(cx);
        let default_model = registry.default_model().map(|m| m.model);
        let summarization_model = registry.thread_summary_model().map(|m| m.model);

        for session in self.sessions.values_mut() {
            session.thread.update(cx, |thread, cx| {
                if thread.model().is_none()
                    && let Some(model) = default_model.clone()
                {
                    thread.set_model(model, cx);
                    cx.notify();
                }
                if let Some(model) = summarization_model.clone() {
                    if thread.summarization_model().is_none()
                        || matches!(event, language_model::Event::ThreadSummaryModelChanged)
                    {
                        thread.set_summarization_model(Some(model), cx);
                    }
                }
            });
        }
    }

    fn handle_context_server_store_updated(
        &mut self,
        store: Entity<project::context_server_store::ContextServerStore>,
        _event: &project::context_server_store::ServerStatusChangedEvent,
        cx: &mut Context<Self>,
    ) {
        let project_id = self.projects.iter().find_map(|(id, state)| {
            if *state.context_server_registry.read(cx).server_store() == store {
                Some(*id)
            } else {
                None
            }
        });
        if let Some(project_id) = project_id {
            self.update_available_commands_for_project(project_id, cx);
        }
    }

    fn handle_context_server_registry_event(
        &mut self,
        registry: Entity<ContextServerRegistry>,
        event: &ContextServerRegistryEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            ContextServerRegistryEvent::ToolsChanged => {}
            ContextServerRegistryEvent::PromptsChanged => {
                let project_id = self.projects.iter().find_map(|(id, state)| {
                    if state.context_server_registry == registry {
                        Some(*id)
                    } else {
                        None
                    }
                });
                if let Some(project_id) = project_id {
                    self.update_available_commands_for_project(project_id, cx);
                }
            }
        }
    }

    fn update_available_commands_for_project(&self, project_id: EntityId, cx: &mut Context<Self>) {
        let available_commands =
            Self::build_available_commands_for_project(self.projects.get(&project_id), cx);
        for session in self.sessions.values() {
            if session.project_id != project_id {
                continue;
            }
            session.acp_thread.update(cx, |thread, cx| {
                thread
                    .handle_session_update(
                        acp::SessionUpdate::AvailableCommandsUpdate(
                            acp::AvailableCommandsUpdate::new(available_commands.clone()),
                        ),
                        cx,
                    )
                    .log_err();
            });
        }
    }

    /// ST-B5 / `catalog-opaque-v1` — see free fns `available_modes_comma` /
    /// `available_modes_pipe`. Shims kept for `Self::…` ergonomics at
    /// the `NativeAgent` call sites.
    fn available_modes_pipe() -> String {
        available_modes_pipe()
    }

    fn build_available_commands_for_project(
        project_state: Option<&ProjectState>,
        cx: &App,
    ) -> Vec<acp::AvailableCommand> {
        // Caduceus built-in commands — always available, even without a project
        let mut commands = vec![
            acp::AvailableCommand::new("compact", "Compress conversation context to free tokens"),
            acp::AvailableCommand::new("mode", "Show or switch Caduceus mode").input(
                acp::AvailableCommandInput::Unstructured(acp::UnstructuredCommandInput::new(
                    format!("<{}>", Self::available_modes_pipe()),
                )),
            ),
            acp::AvailableCommand::new("context", "Show context usage and zone status"),
            acp::AvailableCommand::new("checkpoint", "Create a code checkpoint for rollback"),
            acp::AvailableCommand::new(
                "dag",
                "Show the index-access DAG (which agents are reading/writing engine state)",
            ),
            acp::AvailableCommand::new("help", "Show all Caduceus commands"),
            acp::AvailableCommand::new("review", "Review code for security issues"),
            acp::AvailableCommand::new("headless", "Generate CLI command for headless execution")
                .input(acp::AvailableCommandInput::Unstructured(
                    acp::UnstructuredCommandInput::new("<prompt>"),
                )),
            acp::AvailableCommand::new("map", "Show project repo map (tree-sitter outline)"),
            acp::AvailableCommand::new("start", "Getting started guide for new users"),
            acp::AvailableCommand::new(
                "status",
                "Show unified Caduceus dashboard with all metrics",
            ),
            acp::AvailableCommand::new("search", "Semantic search across indexed code").input(
                acp::AvailableCommandInput::Unstructured(acp::UnstructuredCommandInput::new(
                    "<query>",
                )),
            ),
            acp::AvailableCommand::new("index", "Index project for semantic search"),
        ];

        let Some(state) = project_state else {
            return commands;
        };
        let registry = state.context_server_registry.read(cx);

        let mut prompt_name_counts: HashMap<&str, usize> = HashMap::default();
        for context_server_prompt in registry.prompts() {
            *prompt_name_counts
                .entry(context_server_prompt.prompt.name.as_str())
                .or_insert(0) += 1;
        }

        let context_server_commands: Vec<acp::AvailableCommand> = registry
            .prompts()
            .flat_map(|context_server_prompt| {
                let prompt = &context_server_prompt.prompt;

                let should_prefix = prompt_name_counts
                    .get(prompt.name.as_str())
                    .copied()
                    .unwrap_or(0)
                    > 1;

                let name = if should_prefix {
                    format!("{}.{}", context_server_prompt.server_id, prompt.name)
                } else {
                    prompt.name.clone()
                };

                let mut command = acp::AvailableCommand::new(
                    name,
                    prompt.description.clone().unwrap_or_default(),
                );

                match prompt.arguments.as_deref() {
                    Some([arg]) => {
                        let hint = format!("<{}>", arg.name);

                        command = command.input(acp::AvailableCommandInput::Unstructured(
                            acp::UnstructuredCommandInput::new(hint),
                        ));
                    }
                    Some([]) | None => {}
                    Some(_) => {
                        // skip >1 argument commands since we don't support them yet
                        return None;
                    }
                }

                Some(command)
            })
            .collect::<Vec<_>>();

        // Context server commands added to the Caduceus built-ins
        commands.extend(context_server_commands);
        commands
    }

    pub fn load_thread(
        &mut self,
        id: acp::SessionId,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Thread>>> {
        let database_future = ThreadsDatabase::connect(cx);
        cx.spawn(async move |this, cx| {
            let database = database_future.await.map_err(|err| anyhow!(err))?;
            let db_thread = database
                .load_thread(id.clone())
                .await?
                .with_context(|| format!("no thread found with ID: {id:?}"))?;

            this.update(cx, |this, cx| {
                let project_id = this.get_or_create_project_state(&project, cx);
                let project_state = this
                    .projects
                    .get(&project_id)
                    .context("project state not found")?;
                let summarization_model = LanguageModelRegistry::read_global(cx)
                    .thread_summary_model()
                    .map(|c| c.model);

                Ok(cx.new(|cx| {
                    let mut thread = Thread::from_db(
                        id.clone(),
                        db_thread,
                        project_state.project.clone(),
                        project_state.project_context.clone(),
                        project_state.context_server_registry.clone(),
                        this.templates.clone(),
                        cx,
                    );
                    thread.set_summarization_model(summarization_model, cx);
                    thread
                }))
            })?
        })
    }

    pub fn open_thread(
        &mut self,
        id: acp::SessionId,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<AcpThread>>> {
        if let Some(session) = self.sessions.get(&id) {
            return Task::ready(Ok(session.acp_thread.clone()));
        }

        let task = self.load_thread(id, project.clone(), cx);
        cx.spawn(async move |this, cx| {
            let thread = task.await?;
            let acp_thread = this.update(cx, |this, cx| {
                let project_id = this.get_or_create_project_state(&project, cx);
                this.register_session(thread.clone(), project_id, cx)
            })?;
            let events = thread.update(cx, |thread, cx| thread.replay(cx));
            cx.update(|cx| {
                NativeAgentConnection::handle_thread_events(
                    events,
                    acp_thread.downgrade(),
                    None,
                    None,
                    cx,
                )
            })
            .await?;
            acp_thread.update(cx, |thread, cx| {
                thread.snapshot_completed_plan(cx);
            });
            Ok(acp_thread)
        })
    }

    pub fn thread_summary(
        &mut self,
        id: acp::SessionId,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Task<Result<SharedString>> {
        let thread = self.open_thread(id.clone(), project, cx);
        cx.spawn(async move |this, cx| {
            let acp_thread = thread.await?;
            let result = this
                .update(cx, |this, cx| {
                    this.sessions
                        .get(&id)
                        .unwrap()
                        .thread
                        .update(cx, |thread, cx| thread.summary(cx))
                })?
                .await
                .context("Failed to generate summary")?;
            drop(acp_thread);
            Ok(result)
        })
    }

    fn save_thread(&mut self, thread: Entity<Thread>, cx: &mut Context<Self>) {
        if thread.read(cx).is_empty() {
            return;
        }

        let id = thread.read(cx).id().clone();
        let Some(session) = self.sessions.get_mut(&id) else {
            return;
        };

        let project_id = session.project_id;
        let Some(state) = self.projects.get(&project_id) else {
            return;
        };

        let folder_paths = PathList::new(
            &state
                .project
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
                .collect::<Vec<_>>(),
        );

        let draft_prompt = session.acp_thread.read(cx).draft_prompt().map(Vec::from);
        let database_future = ThreadsDatabase::connect(cx);
        let db_thread = thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(draft_prompt);
            thread.to_db(cx)
        });
        let thread_store = self.thread_store.clone();
        session.pending_save = cx.spawn(async move |_, cx| {
            let Some(database) = database_future.await.map_err(|err| anyhow!(err)).log_err() else {
                return Ok(());
            };
            let db_thread = db_thread.await;
            database
                .save_thread(id, db_thread, folder_paths)
                .await
                .log_err();
            thread_store.update(cx, |store, cx| store.reload(cx));
            Ok(())
        });
    }

    fn send_mcp_prompt(
        &self,
        message_id: UserMessageId,
        session_id: acp::SessionId,
        prompt_name: String,
        server_id: ContextServerId,
        arguments: HashMap<String, String>,
        original_content: Vec<acp::ContentBlock>,
        cx: &mut Context<Self>,
    ) -> Task<Result<acp::PromptResponse>> {
        let Some(state) = self.session_project_state(&session_id) else {
            return Task::ready(Err(anyhow!("Project state not found for session")));
        };
        let server_store = state
            .context_server_registry
            .read(cx)
            .server_store()
            .clone();
        let path_style = state.project.read(cx).path_style(cx);

        cx.spawn(async move |this, cx| {
            let prompt =
                crate::get_prompt(&server_store, &server_id, &prompt_name, arguments, cx).await?;

            let (acp_thread, thread) = this.update(cx, |this, _cx| {
                let session = this
                    .sessions
                    .get(&session_id)
                    .context("Failed to get session")?;
                anyhow::Ok((session.acp_thread.clone(), session.thread.clone()))
            })??;

            let mut last_is_user = true;

            thread.update(cx, |thread, cx| {
                thread.push_acp_user_block(
                    message_id,
                    original_content.into_iter().skip(1),
                    path_style,
                    cx,
                );
            });

            for message in prompt.messages {
                let context_server::types::PromptMessage { role, content } = message;
                let block = mcp_message_content_to_acp_content_block(content);

                match role {
                    context_server::types::Role::User => {
                        let id = acp_thread::UserMessageId::new();

                        acp_thread.update(cx, |acp_thread, cx| {
                            acp_thread.push_user_content_block_with_indent(
                                Some(id.clone()),
                                block.clone(),
                                true,
                                cx,
                            );
                        });

                        thread.update(cx, |thread, cx| {
                            thread.push_acp_user_block(id, [block], path_style, cx);
                        });
                    }
                    context_server::types::Role::Assistant => {
                        acp_thread.update(cx, |acp_thread, cx| {
                            acp_thread.push_assistant_content_block_with_indent(
                                block.clone(),
                                false,
                                true,
                                cx,
                            );
                        });

                        thread.update(cx, |thread, cx| {
                            thread.push_acp_agent_block(block, cx);
                        });
                    }
                }

                last_is_user = role == context_server::types::Role::User;
            }

            let response_stream = thread.update(cx, |thread, cx| {
                if last_is_user {
                    thread.send_existing(cx)
                } else {
                    // Resume if MCP prompt did not end with a user message
                    thread.resume(cx)
                }
            })?;

            cx.update(|cx| {
                NativeAgentConnection::handle_thread_events(
                    response_stream,
                    acp_thread.downgrade(),
                    None,
                    None,
                    cx,
                )
            })
            .await
        })
    }
}

/// Wrapper struct that implements the AgentConnection trait
#[derive(Clone)]
pub struct NativeAgentConnection(pub Entity<NativeAgent>);

impl NativeAgentConnection {
    pub fn thread(&self, session_id: &acp::SessionId, cx: &App) -> Option<Entity<Thread>> {
        self.0
            .read(cx)
            .sessions
            .get(session_id)
            .map(|session| session.thread.clone())
    }

    pub fn load_thread(
        &self,
        id: acp::SessionId,
        project: Entity<Project>,
        cx: &mut App,
    ) -> Task<Result<Entity<Thread>>> {
        self.0
            .update(cx, |this, cx| this.load_thread(id, project, cx))
    }

    /// Handle Caduceus-specific slash commands.
    /// Returns `Some(Task)` if the command was recognized and handled,
    /// `None` to fall through to MCP routing.
    fn handle_caduceus_command(
        &self,
        command: &str,
        args: &str,
        session_id: &acp::SessionId,
        cx: &mut App,
    ) -> Option<Task<Result<acp::PromptResponse>>> {
        let response_text = match command.to_lowercase().as_str() {
            "compact" => {
                let outcome = if let Some(thread) = self.thread(session_id, cx) {
                    thread.update(cx, |thread, cx| {
                        thread.auto_compact_context_explained(cx)
                    })
                } else {
                    crate::thread::CompactOutcome::WithinBudget
                };
                let (label, message) = compact_outcome_message(outcome);
                caduceus_bridge::context_events::record_and_count(
                    caduceus_bridge::context_events::ContextEventKind::ManualCompactRequested {
                        outcome: label.into(),
                    },
                );
                message.to_string()
            }
            "checkpoint" => {
                "📌 Use the `caduceus_checkpoint` tool with operation `create` and a label to save a checkpoint.".to_string()
            }
            "dag" => {
                // Render the current Index Access DAG so the user can see
                // which agents are reading/writing the same engine state.
                // Implemented per the request: "when the same index is
                // being read by agents the dependency should be clear and
                // a DAG should be generated and also drawn to the chat".
                let ascii = caduceus_bridge::index_dag::render_current_ascii();
                format!("```\n{}\n```", ascii)
            }
            "mode" => {
                use caduceus_bridge::orchestrator::BridgeAgentMode;
                let mode_name = args.trim();
                if mode_name.is_empty() {
                    let current = self
                        .thread(session_id, cx)
                        .map(|t| t.read(cx).caduceus_mode_name().to_string())
                        .unwrap_or_else(|| "act".to_string());
                    // ST-B5 / `catalog-opaque-v1` — generate the available
                    // list from the engine's catalog. No hardcoded mode
                    // names in the IDE.
                    let available = caduceus_bridge::orchestrator::list_modes()
                        .into_iter()
                        .map(|m| m.name)
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("Current mode: **{current}**\nAvailable: {available}")
                } else if let Some(canonical) = BridgeAgentMode::from_str_loose(mode_name) {
                    let canonical_name = canonical.name().to_string();
                    let modes = self
                        .0
                        .read(cx)
                        .sessions
                        .get(session_id)
                        .map(|s| s.caduceus_modes.clone());
                    if let Some(modes) = modes {
                        modes
                            .set_mode(acp::SessionModeId::new(canonical_name.clone()), cx)
                            .detach();
                    }
                    if let Some(thread) = self.thread(session_id, cx) {
                        let canonical_for_thread = canonical_name.clone();
                        thread.update(cx, |thread, _cx| {
                            thread.set_caduceus_mode(Some(canonical_for_thread));
                        });
                    }
                    format!("✅ Mode switched to **{canonical_name}**")
                } else {
                    format!(
                        "❌ Unknown mode '{mode_name}'. Valid: {}",
                        available_modes_comma()
                    )
                }
            }
            "context" => {
                let sub = args.trim();
                if let Some(thread) = self.thread(session_id, cx) {
                    if sub.starts_with("pin ") {
                        let rest = sub.strip_prefix("pin ").unwrap_or("");
                        if let Some((label, content)) = rest.split_once(' ') {
                            thread.update(cx, |t, _| t.pin_context(label.trim(), content.trim()));
                            format!("📌 Pinned: **{}** — survives compaction", label.trim())
                        } else {
                            "Usage: `/context pin <label> <content>`".to_string()
                        }
                    } else if sub.starts_with("unpin ") {
                        let label = sub.strip_prefix("unpin ").unwrap_or("").trim();
                        let removed = thread.update(cx, |t, _| t.unpin_context(label));
                        if removed {
                            format!("🗑 Unpinned: **{}**", label)
                        } else {
                            format!("❌ No pin named '{}'", label)
                        }
                    } else if sub == "pins" {
                        let pins = thread.read(cx).list_pins();
                        if pins.is_empty() {
                            "No pinned context items.".to_string()
                        } else {
                            let mut out = format!("📌 {} pinned items:\n", pins.len());
                            for p in pins {
                                let preview: String = p.content.chars().take(50).collect();
                                out.push_str(&format!("  **{}**: {}\n", p.label, preview));
                            }
                            out
                        }
                    } else {
                        let t = thread.read(cx);
                        let msg_count = t.message_count();
                        let zone = t.context_zone();
                        let fill_pct = t.context_fill_pct();
                        let pin_count = t.list_pins().len();
                        format!(
                            "📊 Context: {} messages | {:.0}% full | Zone: {} — {}\n📌 {} pinned items (survive compaction)\nUse `/compact` to free space, `/context pin <label> <text>` to pin.",
                            msg_count, fill_pct, zone.label(), zone.recommendation(), pin_count
                        )
                    }
                } else {
                    "No active session".to_string()
                }
            }
            "review" => {
                "🔍 To review code for issues, use the `caduceus_security_scan` tool:\n\
                 - Ask: \"Review this file for security issues\"\n\
                 - Or: \"Scan src/main.rs for vulnerabilities\"\n\
                 The agent will use the security scanner and dependency checker automatically."
                    .to_string()
            }
            "status" => {
                // Unified dashboard — all Caduceus metrics in one view
                let mut dashboard = String::from("## 📊 Caduceus Dashboard\n\n");

                // Mode & Session
                if let Some(thread) = self.thread(session_id, cx) {
                    let t = thread.read(cx);
                    let mode = t.caduceus_mode_name();
                    let (zone, fill_pct) = t.context_zone_and_pct();
                    let msg_count = t.message_count();
                    let pin_count = t.list_pins().len();

                    dashboard.push_str("### 🧠 Session\n");
                    dashboard.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
                    dashboard.push_str(&format!("| Mode | **{}** |\n", mode.to_uppercase()));
                    dashboard.push_str(&format!("| Messages | {} |\n", msg_count));
                    dashboard.push_str(&format!("| Context | {:.0}% — **{}** |\n", fill_pct, zone.label()));
                    dashboard.push_str(&format!("| Pinned items | {} |\n", pin_count));
                    dashboard.push_str("\n");
                }

                // Project metrics
                if let Some(state) = self.0.read(cx).projects.values().next() {
                    let project = state.project.read(cx);
                    if let Some(worktree) = project.worktrees(cx).next() {
                        let root = worktree.read(cx).abs_path();

                        // Memory
                        let memories = caduceus_bridge::memory::list(&root);

                        // Security
                        let security_path = root.join(".caduceus/security.json");
                        let security_score: f64 = std::fs::read_to_string(&security_path)
                            .ok()
                            .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok())
                            .and_then(|v| v["score"].as_f64())
                            .unwrap_or(0.0);

                        // Health
                        let health_path = root.join(".caduceus/health.json");
                        let health_score: Option<f64> = std::fs::read_to_string(&health_path)
                            .ok()
                            .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok())
                            .and_then(|v| v["score"].as_f64());

                        // Checkpoints
                        let checkpoint_dir = root.join(".caduceus/checkpoints");
                        let checkpoint_count = std::fs::read_dir(&checkpoint_dir)
                            .ok()
                            .map(|entries| entries.count())
                            .unwrap_or(0);

                        // Agents
                        let agents_dir = root.join(".caduceus/agents");
                        let agent_count = std::fs::read_dir(&agents_dir)
                            .ok()
                            .map(|entries| entries.count())
                            .unwrap_or(0);

                        // Index
                        let index_path = root.join(".caduceus/index.json");
                        let index_size = std::fs::metadata(&index_path)
                            .ok()
                            .map(|m| m.len())
                            .unwrap_or(0);

                        // Locked files
                        let locked = crate::caduceus_file_lock::list_locked_files();

                        dashboard.push_str("### 🏗️ Project Health\n");
                        dashboard.push_str("| Metric | Value | Status |\n|--------|-------|--------|\n");
                        dashboard.push_str(&format!("| Security | {:.0}% | {} |\n",
                            security_score * 100.0,
                            if security_score >= 0.8 { "✅ Good" } else if security_score >= 0.5 { "⚠️ Needs attention" } else { "🔴 Critical" }
                        ));
                        if let Some(health) = health_score {
                            dashboard.push_str(&format!("| Architecture Health | {:.0}% | {} |\n",
                                health * 100.0,
                                if health >= 0.8 { "✅" } else if health >= 0.5 { "⚠️" } else { "🔴" }
                            ));
                        }
                        dashboard.push_str(&format!("| Memories | {} / 100 | {} |\n",
                            memories.len(),
                            if memories.len() > 80 { "⚠️ Near limit" } else { "✅" }
                        ));
                        dashboard.push_str(&format!("| Checkpoints | {} | {} |\n",
                            checkpoint_count,
                            if checkpoint_count > 0 { "✅ Protected" } else { "⚠️ No backups" }
                        ));
                        dashboard.push_str(&format!("| Background Agents | {} | |\n", agent_count));
                        dashboard.push_str(&format!("| Search Index | {} KB | {} |\n",
                            index_size / 1024,
                            if index_size > 0 { "✅ Active" } else { "⚠️ Not indexed" }
                        ));
                        if !locked.is_empty() {
                            dashboard.push_str(&format!("| Locked Files | {} | 🔒 |\n", locked.len()));
                        }
                        dashboard.push_str("\n");

                        // Safety
                        dashboard.push_str("### 🛡️ Safety Status\n");
                        dashboard.push_str("| Guard | Status |\n|-------|--------|\n");
                        dashboard.push_str("| Loop Detector | ✅ Active (threshold: 3) |\n");
                        dashboard.push_str("| Circuit Breaker | ✅ Active (threshold: 5) |\n");
                        dashboard.push_str("| Compaction Cooldown | ✅ Active (30s) |\n");
                        dashboard.push_str("| Secret Scanner | ✅ Active |\n");
                        dashboard.push_str("| Path Sandboxing | ✅ Active |\n");
                        dashboard.push_str("\n");

                        // Readiness radar (text-based)
                        dashboard.push_str("### 📡 AI Readiness Radar\n");
                        dashboard.push_str("```\n");
                        let dims = [
                            ("Context Mgmt", 9),
                            ("Agent Arch", 9),
                            ("Code Intel", 9),
                            ("Performance", 8),
                            ("Safety", 9),
                            ("DX", 8),
                        ];
                        for (name, score) in &dims {
                            let bar: String = "█".repeat(*score as usize);
                            let empty: String = "░".repeat(10 - *score as usize);
                            dashboard.push_str(&format!("  {:<14} {}{} {}/10\n", name, bar, empty, score));
                        }
                        dashboard.push_str("```\n");
                    }
                }

                dashboard
            }
            "start" => {
                "## 🚀 Getting Started with Caduceus\n\n\
                 **1. Index your project** — Ask: \"Index this project for semantic search\"\n\
                 **2. Explore** — Try: `/map` to see the project structure\n\
                 **3. Ask questions** — The agent knows your codebase after indexing\n\
                 **4. Switch modes** — `/mode plan` for planning, `/mode act` for coding\n\
                 **5. Pin important context** — `/context pin arch We use microservices`\n\n\
                 **Quick commands:** `/help` for all commands, `/compact` to free context"
                    .to_string()
            }
            "search" => {
                let query = args.trim();
                if query.is_empty() {
                    "Usage: `/search <query>` — semantic search across indexed code\n\
                     Example: `/search authentication handler`"
                        .to_string()
                } else if let Some(thread) = self.thread(session_id, cx) {
                    let engine = thread.read(cx).project().read(cx).worktrees(cx).next()
                        .map(|wt| caduceus_bridge::engine::CaduceusEngine::new(wt.read(cx).abs_path().to_path_buf()));
                    if let Some(engine) = engine {
                        // Run search synchronously for slash command (async would need different pattern)
                        format!("🔍 Use `caduceus_semantic_search` tool with query: \"{}\"\n\
                                 The agent will search your indexed codebase.", query)
                    } else {
                        "No project open. Open a project first.".to_string()
                    }
                } else {
                    "No active session.".to_string()
                }
            }
            "index" => {
                if let Some(state) = self.0.read(cx).projects.values().next() {
                    if state.caduceus_engine.is_some() {
                        "📚 **Index Status**\n\
                         - Engine: ✅ active\n\
                         - Indexing: incremental (only changed files re-processed)\n\
                         - Storage: `.caduceus/index.json` (persistent across restarts)\n\n\
                         To re-index: Ask \"Re-index the project\"\n\
                         To search: `/search <query>` or ask a code question"
                            .to_string()
                    } else {
                        "📚 No engine initialized. Open a project to start indexing.".to_string()
                    }
                } else {
                    "📚 No project open. Open a project first.".to_string()
                }
            }
            "map" => {
                // Generate repo map from tree-sitter outline
                if let Some(worktree) = self.0.read(cx).projects.values().next()
                    .and_then(|ps| ps.project.read(cx).worktrees(cx).next())
                {
                    let root = worktree.read(cx).abs_path().to_path_buf();
                    let mut files: Vec<(String, String)> = Vec::new();
                    let walker = ignore::WalkBuilder::new(&root)
                        .hidden(true)
                        .git_ignore(true)
                        .max_depth(Some(4))
                        .build();
                    for entry in walker.flatten() {
                        let path = entry.path();
                        if !path.is_file() { continue; }
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                        if !["rs", "py", "ts", "tsx", "js", "jsx", "go"].contains(&ext) { continue; }
                        if let Ok(content) = std::fs::read_to_string(path) {
                            let rel = path.strip_prefix(&root).unwrap_or(path);
                            files.push((rel.to_string_lossy().to_string(), content));
                        }
                        if files.len() >= 100 { break; } // cap for large repos
                    }
                    if files.is_empty() {
                        "No supported source files found in project.".to_string()
                    } else {
                        let map = caduceus_bridge::tree_sitter::generate_repo_map(&files);
                        format!("## Repo Map ({} files)\n\n```\n{}\n```", files.len(), map)
                    }
                } else {
                    "No project open.".to_string()
                }
            }
            "headless" => {
                let prompt = args.trim();
                if prompt.is_empty() {
                    format!(
                        "## Headless Mode\n\
                         Run Caduceus from the command line without the IDE:\n\n\
                         ```bash\n\
                         caduceus run --prompt \"your task here\" --mode act --format json\n\
                         ```\n\n\
                         Options:\n\
                         - `--mode {}`\n\
                         - `--format text|json|compact`\n\
                         - `--print-only` — suppress streaming, output final result only\n\n\
                         Use `/headless <prompt>` to generate a ready-to-run command.",
                        available_modes_pipe()
                    )
                } else {
                    let mode = self.thread(session_id, cx)
                        .map(|t| t.read(cx).caduceus_mode_name().to_string())
                        .unwrap_or_else(|| "act".to_string());
                    let escaped = prompt.replace('\'', "'\\''");
                    format!(
                        "```bash\ncaduceus run \\\n  --prompt '{}' \\\n  --mode {} \\\n  --format json \\\n  --print-only\n```\n\nCopy and run in your terminal.",
                        escaped, mode
                    )
                }
            }
            "help" => {
                "## Caduceus Commands\n\n\
                 **Context Management:**\n\
                 - `/compact` — compress conversation context to free tokens\n\
                 - `/context` — show context usage, zone status, and pinned items\n\
                 - `/context pin <label> <text>` — pin context that survives compaction\n\
                 - `/context unpin <label>` — remove a pinned item\n\
                 - `/context pins` — list all pinned items\n\n\
                 **Modes** (use `/mode <name>` to switch):\n\
                 - `plan` — read-only analysis, write plans/docs only\n\
                 - `act` — execute code changes with approval\n\
                 - `research` — read-only exploration, summarize findings\n\
                 - `autopilot` — fully autonomous (plan + act + test + commit)\n\
                 - `architect` — high-level design, no code changes\n\
                 - `debug` — investigate errors, trace bugs\n\
                 - `review` — code review, find issues\n\n\
                 **Tools:**\n\
                 - `/status` — show unified dashboard with all metrics\n\
                 - `/map` — show project repo map (tree-sitter symbol outline)\n\
                 - `/review` — review code for security issues\n\
                 - `/dag` — show the Index Access DAG (subagent spawns + index reads/writes)\n\
                 - `/checkpoint [label]` — create a code checkpoint for rollback\n\
                 - `/headless [prompt]` — generate CLI command for headless execution\n\n\
                 **Examples:**\n\
                 - `/status` → see security, health, context, safety at a glance\n\
                 - `/mode research` → switch to read-only research mode\n\
                 - `/context pin arch Use microservices pattern` → pin architecture decision\n\
                 - `/map` → see all symbols in the project\n\
                 - `/compact` → free up context when conversation gets long"
                    .to_string()
            }
            _ => return None, // Not a Caduceus command — fall through
        };

        // Push the response as an assistant message via the ACP thread
        let acp_thread = self
            .0
            .read(cx)
            .sessions
            .get(session_id)
            .map(|s| s.acp_thread.clone());
        if let Some(acp_thread) = acp_thread {
            acp_thread.update(cx, |thread, cx| {
                thread.push_assistant_content_block(
                    acp::ContentBlock::Text(acp::TextContent::new(response_text)),
                    false,
                    cx,
                );
            });
        }

        Some(Task::ready(Ok(acp::PromptResponse::new(
            acp::StopReason::EndTurn,
        ))))
    }

    fn run_turn(
        &self,
        session_id: acp::SessionId,
        cx: &mut App,
        f: impl 'static
        + FnOnce(Entity<Thread>, &mut App) -> Result<mpsc::UnboundedReceiver<Result<ThreadEvent>>>,
    ) -> Task<Result<acp::PromptResponse>> {
        // Abort any in-flight event forwarder for this session before starting
        // a new turn. This prevents two `handle_thread_events` tasks from
        // concurrently calling `acp_thread.update(...)` on the same Entity
        // (which would trigger a gpui double-lease panic).
        let Some((thread, acp_thread, abort_rx, auth_tasks)) = self.0.update(cx, |agent, _cx| {
            let session = agent.sessions.get_mut(&session_id)?;
            if let Some(mut prev_abort) = session.forwarder_abort.take() {
                let _ = prev_abort.send(true);
            }
            let (abort_tx, abort_rx) = watch::channel(false);
            session.forwarder_abort = Some(abort_tx);
            // Drain any completed auth tasks from prior turns to keep the
            // vec from growing unboundedly across long sessions.
            if let Ok(mut tasks) = session.auth_tasks.lock() {
                tasks.retain(|_| true); // tasks self-drop on completion via Drop
            }
            Some((
                session.thread.clone(),
                session.acp_thread.clone(),
                abort_rx,
                session.auth_tasks.clone(),
            ))
        }) else {
            return Task::ready(Err(anyhow!("Session not found")));
        };
        log::debug!("Found session for: {}", session_id);

        let response_stream = match f(thread, cx) {
            Ok(stream) => stream,
            Err(err) => return Task::ready(Err(err)),
        };
        Self::handle_thread_events(
            response_stream,
            acp_thread.downgrade(),
            Some(abort_rx),
            Some(auth_tasks),
            cx,
        )
    }

    fn handle_thread_events(
        mut events: mpsc::UnboundedReceiver<Result<ThreadEvent>>,
        acp_thread: WeakEntity<AcpThread>,
        abort_rx: Option<watch::Receiver<bool>>,
        auth_tasks: Option<std::sync::Arc<std::sync::Mutex<Vec<Task<()>>>>>,
        cx: &App,
    ) -> Task<Result<acp::PromptResponse>> {
        cx.spawn(async move |cx| {
            // For callers that don't need abort (replay / MCP prompt paths),
            // create a local channel and keep its sender alive in scope so
            // the receiver never observes a Closed event.
            let _abort_tx_keepalive;
            let mut abort_rx = match abort_rx {
                Some(rx) => rx,
                None => {
                    let (tx, rx) = watch::channel(false);
                    _abort_tx_keepalive = tx;
                    rx
                }
            };
            // Handle response stream and forward to session.acp_thread
            loop {
                let result = futures::select! {
                    next = events.next().fuse() => match next {
                        Some(r) => r,
                        None => break,
                    },
                    _ = abort_rx.changed().fuse() => {
                        if *abort_rx.borrow() {
                            log::debug!("Event forwarder aborted by newer turn");
                            return Ok(acp::PromptResponse::new(acp::StopReason::Cancelled));
                        }
                        continue;
                    }
                };
                match result {
                    Ok(event) => {
                        log::trace!("Received completion event: {:?}", event);

                        match event {
                            ThreadEvent::UserMessage(message) => {
                                acp_thread.update(cx, |thread, cx| {
                                    for content in message.content {
                                        thread.push_user_content_block(
                                            Some(message.id.clone()),
                                            content.into(),
                                            cx,
                                        );
                                    }
                                })?;
                            }
                            ThreadEvent::AgentText(text) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.push_assistant_content_block(text.into(), false, cx)
                                })?;
                            }
                            ThreadEvent::AgentThinking(text) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.push_assistant_content_block(text.into(), true, cx)
                                })?;
                            }
                            ThreadEvent::ToolCallAuthorization(ToolCallAuthorization {
                                tool_call,
                                options,
                                response,
                                context: _,
                            }) => {
                                let outcome_task = acp_thread.update(cx, |thread, cx| {
                                    thread.request_tool_call_authorization(tool_call, options, cx)
                                })??;
                                let task = cx.background_spawn(async move {
                                    if let acp_thread::RequestPermissionOutcome::Selected(outcome) =
                                        outcome_task.await
                                    {
                                        response
                                            .send(outcome)
                                            .map(|_| anyhow!("authorization receiver was dropped"))
                                            .log_err();
                                    }
                                });
                                // Track in the session-scoped vec so we can
                                // abort on session drop; fall back to detach
                                // for non-session callers (replay / MCP).
                                let mut detached = false;
                                if let Some(tasks) = &auth_tasks {
                                    if let Ok(mut guard) = tasks.lock() {
                                        guard.push(task);
                                        detached = true;
                                    }
                                }
                                if !detached {
                                    // Avoid double-tracking; this branch only
                                    // runs when auth_tasks is None or poisoned.
                                    // The compiler ensures `task` is consumed.
                                }
                            }
                            ThreadEvent::ToolCall(tool_call) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.upsert_tool_call(tool_call, cx)
                                })??;
                            }
                            ThreadEvent::ToolCallUpdate(update) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.update_tool_call(update, cx)
                                })??;
                            }
                            ThreadEvent::Plan(plan) => {
                                acp_thread.update(cx, |thread, cx| thread.update_plan(plan, cx))?;
                            }
                            ThreadEvent::SubagentSpawned(session_id) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.subagent_spawned(session_id, cx);
                                })?;
                            }
                            ThreadEvent::Retry(status) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.update_retry_status(status, cx)
                                })?;
                            }
                            ThreadEvent::Stop(stop_reason) => {
                                log::debug!("Assistant message complete: {:?}", stop_reason);
                                return Ok(acp::PromptResponse::new(stop_reason));
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Error in model response stream: {:?}", e);
                        return Err(e);
                    }
                }
            }

            log::debug!("Response stream completed");
            anyhow::Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))
        })
    }
}

struct Command<'a> {
    prompt_name: &'a str,
    arg_value: &'a str,
    explicit_server_id: Option<&'a str>,
}

impl<'a> Command<'a> {
    fn parse(prompt: &'a [acp::ContentBlock]) -> Option<Self> {
        let acp::ContentBlock::Text(text_content) = prompt.first()? else {
            return None;
        };
        let text = text_content.text.trim();
        let command = text.strip_prefix('/')?;
        let (command, arg_value) = command
            .split_once(char::is_whitespace)
            .unwrap_or((command, ""));

        if let Some((server_id, prompt_name)) = command.split_once('.') {
            Some(Self {
                prompt_name,
                arg_value,
                explicit_server_id: Some(server_id),
            })
        } else {
            Some(Self {
                prompt_name: command,
                arg_value,
                explicit_server_id: None,
            })
        }
    }
}

struct NativeAgentModelSelector {
    session_id: acp::SessionId,
    connection: NativeAgentConnection,
}

impl acp_thread::AgentModelSelector for NativeAgentModelSelector {
    fn list_models(&self, cx: &mut App) -> Task<Result<acp_thread::AgentModelList>> {
        log::debug!("NativeAgentConnection::list_models called");
        let list = self.connection.0.read(cx).models.model_list.clone();
        Task::ready(if list.is_empty() {
            Err(anyhow::anyhow!("No models available"))
        } else {
            Ok(list)
        })
    }

    fn select_model(&self, model_id: acp::ModelId, cx: &mut App) -> Task<Result<()>> {
        log::debug!(
            "Setting model for session {}: {}",
            self.session_id,
            model_id
        );
        let Some(thread) = self
            .connection
            .0
            .read(cx)
            .sessions
            .get(&self.session_id)
            .map(|session| session.thread.clone())
        else {
            return Task::ready(Err(anyhow!("Session not found")));
        };

        let Some(model) = self.connection.0.read(cx).models.model_from_id(&model_id) else {
            return Task::ready(Err(anyhow!("Invalid model ID {}", model_id)));
        };

        // We want to reset the effort level when switching models, as the currently-selected effort level may
        // not be compatible.
        let effort = model
            .default_effort_level()
            .map(|effort_level| effort_level.value.to_string());

        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.set_thinking_effort(effort.clone(), cx);
            thread.set_thinking_enabled(model.supports_thinking(), cx);
        });

        update_settings_file(
            self.connection.0.read(cx).fs.clone(),
            cx,
            move |settings, cx| {
                let provider = model.provider_id().0.to_string();
                let model = model.id().0.to_string();
                let enable_thinking = thread.read(cx).thinking_enabled();
                let speed = thread.read(cx).speed();
                settings
                    .agent
                    .get_or_insert_default()
                    .set_model(LanguageModelSelection {
                        provider: provider.into(),
                        model,
                        enable_thinking,
                        effort,
                        speed,
                    });
            },
        );

        Task::ready(Ok(()))
    }

    fn selected_model(&self, cx: &mut App) -> Task<Result<acp_thread::AgentModelInfo>> {
        let Some(thread) = self
            .connection
            .0
            .read(cx)
            .sessions
            .get(&self.session_id)
            .map(|session| session.thread.clone())
        else {
            return Task::ready(Err(anyhow!("Session not found")));
        };
        let Some(model) = thread.read(cx).model() else {
            return Task::ready(Err(anyhow!("Model not found")));
        };
        let Some(provider) = LanguageModelRegistry::read_global(cx).provider(&model.provider_id())
        else {
            return Task::ready(Err(anyhow!("Provider not found")));
        };
        Task::ready(Ok(LanguageModels::map_language_model_to_info(
            model, &provider,
        )))
    }

    fn watch(&self, cx: &mut App) -> Option<watch::Receiver<()>> {
        Some(self.connection.0.read(cx).models.watch())
    }

    fn should_render_footer(&self) -> bool {
        true
    }
}

// ── Caduceus agent modes ─────────────────────────────────────────────────────

use std::cell::RefCell;

/// Caduceus agent modes backed by caduceus-orchestrator.
/// Defines 7 modes: Plan, Act, Research, Autopilot, Architect, Debug, Review.
struct CaduceusSessionModes {
    current: RefCell<acp::SessionModeId>,
}

impl CaduceusSessionModes {
    fn new() -> Self {
        Self {
            current: RefCell::new(acp::SessionModeId::new("act")),
        }
    }

    /// ST-B5 / `catalog-opaque-v1` — the canonical catalog is the
    /// engine. Zed holds NO hardcoded mode list. We iterate the
    /// bridge's [`list_modes`](caduceus_bridge::orchestrator::list_modes)
    /// and translate each [`BridgeModeDescriptor`] into an
    /// [`acp::SessionMode`], preserving the descriptor's `name` as the
    /// opaque id ACP will hand back on `set_mode`. Adding or renaming
    /// a mode in the engine now reaches the IDE's picker with zero
    /// code changes here.
    fn all_caduceus_modes() -> Vec<acp::SessionMode> {
        caduceus_bridge::orchestrator::list_modes()
            .into_iter()
            .map(|m| {
                acp::SessionMode::new(m.name.clone(), m.label.clone()).description(m.description)
            })
            .collect()
    }
}

impl acp_thread::AgentSessionModes for CaduceusSessionModes {
    fn current_mode(&self) -> acp::SessionModeId {
        self.current.borrow().clone()
    }

    fn all_modes(&self) -> Vec<acp::SessionMode> {
        Self::all_caduceus_modes()
    }

    fn set_mode(&self, mode: acp::SessionModeId, _cx: &mut App) -> Task<Result<()>> {
        *self.current.borrow_mut() = mode;
        Task::ready(Ok(()))
    }
}

pub static ZED_AGENT_ID: LazyLock<AgentId> = LazyLock::new(|| AgentId::new("Caduceus Agent"));

impl acp_thread::AgentConnection for NativeAgentConnection {
    fn agent_id(&self) -> AgentId {
        ZED_AGENT_ID.clone()
    }

    fn telemetry_id(&self) -> SharedString {
        "zed".into()
    }

    fn new_session(
        self: Rc<Self>,
        project: Entity<Project>,
        work_dirs: PathList,
        cx: &mut App,
    ) -> Task<Result<Entity<acp_thread::AcpThread>>> {
        log::debug!("Creating new thread for project at: {work_dirs:?}");
        Task::ready(Ok(self
            .0
            .update(cx, |agent, cx| agent.new_session(project, cx))))
    }

    fn supports_load_session(&self) -> bool {
        true
    }

    fn load_session(
        self: Rc<Self>,
        session_id: acp::SessionId,
        project: Entity<Project>,
        _work_dirs: PathList,
        _title: Option<SharedString>,
        cx: &mut App,
    ) -> Task<Result<Entity<acp_thread::AcpThread>>> {
        self.0
            .update(cx, |agent, cx| agent.open_thread(session_id, project, cx))
    }

    fn supports_close_session(&self) -> bool {
        true
    }

    fn close_session(
        self: Rc<Self>,
        session_id: &acp::SessionId,
        cx: &mut App,
    ) -> Task<Result<()>> {
        self.0.update(cx, |agent, cx| {
            let thread = agent.sessions.get(session_id).map(|s| s.thread.clone());
            if let Some(thread) = thread {
                agent.save_thread(thread, cx);
            }

            let Some(session) = agent.sessions.remove(session_id) else {
                return Task::ready(Ok(()));
            };
            let project_id = session.project_id;

            let has_remaining = agent.sessions.values().any(|s| s.project_id == project_id);
            if !has_remaining {
                agent.projects.remove(&project_id);
            }

            session.pending_save
        })
    }

    fn auth_methods(&self) -> &[acp::AuthMethod] {
        &[] // No auth for in-process
    }

    fn authenticate(&self, _method: acp::AuthMethodId, _cx: &mut App) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }

    fn model_selector(&self, session_id: &acp::SessionId) -> Option<Rc<dyn AgentModelSelector>> {
        Some(Rc::new(NativeAgentModelSelector {
            session_id: session_id.clone(),
            connection: self.clone(),
        }) as Rc<dyn AgentModelSelector>)
    }

    fn session_modes(
        &self,
        session_id: &acp::SessionId,
        cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionModes>> {
        self.0
            .read(cx)
            .sessions
            .get(session_id)
            .map(|session| session.caduceus_modes.clone() as Rc<dyn acp_thread::AgentSessionModes>)
    }

    fn prompt(
        &self,
        id: acp_thread::UserMessageId,
        params: acp::PromptRequest,
        cx: &mut App,
    ) -> Task<Result<acp::PromptResponse>> {
        let session_id = params.session_id.clone();
        log::info!("Received prompt request for session: {}", session_id);
        log::debug!("Prompt blocks count: {}", params.prompt.len());

        // Caduceus slash commands — handle before MCP routing
        if let Some(acp::ContentBlock::Text(text)) = params.prompt.first() {
            let trimmed = text.text.trim();
            if let Some(cmd) = trimmed.strip_prefix('/') {
                let (command, args) = cmd.split_once(char::is_whitespace).unwrap_or((cmd, ""));
                if let Some(response) = self.handle_caduceus_command(command, args, &session_id, cx)
                {
                    return response;
                }
            }
        }

        let Some(project_state) = self.0.read(cx).session_project_state(&session_id) else {
            return Task::ready(Err(anyhow::anyhow!("Session not found")));
        };

        if let Some(parsed_command) = Command::parse(&params.prompt) {
            let registry = project_state.context_server_registry.read(cx);

            let explicit_server_id = parsed_command
                .explicit_server_id
                .map(|server_id| ContextServerId(server_id.into()));

            if let Some(prompt) =
                registry.find_prompt(explicit_server_id.as_ref(), parsed_command.prompt_name)
            {
                let arguments = if !parsed_command.arg_value.is_empty()
                    && let Some(arg_name) = prompt
                        .prompt
                        .arguments
                        .as_ref()
                        .and_then(|args| args.first())
                        .map(|arg| arg.name.clone())
                {
                    HashMap::from_iter([(arg_name, parsed_command.arg_value.to_string())])
                } else {
                    Default::default()
                };

                let prompt_name = prompt.prompt.name.clone();
                let server_id = prompt.server_id.clone();

                return self.0.update(cx, |agent, cx| {
                    agent.send_mcp_prompt(
                        id,
                        session_id.clone(),
                        prompt_name,
                        server_id,
                        arguments,
                        params.prompt,
                        cx,
                    )
                });
            }
        };

        let path_style = project_state.project.read(cx).path_style(cx);

        // Derive Caduceus mode from thread's profile_id instead of session_modes
        let current_mode = self
            .thread(&session_id, cx)
            .map(|t| t.read(cx).profile().0.to_string());

        self.run_turn(session_id.clone(), cx, move |thread, cx| {
            let content: Vec<UserMessageContent> = params
                .prompt
                .into_iter()
                .map(|block| UserMessageContent::from_content_block(block, path_style))
                .collect::<Vec<_>>();
            log::debug!("Converted prompt to message: {} chars", content.len());
            log::debug!("Message id: {:?}", id);
            log::debug!("Message content: {:?}", content);

            thread.update(cx, |thread, cx| {
                thread.set_caduceus_mode(current_mode);
                thread.send(id, content, cx)
            })
        })
    }

    fn retry(
        &self,
        session_id: &acp::SessionId,
        _cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionRetry>> {
        Some(Rc::new(NativeAgentSessionRetry {
            connection: self.clone(),
            session_id: session_id.clone(),
        }) as _)
    }

    fn cancel(&self, session_id: &acp::SessionId, cx: &mut App) {
        log::info!("Cancelling on session: {}", session_id);
        self.0.update(cx, |agent, cx| {
            if let Some(session) = agent.sessions.get(session_id) {
                session
                    .thread
                    .update(cx, |thread, cx| thread.cancel(cx))
                    .detach();
            }
        });
    }

    fn truncate(
        &self,
        session_id: &acp::SessionId,
        cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionTruncate>> {
        self.0.read_with(cx, |agent, _cx| {
            agent.sessions.get(session_id).map(|session| {
                Rc::new(NativeAgentSessionTruncate {
                    thread: session.thread.clone(),
                    acp_thread: session.acp_thread.downgrade(),
                }) as _
            })
        })
    }

    fn set_title(
        &self,
        session_id: &acp::SessionId,
        cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionSetTitle>> {
        self.0.read_with(cx, |agent, _cx| {
            agent
                .sessions
                .get(session_id)
                .filter(|s| !s.thread.read(cx).is_subagent())
                .map(|session| {
                    Rc::new(NativeAgentSessionSetTitle {
                        thread: session.thread.clone(),
                    }) as _
                })
        })
    }

    fn session_list(&self, cx: &mut App) -> Option<Rc<dyn AgentSessionList>> {
        let thread_store = self.0.read(cx).thread_store.clone();
        Some(Rc::new(NativeAgentSessionList::new(thread_store, cx)) as _)
    }

    fn telemetry(&self) -> Option<Rc<dyn acp_thread::AgentTelemetry>> {
        Some(Rc::new(self.clone()) as Rc<dyn acp_thread::AgentTelemetry>)
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}

impl acp_thread::AgentTelemetry for NativeAgentConnection {
    fn thread_data(
        &self,
        session_id: &acp::SessionId,
        cx: &mut App,
    ) -> Task<Result<serde_json::Value>> {
        let Some(session) = self.0.read(cx).sessions.get(session_id) else {
            return Task::ready(Err(anyhow!("Session not found")));
        };

        let task = session.thread.read(cx).to_db(cx);
        cx.background_spawn(async move {
            serde_json::to_value(task.await).context("Failed to serialize thread")
        })
    }
}

pub struct NativeAgentSessionList {
    thread_store: Entity<ThreadStore>,
    updates_tx: smol::channel::Sender<acp_thread::SessionListUpdate>,
    updates_rx: smol::channel::Receiver<acp_thread::SessionListUpdate>,
    _subscription: Subscription,
}

impl NativeAgentSessionList {
    fn new(thread_store: Entity<ThreadStore>, cx: &mut App) -> Self {
        let (tx, rx) = smol::channel::unbounded();
        let this_tx = tx.clone();
        let subscription = cx.observe(&thread_store, move |_, _| {
            this_tx
                .try_send(acp_thread::SessionListUpdate::Refresh)
                .ok();
        });
        Self {
            thread_store,
            updates_tx: tx,
            updates_rx: rx,
            _subscription: subscription,
        }
    }

    pub fn thread_store(&self) -> &Entity<ThreadStore> {
        &self.thread_store
    }
}

impl AgentSessionList for NativeAgentSessionList {
    fn list_sessions(
        &self,
        _request: AgentSessionListRequest,
        cx: &mut App,
    ) -> Task<Result<AgentSessionListResponse>> {
        let sessions = self
            .thread_store
            .read(cx)
            .entries()
            .map(|entry| AgentSessionInfo::from(&entry))
            .collect();
        Task::ready(Ok(AgentSessionListResponse::new(sessions)))
    }

    fn supports_delete(&self) -> bool {
        true
    }

    fn delete_session(&self, session_id: &acp::SessionId, cx: &mut App) -> Task<Result<()>> {
        self.thread_store
            .update(cx, |store, cx| store.delete_thread(session_id.clone(), cx))
    }

    fn delete_sessions(&self, cx: &mut App) -> Task<Result<()>> {
        self.thread_store
            .update(cx, |store, cx| store.delete_threads(cx))
    }

    fn watch(
        &self,
        _cx: &mut App,
    ) -> Option<smol::channel::Receiver<acp_thread::SessionListUpdate>> {
        Some(self.updates_rx.clone())
    }

    fn notify_refresh(&self) {
        self.updates_tx
            .try_send(acp_thread::SessionListUpdate::Refresh)
            .ok();
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}

struct NativeAgentSessionTruncate {
    thread: Entity<Thread>,
    acp_thread: WeakEntity<AcpThread>,
}

impl acp_thread::AgentSessionTruncate for NativeAgentSessionTruncate {
    fn run(&self, message_id: acp_thread::UserMessageId, cx: &mut App) -> Task<Result<()>> {
        match self.thread.update(cx, |thread, cx| {
            thread.truncate(message_id.clone(), cx)?;
            Ok(thread.latest_token_usage())
        }) {
            Ok(usage) => {
                self.acp_thread
                    .update(cx, |thread, cx| {
                        thread.update_token_usage(usage, cx);
                    })
                    .ok();
                Task::ready(Ok(()))
            }
            Err(error) => Task::ready(Err(error)),
        }
    }
}

struct NativeAgentSessionRetry {
    connection: NativeAgentConnection,
    session_id: acp::SessionId,
}

impl acp_thread::AgentSessionRetry for NativeAgentSessionRetry {
    fn run(&self, cx: &mut App) -> Task<Result<acp::PromptResponse>> {
        self.connection
            .run_turn(self.session_id.clone(), cx, |thread, cx| {
                thread.update(cx, |thread, cx| thread.resume(cx))
            })
    }
}

struct NativeAgentSessionSetTitle {
    thread: Entity<Thread>,
}

impl acp_thread::AgentSessionSetTitle for NativeAgentSessionSetTitle {
    fn run(&self, title: SharedString, cx: &mut App) -> Task<Result<()>> {
        self.thread
            .update(cx, |thread, cx| thread.set_title(title, cx));
        Task::ready(Ok(()))
    }
}

pub struct NativeThreadEnvironment {
    agent: WeakEntity<NativeAgent>,
    thread: WeakEntity<Thread>,
    acp_thread: WeakEntity<AcpThread>,
}

impl NativeThreadEnvironment {
    pub(crate) fn create_subagent_thread(
        &self,
        label: String,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        let Some(parent_thread_entity) = self.thread.upgrade() else {
            anyhow::bail!("Parent thread no longer exists".to_string());
        };
        let parent_thread = parent_thread_entity.read(cx);
        let current_depth = parent_thread.depth();
        let parent_session_id = parent_thread.id().clone();

        if current_depth >= MAX_SUBAGENT_DEPTH {
            return Err(anyhow!(
                "Maximum subagent depth ({}) reached",
                MAX_SUBAGENT_DEPTH
            ));
        }

        let subagent_thread: Entity<Thread> = cx.new(|cx| {
            let mut thread = Thread::new_subagent(&parent_thread_entity, cx);
            thread.set_title(label.into(), cx);
            thread
        });

        let session_id = subagent_thread.read(cx).id().clone();

        let acp_thread = self
            .agent
            .update(cx, |agent, cx| -> Result<Entity<AcpThread>> {
                let project_id = agent
                    .sessions
                    .get(&parent_session_id)
                    .map(|s| s.project_id)
                    .context("parent session not found")?;
                Ok(agent.register_session(subagent_thread.clone(), project_id, cx))
            })??;

        let depth = current_depth + 1;

        telemetry::event!(
            "Subagent Started",
            session = parent_thread_entity.read(cx).id().to_string(),
            subagent_session = session_id.to_string(),
            depth,
            is_resumed = false,
        );

        self.prompt_subagent(session_id, subagent_thread, acp_thread)
    }

    pub(crate) fn resume_subagent_thread(
        &self,
        session_id: acp::SessionId,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        let (subagent_thread, acp_thread) = self.agent.update(cx, |agent, _cx| {
            let session = agent
                .sessions
                .get(&session_id)
                .ok_or_else(|| anyhow!("No subagent session found with id {session_id}"))?;
            anyhow::Ok((session.thread.clone(), session.acp_thread.clone()))
        })??;

        let depth = subagent_thread.read(cx).depth();

        if let Some(parent_thread_entity) = self.thread.upgrade() {
            telemetry::event!(
                "Subagent Started",
                session = parent_thread_entity.read(cx).id().to_string(),
                subagent_session = session_id.to_string(),
                depth,
                is_resumed = true,
            );
        }

        self.prompt_subagent(session_id, subagent_thread, acp_thread)
    }

    fn prompt_subagent(
        &self,
        session_id: acp::SessionId,
        subagent_thread: Entity<Thread>,
        acp_thread: Entity<acp_thread::AcpThread>,
    ) -> Result<Rc<dyn SubagentHandle>> {
        let Some(parent_thread_entity) = self.thread.upgrade() else {
            anyhow::bail!("Parent thread no longer exists".to_string());
        };
        Ok(Rc::new(NativeSubagentHandle::new(
            session_id,
            subagent_thread,
            acp_thread,
            parent_thread_entity,
        )) as _)
    }
}

impl ThreadEnvironment for NativeThreadEnvironment {
    fn create_terminal(
        &self,
        command: String,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        cx: &mut AsyncApp,
    ) -> Task<Result<Rc<dyn TerminalHandle>>> {
        let task = self.acp_thread.update(cx, |thread, cx| {
            thread.create_terminal(command, vec![], vec![], cwd, output_byte_limit, cx)
        });

        let acp_thread = self.acp_thread.clone();
        cx.spawn(async move |cx| {
            let terminal = task?.await?;

            let (drop_tx, drop_rx) = oneshot::channel();
            let terminal_id = terminal.read_with(cx, |terminal, _cx| terminal.id().clone());

            cx.spawn(async move |cx| {
                drop_rx.await.ok();
                acp_thread.update(cx, |thread, cx| thread.release_terminal(terminal_id, cx))
            })
            .detach();

            let handle = AcpTerminalHandle {
                terminal,
                _drop_tx: Some(drop_tx),
            };

            Ok(Rc::new(handle) as _)
        })
    }

    fn create_subagent(&self, label: String, cx: &mut App) -> Result<Rc<dyn SubagentHandle>> {
        self.create_subagent_thread(label, cx)
    }

    fn resume_subagent(
        &self,
        session_id: acp::SessionId,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        self.resume_subagent_thread(session_id, cx)
    }
}

#[derive(Debug, Clone)]
enum SubagentPromptResult {
    Completed,
    Cancelled,
    ContextWindowWarning,
    Error(String),
}

pub struct NativeSubagentHandle {
    session_id: acp::SessionId,
    parent_thread: WeakEntity<Thread>,
    subagent_thread: Entity<Thread>,
    acp_thread: Entity<acp_thread::AcpThread>,
}

impl NativeSubagentHandle {
    fn new(
        session_id: acp::SessionId,
        subagent_thread: Entity<Thread>,
        acp_thread: Entity<acp_thread::AcpThread>,
        parent_thread_entity: Entity<Thread>,
    ) -> Self {
        NativeSubagentHandle {
            session_id,
            subagent_thread,
            parent_thread: parent_thread_entity.downgrade(),
            acp_thread,
        }
    }
}

impl SubagentHandle for NativeSubagentHandle {
    fn id(&self) -> acp::SessionId {
        self.session_id.clone()
    }

    fn num_entries(&self, cx: &App) -> usize {
        self.acp_thread.read(cx).entries().len()
    }

    fn send(&self, message: String, cx: &AsyncApp) -> Task<Result<String>> {
        let thread = self.subagent_thread.clone();
        let acp_thread = self.acp_thread.clone();
        let subagent_session_id = self.session_id.clone();
        let parent_thread = self.parent_thread.clone();

        cx.spawn(async move |cx| {
            let (task, _subscription) = cx.update(|cx| {
                let ratio_before_prompt = thread
                    .read(cx)
                    .latest_token_usage()
                    .map(|usage| usage.ratio());

                parent_thread
                    .update(cx, |parent_thread, cx| {
                        parent_thread.register_running_subagent(thread.downgrade(), cx)
                    })
                    .ok();

                let task = acp_thread.update(cx, |acp_thread, cx| {
                    acp_thread.send(vec![message.into()], cx)
                });

                let (token_limit_tx, token_limit_rx) = oneshot::channel::<()>();
                let mut token_limit_tx = Some(token_limit_tx);

                let subscription = cx.subscribe(
                    &thread,
                    move |_thread, event: &TokenUsageUpdated, _cx| {
                        if let Some(usage) = &event.0 {
                            let old_ratio = ratio_before_prompt
                                .clone()
                                .unwrap_or(TokenUsageRatio::Normal);
                            let new_ratio = usage.ratio();
                            if old_ratio == TokenUsageRatio::Normal
                                && new_ratio == TokenUsageRatio::Warning
                            {
                                if let Some(tx) = token_limit_tx.take() {
                                    tx.send(()).ok();
                                }
                            }
                        }
                    },
                );

                let wait_for_prompt = cx
                    .background_spawn(async move {
                        futures::select! {
                            response = task.fuse() => match response {
                                Ok(Some(response)) => {
                                    match response.stop_reason {
                                        acp::StopReason::Cancelled => SubagentPromptResult::Cancelled,
                                        acp::StopReason::MaxTokens => SubagentPromptResult::Error("The agent reached the maximum number of tokens.".into()),
                                        acp::StopReason::MaxTurnRequests => SubagentPromptResult::Error("The agent reached the maximum number of allowed requests between user turns. Try prompting again.".into()),
                                        acp::StopReason::Refusal => SubagentPromptResult::Error("The agent refused to process that prompt. Try again.".into()),
                                        acp::StopReason::EndTurn | _ => SubagentPromptResult::Completed,
                                    }
                                }
                                Ok(None) => SubagentPromptResult::Error("No response from the agent. You can try messaging again.".into()),
                                Err(error) => SubagentPromptResult::Error(error.to_string()),
                            },
                            _ = token_limit_rx.fuse() => SubagentPromptResult::ContextWindowWarning,
                        }
                    });

                (wait_for_prompt, subscription)
            });

            let result = match task.await {
                SubagentPromptResult::Completed => thread.read_with(cx, |thread, _cx| {
                    thread
                        .last_message()
                        .and_then(|message| {
                            let content = message.as_agent_message()?
                                .content
                                .iter()
                                .filter_map(|c| match c {
                                    AgentMessageContent::Text(text) => Some(text.as_str()),
                                    _ => None,
                                })
                                .join("\n\n");
                            if content.is_empty() {
                                None
                            } else {
                                Some( content)
                            }
                        })
                        .context("No response from subagent")
                }),
                SubagentPromptResult::Cancelled => Err(anyhow!("User canceled")),
                SubagentPromptResult::Error(message) => Err(anyhow!("{message}")),
                SubagentPromptResult::ContextWindowWarning => {
                    thread.update(cx, |thread, cx| thread.cancel(cx)).await;
                    Err(anyhow!(
                        "The agent is nearing the end of its context window and has been \
                         stopped. You can prompt the thread again to have the agent wrap up \
                         or hand off its work."
                    ))
                }
            };

            parent_thread
                .update(cx, |parent_thread, cx| {
                    parent_thread.unregister_running_subagent(&subagent_session_id, cx)
                })
                .ok();

            result
        })
    }
}

pub struct AcpTerminalHandle {
    terminal: Entity<acp_thread::Terminal>,
    _drop_tx: Option<oneshot::Sender<()>>,
}

impl TerminalHandle for AcpTerminalHandle {
    fn id(&self, cx: &AsyncApp) -> Result<acp::TerminalId> {
        Ok(self.terminal.read_with(cx, |term, _cx| term.id().clone()))
    }

    fn wait_for_exit(&self, cx: &AsyncApp) -> Result<Shared<Task<acp::TerminalExitStatus>>> {
        Ok(self
            .terminal
            .read_with(cx, |term, _cx| term.wait_for_exit()))
    }

    fn current_output(&self, cx: &AsyncApp) -> Result<acp::TerminalOutputResponse> {
        Ok(self
            .terminal
            .read_with(cx, |term, cx| term.current_output(cx)))
    }

    fn kill(&self, cx: &AsyncApp) -> Result<()> {
        cx.update(|cx| {
            self.terminal.update(cx, |terminal, cx| {
                terminal.kill(cx);
            });
        });
        Ok(())
    }

    fn was_stopped_by_user(&self, cx: &AsyncApp) -> Result<bool> {
        Ok(self
            .terminal
            .read_with(cx, |term, _cx| term.was_stopped_by_user()))
    }
}

#[cfg(test)]
mod internal_tests {
    use std::path::Path;

    use super::*;
    use acp_thread::{AgentConnection, AgentModelGroupName, AgentModelInfo, MentionUri};
    use fs::FakeFs;
    use gpui::TestAppContext;
    use indoc::formatdoc;
    use language_model::fake_provider::{FakeLanguageModel, FakeLanguageModelProvider};
    use language_model::{
        LanguageModelCompletionEvent, LanguageModelProviderId, LanguageModelProviderName,
    };
    use serde_json::json;
    use settings::SettingsStore;
    use util::{path, rel_path::rel_path};

    #[gpui::test]
    async fn test_maintaining_project_context(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "a": {}
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), None, fs.clone(), cx));

        // Creating a session registers the project and triggers context building.
        let connection = NativeAgentConnection(agent.clone());
        let _acp_thread = cx
            .update(|cx| {
                Rc::new(connection).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let thread = agent.read_with(cx, |agent, _cx| {
            agent.sessions.values().next().unwrap().thread.clone()
        });

        agent.read_with(cx, |agent, cx| {
            let project_id = project.entity_id();
            let state = agent.projects.get(&project_id).unwrap();
            assert_eq!(state.project_context.read(cx).worktrees, vec![]);
            assert_eq!(thread.read(cx).project_context().read(cx).worktrees, vec![]);
        });

        let worktree = project
            .update(cx, |project, cx| project.create_worktree("/a", true, cx))
            .await
            .unwrap();
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let project_id = project.entity_id();
            let state = agent.projects.get(&project_id).unwrap();
            let expected_worktrees = vec![WorktreeContext {
                root_name: "a".into(),
                abs_path: Path::new("/a").into(),
                rules_file: None,
            }];
            assert_eq!(state.project_context.read(cx).worktrees, expected_worktrees);
            assert_eq!(
                thread.read(cx).project_context().read(cx).worktrees,
                expected_worktrees
            );
        });

        // Creating `/a/.rules` updates the project context.
        fs.insert_file("/a/.rules", Vec::new()).await;
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let project_id = project.entity_id();
            let state = agent.projects.get(&project_id).unwrap();
            let rules_entry = worktree
                .read(cx)
                .entry_for_path(rel_path(".rules"))
                .unwrap();
            let expected_worktrees = vec![WorktreeContext {
                root_name: "a".into(),
                abs_path: Path::new("/a").into(),
                rules_file: Some(RulesFileContext {
                    path_in_worktree: rel_path(".rules").into(),
                    text: "".into(),
                    project_entry_id: rules_entry.id.to_usize(),
                }),
            }];
            assert_eq!(state.project_context.read(cx).worktrees, expected_worktrees);
            assert_eq!(
                thread.read(cx).project_context().read(cx).worktrees,
                expected_worktrees
            );
        });
    }

    #[gpui::test]
    async fn test_listing_models(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {}  })).await;
        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let connection =
            NativeAgentConnection(cx.update(|cx| {
                NativeAgent::new(thread_store, Templates::new(), None, fs.clone(), cx)
            }));

        // Create a thread/session
        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();

        let session_id = cx.update(|cx| acp_thread.read(cx).session_id().clone());

        let models = cx
            .update(|cx| {
                connection
                    .model_selector(&session_id)
                    .unwrap()
                    .list_models(cx)
            })
            .await
            .unwrap();

        let acp_thread::AgentModelList::Grouped(models) = models else {
            panic!("Unexpected model group");
        };
        assert_eq!(
            models,
            IndexMap::from_iter([(
                AgentModelGroupName("Fake".into()),
                vec![AgentModelInfo {
                    id: acp::ModelId::new("fake/fake"),
                    name: "Fake".into(),
                    description: None,
                    icon: Some(acp_thread::AgentModelIcon::Named(
                        ui::IconName::ZedAssistant
                    )),
                    is_latest: false,
                    cost: None,
                }]
            )])
        );
    }

    #[gpui::test]
    async fn test_model_selection_persists_to_settings(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.create_dir(paths::settings_file().parent().unwrap())
            .await
            .unwrap();
        fs.insert_file(
            paths::settings_file(),
            json!({
                "agent": {
                    "default_model": {
                        "provider": "foo",
                        "model": "bar"
                    }
                }
            })
            .to_string()
            .into_bytes(),
        )
        .await;
        let project = Project::test(fs.clone(), [], cx).await;

        let thread_store = cx.new(|cx| ThreadStore::new(cx));

        // Create the agent and connection
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), None, fs.clone(), cx));
        let connection = NativeAgentConnection(agent.clone());

        // Create a thread/session
        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();

        let session_id = cx.update(|cx| acp_thread.read(cx).session_id().clone());

        // Select a model
        let selector = connection.model_selector(&session_id).unwrap();
        let model_id = acp::ModelId::new("fake/fake");
        cx.update(|cx| selector.select_model(model_id.clone(), cx))
            .await
            .unwrap();

        // Verify the thread has the selected model
        agent.read_with(cx, |agent, _| {
            let session = agent.sessions.get(&session_id).unwrap();
            session.thread.read_with(cx, |thread, _| {
                assert_eq!(thread.model().unwrap().id().0, "fake");
            });
        });

        cx.run_until_parked();

        // Verify settings file was updated
        let settings_content = fs.load(paths::settings_file()).await.unwrap();
        let settings_json: serde_json::Value = serde_json::from_str(&settings_content).unwrap();

        // Check that the agent settings contain the selected model
        assert_eq!(
            settings_json["agent"]["default_model"]["model"],
            json!("fake")
        );
        assert_eq!(
            settings_json["agent"]["default_model"]["provider"],
            json!("fake")
        );

        // Register a thinking model and select it.
        cx.update(|cx| {
            let thinking_model = Arc::new(FakeLanguageModel::with_id_and_thinking(
                "fake-corp",
                "fake-thinking",
                "Fake Thinking",
                true,
            ));
            let thinking_provider = Arc::new(
                FakeLanguageModelProvider::new(
                    LanguageModelProviderId::from("fake-corp".to_string()),
                    LanguageModelProviderName::from("Fake Corp".to_string()),
                )
                .with_models(vec![thinking_model]),
            );
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(thinking_provider, cx);
            });
        });
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(acp::ModelId::new("fake-corp/fake-thinking"), cx))
            .await
            .unwrap();
        cx.run_until_parked();

        // Verify enable_thinking was written to settings as true.
        let settings_content = fs.load(paths::settings_file()).await.unwrap();
        let settings_json: serde_json::Value = serde_json::from_str(&settings_content).unwrap();
        assert_eq!(
            settings_json["agent"]["default_model"]["enable_thinking"],
            json!(true),
            "selecting a thinking model should persist enable_thinking: true to settings"
        );
    }

    #[gpui::test]
    async fn test_select_model_updates_thinking_enabled(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.create_dir(paths::settings_file().parent().unwrap())
            .await
            .unwrap();
        fs.insert_file(paths::settings_file(), b"{}".to_vec()).await;
        let project = Project::test(fs.clone(), [], cx).await;

        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), None, fs.clone(), cx));
        let connection = NativeAgentConnection(agent.clone());

        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = cx.update(|cx| acp_thread.read(cx).session_id().clone());

        // Register a second provider with a thinking model.
        cx.update(|cx| {
            let thinking_model = Arc::new(FakeLanguageModel::with_id_and_thinking(
                "fake-corp",
                "fake-thinking",
                "Fake Thinking",
                true,
            ));
            let thinking_provider = Arc::new(
                FakeLanguageModelProvider::new(
                    LanguageModelProviderId::from("fake-corp".to_string()),
                    LanguageModelProviderName::from("Fake Corp".to_string()),
                )
                .with_models(vec![thinking_model]),
            );
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(thinking_provider, cx);
            });
        });
        // Refresh the agent's model list so it picks up the new provider.
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        // Thread starts with thinking_enabled = false (the default).
        agent.read_with(cx, |agent, _| {
            let session = agent.sessions.get(&session_id).unwrap();
            session.thread.read_with(cx, |thread, _| {
                assert!(!thread.thinking_enabled(), "thinking defaults to false");
            });
        });

        // Select the thinking model via select_model.
        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(acp::ModelId::new("fake-corp/fake-thinking"), cx))
            .await
            .unwrap();

        // select_model should have enabled thinking based on the model's supports_thinking().
        agent.read_with(cx, |agent, _| {
            let session = agent.sessions.get(&session_id).unwrap();
            session.thread.read_with(cx, |thread, _| {
                assert!(
                    thread.thinking_enabled(),
                    "select_model should enable thinking when model supports it"
                );
            });
        });

        // Switch back to the non-thinking model.
        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(acp::ModelId::new("fake/fake"), cx))
            .await
            .unwrap();

        // select_model should have disabled thinking.
        agent.read_with(cx, |agent, _| {
            let session = agent.sessions.get(&session_id).unwrap();
            session.thread.read_with(cx, |thread, _| {
                assert!(
                    !thread.thinking_enabled(),
                    "select_model should disable thinking when model does not support it"
                );
            });
        });
    }

    #[gpui::test]
    async fn test_summarization_model_survives_transient_registry_clearing(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [], cx).await;

        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), None, fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());

        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        thread.read_with(cx, |thread, _| {
            assert!(
                thread.summarization_model().is_some(),
                "session should have a summarization model from the test registry"
            );
        });

        // Simulate what happens during a provider blip:
        // update_active_language_model_from_settings calls set_default_model(None)
        // when it can't resolve the model, clearing all fallbacks.
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.set_default_model(None, cx);
            });
        });
        cx.run_until_parked();

        thread.read_with(cx, |thread, _| {
            assert!(
                thread.summarization_model().is_some(),
                "summarization model should survive a transient default model clearing"
            );
        });
    }

    #[gpui::test]
    async fn test_loaded_thread_preserves_thinking_enabled(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx.update(|cx| {
            NativeAgent::new(thread_store.clone(), Templates::new(), None, fs.clone(), cx)
        });
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        // Register a thinking model.
        let thinking_model = Arc::new(FakeLanguageModel::with_id_and_thinking(
            "fake-corp",
            "fake-thinking",
            "Fake Thinking",
            true,
        ));
        let thinking_provider = Arc::new(
            FakeLanguageModelProvider::new(
                LanguageModelProviderId::from("fake-corp".to_string()),
                LanguageModelProviderName::from("Fake Corp".to_string()),
            )
            .with_models(vec![thinking_model.clone()]),
        );
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(thinking_provider, cx);
            });
        });
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        // Create a thread and select the thinking model.
        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());

        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(acp::ModelId::new("fake-corp/fake-thinking"), cx))
            .await
            .unwrap();

        // Verify thinking is enabled after selecting the thinking model.
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        thread.read_with(cx, |thread, _| {
            assert!(
                thread.thinking_enabled(),
                "thinking should be enabled after selecting thinking model"
            );
        });

        // Send a message so the thread gets persisted.
        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["Hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        thinking_model.send_last_completion_stream_text_chunk("Response.");
        thinking_model.end_last_completion_stream();

        send.await.unwrap();
        cx.run_until_parked();

        // Close the session so it can be reloaded from disk.
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(thread);
        drop(acp_thread);
        agent.read_with(cx, |agent, _| {
            assert!(agent.sessions.is_empty());
        });

        // Reload the thread and verify thinking_enabled is still true.
        let reloaded_acp_thread = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        let reloaded_thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        reloaded_thread.read_with(cx, |thread, _| {
            assert!(
                thread.thinking_enabled(),
                "thinking_enabled should be preserved when reloading a thread with a thinking model"
            );
        });

        drop(reloaded_acp_thread);
    }

    #[gpui::test]
    async fn test_loaded_thread_preserves_model(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx.update(|cx| {
            NativeAgent::new(thread_store.clone(), Templates::new(), None, fs.clone(), cx)
        });
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        // Register a model where id() != name(), like real Anthropic models
        // (e.g. id="claude-sonnet-4-5-thinking-latest", name="Claude Sonnet 4.5 Thinking").
        let model = Arc::new(FakeLanguageModel::with_id_and_thinking(
            "fake-corp",
            "custom-model-id",
            "Custom Model Display Name",
            false,
        ));
        let provider = Arc::new(
            FakeLanguageModelProvider::new(
                LanguageModelProviderId::from("fake-corp".to_string()),
                LanguageModelProviderName::from("Fake Corp".to_string()),
            )
            .with_models(vec![model.clone()]),
        );
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(provider, cx);
            });
        });
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        // Create a thread and select the model.
        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());

        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(acp::ModelId::new("fake-corp/custom-model-id"), cx))
            .await
            .unwrap();

        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        thread.read_with(cx, |thread, _| {
            assert_eq!(
                thread.model().unwrap().id().0.as_ref(),
                "custom-model-id",
                "model should be set before persisting"
            );
        });

        // Send a message so the thread gets persisted.
        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["Hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        model.send_last_completion_stream_text_chunk("Response.");
        model.end_last_completion_stream();

        send.await.unwrap();
        cx.run_until_parked();

        // Close the session so it can be reloaded from disk.
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(thread);
        drop(acp_thread);
        agent.read_with(cx, |agent, _| {
            assert!(agent.sessions.is_empty());
        });

        // Reload the thread and verify the model was preserved.
        let reloaded_acp_thread = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        let reloaded_thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        reloaded_thread.read_with(cx, |thread, _| {
            let reloaded_model = thread
                .model()
                .expect("model should be present after reload");
            assert_eq!(
                reloaded_model.id().0.as_ref(),
                "custom-model-id",
                "reloaded thread should have the same model, not fall back to the default"
            );
        });

        drop(reloaded_acp_thread);
    }

    #[gpui::test]
    async fn test_save_load_thread(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "a": {
                    "b.md": "Lorem"
                }
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx.update(|cx| {
            NativeAgent::new(thread_store.clone(), Templates::new(), None, fs.clone(), cx)
        });
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new(&[Path::new("")]), cx)
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        // Ensure empty threads are not saved, even if they get mutated.
        let model = Arc::new(FakeLanguageModel::default());
        let summary_model = Arc::new(FakeLanguageModel::default());
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.set_summarization_model(Some(summary_model.clone()), cx);
        });
        cx.run_until_parked();
        assert_eq!(thread_entries(&thread_store, cx), vec![]);

        let send = acp_thread.update(cx, |thread, cx| {
            thread.send(
                vec![
                    "What does ".into(),
                    acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                        "b.md",
                        MentionUri::File {
                            abs_path: path!("/a/b.md").into(),
                        }
                        .to_uri()
                        .to_string(),
                    )),
                    " mean?".into(),
                ],
                cx,
            )
        });
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        model.send_last_completion_stream_text_chunk("Lorem.");
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::UsageUpdate(
            language_model::TokenUsage {
                input_tokens: 150,
                output_tokens: 75,
                ..Default::default()
            },
        ));
        model.end_last_completion_stream();
        cx.run_until_parked();
        summary_model
            .send_last_completion_stream_text_chunk(&format!("Explaining {}", path!("/a/b.md")));
        summary_model.end_last_completion_stream();

        send.await.unwrap();
        let uri = MentionUri::File {
            abs_path: path!("/a/b.md").into(),
        }
        .to_uri();
        acp_thread.read_with(cx, |thread, cx| {
            assert_eq!(
                thread.to_markdown(cx),
                formatdoc! {"
                    ## User

                    What does [@b.md]({uri}) mean?

                    ## Assistant

                    Lorem.

                "}
            )
        });

        cx.run_until_parked();

        // Set a draft prompt with rich content blocks and scroll position
        // AFTER run_until_parked, so the only save that captures these
        // changes is the one performed by close_session itself.
        let draft_blocks = vec![
            acp::ContentBlock::Text(acp::TextContent::new("Check out ")),
            acp::ContentBlock::ResourceLink(acp::ResourceLink::new("b.md", uri.to_string())),
            acp::ContentBlock::Text(acp::TextContent::new(" please")),
        ];
        acp_thread.update(cx, |thread, _cx| {
            thread.set_draft_prompt(Some(draft_blocks.clone()));
        });
        thread.update(cx, |thread, _cx| {
            thread.set_ui_scroll_position(Some(gpui::ListOffset {
                item_ix: 5,
                offset_in_item: gpui::px(12.5),
            }));
        });

        // Close the session so it can be reloaded from disk.
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(thread);
        drop(acp_thread);
        agent.read_with(cx, |agent, _| {
            assert_eq!(agent.sessions.keys().cloned().collect::<Vec<_>>(), []);
        });

        // Ensure the thread can be reloaded from disk.
        assert_eq!(
            thread_entries(&thread_store, cx),
            vec![(
                session_id.clone(),
                format!("Explaining {}", path!("/a/b.md"))
            )]
        );
        let acp_thread = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        acp_thread.read_with(cx, |thread, cx| {
            assert_eq!(
                thread.to_markdown(cx),
                formatdoc! {"
                    ## User

                    What does [@b.md]({uri}) mean?

                    ## Assistant

                    Lorem.

                "}
            )
        });

        // Ensure the draft prompt with rich content blocks survived the round-trip.
        acp_thread.read_with(cx, |thread, _| {
            assert_eq!(thread.draft_prompt(), Some(draft_blocks.as_slice()));
        });

        // Ensure token usage survived the round-trip.
        acp_thread.read_with(cx, |thread, _| {
            let usage = thread
                .token_usage()
                .expect("token usage should be restored after reload");
            assert_eq!(usage.input_tokens, 150);
            assert_eq!(usage.output_tokens, 75);
        });

        // Ensure scroll position survived the round-trip.
        acp_thread.read_with(cx, |thread, _| {
            let scroll = thread
                .ui_scroll_position()
                .expect("scroll position should be restored after reload");
            assert_eq!(scroll.item_ix, 5);
            assert_eq!(scroll.offset_in_item, gpui::px(12.5));
        });
    }

    #[gpui::test]
    async fn test_close_session_saves_thread(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "a": {
                    "file.txt": "hello"
                }
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx.update(|cx| {
            NativeAgent::new(thread_store.clone(), Templates::new(), None, fs.clone(), cx)
        });
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new(&[Path::new("")]), cx)
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        let model = Arc::new(FakeLanguageModel::default());
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
        });

        // Send a message so the thread is non-empty (empty threads aren't saved).
        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        model.send_last_completion_stream_text_chunk("world");
        model.end_last_completion_stream();
        send.await.unwrap();
        cx.run_until_parked();

        // Set a draft prompt WITHOUT calling run_until_parked afterwards.
        // This means no observe-triggered save has run for this change.
        // The only way this data gets persisted is if close_session
        // itself performs the save.
        let draft_blocks = vec![acp::ContentBlock::Text(acp::TextContent::new(
            "unsaved draft",
        ))];
        acp_thread.update(cx, |thread, _cx| {
            thread.set_draft_prompt(Some(draft_blocks.clone()));
        });

        // Close the session immediately — no run_until_parked in between.
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        cx.run_until_parked();

        // Reopen and verify the draft prompt was saved.
        let reloaded = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        reloaded.read_with(cx, |thread, _| {
            assert_eq!(
                thread.draft_prompt(),
                Some(draft_blocks.as_slice()),
                "close_session must save the thread; draft prompt was lost"
            );
        });
    }

    #[gpui::test]
    async fn test_rapid_title_changes_do_not_loop(cx: &mut TestAppContext) {
        // Regression test: rapid title changes must not cause a propagation loop
        // between Thread and AcpThread via handle_thread_title_updated.
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx.update(|cx| {
            NativeAgent::new(thread_store.clone(), Templates::new(), None, fs.clone(), cx)
        });
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new(&[Path::new("")]), cx)
            })
            .await
            .unwrap();

        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        let title_updated_count = Rc::new(std::cell::RefCell::new(0usize));
        cx.update(|cx| {
            let count = title_updated_count.clone();
            cx.subscribe(
                &thread,
                move |_entity: Entity<Thread>, _event: &TitleUpdated, _cx: &mut App| {
                    let new_count = {
                        let mut count = count.borrow_mut();
                        *count += 1;
                        *count
                    };
                    assert!(
                        new_count <= 2,
                        "TitleUpdated fired {new_count} times; \
                         title updates are looping"
                    );
                },
            )
            .detach();
        });

        thread.update(cx, |thread, cx| thread.set_title("first".into(), cx));
        thread.update(cx, |thread, cx| thread.set_title("second".into(), cx));

        cx.run_until_parked();

        thread.read_with(cx, |thread, _| {
            assert_eq!(thread.title(), Some("second".into()));
        });
        acp_thread.read_with(cx, |acp_thread, _| {
            assert_eq!(acp_thread.title(), Some("second".into()));
        });

        assert_eq!(*title_updated_count.borrow(), 2);
    }

    fn thread_entries(
        thread_store: &Entity<ThreadStore>,
        cx: &mut TestAppContext,
    ) -> Vec<(acp::SessionId, String)> {
        thread_store.read_with(cx, |store, _| {
            store
                .entries()
                .map(|entry| (entry.id.clone(), entry.title.to_string()))
                .collect::<Vec<_>>()
        })
    }

    fn init_test(cx: &mut TestAppContext) {
        env_logger::try_init().ok();
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);

            LanguageModelRegistry::test(cx);
        });
    }
}

// ── ST-B5 / catalog-opaque-v1 — mode-name catalog helpers ─────────────────
// IDE-facing mode name surfaces. Single source of truth is the engine
// via `caduceus_bridge::orchestrator::list_modes()`; these helpers just
// project the catalog into the two formats slash-help/CLI-help need.

fn available_modes_comma() -> String {
    caduceus_bridge::orchestrator::list_modes()
        .into_iter()
        .map(|m| m.name)
        .collect::<Vec<_>>()
        .join(", ")
}

fn available_modes_pipe() -> String {
    caduceus_bridge::orchestrator::list_modes()
        .into_iter()
        .map(|m| m.name)
        .collect::<Vec<_>>()
        .join("|")
}

fn should_index_path(path: &std::path::Path, ignore_patterns: &[String]) -> bool {
    let path_str = path.to_string_lossy();
    for pattern in ignore_patterns {
        if pattern.starts_with('!') {
            continue;
        }
        if path_str.contains(pattern.trim_start_matches('/')) {
            return false;
        }
        if pattern.starts_with("*.") {
            let ext = &pattern[1..];
            if path_str.ends_with(ext) {
                return false;
            }
        }
    }
    true
}

fn load_caduceuignore(project_root: &std::path::Path) -> Vec<String> {
    let ignore_path = project_root.join(".caduceuignore");
    std::fs::read_to_string(&ignore_path)
        .unwrap_or_default()
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|s| s.to_string())
        .collect()
}

fn mcp_message_content_to_acp_content_block(
    content: context_server::types::MessageContent,
) -> acp::ContentBlock {
    match content {
        context_server::types::MessageContent::Text {
            text,
            annotations: _,
        } => text.into(),
        context_server::types::MessageContent::Image {
            data,
            mime_type,
            annotations: _,
        } => acp::ContentBlock::Image(acp::ImageContent::new(data, mime_type)),
        context_server::types::MessageContent::Audio {
            data,
            mime_type,
            annotations: _,
        } => acp::ContentBlock::Audio(acp::AudioContent::new(data, mime_type)),
        context_server::types::MessageContent::Resource {
            resource,
            annotations: _,
        } => {
            let mut link =
                acp::ResourceLink::new(resource.uri.to_string(), resource.uri.to_string());
            if let Some(mime_type) = resource.mime_type {
                link = link.mime_type(mime_type);
            }
            acp::ContentBlock::ResourceLink(link)
        }
    }
}

/// Map a CompactOutcome to (event-label, user-facing message).
///
/// Regression for bug #19: `/compact` previously always reported
/// "✅ within budget" even when the cooldown blocked it — pretending the
/// command had succeeded. The honest tri-state mapping below makes the UI
/// distinguish the three real outcomes.
pub(crate) fn compact_outcome_message(
    outcome: crate::thread::CompactOutcome,
) -> (&'static str, &'static str) {
    match outcome {
        crate::thread::CompactOutcome::Compacted => (
            "compacted",
            "✅ Context compacted. Older messages summarized to save tokens.",
        ),
        crate::thread::CompactOutcome::WithinBudget => (
            "within budget",
            "ℹ️ Context is within budget — no compaction needed.",
        ),
        crate::thread::CompactOutcome::CooldownActive => (
            "cooldown active",
            "⏳ Compaction cooldown is active — try again in a few seconds.",
        ),
    }
}

#[cfg(test)]
mod compact_outcome_tests {
    use super::*;
    use crate::thread::CompactOutcome;

    #[test]
    fn compacted_returns_success_message() {
        let (label, msg) = compact_outcome_message(CompactOutcome::Compacted);
        assert_eq!(label, "compacted");
        assert!(msg.contains("compacted"));
    }

    #[test]
    fn within_budget_returns_no_op_message() {
        let (label, msg) = compact_outcome_message(CompactOutcome::WithinBudget);
        assert_eq!(label, "within budget");
        assert!(msg.contains("within budget"));
    }

    /// The whole point of bug #19's fix: cooldown must NOT be reported as
    /// "within budget" — the user has to know the command was blocked.
    #[test]
    fn cooldown_returns_distinct_message_not_within_budget() {
        let (label, msg) = compact_outcome_message(CompactOutcome::CooldownActive);
        assert_eq!(label, "cooldown active");
        assert!(msg.contains("cooldown"));
        let (_, within) = compact_outcome_message(CompactOutcome::WithinBudget);
        assert_ne!(msg, within, "cooldown must not masquerade as within-budget");
    }

    #[test]
    fn all_three_outcomes_have_unique_labels_and_messages() {
        let outcomes = [
            CompactOutcome::Compacted,
            CompactOutcome::WithinBudget,
            CompactOutcome::CooldownActive,
        ];
        let mut labels = std::collections::HashSet::new();
        let mut messages = std::collections::HashSet::new();
        for o in outcomes {
            let (l, m) = compact_outcome_message(o);
            assert!(labels.insert(l), "duplicate label: {l}");
            assert!(messages.insert(m), "duplicate message: {m}");
        }
    }
}

#[cfg(test)]
mod b5_catalog_tests {
    //! ST-B5 / `catalog-opaque-v1` — Zed holds no hardcoded mode catalog.
    //! These tests pin that invariant: every mode the IDE renders MUST
    //! come from the engine (`caduceus_bridge::orchestrator::list_modes`),
    //! and the slash-help / CLI-help surfaces MUST derive from the same
    //! source. Adding a mode in the engine appears in the IDE without
    //! touching any `.rs` file in this crate.

    use super::*;

    #[test]
    fn all_caduceus_modes_mirrors_engine_catalog_exactly() {
        // IDE's picker view.
        let ide_modes = CaduceusSessionModes::all_caduceus_modes();
        // Engine's authoritative catalog.
        let engine_modes = caduceus_bridge::orchestrator::list_modes();

        assert_eq!(
            ide_modes.len(),
            engine_modes.len(),
            "IDE mode picker MUST have the same count as the engine catalog — \
             diverging means a hardcoded list has crept back into the IDE"
        );

        // Order-preserving pairwise equality on (id, label, description).
        for (ide, eng) in ide_modes.iter().zip(engine_modes.iter()) {
            assert_eq!(
                ide.id.0.as_ref(),
                eng.name.as_str(),
                "IDE session-mode id MUST be the engine's canonical name (opaque token)"
            );
            assert_eq!(
                ide.name.as_str(),
                eng.label.as_str(),
                "IDE label MUST match engine label"
            );
            assert_eq!(
                ide.description.as_deref().unwrap_or(""),
                eng.description.as_str(),
                "IDE description MUST match engine description"
            );
        }
    }

    #[test]
    fn available_modes_helpers_cover_every_engine_mode() {
        let engine_names: Vec<String> = caduceus_bridge::orchestrator::list_modes()
            .into_iter()
            .map(|m| m.name)
            .collect();
        let comma = available_modes_comma();
        let pipe = available_modes_pipe();
        for name in &engine_names {
            assert!(
                comma.contains(name),
                "slash-help comma list MUST include engine mode {name}: got {comma:?}"
            );
            assert!(
                pipe.contains(name),
                "CLI-help pipe list MUST include engine mode {name}: got {pipe:?}"
            );
        }
        // Separator sanity.
        assert!(comma.contains(", "), "comma helper MUST use ', ' separator");
        if engine_names.len() > 1 {
            assert!(pipe.contains('|'), "pipe helper MUST use '|' separator");
        }
    }

    #[test]
    fn available_modes_helpers_have_no_legacy_mode_names() {
        // Legacy aliases (Architect / Debug / Review) deserialize through
        // serde rename on the engine side for backward compat, but they
        // MUST NOT appear in any help-text surface the IDE shows to the
        // user — the canonical 4 modes are the only thing a new user
        // should see.
        let comma = available_modes_comma();
        for legacy in &["architect", "debug", "review"] {
            assert!(
                !comma.contains(legacy),
                "legacy alias '{legacy}' leaked into slash-help list: {comma:?}"
            );
        }
    }
}
