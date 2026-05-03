use std::{cmp::Reverse, sync::Arc};

use agent_settings::AgentSettings;
use collections::{HashMap, HashSet, IndexMap};
use fuzzy::{StringMatch, StringMatchCandidate, match_strings};
use gpui::{
    Action, AnyElement, App, BackgroundExecutor, DismissEvent, FocusHandle, ForegroundExecutor,
    Subscription, Task,
};
use language_model::{
    AuthAction, AuthenticateError, ConfiguredModel, IconOrSvg, LanguageModel, LanguageModelId,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelRegistry, ProviderAuthState,
};
use ordered_float::OrderedFloat;
use picker::{Picker, PickerDelegate};
use settings::Settings;
use ui::prelude::*;
use ui::{Chip, ListItem, ListItemSpacing};
use zed_actions::agent::OpenSettings;

use crate::ui::{ModelSelectorFooter, ModelSelectorHeader, ModelSelectorListItem};

type OnModelChanged = Arc<dyn Fn(Arc<dyn LanguageModel>, &mut App) + 'static>;
type GetActiveModel = Arc<dyn Fn(&App) -> Option<ConfiguredModel> + 'static>;
type OnToggleFavorite = Arc<dyn Fn(Arc<dyn LanguageModel>, bool, &mut App) + 'static>;

pub type LanguageModelSelector = Picker<LanguageModelPickerDelegate>;

pub fn language_model_selector(
    get_active_model: impl Fn(&App) -> Option<ConfiguredModel> + 'static,
    on_model_changed: impl Fn(Arc<dyn LanguageModel>, &mut App) + 'static,
    on_toggle_favorite: impl Fn(Arc<dyn LanguageModel>, bool, &mut App) + 'static,
    popover_styles: bool,
    focus_handle: FocusHandle,
    window: &mut Window,
    cx: &mut Context<LanguageModelSelector>,
) -> LanguageModelSelector {
    let delegate = LanguageModelPickerDelegate::new(
        get_active_model,
        on_model_changed,
        on_toggle_favorite,
        popover_styles,
        focus_handle,
        window,
        cx,
    );

    if popover_styles {
        Picker::list(delegate, window, cx)
            .show_scrollbar(true)
            .width(rems(20.))
            .max_height(Some(rems(20.).into()))
    } else {
        Picker::list(delegate, window, cx).show_scrollbar(true)
    }
}

fn all_models(cx: &App) -> GroupedModels {
    let lm_registry = LanguageModelRegistry::global(cx).read(cx);
    let providers = lm_registry.visible_providers();

    let mut favorites_index = FavoritesIndex::default();

    for sel in &AgentSettings::get_global(cx).favorite_models {
        favorites_index
            .entry(sel.provider.0.clone().into())
            .or_default()
            .insert(sel.model.clone().into());
    }

    let recommended = providers
        .iter()
        .flat_map(|provider| {
            provider
                .recommended_models(cx)
                .into_iter()
                .map(|model| ModelInfo::new(&**provider, model, &favorites_index))
        })
        .collect();

    let all = providers
        .iter()
        .flat_map(|provider| {
            provider
                .provided_models(cx)
                .into_iter()
                .map(|model| ModelInfo::new(&**provider, model, &favorites_index))
        })
        .collect();

    GroupedModels::new(all, recommended, unauth_rows(cx))
}

/// Snapshot of an unauthenticated / rate-limited / policy-disabled provider
/// row. Built from the registry independently of `all_models()` so that
/// `GroupedModels` never contains models for unauth providers.
///
/// ST1b: powers the "More providers" section in the language model picker.
#[derive(Clone)]
struct UnauthRow {
    provider: Arc<dyn LanguageModelProvider>,
    auth_state: ProviderAuthState,
    icon: IconOrSvg,
}

fn unauth_rows(cx: &App) -> Vec<UnauthRow> {
    partition_providers(cx).1
}

/// Partitions visible providers into (configured provider IDs, unauth snapshot rows).
/// Single source of truth for the auth/unauth split — both the initial
/// `GroupedModels` build and `update_matches` go through this so they can never
/// drift apart.
fn partition_providers(cx: &App) -> (Vec<LanguageModelProviderId>, Vec<UnauthRow>) {
    let mut configured = Vec::new();
    let mut unauth = Vec::new();
    for provider in LanguageModelRegistry::global(cx)
        .read(cx)
        .visible_providers()
    {
        let auth_state = provider.auth_state(cx);
        if auth_state.can_provide_models() {
            configured.push(provider.id());
        } else {
            unauth.push(UnauthRow {
                icon: provider.icon(),
                provider,
                auth_state,
            });
        }
    }
    (configured, unauth)
}

type FavoritesIndex = HashMap<LanguageModelProviderId, HashSet<LanguageModelId>>;

#[derive(Clone)]
struct ModelInfo {
    model: Arc<dyn LanguageModel>,
    icon: IconOrSvg,
    is_favorite: bool,
}

impl ModelInfo {
    fn new(
        provider: &dyn LanguageModelProvider,
        model: Arc<dyn LanguageModel>,
        favorites_index: &FavoritesIndex,
    ) -> Self {
        let is_favorite = favorites_index
            .get(&provider.id())
            .map_or(false, |set| set.contains(&model.id()));

        Self {
            model,
            icon: provider.icon(),
            is_favorite,
        }
    }
}

pub struct LanguageModelPickerDelegate {
    on_model_changed: OnModelChanged,
    get_active_model: GetActiveModel,
    on_toggle_favorite: OnToggleFavorite,
    all_models: Arc<GroupedModels>,
    filtered_entries: Vec<LanguageModelPickerEntry>,
    selected_index: usize,
    _authenticate_all_providers_task: Task<()>,
    _subscriptions: Vec<Subscription>,
    popover_styles: bool,
    focus_handle: FocusHandle,
}

impl LanguageModelPickerDelegate {
    fn new(
        get_active_model: impl Fn(&App) -> Option<ConfiguredModel> + 'static,
        on_model_changed: impl Fn(Arc<dyn LanguageModel>, &mut App) + 'static,
        on_toggle_favorite: impl Fn(Arc<dyn LanguageModel>, bool, &mut App) + 'static,
        popover_styles: bool,
        focus_handle: FocusHandle,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Self {
        let on_model_changed = Arc::new(on_model_changed);
        let models = all_models(cx);
        let entries = models.entries();

        Self {
            on_model_changed,
            #[allow(clippy::arc_with_non_send_sync)]
            all_models: Arc::new(models),
            selected_index: Self::get_active_model_index(&entries, get_active_model(cx)),
            filtered_entries: entries,
            get_active_model: Arc::new(get_active_model),
            on_toggle_favorite: Arc::new(on_toggle_favorite),
            _authenticate_all_providers_task: Self::authenticate_all_providers(cx),
            _subscriptions: vec![cx.subscribe_in(
                &LanguageModelRegistry::global(cx),
                window,
                |picker, _, event, window, cx| {
                    match event {
                        language_model::Event::ProviderStateChanged(_)
                        | language_model::Event::AddedProvider(_)
                        | language_model::Event::RemovedProvider(_) => {
                            let query = picker.query(cx);
                            #[allow(clippy::arc_with_non_send_sync)]
                            {
                                picker.delegate.all_models = Arc::new(all_models(cx));
                            }
                            // Update matches will automatically drop the previous task
                            // if we get a provider event again
                            picker.update_matches(query, window, cx)
                        }
                        _ => {}
                    }
                },
            )],
            popover_styles,
            focus_handle,
        }
    }

    fn get_active_model_index(
        entries: &[LanguageModelPickerEntry],
        active_model: Option<ConfiguredModel>,
    ) -> usize {
        entries
            .iter()
            .position(|entry| {
                if let LanguageModelPickerEntry::Model(model) = entry {
                    active_model
                        .as_ref()
                        .map(|active_model| {
                            active_model.model.id() == model.model.id()
                                && active_model.provider.id() == model.model.provider_id()
                        })
                        .unwrap_or_default()
                } else {
                    false
                }
            })
            .unwrap_or(0)
    }

    /// Authenticates all providers in the [`LanguageModelRegistry`].
    ///
    /// We do this so that we can populate the language selector with all of the
    /// models from the configured providers.
    fn authenticate_all_providers(cx: &mut App) -> Task<()> {
        let authenticate_all_providers = LanguageModelRegistry::global(cx)
            .read(cx)
            .visible_providers()
            .iter()
            .map(|provider| (provider.id(), provider.name(), provider.authenticate(cx)))
            .collect::<Vec<_>>();

        cx.spawn(async move |_cx| {
            for (provider_id, provider_name, authenticate_task) in authenticate_all_providers {
                if let Err(err) = authenticate_task.await {
                    if matches!(err, AuthenticateError::CredentialsNotFound) {
                        // Since we're authenticating these providers in the
                        // background for the purposes of populating the
                        // language selector, we don't care about providers
                        // where the credentials are not found.
                    } else {
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
        })
    }

    pub fn active_model(&self, cx: &App) -> Option<ConfiguredModel> {
        (self.get_active_model)(cx)
    }

    pub fn favorites_count(&self) -> usize {
        self.all_models.favorites.len()
    }

    pub fn cycle_favorite_models(&mut self, window: &mut Window, cx: &mut Context<Picker<Self>>) {
        if self.all_models.favorites.is_empty() {
            return;
        }

        let active_model = (self.get_active_model)(cx);
        let active_provider_id = active_model.as_ref().map(|m| m.provider.id());
        let active_model_id = active_model.as_ref().map(|m| m.model.id());

        let current_index = self
            .all_models
            .favorites
            .iter()
            .position(|info| {
                Some(info.model.provider_id()) == active_provider_id
                    && Some(info.model.id()) == active_model_id
            })
            .unwrap_or(usize::MAX);

        let next_index = if current_index == usize::MAX {
            0
        } else {
            (current_index + 1) % self.all_models.favorites.len()
        };

        let next_model = self.all_models.favorites[next_index].model.clone();

        (self.on_model_changed)(next_model, cx);

        // Align the picker selection with the newly-active model
        let new_index =
            Self::get_active_model_index(&self.filtered_entries, (self.get_active_model)(cx));
        self.set_selected_index(new_index, window, cx);
    }
}

struct GroupedModels {
    favorites: Vec<ModelInfo>,
    recommended: Vec<ModelInfo>,
    all: IndexMap<LanguageModelProviderId, Vec<ModelInfo>>,
    unauth: Vec<UnauthRow>,
}

impl GroupedModels {
    pub fn new(all: Vec<ModelInfo>, recommended: Vec<ModelInfo>, unauth: Vec<UnauthRow>) -> Self {
        let favorites = all
            .iter()
            .filter(|info| info.is_favorite)
            .cloned()
            .collect();

        let mut all_by_provider: IndexMap<_, Vec<ModelInfo>> = IndexMap::default();
        for model in all {
            let provider = model.model.provider_id();
            if let Some(models) = all_by_provider.get_mut(&provider) {
                models.push(model);
            } else {
                all_by_provider.insert(provider, vec![model]);
            }
        }

        Self {
            favorites,
            recommended,
            all: all_by_provider,
            unauth,
        }
    }

    fn entries(&self) -> Vec<LanguageModelPickerEntry> {
        let mut entries = Vec::new();

        if !self.favorites.is_empty() {
            entries.push(LanguageModelPickerEntry::Separator("Favorite".into()));
            for info in &self.favorites {
                entries.push(LanguageModelPickerEntry::Model(info.clone()));
            }
        }

        if !self.recommended.is_empty() {
            entries.push(LanguageModelPickerEntry::Separator("Recommended".into()));
            for info in &self.recommended {
                entries.push(LanguageModelPickerEntry::Model(info.clone()));
            }
        }

        for models in self.all.values() {
            if models.is_empty() {
                continue;
            }
            entries.push(LanguageModelPickerEntry::Separator(
                models[0].model.provider_name().0,
            ));
            for info in models {
                entries.push(LanguageModelPickerEntry::Model(info.clone()));
            }
        }

        // ST1b: unauthenticated / rate-limited / policy-disabled providers
        // always appear at the bottom so configured stuff comes first.
        if !self.unauth.is_empty() {
            entries.push(LanguageModelPickerEntry::Separator("More providers".into()));
            for row in &self.unauth {
                entries.push(LanguageModelPickerEntry::UnauthProvider(row.clone()));
            }
        }

        entries
    }
}

enum LanguageModelPickerEntry {
    Model(ModelInfo),
    Separator(SharedString),
    /// ST1b: a provider that the user has not signed in to (or is rate-limited
    /// / policy-disabled). Renders as a row with an action badge.
    UnauthProvider(UnauthRow),
}

struct ModelMatcher {
    models: Vec<ModelInfo>,
    fg_executor: ForegroundExecutor,
    bg_executor: BackgroundExecutor,
    candidates: Vec<StringMatchCandidate>,
}

impl ModelMatcher {
    fn new(
        models: Vec<ModelInfo>,
        fg_executor: ForegroundExecutor,
        bg_executor: BackgroundExecutor,
    ) -> ModelMatcher {
        let candidates = Self::make_match_candidates(&models);
        Self {
            models,
            fg_executor,
            bg_executor,
            candidates,
        }
    }

    pub fn fuzzy_search(&self, query: &str) -> Vec<ModelInfo> {
        let mut matches = self.fg_executor.block_on(match_strings(
            &self.candidates,
            query,
            false,
            true,
            100,
            &Default::default(),
            self.bg_executor.clone(),
        ));

        let sorting_key = |mat: &StringMatch| {
            let candidate = &self.candidates[mat.candidate_id];
            (Reverse(OrderedFloat(mat.score)), candidate.id)
        };
        matches.sort_unstable_by_key(sorting_key);

        let matched_models: Vec<_> = matches
            .into_iter()
            .map(|mat| self.models[mat.candidate_id].clone())
            .collect();

        matched_models
    }

    pub fn exact_search(&self, query: &str) -> Vec<ModelInfo> {
        self.models
            .iter()
            .filter(|m| {
                m.model
                    .name()
                    .0
                    .to_lowercase()
                    .contains(&query.to_lowercase())
            })
            .cloned()
            .collect::<Vec<_>>()
    }

    fn make_match_candidates(model_infos: &Vec<ModelInfo>) -> Vec<StringMatchCandidate> {
        model_infos
            .iter()
            .enumerate()
            .map(|(index, model)| {
                StringMatchCandidate::new(
                    index,
                    &format!(
                        "{}/{}",
                        &model.model.provider_name().0,
                        &model.model.name().0
                    ),
                )
            })
            .collect::<Vec<_>>()
    }
}

impl PickerDelegate for LanguageModelPickerDelegate {
    type ListItem = AnyElement;

    fn match_count(&self) -> usize {
        self.filtered_entries.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(&mut self, ix: usize, _: &mut Window, cx: &mut Context<Picker<Self>>) {
        self.selected_index = ix.min(self.filtered_entries.len().saturating_sub(1));
        cx.notify();
    }

    fn can_select(&self, ix: usize, _window: &mut Window, _cx: &mut Context<Picker<Self>>) -> bool {
        match self.filtered_entries.get(ix) {
            Some(LanguageModelPickerEntry::Model(_)) => true,
            Some(LanguageModelPickerEntry::UnauthProvider(row)) => unauth_row_is_actionable(row),
            Some(LanguageModelPickerEntry::Separator(_)) | None => false,
        }
    }

    fn placeholder_text(&self, _window: &mut Window, _cx: &mut App) -> Arc<str> {
        "Select a model…".into()
    }

    fn update_matches(
        &mut self,
        query: String,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        let all_models = self.all_models.clone();
        let active_model = (self.get_active_model)(cx);
        let fg_executor = cx.foreground_executor();
        let bg_executor = cx.background_executor();

        // ST1b: do NOT filter unauth providers from the visible list. We
        // partition them: configured providers contribute models to the
        // matchers; unauth providers contribute UnauthProvider rows.
        let (configured_provider_ids, mut unauth_rows) = partition_providers(cx);

        // ST1b: when the user types a query, also filter the unauth section
        // by case-insensitive substring on provider name so typing "Anthropic"
        // surfaces the unauth row.
        if !query.is_empty() {
            let q = query.to_lowercase();
            unauth_rows.retain(|row| row.provider.name().0.to_lowercase().contains(&q));
        }

        let recommended_models = all_models
            .recommended
            .iter()
            .filter(|m| configured_provider_ids.contains(&m.model.provider_id()))
            .cloned()
            .collect::<Vec<_>>();

        let available_models = all_models
            .all
            .values()
            .flat_map(|models| models.iter())
            .filter(|m| configured_provider_ids.contains(&m.model.provider_id()))
            .cloned()
            .collect::<Vec<_>>();

        let matcher_rec =
            ModelMatcher::new(recommended_models, fg_executor.clone(), bg_executor.clone());
        let matcher_all =
            ModelMatcher::new(available_models, fg_executor.clone(), bg_executor.clone());

        let recommended = matcher_rec.exact_search(&query);
        let all = matcher_all.fuzzy_search(&query);

        let filtered_models = GroupedModels::new(all, recommended, unauth_rows);

        cx.spawn_in(window, async move |this, cx| {
            this.update_in(cx, |this, window, cx| {
                this.delegate.filtered_entries = filtered_models.entries();
                // Finds the currently selected model in the list
                let new_index =
                    Self::get_active_model_index(&this.delegate.filtered_entries, active_model);
                this.set_selected_index(new_index, Some(picker::Direction::Down), true, window, cx);
                cx.notify();
            })
            .ok();
        })
    }

    fn confirm(&mut self, _secondary: bool, window: &mut Window, cx: &mut Context<Picker<Self>>) {
        match self.filtered_entries.get(self.selected_index) {
            Some(LanguageModelPickerEntry::Model(model_info)) => {
                let model = model_info.model.clone();
                (self.on_model_changed)(model.clone(), cx);

                let current_index = self.selected_index;
                self.set_selected_index(current_index, window, cx);

                cx.emit(DismissEvent);
            }
            Some(LanguageModelPickerEntry::UnauthProvider(row)) => {
                if !unauth_row_is_actionable(row) {
                    return;
                }
                let action = unauth_action_for(&row.auth_state);
                dispatch_unauth_action(action, window, cx);
                cx.emit(DismissEvent);
            }
            _ => {}
        }
    }

    fn dismissed(&mut self, _: &mut Window, cx: &mut Context<Picker<Self>>) {
        cx.emit(DismissEvent);
    }

    fn render_match(
        &self,
        ix: usize,
        selected: bool,
        _: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        match self.filtered_entries.get(ix)? {
            LanguageModelPickerEntry::Separator(title) => {
                Some(ModelSelectorHeader::new(title, ix > 1).into_any_element())
            }
            LanguageModelPickerEntry::Model(model_info) => {
                let active_model = (self.get_active_model)(cx);
                let active_provider_id = active_model.as_ref().map(|m| m.provider.id());
                let active_model_id = active_model.map(|m| m.model.id());

                let is_selected = Some(model_info.model.provider_id()) == active_provider_id
                    && Some(model_info.model.id()) == active_model_id;

                let model_cost = model_info
                    .model
                    .model_cost_info()
                    .map(|cost| cost.to_shared_string());

                let is_favorite = model_info.is_favorite;
                let handle_action_click = {
                    let model = model_info.model.clone();
                    let on_toggle_favorite = self.on_toggle_favorite.clone();
                    cx.listener(move |picker, _, window, cx| {
                        on_toggle_favorite(model.clone(), !is_favorite, cx);
                        picker.refresh(window, cx);
                    })
                };

                Some(
                    ModelSelectorListItem::new(ix, model_info.model.name().0)
                        .map(|this| match &model_info.icon {
                            IconOrSvg::Icon(icon_name) => this.icon(*icon_name),
                            IconOrSvg::Svg(icon_path) => this.icon_path(icon_path.clone()),
                        })
                        .is_selected(is_selected)
                        .is_focused(selected)
                        .is_latest(model_info.model.is_latest())
                        .is_favorite(is_favorite)
                        .cost_info(model_cost)
                        .on_toggle_favorite(handle_action_click)
                        .into_any_element(),
                )
            }
            LanguageModelPickerEntry::UnauthProvider(row) => {
                Some(render_unauth_row(ix, row, selected).into_any_element())
            }
        }
    }

    fn render_footer(
        &self,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> Option<gpui::AnyElement> {
        let focus_handle = self.focus_handle.clone();

        if !self.popover_styles {
            return None;
        }

        Some(ModelSelectorFooter::new(OpenSettings.boxed_clone(), focus_handle).into_any_element())
    }
}

// ─── ST1b: unauth-provider row plumbing ───────────────────────────────────

/// Decides which `AuthAction` the dispatcher should run for an `UnauthRow`.
/// `RateLimited` and `DisabledByPolicy` collapse to `None` because there's
/// no meaningful user action — the row is rendered for visibility only.
fn unauth_action_for(state: &ProviderAuthState) -> AuthAction {
    match state {
        ProviderAuthState::Authenticated => AuthAction::None,
        ProviderAuthState::NotAuthenticated { action } => action.clone(),
        ProviderAuthState::RateLimited { .. } | ProviderAuthState::DisabledByPolicy(_) => {
            AuthAction::None
        }
    }
}

/// True iff clicking this row should trigger a meaningful action.
/// Rate-limited / policy-disabled / `AuthAction::None` rows are non-actionable.
fn unauth_row_is_actionable(row: &UnauthRow) -> bool {
    !matches!(unauth_action_for(&row.auth_state), AuthAction::None)
}

/// Human-readable badge for the unauth row's right side.
/// Pattern matches on auth_state so `[Disabled]` is only used for
/// `DisabledByPolicy`, not for unactionable `NotAuthenticated { action: None }`
/// (which uses `[Unavailable]` since there's nothing the user can do here).
fn unauth_row_badge(state: &ProviderAuthState) -> SharedString {
    match state {
        ProviderAuthState::Authenticated => {
            debug_assert!(
                false,
                "Authenticated provider should not appear in unauth rows"
            );
            "Ready".into()
        }
        ProviderAuthState::NotAuthenticated { action } => match action {
            AuthAction::SignInImperative => "Sign In".into(),
            AuthAction::EnterApiKeyInSettings => "Configure".into(),
            AuthAction::OpenUrl(_) => "Open".into(),
            AuthAction::None => "Unavailable".into(),
        },
        ProviderAuthState::RateLimited { .. } => "Rate Limited".into(),
        ProviderAuthState::DisabledByPolicy(_) => "Disabled".into(),
    }
}

/// Selector key for the harness — exposed so render_unauth_row tests can pin
/// rows by `provider_id` without scraping `Label` text.
fn unauth_row_selector(provider_id: &LanguageModelProviderId) -> String {
    format!("unauth-provider-row-{}", provider_id.0)
}

/// Renders an unauth-provider row as a [`ListItem`] with provider icon, name,
/// and an action badge.
fn render_unauth_row(ix: usize, row: &UnauthRow, focused: bool) -> impl IntoElement {
    let badge = unauth_row_badge(&row.auth_state);
    let actionable = unauth_row_is_actionable(row);
    let provider_id = row.provider.id();
    let provider_name = row.provider.name().0;
    let selector_key = unauth_row_selector(&provider_id);

    let icon_color = if actionable {
        Color::Default
    } else {
        Color::Muted
    };

    div()
        .id(("unauth-row-wrapper", ix))
        .debug_selector(move || selector_key)
        .child(
            ListItem::new(("unauth-row", ix))
                .inset(true)
                .spacing(ListItemSpacing::Sparse)
                .toggle_state(focused)
                .disabled(!actionable)
                .child(
                    h_flex()
                        .w_full()
                        .gap_1p5()
                        .child(
                            match &row.icon {
                                IconOrSvg::Icon(icon_name) => Icon::new(*icon_name),
                                IconOrSvg::Svg(icon_path) => {
                                    Icon::from_external_svg(icon_path.clone())
                                }
                            }
                            .color(icon_color)
                            .size(IconSize::Small),
                        )
                        .child(Label::new(provider_name).truncate()),
                )
                .end_slot(div().pr_2().child(Chip::new(badge))),
        )
}

/// Dispatches the AuthAction for an actionable row. `SignInImperative` and
/// `EnterApiKeyInSettings` both route to `OpenSettings` for now — the
/// imperative magic (Copilot device flow, etc.) lives in the Settings page
/// and is not yet plumbed end-to-end from the picker. The toggle-on-second-
/// click behavior of `OpenSettings` is documented as a known limitation
/// (see ST1b-followup).
fn dispatch_unauth_action(action: AuthAction, window: &mut Window, cx: &mut App) {
    match action {
        AuthAction::SignInImperative | AuthAction::EnterApiKeyInSettings => {
            window.dispatch_action(OpenSettings.boxed_clone(), cx);
        }
        AuthAction::OpenUrl(url) => {
            cx.open_url(url.as_str());
        }
        AuthAction::None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{future::BoxFuture, stream::BoxStream};
    use gpui::{AsyncApp, TestAppContext, http_client};
    use language_model::{
        LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelId,
        LanguageModelName, LanguageModelProviderId, LanguageModelProviderName,
        LanguageModelRequest, LanguageModelToolChoice,
    };
    use ui::IconName;

    #[derive(Clone)]
    struct TestLanguageModel {
        name: LanguageModelName,
        id: LanguageModelId,
        provider_id: LanguageModelProviderId,
        provider_name: LanguageModelProviderName,
    }

    impl TestLanguageModel {
        fn new(name: &str, provider: &str) -> Self {
            Self {
                name: LanguageModelName::from(name.to_string()),
                id: LanguageModelId::from(name.to_string()),
                provider_id: LanguageModelProviderId::from(provider.to_string()),
                provider_name: LanguageModelProviderName::from(provider.to_string()),
            }
        }
    }

    impl LanguageModel for TestLanguageModel {
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

        fn telemetry_id(&self) -> String {
            format!("{}/{}", self.provider_id.0, self.name.0)
        }

        fn max_token_count(&self) -> u64 {
            1000
        }

        fn count_tokens(
            &self,
            _: LanguageModelRequest,
            _: &App,
        ) -> BoxFuture<'static, http_client::Result<u64>> {
            unimplemented!()
        }

        fn stream_completion(
            &self,
            _: LanguageModelRequest,
            _: &AsyncApp,
        ) -> BoxFuture<
            'static,
            Result<
                BoxStream<
                    'static,
                    Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
                >,
                LanguageModelCompletionError,
            >,
        > {
            unimplemented!()
        }
    }

    fn create_models(model_specs: Vec<(&str, &str)>) -> Vec<ModelInfo> {
        create_models_with_favorites(model_specs, vec![])
    }

    fn create_models_with_favorites(
        model_specs: Vec<(&str, &str)>,
        favorites: Vec<(&str, &str)>,
    ) -> Vec<ModelInfo> {
        model_specs
            .into_iter()
            .map(|(provider, name)| {
                let is_favorite = favorites
                    .iter()
                    .any(|(fav_provider, fav_name)| *fav_provider == provider && *fav_name == name);
                ModelInfo {
                    model: Arc::new(TestLanguageModel::new(name, provider)),
                    icon: IconOrSvg::Icon(IconName::ZedAgent),
                    is_favorite,
                }
            })
            .collect()
    }

    fn assert_models_eq(result: Vec<ModelInfo>, expected: Vec<&str>) {
        assert_eq!(
            result.len(),
            expected.len(),
            "Number of models doesn't match"
        );

        for (i, expected_name) in expected.iter().enumerate() {
            assert_eq!(
                result[i].model.telemetry_id(),
                *expected_name,
                "Model at position {} doesn't match expected model",
                i
            );
        }
    }

    #[gpui::test]
    fn test_exact_match(cx: &mut TestAppContext) {
        let models = create_models(vec![
            ("zed", "Claude 3.7 Sonnet"),
            ("zed", "Claude 3.7 Sonnet Thinking"),
            ("zed", "gpt-5"),
            ("zed", "gpt-5-mini"),
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-5"),
            ("openai", "gpt-5-mini"),
            ("ollama", "mistral"),
            ("ollama", "deepseek"),
        ]);
        let matcher = ModelMatcher::new(
            models,
            cx.foreground_executor().clone(),
            cx.background_executor.clone(),
        );

        // The order of models should be maintained, case doesn't matter
        let results = matcher.exact_search("GPT-5");
        assert_models_eq(
            results,
            vec![
                "zed/gpt-5",
                "zed/gpt-5-mini",
                "openai/gpt-5",
                "openai/gpt-5-mini",
            ],
        );
    }

    #[gpui::test]
    fn test_fuzzy_match(cx: &mut TestAppContext) {
        let models = create_models(vec![
            ("zed", "Claude 3.7 Sonnet"),
            ("zed", "Claude 3.7 Sonnet Thinking"),
            ("zed", "gpt-5"),
            ("zed", "gpt-5-mini"),
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-5"),
            ("openai", "gpt-5-mini"),
            ("ollama", "mistral"),
            ("ollama", "deepseek"),
        ]);
        let matcher = ModelMatcher::new(
            models,
            cx.foreground_executor().clone(),
            cx.background_executor.clone(),
        );

        // Results should preserve models order whenever possible.
        // In the case below, `zed/gpt-5-mini` and `openai/gpt-5-mini` have identical
        // similarity scores, but `zed/gpt-5-mini` was higher in the models list,
        // so it should appear first in the results.
        let results = matcher.fuzzy_search("mini");
        assert_models_eq(results, vec!["zed/gpt-5-mini", "openai/gpt-5-mini"]);

        // Model provider should be searchable as well
        let results = matcher.fuzzy_search("ol"); // meaning "ollama"
        assert_models_eq(results, vec!["ollama/mistral", "ollama/deepseek"]);

        // Fuzzy search - search for Claude to get the Thinking variant
        let results = matcher.fuzzy_search("thinking");
        assert_models_eq(results, vec!["zed/Claude 3.7 Sonnet Thinking"]);
    }

    #[gpui::test]
    fn test_recommended_models_also_appear_in_other(_cx: &mut TestAppContext) {
        let recommended_models = create_models(vec![("zed", "claude")]);
        let all_models = create_models(vec![
            ("zed", "claude"), // Should also appear in "other"
            ("zed", "gemini"),
            ("copilot", "o3"),
        ]);

        let grouped_models = GroupedModels::new(all_models, recommended_models, Vec::new());

        let actual_all_models = grouped_models
            .all
            .values()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();

        // Recommended models should also appear in "all"
        assert_models_eq(
            actual_all_models,
            vec!["zed/claude", "zed/gemini", "copilot/o3"],
        );
    }

    #[gpui::test]
    fn test_models_from_different_providers(_cx: &mut TestAppContext) {
        let recommended_models = create_models(vec![("zed", "claude")]);
        let all_models = create_models(vec![
            ("zed", "claude"), // Should also appear in "other"
            ("zed", "gemini"),
            ("copilot", "claude"), // Different provider, should appear in "other"
        ]);

        let grouped_models = GroupedModels::new(all_models, recommended_models, Vec::new());

        let actual_all_models = grouped_models
            .all
            .values()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();

        // All models should appear in "all" regardless of recommended status
        assert_models_eq(
            actual_all_models,
            vec!["zed/claude", "zed/gemini", "copilot/claude"],
        );
    }

    #[gpui::test]
    fn test_favorites_section_appears_when_favorites_exist(_cx: &mut TestAppContext) {
        let recommended_models = create_models(vec![("zed", "claude")]);
        let all_models = create_models_with_favorites(
            vec![("zed", "claude"), ("zed", "gemini"), ("openai", "gpt-4")],
            vec![("zed", "gemini")],
        );

        let grouped_models = GroupedModels::new(all_models, recommended_models, Vec::new());
        let entries = grouped_models.entries();

        assert!(matches!(
            entries.first(),
            Some(LanguageModelPickerEntry::Separator(s)) if s == "Favorite"
        ));

        assert_models_eq(grouped_models.favorites, vec!["zed/gemini"]);
    }

    #[gpui::test]
    fn test_no_favorites_section_when_no_favorites(_cx: &mut TestAppContext) {
        let recommended_models = create_models(vec![("zed", "claude")]);
        let all_models = create_models(vec![("zed", "claude"), ("zed", "gemini")]);

        let grouped_models = GroupedModels::new(all_models, recommended_models, Vec::new());
        let entries = grouped_models.entries();

        assert!(matches!(
            entries.first(),
            Some(LanguageModelPickerEntry::Separator(s)) if s == "Recommended"
        ));

        assert!(grouped_models.favorites.is_empty());
    }

    #[gpui::test]
    fn test_models_have_correct_actions(_cx: &mut TestAppContext) {
        let recommended_models =
            create_models_with_favorites(vec![("zed", "claude")], vec![("zed", "claude")]);
        let all_models = create_models_with_favorites(
            vec![("zed", "claude"), ("zed", "gemini"), ("openai", "gpt-4")],
            vec![("zed", "claude")],
        );

        let grouped_models = GroupedModels::new(all_models, recommended_models, Vec::new());
        let entries = grouped_models.entries();

        for entry in &entries {
            if let LanguageModelPickerEntry::Model(info) = entry {
                if info.model.telemetry_id() == "zed/claude" {
                    assert!(info.is_favorite, "zed/claude should be a favorite");
                } else {
                    assert!(
                        !info.is_favorite,
                        "{} should not be a favorite",
                        info.model.telemetry_id()
                    );
                }
            }
        }
    }

    #[gpui::test]
    fn test_favorites_appear_in_other_sections(_cx: &mut TestAppContext) {
        let favorites = vec![("zed", "gemini"), ("openai", "gpt-4")];

        let recommended_models =
            create_models_with_favorites(vec![("zed", "claude")], favorites.clone());

        let all_models = create_models_with_favorites(
            vec![
                ("zed", "claude"),
                ("zed", "gemini"),
                ("openai", "gpt-4"),
                ("openai", "gpt-3.5"),
            ],
            favorites,
        );

        let grouped_models = GroupedModels::new(all_models, recommended_models, Vec::new());

        assert_models_eq(grouped_models.favorites, vec!["zed/gemini", "openai/gpt-4"]);
        assert_models_eq(grouped_models.recommended, vec!["zed/claude"]);
        assert_models_eq(
            grouped_models.all.values().flatten().cloned().collect(),
            vec!["zed/claude", "zed/gemini", "openai/gpt-4", "openai/gpt-3.5"],
        );
    }

    // ─── ST1b: unauth-provider helpers + data-flow ──────────────────────

    use language_model::{SafeUrl, SanitizedReason, fake_provider::FakeLanguageModelProvider};

    fn fake_unauth_row(id: &str, name: &str, state: ProviderAuthState) -> UnauthRow {
        let provider = FakeLanguageModelProvider::new(
            LanguageModelProviderId::from(id.to_string()),
            LanguageModelProviderName::from(name.to_string()),
        );
        provider.set_auth_state(Some(state.clone()));
        UnauthRow {
            provider: Arc::new(provider),
            auth_state: state,
            icon: IconOrSvg::Icon(IconName::AiAnthropic),
        }
    }

    #[test]
    fn unauth_action_for_routes_each_state() {
        assert!(matches!(
            unauth_action_for(&ProviderAuthState::Authenticated),
            AuthAction::None
        ));
        assert!(matches!(
            unauth_action_for(&ProviderAuthState::NotAuthenticated {
                action: AuthAction::SignInImperative
            }),
            AuthAction::SignInImperative
        ));
        assert!(matches!(
            unauth_action_for(&ProviderAuthState::NotAuthenticated {
                action: AuthAction::EnterApiKeyInSettings
            }),
            AuthAction::EnterApiKeyInSettings
        ));
        let url = SafeUrl::https("https://example.com").unwrap();
        assert!(matches!(
            unauth_action_for(&ProviderAuthState::NotAuthenticated {
                action: AuthAction::OpenUrl(url)
            }),
            AuthAction::OpenUrl(_)
        ));
        // RateLimited / DisabledByPolicy collapse to None even if the inner
        // action variant says otherwise.
        assert!(matches!(
            unauth_action_for(&ProviderAuthState::RateLimited {
                retry_after: None,
                action: AuthAction::SignInImperative,
            }),
            AuthAction::None
        ));
        assert!(matches!(
            unauth_action_for(&ProviderAuthState::DisabledByPolicy(SanitizedReason::new(
                "policy"
            ))),
            AuthAction::None
        ));
    }

    #[test]
    fn unauth_row_is_actionable_only_for_meaningful_actions() {
        let signin = fake_unauth_row(
            "fake-signin",
            "Fake SignIn",
            ProviderAuthState::NotAuthenticated {
                action: AuthAction::SignInImperative,
            },
        );
        assert!(unauth_row_is_actionable(&signin));

        let configure = fake_unauth_row(
            "fake-configure",
            "Fake Configure",
            ProviderAuthState::NotAuthenticated {
                action: AuthAction::EnterApiKeyInSettings,
            },
        );
        assert!(unauth_row_is_actionable(&configure));

        let open_url = fake_unauth_row(
            "fake-open",
            "Fake Open",
            ProviderAuthState::NotAuthenticated {
                action: AuthAction::OpenUrl(SafeUrl::https("https://example.com").unwrap()),
            },
        );
        assert!(unauth_row_is_actionable(&open_url));

        let unavailable = fake_unauth_row(
            "fake-unavailable",
            "Fake Unavailable",
            ProviderAuthState::NotAuthenticated {
                action: AuthAction::None,
            },
        );
        assert!(!unauth_row_is_actionable(&unavailable));

        let policy = fake_unauth_row(
            "fake-policy",
            "Fake Policy",
            ProviderAuthState::DisabledByPolicy(SanitizedReason::new("blocked")),
        );
        assert!(!unauth_row_is_actionable(&policy));

        let rate_limited = fake_unauth_row(
            "fake-rate",
            "Fake Rate",
            ProviderAuthState::RateLimited {
                retry_after: None,
                action: AuthAction::None,
            },
        );
        assert!(!unauth_row_is_actionable(&rate_limited));

        // RateLimited is non-actionable even when the inner action says otherwise —
        // pins the invariant in `unauth_action_for`.
        let rate_limited_with_action = fake_unauth_row(
            "fake-rate-signin",
            "Fake Rate SignIn",
            ProviderAuthState::RateLimited {
                retry_after: None,
                action: AuthAction::SignInImperative,
            },
        );
        assert!(
            !unauth_row_is_actionable(&rate_limited_with_action),
            "RateLimited must be non-actionable regardless of inner action"
        );
    }

    #[test]
    fn unauth_row_badge_picks_per_variant_label() {
        assert_eq!(
            unauth_row_badge(&ProviderAuthState::NotAuthenticated {
                action: AuthAction::SignInImperative
            })
            .as_ref(),
            "Sign In"
        );
        assert_eq!(
            unauth_row_badge(&ProviderAuthState::NotAuthenticated {
                action: AuthAction::EnterApiKeyInSettings
            })
            .as_ref(),
            "Configure"
        );
        let url = SafeUrl::https("https://example.com").unwrap();
        assert_eq!(
            unauth_row_badge(&ProviderAuthState::NotAuthenticated {
                action: AuthAction::OpenUrl(url)
            })
            .as_ref(),
            "Open"
        );
        assert_eq!(
            unauth_row_badge(&ProviderAuthState::NotAuthenticated {
                action: AuthAction::None
            })
            .as_ref(),
            "Unavailable"
        );
        assert_eq!(
            unauth_row_badge(&ProviderAuthState::RateLimited {
                retry_after: None,
                action: AuthAction::None,
            })
            .as_ref(),
            "Rate Limited"
        );
        assert_eq!(
            unauth_row_badge(&ProviderAuthState::DisabledByPolicy(SanitizedReason::new(
                "x"
            )))
            .as_ref(),
            "Disabled"
        );
    }

    #[test]
    fn unauth_row_selector_is_provider_id_keyed() {
        let id = LanguageModelProviderId::from("anthropic".to_string());
        assert_eq!(unauth_row_selector(&id), "unauth-provider-row-anthropic");
    }

    #[test]
    fn entries_emit_unauth_section_at_bottom_with_separator() {
        let row = fake_unauth_row(
            "anthropic",
            "Anthropic",
            ProviderAuthState::NotAuthenticated {
                action: AuthAction::EnterApiKeyInSettings,
            },
        );
        let grouped = GroupedModels::new(Vec::new(), Vec::new(), vec![row]);
        let entries = grouped.entries();

        // Last 2 entries: Separator("More providers") + UnauthProvider(...)
        assert!(entries.len() >= 2);
        match &entries[entries.len() - 2] {
            LanguageModelPickerEntry::Separator(s) => {
                assert_eq!(s.as_ref(), "More providers")
            }
            other => panic!(
                "expected separator, got {:?}",
                std::mem::discriminant(other)
            ),
        }
        match &entries[entries.len() - 1] {
            LanguageModelPickerEntry::UnauthProvider(r) => {
                assert_eq!(r.provider.id().0.as_ref(), "anthropic");
            }
            _ => panic!("expected UnauthProvider entry"),
        }
    }

    #[test]
    fn entries_skip_unauth_section_when_empty() {
        let grouped = GroupedModels::new(Vec::new(), Vec::new(), Vec::new());
        let entries = grouped.entries();
        // No "More providers" separator should appear.
        for entry in &entries {
            if let LanguageModelPickerEntry::Separator(s) = entry {
                assert_ne!(s.as_ref(), "More providers");
            }
        }
    }

    #[test]
    fn unauth_rows_do_not_contribute_to_models_list() {
        // GroupedModels is constructed with separate inputs for `all` and `unauth`
        // — this test guards the contract that they don't bleed.
        let row = fake_unauth_row(
            "anthropic",
            "Anthropic",
            ProviderAuthState::NotAuthenticated {
                action: AuthAction::EnterApiKeyInSettings,
            },
        );
        let grouped = GroupedModels::new(Vec::new(), Vec::new(), vec![row]);
        assert!(
            grouped.all.is_empty(),
            "unauth rows must not appear in `all` models map"
        );
        assert!(grouped.favorites.is_empty());
        assert!(grouped.recommended.is_empty());
    }
}
