# Hardcore Review: Index-Access DAG IDE Panel Feature

- **Date**: 2025-07-22 (Iteration 1)
- **Content Type**: Code Changes (multi-file feature)
- **Iteration**: 1
- **Reviewer Model(s)**: claude-opus-4.6, gpt-5.4, gpt-5.3-codex (synthesized by claude-sonnet-4.6)
- **Verdict**: NEEDS WORK

## Files Reviewed
1. `crates/caduceus_bridge/src/index_dag.rs` (630 lines)
2. `crates/agent/src/thread.rs` lines 3870–3904 (register/unregister subagent)
3. `crates/agent_ui/src/agent_panel.rs` (DAG chip, cache, io_task, tests)
4. `crates/agent/src/agent.rs` line 1340–1348 (/dag command) and 1671–1698 (/help)

## Original Requirements vs. Delivery

| Requirement | Delivered? | Notes |
|---|---|---|
| (a) DAG drawn in chat | ✅ | `/dag` renders ASCII into chat wrapped in fences |
| (b) IDE panel/view | ❌ | A chip badge + tooltip in the existing toolbar. No dockable panel. |
| (c) Subagent edges | ⚠️ | Recorded, but using wrong child ID (entity_id ≠ SessionId) |
| (d) Live auto-refresh | ❌ | Refresh only fires when panel renders; idle UI = stale data |

---

## Scores

| # | Dimension | Score | Min-Reviewer | Notes |
|---|-----------|-------|--------------|-------|
| 1 | Correctness | **2/10** | gpt-5.4 | False read-read contention; child ID wrong ID space; "live" refresh is false |
| 2 | Completeness | **1/10** | gpt-5.4 | Requirements (b) and (d) undelivered; /dag undiscoverable; zero E2E tests |
| 3 | Security | **8/10** | all | In-process global statics; no network exposure; Rust safety prevents injection |
| 4 | Clarity | **3/10** | gpt-5.3-codex | "⛓ 3/12" opaque; tooltip lies "live"; /dag invisible in /help |
| 5 | Architecture | **2/10** | gpt-5.4 | Render path triggers I/O refresh; no dedicated DAG view abstraction; SRP blown |
| 6 | Test Coverage | **3/10** | gpt-5.3-codex | No E2E via Thread; no chip color test; no label format test; no /dag fence test |
| 7 | Performance | **6/10** | gpt-5.4 | O(n²) edge dedup in from_snapshots(); double snapshot acquisition per refresh |
| 8 | SRP | **2/10** | gpt-5.4 | agent_panel.rs conflates render, caching, polling, filesystem I/O, and DAG UX |
| 9 | KISS/YAGNI/DRY | **4/10** | gpt-5.4 | Command table duplicated; two identical ring-buffer registries; double snapshot |

**Average: 3.4/10 — NEEDS WORK**

---

## Findings

### 🔴 Critical

---

**1. [thread.rs:3882–3885] — Spawn edges use GPUI entity_id (u64) not acp::SessionId**

`child_id = subagent.entity_id().as_u64().to_string()` (e.g. `"42"`) while `parent_id = self.id.0.to_string()` uses the actual `acp::SessionId` (a UUID-like string). The DAG renders `user:abc-def-123 ──spawns──► subagent:42`. "42" cannot be correlated to anything the user sees (not the thread name, not the session shown in the UI). Access events use `SessionId`; spawn events use `entity_id`. Cross-referencing the two is structurally impossible.

**Fix:** Read the child's SessionId through the upgraded entity:
```rust
pub(crate) fn register_running_subagent(
    &mut self,
    subagent: WeakEntity<Thread>,
    cx: &App,
) {
    let parent_id = self.id.0.to_string();
    let parent_kind = if self.is_subagent() {
        caduceus_bridge::index_dag::AgentKind::Subagent
    } else {
        caduceus_bridge::index_dag::AgentKind::User
    };
    // Use SessionId — same ID space as access events — so the DAG
    // can correlate spawn edges with access edges.
    let child_id = subagent
        .upgrade()
        .map(|s| s.read(cx).id().0.to_string())
        .unwrap_or_else(|| format!("entity:{}", subagent.entity_id().as_u64()));
    caduceus_bridge::index_dag::record_spawn(
        parent_id,
        parent_kind,
        child_id,
        caduceus_bridge::index_dag::AgentKind::Subagent,
    );
    self.running_subagents.push(subagent);
}
```
Update call sites to pass `cx: &App` (the caller already has it).

---

**2. [index_dag.rs:300–315] — `contention_points()` falsely flags read-read as contention**

Two agents concurrently reading the same index is safe. Only write-read or write-write combinations are genuine contention. The current implementation flags any resource touched by >1 distinct agent — including two read-only agents — producing `⚠ contention` in the chip and tooltip when there is no actual conflict.

**Fix:**
```rust
pub fn contention_points(&self) -> Vec<&IndexResource> {
    #[derive(Default)]
    struct ResourceState<'a> {
        agents: std::collections::HashSet<&'a str>,
        has_write: bool,
    }
    let mut by_resource: std::collections::HashMap<&IndexResource, ResourceState<'_>> =
        std::collections::HashMap::new();
    for e in &self.edges {
        if let (DagNode::Agent { id, .. }, DagNode::Resource(r)) = (&e.from, &e.to) {
            let state = by_resource.entry(r).or_default();
            state.agents.insert(id.as_str());
            if matches!(e.access, AccessKind::Write) {
                state.has_write = true;
            }
        }
    }
    // Contention = write from one agent + read/write from another agent.
    // Pure read-read is NOT contention.
    by_resource
        .into_iter()
        .filter_map(|(r, state)| {
            if state.has_write && state.agents.len() > 1 {
                Some(r)
            } else {
                None
            }
        })
        .collect()
}
```

Also update the existing test `two_agents_on_same_resource_show_contention` — it used two readers, which should now NOT show contention. Change one agent to a writer.

---

**3. [agent_panel.rs:4390 + 4850–5124] — "Live, refreshed every 5s" is false**

`refresh_caduceus_stats` is called only from the render path (`render_top_bar` at line 4390). If the panel is visible but idle (no hover, no typing, no subscription firing), `render()` never runs, and the chip never updates. A user can watch 10+ minutes of subagent activity with a completely stale chip. The tooltip text `"live, refreshed every 5s"` is a false guarantee.

**Fix — add a periodic poll task in the panel constructor:**
```rust
// In AgentPanel struct:
_caduceus_stats_poll_task: Task<()>,

// In AgentPanel::new() / construction:
let poll_task = cx.spawn(async move |this, mut cx| {
    loop {
        cx.background_executor().timer(Duration::from_secs(5)).await;
        let _ = this.update(&mut cx, |panel, cx| {
            // Reset last_refresh so the TTL guard doesn't block the forced poll.
            panel.caduceus_stats_cache.last_refresh =
                Instant::now() - Duration::from_secs(60);
            panel.refresh_caduceus_stats(cx);
        });
    }
});
// Store: _caduceus_stats_poll_task: poll_task

// Also remove the render-path trigger:
// DELETE: self.refresh_caduceus_stats(cx); // line 4390
```

Until this is done, change the tooltip text to be honest:
```rust
"Index Access DAG (refreshed when panel is active, up to every 5s)\n\
 spawns: {}  accesses: {}  contention: {}\n\n{}"
```

---

**4. [agent.rs:1671–1698] — `/dag` is absent from `/help` output**

`build_available_commands_for_project` registers `/dag` for ACP autocomplete (line 961), but the hand-written `/help` response (lines 1671–1698) lists every other command and omits `/dag`. A user who types `/help` will never discover this feature.

**Fix:** Add exactly this line in the Tools section of the `/help` response:
```rust
"                 - `/dag` — show the Index Access DAG in chat (subagent spawns + index reads/writes)\n\
```
Insert after the `/status` line.

Also, eliminate the dual-source problem for future commands (see Finding 9).

---

**5. [index_dag.rs:161–167, 182–197] — Lock poisoning permanently silences the DAG**

`if let Ok(mut q) = registry().lock()` silently drops events whenever a thread panics inside the lock. A panic in a background agent during `record()` permanently poisons the mutex, and all subsequent `record()` and `snapshot()` calls return silently without data. The DAG goes dark with no log output and no indication to the user.

**Fix — recover from poison in all lock sites:**
```rust
pub fn record(
    agent_id: impl Into<String>,
    agent_kind: AgentKind,
    resource: IndexResource,
    access: AccessKind,
) {
    let event = AccessEvent {
        agent_id: agent_id.into(),
        agent_kind,
        resource,
        access,
        at: Instant::now(),
    };
    let mut q = match registry().lock() {
        Ok(g) => g,
        Err(poisoned) => {
            log::warn!("[index_dag] registry poisoned; recovering");
            poisoned.into_inner()
        }
    };
    if q.len() == MAX_EVENTS {
        q.pop_front();
    }
    q.push_back(event);
    bump_version();
}
```
Apply the same pattern to `record_spawn()`, `snapshot()`, and `spawn_snapshot()`.

---

### 🟡 Important

---

**6. [agent_panel.rs:4901–4911] — Double snapshot: render_current_ascii + snapshot() diverge**

`render_current_ascii()` (line 4901) internally calls `snapshot()` + `spawn_snapshot()` and builds a DAG. Then lines 4902–4903 call `snapshot()` + `spawn_snapshot()` again to compute counts. Between the two acquisitions, events can arrive, meaning the ASCII string and the numeric counts displayed in the tooltip header are from different logical instants. The user sees counts that don't match the ASCII.

**Fix:** Call snapshot once, derive both ASCII and counts from the same data:
```rust
let (dag_events, dag_spawns) = {
    let now = std::time::Instant::now();
    let accesses = caduceus_bridge::index_dag::snapshot_at(now);
    let spawns = caduceus_bridge::index_dag::spawn_snapshot_at(now);
    (accesses, spawns)
};
let dag = caduceus_bridge::index_dag::Dag::from_snapshots(&dag_events, &dag_spawns);
let dag_ascii = dag.to_ascii();
let dag_contention_count = dag.contention_points().len();
let dag_spawn_count = dag.spawns.len();
let dag_access_count: usize = dag.edges.iter().map(|e| e.count as usize).sum();
```
Add `snapshot_at(Instant)` to `index_dag.rs` so the same `now` can be passed to both calls, or add `atomic_snapshot() -> (Vec<AccessEvent>, Vec<SpawnEvent>)`.

---

**7. [thread.rs:3895–3904] — Subagent termination leaves ghost spawn edges for up to 5 minutes**

`unregister_running_subagent` removes the subagent from `running_subagents` but leaves the `SpawnEvent` in the ring buffer until `EVENT_TTL` (5 min) expires. During rapid spawn/teardown cycles the DAG shows parent→child edges for terminated agents.

**Fix:** Add a removal API and call it on unregister:
```rust
// index_dag.rs — new public function:
pub fn remove_spawn(child_id: &str) {
    match spawn_registry().lock() {
        Ok(mut q) => q.retain(|e| e.child_id != child_id),
        Err(p) => p.into_inner().retain(|e| e.child_id != child_id),
    }
    bump_version();
}

// thread.rs — updated unregister:
pub(crate) fn unregister_running_subagent(
    &mut self,
    subagent_session_id: &acp::SessionId,
    cx: &App,
) {
    caduceus_bridge::index_dag::remove_spawn(&subagent_session_id.0.to_string());
    self.running_subagents.retain(|s| {
        s.upgrade()
            .map_or(false, |s| s.read(cx).id() != subagent_session_id)
    });
}
```

---

**8. [index_dag.rs:167 and 192] — `bump_version()` outside lock creates TOCTOU window**

The version bump happens after the lock is released. A concurrent reader can see the new event in `snapshot()` but still read the old version from `version()`. A UI poller comparing version numbers will incorrectly conclude "no change" and skip re-render, silently staling the chip.

**Fix:** Move `bump_version()` inside the lock scope in both `record()` and `record_spawn()`.

---

**9. [agent.rs:954–962 vs 1671–1698] — Dual command registry will drift**

`build_available_commands_for_project` is the ACP autocomplete source; the `/help` string is hand-written. They already disagree on `/dag`. Every new command added to autocomplete risks being omitted from `/help` and vice versa.

**Fix:** Single source of truth:
```rust
/// All Caduceus slash commands. Single source of truth for both
/// ACP autocomplete registration and the /help output.
const CADUCEUS_COMMANDS: &[(&str, &str)] = &[
    ("compact",    "Compress conversation context to free tokens"),
    ("mode",       "Show or switch Caduceus agent mode"),
    ("context",    "Show context usage, zone status, and pinned items"),
    ("checkpoint", "Create a code checkpoint for rollback"),
    ("dag",        "Show the Index Access DAG in chat (subagent spawns + index reads/writes)"),
    ("status",     "Show unified Caduceus dashboard"),
    ("map",        "Show project repo map (tree-sitter symbol outline)"),
    ("review",     "Review code for security issues"),
    ("headless",   "Generate a CLI command for headless execution"),
    ("help",       "List all Caduceus commands"),
];
```
Then generate both the `acp::AvailableCommand` list and the `/help` Markdown from this table.

---

**10. [index_dag.rs:269–289] — O(n²) edge deduplication**

`from_snapshots()` calls `.find()` on `edges` for every event — O(n×e) where n = events and e = unique edges. At MAX_EVENTS=256 this is negligible, but the edge dedup pattern is incorrect if the cap is raised.

**Fix:**
```rust
pub fn from_snapshots(events: &[AccessEvent], spawns: &[SpawnEvent]) -> Self {
    use std::collections::HashMap;
    type EdgeKey = (String, String, u8); // (agent_id, resource_label, access as u8)
    let mut counts: HashMap<EdgeKey, (DagNode, DagNode, AccessKind)> = HashMap::new();
    let mut count_map: HashMap<EdgeKey, u32> = HashMap::new();

    for ev in events {
        let key = (
            format!("{}:{}", ev.agent_kind.label(), ev.agent_id),
            ev.resource.label().to_string(),
            ev.access as u8,
        );
        count_map.entry(key.clone()).and_modify(|c| *c += 1).or_insert(1);
        counts.entry(key).or_insert_with(|| (
            DagNode::Agent { id: ev.agent_id.clone(), kind: ev.agent_kind },
            DagNode::Resource(ev.resource.clone()),
            ev.access,
        ));
    }

    let edges = count_map
        .into_iter()
        .map(|(key, count)| {
            let (from, to, access) = counts.remove(&key).unwrap();
            DagEdge { from, to, access, count }
        })
        .collect();

    Dag { edges, spawns: spawns.to_vec() }
}
```

---

### 🟢 Suggestions

---

**11. [agent_panel.rs:4785–4789] — Chip label "⛓ 3/12" is opaque**

A first-time user cannot determine what the two numbers mean without hovering to read the tooltip. The chain emoji doesn't suggest "DAG" or "index access".

**Fix:**
```rust
let label = if contention > 0 {
    format!("⚠ DAG S:{} A:{} C:{}", spawns, accesses, contention)
} else {
    format!("DAG S:{} A:{}", spawns, accesses)
};
```
S = spawns, A = accesses, C = contention. Still short but now self-annotated. Or consider `"DAG ↑{} ↷{}"` for spawns/accesses if character count is a concern.

---

**12. [Missing tests] — E2E through Thread, chip color/label/tooltip, /dag fences**

None of these code paths have regression coverage:

```rust
// Add to thread.rs or integration tests:
#[test]
fn register_running_subagent_records_spawn_in_dag() {
    caduceus_bridge::index_dag::clear(); // use test helper
    // Construct a parent Thread with a known SessionId
    // Spawn a child Thread with a known SessionId
    // Call parent.register_running_subagent(child.downgrade(), cx)
    let snap = caduceus_bridge::index_dag::spawn_snapshot();
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].parent_id, "parent-session-id");
    assert_eq!(snap[0].child_id, "child-session-id"); // NOT entity_id
}

// Add to index_dag.rs tests:
#[test]
fn two_readers_do_not_show_contention() {
    let _g = test_lock();
    clear();
    record("u1", AgentKind::User, IndexResource::SemanticIndex, AccessKind::Read);
    record("bg-1", AgentKind::Subagent, IndexResource::SemanticIndex, AccessKind::Read);
    let dag = Dag::from_snapshot(&snapshot());
    assert!(dag.contention_points().is_empty(),
        "read-read must not be flagged as contention:\n{}", dag.to_ascii());
}

#[test]
fn writer_plus_reader_shows_contention() {
    let _g = test_lock();
    clear();
    record("u1", AgentKind::User, IndexResource::SemanticIndex, AccessKind::Write);
    record("bg-1", AgentKind::Subagent, IndexResource::SemanticIndex, AccessKind::Read);
    let dag = Dag::from_snapshot(&snapshot());
    assert_eq!(dag.contention_points().len(), 1);
}
```

For chip label/color/tooltip tests: extract the chip construction into a `render_dag_chip_data()` method returning a plain struct, so tests don't require a full GPUI context.

---

**13. [index_dag.rs:124–144, 145–157] — DRY: two identical ring-buffer registries**

`registry()` and `spawn_registry()` are structural clones: same `Mutex<VecDeque<T>>`, same `OnceLock`, same cap check in `push`. Consider a `RingBuffer<T>` newtype, or at minimum extract a `push_to_ring` helper to avoid divergence when fixing bugs (e.g., lock poison recovery needs to be applied to both separately right now).

---

**14. [index_dag.rs tests] — No concurrent stress test for global static mutexes**

All tests run sequentially behind `test_lock()`. There is no test that exercises concurrent `record()` calls from multiple threads.

```rust
#[test]
fn concurrent_record_does_not_corrupt_or_panic() {
    let _g = test_lock();
    clear();
    let handles: Vec<_> = (0..20).map(|i| {
        std::thread::spawn(move || {
            for j in 0..20 {
                record(
                    format!("agent-{i}"),
                    AgentKind::Subagent,
                    IndexResource::SemanticIndex,
                    if j % 2 == 0 { AccessKind::Read } else { AccessKind::Write },
                );
            }
        })
    }).collect();
    for h in handles { h.join().expect("thread panicked"); }
    assert!(snapshot().len() <= MAX_EVENTS);
}
```

---

## Recommended Actions (Priority Order)

1. **Fix child ID in `register_running_subagent`** (Finding 1) — every spawn edge shown in the DAG is using the wrong identity. Fix before any other DAG work.
2. **Fix `contention_points()` to require a writer** (Finding 2) — false warnings destroy trust in the feature.
3. **Add periodic poll task** (Finding 3) — the tooltip says "live"; make it true.
4. **Add `/dag` to `/help`** (Finding 4) — the feature is invisible to users who type `/help`.
5. **Add lock poison recovery** (Finding 5) — a background agent panic must not permanently silence the DAG.
6. **Fix double snapshot / count drift** (Finding 6) — counts in the tooltip header should match the ASCII body.
7. **Add `remove_spawn` on unregister** (Finding 7) — terminated subagents show as ghost edges.
8. **Move `bump_version` inside lock** (Finding 8) — eliminates TOCTOU between data and version.
9. **Unify command registry** (Finding 9) — prevents `/help` and autocomplete drifting again.
10. **Add missing tests** (Finding 12) — E2E through Thread, chip invariants, correct contention semantics.
