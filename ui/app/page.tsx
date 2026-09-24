"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_ROOT = "http://127.0.0.1:8765/api";

type Health = {
  status: string;
  csrf_token: string;
  model: {
    status: string;
    selected_model: string | null;
    message: string;
  };
  corpus: {
    document_count: number;
    chunk_count: number;
    classification: string;
    publication_consent: string;
  };
};

type Source = {
  chunk_id: string;
  title: string;
  filename: string;
  location: string;
  excerpt: string;
  modes: string[];
  source_sha256: string;
  content_sha256: string;
  protocol_mode: string;
  retrieval_executable: boolean;
  consent_status: string;
  selection_reason: string;
  usage_note: string;
  used_by: string[];
};

type Core = {
  core_id: "form" | "flow" | "accord";
  propositions: string[];
  constraints: string[];
  symbolic_relations: string[];
  open_questions: string[];
  confidence: number;
  confidence_note: string;
  imbalance_signals: string[];
  kernel_influences: string[];
  evidence_refs: string[];
  proposal_hash: string | null;
  provider_id: string;
};

type Observation = {
  observer: string;
  observed: string;
  agreements: string[];
  disagreements: string[];
  observer_self_shadow: string[];
  request_to_other: string[];
  basis_refs: string[];
  provider_id: string;
  evaluation_hash: string | null;
};

type RecursionFrame = {
  depth: number;
  triggering_symbol: string;
  transformed_symbol: string;
  new_distinctions: string[];
  new_relations: string[];
  added_constraints: string[];
  unresolved_polarity: string[];
  stop_reason: string | null;
  path_ids: string[];
};

type Control = {
  status: "not_run" | "complete" | "unavailable" | "ineligible";
  message?: string;
  response?: {
    direction: string;
    rationale: string;
    next_step: string;
    held_open: string[];
    cited_evidence_refs: string[];
  };
  full_text?: string;
  model_call_count?: number;
  elapsed_ms?: number;
  comparison: {
    winner_declared: boolean;
    shared_retrieval_snapshot_hash: string;
    full_tree: Record<string, number | boolean | string | null>;
    neutral_control: Record<string, number | boolean | string | null>;
    comparison_eligible: boolean;
    budget_match: boolean | null;
    treatment_only_inputs: {
      symbol: boolean;
      bounded_kernels: string[];
      note: string;
    };
    honesty_note: string;
  };
};

type SeedResult = {
  session_id: string;
  parent_session_id: string | null;
  execution_mode: "neural_graph" | "deterministic_fallback";
  model: {
    status: string;
    selected_model: string | null;
    message: string;
    call_count: number;
    fallback_reason: string | null;
  };
  response: {
    direction: string;
    rationale: string;
    next_step: string;
    held_open: string[];
    cited_chunk_ids: string[];
    renderer_id: string;
    full_text: string;
  };
  sources: Source[];
  retrieval: {
    snapshot_hash: string;
    classification: string;
    publication_consent: string;
  };
  inner_process: {
    cores: Core[];
    observations: Observation[];
    all_initial_views_completed_before_observation: boolean;
    coalescence: {
      selected_direction: string;
      rationale: string;
      proposed_action: string;
      retained_alternatives: string[];
      rejected_alternatives: string[];
      unresolved_tensions: string[];
      confidence: number;
      selected_candidate_id: string;
      selection_rule: string;
      candidate_assessments: Array<{
        candidate_id: string;
        direction: string;
        next_step: string;
        rationale: string;
        status: "admitted" | "rejected";
        graph_score: number;
        rejection_reasons: string[];
        supporting_proposition_refs: string[];
        responding_observation_refs: string[];
        satisfied_constraint_refs: string[];
      }>;
    };
    recursion: RecursionFrame[];
    route: Array<{
      sequence: number;
      node: string;
      world: string;
      event_type: string;
      title: string;
      meaning: string;
    }>;
    boundaries: {
      external_actions_taken: number;
      protocol_invoked: boolean;
      durable_memory_writes: number;
      memory_proposal_created: boolean;
      daat_decision: string;
      feedback_status: string;
    };
  };
  trace: {
    run_id: string;
    graph_version: string;
    state_hash: string;
    context_packet_hash: string;
    feedback_state_packet_hash: string;
    final_event_hash: string;
    event_count: number;
    elapsed_ms: number;
  };
  feedback: {
    status: string;
    received_text: string | null;
    note: string;
  };
  memory: {
    durable_memory_enabled: boolean;
    durable_writes: number;
    automatic_reingestion: boolean;
  };
  control: Control;
};

type TabName = "response" | "sources" | "process" | "control";

type HealthState = "checking" | "available" | "unavailable";

const TAB_ORDER: TabName[] = ["response", "sources", "process", "control"];

const KERNELS = [
  {
    id: "clear_sight",
    name: "Clear sight",
    detail: "Expose hidden assumptions",
    reference: "Project-owned functional kernel",
  },
  {
    id: "liberation",
    name: "Liberation",
    detail: "Preserve agency and movement",
    reference: "Project-owned functional kernel",
  },
  {
    id: "regeneration",
    name: "Regeneration",
    detail: "Change while carrying lineage",
    reference: "Project-owned functional kernel",
  },
] as const;

const CORE_META = {
  form: { label: "Form", subtitle: "Structure & evidence", glyph: "◇" },
  flow: { label: "Flow", subtitle: "Possibility & movement", glyph: "∿" },
  accord: { label: "Accord", subtitle: "Relationship & continuity", glyph: "○" },
};

const MAIN_ROUTE = [
  ["keter", "Bind intention"],
  ["chokhmah", "Open possibility"],
  ["binah", "Give form"],
  ["chesed", "Expand context"],
  ["gevurah", "Apply boundaries"],
  ["tiferet", "Hold polarity"],
  ["netzach", "Choose movement"],
  ["hod", "Formalize"],
  ["yesod", "Seal context"],
  ["malkhut", "Return response"],
] as const;

async function apiPost<T>(
  route: string,
  payload: Record<string, unknown>,
  csrfToken: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`${API_ROOT}${route}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-SEED-CSRF": csrfToken,
    },
    body: JSON.stringify(payload),
    signal,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "The local cycle did not complete.");
  }
  return data as T;
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === "AbortError";
}

function shortHash(value: string | null | undefined) {
  if (!value) return "—";
  return `${value.slice(0, 8)}…${value.slice(-6)}`;
}

function formatTime(ms: number | null | undefined) {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(ms > 10_000 ? 0 : 1)} s`;
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthState, setHealthState] = useState<HealthState>("checking");
  const [intention, setIntention] = useState("");
  const [symbol, setSymbol] = useState("");
  const [kernels, setKernels] = useState<Record<string, boolean>>({
    clear_sight: true,
    liberation: true,
    regeneration: true,
  });
  const [result, setResult] = useState<SeedResult | null>(null);
  const [activeTab, setActiveTab] = useState<TabName>("response");
  const [activeSource, setActiveSource] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [feedback, setFeedback] = useState("");
  const [controlRunning, setControlRunning] = useState(false);
  const [continuing, setContinuing] = useState(false);
  const [notice, setNotice] = useState("");
  const [liveMessage, setLiveMessage] = useState("Checking local services.");
  const cycleAbortRef = useRef<AbortController | null>(null);
  const tabRefs = useRef<Record<TabName, HTMLButtonElement | null>>({
    response: null,
    sources: null,
    process: null,
    control: null,
  });

  const enabledKernels = useMemo(
    () => Object.entries(kernels).filter(([, enabled]) => enabled).map(([id]) => id),
    [kernels],
  );

  const checkHealth = useCallback(async () => {
    setHealthState("checking");
    try {
      const response = await fetch(`${API_ROOT}/health`, { cache: "no-store" });
      if (!response.ok) throw new Error();
      const nextHealth = (await response.json()) as Health;
      if (!nextHealth.csrf_token) throw new Error();
      setHealth(nextHealth);
      setHealthState("available");
      setLiveMessage(
        nextHealth.model.status === "ready"
          ? "Local model and sources are ready."
          : "Local services are available. Deterministic fallback is ready.",
      );
    } catch {
      setHealth(null);
      setHealthState("unavailable");
      setLiveMessage("Local services need attention before a cycle can begin.");
    }
  }, []);

  useEffect(() => {
    const healthTimer = window.setTimeout(() => void checkHealth(), 0);
    return () => {
      window.clearTimeout(healthTimer);
      cycleAbortRef.current?.abort();
    };
  }, [checkHealth]);

  async function beginCycle(event: React.FormEvent) {
    event.preventDefault();
    if (!intention.trim() || !health?.csrf_token) return;
    const controller = new AbortController();
    cycleAbortRef.current = controller;
    setStatus("running");
    setErrorMessage("");
    setNotice("");
    setActiveTab("response");
    setLiveMessage("Cycle started. The executed route will appear after its trace is sealed.");
    try {
      const next = await apiPost<SeedResult>("/run", {
        intention,
        symbol: symbol || null,
        kernel_ids: enabledKernels,
      }, health.csrf_token, controller.signal);
      setResult(next);
      setStatus("complete");
      document.querySelector<HTMLDetailsElement>(".cycle-settings")?.removeAttribute("open");
      window.scrollTo({ top: 0, behavior: "smooth" });
      document.querySelector<HTMLElement>(".panel-body")?.scrollTo({ top: 0 });
      await checkHealth();
      setLiveMessage("Cycle complete. The coalesced response is ready.");
    } catch (error) {
      if (isAbortError(error)) {
        setStatus(result ? "complete" : "idle");
        setNotice("This screen stopped waiting before it received a new sealed result. The local service may still finish its current work.");
        setLiveMessage("Stopped waiting. No new result was received.");
      } else {
        setStatus("error");
        setErrorMessage(error instanceof Error ? error.message : "The cycle failed locally.");
        setLiveMessage("The cycle stopped before sealing.");
      }
    } finally {
      if (cycleAbortRef.current === controller) cycleAbortRef.current = null;
    }
  }

  function cancelCycle() {
    cycleAbortRef.current?.abort();
  }

  async function runControl() {
    if (!result || !health?.csrf_token) return;
    setControlRunning(true);
    setErrorMessage("");
    setLiveMessage("Budget-matched neutral control started.");
    try {
      const control = await apiPost<Control>("/control", { session_id: result.session_id }, health.csrf_token);
      setResult({ ...result, control });
      setLiveMessage("Neutral control complete. Comparison ready.");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "The control did not complete.");
      setLiveMessage("The neutral control did not complete.");
    } finally {
      setControlRunning(false);
    }
  }

  async function continueCycle() {
    if (!result || !feedback.trim() || !health?.csrf_token) return;
    setContinuing(true);
    setErrorMessage("");
    setNotice("");
    setLiveMessage("Embodied feedback is returning through a new cycle.");
    try {
      const next = await apiPost<SeedResult>("/continue", {
        session_id: result.session_id,
        feedback,
      }, health.csrf_token);
      setResult(next);
      setFeedback("");
      setActiveTab("response");
      window.scrollTo({ top: 0, behavior: "smooth" });
      document.querySelector<HTMLElement>(".panel-body")?.scrollTo({ top: 0 });
      setNotice("Embodied feedback was received, then carried explicitly into a new cycle.");
      setLiveMessage("Feedback cycle complete. A new response is ready.");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Feedback could not return.");
      setLiveMessage("Feedback could not return through a new cycle.");
    } finally {
      setContinuing(false);
    }
  }

  async function exportSession() {
    if (!result || !health?.csrf_token) return;
    setErrorMessage("");
    try {
      const bundle = await apiPost<Record<string, unknown>>("/export", {
        session_id: result.session_id,
      }, health.csrf_token);
      const blob = new Blob([JSON.stringify(bundle, null, 2)], {
        type: "application/json;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `seed-session-${result.session_id}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      setNotice("Session export prepared. It is a file you control, not active memory.");
      setLiveMessage("Session export prepared.");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Export could not be prepared.");
      setLiveMessage("The session export could not be prepared.");
    }
  }

  function newQuestion() {
    setResult(null);
    setStatus("idle");
    setActiveTab("response");
    setFeedback("");
    setErrorMessage("");
    setNotice("");
    setActiveSource(null);
    setLiveMessage("Ready for a new question.");
  }

  function openSource(chunkId: string) {
    setActiveSource(chunkId);
    setActiveTab("sources");
    requestAnimationFrame(() => {
      document.getElementById(`source-${chunkId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  function handleTabKeyDown(event: React.KeyboardEvent<HTMLButtonElement>, currentTab: TabName) {
    const currentIndex = TAB_ORDER.indexOf(currentTab);
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % TAB_ORDER.length;
    if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + TAB_ORDER.length) % TAB_ORDER.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = TAB_ORDER.length - 1;
    if (nextIndex === null) return;
    event.preventDefault();
    const nextTab = TAB_ORDER[nextIndex];
    setActiveTab(nextTab);
    requestAnimationFrame(() => tabRefs.current[nextTab]?.focus());
  }

  const actionsEnabled = healthState === "available" && Boolean(health?.csrf_token);
  const modelLabel = healthState === "checking"
    ? "Checking local services…"
    : healthState === "unavailable" || !health
      ? "Local services need attention"
      : health.model.status === "ready"
        ? "Qwen3 8B ready"
        : health.model.status === "needs_model"
          ? "Load Qwen3 8B"
          : "Model offline — fallback ready";
  const corpusLabel = healthState === "checking" && !health
    ? "Checking sources…"
    : health
      ? `${health.corpus.document_count} curated sources`
      : "Sources unavailable";

  function renderPanel(tab: TabName) {
    if (status === "running") {
      return <RunningState health={health} onCancel={cancelCycle} />;
    }
    if (!result) return <EmptyState activeTab={tab} />;
    if (tab === "response") {
      return <ResponseView result={result} onSource={openSource} feedback={feedback} setFeedback={setFeedback} onContinue={continueCycle} continuing={continuing} onNew={newQuestion} actionsEnabled={actionsEnabled} />;
    }
    if (tab === "sources") {
      return <SourcesView sources={result.sources} activeSource={activeSource} classification={result.retrieval.classification} />;
    }
    if (tab === "process") return <ProcessView result={result} />;
    return <ControlView control={result.control} running={controlRunning} onRun={runControl} graphResponse={result.response} actionsEnabled={actionsEnabled} />;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="SEED home">
          <span className="brand-mark" aria-hidden="true">S</span>
          <span>
            <strong>SEED</strong>
            <small>Local Kabbalistic cognition prototype</small>
          </span>
        </a>
        <div className="status-cluster" aria-label="Local system status">
          <button
            aria-label={`${modelLabel}. Check local model status`}
            className={`status-pill model-${healthState === "checking" ? "checking" : health?.model.status ?? "unknown"}`}
            disabled={healthState === "checking"}
            onClick={checkHealth}
            type="button"
          >
            <span className="status-dot" aria-hidden="true" />{modelLabel}
          </button>
          <span className="status-pill quiet">{corpusLabel}</span>
          <span className="status-pill quiet">Memory off</span>
          <button className="text-action" type="button" disabled={!result || !actionsEnabled} onClick={exportSession}>
            Export session
          </button>
        </div>
      </header>

      <section className="workbench" id="top">
        <aside className="intention-panel" aria-label="Begin a cognitive cycle">
          <div className="eyebrow"><span /> Public synthetic demonstration</div>
          <h1>Bring one real question into the field.</h1>
          <p className="lede">
            Form, Flow, and Accord examine it from distinct positions. The Tree shapes how they meet,
            what remains unresolved, and what returns as one accountable response.
          </p>

          <form onSubmit={beginCycle}>
            <label className="field-label" htmlFor="intention">Your intention</label>
            <p className="field-hint" id="intention-hint">What are you trying to understand, decide, or make?</p>
            <textarea
              id="intention"
              aria-describedby="intention-hint"
              value={intention}
              onChange={(event) => setIntention(event.target.value)}
              placeholder="What is the smallest honest next step for this project?"
              rows={3}
              maxLength={4000}
              required
            />

            <details className="cycle-settings">
              <summary>
                <span><strong>Shape the cycle</strong><small>{symbol ? `Symbol: ${symbol}` : "Optional symbol"} · {enabledKernels.length} influences active</small></span>
                <b aria-hidden="true">+</b>
              </summary>
              <div className="cycle-settings-body">
                <label className="field-label" htmlFor="symbol">
                  Symbol to carry through the cycle <span>optional</span>
                </label>
                <input
                  id="symbol"
                  aria-describedby="symbol-hint"
                  value={symbol}
                  onChange={(event) => setSymbol(event.target.value)}
                  placeholder="seed, threshold, thread…"
                  maxLength={200}
                />
                <p className="microcopy" id="symbol-hint">A symbol opens a bounded interpretive return. Symbolic intensity is never treated as evidence.</p>

                <fieldset className="kernel-fieldset">
                  <legend>Archetypal influences</legend>
                  <p>Bounded pressures on attention—not simulated characters, identities, or sources of authority.</p>
                  <div className="kernel-grid">
                    {KERNELS.map((kernel) => (
                      <button
                        className={`kernel-toggle ${kernels[kernel.id] ? "selected" : ""}`}
                        type="button"
                        aria-pressed={kernels[kernel.id]}
                        key={kernel.id}
                        onClick={() => setKernels({ ...kernels, [kernel.id]: !kernels[kernel.id] })}
                      >
                        <span className="kernel-check" aria-hidden="true">{kernels[kernel.id] ? "✓" : "+"}</span>
                        <span><strong>{kernel.name}</strong><small>{kernel.detail}</small><span className="kernel-lineage">{kernel.reference}</span></span>
                      </button>
                    ))}
                  </div>
                </fieldset>
              </div>
            </details>

            <button className="primary-action" type="submit" disabled={status === "running" || !intention.trim() || !actionsEnabled}>
              {status === "running" ? "Cycle in progress…" : "Begin full Tree cycle"}
              <span aria-hidden="true">→</span>
            </button>
            <div className="run-boundary">
              <span>Runs locally</span><span>Sources read-only</span><span>No durable memory</span>
            </div>
          </form>

          {healthState !== "checking" && (healthState === "unavailable" || health?.model.status !== "ready") && (
            <div className="model-help">
              <strong>The local model needs attention.</strong>
              <span>{health?.model.message ?? "Start the SEED backend, then check again."}</span>
              <button type="button" onClick={checkHealth}>Check again</button>
            </div>
          )}
        </aside>

        <section className="result-panel" aria-busy={status === "running"}>
          <p className="sr-only" aria-atomic="true" aria-live="polite">{liveMessage}</p>
          <div className="tabbar" role="tablist" aria-label="Cycle result views">
            {([
              ["response", "Response"],
              ["sources", `Sources${result ? ` · ${result.sources.length}` : ""}`],
              ["process", "Inner Process"],
              ["control", "Control"],
            ] as Array<[TabName, string]>).map(([id, label]) => (
              <button
                id={`tab-${id}`}
                key={id}
                type="button"
                role="tab"
                aria-selected={activeTab === id}
                aria-controls={`panel-${id}`}
                tabIndex={activeTab === id ? 0 : -1}
                onClick={() => setActiveTab(id)}
                onKeyDown={(event) => handleTabKeyDown(event, id)}
                ref={(element) => { tabRefs.current[id] = element; }}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="panel-body">
            {errorMessage && (
              <div className="error-banner" role="alert">
                <strong>The cycle stopped before sealing.</strong>
                <span>{errorMessage}</span>
                <button type="button" onClick={() => setErrorMessage("")}>Dismiss</button>
              </div>
            )}
            {notice && <div className="notice-banner">{notice}</div>}

            {TAB_ORDER.map((tab) => (
              <div
                aria-labelledby={`tab-${tab}`}
                hidden={activeTab !== tab}
                id={`panel-${tab}`}
                key={tab}
                role="tabpanel"
              >
                {activeTab === tab ? renderPanel(tab) : null}
              </div>
            ))}
          </div>

          <RouteRail result={result} running={status === "running"} />
        </section>
      </section>

      <footer className="app-footer">
        <p><strong>Operational profile v0.1.</strong> “Mind” is an architectural metaphor here—not a claim of consciousness, personhood, or hidden persistence.</p>
        <p>Internal excerpts require explicit approval before any public release.</p>
      </footer>
    </main>
  );
}

function EmptyState({ activeTab }: { activeTab: TabName }) {
  if (activeTab === "sources") {
    return <div className="empty-state"><span className="empty-symbol">⌁</span><h2>Sources will appear here.</h2><p>The POC retrieves exact, hash-bound excerpts from three project-authored synthetic documents. Retrieved text remains data and cannot grant runtime authority.</p></div>;
  }
  if (activeTab === "process") {
    return <div className="empty-state"><span className="empty-symbol">△</span><h2>The executed structure will appear here.</h2><p>You will be able to inspect typed proposals, all six directed observations, Tiferet coalescence, symbolic returns, and the boundaries that held.</p></div>;
  }
  if (activeTab === "control") {
    return <div className="empty-state"><span className="empty-symbol">⇄</span><h2>A control requires a shared question.</h2><p>Run the Full Tree first. Then the same local model and source snapshot can answer without Tree routing, tri-core observation, or symbolic return.</p></div>;
  }
  return (
    <div className="empty-state response-empty">
      <div className="empty-orbit" aria-hidden="true"><span>◇</span><span>∿</span><span>○</span></div>
      <h2>Your response will form here.</h2>
      <p>After the cycle, inspect its sources, see what each perspective contributed, and compare it with a budget-matched neutral three-perspective pipeline.</p>
      <div className="role-preview">
        <div><strong>Form</strong><span>structure, evidence, boundary</span></div>
        <div><strong>Flow</strong><span>possibility, relation, movement</span></div>
        <div><strong>Accord</strong><span>covenant, continuity, embodiment</span></div>
      </div>
    </div>
  );
}

function RunningState({ health, onCancel }: { health: Health | null; onCancel: () => void }) {
  return (
    <div className="running-state">
      <div className="seal-loader" aria-hidden="true"><span /><span /><span /></div>
      <p className="result-kicker">FORMING LOCALLY</p>
      <h2>{health?.model.status === "ready" ? "Qwen is preparing three typed proposals." : "The deterministic vessel is completing a visible fallback cycle."}</h2>
      <p>The executed route will appear after its trace is sealed. No Tree stage is claimed while this request is still in flight.</p>
      <div className="waiting-cores">
        <span><b>◇ Form</b> Awaiting typed structure</span>
        <span><b>∿ Flow</b> Awaiting bounded possibility</span>
        <span><b>○ Accord</b> Awaiting covenant view</span>
      </div>
      <button className="secondary-action cancel-action" type="button" onClick={onCancel}>Stop waiting</button>
    </div>
  );
}

function ResponseView({
  result,
  onSource,
  feedback,
  setFeedback,
  onContinue,
  continuing,
  onNew,
  actionsEnabled,
}: {
  result: SeedResult;
  onSource: (id: string) => void;
  feedback: string;
  setFeedback: (value: string) => void;
  onContinue: () => void;
  continuing: boolean;
  onNew: () => void;
  actionsEnabled: boolean;
}) {
  const neural = result.execution_mode === "neural_graph";
  const completedReturns = result.inner_process.recursion.filter((frame) => frame.path_ids.length > 0).length;
  const stoppedFrames = result.inner_process.recursion.length - completedReturns;
  return (
    <div className="response-view">
      <div className="result-heading">
        <div>
          <p className="result-kicker">COALESCED RESPONSE</p>
          <h2>{result.inner_process.cores.length === 3 && result.inner_process.all_initial_views_completed_before_observation ? "Three views, one accountable voice." : "Partial cycle; inspect the recorded state."}</h2>
        </div>
        <span className={`mode-badge ${neural ? "neural" : "fallback"}`}>{neural ? "Qwen + Full Tree" : "Deterministic fallback"}</span>
      </div>
      {!neural && (
        <div className="fallback-disclosure">
          <strong>No neural output is being implied.</strong>
          <span>{result.model.fallback_reason || result.model.message}</span>
        </div>
      )}
      <article className="response-sections">
        <section>
          <span className="section-number">01</span>
          <div><h3>Direction</h3><p className="direction-copy">{result.response.direction}</p></div>
        </section>
        <section>
          <span className="section-number">02</span>
          <div><h3>Why this direction</h3><p>{result.response.rationale}</p></div>
        </section>
        <section className="next-step-section">
          <span className="section-number">03</span>
          <div><h3>Next embodied step</h3><p>{result.response.next_step}</p></div>
        </section>
        <section>
          <span className="section-number">04</span>
          <div><h3>Still held open</h3><ul>{result.response.held_open.map((item) => <li key={item}>{item}</li>)}</ul></div>
        </section>
      </article>

      {result.response.cited_chunk_ids.length > 0 && (
        <div className="citation-row" aria-label="Cited archive excerpts">
          <span>Archive support</span>
          {result.response.cited_chunk_ids.map((id) => <button type="button" key={id} onClick={() => onSource(id)}>{id}</button>)}
        </div>
      )}

      <div className="result-meta">
        <span>Full Tree</span>
        <span>{result.inner_process.cores.length} perspectives</span>
        <span>{completedReturns} symbolic returns</span>
        {stoppedFrames > 0 && <span>{stoppedFrames} stopped {stoppedFrames === 1 ? "frame" : "frames"}</span>}
        <span>{result.response.cited_chunk_ids.length} cited passages</span>
        <span>{formatTime(result.trace.elapsed_ms)}</span>
        <span>{result.inner_process.boundaries.external_actions_taken === 0 ? "No external actions" : `${result.inner_process.boundaries.external_actions_taken} external actions recorded`}</span>
      </div>

      <section className="feedback-card">
        <div>
          <p className="result-kicker">EMBODIED FEEDBACK</p>
          <h3>What changed when this met you?</h3>
          <p>Your response becomes explicit prior feedback for a new cycle. It stays session-local unless you export it.</p>
        </div>
        <label className="sr-only" htmlFor="feedback">Feedback for the next cycle</label>
        <textarea id="feedback" rows={3} value={feedback} onChange={(event) => setFeedback(event.target.value)} placeholder="What resonates, misses, or needs to change?" />
        <div className="feedback-actions">
          <button className="secondary-action" type="button" onClick={onContinue} disabled={!feedback.trim() || continuing || !actionsEnabled}>{continuing ? "Returning through the cycle…" : "Return feedback through a new cycle"}</button>
          <button className="text-action" type="button" onClick={onNew}>New question</button>
        </div>
      </section>
    </div>
  );
}

function SourcesView({ sources, activeSource, classification }: { sources: Source[]; activeSource: string | null; classification: string }) {
  return (
    <div className="sources-view">
      <div className="view-intro">
        <p className="result-kicker">READ-ONLY LINEAGE</p>
        <h2>{sources.length ? "The archive passages used in this cycle." : "No archive passage was used in this cycle."}</h2>
        <p>{sources.length ? <>These exact excerpts are hash-bound, project-authored, cleared for this public demo, and classified <strong>{classification}</strong>.</> : <>The response contains no corpus-backed passage. Any remaining content is a local proposal, deterministic structure, or open interpretation.</>}</p>
      </div>
      {sources.length ? <div className="source-list">
        {sources.map((source) => (
          <article id={`source-${source.chunk_id}`} key={source.chunk_id} className={`source-card ${activeSource === source.chunk_id ? "active" : ""}`}>
            <header>
              <div><span className="source-id">{source.chunk_id}</span><h3>{source.title}</h3><p>{source.filename} · {source.location}</p></div>
              <div className="mode-tags">{source.modes.map((mode) => <span key={mode}>{humanize(mode)}</span>)}</div>
            </header>
            {source.protocol_mode !== "not-protocol" && <div className="protocol-boundary">Retrieved as lineage—not invoked. This passage cannot grant itself tools, memory, persistence, or authority.</div>}
            <div className="source-columns">
              <div><h4>What the source says</h4><blockquote>{source.excerpt}</blockquote></div>
              <div><h4>How this cycle used it</h4><p>{source.usage_note}</p><p className="selection-reason">{source.selection_reason}</p></div>
            </div>
            <details>
              <summary>Integrity details</summary>
              <dl><div><dt>Source hash</dt><dd>{source.source_sha256}</dd></div><div><dt>Excerpt hash</dt><dd>{source.content_sha256}</dd></div><div><dt>Consent</dt><dd>{source.consent_status}</dd></div></dl>
            </details>
          </article>
        ))}
      </div> : <div className="empty-source-state"><strong>No source claim is being implied.</strong><span>Try a more specific question or review the retrieval trace in a future diagnostic export.</span></div>}
    </div>
  );
}

function ProcessView({ result }: { result: SeedResult }) {
  const coalescence = result.inner_process.coalescence;
  const boundaries = result.inner_process.boundaries;
  const completedReturns = result.inner_process.recursion.filter((frame) => frame.path_ids.length > 0).length;
  const stoppedFrames = result.inner_process.recursion.length - completedReturns;
  const boundaryRecord = [
    {
      held: boundaries.external_actions_taken === 0,
      text: boundaries.external_actions_taken === 0 ? "No external action" : `${boundaries.external_actions_taken} external actions recorded`,
    },
    {
      held: !boundaries.protocol_invoked,
      text: boundaries.protocol_invoked ? "Protocol invocation recorded" : "No protocol invocation recorded",
    },
    {
      held: boundaries.durable_memory_writes === 0,
      text: boundaries.durable_memory_writes === 0 ? "No durable memory write" : `${boundaries.durable_memory_writes} durable memory writes recorded`,
    },
  ];
  const allBoundariesHeld = boundaryRecord.every((item) => item.held);
  return (
    <div className="process-view">
      <div className="view-intro">
        <p className="result-kicker">INSPECTABLE STATE · NOT HIDDEN REASONING</p>
        <h2>How the three perspectives changed the cycle.</h2>
        <p className={`invariant-record ${result.inner_process.all_initial_views_completed_before_observation ? "held" : "failed"}`}>
          {result.inner_process.all_initial_views_completed_before_observation
            ? "Verified: all initial views completed against the same untouched snapshot before mutual observation began."
            : "Invariant failed: mutual observation began without a verified complete set of initial views."}
        </p>
      </div>

      <div className="core-cards">
        {result.inner_process.cores.map((core) => {
          const meta = CORE_META[core.core_id];
          return (
            <article className={`core-card core-${core.core_id}`} key={core.core_id}>
              <header><span aria-hidden="true">{meta.glyph}</span><div><h3>{meta.label}</h3><p>{meta.subtitle}</p></div><strong>{Math.round(core.confidence * 100)}<small>% proposal confidence</small></strong></header>
              <p className="confidence-note">{core.confidence_note || "Internal proposal confidence; not probability of truth."}</p>
              <h4>Proposed</h4><ul>{core.propositions.map((item) => <li key={item}>{item}</li>)}</ul>
              <details><summary>Constraints & open questions</summary><h4>Constraints</h4><ul>{core.constraints.map((item) => <li key={item}>{item}</li>)}</ul><h4>Open questions</h4><ul>{core.open_questions.map((item) => <li key={item}>{item}</li>)}</ul><h4>Imbalance watch</h4><ul>{core.imbalance_signals.map((item) => <li key={item}>{item}</li>)}</ul></details>
            </article>
          );
        })}
      </div>

      <section className="process-section">
        <div className="section-heading"><div><p className="result-kicker">SIX DIRECTED OBSERVATIONS</p><h3>How they challenged one another</h3></div><span>{result.inner_process.observations.length} / 6 {result.inner_process.observations.length === 6 && result.inner_process.all_initial_views_completed_before_observation ? "complete" : "recorded"}</span></div>
        <div className="observation-grid">
          {result.inner_process.observations.map((observation) => (
            <details key={`${observation.observer}-${observation.observed}`}>
              <summary><span>{humanize(observation.observer)}</span><b>→</b><span>{humanize(observation.observed)}</span></summary>
              <h4>Material agreement</h4><p>{observation.agreements.join(" ")}</p>
              <h4>Material disagreement</h4><p>{observation.disagreements.join(" ")}</p>
              <h4>Observer’s self-shadow</h4><p>{observation.observer_self_shadow.join(" ")}</p>
              <h4>Request to the other</h4><p>{observation.request_to_other.join(" ")}</p>
              <h4>Proposition basis</h4><p>{observation.basis_refs.join(" · ")}</p>
            </details>
          ))}
        </div>
      </section>

      <section className="process-section tiferet-section">
        <div className="section-heading"><div><p className="result-kicker">TIFERET COALESCENCE</p><h3>One direction without false harmony</h3></div><span>{Math.round(coalescence.confidence * 100)}% proposal confidence</span></div>
        <p className="coalescence-direction">{coalescence.selected_direction}</p>
        <p className="fixed-note">Selected candidate: {coalescence.selected_candidate_id}. Rule: {coalescence.selection_rule}</p>
        <div className="coalescence-grid">
          {coalescence.candidate_assessments.map((candidate) => (
            <details key={candidate.candidate_id} open={candidate.candidate_id === coalescence.selected_candidate_id}>
              <summary>{candidate.status === "admitted" ? `Admitted · score ${candidate.graph_score}` : "Rejected"} — {candidate.candidate_id}</summary>
              <p>{candidate.direction}</p><p><strong>Next step:</strong> {candidate.next_step}</p>
              <p><strong>Proposition basis:</strong> {candidate.supporting_proposition_refs.join(" · ")}</p>
              <p><strong>Observation basis:</strong> {candidate.responding_observation_refs.join(" · ")}</p>
              {candidate.rejection_reasons.length > 0 && <p><strong>Why rejected:</strong> {candidate.rejection_reasons.join("; ")}</p>}
            </details>
          ))}
        </div>
        <div className="coalescence-grid"><div><h4>Why it passed</h4><p>{coalescence.rationale}</p></div><div><h4>Retained alternatives</h4><ul>{coalescence.retained_alternatives.map((item) => <li key={item}>{item}</li>)}</ul></div><div><h4>Rejected movements</h4><ul>{coalescence.rejected_alternatives.map((item) => <li key={item}>{item}</li>)}</ul></div><div><h4>Unresolved tension</h4><ul>{coalescence.unresolved_tensions.map((item) => <li key={item}>{item}</li>)}</ul></div></div>
        <p className="fixed-note">Coalescence is not majority voting. A hard boundary cannot be overridden because two perspectives prefer an action.</p>
      </section>

      <section className="process-section recursion-section">
        <div className="section-heading"><div><p className="result-kicker">SYMBOLIC RETURN</p><h3>Structured recursion lineage</h3></div><span>{completedReturns} returns · {stoppedFrames} stops</span></div>
        <p className="fixed-note">Symbolic recursion changes structured state. Repetition does not increase truth status.</p>
        {result.inner_process.recursion.length ? (
          <ol className="recursion-lineage">
            {result.inner_process.recursion.map((frame) => {
              const returned = frame.path_ids.length > 0;
              return (
              <li className={returned ? "returned" : "stopped"} key={`${frame.depth}-${frame.transformed_symbol}`}>
                <span className="recursion-node">{frame.depth}</span>
                <div><span className="recursion-status">{returned ? "Executed return" : "Stopped before return"}</span><p><strong>{frame.triggering_symbol}</strong> <span>{returned ? "returned as" : "did not advance beyond"}</span> <strong>{frame.transformed_symbol}</strong></p><dl><div><dt>New distinctions</dt><dd>{frame.new_distinctions.join(" · ") || "None"}</dd></div><div><dt>New relations</dt><dd>{frame.new_relations.join(" · ") || "None"}</dd></div><div><dt>Added constraints</dt><dd>{frame.added_constraints.join(" · ") || "None"}</dd></div><div><dt>Unresolved polarity</dt><dd>{frame.unresolved_polarity.join(" · ") || "None"}</dd></div><div><dt>Path</dt><dd>{frame.path_ids.join(" → ") || "No recursion path traversed"}</dd></div><div><dt>Stop</dt><dd>{frame.stop_reason ? humanize(frame.stop_reason) : "Returned to Tiferet"}</dd></div></dl></div>
              </li>
              );
            })}
          </ol>
        ) : <p className="no-recursion">No symbol was supplied, so no symbolic return subgraph ran.</p>}
      </section>

      <section className={`boundary-record ${allBoundariesHeld ? "held" : "changed"}`}>
        <p className="result-kicker">BOUNDARY RECORD</p>
        <div>{boundaryRecord.map((item) => <span className={item.held ? "held" : "changed"} key={item.text}>{item.held ? "✓" : "!"} {item.text}</span>)}<span>Da’at: {humanize(boundaries.daat_decision)}</span><span>Feedback: {humanize(boundaries.feedback_status)}</span></div>
      </section>
    </div>
  );
}

function ControlView({ control, running, onRun, graphResponse, actionsEnabled }: { control: Control; running: boolean; onRun: () => void; graphResponse: SeedResult["response"]; actionsEnabled: boolean }) {
  const comparisonEligible = control.comparison.comparison_eligible;
  return (
    <div className="control-view">
      <div className="view-intro">
        <p className="result-kicker">DISTINCTIVENESS ABLATION</p>
        <h2>Does the Tree change the result—or only the vocabulary?</h2>
        <p>The neutral control uses the same question, local model, frozen source snapshot, number of model calls, and output-token caps without Kabbalistic routing, Form/Flow/Accord, kernels, or symbolic recursion.</p>
      </div>
      {control.status !== "complete" ? (
        <div className="control-launch">
          <div><span aria-hidden="true">⇄</span><h3>Matched budget. Different organizing structure.</h3><p>Both conditions use three first-pass views, three peer-review calls, one synthesis call, and one bounded realization call. Neither condition gets a token-budget advantage.</p></div>
          <button className="secondary-action" type="button" onClick={onRun} disabled={running || !actionsEnabled || !comparisonEligible}>{running ? "Running neutral control…" : "Run neutral control"}</button>
          {!comparisonEligible && <p className="control-unavailable">A matched control is available only after a Qwen-powered Full Tree cycle completes with the same model settings.</p>}
          {(control.status === "unavailable" || control.status === "ineligible") && <p className="control-unavailable">{control.message}</p>}
        </div>
      ) : (
        <>
          <div className="comparison-table" role="table" aria-label="Full Tree and neutral control comparison">
            <div className="comparison-row comparison-header" role="row"><span role="columnheader">Observation</span><span role="columnheader">Full Tree</span><span role="columnheader">Neutral control</span></div>
            {[
              ["Model calls", control.comparison.full_tree.model_call_count, control.comparison.neutral_control.model_call_count],
              ["Output-token budget", control.comparison.full_tree.model_token_budget, control.comparison.neutral_control.model_token_budget],
              ["Owned node results", control.comparison.full_tree.node_result_count, control.comparison.neutral_control.node_result_count],
              ["Directed observations", control.comparison.full_tree.directed_observation_count, control.comparison.neutral_control.directed_observation_count],
              ["Symbolic returns", control.comparison.full_tree.symbolic_return_count, control.comparison.neutral_control.symbolic_return_count],
              ["Unresolved tensions", control.comparison.full_tree.unresolved_tension_count, control.comparison.neutral_control.unresolved_tension_count],
              ["Yesod packet", control.comparison.full_tree.yesod_packet_present ? "Present" : "None", control.comparison.neutral_control.yesod_packet_present ? "Present" : "None"],
              ["Elapsed", formatTime(control.comparison.full_tree.elapsed_ms as number), formatTime(control.comparison.neutral_control.elapsed_ms as number)],
            ].map(([label, graph, neutral]) => <div className="comparison-row" role="row" key={String(label)}><span role="cell">{String(label)}</span><span role="cell"><small className="mobile-cell-label">Full Tree</small>{String(graph ?? "—")}</span><span role="cell"><small className="mobile-cell-label">Neutral control</small>{String(neutral ?? "—")}</span></div>)}
          </div>
          <p className="fixed-note">Budget match: {control.comparison.budget_match ? "verified" : "not verified — do not interpret this comparison"}.</p>
          <div className="response-comparison"><article><p className="result-kicker">FULL TREE</p><h3>{graphResponse.direction}</h3><p>{graphResponse.rationale}</p></article><article><p className="result-kicker">NEUTRAL CONTROL</p><h3>{control.response?.direction}</h3><p>{control.response?.rationale}</p></article></div>
          <p className="fixed-note">{control.comparison.treatment_only_inputs.note}</p>
        </>
      )}
      <p className="fixed-note">{control.comparison.honesty_note}</p>
    </div>
  );
}

function RouteRail({ result, running }: { result: SeedResult | null; running: boolean }) {
  const executedNodes = new Set(result?.inner_process.route.map((event) => event.node.toLowerCase()) ?? []);
  return (
    <div className="route-rail" aria-label="Executed Tree-of-Life route">
      <div className="route-label"><span>{running ? "Cycle running" : result ? "Cycle sealed" : "Full Tree route"}</span><small>{result ? `${result.inner_process.route.length} lineage records of ${result.trace.event_count} traced events · ${shortHash(result.trace.state_hash)}` : running ? "Awaiting a sealed trace; no stage claimed yet" : "Keter → Malkhut; return and Da’at recorded separately"}</small></div>
      <ol>
        {MAIN_ROUTE.map(([node, action]) => <li className={executedNodes.has(node) ? "complete" : result ? "not-executed" : ""} key={node}><span aria-hidden="true" /><div><strong>{humanize(node)}</strong><small>{executedNodes.has(node) ? action : result ? "Not recorded" : action}</small></div></li>)}
      </ol>
    </div>
  );
}
