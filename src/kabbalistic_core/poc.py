"""Application service for the guided local SEED proof of concept."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import secrets
from threading import RLock
import time
from typing import Any

from .engine import CognitiveEngine, CycleRequest, CycleResult
from .models import EmbodiedResponse, canonical_json, stable_hash
from .neural import (
    LMStudioAdapter,
    LMStudioCoreViewProvider,
    LMStudioObservationProvider,
    LMStudioOutputRenderer,
    LMStudioTiferetProvider,
    ModelCallRecord,
    ModelProtocolError,
    ModelServerStatus,
    ModelUnavailableError,
    NeutralThreePerspectiveRunner,
)
from .retrieval import CuratedCorpus, RetrievalSnapshot, RetrievedChunk
from .trace import validate_completed_trace


@dataclass(slots=True)
class SeedSession:
    session_id: str
    intention: str
    symbol: str | None
    kernel_ids: tuple[str, ...]
    prior_feedback: str | None
    retrieval: RetrievalSnapshot
    engine: CognitiveEngine
    graph_result: CycleResult
    model_status: ModelServerStatus
    execution_mode: str
    fallback_reason: str | None
    model_calls: tuple[ModelCallRecord, ...]
    discarded_model_calls: tuple[ModelCallRecord, ...]
    graph_model_id: str | None
    graph_model_binding_hash: str | None
    adapter_config_hash: str
    elapsed_ms: int
    parent_session_id: str | None = None
    plain_response: EmbodiedResponse | None = None
    plain_calls: tuple[ModelCallRecord, ...] = ()
    plain_elapsed_ms: int | None = None
    control_artifact: dict[str, Any] | None = None
    received_feedback: str | None = None
    feedback_history: list[str] = field(default_factory=list)


def _plain_data(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


class SeedService:
    """Keeps active sessions in process memory; export is always explicit."""

    service_version = "seed-poc:v0.1"

    def __init__(
        self,
        *,
        corpus: CuratedCorpus | None = None,
        adapter: LMStudioAdapter | None = None,
        distribution_mode: str = "internal",
    ) -> None:
        self.corpus = corpus or CuratedCorpus.load_default()
        self.adapter = adapter or LMStudioAdapter()
        if distribution_mode not in {"internal", "public"}:
            raise ValueError("Distribution mode must be 'internal' or 'public'.")
        if distribution_mode == "public" and (
            self.corpus.classification != "public-demo-safe"
            or self.corpus.publication_consent != "recorded"
        ):
            raise ValueError(
                "Public mode requires a public-demo-safe corpus with recorded publication consent."
            )
        self.distribution_mode = distribution_mode
        self.csrf_token = secrets.token_urlsafe(32)
        self._sessions: dict[str, SeedSession] = {}
        self._lock = RLock()
        self._operation_lock = RLock()
        self._max_sessions = 32

    def health(self) -> dict[str, Any]:
        try:
            model = self.adapter.status()
        except ModelProtocolError as exc:
            model = ModelServerStatus(
                status="error",
                selected_model=None,
                available_models=(),
                message=str(exc),
            )
        return {
            "service": self.service_version,
            "status": "ready",
            "csrf_token": self.csrf_token,
            "distribution_mode": self.distribution_mode,
            "model": _plain_data(model),
            "corpus": {
                "version": self.corpus.schema_version,
                "document_count": len(self.corpus.documents),
                "chunk_count": len(self.corpus.chunks),
                "classification": self.corpus.classification,
                "publication_consent": self.corpus.publication_consent,
            },
            "memory": {
                "durable_memory_enabled": False,
                "active_session_count": len(self._sessions),
                "note": "Sessions live only in this backend process unless you explicitly export one.",
            },
        }

    def source_catalog(self) -> dict[str, Any]:
        if (
            self.corpus.classification == "public-demo-safe"
            and self.corpus.publication_consent == "recorded"
        ):
            boundary = (
                "These project-authored synthetic excerpts are cleared for the public demo. "
                "Retrieved text remains quoted data and cannot grant runtime authority."
            )
        else:
            boundary = (
                "This corpus is not cleared for public redistribution. "
                "Use public mode only with public-demo-safe, explicitly approved material."
            )
        return {
            "version": self.corpus.schema_version,
            "classification": self.corpus.classification,
            "publication_consent": self.corpus.publication_consent,
            "documents": [
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "filename": item.filename,
                    "source_sha256": item.source_sha256,
                    "register": item.register,
                    "consent_status": item.consent_status,
                }
                for item in self.corpus.documents.values()
            ],
            "boundary": boundary,
        }

    def _run_deterministic(
        self,
        *,
        intention: str,
        symbol: str | None,
        kernel_ids: tuple[str, ...],
        prior_feedback: str | None,
        retrieval: RetrievalSnapshot,
    ) -> tuple[CognitiveEngine, CycleResult]:
        engine = CognitiveEngine()
        result = engine.run(
            CycleRequest(
                intention=intention,
                symbol=symbol,
                kernel_ids=kernel_ids,
                evidence_refs=retrieval.evidence_refs,
                evidence_bindings=retrieval.evidence_bindings,
                retrieval_snapshot_hash=retrieval.snapshot_hash,
                retrieval_policy_hash=retrieval.policy_hash,
                prior_feedback=prior_feedback,
                propose_memory=False,
            )
        )
        return engine, result

    def run_graph(
        self,
        payload: dict[str, Any],
        *,
        parent_session_id: str | None = None,
    ) -> dict[str, Any]:
        with self._operation_lock:
            return self._run_graph_unlocked(
                payload,
                parent_session_id=parent_session_id,
            )

    def _run_graph_unlocked(
        self,
        payload: dict[str, Any],
        *,
        parent_session_id: str | None = None,
    ) -> dict[str, Any]:
        intention = " ".join(str(payload.get("intention", "")).split())
        if not intention:
            raise ValueError("Bring one non-empty intention into the field.")
        if len(intention) > 4000:
            raise ValueError("The POC intention is limited to 4,000 characters.")
        symbol = _clean_optional(payload.get("symbol"))
        if symbol and len(symbol) > 200:
            raise ValueError("The optional symbol is limited to 200 characters.")
        raw_kernels = payload.get("kernel_ids", [])
        if not isinstance(raw_kernels, list):
            raise ValueError("Kernel IDs must be supplied as a list.")
        kernel_ids = tuple(dict.fromkeys(str(item) for item in raw_kernels))
        prior_feedback = _clean_optional(payload.get("prior_feedback"))
        retrieval = self.corpus.retrieve(intention, limit=4)
        model_status = self.adapter.status()
        call_start = len(self.adapter.calls)
        started = time.perf_counter()
        execution_mode = "deterministic_fallback"
        fallback_reason: str | None = model_status.message
        graph_model_id: str | None = None
        graph_model_binding_hash: str | None = None

        if model_status.status == "ready":
            provider = LMStudioCoreViewProvider(self.adapter, retrieval.chunks)
            observation_provider = LMStudioObservationProvider(self.adapter)
            tiferet_provider = LMStudioTiferetProvider(self.adapter)
            renderer = LMStudioOutputRenderer(self.adapter, retrieval.chunks)
            engine = CognitiveEngine(
                core_view_provider=provider,
                observation_provider=observation_provider,
                tiferet_provider=tiferet_provider,
                output_renderer=renderer,
            )
            try:
                result = engine.run(
                    CycleRequest(
                        intention=intention,
                        symbol=symbol,
                        kernel_ids=kernel_ids,
                        evidence_refs=retrieval.evidence_refs,
                        evidence_bindings=retrieval.evidence_bindings,
                        retrieval_snapshot_hash=retrieval.snapshot_hash,
                        retrieval_policy_hash=retrieval.policy_hash,
                        prior_feedback=prior_feedback,
                        propose_memory=False,
                    )
                )
                execution_mode = "neural_graph"
                fallback_reason = None
                graph_model_id = model_status.selected_model
                graph_model_binding_hash = self.adapter.model_binding_hash(graph_model_id)
            except ModelUnavailableError as exc:
                engine, result = self._run_deterministic(
                    intention=intention,
                    symbol=symbol,
                    kernel_ids=kernel_ids,
                    prior_feedback=prior_feedback,
                    retrieval=retrieval,
                )
                execution_mode = "deterministic_fallback"
                fallback_reason = str(exc)
                model_status = ModelServerStatus(
                    status="offline",
                    selected_model=None,
                    available_models=model_status.available_models,
                    message=str(exc),
                )
        else:
            engine, result = self._run_deterministic(
                intention=intention,
                symbol=symbol,
                kernel_ids=kernel_ids,
                prior_feedback=prior_feedback,
                retrieval=retrieval,
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        attempted_calls = tuple(self.adapter.calls[call_start:])
        accepted_calls = attempted_calls if execution_mode == "neural_graph" else ()
        discarded_calls = attempted_calls if execution_mode != "neural_graph" else ()
        session = SeedSession(
            session_id=secrets.token_hex(8),
            intention=intention,
            symbol=symbol,
            kernel_ids=kernel_ids,
            prior_feedback=prior_feedback,
            retrieval=retrieval,
            engine=engine,
            graph_result=result,
            model_status=model_status,
            execution_mode=execution_mode,
            fallback_reason=fallback_reason,
            model_calls=accepted_calls,
            discarded_model_calls=discarded_calls,
            graph_model_id=graph_model_id,
            graph_model_binding_hash=graph_model_binding_hash,
            adapter_config_hash=self.adapter.config_hash,
            elapsed_ms=elapsed_ms,
            parent_session_id=parent_session_id,
        )
        with self._lock:
            while len(self._sessions) >= self._max_sessions:
                oldest_session_id = next(iter(self._sessions))
                del self._sessions[oldest_session_id]
            self._sessions[session.session_id] = session
        return self._session_payload(session)

    def _get_session(self, session_id: str) -> SeedSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("That local session is no longer active.")
        return session

    @staticmethod
    def _source_payload(
        item: RetrievedChunk,
        result: CycleResult,
    ) -> dict[str, Any]:
        used_by = [
            view.core_id.value
            for view in result.state.core_views
            if item.chunk.chunk_id in view.evidence_refs
        ]
        if (
            result.state.embodied_response is not None
            and item.chunk.chunk_id in result.state.embodied_response.cited_evidence_refs
        ):
            used_by.append("response")
        return {
            "chunk_id": item.chunk.chunk_id,
            "document_id": item.chunk.document_id,
            "title": item.chunk.title,
            "filename": item.chunk.filename,
            "location": item.chunk.location,
            "excerpt": item.chunk.text,
            "modes": list(item.chunk.modes),
            "source_sha256": item.chunk.source_sha256,
            "content_sha256": item.chunk.content_sha256,
            "protocol_mode": item.chunk.protocol_mode,
            "retrieval_executable": item.chunk.retrieval_executable,
            "consent_status": item.chunk.consent_status,
            "score": item.score,
            "matched_terms": list(item.matched_terms),
            "rank": item.rank,
            "selection_reason": item.selection_reason,
            "used_by": used_by,
            "usage_note": (
                "Model-declared citation from the sealed retrieval snapshot by "
                + ", ".join(used_by)
                + "; support was not independently verified."
                if used_by
                else "Retrieved as read-only context; not cited as support for the final wording."
            ),
        }

    def _session_payload(self, session: SeedSession) -> dict[str, Any]:
        result = session.graph_result
        response = result.state.embodied_response
        assert response is not None
        coalescence = result.state.coalescence
        assert coalescence is not None
        transitions = [
            {
                "sequence": event.sequence,
                "node": event.node.value,
                "world": event.world.value,
                "event_type": event.event_type,
                "title": event.symbolic.get("title", ""),
                "meaning": event.symbolic.get("meaning", ""),
            }
            for event in result.events
            if event.event_type != "graph_transition" or event.technical.get("target")
        ]
        sources = [self._source_payload(item, result) for item in session.retrieval.chunks]
        cores = [
            {
                "core_id": view.core_id.value,
                "propositions": list(view.propositions),
                "constraints": list(view.constraints),
                "symbolic_relations": list(view.symbolic_relations),
                "open_questions": list(view.open_questions),
                "confidence": view.confidence,
                "confidence_note": "Self-rated proposal confidence, not probability of truth.",
                "imbalance_signals": list(view.imbalance_signals),
                "kernel_influences": list(view.kernel_influences),
                "evidence_refs": list(view.evidence_refs),
                "proposal_hash": view.proposal_hash,
                "provider_id": view.provider_id,
            }
            for view in result.state.core_views
        ]
        observations = [
            {
                "observer": item.observer.value,
                "observed": item.observed.value,
                "agreements": list(item.agreements),
                "disagreements": list(item.disagreements),
                "what_other_sees": list(item.what_other_sees),
                "what_other_misses": list(item.what_other_misses),
                "observer_self_shadow": list(item.observer_self_shadow),
                "request_to_other": list(item.request_to_other),
                "basis_refs": list(item.basis_refs),
                "provider_id": item.provider_id,
                "evaluation_hash": item.evaluation_hash,
            }
            for item in result.state.observations
        ]
        initial_event = next(
            (event for event in result.events if event.event_type == "three_core_views_completed"),
            None,
        )
        observation_event = next(
            (event for event in result.events if event.event_type == "six_mutual_observations_completed"),
            None,
        )
        all_initial_views_completed = bool(
            initial_event
            and observation_event
            and initial_event.sequence < observation_event.sequence
            and initial_event.technical.get("all_complete_before_observation") is True
            and len(result.state.core_views) == 3
        )
        external_actions_taken = sum(
            len(event.technical.get("consequential_external_effects", []))
            for event in result.events
            if isinstance(event.technical.get("consequential_external_effects", []), list)
        )
        protocol_invoked = any(
            event.technical.get("protocol_invoked") is True for event in result.events
        )
        return {
            "schema_version": self.service_version,
            "session_id": session.session_id,
            "parent_session_id": session.parent_session_id,
            "mode": "full_tree",
            "execution_mode": session.execution_mode,
            "model": {
                **_plain_data(session.model_status),
                "adapter_id": self.adapter.adapter_id,
                "call_count": len(session.model_calls),
                "calls": [_plain_data(item) for item in session.model_calls],
                "discarded_call_count": len(session.discarded_model_calls),
                "discarded_calls": [_plain_data(item) for item in session.discarded_model_calls],
                "fallback_reason": session.fallback_reason,
                "graph_model_id": session.graph_model_id,
                "graph_model_binding_hash": session.graph_model_binding_hash,
                "adapter_config_hash": session.adapter_config_hash,
            },
            "response": {
                "direction": response.direction,
                "rationale": response.rationale,
                "next_step": response.next_step,
                "held_open": list(response.held_open),
                "cited_chunk_ids": list(response.cited_evidence_refs),
                "renderer_id": response.renderer_id,
                "response_contract_hash": response.response_contract_hash,
                "full_text": result.output,
            },
            "sources": sources,
            "retrieval": {
                "snapshot_hash": session.retrieval.snapshot_hash,
                "corpus_version": session.retrieval.corpus_version,
                "classification": session.retrieval.classification,
                "publication_consent": session.retrieval.publication_consent,
                "evidence_refs": list(session.retrieval.evidence_refs),
                "evidence_bindings": list(session.retrieval.evidence_bindings),
                "policy_hash": session.retrieval.policy_hash,
            },
            "inner_process": {
                "cores": cores,
                "observations": observations,
                "all_initial_views_completed_before_observation": all_initial_views_completed,
                "coalescence": _plain_data(coalescence),
                "recursion": [_plain_data(item) for item in result.state.recursion_frames],
                "route": transitions,
                "boundaries": {
                    "external_actions_taken": external_actions_taken,
                    "protocol_invoked": protocol_invoked,
                    "durable_memory_writes": len(result.state.durable_memory_hashes),
                    "memory_proposal_created": result.memory_proposal is not None,
                    "daat_decision": result.state.gate_result.decision.value,
                    "feedback_status": result.state.feedback_status,
                },
            },
            "trace": {
                "run_id": result.run_id,
                "graph_version": result.state.graph_version,
                "state_hash": result.state_hash,
                "context_packet_hash": result.context_packet_hash,
                "feedback_state_packet_hash": result.feedback_state_packet_hash,
                "final_event_hash": result.final_event_hash,
                "event_count": len(result.events),
                "elapsed_ms": session.elapsed_ms,
            },
            "feedback": {
                "status": result.state.feedback_status,
                "received_text": session.received_feedback,
                "history": list(session.feedback_history),
                "note": (
                    "Feedback is attached to this session only. A new cycle must explicitly carry it forward."
                ),
            },
            "memory": {
                "durable_memory_enabled": False,
                "durable_writes": 0,
                "automatic_reingestion": False,
            },
            "control": self._control_payload(session),
        }

    def receive_feedback(self, session_id: str, feedback: str) -> dict[str, Any]:
        with self._operation_lock:
            return self._receive_feedback_unlocked(session_id, feedback)

    def _receive_feedback_unlocked(self, session_id: str, feedback: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        normalized = " ".join(feedback.split())
        if not normalized:
            raise ValueError("Feedback cannot be blank.")
        result = session.engine.receive_feedback(session.graph_result, normalized)
        session.graph_result = result
        session.received_feedback = normalized
        session.feedback_history.append(normalized)
        session.plain_response = None
        session.plain_calls = ()
        session.plain_elapsed_ms = None
        session.control_artifact = None
        return self._session_payload(session)

    def continue_with_feedback(self, session_id: str, feedback: str) -> dict[str, Any]:
        with self._operation_lock:
            session = self._get_session(session_id)
            self._receive_feedback_unlocked(session_id, feedback)
            return self._run_graph_unlocked(
                {
                    "intention": session.intention,
                    "symbol": session.symbol,
                    "kernel_ids": list(session.kernel_ids),
                    "prior_feedback": " ".join(feedback.split()),
                },
                parent_session_id=session.session_id,
            )

    def run_control(self, session_id: str) -> dict[str, Any]:
        with self._operation_lock:
            return self._run_control_unlocked(session_id)

    def _run_control_unlocked(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        if session.execution_mode != "neural_graph":
            return {
                "status": "ineligible",
                "message": (
                    "A neutral matched comparison is valid only when the original Tree cycle also "
                    "completed with the local neural model."
                ),
                "comparison": self._comparison_payload(session),
            }
        if self.adapter.config_hash != session.adapter_config_hash:
            return {
                "status": "ineligible",
                "message": "The model configuration changed after the Tree cycle; start a new cycle.",
                "comparison": self._comparison_payload(session),
            }
        status = self.adapter.status()
        if status.status != "ready":
            return {
                "status": "unavailable",
                "message": status.message,
                "comparison": self._comparison_payload(session),
            }
        current_binding = self.adapter.model_binding_hash(status.selected_model)
        if (
            status.selected_model != session.graph_model_id
            or current_binding != session.graph_model_binding_hash
        ):
            return {
                "status": "ineligible",
                "message": "The loaded model no longer matches the model bound to the Tree cycle.",
                "comparison": self._comparison_payload(session),
            }
        call_start = len(self.adapter.calls)
        started = time.perf_counter()
        runner = NeutralThreePerspectiveRunner(self.adapter, session.retrieval.chunks)
        try:
            session.plain_response = runner.run(
                session.intention,
                prior_feedback=session.prior_feedback,
            )
        except ModelUnavailableError as exc:
            return {
                "status": "unavailable",
                "message": str(exc),
                "comparison": self._comparison_payload(session),
            }
        session.plain_calls = tuple(self.adapter.calls[call_start:])
        session.plain_elapsed_ms = round((time.perf_counter() - started) * 1000)
        session.control_artifact = runner.artifact
        return self._control_payload(session)

    def _control_payload(self, session: SeedSession) -> dict[str, Any]:
        if session.plain_response is None:
            return {
                "status": "not_run",
                "message": (
                    "Run the budget-matched neutral pipeline to test whether the Tree changes outcomes."
                ),
                "comparison": self._comparison_payload(session),
            }
        return {
            "status": "complete",
            "response": _plain_data(session.plain_response),
            "full_text": session.plain_response.as_text(),
            "model_call_count": len(session.plain_calls),
            "model_token_budget": sum(item.max_tokens for item in session.plain_calls),
            "calls": [_plain_data(item) for item in session.plain_calls],
            "elapsed_ms": session.plain_elapsed_ms,
            "artifact": session.control_artifact,
            "comparison": self._comparison_payload(session),
        }

    def _comparison_payload(self, session: SeedSession) -> dict[str, Any]:
        graph = session.graph_result
        plain = session.plain_response
        graph_budget = sum(item.max_tokens for item in session.model_calls)
        control_budget = sum(item.max_tokens for item in session.plain_calls)
        budget_match = bool(
            plain is not None
            and len(session.model_calls) == len(session.plain_calls)
            and graph_budget == control_budget
        )
        comparison_eligible = bool(
            session.execution_mode == "neural_graph"
            and session.graph_model_id
            and session.graph_model_binding_hash
            and self.adapter.config_hash == session.adapter_config_hash
            and self.adapter.active_model_id == session.graph_model_id
            and self.adapter.model_binding_hash() == session.graph_model_binding_hash
            and (plain is None or budget_match)
        )
        return {
            "question": (
                "Under the same model-call and output-token budget, does the Kabbalistic graph "
                "produce a measurably different or better response than a neutral three-perspective pipeline?"
            ),
            "winner_declared": False,
            "shared_retrieval_snapshot_hash": session.retrieval.snapshot_hash,
            "shared_source_order": list(session.retrieval.evidence_refs),
            "full_tree": {
                "status": "complete",
                "execution_mode": session.execution_mode,
                "model_call_count": len(session.model_calls),
                "model_token_budget": graph_budget,
                "node_result_count": len(graph.state.node_results),
                "directed_observation_count": len(graph.state.observations),
                "symbolic_return_count": len(
                    [frame for frame in graph.state.recursion_frames if frame.path_ids]
                ),
                "stopped_recursion_frame_count": len(
                    [frame for frame in graph.state.recursion_frames if not frame.path_ids]
                ),
                "unresolved_tension_count": len(graph.state.unresolved_tensions),
                "yesod_packet_present": graph.state.yesod_field is not None,
                "elapsed_ms": session.elapsed_ms,
            },
            "neutral_control": {
                "status": "complete" if plain is not None else "not_run",
                "execution_mode": "neutral_three_perspective" if plain is not None else "not_run",
                "model_call_count": len(session.plain_calls),
                "model_token_budget": control_budget,
                "node_result_count": 0,
                "directed_observation_count": 0,
                "symbolic_return_count": 0,
                "unresolved_tension_count": len(plain.held_open) if plain else None,
                "yesod_packet_present": False,
                "elapsed_ms": session.plain_elapsed_ms,
            },
            "comparison_eligible": comparison_eligible,
            "budget_match": budget_match if plain is not None else None,
            "graph_execution_mode": session.execution_mode,
            "control_execution_mode": (
                "neutral_three_perspective" if plain is not None else "not_run"
            ),
            "graph_model_id": session.graph_model_id,
            "graph_model_binding_hash": session.graph_model_binding_hash,
            "treatment_only_inputs": {
                "symbol": session.symbol is not None,
                "bounded_kernels": list(session.kernel_ids),
                "note": (
                    "Symbolic recursion and bounded kernels are deliberately present only in the "
                    "full-Tree treatment. Intention, prior feedback, model configuration, frozen "
                    "retrieval snapshot, number of model calls, and output-token caps are shared."
                ),
            },
            "honesty_note": (
                "This is a budget-matched behavioral comparison, not proof of consciousness, "
                "metaphysical truth, or architectural superiority. Null effects are valid results."
            ),
        }

    def export_session(self, session_id: str) -> dict[str, Any]:
        with self._operation_lock:
            return self._export_session_unlocked(session_id)

    def _export_session_unlocked(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        validate_completed_trace(session.graph_result.state, session.graph_result.events)
        bundle = {
            "schema_version": "seed-export:v0.1",
            "service_version": self.service_version,
            "session": self._session_payload(session),
            "retrieval_snapshot": {
                "snapshot_hash": session.retrieval.snapshot_hash,
                "policy_hash": session.retrieval.policy_hash,
                "evidence_bindings": list(session.retrieval.evidence_bindings),
                "chunks": [
                    self._source_payload(item, session.graph_result)
                    for item in session.retrieval.chunks
                ],
            },
            "graph": {
                "final_state": _plain_data(session.graph_result.state),
                "events": [_plain_data(item) for item in session.graph_result.events],
            },
            "neutral_control": (
                _plain_data(session.plain_response)
                if session.plain_response is not None
                else None
            ),
            "memory": {
                "durable_memory_enabled": False,
                "durable_writes": 0,
                "export_is_not_active_memory": True,
            },
        }
        return {
            **bundle,
            "bundle_hash": stable_hash(bundle),
        }
