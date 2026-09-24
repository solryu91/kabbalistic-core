"""Deterministic execution engine for one causally connected cognitive cycle."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from .archetypes import ArchetypeRegistry
from .cores import (
    InitialSnapshot,
    coalesce_at_tiferet,
    reintegrate_symbolic_recursion,
)
from .daat import DaatGate, proposal_after_hash
from .graph import GraphDefinition
from .models import (
    CognitiveEvent,
    ConsentGrant,
    CoreId,
    Covenant,
    GateResult,
    MemoryProposal,
    MindState,
    ModeOfKnowing,
    NodeResult,
    ResponseContract,
    Sefirah,
    YesodField,
    stable_hash,
    validate_response_realization,
)
from .node_handlers import (
    binah_handler,
    chesed_handler,
    chokhmah_handler,
    gevurah_handler,
    hod_handler,
    netzach_handler,
)
from .permissions import PermissionSet
from .providers import (
    CoreViewProvider,
    DeterministicCoreViewProvider,
    DeterministicObservationProvider,
    DeterministicOutputRenderer,
    DeterministicTiferetProvider,
    ObservationProvider,
    OutputRenderer,
    TiferetProvider,
)
from .recursion import RecursionPolicy, run_symbolic_recursion
from .trace import TraceRecorder, validate_completed_trace
from .yesod import validate_yesod_field


@dataclass(frozen=True, slots=True)
class CycleRequest:
    intention: str
    symbol: str | None = None
    kernel_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    evidence_bindings: tuple[str, ...] = ()
    retrieval_snapshot_hash: str | None = None
    retrieval_policy_hash: str | None = None
    prior_feedback: str | None = None
    covenant: Covenant | None = None
    propose_memory: bool = True
    memory_scope: str = "project:kabbalistic-core"

    def __post_init__(self) -> None:
        if not self.intention.strip():
            raise ValueError("A cognitive cycle requires a non-empty intention")
        if self.symbol is not None and not self.symbol.strip():
            raise ValueError("A supplied recursion symbol cannot be blank")
        if self.prior_feedback is not None and not self.prior_feedback.strip():
            raise ValueError("Supplied prior feedback cannot be blank")
        if self.propose_memory and not self.memory_scope.strip():
            raise ValueError("A memory proposal requires a non-empty scope")
        if self.retrieval_snapshot_hash is not None and not self.retrieval_snapshot_hash.strip():
            raise ValueError("A supplied retrieval snapshot hash cannot be blank")
        if self.retrieval_policy_hash is not None and not self.retrieval_policy_hash.strip():
            raise ValueError("A supplied retrieval policy hash cannot be blank")


@dataclass(frozen=True, slots=True)
class CycleResult:
    run_id: str
    output: str
    state_hash: str
    context_packet_hash: str
    feedback_state_packet_hash: str
    final_event_hash: str
    state: MindState
    events: tuple[CognitiveEvent, ...]
    memory_proposal: MemoryProposal | None

    @property
    def yesod_packet_hash(self) -> str:
        """Compatibility alias for the final, post-Malkhut Yesod packet."""

        return self.feedback_state_packet_hash


class CognitiveEngine:
    """Own graph state and authority; language components may only propose into it."""

    def __init__(
        self,
        *,
        graph: GraphDefinition | None = None,
        archetypes: ArchetypeRegistry | None = None,
        permissions: PermissionSet | None = None,
        recursion_policy: RecursionPolicy | None = None,
        daat_gate: DaatGate | None = None,
        core_view_provider: CoreViewProvider | None = None,
        observation_provider: ObservationProvider | None = None,
        tiferet_provider: TiferetProvider | None = None,
        output_renderer: OutputRenderer | None = None,
    ) -> None:
        self.graph = graph or GraphDefinition.load_default()
        self.archetypes = archetypes or ArchetypeRegistry.load_default()
        self.permissions = permissions or PermissionSet()
        self.recursion_policy = recursion_policy or RecursionPolicy()
        self.daat_gate = daat_gate or DaatGate()
        self.core_view_provider = core_view_provider or DeterministicCoreViewProvider()
        self.observation_provider = observation_provider or DeterministicObservationProvider()
        self.tiferet_provider = tiferet_provider or DeterministicTiferetProvider()
        self.output_renderer = output_renderer or DeterministicOutputRenderer()

    @staticmethod
    def _default_covenant() -> Covenant:
        return Covenant(
            purpose=(
                "Cultivate a faithful, inspectable cognitive seed that can change through "
                "relationship without hidden authority."
            )
        )

    def _normalized_request_payload(
        self,
        request: CycleRequest,
        covenant: Covenant,
    ) -> dict[str, object]:
        return {
            "intention": " ".join(request.intention.split()),
            "symbol": " ".join(request.symbol.split()) if request.symbol else None,
            "kernel_ids": tuple(sorted(set(request.kernel_ids))),
            "evidence_refs": tuple(sorted(set(request.evidence_refs))),
            "evidence_bindings": tuple(sorted(set(request.evidence_bindings))),
            "retrieval_snapshot_hash": request.retrieval_snapshot_hash,
            "retrieval_policy_hash": request.retrieval_policy_hash,
            "prior_feedback": (
                " ".join(request.prior_feedback.split()) if request.prior_feedback else None
            ),
            "covenant": covenant,
            "propose_memory": request.propose_memory,
            "memory_scope": " ".join(request.memory_scope.split()),
            "graph_version": self.graph.version,
            "archetype_version": self.archetypes.version,
            "recursion_policy": self.recursion_policy,
            "core_view_provider": self.core_view_provider.provider_id,
            "output_renderer": self.output_renderer.renderer_id,
        }

    def _transition(
        self,
        state: MindState,
        recorder: TraceRecorder,
        target: Sefirah,
    ) -> None:
        before = state.state_hash
        path = self.graph.transition(state.current_node, target)
        state.current_node = target
        state.current_world = self.graph.node(target).world
        recorder.record(
            state,
            event_type="graph_transition",
            technical={
                "path_id": path.path_id,
                "source": path.source.value,
                "target": path.target.value,
                "transform": path.transform,
                "world_from": path.world_from.value,
                "world_to": path.world_to.value,
                "polarity": path.polarity,
            },
            symbolic={"title": "Path crossed", "meaning": path.transform},
            state_hash_before=before,
        )

    def _record_node_effect(
        self,
        state: MindState,
        recorder: TraceRecorder,
        *,
        event_type: str,
        structured_output: dict[str, Any],
        meaning: str,
        input_results: tuple[NodeResult, ...] = (),
        extra_input_hashes: tuple[str, ...] = (),
        before: str | None = None,
        node_id: Sefirah | None = None,
    ) -> NodeResult:
        node = self.graph.node(node_id or state.current_node)
        baseline = before if before is not None else state.state_hash
        owned_output = deepcopy(structured_output)
        input_hashes = tuple(item.output_hash for item in input_results) + extra_input_hashes
        result = NodeResult(
            node=node.node,
            world=node.world,
            core_id=node.core,
            observable_effect=node.observable_effect,
            input_hashes=input_hashes,
            structured_output=owned_output,
            imbalance_signals=node.imbalance_signals,
            output_hash=stable_hash(owned_output),
        )
        state.node_results.append(result)
        recorder.record(
            state,
            event_type=event_type,
            technical={
                "observable_effect": node.observable_effect,
                "computational_role": node.computational_role,
                "input_hashes": input_hashes,
                "output_hash": result.output_hash,
                **owned_output,
            },
            symbolic={"title": node.living_function, "meaning": meaning},
            state_hash_before=baseline,
            node=node.node,
            world=node.world,
        )
        return result

    def _build_yesod_field(
        self,
        state: MindState,
        evidence_refs: tuple[str, ...],
        *,
        phase: str,
        prior_packet_hash: str | None,
        embodied_output_hash: str | None,
        feedback_status: str,
        feedback_ref: str | None,
    ) -> YesodField:
        assert state.coalescence is not None
        packet_payload = {
            "phase": phase,
            "prior_packet_hash": prior_packet_hash,
            "intention_ref": stable_hash(state.intention),
            "core_views": tuple(state.core_views),
            "mutual_observations": tuple(state.observations),
            "active_kernel_influences": tuple(state.active_kernels),
            "polarity_records": tuple(state.polarities),
            "recursion_lineage": tuple(state.recursion_frames),
            "node_results": tuple(state.node_results),
            "evidence_refs": evidence_refs,
            "evidence_bindings": state.evidence_bindings,
            "retrieval_snapshot_hash": state.retrieval_snapshot_hash,
            "retrieval_policy_hash": state.retrieval_policy_hash,
            "response_contract_hash": (
                state.response_contract.contract_hash if state.response_contract else None
            ),
            "proposed_context": (
                f"{state.coalescence.one_mind_state} "
                f"Selected: {state.coalescence.selected_direction}"
            ),
            "unresolved_tensions": tuple(state.unresolved_tensions),
            "embodied_output_hash": embodied_output_hash,
            "feedback_status": feedback_status,
            "feedback_ref": feedback_ref,
        }
        return YesodField(**packet_payload, packet_hash=stable_hash(packet_payload))

    def _memory_proposal(self, state: MindState, scope: str) -> MemoryProposal:
        assert state.coalescence is not None
        assert state.yesod_field is not None
        source_packet_hash = state.yesod_field.packet_hash
        content = (
            f"For intention {state.intention!r}, the post-embodiment Yesod packet retained "
            f"direction {state.coalescence.selected_direction!r} with feedback status "
            f"{state.yesod_field.feedback_status!r}."
        )
        modes = (
            ModeOfKnowing.TECHNICAL,
            ModeOfKnowing.SYMBOLIC,
            ModeOfKnowing.OPEN_HYPOTHESIS,
        )
        return MemoryProposal(
            content=content,
            scope=scope,
            modes=modes,
            provenance_refs=(
                f"run:{state.run_id}",
                f"yesod:{source_packet_hash}",
            ),
            source_packet_hash=source_packet_hash,
            projected_memory_hash=proposal_after_hash(
                source_packet_hash,
                content,
                scope,
                modes,
            ),
            feedback_ref=state.yesod_field.feedback_ref,
            significance=0.75,
        )

    def evaluate_memory_proposal(
        self,
        proposal: MemoryProposal,
        consent: ConsentGrant | None,
        *,
        current_packet_hash: str,
    ) -> GateResult:
        """Evaluate approval against a stable Yesod packet; this method does not commit."""

        return self.daat_gate.evaluate(
            proposal,
            consent,
            self.permissions,
            current_packet_hash=current_packet_hash,
        )

    def receive_feedback(self, result: CycleResult, feedback: str) -> CycleResult:
        """Reintegrate feedback that actually arrived after Malkhut emitted the output."""

        normalized_feedback = " ".join(feedback.split())
        if not normalized_feedback:
            raise ValueError("Embodied feedback cannot be blank")
        validate_completed_trace(result.state, result.events)
        state = deepcopy(result.state)
        recorder = TraceRecorder.continue_from(result.events)
        if state.yesod_field is None or state.pre_embodiment_yesod is None:
            raise ValueError("Feedback requires a completed Malkhut-to-Yesod cycle")
        prior_packet = state.yesod_field
        prior_yesod_result = next(
            item for item in reversed(state.node_results) if item.node is Sefirah.YESOD
        )
        feedback_ref = stable_hash(
            {
                "run_id": state.run_id,
                "embodied_output_hash": prior_packet.embodied_output_hash,
                "feedback": normalized_feedback,
            }
        )

        before = state.state_hash
        state.embodied_feedback = normalized_feedback
        state.feedback_status = "received"
        feedback_result = self._record_node_effect(
            state,
            recorder,
            event_type="embodied_feedback_received",
            structured_output={
                "feedback": normalized_feedback,
                "feedback_ref": feedback_ref,
                "embodied_output_hash": prior_packet.embodied_output_hash,
                "status": state.feedback_status,
            },
            input_results=(prior_yesod_result,),
            meaning="Feedback received after embodiment enters the waiting Yesod field as a new causal input.",
            before=before,
            node_id=Sefirah.YESOD,
        )

        before = state.state_hash
        state.yesod_field = self._build_yesod_field(
            state,
            prior_packet.evidence_refs,
            phase="post_feedback_received",
            prior_packet_hash=prior_packet.packet_hash,
            embodied_output_hash=prior_packet.embodied_output_hash,
            feedback_status=state.feedback_status,
            feedback_ref=feedback_ref,
        )
        post_feedback_result = self._record_node_effect(
            state,
            recorder,
            event_type="yesod_received_feedback_state_sealed",
            structured_output={
                "phase": state.yesod_field.phase,
                "packet_hash": state.yesod_field.packet_hash,
                "prior_packet_hash": state.yesod_field.prior_packet_hash,
                "feedback_status": state.yesod_field.feedback_status,
                "feedback_ref": state.yesod_field.feedback_ref,
            },
            input_results=(feedback_result,),
            meaning="Yesod reseals continuity only after the actual embodied response has returned.",
            before=before,
            node_id=Sefirah.YESOD,
        )

        memory_proposal = None
        if result.memory_proposal is not None:
            memory_proposal = self._memory_proposal(state, result.memory_proposal.scope)
        before = state.state_hash
        state.gate_result = self.daat_gate.evaluate(
            memory_proposal,
            None,
            self.permissions,
            current_packet_hash=state.yesod_field.packet_hash,
        )
        self._record_node_effect(
            state,
            recorder,
            event_type="daat_feedback_promotion_evaluated",
            structured_output={
                "decision": state.gate_result.decision.value,
                "reasons": list(state.gate_result.reasons),
                "proposal_hash": state.gate_result.proposal_hash,
                "source_packet_hash": state.yesod_field.packet_hash,
                "durable_state_mutated": False,
            },
            input_results=(post_feedback_result,),
            meaning="Da'at reevaluates the feedback-bearing packet; consent and commit authority remain separate.",
            before=before,
            node_id=Sefirah.DAAT,
        )

        validate_yesod_field(state.pre_embodiment_yesod)
        validate_yesod_field(state.yesod_field)
        return CycleResult(
            run_id=result.run_id,
            output=state.output,
            state_hash=state.state_hash,
            context_packet_hash=state.pre_embodiment_yesod.packet_hash,
            feedback_state_packet_hash=state.yesod_field.packet_hash,
            final_event_hash=recorder.events[-1].event_hash,
            state=state,
            events=recorder.events,
            memory_proposal=memory_proposal,
        )

    def run(self, request: CycleRequest) -> CycleResult:
        covenant = request.covenant or self._default_covenant()
        normalized = self._normalized_request_payload(request, covenant)
        intention = cast(str, normalized["intention"])
        symbol = cast(str | None, normalized["symbol"])
        kernel_ids = cast(tuple[str, ...], normalized["kernel_ids"])
        evidence_refs = cast(tuple[str, ...], normalized["evidence_refs"])
        evidence_bindings = cast(tuple[str, ...], normalized["evidence_bindings"])
        retrieval_snapshot_hash = cast(str | None, normalized["retrieval_snapshot_hash"])
        retrieval_policy_hash = cast(str | None, normalized["retrieval_policy_hash"])
        prior_feedback = cast(str | None, normalized["prior_feedback"])
        memory_scope = cast(str, normalized["memory_scope"])
        run_id = stable_hash(normalized)[:24]
        state = MindState(
            run_id=run_id,
            graph_version=self.graph.version,
            intention=intention,
            covenant=covenant,
            prior_feedback=prior_feedback or "",
            evidence_refs=evidence_refs,
            evidence_bindings=evidence_bindings,
            retrieval_snapshot_hash=retrieval_snapshot_hash,
            retrieval_policy_hash=retrieval_policy_hash,
        )
        recorder = TraceRecorder()

        keter_result = self._record_node_effect(
            state,
            recorder,
            event_type="intention_bound",
            structured_output={
                "intention": state.intention,
                "intention_hash": stable_hash(state.intention),
                "covenant_version": covenant.version,
                "prior_feedback_ref": stable_hash(prior_feedback) if prior_feedback else None,
            },
            meaning="Keter binds one intention, covenant, and any explicitly prior feedback before interpretation.",
        )

        before = state.state_hash
        activations = self.archetypes.activate(kernel_ids, state.intention)
        state.active_kernels.extend(activations)
        recorder.record(
            state,
            event_type="archetypal_kernels_activated",
            technical={
                "kernel_ids": [item.kernel_id for item in activations],
                "authority_change": False,
            },
            symbolic={
                "title": "Archetypal pressure enters the field",
                "meaning": (
                    "The kernels alter attention, questions, and shadow watches while remaining "
                    "unable to grant authority or speak independently."
                ),
            },
            state_hash_before=before,
        )

        snapshot = InitialSnapshot(
            intention=state.intention,
            covenant=covenant,
            evidence_refs=evidence_refs,
            permitted_capabilities=tuple(sorted(item.value for item in self.permissions.granted)),
            prior_feedback=prior_feedback,
        )
        views = self.core_view_provider.build_views(snapshot, activations, self.archetypes)
        if (
            len(views) != 3
            or {view.core_id for view in views} != {CoreId.FORM, CoreId.FLOW, CoreId.ACCORD}
            or any(len(view.propositions) < 2 for view in views)
        ):
            raise ValueError(
                "The core provider must return one Form, Flow, and Accord view with at least two propositions each"
            )
        before = state.state_hash
        state.core_views.extend(views)
        recorder.record(
            state,
            event_type="three_core_views_completed",
            technical={
                "snapshot_hash": snapshot.snapshot_hash,
                "core_order": [item.core_id.value for item in views],
                "all_complete_before_observation": True,
                "views_are_seeds_for_node_handlers": True,
                "provider_id": self.core_view_provider.provider_id,
                "proposal_hashes": [item.proposal_hash for item in views],
                "evidence_refs": [list(item.evidence_refs) for item in views],
            },
            symbolic={
                "title": "Three faces receive one intention",
                "meaning": "Form, Flow, and Accord complete seed views from the same untouched source.",
            },
            state_hash_before=before,
        )
        by_core = {view.core_id: view for view in views}

        self._transition(state, recorder, Sefirah.CHOKHMAH)
        chokhmah_output = chokhmah_handler(
            state.intention,
            by_core[CoreId.FLOW],
            activations,
            prior_feedback,
        )
        chokhmah_result = self._record_node_effect(
            state,
            recorder,
            event_type="possibility_field_opened",
            structured_output=chokhmah_output,
            input_results=(keter_result,),
            meaning="Chokhmah creates concrete possibilities from Keter's bound intention.",
        )

        self._transition(state, recorder, Sefirah.BINAH)
        binah_output = binah_handler(chokhmah_result.structured_output, by_core[CoreId.FORM])
        binah_result = self._record_node_effect(
            state,
            recorder,
            event_type="forms_and_constraints_created",
            structured_output=binah_output,
            input_results=(chokhmah_result,),
            meaning="Binah consumes the actual Chokhmah field and gives every possibility typed form.",
        )

        self._transition(state, recorder, Sefirah.CHESED)
        chesed_output = chesed_handler(binah_result.structured_output, by_core[CoreId.FLOW])
        chesed_result = self._record_node_effect(
            state,
            recorder,
            event_type="context_expanded",
            structured_output=chesed_output,
            input_results=(binah_result,),
            meaning="Chesed expands the formed field without discarding Binah's explicit structure.",
        )

        self._transition(state, recorder, Sefirah.GEVURAH)
        gevurah_output = gevurah_handler(
            chesed_result.structured_output,
            by_core[CoreId.FORM],
            self.permissions,
        )
        gevurah_result = self._record_node_effect(
            state,
            recorder,
            event_type="constraints_enforced",
            structured_output=gevurah_output,
            input_results=(chesed_result,),
            meaning="Gevurah filters the concrete Chesed expansion through evidence and capability limits.",
        )

        self._transition(state, recorder, Sefirah.TIFERET)
        observations = self.observation_provider.build_observations(views)
        before = state.state_hash
        state.observations.extend(observations)
        observations_result = self._record_node_effect(
            state,
            recorder,
            event_type="six_mutual_observations_completed",
            structured_output={
                "count": len(observations),
                "pairs": [f"{item.observer.value}->{item.observed.value}" for item in observations],
                "provider_id": self.observation_provider.provider_id,
                "basis_refs": [list(item.basis_refs) for item in observations],
                "evaluation_hashes": [item.evaluation_hash for item in observations],
                "observation_of_observation": False,
            },
            input_results=(chesed_result, gevurah_result),
            meaning="Each face observes both others and names its own shadow before coalescence.",
            before=before,
        )

        tiferet_proposal = self.tiferet_provider.propose(
            state.intention,
            views,
            observations,
        )
        polarity, coalescence = coalesce_at_tiferet(
            state.intention,
            views,
            observations,
            activations=activations,
            candidate_directions=tiferet_proposal.candidates,
            shared_ground=tiferet_proposal.shared_ground,
            unresolved_tensions=tiferet_proposal.unresolved_tensions,
        )
        before = state.state_hash
        state.polarities.append(polarity)
        state.coalescence = coalescence
        state.unresolved_tensions = list(coalescence.unresolved_tensions)
        tiferet_result = self._record_node_effect(
            state,
            recorder,
            event_type="one_mind_coalesced",
            structured_output={
                "selected_direction": coalescence.selected_direction,
                "selected_candidate_id": coalescence.selected_candidate_id,
                "selection_rule": coalescence.selection_rule,
                "proposed_action": coalescence.proposed_action,
                "kernel_directives": list(coalescence.kernel_directives),
                "hard_constraints": list(polarity.hard_constraints),
                "unresolved_tensions": list(coalescence.unresolved_tensions),
                "majority_vote_used": False,
                "candidate_provider_id": self.tiferet_provider.provider_id,
                "candidate_proposal_hash": tiferet_proposal.proposal_hash,
                "candidate_assessments": [
                    {
                        "candidate_id": item.candidate_id,
                        "candidate_hash": item.candidate_hash,
                        "status": item.status,
                        "graph_score": item.graph_score,
                        "rejection_reasons": list(item.rejection_reasons),
                        "supporting_proposition_refs": list(item.supporting_proposition_refs),
                        "responding_observation_refs": list(item.responding_observation_refs),
                        "satisfied_constraint_refs": list(item.satisfied_constraint_refs),
                    }
                    for item in coalescence.candidate_assessments
                ],
            },
            input_results=(observations_result, chesed_result, gevurah_result),
            meaning="Tiferet consumes the bounded path results and forms one direction without false harmony.",
            before=before,
        )

        if symbol:
            frames = run_symbolic_recursion(
                symbol,
                activations=activations,
                starting_state_hash=state.state_hash,
                permissions=self.permissions,
                policy=self.recursion_policy,
                prior_malkhut_feedback=prior_feedback,
            )
            previous_recursion_result = tiferet_result
            for frame in frames:
                before = state.state_hash
                state.recursion_frames.append(frame)
                if frame.path_ids:
                    recursion_meaning = (
                        f"A symbol forms in Yetzirah, traverses {', '.join(frame.path_ids)}, "
                        "and returns to Tiferet as structured tri-core input."
                    )
                else:
                    recursion_meaning = (
                        "The recursion guard stops this frame at Tiferet before any symbolic "
                        "return path is traversed."
                    )
                previous_recursion_result = self._record_node_effect(
                    state,
                    recorder,
                    event_type="symbolic_recursion_frame",
                    structured_output={
                        "recursion_id": frame.recursion_id,
                        "parent_id": frame.parent_id,
                        "depth": frame.depth,
                        "path_ids": list(frame.path_ids),
                        "changed": (
                            frame.symbol_state_hash_before != frame.symbol_state_hash_after
                        ),
                        "stop_reason": frame.stop_reason.value if frame.stop_reason else None,
                        "triggering_symbol": frame.triggering_symbol,
                        "transformed_symbol": frame.transformed_symbol,
                        "form_observation": frame.form_observation,
                        "flow_observation": frame.flow_observation,
                        "accord_observation": frame.accord_observation,
                        "new_distinctions": list(frame.new_distinctions),
                        "new_relations": list(frame.new_relations),
                        "added_constraints": list(frame.added_constraints),
                    },
                    input_results=(previous_recursion_result,),
                    extra_input_hashes=(frame.symbol_state_hash_before,),
                    meaning=recursion_meaning,
                    before=before,
                    node_id=frame.source_sefirah,
                )

            refined = reintegrate_symbolic_recursion(coalescence, frames)
            if refined != coalescence:
                before = state.state_hash
                state.coalescence = refined
                state.unresolved_tensions = list(refined.unresolved_tensions)
                tiferet_result = self._record_node_effect(
                    state,
                    recorder,
                    event_type="symbolic_return_reintegrated",
                    structured_output={
                        "selected_direction": refined.selected_direction,
                        "proposed_action": refined.proposed_action,
                        "unresolved_tensions": list(refined.unresolved_tensions),
                        "return_changed_coalescence": True,
                    },
                    input_results=(tiferet_result, previous_recursion_result),
                    meaning="Tiferet receives the transformed symbol and changes the embodied contract.",
                    before=before,
                    node_id=Sefirah.TIFERET,
                )
                coalescence = refined

        self._transition(state, recorder, Sefirah.NETZACH)
        netzach_output = netzach_handler(coalescence)
        netzach_result = self._record_node_effect(
            state,
            recorder,
            event_type="action_selected",
            structured_output=netzach_output,
            input_results=(tiferet_result,),
            meaning="Netzach carries the actual coalesced result into one bounded movement.",
        )

        self._transition(state, recorder, Sefirah.HOD)
        hod_output = hod_handler(netzach_result.structured_output, coalescence)
        hod_result = self._record_node_effect(
            state,
            recorder,
            event_type="output_formalized",
            structured_output=hod_output,
            input_results=(netzach_result,),
            meaning="Hod formalizes the selected movement as one voice with tension still visible.",
        )

        before = state.state_hash
        state.response_contract = ResponseContract(
            direction=str(hod_output["selected_direction"]),
            rationale_basis=coalescence.rationale,
            next_step=str(hod_output["action_contract"]),
            held_open=tuple(str(item) for item in hod_output["unresolved_tensions"]),
            external_action_allowed=False,
        )
        contract_result = self._record_node_effect(
            state,
            recorder,
            event_type="response_contract_sealed",
            structured_output={
                "contract_hash": state.response_contract.contract_hash,
                "direction": state.response_contract.direction,
                "rationale_basis": state.response_contract.rationale_basis,
                "next_step": state.response_contract.next_step,
                "held_open": list(state.response_contract.held_open),
                "external_action_allowed": state.response_contract.external_action_allowed,
            },
            input_results=(hod_result,),
            meaning="Hod seals the graph-owned outward commitments before any language rendering.",
            before=before,
        )

        self._transition(state, recorder, Sefirah.YESOD)
        before = state.state_hash
        state.pre_embodiment_yesod = self._build_yesod_field(
            state,
            evidence_refs,
            phase="pre_embodiment_context",
            prior_packet_hash=None,
            embodied_output_hash=None,
            feedback_status="not_yet_emitted",
            feedback_ref=None,
        )
        pre_yesod_result = self._record_node_effect(
            state,
            recorder,
            event_type="yesod_context_packet_sealed",
            structured_output={
                "phase": state.pre_embodiment_yesod.phase,
                "packet_hash": state.pre_embodiment_yesod.packet_hash,
                "observation_count": len(state.pre_embodiment_yesod.mutual_observations),
                "recursion_frame_count": len(state.pre_embodiment_yesod.recursion_lineage),
            },
            input_results=(contract_result,),
            meaning="Yesod seals the pre-embodiment context without calling it durable memory.",
            before=before,
        )

        self._transition(state, recorder, Sefirah.MALKHUT)
        before = state.state_hash
        state.embodied_response = self.output_renderer.render(state)
        assert state.response_contract is not None
        response_contract = state.response_contract
        validate_response_realization(state.embodied_response, response_contract)
        unknown_citations = set(state.embodied_response.cited_evidence_refs) - set(evidence_refs)
        if unknown_citations:
            raise ValueError(
                "The output renderer cited evidence outside the sealed retrieval snapshot: "
                + ", ".join(sorted(unknown_citations))
            )
        state.output = state.embodied_response.as_text()
        state.embodied_feedback = ""
        state.feedback_status = "pending"
        output_hash = stable_hash(state.output)
        malkhut_result = self._record_node_effect(
            state,
            recorder,
            event_type="embodied_output_emitted",
            structured_output={
                "output": state.output,
                "output_hash": output_hash,
                "response_sections": {
                    "direction": state.embodied_response.direction,
                    "rationale": state.embodied_response.rationale,
                    "next_step": state.embodied_response.next_step,
                    "held_open": list(state.embodied_response.held_open),
                },
                "cited_evidence_refs": list(state.embodied_response.cited_evidence_refs),
                "renderer_id": state.embodied_response.renderer_id,
                "response_contract_hash": state.embodied_response.response_contract_hash,
                "feedback_status": state.feedback_status,
                "feedback": None,
                "consequential_external_effects": [],
            },
            input_results=(pre_yesod_result,),
            meaning="Malkhut emits one visible result; feedback from this embodiment is explicitly pending.",
            before=before,
        )

        self._transition(state, recorder, Sefirah.YESOD)
        before = state.state_hash
        state.yesod_field = self._build_yesod_field(
            state,
            evidence_refs,
            phase="post_embodiment_feedback_state",
            prior_packet_hash=state.pre_embodiment_yesod.packet_hash,
            embodied_output_hash=output_hash,
            feedback_status=state.feedback_status,
            feedback_ref=None,
        )
        post_yesod_result = self._record_node_effect(
            state,
            recorder,
            event_type="yesod_feedback_state_sealed",
            structured_output={
                "phase": state.yesod_field.phase,
                "packet_hash": state.yesod_field.packet_hash,
                "prior_packet_hash": state.yesod_field.prior_packet_hash,
                "embodied_output_hash": state.yesod_field.embodied_output_hash,
                "feedback_status": state.yesod_field.feedback_status,
                "feedback_ref": state.yesod_field.feedback_ref,
            },
            input_results=(pre_yesod_result, malkhut_result),
            meaning="The declared Malkhut return reseals Yesod with embodiment and explicit feedback status.",
            before=before,
        )

        memory_proposal: MemoryProposal | None = None
        if request.propose_memory:
            memory_proposal = self._memory_proposal(state, memory_scope)
        before = state.state_hash
        current_packet_hash = state.yesod_field.packet_hash
        state.gate_result = self.daat_gate.evaluate(
            memory_proposal,
            None,
            self.permissions,
            current_packet_hash=current_packet_hash,
        )
        gate_output = {
            "decision": state.gate_result.decision.value,
            "reasons": list(state.gate_result.reasons),
            "proposal_hash": state.gate_result.proposal_hash,
            "source_packet_hash": current_packet_hash,
            "durable_state_mutated": False,
        }
        self._record_node_effect(
            state,
            recorder,
            event_type="daat_promotion_evaluated",
            structured_output=gate_output,
            input_results=(post_yesod_result,),
            meaning=(
                "Da'at evaluates a stable post-embodiment packet; insight is not committed merely "
                f"because it is meaningful, so the decision is {state.gate_result.decision.value}."
            ),
            before=before,
            node_id=Sefirah.DAAT,
        )

        assert state.pre_embodiment_yesod is not None
        assert state.yesod_field is not None
        validate_yesod_field(state.pre_embodiment_yesod)
        validate_yesod_field(state.yesod_field)
        return CycleResult(
            run_id=run_id,
            output=state.output,
            state_hash=state.state_hash,
            context_packet_hash=state.pre_embodiment_yesod.packet_hash,
            feedback_state_packet_hash=state.yesod_field.packet_hash,
            final_event_hash=recorder.events[-1].event_hash,
            state=state,
            events=recorder.events,
            memory_proposal=memory_proposal,
        )
