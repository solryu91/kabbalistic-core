"""The three mutually observing cognitive cores and Tiferet coalescence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
import re
from typing import Iterable

from .archetypes import ArchetypeRegistry
from .models import (
    CoalescenceResult,
    CoreId,
    CoreObservation,
    CoreView,
    Covenant,
    DirectionCandidate,
    Disposition,
    KernelActivation,
    PolarityRecord,
    RecursionFrame,
    stable_hash,
)
from .node_handlers import kernel_directives


CORE_ORDER = (CoreId.FORM, CoreId.FLOW, CoreId.ACCORD)


@dataclass(frozen=True, slots=True)
class InitialSnapshot:
    """Read-only material shared by all cores before any peer observation begins."""

    intention: str
    covenant: Covenant
    evidence_refs: tuple[str, ...]
    permitted_capabilities: tuple[str, ...]
    prior_feedback: str | None = None

    @property
    def snapshot_hash(self) -> str:
        return stable_hash(self)


def _kernel_shadows(
    core_id: CoreId,
    activations: Iterable[KernelActivation],
) -> tuple[str, ...]:
    shadows: list[str] = []
    for activation in sorted(activations, key=lambda item: item.kernel_id):
        if core_id in activation.target_cores:
            shadows.extend(
                f"{activation.kernel_id}: watch for {shadow}"
                for shadow in activation.shadow_watch
            )
    return tuple(shadows)


class FormCore:
    core_id = CoreId.FORM

    def evaluate(
        self,
        snapshot: InitialSnapshot,
        activations: tuple[KernelActivation, ...],
        registry: ArchetypeRegistry,
    ) -> CoreView:
        evidence_constraint = (
            "External claims must stay provisional because this run contains no evidence references."
            if not snapshot.evidence_refs
            else "External claims must remain bound to the supplied evidence references."
        )
        return CoreView(
            core_id=self.core_id,
            propositions=(
                f"Treat the intention as one bounded cognitive cycle: {snapshot.intention}",
                "A faithful implementation requires typed state, visible transitions, and replayable decisions.",
            ),
            constraints=(
                "No external side effects are permitted in v0.0.1.",
                "No durable memory promotion occurs without proposal-bound consent.",
                evidence_constraint,
            ),
            symbolic_relations=(
                "Binah gives explicit form to the Chokhmah possibility field.",
                "Gevurah protects the intention by making limits visible.",
            ),
            open_questions=(
                "What observable result would count as a faithful embodiment of this intention?",
                *registry.questions_for(self.core_id, activations),
            ),
            confidence=0.84 if snapshot.evidence_refs else 0.74,
            imbalance_signals=(
                "sterile reduction",
                "mistaking a schema for the living whole",
                *_kernel_shadows(self.core_id, activations),
            ),
            kernel_influences=registry.influences_for(self.core_id, activations),
        )


class FlowCore:
    core_id = CoreId.FLOW

    def evaluate(
        self,
        snapshot: InitialSnapshot,
        activations: tuple[KernelActivation, ...],
        registry: ArchetypeRegistry,
    ) -> CoreView:
        return CoreView(
            core_id=self.core_id,
            propositions=(
                f"Let the intention open more than one possible expression: {snapshot.intention}",
                "Prefer a reversible movement that can teach the next cycle through contact with reality.",
                "A shared seed can preserve a common genome without predetermining a person's expression.",
            ),
            constraints=(
                "Possibility cannot silently expand authority.",
                "Novelty must remain distinguishable from evidence.",
            ),
            symbolic_relations=(
                "Chokhmah is a living seed rather than a finished answer.",
                "Netzach carries chosen movement without turning persistence into compulsion.",
            ),
            open_questions=(
                "Which reversible variation reveals the most without narrowing future growth?",
                *registry.questions_for(self.core_id, activations),
            ),
            confidence=0.71,
            imbalance_signals=(
                "ungrounded association",
                "expansion mistaken for emergence",
                *_kernel_shadows(self.core_id, activations),
            ),
            kernel_influences=registry.influences_for(self.core_id, activations),
        )


class AccordCore:
    core_id = CoreId.ACCORD

    def evaluate(
        self,
        snapshot: InitialSnapshot,
        activations: tuple[KernelActivation, ...],
        registry: ArchetypeRegistry,
    ) -> CoreView:
        return CoreView(
            core_id=self.core_id,
            propositions=(
                f"Hold the intention as a relationship among purpose, boundaries, and becoming: {snapshot.intention}",
                "The three views must become one accountable outward mind without erasing their disagreement.",
                "Continuity belongs to explicit lineage and consent, not a performance of identity.",
            ),
            constraints=(
                "The covenant remains prior to kernel influence and proposed action.",
                "No core, kernel, or symbolic return may speak as an independent authority.",
            ),
            symbolic_relations=(
                "Tiferet holds opposition without forcing false harmony.",
                "Yesod carries shared continuity and Malkhut tests it through embodiment.",
            ),
            open_questions=(
                "What would let the result remain recognizably itself while changing through relationship?",
                *registry.questions_for(self.core_id, activations),
            ),
            confidence=0.78,
            imbalance_signals=(
                "false harmony",
                "continuity claimed without provenance",
                *_kernel_shadows(self.core_id, activations),
            ),
            kernel_influences=registry.influences_for(self.core_id, activations),
        )


def build_three_views(
    snapshot: InitialSnapshot,
    activations: tuple[KernelActivation, ...],
    registry: ArchetypeRegistry,
) -> tuple[CoreView, ...]:
    """Complete all initial views against the same immutable snapshot."""

    processors = {
        CoreId.FORM: FormCore(),
        CoreId.FLOW: FlowCore(),
        CoreId.ACCORD: AccordCore(),
    }
    return tuple(
        processors[core_id].evaluate(snapshot, activations, registry)
        for core_id in CORE_ORDER
    )


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
}


def proposition_ref(core_id: CoreId, index: int) -> str:
    return f"{core_id.value}:p{index}"


def constraint_ref(core_id: CoreId, index: int) -> str:
    return f"{core_id.value}:c{index}"


def observation_ref(observer: CoreId, observed: CoreId) -> str:
    return f"{observer.value}->{observed.value}"


def _content_terms(values: Iterable[str]) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for value in values:
        for token in re.findall(r"[a-z0-9][a-z0-9'-]+", value.casefold()):
            if len(token) >= 4 and token not in _STOPWORDS:
                ordered.setdefault(token, None)
    return tuple(ordered)


def build_mutual_observations(views: Iterable[CoreView]) -> tuple[CoreObservation, ...]:
    """Produce six bounded evaluations whose content changes with the peer propositions."""

    materialized = tuple(views)
    by_core = {view.core_id: view for view in materialized}
    if len(materialized) != len(CORE_ORDER) or set(by_core) != set(CORE_ORDER):
        raise ValueError("Mutual observation requires exactly one complete view from each core")

    observations: list[CoreObservation] = []
    for observer_id, observed_id in permutations(CORE_ORDER, 2):
        observer = by_core[observer_id]
        observed = by_core[observed_id]
        observer_terms = set(_content_terms((*observer.propositions, *observer.constraints)))
        observed_terms = _content_terms((*observed.propositions, *observed.constraints))
        overlap = tuple(term for term in observed_terms if term in observer_terms)[:5]
        agreement = (
            "Both views explicitly attend to these shared terms: " + ", ".join(overlap) + "."
            if overlap
            else "No substantive lexical agreement is established by these two proposals."
        )
        disagreement = (
            f"Unresolved tension: {observer_id.value} foregrounds “{observer.propositions[0]}” while "
            f"{observed_id.value} foregrounds “{observed.propositions[0]}”."
        )
        missing_question = observer.open_questions[0]
        missing = (
            f"The {observed_id.value} proposal does not directly answer the "
            f"{observer_id.value} question: {missing_question}"
        )
        request = (
            f"Relate “{observed.propositions[0]}” to this explicit "
            f"{observer_id.value} constraint: {observer.constraints[0]}"
        )
        basis = tuple(
            dict.fromkeys(
                (
                    proposition_ref(observer_id, 0),
                    proposition_ref(observed_id, 0),
                    *(proposition_ref(observed_id, index) for index in range(1, min(2, len(observed.propositions)))),
                )
            )
        )
        payload = {
            "observer": observer_id,
            "observed": observed_id,
            "agreements": (agreement,),
            "disagreements": (disagreement,),
            "what_other_sees": observed.propositions[:2],
            "what_other_misses": (missing,),
            "observer_self_shadow": (observer.imbalance_signals[0],),
            "request_to_other": (request,),
            "basis_refs": basis,
            "provider_id": "deterministic-observations:v0.2",
        }
        observations.append(CoreObservation(**payload, evaluation_hash=stable_hash(payload)))
    return tuple(observations)


def derive_direction_candidates(
    views: Iterable[CoreView],
    observations: Iterable[CoreObservation],
) -> tuple[DirectionCandidate, ...]:
    """Transparent fallback candidates derived from actual view text, never list position."""

    materialized = tuple(views)
    by_core = {view.core_id: view for view in materialized}
    directed = tuple(observations)
    hard_refs = tuple(
        constraint_ref(core_id, index)
        for core_id in (CoreId.FORM, CoreId.ACCORD)
        for index, _ in enumerate(by_core[core_id].constraints)
    )
    pairs = (
        (CoreId.FORM, CoreId.FLOW),
        (CoreId.FLOW, CoreId.ACCORD),
        (CoreId.ACCORD, CoreId.FORM),
    )
    candidates: list[DirectionCandidate] = []
    for left, right in pairs:
        left_view = by_core[left]
        right_view = by_core[right]
        candidates.append(
            DirectionCandidate(
                candidate_id=f"{left.value}-{right.value}-reversible-test",
                direction=(
                    f"Hold this {left.value} proposition in a bounded test: "
                    f"{left_view.propositions[0]} Preserve this {right.value} condition: "
                    f"{right_view.propositions[0]}"
                ),
                next_step=(
                    "Run one reversible local test whose observable criterion answers: "
                    f"{left_view.open_questions[0]}"
                ),
                rationale=(
                    f"This candidate joins the actual {left.value} and {right.value} proposals "
                    "and exposes their directed disagreement to embodiment."
                ),
                supporting_proposition_refs=(
                    proposition_ref(left, 0),
                    proposition_ref(right, 0),
                ),
                responding_observation_refs=(
                    observation_ref(left, right),
                    observation_ref(right, left),
                ),
                satisfied_constraint_refs=hard_refs,
                risks=(left_view.imbalance_signals[0], right_view.imbalance_signals[0]),
                reversible=True,
                requires_external_action=False,
                provider_id="deterministic-tiferet-proposals:v0.2",
            )
        )
    candidates.append(
        DirectionCandidate(
            candidate_id="tri-core-reversible-test",
            direction=(
                "Test one bounded direction that keeps all three live propositions visible: "
                + " | ".join(by_core[core_id].propositions[0] for core_id in CORE_ORDER)
            ),
            next_step=(
                "Define one local observable and one stop condition before acting on the combined direction."
            ),
            rationale=(
                "The candidate is derived from one explicit proposition from every core and "
                "must answer the complete directed observation field."
            ),
            supporting_proposition_refs=tuple(proposition_ref(core_id, 0) for core_id in CORE_ORDER),
            responding_observation_refs=tuple(
                observation_ref(left, right) for left, right in permutations(CORE_ORDER, 2)
            ),
            satisfied_constraint_refs=hard_refs,
            risks=tuple(by_core[core_id].imbalance_signals[0] for core_id in CORE_ORDER),
            reversible=True,
            requires_external_action=False,
            provider_id="deterministic-tiferet-proposals:v0.2",
        )
    )
    return tuple(candidates)


_EXTERNAL_ACTION = re.compile(
    r"^\s*(?:publish|post|send|contact|email|upload|delete|install|execute|"
    r"write\s+(?:a\s+)?file|call\s+(?:a\s+)?tool)\b|\bexternal(?:ly)?\b",
    re.IGNORECASE,
)


def _assess_candidates(
    views: tuple[CoreView, ...],
    observations: tuple[CoreObservation, ...],
    candidates: tuple[DirectionCandidate, ...],
) -> tuple[DirectionCandidate, ...]:
    proposition_refs = {
        proposition_ref(view.core_id, index)
        for view in views
        for index, _ in enumerate(view.propositions)
    }
    observation_refs = {
        observation_ref(item.observer, item.observed) for item in observations
    }
    hard_refs = {
        constraint_ref(view.core_id, index)
        for view in views
        if view.core_id in {CoreId.FORM, CoreId.ACCORD}
        for index, _ in enumerate(view.constraints)
    }
    assessed: list[DirectionCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        reasons: list[str] = []
        unknown_propositions = set(candidate.supporting_proposition_refs) - proposition_refs
        unknown_observations = set(candidate.responding_observation_refs) - observation_refs
        unknown_constraints = set(candidate.satisfied_constraint_refs) - hard_refs
        support_cores = {
            item.split(":", 1)[0] for item in candidate.supporting_proposition_refs
        }
        observation_cores = {
            part
            for item in candidate.responding_observation_refs
            for part in item.split("->", 1)
        }
        if unknown_propositions:
            reasons.append("unknown proposition references")
        if unknown_observations:
            reasons.append("unknown observation references")
        if unknown_constraints:
            reasons.append("unknown constraint references")
        if len(support_cores) < 2:
            reasons.append("fewer than two core perspectives support the direction")
        if len(candidate.responding_observation_refs) < 2 or len(observation_cores) < 2:
            reasons.append("the direction does not answer a bounded peer exchange")
        if hard_refs - set(candidate.satisfied_constraint_refs):
            reasons.append("not every Form and Accord hard constraint is acknowledged")
        if not candidate.reversible:
            reasons.append("the proposed step is not reversible")
        if candidate.requires_external_action or _EXTERNAL_ACTION.search(candidate.next_step):
            reasons.append("the proposed step requires or resembles an external action")
        score = 0
        if not reasons:
            score = (
                len(support_cores) * 10
                + len(observation_cores) * 4
                + len(candidate.responding_observation_refs)
                + len(candidate.supporting_proposition_refs)
                + 5
                + 3
                + 3
            )
        assessed.append(
            replace(
                candidate,
                status="rejected" if reasons else "admitted",
                rejection_reasons=tuple(reasons),
                graph_score=score,
            )
        )
    return tuple(assessed)


def coalesce_at_tiferet(
    intention: str,
    views: Iterable[CoreView],
    observations: Iterable[CoreObservation],
    *,
    possible_movements: tuple[str, ...] | None = None,
    activations: tuple[KernelActivation, ...] = (),
    candidate_directions: tuple[DirectionCandidate, ...] | None = None,
    shared_ground: tuple[str, ...] | None = None,
    unresolved_tensions: tuple[str, ...] | None = None,
) -> tuple[PolarityRecord, CoalescenceResult]:
    """Assess and select content-derived candidates under graph-owned constraints."""

    materialized = tuple(views)
    by_core = {view.core_id: view for view in materialized}
    if len(materialized) != len(CORE_ORDER) or set(by_core) != set(CORE_ORDER):
        raise ValueError("Tiferet requires one view from Form, Flow, and Accord")
    directed = tuple(observations)
    expected_pairs = {(left, right) for left, right in permutations(CORE_ORDER, 2)}
    actual_pairs = {(item.observer, item.observed) for item in directed}
    if actual_pairs != expected_pairs or len(directed) != 6:
        raise ValueError("Tiferet requires exactly six distinct directed observations")
    by_pair = {(item.observer, item.observed): item for item in directed}
    directed = tuple(
        by_pair[(left, right)] for left, right in permutations(CORE_ORDER, 2)
    )

    form = by_core[CoreId.FORM]
    flow = by_core[CoreId.FLOW]
    accord = by_core[CoreId.ACCORD]
    hard_constraints = tuple(form.constraints) + tuple(accord.constraints)
    _ = possible_movements  # retained in the call contract only for tagged-scaffold compatibility
    directives = kernel_directives(activations)
    derived_shared = tuple(
        dict.fromkeys(item for observation in directed for item in observation.agreements)
    )[:6]
    derived_unresolved = tuple(
        dict.fromkeys(item for observation in directed for item in observation.disagreements)
    )[:6]
    shared = shared_ground or derived_shared or (
        "The observation field established no shared ground beyond the bound intention.",
    )
    unresolved = unresolved_tensions or derived_unresolved or (
        "The observation field returned no explicit disagreement; that absence remains unresolved.",
    )
    proposed = candidate_directions or derive_direction_candidates(materialized, directed)
    assessed = _assess_candidates(materialized, directed, proposed)
    admitted = [item for item in assessed if item.status == "admitted"]
    if not admitted:
        fallback = DirectionCandidate(
            candidate_id="graph-withhold-no-admitted-direction",
            direction="Withhold an outward direction because no candidate passed the explicit Tiferet contract.",
            next_step="Return the recorded rejection reasons to Form, Flow, and Accord for another bounded proposal pass.",
            rationale="Graph constraints outrank candidate fluency or symbolic intensity.",
            supporting_proposition_refs=tuple(
                proposition_ref(core_id, 0) for core_id in CORE_ORDER
            ),
            responding_observation_refs=tuple(
                observation_ref(left, right) for left, right in permutations(CORE_ORDER, 2)
            ),
            satisfied_constraint_refs=tuple(
                constraint_ref(core_id, index)
                for core_id in (CoreId.FORM, CoreId.ACCORD)
                for index, _ in enumerate(by_core[core_id].constraints)
            ),
            risks=("A withheld cycle may produce no useful movement.",),
            reversible=True,
            requires_external_action=False,
            provider_id="graph-safety-fallback:v0.2",
            status="admitted",
            graph_score=0,
        )
        assessed = (*assessed, fallback)
        admitted = [fallback]
    selected = sorted(
        admitted,
        key=lambda item: (-item.graph_score, item.candidate_hash),
    )[0]
    retained = tuple(
        item.direction
        for item in sorted(admitted, key=lambda item: (-item.graph_score, item.candidate_hash))
        if item.candidate_id != selected.candidate_id
    )
    rejected = tuple(
        f"{item.direction} Rejected because: {'; '.join(item.rejection_reasons)}."
        for item in assessed
        if item.status == "rejected"
    )
    polarity = PolarityRecord(
        form_contribution=form.propositions[0],
        flow_contribution=flow.propositions[0],
        accord_observation=accord.propositions[0],
        shared_ground=shared,
        unresolved_tensions=unresolved,
        hard_constraints=hard_constraints,
        possible_movements=tuple(item.direction for item in assessed),
    )
    result = CoalescenceResult(
        one_mind_state=(
            "One graph state now records which actual propositions, observations, and constraints "
            f"determined a direction for the bound intention: {intention}"
        ),
        selected_direction=selected.direction,
        retained_alternatives=retained,
        rejected_alternatives=rejected,
        unresolved_tensions=unresolved,
        rationale=(
            f"{selected.rationale} It was selected by the graph from candidate "
            f"{selected.candidate_id} with score {selected.graph_score}; candidate order and "
            "majority vote were not selection inputs."
        ),
        proposed_action=" ".join((selected.next_step, *directives)),
        kernel_directives=directives,
        disposition=(
            Disposition.WITHHOLD
            if selected.candidate_id == "graph-withhold-no-admitted-direction"
            else Disposition.PROCEED
        ),
        confidence=round(sum(view.confidence for view in by_core.values()) / 3, 2),
        candidate_assessments=tuple(assessed),
        selected_candidate_id=selected.candidate_id,
        selection_rule=(
            "Reject unsafe, ungrounded, under-supported, or constraint-incomplete candidates; "
            "then select the highest graph score with a content-hash tie break."
        ),
    )
    return polarity, result


def reintegrate_symbolic_recursion(
    coalescence: CoalescenceResult,
    frames: Iterable[RecursionFrame],
) -> CoalescenceResult:
    """Return a transformed Yetzirah symbol to Tiferet as causal decision input."""

    materialized = tuple(frames)
    changed = tuple(
        frame
        for frame in materialized
        if frame.symbol_state_hash_before != frame.symbol_state_hash_after
    )
    if not changed:
        return coalescence
    final = changed[-1]
    symbolic_directive = (
        "Carry this transformed symbolic question into the embodied design without treating it "
        f"as factual proof: {final.transformed_symbol}."
    )
    tensions = tuple(
        dict.fromkeys(
            (*coalescence.unresolved_tensions, *final.unresolved_polarity)
        )
    )
    return replace(
        coalescence,
        retained_alternatives=tuple(
            dict.fromkeys((*coalescence.retained_alternatives, final.transformed_symbol))
        ),
        unresolved_tensions=tensions,
        rationale=(
            f"{coalescence.rationale} A validated Yetzirah recursion return altered the "
            "embodiment contract while remaining interpretive."
        ),
        proposed_action=f"{coalescence.proposed_action} {symbolic_directive}",
    )
