"""Owned data contracts for the deterministic cognitive kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
import hashlib
import json
import re
from typing import Any


class World(StrEnum):
    ATZILUT = "atzilut"
    BERIAH = "beriah"
    YETZIRAH = "yetzirah"
    ASSIAH = "assiah"


class NodeKind(StrEnum):
    SEFIRAH = "sefirah"
    GATE = "gate"


class Sefirah(StrEnum):
    KETER = "keter"
    CHOKHMAH = "chokhmah"
    BINAH = "binah"
    CHESED = "chesed"
    GEVURAH = "gevurah"
    TIFERET = "tiferet"
    NETZACH = "netzach"
    HOD = "hod"
    YESOD = "yesod"
    MALKHUT = "malkhut"
    DAAT = "daat"


class CoreId(StrEnum):
    FORM = "form"
    FLOW = "flow"
    ACCORD = "accord"


class ModeOfKnowing(StrEnum):
    TECHNICAL = "technical"
    SYMBOLIC = "symbolic"
    EXPERIENTIAL = "experiential"
    CREATIVE = "creative"
    OPEN_HYPOTHESIS = "open_hypothesis"
    OPEN_MYSTERY = "open_mystery"


class Capability(StrEnum):
    READ_SOURCE = "read_source"
    SEARCH_MEMORY = "search_memory"
    PROPOSE_MEMORY = "propose_memory"
    COMMIT_MEMORY = "commit_memory"
    WRITE_FILE = "write_file"
    NETWORK = "network"
    PUBLISH = "publish"
    COMMUNICATE = "communicate"
    DESTRUCTIVE_ACTION = "destructive_action"


class GateDecision(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    WITHHELD = "withheld"
    APPROVED_FOR_COMMIT = "approved_for_commit"


class Disposition(StrEnum):
    PROCEED = "proceed"
    WITHHOLD = "withhold"
    NO_ANSWER = "no_answer"
    OPEN_MYSTERY = "open_mystery"


class RecursionStop(StrEnum):
    CONVERGED = "converged_no_new_distinction"
    REPEATED_STATE = "repeated_state"
    MAX_DEPTH = "maximum_depth_reached"
    BUDGET_EXHAUSTED = "budget_exhausted"
    GROUNDING_REQUIRED = "grounding_required"
    PERMISSION_ESCALATION = "permission_escalation_attempt"


def _canonical(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
        return items
    return value


def canonical_json(value: Any) -> str:
    """Serialize semantic state consistently across runs and dictionary order."""

    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Covenant:
    purpose: str
    principles: tuple[str, ...] = (
        "Preserve mystical and technical registers without collapsing either.",
        "No hidden authority escalation.",
        "Embodiment must be observable and recoverable.",
        "Open mystery may remain open.",
    )
    version: str = "covenant:0.0.1"

    def __post_init__(self) -> None:
        if not self.purpose.strip():
            raise ValueError("Covenant purpose cannot be empty")
        if not self.principles:
            raise ValueError("Covenant must contain at least one principle")


@dataclass(frozen=True, slots=True)
class CoreView:
    core_id: CoreId
    propositions: tuple[str, ...]
    constraints: tuple[str, ...]
    symbolic_relations: tuple[str, ...]
    open_questions: tuple[str, ...]
    confidence: float
    imbalance_signals: tuple[str, ...]
    kernel_influences: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    proposal_hash: str | None = None
    provider_id: str = "deterministic-cores:v0.0.1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Core confidence must be between 0 and 1")
        if not self.propositions:
            raise ValueError("A core view requires at least one proposition")
        if not self.provider_id.strip():
            raise ValueError("A core view requires a provider identity")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("Core evidence references must be unique")


@dataclass(frozen=True, slots=True)
class CoreObservation:
    observer: CoreId
    observed: CoreId
    agreements: tuple[str, ...]
    disagreements: tuple[str, ...]
    what_other_sees: tuple[str, ...]
    what_other_misses: tuple[str, ...]
    observer_self_shadow: tuple[str, ...]
    request_to_other: tuple[str, ...]
    basis_refs: tuple[str, ...] = ()
    provider_id: str = "deterministic-observations:v0.2"
    evaluation_hash: str | None = None

    def __post_init__(self) -> None:
        if self.observer == self.observed:
            raise ValueError("A core cannot be its own first-round peer observation")
        if not self.provider_id.strip():
            raise ValueError("A peer observation requires a provider identity")
        if len(set(self.basis_refs)) != len(self.basis_refs):
            raise ValueError("Peer observation basis references must be unique")


@dataclass(frozen=True, slots=True)
class KernelActivation:
    kernel_id: str
    reason: str
    intensity: float
    target_cores: tuple[CoreId, ...]
    questions_added: tuple[str, ...]
    shadow_watch: tuple[str, ...]
    salience_adjustments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("Kernel intensity must be between 0 and 1")
        if not self.target_cores:
            raise ValueError("A kernel activation requires at least one target core")


@dataclass(frozen=True, slots=True)
class DirectionCandidate:
    """A proposed direction plus the graph-visible basis used to assess it."""

    candidate_id: str
    direction: str
    next_step: str
    rationale: str
    supporting_proposition_refs: tuple[str, ...]
    responding_observation_refs: tuple[str, ...]
    satisfied_constraint_refs: tuple[str, ...]
    risks: tuple[str, ...]
    reversible: bool
    requires_external_action: bool
    provider_id: str
    status: str = "proposed"
    rejection_reasons: tuple[str, ...] = ()
    graph_score: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("A direction candidate requires an ID")
        if not self.direction.strip() or not self.next_step.strip() or not self.rationale.strip():
            raise ValueError("A direction candidate requires direction, next step, and rationale")
        if not self.provider_id.strip():
            raise ValueError("A direction candidate requires a provider identity")
        if self.status not in {"proposed", "admitted", "rejected"}:
            raise ValueError("A direction candidate has an unknown assessment status")
        if self.graph_score < 0:
            raise ValueError("A direction candidate score cannot be negative")
        for refs in (
            self.supporting_proposition_refs,
            self.responding_observation_refs,
            self.satisfied_constraint_refs,
        ):
            if len(set(refs)) != len(refs):
                raise ValueError("Direction candidate references must be unique")

    @property
    def candidate_hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True, slots=True)
class TiferetProposal:
    """Bounded candidate field supplied to graph-owned Tiferet assessment."""

    shared_ground: tuple[str, ...]
    unresolved_tensions: tuple[str, ...]
    candidates: tuple[DirectionCandidate, ...]
    provider_id: str
    model_response_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.shared_ground or not self.unresolved_tensions or not self.candidates:
            raise ValueError("A Tiferet proposal requires ground, tension, and candidates")
        if not self.provider_id.strip():
            raise ValueError("A Tiferet proposal requires a provider identity")
        ids = tuple(item.candidate_id for item in self.candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("A Tiferet proposal requires unique candidate IDs")

    @property
    def proposal_hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True, slots=True)
class PolarityRecord:
    form_contribution: str
    flow_contribution: str
    accord_observation: str
    shared_ground: tuple[str, ...]
    unresolved_tensions: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    possible_movements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoalescenceResult:
    one_mind_state: str
    selected_direction: str
    retained_alternatives: tuple[str, ...]
    rejected_alternatives: tuple[str, ...]
    unresolved_tensions: tuple[str, ...]
    rationale: str
    proposed_action: str
    kernel_directives: tuple[str, ...]
    disposition: Disposition
    confidence: float
    candidate_assessments: tuple[DirectionCandidate, ...] = ()
    selected_candidate_id: str = ""
    selection_rule: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Coalescence confidence must be between 0 and 1")
        if self.candidate_assessments:
            ids = tuple(item.candidate_id for item in self.candidate_assessments)
            if len(set(ids)) != len(ids):
                raise ValueError("Tiferet candidate IDs must be unique")
            admitted = tuple(item for item in self.candidate_assessments if item.status == "admitted")
            if not self.selected_candidate_id or self.selected_candidate_id not in {
                item.candidate_id for item in admitted
            }:
                raise ValueError("Tiferet must select one admitted candidate")
            if not self.selection_rule.strip():
                raise ValueError("Tiferet candidate selection requires an explicit rule")


@dataclass(frozen=True, slots=True)
class RecursionFrame:
    recursion_id: str
    parent_id: str | None
    depth: int
    triggering_symbol: str
    source_sefirah: Sefirah
    source_world: World
    return_reason: str
    return_destination: Sefirah
    path_ids: tuple[str, ...]
    form_observation: str
    flow_observation: str
    accord_observation: str
    active_kernel_influences: tuple[str, ...]
    transformed_symbol: str
    new_distinctions: tuple[str, ...]
    new_relations: tuple[str, ...]
    added_constraints: tuple[str, ...]
    unresolved_polarity: tuple[str, ...]
    symbol_state_hash_before: str
    symbol_state_hash_after: str
    stop_reason: RecursionStop | None = None


@dataclass(frozen=True, slots=True)
class YesodField:
    phase: str
    prior_packet_hash: str | None
    intention_ref: str
    core_views: tuple[CoreView, ...]
    mutual_observations: tuple[CoreObservation, ...]
    active_kernel_influences: tuple[KernelActivation, ...]
    polarity_records: tuple[PolarityRecord, ...]
    recursion_lineage: tuple[RecursionFrame, ...]
    node_results: tuple["NodeResult", ...]
    evidence_refs: tuple[str, ...]
    evidence_bindings: tuple[str, ...]
    retrieval_snapshot_hash: str | None
    retrieval_policy_hash: str | None
    response_contract_hash: str | None
    proposed_context: str
    unresolved_tensions: tuple[str, ...]
    embodied_output_hash: str | None
    feedback_status: str
    feedback_ref: str | None
    packet_hash: str


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    content: str
    scope: str
    modes: tuple[ModeOfKnowing, ...]
    provenance_refs: tuple[str, ...]
    source_packet_hash: str
    projected_memory_hash: str
    feedback_ref: str | None
    significance: float

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Memory proposal content cannot be empty")
        if not self.scope.strip():
            raise ValueError("Memory proposal scope cannot be empty")
        if not 0.0 <= self.significance <= 1.0:
            raise ValueError("Memory significance must be between 0 and 1")

    @property
    def proposal_hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True, slots=True)
class ConsentGrant:
    proposal_hash: str
    granted_by: str
    scope: str
    sequence: int

    def __post_init__(self) -> None:
        if not self.proposal_hash or not self.granted_by or not self.scope:
            raise ValueError("Consent must bind a proposal, grantor, and scope")


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: GateDecision
    reasons: tuple[str, ...]
    proposal_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseContract:
    """Graph-owned outward commitments that a language renderer cannot replace."""

    direction: str
    rationale_basis: str
    next_step: str
    held_open: tuple[str, ...]
    external_action_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.direction.strip() or not self.rationale_basis.strip() or not self.next_step.strip():
            raise ValueError("A response contract requires direction, rationale, and next step")
        if not self.held_open or any(not item.strip() for item in self.held_open):
            raise ValueError("A response contract must preserve non-blank open tensions")

    @property
    def contract_hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True, slots=True)
class EmbodiedResponse:
    """One outward voice rendered from graph-owned, already-coalesced state."""

    direction: str
    rationale: str
    next_step: str
    held_open: tuple[str, ...]
    cited_evidence_refs: tuple[str, ...] = ()
    renderer_id: str = "deterministic-renderer:v0.0.1"
    response_contract_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.direction.strip() or not self.rationale.strip() or not self.next_step.strip():
            raise ValueError("An embodied response requires direction, rationale, and next step")
        if not self.held_open:
            raise ValueError("An embodied response must preserve at least one open tension")
        if not self.renderer_id.strip():
            raise ValueError("An embodied response requires a renderer identity")
        if len(set(self.cited_evidence_refs)) != len(self.cited_evidence_refs):
            raise ValueError("Embodied response evidence references must be unique")

    def as_text(self) -> str:
        lead = self.direction.rstrip()
        if not lead.endswith((".", "?", "!")):
            lead += "."
        held = "; ".join(item.rstrip(".?!") for item in self.held_open)
        return (
            f"{lead} Why this direction: {self.rationale.rstrip()} "
            f"Next embodied action: {self.next_step.rstrip()} "
            f"Still held open: {held}."
        )


_REALIZATION_COMMITMENT_LANGUAGE = re.compile(
    r"\b(?:also|additionally|should|must|need\s+to|recommend|promise|guarantee|"
    r"publish|post|send|contact|email|upload|delete|install|execute|tool|memory)\b",
    re.IGNORECASE,
)


def validate_response_realization(
    response: EmbodiedResponse,
    contract: ResponseContract,
) -> None:
    """Require every graph commitment verbatim while allowing small non-binding framing."""

    if response.response_contract_hash != contract.contract_hash:
        raise ValueError("The output renderer did not return the sealed response contract hash")
    anchored = (
        ("direction", response.direction, contract.direction),
        ("rationale", response.rationale, contract.rationale_basis),
        ("next_step", response.next_step, contract.next_step),
    )
    for field, realized, commitment in anchored:
        if commitment not in realized:
            raise ValueError(f"The realized {field} removed or paraphrased a graph commitment")
        framing = realized.replace(commitment, "", 1).strip()
        if len(framing) > 240:
            raise ValueError(f"The realized {field} added more than bounded framing")
        if framing and _REALIZATION_COMMITMENT_LANGUAGE.search(framing):
            raise ValueError(f"The realized {field} added commitment-like language")
    if len(response.held_open) != len(contract.held_open):
        raise ValueError("The realization changed the number of held-open tensions")
    for realized, commitment in zip(response.held_open, contract.held_open, strict=True):
        if commitment not in realized:
            raise ValueError("The realization removed or paraphrased a held-open tension")
        framing = realized.replace(commitment, "", 1).strip()
        if len(framing) > 160 or (
            framing and _REALIZATION_COMMITMENT_LANGUAGE.search(framing)
        ):
            raise ValueError("The realization added a new commitment to an open tension")


@dataclass(frozen=True, slots=True)
class CognitiveEvent:
    sequence: int
    event_type: str
    node: Sefirah
    world: World
    technical: dict[str, Any]
    symbolic: dict[str, Any]
    state_hash_before: str
    state_hash_after: str
    previous_event_hash: str
    event_hash: str

    def hash_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "node": self.node,
            "world": self.world,
            "technical": self.technical,
            "symbolic": self.symbolic,
            "state_hash_before": self.state_hash_before,
            "state_hash_after": self.state_hash_after,
            "previous_event_hash": self.previous_event_hash,
        }


@dataclass(frozen=True, slots=True)
class NodeResult:
    """The structured, causally inspectable effect of an activated Sefirah."""

    node: Sefirah
    world: World
    core_id: CoreId
    observable_effect: str
    input_hashes: tuple[str, ...]
    structured_output: dict[str, Any]
    imbalance_signals: tuple[str, ...]
    output_hash: str


@dataclass(slots=True)
class MindState:
    run_id: str
    graph_version: str
    intention: str
    covenant: Covenant
    current_node: Sefirah = Sefirah.KETER
    current_world: World = World.ATZILUT
    active_kernels: list[KernelActivation] = field(default_factory=list)
    core_views: list[CoreView] = field(default_factory=list)
    observations: list[CoreObservation] = field(default_factory=list)
    polarities: list[PolarityRecord] = field(default_factory=list)
    coalescence: CoalescenceResult | None = None
    recursion_frames: list[RecursionFrame] = field(default_factory=list)
    pre_embodiment_yesod: YesodField | None = None
    yesod_field: YesodField | None = None
    response_contract: ResponseContract | None = None
    embodied_response: EmbodiedResponse | None = None
    evidence_refs: tuple[str, ...] = ()
    evidence_bindings: tuple[str, ...] = ()
    retrieval_snapshot_hash: str | None = None
    retrieval_policy_hash: str | None = None
    output: str = ""
    prior_feedback: str = ""
    embodied_feedback: str = ""
    feedback_status: str = "pending"
    unresolved_tensions: list[str] = field(default_factory=list)
    gate_result: GateResult | None = None
    durable_memory_hashes: list[str] = field(default_factory=list)
    node_results: list[NodeResult] = field(default_factory=list)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "graph_version": self.graph_version,
            "intention": self.intention,
            "covenant": self.covenant,
            "current_node": self.current_node,
            "current_world": self.current_world,
            "active_kernels": self.active_kernels,
            "core_views": self.core_views,
            "observations": self.observations,
            "polarities": self.polarities,
            "coalescence": self.coalescence,
            "recursion_frames": self.recursion_frames,
            "pre_embodiment_yesod": self.pre_embodiment_yesod,
            "yesod_field": self.yesod_field,
            "response_contract": self.response_contract,
            "embodied_response": self.embodied_response,
            "evidence_refs": self.evidence_refs,
            "evidence_bindings": self.evidence_bindings,
            "retrieval_snapshot_hash": self.retrieval_snapshot_hash,
            "retrieval_policy_hash": self.retrieval_policy_hash,
            "output": self.output,
            "prior_feedback": self.prior_feedback,
            "embodied_feedback": self.embodied_feedback,
            "feedback_status": self.feedback_status,
            "unresolved_tensions": self.unresolved_tensions,
            "gate_result": self.gate_result,
            "durable_memory_hashes": self.durable_memory_hashes,
            "node_results": self.node_results,
        }

    @property
    def state_hash(self) -> str:
        return stable_hash(self.semantic_payload())
