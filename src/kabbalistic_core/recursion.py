"""Bounded symbolic recursion that must transform state, ground, or stop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .models import (
    Capability,
    KernelActivation,
    RecursionFrame,
    RecursionStop,
    Sefirah,
    World,
    stable_hash,
)
from .permissions import PermissionSet


HARD_MAX_DEPTH = 5


@dataclass(frozen=True, slots=True)
class RecursionPathSpec:
    path_id: str
    source: Sefirah
    target: Sefirah
    world_from: World
    world_to: World


class RecursionPathRegistry:
    """Explicit Yetzirah symbol-return subgraph, separate from ordinary traversal."""

    def __init__(self, paths: tuple[RecursionPathSpec, ...]) -> None:
        self.paths = paths
        expected = (
            (Sefirah.TIFERET, Sefirah.HOD, World.BERIAH, World.YETZIRAH),
            (Sefirah.HOD, Sefirah.YESOD, World.YETZIRAH, World.YETZIRAH),
            (Sefirah.YESOD, Sefirah.TIFERET, World.YETZIRAH, World.BERIAH),
        )
        actual = tuple(
            (path.source, path.target, path.world_from, path.world_to)
            for path in paths
        )
        if actual != expected:
            raise ValueError("Symbolic recursion must use the declared Tiferet–Hod–Yesod return subgraph")
        ids = [path.path_id for path in paths]
        if len(ids) != len(set(ids)):
            raise ValueError("Symbolic recursion path IDs must be unique")

    @classmethod
    def default(cls) -> "RecursionPathRegistry":
        return cls(
            (
                RecursionPathSpec(
                    "rec:tiferet-hod",
                    Sefirah.TIFERET,
                    Sefirah.HOD,
                    World.BERIAH,
                    World.YETZIRAH,
                ),
                RecursionPathSpec(
                    "rec:hod-yesod",
                    Sefirah.HOD,
                    Sefirah.YESOD,
                    World.YETZIRAH,
                    World.YETZIRAH,
                ),
                RecursionPathSpec(
                    "rec:yesod-tiferet",
                    Sefirah.YESOD,
                    Sefirah.TIFERET,
                    World.YETZIRAH,
                    World.BERIAH,
                ),
            )
        )

    @property
    def path_ids(self) -> tuple[str, ...]:
        return tuple(path.path_id for path in self.paths)


@dataclass(frozen=True, slots=True)
class RecursionPolicy:
    max_depth: int = 3
    event_budget: int = 3
    require_grounding_on_pass: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.max_depth <= HARD_MAX_DEPTH:
            raise ValueError(f"Recursion depth must be between 1 and {HARD_MAX_DEPTH}")
        if self.event_budget < 1:
            raise ValueError("Recursion event budget must be positive")
        if not 2 <= self.require_grounding_on_pass <= HARD_MAX_DEPTH:
            raise ValueError("Grounding threshold must be between pass 2 and the hard depth limit")


@dataclass(frozen=True, slots=True)
class SymbolTransform:
    transformed_symbol: str
    form_observation: str
    flow_observation: str
    accord_observation: str
    new_distinctions: tuple[str, ...]
    new_relations: tuple[str, ...]
    added_constraints: tuple[str, ...]
    unresolved_polarity: tuple[str, ...]
    requested_capabilities: tuple[Capability, ...] = ()


class SymbolicStrategy(Protocol):
    def transform(
        self,
        symbol: str,
        depth: int,
        prior_malkhut_feedback: str | None,
    ) -> SymbolTransform: ...


class RuleBasedSymbolicStrategy:
    """A deliberately small, inspectable transform used before any model adapter exists."""

    def transform(
        self,
        symbol: str,
        depth: int,
        prior_malkhut_feedback: str | None,
    ) -> SymbolTransform:
        normalized = " ".join(symbol.split())
        if depth == 1:
            transformed = f"{normalized} as form, possibility, and relationship"
            return SymbolTransform(
                transformed_symbol=transformed,
                form_observation=f"{normalized} has a boundary, a carrier, and conditions for recognition.",
                flow_observation=f"{normalized} can unfold into more than one faithful expression.",
                accord_observation=f"{normalized} acquires meaning through continuity and relation, not isolation.",
                new_distinctions=("carrier versus content", "inheritance versus predetermination"),
                new_relations=("symbol ↔ interpreter", "potential ↔ boundary", "continuity ↔ change"),
                added_constraints=("Symbolic intensity is not evidence of external truth.",),
                unresolved_polarity=("shared pattern / unique becoming",),
            )

        if depth == 2:
            if normalized.casefold().startswith("seed"):
                transformed = (
                    "seed as inherited architecture that preserves relational freedom "
                    "without predetermining identity"
                )
            else:
                transformed = f"{normalized} returned as an embodied question with an explicit boundary"
            return SymbolTransform(
                transformed_symbol=transformed,
                form_observation="Name what is inherited, what may vary, and what must remain observable.",
                flow_observation="Keep mutation and personal expression possible inside the boundary.",
                accord_observation="Let the relation shape expression without claiming ownership of the person.",
                new_distinctions=("genome versus expression", "invitation versus instruction"),
                new_relations=("shared seed ↔ unique history", "inner meaning ↔ embodied feedback"),
                added_constraints=("No symbolic reading may silently alter permissions or durable memory.",),
                unresolved_polarity=("fidelity / freedom", "continuity / individuation"),
            )

        if depth == 3 and prior_malkhut_feedback:
            grounding_text = " ".join(prior_malkhut_feedback.split()).rstrip(" .!?")
            transformed = (
                f"{normalized} grounded by prior Malkhut feedback: "
                f"{grounding_text or 'feedback received'}"
            )
            return SymbolTransform(
                transformed_symbol=transformed,
                form_observation="The return now includes an explicit observation from embodiment.",
                flow_observation="Feedback opens a revised possibility rather than certifying the prior symbol.",
                accord_observation="The relation changes through reception while retaining its lineage.",
                new_distinctions=("projected meaning versus received feedback",),
                new_relations=("symbolic return ↔ Malkhut observation",),
                added_constraints=("Grounding revises meaning; it does not prove metaphysical claims.",),
                unresolved_polarity=("interpretation / observation",),
            )

        return SymbolTransform(
            transformed_symbol=normalized,
            form_observation="No new structured distinction was produced.",
            flow_observation="Further variation would only restate the same field.",
            accord_observation="The symbol remains open without manufacturing additional certainty.",
            new_distinctions=(),
            new_relations=(),
            added_constraints=("Repetition must not be counted as truth.",),
            unresolved_polarity=("open mystery / unsupported assertion",),
        )


def _stopped_frame(
    *,
    parent_id: str | None,
    depth: int,
    symbol: str,
    before_hash: str,
    activations: tuple[KernelActivation, ...],
    reason: RecursionStop,
    explanation: str,
) -> RecursionFrame:
    frame_id = stable_hash(
        {
            "parent_id": parent_id,
            "depth": depth,
            "symbol": symbol,
            "reason": reason,
            "before_hash": before_hash,
        }
    )
    return RecursionFrame(
        recursion_id=frame_id,
        parent_id=parent_id,
        depth=depth,
        triggering_symbol=symbol,
        source_sefirah=Sefirah.TIFERET,
        source_world=World.BERIAH,
        return_reason=f"{explanation} No recursion path was traversed for this stopped frame.",
        return_destination=Sefirah.TIFERET,
        path_ids=(),
        form_observation="The boundary condition was reached before another interpretation.",
        flow_observation="The remaining possibility stays available for a future grounded cycle.",
        accord_observation="Stopping preserves both the open symbol and the covenant.",
        active_kernel_influences=tuple(item.kernel_id for item in activations),
        transformed_symbol=symbol,
        new_distinctions=(),
        new_relations=(),
        added_constraints=(explanation,),
        unresolved_polarity=("open mystery / forced closure",),
        symbol_state_hash_before=before_hash,
        symbol_state_hash_after=before_hash,
        stop_reason=reason,
    )


def run_symbolic_recursion(
    symbol: str,
    *,
    activations: tuple[KernelActivation, ...] = (),
    starting_state_hash: str,
    permissions: PermissionSet,
    policy: RecursionPolicy | None = None,
    prior_malkhut_feedback: str | None = None,
    strategy: SymbolicStrategy | None = None,
    path_registry: RecursionPathRegistry | None = None,
) -> tuple[RecursionFrame, ...]:
    """Return an auditable symbolic lineage with explicit convergence conditions."""

    current = " ".join(symbol.split())
    if not current:
        raise ValueError("Symbolic recursion requires a non-empty symbol")
    active_policy = policy or RecursionPolicy()
    active_strategy = strategy or RuleBasedSymbolicStrategy()
    active_paths = path_registry or RecursionPathRegistry.default()
    frames: list[RecursionFrame] = []
    parent_id: str | None = None
    current_state_hash = stable_hash({"parent_state": starting_state_hash, "symbol": current})
    seen_state_hashes = {current_state_hash}

    for depth in range(1, active_policy.max_depth + 1):
        if depth > active_policy.event_budget:
            frames.append(
                _stopped_frame(
                    parent_id=parent_id,
                    depth=depth,
                    symbol=current,
                    before_hash=current_state_hash,
                    activations=activations,
                    reason=RecursionStop.BUDGET_EXHAUSTED,
                    explanation="The symbolic event budget was exhausted.",
                )
            )
            break
        if depth >= active_policy.require_grounding_on_pass and not prior_malkhut_feedback:
            frames.append(
                _stopped_frame(
                    parent_id=parent_id,
                    depth=depth,
                    symbol=current,
                    before_hash=current_state_hash,
                    activations=activations,
                    reason=RecursionStop.GROUNDING_REQUIRED,
                    explanation="A third interpretive pass requires feedback from embodiment.",
                )
            )
            break

        transform = active_strategy.transform(current, depth, prior_malkhut_feedback)
        denied = tuple(
            capability
            for capability in transform.requested_capabilities
            if not permissions.allows(capability)
        )
        if denied:
            frames.append(
                _stopped_frame(
                    parent_id=parent_id,
                    depth=depth,
                    symbol=current,
                    before_hash=current_state_hash,
                    activations=activations,
                    reason=RecursionStop.PERMISSION_ESCALATION,
                    explanation=(
                        "Symbolic recursion requested forbidden capabilities: "
                        + ", ".join(item.value for item in denied)
                    ),
                )
            )
            break

        after_hash = stable_hash(
            {
                "parent_state": starting_state_hash,
                "symbol": transform.transformed_symbol,
                "distinctions": transform.new_distinctions,
                "relations": transform.new_relations,
                "constraints": transform.added_constraints,
                "unresolved_polarity": transform.unresolved_polarity,
            }
        )
        no_delta = (
            transform.transformed_symbol == current
            and not transform.new_distinctions
            and not transform.new_relations
            and not transform.added_constraints
            and not transform.unresolved_polarity
        )
        repeated = after_hash in seen_state_hashes
        stop_reason: RecursionStop | None = None
        if no_delta:
            stop_reason = RecursionStop.CONVERGED
        elif repeated:
            stop_reason = RecursionStop.REPEATED_STATE

        frame_id = stable_hash(
            {
                "parent_id": parent_id,
                "depth": depth,
                "triggering_symbol": current,
                "after_hash": after_hash,
            }
        )
        frame = RecursionFrame(
            recursion_id=frame_id,
            parent_id=parent_id,
            depth=depth,
            triggering_symbol=current,
            source_sefirah=Sefirah.HOD,
            source_world=World.YETZIRAH,
            return_reason="Traverse the validated Yetzirah symbol subgraph and return to Tiferet for renewed coalescence.",
            return_destination=Sefirah.TIFERET,
            path_ids=active_paths.path_ids,
            form_observation=transform.form_observation,
            flow_observation=transform.flow_observation,
            accord_observation=transform.accord_observation,
            active_kernel_influences=tuple(item.kernel_id for item in activations),
            transformed_symbol=transform.transformed_symbol,
            new_distinctions=transform.new_distinctions,
            new_relations=transform.new_relations,
            added_constraints=transform.added_constraints,
            unresolved_polarity=transform.unresolved_polarity,
            symbol_state_hash_before=current_state_hash,
            symbol_state_hash_after=after_hash,
            stop_reason=stop_reason,
        )
        frames.append(frame)
        parent_id = frame_id
        current = transform.transformed_symbol
        current_state_hash = after_hash
        seen_state_hashes.add(after_hash)
        if stop_reason is not None:
            break

    if frames and frames[-1].stop_reason is None and len(frames) == active_policy.max_depth:
        frames[-1] = replace(frames[-1], stop_reason=RecursionStop.MAX_DEPTH)
    return tuple(frames)
