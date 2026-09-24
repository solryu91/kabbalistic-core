"""Shared deterministic fixtures for the standard-library test suite.

The project deliberately does not require installation for its first test run.  Importing
this module places ``src`` at the front of ``sys.path`` so that
``python -m unittest discover -s tests -v`` works from the repository root.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from kabbalistic_core.archetypes import ArchetypeRegistry
from kabbalistic_core.cores import InitialSnapshot, build_three_views
from kabbalistic_core.daat import proposal_after_hash
from kabbalistic_core.models import (
    Capability,
    ConsentGrant,
    CoreId,
    CoreView,
    Covenant,
    MemoryProposal,
    ModeOfKnowing,
    stable_hash,
)
from kabbalistic_core.permissions import PermissionSet


def default_snapshot(*, evidence: tuple[str, ...] = ("source:test",)) -> InitialSnapshot:
    return InitialSnapshot(
        intention="Build one faithful, inspectable cognitive cycle.",
        covenant=Covenant(purpose="Preserve fidelity while making computation observable."),
        evidence_refs=evidence,
        permitted_capabilities=tuple(
            sorted(item.value for item in PermissionSet().granted)
        ),
    )


def default_views() -> tuple[CoreView, ...]:
    registry = ArchetypeRegistry.load_default()
    return build_three_views(default_snapshot(), (), registry)


def custom_view(
    core_id: CoreId,
    *,
    proposition: str,
    constraint: str,
    confidence: float = 0.75,
) -> CoreView:
    return CoreView(
        core_id=core_id,
        propositions=(proposition, proposition),
        constraints=(constraint,),
        symbolic_relations=(f"{core_id.value} relation",),
        open_questions=(f"What does {core_id.value} still need to learn?",),
        confidence=confidence,
        imbalance_signals=(f"{core_id.value} shadow",),
    )


def valid_memory_proposal(
    *,
    source_packet_hash: str | None = None,
    scope: str = "profile:test",
) -> MemoryProposal:
    packet_hash = source_packet_hash or stable_hash({"yesod": "sealed-packet"})
    content = "Preserve a tested distinction while retaining unresolved mystery."
    modes = (ModeOfKnowing.TECHNICAL, ModeOfKnowing.SYMBOLIC)
    return MemoryProposal(
        content=content,
        scope=scope,
        modes=modes,
        provenance_refs=("run:test", "event:test"),
        source_packet_hash=packet_hash,
        projected_memory_hash=proposal_after_hash(packet_hash, content, scope, modes),
        feedback_ref="feedback:test",
        significance=0.80,
    )


def exact_consent(proposal: MemoryProposal) -> ConsentGrant:
    return ConsentGrant(
        proposal_hash=proposal.proposal_hash,
        granted_by="user:test",
        scope=proposal.scope,
        sequence=1,
    )


def memory_permissions() -> PermissionSet:
    return PermissionSet(
        frozenset({Capability.PROPOSE_MEMORY, Capability.COMMIT_MEMORY})
    )


def with_proposition(view: CoreView, proposition: str) -> CoreView:
    """Return a structurally valid altered view for adversarial coalescence tests."""

    return replace(view, propositions=(proposition, proposition))
