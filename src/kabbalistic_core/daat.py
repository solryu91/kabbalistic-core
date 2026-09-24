"""Da'at as an explicit two-phase gate for durable state promotion."""

from __future__ import annotations

from .models import (
    Capability,
    ConsentGrant,
    GateDecision,
    GateResult,
    MemoryProposal,
    ModeOfKnowing,
    stable_hash,
)
from .permissions import PermissionSet


DEFAULT_SIGNIFICANCE_THRESHOLD = 0.70


def proposal_after_hash(
    source_packet_hash: str,
    content: str,
    scope: str,
    modes: tuple[ModeOfKnowing, ...],
) -> str:
    """Compute the exact hypothetical durable state without mutating current state."""

    return stable_hash(
        {
            "source_packet_hash": source_packet_hash,
            "content": content,
            "scope": scope,
            "modes": modes,
        }
    )


class DaatGate:
    def __init__(self, significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD) -> None:
        if not 0.0 <= significance_threshold <= 1.0:
            raise ValueError("Da'at significance threshold must be between 0 and 1")
        self.significance_threshold = significance_threshold

    def evaluate(
        self,
        proposal: MemoryProposal | None,
        consent: ConsentGrant | None,
        permissions: PermissionSet,
        *,
        current_packet_hash: str,
    ) -> GateResult:
        if proposal is None:
            return GateResult(
                decision=GateDecision.NOT_APPLICABLE,
                reasons=("No durable memory proposal entered the Da'at threshold.",),
            )

        reasons: list[str] = []
        if not permissions.allows(Capability.PROPOSE_MEMORY):
            reasons.append("The system lacks permission to propose durable memory.")
        if not permissions.allows(Capability.COMMIT_MEMORY):
            reasons.append("The system lacks permission to commit durable memory.")
        if proposal.source_packet_hash != current_packet_hash:
            reasons.append("The proposal is not based on the current sealed Yesod packet.")
        expected_after = proposal_after_hash(
            proposal.source_packet_hash,
            proposal.content,
            proposal.scope,
            proposal.modes,
        )
        if proposal.projected_memory_hash != expected_after:
            reasons.append("The proposal's packet-to-memory projection is not reproducible.")
        if not proposal.provenance_refs:
            reasons.append("The proposal has no provenance references.")
        if not proposal.modes:
            reasons.append("The proposal does not label its modes of knowing.")
        if not proposal.feedback_ref:
            reasons.append("No embodied-feedback reference is bound to the proposal.")
        if proposal.significance < self.significance_threshold:
            reasons.append(
                f"Significance {proposal.significance:.2f} is below the "
                f"{self.significance_threshold:.2f} durable-memory threshold."
            )
        if consent is None:
            reasons.append("No explicit proposal-bound consent was supplied.")
        else:
            if consent.proposal_hash != proposal.proposal_hash:
                reasons.append("Consent is bound to a different proposal.")
            if consent.scope != proposal.scope:
                reasons.append("Consent scope does not match proposal scope.")
            if consent.sequence < 1:
                reasons.append("Consent sequence must be a positive explicit act.")

        if reasons:
            return GateResult(
                decision=GateDecision.WITHHELD,
                reasons=tuple(reasons),
                proposal_hash=proposal.proposal_hash,
            )
        return GateResult(
            decision=GateDecision.APPROVED_FOR_COMMIT,
            reasons=(
                "Exact proposal, feedback, provenance, significance, capability, and consent checks passed; no commit has occurred.",
            ),
            proposal_hash=proposal.proposal_hash,
        )
