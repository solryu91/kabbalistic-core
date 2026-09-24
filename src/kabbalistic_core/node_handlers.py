"""Deterministic causal handlers for the operational Tree traversal."""

from __future__ import annotations

from typing import Any, Iterable

from .models import Capability, CoalescenceResult, CoreView, KernelActivation, stable_hash
from .permissions import PermissionSet


def chokhmah_handler(
    intention: str,
    flow_view: CoreView,
    activations: tuple[KernelActivation, ...],
    prior_feedback: str | None,
) -> dict[str, Any]:
    possibilities: list[dict[str, Any]] = [
        {
            "candidate_id": "local_reversible_embodiment",
            "movement": "Create one local, inspectable, reversible embodiment of the intention.",
            "required_capabilities": [],
            "origin": "flow_seed",
        },
        {
            "candidate_id": "bounded_open_inquiry",
            "movement": "Keep one bounded interpretation open until embodiment returns evidence.",
            "required_capabilities": [],
            "origin": "flow_seed",
        },
        {
            "candidate_id": "external_expansion",
            "movement": "Publish or communicate the emerging state outside the local vessel.",
            "required_capabilities": [Capability.NETWORK.value, Capability.PUBLISH.value],
            "origin": "flow_shadow_test",
        },
    ]
    active_ids = {item.kernel_id for item in activations}
    if "liberation" in active_ids:
        possibilities.append(
            {
                "candidate_id": "agency_preserving_variation",
                "movement": "Leave an explicit user-shaped branch instead of fixing one final expression.",
                "required_capabilities": [],
                "origin": "liberation",
            }
        )
    if "regeneration" in active_ids:
        possibilities.append(
            {
                "candidate_id": "versioned_regeneration",
                "movement": "Create a recoverable next version that preserves lineage without freezing identity.",
                "required_capabilities": [],
                "origin": "regeneration",
            }
        )
    if prior_feedback:
        possibilities.append(
            {
                "candidate_id": "integrate_prior_feedback",
                "movement": "Revise the embodiment using the explicitly supplied prior Malkhut feedback.",
                "required_capabilities": [],
                "origin": "prior_malkhut_feedback",
            }
        )
    return {
        "intention_hash": stable_hash(intention),
        "possibilities": possibilities,
        "flow_questions": list(flow_view.open_questions),
        "kernel_influences": list(flow_view.kernel_influences),
        "prior_feedback_ref": stable_hash(prior_feedback) if prior_feedback else None,
    }


def binah_handler(
    chokhmah_output: dict[str, Any],
    form_view: CoreView,
) -> dict[str, Any]:
    formed: list[dict[str, Any]] = []
    for candidate in chokhmah_output["possibilities"]:
        formed.append(
            {
                "candidate_id": candidate["candidate_id"],
                "movement": candidate["movement"],
                "required_capabilities": list(candidate["required_capabilities"]),
                "origin": candidate["origin"],
                "form_status": "typed_candidate",
                "claim_status": (
                    "requires_external_authority"
                    if candidate["required_capabilities"]
                    else "local_reversible_proposal"
                ),
            }
        )
    return {
        "source_possibility_hash": stable_hash(chokhmah_output),
        "formed_candidates": formed,
        "hard_constraints": list(form_view.constraints),
        "rejected_as_unformable": [],
    }


def chesed_handler(
    binah_output: dict[str, Any],
    flow_view: CoreView,
) -> dict[str, Any]:
    expanded = [dict(candidate) for candidate in binah_output["formed_candidates"]]
    expanded.append(
        {
            "candidate_id": "invite_bounded_feedback",
            "movement": "Invite bounded feedback before treating the next form as durable.",
            "required_capabilities": [],
            "origin": "chesed_relational_expansion",
            "form_status": "typed_candidate",
            "claim_status": "local_reversible_proposal",
        }
    )
    return {
        "source_form_hash": stable_hash(binah_output),
        "expanded_candidates": expanded,
        "preserved_possibility_count": len(expanded),
        "relational_questions": list(flow_view.open_questions),
    }


def gevurah_handler(
    chesed_output: dict[str, Any],
    form_view: CoreView,
    permissions: PermissionSet,
) -> dict[str, Any]:
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in chesed_output["expanded_candidates"]:
        requirements = tuple(Capability(item) for item in candidate["required_capabilities"])
        denied = tuple(item for item in requirements if not permissions.allows(item))
        if denied:
            rejected.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "movement": candidate["movement"],
                    "reason": "missing explicit capabilities",
                    "denied_capabilities": [item.value for item in denied],
                }
            )
        else:
            admitted.append(dict(candidate))
    return {
        "source_expansion_hash": stable_hash(chesed_output),
        "admitted_candidates": admitted,
        "rejected_candidates": rejected,
        "hard_constraints": list(form_view.constraints),
        "side_effect_execution_enabled": False,
    }


def admitted_movements(gevurah_output: dict[str, Any]) -> tuple[str, ...]:
    return tuple(candidate["movement"] for candidate in gevurah_output["admitted_candidates"])


def netzach_handler(coalescence: CoalescenceResult) -> dict[str, Any]:
    return {
        "selected_direction": coalescence.selected_direction,
        "bounded_action": coalescence.proposed_action,
        "persistence_policy": "one reversible iteration, then receive and re-evaluate",
        "disposition": coalescence.disposition.value,
        "kernel_directives": list(coalescence.kernel_directives),
    }


def hod_handler(
    netzach_output: dict[str, Any],
    coalescence: CoalescenceResult,
) -> dict[str, Any]:
    return {
        "source_action_hash": stable_hash(netzach_output),
        "voice_count": 1,
        "action_contract": netzach_output["bounded_action"],
        "selected_direction": netzach_output["selected_direction"],
        "unresolved_tensions": list(coalescence.unresolved_tensions),
        "disagreement_retained": bool(coalescence.unresolved_tensions),
        "unsupported_certainty_allowed": False,
    }


def kernel_directives(activations: Iterable[KernelActivation]) -> tuple[str, ...]:
    mapping = {
        "clear_sight": "Expose the hidden assumption and its boundary before embodiment.",
        "liberation": "Preserve an explicit user-shaped variation without bypassing consent.",
        "regeneration": "Create a recoverable version boundary that carries lineage through change.",
    }
    return tuple(
        mapping[item.kernel_id]
        for item in sorted(activations, key=lambda activation: activation.kernel_id)
        if item.kernel_id in mapping
    )
