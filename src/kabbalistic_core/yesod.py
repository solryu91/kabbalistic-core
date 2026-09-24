"""Yesod packet integrity helpers."""

from __future__ import annotations

from typing import Any

from .models import YesodField, stable_hash


class YesodIntegrityError(ValueError):
    pass


def yesod_payload(field: YesodField) -> dict[str, Any]:
    return {
        "phase": field.phase,
        "prior_packet_hash": field.prior_packet_hash,
        "intention_ref": field.intention_ref,
        "core_views": field.core_views,
        "mutual_observations": field.mutual_observations,
        "active_kernel_influences": field.active_kernel_influences,
        "polarity_records": field.polarity_records,
        "recursion_lineage": field.recursion_lineage,
        "node_results": field.node_results,
        "evidence_refs": field.evidence_refs,
        "evidence_bindings": field.evidence_bindings,
        "retrieval_snapshot_hash": field.retrieval_snapshot_hash,
        "retrieval_policy_hash": field.retrieval_policy_hash,
        "response_contract_hash": field.response_contract_hash,
        "proposed_context": field.proposed_context,
        "unresolved_tensions": field.unresolved_tensions,
        "embodied_output_hash": field.embodied_output_hash,
        "feedback_status": field.feedback_status,
        "feedback_ref": field.feedback_ref,
    }


def validate_yesod_field(field: YesodField) -> None:
    expected = stable_hash(yesod_payload(field))
    if field.packet_hash != expected:
        raise YesodIntegrityError("The Yesod packet content no longer matches its sealed hash")
