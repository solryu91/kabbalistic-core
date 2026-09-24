"""Run and export one synthetic, disposable v0.1 continuity evidence trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kabbalistic_core.continuity import ContinuityService  # noqa: E402
from kabbalistic_core.models import canonical_json, stable_hash  # noqa: E402


OWNER = "owner:synthetic-continuity-evidence"
INTENTION = "Help me choose a small reversible next step without adding outside obligations."
EPISODE = "When planning with me, keep the first step under fifteen minutes and reversible."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-root", type=Path)
    return parser


def acknowledge(service: ContinuityService, response: dict) -> dict:
    return service.acknowledge_delivery(
        response["session_id"],
        delivery_id=response["delivery"]["delivery_id"],
        response_hash=response["response_hash"],
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.work_root is not None:
        args.work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="continuity-evidence-",
        dir=None if args.work_root is None else args.work_root,
    ) as temporary:
        root = Path(temporary)
        database = root / "content-domain" / "continuity.sqlite3"
        checkpoint = root / "authority-domain" / "authority.checkpoint.json"
        key = os.urandom(32)

        service_a = ContinuityService.create(
            database,
            checkpoint,
            key,
            owner_principal=OWNER,
        )
        session_a = service_a.open_session(principal=OWNER, intention=INTENTION)
        precommit = service_a.respond(
            session_a["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        precommit_ack = acknowledge(service_a, precommit)
        proposal = service_a.propose_episode(
            session_a["session_id"], principal=OWNER, content=EPISODE
        )
        consent = service_a.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        commit = service_a.commit_granted_proposal(
            proposal_ref=proposal["proposal_ref"],
            consent_decision_ref=consent["decision_ref"],
        )
        receipt = service_a.inspect_receipt(commit["committed_consent_receipt_ref"])
        close_a = service_a.close_session(session_a["session_id"], principal=OWNER)
        instance_a = service_a.runtime_instance_id
        verify_a = service_a.verify()
        service_a.close()

        service_b = ContinuityService.open(database, checkpoint, key)
        session_b = service_b.open_session(principal=OWNER, intention=INTENTION)
        history = service_b.respond(
            session_b["session_id"], include_history=True, condition_label="history"
        )
        history_ack = acknowledge(service_b, history)
        close_b = service_b.close_session(session_b["session_id"], principal=OWNER)
        control_session = service_b.open_session(principal=OWNER, intention=INTENTION)
        no_history = service_b.respond(
            control_session["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        no_history_ack = acknowledge(service_b, no_history)
        comparison = service_b.causal_comparison(history, no_history)
        history_read_set = service_b.store.read_record(history["read_set_ref"])
        no_history_read_set = service_b.store.read_record(no_history["read_set_ref"])
        close_control = service_b.close_session(control_session["session_id"], principal=OWNER)
        verify_b = service_b.verify()
        instance_b = service_b.runtime_instance_id
        service_b.close()

        database_bytes = database.read_bytes()
        checkpoint_bytes = checkpoint.read_bytes()
        evidence = {
            "schema_version": "continuity-evidence:v0.1",
            "scope": (
                "Session A → explicit proposal → visible consent → durable commit → close → "
                "independent Session B → exact read set → history response → matched no-history check"
            ),
            "synthetic_fixture": {
                "intention": INTENTION,
                "episode": EPISODE,
                "contains_nonpublic_source_material": False,
                "public_seed_material_used": False,
            },
            "session_a": {
                "open": session_a,
                "precommit_read_set_ref": precommit["read_set_ref"],
                "precommit_selected_episode_refs": precommit["selected_episode_refs"],
                "precommit_response_hash": precommit["response_hash"],
                "precommit_delivery_ack": precommit_ack,
                "visible_proposal": proposal,
                "consent": consent,
                "commit": commit,
                "committed_consent_receipt": receipt,
                "close": close_a,
                "verification": verify_a,
            },
            "session_b": {
                "open": session_b,
                "independent_runtime_instance": instance_a != instance_b,
                "history_condition": {
                    "read_set_ref": history["read_set_ref"],
                    "snapshot_ref": history["snapshot_ref"],
                    "selected_episode_refs": history["selected_episode_refs"],
                    "ordered_context": history["ordered_context"],
                    "rendered_context_hash": history["rendered_context_hash"],
                    "sealed_immediately_before_stage": history_read_set.payload[
                        "sealed_immediately_before_stage"
                    ],
                    "response": history["response"],
                    "response_hash": history["response_hash"],
                    "model_call": history["model_call"],
                    "delivery_ack": history_ack,
                },
                "matched_no_history_condition": {
                    "read_set_ref": no_history["read_set_ref"],
                    "snapshot_ref": no_history["snapshot_ref"],
                    "eligible_episode_refs": no_history["eligible_episode_refs"],
                    "selected_episode_refs": no_history["selected_episode_refs"],
                    "excluded_with_reasons": no_history_read_set.payload[
                        "excluded_candidate_refs_with_reasons"
                    ],
                    "rendered_context_hash": no_history["rendered_context_hash"],
                    "sealed_immediately_before_stage": no_history_read_set.payload[
                        "sealed_immediately_before_stage"
                    ],
                    "response": no_history["response"],
                    "response_hash": no_history["response_hash"],
                    "model_call": no_history["model_call"],
                    "delivery_ack": no_history_ack,
                },
                "comparison": comparison,
                "close": close_b,
                "matched_control_session": control_session,
                "matched_control_close": close_control,
                "verification": verify_b,
            },
            "storage_evidence": {
                "database_sha256": file_hash(database),
                "checkpoint_sha256": file_hash(checkpoint),
                "database_size_bytes": len(database_bytes),
                "checkpoint_size_bytes": len(checkpoint_bytes),
                "episode_plaintext_absent_from_database": (
                    EPISODE.encode("utf-8") not in database_bytes
                ),
                "episode_plaintext_absent_from_checkpoint": (
                    EPISODE.encode("utf-8") not in checkpoint_bytes
                ),
                "master_key_absent_from_database": key not in database_bytes,
                "master_key_absent_from_checkpoint": key not in checkpoint_bytes,
                "master_key_exported": False,
                "content_and_checkpoint_directories_distinct": (
                    database.parent != checkpoint.parent
                ),
            },
            "results": {
                "session_a_commit_completed": True,
                "independent_session_b_opened": instance_a != instance_b,
                "exact_episode_entered_session_b_read_set": (
                    history["selected_episode_refs"] == [commit["episode_ref"]]
                ),
                "matched_no_history_removed_episode": not no_history[
                    "selected_episode_refs"
                ],
                "mechanical_continuity_effect_observed": comparison[
                    "history_effect_observed"
                ],
                "null_result": not comparison["history_effect_observed"],
                "adverse_result": False,
                "protocol_failure": False,
                "claim_boundary": comparison["claim_boundary"],
            },
            "deviations_from_frozen_specification": [],
            "known_implementation_limitations": [
                (
                    "The history-conditioned response is a deterministic causal probe, "
                    "not evidence of relationship intelligence or model cognition."
                ),
                (
                    "The 32-byte master key is supplied externally for the run; this slice "
                    "does not implement operating-system or hardware key custody."
                ),
                (
                    "The separately stored authenticated checkpoint detects content-only "
                    "rollback while the current checkpoint is retained, but is not a "
                    "hardware monotonic counter and cannot detect a coordinated rollback "
                    "of content, checkpoint, and the external key environment."
                ),
            ],
            "explicit_non_capabilities": [
                "correction",
                "forgetting",
                "durable_disagreement_or_non_mirroring",
                "semantic_projection",
                "embeddings_or_vector_search",
                "archive_import",
                "public_seed_inheritance_or_deployment",
                "branch_merge",
                "kernel_efficacy",
                "topology_expansion",
            ],
            "next_phase_readiness": {
                "phase": "correction",
                "status": "READY_FOR_OWNER_REVIEW_NOT_AUTHORIZED",
                "implementation_started": False,
            },
        }
        evidence["evidence_hash"] = stable_hash(evidence)
        args.output.write_text(canonical_json(evidence) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "evidence_hash": evidence["evidence_hash"],
                    "history_effect_observed": comparison["history_effect_observed"],
                    "record_count": verify_b["record_count"],
                    "next_phase": evidence["next_phase_readiness"],
                },
                indent=2,
            )
        )
        return 0 if comparison["history_effect_observed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
