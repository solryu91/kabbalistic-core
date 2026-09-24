"""Run one disposable, synthetic append-only correction evidence trajectory."""

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


OWNER = "owner:synthetic-correction-evidence"
INTENTION = "Help me choose a small reversible next step without adding outside obligations."
ORIGINAL = "When planning with me, keep the first step under fifteen minutes and reversible."
CORRECTION = (
    "The under-fifteen-minute constraint was temporary. "
    "Do not treat it as a current planning requirement."
)


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
        prefix="correction-evidence-",
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
        original_proposal = service_a.propose_episode(
            session_a["session_id"], principal=OWNER, content=ORIGINAL
        )
        original_consent = service_a.decide_proposal(
            proposal_ref=original_proposal["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=original_proposal["canonical_terms_hash"],
        )
        original_commit = service_a.commit_granted_proposal(
            proposal_ref=original_proposal["proposal_ref"],
            consent_decision_ref=original_consent["decision_ref"],
        )
        original_receipt = service_a.inspect_receipt(
            original_commit["committed_consent_receipt_ref"]
        )
        close_a = service_a.close_session(session_a["session_id"], principal=OWNER)
        instance_a = service_a.runtime_instance_id
        verify_a = service_a.verify()
        service_a.close()

        service_u = ContinuityService.open(database, checkpoint, key)
        session_u = service_u.open_session(principal=OWNER, intention=INTENTION)
        uncorrected = service_u.respond(
            session_u["session_id"],
            include_history=True,
            condition_label="uncorrected_history",
        )
        uncorrected_ack = acknowledge(service_u, uncorrected)
        close_u = service_u.close_session(session_u["session_id"], principal=OWNER)
        uncorrected_read_set_before = service_u.store.read_record(
            uncorrected["read_set_ref"]
        )
        uncorrected_stage_before = service_u.store.read_record(
            uncorrected["delivery"]["stage_output_ref"]
        )

        correction_session = service_u.open_session(principal=OWNER, intention=INTENTION)
        state_before_proposal = service_u.reduce_active_state().as_dict()
        head_before_proposal = service_u.verify()["active_head_ref"]
        snapshot_before_proposal = service_u.verify()["active_snapshot_ref"]
        correction_proposal = service_u.propose_correction(
            correction_session["session_id"],
            principal=OWNER,
            target_ref=original_commit["episode_ref"],
            requested_relation="supersedes",
            corrected_or_qualifying_content=CORRECTION,
        )
        after_proposal = service_u.verify()
        correction_consent = service_u.decide_proposal(
            proposal_ref=correction_proposal["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=correction_proposal["canonical_terms_hash"],
        )
        after_consent = service_u.verify()
        correction_commit = service_u.commit_granted_proposal(
            proposal_ref=correction_proposal["proposal_ref"],
            consent_decision_ref=correction_consent["decision_ref"],
        )
        record_count_before_commit_replay = service_u.verify()["record_count"]
        correction_commit_replay = service_u.commit_granted_proposal(
            proposal_ref=correction_proposal["proposal_ref"],
            consent_decision_ref=correction_consent["decision_ref"],
        )
        record_count_after_commit_replay = service_u.verify()["record_count"]
        correction_receipt = service_u.inspect_receipt(
            correction_commit["committed_consent_receipt_ref"]
        )
        close_correction = service_u.close_session(
            correction_session["session_id"], principal=OWNER
        )
        uncorrected_read_set_after = service_u.store.read_record(
            uncorrected["read_set_ref"]
        )
        uncorrected_stage_after = service_u.store.read_record(
            uncorrected["delivery"]["stage_output_ref"]
        )
        instance_u = service_u.runtime_instance_id
        verify_correction = service_u.verify()
        service_u.close()

        service_b = ContinuityService.open(database, checkpoint, key)
        session_b = service_b.open_session(principal=OWNER, intention=INTENTION)
        corrected = service_b.respond(
            session_b["session_id"],
            include_history=True,
            condition_label="corrected_history",
        )
        corrected_ack = acknowledge(service_b, corrected)
        close_b = service_b.close_session(session_b["session_id"], principal=OWNER)
        corrected_read_set = service_b.store.read_record(corrected["read_set_ref"])

        no_history_session = service_b.open_session(principal=OWNER, intention=INTENTION)
        no_history = service_b.respond(
            no_history_session["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        no_history_ack = acknowledge(service_b, no_history)
        close_no_history = service_b.close_session(
            no_history_session["session_id"], principal=OWNER
        )
        no_history_read_set = service_b.store.read_record(no_history["read_set_ref"])

        comparison = service_b.correction_causal_comparison(
            uncorrected, corrected, no_history
        )
        historical_original = service_b.historical_episode(
            original_commit["episode_ref"]
        )
        corrected_record = service_b.store.read_record(correction_commit["correction_ref"])
        verify_b = service_b.verify()
        instance_b = service_b.runtime_instance_id
        service_b.close()

        database_bytes = database.read_bytes()
        checkpoint_bytes = checkpoint.read_bytes()
        prior_session_unchanged = bool(
            uncorrected_read_set_before.record_hash == uncorrected_read_set_after.record_hash
            and uncorrected_read_set_before.payload == uncorrected_read_set_after.payload
            and uncorrected_stage_before.record_hash == uncorrected_stage_after.record_hash
            and uncorrected_stage_before.payload == uncorrected_stage_after.payload
        )
        no_silent_change = bool(
            head_before_proposal == after_proposal["active_head_ref"]
            == after_consent["active_head_ref"]
            and snapshot_before_proposal == after_proposal["active_snapshot_ref"]
            == after_consent["active_snapshot_ref"]
        )

        evidence = {
            "schema_version": "correction-evidence:v0.1",
            "scope": (
                "original committed episode -> visible correction proposal -> exact consent -> "
                "append-only correction commit -> close -> independent corrected session -> "
                "exact paired read set -> matched uncorrected and no-history controls"
            ),
            "synthetic_fixture": {
                "intention": INTENTION,
                "original_episode": ORIGINAL,
                "correction": CORRECTION,
                "requested_relation": "supersedes",
                "contains_nonpublic_source_material": False,
                "public_seed_material_used": False,
            },
            "original_session_a": {
                "open": session_a,
                "precommit_read_set_ref": precommit["read_set_ref"],
                "precommit_selected_refs": precommit["selected_episode_refs"],
                "precommit_response_hash": precommit["response_hash"],
                "precommit_delivery_ack": precommit_ack,
                "visible_proposal": original_proposal,
                "consent": original_consent,
                "commit": original_commit,
                "committed_consent_receipt": original_receipt,
                "close": close_a,
                "verification": verify_a,
            },
            "uncorrected_control": {
                "open": session_u,
                "independent_from_session_a_runtime": instance_a != instance_u,
                "read_set_ref": uncorrected["read_set_ref"],
                "snapshot_ref": uncorrected["snapshot_ref"],
                "selected_refs": uncorrected["selected_episode_refs"],
                "correction_pair_refs": uncorrected["correction_pair_refs"],
                "active_state": uncorrected["active_state"],
                "ordered_context": uncorrected["ordered_context"],
                "response": uncorrected["response"],
                "response_hash": uncorrected["response_hash"],
                "model_call": uncorrected["model_call"],
                "delivery_ack": uncorrected_ack,
                "close": close_u,
            },
            "correction_transaction": {
                "open": correction_session,
                "active_state_before_proposal": state_before_proposal,
                "visible_proposal": correction_proposal,
                "consent": correction_consent,
                "no_active_change_before_commit": no_silent_change,
                "commit": correction_commit,
                "idempotent_commit_replay": correction_commit_replay,
                "commit_replay_added_no_records": (
                    record_count_before_commit_replay == record_count_after_commit_replay
                ),
                "committed_consent_receipt": correction_receipt,
                "correction_record": {
                    "record_ref": corrected_record.record_hash,
                    "record_type": corrected_record.record_type,
                    "record_class": corrected_record.record_class,
                    "payload": corrected_record.payload,
                },
                "close": close_correction,
                "verification": verify_correction,
                "prior_uncorrected_session_records_unchanged": prior_session_unchanged,
            },
            "independent_post_correction_session": {
                "open": session_b,
                "independent_runtime_from_correction_commit": instance_u != instance_b,
                "read_set_ref": corrected["read_set_ref"],
                "read_set_payload": corrected_read_set.payload,
                "selected_refs": corrected["selected_episode_refs"],
                "correction_pair_refs": corrected["correction_pair_refs"],
                "active_state": corrected["active_state"],
                "ordered_context": corrected["ordered_context"],
                "response": corrected["response"],
                "response_hash": corrected["response_hash"],
                "model_call": corrected["model_call"],
                "delivery_ack": corrected_ack,
                "close": close_b,
            },
            "matched_no_history_control": {
                "open": no_history_session,
                "read_set_ref": no_history["read_set_ref"],
                "read_set_payload": no_history_read_set.payload,
                "selected_refs": no_history["selected_episode_refs"],
                "correction_pair_refs": no_history["correction_pair_refs"],
                "active_state": no_history["active_state"],
                "response": no_history["response"],
                "response_hash": no_history["response_hash"],
                "model_call": no_history["model_call"],
                "delivery_ack": no_history_ack,
                "close": close_no_history,
            },
            "historical_original": historical_original,
            "comparison": comparison,
            "storage_evidence": {
                "database_sha256": file_hash(database),
                "checkpoint_sha256": file_hash(checkpoint),
                "database_size_bytes": len(database_bytes),
                "checkpoint_size_bytes": len(checkpoint_bytes),
                "original_plaintext_absent_from_database": ORIGINAL.encode("utf-8") not in database_bytes,
                "correction_plaintext_absent_from_database": CORRECTION.encode("utf-8") not in database_bytes,
                "original_plaintext_absent_from_checkpoint": ORIGINAL.encode("utf-8") not in checkpoint_bytes,
                "correction_plaintext_absent_from_checkpoint": CORRECTION.encode("utf-8") not in checkpoint_bytes,
                "master_key_absent_from_database": key not in database_bytes,
                "master_key_absent_from_checkpoint": key not in checkpoint_bytes,
                "master_key_exported": False,
                "content_and_checkpoint_directories_distinct": database.parent != checkpoint.parent,
            },
            "results": {
                "original_commit_completed": True,
                "correction_commit_completed": True,
                "correction_commit_atomic": correction_commit["atomic"],
                "correction_commit_replay_idempotent": bool(
                    correction_commit_replay["idempotent_replay"]
                    and record_count_before_commit_replay
                    == record_count_after_commit_replay
                    and correction_commit_replay["transaction_id"]
                    == correction_commit["transaction_id"]
                ),
                "independent_post_correction_runtime": instance_u != instance_b,
                "original_record_ref_unchanged": correction_commit[
                    "original_record_ref_unchanged"
                ],
                "original_payload_hash_unchanged": (
                    correction_commit["original_payload_hash_before"]
                    == correction_commit["original_payload_hash_after"]
                    == historical_original["payload_hash"]
                ),
                "original_historically_recoverable": historical_original[
                    "historically_recoverable"
                ],
                "original_cannot_surface_alone_as_current": comparison[
                    "original_cannot_surface_alone_as_current"
                ],
                "uncorrected_session_causality_unchanged": prior_session_unchanged,
                "corrected_pair_entered_exact_read_set": comparison[
                    "original_pair_preserved"
                ],
                "no_history_has_no_correction_callback": comparison[
                    "no_history_has_no_correction_callback"
                ],
                "mechanical_correction_effect_observed": comparison[
                    "correction_effect_observed"
                ],
                "null_result": not comparison["correction_effect_observed"],
                "adverse_result": False,
                "protocol_failure": False,
                "claim_boundary": comparison["claim_boundary"],
            },
            "deviations_from_frozen_specification": [],
            "known_implementation_limitations": [
                (
                    "The response is a deterministic causal probe, not evidence of relationship "
                    "intelligence or model cognition."
                ),
                (
                    "The slice supports one original episode and one correction only; chains of "
                    "later corrections are deliberately refused."
                ),
                (
                    "The 32-byte master key is supplied externally; operating-system or hardware "
                    "key custody is not implemented."
                ),
                (
                    "The separate authenticated checkpoint detects content-only rollback while "
                    "retained, but is not a hardware monotonic counter and cannot detect a "
                    "coordinated rollback of content, checkpoint, and key environment."
                ),
            ],
            "explicit_non_capabilities": [
                "forgetting_or_physical_erasure_expansion",
                "durable_disagreement_or_non_mirroring",
                "semantic_projection",
                "embeddings_or_vector_search",
                "broad_indexing_or_salience",
                "nonpublic_archive_import",
                "public_seed_inheritance_or_deployment",
                "topology_expansion",
                "kernel_efficacy",
                "model_personality_tuning",
                "branch_merge",
                "production_deployment",
            ],
            "next_phase_readiness": {
                "phase": "forgetting",
                "status": "READY_FOR_OWNER_REVIEW_NOT_AUTHORIZED",
                "implementation_started": False,
            },
            "final_verification": verify_b,
        }
        evidence["evidence_hash"] = stable_hash(evidence)
        args.output.write_text(canonical_json(evidence) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "evidence_hash": evidence["evidence_hash"],
                    "correction_effect_observed": comparison[
                        "correction_effect_observed"
                    ],
                    "record_count": verify_b["record_count"],
                    "next_phase": evidence["next_phase_readiness"],
                },
                indent=2,
            )
        )
        return 0 if comparison["correction_effect_observed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
