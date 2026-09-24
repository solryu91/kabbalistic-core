from __future__ import annotations

import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from helpers import SRC_ROOT  # noqa: F401  # ensures the local src tree is importable

from kabbalistic_core.continuity import (
    ContinuityProtocolError,
    ContinuityService,
)
from kabbalistic_core.continuity_store import (
    CheckpointMismatchError,
    PayloadUnavailableError,
    RecordIntegrityError,
)
from kabbalistic_core.continuity_cli import build_parser


OWNER = "local-owner:test"
INTENTION = "Help me choose a small reversible next step without adding outside obligations."
EPISODE = "When planning with me, keep the first step under fifteen minutes and reversible."


def acknowledge(service: ContinuityService, result: dict) -> dict:
    return service.acknowledge_delivery(
        result["session_id"],
        delivery_id=result["delivery"]["delivery_id"],
        response_hash=result["response_hash"],
    )


class ContinuityVerticalSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "content" / "continuity.sqlite3"
        self.checkpoint = self.root / "authority" / "authority.checkpoint.json"
        self.key = os.urandom(32)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create_service(self) -> ContinuityService:
        return ContinuityService.create(
            self.database,
            self.checkpoint,
            self.key,
            owner_principal=OWNER,
        )

    def test_cli_help_is_safe_for_the_default_windows_console_encoding(self) -> None:
        help_text = build_parser().format_help()
        help_text.encode("cp1252")
        self.assertIn("Session A -> consented commit -> independent Session B", help_text)

    def perform_commit(self, service: ContinuityService) -> tuple[dict, dict, dict, dict]:
        session = service.open_session(principal=OWNER, intention=INTENTION)
        before = service.respond(
            session["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        acknowledge(service, before)
        proposal = service.propose_episode(
            session["session_id"], principal=OWNER, content=EPISODE
        )
        consent = service.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        commit = service.commit_granted_proposal(
            proposal_ref=proposal["proposal_ref"],
            consent_decision_ref=consent["decision_ref"],
        )
        close = service.close_session(session["session_id"], principal=OWNER)
        return before, proposal, commit, close

    def test_complete_session_a_to_independent_session_b_causal_proof(self) -> None:
        service_a = self.create_service()
        instance_a = service_a.runtime_instance_id
        before, proposal, commit, close_a = self.perform_commit(service_a)

        self.assertEqual([], before["selected_episode_refs"])
        read_set_a = service_a.store.read_record(before["read_set_ref"])
        self.assertEqual([], read_set_a.payload["selected_memory_refs_in_order"])
        self.assertEqual(commit["after_head_ref"], close_a["observed_lineage_head_ref_at_close"])
        self.assertEqual([commit["memory_commit_ref"]], close_a["memory_commit_refs"])
        self.assertTrue(commit["atomic"])
        self.assertTrue(commit["proposal_staging_payload_erased"])
        with self.assertRaises(PayloadUnavailableError):
            service_a.store.read_record(proposal["proposal_ref"])
        receipt = service_a.inspect_receipt(commit["committed_consent_receipt_ref"])
        self.assertEqual(proposal["proposal_content_root"], receipt["proposal_content_root"])
        self.assertEqual([], receipt["canonical_allowed_derivations"])
        service_a.close()

        service_b = ContinuityService.open(self.database, self.checkpoint, self.key)
        instance_b = service_b.runtime_instance_id
        self.assertNotEqual(instance_a, instance_b)
        session_b = service_b.open_session(principal=OWNER, intention=INTENTION)
        self.assertEqual(commit["after_snapshot_ref"], session_b["start_snapshot_ref"])

        history = service_b.respond(
            session_b["session_id"], include_history=True, condition_label="history"
        )
        acknowledge(service_b, history)
        close_b = service_b.close_session(session_b["session_id"], principal=OWNER)
        control_session = service_b.open_session(principal=OWNER, intention=INTENTION)
        no_history = service_b.respond(
            control_session["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        acknowledge(service_b, no_history)
        close_control = service_b.close_session(
            control_session["session_id"], principal=OWNER
        )
        comparison = service_b.causal_comparison(history, no_history)

        self.assertEqual([commit["episode_ref"]], history["selected_episode_refs"])
        self.assertEqual([], no_history["selected_episode_refs"])
        self.assertIn(EPISODE, history["response"])
        self.assertNotIn(EPISODE, no_history["response"])
        self.assertNotEqual(session_b["session_id"], control_session["session_id"])
        self.assertTrue(comparison["same_start_snapshot"])
        self.assertTrue(comparison["same_provider_and_budget"])
        self.assertTrue(comparison["response_hashes_differ"])
        self.assertTrue(comparison["history_effect_observed"])
        self.assertEqual("mechanical_continuity_effect", comparison["result"])

        read_set = service_b.store.read_record(history["read_set_ref"])
        self.assertTrue(read_set.payload["sealed_immediately_before_stage"])
        self.assertEqual(
            [commit["episode_ref"]], read_set.payload["selected_memory_refs_in_order"]
        )
        self.assertEqual(
            history["rendered_context_hash"], read_set.payload["rendered_context_hash"]
        )
        output_ref = history["delivery"]["stage_output_ref"]
        output_binding = service_b.store.read_record(output_ref)
        self.assertEqual(history["read_set_ref"], output_binding.payload["causal_read_set_ref"])
        self.assertEqual("volatile_dependency_only", output_binding.payload["capture_mode"])
        self.assertIsNone(output_binding.payload["output_payload_locator"])
        self.assertEqual("acknowledged", close_b["delivery_status"])
        self.assertEqual("acknowledged", close_control["delivery_status"])
        self.assertTrue(service_b.verify()["chain_valid"])
        service_b.close()

    def test_proposal_and_grant_do_not_silently_change_active_memory(self) -> None:
        service = self.create_service()
        session = service.open_session(principal=OWNER, intention=INTENTION)
        initial_head = session["start_head_ref"]
        proposal = service.propose_episode(
            session["session_id"], principal=OWNER, content=EPISODE
        )
        self.assertEqual((), service.active_episode_refs())
        consent = service.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        self.assertTrue(consent["commit_authorized"])
        self.assertFalse(consent["active_memory_changed"])
        self.assertEqual((), service.active_episode_refs())
        self.assertEqual(initial_head, service.store.latest_head().record_hash)
        service.close()

    def test_consent_substitution_and_wrong_owner_fail_closed(self) -> None:
        service = self.create_service()
        session = service.open_session(principal=OWNER, intention=INTENTION)
        proposal = service.propose_episode(
            session["session_id"], principal=OWNER, content=EPISODE
        )
        with self.assertRaises(ContinuityProtocolError):
            service.decide_proposal(
                proposal_ref=proposal["proposal_ref"],
                principal=OWNER,
                decision="grant",
                canonical_terms_hash="0" * 64,
            )
        with self.assertRaises(PermissionError):
            service.decide_proposal(
                proposal_ref=proposal["proposal_ref"],
                principal="someone-else",
                decision="grant",
                canonical_terms_hash=proposal["canonical_terms_hash"],
            )
        self.assertEqual((), service.active_episode_refs())
        service.close()

    def test_rejection_erases_staging_payload_without_changing_head(self) -> None:
        service = self.create_service()
        session = service.open_session(principal=OWNER, intention=INTENTION)
        initial_head = service.store.latest_head().record_hash
        proposal = service.propose_episode(
            session["session_id"], principal=OWNER, content=EPISODE
        )
        decision = service.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=OWNER,
            decision="reject",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        self.assertEqual("reject", decision["decision"])
        self.assertEqual(initial_head, service.store.latest_head().record_hash)
        self.assertEqual((), service.active_episode_refs())
        with self.assertRaises(PayloadUnavailableError):
            service.store.read_record(proposal["proposal_ref"])
        close = service.close_session(session["session_id"], principal=OWNER)
        close_record = service.store.read_record(close["session_close_ref"])
        self.assertEqual(
            "REJECTED", close_record.payload["proposal_status_at_close"][0]["reducer_status"]
        )
        service.close()

    def test_payload_is_encrypted_and_master_key_is_external(self) -> None:
        service = self.create_service()
        _, _, commit, _ = self.perform_commit(service)
        database_bytes = self.database.read_bytes()
        checkpoint_bytes = self.checkpoint.read_bytes()
        self.assertNotIn(EPISODE.encode("utf-8"), database_bytes)
        self.assertNotIn(self.key, database_bytes)
        self.assertNotIn(self.key, checkpoint_bytes)
        self.assertNotIn(OWNER.encode("utf-8"), database_bytes)
        self.assertNotIn(OWNER.encode("utf-8"), checkpoint_bytes)
        self.assertEqual([commit["episode_ref"]], list(service.active_episode_refs()))
        service.close()
        wrong_key = os.urandom(32)
        with self.assertRaises(CheckpointMismatchError):
            ContinuityService.open(self.database, self.checkpoint, wrong_key)

    def test_append_only_records_reject_update_and_delete(self) -> None:
        service = self.create_service()
        record_ref = service.store.latest_head().record_hash
        connection = sqlite3.connect(self.database)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE records SET record_type = 'tampered' WHERE record_hash = ?",
                    (record_ref,),
                )
            connection.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("DELETE FROM records WHERE record_hash = ?", (record_ref,))
        finally:
            connection.close()
            service.close()

    def test_failed_transaction_leaves_chain_and_checkpoint_unchanged(self) -> None:
        service = self.create_service()
        before = service.verify()
        with self.assertRaises(RuntimeError):
            with service.store.write_batch() as batch:
                batch.append(
                    "TestOnlyRecord",
                    "G",
                    {"relationship_retrieval_eligible": False},
                )
                raise RuntimeError("injected failure before checkpoint prepare")
        after = service.verify()
        self.assertEqual(before["record_count"], after["record_count"])
        self.assertEqual(before["record_chain_head_hash"], after["record_chain_head_hash"])
        self.assertEqual(
            before["authority_checkpoint_hash"], after["authority_checkpoint_hash"]
        )
        service.close()

    def test_checkpoint_finalize_failure_leaves_store_pending_and_fail_closed(self) -> None:
        service = self.create_service()
        original_write = service.store._write_checkpoint

        def fail_finalize(checkpoint):
            if checkpoint.pending_or_finalized == "pending":
                original_write(checkpoint)
                return
            raise OSError("injected authority-checkpoint finalization failure")

        with patch.object(service.store, "_write_checkpoint", side_effect=fail_finalize):
            with self.assertRaises(OSError):
                service.open_session(principal=OWNER, intention=INTENTION)
        service.close()
        with self.assertRaises(CheckpointMismatchError):
            ContinuityService.open(self.database, self.checkpoint, self.key)

    def test_stale_content_backup_is_rejected_by_current_checkpoint(self) -> None:
        service = self.create_service()
        stale_database = self.root / "stale-content.sqlite3"
        shutil.copy2(self.database, stale_database)
        self.perform_commit(service)
        service.close()
        with self.assertRaises(CheckpointMismatchError):
            ContinuityService.open(stale_database, self.checkpoint, self.key)

    def test_content_and_authority_checkpoint_require_distinct_directories(self) -> None:
        shared_directory = self.root / "shared-domain"
        with self.assertRaises(ValueError):
            ContinuityService.create(
                shared_directory / "continuity.sqlite3",
                shared_directory / "authority.checkpoint.json",
                self.key,
                owner_principal=OWNER,
            )

    def test_physical_ciphertext_tampering_fails_closed_on_reopen(self) -> None:
        service = self.create_service()
        _, _, commit, _ = self.perform_commit(service)
        service.close()
        connection = sqlite3.connect(self.database)
        try:
            connection.execute("DROP TRIGGER records_no_update")
            connection.execute(
                "UPDATE records SET payload_ciphertext = ? WHERE record_hash = ?",
                (os.urandom(64), commit["episode_ref"]),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(RecordIntegrityError):
            ContinuityService.open(self.database, self.checkpoint, self.key)

    def test_vertical_slice_refuses_second_episode(self) -> None:
        service = self.create_service()
        self.perform_commit(service)
        session = service.open_session(principal=OWNER, intention=INTENTION)
        proposal = service.propose_episode(
            session["session_id"],
            principal=OWNER,
            content="A second memory must wait for the next reviewed phase.",
        )
        consent = service.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        with self.assertRaises(ContinuityProtocolError):
            service.commit_granted_proposal(
                proposal_ref=proposal["proposal_ref"],
                consent_decision_ref=consent["decision_ref"],
            )
        self.assertEqual(1, len(service.active_episode_refs()))
        service.close()

    def test_completed_episode_commit_retry_returns_existing_transaction(self) -> None:
        service = self.create_service()
        _, proposal, commit, _ = self.perform_commit(service)
        consent_ref = commit["consent_decision_ref"]
        replay = service.commit_granted_proposal(
            proposal_ref=proposal["proposal_ref"],
            consent_decision_ref=consent_ref,
        )
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(commit["memory_commit_ref"], replay["memory_commit_ref"])
        self.assertEqual(commit["episode_ref"], replay["episode_ref"])
        self.assertEqual(commit["transaction_id"], replay["transaction_id"])
        self.assertEqual(1, len(service.store.list_record_refs(record_type="EpisodicEventRecord")))
        service.close()


if __name__ == "__main__":
    unittest.main()
