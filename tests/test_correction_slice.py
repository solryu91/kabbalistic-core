from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helpers import SRC_ROOT  # noqa: F401

from kabbalistic_core.continuity import ContinuityProtocolError, ContinuityService
from kabbalistic_core.continuity_store import PayloadUnavailableError


OWNER = "local-owner:synthetic-correction"
INTENTION = "Help me choose a small reversible next step without adding outside obligations."
ORIGINAL = "When planning with me, keep the first step under fifteen minutes and reversible."
CORRECTION = (
    "The under-fifteen-minute constraint was temporary. "
    "Do not treat it as a current planning requirement."
)


def acknowledge(service: ContinuityService, result: dict) -> dict:
    return service.acknowledge_delivery(
        result["session_id"],
        delivery_id=result["delivery"]["delivery_id"],
        response_hash=result["response_hash"],
    )


class CorrectionSliceTests(unittest.TestCase):
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

    def commit_original(self, service: ContinuityService) -> tuple[dict, dict]:
        session = service.open_session(principal=OWNER, intention=INTENTION)
        precommit = service.respond(
            session["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        acknowledge(service, precommit)
        proposal = service.propose_episode(
            session["session_id"], principal=OWNER, content=ORIGINAL
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
        return commit, close

    def propose_and_commit_correction(
        self,
        service: ContinuityService,
        original_ref: str,
        *,
        relation: str = "supersedes",
    ) -> tuple[dict, dict, dict, dict]:
        session = service.open_session(principal=OWNER, intention=INTENTION)
        proposal = service.propose_correction(
            session["session_id"],
            principal=OWNER,
            target_ref=original_ref,
            requested_relation=relation,
            corrected_or_qualifying_content=CORRECTION,
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
        return session, proposal, consent, {**commit, "close": close}

    def test_complete_append_only_correction_causal_proof(self) -> None:
        service_a = self.create_service()
        original_commit, original_close = self.commit_original(service_a)
        original_ref = original_commit["episode_ref"]
        instance_a = service_a.runtime_instance_id
        service_a.close()

        service_uncorrected = ContinuityService.open(
            self.database, self.checkpoint, self.key
        )
        self.assertNotEqual(instance_a, service_uncorrected.runtime_instance_id)
        uncorrected_session = service_uncorrected.open_session(
            principal=OWNER, intention=INTENTION
        )
        uncorrected = service_uncorrected.respond(
            uncorrected_session["session_id"],
            include_history=True,
            condition_label="uncorrected_history",
        )
        acknowledge(service_uncorrected, uncorrected)
        uncorrected_close = service_uncorrected.close_session(
            uncorrected_session["session_id"], principal=OWNER
        )
        uncorrected_read_set_before = service_uncorrected.store.read_record(
            uncorrected["read_set_ref"]
        )
        uncorrected_stage_before = service_uncorrected.store.read_record(
            uncorrected["delivery"]["stage_output_ref"]
        )

        correction_session, proposal, consent, correction_commit = (
            self.propose_and_commit_correction(service_uncorrected, original_ref)
        )
        corrected_snapshot = correction_commit["after_snapshot_ref"]
        original_payload_hash = correction_commit["original_payload_hash_before"]
        instance_uncorrected = service_uncorrected.runtime_instance_id
        service_uncorrected.close()

        service_b = ContinuityService.open(self.database, self.checkpoint, self.key)
        self.assertNotEqual(instance_uncorrected, service_b.runtime_instance_id)
        corrected_session = service_b.open_session(principal=OWNER, intention=INTENTION)
        corrected = service_b.respond(
            corrected_session["session_id"],
            include_history=True,
            condition_label="corrected_history",
        )
        acknowledge(service_b, corrected)
        service_b.close_session(corrected_session["session_id"], principal=OWNER)

        no_history_session = service_b.open_session(principal=OWNER, intention=INTENTION)
        no_history = service_b.respond(
            no_history_session["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        acknowledge(service_b, no_history)
        service_b.close_session(no_history_session["session_id"], principal=OWNER)

        comparison = service_b.correction_causal_comparison(
            uncorrected, corrected, no_history
        )
        historical = service_b.historical_episode(original_ref)
        corrected_read_set = service_b.store.read_record(corrected["read_set_ref"])
        uncorrected_read_set_after = service_b.store.read_record(
            uncorrected["read_set_ref"]
        )
        uncorrected_stage_after = service_b.store.read_record(
            uncorrected["delivery"]["stage_output_ref"]
        )

        self.assertTrue(comparison["correction_effect_observed"])
        self.assertEqual("mechanical_correction_effect", comparison["result"])
        self.assertEqual("active", comparison["uncorrected_status"])
        self.assertEqual("superseded", comparison["corrected_status"])
        self.assertEqual(corrected_snapshot, corrected["snapshot_ref"])
        self.assertEqual(corrected_snapshot, no_history["snapshot_ref"])
        self.assertEqual([original_ref], uncorrected["selected_episode_refs"])
        self.assertEqual(
            [original_ref, correction_commit["correction_ref"]],
            corrected["selected_episode_refs"],
        )
        self.assertEqual([], no_history["selected_episode_refs"])
        self.assertTrue(corrected["correction_pair_refs"])
        self.assertFalse(
            corrected["ordered_context"][0]["may_surface_alone_as_current"]
        )
        self.assertIn("now superseded", corrected["response"])
        self.assertIn(
            "will not treat the original constraint as a current planning requirement",
            corrected["response"],
        )
        self.assertNotIn(CORRECTION, no_history["response"])
        self.assertEqual("superseded", historical["current_status"])
        self.assertTrue(historical["historically_recoverable"])
        self.assertFalse(historical["may_surface_alone_as_current"])
        self.assertEqual(original_payload_hash, historical["payload_hash"])
        self.assertEqual(
            uncorrected_read_set_before.record_hash,
            uncorrected_read_set_after.record_hash,
        )
        self.assertEqual(
            uncorrected_read_set_before.payload,
            uncorrected_read_set_after.payload,
        )
        self.assertEqual(uncorrected_stage_before.payload, uncorrected_stage_after.payload)
        self.assertEqual(
            [original_ref],
            uncorrected_read_set_after.payload["selected_memory_refs_in_order"],
        )
        self.assertEqual(
            [
                {
                    "target_ref": original_ref,
                    "correction_ref": correction_commit["correction_ref"],
                    "requested_relation": "supersedes",
                    "computed_target_status": "superseded",
                }
            ],
            corrected_read_set.payload["correction_pair_refs"],
        )
        self.assertEqual(original_ref, correction_commit["target_ref"])
        self.assertTrue(correction_commit["original_record_ref_unchanged"])
        self.assertEqual(
            correction_commit["original_payload_hash_before"],
            correction_commit["original_payload_hash_after"],
        )
        self.assertEqual("complete", original_close["completion_status"])
        self.assertEqual("complete", uncorrected_close["completion_status"])
        self.assertEqual("correct_or_supersede", proposal["proposal_kind"])
        self.assertEqual("grant", consent["decision"])
        self.assertEqual("complete", correction_commit["close"]["completion_status"])
        self.assertTrue(service_b.verify()["record_count"] > 0)
        service_b.close()

    def test_reducer_supports_qualification_dispute_and_supersession(self) -> None:
        expected = {
            "qualifies": ("qualified", 2, 0),
            "disputes": ("disputed", 1, 1),
            "supersedes": ("superseded", 1, 1),
        }
        for relation, (status, active_count, inactive_count) in expected.items():
            with self.subTest(relation=relation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                service = ContinuityService.create(
                    root / "content" / "continuity.sqlite3",
                    root / "authority" / "authority.checkpoint.json",
                    os.urandom(32),
                    owner_principal=OWNER,
                )
                original_commit, _ = self.commit_original(service)
                _, _, _, correction = self.propose_and_commit_correction(
                    service, original_commit["episode_ref"], relation=relation
                )
                view = service.reduce_active_state()
                self.assertEqual(status, view.original_status)
                self.assertEqual(relation, view.correction_relation)
                self.assertEqual(active_count, len(view.active_refs))
                self.assertEqual(inactive_count, len(view.inactive_or_disputed_refs))
                self.assertFalse(view.as_dict()["original_may_surface_alone_as_current"])
                self.assertEqual(
                    correction["correction_ref"], view.correction_ref
                )
                service.close()

    def test_correction_proposal_and_grant_do_not_change_active_state(self) -> None:
        service = self.create_service()
        original_commit, _ = self.commit_original(service)
        original_ref = original_commit["episode_ref"]
        session = service.open_session(principal=OWNER, intention=INTENTION)
        before = service.verify()
        proposal = service.propose_correction(
            session["session_id"],
            principal=OWNER,
            target_ref=original_ref,
            requested_relation="supersedes",
            corrected_or_qualifying_content=CORRECTION,
        )
        after_proposal = service.verify()
        consent = service.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        after_grant = service.verify()
        self.assertEqual(before["active_head_ref"], after_proposal["active_head_ref"])
        self.assertEqual(before["active_snapshot_ref"], after_proposal["active_snapshot_ref"])
        self.assertEqual(before["active_head_ref"], after_grant["active_head_ref"])
        self.assertEqual(before["active_snapshot_ref"], after_grant["active_snapshot_ref"])
        self.assertEqual("active", service.reduce_active_state().original_status)
        self.assertFalse(proposal["active_memory_changed"])
        self.assertFalse(consent["active_memory_changed"])
        service.close()

    def test_rejected_correction_erases_staging_without_changing_original(self) -> None:
        service = self.create_service()
        original_commit, _ = self.commit_original(service)
        original_ref = original_commit["episode_ref"]
        historical_before = service.historical_episode(original_ref)
        session = service.open_session(principal=OWNER, intention=INTENTION)
        proposal = service.propose_correction(
            session["session_id"],
            principal=OWNER,
            target_ref=original_ref,
            requested_relation="disputes",
            corrected_or_qualifying_content=CORRECTION,
        )
        decision = service.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=OWNER,
            decision="reject",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        with self.assertRaises(PayloadUnavailableError):
            service.store.read_record(proposal["proposal_ref"])
        historical_after = service.historical_episode(original_ref)
        self.assertEqual("reject", decision["decision"])
        self.assertEqual("active", historical_after["current_status"])
        self.assertEqual(
            historical_before["payload_hash"], historical_after["payload_hash"]
        )
        self.assertFalse(service.store.list_record_refs(record_type="CorrectionRecord"))
        service.close()

    def test_consent_for_another_correction_cannot_be_substituted(self) -> None:
        service = self.create_service()
        original_commit, _ = self.commit_original(service)
        original_ref = original_commit["episode_ref"]
        session = service.open_session(principal=OWNER, intention=INTENTION)
        first = service.propose_correction(
            session["session_id"],
            principal=OWNER,
            target_ref=original_ref,
            requested_relation="supersedes",
            corrected_or_qualifying_content=CORRECTION,
        )
        second = service.propose_correction(
            session["session_id"],
            principal=OWNER,
            target_ref=original_ref,
            requested_relation="supersedes",
            corrected_or_qualifying_content="A different correction that was not granted.",
        )
        consent = service.decide_proposal(
            proposal_ref=first["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=first["canonical_terms_hash"],
        )
        with self.assertRaises(ContinuityProtocolError):
            service.commit_granted_proposal(
                proposal_ref=second["proposal_ref"],
                consent_decision_ref=consent["decision_ref"],
            )
        self.assertEqual("active", service.reduce_active_state().original_status)
        service.close()

    def test_failed_correction_transaction_keeps_original_active(self) -> None:
        service = self.create_service()
        original_commit, _ = self.commit_original(service)
        original_ref = original_commit["episode_ref"]
        session = service.open_session(principal=OWNER, intention=INTENTION)
        proposal = service.propose_correction(
            session["session_id"],
            principal=OWNER,
            target_ref=original_ref,
            requested_relation="supersedes",
            corrected_or_qualifying_content=CORRECTION,
        )
        consent = service.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=OWNER,
            decision="grant",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        before = service.verify()
        with patch.object(
            service,
            "_append_transition",
            side_effect=RuntimeError("injected correction commit failure"),
        ):
            with self.assertRaises(RuntimeError):
                service.commit_granted_proposal(
                    proposal_ref=proposal["proposal_ref"],
                    consent_decision_ref=consent["decision_ref"],
                )
        after = service.verify()
        self.assertEqual(before["active_head_ref"], after["active_head_ref"])
        self.assertEqual(before["active_snapshot_ref"], after["active_snapshot_ref"])
        self.assertEqual("active", service.reduce_active_state().original_status)
        self.assertFalse(service.store.list_record_refs(record_type="CorrectionRecord"))
        service.close()

    def test_second_correction_is_refused_by_slice_boundary(self) -> None:
        service = self.create_service()
        original_commit, _ = self.commit_original(service)
        original_ref = original_commit["episode_ref"]
        self.propose_and_commit_correction(service, original_ref)
        session = service.open_session(principal=OWNER, intention=INTENTION)
        with self.assertRaises(ContinuityProtocolError):
            service.propose_correction(
                session["session_id"],
                principal=OWNER,
                target_ref=original_ref,
                requested_relation="qualifies",
                corrected_or_qualifying_content="A second correction is out of scope.",
            )
        service.close()

    def test_completed_correction_retry_returns_the_existing_transaction(self) -> None:
        service = self.create_service()
        original_commit, _ = self.commit_original(service)
        original_ref = original_commit["episode_ref"]
        _, proposal, consent, committed = self.propose_and_commit_correction(
            service, original_ref
        )
        record_count_before_replay = service.verify()["record_count"]
        replay = service.commit_granted_proposal(
            proposal_ref=proposal["proposal_ref"],
            consent_decision_ref=consent["decision_ref"],
        )
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(committed["memory_commit_ref"], replay["memory_commit_ref"])
        self.assertEqual(committed["correction_ref"], replay["correction_ref"])
        self.assertEqual(committed["transaction_id"], replay["transaction_id"])
        self.assertEqual(
            committed["committed_consent_receipt_ref"],
            replay["committed_consent_receipt_ref"],
        )
        self.assertEqual(
            committed["original_payload_hash_before"],
            replay["original_payload_hash_before"],
        )
        self.assertEqual(
            committed["original_payload_hash_after"],
            replay["original_payload_hash_after"],
        )
        self.assertEqual(record_count_before_replay, service.verify()["record_count"])
        self.assertEqual(1, len(service.store.list_record_refs(record_type="CorrectionRecord")))
        service.close()

    def test_original_and_correction_plaintext_are_encrypted_at_rest(self) -> None:
        service = self.create_service()
        original_commit, _ = self.commit_original(service)
        self.propose_and_commit_correction(service, original_commit["episode_ref"])
        service.close()

        database_bytes = self.database.read_bytes()
        checkpoint_bytes = self.checkpoint.read_bytes()
        for plaintext in (ORIGINAL.encode("utf-8"), CORRECTION.encode("utf-8")):
            self.assertNotIn(plaintext, database_bytes)
            self.assertNotIn(plaintext, checkpoint_bytes)
        self.assertNotIn(self.key, database_bytes)
        self.assertNotIn(self.key, checkpoint_bytes)

    def test_correction_schema_is_model_agnostic_and_grants_no_authority(self) -> None:
        service = self.create_service()
        original_commit, _ = self.commit_original(service)
        original_ref = original_commit["episode_ref"]
        _, _, _, correction = self.propose_and_commit_correction(service, original_ref)
        record = service.store.read_record(correction["correction_ref"])
        forbidden = {
            "provider_id",
            "model_id",
            "qwen",
            "tool_permissions",
            "storage_authority",
            "external_action_authority",
            "covenant_authority",
        }
        self.assertTrue(forbidden.isdisjoint(record.payload))
        self.assertEqual("CorrectionRecord", record.record_type)
        self.assertEqual("E", record.record_class)
        self.assertEqual([original_ref], record.payload["target_refs"])
        self.assertEqual(
            "owner_attributed_correction_not_objective_fact",
            record.payload["epistemic_status"],
        )
        service.close()


if __name__ == "__main__":
    unittest.main()
