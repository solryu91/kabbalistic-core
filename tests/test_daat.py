from __future__ import annotations

from dataclasses import replace
import unittest

from helpers import exact_consent, memory_permissions, valid_memory_proposal

from kabbalistic_core.daat import DaatGate
from kabbalistic_core.models import ConsentGrant, GateDecision, stable_hash
from kabbalistic_core.permissions import PermissionSet


class DaatGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = DaatGate()
        self.proposal = valid_memory_proposal()
        self.permissions = memory_permissions()

    def evaluate(self, proposal=None, consent=None, *, current_state_hash=None):
        active_proposal = self.proposal if proposal is None else proposal
        return self.gate.evaluate(
            active_proposal,
            consent,
            self.permissions,
            current_packet_hash=current_state_hash or active_proposal.source_packet_hash,
        )

    def test_no_proposal_is_not_applicable(self) -> None:
        result = self.gate.evaluate(
            None,
            None,
            self.permissions,
            current_packet_hash=stable_hash("current"),
        )
        self.assertEqual(GateDecision.NOT_APPLICABLE, result.decision)

    def test_exact_packet_consent_provenance_and_projection_approve_for_commit(self) -> None:
        result = self.evaluate(consent=exact_consent(self.proposal))
        self.assertEqual(GateDecision.APPROVED_FOR_COMMIT, result.decision)
        self.assertEqual(self.proposal.proposal_hash, result.proposal_hash)
        self.assertTrue(any("no commit has occurred" in reason for reason in result.reasons))

    def test_missing_consent_withholds(self) -> None:
        result = self.evaluate()
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("consent" in reason.casefold() for reason in result.reasons))

    def test_consent_for_different_proposal_withholds(self) -> None:
        consent = replace(exact_consent(self.proposal), proposal_hash="f" * 64)
        result = self.evaluate(consent=consent)
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("different proposal" in reason for reason in result.reasons))

    def test_consent_scope_must_match_exactly(self) -> None:
        consent = replace(exact_consent(self.proposal), scope="profile:someone-else")
        result = self.evaluate(consent=consent)
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("scope" in reason.casefold() for reason in result.reasons))

    def test_consent_sequence_must_be_positive(self) -> None:
        consent = ConsentGrant(
            proposal_hash=self.proposal.proposal_hash,
            granted_by="user:test",
            scope=self.proposal.scope,
            sequence=0,
        )
        result = self.evaluate(consent=consent)
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("positive" in reason.casefold() for reason in result.reasons))

    def test_missing_provenance_withholds(self) -> None:
        proposal = replace(self.proposal, provenance_refs=())
        result = self.evaluate(proposal=proposal, consent=exact_consent(proposal))
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("provenance" in reason.casefold() for reason in result.reasons))

    def test_stale_before_state_withholds(self) -> None:
        result = self.evaluate(
            consent=exact_consent(self.proposal),
            current_state_hash=stable_hash("different sealed packet"),
        )
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("current sealed Yesod packet" in reason for reason in result.reasons))

    def test_unreproducible_after_state_withholds(self) -> None:
        proposal = replace(self.proposal, projected_memory_hash="0" * 64)
        result = self.evaluate(proposal=proposal, consent=exact_consent(proposal))
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("not reproducible" in reason for reason in result.reasons))

    def test_missing_embodied_feedback_reference_withholds(self) -> None:
        proposal = replace(self.proposal, feedback_ref=None)
        result = self.evaluate(proposal=proposal, consent=exact_consent(proposal))
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("feedback" in reason.casefold() for reason in result.reasons))

    def test_missing_capabilities_withhold(self) -> None:
        result = self.gate.evaluate(
            self.proposal,
            exact_consent(self.proposal),
            PermissionSet(frozenset()),
            current_packet_hash=self.proposal.source_packet_hash,
        )
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        joined = " ".join(result.reasons).casefold()
        self.assertIn("propose", joined)
        self.assertIn("commit", joined)

    def test_low_significance_withholds(self) -> None:
        proposal = replace(self.proposal, significance=0.69)
        result = self.evaluate(proposal=proposal, consent=exact_consent(proposal))
        self.assertEqual(GateDecision.WITHHELD, result.decision)
        self.assertTrue(any("significance" in reason.casefold() for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
