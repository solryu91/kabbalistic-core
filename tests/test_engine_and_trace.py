from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

from helpers import ArchetypeRegistry

from kabbalistic_core.engine import CognitiveEngine, CycleRequest
from kabbalistic_core.models import (
    Capability,
    ConsentGrant,
    GateDecision,
    Sefirah,
    stable_hash,
)
from kabbalistic_core.permissions import PermissionSet
from kabbalistic_core.trace import (
    TraceIntegrityError,
    validate_completed_trace,
    validate_trace_chain,
)
from kabbalistic_core.yesod import yesod_payload


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = CycleRequest(
            intention="Build one faithful, inspectable cycle.",
            symbol="seed",
            kernel_ids=("clear_sight", "liberation", "regeneration"),
            evidence_refs=("source:one", "source:two"),
        )

    def test_identical_run_is_deterministically_replayable(self) -> None:
        first = CognitiveEngine().run(self.request)
        second = CognitiveEngine().run(self.request)
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.output, second.output)
        self.assertEqual(first.state_hash, second.state_hash)
        self.assertEqual(first.yesod_packet_hash, second.yesod_packet_hash)
        self.assertEqual(first.final_event_hash, second.final_event_hash)
        self.assertEqual(first.events, second.events)

    def test_set_like_request_fields_have_canonical_order(self) -> None:
        registry = ArchetypeRegistry.load_default()
        first = CognitiveEngine().run(
            CycleRequest(
                intention="Canonicalize this cycle.",
                kernel_ids=("regeneration", "clear_sight", "liberation", "clear_sight"),
                evidence_refs=("source:b", "source:a", "source:a"),
            )
        )
        second = CognitiveEngine().run(
            CycleRequest(
                intention="Canonicalize this cycle.",
                kernel_ids=registry.kernel_ids,
                evidence_refs=("source:a", "source:b"),
            )
        )
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.final_event_hash, second.final_event_hash)

    def test_whitespace_equivalent_request_has_identical_full_replay(self) -> None:
        first = CognitiveEngine().run(
            CycleRequest(
                intention="Observe the seed through one bounded cycle.",
                symbol="shared seed",
                kernel_ids=("regeneration", "clear_sight"),
                evidence_refs=("source:a", "source:b"),
                prior_feedback="A bounded observation returned from embodiment.",
                memory_scope="profile:test",
            )
        )
        second = CognitiveEngine().run(
            CycleRequest(
                intention="  Observe   the seed through one bounded cycle.  ",
                symbol=" shared   seed ",
                kernel_ids=("clear_sight", "regeneration", "clear_sight"),
                evidence_refs=("source:b", "source:a", "source:a"),
                prior_feedback=" A bounded   observation returned from embodiment. ",
                memory_scope=" profile:test ",
            )
        )
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.output, second.output)
        self.assertEqual(first.state_hash, second.state_hash)
        self.assertEqual(first.yesod_packet_hash, second.yesod_packet_hash)
        self.assertEqual(first.final_event_hash, second.final_event_hash)
        self.assertEqual(first.events, second.events)

    def test_engine_emits_one_public_voice(self) -> None:
        result = CognitiveEngine().run(self.request)
        output_events = [
            event for event in result.events if event.event_type == "embodied_output_emitted"
        ]
        formalization_events = [
            event for event in result.events if event.event_type == "output_formalized"
        ]
        self.assertEqual(1, len(output_events))
        self.assertEqual(1, len(formalization_events))
        self.assertEqual(1, formalization_events[0].technical["voice_count"])
        self.assertEqual(result.output, output_events[0].technical["output"])
        for heading in ("Form:", "Flow:", "Accord:"):
            self.assertNotIn(heading, result.output)

    def test_complete_main_cycle_executes_in_declared_order(self) -> None:
        engine = CognitiveEngine()
        result = engine.run(self.request)
        targets = [
            Sefirah(event.technical["target"])
            for event in result.events
            if event.event_type == "graph_transition"
        ]
        self.assertEqual(
            [*engine.graph.main_cycle[1:], Sefirah.YESOD],
            targets,
        )
        self.assertEqual(Sefirah.YESOD, result.state.current_node)

    def test_exactly_six_observations_enter_one_yesod_field(self) -> None:
        result = CognitiveEngine().run(self.request)
        field = result.state.yesod_field
        self.assertIsNotNone(field)
        assert field is not None
        self.assertEqual(6, len(result.state.observations))
        self.assertEqual(tuple(result.state.observations), field.mutual_observations)
        self.assertEqual(6, len(field.mutual_observations))

    def test_yesod_packet_hash_is_reproducible_from_its_semantic_payload(self) -> None:
        result = CognitiveEngine().run(self.request)
        field = result.state.yesod_field
        self.assertIsNotNone(field)
        assert field is not None
        self.assertEqual(stable_hash(yesod_payload(field)), field.packet_hash)
        self.assertEqual(field.packet_hash, result.yesod_packet_hash)

    def test_node_results_are_causal_and_hash_their_owned_outputs(self) -> None:
        result = CognitiveEngine().run(self.request)
        node_results = result.state.node_results
        self.assertTrue(node_results)
        for node_result in node_results:
            with self.subTest(node=node_result.node, effect=node_result.observable_effect):
                self.assertEqual(
                    stable_hash(node_result.structured_output),
                    node_result.output_hash,
                )
                self.assertTrue(node_result.structured_output)

        by_node = {}
        for node_result in node_results:
            by_node.setdefault(node_result.node, node_result)
        causal_chain = (
            Sefirah.KETER,
            Sefirah.CHOKHMAH,
            Sefirah.BINAH,
            Sefirah.CHESED,
            Sefirah.GEVURAH,
        )
        self.assertEqual((), by_node[Sefirah.KETER].input_hashes)
        for source, target in zip(causal_chain, causal_chain[1:]):
            self.assertIn(
                by_node[source].output_hash,
                by_node[target].input_hashes,
            )

        self.assertEqual(
            by_node[Sefirah.CHOKHMAH].output_hash,
            by_node[Sefirah.BINAH].structured_output["source_possibility_hash"],
        )
        self.assertEqual(
            by_node[Sefirah.BINAH].output_hash,
            by_node[Sefirah.CHESED].structured_output["source_form_hash"],
        )
        self.assertEqual(
            by_node[Sefirah.CHESED].output_hash,
            by_node[Sefirah.GEVURAH].structured_output["source_expansion_hash"],
        )

    def test_prior_feedback_is_input_to_keter_chokhmah_and_recursion_not_current_feedback(self) -> None:
        feedback = "The prior embodiment revealed a recoverable boundary."
        result = CognitiveEngine().run(
            CycleRequest(
                intention="Revise the next embodiment.",
                symbol="seed",
                prior_feedback=feedback,
            )
        )
        self.assertEqual(feedback, result.state.prior_feedback)
        self.assertEqual("", result.state.embodied_feedback)
        self.assertEqual("pending", result.state.feedback_status)
        chokhmah = next(
            item for item in result.state.node_results if item.node is Sefirah.CHOKHMAH
        )
        candidate_ids = {
            item["candidate_id"] for item in chokhmah.structured_output["possibilities"]
        }
        self.assertIn("integrate_prior_feedback", candidate_ids)
        self.assertEqual(stable_hash(feedback), chokhmah.structured_output["prior_feedback_ref"])
        self.assertIn("prior Malkhut feedback", result.state.recursion_frames[-1].transformed_symbol)

    def test_malkhut_returns_into_a_distinct_post_embodiment_yesod_packet(self) -> None:
        result = CognitiveEngine().run(self.request)
        pre = result.state.pre_embodiment_yesod
        post = result.state.yesod_field
        self.assertIsNotNone(pre)
        self.assertIsNotNone(post)
        assert pre is not None and post is not None
        self.assertEqual("pre_embodiment_context", pre.phase)
        self.assertEqual("post_embodiment_feedback_state", post.phase)
        self.assertIsNone(pre.prior_packet_hash)
        self.assertEqual(pre.packet_hash, post.prior_packet_hash)
        self.assertIsNone(pre.embodied_output_hash)
        self.assertEqual(stable_hash(result.output), post.embodied_output_hash)
        self.assertEqual("pending", post.feedback_status)
        self.assertEqual(result.context_packet_hash, pre.packet_hash)
        self.assertEqual(result.feedback_state_packet_hash, post.packet_hash)
        self.assertNotEqual(pre.packet_hash, post.packet_hash)

        malkhut = next(
            item
            for item in result.state.node_results
            if item.node is Sefirah.MALKHUT
        )
        post_yesod = next(
            item
            for item in reversed(result.state.node_results)
            if item.node is Sefirah.YESOD
        )
        self.assertIn(malkhut.output_hash, post_yesod.input_hashes)

    def test_yesod_is_not_silent_durable_memory(self) -> None:
        result = CognitiveEngine().run(self.request)
        self.assertIsNotNone(result.memory_proposal)
        self.assertEqual([], result.state.durable_memory_hashes)
        self.assertEqual(GateDecision.WITHHELD, result.state.gate_result.decision)

    def test_memory_proposal_is_bound_to_final_yesod_packet_not_live_mind_state(self) -> None:
        result = CognitiveEngine().run(self.request)
        proposal = result.memory_proposal
        post = result.state.yesod_field
        self.assertIsNotNone(proposal)
        self.assertIsNotNone(post)
        assert proposal is not None and post is not None
        self.assertEqual(post.packet_hash, proposal.source_packet_hash)
        self.assertIn(f"yesod:{post.packet_hash}", proposal.provenance_refs)
        self.assertNotEqual(result.state_hash, proposal.source_packet_hash)
        self.assertEqual(GateDecision.WITHHELD, result.state.gate_result.decision)
        self.assertEqual([], result.state.durable_memory_hashes)

    def test_cycle_without_memory_proposal_is_not_applicable_at_daat(self) -> None:
        result = CognitiveEngine().run(
            CycleRequest(
                intention="Run without proposing durable memory.",
                propose_memory=False,
            )
        )
        self.assertIsNone(result.memory_proposal)
        self.assertEqual(GateDecision.NOT_APPLICABLE, result.state.gate_result.decision)
        self.assertEqual([], result.state.durable_memory_hashes)

    def test_receive_feedback_reseals_yesod_and_allows_exact_approval(self) -> None:
        engine = CognitiveEngine(
            permissions=PermissionSet(
                frozenset({Capability.PROPOSE_MEMORY, Capability.COMMIT_MEMORY})
            )
        )
        pending = engine.run(self.request)
        received = engine.receive_feedback(
            pending,
            "  The embodied form remained recognizable and revealed a clearer boundary.  ",
        )
        self.assertEqual("pending", pending.state.feedback_status)
        self.assertEqual("received", received.state.feedback_status)
        self.assertEqual(
            "The embodied form remained recognizable and revealed a clearer boundary.",
            received.state.embodied_feedback,
        )
        self.assertGreater(len(received.events), len(pending.events))
        self.assertEqual(
            pending.feedback_state_packet_hash,
            received.state.yesod_field.prior_packet_hash,
        )
        self.assertIsNotNone(received.state.yesod_field.feedback_ref)
        self.assertEqual(
            received.state.yesod_field.feedback_ref,
            received.memory_proposal.feedback_ref,
        )
        self.assertEqual(
            received.feedback_state_packet_hash,
            received.memory_proposal.source_packet_hash,
        )
        validate_completed_trace(received.state, received.events)

        proposal = received.memory_proposal
        assert proposal is not None
        consent = ConsentGrant(
            proposal_hash=proposal.proposal_hash,
            granted_by="user:local",
            scope=proposal.scope,
            sequence=1,
        )
        gate = engine.evaluate_memory_proposal(
            proposal,
            consent,
            current_packet_hash=received.feedback_state_packet_hash,
        )
        self.assertEqual(GateDecision.APPROVED_FOR_COMMIT, gate.decision)
        self.assertEqual([], received.state.durable_memory_hashes)
        self.assertTrue(any("no commit" in reason for reason in gate.reasons))

    def test_receive_feedback_rejects_blank_or_tampered_completed_state(self) -> None:
        engine = CognitiveEngine()
        pending = engine.run(self.request)
        with self.assertRaisesRegex(ValueError, "cannot be blank"):
            engine.receive_feedback(pending, "   ")

        tampered_state = deepcopy(pending.state)
        tampered_state.output += " forged"
        tampered_result = replace(pending, state=tampered_state)
        with self.assertRaises(TraceIntegrityError):
            engine.receive_feedback(tampered_result, "Observed feedback")

    def test_every_event_has_paired_technical_and_symbolic_records(self) -> None:
        result = CognitiveEngine().run(self.request)
        for event in result.events:
            with self.subTest(sequence=event.sequence, event_type=event.event_type):
                self.assertTrue(event.technical)
                self.assertTrue(event.symbolic)
                self.assertIn("title", event.symbolic)
                self.assertIn("meaning", event.symbolic)


class TraceIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = CognitiveEngine().run(
            CycleRequest(
                intention="Create a trace whose integrity can be checked.",
                symbol="seed",
            )
        )

    def test_unmodified_trace_chain_validates(self) -> None:
        validate_trace_chain(self.result.events)
        validate_completed_trace(self.result.state, self.result.events)

    def test_completed_trace_is_bound_to_exact_final_state(self) -> None:
        altered_state = deepcopy(self.result.state)
        altered_state.output += " tampered after recording"
        with self.assertRaisesRegex(TraceIntegrityError, "final state"):
            validate_completed_trace(altered_state, self.result.events)

    def test_payload_tampering_is_detected(self) -> None:
        events = list(self.result.events)
        target = events[len(events) // 2]
        events[len(events) // 2] = replace(
            target,
            technical={**target.technical, "tampered": True},
        )
        with self.assertRaisesRegex(TraceIntegrityError, "altered"):
            validate_trace_chain(events)

    def test_event_reordering_is_detected(self) -> None:
        events = list(self.result.events)
        events[1], events[2] = events[2], events[1]
        with self.assertRaises(TraceIntegrityError):
            validate_trace_chain(events)

    def test_event_deletion_is_detected(self) -> None:
        events = list(self.result.events)
        del events[len(events) // 2]
        with self.assertRaises(TraceIntegrityError):
            validate_trace_chain(events)

    def test_previous_event_hash_tampering_is_detected(self) -> None:
        events = list(self.result.events)
        events[-1] = replace(events[-1], previous_event_hash="f" * 64)
        with self.assertRaisesRegex(TraceIntegrityError, "previous-event link"):
            validate_trace_chain(events)

    def test_state_link_tampering_is_detected(self) -> None:
        events = list(self.result.events)
        target = events[-1]
        forged = replace(target, state_hash_before="0" * 64, event_hash="")
        forged = replace(forged, event_hash=stable_hash(forged.hash_payload()))
        events[-1] = forged
        with self.assertRaisesRegex(TraceIntegrityError, "semantic-state link"):
            validate_trace_chain(events)


if __name__ == "__main__":
    unittest.main()
