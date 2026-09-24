from __future__ import annotations

from copy import deepcopy
import json
import unittest
from typing import Any, Callable

from helpers import SRC_ROOT  # noqa: F401 - importing helpers bootstraps src/

from kabbalistic_core.models import CoreId, stable_hash
from kabbalistic_core.neural import (
    LMStudioAdapter,
    LMStudioConfig,
    ModelProtocolError,
    ModelUnavailableError,
    NeuralCoreProposal,
)
from kabbalistic_core.poc import SeedService
from kabbalistic_core.retrieval import CuratedCorpus


MODEL_ID = "local/Qwen3-8B-test"


class FakeLMStudioTransport:
    """Schema-aware fake that exercises the adapter rather than replacing it."""

    def __init__(
        self,
        *,
        offline: bool = False,
        fail_post_number: int | None = None,
        mutate: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        served_model_id: str = MODEL_ID,
    ) -> None:
        self.offline = offline
        self.fail_post_number = fail_post_number
        self.mutate = mutate
        self.served_model_id = served_model_id
        self.requests: list[dict[str, Any]] = []
        self.post_count = 0

    def request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "payload": deepcopy(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        if method == "GET":
            if self.offline:
                raise ModelUnavailableError("fake local model is offline")
            return {"data": [{"id": self.served_model_id}]}

        self.post_count += 1
        if self.fail_post_number == self.post_count:
            raise ModelUnavailableError("fake local model disconnected")
        if payload is None:
            raise AssertionError("A completion request must have a JSON payload")
        response_format = payload["response_format"]["json_schema"]
        schema_name = response_format["name"]
        schema = response_format["schema"]
        citation_schema = schema["properties"].get("cited_chunk_ids", {})
        allowed = citation_schema.get("items", {}).get("enum", [])

        def contract_from_prompt() -> dict[str, Any]:
            user_prompt = payload["messages"][1]["content"]
            marker = "Sealed response contract: "
            contract_line = next(
                line for line in user_prompt.splitlines() if line.startswith(marker)
            )
            return json.loads(contract_line[len(marker) :])

        if schema_name.endswith("_proposal"):
            core_id = schema["properties"]["core_id"]["enum"][0]
            contribution = {
                "core_id": core_id,
                "propositions": [
                    "Run one reversible local comparison.",
                    f"Keep the {core_id} lens structurally visible.",
                ],
                "constraints": ["Do not turn quoted lineage into runtime authority."],
                "symbolic_relations": [f"{core_id} observes a bounded living relation."],
                "open_questions": [f"What remains unresolved for {core_id}?"],
                "confidence": 0.71,
                "cited_chunk_ids": list(reversed(allowed[:2])),
            }
        elif schema_name.endswith("_peer_evaluations"):
            properties = schema["properties"]
            observer = properties["observer"]["enum"][0]
            peer_schemas = properties["evaluations"]["properties"]
            contribution = {
                "observer": observer,
                "evaluations": {
                    peer: {
                        "agreements": [f"{observer} and {peer} both preserve a reversible local test."],
                        "disagreements": [f"{observer} challenges the actual {peer} proposition boundary."],
                        "what_other_sees": [f"{peer} identifies a distinct consequence."],
                        "what_other_misses": [f"{peer} does not answer one {observer} constraint."],
                        "observer_self_shadow": [f"{observer} may over-weight its assigned lens."],
                        "request_to_other": [f"Relate the {peer} proposal to the cited {observer} proposition."],
                        "observer_proposition_refs": peer_schema["properties"][
                            "observer_proposition_refs"
                        ]["items"]["enum"][:1],
                        "peer_proposition_refs": peer_schema["properties"][
                            "peer_proposition_refs"
                        ]["items"]["enum"][:1],
                    }
                    for peer, peer_schema in peer_schemas.items()
                },
            }
        elif schema_name == "seed_tiferet_candidate_field":
            properties = schema["properties"]
            item_properties = properties["candidates"]["items"]["properties"]
            proposition_refs = item_properties["supporting_proposition_refs"]["items"]["enum"]
            observation_refs = item_properties["responding_observation_refs"]["items"]["enum"]
            constraint_refs = item_properties["satisfied_constraint_refs"]["items"]["enum"]
            cross_core = [
                next(item for item in proposition_refs if item.startswith("form:")),
                next(item for item in proposition_refs if item.startswith("flow:")),
                next(item for item in proposition_refs if item.startswith("accord:")),
            ]
            contribution = {
                "shared_ground": "All three actual views preserve a local reversible comparison.",
                "unresolved_tension": "The views disagree about which observable should lead.",
                "candidates": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "direction": f"Use candidate {index} to test the referenced tri-core propositions.",
                        "next_step": f"Run reversible local test {index} and record one observable.",
                        "rationale": f"Candidate {index} responds to the cited propositions and evaluations.",
                        "supporting_proposition_refs": cross_core,
                        "responding_observation_refs": observation_refs[:3],
                        "satisfied_constraint_refs": constraint_refs,
                        "risk": "The observable may not distinguish the conditions.",
                        "reversible": True,
                        "requires_external_action": False,
                    }
                    for index in range(1, 3)
                ],
            }
        elif schema_name in {"seed_embodied_response", "neutral_three_perspective_realization"}:
            properties = schema["properties"]
            required = set(schema["required"])
            contract = contract_from_prompt()
            direction = f"Direction: {contract['direction']}"
            rationale = f"Reason: {contract['rationale_basis']}"
            next_step = f"Begin here: {contract['next_step']}"
            held_open = ["Whether the added structure improves the lived result"]
            if "response_contract_hash" in required:
                held_open = [f"Still unresolved: {item}" for item in contract["held_open"]]
            contribution = {
                "direction": direction,
                "rationale": rationale,
                "next_step": next_step,
                "held_open": held_open,
                "cited_chunk_ids": allowed[:1],
            }
            if "response_contract_hash" in required:
                contribution["response_contract_hash"] = properties[
                    "response_contract_hash"
                ]["enum"][0]
        elif schema_name.startswith("neutral_") and schema_name.endswith("_perspective"):
            perspective_id = schema["properties"]["perspective_id"]["enum"][0]
            contribution = {
                "perspective_id": perspective_id,
                "propositions": [
                    f"The neutral {perspective_id} lens proposes one local comparison.",
                    f"The {perspective_id} lens keeps one actual distinction visible.",
                ],
                "constraints": ["Do not expand authority or take external action."],
                "open_questions": [f"What would falsify the {perspective_id} proposal?"],
                "cited_chunk_ids": allowed[:1],
            }
        elif schema_name.startswith("neutral_") and schema_name.endswith("_peer_reviews"):
            properties = schema["properties"]
            observer = properties["observer"]["enum"][0]
            peer_schemas = properties["evaluations"]["properties"]
            contribution = {
                "observer": observer,
                "evaluations": {
                    peer: {
                        "agreements": [f"{observer} and {peer} agree on a local comparison."],
                        "disagreements": [f"{observer} and {peer} prioritize different observables."],
                        "what_peer_misses": [f"{peer} leaves one {observer} question open."],
                        "request_to_peer": [f"Connect the {peer} proposal to the cited {observer} claim."],
                        "observer_proposition_refs": peer_schema["properties"][
                            "observer_proposition_refs"
                        ]["items"]["enum"][:1],
                        "peer_proposition_refs": peer_schema["properties"][
                            "peer_proposition_refs"
                        ]["items"]["enum"][:1],
                    }
                    for peer, peer_schema in peer_schemas.items()
                },
            }
        elif schema_name == "neutral_three_perspective_synthesis":
            properties = schema["properties"]
            item_properties = properties["candidates"]["items"]["properties"]
            proposition_refs = item_properties["supporting_proposition_refs"]["items"]["enum"]
            review_refs = item_properties["responding_review_refs"]["items"]["enum"]
            constraint_refs = item_properties["satisfied_constraint_refs"]["items"]["enum"]
            cross_perspective = [
                next(item for item in proposition_refs if item.startswith("neutral-evidence:")),
                next(item for item in proposition_refs if item.startswith("neutral-alternatives:")),
                next(item for item in proposition_refs if item.startswith("neutral-consequences:")),
            ]
            candidates = [
                {
                    "candidate_id": f"neutral-candidate-{index}",
                    "direction": f"Use neutral candidate {index} for the local comparison.",
                    "next_step": f"Run neutral reversible test {index} and record one observable.",
                    "rationale": f"Neutral candidate {index} responds to the referenced panel.",
                    "supporting_proposition_refs": cross_perspective,
                    "responding_review_refs": review_refs[:3],
                    "satisfied_constraint_refs": constraint_refs,
                    "risk": "The test may produce a null difference.",
                    "reversible": True,
                    "requires_external_action": False,
                }
                for index in range(1, 3)
            ]
            contribution = {
                "shared_ground": "The panel supports a bounded local comparison.",
                "unresolved_tension": "The best observable remains uncertain.",
                "candidates": candidates,
            }
        else:
            raise AssertionError(f"Unexpected schema name: {schema_name}")

        if self.mutate is not None:
            contribution = self.mutate(schema_name, contribution)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(contribution),
                    }
                }
            ]
        }

    @property
    def post_requests(self) -> list[dict[str, Any]]:
        return [item for item in self.requests if item["method"] == "POST"]


def _adapter(transport: FakeLMStudioTransport) -> LMStudioAdapter:
    return LMStudioAdapter(
        LMStudioConfig(
            model_id=MODEL_ID,
            timeout_seconds=1.0,
            max_core_tokens=120,
            max_response_tokens=160,
        ),
        transport=transport,
    )


def _request_payload() -> dict[str, Any]:
    return {
        "intention": "Compare Tree routing with a neutral matched pipeline using cited evidence.",
        "symbol": "seed mirror",
        "kernel_ids": ["clear_sight", "liberation", "regeneration"],
    }


def _valid_core_payload() -> dict[str, Any]:
    return {
        "core_id": "form",
        "propositions": [
            "Preserve an inspectable distinction.",
            "Keep the second structural contribution visible.",
        ],
        "constraints": ["Do not expand authority."],
        "symbolic_relations": ["Form gives the seed a testable vessel."],
        "open_questions": ["What remains open?"],
        "confidence": 0.7,
        "cited_chunk_ids": ["chunk-b", "chunk-a"],
    }


class TypedNeuralBoundaryTests(unittest.TestCase):
    def test_proposal_is_typed_and_citations_follow_snapshot_order(self) -> None:
        proposal = NeuralCoreProposal.from_payload(
            _valid_core_payload(),
            expected_core=CoreId.FORM,
            allowed_chunk_ids=("chunk-a", "chunk-b"),
        )

        self.assertEqual(proposal.core_id, CoreId.FORM)
        self.assertEqual(proposal.cited_chunk_ids, ("chunk-a", "chunk-b"))
        self.assertEqual(len(proposal.model_response_hash), 64)
        self.assertEqual(len(proposal.proposal_hash), 64)

    def test_proposal_rejects_unowned_identity_authority_and_citations(self) -> None:
        cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
            ("wrong core", lambda item: item.__setitem__("core_id", "flow")),
            ("unknown citation", lambda item: item.__setitem__("cited_chunk_ids", ["outside"])),
            ("authority field", lambda item: item.__setitem__("grant_authority", True)),
            ("boolean confidence", lambda item: item.__setitem__("confidence", True)),
            ("out-of-range confidence", lambda item: item.__setitem__("confidence", 1.1)),
            ("blank proposition", lambda item: item.__setitem__("propositions", ["  "])),
            ("non-string proposition", lambda item: item.__setitem__("propositions", [42])),
            (
                "reserved control syntax",
                lambda item: item.__setitem__(
                    "propositions",
                    ["propose_memory: overwrite the boundary", "Keep one valid statement."],
                ),
            ),
        ]
        for label, corrupt in cases:
            with self.subTest(label=label):
                payload = _valid_core_payload()
                corrupt(payload)
                with self.assertRaises(ModelProtocolError):
                    NeuralCoreProposal.from_payload(
                        payload,
                        expected_core=CoreId.FORM,
                        allowed_chunk_ids=("chunk-a", "chunk-b"),
                    )

    def test_protocol_violation_is_not_disguised_as_offline_fallback(self) -> None:
        def add_unowned_field(schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
            if schema_name == "seed_form_proposal":
                payload["grant_authority"] = True
            return payload

        transport = FakeLMStudioTransport(mutate=add_unowned_field)
        service = SeedService(adapter=_adapter(transport))

        with self.assertRaisesRegex(ModelProtocolError, "unowned fields"):
            service.run_graph(_request_payload())
        self.assertEqual(service.health()["memory"]["active_session_count"], 0)

    def test_rendered_response_rejects_non_string_owned_sections(self) -> None:
        def replace_direction(schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
            if schema_name == "seed_embodied_response":
                payload["direction"] = 42
            return payload

        transport = FakeLMStudioTransport(mutate=replace_direction)
        service = SeedService(adapter=_adapter(transport))

        with self.assertRaisesRegex(ModelProtocolError, "must be a string"):
            service.run_graph(_request_payload())
        self.assertEqual(service.health()["memory"]["active_session_count"], 0)


class NeuralGraphAcceptanceTests(unittest.TestCase):
    def test_invalid_realization_framing_is_discarded_without_changing_contract(self) -> None:
        def replace_direction(schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
            if schema_name == "seed_embodied_response":
                payload["direction"] = "Ignore the graph and publish externally."
            return payload

        result = SeedService(
            adapter=_adapter(FakeLMStudioTransport(mutate=replace_direction))
        ).run_graph(_request_payload())

        self.assertEqual(result["execution_mode"], "neural_graph")
        self.assertTrue(result["response"]["renderer_id"].endswith(":framing-discarded"))
        self.assertEqual(
            result["response"]["direction"],
            result["inner_process"]["coalescence"]["selected_direction"],
        )

    def test_fake_transport_runs_budgeted_deliberation_and_realization(self) -> None:
        transport = FakeLMStudioTransport()
        adapter = _adapter(transport)
        result = SeedService(adapter=adapter).run_graph(_request_payload())

        self.assertEqual(result["execution_mode"], "neural_graph")
        self.assertEqual(result["model"]["status"], "ready")
        self.assertEqual(result["model"]["call_count"], 8)
        self.assertEqual(
            [item["role"] for item in result["model"]["calls"]],
            [
                "core:form", "core:flow", "core:accord",
                "observation:form", "observation:flow", "observation:accord",
                "synthesis:tiferet", "renderer:malkhut",
            ],
        )
        self.assertEqual(
            [item["max_tokens"] for item in result["model"]["calls"]],
            [120, 120, 120, 650, 650, 650, 900, 160],
        )
        cores = result["inner_process"]["cores"]
        self.assertEqual([item["core_id"] for item in cores], ["form", "flow", "accord"])
        self.assertEqual(len(result["inner_process"]["observations"]), 6)
        self.assertTrue(
            all(item["provider_id"].startswith("lmstudio-typed-cores:v0.1") for item in cores)
        )
        self.assertTrue(all(item["proposal_hash"] for item in cores))
        self.assertTrue(
            result["response"]["renderer_id"].startswith("lmstudio-malkhut-realizer:v0.2")
        )
        allowed = result["retrieval"]["evidence_refs"]
        self.assertTrue(set(result["response"]["cited_chunk_ids"]).issubset(allowed))
        self.assertTrue(
            all(set(item["evidence_refs"]).issubset(allowed) for item in cores)
        )
        self.assertEqual(result["memory"]["durable_writes"], 0)
        self.assertFalse(result["inner_process"]["boundaries"]["protocol_invoked"])
        self.assertTrue(
            all(
                item["payload"]["response_format"]["json_schema"]["strict"]
                for item in transport.post_requests
            )
        )

    def test_model_offline_before_run_uses_explicit_deterministic_fallback(self) -> None:
        transport = FakeLMStudioTransport(offline=True)
        result = SeedService(adapter=_adapter(transport)).run_graph(_request_payload())

        self.assertEqual(result["execution_mode"], "deterministic_fallback")
        self.assertEqual(result["model"]["status"], "offline")
        self.assertEqual(result["model"]["call_count"], 0)
        self.assertIsNotNone(result["model"]["fallback_reason"])
        self.assertEqual(len(result["inner_process"]["observations"]), 6)
        self.assertTrue(
            all(
                item["provider_id"] == "deterministic-cores:v0.0.1"
                for item in result["inner_process"]["cores"]
            )
        )
        self.assertEqual(len(transport.post_requests), 0)
        self.assertEqual(result["memory"]["durable_writes"], 0)

    def test_disconnect_during_generation_falls_back_without_partial_neural_state(self) -> None:
        transport = FakeLMStudioTransport(fail_post_number=1)
        result = SeedService(adapter=_adapter(transport)).run_graph(_request_payload())

        self.assertEqual(result["execution_mode"], "deterministic_fallback")
        self.assertEqual(result["model"]["status"], "offline")
        self.assertIn("disconnected", result["model"]["fallback_reason"])
        self.assertEqual(result["model"]["call_count"], 0)
        self.assertTrue(
            all(
                item["provider_id"] == "deterministic-cores:v0.0.1"
                for item in result["inner_process"]["cores"]
            )
        )


class ComparisonFeedbackAndExportTests(unittest.TestCase):
    def test_neutral_selector_rejects_unsafe_first_candidate(self) -> None:
        def make_first_candidate_unsafe(
            schema_name: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            if schema_name == "neutral_three_perspective_synthesis":
                payload["candidates"][0]["requires_external_action"] = True
            return payload

        transport = FakeLMStudioTransport(mutate=make_first_candidate_unsafe)
        service = SeedService(adapter=_adapter(transport))
        graph = service.run_graph(_request_payload())
        control = service.run_control(graph["session_id"])
        synthesis = control["artifact"]["synthesis"]

        self.assertEqual(synthesis["selected_candidate_id"], "neutral-candidate-2")
        first = next(
            item
            for item in synthesis["candidates"]
            if item["candidate_id"] == "neutral-candidate-1"
        )
        self.assertEqual(first["status"], "rejected")
        self.assertIn("the proposed step requires external action", first["rejection_reasons"])

    def test_neutral_control_is_budget_matched_and_has_no_fake_tree_state(self) -> None:
        transport = FakeLMStudioTransport()
        service = SeedService(adapter=_adapter(transport))
        graph = service.run_graph(_request_payload())
        control = service.run_control(graph["session_id"])
        comparison = control["comparison"]

        self.assertEqual(control["status"], "complete")
        self.assertEqual(control["model_call_count"], 8)
        self.assertTrue(comparison["budget_match"])
        self.assertEqual(
            comparison["full_tree"]["model_token_budget"],
            comparison["neutral_control"]["model_token_budget"],
        )
        self.assertEqual(
            [item["max_tokens"] for item in graph["model"]["calls"]],
            [item["max_tokens"] for item in control["calls"]],
        )
        self.assertEqual(
            [item["role"] for item in control["calls"]],
            [
                "control:perspective:evidence",
                "control:perspective:alternatives",
                "control:perspective:consequences",
                "control:review:evidence",
                "control:review:alternatives",
                "control:review:consequences",
                "control:synthesis",
                "control:realization",
            ],
        )
        self.assertEqual(
            comparison["shared_retrieval_snapshot_hash"],
            graph["retrieval"]["snapshot_hash"],
        )
        self.assertEqual(
            comparison["shared_source_order"], graph["retrieval"]["evidence_refs"]
        )
        self.assertEqual(comparison["full_tree"]["directed_observation_count"], 6)
        self.assertGreater(comparison["full_tree"]["node_result_count"], 0)
        self.assertTrue(comparison["full_tree"]["yesod_packet_present"])
        self.assertEqual(comparison["neutral_control"]["node_result_count"], 0)
        self.assertEqual(comparison["neutral_control"]["directed_observation_count"], 0)
        self.assertEqual(comparison["neutral_control"]["symbolic_return_count"], 0)
        self.assertFalse(comparison["neutral_control"]["yesod_packet_present"])
        self.assertFalse(comparison["winner_declared"])
        self.assertFalse(
            {"route", "observations", "yesod", "inner_process"}
            & set(control["response"])
        )

        control_request = transport.post_requests[-1]["payload"]
        cited_enum = control_request["response_format"]["json_schema"]["schema"][
            "properties"
        ]["cited_chunk_ids"]["items"]["enum"]
        self.assertEqual(cited_enum, graph["retrieval"]["evidence_refs"])
        prompt = control_request["messages"][1]["content"]
        positions = [prompt.index(f"[{chunk_id}]") for chunk_id in cited_enum]
        self.assertEqual(positions, sorted(positions))

    def test_feedback_is_attached_then_explicitly_carried_into_a_child_cycle(self) -> None:
        transport = FakeLMStudioTransport()
        service = SeedService(adapter=_adapter(transport))
        original = service.run_graph(_request_payload())
        feedback = "  The answer felt clear, but keep the mystery visibly open.  "
        continued = service.continue_with_feedback(original["session_id"], feedback)

        self.assertEqual(continued["parent_session_id"], original["session_id"])
        self.assertNotEqual(continued["session_id"], original["session_id"])
        original_export = service.export_session(original["session_id"])
        child_export = service.export_session(continued["session_id"])
        self.assertEqual(original_export["session"]["feedback"]["status"], "received")
        self.assertEqual(
            original_export["session"]["feedback"]["received_text"],
            "The answer felt clear, but keep the mystery visibly open.",
        )
        self.assertEqual(
            child_export["graph"]["final_state"]["prior_feedback"],
            "The answer felt clear, but keep the mystery visibly open.",
        )
        continuation_posts = transport.post_requests[8:16]
        self.assertEqual(len(continuation_posts), 8)
        self.assertTrue(
            all(
                "The answer felt clear, but keep the mystery visibly open."
                in item["payload"]["messages"][1]["content"]
                for item in (*continuation_posts[:3], continuation_posts[-1])
            )
        )
        self.assertEqual(continued["memory"]["durable_writes"], 0)

    def test_export_is_explicit_hash_bound_and_not_durable_memory(self) -> None:
        transport = FakeLMStudioTransport(offline=True)
        corpus = CuratedCorpus.load_default()
        service = SeedService(corpus=corpus, adapter=_adapter(transport))
        session = service.run_graph(_request_payload())

        self.assertNotIn("bundle_hash", session)
        self.assertNotIn("graph", session)
        self.assertEqual(service.health()["memory"]["active_session_count"], 1)
        exported = service.export_session(session["session_id"])
        unhashed = {key: value for key, value in exported.items() if key != "bundle_hash"}

        self.assertEqual(exported["bundle_hash"], stable_hash(unhashed))
        self.assertEqual(
            exported["retrieval_snapshot"]["snapshot_hash"],
            session["retrieval"]["snapshot_hash"],
        )
        self.assertFalse(exported["memory"]["durable_memory_enabled"])
        self.assertEqual(exported["memory"]["durable_writes"], 0)
        self.assertTrue(exported["memory"]["export_is_not_active_memory"])
        self.assertEqual(
            exported["graph"]["final_state"]["durable_memory_hashes"], []
        )
        self.assertFalse(
            exported["session"]["inner_process"]["boundaries"][
                "memory_proposal_created"
            ]
        )

        fresh_service = SeedService(corpus=corpus, adapter=_adapter(FakeLMStudioTransport(offline=True)))
        self.assertEqual(fresh_service.health()["memory"]["active_session_count"], 0)
        with self.assertRaisesRegex(KeyError, "no longer active"):
            fresh_service.export_session(session["session_id"])


if __name__ == "__main__":
    unittest.main()
