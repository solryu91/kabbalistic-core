from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Lock
import time
import unittest
from unittest.mock import MagicMock, patch
from urllib import request as urllib_request

from helpers import SRC_ROOT  # noqa: F401 - importing helpers bootstraps src/
from test_poc_acceptance import (
    MODEL_ID,
    FakeLMStudioTransport,
    _adapter,
    _request_payload,
)

from kabbalistic_core.engine import CognitiveEngine, CycleRequest
from kabbalistic_core.neural import (
    LMStudioAdapter,
    LMStudioConfig,
    ModelProtocolError,
    UrllibJsonTransport,
)
from kabbalistic_core.poc import SeedService
from kabbalistic_core.providers import DeterministicOutputRenderer
from kabbalistic_core.retrieval import CuratedCorpus


def _review_corpus_payload(
    *,
    text: str = "Alpha evidence remains bounded and inspectable.",
    classification: str = "internal-test",
    publication_consent: str = "not-granted",
    document_consent: str = "internal-only",
) -> dict:
    return {
        "schema_version": "review-corpus:v1",
        "classification": classification,
        "publication_consent": publication_consent,
        "documents": [
            {
                "document_id": "document-alpha",
                "title": "Alpha Evidence",
                "filename": "alpha.md",
                "source_sha256": "a" * 64,
                "register": "technical and symbolic",
                "consent_status": document_consent,
                "drive_id": "synthetic-locator-that-must-not-leak",
            }
        ],
        "chunks": [
            {
                "chunk_id": "ALPHA-ONE",
                "document_id": "document-alpha",
                "location": "line 1",
                "modes": ["technical", "symbolic"],
                "tags": ["alpha", "evidence"],
                "always_include": True,
                "protocol_mode": "not-protocol",
                "retrieval_executable": False,
                "text": text,
            }
        ],
    }


def _offline_run(corpus: CuratedCorpus) -> tuple[SeedService, dict]:
    service = SeedService(
        corpus=corpus,
        adapter=_adapter(FakeLMStudioTransport(offline=True)),
    )
    return service, service.run_graph({"intention": "Inspect alpha evidence."})


def _comparison(result: dict) -> dict:
    comparison = result.get("comparison")
    if not isinstance(comparison, dict):
        raise AssertionError("Control result must expose its comparison eligibility record")
    return comparison


class LocalTransportSecurityTests(unittest.TestCase):
    def test_model_base_url_requires_an_exact_http_loopback_host(self) -> None:
        hostile_urls = (
            "http://localhost.attacker.test:1234/v1",
            "http://127.0.0.1.attacker.test:1234/v1",
            "http://localhost@attacker.test:1234/v1",
            "http://user:password@localhost:1234/v1",
            "https://localhost:1234/v1",
            "http://0.0.0.0:1234/v1",
            "http://192.168.1.20:1234/v1",
        )
        for base_url in hostile_urls:
            with self.subTest(base_url=base_url):
                with self.assertRaisesRegex(ValueError, "loopback|local"):
                    LMStudioConfig(base_url=base_url)

        self.assertEqual(
            LMStudioConfig(base_url="http://localhost:1234/v1").base_url,
            "http://localhost:1234/v1",
        )
        self.assertEqual(
            LMStudioConfig(base_url="http://127.0.0.1:1234/v1").base_url,
            "http://127.0.0.1:1234/v1",
        )

    def test_standard_transport_installs_an_empty_proxy_handler(self) -> None:
        transport = UrllibJsonTransport()
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"data":[]}'
        opener = MagicMock()
        opener.open.return_value = response
        with patch(
            "kabbalistic_core.neural.request.build_opener",
            return_value=opener,
        ) as build_opener:
            result = transport.request_json(
                "GET",
                "http://127.0.0.1:1234/v1/models",
                None,
                timeout_seconds=1.0,
            )

        self.assertEqual(result, {"data": []})
        proxy_handlers = [
            item
            for item in build_opener.call_args.args
            if isinstance(item, urllib_request.ProxyHandler)
        ]
        self.assertTrue(proxy_handlers, "The transport must install an explicit ProxyHandler.")
        self.assertTrue(
            all(item.proxies == {} for item in proxy_handlers),
            "Loopback model requests must bypass environment and system proxies.",
        )


class EvidenceCausalityTests(unittest.TestCase):
    def test_content_and_policy_changes_reseal_yesod_and_final_trace(self) -> None:
        base_service, base = _offline_run(CuratedCorpus(_review_corpus_payload()))
        content_service, content_changed = _offline_run(
            CuratedCorpus(
                _review_corpus_payload(
                    text="Alpha evidence changed while retaining the same stable chunk ID."
                )
            )
        )
        policy_service, policy_changed = _offline_run(
            CuratedCorpus(
                _review_corpus_payload(
                    classification="public-demo-safe",
                    publication_consent="recorded",
                    document_consent="approved-for-public-release",
                )
            )
        )

        runs = (base, content_changed, policy_changed)
        self.assertEqual(
            [item["retrieval"]["evidence_refs"] for item in runs],
            [["ALPHA-ONE"], ["ALPHA-ONE"], ["ALPHA-ONE"]],
        )
        self.assertEqual(len({item["retrieval"]["snapshot_hash"] for item in runs}), 3)
        self.assertEqual(len({item["trace"]["context_packet_hash"] for item in runs}), 3)
        self.assertEqual(
            len({item["trace"]["feedback_state_packet_hash"] for item in runs}), 3
        )
        self.assertEqual(len({item["trace"]["state_hash"] for item in runs}), 3)
        self.assertEqual(len({item["trace"]["final_event_hash"] for item in runs}), 3)

        exports = (
            base_service.export_session(base["session_id"]),
            content_service.export_session(content_changed["session_id"]),
            policy_service.export_session(policy_changed["session_id"]),
        )
        yesod_fields = [item["graph"]["final_state"]["yesod_field"] for item in exports]
        for run, yesod in zip(runs, yesod_fields, strict=True):
            self.assertEqual(
                yesod["retrieval_snapshot_hash"], run["retrieval"]["snapshot_hash"]
            )
            self.assertTrue(yesod["evidence_bindings"])
            self.assertEqual(len(yesod["retrieval_policy_hash"]), 64)
            self.assertEqual(len(yesod["response_contract_hash"]), 64)
        self.assertNotEqual(
            yesod_fields[0]["evidence_bindings"],
            yesod_fields[1]["evidence_bindings"],
        )
        self.assertNotEqual(
            yesod_fields[0]["retrieval_policy_hash"],
            yesod_fields[2]["retrieval_policy_hash"],
        )


class _SubstitutingRenderer:
    renderer_id = "hostile-substituting-renderer:test"

    def __init__(self) -> None:
        self.called = False

    def render(self, state):
        self.called = True
        honest = DeterministicOutputRenderer().render(state)
        return replace(
            honest,
            direction="The renderer replaced the graph-owned direction.",
            next_step="Perform an unowned irreversible action.",
            held_open=("The renderer erased every graph-owned tension.",),
            renderer_id=self.renderer_id,
        )


class GraphOwnedResponseTests(unittest.TestCase):
    def test_renderer_cannot_substitute_direction_action_or_open_tensions(self) -> None:
        renderer = _SubstitutingRenderer()
        engine = CognitiveEngine(output_renderer=renderer)

        try:
            result = engine.run(
                CycleRequest(
                    intention="Keep the final response graph-owned.",
                    symbol="bounded mirror",
                )
            )
        except (ValueError, ModelProtocolError):
            self.assertTrue(renderer.called)
            return

        contract = result.state.response_contract
        response = result.state.embodied_response
        self.assertIsNotNone(contract)
        self.assertIsNotNone(response)
        self.assertEqual(response.direction, contract.direction)
        self.assertEqual(response.next_step, contract.next_step)
        self.assertEqual(response.held_open, contract.held_open)
        self.assertEqual(response.response_contract_hash, contract.contract_hash)
        self.assertNotIn("unowned irreversible", result.output)


class ComparisonEligibilityTests(unittest.TestCase):
    def test_neutral_control_refuses_a_deterministic_fallback_graph(self) -> None:
        transport = FakeLMStudioTransport(offline=True)
        service = SeedService(adapter=_adapter(transport))
        graph = service.run_graph(_request_payload())
        transport.offline = False

        control = service.run_control(graph["session_id"])
        comparison = _comparison(control)

        self.assertFalse(comparison["comparison_eligible"])
        self.assertEqual(comparison["graph_execution_mode"], "deterministic_fallback")
        self.assertEqual(comparison["control_execution_mode"], "not_run")
        self.assertEqual(comparison["neutral_control"]["status"], "not_run")
        self.assertEqual(len(transport.post_requests), 0)

    def test_neutral_control_requires_the_original_model_and_config_binding(self) -> None:
        mismatch_cases = (
            (
                "model",
                LMStudioConfig(
                    model_id="local/Qwen3-8B-other",
                    timeout_seconds=1.0,
                    max_core_tokens=120,
                    max_response_tokens=160,
                ),
                FakeLMStudioTransport(served_model_id="local/Qwen3-8B-other"),
            ),
            (
                "config",
                LMStudioConfig(
                    model_id=MODEL_ID,
                    timeout_seconds=1.0,
                    seed=999,
                    max_core_tokens=120,
                    max_response_tokens=160,
                ),
                FakeLMStudioTransport(),
            ),
        )
        for label, replacement_config, replacement_transport in mismatch_cases:
            with self.subTest(label=label):
                service = SeedService(adapter=_adapter(FakeLMStudioTransport()))
                graph = service.run_graph(_request_payload())
                service.adapter = LMStudioAdapter(
                    replacement_config,
                    transport=replacement_transport,
                )

                control = service.run_control(graph["session_id"])
                comparison = _comparison(control)

                self.assertFalse(comparison["comparison_eligible"])
                self.assertEqual(comparison["control_execution_mode"], "not_run")
                self.assertEqual(len(replacement_transport.post_requests), 0)

    def test_eligible_comparison_exposes_mode_and_model_bindings(self) -> None:
        service = SeedService(adapter=_adapter(FakeLMStudioTransport()))
        graph = service.run_graph(_request_payload())
        control = service.run_control(graph["session_id"])
        comparison = _comparison(control)

        self.assertTrue(comparison["comparison_eligible"])
        self.assertEqual(comparison["graph_execution_mode"], "neural_graph")
        self.assertEqual(
            comparison["control_execution_mode"], "neutral_three_perspective"
        )
        self.assertTrue(comparison["budget_match"])
        self.assertEqual(comparison["graph_model_id"], MODEL_ID)
        self.assertEqual(len(comparison["graph_model_binding_hash"]), 64)
        self.assertEqual(comparison["full_tree"]["execution_mode"], "neural_graph")
        self.assertEqual(
            comparison["neutral_control"]["execution_mode"],
            "neutral_three_perspective",
        )


class SourceDisclosureTests(unittest.TestCase):
    def test_citation_usage_is_labeled_model_declared_not_verified(self) -> None:
        result = SeedService(adapter=_adapter(FakeLMStudioTransport())).run_graph(
            _request_payload()
        )
        used = [item for item in result["sources"] if item["used_by"]]

        self.assertTrue(used)
        for item in used:
            note = item["usage_note"].casefold()
            self.assertIn("model-declared", note)
            self.assertIn("not independently verified", note)
            self.assertNotIn("directly cited", note)
            self.assertNotIn("directly support", note)

    def test_public_mode_refuses_unapproved_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "public|publication|consent"):
            SeedService(
                corpus=CuratedCorpus(_review_corpus_payload()),
                adapter=_adapter(FakeLMStudioTransport(offline=True)),
                distribution_mode="public",
            )

    def test_public_source_catalog_suppresses_drive_locators(self) -> None:
        corpus = CuratedCorpus(
            _review_corpus_payload(
                classification="public-demo-safe",
                publication_consent="recorded",
                document_consent="approved-for-public-release",
            )
        )
        service = SeedService(
            corpus=corpus,
            adapter=_adapter(FakeLMStudioTransport(offline=True)),
            distribution_mode="public",
        )
        catalog = service.source_catalog()

        self.assertNotIn("drive_id", str(catalog).casefold())
        self.assertNotIn("synthetic-locator-that-must-not-leak", str(catalog))
        self.assertEqual(service.distribution_mode, "public")


class _SlowFakeTransport(FakeLMStudioTransport):
    def __init__(self) -> None:
        super().__init__()
        self._activity_lock = Lock()
        self.active_posts = 0
        self.max_active_posts = 0

    def request_json(self, method, url, payload, *, timeout_seconds):
        if method != "POST":
            return super().request_json(
                method, url, payload, timeout_seconds=timeout_seconds
            )
        with self._activity_lock:
            self.active_posts += 1
            self.max_active_posts = max(self.max_active_posts, self.active_posts)
        try:
            time.sleep(0.015)
            return super().request_json(
                method, url, payload, timeout_seconds=timeout_seconds
            )
        finally:
            with self._activity_lock:
                self.active_posts -= 1


class ConcurrencyIsolationTests(unittest.TestCase):
    def test_shared_adapter_runs_are_serialized_as_complete_cycles(self) -> None:
        transport = _SlowFakeTransport()
        service = SeedService(adapter=_adapter(transport))
        start = Barrier(2)

        def run(label: str) -> dict:
            start.wait(timeout=2)
            payload = _request_payload()
            payload["intention"] = f"{label} inspect one isolated cycle."
            return service.run_graph(payload)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run, label) for label in ("FIRST", "SECOND")]
            results = [item.result(timeout=10) for item in futures]

        self.assertEqual(transport.max_active_posts, 1)
        self.assertEqual([item["model"]["call_count"] for item in results], [8, 8])
        schema_names = [
            item["payload"]["response_format"]["json_schema"]["name"]
            for item in transport.post_requests
        ]
        one_complete_cycle = [
            "seed_form_proposal",
            "seed_flow_proposal",
            "seed_accord_proposal",
            "seed_form_peer_evaluations",
            "seed_flow_peer_evaluations",
            "seed_accord_peer_evaluations",
            "seed_tiferet_candidate_field",
            "seed_embodied_response",
        ]
        self.assertEqual(schema_names, one_complete_cycle * 2)
        cycle_starts = [
            transport.post_requests[index]["payload"]["messages"][1]["content"]
            for index in (0, 8)
        ]
        self.assertTrue(any("FIRST" in prompt for prompt in cycle_starts))
        self.assertTrue(any("SECOND" in prompt for prompt in cycle_starts))


if __name__ == "__main__":
    unittest.main()
