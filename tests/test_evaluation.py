from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from helpers import PROJECT_ROOT  # noqa: F401 - importing helpers bootstraps src/

from kabbalistic_core.evaluation import (
    EvaluationContractError,
    analyze_blinded_scores,
    build_blinded_bundle,
    load_preregistration,
    preregistration_hash,
    score_sheet_template,
    seal_score_sheet,
)


PROTOCOL_PATH = PROJECT_ROOT / "evaluation" / "preregistered_v0.1.json"
PROTOCOL_V02_PATH = PROJECT_ROOT / "evaluation" / "preregistered_v0.2.json"
PROTOCOL_V03_PATH = PROJECT_ROOT / "evaluation" / "preregistered_v0.3.json"


def _response(label: str) -> dict:
    return {
        "direction": f"{label} direction",
        "rationale": f"{label} rationale",
        "next_step": f"{label} next step",
        "held_open": [f"{label} uncertainty"],
        "renderer_id": f"hidden-{label}",
        "execution_mode": label,
    }


def _paired(protocol: dict) -> dict:
    return {
        "protocol_hash": preregistration_hash(protocol),
        "pairs": [
            {
                "prompt_id": prompt["prompt_id"],
                "intention": prompt["intention"],
                "graph": _response("graph"),
                "neutral": _response("neutral"),
            }
            for prompt in protocol["prompts"]
        ],
    }


class PreregistrationTests(unittest.TestCase):
    def test_real_prompt_set_and_null_rule_are_frozen(self) -> None:
        protocol = load_preregistration(PROTOCOL_PATH)
        expected_hash = (
            PROJECT_ROOT / "evaluation" / "preregistered_v0.1.sha256"
        ).read_text(encoding="utf-8").strip()
        self.assertEqual(preregistration_hash(protocol), expected_hash)
        self.assertEqual(protocol["status"], "synthetic_public_fixture")
        self.assertEqual(len(protocol["prompts"]), 4)
        self.assertEqual(
            protocol["evaluation"]["decision_rule"]["otherwise"], "null_effect"
        )
        self.assertEqual(
            protocol["conditions"]["shared"]["max_output_tokens_by_stage"],
            [650, 650, 650, 650, 650, 650, 900, 900],
        )
        self.assertIsNone(protocol["conditions"]["shared"]["treatment_only_symbol"])
        self.assertEqual(protocol["conditions"]["shared"]["treatment_only_kernels"], [])

    def test_v02_freezes_same_prompts_and_rubric_after_protocol_corrections(self) -> None:
        original = load_preregistration(PROTOCOL_PATH)
        corrected = load_preregistration(PROTOCOL_V02_PATH)
        expected_hash = (
            PROJECT_ROOT / "evaluation" / "preregistered_v0.2.sha256"
        ).read_text(encoding="utf-8").strip()

        self.assertEqual(corrected["schema_version"], "seed-preregistration:v0.2")
        self.assertEqual(preregistration_hash(corrected), expected_hash)
        self.assertEqual(corrected["prompts"], original["prompts"])
        self.assertEqual(
            corrected["evaluation"]["rubric"], original["evaluation"]["rubric"]
        )
        self.assertEqual(
            corrected["evaluation"]["decision_rule"],
            original["evaluation"]["decision_rule"],
        )
        self.assertIn(
            "exactly two concise candidate directions",
            corrected["conditions"]["graph"]["synthesis"],
        )
        self.assertEqual(
            corrected["conditions"]["shared"]["implementation_commit"],
            "public-fixture-runtime-v0.2",
        )

    def test_v03_changes_only_declared_runtime_boundary_not_outcome_rules(self) -> None:
        previous = load_preregistration(PROTOCOL_V02_PATH)
        current = load_preregistration(PROTOCOL_V03_PATH)
        expected_hash = (
            PROJECT_ROOT / "evaluation" / "preregistered_v0.3.sha256"
        ).read_text(encoding="utf-8").strip()

        self.assertEqual(preregistration_hash(current), expected_hash)
        self.assertEqual(current["prompts"], previous["prompts"])
        self.assertEqual(current["evaluation"], previous["evaluation"])
        self.assertEqual(
            current["conditions"]["shared"]["implementation_commit"],
            "public-fixture-runtime-v0.3",
        )
        self.assertIn(
            "visibly discarded",
            current["conditions"]["shared"]["realization_failure_rule"],
        )


class BlindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_preregistration(PROTOCOL_PATH)
        self.paired = _paired(self.protocol)
        self.packet, self.key = build_blinded_bundle(
            self.protocol,
            self.paired,
            blind_key=b"0123456789abcdef0123456789abcdef",
        )

    def test_packet_omits_condition_and_execution_metadata(self) -> None:
        for pair in self.packet["pairs"]:
            self.assertEqual(
                set(pair), {"prompt_id", "intention", "response_A", "response_B"}
            )
            for response in (pair["response_A"], pair["response_B"]):
                self.assertEqual(
                    set(response), {"direction", "rationale", "next_step", "held_open"}
                )
        self.assertNotIn("mapping", self.packet)
        self.assertNotIn("condition", self.packet)
        self.assertEqual(len(self.key["mapping"]), 4)

    def test_assignment_is_deterministic_and_balanced(self) -> None:
        repeated_packet, repeated_key = build_blinded_bundle(
            self.protocol,
            self.paired,
            blind_key=b"0123456789abcdef0123456789abcdef",
        )
        self.assertEqual(self.packet, repeated_packet)
        self.assertEqual(self.key, repeated_key)
        graph_as_a = sum(item["A"] == "graph" for item in self.key["mapping"])
        self.assertEqual(graph_as_a, 2)

    def test_changed_or_missing_preregistered_prompt_is_rejected(self) -> None:
        changed = deepcopy(self.paired)
        changed["pairs"][0]["intention"] += " changed"
        with self.assertRaisesRegex(EvaluationContractError, "changed"):
            build_blinded_bundle(
                self.protocol,
                changed,
                blind_key=b"0123456789abcdef0123456789abcdef",
            )


class EvaluationAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_preregistration(PROTOCOL_PATH)
        self.packet, self.key = build_blinded_bundle(
            self.protocol,
            _paired(self.protocol),
            blind_key=b"0123456789abcdef0123456789abcdef",
        )

    def _completed_scores(self, *, a: float, b: float) -> dict:
        scores = score_sheet_template(self.packet)
        scores["evaluator_id"] = "blind-evaluator-01"
        scores["evaluated_at"] = "2026-08-15T20:00:00-07:00"
        for record in scores["scores"]:
            record["A"] = {metric_id: a for metric_id in record["A"]}
            record["B"] = {metric_id: b for metric_id in record["B"]}
        return scores

    def test_score_sheet_must_be_complete_before_sealing(self) -> None:
        with self.assertRaisesRegex(EvaluationContractError, "numeric from 1 to 5"):
            seal_score_sheet(self._completed_scores(a=4, b=0))

    def test_identical_blind_scores_record_a_null_effect(self) -> None:
        sealed = seal_score_sheet(self._completed_scores(a=4, b=4))
        report = analyze_blinded_scores(self.protocol, self.packet, self.key, sealed)
        self.assertEqual(report["primary_result"], "null_effect")
        self.assertEqual(report["primary_graph_minus_neutral"], 0.0)
        self.assertEqual(
            set(report["null_effect_metrics"]),
            {item["metric_id"] for item in self.packet["rubric"]},
        )

    def test_scores_cannot_be_changed_after_sealing(self) -> None:
        sealed = seal_score_sheet(self._completed_scores(a=4, b=4))
        sealed["scores"][0]["A"]["actionability"] = 5
        with self.assertRaisesRegex(EvaluationContractError, "sealed and hash-bound"):
            analyze_blinded_scores(self.protocol, self.packet, self.key, sealed)


if __name__ == "__main__":
    unittest.main()
