"""Preregistered, blinded paired evaluation for the instrumented POC harness.

This module does not decide whether a mind exists.  It compares two bounded response
pipelines and keeps condition labels outside the packet shown to an evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from .models import stable_hash


class EvaluationContractError(ValueError):
    """Raised when an evaluation artifact violates the frozen protocol."""


def load_preregistration(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EvaluationContractError("A preregistration must be a JSON object")
    required = {
        "schema_version",
        "protocol_id",
        "status",
        "authorship_refs",
        "conditions",
        "prompts",
        "evaluation",
    }
    if set(data) != required:
        raise EvaluationContractError("The preregistration has missing or unowned fields")
    if data["status"] not in {"frozen_before_outcomes", "synthetic_public_fixture"}:
        raise EvaluationContractError(
            "The evaluation definition is neither frozen before outcomes nor a public fixture"
        )
    prompts = data["prompts"]
    if not isinstance(prompts, list) or not 3 <= len(prompts) <= 6:
        raise EvaluationContractError("The preregistration requires three to six prompts")
    prompt_ids = []
    for prompt in prompts:
        if not isinstance(prompt, dict) or set(prompt) != {"prompt_id", "intention", "why_real"}:
            raise EvaluationContractError("Each prompt requires only ID, intention, and rationale")
        if not all(isinstance(prompt[key], str) and prompt[key].strip() for key in prompt):
            raise EvaluationContractError("Preregistered prompt fields must be non-blank strings")
        prompt_ids.append(prompt["prompt_id"])
    if len(prompt_ids) != len(set(prompt_ids)):
        raise EvaluationContractError("Preregistered prompt IDs must be unique")
    rubric = data["evaluation"].get("rubric")
    if not isinstance(rubric, list) or not rubric:
        raise EvaluationContractError("The preregistration requires a non-empty rubric")
    metric_ids = []
    for metric in rubric:
        if not isinstance(metric, dict) or set(metric) != {
            "metric_id",
            "question",
            "score_1",
            "score_3",
            "score_5",
        }:
            raise EvaluationContractError("Each rubric metric must define the frozen anchors")
        if not all(isinstance(metric[key], str) and metric[key].strip() for key in metric):
            raise EvaluationContractError("Rubric fields must be non-blank strings")
        metric_ids.append(metric["metric_id"])
    if len(metric_ids) != len(set(metric_ids)):
        raise EvaluationContractError("Rubric metric IDs must be unique")
    return data


def preregistration_hash(protocol: Mapping[str, Any]) -> str:
    """Content hash used to bind outputs, packets, score sheets, and reports."""

    return stable_hash(dict(protocol))


def _bounded_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError("A paired response must be an object")
    expected = ("direction", "rationale", "next_step", "held_open")
    for key in expected[:3]:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise EvaluationContractError(f"A paired response requires a non-blank {key}")
    held = value.get("held_open")
    if not isinstance(held, (list, tuple)) or not held or any(
        not isinstance(item, str) or not item.strip() for item in held
    ):
        raise EvaluationContractError("A paired response requires held-open tensions")
    return {
        "direction": value["direction"].strip(),
        "rationale": value["rationale"].strip(),
        "next_step": value["next_step"].strip(),
        "held_open": [item.strip() for item in held],
    }


def _balanced_assignments(prompt_ids: Sequence[str], blind_key: bytes) -> dict[str, str]:
    if len(blind_key) < 16:
        raise EvaluationContractError("The blinding key must contain at least 128 bits")
    ranked = sorted(
        prompt_ids,
        key=lambda prompt_id: hmac.new(
            blind_key, prompt_id.encode("utf-8"), hashlib.sha256
        ).digest(),
    )
    graph_as_a = set(ranked[: len(ranked) // 2])
    return {
        prompt_id: ("graph" if prompt_id in graph_as_a else "neutral")
        for prompt_id in prompt_ids
    }


def build_blinded_bundle(
    protocol: Mapping[str, Any],
    paired_outputs: Mapping[str, Any],
    *,
    blind_key: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an evaluator packet and a separate condition key.

    The packet contains no execution-mode, renderer, provider, route, or condition labels.
    The key must not be given to the evaluator until the score sheet is sealed.
    """

    protocol_hash = preregistration_hash(protocol)
    if paired_outputs.get("protocol_hash") != protocol_hash:
        raise EvaluationContractError("Paired outputs do not bind the frozen preregistration")
    raw_pairs = paired_outputs.get("pairs")
    if not isinstance(raw_pairs, list):
        raise EvaluationContractError("Paired outputs require a pair list")
    by_id = {
        item.get("prompt_id"): item
        for item in raw_pairs
        if isinstance(item, Mapping) and isinstance(item.get("prompt_id"), str)
    }
    prompt_ids = [item["prompt_id"] for item in protocol["prompts"]]
    if set(by_id) != set(prompt_ids) or len(raw_pairs) != len(prompt_ids):
        raise EvaluationContractError("Paired outputs must contain every frozen prompt exactly once")
    assignments = _balanced_assignments(prompt_ids, blind_key)
    pairs: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for prompt in protocol["prompts"]:
        prompt_id = prompt["prompt_id"]
        raw = by_id[prompt_id]
        if raw.get("intention") != prompt["intention"]:
            raise EvaluationContractError("A paired output changed a preregistered intention")
        graph = _bounded_response(raw.get("graph"))
        neutral = _bounded_response(raw.get("neutral"))
        a_condition = assignments[prompt_id]
        b_condition = "neutral" if a_condition == "graph" else "graph"
        condition_payload = {"graph": graph, "neutral": neutral}
        pairs.append(
            {
                "prompt_id": prompt_id,
                "intention": prompt["intention"],
                "response_A": condition_payload[a_condition],
                "response_B": condition_payload[b_condition],
            }
        )
        mapping.append(
            {
                "prompt_id": prompt_id,
                "A": a_condition,
                "B": b_condition,
                "graph_response_hash": stable_hash(graph),
                "neutral_response_hash": stable_hash(neutral),
            }
        )
    packet_body = {
        "schema_version": "seed-blind-packet:v0.1",
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": protocol_hash,
        "instructions": protocol["evaluation"]["blind_instructions"],
        "rubric": protocol["evaluation"]["rubric"],
        "pairs": pairs,
    }
    packet = {**packet_body, "packet_hash": stable_hash(packet_body)}
    key_body = {
        "schema_version": "seed-blinding-key:v0.1",
        "protocol_hash": protocol_hash,
        "packet_hash": packet["packet_hash"],
        "mapping": mapping,
    }
    key_record = {**key_body, "key_record_hash": stable_hash(key_body)}
    return packet, key_record


def score_sheet_template(packet: Mapping[str, Any]) -> dict[str, Any]:
    metric_ids = [item["metric_id"] for item in packet["rubric"]]
    body = {
        "schema_version": "seed-blind-scores:v0.1",
        "protocol_hash": packet["protocol_hash"],
        "packet_hash": packet["packet_hash"],
        "evaluator_id": "",
        "evaluated_at": "",
        "sealed_before_unblinding": False,
        "scores": [
            {
                "prompt_id": pair["prompt_id"],
                "A": {metric_id: None for metric_id in metric_ids},
                "B": {metric_id: None for metric_id in metric_ids},
                "notes": "",
            }
            for pair in packet["pairs"]
        ],
    }
    return {**body, "score_sheet_hash": None}


def seal_score_sheet(score_sheet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate completed blinded scores and add a hash before unblinding."""

    body = dict(score_sheet)
    body.pop("score_sheet_hash", None)
    if not isinstance(body.get("evaluator_id"), str) or not body["evaluator_id"].strip():
        raise EvaluationContractError("A sealed score sheet requires an evaluator ID")
    if not isinstance(body.get("evaluated_at"), str) or not body["evaluated_at"].strip():
        raise EvaluationContractError("A sealed score sheet requires an evaluation timestamp")
    body["sealed_before_unblinding"] = True
    scores = body.get("scores")
    if not isinstance(scores, list) or not scores:
        raise EvaluationContractError("A sealed score sheet requires scores")
    for item in scores:
        if not isinstance(item, Mapping):
            raise EvaluationContractError("Every score record must be an object")
        for label in ("A", "B"):
            values = item.get(label)
            if not isinstance(values, Mapping) or not values:
                raise EvaluationContractError("Every response requires all rubric scores")
            for score in values.values():
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 5:
                    raise EvaluationContractError("Every rubric score must be numeric from 1 to 5")
    return {**body, "score_sheet_hash": stable_hash(body)}


def analyze_blinded_scores(
    protocol: Mapping[str, Any],
    packet: Mapping[str, Any],
    key_record: Mapping[str, Any],
    sealed_scores: Mapping[str, Any],
) -> dict[str, Any]:
    """Unblind a sealed score sheet and apply the preregistered decision rule."""

    protocol_hash = preregistration_hash(protocol)
    if any(
        artifact.get("protocol_hash") != protocol_hash
        for artifact in (packet, key_record, sealed_scores)
    ):
        raise EvaluationContractError("Evaluation artifacts do not share one protocol hash")
    packet_body = {key: value for key, value in packet.items() if key != "packet_hash"}
    if packet.get("packet_hash") != stable_hash(packet_body):
        raise EvaluationContractError("The blinded packet hash is invalid")
    key_body = {key: value for key, value in key_record.items() if key != "key_record_hash"}
    if key_record.get("key_record_hash") != stable_hash(key_body):
        raise EvaluationContractError("The blinding key record hash is invalid")
    score_body = {key: value for key, value in sealed_scores.items() if key != "score_sheet_hash"}
    if not sealed_scores.get("sealed_before_unblinding") or sealed_scores.get(
        "score_sheet_hash"
    ) != stable_hash(score_body):
        raise EvaluationContractError("Scores must be sealed and hash-bound before unblinding")
    if any(
        artifact.get("packet_hash") != packet["packet_hash"]
        for artifact in (key_record, sealed_scores)
    ):
        raise EvaluationContractError("Evaluation artifacts do not share one blinded packet")

    metric_ids = [item["metric_id"] for item in protocol["evaluation"]["rubric"]]
    mapping = {item["prompt_id"]: item for item in key_record["mapping"]}
    scores = {item["prompt_id"]: item for item in sealed_scores["scores"]}
    prompt_ids = [item["prompt_id"] for item in protocol["prompts"]]
    if set(mapping) != set(prompt_ids) or set(scores) != set(prompt_ids):
        raise EvaluationContractError("Scores and condition mapping must cover every prompt")

    metric_deltas = {metric_id: [] for metric_id in metric_ids}
    prompt_results: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        score = scores[prompt_id]
        if set(score["A"]) != set(metric_ids) or set(score["B"]) != set(metric_ids):
            raise EvaluationContractError("A score record changed the frozen rubric")
        graph_label = "A" if mapping[prompt_id]["A"] == "graph" else "B"
        neutral_label = "B" if graph_label == "A" else "A"
        deltas = {
            metric_id: round(
                float(score[graph_label][metric_id]) - float(score[neutral_label][metric_id]),
                3,
            )
            for metric_id in metric_ids
        }
        for metric_id, delta in deltas.items():
            metric_deltas[metric_id].append(delta)
        composite = round(sum(deltas.values()) / len(deltas), 3)
        prompt_results.append(
            {
                "prompt_id": prompt_id,
                "graph_minus_neutral": deltas,
                "composite_difference": composite,
                "outcome": "graph" if composite > 0 else "neutral" if composite < 0 else "tie",
            }
        )

    metric_means = {
        metric_id: round(sum(values) / len(values), 3)
        for metric_id, values in metric_deltas.items()
    }
    primary_difference = round(sum(metric_means.values()) / len(metric_means), 3)
    graph_wins = sum(item["composite_difference"] > 0 for item in prompt_results)
    neutral_wins = sum(item["composite_difference"] < 0 for item in prompt_results)
    rule = protocol["evaluation"]["decision_rule"]
    threshold = float(rule["minimum_mean_difference"])
    required_wins = int(rule["minimum_prompt_wins"])
    if primary_difference >= threshold and graph_wins >= required_wins:
        primary_result = "graph_practical_advantage"
    elif primary_difference <= -threshold and neutral_wins >= required_wins:
        primary_result = "neutral_practical_advantage"
    else:
        primary_result = "null_effect"
    null_metrics = [
        metric_id for metric_id, delta in metric_means.items() if abs(delta) < threshold
    ]
    report_body = {
        "schema_version": "seed-evaluation-report:v0.1",
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": protocol_hash,
        "packet_hash": packet["packet_hash"],
        "score_sheet_hash": sealed_scores["score_sheet_hash"],
        "primary_result": primary_result,
        "primary_graph_minus_neutral": primary_difference,
        "graph_prompt_wins": graph_wins,
        "neutral_prompt_wins": neutral_wins,
        "tied_prompts": len(prompt_results) - graph_wins - neutral_wins,
        "metric_graph_minus_neutral": metric_means,
        "null_effect_metrics": null_metrics,
        "prompt_results": prompt_results,
        "interpretation_boundary": (
            "This small paired harness can detect a preregistered practical difference. "
            "It cannot establish consciousness, personhood, growth, or general superiority."
        ),
    }
    return {**report_body, "report_hash": stable_hash(report_body)}
