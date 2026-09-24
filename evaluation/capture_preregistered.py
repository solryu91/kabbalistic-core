"""Capture the frozen prompt pairs and create separate blinded artifacts.

Run from the repository root with ``PYTHONPATH=src`` while the local model endpoint is
ready.  The script never starts a remote service and never writes durable Seed memory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
from typing import Any

from kabbalistic_core.evaluation import (
    build_blinded_bundle,
    load_preregistration,
    preregistration_hash,
    score_sheet_template,
)
from kabbalistic_core.poc import SeedService


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace frozen evaluation artifact: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _response_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "direction": value["direction"],
        "rationale": value["rationale"],
        "next_step": value["next_step"],
        "held_open": value["held_open"],
    }


def capture(protocol: dict[str, Any], service: SeedService) -> dict[str, Any]:
    protocol_hash = preregistration_hash(protocol)
    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    prompt_total = len(protocol["prompts"])
    for prompt_index, prompt in enumerate(protocol["prompts"], start=1):
        prompt_id = prompt["prompt_id"]
        print(f"[{prompt_index}/{prompt_total}] Capturing frozen pair {prompt_id}...", flush=True)
        try:
            graph = service.run_graph(
                {
                    "intention": prompt["intention"],
                    "symbol": None,
                    "kernel_ids": [],
                    "prior_feedback": None,
                }
            )
            control = service.run_control(graph["session_id"])
            comparison = control["comparison"]
            if graph["execution_mode"] != "neural_graph":
                raise RuntimeError("graph condition used fallback")
            if graph["model"]["call_count"] != 8 or control.get("model_call_count") != 8:
                raise RuntimeError("a condition did not complete exactly eight calls")
            if not comparison.get("comparison_eligible") or not comparison.get("budget_match"):
                raise RuntimeError("the pair did not preserve its model/config/budget binding")
            pairs.append(
                {
                    "prompt_id": prompt_id,
                    "intention": prompt["intention"],
                    "graph": _response_fields(graph["response"]),
                    "neutral": _response_fields(control["response"]),
                    "bindings": {
                        "model_id": comparison["graph_model_id"],
                        "model_binding_hash": comparison["graph_model_binding_hash"],
                        "adapter_config_hash": graph["model"]["adapter_config_hash"],
                        "retrieval_snapshot_hash": comparison[
                            "shared_retrieval_snapshot_hash"
                        ],
                        "graph_calls": comparison["full_tree"]["model_call_count"],
                        "neutral_calls": comparison["neutral_control"]["model_call_count"],
                        "graph_token_budget": comparison["full_tree"][
                            "model_token_budget"
                        ],
                        "neutral_token_budget": comparison["neutral_control"][
                            "model_token_budget"
                        ],
                    },
                }
            )
            print(f"[{prompt_index}/{prompt_total}] {prompt_id}: eligible pair captured.", flush=True)
        except (KeyError, RuntimeError, ValueError) as exc:
            exclusions.append({"prompt_id": prompt_id, "reason": str(exc)})
            print(f"[{prompt_index}/{prompt_total}] {prompt_id}: excluded — {exc}", flush=True)
    return {
        "schema_version": "seed-paired-outputs:v0.1",
        "protocol_hash": protocol_hash,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "pairs": pairs,
        "exclusions": exclusions,
        "durable_memory_writes": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the frozen local graph/neutral response pairs."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("evaluation/preregistered_v0.1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/v0.1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_preregistration(args.protocol)
    service = SeedService()
    status = service.health()["model"]
    if status["status"] != "ready":
        raise RuntimeError(
            "The preregistered run requires the bound local Qwen model; fallback is excluded."
        )
    captured = capture(protocol, service)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "paired_outputs.json"
    _write_new(paired_path, captured)
    if captured["exclusions"] or len(captured["pairs"]) != len(protocol["prompts"]):
        print(f"Captured incomplete run with exclusions: {paired_path.resolve()}")
        print("No blinded packet was created and no prompt was replaced.")
        return 2

    packet, key_record = build_blinded_bundle(
        protocol,
        captured,
        blind_key=secrets.token_bytes(32),
    )
    packet_path = args.output_dir / "blinded_packet.json"
    key_path = args.output_dir / "condition_key.keep_separate.json"
    scores_path = args.output_dir / "blank_scores.json"
    _write_new(packet_path, packet)
    _write_new(key_path, key_record)
    _write_new(scores_path, score_sheet_template(packet))
    print(f"Paired outputs: {paired_path.resolve()}")
    print(f"Evaluator packet: {packet_path.resolve()}")
    print(f"Blank scores: {scores_path.resolve()}")
    print(f"Condition key (keep separate): {key_path.resolve()}")
    print("No condition was declared a winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
