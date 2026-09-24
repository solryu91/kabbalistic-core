"""Command-line doorway for one deterministic cognitive cycle."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import CognitiveEngine, CycleRequest
from .trace import write_run_artifacts


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kabbalistic-core",
        description="Run one deterministic Form–Flow–Accord cognitive cycle.",
    )
    parser.add_argument("--intention", required=True, help="The single intention bound at Keter.")
    parser.add_argument("--symbol", help="Optional symbol to pass through bounded recursion.")
    parser.add_argument(
        "--kernels",
        type=_csv,
        default=(),
        help="Comma-separated functional kernel IDs.",
    )
    parser.add_argument(
        "--evidence",
        type=_csv,
        default=(),
        help="Comma-separated local evidence references; no files are opened by this release.",
    )
    parser.add_argument(
        "--prior-feedback",
        help="Optional feedback already observed from a previous Malkhut embodiment.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Directory for the paired traces and final semantic state.",
    )
    parser.add_argument(
        "--no-memory-proposal",
        action="store_true",
        help="Skip creation of the consent-gated durable-memory proposal.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = CognitiveEngine()
    request = CycleRequest(
        intention=args.intention,
        symbol=args.symbol,
        kernel_ids=args.kernels,
        evidence_refs=args.evidence,
        prior_feedback=args.prior_feedback,
        propose_memory=not args.no_memory_proposal,
    )
    result = engine.run(request)
    artifact_paths = write_run_artifacts(Path(args.output_dir), result.state, result.events)

    print(result.output)
    print()
    print(f"Run: {result.run_id}")
    print(f"Yesod context packet: {result.context_packet_hash}")
    print(f"Yesod feedback-state packet: {result.feedback_state_packet_hash}")
    print(f"Da'at: {result.state.gate_result.decision.value}")
    for path in artifact_paths:
        print(f"Wrote: {path.resolve()}")
    return 0
