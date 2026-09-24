"""Guided local CLI for the single released continuity proof."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys

from .continuity import ContinuityService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seed-continuity",
        description=(
            "Run the frozen v0.1 Session A -> consented commit -> independent "
            "Session B continuity proof. This command stops before correction."
        ),
    )
    parser.add_argument("--store", required=True, type=Path, help="New SQLite content-store path.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Separate authority-checkpoint path; exclude it from content-only backups.",
    )
    parser.add_argument("--owner", required=True, help="Local owner principal label.")
    parser.add_argument("--intention", required=True, help="One bounded test intention.")
    parser.add_argument("--memory", required=True, help="One exact episodic statement to propose.")
    parser.add_argument(
        "--key-env",
        default="SEED_CONTINUITY_MASTER_KEY_B64",
        help="Environment variable containing a base64-encoded 32-byte master key.",
    )
    return parser


def _load_key(environment_name: str) -> bytes:
    encoded = os.environ.get(environment_name)
    if not encoded:
        raise ValueError(
            f"Set {environment_name} to a base64-encoded 32-byte key. "
            "The key is never stored in the database or checkpoint."
        )
    try:
        key = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError(f"{environment_name} is not valid base64.") from exc
    if len(key) != 32:
        raise ValueError(f"{environment_name} must decode to exactly 32 bytes.")
    return key


def _acknowledge(service: ContinuityService, result: dict[str, object]) -> dict[str, object]:
    delivery = result["delivery"]
    assert isinstance(delivery, dict)
    return service.acknowledge_delivery(
        str(result["session_id"]),
        delivery_id=str(delivery["delivery_id"]),
        response_hash=str(result["response_hash"]),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        key = _load_key(args.key_env)
        if args.store.exists() or args.checkpoint.exists():
            raise FileExistsError(
                "This proof command refuses existing state. Choose new store and checkpoint paths."
            )
        service_a = ContinuityService.create(
            args.store,
            args.checkpoint,
            key,
            owner_principal=args.owner,
        )
        session_a = service_a.open_session(principal=args.owner, intention=args.intention)
        response_a = service_a.respond(
            session_a["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        ack_a = _acknowledge(service_a, response_a)
        proposal = service_a.propose_episode(
            session_a["session_id"],
            principal=args.owner,
            content=args.memory,
        )
        print("\nVISIBLE MEMORY PROPOSAL\n")
        print(json.dumps(proposal, indent=2, ensure_ascii=False, sort_keys=True))
        expected = f"GRANT {proposal['canonical_terms_hash']}"
        entered = input(
            "\nTo grant this exact proposal, type the following line exactly:\n"
            f"{expected}\n> "
        ).strip()
        if entered != expected:
            decision = service_a.decide_proposal(
                proposal_ref=proposal["proposal_ref"],
                principal=args.owner,
                decision="reject",
                canonical_terms_hash=proposal["canonical_terms_hash"],
                reason="Exact visible grant string was not entered.",
            )
            closed_a = service_a.close_session(session_a["session_id"], principal=args.owner)
            service_a.close()
            print(json.dumps({"decision": decision, "session_a_close": closed_a}, indent=2))
            return 2
        consent = service_a.decide_proposal(
            proposal_ref=proposal["proposal_ref"],
            principal=args.owner,
            decision="grant",
            canonical_terms_hash=proposal["canonical_terms_hash"],
        )
        commit = service_a.commit_granted_proposal(
            proposal_ref=proposal["proposal_ref"],
            consent_decision_ref=consent["decision_ref"],
        )
        close_a = service_a.close_session(session_a["session_id"], principal=args.owner)
        instance_a = service_a.runtime_instance_id
        service_a.close()

        service_b = ContinuityService.open(args.store, args.checkpoint, key)
        session_b = service_b.open_session(principal=args.owner, intention=args.intention)
        history = service_b.respond(
            session_b["session_id"], include_history=True, condition_label="history"
        )
        ack_history = _acknowledge(service_b, history)
        close_b = service_b.close_session(session_b["session_id"], principal=args.owner)
        control_session = service_b.open_session(principal=args.owner, intention=args.intention)
        no_history = service_b.respond(
            control_session["session_id"],
            include_history=False,
            condition_label="matched_no_history",
        )
        ack_no_history = _acknowledge(service_b, no_history)
        comparison = service_b.causal_comparison(history, no_history)
        close_control = service_b.close_session(
            control_session["session_id"], principal=args.owner
        )
        verification = service_b.verify()
        instance_b = service_b.runtime_instance_id
        service_b.close()

        evidence = {
            "schema_version": "continuity-evidence:v0.1",
            "session_a": {
                "open": session_a,
                "precommit_response": response_a,
                "delivery_acknowledgement": ack_a,
                "proposal": proposal,
                "consent": consent,
                "commit": commit,
                "close": close_a,
            },
            "session_b": {
                "open": session_b,
                "history": history,
                "history_acknowledgement": ack_history,
                "matched_no_history": no_history,
                "matched_no_history_acknowledgement": ack_no_history,
                "comparison": comparison,
                "close": close_b,
                "matched_control_session": control_session,
                "matched_control_close": close_control,
            },
            "independent_runtime_instances": instance_a != instance_b,
            "verification": verification,
            "next_phase": "correction_review_required_not_implemented",
        }
        print("\nCONTINUITY EVIDENCE\n")
        print(json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if comparison["history_effect_observed"] else 3
    except Exception as exc:
        print(f"continuity proof failed closed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
