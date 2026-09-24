"""Run one non-evaluation smoke pair against the local model boundary."""

from __future__ import annotations

from kabbalistic_core.poc import SeedService


PILOT_INTENTION = (
    "Engineering smoke test: choose one reversible local check for a structured-output "
    "pipeline, name the stop condition, and preserve one uncertainty."
)


def main() -> int:
    service = SeedService()
    graph = service.run_graph(
        {"intention": PILOT_INTENTION, "symbol": None, "kernel_ids": []}
    )
    control = service.run_control(graph["session_id"])
    comparison = control["comparison"]
    print(f"graph_mode={graph['execution_mode']}")
    print(f"graph_calls={graph['model']['call_count']}")
    print(f"control_status={control['status']}")
    print(f"control_calls={control.get('model_call_count')}")
    print(f"budget_match={comparison['budget_match']}")
    print(f"comparison_eligible={comparison['comparison_eligible']}")
    return 0 if comparison["comparison_eligible"] and comparison["budget_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
