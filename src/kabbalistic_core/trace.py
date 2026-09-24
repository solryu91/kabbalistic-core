"""Paired technical and symbolic traces derived from the same executed events."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable

from .models import CognitiveEvent, MindState, Sefirah, World, canonical_json, stable_hash
from .yesod import validate_yesod_field


GENESIS_EVENT_HASH = "0" * 64


class TraceIntegrityError(ValueError):
    pass


class TraceRecorder:
    def __init__(self) -> None:
        self._events: list[CognitiveEvent] = []

    @property
    def events(self) -> tuple[CognitiveEvent, ...]:
        return tuple(self._events)

    @classmethod
    def continue_from(cls, events: Iterable[CognitiveEvent]) -> "TraceRecorder":
        materialized = tuple(events)
        validate_trace_chain(materialized)
        recorder = cls()
        recorder._events.extend(materialized)
        return recorder

    def record(
        self,
        state: MindState,
        *,
        event_type: str,
        technical: dict[str, Any],
        symbolic: dict[str, Any],
        state_hash_before: str,
        node: Sefirah | None = None,
        world: World | None = None,
    ) -> CognitiveEvent:
        if self._events and state_hash_before != self._events[-1].state_hash_after:
            raise TraceIntegrityError(
                "New event does not begin from the previous event's semantic state"
            )
        event = CognitiveEvent(
            sequence=len(self._events) + 1,
            event_type=event_type,
            node=node or state.current_node,
            world=world or state.current_world,
            technical=technical,
            symbolic=symbolic,
            state_hash_before=state_hash_before,
            state_hash_after=state.state_hash,
            previous_event_hash=(self._events[-1].event_hash if self._events else GENESIS_EVENT_HASH),
            event_hash="",
        )
        event = replace(event, event_hash=stable_hash(event.hash_payload()))
        self._events.append(event)
        return event


def validate_trace_chain(events: Iterable[CognitiveEvent]) -> None:
    previous_hash = GENESIS_EVENT_HASH
    previous_state_hash: str | None = None
    for expected_sequence, event in enumerate(events, start=1):
        if event.sequence != expected_sequence:
            raise TraceIntegrityError(
                f"Trace sequence mismatch: expected {expected_sequence}, got {event.sequence}"
            )
        if event.previous_event_hash != previous_hash:
            raise TraceIntegrityError(f"Broken previous-event link at sequence {event.sequence}")
        if previous_state_hash is not None and event.state_hash_before != previous_state_hash:
            raise TraceIntegrityError(f"Broken semantic-state link at sequence {event.sequence}")
        expected_hash = stable_hash(event.hash_payload())
        if event.event_hash != expected_hash:
            raise TraceIntegrityError(f"Event payload was altered at sequence {event.sequence}")
        previous_hash = event.event_hash
        previous_state_hash = event.state_hash_after


def validate_completed_trace(state: MindState, events: Iterable[CognitiveEvent]) -> None:
    materialized = tuple(events)
    if not materialized:
        raise TraceIntegrityError("A completed run cannot have an empty event trace")
    validate_trace_chain(materialized)
    if materialized[-1].state_hash_after != state.state_hash:
        raise TraceIntegrityError("The supplied final state is not bound to the final trace event")
    if state.pre_embodiment_yesod is not None:
        validate_yesod_field(state.pre_embodiment_yesod)
    if state.yesod_field is None:
        raise TraceIntegrityError("A completed run is missing its post-embodiment Yesod packet")
    validate_yesod_field(state.yesod_field)


def _json_value(value: Any) -> Any:
    return json.loads(canonical_json(value))


def technical_trace_document(
    state: MindState,
    events: Iterable[CognitiveEvent],
) -> dict[str, Any]:
    materialized = tuple(events)
    validate_completed_trace(state, materialized)
    return {
        "format": "kabbalistic-core-technical-trace:v0.0.1",
        "run_id": state.run_id,
        "graph_version": state.graph_version,
        "final_state_hash": state.state_hash,
        "final_event_hash": materialized[-1].event_hash if materialized else GENESIS_EVENT_HASH,
        "events": [_json_value(event) for event in materialized],
    }


def symbolic_trace_text(state: MindState, events: Iterable[CognitiveEvent]) -> str:
    """Render symbolic language from recorded facts; it cannot diverge from execution."""

    materialized = tuple(events)
    validate_completed_trace(state, materialized)
    lines = [
        "Kabbalistic Core — Symbolic Execution Trace",
        f"Run: {state.run_id}",
        f"Graph: {state.graph_version}",
        "",
    ]
    for event in materialized:
        title = str(event.symbolic.get("title", event.event_type.replace("_", " ").title()))
        meaning = str(event.symbolic.get("meaning", "No symbolic annotation supplied."))
        lines.extend(
            (
                f"{event.sequence:02d}. {event.node.value.title()} · {title}",
                f"    {meaning}",
                f"    State: {event.state_hash_after[:12]} · Event: {event.event_hash[:12]}",
                "",
            )
        )
    lines.extend(
        (
            "Embodied response",
            f"    {state.output}",
            "",
            "Unresolved tensions",
            *(f"    - {item}" for item in state.unresolved_tensions),
            "",
            f"Final semantic state: {state.state_hash}",
        )
    )
    return "\n".join(lines) + "\n"


def write_run_artifacts(
    output_dir: str | Path,
    state: MindState,
    events: Iterable[CognitiveEvent],
) -> tuple[Path, Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    materialized = tuple(events)
    technical_path = target / "technical_trace.json"
    symbolic_path = target / "symbolic_trace.txt"
    final_state_path = target / "final_state.json"
    technical_path.write_text(
        json.dumps(technical_trace_document(state, materialized), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    symbolic_path.write_text(symbolic_trace_text(state, materialized), encoding="utf-8")
    final_state_path.write_text(
        json.dumps(
            {
                "format": "kabbalistic-core-final-state:v0.0.1",
                "run_id": state.run_id,
                "state_hash": state.state_hash,
                "semantic_state": _json_value(state.semantic_payload()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return technical_path, symbolic_path, final_state_path
