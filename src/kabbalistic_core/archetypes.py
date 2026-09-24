"""Bounded archetypal influences for salience, questions, and shadow checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Iterable

from .models import CoreId, KernelActivation


@dataclass(frozen=True, slots=True)
class ArchetypeKernelSpec:
    """A functional influence contract, never an independent persona or authority."""

    kernel_id: str
    values: tuple[str, ...]
    questions: tuple[str, ...]
    shadow_watch: tuple[str, ...]
    salience_adjustments: tuple[str, ...]
    target_cores: tuple[CoreId, ...]
    keyword_triggers: tuple[str, ...]
    forbidden_authorities: tuple[str, ...]


class ArchetypeRegistry:
    def __init__(self, version: str, specs: dict[str, ArchetypeKernelSpec]) -> None:
        self.version = version
        self._specs = dict(specs)

    @classmethod
    def load_default(cls) -> "ArchetypeRegistry":
        resource = files("kabbalistic_core.data").joinpath("archetypes.json")
        data = json.loads(resource.read_text(encoding="utf-8"))
        specs = {
            item["id"]: ArchetypeKernelSpec(
                kernel_id=item["id"],
                values=tuple(item["values"]),
                questions=tuple(item["questions"]),
                shadow_watch=tuple(item["shadow_watch"]),
                salience_adjustments=tuple(item["salience_adjustments"]),
                target_cores=tuple(CoreId(core) for core in item["target_cores"]),
                keyword_triggers=tuple(item["keyword_triggers"]),
                forbidden_authorities=tuple(item["forbidden_authorities"]),
            )
            for item in data["kernels"]
        }
        return cls(data["version"], specs)

    @property
    def kernel_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def spec(self, kernel_id: str) -> ArchetypeKernelSpec:
        try:
            return self._specs[kernel_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown archetypal kernel {kernel_id!r}; available: {', '.join(self.kernel_ids)}"
            ) from exc

    def activate(
        self,
        explicit_ids: Iterable[str],
        intention: str,
        *,
        infer_from_intention: bool = False,
    ) -> tuple[KernelActivation, ...]:
        """Activate explicit kernels, with optional visible keyword inference.

        Hidden activation is deliberately off by default. Duplicate identifiers collapse
        to one activation and canonical identifier sorting makes replay order-independent.
        """

        requested = {kernel_id.strip() for kernel_id in explicit_ids if kernel_id.strip()}
        inferred: set[str] = set()
        if infer_from_intention:
            normalized = intention.casefold()
            for spec in self._specs.values():
                if any(keyword.casefold() in normalized for keyword in spec.keyword_triggers):
                    inferred.add(spec.kernel_id)

        unknown = requested.difference(self._specs)
        if unknown:
            raise KeyError(f"Unknown archetypal kernels: {', '.join(sorted(unknown))}")

        activations: list[KernelActivation] = []
        for kernel_id in sorted(requested | inferred):
            spec = self._specs[kernel_id]
            explicit = kernel_id in requested
            reason = "explicitly requested" if explicit else "visible keyword match"
            activations.append(
                KernelActivation(
                    kernel_id=kernel_id,
                    reason=reason,
                    intensity=0.8 if explicit else 0.55,
                    target_cores=spec.target_cores,
                    questions_added=spec.questions,
                    shadow_watch=spec.shadow_watch,
                    salience_adjustments=spec.salience_adjustments,
                )
            )
        return tuple(activations)

    def influences_for(
        self,
        core_id: CoreId,
        activations: Iterable[KernelActivation],
    ) -> tuple[str, ...]:
        influences: list[str] = []
        for activation in sorted(activations, key=lambda item: item.kernel_id):
            if core_id in activation.target_cores:
                influences.extend(
                    f"{activation.kernel_id}: attend to {item}"
                    for item in activation.salience_adjustments
                )
        return tuple(influences)

    def questions_for(
        self,
        core_id: CoreId,
        activations: Iterable[KernelActivation],
    ) -> tuple[str, ...]:
        questions: list[str] = []
        for activation in sorted(activations, key=lambda item: item.kernel_id):
            if core_id in activation.target_cores:
                questions.extend(activation.questions_added)
        return tuple(questions)
