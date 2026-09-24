"""Pluggable proposal and language boundaries for the cognitive engine.

Providers may contribute typed language, but the graph remains the owner of state,
permissions, transitions, feedback lineage, and memory decisions.
"""

from __future__ import annotations

from typing import Protocol

from .archetypes import ArchetypeRegistry
from .cores import (
    InitialSnapshot,
    build_mutual_observations,
    build_three_views,
    derive_direction_candidates,
)
from .models import (
    CoreObservation,
    CoreView,
    EmbodiedResponse,
    KernelActivation,
    MindState,
    TiferetProposal,
)


class CoreViewProvider(Protocol):
    """Build exactly one initial view for each of Form, Flow, and Accord."""

    provider_id: str

    def build_views(
        self,
        snapshot: InitialSnapshot,
        activations: tuple[KernelActivation, ...],
        registry: ArchetypeRegistry,
    ) -> tuple[CoreView, ...]:
        """Return the three typed core views from one untouched snapshot."""


class OutputRenderer(Protocol):
    """Render an already-coalesced graph state into one embodied response."""

    renderer_id: str

    def render(self, state: MindState) -> EmbodiedResponse:
        """Return language only; do not mutate graph state or authority."""


class ObservationProvider(Protocol):
    """Run the bounded second pass after all initial core views exist."""

    provider_id: str

    def build_observations(
        self,
        views: tuple[CoreView, ...],
    ) -> tuple[CoreObservation, ...]:
        """Return exactly six directed, proposition-responsive evaluations."""


class TiferetProvider(Protocol):
    """Derive candidate directions from the actual view and observation field."""

    provider_id: str

    def propose(
        self,
        intention: str,
        views: tuple[CoreView, ...],
        observations: tuple[CoreObservation, ...],
    ) -> TiferetProposal:
        """Return candidates and tensions; the graph still assesses and selects them."""


class DeterministicCoreViewProvider:
    """The original rule-based provider retained as the trusted default."""

    provider_id = "deterministic-cores:v0.0.1"

    def build_views(
        self,
        snapshot: InitialSnapshot,
        activations: tuple[KernelActivation, ...],
        registry: ArchetypeRegistry,
    ) -> tuple[CoreView, ...]:
        return build_three_views(snapshot, activations, registry)


class DeterministicObservationProvider:
    """Content-addressed fallback evaluation without a language model."""

    provider_id = "deterministic-observations:v0.2"

    def build_observations(
        self,
        views: tuple[CoreView, ...],
    ) -> tuple[CoreObservation, ...]:
        return build_mutual_observations(views)


class DeterministicTiferetProvider:
    """Transparent fallback proposal field derived from the supplied text."""

    provider_id = "deterministic-tiferet-proposals:v0.2"

    def propose(
        self,
        intention: str,
        views: tuple[CoreView, ...],
        observations: tuple[CoreObservation, ...],
    ) -> TiferetProposal:
        _ = intention
        candidates = derive_direction_candidates(views, observations)
        shared_ground = tuple(
            dict.fromkeys(
                item
                for observation in observations
                for item in observation.agreements
            )
        )[:6]
        unresolved = tuple(
            dict.fromkeys(
                item
                for observation in observations
                for item in observation.disagreements
            )
        )[:6]
        return TiferetProposal(
            shared_ground=shared_ground
            or ("The deterministic evaluation established no shared proposition.",),
            unresolved_tensions=unresolved
            or ("The deterministic evaluation established no explicit disagreement.",),
            candidates=candidates,
            provider_id=self.provider_id,
        )


class DeterministicOutputRenderer:
    """The original inspectable wording path, expressed as typed sections."""

    renderer_id = "deterministic-renderer:v0.0.1"

    def render(self, state: MindState) -> EmbodiedResponse:
        if state.coalescence is None:
            raise ValueError("An embodied response requires a Tiferet coalescence")
        if state.response_contract is None:
            raise ValueError("An embodied response requires a graph-owned response contract")
        return EmbodiedResponse(
            direction=state.response_contract.direction,
            rationale=state.response_contract.rationale_basis,
            next_step=state.response_contract.next_step,
            held_open=state.response_contract.held_open,
            cited_evidence_refs=(),
            renderer_id=self.renderer_id,
            response_contract_hash=state.response_contract.contract_hash,
        )
