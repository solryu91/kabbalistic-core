"""Explicit capability membrane. v0.0.1 performs no external effects."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Capability


@dataclass(frozen=True, slots=True)
class PermissionSet:
    granted: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.PROPOSE_MEMORY})
    )

    def allows(self, capability: Capability) -> bool:
        return capability in self.granted

    def require(self, capability: Capability) -> None:
        if not self.allows(capability):
            raise PermissionError(f"Capability is not granted: {capability.value}")

    def with_grant(self, capability: Capability) -> "PermissionSet":
        return PermissionSet(self.granted | {capability})

    def without(self, capability: Capability) -> "PermissionSet":
        return PermissionSet(self.granted - {capability})
