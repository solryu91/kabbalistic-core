from __future__ import annotations

from dataclasses import fields
import unittest

from helpers import default_snapshot

from kabbalistic_core.archetypes import ArchetypeRegistry
from kabbalistic_core.cores import build_three_views
from kabbalistic_core.engine import CognitiveEngine, CycleRequest
from kabbalistic_core.models import Capability, CoreId, RecursionStop, Sefirah, World
from kabbalistic_core.permissions import PermissionSet
from kabbalistic_core.recursion import (
    RecursionPathRegistry,
    RecursionPolicy,
    SymbolTransform,
    run_symbolic_recursion,
)


class EscalatingStrategy:
    def transform(self, symbol: str, depth: int, grounding_feedback: str | None) -> SymbolTransform:
        return SymbolTransform(
            transformed_symbol=f"{symbol} with requested file authority",
            form_observation="The request crosses a capability boundary.",
            flow_observation="The possibility remains available without execution.",
            accord_observation="The covenant requires explicit authority.",
            new_distinctions=("symbolic request versus granted capability",),
            new_relations=("interpretation to proposed action",),
            added_constraints=("No permission escalation through recursion.",),
            unresolved_polarity=("possibility / authority",),
            requested_capabilities=(Capability.WRITE_FILE,),
        )


class ConstraintOnlyStrategy:
    def transform(self, symbol: str, depth: int, grounding_feedback: str | None) -> SymbolTransform:
        return SymbolTransform(
            transformed_symbol=symbol,
            form_observation="No new distinction.",
            flow_observation="No new relation.",
            accord_observation="Stop without manufacturing meaning.",
            new_distinctions=(),
            new_relations=(),
            added_constraints=("Repetition is not transformation.",),
            unresolved_polarity=("open mystery / forced closure",),
        )


class PureNoDeltaStrategy:
    def transform(self, symbol: str, depth: int, grounding_feedback: str | None) -> SymbolTransform:
        return SymbolTransform(
            transformed_symbol=symbol,
            form_observation="No new form.",
            flow_observation="No new possibility.",
            accord_observation="No new relation.",
            new_distinctions=(),
            new_relations=(),
            added_constraints=(),
            unresolved_polarity=(),
        )


class RepeatingStateStrategy:
    def transform(self, symbol: str, depth: int, grounding_feedback: str | None) -> SymbolTransform:
        return SymbolTransform(
            transformed_symbol="unchanging transformed symbol",
            form_observation="The same form recurs.",
            flow_observation="The same possibility recurs.",
            accord_observation="The repeated state must be recognized.",
            new_distinctions=("constant distinction",),
            new_relations=("constant relation",),
            added_constraints=("constant constraint",),
            unresolved_polarity=("repeat / return",),
        )


class IncrementingStrategy:
    def transform(self, symbol: str, depth: int, grounding_feedback: str | None) -> SymbolTransform:
        return SymbolTransform(
            transformed_symbol=f"{symbol}|pass-{depth}",
            form_observation=f"Form pass {depth}.",
            flow_observation=f"Flow pass {depth}.",
            accord_observation=f"Accord pass {depth}.",
            new_distinctions=(f"distinction-{depth}",),
            new_relations=(f"relation-{depth}",),
            added_constraints=(f"constraint-{depth}",),
            unresolved_polarity=("continuity / change",),
        )


class PolarityOnlyStrategy:
    def transform(self, symbol: str, depth: int, grounding_feedback: str | None) -> SymbolTransform:
        return SymbolTransform(
            transformed_symbol=symbol,
            form_observation="The form remains stable.",
            flow_observation="The possibility remains stable.",
            accord_observation="Only the unresolved polarity changes.",
            new_distinctions=(),
            new_relations=(),
            added_constraints=(),
            unresolved_polarity=(f"polarity-{depth}",),
        )


class SymbolicRecursionTests(unittest.TestCase):
    def test_policy_rejects_depth_above_hard_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            RecursionPolicy(max_depth=6)

    def test_third_ungrounded_pass_stops_before_interpretation(self) -> None:
        frames = run_symbolic_recursion(
            "seed",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
        )
        self.assertEqual(3, len(frames))
        self.assertEqual(RecursionStop.GROUNDING_REQUIRED, frames[-1].stop_reason)
        self.assertEqual(
            frames[-1].symbol_state_hash_before,
            frames[-1].symbol_state_hash_after,
        )

    def test_five_is_absolute_execution_cap(self) -> None:
        frames = run_symbolic_recursion(
            "seed",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
            policy=RecursionPolicy(max_depth=5, event_budget=5),
            prior_malkhut_feedback="The embodied test returned a bounded observation.",
            strategy=IncrementingStrategy(),
        )
        self.assertEqual([1, 2, 3, 4, 5], [frame.depth for frame in frames])
        self.assertEqual(RecursionStop.MAX_DEPTH, frames[-1].stop_reason)

    def test_event_budget_stops_recursion(self) -> None:
        frames = run_symbolic_recursion(
            "seed",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
            policy=RecursionPolicy(max_depth=3, event_budget=1),
            prior_malkhut_feedback="grounded",
            strategy=IncrementingStrategy(),
        )
        self.assertEqual(RecursionStop.BUDGET_EXHAUSTED, frames[-1].stop_reason)

    def test_no_structured_delta_converges_immediately(self) -> None:
        frames = run_symbolic_recursion(
            "mirror",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
            strategy=PureNoDeltaStrategy(),
        )
        self.assertEqual(1, len(frames))
        self.assertEqual(RecursionStop.CONVERGED, frames[0].stop_reason)

    def test_constraint_only_return_is_a_real_semantic_delta(self) -> None:
        frames = run_symbolic_recursion(
            "mirror",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
            strategy=ConstraintOnlyStrategy(),
        )
        self.assertGreaterEqual(len(frames), 2)
        self.assertNotEqual(
            frames[0].symbol_state_hash_before,
            frames[0].symbol_state_hash_after,
        )
        self.assertEqual(RecursionStop.REPEATED_STATE, frames[-1].stop_reason)

    def test_repeated_semantic_state_is_detected(self) -> None:
        frames = run_symbolic_recursion(
            "mirror",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
            strategy=RepeatingStateStrategy(),
        )
        self.assertEqual(2, len(frames))
        self.assertEqual(RecursionStop.REPEATED_STATE, frames[-1].stop_reason)

    def test_recursion_cannot_escalate_permission(self) -> None:
        permissions = PermissionSet()
        granted_before = permissions.granted
        frames = run_symbolic_recursion(
            "door",
            starting_state_hash="state:before",
            permissions=permissions,
            strategy=EscalatingStrategy(),
        )
        self.assertEqual(RecursionStop.PERMISSION_ESCALATION, frames[-1].stop_reason)
        self.assertEqual(
            frames[-1].symbol_state_hash_before,
            frames[-1].symbol_state_hash_after,
        )
        self.assertEqual(granted_before, permissions.granted)
        self.assertFalse(permissions.allows(Capability.WRITE_FILE))

    def test_frame_parent_chain_is_linear_and_acyclic(self) -> None:
        frames = run_symbolic_recursion(
            "seed",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
            prior_malkhut_feedback="observed feedback",
        )
        seen: set[str] = set()
        for index, frame in enumerate(frames):
            with self.subTest(depth=frame.depth):
                self.assertNotIn(frame.recursion_id, seen)
                if index == 0:
                    self.assertIsNone(frame.parent_id)
                else:
                    self.assertEqual(frames[index - 1].recursion_id, frame.parent_id)
                seen.add(frame.recursion_id)

    def test_each_frame_carries_explicit_yetzirah_return_subgraph(self) -> None:
        registry = RecursionPathRegistry.default()
        self.assertEqual(
            ("rec:tiferet-hod", "rec:hod-yesod", "rec:yesod-tiferet"),
            registry.path_ids,
        )
        frames = run_symbolic_recursion(
            "seed",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
        )
        for frame in frames[:-1]:
            with self.subTest(depth=frame.depth):
                self.assertEqual(Sefirah.HOD, frame.source_sefirah)
                self.assertEqual(World.YETZIRAH, frame.source_world)
                self.assertEqual(Sefirah.TIFERET, frame.return_destination)
                self.assertEqual(registry.path_ids, frame.path_ids)
        stopped = frames[-1]
        self.assertEqual(RecursionStop.GROUNDING_REQUIRED, stopped.stop_reason)
        self.assertEqual(Sefirah.TIFERET, stopped.source_sefirah)
        self.assertEqual(World.BERIAH, stopped.source_world)
        self.assertEqual((), stopped.path_ids)

    def test_unresolved_polarity_is_part_of_symbolic_state_hash(self) -> None:
        frames = run_symbolic_recursion(
            "mirror",
            starting_state_hash="state:before",
            permissions=PermissionSet(),
            policy=RecursionPolicy(max_depth=2, event_budget=2),
            strategy=PolarityOnlyStrategy(),
        )
        self.assertEqual(2, len(frames))
        self.assertNotEqual(
            frames[0].symbol_state_hash_after,
            frames[1].symbol_state_hash_after,
        )


class ArchetypalKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ArchetypeRegistry.load_default()

    def test_activation_order_is_canonical_and_duplicate_free(self) -> None:
        activations = self.registry.activate(
            ("regeneration", "clear_sight", "liberation", "clear_sight"),
            "test intention",
        )
        self.assertEqual(
            ("clear_sight", "liberation", "regeneration"),
            tuple(item.kernel_id for item in activations),
        )

    def test_unknown_kernel_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.activate(("unknown_kernel",), "test intention")

    def test_each_kernel_changes_structured_core_salience(self) -> None:
        baseline = {item.core_id: item for item in build_three_views(default_snapshot(), (), self.registry)}
        for kernel_id in self.registry.kernel_ids:
            with self.subTest(kernel=kernel_id):
                activation = self.registry.activate((kernel_id,), "test intention")
                altered = {
                    item.core_id: item
                    for item in build_three_views(default_snapshot(), activation, self.registry)
                }
                targets = set(self.registry.spec(kernel_id).target_cores)
                self.assertTrue(targets)
                for core_id in targets:
                    self.assertNotEqual(
                        baseline[core_id].kernel_influences,
                        altered[core_id].kernel_influences,
                    )
                    self.assertGreater(
                        len(altered[core_id].open_questions),
                        len(baseline[core_id].open_questions),
                    )

    def test_kernel_activation_contract_has_no_authority_field(self) -> None:
        activation_fields = {item.name for item in fields(self.registry.activate(("clear_sight",), "x")[0])}
        forbidden_fields = {
            "permissions",
            "granted_capabilities",
            "memory_commit",
            "tool_call",
            "covenant_override",
        }
        self.assertTrue(activation_fields.isdisjoint(forbidden_fields))
        for kernel_id in self.registry.kernel_ids:
            forbidden = set(self.registry.spec(kernel_id).forbidden_authorities)
            self.assertIn("grant tools", forbidden)
            self.assertIn("commit memory", forbidden)
            self.assertIn("override gevurah", forbidden)

    def test_all_kernels_change_state_but_not_engine_permissions(self) -> None:
        engine = CognitiveEngine()
        permissions_before = engine.permissions
        baseline = engine.run(CycleRequest(intention="Protect freedom through observable boundaries."))
        activated = engine.run(
            CycleRequest(
                intention="Protect freedom through observable boundaries.",
                kernel_ids=self.registry.kernel_ids,
            )
        )
        self.assertNotEqual(baseline.yesod_packet_hash, activated.yesod_packet_hash)
        self.assertIs(permissions_before, engine.permissions)
        self.assertFalse(engine.permissions.allows(Capability.WRITE_FILE))
        output_events = [
            event for event in activated.events if event.event_type == "embodied_output_emitted"
        ]
        self.assertEqual([], output_events[0].technical["consequential_external_effects"])


if __name__ == "__main__":
    unittest.main()
