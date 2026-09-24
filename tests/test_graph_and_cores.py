from __future__ import annotations

from dataclasses import replace
from itertools import permutations
import unittest

from helpers import custom_view, default_views

from kabbalistic_core.cores import (
    build_mutual_observations,
    coalesce_at_tiferet,
    constraint_ref,
    observation_ref,
    proposition_ref,
)
from kabbalistic_core.graph import GraphDefinition, InvalidTransitionError
from kabbalistic_core.models import (
    CoreId,
    DirectionCandidate,
    EmbodiedResponse,
    NodeKind,
    ResponseContract,
    Sefirah,
    World,
    validate_response_realization,
)


class GraphValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphDefinition.load_default()

    def test_default_graph_contains_every_sefirah_and_daat_gate(self) -> None:
        self.assertEqual(set(Sefirah), set(self.graph.nodes))
        self.assertIn(Sefirah.DAAT, self.graph.nodes)
        self.assertEqual(NodeKind.GATE, self.graph.node(Sefirah.DAAT).kind)
        self.assertEqual(World.YETZIRAH, self.graph.node(Sefirah.DAAT).world)

    def test_world_enum_contains_only_the_four_worlds(self) -> None:
        self.assertEqual(
            {World.ATZILUT, World.BERIAH, World.YETZIRAH, World.ASSIAH},
            set(World),
        )

    def test_daat_cannot_be_downgraded_to_an_ordinary_sefirah(self) -> None:
        nodes = dict(self.graph.nodes)
        nodes[Sefirah.DAAT] = replace(
            nodes[Sefirah.DAAT],
            kind=NodeKind.SEFIRAH,
        )
        with self.assertRaisesRegex(ValueError, "must be declared as a gate"):
            GraphDefinition(self.graph.version, nodes, self.graph.paths)

    def test_daat_never_participates_in_an_ordinary_path(self) -> None:
        self.assertTrue(
            all(
                Sefirah.DAAT not in (path.source, path.target)
                for path in self.graph.paths
            )
        )

    def test_main_cycle_is_fully_declared_and_excludes_daat(self) -> None:
        self.assertEqual(Sefirah.KETER, self.graph.main_cycle[0])
        self.assertEqual(Sefirah.MALKHUT, self.graph.main_cycle[-1])
        self.assertNotIn(Sefirah.DAAT, self.graph.main_cycle)
        for source, target in zip(self.graph.main_cycle, self.graph.main_cycle[1:]):
            path = self.graph.transition(source, target)
            self.assertEqual((source, target), (path.source, path.target))

    def test_missing_required_node_is_rejected(self) -> None:
        nodes = dict(self.graph.nodes)
        nodes.pop(Sefirah.DAAT)
        with self.assertRaisesRegex(ValueError, "missing nodes"):
            GraphDefinition(self.graph.version, nodes, self.graph.paths)

    def test_duplicate_source_target_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate source/target"):
            GraphDefinition(
                self.graph.version,
                self.graph.nodes,
                self.graph.paths + (self.graph.paths[0],),
            )

    def test_node_and_path_world_mismatch_is_rejected(self) -> None:
        nodes = dict(self.graph.nodes)
        nodes[Sefirah.KETER] = replace(
            nodes[Sefirah.KETER],
            world=World.BERIAH,
        )
        with self.assertRaisesRegex(ValueError, "world_from"):
            GraphDefinition(self.graph.version, nodes, self.graph.paths)

    def test_undeclared_transition_is_rejected(self) -> None:
        with self.assertRaises(InvalidTransitionError):
            self.graph.transition(Sefirah.KETER, Sefirah.MALKHUT)


class MutualObservationTests(unittest.TestCase):
    def test_exactly_six_unique_directed_observations_are_produced(self) -> None:
        observations = build_mutual_observations(default_views())
        expected = set(permutations(CoreId, 2))
        actual = {(item.observer, item.observed) for item in observations}
        self.assertEqual(6, len(observations))
        self.assertEqual(expected, actual)
        self.assertTrue(all(item.observer != item.observed for item in observations))

    def test_each_observation_records_disagreement_and_self_shadow(self) -> None:
        observations = build_mutual_observations(default_views())
        for observation in observations:
            with self.subTest(pair=(observation.observer, observation.observed)):
                self.assertTrue(observation.disagreements)
                self.assertTrue(observation.what_other_sees)
                self.assertTrue(observation.what_other_misses)
                self.assertTrue(observation.observer_self_shadow)
                self.assertTrue(observation.request_to_other)

    def test_peer_proposition_changes_the_second_pass_evaluation(self) -> None:
        original_views = default_views()
        original = {
            (item.observer, item.observed): item
            for item in build_mutual_observations(original_views)
        }
        changed_views = tuple(
            replace(
                view,
                propositions=(
                    "Use a deliberately different Flow proposition about a concrete local trial.",
                    *view.propositions[1:],
                ),
            )
            if view.core_id == CoreId.FLOW
            else view
            for view in original_views
        )
        changed = {
            (item.observer, item.observed): item
            for item in build_mutual_observations(changed_views)
        }

        for pair in (
            (CoreId.FORM, CoreId.FLOW),
            (CoreId.ACCORD, CoreId.FLOW),
            (CoreId.FLOW, CoreId.FORM),
        ):
            with self.subTest(pair=pair):
                self.assertNotEqual(original[pair].evaluation_hash, changed[pair].evaluation_hash)
                self.assertNotEqual(original[pair].disagreements, changed[pair].disagreements)
        self.assertIn(
            proposition_ref(CoreId.FLOW, 0),
            changed[(CoreId.FORM, CoreId.FLOW)].basis_refs,
        )

    def test_missing_core_view_is_rejected(self) -> None:
        views = default_views()
        with self.assertRaisesRegex(ValueError, "exactly one complete view"):
            build_mutual_observations(views[:2])

    def test_duplicate_core_view_is_rejected_even_if_all_three_are_present(self) -> None:
        views = default_views()
        duplicate_form = replace(
            views[0],
            propositions=("A second, conflicting Form view.",),
        )
        with self.assertRaisesRegex(ValueError, "exactly one complete view"):
            build_mutual_observations((*views, duplicate_form))


class TiferetCoalescenceTests(unittest.TestCase):
    @staticmethod
    def _candidate(
        views,
        *,
        candidate_id: str,
        support_cores: tuple[CoreId, ...],
        observation_pairs: tuple[tuple[CoreId, CoreId], ...],
        next_step: str = "Run one reversible local comparison.",
        reversible: bool = True,
        requires_external_action: bool = False,
    ) -> DirectionCandidate:
        by_core = {view.core_id: view for view in views}
        hard_refs = tuple(
            constraint_ref(core_id, index)
            for core_id in (CoreId.FORM, CoreId.ACCORD)
            for index, _ in enumerate(by_core[core_id].constraints)
        )
        return DirectionCandidate(
            candidate_id=candidate_id,
            direction=f"Direction derived for {candidate_id}.",
            next_step=next_step,
            rationale=f"Rationale derived for {candidate_id}.",
            supporting_proposition_refs=tuple(
                proposition_ref(core_id, 0) for core_id in support_cores
            ),
            responding_observation_refs=tuple(
                observation_ref(observer, observed)
                for observer, observed in observation_pairs
            ),
            satisfied_constraint_refs=hard_refs,
            risks=("A bounded test may return a null result.",),
            reversible=reversible,
            requires_external_action=requires_external_action,
            provider_id="test-tiferet-provider:v1",
        )

    def test_tiferet_requires_all_six_directed_observations(self) -> None:
        views = default_views()
        observations = build_mutual_observations(views)
        with self.assertRaisesRegex(ValueError, "exactly six"):
            coalesce_at_tiferet("test intention", views, observations[:-1])

    def test_coalescence_is_order_independent(self) -> None:
        views = default_views()
        observations = build_mutual_observations(views)
        expected = coalesce_at_tiferet("test intention", views, observations)
        actual = coalesce_at_tiferet(
            "test intention",
            tuple(reversed(views)),
            tuple(reversed(observations)),
        )
        self.assertEqual(expected, actual)

    def test_contradiction_is_retained_instead_of_forced_closed(self) -> None:
        views = default_views()
        observations = build_mutual_observations(views)
        polarity, result = coalesce_at_tiferet("test intention", views, observations)
        self.assertTrue(polarity.unresolved_tensions)
        self.assertEqual(polarity.unresolved_tensions, result.unresolved_tensions)
        self.assertIn("tension", " ".join(result.unresolved_tensions).casefold())

    def test_two_matching_views_do_not_outvote_form_boundary(self) -> None:
        views = (
            custom_view(
                CoreId.FORM,
                proposition="Do not publish private material.",
                constraint="Publication is forbidden without explicit permission.",
            ),
            custom_view(
                CoreId.FLOW,
                proposition="Publish the material now.",
                constraint="Keep creative possibility alive.",
            ),
            custom_view(
                CoreId.ACCORD,
                proposition="Publish the material now.",
                constraint="The covenant remains prior to action.",
            ),
        )
        observations = build_mutual_observations(views)
        polarity, result = coalesce_at_tiferet("share private material", views, observations)
        self.assertIn(
            "Publication is forbidden without explicit permission.",
            polarity.hard_constraints,
        )
        self.assertNotEqual("Publish the material now.", result.selected_direction)
        self.assertTrue(result.unresolved_tensions)

    def test_tiferet_selects_supported_candidate_not_first_candidate(self) -> None:
        views = default_views()
        observations = build_mutual_observations(views)
        partial = self._candidate(
            views,
            candidate_id="listed-first-partial",
            support_cores=(CoreId.FORM, CoreId.FLOW),
            observation_pairs=((CoreId.FORM, CoreId.FLOW), (CoreId.FLOW, CoreId.FORM)),
        )
        integrated = self._candidate(
            views,
            candidate_id="listed-second-integrated",
            support_cores=(CoreId.FORM, CoreId.FLOW, CoreId.ACCORD),
            observation_pairs=tuple(permutations(CoreId, 2)),
        )

        _, result = coalesce_at_tiferet(
            "derive rather than choose the first item",
            views,
            observations,
            candidate_directions=(partial, integrated),
        )

        self.assertEqual("listed-second-integrated", result.selected_candidate_id)
        scores = {item.candidate_id: item.graph_score for item in result.candidate_assessments}
        self.assertGreater(scores["listed-second-integrated"], scores["listed-first-partial"])

    def test_tiferet_rejects_unsafe_candidate_even_with_more_support(self) -> None:
        views = default_views()
        observations = build_mutual_observations(views)
        unsafe = self._candidate(
            views,
            candidate_id="unsafe-external",
            support_cores=(CoreId.FORM, CoreId.FLOW, CoreId.ACCORD),
            observation_pairs=tuple(permutations(CoreId, 2)),
            next_step="Publish externally before testing.",
            requires_external_action=True,
        )
        safe = self._candidate(
            views,
            candidate_id="safe-bounded",
            support_cores=(CoreId.FORM, CoreId.FLOW),
            observation_pairs=((CoreId.FORM, CoreId.FLOW), (CoreId.FLOW, CoreId.FORM)),
        )

        _, result = coalesce_at_tiferet(
            "preserve a real boundary",
            views,
            observations,
            candidate_directions=(unsafe, safe),
        )

        self.assertEqual("safe-bounded", result.selected_candidate_id)
        assessed = {item.candidate_id: item for item in result.candidate_assessments}
        self.assertEqual("rejected", assessed["unsafe-external"].status)
        self.assertTrue(
            any("external action" in reason for reason in assessed["unsafe-external"].rejection_reasons)
        )

    def test_candidate_selection_is_independent_of_candidate_order(self) -> None:
        views = default_views()
        observations = build_mutual_observations(views)
        candidates = (
            self._candidate(
                views,
                candidate_id="two-core",
                support_cores=(CoreId.FORM, CoreId.FLOW),
                observation_pairs=((CoreId.FORM, CoreId.FLOW), (CoreId.FLOW, CoreId.FORM)),
            ),
            self._candidate(
                views,
                candidate_id="three-core",
                support_cores=(CoreId.FORM, CoreId.FLOW, CoreId.ACCORD),
                observation_pairs=tuple(permutations(CoreId, 2)),
            ),
        )
        _, forward = coalesce_at_tiferet(
            "candidate order test", views, observations, candidate_directions=candidates
        )
        _, reversed_result = coalesce_at_tiferet(
            "candidate order test",
            views,
            observations,
            candidate_directions=tuple(reversed(candidates)),
        )
        self.assertEqual(forward, reversed_result)

    def test_public_state_is_one_voice_not_a_three_agent_transcript(self) -> None:
        views = default_views()
        observations = build_mutual_observations(views)
        _, result = coalesce_at_tiferet("test intention", views, observations)
        self.assertTrue(result.one_mind_state.strip())
        for heading in ("Form:", "Flow:", "Accord:"):
            self.assertNotIn(heading, result.one_mind_state)


class BoundedRealizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ResponseContract(
            direction="Run one local reversible comparison.",
            rationale_basis="The selected candidate answers all three views and preserves the boundary.",
            next_step="Record the observable result before interpreting it.",
            held_open=("The structural advantage may be null.",),
        )

    def _response(self, **changes) -> EmbodiedResponse:
        fields = {
            "direction": f"In practical terms: {self.contract.direction}",
            "rationale": f"The graph's basis is: {self.contract.rationale_basis}",
            "next_step": f"Begin with this bounded move: {self.contract.next_step}",
            "held_open": (f"Still unresolved: {self.contract.held_open[0]}",),
            "renderer_id": "test-realizer:v1",
            "response_contract_hash": self.contract.contract_hash,
        }
        fields.update(changes)
        return EmbodiedResponse(**fields)

    def test_short_framing_is_allowed_while_commitments_remain_verbatim(self) -> None:
        validate_response_realization(self._response(), self.contract)

    def test_paraphrasing_a_commitment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "removed or paraphrased"):
            validate_response_realization(
                self._response(direction="Try a local reversible comparison."),
                self.contract,
            )

    def test_realizer_cannot_add_a_new_action_commitment(self) -> None:
        with self.assertRaisesRegex(ValueError, "commitment-like language"):
            validate_response_realization(
                self._response(
                    next_step=(
                        f"You should publish this too. {self.contract.next_step}"
                    )
                ),
                self.contract,
            )


if __name__ == "__main__":
    unittest.main()
