"""STAGE E/F tests: autonomy engine (self-generated goals, compositional
actions, brain-driven modulation), embodiment body, organism integration,
checkpoint continuation of autonomy state."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import math
import numpy as np

from src.autonomy.engine import AutonomyEngine, ActionCandidate, Goal, GoalStatus
from src.embodiment.body import BodyState
from src.genome.schema import Genome
from src.common.determinism import SeedBundle
from src.organism.organism import Organism
from src.world.environment import GridWorld, WorldConfig
from src.connectome.types import GraphMode


class TestGoalGeneration(unittest.TestCase):
    def test_low_energy_generates_forage_goal(self):
        eng = AutonomyEngine(seed=1)
        goals = eng.generate_goals(10, energy=0.2, health=1.0, drives={},
                                   prediction_error=0.0,
                                   world_sense={"food_gradient": 0.0,
                                                "hazard_gradient": 0.0,
                                                "nearby_organisms": 0},
                                   position=(0, 0), skills={})
        self.assertTrue(any(g.kind == "forage" and g.source == "need" for g in goals))

    def test_curiosity_generates_explore_with_target(self):
        eng = AutonomyEngine(seed=2)
        goals = eng.generate_goals(10, energy=1.0, health=1.0,
                                   drives={"curiosity": 0.95},
                                   prediction_error=0.0,
                                   world_sense={"food_gradient": 0.0,
                                                "hazard_gradient": 0.0,
                                                "nearby_organisms": 0},
                                   position=(5, 5), skills={})
        explore = [g for g in goals if g.kind == "explore"]
        self.assertTrue(explore and explore[0].target is not None)
        self.assertEqual(explore[0].source, "curiosity")

    def test_prediction_error_generates_investigation(self):
        eng = AutonomyEngine(seed=3)
        goals = eng.generate_goals(10, energy=1.0, health=1.0, drives={},
                                   prediction_error=0.8,
                                   world_sense={"food_gradient": 0.0,
                                                "hazard_gradient": 0.0,
                                                "nearby_organisms": 0},
                                   position=(0, 0), skills={})
        self.assertTrue(any(g.kind == "investigate" and g.source == "prediction_error"
                            for g in goals))

    def test_no_human_tasks_required(self):
        """Goals emerge purely from internal state + world observation."""
        eng = AutonomyEngine(seed=4)
        kinds = set()
        pos = (0, 0)
        for t in range(20):
            eng.generate_goals(t, energy=0.3 + 0.02 * t, health=1.0,
                               drives={"curiosity": 0.9, "social": 0.95},
                               prediction_error=0.5 if t % 3 == 0 else 0.0,
                               world_sense={"food_gradient": 1.5, "hazard_gradient": 0.7,
                                            "nearby_organisms": 2},
                               position=pos, skills={})
            kinds |= {g.kind for g in eng.goals}
            pos = (pos[0] + 1, pos[1])
        self.assertTrue({"forage", "explore", "investigate", "avoid", "socialize"} <= kinds)


class TestSelfEvaluation(unittest.TestCase):
    def test_progress_achievement_and_abandonment(self):
        eng = AutonomyEngine(seed=5, goal_patience=3)
        eng.generate_goals(0, energy=0.2, health=1.0, drives={}, prediction_error=0.0,
                           world_sense={"food_gradient": 0.0, "hazard_gradient": 0.0,
                                        "nearby_organisms": 0}, position=(0, 0), skills={})
        goal = eng.active_goal()
        self.assertIsNotNone(goal)
        for t in range(1, 5):
            eng.update_goals(t, energy_delta=0.3, novelty_gained=False, prediction_error=0.0)
        finished = [g for g in eng.completed if g.goal_id == goal.goal_id]
        self.assertTrue(finished and finished[0].status in
                        (GoalStatus.ACHIEVED.value, GoalStatus.ABANDONED.value))

    def test_stale_goal_abandoned_after_patience(self):
        eng = AutonomyEngine(seed=6, goal_patience=2)
        eng.generate_goals(0, energy=0.3, health=1.0, drives={}, prediction_error=0.0,
                           world_sense={"food_gradient": 0.0, "hazard_gradient": 0.0,
                                        "nearby_organisms": 0}, position=(0, 0), skills={})
        gid = eng.active_goal().goal_id
        for t in range(1, 4):
            eng.update_goals(t, energy_delta=0.0, novelty_gained=False, prediction_error=0.9)
        statuses = {g.goal_id: g.status for g in eng.completed}
        self.assertEqual(statuses.get(gid), GoalStatus.ABANDONED.value)


class TestCompositionalActions(unittest.TestCase):
    def _brain_out(self, act=0.5, remember=0.1, curiosity=0.6, pred=0.2):
        return {"drives": {"curiosity": curiosity, "energy": 1.0, "social": 0.5},
                "tool_associations": {"act_in_environment": act, "remember": remember,
                                      "speak": 0.2, "generate_image": 0.1},
                "prediction_error": pred}

    def test_action_has_continuous_components(self):
        eng = AutonomyEngine(seed=7)
        cand = eng.synthesize_action(0, None, self._brain_out(),
                                     {"position": (0, 0)}, 0.8, 1.0, exploration=0.5)
        self.assertIsInstance(cand, ActionCandidate)
        self.assertTrue(0.0 <= cand.speed <= 1.0)
        self.assertTrue(0.0 <= cand.heading < 2 * math.pi)
        self.assertGreaterEqual(cand.duration, 1)
        self.assertTrue(0.0 <= cand.intensity <= 1.0)

    def test_rest_action_when_low_health(self):
        eng = AutonomyEngine(seed=8)
        cand = eng.synthesize_action(0, None, self._brain_out(), {"position": (0, 0)},
                                     0.9, 0.2, exploration=0.5)
        self.assertEqual(cand.kind, "rest")
        self.assertEqual(cand.speed, 0.0)

    def test_brain_state_changes_candidate(self):
        """Neural evidence must modulate output (rule 13: brain-driven)."""
        eng = AutonomyEngine(seed=9)
        a1 = eng.synthesize_action(0, None, self._brain_out(act=0.9, remember=0.0),
                                   {"position": (0, 0)}, 0.8, 1.0)
        a2 = eng.synthesize_action(0, None, self._brain_out(act=0.0, remember=0.9),
                                   {"position": (0, 0)}, 0.8, 1.0)
        self.assertNotEqual(round(a1.heading, 3), round(a2.heading, 3),
                            "motor asymmetry must bias heading")
        self.assertNotEqual(a1.neural_evidence, a2.neural_evidence)

    def test_deterministic_under_same_seed(self):
        e1, e2 = AutonomyEngine(seed=10), AutonomyEngine(seed=10)
        c1 = e1.synthesize_action(3, None, self._brain_out(), {"position": (1, 1)}, 0.5, 1.0)
        c2 = e2.synthesize_action(3, None, self._brain_out(), {"position": (1, 1)}, 0.5, 1.0)
        self.assertEqual(c1.to_dict(), c2.to_dict())


class TestBodyState(unittest.TestCase):
    def test_damage_limits_speed_and_recovery_restores(self):
        b = BodyState(health=1.0)
        self.assertEqual(b.speed_capacity(), 1.0)
        b.apply_damage(0.5, "test")
        self.assertLess(b.speed_capacity(), 1.0)
        self.assertEqual(b.damage_events, 1)
        while b.damage > 0:
            b.recover(0.1)
        self.assertEqual(b.speed_capacity(), 1.0)

    def test_energy_bounds_and_mobility(self):
        b = BodyState(energy=1.4)
        b.gain_energy(10.0)
        self.assertEqual(b.energy, b.MAX_ENERGY)
        b.consume_energy(99.0)
        self.assertEqual(b.energy, 0.0)
        self.assertFalse(b.is_mobile())

    def test_nonpositive_damage_is_noop(self):
        b = BodyState()
        self.assertFalse(b.apply_damage(0.0))
        self.assertEqual(b.damage_events, 0)


def make_org(seed=21, autonomy=True, size=32):
    seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                       organism_seed=seed + 2, development_seed=seed + 3,
                       mutation_seed=seed + 4, world_seed=seed + 5,
                       teacher_seed=seed + 6)
    genome = Genome.founder(seed + 7)
    org = Organism(genome, f"org-{seed}", seeds={"organism_seed": seeds.organism_seed,
                                                 "development_seed": seeds.development_seed},
                   graph_mode=GraphMode.SYNTHETIC_TEST, circuit_size=size,
                   autonomy_mode=autonomy)
    world = GridWorld(WorldConfig(width=16, height=16, n_resources=24, n_hazards=2,
                                  world_seed=seeds.world_seed))
    world.place(org.id, (2, 2))
    return org, world


class TestOrganismIntegration(unittest.TestCase):
    def test_autonomy_organism_runs_and_generates_goals(self):
        org, world = make_org()
        kinds = set()
        for t in range(30):
            out = org.step(world)
            kinds.add(out.get("goal") or out.get("action_kind"))
            self.assertTrue(org.alive or t > 20)
        summary = org.autonomy.autonomy_summary()
        self.assertGreater(summary["goals_generated"], 0)
        self.assertGreater(len(org.autonomy._visited), 1, "organism must explore")
        if org.living is not None:
            org.living.validate()

    def test_organisms_differ_by_mode(self):
        """Autonomy behavior is real: it must differ from legacy chemotaxis."""
        org_a, world_a = make_org(seed=31, autonomy=True)
        org_b, world_b = make_org(seed=31, autonomy=False)
        positions_a, positions_b = [], []
        for _ in range(25):
            org_a.step(world_a); org_b.step(world_b)
            positions_a.append(org_a.position); positions_b.append(org_b.position)
        self.assertNotEqual(positions_a, positions_b)

    def test_autonomy_snapshot_restore_continuation(self):
        org, world = make_org(seed=41)
        for _ in range(12):
            org.step(world)
        snap = org.snapshot()
        org2 = Organism.restore(snap)
        self.assertTrue(org2.autonomy_mode)
        self.assertIsNotNone(org2.autonomy)
        self.assertIsNotNone(org2.body)
        self.assertIsNotNone(org2.living)
        org2.living.validate()
        # engine state carried over (compare BEFORE the extra step)
        self.assertEqual(org2.autonomy._goal_seq, org.autonomy._goal_seq)
        self.assertEqual(org2.autonomy._visited, org.autonomy._visited)
        out = org2.step(world)
        self.assertIn("goal", out)

    def test_legacy_snapshot_still_loads(self):
        org, world = make_org(seed=51, autonomy=False)
        for _ in range(4):
            org.step(world)
        snap = org.snapshot()
        self.assertFalse(snap.get("autonomy_mode"))
        org2 = Organism.restore(snap)
        self.assertFalse(org2.autonomy_mode)
        self.assertIsNone(org2.autonomy)
        org2.step(world)  # legacy path intact

    def test_resource_constrained_growth(self):
        org, world = make_org(seed=61)
        org.energy = 0.2  # poor: no growth budget
        n0 = org.graph.num_neurons
        for _ in range(6):
            org.step(world)
        poor_growth = org.graph.num_neurons - n0
        org2, world2 = make_org(seed=62)
        org2.energy = 1.4  # rich: budget available
        n0b = org2.graph.num_neurons
        for _ in range(6):
            org2.step(world2)
        rich_growth = org2.graph.num_neurons - n0b
        self.assertLessEqual(poor_growth, rich_growth)


if __name__ == "__main__":
    unittest.main(verbosity=2)
