"""STAGE L/M tests: deep-time acceleration (documented approximation, milestone
escalation, high-res replay) and milestone detection with evidence."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from src.common.determinism import SeedBundle
from src.connectome.types import GraphMode
from src.population.population import Population
from src.timeline.deeptime import DeepTimeConfig, DeepTimeRunner, APPROXIMATION_MODEL
from src.science.milestones import detect_milestones, milestone_certificate


def make_pop(seed, size=4, autonomy=True):
    seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                       organism_seed=seed + 2, development_seed=seed + 3,
                       mutation_seed=seed + 4, world_seed=seed + 5,
                       teacher_seed=seed + 6)
    return Population(size, seeds, GraphMode.SYNTHETIC_TEST, 32,
                      experiment_seed=seed, autonomy_mode=autonomy,
                      genome_version="2.0" if autonomy else "1.0")


class TestDeepTime(unittest.TestCase):
    def test_ledger_carries_approximation_provenance(self):
        pop = make_pop(81)
        runner = DeepTimeRunner(pop)
        s = runner.fast_forward(2)
        self.assertEqual(s["resolution"], "coarse")
        self.assertEqual(s["approximation_model"], APPROXIMATION_MODEL)

    def test_generations_and_births_advance(self):
        pop = make_pop(82)
        runner = DeepTimeRunner(pop)
        s = runner.fast_forward(5)
        self.assertGreater(s["coarse_steps"], 0)
        self.assertGreater(s["sim_ticks"], 0)
        self.assertGreater(s["births"], 0, "reproduction must occur in deep time")
        self.assertTrue(s["generations_seen"], "generations must be recorded")

    def test_events_recorded(self):
        pop = make_pop(83)
        runner = DeepTimeRunner(pop)
        runner.fast_forward(4)
        kinds = {e.kind for e in runner.events}
        self.assertIn("BIRTH", kinds)
        self.assertTrue(runner.events)

    def test_escalation_and_high_res_replay(self):
        pop = make_pop(84)
        runner = DeepTimeRunner(pop)
        runner.fast_forward(3)
        esc = runner.escalate("milestone-1")
        self.assertEqual(esc["population_hash"], pop.population_hash())
        rep = runner.replay_high_resolution("milestone-1", ticks=5)
        self.assertEqual(rep["status"], "EXECUTED")
        self.assertTrue(rep["checkpoint_hash_verified"],
                        "restored state must match checkpoint hash exactly")
        self.assertEqual(rep["resolution"], "full")
        # unknown checkpoint -> honest failure
        rep2 = runner.replay_high_resolution("nope", ticks=5)
        self.assertEqual(rep2["status"], "FAILED")

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            DeepTimeConfig(coarse_ticks_per_step=0).validate()
        with self.assertRaises(ValueError):
            DeepTimeConfig(divergence_threshold=1.5).validate()

    def test_milestone_hook_invoked(self):
        pop = make_pop(85)
        runner = DeepTimeRunner(pop)
        calls = []
        runner.config.milestone_hooks.append(lambda p: (calls.append(p.tick) or None))
        runner.fast_forward(2)
        self.assertGreater(len(calls), 0)


class TestMilestones(unittest.TestCase):
    def test_overlapping_generations_detected_with_evidence(self):
        pop = make_pop(91)
        pop.organisms[0].age = 70  # eligible parent territory (adult/elder)
        pop.organisms[1].age = 70  # sexual reproduction needs two parents
        pop.step(10)
        pop.reproduce(2)
        ms = detect_milestones(pop)
        kinds = [m["milestone"] for m in ms]
        self.assertIn("OVERLAPPING_GENERATIONS", kinds)
        rec = next(m for m in ms if m["milestone"] == "OVERLAPPING_GENERATIONS")
        self.assertGreaterEqual(len(rec["evidence"]["generations"]), 2)
        self.assertTrue(rec["population_hash"])

    def test_structural_expansion_requires_real_growth(self):
        pop = make_pop(92)
        ms = detect_milestones(pop)
        kinds = [m["milestone"] for m in ms]
        # either real expansion (with provenance chain) or honest absence
        if "STRUCTURAL_EXPANSION" in kinds:
            rec = next(m for m in ms if m["milestone"] == "STRUCTURAL_EXPANSION")
            self.assertIn("EMERGENT", rec["evidence"]["provenance_chain"])
            self.assertGreater(rec["evidence"]["current_size"], rec["evidence"]["seed_size"])

    def test_cultural_transmission_requires_measured_gain(self):
        pop = make_pop(93)
        teacher = pop.organisms[0]
        student = pop.organisms[1]
        teacher.age = 100
        teacher.skills["forage"] = 0.9
        student.skills["forage"] = 0.1
        pop.world.place(teacher.id, (2, 2))
        pop.world.place(student.id, (3, 2))
        pop.step(5)
        self.assertTrue(pop.teaching_sessions, "test setup must produce a session")
        ms = detect_milestones(pop)
        rec = next(m for m in ms if m["milestone"] == "CULTURAL_TRANSMISSION")
        self.assertIn("mean_gain", rec["evidence"])
        self.assertGreater(rec["evidence"]["mean_gain"], 0.0)

    def test_no_fake_milestones_on_fresh_population(self):
        pop = make_pop(94)
        ms = detect_milestones(pop)
        for m in ms:
            self.assertTrue(m.get("evidence"), f"milestone without evidence: {m}")

    def test_certificate_binds_evidence_and_hashes(self):
        pop = make_pop(95)
        pop.step(45)
        pop.reproduce(2)
        ms = detect_milestones(pop)
        self.assertTrue(ms)
        cert = milestone_certificate(ms[0], experiment_seed=95)
        self.assertEqual(cert["milestone"], ms[0]["milestone"])
        self.assertTrue(cert["valid"])
        self.assertEqual(cert["population_hash"], ms[0]["population_hash"])
        self.assertEqual(cert["experiment_seed"], 95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
