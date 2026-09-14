"""Long-run campaign + checkpoint/resume equivalence (REAL, INDEPENDENT)."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from src.common.determinism import SeedBundle
from src.connectome.types import GraphMode
from src.population.population import Population
from scripts.run_long_campaign import TICKS_PER_GEN, seeds_for


def advance(pop, generations):
    for _ in range(generations):
        pop.step(TICKS_PER_GEN)
        pop.reproduce(2, mode="sexual")
    return pop


class TestCampaignCheckpoint(unittest.TestCase):
    def test_resume_equals_uninterrupted(self):
        seed, pop0 = 31, 5
        fresh = Population(pop0, seeds_for(seed), GraphMode.SYNTHETIC_TEST, 32,
                           experiment_seed=seed)
        advance(fresh, 6)
        h_fresh = fresh.population_hash()

        # interrupted branch: 3 generations, checkpoint, restore, 3 more
        interrupted = Population(pop0, seeds_for(seed), GraphMode.SYNTHETIC_TEST, 32,
                                 experiment_seed=seed)
        advance(interrupted, 3)
        snap = interrupted.snapshot()
        resumed = Population.restore(snap, seeds_for(seed))
        advance(resumed, 3)
        self.assertEqual(resumed.population_hash(), h_fresh,
                         "checkpoint/resume must equal uninterrupted execution")

    def test_campaign_survives_extended_horizon(self):
        # Modest long-run: 12 generations must not corrupt state or crash.
        pop = Population(5, seeds_for(41), GraphMode.SYNTHETIC_TEST, 32, experiment_seed=41)
        for _ in range(12):
            pop.step(TICKS_PER_GEN)
            pop.reproduce(2, mode="sexual")
        for o in pop.organisms:
            o.graph.validate_invariants()
        self.assertGreater(len(pop.events.events), 0)
        self.assertGreaterEqual(len(pop.organisms), 5)
        # reproducibility across independent construction
        pop2 = Population(5, seeds_for(41), GraphMode.SYNTHETIC_TEST, 32, experiment_seed=41)
        for _ in range(12):
            pop2.step(TICKS_PER_GEN)
            pop2.reproduce(2, mode="sexual")
        self.assertEqual(pop2.population_hash(), pop.population_hash())


if __name__ == "__main__":
    unittest.main(verbosity=2)