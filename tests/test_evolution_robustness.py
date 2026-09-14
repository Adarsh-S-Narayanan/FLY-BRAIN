"""Evolution robustness: determinism, isolation, checkpoint/resume, multi-generation (P9)."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode
from src.evolution.scheduler import EvolutionScheduler


def fresh_sched(history="diagnostics/test_evo_robust.json", seed=500):
    g = get_or_create_circuit(64, mode=GraphMode.SYNTHETIC_TEST, seed=seed,
                              cache_name="evo_robust_synth_64.npz")
    return EvolutionScheduler(g, history_file=history, seed=seed)


class TestEvolutionRobustness(unittest.TestCase):
    def test_deterministic_ids_and_repeatability(self):
        s1 = fresh_sched("diagnostics/test_evo_r1.json")
        r1 = s1.run_generation(num_candidates=3, seed=501)
        s2 = fresh_sched("diagnostics/test_evo_r2.json")
        r2 = s2.run_generation(num_candidates=3, seed=501)
        self.assertEqual(r1["best_score"], r2["best_score"])
        self.assertEqual([c["candidate_id"] for c in r1["candidates"]],
                         [c["candidate_id"] for c in r2["candidates"]])
        self.assertEqual(s1.current_graph.graph_hash, s2.current_graph.graph_hash)

    def test_failed_candidates_cannot_corrupt_parent(self):
        s = fresh_sched("diagnostics/test_evo_iso.json")
        before = s.current_graph.weights.copy()
        before_hash = s.current_graph.graph_hash
        s.run_generation(num_candidates=4, seed=502)
        # parent weights only change via accepted candidate replacement
        if not s.history[-1]["improved"]:
            np.testing.assert_array_equal(s.current_graph.weights, before)
            self.assertEqual(s.current_graph.graph_hash, before_hash)
        # mode/provenance preserved through cloning
        self.assertEqual(s.current_graph.mode, GraphMode.SYNTHETIC_TEST)

    def test_multi_generation_and_checkpoint_resume(self):
        s = fresh_sched("diagnostics/test_evo_ckpt.json")
        for i in range(2):
            s.run_generation(num_candidates=2, seed=600 + i)
        ckpt = s.save_checkpoint()
        self.assertTrue(os.path.exists(ckpt))
        hash_at_ckpt = s.current_graph.graph_hash

        resumed = EvolutionScheduler.load_checkpoint(ckpt, history_file="diagnostics/test_evo_ckpt2.json")
        self.assertEqual(resumed.generation, 2)
        self.assertEqual(resumed.current_graph.graph_hash, hash_at_ckpt)
        self.assertEqual(resumed.current_brain_id, s.current_brain_id)

        for i in range(2, 4):
            s.run_generation(num_candidates=2, seed=600 + i)
            resumed.run_generation(num_candidates=2, seed=600 + i)
        self.assertEqual(s.current_graph.graph_hash, resumed.current_graph.graph_hash)
        self.assertEqual(s.history[-1]["best_score"], resumed.history[-1]["best_score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
