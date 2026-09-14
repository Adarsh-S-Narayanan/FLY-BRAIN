"""Adversarial tests (P31): the system must fail safely, never silently corrupt."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import shutil
import unittest
import numpy as np

from src.connectome.types import ConnectomeGraph, GraphMode, SyntheticTestGraph
from src.connectome.loader import get_or_create_circuit
from src.common.provenance import (build_layers, experiment_fingerprint, model_hash,
                                   dataset_hash)
from src.genome.schema import Genome
from src.genome.operators import mutate_genome


def base_graph_kwargs(n=4):
    return dict(neuron_ids=np.arange(n, dtype=np.int64),
                coordinates=np.zeros((n, 3), dtype=np.float32),
                tbars=np.ones(n, dtype=np.int32), sides=["L"] * n,
                row_offsets=np.zeros(n + 1, dtype=np.int32),
                col_indices=np.array([], dtype=np.int32),
                weights=np.array([], dtype=np.float32),
                mode=GraphMode.SYNTHETIC_TEST)


class TestGraphAdversarial(unittest.TestCase):
    def test_wrong_dimensions_rejected(self):
        kw = base_graph_kwargs()
        kw["row_offsets"] = np.array([0, 0], dtype=np.int32)  # too short
        with self.assertRaises(ValueError):
            ConnectomeGraph(**kw)

    def test_negative_and_oob_indices_rejected(self):
        kw = base_graph_kwargs()
        kw["row_offsets"] = np.array([0, 1, 1, 1, 1], dtype=np.int32)
        for bad in (-1, 99):
            kw["col_indices"] = np.array([bad], dtype=np.int32)
            with self.assertRaises(ValueError):
                ConnectomeGraph(**kw)

    def test_duplicate_neuron_ids_rejected(self):
        kw = base_graph_kwargs()
        kw["neuron_ids"] = np.array([5, 5, 6, 7], dtype=np.int64)
        with self.assertRaises(ValueError):
            ConnectomeGraph(**kw)

    def test_nonmonotonic_row_offsets_rejected(self):
        kw = base_graph_kwargs()
        kw["row_offsets"] = np.array([0, 2, 1, 1, 1], dtype=np.int32)
        with self.assertRaises(ValueError):
            ConnectomeGraph(**kw)

    def test_nan_inf_weights_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            kw = base_graph_kwargs()
            kw["row_offsets"] = np.array([0, 1, 1, 1, 1], dtype=np.int32)
            kw["col_indices"] = np.array([1], dtype=np.int32)
            kw["weights"] = np.array([bad], dtype=np.float32)
            with self.assertRaises(ValueError):
                ConnectomeGraph(**kw)


class TestCacheAdversarial(unittest.TestCase):
    def test_corrupt_cache_is_rebuilt_not_loaded(self):
        cache = "diagnostics/connectome_cache/adv_corrupt.npz"
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            f.write(b"\x00" * 512)
        g = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=3,
                                  cache_name="adv_corrupt.npz")
        self.assertEqual(g.num_neurons, 32)
        g.validate_invariants()
        # and the rebuilt cache is valid on reload
        g2 = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=3,
                                   cache_name="adv_corrupt.npz")
        self.assertEqual(g.graph_hash, g2.graph_hash)

    def test_version_mismatch_ignored(self):
        import numpy as np
        cache = "diagnostics/connectome_cache/adv_version.npz"
        g = get_or_create_circuit(24, mode=GraphMode.SYNTHETIC_TEST, seed=4,
                                  cache_name="adv_version.npz")
        data = dict(np.load(cache, allow_pickle=True))
        data["loader_version"] = np.array(1)  # stale semantics
        np.savez_compressed(cache, **data)
        g2 = get_or_create_circuit(24, mode=GraphMode.SYNTHETIC_TEST, seed=4,
                                   cache_name="adv_version.npz")
        self.assertEqual(g.graph_hash, g2.graph_hash)
        self.assertEqual(str(np.load(cache, allow_pickle=True)["loader_version"]), "3")


class TestGenomeAdversarial(unittest.TestCase):
    def test_out_of_bounds_and_nan_rejected(self):
        g = Genome.founder(1)
        for bad in ({"exploration": 5.0}, {"plasticity_rate": float("nan")},
                    {"metabolism_rate": -1.0}):
            with self.assertRaises(ValueError):
                Genome(params=dict(g.params, **bad)).validate()

    def test_mutation_never_escapes_bounds(self):
        for s in range(200):
            c, _ = mutate_genome(Genome.founder(s), s, rate=1.0, scale=1.0)
            c.validate()  # would raise on any bound violation


class TestProvenanceLayers(unittest.TestCase):
    def test_layers_and_fingerprint_change_with_children(self):
        g = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=5,
                                  cache_name="adv_layers.npz")
        l1 = build_layers(graph=g, model_path=None)
        g.weights = g.weights.copy()
        g.weights[0] = float(g.weights[0]) + 0.01
        g.graph_hash = g.compute_graph_hash()
        l2 = build_layers(graph=g, model_path=None)
        self.assertNotEqual(l1["graph"], l2["graph"])
        f1, f2 = experiment_fingerprint(l1), experiment_fingerprint(l2)
        self.assertNotEqual(f1, f2, "parent fingerprint must change when a child changes")

    def test_unavailable_model_hash_is_explicit_and_stable(self):
        # Unavailable models get a deterministic sentinel hash (distinct from a
        # real model hash) so provenance never silently omits the model layer.
        h_none = model_hash(None)
        self.assertEqual(h_none, model_hash("no/such/model.gguf"))
        real = model_hash(os.path.join("llm", "gguf", "MiniCPM5-2B-Q8_0.gguf"))
        self.assertNotEqual(h_none, real)
        self.assertEqual(len(real), 64)


class TestPopulationEdgeCases(unittest.TestCase):
    def test_empty_population_hash_stable(self):
        from src.common.determinism import SeedBundle
        from src.population.population import Population
        seeds = SeedBundle(experiment_seed=61, generation_seed=62, organism_seed=63,
                           development_seed=64, mutation_seed=65, world_seed=66,
                           teacher_seed=67)
        pop = Population(2, seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=61)
        for o in pop.organisms:
            o.die("test")
        self.assertEqual(len(pop.living()), 0)
        self.assertEqual(pop.population_hash(), pop.population_hash())
        # no parents eligible, reproduce is a safe no-op
        self.assertEqual(pop.reproduce(2), [])

    def test_double_death_is_idempotent(self):
        from src.common.determinism import SeedBundle
        from src.population.population import Population
        seeds = SeedBundle(experiment_seed=71, generation_seed=72, organism_seed=73,
                           development_seed=74, mutation_seed=75, world_seed=76,
                           teacher_seed=77)
        pop = Population(2, seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=71)
        o = pop.organisms[0]
        o.die("first")
        h = o.organism_hash()
        o.die("second")
        self.assertEqual(o.cause_of_death, "first")
        self.assertEqual(o.organism_hash(), h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
