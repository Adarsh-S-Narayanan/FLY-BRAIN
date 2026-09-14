"""LivingBrain tests (STAGE B/C): identity, provenance classes, resource-
constrained growth, checkpoint continuation, migration from old checkpoints."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.brain.living import LivingBrain, ProvenanceClass, SCHEMA_VERSION
from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode
from src.development.engine import DevelopmentEngine, DevelopmentState


def make_brain(mode=GraphMode.SYNTHETIC_TEST, size=32, seed=7, cache=None):
    g = get_or_create_circuit(size, mode=mode, seed=seed,
                              cache_name=cache or f"lb_{mode.value.lower()}_{size}_{seed}.npz")
    dev = DevelopmentState.initialize(g.num_neurons)
    eng = DevelopmentEngine({}, seed + 1)
    return LivingBrain(g, dev=dev, engine=eng, experiment_seed=seed)


class TestSeedClassification(unittest.TestCase):
    def test_seed_class_matches_mode(self):
        self.assertEqual(make_brain(GraphMode.REAL, 64, 42,
                                    "lb_real_64_42.npz").seed_class,
                         ProvenanceClass.BIOLOGICAL)
        self.assertEqual(make_brain(GraphMode.SPATIAL_SURROGATE, 64, 42,
                                    "lb_surr_64_42.npz").seed_class,
                         ProvenanceClass.DERIVED)
        self.assertEqual(make_brain(GraphMode.SYNTHETIC_TEST, 32, 7).seed_class,
                         ProvenanceClass.SYNTHETIC)

    def test_registry_covers_graph(self):
        lb = make_brain()
        lb.validate()
        self.assertEqual(len(lb._neurons), lb.graph.num_neurons)
        self.assertEqual(len(lb._edge_index), lb.graph.num_synapses)


class TestLifetimeGrowth(unittest.TestCase):
    def test_growth_creates_emergent_neurons_with_persistent_ids(self):
        lb = make_brain()
        before_ids = set(int(x) for x in lb.graph.neuron_ids)
        before_n = lb.graph.num_neurons
        summary = lb.run_development_cycle(tick=5, growth_budget=3)
        born = summary["ops"][0]["neurons_born"]
        if born == 0:
            self.skipTest("Poisson neurogenesis produced 0 this seed")
        new_ids = set(int(x) for x in lb.graph.neuron_ids) - before_ids
        self.assertEqual(lb.graph.num_neurons, before_n + born)
        for nid in new_ids:
            rec = lb._neurons[nid]
            self.assertEqual(rec.provenance_class, "EMERGENT")
            self.assertEqual(rec.birth_op, "neurogenesis")
        self.assertFalse(before_ids & new_ids, "ids must never be reused")
        lb.validate()

    def test_growth_budget_zero_blocks_neurogenesis(self):
        lb = make_brain()
        n0 = lb.graph.num_neurons
        summary = lb.run_development_cycle(tick=5, growth_budget=0)
        self.assertEqual(summary["ops"][0]["neurons_born"], 0)
        self.assertEqual(lb.graph.num_neurons, n0)

    def test_exceeds_seed_and_provenance_chain(self):
        lb = make_brain()
        seed_size = lb.seed_size
        for t in (5, 10, 15, 20, 25, 30):
            lb.run_development_cycle(tick=t, growth_budget=4)
        if lb.graph.num_neurons <= seed_size:
            self.skipTest("no net growth this seed")
        self.assertTrue(lb.structural_summary()["exceeds_seed"])
        chain = lb.provenance_chain()
        self.assertEqual(chain[0], "SYNTHETIC_SEED")
        self.assertIn("EMERGENT", chain)
        # emergent neurons are NEVER relabeled as the seed class
        emergent = [r for r in lb._neurons.values()
                    if r.provenance_class == "EMERGENT"]
        self.assertTrue(emergent)
        self.assertNotEqual(emergent[0].provenance_class, lb.seed_class.value)

    def test_synapse_lifecycle_records(self):
        lb = make_brain()
        for t in (5, 10):
            lb.run_development_cycle(tick=t, growth_budget=2)
        born = [r for r in lb._synapses.values() if r.birth_op not in ("seed",)]
        for r in born:
            self.assertEqual(r.provenance_class, "EMERGENT")
            self.assertGreaterEqual(r.birth_tick, 5)
        removed = [r for r in lb._synapses.values() if r.death_tick is not None]
        for r in removed:
            self.assertGreaterEqual(r.death_tick, r.birth_tick)
        lb.validate()

    def test_structural_events_logged(self):
        lb = make_brain()
        lb.run_development_cycle(tick=5, growth_budget=2)
        types = {e["type"] for e in lb.structural_events.events}
        self.assertIn("NEURON_BORN", types)  # engine emits its own events
        self.assertTrue(len(lb.structural_events) > 0)


class TestCheckpointContinuation(unittest.TestCase):
    def test_snapshot_restore_registry_equivalence(self):
        lb = make_brain()
        for t in (5, 10, 15):
            lb.run_development_cycle(tick=t, growth_budget=3)
        h_before = lb.brain_hash()
        snap = lb.snapshot()
        # restore into a fresh living brain over the SAME graph state
        g2 = get_or_create_circuit(lb.graph.num_neurons, mode=lb.graph.mode, seed=7,
                                   cache_name=lb.graph.mode.value.lower() + "_lbck.npz")
        # build graph identical to lb.graph arrays
        g2.neuron_ids = lb.graph.neuron_ids
        g2.coordinates = lb.graph.coordinates
        g2.tbars = lb.graph.tbars
        g2.sides = list(lb.graph.sides)
        g2.row_offsets = lb.graph.row_offsets
        g2.col_indices = lb.graph.col_indices
        g2.weights = lb.graph.weights
        g2.graph_hash = g2.compute_graph_hash()
        dev2 = DevelopmentState.initialize(lb.graph.num_neurons)
        dev2.cell_types = list(lb.dev.cell_types)
        dev2.birth_ticks = list(lb.dev.birth_ticks)
        dev2.lineage_ids = list(lb.dev.lineage_ids)
        dev2.developmental_states = list(lb.dev.developmental_states)
        dev2.alive = list(lb.dev.alive)
        eng2 = DevelopmentEngine({}, 8)
        lb2 = LivingBrain.attach(g2, dev2, eng2, 7, payload=snap)
        self.assertEqual(lb2.brain_hash(), h_before,
                         "restored registry must hash identically")

    def test_migration_from_legacy_checkpoint(self):
        lb = make_brain()
        for t in (5, 10):
            lb.run_development_cycle(tick=t, growth_budget=2)
        # legacy payload = None -> registries rebuilt from graph + dev
        g2 = lb.graph
        dev2 = DevelopmentState.initialize(g2.num_neurons)
        eng2 = DevelopmentEngine({}, 9)
        lb2 = LivingBrain.attach(g2, dev2, eng2, 7, payload=None)
        lb2.validate()
        self.assertEqual(lb2.graph.num_neurons, lb.graph.num_neurons)

    def test_invalid_schema_rejected(self):
        lb = make_brain()
        with self.assertRaises(ValueError):
            LivingBrain.attach(lb.graph, lb.dev, lb.engine, 7,
                               payload={"schema_version": "living_brain_v0"})


class TestInvariants(unittest.TestCase):
    def test_never_reuse_after_death(self):
        lb = make_brain()
        max_id_seen = max(int(x) for x in lb.graph.neuron_ids)
        known = set(int(x) for x in lb.graph.neuron_ids)
        for t in range(5, 60, 5):
            lb.run_development_cycle(tick=t, growth_budget=2)
            ids = set(int(x) for x in lb.graph.neuron_ids)
            # ids never disappear (death keeps the record in arrays)
            self.assertTrue(known <= ids, f"neuron ids disappeared at tick {t}")
            new = ids - known
            if new:
                self.assertGreater(min(new), max_id_seen,
                                   f"reused/decreasing id at tick {t}: {sorted(new)}")
                max_id_seen = max(max_id_seen, max(new))
            known = ids
        lb.validate()

    def test_validate_catches_registry_tamper(self):
        lb = make_brain()
        some_id = next(iter(lb._edge_index))
        lb._edge_index.pop(some_id)
        with self.assertRaises(ValueError):
            lb.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
