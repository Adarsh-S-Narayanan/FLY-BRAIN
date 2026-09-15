"""PHASE C tests: immutable biological baseline, synapse identity, layered
brain identity, graph state model, HEREDITY SEPARATION (parent learned weights
never copied into child), evolution scheduler parent-identity integrity."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.provenance.v4 import (BiologicalBaseline, BiologicalSynapse, BrainIdentity,
                               ProvenanceClassV4, build_biological_baseline, GRAPH_STATES)
from src.brain.living import LivingBrain, DevelopmentState, DevelopmentEngine
from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode
from src.common.determinism import SeedBundle
from src.population.population import Population


def make_lb(mode=GraphMode.SYNTHETIC_TEST, size=32, seed=7, cache=None):
    g = get_or_create_circuit(size, mode=mode, seed=seed,
                              cache_name=cache or f"v4_{mode.value.lower()}_{size}_{seed}.npz")
    return LivingBrain(g, dev=DevelopmentState.initialize(g.num_neurons),
                       engine=DevelopmentEngine({}, seed + 1), experiment_seed=seed)


class TestImmutableBiologicalBaseline(unittest.TestCase):
    def test_baseline_frozen_and_verifiable(self):
        lb = make_lb()
        self.assertTrue(lb.biological_baseline_intact())
        fp0 = lb.bio_baseline.seed_fingerprint
        # lifetime development must NEVER rewrite the baseline
        for t in (5, 10, 15):
            lb.run_development_cycle(tick=t, growth_budget=2)
        self.assertTrue(lb.biological_baseline_intact())
        self.assertEqual(lb.bio_baseline.seed_fingerprint, fp0)
        self.assertGreater(len(lb.bio_baseline.synapses), 0)

    def test_plasticity_cannot_touch_baseline(self):
        """Plasticity changes effective weights — the biological measurement
        (original synapse count per record) is untouched."""
        lb = make_lb()
        before = {s.dataset_record_id: s.original_synapse_count
                  for s in lb.bio_baseline.synapses}
        w_before = lb.graph.weights.copy()
        lb.graph.weights = np.clip(w_before + 0.05, 0.01, 1.0).astype(np.float32)
        after = {s.dataset_record_id: s.original_synapse_count
                 for s in lb.bio_baseline.synapses}
        self.assertEqual(before, after)
        self.assertTrue(lb.biological_baseline_intact())

    def test_baseline_records_dataset_provenance(self):
        lb_real = make_lb(GraphMode.REAL, 64, 42, "v4_real_64_42.npz")
        self.assertEqual(lb_real.bio_baseline.dataset_name, "malecns")
        for s in lb_real.bio_baseline.synapses:
            self.assertTrue(s.dataset_record_id.startswith("malecns:"))
        lb_syn = make_lb(cache=None)  # synthetic
        self.assertTrue(lb_syn.bio_baseline.dataset_name.startswith("seed:"))

    def test_snapshot_restore_preserves_baseline(self):
        lb = make_lb()
        for t in (5, 10):
            lb.run_development_cycle(tick=t, growth_budget=2)
        snap = lb.snapshot()
        lb2 = LivingBrain.attach(lb.graph, lb.dev, lb.engine, 7, payload=snap)
        self.assertTrue(lb2.biological_baseline_intact())
        self.assertEqual(lb2.bio_baseline.seed_fingerprint,
                         lb.bio_baseline.seed_fingerprint)


class TestSynapseIdentity(unittest.TestCase):
    def test_records_have_stable_identity_and_source(self):
        lb = make_lb()
        seed_syns = [r for r in lb._synapses.values() if r.birth_op == "seed"]
        for r in seed_syns:
            self.assertTrue(r.synapse_id and len(r.synapse_id) == 16)
            self.assertEqual(r.creation_generation, 0)
            self.assertEqual(r.parent_synapse_ids, [])
            self.assertTrue(r.source_dataset)
            self.assertTrue(r.source_record_id)
        # emergent synapses are EMERGENT with empty source_record (not biological)
        lb.run_development_cycle(tick=5, growth_budget=2)
        emergent = [r for r in lb._synapses.values() if r.birth_op != "seed"]
        for r in emergent:
            self.assertEqual(r.provenance_class, "EMERGENT")
            self.assertEqual(r.source_record_id, "")

    def test_new_synapses_never_claim_biological(self):
        lb = make_lb()
        for t in (5, 10, 15):
            lb.run_development_cycle(tick=t, growth_budget=2)
        for r in lb._synapses.values():
            if r.birth_op != "seed":
                self.assertNotEqual(r.provenance_class, "BIOLOGICAL")


class TestBrainIdentityLayers(unittest.TestCase):
    def _identity(self, g, **kw):
        return BrainIdentity.from_components(g, **kw)

    def test_all_layers_present_and_identity_changes_per_component(self):
        g = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=11,
                                  cache_name="v4_ident.npz")
        id0 = self._identity(g).full_identity()
        # weights change -> identity changes
        g.weights = np.clip(g.weights + 0.05, 0.01, 1.0)
        id1 = self._identity(g).full_identity()
        self.assertNotEqual(id0, id1)
        # topology change -> identity changes
        g2 = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=11,
                                   cache_name="v4_ident.npz")
        g2.col_indices = g2.col_indices[:-1].copy()
        g2.row_offsets = g2.row_offsets.copy()
        g2.row_offsets[-1] = len(g2.col_indices)
        id2 = self._identity(g2).full_identity()
        self.assertNotEqual(id2, id1)
        # geometry change -> identity changes
        g3 = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=11,
                                   cache_name="v4_ident.npz")
        g3.coordinates = g3.coordinates + 10.0
        id3 = self._identity(g3).full_identity()
        self.assertNotEqual(id3, id2)
        # identical state -> identical identity
        g4 = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=11,
                                   cache_name="v4_ident.npz")
        self.assertEqual(self._identity(g4).full_identity(), id0)

    def test_layered_dict_contains_expected_layers(self):
        g = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=11,
                                  cache_name="v4_ident.npz")
        ident = self._identity(g, population_hash="p", event_stream_hash="e",
                               synapse_provenance_hash="s", development_hash="d")
        d = ident.to_dict()
        for layer in BrainIdentity.LAYERS:
            self.assertIn(layer, d["layers"])
        self.assertEqual(d["layers"]["population"], "p")
        self.assertEqual(d["layers"]["event_stream"], "e")


class TestGraphStateModel(unittest.TestCase):
    def test_state_progression(self):
        lb = make_lb()
        self.assertEqual(lb.state_class, "SYNTHETIC")
        for t in (5, 10):
            lb.run_development_cycle(tick=t, growth_budget=2)
        if any(r.birth_op != "seed" for r in lb._synapses.values()) or \
                any(r.provenance_class == "EMERGENT" for r in lb._neurons.values()):
            self.assertEqual(lb.state_class, "LIVING_EMERGENT")
        self.assertIn(lb.state_class, GRAPH_STATES)

    def test_provenance_class_completeness(self):
        names = {c.value for c in ProvenanceClassV4}
        self.assertEqual(names, {"BIOLOGICAL", "DERIVED", "HEURISTIC", "EMERGENT",
                                 "EVOLVED", "SYNTHETIC", "UNKNOWN"})


class TestHereditySeparation(unittest.TestCase):
    def test_parent_learned_weights_not_inherited(self):
        """GENOME is inherited; LEARNED NEURAL STATE is not (mission §20/§21)."""
        seed = 221
        seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                           organism_seed=seed + 2, development_seed=seed + 3,
                           mutation_seed=seed + 4, world_seed=seed + 5,
                           teacher_seed=seed + 6)
        pop = Population(2, seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=seed)
        # parent learns via stimulated brain steps with reward -> weights change
        parent = pop.living()[0]
        parent.energy = 1.0
        world = pop.world
        world.place(parent.id, (1, 1))
        w0 = parent.graph.weights.copy()
        rng = np.random.RandomState(7)
        for _ in range(6):
            stim = {"visual": rng.uniform(0, 1.2, 16).astype(np.float32)}
            parent.brain.step(sensory_inputs=stim, reward=1.0)
        self.assertTrue(np.any(np.abs(parent.graph.weights - w0) > 1e-6),
                        "parent must actually learn in this test")
        # birth child from same (learned) parent
        pop.tick = 100
        parent.age = 70
        parent.stage = __import__("src.organism.organism", fromlist=["LifeStage"]) \
            .LifeStage.ADULT
        parent2 = pop.living()[1]
        parent2.age = 70
        parent2.stage = __import__("src.organism.organism", fromlist=["LifeStage"]) \
            .LifeStage.ADULT
        children = pop.reproduce(1, mode="sexual")
        self.assertEqual(len(children), 1)
        child = children[0]
        # child weights == fresh-development weights for the child's seeds,
        # NOT the parent's learned weights
        child_seeds = child.seeds
        fresh = get_or_create_circuit(
            pop.circuit_size, mode=pop.graph_mode, seed=child_seeds["organism_seed"],
            cache_name="v4_heredity_fresh_221.npz")
        fresh_w = fresh.weights.copy()
        self.assertTrue(np.allclose(child.graph.weights, fresh_w, atol=1e-7),
                        "child brain must start from deterministic development, "
                        "not the parent's learned weights")
        # and the child's genome IS derived from parents (inherited)
        self.assertIn(child.id, pop.genetic_lineage)
        self.assertEqual(len(pop.genetic_lineage[child.id]), 2)

    def test_emergent_structures_not_automatic_inheritance(self):
        """A parent with grown brain does not birth a child with the same grown
        structure; the child develops from its own genome program."""
        seed = 231
        seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                           organism_seed=seed + 2, development_seed=seed + 3,
                           mutation_seed=seed + 4, world_seed=seed + 5,
                           teacher_seed=seed + 6)
        pop = Population(2, seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=seed,
                         autonomy_mode=True, genome_version="2.0")
        for o in pop.living():
            o.age = 70
        pop.step(30)
        pop.reproduce(1, mode="sexual")
        children = [o for o in pop.organisms if o.generation > 0]
        if not children:
            self.skipTest("no reproduction occurred")
        child = children[0]
        self.assertEqual(child.living.seed_size, child.graph.num_neurons,
                         "child starts at its own developmental seed size")
        self.assertTrue(child.living.biological_baseline_intact())


class TestSchedulerParentIdentity(unittest.TestCase):
    def test_accepted_child_never_its_own_parent(self):
        from src.evolution.scheduler import EvolutionScheduler
        g = get_or_create_circuit(64, mode=GraphMode.SYNTHETIC_TEST, seed=51,
                                  cache_name="v4_sched.npz")
        sched = EvolutionScheduler(g, history_file="diagnostics/v4_sched_hist.json",
                                   seed=100)
        prev_child = sched.current_brain_id
        for _ in range(3):
            summary = sched.run_generation(num_candidates=4, seed=100)
            # the generation's parent is ALWAYS the pre-selection brain
            self.assertEqual(summary["parent_id"], prev_child)
            # the child (selected or rollback) is the current brain after run
            self.assertEqual(summary["child_id"], sched.current_brain_id)
            if summary["improved"]:
                self.assertNotEqual(summary["child_id"], summary["parent_id"],
                                    "accepted child must not become its own parent")
            for c in summary["candidates"]:
                self.assertEqual(c["parent_id"], summary["parent_id"])
                self.assertNotEqual(c["candidate_id"], c["parent_id"])
            prev_child = summary["child_id"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
