"""Provenance regression tests: REAL contains ONLY empirical edges (C4, R5, R6)."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import csv
import unittest
import numpy as np

from src.connectome.loader import (
    get_or_create_circuit, build_real_connectome, DEFAULT_CONNECTIONS_PATH,
)
from src.connectome.types import GraphMode, ProvenanceStatus


def load_bio_pairs():
    pairs = set()
    with open(DEFAULT_CONNECTIONS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pairs.add((int(row["pre_body_id"]), int(row["post_body_id"])))
            except (ValueError, KeyError):
                continue
    return pairs


def graph_bio_edges(g):
    """CSR edges as (pre_body_id, post_body_id) with proper index mapping."""
    body = [int(x) for x in g.neuron_ids]
    out = set()
    for i in range(g.num_neurons):
        s, e = int(g.row_offsets[i]), int(g.row_offsets[i + 1])
        for k in range(s, e):
            out.add((body[i], body[int(g.col_indices[k])]))
    return out


class TestProvenanceIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bio_pairs = load_bio_pairs()

    def test_real_edges_subset_of_biological_table(self):
        g = get_or_create_circuit(128, mode=GraphMode.REAL, seed=42,
                                  cache_name="prov_test_real_128.npz")
        self.assertEqual(g.mode, GraphMode.REAL)
        self.assertEqual(g.provenance_status, ProvenanceStatus.VERIFIED)
        edges = graph_bio_edges(g)
        self.assertGreater(len(edges), 0, "REAL graph must contain empirical edges")
        bad = sorted(edges - self.bio_pairs)[:5]
        self.assertEqual(bad, [], f"{len(edges - self.bio_pairs)} REAL edges missing from biological table")
        # no invented edges: every edge accounted for above; metadata agrees
        pm = g.provenance_metadata
        self.assertEqual(pm.get("surrogate_edge_count", 0), 0)
        self.assertEqual(pm.get("fallback_edges_added", 0), 0)

    def test_disconnected_neurons_stay_disconnected(self):
        g = build_real_connectome(max_neurons=128, seed=42)
        degs = np.diff(g.row_offsets)
        n_isolated = int(np.sum(degs == 0))
        # Hub-sampled real data leaves some neurons without in-sample edges;
        # they must remain edgeless, not wired to invented neighbors.
        print(f"[prov] isolated-but-real neurons: {n_isolated}/{g.num_neurons}")
        edges = graph_bio_edges(g)
        bad = sorted(edges - self.bio_pairs)[:5]
        self.assertEqual(bad, [], f"{len(edges - self.bio_pairs)} non-empirical edges found")

    def test_modes_structurally_distinct(self):
        real = get_or_create_circuit(64, mode=GraphMode.REAL, seed=7,
                                     cache_name="prov_test_real_64.npz")
        surr = get_or_create_circuit(64, mode=GraphMode.SPATIAL_SURROGATE, seed=7,
                                     cache_name="prov_test_surr_64.npz")

        re_, se_ = graph_bio_edges(real), graph_bio_edges(surr)
        self.assertNotEqual(re_, se_)
        self.assertEqual(surr.provenance_status, ProvenanceStatus.SURROGATE)
        self.assertGreater(surr.provenance_metadata.get("surrogate_edge_count", 0), 0)

    def test_synthetic_never_verified(self):
        from src.connectome.types import SyntheticTestGraph
        g = get_or_create_circuit(64, mode=GraphMode.SYNTHETIC_TEST, seed=7,
                                  cache_name="prov_test_synth_64.npz")
        self.assertNotEqual(g.provenance_status, ProvenanceStatus.VERIFIED)
        self.assertEqual(g.provenance_status, ProvenanceStatus.EXPERIMENTAL)
        self.assertIsInstance(g, SyntheticTestGraph)

    def test_cache_preserves_provenance_and_hash(self):
        a = get_or_create_circuit(64, mode=GraphMode.REAL, seed=11,
                                  cache_name="prov_test_cache_64.npz")
        b = get_or_create_circuit(64, mode=GraphMode.REAL, seed=11,
                                  cache_name="prov_test_cache_64.npz")
        self.assertEqual(a.graph_hash, b.graph_hash)
        self.assertEqual(b.mode, GraphMode.REAL)
        self.assertIn("mode", b.provenance_metadata)

    def test_population_metadata_cannot_pose_as_annotated(self):
        for mode, cache in ((GraphMode.REAL, "prov_test_real_128.npz"),
                            (GraphMode.SYNTHETIC_TEST, "prov_test_synth_64.npz")):
            n = 128 if mode == GraphMode.REAL else 64
            g = get_or_create_circuit(n, mode=mode, seed=42, cache_name=cache)
            self.assertGreater(len(g.populations.populations), 0)
            for name, pop in g.populations.populations.items():
                pop.assert_not_empirical()  # raises if VERIFIED or non-heuristic
                d = pop.to_dict()
                self.assertTrue(d["heuristic"])
                self.assertEqual(d["annotation_status"], "no_em_annotation_available")


if __name__ == "__main__":
    unittest.main(verbosity=2)
