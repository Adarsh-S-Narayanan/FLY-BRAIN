"""CSR directionality regression tests (P0): A->B must drive B, never A.

Convention (loader v3): row i stores INCOMING edges (presynaptic sources of i).
These tests fail if the convention ever inverts again.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.compute.cpu_reference import cpu_lif_step


def ab_graph(weight: float = 5.0):
    """Single directed edge 0 -> 1 in incoming-per-row CSR."""
    ro = np.array([0, 0, 1], dtype=np.int32)
    ci = np.array([0], dtype=np.int32)
    w = np.array([weight], dtype=np.float32)
    return ro, ci, w


def lif_step(ro, ci, w, prev, ext=None):
    n = 2
    if ext is None:
        ext = np.zeros(n, dtype=np.float32)
    return cpu_lif_step(ro, ci, w, np.asarray(prev, dtype=np.float32), ext,
                        np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.int32),
                        decay=0.85, threshold=1.0, v_reset=0.0, v_rest=0.0, t_ref=2)


class TestCSRDirectionality(unittest.TestCase):
    def test_csr_directionality_cpu(self):
        ro, ci, w = ab_graph()
        # A spikes, B silent -> B must receive the signal (V_B = 5.0 >= thresh -> fires)
        p, s, _ = lif_step(ro, ci, w, [1.0, 0.0])
        self.assertEqual(s[1], 1.0, "postsynaptic neuron must fire when presynaptic spikes")
        self.assertAlmostEqual(float(p[1]), 0.0, places=5)  # reset after spike
        # B spikes, A silent -> A must NOT receive anything through edge A->B
        p2, s2, _ = lif_step(ro, ci, w, [0.0, 1.0])
        self.assertEqual(s2[0], 0.0, "presynaptic neuron must stay silent")
        self.assertAlmostEqual(float(p2[0]), 0.0, places=5)

    def test_csr_directionality_subthreshold(self):
        ro, ci, w = ab_graph(weight=0.4)
        p, s, _ = lif_step(ro, ci, w, [1.0, 0.0])
        self.assertEqual(s[1], 0.0)
        self.assertAlmostEqual(float(p[1]), 0.4, places=5)
        p2, _, _ = lif_step(ro, ci, w, [0.0, 1.0])
        self.assertAlmostEqual(float(p2[0]), 0.0, places=5)

    def test_csr_directionality_vulkan(self):
        try:
            from src.compute.vulkan_backend import VulkanComputeEngine
            eng = VulkanComputeEngine()
        except Exception as e:
            self.skipTest(f"Vulkan unavailable: {e}")
            return
        try:
            ro, ci, w = ab_graph()
            n = 2
            eng.load_circuit(ro, ci, w, np.zeros(n, dtype=np.float32),
                             np.array([1.0, 0.0], dtype=np.float32))
            gp, gs, _ = eng.run_step_persistent(
                external_inputs=np.zeros(n, dtype=np.float32), decay=0.85,
                threshold=1.0, v_reset=0.0, v_rest=0.0, t_ref=2, readback=True)
            self.assertEqual(float(gs[1]), 1.0)
            # reverse: B spikes -> A silent
            eng2 = VulkanComputeEngine()
            eng2.load_circuit(ro, ci, w, np.zeros(n, dtype=np.float32),
                              np.array([0.0, 1.0], dtype=np.float32))
            gp2, gs2, _ = eng2.run_step_persistent(
                external_inputs=np.zeros(n, dtype=np.float32), decay=0.85,
                threshold=1.0, v_reset=0.0, v_rest=0.0, t_ref=2, readback=True)
            self.assertEqual(float(gs2[0]), 0.0)
            eng2.cleanup()
        finally:
            eng.cleanup()

    def test_brainruntime_directionality(self):
        from src.connectome.types import SyntheticTestGraph
        from src.brain.runtime import BrainRuntime
        ro, ci, w = ab_graph()
        g = SyntheticTestGraph(
            neuron_ids=np.array([1, 2], dtype=np.int64),
            coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            tbars=np.array([10, 10], dtype=np.int32),
            sides=["L", "R"],
            row_offsets=ro, col_indices=ci, weights=w)
        rt = BrainRuntime(g, use_gpu=False, enable_plasticity=False, seed=1)
        # force A spiking by direct state, then step with no input
        rt.state.spikes = np.array([1.0, 0.0], dtype=np.float32)
        out = rt.step(sensory_inputs=None, reward=0.0)
        self.assertGreater(out["spikes"], 0)
        self.assertEqual(float(rt.state.spikes[1]), 1.0)
        rt.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
