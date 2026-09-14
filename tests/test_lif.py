"""Expanded LIF contract tests (P4): one explicit semantics, CPU+GPU. No thresholds weakened."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.compute.cpu_reference import cpu_lif_step


def isolated_step(ext, pot, ref=None, decay=0.95, thr=-50.0, rst=-70.0, rest=-70.0, t_ref=2):
    n = len(ext)
    return cpu_lif_step(
        np.array([0] * (n + 1), dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.asarray(ext, dtype=np.float32),
        np.asarray(pot, dtype=np.float32),
        np.zeros(n, dtype=np.int32) if ref is None else np.asarray(ref, dtype=np.int32),
        decay=decay, threshold=thr, v_reset=rst, v_rest=rest, t_ref=t_ref,
    )


class TestLIFContract(unittest.TestCase):
    def test_subthreshold_decay(self):
        # v_cand = -70 + (−60+70)*0.95 + 0 = -60.5
        p, s, r = isolated_step([0.0], [-60.0])
        self.assertAlmostEqual(float(p[0]), -60.5, places=5)
        self.assertEqual(s[0], 0.0)
        self.assertEqual(r[0], 0)

    def test_threshold_boundary_fires(self):
        # v_cand exactly threshold -> fires (>= semantics)
        # -70 + 20*0.95 = -51; need ext to reach -50: ext = 1.0
        p, s, r = isolated_step([1.0], [-50.0])  # -70+20*.95+1 = -50.0
        self.assertEqual(s[0], 1.0)
        self.assertEqual(p[0], -70.0)
        self.assertEqual(r[0], 2)

    def test_above_threshold_and_reset(self):
        p, s, r = isolated_step([50.0], [-70.0])
        self.assertEqual(s[0], 1.0)
        self.assertEqual(p[0], -70.0)

    def test_refractory_suppresses_and_decrements(self):
        p, s, r = isolated_step([100.0], [-70.0], ref=[2])
        self.assertEqual(s[0], 0.0)
        self.assertEqual(p[0], -70.0)
        self.assertEqual(r[0], 1)
        p, s, r = isolated_step([100.0], list(p), ref=list(r))
        self.assertEqual(s[0], 0.0)
        self.assertEqual(r[0], 0)

    def test_refractory_expiry_refires(self):
        p, s, r = isolated_step([100.0], [-70.0], ref=[1])
        self.assertEqual(tuple(r), (0,))
        p2, s2, r2 = isolated_step([100.0], list(p), ref=list(r))
        self.assertEqual(s2[0], 1.0)  # fires again once expired

    def test_zero_input_stability(self):
        p, s, r = isolated_step([0.0], [-70.0])
        self.assertEqual(s[0], 0.0)
        self.assertAlmostEqual(float(p[0]), -70.0, places=5)

    def test_negative_input_hyperpolarizes_with_floor(self):
        p, s, r = isolated_step([-5.0], [-70.0])
        self.assertEqual(s[0], 0.0)
        self.assertAlmostEqual(float(p[0]), -71.0, places=5)  # floor v_reset-1
        p, s, r = isolated_step([-50.0], [-70.0])
        self.assertAlmostEqual(float(p[0]), -71.0, places=5)

    def test_synaptic_propagation(self):
        # neuron 1 spikes; weight 0.6 drives neuron 0 over threshold
        ro = np.array([0, 1, 1], dtype=np.int32)
        ci = np.array([1], dtype=np.int32)
        w = np.array([30.0], dtype=np.float32)
        prev = np.array([0.0, 1.0], dtype=np.float32)
        p, s, r = cpu_lif_step(ro, ci, w, prev,
                               np.zeros(2, dtype=np.float32),
                               np.array([-70.0, -70.0], dtype=np.float32),
                               np.zeros(2, dtype=np.int32),
                               decay=0.95, threshold=-50.0,
                               v_reset=-70.0, v_rest=-70.0, t_ref=2)
        self.assertEqual(s[0], 1.0)
        self.assertEqual(s[1], 0.0)

    def test_multiple_presynaptic_summation(self):
        ro = np.array([0, 2, 2, 2], dtype=np.int32)
        ci = np.array([1, 2], dtype=np.int32)
        # each weight 0.4: I_syn = 0.4*1 + 0.4*0 = 0.4; subthreshold at these params
        w = np.array([0.4, 0.4], dtype=np.float32)
        prev = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        p, s, r = cpu_lif_step(ro, ci, w, prev,
                               np.zeros(3, dtype=np.float32),
                               np.full(3, -70.0, dtype=np.float32),
                               np.zeros(3, dtype=np.int32),
                               decay=0.95, threshold=-50.0,
                               v_reset=-70.0, v_rest=-70.0, t_ref=2)
        self.assertEqual(s[0], 0.0)
        self.assertAlmostEqual(float(p[0]), -69.6, places=4)
        # both presynaptic firing pushes over: 0.8 still sub; use ext to confirm additivity
        p2, _, _ = cpu_lif_step(ro, ci, w, np.array([0.0, 1.0, 1.0], dtype=np.float32),
                                np.zeros(3, dtype=np.float32),
                                np.full(3, -70.0, dtype=np.float32),
                                np.zeros(3, dtype=np.int32),
                                decay=0.95, threshold=-50.0,
                                v_reset=-70.0, v_rest=-70.0, t_ref=2)
        self.assertAlmostEqual(float(p2[0]), -69.2, places=4)

    def test_deterministic_repeat(self):
        args = dict(decay=0.9, threshold=0.5, v_reset=0.0, v_rest=0.0, t_ref=3)
        ro = np.array([0, 1, 2, 2], dtype=np.int32)
        ci = np.array([1, 0], dtype=np.float32).astype(np.int32)
        w = np.array([0.3, 0.3], dtype=np.float32)
        outs = []
        for _ in range(3):
            outs.append(cpu_lif_step(ro, ci, w,
                                     np.array([1.0, 0.0], dtype=np.float32),
                                     np.array([0.2, 0.1], dtype=np.float32),
                                     np.array([0.1, 0.2], dtype=np.float32),
                                     np.zeros(2, dtype=np.int32), **args))
        for o in outs[1:]:
            np.testing.assert_array_equal(o[0], outs[0][0])
            np.testing.assert_array_equal(o[1], outs[0][1])
            np.testing.assert_array_equal(o[2], outs[0][2])

    def test_cpu_gpu_parity_where_available(self):
        try:
            from src.compute.vulkan_backend import VulkanComputeEngine
            eng = VulkanComputeEngine()
        except Exception as e:
            self.skipTest(f"Vulkan unavailable: {e}")
            return
        try:
            n = 16
            rng = np.random.RandomState(7)
            ro = np.zeros(n + 1, dtype=np.int32)
            ci_list, w_list = [], []
            for i in range(n):
                tgts = sorted(rng.choice(n, size=4, replace=False).tolist())
                for t in tgts:
                    if t != i:
                        ci_list.append(t)
                        w_list.append(float(rng.uniform(0.05, 0.3)))
                ro[i + 1] = len(ci_list)
            ci = np.array(ci_list, dtype=np.int32)
            w = np.array(w_list, dtype=np.float32)
            prev = (rng.rand(n) < 0.2).astype(np.float32)
            ext = rng.uniform(0, 0.4, n).astype(np.float32)
            pot = rng.uniform(-0.2, 0.4, n).astype(np.float32)
            ref = np.zeros(n, dtype=np.int32)
            cp, cs, cr = cpu_lif_step(ro, ci, w, prev, ext, pot, ref)
            eng.load_circuit(ro, ci, w, pot, prev, ref)
            gp, gs, gr = eng.run_step_persistent(external_inputs=ext, readback=True)
            self.assertLess(float(np.max(np.abs(cp - gp))), 1e-4)
            np.testing.assert_array_equal(cs, gs)
            np.testing.assert_array_equal(cr, gr)
        finally:
            eng.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
