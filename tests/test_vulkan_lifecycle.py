"""Vulkan engine hardening tests (P5). Honest SKIP when no hardware is present."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np


def make_engine(testcase):
    try:
        from src.compute.vulkan_backend import VulkanComputeEngine
        return VulkanComputeEngine()
    except Exception as e:
        testcase.skipTest(f"Vulkan unavailable: {e}")


def tiny_circuit(n=32, seed=3):
    from src.connectome.loader import get_or_create_circuit
    from src.connectome.types import GraphMode
    g = get_or_create_circuit(n, mode=GraphMode.SYNTHETIC_TEST, seed=seed,
                              cache_name="vk_lifecycle_synth_32.npz")
    return g


class TestVulkanLifecycle(unittest.TestCase):
    def test_init_and_discovery(self):
        eng = make_engine(self)
        try:
            self.assertTrue(eng.device_name and eng.device_name != "Unknown")
            self.assertIsNotNone(eng.device)
            self.assertIsNotNone(eng.queue)
            self.assertGreaterEqual(eng.queue_family_idx, 0)
            d = eng.get_diagnostics()
            self.assertEqual(d["status"], "OPERATIONAL")
        finally:
            eng.cleanup()

    def test_repeated_steps_stable(self):
        eng = make_engine(self)
        try:
            g = tiny_circuit()
            n = g.num_neurons
            eng.load_circuit(g.row_offsets, g.col_indices, g.weights,
                             np.zeros(n, dtype=np.float32),
                             np.zeros(n, dtype=np.float32))
            rng = np.random.RandomState(11)
            for _ in range(10):
                ext = rng.uniform(0, 0.5, n).astype(np.float32)
                pot, spk, ref = eng.run_step_persistent(external_inputs=ext, readback=True)
                self.assertEqual(len(pot), n)
                self.assertTrue(bool(np.all((spk == 0.0) | (spk == 1.0))))
                self.assertTrue(bool(np.all(ref >= 0)))
                self.assertTrue(bool(np.all(np.isfinite(pot))))
            self.assertEqual(eng.total_steps_executed, 10)
        finally:
            eng.cleanup()

    def test_repeated_reload_no_corruption(self):
        eng = make_engine(self)
        try:
            g = tiny_circuit()
            n = g.num_neurons
            hashes = []
            for i in range(3):
                eng.load_circuit(g.row_offsets, g.col_indices, g.weights,
                                 np.full(n, 0.1, dtype=np.float32),
                                 np.zeros(n, dtype=np.float32))
                pot, spk, ref = eng.run_step_persistent(
                    external_inputs=np.full(n, 0.2, dtype=np.float32), readback=True)
                h = (pot.tobytes(), spk.tobytes(), ref.tobytes())
                hashes.append(h)
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(hashes[1], hashes[2])
        finally:
            eng.cleanup()

    def test_cleanup_idempotent_and_reinit(self):
        from src.compute.vulkan_backend import VulkanComputeEngine
        eng = make_engine(self)
        try:
            g = tiny_circuit()
            n = g.num_neurons
            eng.load_circuit(g.row_offsets, g.col_indices, g.weights,
                             np.zeros(n, dtype=np.float32),
                             np.zeros(n, dtype=np.float32))
        finally:
            eng.cleanup()
        eng.cleanup()  # second cleanup must not raise
        self.assertIsNone(eng.device)
        eng2 = VulkanComputeEngine()  # fresh init after cleanup works
        try:
            self.assertIsNotNone(eng2.device)
        finally:
            eng2.cleanup()

    def test_persistent_resource_identity_across_steps(self):
        eng = make_engine(self)
        try:
            g = tiny_circuit()
            n = g.num_neurons
            eng.load_circuit(g.row_offsets, g.col_indices, g.weights,
                             np.zeros(n, dtype=np.float32),
                             np.zeros(n, dtype=np.float32))
            pipe0, cmd0 = eng.brain_pipeline, eng.brain_command_buffer
            eng.run_step_persistent(external_inputs=np.zeros(n, dtype=np.float32))
            self.assertEqual(eng.brain_pipeline, pipe0)
            self.assertEqual(eng.brain_command_buffer, cmd0)
        finally:
            eng.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
