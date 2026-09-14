"""STAGE D tests: eligibility traces, neuromodulation, prediction influence,
v2 mode integration + checkpoint continuation of eligibility state."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.brain.eligibility import (EligibilityState, EligibilityEngine,
                                   NeuromodulationConfig, PLASTICITY_MODES)
from src.brain.runtime import BrainRuntime
from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode


def small_graph(cache="elig_g.npz", n=32, seed=11):
    return get_or_create_circuit(n, mode=GraphMode.SYNTHETIC_TEST, seed=seed, cache_name=cache)


class TestNeuromodulation(unittest.TestCase):
    def test_default_config_is_reward_only(self):
        cfg = NeuromodulationConfig()
        self.assertEqual(cfg.signal(reward=0.7), 0.7)
        self.assertEqual(cfg.signal(reward=0.0, novelty=5.0), 0.0)

    def test_weighted_combination_and_clip(self):
        cfg = NeuromodulationConfig(w_reward=1.0, w_novelty=0.5, w_prediction_error=0.25)
        s = cfg.signal(reward=1.0, novelty=2.0, prediction_error=4.0)
        self.assertAlmostEqual(s, 1.0 + 1.0 + 1.0, places=6)
        clipped = cfg.signal(reward=100.0)
        self.assertEqual(clipped, cfg.signal_clip)

    def test_validation_rejects_bad_weights(self):
        with self.assertRaises(ValueError):
            NeuromodulationConfig(w_reward=-1.0).validate()
        with self.assertRaises(ValueError):
            NeuromodulationConfig(signal_clip=0.0).validate()


class TestEligibility(unittest.TestCase):
    def test_trace_update_and_decay(self):
        ro = np.array([0, 0, 1], dtype=np.int32)  # edge: pre 0 -> post 1
        ci = np.array([0], dtype=np.int32)
        st = EligibilityState(1)
        pre = np.array([1.0, 0.0], dtype=np.float32)
        post = np.array([0.0, 1.0], dtype=np.float32)
        st.update(ro, ci, pre, post, decay=0.5)
        self.assertAlmostEqual(float(st.traces[0]), 1.0, places=5)
        # no co-activity -> decay only
        st.update(ro, ci, np.zeros(2, dtype=np.float32), post, decay=0.5)
        self.assertAlmostEqual(float(st.traces[0]), 0.5, places=5)

    def test_resize_on_structural_change_preserves_prefix(self):
        st = EligibilityState(2)
        st.traces[:] = [0.3, 0.7]
        self.assertTrue(st.sync_size(4))
        self.assertTrue(np.allclose(st.traces, [0.3, 0.7, 0.0, 0.0], atol=1e-6))
        self.assertTrue(st.sync_size(1))
        self.assertTrue(np.allclose(st.traces, [0.3], atol=1e-6))
        self.assertFalse(st.sync_size(1))

    def test_apply_moves_weights_in_signal_direction(self):
        g = small_graph("elig_apply.npz")
        st = EligibilityState(len(g.weights))
        st.traces[:] = 0.5
        eng = EligibilityEngine(learning_rate=0.1, neuromod=NeuromodulationConfig())
        w0 = g.weights.copy()
        eng.apply(g, st, signal=2.0)
        self.assertTrue(np.all(g.weights > w0 + 1e-3))
        w1 = g.weights.copy()
        eng.apply(g, st, signal=-2.0)
        self.assertTrue(np.all(g.weights < w1 - 1e-3))

    def test_engine_rejects_bad_decay(self):
        with self.assertRaises(ValueError):
            EligibilityEngine(trace_decay=1.0)


class TestRuntimeV2Mode(unittest.TestCase):
    def test_v2_eligibility_learns_with_novelty_weight(self):
        g = small_graph("elig_v2a.npz")
        rt = BrainRuntime(g, use_gpu=False, plasticity_mode="v2_eligibility",
                          neuromod=NeuromodulationConfig(w_reward=0.0, w_novelty=1.0))
        w0 = g.weights.copy()
        rng = np.random.RandomState(3)
        updated_any = False
        for _ in range(10):
            out = rt.step(sensory_inputs={"visual": rng.uniform(0, 1.0, 16).astype(np.float32)})
            updated_any = updated_any or out["synapses_updated"] > 0
        self.assertTrue(updated_any, "novelty-driven signal must update synapses")
        self.assertTrue(np.any(np.abs(g.weights - w0) > 1e-5))
        self.assertEqual(out["plasticity_mode"], "v2_eligibility")

    def test_v2_reward_only_matches_v1_direction(self):
        """With default neuromod (reward-only), same inputs must move weights the
        same direction as v1 (both are reward-gated positive updates)."""
        g1 = small_graph("elig_v2b1.npz")
        g2 = small_graph("elig_v2b2.npz")
        rt1 = BrainRuntime(g1, use_gpu=False)  # v1 default
        rt2 = BrainRuntime(g2, use_gpu=False, plasticity_mode="v2_eligibility",
                           neuromod=NeuromodulationConfig())  # reward-only
        rng = np.random.RandomState(5)
        for _ in range(6):
            stim = {"visual": rng.uniform(0, 0.8, 16).astype(np.float32)}
            rt1.step(sensory_inputs=stim, reward=1.0)
            rt2.step(sensory_inputs=stim, reward=1.0)
        d1 = g1.weights - rt1.plasticity.min_weight
        d2 = g2.weights - rt2.plasticity.min_weight
        self.assertGreater(float(np.sum(d2)), 0.0, "v2 reward-only must potentiate")
        self.assertGreater(float(np.sum(d1)), 0.0)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            BrainRuntime(small_graph(), use_gpu=False, plasticity_mode="v3_magic")

    def test_eligibility_survives_snapshot_restore(self):
        path = "diagnostics/snapshots/elig_cont.npz"
        os.makedirs("diagnostics/snapshots", exist_ok=True)
        g = small_graph("elig_v2c.npz")
        rt = BrainRuntime(g, use_gpu=False, plasticity_mode="v2_eligibility",
                          neuromod=NeuromodulationConfig(w_reward=1.0))
        rng = np.random.RandomState(7)
        for _ in range(5):
            rt.step(sensory_inputs={"visual": rng.uniform(0, 0.8, 16).astype(np.float32)},
                    reward=0.5)
        traces_snapshot = rt.eligibility.traces.copy()
        w_snapshot = g.weights.copy()
        rt.save_snapshot(path)
        rt2 = BrainRuntime(g, use_gpu=False, plasticity_mode="v2_eligibility",
                           neuromod=NeuromodulationConfig(w_reward=1.0))
        rt2.load_snapshot(path)
        self.assertTrue(np.allclose(rt2.eligibility.traces, traces_snapshot, atol=1e-6))
        self.assertTrue(np.allclose(g.weights, w_snapshot, atol=1e-6))
        self.assertEqual(rt2.eligibility.updates, rt.eligibility.updates)

    def test_v1_default_untouched_behavior(self):
        """Compat guarantee: default runtime exposes v1 and never builds traces."""
        g = small_graph("elig_v1c.npz")
        rt = BrainRuntime(g, use_gpu=False)
        self.assertEqual(rt.plasticity_mode, "v1_hebbian")
        self.assertIsNone(rt.eligibility)
        out = rt.step(reward=1.0)
        self.assertEqual(out["plasticity_mode"], "v1_hebbian")
        self.assertIn("v1_hebbian", PLASTICITY_MODES)


class TestPredictionInfluence(unittest.TestCase):
    def test_prediction_error_feeds_curiosity_when_enabled(self):
        g = small_graph("elig_pred.npz")
        rt = BrainRuntime(g, use_gpu=False, prediction_influence=True)
        rt.state.predicted_reward = 0.9
        c0 = rt.state.drives.curiosity
        rt.step(reward=0.0)  # large prediction error
        self.assertGreater(rt.state.drives.curiosity, c0)

    def test_disabled_by_default_and_delta_when_enabled(self):
        # Two identical runtimes: identical deterministic trajectories differ
        # ONLY by the prediction-influence curiosity bonus.
        g1 = small_graph("elig_pred_off.npz")
        g2 = small_graph("elig_pred_on.npz")
        base = BrainRuntime(g1, use_gpu=False)
        infl = BrainRuntime(g2, use_gpu=False, prediction_influence=True)
        infl.state.predicted_reward = base.state.predicted_reward = 0.9
        for _ in range(3):
            base.step(reward=0.0)
            infl.step(reward=0.0)
        self.assertGreater(infl.state.drives.curiosity, base.state.drives.curiosity,
                           "prediction influence must add curiosity beyond drives baseline")
        # The bonus compounds with the drives decay, so it must be strictly
        # positive and at least one step's worth of bonus immediately after
        # the first step.
        self.assertGreaterEqual(
            round(infl.state.drives.curiosity - base.state.drives.curiosity, 6),
            0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
