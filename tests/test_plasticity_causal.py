"""Plasticity causal chain (P11): experience -> reward -> weights -> future behavior."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.compute.cpu_reference import cpu_plasticity_step
from src.connectome.types import SyntheticTestGraph
from src.brain.runtime import BrainRuntime


def two_neuron(w0: float = 0.5):
    ro = np.array([0, 0, 1], dtype=np.int32)
    ci = np.array([0], dtype=np.int32)
    return ro, ci, np.array([w0], dtype=np.float32)


class TestPlasticityCausal(unittest.TestCase):
    def test_reward_zero_no_learning(self):
        ro, ci, w = two_neuron()
        w2 = cpu_plasticity_step(ci, w, np.array([1.0, 1.0], dtype=np.float32),
                                 np.array([0.0, 1.0], dtype=np.float32), ro,
                                 learning_rate=0.05, reward=0.0)
        np.testing.assert_array_equal(w2, w)

    def test_reward_sign_and_formula(self):
        ro, ci, w = two_neuron(0.5)
        pre = np.array([1.0, 0.0], dtype=np.float32)
        post = np.array([0.0, 1.0], dtype=np.float32)
        w_up = cpu_plasticity_step(ci, w, pre, post, ro, learning_rate=0.05, reward=1.0)
        self.assertAlmostEqual(float(w_up[0]), 0.5 + 0.05 * (1.0 - 0.01 * 0.5), places=6)
        w_dn = cpu_plasticity_step(ci, w, pre, post, ro, learning_rate=0.05, reward=-1.0)
        self.assertAlmostEqual(float(w_dn[0]), 0.5 - 0.05 * (1.0 - 0.01 * 0.5), places=6)
        # inactive pathway: only decay term
        w_idle = cpu_plasticity_step(ci, w, np.array([0.0, 0.0], dtype=np.float32), post,
                                     ro, learning_rate=0.05, reward=1.0)
        self.assertAlmostEqual(float(w_idle[0]), 0.5 + 0.05 * (0.0 - 0.01 * 0.5), places=6)

    def test_weight_change_alters_future_trajectory(self):
        # Closed loop: rewarded co-firing strengthens A->B until a probe that was
        # subthreshold reliably drives B; unrewarded control never learns.
        def make_rt(w0):
            ro, ci, w = two_neuron(w0)
            g = SyntheticTestGraph(
                neuron_ids=np.array([1, 2], dtype=np.int64),
                coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
                tbars=np.array([10, 10], dtype=np.int32), sides=["L", "R"],
                row_offsets=ro, col_indices=ci, weights=w)
            return BrainRuntime(g, use_gpu=False, enable_plasticity=True, seed=1)

        trained, control = make_rt(0.6), make_rt(0.6)
        # Staggered drive: A fires on even steps, B on odd steps, so each odd
        # step pairs pre-spike S_A(t-1)=1 with post-spike S_B(t)=1 (lockstep
        # driving cannot potentiate under absolute refractoriness).
        # Reward only on odd steps: potentiation is rewarded, idle decay is not.
        stim_a = np.array([10.0, 0.0], dtype=np.float32)
        stim_b = np.array([0.0, 10.0], dtype=np.float32)
        # A fires every 4th step under t_ref=2; odd followers potentiate.
        for i in range(61):
            stim = stim_a if i % 2 == 0 else stim_b
            rew = 1.0 if i % 2 == 1 else 0.0
            trained.step(sensory_inputs={"visual": stim}, reward=rew)
            control.step(sensory_inputs={"visual": stim}, reward=0.0)
        wt = float(trained.graph.weights[0])
        wc = float(control.graph.weights[0])
        self.assertGreater(wt, wc + 0.2, f"trained weight should grow, got {wt} vs {wc}")
        self.assertAlmostEqual(wc, 0.6, places=6)
        # Probe: identical A-spike input. The trained pathway must produce a
        # measurably stronger future response (changed trajectory from changed weights).
        for rt in (trained, control):
            rt.state.membrane_potentials = np.zeros(2, dtype=np.float32)
            rt.state.spikes = np.array([1.0, 0.0], dtype=np.float32)
            rt.state.refractory_steps = np.zeros(2, dtype=np.int32)
            rt.step(sensory_inputs=None, reward=0.0)
        vt = float(trained.state.membrane_potentials[1])
        vc = float(control.state.membrane_potentials[1])
        self.assertGreater(vt - vc, 0.2, f"future response must differ: {vt} vs {vc}")
        # Control pathway stays subthreshold and silent.
        self.assertEqual(float(control.state.spikes[1]), 0.0)
        trained.cleanup()
        control.cleanup()

    def test_organism_plasticity_on_off_diverges_weights(self):
        # P12: identical organisms in identical worlds; only neural plasticity differs.
        from src.genome.schema import Genome
        from src.organism.organism import Organism
        from src.world.environment import GridWorld, WorldConfig
        from src.connectome.types import GraphMode

        def make_pair(plasticity: bool):
            g = Genome.founder(900)
            o = Organism(g, "learn-org", 0,
                         {"organism_seed": 901, "development_seed": 902},
                         GraphMode.SYNTHETIC_TEST, 32, start_pos=(2, 2))
            o.brain.enable_plasticity = plasticity
            w = GridWorld(WorldConfig(width=8, height=8, n_resources=10,
                                      n_hazards=0, world_seed=903))
            w.place(o.id, (2, 2))
            return o, w

        o_on, w_on = make_pair(True)
        o_off, w_off = make_pair(False)
        for _ in range(40):
            o_on.step(w_on)
            o_off.step(w_off)
        # Neural plasticity fired while foraging (real rewards), and only there.
        self.assertGreater(o_on.brain.plasticity_updates, 0,
                           "plastic organism must apply synaptic updates while foraging")
        self.assertEqual(o_off.brain.plasticity_updates, 0,
                         "plasticity-disabled brain must never update")
        # ...and the updates changed the executable substrate.
        sig_on = (o_on.graph.weights.shape, o_on.graph.graph_hash)
        sig_off = (o_off.graph.weights.shape, o_off.graph.graph_hash)
        self.assertNotEqual(sig_on, sig_off)


if __name__ == "__main__":
    unittest.main(verbosity=2)
