"""SimulationEngine concurrency stress test (P12): real threads, real contention."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import threading
import time
import unittest
import numpy as np

from src.connectome.types import GraphMode
from src.brain.simulation_engine import SimulationEngine


class TestEngineConcurrencyStress(unittest.TestCase):
    def test_stress_no_races_no_deadlock(self):
        eng = SimulationEngine(circuit_size=64, graph_mode=GraphMode.SYNTHETIC_TEST,
                               use_gpu=False, seed=21,
                               db_path="diagnostics/test_stress_memory.db")
        eng.target_hz = 50.0
        errors = []
        stop = threading.Event()
        steps_seen = []

        def reader(tid):
            try:
                for _ in range(120):
                    t = eng.get_latest_telemetry()
                    s = eng.get_current_state()
                    c = eng.get_connectome_3d_view(max_nodes=32, max_edges=64)
                    assert t["step"] >= 0
                    assert s["connectome"]["num_neurons"] == 64
                    assert len(c["nodes"]) == 32
                    assert t["total_spikes"] >= 0
                    steps_seen.append(t["step"])
                    eng.set_sensory_stimulus("visual", np.full(8, 0.2, dtype=np.float32))
                    eng.add_reward(0.01)
                    if stop.is_set():
                        return
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        def stepper():
            try:
                for _ in range(30):
                    eng.step_single(n_steps=2)
                    if stop.is_set():
                        return
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        def cycler():
            try:
                for _ in range(6):
                    eng.start()
                    time.sleep(0.05)
                    eng.pause()
                    if stop.is_set():
                        return
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        eng.start()
        threads = ([threading.Thread(target=reader, args=(i,)) for i in range(4)]
                   + [threading.Thread(target=stepper) for _ in range(2)]
                   + [threading.Thread(target=cycler)])
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        stop.set()
        alive = [t for t in threads if t.is_alive()]
        eng.pause()
        self.assertEqual(alive, [], f"deadlocked threads: {alive}")
        self.assertEqual(errors, [])
        # step counter advanced and never went negative/reset
        self.assertGreater(eng.brain.state.step_count, 0)
        self.assertTrue(all(s >= 0 for s in steps_seen))
        eng.circuit.validate_invariants()
        eng.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
