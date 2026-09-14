"""Memory + dream robustness: capacity, restart, malformed input, concurrency (P11)."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import threading
import unittest
import numpy as np

from src.memory.persistence import PersistentMemoryManager
from src.memory.working import WorkingMemory

DB_BASE = "diagnostics/test_memory_robust"
_DB_COUNTER = [0]
_last_restart_db = [""]


def fresh_db(tag="default"):
    _DB_COUNTER[0] += 1
    path = f"{DB_BASE}_{tag}_{_DB_COUNTER[0]}.db"
    if os.path.exists(path):
        os.remove(path)
    return PersistentMemoryManager(db_path=path)


class TestMemoryRobustness(unittest.TestCase):
    def test_working_memory_capacity_and_eviction(self):
        wm = WorkingMemory(capacity=7)
        for i in range(10):
            wm.put(f"k{i}", f"v{i}", salience=0.1 + 0.01 * i)
        self.assertLessEqual(len(wm.items), 7)
        # highest salience survives
        self.assertIn("k9", wm.items)
        # decay removes exhausted items
        for _ in range(60):
            wm.decay_step()
        self.assertEqual(len(wm.items), 0)

    def test_restart_persistence_all_stores(self):
        global _last_restart_db
        m1 = fresh_db("restart")
        _last_restart_db = [m1.db_path]
        m1.record_episode(1, {"cue": "x"}, "act", 0.5, 0.1, {"ok": True})
        m1.store_concept("c1", "seeded placeholder", np.ones(4, dtype=np.float32))
        m1.record_skill("s1", ["a", "b"], success=True)
        m1.record_dream(seed=3, base_episode_id=1, simulated_action="explore",
                        counterfactual_reward=0.5, insight="i")
        del m1
        m2 = PersistentMemoryManager(db_path=_last_restart_db[0])
        self.assertEqual(len(m2.get_recent_episodes(5)), 1)
        self.assertEqual(len(m2.query_semantic(np.ones(4, dtype=np.float32), top_k=2)), 1)
        self.assertEqual(len(m2.get_skills()), 1)
        self.assertEqual(len(m2.get_recent_dreams(5)), 1)

    def test_malformed_input_fails_safely(self):
        m = fresh_db("case")
        with self.assertRaises(Exception):
            m.record_episode(1, {"bad": {1, 2, 3}}, "act", 0.5, 0.1, {"ok": True})
        # DB still usable afterwards (no corruption)
        eid = m.record_episode(2, {"good": 1}, "act", 0.5, 0.1, {"ok": True})
        self.assertGreater(eid, 0)
        self.assertEqual(len(m.get_recent_episodes(5)), 1)

    def test_empty_semantic_query(self):
        m = fresh_db("case")
        self.assertEqual(m.query_semantic(np.ones(4, dtype=np.float32)), [])

    def test_concurrent_writes(self):
        m = fresh_db("case")
        errors = []

        def writer(tid):
            try:
                for i in range(20):
                    m.record_episode(tid * 100 + i, {"t": tid}, "act", 0.1, 0.0, {})
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [])
        self.assertEqual(len(m.get_recent_episodes(limit=500)), 160)


if __name__ == "__main__":
    unittest.main(verbosity=2)
