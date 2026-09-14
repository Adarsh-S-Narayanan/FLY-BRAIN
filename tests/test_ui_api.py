"""UI/API behavioral verification (P15): every endpoint returns real backend state."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from fastapi.testclient import TestClient
import src.ui.server as srv

client = TestClient(srv.app)


class TestUIAPI(unittest.TestCase):
    def test_health_state_telemetry_real(self):
        h = client.get("/api/health").json()
        self.assertEqual(h["status"], "HEALTHY")
        self.assertIn(h["graph_mode"], ("REAL", "SPATIAL_SURROGATE", "SYNTHETIC_TEST"))
        t = client.get("/api/telemetry").json()
        for k in ("step", "spikes", "drives", "graph_mode", "backend"):
            self.assertIn(k, t)
        s = client.get("/api/state").json()
        self.assertIn("simulation", s)
        self.assertIn("connectome", s)

    def test_connectome_shows_provenance(self):
        c = client.get("/api/connectome", params={"max_nodes": 64, "max_edges": 64}).json()
        self.assertIn(c["mode"], ("REAL", "SPATIAL_SURROGATE", "SYNTHETIC_TEST"))
        self.assertIn("provenance_status", c)
        self.assertIn("graph_hash", c)
        self.assertLessEqual(len(c["nodes"]), 64)
        # nodes carry real biological metadata
        n0 = c["nodes"][0]
        for k in ("id", "pos", "side", "tbars", "act", "pot", "spk"):
            self.assertIn(k, n0)

    def test_simulation_controls_work(self):
        r = client.post("/api/simulation/step", json={"steps": 2, "reward": 0.1}).json()
        self.assertIn("spikes", r)
        self.assertGreaterEqual(r["step"], 2)

    def test_memory_dreams_lineage_real(self):
        m = client.get("/api/memory", params={"limit": 3}).json()
        for k in ("working", "episodes", "skills", "dreams"):
            self.assertIn(k, m)
        d = client.post("/api/dreams/replay", params={"mode": "deterministic", "count": 1}).json()
        self.assertTrue(isinstance(d, list) and len(d) == 1)
        self.assertIn("counterfactual_action", d[0])
        self.assertIn("consolidation_result", d[0])
        lin = client.get("/api/evolution/lineage").json()
        self.assertIn("history", lin)

    def test_colony_lifecycle(self):
        r = client.post("/api/colony/reset", json={"size": 4, "seed": 77, "circuit_size": 32}).json()
        self.assertEqual(r["status"], "COLONY_RESET")
        c0 = client.get("/api/colony").json()
        self.assertEqual(c0["total"], 4)
        self.assertIn("population_hash", c0)
        oid = c0["organisms"][0]["id"]
        det = client.get(f"/api/colony/organism/{oid}").json()
        for k in ("genome", "genome_hash", "brain", "memory", "culture", "organism_hash"):
            self.assertIn(k, det)
        st = client.post("/api/colony/step", params={"ticks": 10}).json()
        self.assertEqual(st["status"], "STEPPED")
        c1 = client.get("/api/colony").json()
        self.assertNotEqual(c1["population_hash"], c0["population_hash"])
        lin = client.get("/api/colony/lineage").json()
        self.assertIn("genetic", lin)
        self.assertIn("cultural", lin)
        self.assertEqual(client.get("/api/colony/organism/nonexistent").status_code, 404)

    def test_diagnostics_real(self):
        d = client.get("/api/diagnostics").json()
        self.assertIn("system", d)
        self.assertIn("dataset_provenance", d)
        self.assertIn("runtime", d)
        self.assertTrue(d["dataset_provenance"]["soma_sha256"])

    def test_llm_status_discovery_and_tools(self):
        st = client.get("/api/llm/status").json()
        self.assertIn("runtime_status", st)
        self.assertIn("discovered", st)
        self.assertGreaterEqual(len(st["discovered"]), 1)
        self.assertEqual(st["discovered"][0]["architecture"], "llama")
        tools = client.get("/api/llm/tools").json()
        for required in ("inspect_brain_state", "inspect_provenance", "query_memory",
                         "run_bounded_experiment", "inspect_population"):
            self.assertIn(required, tools)

    def test_llm_tool_safety_rejects_unknown(self):
        r = client.post("/api/llm/tool", json={"tool": "run_shell", "params": {}}).json()
        self.assertEqual(r["status"], "REJECTED")

    def test_llm_tool_inspection_returns_real_state(self):
        r = client.post("/api/llm/tool",
                        json={"tool": "inspect_provenance", "params": {}}).json()
        self.assertEqual(r["status"], "SUCCESS")
        self.assertIn("graph_hash", r["result"])
        self.assertIn("csr_convention", r["result"])

    def test_llm_generate_real(self):
        r = client.post("/api/llm/generate",
                        json={"prompt": "1 + 1 =", "max_tokens": 8, "seed": 3}).json()
        self.assertEqual(r["status"], "SUCCESS")
        self.assertIn("model_sha256", r["provenance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
