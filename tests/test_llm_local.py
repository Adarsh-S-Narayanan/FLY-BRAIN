"""Local LLM integration tests (REAL inference, REAL failure modes)."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from src.llm.discovery import discover_models, ModelInfo
from src.llm.runtime import LocalLLM, GenerationConfig, RESEARCH_DETERMINISTIC
from src.llm.scientist import ScientistLoop, ToolSpec

MODEL_PATH = os.path.join("llm", "gguf", "MiniCPM5-2B-Q8_0.gguf")


class TestLLMDiscovery(unittest.TestCase):
    def test_discovers_real_model_with_metadata(self):
        models = [m for m in discover_models() if m.status == "DISCOVERED"]
        self.assertGreater(len(models), 0, "llm/ must contain the GGUF model")
        m = next(x for x in models if x.filename == "MiniCPM5-2B-Q8_0.gguf")
        self.assertEqual(m.architecture, "llama")
        self.assertEqual(m.quantization, "MOSTLY_Q8_0")
        self.assertEqual(m.context_length, 131072)
        self.assertGreater(m.block_count, 0)
        self.assertEqual(len(m.sha256), 64)
        self.assertGreater(m.size_bytes, 2_000_000_000)

    def test_missing_dir_is_unavailable_not_error(self):
        self.assertEqual(discover_models(roots=("no_such_dir_xyz",)), [])

    def test_corrupt_file_reported_never_loaded(self):
        os.makedirs("diagnostics/llm_corrupt", exist_ok=True)
        bad = "diagnostics/llm_corrupt/bad.gguf"
        with open(bad, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 100)
        models = discover_models(roots=("diagnostics/llm_corrupt",))
        self.assertEqual(len(models), 1)
        self.assertTrue(models[0].status.startswith("CORRUPT"))
        llm = LocalLLM(models[0])
        self.assertFalse(llm.load())
        self.assertEqual(llm.status, "MODEL_LOAD_ERROR")


class TestLLMRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = LocalLLM.auto(n_ctx=1024)
        if cls.llm.model is None:
            raise unittest.SkipTest("no GGUF model discovered")
        if not cls.llm.load():
            raise unittest.SkipTest(f"model failed to load: {cls.llm.last_error}")

    @classmethod
    def tearDownClass(cls):
        cls.llm.unload()

    def test_real_inference_returns_text(self):
        res = self.llm.generate("Count: 1, 2,",
                                GenerationConfig(max_tokens=8, seed=1))
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["text"] and len(res["text"].strip()) > 0)
        self.assertIn("model_sha256", res["provenance"])
        self.assertEqual(res["provenance"]["mode"], RESEARCH_DETERMINISTIC)

    def test_deterministic_mode_reproduces(self):
        cfg = GenerationConfig(max_tokens=16, seed=7)
        a = self.llm.generate("The sky is", cfg)["text"]
        b = self.llm.generate("The sky is", cfg)["text"]
        self.assertEqual(a, b)

    def test_missing_model_structured_error(self):
        llm = LocalLLM(None)
        res = llm.generate("hello?")
        self.assertEqual(res["status"], "MODEL_UNAVAILABLE")
        self.assertIsNone(res["text"])


class TestScientistSafety(unittest.TestCase):
    def test_unknown_tool_rejected(self):
        llm = LocalLLM(None)
        loop = ScientistLoop(llm)
        res = loop.execute_tool_request({"tool": "run_shell", "params": {"cmd": "x"}})
        self.assertEqual(res["status"], "REJECTED")

    def test_shell_patterns_rejected(self):
        llm = LocalLLM(None)
        loop = ScientistLoop(llm)
        loop.register(ToolSpec("echo", "d", {}, lambda p: p))
        for bad in ("os.system", "subprocess", "rm -rf", "DROP TABLE"):
            res = loop.execute_tool_request({"tool": "echo", "params": {"x": f"a {bad} b"}})
            self.assertEqual(res["status"], "REJECTED")

    def test_malformed_request_rejected(self):
        llm = LocalLLM(None)
        loop = ScientistLoop(llm)
        self.assertEqual(loop.execute_tool_request("run it")["status"], "REJECTED")
        self.assertEqual(loop.execute_tool_request({"params": {}})["status"], "REJECTED")

    def test_unverified_claims_flagged(self):
        loop = ScientistLoop(LocalLLM(None))
        claims = loop.ground_claims("the neuron fired 999 times", {"spikes": 4})
        self.assertTrue(all(c["status"] == "UNVERIFIED" for c in claims))
        ok = loop.ground_claims("measured spikes=4 today", {"spikes": 4})
        self.assertIn("VERIFIED", [c["status"] for c in ok])

    def test_loop_without_model_is_honest(self):
        loop = ScientistLoop(LocalLLM(None))
        rec = loop.run_iteration("context", "statehash")
        self.assertIn("MODEL_UNAVAILABLE", rec.hypothesis)
        self.assertEqual(rec.tool_calls, [])


class TestCognitiveTrainerLLM(unittest.TestCase):
    def test_trainer_discovers_model_and_hypothesizes(self):
        from src.trainer.cognitive import CognitiveTrainer
        c = CognitiveTrainer()
        self.assertEqual(c.model_status, "OPERATIONAL")
        r = c.generate_curriculum_hypothesis({"step_count": 3, "drives": {}})
        self.assertEqual(r["status"], "SUCCESS")
        self.assertTrue((r["hypothesis"] or "").strip())
        self.assertIn("model_sha256", r)
        self.assertTrue(r.get("advisory_only"))

    def test_trainer_rule_based_labeling(self):
        from src.trainer.cognitive import CognitiveTrainer
        c = CognitiveTrainer()
        p = c.propose_curriculum_step({"energy": 0.5}, [{"name": "t"}])
        self.assertEqual(p.status, "RULE_BASED")

    def test_trainer_unavailable_path(self):
        from src.trainer.cognitive import CognitiveTrainer
        c = CognitiveTrainer(model_path="no/such/model.gguf")
        self.assertEqual(c.model_status, "MODEL_UNAVAILABLE")
        r = c.generate_curriculum_hypothesis({})
        self.assertEqual(r["status"], "MODEL_UNAVAILABLE")
        self.assertIsNone(r["hypothesis"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
