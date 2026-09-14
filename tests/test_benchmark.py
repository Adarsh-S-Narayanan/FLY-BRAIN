"""STAGE N tests: benchmark suite measures real behavior, documents budgets,
never claims superiority, skips LLM arms honestly without a model."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from src.research.benchmark import (run_flybrain_arm, run_llm_only_arm,
                                    run_llm_tools_arm, run_benchmark_suite,
                                    CATEGORIES, BENCHMARK_VERSION)


class TestFlyBrainArm(unittest.TestCase):
    def test_categories_measured_with_budgets(self):
        res = run_flybrain_arm(seed=111, train_ticks=25, delay_ticks=15)
        self.assertEqual(res["arm"], "flybrain")
        self.assertEqual(set(res["categories"]), set(CATEGORIES))
        for cat, data in res["categories"].items():
            self.assertEqual(data["status"], "MEASURED", cat)
        self.assertGreater(res["budget"]["compute_ticks"], 0)
        self.assertEqual(res["budget"]["training_exposure_ticks"], 25)
        self.assertIn("population_hash", res)

    def test_training_improves_or_preserves_skill(self):
        res = run_flybrain_arm(seed=112, train_ticks=40, delay_ticks=10)
        cat = res["categories"]["memory_retention"]
        self.assertGreaterEqual(cat["skill_after_training"], cat["skill_before"] - 0.05,
                                "training must not destroy skill")

    def test_deterministic_reruns_match(self):
        a = run_flybrain_arm(seed=113, train_ticks=20, delay_ticks=10)
        b = run_flybrain_arm(seed=113, train_ticks=20, delay_ticks=10)
        self.assertEqual(a["population_hash"], b["population_hash"])


class TestLLMArms(unittest.TestCase):
    def test_llm_arms_skip_or_measure_honestly(self):
        for arm_fn, name in ((run_llm_only_arm, "llm_only"),
                             (run_llm_tools_arm, "llm_tools")):
            res = arm_fn(seed=111)
            self.assertIn(res["status"], ("MEASURED", "SKIP"), name)
            if res["status"] == "SKIP":
                self.assertTrue(res.get("reason"), "SKIP requires concrete reason")

    def test_llm_only_embodied_categories_not_applicable(self):
        res = run_llm_only_arm(seed=111)
        if res["status"] != "MEASURED":
            self.skipTest(res.get("reason", "model unavailable"))
        for cat in CATEGORIES:
            self.assertEqual(res["categories"][cat]["status"], "NOT_APPLICABLE")

    def test_suite_report_has_no_superiority_claim(self):
        report = run_benchmark_suite(seed=111)
        self.assertEqual(report["benchmark_version"], BENCHMARK_VERSION)
        self.assertEqual(len(report["arms"]), 3)
        blob = str(report).lower()
        for claim in ("flybrain wins", "superior to", "beats gpt", "outperforms all"):
            self.assertNotIn(claim, blob, "no aggregate superiority claims")
        self.assertIn("fairness_note", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
