"""STAGE K tests: LLM control plane (schema validation, safety, execution),
research memory (hash chain, queries). The LLM can command research but never
mutate source, touch the shell, or fabricate results."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import tempfile
import unittest

from src.llm.control import (CommandEnvelope, ControlPlane, ResearchRuntime,
                             validate_envelope, COMMAND_SCHEMA_VERSION)
from src.llm.research_memory import ResearchMemory


class TestValidation(unittest.TestCase):
    def test_valid_commands_pass(self):
        ok, err = validate_envelope(CommandEnvelope("SPAWN_POPULATION", {"size": 4}))
        self.assertTrue(ok, err)
        ok, err = validate_envelope(CommandEnvelope(
            "PROPOSE_HYPOTHESIS", {"text": "growth correlates with navigation",
                                   "based_on_experiments": ["exp-baseline-1"]}))
        self.assertTrue(ok, err)

    def test_unknown_command_and_schema_version(self):
        ok, err = validate_envelope(CommandEnvelope("SELF_DESTRUCT", {}))
        self.assertFalse(ok)
        ok, err = validate_envelope(CommandEnvelope("START_RUN", {},
                                                    schema_version="control_v0"))
        self.assertFalse(ok)

    def test_missing_and_unknown_params_rejected(self):
        ok, err = validate_envelope(CommandEnvelope("SPAWN_POPULATION", {}))
        self.assertFalse(ok and "size" not in err)
        ok, err = validate_envelope(CommandEnvelope("SPAWN_POPULATION",
                                                    {"size": 4, "evil": 1}))
        self.assertFalse(ok)

    def test_range_and_type_constraints(self):
        ok, err = validate_envelope(CommandEnvelope("SPAWN_POPULATION", {"size": 999}))
        self.assertFalse(ok)
        ok, err = validate_envelope(CommandEnvelope("SPAWN_POPULATION", {"size": True}))
        self.assertFalse(ok)
        ok, err = validate_envelope(CommandEnvelope("SAVE_CHECKPOINT", {"name": ""}))
        self.assertFalse(ok)

    def test_non_envelope_rejected(self):
        ok, err = validate_envelope({"command": "START_RUN"})
        self.assertFalse(ok)

    def test_shell_and_code_injection_rejected(self):
        attacks = [
            {"name": "cp; import os; os.system('rm -rf /')"},
            {"name": "checkpoint' subprocess.call('x')"},
            {"name": "eval('1+1')"},
            {"name": "__import__('os')"},
            {"name": "powershell -c Get-Process"},
        ]
        for params in attacks:
            ok, err = validate_envelope(CommandEnvelope("SAVE_CHECKPOINT", params))
            self.assertFalse(ok, f"attack not rejected: {params}")


class TestExecution(unittest.TestCase):
    def setUp(self):
        self.cp = ControlPlane(ResearchRuntime(experiment_seed=9))

    def _env(self, cmd, params):
        return CommandEnvelope(cmd, params, requested_by="llm-scientist")

    def test_full_research_workflow(self):
        r = self.cp.execute(self._env("SPAWN_POPULATION", {"size": 3}))
        self.assertEqual(r["status"], "EXECUTED")
        r = self.cp.execute(self._env("START_RUN", {"ticks": 8}))
        self.assertEqual(r["status"], "EXECUTED")
        r = self.cp.execute(self._env("SAVE_CHECKPOINT", {"name": "ck1"}))
        self.assertEqual(r["status"], "EXECUTED")
        r = self.cp.execute(self._env("STOP_RUN", {}))
        self.assertEqual(r["status"], "EXECUTED")
        r = self.cp.execute(self._env("LOAD_CHECKPOINT", {"name": "ck1"}))
        self.assertEqual(r["status"], "EXECUTED")

    def test_experiment_and_comparison(self):
        self.cp.execute(self._env("REQUEST_EXPERIMENT",
                                  {"experiment_type": "baseline", "seed": 5, "ticks": 6}))
        self.cp.execute(self._env("REQUEST_EXPERIMENT",
                                  {"experiment_type": "ablation_no_teaching",
                                   "seed": 5, "ticks": 6}))
        r = self.cp.execute(self._env("REQUEST_COMPARISON",
                                      {"experiment_a": "exp-baseline-5",
                                       "experiment_b": "exp-ablation_no_teaching-5"}))
        self.assertEqual(r["status"], "EXECUTED")
        self.assertIn("hash_equal", r["comparison"])
        # unknown ids -> honest failure
        r = self.cp.execute(self._env("REQUEST_COMPARISON",
                                      {"experiment_a": "nope", "experiment_b": "nada"}))
        self.assertEqual(r["status"], "FAILED")

    def test_proposals_are_records_not_executions(self):
        r = self.cp.execute(self._env("PROPOSE_HYPOTHESIS",
                                      {"text": "structural growth follows energy abundance",
                                       "based_on_experiments": []}))
        self.assertEqual(r["status"], "EXECUTED")
        self.assertEqual(r["hypothesis"]["status"], "HYPOTHESIS")
        r2 = self.cp.execute(self._env("PROPOSE_CURRICULUM",
                                       {"stages": ["forage", "communicate", "coordinate"]}))
        self.assertEqual(r2["status"], "EXECUTED")
        self.assertEqual(len(self.cp.runtime.curricula), 1)

    def test_every_execution_is_logged(self):
        n0 = len(self.cp.runtime.execution_log)
        self.cp.execute(self._env("SPAWN_POPULATION", {"size": 2}))
        self.cp.execute(self._env("START_RUN", {"ticks": 2}))
        self.cp.execute(CommandEnvelope("SPAWN_POPULATION", {"size": 0}))  # rejected
        self.assertEqual(len(self.cp.runtime.execution_log), n0 + 3)
        self.assertEqual(self.cp.runtime.execution_log[-1]["status"], "REJECTED")

    def test_rejection_does_not_execute(self):
        r = self.cp.execute(self._env("SPAWN_POPULATION", {"size": 2**40}))
        self.assertEqual(r["status"], "REJECTED")
        self.assertIsNone(self.cp.runtime.population)

    def test_hypothesis_references_validated(self):
        self.cp.execute(self._env("REQUEST_EXPERIMENT",
                                  {"experiment_type": "baseline", "seed": 3}))
        r = self.cp.execute(self._env("PROPOSE_HYPOTHESIS",
                                      {"text": "h", "based_on_experiments":
                                       ["exp-baseline-3", "ghost-exp"]}))
        self.assertEqual(r["hypothesis"]["based_on_experiments"], ["exp-baseline-3"])
        self.assertEqual(r["hypothesis"]["unknown_references"], ["ghost-exp"])


class TestResearchMemory(unittest.TestCase):
    def test_append_and_chain_verification(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rm.jsonl")
            mem = ResearchMemory(path)
            mem.append("experiment", {"experiment_id": "e1", "status": "EXECUTED"})
            mem.append("hypothesis", {"hypothesis_id": "h1", "status": "HYPOTHESIS"})
            mem.append("experiment", {"experiment_id": "e2", "status": "FAILED"})
            self.assertTrue(mem.verify_chain())
            # reload from disk -> chain still valid
            mem2 = ResearchMemory(path)
            self.assertTrue(mem2.verify_chain())
            self.assertEqual(len(mem2.records), 3)

    def test_tamper_detection(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rm.jsonl")
            mem = ResearchMemory(path)
            mem.append("experiment", {"result": 1})
            mem.append("experiment", {"result": 2})
            # tamper with the file
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            tampered = lines[0].replace('"result": 1', '"result": 999')
            with open(path, "w", encoding="utf-8") as f:
                f.write(tampered + lines[1])
            mem2 = ResearchMemory(path)
            self.assertFalse(mem2.verify_chain())

    def test_scientist_queries(self):
        with tempfile.TemporaryDirectory() as td:
            mem = ResearchMemory(os.path.join(td, "rm.jsonl"))
            mem.append("experiment", {"experiment_id": "e1", "status": "EXECUTED"})
            mem.append("experiment", {"experiment_id": "e2", "status": "FAILED"})
            mem.append("hypothesis", {"hypothesis_id": "h1", "status": "HYPOTHESIS"})
            self.assertEqual(len(mem.experiments()), 2)
            self.assertEqual(len(mem.failed_experiments()), 1)
            self.assertEqual(len(mem.successful_protocols()), 1)
            self.assertEqual(len(mem.hypotheses()), 1)
            s = mem.summary()
            self.assertTrue(s["chain_valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
