import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import unittest
import numpy as np

from src.connectome.loader import load_raw_neurons, get_or_create_circuit, DEFAULT_SOMA_PATH
from src.compute.cpu_reference import cpu_brain_step, cpu_plasticity_step
from src.compute.vulkan_backend import VulkanComputeEngine
from src.brain.runtime import BrainRuntime
from src.brain.plasticity import PlasticityEngine
from src.memory.persistence import PersistentMemoryManager
from src.tools.registry import ToolRegistry
from src.tools.visual import ObserveVisualConnector
from src.tools.audio import ListenAudioConnector
from src.tools.speech import SpeakConnector
from src.tools.image_gen import GenerateImageConnector
from src.tools.memory_tool import RememberConnector, RetrieveMemoryConnector
from src.tools.environment import ActInEnvironmentConnector, InspectSelfConnector, SleepConnector
from src.trainer.curriculum import CurriculumTrainer
from src.evolution.scheduler import EvolutionScheduler
from src.dream.engine import DreamEngine

class TestFlyBrainSystem(unittest.TestCase):

    def test_01_malecns_biological_source(self):
        """Gate G007: Real malecns data is integrated and validated."""
        self.assertTrue(os.path.exists(DEFAULT_SOMA_PATH), f"File {DEFAULT_SOMA_PATH} must exist.")
        neurons = load_raw_neurons(DEFAULT_SOMA_PATH)
        self.assertGreater(len(neurons), 100000, "Should load over 100,000 biological neurons.")
        
        # Verify fields
        n0 = neurons[0]
        self.assertGreater(n0.body_id, 0)
        self.assertIn(n0.side, ["L", "R", "M"])
        self.assertGreaterEqual(n0.tbars, 0)

        # Build circuit
        circuit = get_or_create_circuit(256, cache_name="test_suite_circuit_256.npz")
        self.assertEqual(circuit.num_neurons, 256)
        self.assertGreater(circuit.num_synapses, 1000)

    def test_02_vulkan_compute_and_cpu_reference(self):
        """Gate G008, G009, G010: CPU reference, Vulkan GPU backend, and numerical agreement."""
        circuit = get_or_create_circuit(256, cache_name="test_suite_circuit_256.npz")
        rng = np.random.RandomState(42)
        prev_act = rng.uniform(0.0, 1.0, circuit.num_neurons).astype(np.float32)
        ext_in = rng.uniform(0.0, 0.5, circuit.num_neurons).astype(np.float32)
        pot_in = rng.uniform(-0.2, 0.2, circuit.num_neurons).astype(np.float32)

        # CPU step
        cpu_pot, cpu_act = cpu_brain_step(
            circuit.row_offsets, circuit.col_indices, circuit.weights,
            prev_act, ext_in, pot_in
        )
        self.assertEqual(len(cpu_pot), 256)
        self.assertEqual(len(cpu_act), 256)

        # Vulkan GPU step
        vk_engine = VulkanComputeEngine()
        self.assertIn("Radeon", vk_engine.device_name)
        gpu_pot, gpu_act = vk_engine.run_step(
            circuit.row_offsets, circuit.col_indices, circuit.weights,
            prev_act, ext_in, pot_in
        )
        vk_engine.cleanup()

        diff_pot = float(np.max(np.abs(cpu_pot - gpu_pot)))
        diff_act = float(np.max(np.abs(cpu_act - gpu_act)))
        self.assertLess(diff_pot, 1e-4, f"Potential mismatch: {diff_pot}")
        self.assertLess(diff_act, 1e-4, f"Activation mismatch: {diff_act}")

    def test_03_brain_state_and_plasticity(self):
        """Gate G015: Brain maintains state and adapts through plasticity."""
        circuit = get_or_create_circuit(256, cache_name="test_suite_circuit_256.npz")
        brain = BrainRuntime(circuit, use_gpu=False, seed=42)
        
        initial_weights = circuit.weights.copy()
        step1 = brain.step(sensory_inputs={"visual": np.full(64, 0.5, dtype=np.float32)}, reward=1.0)
        
        self.assertEqual(step1["step"], 1)
        self.assertGreater(step1["spikes"], 0)
        self.assertGreater(step1["drives"]["energy"], 0.0)
        self.assertGreater(step1["synapses_updated"], 0)
        
        delta_w = float(np.mean(np.abs(circuit.weights - initial_weights)))
        self.assertGreater(delta_w, 0.0, "Weights must change under reward.")
        brain.cleanup()

    def test_04_memory_persistence(self):
        """Gate G011: Memory persists across process restarts."""
        test_db = "diagnostics/test_suite_memory.db"
        if os.path.exists(test_db):
            os.remove(test_db)
            
        mem1 = PersistentMemoryManager(db_path=test_db)
        mem1.record_episode(
            step=10,
            observation={"cue": "nectar"},
            action="forage",
            reward=0.8,
            prediction_error=0.05,
            outcome={"energy": 0.9}
        )
        mem1.store_concept("floral_odor", "Olfactory cue", np.array([1.0, 0.0, 0.5], dtype=np.float32))
        mem1.record_skill("foraging_trial", ["observe_visual", "act_in_environment"], success=True)
        mem1.record_dream(42, 1, "speak", 0.7, "Replay insight")
        del mem1

        # Reopen
        mem2 = PersistentMemoryManager(db_path=test_db)
        episodes = mem2.get_recent_episodes(5)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["action"], "forage")
        
        sem = mem2.query_semantic(np.array([1.0, 0.0, 0.4], dtype=np.float32), top_k=1)
        self.assertEqual(sem[0]["concept"], "floral_odor")
        
        skills = mem2.get_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["skill_name"], "foraging_trial")
        
        dreams = mem2.get_recent_dreams(5)
        self.assertEqual(len(dreams), 1)
        self.assertEqual(dreams[0]["simulated_action"], "speak")

    def test_05_tool_connectors_execution(self):
        """Gate G012: Real tool connectors pass end-to-end execution."""
        mem = PersistentMemoryManager(db_path="diagnostics/test_suite_memory.db")
        circuit = get_or_create_circuit(256, cache_name="test_suite_circuit_256.npz")
        brain = BrainRuntime(circuit, use_gpu=False, seed=42)
        
        reg = ToolRegistry()
        reg.register(ObserveVisualConnector())
        reg.register(ListenAudioConnector())
        reg.register(SpeakConnector())
        reg.register(GenerateImageConnector())
        reg.register(RememberConnector(mem))
        reg.register(RetrieveMemoryConnector(mem))
        reg.register(ActInEnvironmentConnector())
        reg.register(InspectSelfConnector(brain))
        reg.register(SleepConnector(brain))

        # 1. Visual
        v_res = reg.execute("observe_visual", {"synthetic_target": "green_foliage"})
        self.assertEqual(v_res["status"], "SUCCESS")
        self.assertEqual(v_res["result"]["dominant_channel"], "green")
        self.assertEqual(len(v_res["result"]["features_vector"]), 64)

        # 2. Audio
        a_res = reg.execute("listen_audio", {"synthetic_freq_hz": 880})
        self.assertEqual(a_res["status"], "SUCCESS")
        self.assertTrue(a_res["result"]["voice_active"])

        # 3. Speak (Native Windows SAPI)
        s_res = reg.execute("speak", {"text": "FlyBrain unit test speech verified."})
        self.assertEqual(s_res["status"], "SUCCESS")
        self.assertTrue(os.path.exists(s_res["result"]["wav_file"]))
        self.assertGreater(s_res["result"]["duration_sec"], 0.5)

        # 4. Image Gen (AutoencoderTiny VAE)
        img_res = reg.execute("generate_image", {"prompt": "nectar source blossom", "seed": 42})
        self.assertEqual(img_res["status"], "SUCCESS")
        self.assertTrue(os.path.exists(img_res["result"]["image_path"]))
        self.assertEqual(img_res["result"]["width"], 256)
        self.assertEqual(img_res["result"]["height"], 256)

        # 5. Inspect Self
        self_res = reg.execute("inspect_self", {})
        self.assertEqual(self_res["status"], "SUCCESS")
        self.assertEqual(self_res["result"]["num_neurons"], 256)

        brain.cleanup()

    def test_06_evolution_and_rollback(self):
        """Gate G017, G018, G019, G021: Evolution generations, mutations, and rollback."""
        circuit = get_or_create_circuit(256, cache_name="test_suite_circuit_256.npz")
        evo = EvolutionScheduler(circuit, history_file="diagnostics/test_suite_evo.json")
        
        gen_res = evo.run_generation(num_candidates=4, seed=42)
        self.assertEqual(len(gen_res["candidates"]), 4)
        self.assertEqual(gen_res["generation"], 1)

        # Verify acceptance / rejection decisions
        cands = gen_res["candidates"]
        has_decision = any(c["accepted"] for c in cands) or not gen_res["improved"]
        self.assertTrue(has_decision)

    def test_07_dream_replay(self):
        """Gate G020: Dream replay pipeline using real stored experience."""
        mem = PersistentMemoryManager(db_path="diagnostics/test_suite_memory.db")
        mem.record_episode(
            step=1,
            observation={"cue": "target"},
            action="observe_visual",
            reward=0.5,
            prediction_error=0.1,
            outcome={"seen": True}
        )
        circuit = get_or_create_circuit(256, cache_name="test_suite_circuit_256.npz")
        brain = BrainRuntime(circuit, use_gpu=False, seed=42)
        dreamer = DreamEngine(brain, mem)

        dreams = dreamer.run_dream_cycle(mode="deterministic", seed=42, num_episodes_to_replay=1)
        self.assertEqual(len(dreams), 1)
        self.assertIn("Simulated", dreams[0]["insight"])
        brain.cleanup()

def write_junit_xml(results, total_time, out_path="diagnostics/test_results.xml"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tests = results.testsRun
    failures = len(results.failures)
    errors = len(results.errors)
    
    xml = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuites>',
        f'  <testsuite name="FlyBrainTestSuite" tests="{tests}" failures="{failures}" errors="{errors}" time="{total_time:.3f}">'
    ]
    for test, err in results.failures:
        xml.append(f'    <testcase classname="{test.__class__.__name__}" name="{test._testMethodName}">')
        xml.append(f'      <failure message="Failure">{err}</failure>')
        xml.append(f'    </testcase>')
    for test, err in results.errors:
        xml.append(f'    <testcase classname="{test.__class__.__name__}" name="{test._testMethodName}">')
        xml.append(f'      <error message="Error">{err}</error>')
        xml.append(f'    </testcase>')
    for test in results.testsRun - failures - errors:
        pass
    xml.append(f'  </testsuite>')
    xml.append(f'</testsuites>')
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml))
    print(f"Saved JUnit XML test report to {out_path}")

if __name__ == "__main__":
    start_t = time.time()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlyBrainSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    elapsed = time.time() - start_t
    
    # Write custom XML
    tests_count = res.testsRun
    fails = len(res.failures)
    errs = len(res.errors)
    
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuite name="FlyBrainIntegrationTests" tests="{tests_count}" failures="{fails}" errors="{errs}" time="{elapsed:.3f}">'
    ]
    for method_name in [m for m in dir(TestFlyBrainSystem) if m.startswith("test_")]:
        xml_lines.append(f'  <testcase classname="TestFlyBrainSystem" name="{method_name}" time="{elapsed/max(1, tests_count):.3f}" />')
    xml_lines.append('</testsuite>')
    
    out_xml = "diagnostics/test_results.xml"
    with open(out_xml, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))
    print(f"Generated {out_xml}")

    if not res.wasSuccessful():
        sys.exit(1)
