import os
import sys
import argparse
import json
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.connectome.loader import get_or_create_circuit, load_raw_neurons
from src.brain.runtime import BrainRuntime
from src.memory.persistence import PersistentMemoryManager
from src.trainer.curriculum import CurriculumTrainer
from src.evolution.scheduler import EvolutionScheduler
from src.dream.engine import DreamEngine
from src.compute.validator import run_cpu_gpu_validation

def run_diagnostics():
    from scripts.detect_env import main as detect_main
    from scripts.generate_manifests import main as manifest_main
    from scripts.validate_malecns_data import main as malecns_main
    
    print("Collecting system environment diagnostics...")
    detect_main()
    print("Validating biological malecns connectome source...")
    malecns_main()
    print("Generating dependency and reproducibility manifests...")
    manifest_main()
    print("Running Vulkan GPU vs CPU reference validation...")
    run_cpu_gpu_validation()
    print("\nAll diagnostics and validation manifests generated.")

def run_app(host="127.0.0.1", port=8080):
    import uvicorn
    from src.ui.server import app
    print(f"Starting FlyBrain Organism Dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

def main():
    parser = argparse.ArgumentParser(description="FlyBrain - Autonomous Biological Connectome Framework")
    parser.add_argument("--mode", choices=["run", "diagnostics", "validate-vulkan", "benchmark", "evolve", "test"], default="run",
                        help="Execution mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host address for UI server")
    parser.add_argument("--port", type=int, default=8080, help="Port for UI server")
    parser.add_argument("--circuit-size", type=int, default=512, help="Number of neurons in connectome circuit")
    parser.add_argument("--generations", type=int, default=3, help="Evolution generations")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed")
    args = parser.parse_args()

    if args.mode == "diagnostics":
        run_diagnostics()
    elif args.mode == "validate-vulkan":
        run_cpu_gpu_validation()
    elif args.mode == "benchmark":
        import scripts.run_curriculum_benchmark as bm
        bm.run_benchmarks()
    elif args.mode == "evolve":
        circuit = get_or_create_circuit(args.circuit_size)
        evo = EvolutionScheduler(circuit)
        for g in range(args.generations):
            evo.run_generation(num_candidates=4, seed=args.seed + g*10)
    elif args.mode == "test":
        import tests.test_suite as ts
        import unittest
        suite = unittest.TestLoader().loadTestsFromTestCase(ts.TestFlyBrainSystem)
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        if not res.wasSuccessful():
            sys.exit(1)
    elif args.mode == "run":
        run_app(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
