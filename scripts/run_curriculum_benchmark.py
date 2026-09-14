import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from src.connectome.loader import get_or_create_circuit
from src.brain.runtime import BrainRuntime
from src.memory.persistence import PersistentMemoryManager
from src.trainer.curriculum import CurriculumTrainer

def run_benchmarks():
    print("=== Commencing Controlled Curriculum & Tool Learning Benchmarks ===")
    os.makedirs("diagnostics", exist_ok=True)
    db_path = "diagnostics/benchmark_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    memory = PersistentMemoryManager(db_path=db_path)
    circuit = get_or_create_circuit(512, cache_name="benchmark_circuit_512.npz")
    brain = BrainRuntime(circuit, use_gpu=True, seed=42)
    trainer = CurriculumTrainer(brain, memory)

    # 1. Train tool selection skill (Gate G015)
    learning_result = trainer.train_tool_selection_skill(target_tool="speak", num_trials=15)
    assert learning_result["plasticity_occurred"], "Plasticity failed to occur during training!"
    assert learning_result["score_improvement"] > 0, "Tool score failed to improve!"
    print(f"[PASS Gate G015] Measurable learning achieved: delta score = +{learning_result['score_improvement']:.4f}")

    # 2. Multi-step learned tool sequence (Gate G016)
    seq_result = trainer.execute_multi_step_tool_sequence()
    assert seq_result["all_success"], "Multi-step tool sequence failed!"
    print(f"[PASS Gate G016] Multi-step sequence completed successfully ({seq_result['steps_completed']} steps).")

    brain.cleanup()

    # Save benchmark report
    benchmark_report = {
        "gate_g015_tool_learning": learning_result,
        "gate_g016_tool_sequence": {
            "sequence_name": seq_result["sequence_name"],
            "steps_completed": seq_result["steps_completed"],
            "all_success": seq_result["all_success"]
        },
        "overall_status": "SUCCESS"
    }
    out_file = "diagnostics/benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    run_benchmarks()
