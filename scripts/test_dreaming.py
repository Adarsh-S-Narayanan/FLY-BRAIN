import os
import sys
sys.path.insert(0, os.path.abspath("."))
from src.connectome.loader import get_or_create_circuit
from src.brain.runtime import BrainRuntime
from src.memory.persistence import PersistentMemoryManager
from src.dream.engine import DreamEngine

def test_dream_pipeline():
    print("=== Testing Experience Replay & Dream System ===")
    test_db = "diagnostics/test_dream_memory.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    memory = PersistentMemoryManager(db_path=test_db)
    
    # 1. Record authentic waking experiences
    print("Recording real-world waking episodes...")
    for i in range(3):
        memory.record_episode(
            step=i + 1,
            observation={"target": f"stimulus_{i}"},
            action=["observe_visual", "speak", "act_in_environment"][i],
            reward=0.6 * (i + 1),
            prediction_error=0.1,
            outcome={"success": True}
        )

    circuit = get_or_create_circuit(256, cache_name="test_dream_circuit_256.npz")
    brain = BrainRuntime(circuit, use_gpu=False, seed=42)
    dream_engine = DreamEngine(brain, memory)

    # 2. Run deterministic dream cycle
    print("\nRunning deterministic dream cycle (Seed 42)...")
    dreams_det = dream_engine.run_dream_cycle(mode="deterministic", seed=42, num_episodes_to_replay=2)
    assert len(dreams_det) == 2, f"Expected 2 dreams, got {len(dreams_det)}"
    print(f"Deterministic Dream 1: {dreams_det[0]['insight']}")
    print(f"Deterministic Dream 2: {dreams_det[1]['insight']}")

    # 3. Run exploratory dream cycle
    print("\nRunning exploratory dream cycle (Seed 99)...")
    dreams_exp = dream_engine.run_dream_cycle(mode="exploratory", seed=99, num_episodes_to_replay=2)
    assert len(dreams_exp) == 2
    print(f"Exploratory Dream 1: {dreams_exp[0]['insight']}")

    # 4. Verify separate dream persistence
    all_dreams = memory.get_recent_dreams(10)
    assert len(all_dreams) == 4, f"Expected 4 persisted dreams, got {len(all_dreams)}"
    print(f"\nPersisted dreams verified: {len(all_dreams)} separate records in dream store.")
    print("SUCCESS: Dream and experience replay pipeline verified!")

if __name__ == "__main__":
    test_dream_pipeline()
