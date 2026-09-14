import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from src.memory.persistence import PersistentMemoryManager

def test_memory_restart_survival():
    print("=== Testing Memory Persistence Across Process Restart ===")
    test_db = "diagnostics/test_memory.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    # Session 1: write items into all persistent stores
    print("Session 1: Writing experiences, concepts, skills, and dreams...")
    mem1 = PersistentMemoryManager(db_path=test_db)
    
    ep_id = mem1.record_episode(
        step=42,
        observation={"visual_target": "red_flower", "audio_tone": 440},
        action="speak",
        reward=1.0,
        prediction_error=0.2,
        outcome={"speech_emitted": "flower detected"}
    )
    
    concept_vec = np.array([0.5, 0.2, 0.8, -0.1], dtype=np.float32)
    mem1.store_concept(
        concept="nectar_source",
        description="Floral visual cue associated with positive caloric reward",
        embedding=concept_vec,
        associations={"color": "red", "reward": 1.0}
    )
    
    mem1.record_skill("visual_foraging", ["observe_visual", "speak"], success=True)
    mem1.record_dream(
        seed=1234,
        base_episode_id=ep_id,
        simulated_action="generate_image",
        counterfactual_reward=0.9,
        insight="Visual association confirmed during replay"
    )
    
    # Simulate restart by deleting mem1 instance
    del mem1

    # Session 2: open new memory instance and retrieve
    print("Session 2: Spawning fresh memory instance and verifying persistence...")
    mem2 = PersistentMemoryManager(db_path=test_db)
    
    episodes = mem2.get_recent_episodes(5)
    assert len(episodes) == 1, f"Expected 1 episode, got {len(episodes)}"
    assert episodes[0]["step"] == 42
    assert episodes[0]["action"] == "speak"
    assert episodes[0]["reward"] == 1.0
    print("[PASS] Episodic memory survived restart.")

    semantic_res = mem2.query_semantic(np.array([0.5, 0.2, 0.7, -0.1], dtype=np.float32), top_k=1)
    assert len(semantic_res) == 1
    assert semantic_res[0]["concept"] == "nectar_source"
    assert semantic_res[0]["similarity"] > 0.95
    print("[PASS] Semantic associative memory survived restart.")

    skills = mem2.get_skills()
    assert len(skills) == 1
    assert skills[0]["skill_name"] == "visual_foraging"
    assert skills[0]["success_rate"] == 1.0
    print("[PASS] Skill memory survived restart.")

    dreams = mem2.get_recent_dreams(5)
    assert len(dreams) == 1
    assert dreams[0]["simulated_action"] == "generate_image"
    assert dreams[0]["counterfactual_reward"] == 0.9
    print("[PASS] Dream replay memory survived restart.")

    print("SUCCESS: All persistent memory categories verified across restart!")

if __name__ == "__main__":
    test_memory_restart_survival()
