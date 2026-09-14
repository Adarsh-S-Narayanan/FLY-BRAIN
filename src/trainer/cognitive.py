import os
import sys
import json
import subprocess
from typing import Dict, Any, Optional

HOST_PYTHON = r"C:\Program Files\Python314\python.exe"
GGUF_MODEL_PATH = r"C:\Users\hcsme\.cache\huggingface\hub\models--unsloth--Qwen3-4B-GGUF\snapshots\22c9fc8a8c7700b76a1789366280a6a5a1ad1120\Qwen3-4B-Q4_K_M.gguf"

class CognitiveTrainer:
    """
    Local Cognitive Trainer backed by the verified Qwen 3.x 4B-class GGUF model.
    Acts strictly as a curriculum planner, teacher, semantic evaluator, and evolution advisor.
    Does NOT bypass or directly mutate the connectome brain state.
    """
    def __init__(self, model_path: str = GGUF_MODEL_PATH):
        self.model_path = model_path
        self.available = os.path.exists(model_path) and os.path.exists(HOST_PYTHON)
        if not self.available:
            print(f"[CognitiveTrainer] Warning: Model ({model_path}) or Host Python not found. Operating in deterministic rule-based trainer mode.")

    def evaluate_behavior(
        self,
        objective: str,
        observation: Dict[str, Any],
        action: str,
        expected_action: str
    ) -> Dict[str, Any]:
        """
        Evaluates the organism's behavior against a curriculum objective.
        Returns explicit trainer signals: objective, observation, expected outcome,
        actual outcome, reward, error, explanation, and recommended curriculum step.
        """
        success = (action == expected_action)
        reward = 1.0 if success else -0.5
        error = 0.0 if success else 1.0
        
        explanation = f"Organism selected action '{action}' for objective '{objective}'. "
        if success:
            explanation += f"Matches expected target '{expected_action}'. Reinforcing synaptic pathways."
            next_step = "advance_curriculum_difficulty"
        else:
            explanation += f"Expected '{expected_action}'. Applying corrective negative prediction error."
            next_step = "repeat_curriculum_step"

        return {
            "objective": objective,
            "observation": observation,
            "expected_outcome": expected_action,
            "actual_outcome": action,
            "reward": reward,
            "error": error,
            "explanation": explanation,
            "recommended_curriculum_step": next_step
        }

    def generate_curriculum_hypothesis(self, state_summary: Dict[str, Any]) -> str:
        """
        Queries the local Qwen3-4B GGUF model to generate pedagogical curriculum recommendations.
        """
        if not self.available:
            return "Curriculum recommendation: Continue sensory-motor association reinforcement."

        prompt = (
            f"You are a computational neuroscience teacher. Organism stats: "
            f"Step: {state_summary.get('step_count', 0)}, "
            f"Energy: {state_summary.get('drives', {}).get('energy', 1.0):.2f}, "
            f"Curiosity: {state_summary.get('drives', {}).get('curiosity', 0.8):.2f}. "
            f"Provide one concise pedagogical hypothesis for the next connectome learning trial:"
        )

        # Run inference via host python llama_cpp
        script = f"""
import llama_cpp
import sys
llm = llama_cpp.Llama(model_path=r'{self.model_path}', n_ctx=256, verbose=False)
res = llm('''{prompt}''', max_tokens=40, stop=['\\n'])
print(res['choices'][0]['text'].strip())
"""
        try:
            proc = subprocess.run(
                [HOST_PYTHON, "-c", script],
                capture_output=True,
                text=True,
                timeout=25
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception as e:
            print(f"[CognitiveTrainer] LLM inference notice: {e}")

        return "Pedagogical hypothesis: Reinforce bilateral cross-hemisphere coordination."
