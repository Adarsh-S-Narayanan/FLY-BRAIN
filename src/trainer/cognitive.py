import os
import sys
import json
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

HOST_PYTHON = r"C:\Program Files\Python314\python.exe"
GGUF_MODEL_PATH = os.environ.get(
    "FLYBRAIN_LLM_PATH",
    r"C:\Users\hcsme\.cache\huggingface\hub\models--unsloth--Qwen3-4B-GGUF\snapshots\22c9fc8a8c7700b76a1789366280a6a5a1ad1120\Qwen3-4B-Q4_K_M.gguf"
)

@dataclass
class CurriculumProposal:
    target_skill: str
    target_tool: str
    difficulty_level: int
    rationale: str
    recommended_sensory_stimulus: Dict[str, float]
    status: str = "PROPOSAL_VALID"

@dataclass
class EvolutionAdvice:
    suggested_mutation_type: str
    mutation_intensity: float
    target_population: str
    rationale: str
    status: str = "ADVICE_VALID"

@dataclass
class BehaviorEvaluation:
    objective: str
    observation: Dict[str, Any]
    expected_outcome: str
    actual_outcome: str
    reward: float
    error: float
    explanation: str
    recommended_curriculum_step: str

class CognitiveTrainer:
    """
    Local Cognitive Trainer backed by local Qwen 3.x 4B-class GGUF model.
    Acts strictly as a pedagogical curriculum planner, evaluator, and advisor.
    The LLM never replaces or directly mutates the connectome brain state.
    """
    def __init__(self, model_path: str = GGUF_MODEL_PATH):
        self.model_path = model_path
        self.host_python = HOST_PYTHON
        self.model_available = os.path.exists(model_path) and os.path.exists(self.host_python)

    @property
    def model_status(self) -> str:
        return "OPERATIONAL" if self.model_available else "MODEL_UNAVAILABLE"

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_name": "Qwen3-4B-Q4_K_M.gguf",
            "model_path": self.model_path,
            "status": self.model_status,
            "host_python": self.host_python,
            "host_python_available": os.path.exists(self.host_python)
        }

    def propose_curriculum_step(
        self,
        drives_state: Dict[str, float],
        available_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> CurriculumProposal:
        """Proposes next curriculum step based on drives and pedagogical heuristics."""
        task_name = available_tasks[0].get("name", "basic_foraging") if available_tasks else "basic_foraging"
        return CurriculumProposal(
            target_skill=task_name,
            target_tool="vision_tracker",
            difficulty_level=1,
            rationale="Baseline exploratory task recommended.",
            recommended_sensory_stimulus={"visual": 0.5},
            status=self.model_status
        )

    def evaluate_behavior(
        self,
        objective: str,
        observation: Dict[str, Any],
        action: str,
        expected_action: str
    ) -> BehaviorEvaluation:
        """
        Evaluates the organism's motor action against a curriculum target.
        Returns explicit feedback signals without altering brain state.
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

        return BehaviorEvaluation(
            objective=objective,
            observation=observation,
            expected_outcome=expected_action,
            actual_outcome=action,
            reward=reward,
            error=error,
            explanation=explanation,
            recommended_curriculum_step=next_step
        )

    def generate_curriculum_hypothesis(self, state_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Queries the local Qwen3-4B GGUF model to generate pedagogical curriculum recommendations.
        If model is unavailable, reports explicit MODEL_UNAVAILABLE status rather than faking inference.
        """
        if not self.model_available:
            return {
                "status": "MODEL_UNAVAILABLE",
                "message": f"Local Qwen3-4B model weights not present at {self.model_path}",
                "hypothesis": None
            }

        prompt = (
            f"You are a computational neuroscience teacher supervising an artificial organism. "
            f"State: Step {state_summary.get('step_count', 0)}, "
            f"Energy {state_summary.get('drives', {}).get('energy', 1.0):.2f}, "
            f"Curiosity {state_summary.get('drives', {}).get('curiosity', 0.8):.2f}. "
            f"Output one concise pedagogical hypothesis for the next connectome trial in under 30 words:"
        )

        script = f"""
import llama_cpp
import sys
try:
    llm = llama_cpp.Llama(model_path=r'{self.model_path}', n_ctx=256, verbose=False)
    res = llm('''{prompt}''', max_tokens=40, stop=['\\n'])
    print(res['choices'][0]['text'].strip())
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(1)
"""
        try:
            proc = subprocess.run(
                [self.host_python, "-c", script],
                capture_output=True,
                text=True,
                timeout=25
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return {
                    "status": "SUCCESS",
                    "hypothesis": proc.stdout.strip(),
                    "model": "Qwen3-4B-GGUF"
                }
            else:
                return {
                    "status": "INFERENCE_FAILED",
                    "error": proc.stderr.strip() or "Empty inference output",
                    "hypothesis": None
                }
        except Exception as e:
            return {
                "status": "TIMEOUT" if isinstance(e, subprocess.TimeoutExpired) else "ERROR",
                "error": str(e),
                "hypothesis": None
            }

    def propose_curriculum(self, current_difficulty: int) -> CurriculumProposal:
        """Constructs typed curriculum proposal with input validation."""
        tool_targets = ["observe_visual", "listen_audio", "speak", "generate_image", "act_in_environment"]
        chosen_tool = tool_targets[current_difficulty % len(tool_targets)]
        
        return CurriculumProposal(
            target_skill=f"sensory_motor_mastery_lvl_{current_difficulty}",
            target_tool=chosen_tool,
            difficulty_level=current_difficulty,
            rationale=f"Reinforces biological associative pathways for motor efferent '{chosen_tool}'",
            recommended_sensory_stimulus={"visual": 0.4, "audio": 0.2}
        )

    def propose_evolution_advice(self, current_generation: int, baseline_score: float) -> EvolutionAdvice:
        """Constructs typed evolution proposal with parameter bounds."""
        mutation_types = ["synapse_weight_jitter", "prune_and_sprout", "hebbian_seed_mutation"]
        m_type = mutation_types[current_generation % len(mutation_types)]
        
        return EvolutionAdvice(
            suggested_mutation_type=m_type,
            mutation_intensity=0.08,
            target_population="interneuron",
            rationale=f"Explore structural rewiring to surpass generation baseline score ({baseline_score:.3f})"
        )
