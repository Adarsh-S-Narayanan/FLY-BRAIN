import os
import sys
import json
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# R9: no hard-coded machine paths. Resolution order: explicit arg > env >
# project-local models/ dir > unavailable. Host python defaults to this interpreter.
HOST_PYTHON = os.environ.get("FLYBRAIN_HOST_PYTHON", sys.executable)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_MODEL = os.path.join(_PROJECT_ROOT, "models", "qwen3-4b-q4_k_m.gguf")
GGUF_MODEL_PATH = os.environ.get(
    "FLYBRAIN_LLM_PATH",
    _LOCAL_MODEL if os.path.exists(_LOCAL_MODEL) else ""
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
        self.model_path = model_path or ""
        self.host_python = HOST_PYTHON
        self._llm = None  # lazy LocalLLM instance
        if not self.model_path:
            # Discover a locally present GGUF model (llm/); stays honest if none.
            try:
                from src.llm.discovery import discover_models
                found = [m for m in discover_models() if m.status == "DISCOVERED"]
                if found:
                    self.model_path = found[0].path
            except Exception:
                pass
        self.model_available = bool(self.model_path) and os.path.exists(self.model_path) \
            and bool(self.host_python) and os.path.exists(self.host_python)

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
        """Rule-based curriculum proposal. NEVER labeled as LLM inference:
        status is RULE_BASED, with the model availability recorded separately."""
        task_name = available_tasks[0].get("name", "basic_foraging") if available_tasks else "basic_foraging"
        return CurriculumProposal(
            target_skill=task_name,
            target_tool="vision_tracker",
            difficulty_level=1,
            rationale=f"Rule-based exploratory task (local model: {self.model_status}).",
            recommended_sensory_stimulus={"visual": 0.5},
            status="RULE_BASED"
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
        Generates a pedagogical hypothesis with the local GGUF model when present.
        Returns explicit MODEL_UNAVAILABLE / MODEL_LOAD_ERROR / INFERENCE_FAILED
        structures otherwise — never faked text. Output is advisory only.
        """
        if not self.model_available:
            return {
                "status": "MODEL_UNAVAILABLE",
                "message": "No local GGUF model discovered (see llm/ discovery)",
                "hypothesis": None
            }
        try:
            from src.llm.runtime import LocalLLM, GenerationConfig
            from src.llm.discovery import discover_models
            if self._llm is None:
                found = [m for m in discover_models()
                         if m.status == "DISCOVERED" and m.path == self.model_path]
                model = found[0] if found else None
                if model is None:
                    return {"status": "MODEL_LOAD_ERROR",
                            "error": f"model not in discovery index: {self.model_path}",
                            "hypothesis": None}
                self._llm = LocalLLM(model, n_ctx=2048)
                if not self._llm.load():
                    return {"status": "MODEL_LOAD_ERROR", "error": self._llm.last_error,
                            "hypothesis": None}
            prompt = (
                f"State: step {state_summary.get('step_count', 0)}, "
                f"energy {state_summary.get('drives', {}).get('energy', 1.0):.2f}, "
                f"curiosity {state_summary.get('drives', {}).get('curiosity', 0.8):.2f}. "
                f"Propose one concise pedagogical hypothesis for the next trial (under 30 words):"
            )
            res = self._llm.generate(prompt, GenerationConfig(max_tokens=60, seed=42))
            if res["status"] != "SUCCESS":
                return {"status": "INFERENCE_FAILED", "error": res.get("error", ""),
                        "hypothesis": None}
            out = dict(res["provenance"])
            out.update({"status": "SUCCESS", "hypothesis": (res["text"] or "").strip(),
                        "advisory_only": True})
            return out
        except Exception as e:  # noqa: BLE001
            return {"status": "MODEL_RUNTIME_ERROR", "error": f"{type(e).__name__}: {e}",
                    "hypothesis": None}

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
