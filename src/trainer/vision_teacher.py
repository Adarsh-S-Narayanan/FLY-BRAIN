import os
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

class VisionTeacher:
    """
    Local Vision Teacher evaluating visual observations and behaviors.
    Produces structured observations and explicit training feedback without bypassing the brain.
    """
    def __init__(self):
        pass

    def evaluate_visual_observation(
        self,
        image_path: str,
        target_concept: str = "red_flower"
    ) -> Dict[str, Any]:
        """
        Inspects an image, extracts visual features, and evaluates semantic match against target concept.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB").resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0

        mean_r = float(arr[:, :, 0].mean())
        mean_g = float(arr[:, :, 1].mean())
        mean_b = float(arr[:, :, 2].mean())
        contrast = float(arr.std())

        # Determine target concept match
        match_score = 0.0
        if "red" in target_concept:
            match_score = max(0.0, min(1.0, (mean_r - max(mean_g, mean_b)) * 2.0 + 0.5))
        elif "green" in target_concept:
            match_score = max(0.0, min(1.0, (mean_g - max(mean_r, mean_b)) * 2.0 + 0.5))
        elif "blue" in target_concept:
            match_score = max(0.0, min(1.0, (mean_b - max(mean_r, mean_g)) * 2.0 + 0.5))
        else:
            match_score = min(1.0, contrast * 2.0)

        observation = {
            "image_path": image_path,
            "rgb_means": [round(mean_r, 3), round(mean_g, 3), round(mean_b, 3)],
            "contrast": round(contrast, 3),
            "target_concept": target_concept,
            "semantic_match_score": round(match_score, 3)
        }

        reward = 1.0 if match_score > 0.6 else -0.3
        explanation = f"Visual target '{target_concept}' evaluated with match score {match_score:.2f}."

        return {
            "objective": f"identify_{target_concept}",
            "observation": observation,
            "expected_outcome": "positive_cue",
            "actual_outcome": "match" if match_score > 0.6 else "mismatch",
            "reward": reward,
            "error": round(1.0 - match_score, 3),
            "explanation": explanation,
            "recommended_curriculum_step": "proceed_to_speech_association" if match_score > 0.6 else "refine_visual_filter"
        }
