import os
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
from src.tools.base import ToolConnector

class ObserveVisualConnector(ToolConnector):
    def __init__(self):
        super().__init__(
            name="observe_visual",
            description="Observes and extracts 64-dimensional biophysical visual features from an image file or synthetic visual field.",
            timeout_sec=10.0
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "synthetic_target": {"type": "string"}
            }
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "features_vector": {"type": "array", "items": {"type": "number"}},
                "mean_brightness": {"type": "number"},
                "contrast": {"type": "number"},
                "dominant_channel": {"type": "string"}
            },
            "required": ["features_vector", "mean_brightness", "contrast", "dominant_channel"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        img_path = params.get("image_path")
        
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB").resize((64, 64))
            arr = np.array(img, dtype=np.float32) / 255.0
        else:
            # Generate synthetic sensory visual field based on synthetic_target
            target = params.get("synthetic_target", "neutral")
            arr = np.zeros((64, 64, 3), dtype=np.float32)
            if "red" in target:
                arr[:, :, 0] = 0.8
            elif "green" in target:
                arr[:, :, 1] = 0.8
            elif "blue" in target:
                arr[:, :, 2] = 0.8
            elif "bright" in target:
                arr[:] = 0.9
            else:
                # Gradient field
                for y in range(64):
                    for x in range(64):
                        arr[y, x, 0] = x / 64.0
                        arr[y, x, 1] = y / 64.0
                        arr[y, x, 2] = 0.5

        # Extract 64-d feature vector
        # 16 values: 4x4 spatial grid of mean luminance
        grid = arr.mean(axis=2).reshape(4, 16, 4, 16).mean(axis=(1, 3)).flatten()
        # 16 values: color distribution histograms
        hist_r, _ = np.histogram(arr[:, :, 0], bins=8, range=(0, 1))
        hist_g, _ = np.histogram(arr[:, :, 1], bins=8, range=(0, 1))
        # 16 values: horizontal gradients
        grad_x = np.abs(np.diff(arr.mean(axis=2), axis=1))
        grad_pool_x = grad_x.reshape(4, 16, 63).mean(axis=(1, 2))
        grad_pool_pad = np.pad(grad_pool_x, (0, 12), mode="edge")
        # 16 values: vertical gradients
        grad_y = np.abs(np.diff(arr.mean(axis=2), axis=0))
        grad_pool_y = grad_y.reshape(63, 4, 16).mean(axis=(0, 2))
        grad_pool_y_pad = np.pad(grad_pool_y, (0, 12), mode="edge")

        feat = np.concatenate([grid, hist_r / 4096.0, hist_g / 4096.0, grad_pool_pad[:16], grad_pool_y_pad[:16]])
        feat = feat[:64].astype(np.float32)

        mean_b = float(arr.mean())
        contrast = float(arr.std())
        channel_means = [arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()]
        dom_idx = int(np.argmax(channel_means))
        dominant_channel = ["red", "green", "blue"][dom_idx]

        return {
            "features_vector": feat.tolist(),
            "mean_brightness": round(mean_b, 4),
            "contrast": round(contrast, 4),
            "dominant_channel": dominant_channel
        }
