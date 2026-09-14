"""Local LLM runtime on llama-cpp-python (REAL, IMPLEMENTED).

Two reproducibility modes:
  RESEARCH_DETERMINISTIC: temperature=0, fixed seed, full provenance recorded.
  NORMAL_GENERATION: caller-supplied sampling.
Never returns fake text: every failure yields a structured error status.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.llm.discovery import ModelInfo, discover_models

RESEARCH_DETERMINISTIC = "RESEARCH_DETERMINISTIC"
NORMAL_GENERATION = "NORMAL_GENERATION"


def _runtime_version() -> str:
    try:
        import llama_cpp
        return getattr(llama_cpp, "__version__", "unknown")
    except Exception:
        return "not_installed"


@dataclass
class GenerationConfig:
    mode: str = RESEARCH_DETERMINISTIC
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.0
    seed: int = 42
    max_tokens: int = 128
    n_ctx: int = 4096

    def effective(self) -> Dict[str, Any]:
        if self.mode == RESEARCH_DETERMINISTIC:
            return {"temperature": 0.0, "seed": int(self.seed)}
        return {"temperature": float(self.temperature), "seed": int(self.seed)}


class LocalLLM:
    """Loads one discovered GGUF model. Untrusted output; provenance always attached."""

    def __init__(self, model: Optional[ModelInfo] = None, n_ctx: int = 4096):
        self.model = model
        self.n_ctx = int(n_ctx)
        self._llm = None
        self.status = "MODEL_UNAVAILABLE"
        self.last_error = ""
        if model is not None and not model.status.startswith("CORRUPT"):
            self.status = "DISCOVERED_NOT_LOADED"

    @classmethod
    def auto(cls, n_ctx: int = 4096) -> "LocalLLM":
        models = [m for m in discover_models() if m.status == "DISCOVERED"]
        if not models:
            inst = cls(None, n_ctx)
            inst.last_error = "no usable GGUF model discovered under llm/"
            return inst
        return cls(models[0], n_ctx)

    def load(self) -> bool:
        if self.model is None:
            self.status = "MODEL_UNAVAILABLE"
            self.last_error = "no model discovered"
            return False
        if self.model.status.startswith("CORRUPT"):
            self.status = "MODEL_LOAD_ERROR"
            self.last_error = self.model.status
            return False
        try:
            from llama_cpp import Llama
            self._llm = Llama(model_path=self.model.path, n_ctx=self.n_ctx, verbose=False)
            self.status = "OPERATIONAL"
            self.last_error = ""
            return True
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            self.status = "MODEL_OOM" if "memory" in msg.lower() or "alloc" in msg.lower() \
                else "MODEL_LOAD_ERROR"
            self.last_error = msg
            self._llm = None
            return False

    def is_ready(self) -> bool:
        return self._llm is not None and self.status == "OPERATIONAL"

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None,
                 system: str = "", timeout_note: str = "") -> Dict[str, Any]:
        cfg = config or GenerationConfig()
        if not self.is_ready():
            return {"status": self.status, "error": self.last_error or "model not loaded",
                    "text": None, "provenance": self._prov(cfg, prompt, system)}
        eff = cfg.effective()
        full = f"{system}\n{prompt}" if system else prompt
        t0 = time.time()
        try:
            out = self._llm(full, max_tokens=cfg.max_tokens,
                            temperature=eff["temperature"], top_k=cfg.top_k,
                            top_p=cfg.top_p, min_p=cfg.min_p, seed=eff["seed"])
            text = out["choices"][0]["text"]
            usage = out.get("usage", {})
        except Exception as e:  # noqa: BLE001
            return {"status": "MODEL_RUNTIME_ERROR", "error": f"{type(e).__name__}: {e}",
                    "text": None, "provenance": self._prov(cfg, prompt, system)}
        prov = self._prov(cfg, prompt, system)
        prov.update({"latency_sec": round(time.time() - t0, 2),
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens")})
        return {"status": "SUCCESS", "text": text, "provenance": prov}

    def _prov(self, cfg: GenerationConfig, prompt: str, system: str) -> Dict[str, Any]:
        m = self.model
        return {
            "model_path": m.path if m else None,
            "model_sha256": m.sha256 if m else None,
            "model_size_bytes": m.size_bytes if m else None,
            "architecture": m.architecture if m else None,
            "quantization": m.quantization if m else None,
            "runtime": "llama-cpp-python",
            "runtime_version": _runtime_version(),
            "mode": cfg.mode,
            "generation_parameters": {"temperature": cfg.effective()["temperature"],
                                      "top_k": cfg.top_k, "top_p": cfg.top_p,
                                      "min_p": cfg.min_p, "seed": cfg.seed,
                                      "max_tokens": cfg.max_tokens, "n_ctx": self.n_ctx},
            "system_prompt_hash": hashlib.sha256(system.encode()).hexdigest()[:16],
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        }

    def unload(self):
        self._llm = None
        if self.model is not None:
            self.status = "DISCOVERED_NOT_LOADED"
