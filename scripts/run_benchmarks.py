"""Reproducible benchmarking harness (REAL measurements only).

Writes timestamped, never-overwritten reports with full provenance.
"""
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import numpy as np

from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode
from src.brain.runtime import BrainRuntime

OUT_DIR = "diagnostics/benchmarks"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def _env() -> dict:
    return {}


def bench_cpu_step(sizes=(64, 128, 256)):
    results = []
    for size in sizes:
        g = get_or_create_circuit(size, mode=GraphMode.SYNTHETIC_TEST, seed=42,
                                  cache_name=f"bench_cpu_{size}.npz")
        from copy import deepcopy
        rt = BrainRuntime(deepcopy(g), use_gpu=False, enable_plasticity=False, seed=42)
        rng = np.random.RandomState(1)
        for _ in range(3):
            rt.step(sensory_inputs={"visual": rng.uniform(0, 0.4, 16).astype(np.float32)},
                    reward=0.0)
        lat = []
        for _ in range(30):
            t0 = time.perf_counter()
            rt.step(sensory_inputs={"visual": rng.uniform(0, 0.4, 16).astype(np.float32)},
                    reward=0.0)
            lat.append((time.perf_counter() - t0) * 1000.0)
        rt.cleanup()
        mean_ms = float(np.mean(lat))
        results.append({"neurons": size, "synapses": g.num_synapses,
                        "mean_step_ms": round(mean_ms, 4),
                        "steps_per_sec": round(1000.0 / mean_ms, 1),
                        "synapses_per_sec": round(1000.0 / mean_ms * g.num_synapses)})
    return results


def bench_vulkan_step(sizes=(64, 128, 256)):
    try:
        from src.compute.vulkan_backend import VulkanComputeEngine
        eng = VulkanComputeEngine()
    except Exception as e:
        return {"status": "UNAVAILABLE", "error": str(e)}
    try:
        results = []
        for size in sizes:
            g = get_or_create_circuit(size, mode=GraphMode.REAL, seed=42,
                                      cache_name=f"bench_vk_{size}.npz")
            n = g.num_neurons
            eng.load_circuit(g.row_offsets, g.col_indices, g.weights,
                             np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))
            ext = np.full(n, 0.3, dtype=np.float32)
            for _ in range(3):
                eng.run_step_persistent(external_inputs=ext, readback=True)
            lat = []
            for _ in range(20):
                t0 = time.perf_counter()
                eng.run_step_persistent(external_inputs=ext, readback=True)
                lat.append((time.perf_counter() - t0) * 1000.0)
            mean_ms = float(np.mean(lat))
            results.append({"neurons": n, "synapses": g.num_synapses,
                            "mean_step_ms": round(mean_ms, 4),
                            "steps_per_sec": round(1000.0 / mean_ms, 1),
                            "synapses_per_sec": round(1000.0 / mean_ms * g.num_synapses)})
        return {"status": "MEASURED", "device": eng.device_name, "cases": results}
    finally:
        eng.cleanup()


def bench_llm():
    try:
        from src.llm.runtime import LocalLLM, GenerationConfig
        llm = LocalLLM.auto(n_ctx=2048)
        if not llm.load():
            return {"status": llm.status, "error": llm.last_error}
        cfg = GenerationConfig(max_tokens=32, seed=1)
        lat = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = llm.generate("The fly brain", cfg)
            lat.append(time.perf_counter() - t0)
            if r["status"] != "SUCCESS":
                return {"status": r["status"], "error": r.get("error")}
        return {"status": "MEASURED", "model": llm.model.filename,
                "model_sha256": llm.model.sha256,
                "mean_gen_sec_32tok": round(float(np.mean(lat)), 3),
                "tokens_per_sec": round(32 / float(np.mean(lat)), 1)}
    except Exception as e:  # noqa: BLE001
        return {"status": "UNAVAILABLE", "error": f"{type(e).__name__}: {e}"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_step": bench_cpu_step(),
        "vulkan_step": bench_vulkan_step(),
        "llm_inference": bench_llm(),
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"bench_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"cpu_step": report["cpu_step"],
                      "vulkan_step": report["vulkan_step"].get("status"),
                      "llm_inference": report["llm_inference"].get("status"),
                      "report": path}, indent=2))


if __name__ == "__main__":
    main()
