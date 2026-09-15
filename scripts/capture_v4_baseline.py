"""Capture the IMMUTABLE V4 baseline (mission §6).

Writes release/v4/baseline/*.json. These records are never modified afterwards.
"""
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath("."))

OUT = os.path.join("release", "v4", "baseline")


def _git(args):
    return subprocess.run(["git"] + args, capture_output=True, text=True,
                          timeout=10).stdout.strip()


def _write(name, data):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[baseline] wrote {path}")
    return path


def repository_manifest():
    files = {}
    for root, dirs, names in os.walk("."):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "__pycache__",
                                                "diagnostics", "release",
                                                "build", "dist", "*.egg-info")]
        for n in names:
            if n.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(root, n)
            rel = os.path.relpath(p, ".")
            files[rel.replace("\\", "/")] = os.path.getsize(p)
    return {
        "version": "v4.0.0-baseline",
        "captured_ts": time.time(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_tags": _git(["tag", "--list"]),
        "file_count": len(files),
        "total_bytes": sum(files.values()),
        "tracked_files_sample": dict(sorted(files.items())[:400]),
    }


def test_baseline():
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
    result = runner.run(suite)
    return {
        "captured_ts": time.time(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(getattr(result, "skipped", []) or []),
        "success": result.wasSuccessful(),
    }


def acceptance_baseline():
    from scripts.run_acceptance_matrix import evaluate_acceptance_matrix
    rep = evaluate_acceptance_matrix()
    return {
        "captured_ts": time.time(),
        "gates": rep["total_categories"],
        "passed": rep["passed"],
        "failed": rep["failed"],
        "skipped": rep["skipped"],
        "overall": rep["overall_status"],
        "categories": rep["categories"],
    }


def environment_baseline():
    env = {
        "captured_ts": time.time(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version,
        "processor": platform.processor(),
    }
    try:
        from src.compute.vulkan_backend import VulkanComputeEngine
        eng = VulkanComputeEngine()
        env["vulkan_device"] = eng.device_name
        eng.cleanup()
    except Exception as e:
        env["vulkan_device"] = f"UNAVAILABLE: {type(e).__name__}: {e}"
    try:
        from src.llm.discovery import discover_models
        models = [m.filename for m in discover_models() if m.status == "DISCOVERED"]
        env["gguf_models"] = models
    except Exception as e:
        env["gguf_models"] = f"ERROR: {e}"
    return env


def dependency_baseline():
    import importlib.metadata as im
    deps = {}
    for dist in sorted(im.distributions(), key=lambda d: (d.metadata["Name"] or "")):
        name = dist.metadata["Name"]
        if name:
            deps[name] = dist.version
    return {"captured_ts": time.time(), "python_packages": deps}


def diagnostics_baseline():
    """Hashes of the current authoritative diagnostics artifacts."""
    import hashlib
    out = {"captured_ts": time.time()}
    for rel in ("diagnostics/acceptance_matrix.json",
                "diagnostics/cpu_gpu_validation_report.json",
                "diagnostics/milestones/milestone_evidence.json",
                "diagnostics/alife_experiments/alife_p6_t60_s7.json"):
        p = os.path.join(rel)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
        else:
            out[rel] = "missing"
    return out


if __name__ == "__main__":
    _write("repository_manifest.json", repository_manifest())
    _write("test_baseline.json", test_baseline())
    _write("acceptance_baseline.json", acceptance_baseline())
    _write("environment_baseline.json", environment_baseline())
    _write("dependency_baseline.json", dependency_baseline())
    _write("diagnostics_baseline.json", diagnostics_baseline())
    print("[baseline] V4 baseline capture complete (IMMUTABLE)")
