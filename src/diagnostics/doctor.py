"""FlyBrain doctor (mission §47): REAL environment verification.

Every check reports an actual observed value — never a fabricated one.
Statuses: OK | DEGRADED | UNAVAILABLE | ERROR, with exact reasons.
"""
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.version import VERSION  # noqa: E402


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_os() -> Dict[str, Any]:
    return {"name": "os", "status": "OK",
            "detail": f"{platform.system()} {platform.release()} "
                      f"({platform.machine()})"}


def check_python() -> Dict[str, Any]:
    v = sys.version_info
    status = "OK" if v >= (3, 11) else "DEGRADED"
    return {"name": "python", "status": status,
            "detail": f"{v.major}.{v.minor}.{v.micro} "
                      f"({'supported (>=3.11)' if status == 'OK' else 'below supported 3.11'})"}


def check_dependencies() -> List[Dict[str, Any]]:
    out = []
    for mod, min_ver in (("numpy", None), ("scipy", None)):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "unknown")
            out.append({"name": f"dep:{mod}", "status": "OK", "detail": ver})
        except Exception as e:  # noqa: BLE001
            out.append({"name": f"dep:{mod}", "status": "ERROR",
                        "detail": f"{type(e).__name__}: {e}"})
    # optional LLM runtime (may be excluded from portable builds by design)
    try:
        import llama_cpp  # noqa: F401
        out.append({"name": "dep:llama_cpp", "status": "OK",
                    "detail": getattr(__import__("llama_cpp"), "__version__", "unknown")})
    except Exception:
        out.append({"name": "dep:llama_cpp", "status": "UNAVAILABLE",
                    "detail": "local LLM runtime not installed (optional model pack); "
                              "scientist loop will report MODEL_UNAVAILABLE"})
    return out


def check_dataset() -> Dict[str, Any]:
    soma = os.path.join("malecns", "data-raw", "2023-27-2 soma_sides.csv")
    conn = os.path.join("malecns", "data-raw", "malecns_v1_0_connections.csv")
    manifest = os.path.join("manifests", "malecns_provenance.json")
    if not (os.path.exists(soma) and os.path.exists(conn)):
        return {"name": "dataset", "status": "UNAVAILABLE",
                "detail": "MaleCNS data pack missing (optional pack, see docs)"}
    soma_sha = _file_sha256(soma)
    conn_sha = _file_sha256(conn)
    detail = (f"soma_sha256={soma_sha[:16]}.. conn_sha256={conn_sha[:16]}..")
    status = "OK"
    if os.path.exists(manifest):
        try:
            with open(manifest, encoding="utf-8") as f:
                pm = json.load(f)
            pm_meta = pm.get("provenance_metadata", pm)
            exp_soma = pm_meta.get("soma_sha256")
            exp_conn = pm_meta.get("connections_sha256")
            if exp_soma and exp_soma != soma_sha:
                status, detail = "ERROR", "soma hash MISMATCH vs provenance manifest"
            elif exp_conn and exp_conn != conn_sha:
                status, detail = "ERROR", "connections hash MISMATCH vs provenance manifest"
            else:
                detail += " | manifest match=True"
        except Exception as e:  # noqa: BLE001
            status, detail = "DEGRADED", f"{detail} | manifest unreadable: {e}"
    else:
        status = "DEGRADED"
        detail += " | provenance manifest missing"
    return {"name": "dataset", "status": status, "detail": detail}


def check_shaders() -> List[Dict[str, Any]]:
    out = []
    for spv in ("shaders/brain_step.spv", "shaders/plasticity.spv"):
        if os.path.exists(spv):
            out.append({"name": f"shader:{os.path.basename(spv)}", "status": "OK",
                        "detail": f"sha256={_file_sha256(spv)[:16]}.. "
                                  f"size={os.path.getsize(spv)}"})
        else:
            out.append({"name": f"shader:{os.path.basename(spv)}",
                        "status": "UNAVAILABLE",
                        "detail": "compiled SPIR-V missing"})
    return out


def check_gpu() -> Dict[str, Any]:
    try:
        from src.compute.vulkan_backend import VulkanComputeEngine
        eng = VulkanComputeEngine()
        detail = f"device={eng.device_name}"
        eng.cleanup()
        return {"name": "gpu", "status": "OK", "detail": detail}
    except Exception as e:  # noqa: BLE001
        return {"name": "gpu", "status": "UNAVAILABLE",
                "detail": f"Vulkan device unavailable ({type(e).__name__}: {e}); "
                          f"CPU reference engine will be used"}


def check_storage() -> Dict[str, Any]:
    base = "diagnostics"
    try:
        os.makedirs(base, exist_ok=True)
        probe = os.path.join(base, ".doctor_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return {"name": "storage", "status": "OK",
                "detail": f"'{base}' writable"}
    except Exception as e:  # noqa: BLE001
        return {"name": "storage", "status": "ERROR",
                "detail": f"cannot write '{base}': {e}"}


def check_model() -> Dict[str, Any]:
    try:
        from src.llm.discovery import discover_models
        models = [m for m in discover_models() if m.status == "DISCOVERED"]
        if models:
            return {"name": "llm_model", "status": "OK",
                    "detail": f"{models[0].filename} (sha256={models[0].sha256[:16]}..)"}
        return {"name": "llm_model", "status": "UNAVAILABLE",
                "detail": "no GGUF under llm/gguf (optional model pack); "
                          "LLM arms will report MODEL_UNAVAILABLE honestly"}
    except Exception as e:  # noqa: BLE001
        return {"name": "llm_model", "status": "ERROR", "detail": str(e)}


def check_core_runtime() -> Dict[str, Any]:
    try:
        from src.connectome.loader import get_or_create_circuit
        from src.connectome.types import GraphMode
        from src.brain.runtime import BrainRuntime
        g = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=42,
                                  cache_name="doctor_core.npz")
        rt = BrainRuntime(g, use_gpu=False, enable_plasticity=False, seed=42)
        out = rt.step(sensory_inputs={"visual": __import__("numpy").zeros(
            16, dtype="float32")}, reward=0.0)
        rt.cleanup()
        return {"name": "core_runtime", "status": "OK",
                "detail": f"LIF step executed (backend={out['backend']}, "
                          f"step={out['step']})"}
    except Exception as e:  # noqa: BLE001
        return {"name": "core_runtime", "status": "ERROR",
                "detail": f"{type(e).__name__}: {e}"}


def run_doctor() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = [
        check_os(), check_python(),
    ]
    checks += check_dependencies()
    checks.append(check_dataset())
    checks += check_shaders()
    checks.append(check_gpu())
    checks.append(check_storage())
    checks.append(check_model())
    checks.append(check_core_runtime())
    overall = "READY"
    if any(c["status"] == "ERROR" for c in checks):
        overall = "DEGRADED"
    elif any(c["status"] in ("UNAVAILABLE", "DEGRADED") for c in checks):
        overall = "READY_DEGRADED"
    return {
        "doctor_version": "v1",
        "flybrain_version": VERSION,
        "overall": overall,
        "checks": checks,
        "summary": {
            "ok": sum(1 for c in checks if c["status"] == "OK"),
            "degraded": sum(1 for c in checks if c["status"] == "DEGRADED"),
            "unavailable": sum(1 for c in checks if c["status"] == "UNAVAILABLE"),
            "errors": sum(1 for c in checks if c["status"] == "ERROR"),
        },
    }


def main() -> int:
    report = run_doctor()
    print(json.dumps(report, indent=2))
    return 0 if report["overall"].startswith("READY") else 1


if __name__ == "__main__":
    sys.exit(main())
