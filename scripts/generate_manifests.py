import json
import os
import sys
import subprocess
import hashlib
import platform

def get_git_info():
    info = {}
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if res.returncode == 0:
            info["main_repo_head"] = res.stdout.strip()
    except Exception as e:
        info["main_repo_head"] = str(e)
    
    try:
        res = subprocess.run(["git", "-C", "malecns", "rev-parse", "HEAD"], capture_output=True, text=True)
        if res.returncode == 0:
            info["malecns_head"] = res.stdout.strip()
    except Exception as e:
        info["malecns_head"] = str(e)
    return info

def file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    os.makedirs("diagnostics", exist_ok=True)
    git_info = get_git_info()
    
    soma_csv = os.path.join("malecns", "data-raw", "2023-27-2 soma_sides.csv")
    soma_hash = file_hash(soma_csv)

    # Dependency manifest
    dep_manifest = {
        "python_runtime": {
            "version": sys.version,
            "executable": sys.executable
        },
        "vulkan_sdk": {
            "env_path": os.environ.get("VULKAN_SDK"),
            "version": "1.4.357.0",
            "glslc_path": "C:\\VulkanSDK\\1.4.357.0\\Bin\\glslc.exe"
        },
        "git_provenance": git_info,
        "biological_source_hashes": {
            "soma_sides_csv": {
                "path": soma_csv,
                "sha256": soma_hash
            }
        },
        "target_qwen_model": {
            "path": "C:\\Users\\hcsme\\.cache\\huggingface\\hub\\models--unsloth--Qwen3-4B-GGUF\\snapshots\\22c9fc8a8c7700b76a1789366280a6a5a1ad1120\\Qwen3-4B-Q4_K_M.gguf",
            "exists": os.path.exists("C:\\Users\\hcsme\\.cache\\huggingface\\hub\\models--unsloth--Qwen3-4B-GGUF\\snapshots\\22c9fc8a8c7700b76a1789366280a6a5a1ad1120\\Qwen3-4B-Q4_K_M.gguf")
        }
    }
    
    with open("diagnostics/dependency_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dep_manifest, f, indent=2)
    print("Wrote diagnostics/dependency_manifest.json")

    # Reproducibility manifest
    repro_manifest = {
        "seed_baseline": 42,
        "platform": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": git_info,
        "data_hashes": {
            "malecns_soma": soma_hash
        },
        "precision_tolerances": {
            "vulkan_vs_cpu_float32_abs_tol": 1e-4,
            "vulkan_vs_cpu_float32_rel_tol": 1e-3
        },
        "deterministic_flags": {
            "torch_deterministic": True,
            "numpy_seed": 42
        }
    }
    with open("diagnostics/reproducibility_manifest.json", "w", encoding="utf-8") as f:
        json.dump(repro_manifest, f, indent=2)
    print("Wrote diagnostics/reproducibility_manifest.json")

if __name__ == "__main__":
    main()
