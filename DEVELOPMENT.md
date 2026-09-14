# FlyBrain Development Guide

## 1. Environment & Prerequisites
- **OS**: Windows 11 64-bit
- **Python**: Python 3.12 (`.venv` virtual environment managed by `uv`)
- **Vulkan SDK**: LunarG Vulkan SDK 1.4.357.0 (with `glslc` SPIR-V compiler in PATH or SDK bin)
- **Host Dependencies**: `git`, `powershell`, Microsoft Edge (for visual evidence capture)

## 2. Directory Layout
```
FlyBrain/
├── malecns/                      # Biological Male CNS connectome data (125k somas)
├── diagnostics/                  # Evidence artifacts (manifests, reports, logs, DBs)
├── shaders/                      # GLSL compute shaders and compiled SPIR-V
│   ├── brain_step.comp / .spv
│   └── plasticity.comp / .spv
├── src/
│   ├── connectome/               # Biological data loading, KD-tree, CSR layout
│   ├── compute/                  # Vulkan GPU compute engine & CPU reference baseline
│   ├── brain/                    # BrainState, BrainRuntime, PlasticityEngine, Drives
│   ├── memory/                   # Working, episodic, semantic, dream persistence (SQLite)
│   ├── tools/                    # Typed connector contracts and implementations
│   ├── trainer/                  # Cognitive trainer (Qwen3-4B GGUF) and Vision teacher
│   ├── evolution/                # Parallel candidate evolution, mutations, rollback
│   ├── dream/                    # Experience replay and counterfactual simulation
│   ├── ui/                       # FastAPI live dashboard and HTML5 Canvas frontend
│   └── main.py                   # Central CLI entrypoint
├── tests/                        # Automated unit and integration test suite
├── visual_evidence/              # Screenshots and live output audio/image artifacts
└── launch.bat                    # Windows batch launcher
```

## 3. Shader Compilation
Whenever modifying GLSL compute shaders in `shaders/`:
```powershell
& "$env:VULKAN_SDK\Bin\glslc.exe" shaders/brain_step.comp -o shaders/brain_step.spv
& "$env:VULKAN_SDK\Bin\glslc.exe" shaders/plasticity.comp -o shaders/plasticity.spv
```

## 4. Coding Standards & Integrity
- All PRNG seeds must be recorded explicitly.
- No mocks, development stubs, or fake demo behaviors are permitted.
- All structural changes must be measurable and verifiable via SHA-256 hashes.
