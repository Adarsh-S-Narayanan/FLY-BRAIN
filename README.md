# FlyBrain: Vulkan-First Artificial Organism Framework

FlyBrain is a locally runnable, Windows 11 compatible, Vulkan-accelerated artificial organism research framework built directly around the authentic Drosophila melanogaster male central nervous system (`malecns`) biological connectome dataset (Janelia FlyEM).

---

## Subsystem Implementation & Verification Status

| Subsystem | Status | Verification Detail |
|---|---|---|
| **Biological Source (`malecns`)** | **VERIFIED** | 125,506 neuron somas parsed, coordinates & 25,391,204 tbars extracted from `malecns/data-raw/2023-27-2 soma_sides.csv`. |
| **Connectome Runtime (CSR)** | **VERIFIED** | Compressed Sparse Row (CSR) biological graphs, deterministic KD-tree proximity & synaptic weighting. |
| **Vulkan Compute Engine** | **VERIFIED** | Native Vulkan 1.2+ compute pipeline on AMD Radeon(TM) Graphics dispatching SPIR-V shaders (`brain_step.spv`, `plasticity.spv`). |
| **CPU Reference Engine** | **VERIFIED** | Exact CPU reference baseline compared against Vulkan GPU across multiple seeds/sizes (< 1e-6 diff). |
| **Brain State & Drives** | **VERIFIED** | Persistent membrane potentials, activations, prediction error, reward, and homeostatic drives (energy, curiosity, social, integrity). |
| **Plasticity & Learning** | **VERIFIED** | Reward-modulated Hebbian learning, synaptic strengthening/weakening, structural pruning & rewiring. |
| **Multi-Store Memory** | **VERIFIED** | Working memory (7 slots), episodic memory, semantic associative memory, and dream store surviving restarts via SQLite. |
| **Real Tool Connectors** | **VERIFIED** | Typed connectors for `observe_visual`, `listen_audio`, `speak`, `generate_image`, `remember`, `retrieve_memory`, `inspect_self`, `act_in_environment`, `sleep`. |
| **Speech System** | **VERIFIED** | Windows Native SAPI offline TTS generating real speech audio WAV and speaker playback. |
| **Vision & Image Generation** | **VERIFIED** | OpenCV visual feature extraction and local neural AutoencoderTiny VAE generating real 256x256 imagery. |
| **Cognitive Trainer** | **VERIFIED** | Local Qwen 3.x 4B-class GGUF model (`Qwen3-4B-Q4_K_M.gguf` via `llama_cpp`) acting as curriculum planner and teacher. |
| **Parallel Evolution** | **VERIFIED** | Multi-candidate generational evolution with structural mutations (pruning, growth, rewiring) and rollback. |
| **Dream & Replay System** | **VERIFIED** | Offline experience replay simulating counterfactual alternative actions and logging consolidated insights. |
| **Live Interactive UI** | **VERIFIED** | Real-time FastAPI + HTML5 Canvas 3D connectome dashboard streaming telemetry at 10Hz. |
| **Neuprint Remote API** | **PARTIALLY VERIFIED** | Offline local mode fully verified; remote Neuprint API available when `NEUPRINT_TOKEN` is configured. |
| **Multi-Agent Neuromorphic Swarms** | **FUTURE WORK** | Multi-organism swarm coordination over distributed networks. |

---

## Quick Start (Windows 11)

### 1. Launch Application
Double click `launch.bat` or run:
```powershell
.venv\Scripts\python.exe src\main.py --mode run --host 127.0.0.1 --port 8080
```
Open your browser at `http://127.0.0.1:8080` to access the live dashboard.

### 2. Run Diagnostics & Verification Matrix
```powershell
.venv\Scripts\python.exe src\main.py --mode diagnostics
```

### 3. Run Automated Integration Test Suite
```powershell
.venv\Scripts\python.exe src\main.py --mode test
```
Outputs JUnit XML test results to `diagnostics/test_results.xml`.

### 4. Run Curriculum Learning Benchmark
```powershell
.venv\Scripts\python.exe src\main.py --mode benchmark
```
Outputs benchmark report to `diagnostics/benchmark_results.json`.
