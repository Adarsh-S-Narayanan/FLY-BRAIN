# MaleCNS FlyBrain Architecture Specification

## 1. System Overview
FlyBrain is a real, local, Windows 11 compatible, Vulkan-first artificial-organism research framework built directly around the authentic Drosophila melanogaster male central nervous system (`malecns`) biological connectome dataset.

The system integrates:
1. **Biological Source (`malecns`)**: 125,506 verified neuron somas, coordinates, sides (L/R/M), and 25,391,204 presynaptic sites (`tbars`) from Janelia FlyEM.
2. **Computational Connectome Runtime**: High-performance sparse representation (CSR), deterministic indexing, biophysically grounded leaky integrate-and-fire / activation dynamics.
3. **Vulkan GPU Compute Engine**: Native Vulkan 1.2+ compute pipelines running on AMD Radeon Graphics (Device 0x1681), dispatching compiled SPIR-V compute shaders with a 100% bit-exact reference CPU baseline.
4. **Brain State & Plasticity**: Multi-compartment dynamic state tracking membrane potential, spiking activations, prediction errors, reward signals, homeostatic drives, and reward-modulated Hebbian / STDP plasticity.
5. **Structural Evolution**: Measurable synaptic growth, pruning, rewiring, population expansion, mutation rollback, and parallel candidate selection.
6. **Multi-Store Persistent Memory**: Working memory, episodic replay buffer, semantic associative memory, and dream stores surviving process restarts.
7. **Local Cognitive Trainer**: Local Qwen 3.x 4B-class model (`Qwen3-4B-Q4_K_M.gguf` via `llama_cpp`) acting as curriculum planner, semantic evaluator, and evolution advisor without replacing internal brain state.
8. **Real Tool Connectors**: Typed connectors for visual observation, audio listening, speech recognition, speech synthesis (Windows SAPI), local image generation (diffusion), memory storage/retrieval, and environmental action.
9. **Interactive Live Dashboard & Visualization**: Real-time telemetry displaying neural activity, connectome graph, memory logs, tool activations, evolution scores, and diagnostics.

---

## 2. Hardware & Environment Grounding
- **Operating System**: Microsoft Windows 11 Pro 64-bit (Build 10.0.26200)
- **CPU**: AMD Ryzen 7 7735HS (8 Cores, 16 Logical Processors)
- **RAM**: 19.79 GB physical memory (10.5+ GB available)
- **GPU**: AMD Radeon(TM) Graphics (RDNA2 680M architecture, Device ID 0x1681, Driver 32.0.21043.12001)
- **Vulkan Driver**: LunarG Vulkan SDK 1.4.357.0 with `glslc` SPIR-V compiler
- **Python Runtime**: Python 3.12 (CPython virtual environment `.venv`) with NumPy, SciPy, PyTorch, Pillow, SoundDevice, SoundFile, Vulkan, FastAPI, Uvicorn, WebSockets, PyWebView.

---

## 3. Subsystem Boundaries & Dataflow

```
   +-----------------------------------------------------------+
   |            Biological Source Data (malecns)               |
   |   125,506 neurons, 3D coordinates, tbars, soma_sides       |
   +-----------------------------+-----------------------------+
                                 |
                                 v
   +-----------------------------------------------------------+
   |             Connectome Runtime Graph (CSR)                |
   |       Deterministic Sparse Matrix & Synapse Index         |
   +-----------------------------+-----------------------------+
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
+-------------------------------+             +-------------------------------+
|     Vulkan Compute Shader     |             |      CPU Reference Engine     |
|   (GPU SPIR-V via glslc)      |<---Verify-->|    (Deterministic Reference)  |
+---------------+---------------+             +---------------+---------------+
                |                                             |
                +----------------------+----------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |       Computational Brain         |
                     |  - Membrane Potential (V)         |
                     |  - Spikes / Activations (A)       |
                     |  - Homeostatic Drives & Reward    |
                     |  - Prediction Error State         |
                     +-----------------+-----------------+
                                       |
        +------------------------------+------------------------------+
        |                              |                              |
        v                              v                              v
+----------------+            +-----------------+           +--------------------+
| Plasticity &   |            | Multi-Store     |           | Typed Tool         |
| Evolution      |            | Persistent      |           | Connectors         |
| - STDP/Hebbian |            | Memory          |           | - Observe Visual   |
| - Pruning      |            | - Working       |           | - Listen Audio     |
| - Rewiring     |            | - Episodic      |           | - Speak (TTS)      |
| - Growth       |            | - Semantic      |           | - Image Gen (Diff) |
| - Rollback     |            | - Dream Replay  |           | - Environment Act  |
+----------------+            +-----------------+           +--------------------+
        ^                              ^                              ^
        |                              |                              |
        +------------------------------+------------------------------+
                                       |
                     +-----------------+-----------------+
                     |  Local Cognitive & Vision Trainer |
                     |  (Qwen 3.x 4B-class GGUF / OpenCV)|
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |     Live Visualizer & UI          |
                     |  (FastAPI + WebSockets + Canvas)  |
                     +-----------------------------------+
```

---

## 4. Determinism & Numerical Precision Policy
- All experiments and evolution rollouts record explicit PRNG seeds (`seed_baseline = 42`).
- Vulkan float32 compute outputs must match CPU 64/32-bit reference outputs within:
  - Absolute tolerance: $1 \times 10^{-4}$
  - Relative tolerance: $1 \times 10^{-3}$
- All structural mutations, snapshots, and memory stores must be serializable to JSON/NPZ and verifiable by SHA-256 hashes.
