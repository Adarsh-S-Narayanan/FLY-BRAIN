# FlyBrain Reproducibility Guide

## 1. Deterministic Execution Policy
FlyBrain guarantees reproducible execution through:
1. **Explicit PRNG Seeds**: All simulations, mutations, and replays record their seed (default: `seed = 42`).
2. **Biological Source Integrity**: The biological dataset is pinned to SHA-256 hash `d20bb1b48b99cfbe1a0efd0614f85108ce8a30644e5917fa97bc8a873138b309` (`malecns/data-raw/2023-27-2 soma_sides.csv`).
3. **Deterministic Connectome Extraction**: KD-tree spatial queries and CSR matrix layout are deterministic given the seed and neuron count.
4. **Vulkan Kernel Bit-Consistency**: Shaders use standard IEEE-754 float32 arithmetic matching CPU reference to within $10^{-6}$.
5. **Snapshot Hash Verification**: All snapshots record and verify SHA-256 checksums before and after restoration.

## 2. Pinned Environment Hashes & Versions
- Windows 11 Build: 10.0.26200
- Python: 3.12.13 (x86_64)
- Vulkan SDK: 1.4.357.0
- GPU Driver: 32.0.21043.12001
- Model: `Qwen3-4B-Q4_K_M.gguf` (SHA-256 recorded in `diagnostics/dependency_manifest.json`)
- VAE: `AutoencoderTiny` (safetensors format)

## 3. Replay Test Command
```powershell
.venv\Scripts\python.exe scripts/test_state_persistence.py
```
Outputs `diagnostics/brain_snapshot_manifest.json` showing identical bitwise continuity.
