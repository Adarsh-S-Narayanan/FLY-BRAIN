# FlyBrain Testing & Verification Protocol

## 1. Automated Test Execution
Run the complete integration and behavioral test suite:
```powershell
.venv\Scripts\python.exe tests/test_suite.py
```
This executes all unit and behavioral tests and writes JUnit XML to `diagnostics/test_results.xml`.

## 2. CPU vs Vulkan GPU Validation Matrix
To verify that GPU execution on the physical AMD Radeon GPU exactly matches the CPU reference:
```powershell
.venv\Scripts\python.exe src/compute/validator.py
```
Outputs validation matrix across circuit sizes (256, 512, 1024) and seeds (42, 100, 2026) to `diagnostics/cpu_gpu_validation_report.json`.
Tolerances:
- Absolute tolerance: $1 \times 10^{-4}$
- Relative tolerance: $1 \times 10^{-3}$
- Observed difference: $\le 9.54 \times 10^{-7}$ (Activation) and $\le 5.96 \times 10^{-8}$ (Weights).

## 3. Curriculum & Tool Learning Benchmark
```powershell
.venv\Scripts\python.exe scripts/run_curriculum_benchmark.py
```
Verifies:
- Gate G015: Measurable synaptic weight changes ($\Delta W > 0.67$) and tool selection accuracy (+0.63 score improvement).
- Gate G016: Multi-step tool sequence execution (Observe -> Remember -> Speak -> Generate Image -> Sleep).

## 4. State Persistence & Replay Verification
```powershell
.venv\Scripts\python.exe scripts/test_state_persistence.py
```
Verifies:
- Saving snapshot to disk
- Spawning a fresh brain instance with different seed
- Restoring snapshot and proving exact 0.00e+00 difference
- Replaying identical inputs and proving deterministic output trajectory.

## 5. Visual Evidence Verification
```powershell
.venv\Scripts\python.exe scripts/capture_visual_evidence.py
```
Captures all 11 required visual evidence screens from the running application into `visual_evidence/screens/`.
