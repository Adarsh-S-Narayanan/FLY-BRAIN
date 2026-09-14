# FlyBrain: Final Technical & Behavioral Verification Report

**Framework**: FlyBrain — biologically grounded artificial-life research framework
**Target Biological Dataset**: Janelia MaleCNS v1.0 (`male-cns:v1.0`)
**Host Environment**: Windows 11, AMD Ryzen 7 7735HS, AMD Radeon(TM) Graphics, Vulkan SDK 1.4.357
**Report date**: 2026-09-14 · **Overall Status**: **PASSED (25/25 acceptance gates, 63/63 tests)**

> Supersedes all earlier verification reports. Historical claims (e.g. a locally-present LLM,
> 1,934-synapse REAL-256 circuits, "no blockers") are obsolete and retained nowhere as current.

---

## 1. Verification commands (all runnable, all green)

```powershell
.venv\Scripts\python.exe src/main.py test                # 63/63 pass (all suites incl. ALife)
.venv\Scripts\python.exe src/main.py acceptance-matrix   # 25/25 PASS, 0 failed, 0 skipped
.venv\Scripts\python.exe src/main.py docs-verify         # consistency checks pass
.venv\Scripts\python.exe src/main.py validate-vulkan     # 9/9 CPU/GPU parity (diffs < 1.79e-07)
.venv\Scripts\python.exe scripts/run_alife_experiment.py --population 10 --ticks 150 --seed 7
.venv\Scripts\python.exe scripts/run_alife_experiment.py --verify diagnostics/alife_experiments/alife_p10_t150_s7.json  # match=True
```

## 2. Architecture (as built)

1. **Biological source**: 125,506 somas; 99,301 connection rows across 2,045 unique bodies.
2. **Connectome modes**: `REAL` (VERIFIED, 100% empirical edges, no fallback wiring),
   `SPATIAL_SURROGATE` (SURROGATE k-d tree), `SYNTHETIC_TEST` (EXPERIMENTAL). Loader raises
   instead of substituting modes; caches version-guarded (`LOADER_VERSION`).
3. **Population metadata**: coordinate heuristics flagged `heuristic: true`,
   `annotation_status: no_em_annotation_available`; never `VERIFIED`.
4. **LIF contract**: normalized units by default with documented mV mapping
   ($V_{\text{norm}}=(V_{\text{mV}}+70)/20$); identical parametrized equations in
   `cpu_lif_step`, `brain_step.comp`, runtime; GPU refractory counters read back authoritatively.
5. **Plasticity**: one three-factor rule everywhere
   ($\Delta W=\eta r(a_{\text{pre}}a_{\text{post}}-\beta W)$, $\eta=0.05$, $\beta=0.01$);
   shader resolves CSR row-owners; runtime feeds identical pre/post spikes;
   12-step CPU/GPU trajectory parity (spikes exact, weights < 2e-08).
6. **Vulkan engine**: scored device selection, persistent buffers/pipelines, barriers,
   idempotent cleanup, reload-safe; lifecycle stress-tested.
7. **Runtime**: snapshot/restore covers topology, weights, all dynamic state, drives,
   goals, attention, memory refs; continuation bit-exact on CPU and GPU.
8. **Determinism**: all RNG seeded with provenance (no uuid4 in scientific records);
   evolution IDs deterministic; scheduler checkpoint/resume bit-exact;
   world config (incl. world_seed) part of snapshots.
9. **ALife stack**: genome v1.0, development pipeline, grid world, organisms, overlapping
   generations, Pareto selection, cultural transmission, sleep/dream replay, colony API.
10. **Evolution**: candidate isolation, parent preservation, rollback, deterministic lineage,
    checkpoint/resume; `StructuralMutator` refreshes `graph_hash` after every mutation and
    preserves graph subclass/mode on clone.
11. **Memory**: SQLite with write lock; capacity/eviction, restart, malformed-input safety,
    and 8-thread concurrent-write tests green.
12. **Concurrency**: RLock engine + real multi-thread stress test (readers, steppers,
    start/pause cycler; no deadlock, no races, invariants hold).
13. **CLI**: `flybrain test` runs the complete suite (63 tests); matrix/docs-verify exit
    non-zero on failure.
14. **UI/API**: all fetched routes behaviorally tested (`tests/test_ui_api.py`); connectome
    payloads expose mode + provenance; dream replay seed deterministic by default.

## 3. Canonical results

- **ALife** (seed 7, pop 10, 150 ticks): 8 living, 4 births, 6 deaths, generations `[0, 1]`
  coexisting, 131 teaching sessions, replay hash `70a5e4b29e06c320`, match `True`.
- **Vulkan benchmarks** (2026-09-14, current REAL topology): 256/1,449 @ 0.374 ms;
  512/6,557 @ 0.237 ms; 1,024/25,749 @ 0.260 ms.
- **Shaders**: committed SPV byte-identical to fresh `glslc` compiles.

## 4. Capability classification

### IMPLEMENTED + VERIFIED
Connectome loading (3 modes), LIF CPU/GPU + parity, plasticity (unified rule), snapshots,
deterministic experiments, memory + dreams, evolution + rollback + checkpoints, ALife
(genome/development/organism/world/population/culture/sleep), colony API, CLI, 25-gate matrix.

### IMPLEMENTED + EXPERIMENTAL
Population simulation runs CPU-only (no batched multi-organism GPU stepping); dream
consolidation is replay-based; long-run (>4-generation) campaigns untested beyond the
canonical 150-tick run.

### NOT IMPLEMENTED
LLM/VLM scientist loop (slot raises `MODEL_UNAVAILABLE`); 3D colony/development-timeline
visualizations (data APIs exist); 100+ generation checkpointed campaigns; device-local
GPU memory staging (portable host-visible buffers used — documented tradeoff).

### UNAVAILABLE ON CURRENT ENVIRONMENT
None — Vulkan hardware present; all GPU gates execute (0 skips).

## 5. Bugs fixed during this release pass
1. REAL-mode invented fallback edges (removed; 3 honestly isolated neurons in REAL-256).
2. Silent REAL→surrogate fallback on loader error (now raises).
3. Stale connectome caches (version-guarded rebuilds).
4. Provenance metadata lost on cache round-trip (reconstructed with edge counts).
5. GPU refractory recomputed locally with hardcoded `t_ref` (authoritative readback now).
6. CPU/GPU plasticity rule mismatch — shader ignored post-synaptic input (three-factor
   shader with CSR row lookup; unified decay 0.01; runtime restores S(t-1) around ping-pong).
7. `graph_hash` never refreshed after structural mutation (stale provenance).
8. Clone dropped graph subclass/mode/populations.
9. Scheduler uuid4 IDs (deterministic IDs now).
10. API dream seed from wall-clock (deterministic default now).
11. World config dropped on restore (regrowth diverged; fixed + tested).
12. Sensorimotor maps, drives, reward trackers missing from organism snapshots.
13. Hard-coded machine paths (LLM, VAE, Edge, manifests → env/discovery).
14. VAE renderer honesty (`renderer` field per call); placeholder-embedding labeling.
15. SQLite write contention (RLock).
16. `flybrain test` ran legacy suite only (now full discovery: 63 tests).
17. Stale docs (19→25 gates, 2,048→2,045 bodies, λ wording, benchmark re-measurement).

## 6. Proof locations
- `diagnostics/acceptance_matrix.json` (25/25), `diagnostics/cpu_gpu_validation_report.json` (9/9),
  `diagnostics/alife_experiments/alife_p10_t150_s7.json`, `diagnostics/alife_baseline.json`,
  `diagnostics/dependency_manifest.json`, `manifests/malecns_provenance.json`.
- Test files: `tests/test_suite.py` (11), `tests/test_alife.py` (12), `tests/test_provenance.py` (6),
  `tests/test_lif.py` (11), `tests/test_vulkan_lifecycle.py` (5), `tests/test_runtime_continuation.py` (3),
  `tests/test_evolution_robustness.py` (3), `tests/test_memory_robustness.py` (5),
  `tests/test_engine_stress.py` (1), `tests/test_ui_api.py` (6).
