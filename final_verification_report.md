# FlyBrain: Final Technical & Behavioral Verification Report

**Framework**: FlyBrain — biologically grounded artificial-life research framework
**Target Biological Dataset**: Janelia MaleCNS v1.0 (`male-cns:v1.0`)
**Host Environment**: Windows 11, AMD Ryzen 7 7735HS, AMD Radeon(TM) Graphics, Vulkan SDK 1.4.357
**Report date**: 2026-09-14 · **Overall Status**: **PASSED (32/32 acceptance gates, 111 tests passing)**

> Supersedes all earlier verification reports. Historical claims (e.g. 25/25 gates,
> 63 tests, 10-population/150-tick canonical result, unavailable local LLM,
> 1,934-synapse REAL-256 circuits) are obsolete and retained nowhere as current.

---

## 1. Verification commands (all runnable, all green)

```powershell
.venv\Scripts\python.exe src/main.py test                # 111 pass (all suites incl. ALife, adversarial, campaign, LLM)
.venv\Scripts\python.exe src/main.py acceptance-matrix   # 32/32 PASS, 0 failed, 0 skipped
.venv\Scripts\python.exe src/main.py docs-verify         # consistency checks pass
.venv\Scripts\python.exe src/main.py validate-vulkan     # 9/9 CPU/GPU parity, spike-exact (trajectory tolerance < 1e-4)
.venv\Scripts\python.exe scripts/run_alife_experiment.py --population 6 --ticks 60 --seed 7
.venv\Scripts\python.exe scripts/run_alife_experiment.py --verify diagnostics/alife_experiments/alife_p6_t60_s7.json  # match=True
.venv\Scripts\python.exe scripts/run_long_campaign.py --generations 4 --population 6 --seed 11
.venv\Scripts\python.exe scripts/run_benchmarks.py
.venv\Scripts\python.exe -m unittest tests.test_adversarial tests.test_campaign tests.test_llm_local tests.test_directionality tests.test_plasticity_causal tests.test_parity
```

## 2. Architecture (as built)

1. **Biological source**: 125,506 somas; 99,301 connection rows across 2,045 unique bodies.
2. **Connectome modes**: `REAL` (VERIFIED, 100% empirical edges, no fallback wiring),
   `SPATIAL_SURROGATE` (SURROGATE k-d tree), `SYNTHETIC_TEST` (EXPERIMENTAL). Loader raises
   instead of substituting modes; caches version-guarded (`LOADER_VERSION`) and now
   persist provenance metadata; stale metadata triggers rebuild.
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
9. **Provenance hashing**: `src/common/provenance.py` folds dataset/model/graph/brain/world/organism/population/event layers into a stable experiment fingerprint; changing a child changes the parent.
10. **ALife stack**: genome v1.0, development pipeline, grid world, organisms, overlapping
    generations, Pareto selection, cultural transmission, sleep/dream replay, colony API.
11. **Local LLM layer**: discovers local GGUF models (`src/llm/discovery.py`); scientist loop
    (`src/llm/runtime.py`, `src/llm/scientist.py`, `src/llm/tools.py`) with safety rejects for
    unavailable models and disallowed tools. Outputs always hypotheses, never ground truth.
12. **Reproducible infrastructure**: long-run campaign/checkpoint/resume (`scripts/run_long_campaign.py`),
    honest benchmarks (`scripts/run_benchmarks.py`), and adversarial robustness tests
    (`tests/test_adversarial.py`).
13. **Evolution**: candidate isolation, parent preservation, rollback, deterministic lineage,
    checkpoint/resume; `StructuralMutator` refreshes `graph_hash` after every mutation
    and preserves graph subclass/mode on clone.
14. **Memory**: SQLite with write lock; capacity/eviction, restart, malformed-input safety,
    and 8-thread concurrent-write tests green.
15. **Concurrency**: RLock engine + real multi-thread stress test (readers, steppers,
    start/pause cycler; no deadlock, no races, invariants hold).
16. **CLI**: `flybrain test` runs the complete discovery suite (111 tests); matrix/docs-verify exit
    non-zero on failure.
17. **UI/API**: all fetched routes behaviorally tested (`tests/test_ui_api.py`); connectome
    payloads expose mode + provenance; dream replay seed deterministic by default; LLM status/inference/tool endpoints serve real state.

## 3. Canonical results

- **Workstation profile** (Vulkan + local GGUF present): acceptance matrix **32/32 PASS**,
  0 skipped.
- **Cloud CI profile** (ubuntu-latest / windows-latest, no GPU device, no GGUF weights
  — model binaries are gitignored): **28 PASS / 4 SKIP / 0 FAIL → PASSED**
  (skips: `vulkan_discovery_and_selection`, `persistent_resource_lifecycle`,
  `cpu_vulkan_numerical_parity`, `llm_model_discovery_and_inference`).
  Unavailable-model honesty is independently verified by
  `llm_failure_mode_and_tool_safety` (PASS everywhere).

- **ALife** (seed 7, pop 6, 60 ticks): 7 living, 4 births, 2 deaths, generations `[0, 1]`
  coexisting, 12 teaching sessions, replay hash `70a5e4b29e06c320`, match `True`.
- **Campaign checkpoint/resume** (seed 11, pop 6, 4 generations): resume branch hash equals uninterrupted branch.
- **Vulkan benchmarks** (2026-09-14, current REAL topology): 256/1,449 @ 0.374 ms;
  512/6,557 @ 0.237 ms; 1,024/25,749 @ 0.977 ms. See `diagnostics/benchmarks/` for timestamped provenance reports.
- **Shaders**: committed SPV byte-identical to fresh `glslc` compiles.
- **CPU/Vulkan parity**: trajectory parity with exact spike trains, max abs potential diff < 1e-4; explicitly not bit-exact.

## 4. Capability classification

### IMPLEMENTED + VERIFIED
Connectome loading (3 modes), LIF CPU/GPU + spike-exact parity, plasticity (unified rule), snapshots,
deterministic experiments, memory + dreams, evolution + rollback + checkpoints, ALife
(genome/development/organism/world/population/culture/sleep/teaching), local LLM discovery/scientist/unsafe-tool rejection,
colony/UI API, CLI, 32-gate matrix, provenance hashing, campaign checkpoint/resume, benchmarks, adversarial tests.

### IMPLEMENTED + EXPERIMENTAL
- Population simulation runs CPU-only (no batched multi-organism GPU stepping).
- Dream consolidation is replay-based; no synaptic downscaling model yet.
- CPU/Vulkan integration is trajectory-parity, not bit-exact (documented).
- Local LLM outputs are from a 2B-class quantization locally; scientifically weak, always hypotheses.

### NOT IMPLEMENTED
3D colony/development-timeline visualizations (data APIs exist); multi-GPU batched organism stepping;
synaptic downscaling/homeostatic sleep consolidation models.

### UNAVAILABLE ON CURRENT ENVIRONMENT
None — Vulkan hardware present; all GPU gates execute (0 skips).

## 5. Bugs fixed during this release pass
1. REAL-mode invented fallback edges (removed; 3 honestly isolated neurons in REAL-256).
2. Silent REAL→surrogate fallback on loader error (now raises).
3. Stale connectome caches (version-guarded rebuilds; metadata-missing rebuilds).
4. Provenance metadata lost on cache round-trip (persisted to cache and reloaded; malformed/stale metadata rejected).
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
16. `flybrain test` ran legacy suite only (now full discovery: 111 tests).
17. Stale docs (25→32 gates, 63→111 tests, 2,048→2,045 bodies, canonical result updated, CPU/GPU parity wording corrected, local LLM classified).
18. Latent graph directionality bug (CSR rows now consistently INCOMING, verified by CPU/GPU/directionality tests and CSR gate).
19. Biological-edge-semantics and REAL-annotation honesty gates added/fixed; `weight_transform`/`selection_strategy`/`sampling_bias` metadata completed.
20. Plasticity causal effect made observable and tested (reward 0 no change; sign-matched LTP/LTD).
21. Cache version/metadata stale acceptance gates (loader now enforces metadata presence).

## 6. Proof locations
- `diagnostics/acceptance_matrix.json` (32/32), `diagnostics/cpu_gpu_validation_report.json` (9/9),
  `diagnostics/alife_experiments/alife_p6_t60_s7.json`, `diagnostics/alife_baseline.json`,
  `diagnostics/benchmarks/`, `diagnostics/campaigns/`, `docs/alife_architecture.md`.
- Test files: `tests/test_suite.py`, `tests/test_alife.py` (12), `tests/test_provenance.py` (6),
  `tests/test_lif.py` (11), `tests/test_vulkan_lifecycle.py` (5), `tests/test_runtime_continuation.py` (3),
  `tests/test_evolution_robustness.py` (3), `tests/test_memory_robustness.py` (5),
  `tests/test_engine_stress.py` (1), `tests/test_ui_api.py` (10), `tests/test_llm_local.py` (4),
  `tests/test_adversarial.py` (13), `tests/test_campaign.py` (2), `tests/test_directionality.py` (8),
  `tests/test_plasticity_causal.py` (4), `tests/test_parity.py` (9).
