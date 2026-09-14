# FlyBrain Artificial-Life Layer — Implementation Status (2026-09-14)

> Scientific positioning: FlyBrain is a **biologically grounded artificial-life research
> framework**, not a literal brain model. Research question: *What structures and behaviors
> emerge when a biologically grounded neural substrate develops, learns, socializes,
> transmits knowledge, reproduces, mutates, and evolves across overlapping generations?*

## IMPLEMENTED + VERIFIED (`tests/test_alife.py` 12/12, matrix gates 20–24)

| Layer | Module | Evidence |
|---|---|---|
| Deterministic identity/seeds | `src/common/determinism.py` | stable SHA-256 IDs, derived subseeds |
| Event sourcing (26 types) | `src/common/events.py` | hash-chained, JSONL serializable |
| State hashing | `src/common/hashing.py` | array/dict hashes |
| Genome v1.0 (15 params) | `src/genome/schema.py` | validate/hash/serialize, bounds-checked |
| Mutation + crossover | `src/genome/operators.py` | deterministic, full provenance records |
| Development (birth/diff/migrate/grow/synaptogenesis/prune/apoptosis) | `src/development/engine.py` | graph invariants enforced after every op |
| Grid world + sensors/actuators | `src/world/environment.py` | grazing, hazards, regrowth w/ tracked influx; config in snapshots |
| Organism + lifecycle + metabolism | `src/organism/organism.py` | stages, energy books, sleep/dream replay; maps+drives in snapshots |
| Population + overlapping generations | `src/population/population.py` | coexistence, Pareto selection, lineage; branch-replay tested |
| Teaching + cultural transmission | `src/culture/transmission.py` | gain measured, provenance chains |
| Scripted teacher protocol | `src/agents/protocols.py` | acts only via cultural channel |
| Canonical experiment + verify | `scripts/run_alife_experiment.py` | bit-exact replay (hash match) |
| Colony UI API | `src/ui/server.py` (`/api/colony*`) | serves real live state, tested |

Canonical result (`seed 7, pop 10, 150 ticks`): 8 living, 4 births, 6 deaths,
generations `[0, 1]` coexisting, 131 teaching sessions, deterministic replay hash
match `True` (`70a5e4b29e06c320`).

## EXPERIMENTAL

- Population simulation runs on **CPU** (single-brain Vulkan path is verified with
  trajectory parity, but batched multi-organism GPU stepping is not implemented).
- Dream consolidation is replay-based; no synaptic downscaling model yet.
- Long runs beyond the canonical 150-tick campaign are untested.

## NOT IMPLEMENTED (explicitly unavailable, never faked)

- LLM/VLM scientist loop (`UnavailableLLMTeacher` + `CognitiveTrainer` report
  `MODEL_UNAVAILABLE`/`RULE_BASED` honestly).
- 3D colony / development-timeline UI panels (data APIs exist; visualization pending).
- 100-generation checkpointed campaigns (scheduler checkpoints exist and are tested
  for exact resume, but no 100-gen campaign has been run).

## Key model parameters (documented, not hidden)

- Metabolism: `cost = rate*(0.5 + activity + 0.25*move)`; growth `0.02` energy/neuron;
  grazing radius 1, assimilation `0.8`; reproduction cost `0.25` energy/parent.
- Ecology: resources scale `max(12, 4*pop)`, regrowth every 3 ticks (+0.4–0.8, tracked
  in `energy_injected`); energy invariant tested as
  `total <= initial + injected`.
- Plasticity inside organism brains never fires in current runs (reward passed as 0.0);
  organism learning is behavioral (foraging/social skills) + cultural.
