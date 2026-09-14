# FlyBrain Artificial-Life Layer — Implementation Status

> Scientific positioning: FlyBrain is a **biologically grounded artificial-life research
> framework**, not a literal brain model. Research question: *What structures and behaviors
> emerge when a biologically grounded neural substrate develops, learns, socializes,
> transmits knowledge, reproduces, mutates, and evolves across overlapping generations?*

## IMPLEMENTED + VERIFIED (tests in `tests/test_alife.py`, 10/10 pass)

| Layer | Module | Evidence |
|---|---|---|
| Deterministic identity/seeds | `src/common/determinism.py` | stable SHA-256 IDs, derived subseeds |
| Event sourcing (26 types) | `src/common/events.py` | hash-chained, JSONL serializable |
| State hashing | `src/common/hashing.py` | array/dict hashes |
| Genome v1.0 (15 params) | `src/genome/schema.py` | validate/hash/serialize, bounds-checked |
| Mutation + crossover | `src/genome/operators.py` | deterministic, full provenance records |
| Development (birth/diff/migrate/grow/synaptogenesis/prune/apoptosis) | `src/development/engine.py` | graph invariants enforced after every op |
| Grid world + sensors/actuators | `src/world/environment.py` | grazing, hazards, regrowth w/ tracked influx |
| Organism + lifecycle + metabolism | `src/organism/organism.py` | stages, energy books, sleep/dream replay |
| Population + overlapping generations | `src/population/population.py` | coexistence, Pareto selection, lineage |
| Teaching + cultural transmission | `src/culture/transmission.py` | gain measured, provenance chains |
| Scripted teacher protocol | `src/agents/protocols.py` | acts only via cultural channel |
| Canonical experiment + verify | `scripts/run_alife_experiment.py` | bit-exact replay (hash match) |
| Colony UI API | `src/ui/server.py` (`/api/colony*`) | serves real live state |

Canonical result (`seed 7, pop 10, 150 ticks`): 8 living, 4 births, 2 overlapping
generations, 131 teaching sessions, deterministic replay hash match `True`.

## EXPERIMENTAL

- Vulkan acceleration for populations (single-brain Vulkan path exists and is verified;
  batched multi-organism GPU stepping not yet implemented — populations run CPU).
- Dream consolidation is replay-based; no synaptic downscaling model yet.

## PLANNED / NOT IMPLEMENTED

- LLM/VLM scientist loop (`UnavailableLLMTeacher` raises honestly — no mock).
- 3D colony/development-timeline UI panels (data APIs exist; visualization pending).
- 100-generation long-run campaign with checkpoints (script supports ticks; checkpoints pending).
- Modularity/motif-diversity complexity metrics (counts/diversity implemented).

## Key model parameters (documented, not hidden)

- Metabolism: `cost = rate*(0.5 + activity + 0.25*move)`; growth `0.02` energy/neuron;
  grazing radius 1, assimilation `0.8`; reproduction cost `0.25` energy/parent.
- Ecology: resources scale `max(12, 4*pop)`, regrowth every 3 ticks (+0.4–0.8, tracked
  in `energy_injected`); energy invariant tested as
  `total <= initial + injected`.
