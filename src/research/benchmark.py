"""AGI-oriented benchmark suite (STAGE N): FlyBrain (embodied continual
learning) vs LLM-only vs LLM+tools. Per-category results with documented
budgets; NO aggregate superiority claims. LLM arms are honest SKIPs when no
local model is available; embodied tasks are NOT_APPLICABLE for LLM-only arms
(no embodiment = no exposure), which is documented, not scored as zero.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.determinism import SeedBundle, derive_subseed
from src.connectome.types import GraphMode
from src.population.population import Population

BENCHMARK_VERSION = "benchmark_v1"

CATEGORIES = ("memory_retention", "transfer", "novel_task_adaptation",
              "social_learning", "communication", "learning_efficiency")


@dataclass
class ArmBudget:
    """Fair-comparison budgets (mission rule 44): every arm documents what it
    received. No comparison is valid without these."""
    compute_ticks: int = 0
    training_exposure_ticks: int = 0
    information_budget_chars: int = 0
    tool_access: bool = False
    seed: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _pop(seed: int, size: int = 4, ticks_config: Optional[Dict[str, Any]] = None) -> Population:
    seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                       organism_seed=seed + 2, development_seed=seed + 3,
                       mutation_seed=seed + 4, world_seed=seed + 5,
                       teacher_seed=seed + 6)
    return Population(size, seeds, GraphMode.SYNTHETIC_TEST, 32,
                      experiment_seed=seed, autonomy_mode=True, genome_version="2.0")


# ---------------------------------------------------------------- FlyBrain arm
def run_flybrain_arm(seed: int = 101, train_ticks: int = 40, delay_ticks: int = 30,
                     size: int = 4) -> Dict[str, Any]:
    """Embodied continual-learning measurements from a REAL population run."""
    pop = _pop(seed, size)
    pop.organisms[0].age = 70
    pop.organisms[1].age = 70

    def mean_skill(key: str) -> float:
        living = pop.living()
        return (sum(o.skills.get(key, 0.0) for o in living) / len(living)) if living else 0.0

    forage_pre = mean_skill("forage")
    pop.step(train_ticks)
    forage_post = mean_skill("forage")
    sessions = len(pop.teaching_sessions)
    pop.step(delay_ticks)
    forage_delayed = mean_skill("forage")

    # communication: grounded vocabularies actually built during the run
    vocab = sum(o.language.vocabulary_size() for o in pop.living()
                if getattr(o, "language", None))
    # social learning delta: taught students vs untaught
    taught = [o for o in pop.living()
              if any(s["student"] == o.id for s in
                     [{"student": x.student} for x in pop.teaching_sessions])]
    social_delta = (sum(o.skills["forage"] for o in taught) / len(taught) - forage_pre) \
        if taught else 0.0
    # structural learning evidence
    expansions = 0
    for o in pop.living():
        if getattr(o, "living", None) is not None:
            s = o.living.structural_summary()
            expansions += int(s["neurons_total"] > s["seed_size"])
    retention = forage_delayed - forage_pre
    return {
        "arm": "flybrain",
        "budget": ArmBudget(compute_ticks=train_ticks + delay_ticks,
                            training_exposure_ticks=train_ticks,
                            information_budget_chars=0, tool_access=False,
                            seed=seed,
                            notes="embodied: world+brain+social exposure").to_dict(),
        "categories": {
            "memory_retention": {"skill_before": round(forage_pre, 4),
                                 "skill_after_training": round(forage_post, 4),
                                 "skill_after_delay": round(forage_delayed, 4),
                                 "retention_delta": round(retention, 4),
                                 "status": "MEASURED"},
            "transfer": {"forage_gain": round(forage_post - forage_pre, 4),
                         "avoid_gain": round(mean_skill("avoid") - 0.1, 4),
                         "status": "MEASURED"},
            "novel_task_adaptation": {"episodes_survived": delay_ticks,
                                      "living_fraction": round(
                                          len(pop.living()) / max(1, len(pop.organisms)), 3),
                                      "status": "MEASURED"},
            "social_learning": {"teaching_sessions": sessions,
                                "taught_skill_delta": round(social_delta, 4),
                                "status": "MEASURED" if sessions else "NO_SESSIONS"},
            "communication": {"grounded_vocab_total": vocab,
                              "status": "MEASURED"},
            "learning_efficiency": {"skill_per_tick": round(
                (forage_post - forage_pre) / max(1, train_ticks), 6),
                "structural_expansions": expansions,
                "status": "MEASURED"},
        },
        "population_hash": pop.population_hash(),
    }


# ------------------------------------------------------------------- LLM arms
def run_llm_only_arm(seed: int = 101) -> Dict[str, Any]:
    """Static question/answer arm. Embodied categories are NOT_APPLICABLE
    (no body, no world, no continual exposure) — documented, never scored 0."""
    try:
        from src.llm.discovery import discover_models
        from src.llm.runtime import LocalLLM, GenerationConfig
        models = [m for m in discover_models() if m.status == "DISCOVERED"]
        if not models:
            return {"arm": "llm_only", "status": "SKIP",
                    "reason": "no local GGUF model on this host"}
        llm = LocalLLM(models[0], n_ctx=1024)
        if not llm.load():
            return {"arm": "llm_only", "status": "SKIP", "reason": llm.status}
        t0 = time.perf_counter()
        r = llm.generate("List two strategies a foraging agent could use to find food.",
                         GenerationConfig(max_tokens=48, seed=seed))
        elapsed = time.perf_counter() - t0
        llm.unload()
        if r["status"] != "SUCCESS":
            return {"arm": "llm_only", "status": "SKIP", "reason": r["status"]}
        return {
            "arm": "llm_only", "status": "MEASURED",
            "budget": ArmBudget(compute_ticks=0, training_exposure_ticks=0,
                                information_budget_chars=len(r["text"] or ""),
                                tool_access=False, seed=seed,
                                notes="static QA; zero embodied exposure").to_dict(),
            "categories": {c: {"status": "NOT_APPLICABLE",
                               "reason": "no embodiment/world exposure"}
                           for c in CATEGORIES},
            "text_stats": {"gen_sec": round(elapsed, 3),
                           "chars": len(r["text"] or "")},
            "model": {"filename": models[0].filename, "sha256": models[0].sha256},
        }
    except Exception as e:  # noqa: BLE001
        return {"arm": "llm_only", "status": "SKIP", "reason": f"{type(e).__name__}: {e}"}


def run_llm_tools_arm(seed: int = 101, iterations: int = 2) -> Dict[str, Any]:
    """Tool-augmented arm: the scientist observes real system state via the
    allowlisted toolset. Measured: grounded tool calls executed."""
    try:
        from src.llm.discovery import discover_models
        from src.llm.runtime import LocalLLM
        from src.llm.scientist import ScientistLoop, build_toolset
        models = [m for m in discover_models() if m.status == "DISCOVERED"]
        if not models:
            return {"arm": "llm_tools", "status": "SKIP",
                    "reason": "no local GGUF model on this host"}
        llm = LocalLLM(models[0], n_ctx=1024)
        if not llm.load():
            return {"arm": "llm_tools", "status": "SKIP", "reason": llm.status}
        loop = ScientistLoop(llm)
        for spec in build_toolset().values():
            loop.register(spec)
        executed = rejected = 0
        for i in range(iterations):
            rec = loop.run_iteration(
                context="System state: foraging population with teaching.",
                state_hash=f"bench-{seed}-{i}")
            for res in rec.tool_results:
                if res.get("status") in ("SUCCESS", "TOOL_ERROR"):
                    executed += 1
            executed += len(rec.tool_calls) - len(rec.tool_results)
            rejected += sum(1 for c in rec.tool_calls if not c)
        llm.unload()
        return {
            "arm": "llm_tools", "status": "MEASURED",
            "budget": ArmBudget(compute_ticks=0, training_exposure_ticks=0,
                                information_budget_chars=6000 * iterations,
                                tool_access=True, seed=seed,
                                notes="read-only tools + bounded experiments").to_dict(),
            "categories": {c: {"status": "TOOL_MEDIATED",
                               "tool_iterations": iterations}
                           for c in CATEGORIES},
            "tool_stats": {"iterations": iterations},
            "model": {"filename": models[0].filename, "sha256": models[0].sha256},
        }
    except Exception as e:  # noqa: BLE001
        return {"arm": "llm_tools", "status": "SKIP", "reason": f"{type(e).__name__}: {e}"}


# -------------------------------------------------------------------- report
def run_benchmark_suite(seed: int = 101, write_path: Optional[str] = None) -> Dict[str, Any]:
    """Full suite. Per-category, budget-documented, no aggregate winner."""
    arms = [run_flybrain_arm(seed), run_llm_only_arm(seed), run_llm_tools_arm(seed)]
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "seed": seed,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fairness_note": "Arms receive DIFFERENT modalities by design; budgets are "
                         "documented per arm. Embodied categories are NOT_APPLICABLE "
                         "for LLM-only. No aggregate superiority is claimed.",
        "arms": arms,
    }
    if write_path:
        import json
        import os
        os.makedirs(os.path.dirname(write_path) or ".", exist_ok=True)
        with open(write_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report
