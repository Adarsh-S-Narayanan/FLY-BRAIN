"""Milestone evidence capture (STAGE M evidence artifact generator).

Runs a REAL autonomous v2 population through a deep-time campaign, detects
milestones, escalates a checkpoint and writes machine-readable evidence.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from src.common.determinism import SeedBundle
from src.connectome.types import GraphMode
from src.population.population import Population
from src.timeline.deeptime import DeepTimeRunner, DeepTimeConfig
from src.science.milestones import detect_milestones, milestone_certificate
from src.research.benchmark import run_benchmark_suite
from src.common.provenance import build_layers, experiment_fingerprint


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def main(seed: int = 301, coarse_steps: int = 6, size: int = 5):
    seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                       organism_seed=seed + 2, development_seed=seed + 3,
                       mutation_seed=seed + 4, world_seed=seed + 5,
                       teacher_seed=seed + 6)
    pop = Population(size, seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=seed,
                     autonomy_mode=True, genome_version="2.0")
    runner = DeepTimeRunner(pop, DeepTimeConfig(coarse_ticks_per_step=20,
                                                offspring_per_generation=2))
    milestones_all = []
    t0 = time.time()
    for _ in range(coarse_steps):
        runner.fast_forward(1, milestone_fn=detect_milestones)
        fresh = detect_milestones(pop)
        milestones_all.extend(fresh)
    escalation = runner.escalate("evidence_checkpoint")
    replay = runner.replay_high_resolution("evidence_checkpoint", ticks=5)
    suite = run_benchmark_suite(seed=seed)

    # dedupe milestones by (milestone, tick)
    seen = set()
    uniq = []
    for m in milestones_all:
        key = (m["milestone"], m["tick"])
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    certificates = [milestone_certificate(m, seed) for m in uniq]

    report = {
        "artifact": "milestone_evidence_v1",
        "seed": seed,
        "git_commit": git_commit(),
        "captured_ts": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"population_size": size, "coarse_steps": coarse_steps,
                   "coarse_ticks_per_step": 20,
                   "approximation_model": runner.ledger["approximation_model"]},
        "ledger": runner.ledger_summary(),
        "events": [e.to_dict() for e in runner.events[:50]],
        "milestones": uniq,
        "certificates": certificates,
        "escalation": {"checkpoint": "evidence_checkpoint",
                       "population_hash": escalation["population_hash"],
                       "sim_tick": escalation["sim_tick"]},
        "replay_verification": replay,
        "benchmark_suite": suite,
        "provenance_layers": build_layers(
            population=pop, world=pop.world, event_log=pop.events),
        "experiment_fingerprint": experiment_fingerprint(build_layers(
            population=pop, world=pop.world, event_log=pop.events)),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out = "diagnostics/milestones/milestone_evidence.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({
        "coarse_steps": report["ledger"]["coarse_steps"],
        "sim_ticks": report["ledger"]["sim_ticks"],
        "generations": report["ledger"]["generations_seen"],
        "milestones": [m["milestone"] for m in uniq],
        "replay_verified": replay.get("checkpoint_hash_verified"),
        "fingerprint": report["experiment_fingerprint"][:16],
        "artifact": out}, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--coarse-steps", type=int, default=6)
    ap.add_argument("--size", type=int, default=5)
    a = ap.parse_args()
    main(a.seed, a.coarse_steps, a.size)
