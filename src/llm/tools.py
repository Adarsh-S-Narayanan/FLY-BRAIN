"""Scientist tool bindings: read-only inspection + bounded experiments (REAL).

Every tool executes against live subsystems and returns measured data.
The LLM never writes neural state; the only stateful tool (run_experiment)
goes through ExperimentManager with full provenance.
"""
from typing import Any, Dict

from src.llm.scientist import ToolSpec


def build_toolset(get_engine=None, get_colony=None, memory=None) -> Dict[str, ToolSpec]:
    tools: Dict[str, ToolSpec] = {}

    def inspect_brain_state(_p: Dict[str, Any]) -> Dict[str, Any]:
        if get_engine is None:
            return {"error": "no engine bound"}
        return get_engine().get_telemetry_payload()

    def inspect_provenance(_p: Dict[str, Any]) -> Dict[str, Any]:
        if get_engine is None:
            return {"error": "no engine bound"}
        eng = get_engine()
        return {"graph_mode": eng.circuit.mode.value,
                "provenance_status": eng.circuit.provenance_status.value,
                "graph_hash": eng.circuit.graph_hash,
                "selection_strategy": eng.circuit.provenance_metadata.get("selection_strategy"),
                "csr_convention": eng.circuit.provenance_metadata.get("csr_convention")}

    def query_memory(p: Dict[str, Any]) -> Dict[str, Any]:
        if memory is None:
            return {"error": "no memory bound"}
        limit = max(1, min(int(p.get("limit", 3)), 10))
        return {"episodes": memory.get_recent_episodes(limit=limit),
                "skills": memory.get_skills()}

    def run_bounded_experiment(p: Dict[str, Any]) -> Dict[str, Any]:
        from src.experiment.manager import ExperimentManager
        from src.connectome.types import GraphMode
        mgr = ExperimentManager()
        man = mgr.run_experiment(
            experiment_id=None,
            seed=int(p.get("seed", 42)),
            graph_mode=GraphMode.SYNTHETIC_TEST,
            neuron_scale=max(16, min(int(p.get("scale", 64)), 128)),
            duration_steps=max(5, min(int(p.get("steps", 15)), 30)),
            use_gpu=False,
        )
        return {"experiment_id": man.experiment_id,
                "final_state_hash": man.final_state_hash,
                "graph_provenance": man.graph_provenance,
                "metrics": man.metrics}

    def inspect_population(_p: Dict[str, Any]) -> Dict[str, Any]:
        if get_colony is None:
            return {"error": "no colony bound"}
        pop = get_colony()
        return {"tick": pop.tick, "living": len(pop.living()), "total": len(pop.organisms),
                "population_hash": pop.population_hash(),
                "teaching_sessions": len(pop.teaching_sessions)}

    tools["inspect_brain_state"] = ToolSpec("inspect_brain_state",
        "Read live brain telemetry (authoritative simulation state).", {}, inspect_brain_state)
    tools["inspect_provenance"] = ToolSpec("inspect_provenance",
        "Read connectome provenance and selection metadata.", {}, inspect_provenance)
    tools["query_memory"] = ToolSpec("query_memory",
        "Retrieve recent episodes and skills with provenance.",
        {"limit": "integer 1..10"}, query_memory)
    tools["run_bounded_experiment"] = ToolSpec("run_bounded_experiment",
        "Run a small bounded synthetic experiment (scale<=128, steps<=30, CPU).",
        {"seed": "integer", "scale": "integer", "steps": "integer"}, run_bounded_experiment)
    tools["inspect_population"] = ToolSpec("inspect_population",
        "Read live colony summary.", {}, inspect_population)
    return tools
