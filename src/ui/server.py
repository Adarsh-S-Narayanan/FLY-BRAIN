import os
import sys
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from src.connectome.loader import get_or_create_circuit
from src.brain.runtime import BrainRuntime
from src.memory.persistence import PersistentMemoryManager
from src.trainer.curriculum import CurriculumTrainer
from src.evolution.scheduler import EvolutionScheduler
from src.dream.engine import DreamEngine

app = FastAPI(title="FlyBrain Autonomous Organism Dashboard")

# Global runtime state
STATE = {
    "circuit": None,
    "brain": None,
    "memory": None,
    "trainer": None,
    "evolution": None,
    "dream_engine": None,
    "is_running": False
}

def init_app_state():
    if STATE["circuit"] is None:
        circuit = get_or_create_circuit(512, cache_name="flybrain_ui_512.npz")
        STATE["circuit"] = circuit
        memory = PersistentMemoryManager(db_path="diagnostics/flybrain_live_memory.db")
        STATE["memory"] = memory
        brain = BrainRuntime(circuit, use_gpu=True, seed=42)
        STATE["brain"] = brain
        STATE["trainer"] = CurriculumTrainer(brain, memory)
        STATE["evolution"] = EvolutionScheduler(circuit, history_file="diagnostics/evolution_history.json")
        STATE["dream_engine"] = DreamEngine(brain, memory)

init_app_state()

# Serve static directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs("visual_evidence", exist_ok=True)
app.mount("/visual_evidence", StaticFiles(directory="visual_evidence"), name="visual_evidence")

@app.get("/")
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>FlyBrain Dashboard loading...</h1>")

@app.get("/api/state")
def get_state():
    brain: BrainRuntime = STATE["brain"]
    return {
        "step": brain.state.step_count,
        "total_spikes": brain.state.total_spikes,
        "mean_activation": float(np.mean(brain.state.activations)),
        "max_activation": float(np.max(brain.state.activations)),
        "prediction_error": brain.state.prediction_error,
        "current_reward": brain.state.current_reward,
        "predicted_reward": brain.state.predicted_reward,
        "active_goal": brain.state.active_goal,
        "tool_associations": brain.state.tool_associations,
        "drives": {
            "energy": brain.state.drives.energy,
            "curiosity": brain.state.drives.curiosity,
            "social": brain.state.drives.social,
            "integrity": brain.state.drives.integrity
        },
        "backend": "vulkan_gpu" if (brain.gpu_engine and brain.use_gpu) else "cpu_reference",
        "gpu_device": brain.gpu_engine.device_name if brain.gpu_engine else "N/A",
        "num_neurons": brain.graph.num_neurons,
        "num_synapses": brain.graph.num_synapses
    }

@app.get("/api/connectome")
def get_connectome():
    brain: BrainRuntime = STATE["brain"]
    graph = brain.graph
    
    # Send sampled nodes with coordinates and live activation
    nodes = []
    coords_min = graph.coordinates.min(axis=0)
    coords_max = graph.coordinates.max(axis=0)
    norm_coords = (graph.coordinates - coords_min) / (coords_max - coords_min + 1e-5)

    for i in range(min(512, graph.num_neurons)):
        nodes.append({
            "id": int(graph.neuron_ids[i]),
            "idx": i,
            "pos": [round(float(c), 3) for c in norm_coords[i]],
            "side": graph.sides[i],
            "tbars": int(graph.tbars[i]),
            "act": round(float(brain.state.activations[i]), 3)
        })

    # Sample top active connections
    edges = []
    for i in range(min(128, graph.num_neurons)):
        start = graph.row_offsets[i]
        end = min(start + 4, graph.row_offsets[i+1])
        for k in range(start, end):
            target = int(graph.col_indices[k])
            if target < min(512, graph.num_neurons):
                edges.append({
                    "src": i,
                    "tgt": target,
                    "w": round(float(graph.weights[k]), 2)
                })

    return {"nodes": nodes, "edges": edges}

@app.post("/api/step")
def post_step(steps: int = 1):
    brain: BrainRuntime = STATE["brain"]
    results = []
    for _ in range(steps):
        # Simulate slight sensory stimulation
        sensory = {"visual": np.random.uniform(0.1, 0.3, 64).astype(np.float32)}
        out = brain.step(sensory_inputs=sensory, reward=0.1)
        results.append(out)
    return results[-1] if results else {}

@app.post("/api/train")
def post_train(tool: str = "speak"):
    trainer: CurriculumTrainer = STATE["trainer"]
    res = trainer.train_tool_selection_skill(target_tool=tool, num_trials=5)
    return res

@app.post("/api/dream")
def post_dream():
    dream_engine: DreamEngine = STATE["dream_engine"]
    res = dream_engine.run_dream_cycle(mode="exploratory", seed=42, num_episodes_to_replay=2)
    return res

@app.post("/api/evolve")
def post_evolve():
    evo: EvolutionScheduler = STATE["evolution"]
    res = evo.run_generation(num_candidates=4)
    return res

@app.get("/api/memory")
def get_memory():
    mem: PersistentMemoryManager = STATE["memory"]
    return {
        "working": mem.working.get_all_active(),
        "episodes": mem.get_recent_episodes(limit=8),
        "skills": mem.get_skills(),
        "dreams": mem.get_recent_dreams(limit=5)
    }

@app.get("/api/tools")
def get_tools():
    trainer: CurriculumTrainer = STATE["trainer"]
    return trainer.registry.list_tools()

@app.post("/api/tools/execute")
def execute_tool(tool_name: str, params: Dict[str, Any] = None):
    trainer: CurriculumTrainer = STATE["trainer"]
    return trainer.registry.execute(tool_name, params or {})

@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    await ws.accept()
    brain: BrainRuntime = STATE["brain"]
    try:
        while True:
            # Step brain in background if active
            sensory = {"visual": np.random.uniform(0.1, 0.3, 64).astype(np.float32)}
            brain.step(sensory_inputs=sensory, reward=0.05)
            
            payload = {
                "step": brain.state.step_count,
                "spikes": int(np.sum(brain.state.activations > 0.5)),
                "mean_act": round(float(np.mean(brain.state.activations)), 3),
                "prediction_error": round(brain.state.prediction_error, 3),
                "energy": round(brain.state.drives.energy, 3),
                "curiosity": round(brain.state.drives.curiosity, 3),
                "social": round(brain.state.drives.social, 3),
                "tool_scores": {k: round(v, 3) for k, v in brain.state.tool_associations.items()},
                "active_neurons": [
                    {"idx": int(i), "act": round(float(brain.state.activations[i]), 2)}
                    for i in np.argsort(brain.state.activations)[-10:]
                ]
            }
            await ws.send_json(payload)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
