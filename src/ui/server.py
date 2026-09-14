import os
import sys
import json
import time
import asyncio
import platform
import psutil
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from dataclasses import asdict
from typing import Dict, Any, Optional, List

from src.connectome.types import GraphMode, ProvenanceStatus
from src.brain.simulation_engine import SimulationEngine
from src.experiment.manager import ExperimentManager, get_file_sha256, get_git_commit

app = FastAPI(title="FlyBrain Lab — Biological Connectome Research Platform")

# Central Simulation Engine instance
SIMULATION_ENGINE: Optional[SimulationEngine] = None
EXPERIMENT_MGR = ExperimentManager()

def get_engine() -> SimulationEngine:
    global SIMULATION_ENGINE
    if SIMULATION_ENGINE is None:
        SIMULATION_ENGINE = SimulationEngine(
            circuit_size=512,
            graph_mode=GraphMode.REAL,
            use_gpu=True,
            seed=42
        )
    return SIMULATION_ENGINE

@app.on_event("startup")
async def startup_event():
    engine = get_engine()
    engine.set_event_loop(asyncio.get_event_loop())

@app.on_event("shutdown")
def shutdown_event():
    if SIMULATION_ENGINE:
        SIMULATION_ENGINE.close()

# Static directories
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs("visual_evidence", exist_ok=True)
app.mount("/visual_evidence", StaticFiles(directory="visual_evidence"), name="visual_evidence")

@app.get("/")
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>FlyBrain Lab is initializing...</h1>")

@app.get("/api/health")
def get_health():
    engine = get_engine()
    return {
        "status": "HEALTHY",
        "timestamp": time.time(),
        "graph_mode": engine.circuit.mode.value,
        "provenance_status": engine.circuit.provenance_status.value,
        "backend": "vulkan_gpu" if (engine.brain.gpu_engine and engine.brain.use_gpu) else "cpu_reference",
        "device_name": engine.brain.gpu_engine.device_name if engine.brain.gpu_engine else "CPU Reference Mode"
    }

@app.get("/api/state")
def get_state():
    engine = get_engine()
    return engine.get_full_state()

@app.get("/api/telemetry")
def get_telemetry():
    engine = get_engine()
    return engine.get_telemetry_payload()

@app.get("/api/connectome")
def get_connectome(max_nodes: int = 512, max_edges: int = 384):
    engine = get_engine()
    return engine.get_connectome_3d_view(max_nodes=max_nodes, max_edges=max_edges)

@app.post("/api/simulation/start")
def post_start():
    engine = get_engine()
    engine.start()
    return {"status": "STARTED", "is_running": True}

@app.post("/api/simulation/pause")
def post_pause():
    engine = get_engine()
    engine.pause()
    return {"status": "PAUSED", "is_running": False}

class StepRequest(BaseModel):
    steps: int = 1
    sensory_inputs: Optional[Dict[str, List[float]]] = None
    reward: float = 0.0

@app.post("/api/simulation/step")
def post_step(req: StepRequest):
    engine = get_engine()
    s_in = None
    if req.sensory_inputs:
        s_in = {k: np.array(v, dtype=np.float32) for k, v in req.sensory_inputs.items()}
    res = engine.step_single(n_steps=req.steps, sensory_inputs=s_in, reward=req.reward)
    return res

class ResetRequest(BaseModel):
    circuit_size: int = 512
    graph_mode: str = "REAL"
    seed: int = 42

@app.post("/api/simulation/reset")
def post_reset(req: ResetRequest):
    global SIMULATION_ENGINE
    if SIMULATION_ENGINE:
        SIMULATION_ENGINE.close()
    mode = GraphMode(req.graph_mode)
    SIMULATION_ENGINE = SimulationEngine(
        circuit_size=req.circuit_size,
        graph_mode=mode,
        use_gpu=True,
        seed=req.seed
    )
    SIMULATION_ENGINE.set_event_loop(asyncio.get_event_loop())
    return {"status": "RESET_COMPLETE", "graph_mode": mode.value, "neurons": req.circuit_size}

@app.get("/api/memory")
def get_memory(query: Optional[str] = None, limit: int = 10):
    engine = get_engine()
    mem = engine.memory
    episodes = mem.get_recent_episodes(limit=limit)
    skills = mem.get_skills()
    dreams = mem.get_recent_dreams(limit=limit)
    
    if query:
        # Filter episodes containing query
        episodes = [e for e in episodes if query.lower() in json.dumps(e).lower()]
        
    return {
        "working": mem.working.get_all_active(),
        "episodes": episodes,
        "skills": skills,
        "dreams": dreams
    }

@app.get("/api/evolution/lineage")
def get_evolution_lineage():
    engine = get_engine()
    return engine.evolution.get_lineage()

@app.post("/api/evolution/generation")
def post_evolution_generation(num_candidates: int = 4):
    engine = get_engine()
    with engine.lock:
        res = engine.evolution.run_generation(num_candidates=num_candidates)
    return res

@app.get("/api/dreams")
def get_dreams(limit: int = 10):
    engine = get_engine()
    return engine.memory.get_recent_dreams(limit=limit)

@app.post("/api/dreams/replay")
def post_dream_replay(mode: str = "deterministic", count: int = 2):
    engine = get_engine()
    with engine.lock:
        res = engine.dream_engine.run_dream_cycle(mode=mode, seed=int(time.time()), num_episodes_to_replay=count)
    return res

@app.get("/api/tools")
def get_tools():
    engine = get_engine()
    return engine.trainer.registry.list_tools()

class ToolExecRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any] = {}

@app.post("/api/tools/execute")
def post_execute_tool(req: ToolExecRequest):
    engine = get_engine()
    return engine.trainer.registry.execute(req.tool_name, req.params)

@app.get("/api/experiments")
def list_experiments():
    exp_dir = EXPERIMENT_MGR.exp_dir
    files = [f for f in os.listdir(exp_dir) if f.endswith(".json")]
    manifests = []
    for f in sorted(files, reverse=True)[:20]:
        try:
            with open(os.path.join(exp_dir, f), "r", encoding="utf-8") as fp:
                manifests.append(json.load(fp))
        except Exception:
            pass
    return manifests

class RunExpRequest(BaseModel):
    experiment_id: Optional[str] = None
    seed: int = 42
    graph_mode: str = "REAL"
    neuron_scale: int = 256
    duration_steps: int = 50

@app.post("/api/experiments/run")
def post_run_experiment(req: RunExpRequest):
    mode = GraphMode(req.graph_mode)
    manifest = EXPERIMENT_MGR.run_experiment(
        experiment_id=req.experiment_id,
        seed=req.seed,
        graph_mode=mode,
        neuron_scale=req.neuron_scale,
        duration_steps=req.duration_steps
    )
    return asdict(manifest)

@app.post("/api/experiments/verify")
def post_verify_experiment(experiment_id: str):
    res = EXPERIMENT_MGR.verify_experiment(experiment_id)
    return res

@app.get("/api/diagnostics")
def get_diagnostics():
    engine = get_engine()
    brain = engine.brain
    gpu_diag = brain.gpu_engine.get_diagnostics() if brain.gpu_engine else {
        "status": "UNAVAILABLE",
        "device_name": "CPU Reference Mode (Headless / No Vulkan Device)"
    }

    vm = psutil.virtual_memory()
    return {
        "system": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_available_gb": round(vm.available / (1024**3), 2),
            "ram_percent": vm.percent,
            "git_commit": get_git_commit()
        },
        "vulkan": gpu_diag,
        "dataset_provenance": {
            "dataset_name": "Janelia MaleCNS",
            "version": "male-cns:v1.0",
            "soma_sha256": get_file_sha256(os.path.join("malecns", "data-raw", "2023-27-2 soma_sides.csv")),
            "connections_sha256": get_file_sha256(os.path.join("malecns", "data-raw", "malecns_v1_0_connections.csv")),
            "brain_shader_sha256": get_file_sha256(os.path.join("shaders", "brain_step.spv")),
            "plasticity_shader_sha256": get_file_sha256(os.path.join("shaders", "plasticity.spv"))
        },
        "runtime": {
            "is_running": engine.is_running,
            "circuit_neurons": engine.circuit.num_neurons,
            "circuit_synapses": engine.circuit.num_synapses,
            "graph_mode": engine.circuit.mode.value,
            "graph_hash": engine.circuit.graph_hash,
            "step_count": brain.state.step_count,
            "total_spikes": brain.state.total_spikes,
            "last_step_latency_ms": engine.last_step_time_ms
        }
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    """
    Real-time streaming telemetry WebSocket.
    Clients receive updates from the SimulationEngine broadcast queue.
    Clients do NOT independently advance or step the brain.
    """
    await ws.accept()
    engine = get_engine()
    queue = asyncio.Queue(maxsize=10)
    engine.register_telemetry_queue(queue)
    try:
        # Send initial state immediately
        await ws.send_json(engine.get_telemetry_payload())
        while True:
            payload = await queue.get()
            await ws.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        engine.unregister_telemetry_queue(queue)
