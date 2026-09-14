import os
import csv
import json
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial import cKDTree
from src.connectome.types import ConnectomeGraph, NeuronMetadata

DEFAULT_SOMA_PATH = os.path.join("malecns", "data-raw", "2023-27-2 soma_sides.csv")
CACHE_DIR = os.path.join("diagnostics", "connectome_cache")

def load_raw_neurons(csv_path: str = DEFAULT_SOMA_PATH) -> List[NeuronMetadata]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MaleCNS soma file not found at: {csv_path}")
    
    neurons = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                body_id = int(row["body"])
                nucleus_id = int(row["nucleus_id"])
                nx = float(row["nx"])
                ny = float(row["ny"])
                nz = float(row["nz"])
                side = row["soma_side"].strip()
                tbars = int(row["tbars"])
                body_size = int(row["body_size"])
                
                neurons.append(NeuronMetadata(
                    body_id=body_id,
                    nucleus_id=nucleus_id,
                    x=nx,
                    y=ny,
                    z=nz,
                    side=side,
                    tbars=tbars,
                    body_size=body_size
                ))
            except (ValueError, KeyError):
                continue
    return neurons

def build_connectome_circuit(
    neurons: List[NeuronMetadata],
    max_neurons: int = 1024,
    interaction_radius: float = 6000.0,
    max_degree: int = 32,
    seed: int = 42
) -> ConnectomeGraph:
    """
    Constructs a deterministic biologically-grounded circuit from the malecns dataset.
    Selects top tbar hub neurons paired across left and right hemilaterals,
    and calculates connectivity using KD-tree spatial proximity and tbars.
    """
    rng = np.random.RandomState(seed)
    
    # Sort neurons by tbars descending to prioritize major projection and interneurons
    sorted_neurons = sorted(neurons, key=lambda n: n.tbars, reverse=True)
    
    # Balance left, right, and midline
    left_neurons = [n for n in sorted_neurons if n.side == "L"]
    right_neurons = [n for n in sorted_neurons if n.side == "R"]
    mid_neurons = [n for n in sorted_neurons if n.side == "M"]
    
    n_per_side = max_neurons // 2
    selected = []
    selected.extend(left_neurons[:n_per_side])
    selected.extend(right_neurons[:n_per_side])
    if len(selected) < max_neurons and mid_neurons:
        remaining = max_neurons - len(selected)
        selected.extend(mid_neurons[:remaining])
    
    # Sort deterministically by body_id
    selected.sort(key=lambda n: n.body_id)
    N = len(selected)
    
    neuron_ids = np.array([n.body_id for n in selected], dtype=np.int64)
    coordinates = np.array([[n.x, n.y, n.z] for n in selected], dtype=np.float32)
    tbars = np.array([n.tbars for n in selected], dtype=np.int32)
    sides = [n.side for n in selected]
    
    # Build KD-tree for spatial proximity connectivity
    tree = cKDTree(coordinates)
    
    row_offsets = [0]
    col_indices = []
    weights = []
    
    for i in range(N):
        # Query neighbors within interaction radius
        dists, indices = tree.query(coordinates[i], k=min(max_degree + 1, N), distance_upper_bound=interaction_radius)
        
        # Filter out self-connection and infs
        valid_targets = []
        for d, j in zip(dists, indices):
            if j < N and j != i and not np.isinf(d):
                # Connection weight is proportional to presynaptic tbars and inverse distance
                # Normalized to initial baseline [0.05, 0.5]
                dist_factor = np.exp(- (d / (interaction_radius * 0.5)) ** 2)
                tbar_factor = np.log1p(tbars[i]) / (np.log1p(np.max(tbars)) + 1e-5)
                # Random variation based on deterministic seed
                w = float(0.1 + 0.4 * (dist_factor * 0.7 + tbar_factor * 0.3) * (0.8 + 0.4 * rng.rand()))
                valid_targets.append((int(j), w))
        
        # Sort targets by index for deterministic CSR layout
        valid_targets.sort(key=lambda x: x[0])
        
        for target_idx, w in valid_targets:
            col_indices.append(target_idx)
            weights.append(w)
            
        row_offsets.append(len(col_indices))
        
    row_offsets = np.array(row_offsets, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    weights = np.array(weights, dtype=np.float32)
    
    return ConnectomeGraph(
        neuron_ids=neuron_ids,
        coordinates=coordinates,
        tbars=tbars,
        sides=sides,
        row_offsets=row_offsets,
        col_indices=col_indices,
        weights=weights
    )

def save_connectome_cache(graph: ConnectomeGraph, cache_path: str):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        neuron_ids=graph.neuron_ids,
        coordinates=graph.coordinates,
        tbars=graph.tbars,
        sides=np.array(graph.sides),
        row_offsets=graph.row_offsets,
        col_indices=graph.col_indices,
        weights=graph.weights
    )

def load_connectome_cache(cache_path: str) -> ConnectomeGraph:
    data = np.load(cache_path, allow_pickle=True)
    return ConnectomeGraph(
        neuron_ids=data["neuron_ids"],
        coordinates=data["coordinates"],
        tbars=data["tbars"],
        sides=list(data["sides"]),
        row_offsets=data["row_offsets"],
        col_indices=data["col_indices"],
        weights=data["weights"]
    )

def get_or_create_circuit(
    max_neurons: int = 1024,
    cache_name: str = "flybrain_circuit_1024.npz"
) -> ConnectomeGraph:
    cache_path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        return load_connectome_cache(cache_path)
    
    raw = load_raw_neurons()
    circuit = build_connectome_circuit(raw, max_neurons=max_neurons)
    save_connectome_cache(circuit, cache_path)
    return circuit
