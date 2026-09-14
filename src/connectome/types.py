from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

@dataclass
class NeuronMetadata:
    body_id: int
    nucleus_id: int
    x: float
    y: float
    z: float
    side: str  # 'L', 'R', or 'M'
    tbars: int
    body_size: int
    region: str = "cns"

@dataclass
class ConnectomeGraph:
    neuron_ids: np.ndarray          # int64 array of body IDs [N]
    coordinates: np.ndarray         # float32 array [N, 3]
    tbars: np.ndarray               # int32 array [N]
    sides: List[str]                # list of side strings [N]
    row_offsets: np.ndarray         # int32 array [N + 1] (CSR)
    col_indices: np.ndarray         # int32 array [M] (CSR target neurons)
    weights: np.ndarray             # float32 array [M] (synaptic weights)
    
    @property
    def num_neurons(self) -> int:
        return len(self.neuron_ids)
    
    @property
    def num_synapses(self) -> int:
        return len(self.col_indices)
