import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

class GraphMode(str, Enum):
    REAL = "REAL"
    SPATIAL_SURROGATE = "SPATIAL_SURROGATE"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"

class ProvenanceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    IMPLEMENTED = "IMPLEMENTED"
    DERIVED = "DERIVED"
    SURROGATE = "SURROGATE"
    EXPERIMENTAL = "EXPERIMENTAL"
    PLANNED = "PLANNED"
    UNAVAILABLE = "UNAVAILABLE"

class AnnotationLevel(str, Enum):
    """Per-metadata annotation provenance vocabulary (P7)."""
    EMPIRICAL = "EMPIRICAL"    # directly measured in the MaleCNS tables
    DERIVED = "DERIVED"        # computed from empirical fields (no new claims)
    HEURISTIC = "HEURISTIC"    # coordinate/rule guess, explicitly uncertain
    SURROGATE = "SURROGATE"    # synthetic stand-in, never biological
    UNKNOWN = "UNKNOWN"        # not available locally (e.g. cell type, hemilineage, NT)

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
    # DERIVED morphology from the soma table (soma->tail stub length).
    tail_x: float = 0.0
    tail_y: float = 0.0
    tail_z: float = 0.0
    tail_distance: float = 0.0
    annotation_levels: Dict[str, str] = field(default_factory=lambda: {
        "position": AnnotationLevel.EMPIRICAL.value,
        "side": AnnotationLevel.EMPIRICAL.value,
        "tbars": AnnotationLevel.EMPIRICAL.value,
        "body_size": AnnotationLevel.EMPIRICAL.value,
        "tail_distance": AnnotationLevel.DERIVED.value,
        "cell_type": AnnotationLevel.UNKNOWN.value,
        "hemilineage": AnnotationLevel.UNKNOWN.value,
        "neurotransmitter": AnnotationLevel.UNKNOWN.value,
    })

@dataclass
class PopulationMetadata:
    name: str
    source: str
    selection_rule: str
    neuron_ids: np.ndarray
    neuron_indices: np.ndarray
    count: int
    provenance_status: ProvenanceStatus = ProvenanceStatus.DERIVED
    confidence: float = 0.95
    # P3 honesty fields: coordinate-heuristic populations are NOT EM-annotated.
    classification_method: str = "coordinate_heuristic"
    biological_source: str = "none"
    annotation_status: str = "no_em_annotation_available"
    heuristic: bool = True

    def assert_not_empirical(self):
        if not self.heuristic:
            raise AssertionError(f"Population '{self.name}' claims non-heuristic status without EM annotation")
        if self.provenance_status == ProvenanceStatus.VERIFIED:
            raise AssertionError(f"Heuristic population '{self.name}' must not be VERIFIED")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "selection_rule": self.selection_rule,
            "count": self.count,
            "provenance_status": self.provenance_status.value,
            "confidence": self.confidence,
            "classification_method": self.classification_method,
            "biological_source": self.biological_source,
            "annotation_status": self.annotation_status,
            "annotation_level": AnnotationLevel.HEURISTIC.value,
            "heuristic": self.heuristic,
            "neuron_indices": self.neuron_indices.tolist()
        }

@dataclass
class PopulationRegistry:
    populations: Dict[str, PopulationMetadata] = field(default_factory=dict)

    def register(self, pop: PopulationMetadata):
        self.populations[pop.name] = pop

    def get(self, name: str) -> Optional[PopulationMetadata]:
        return self.populations.get(name)

    def get_indices(self, name: str) -> np.ndarray:
        if name in self.populations:
            return self.populations[name].neuron_indices
        return np.array([], dtype=np.int32)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.populations.items()}

@dataclass
class ConnectomeGraph:
    neuron_ids: np.ndarray          # int64 array of body IDs [N]
    coordinates: np.ndarray         # float32 array [N, 3]
    tbars: np.ndarray               # int32 array [N]
    sides: List[str]                # list of side strings [N]
    row_offsets: np.ndarray         # int32 array [N + 1] (CSR)
    col_indices: np.ndarray         # int32 array [M] (CSR target neurons)
    weights: np.ndarray             # float32 array [M] (synaptic weights)
    mode: GraphMode = GraphMode.SPATIAL_SURROGATE
    provenance_status: ProvenanceStatus = ProvenanceStatus.SURROGATE
    graph_hash: str = ""
    populations: Optional[PopulationRegistry] = None
    provenance_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.graph_hash:
            self.graph_hash = self.compute_graph_hash()
        self.validate_invariants()

    @property
    def num_neurons(self) -> int:
        return len(self.neuron_ids)
    
    @property
    def num_synapses(self) -> int:
        return len(self.col_indices)

    def compute_graph_hash(self) -> str:
        """Computes deterministic SHA-256 fingerprint of the graph topology and parameters."""
        h = hashlib.sha256()
        h.update(self.mode.value.encode())
        h.update(np.ascontiguousarray(self.neuron_ids, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(self.row_offsets, dtype=np.int32).tobytes())
        h.update(np.ascontiguousarray(self.col_indices, dtype=np.int32).tobytes())
        # Quantize weights to 1e-6 for float stability in hashing
        w_quant = np.round(np.ascontiguousarray(self.weights, dtype=np.float32) * 1e6).astype(np.int64)
        h.update(w_quant.tobytes())
        return h.hexdigest()

    def validate_invariants(self) -> bool:
        """Strict mathematical and structural invariant validation."""
        N = self.num_neurons
        M = self.num_synapses

        if len(self.row_offsets) != N + 1:
            raise ValueError(f"row_offsets length {len(self.row_offsets)} must equal num_neurons + 1 ({N + 1})")
        
        if self.row_offsets[0] != 0:
            raise ValueError(f"row_offsets[0] must be 0, got {self.row_offsets[0]}")
            
        if self.row_offsets[-1] != M:
            raise ValueError(f"row_offsets[-1] must equal num_synapses ({M}), got {self.row_offsets[-1]}")

        # Monotonicity check
        diffs = np.diff(self.row_offsets)
        if np.any(diffs < 0):
            raise ValueError("row_offsets must be monotonically non-decreasing.")

        # Range check on target indices
        if M > 0:
            if np.any(self.col_indices < 0) or np.any(self.col_indices >= N):
                raise ValueError("col_indices contains out-of-bounds target neuron indices.")
            if np.any(~np.isfinite(self.weights)):
                raise ValueError("weights array contains NaN or Inf values.")

        # Check unique neuron body IDs
        if len(np.unique(self.neuron_ids)) != N:
            raise ValueError("neuron_ids contains non-unique body IDs.")

        # Coordinates check
        if self.coordinates.shape != (N, 3):
            raise ValueError(f"coordinates shape must be ({N}, 3), got {self.coordinates.shape}")
        if np.any(~np.isfinite(self.coordinates)):
            raise ValueError("coordinates array contains NaN or Inf values.")

        return True

    def get_population_indices(self, pop_name: str) -> np.ndarray:
        if self.populations:
            return self.populations.get_indices(pop_name)
        return np.array([], dtype=np.int32)

class MaleCNSRealGraph(ConnectomeGraph):
    """Authentic biological connectome graph constructed directly from Janelia MaleCNS v1.0 EM synapse tables."""
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = GraphMode.REAL
        kwargs["provenance_status"] = ProvenanceStatus.VERIFIED
        super().__init__(*args, **kwargs)

class MaleCNSSpatialSurrogateGraph(ConnectomeGraph):
    """Spatial surrogate graph constructed from Janelia MaleCNS soma coordinates and presynaptic capacities."""
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = GraphMode.SPATIAL_SURROGATE
        kwargs["provenance_status"] = ProvenanceStatus.SURROGATE
        super().__init__(*args, **kwargs)

class SyntheticTestGraph(ConnectomeGraph):
    """Deterministic synthetic test graph for regression and unit testing."""
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = GraphMode.SYNTHETIC_TEST
        kwargs["provenance_status"] = ProvenanceStatus.EXPERIMENTAL
        super().__init__(*args, **kwargs)
