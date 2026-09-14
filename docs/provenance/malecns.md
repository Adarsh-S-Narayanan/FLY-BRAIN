# Janelia MaleCNS v1.0 Biological Provenance

## 1. Overview & Dataset Identity

The connectome subsystem in FlyBrain is built directly on the **Janelia MaleCNS (male-cns:v1.0)** electron-microscopy whole central nervous system reconstruction of adult *Drosophila melanogaster*, published and maintained by the Janelia FlyEM project and the Cambridge Department of Zoology (Jefferis Lab / natverse).

| Property | Value |
| :--- | :--- |
| **Dataset Name** | Janelia MaleCNS |
| **Snapshot Version** | `male-cns:v1.0` |
| **Upstream Repository** | [https://github.com/natverse/malecns](https://github.com/natverse/malecns) |
| **Primary Soma File** | `malecns/data-raw/2023-27-2 soma_sides.csv` |
| **Soma File SHA-256** | `6c1c415ab748ab1cc65c9a274885cfadc880a88eabb17915afe643c105bac84d` |
| **Connections File** | `malecns/data-raw/malecns_v1_0_connections.csv` |
| **Connections SHA-256** | `039e929b4ccc776f13a8d17eb9833d85de9454b8f43a0a0de34fdabb7ca2a668` |
| **Total Mapped Neurons** | 125,506 neurons |
| **Total Presynaptic T-Bars** | 25,394,402 presynaptic sites |
| **Organism / Sex** | *Drosophila melanogaster* (Male adult) |

---

## 2. Graph Modes & Scientific Classification

To prevent ambiguity between authentic biological observations, derived models, and testing harnesses, FlyBrain establishes three mutually exclusive graph modes:

```text
               ┌────────────────────────────────────────────────────────┐
               │              Connectome Subsystem Modes                │
               └───────────────────────────┬────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼────────────────────────────────┐
         ▼                                 ▼                                ▼
┌─────────────────┐              ┌───────────────────┐            ┌───────────────────┐
│ GraphMode.REAL  │              │SPATIAL_SURROGATE  │            │  SYNTHETIC_TEST   │
│                 │              │                   │            │                   │
│ Provenance:     │              │ Provenance:       │            │ Provenance:       │
│ VERIFIED        │              │ SURROGATE         │            │ EXPERIMENTAL      │
│ Authentic EM    │              │ Spatial KD-tree   │            │ Synthetic test    │
│ synapse tables  │              │ proximity model   │            │ circuit topology  │
└─────────────────┘              └───────────────────┘            └───────────────────┘
```

### 2.1 GraphMode.REAL (`MaleCNSRealGraph`)
- **Category**: `VERIFIED`
- **Source**: Ingested directly from canonical Janelia MaleCNS v1.0 synaptic connection tables (`malecns_v1_0_connections.csv`).
- **Semantics**: Edges represent real physical synapses reconstructed via volume electron microscopy.
- **Weights**: Normalized from observed biological synapse counts between partner body IDs ($W_{ij} \propto \text{synapse\_count}$).
- **Invariants**: All neuron IDs correspond to validated biological body IDs in `2023-27-2 soma_sides.csv`.

### 2.2 GraphMode.SPATIAL_SURROGATE (`MaleCNSSpatialSurrogateGraph`)
- **Category**: `SURROGATE`
- **Source**: Derived from the 125,506 3D soma coordinates (`nx, ny, nz`) and presynaptic T-bar capacities in `2023-27-2 soma_sides.csv`.
- **Semantics**: Connections are computed via spatial proximity (k-d tree Euclidean distance within interaction radius $R \le 6000\,\text{nm}$) modulated by presynaptic T-bar counts.
- **Purpose**: High-performance spatial morphometry approximation for scalable whole-brain emulation when full synaptic graphs exceed physical GPU memory bounds.
- **Reporting Rule**: Must always be referred to as `MaleCNS Spatial Surrogate`, never as authentic synaptic connectivity.

### 2.3 GraphMode.SYNTHETIC_TEST (`SyntheticTestGraph`)
- **Category**: `EXPERIMENTAL`
- **Source**: Pure algorithmic synthetic graph generated with fixed pseudo-random seeds.
- **Purpose**: Unit tests, invariant verification, numerical stability tests, and portable CI environments.

---

## 3. Biological Population Mapping

Arbitrary index slicing is strictly prohibited. Functional populations are defined via explicit anatomical and morphological criteria:

| Population | Biological Locus | Selection Criteria | Typical Provenance |
| :--- | :--- | :--- | :--- |
| **visual** | Optic Lobes / Medulla / Lobula | Anterior lateral somas ($x < 30000$ or $x > 70000$, $z < 35000$) | `DERIVED` |
| **auditory** | AMMC / Johnston's Organ | Anterior central somas ($35000 \le x \le 65000, 15000 \le y \le 25000, z < 25000$) | `DERIVED` |
| **olfactory** | Antennal Lobe / Mushroom Body | Rostral medial cluster ($40000 \le x \le 60000, y < 22000$) | `DERIVED` |
| **mechanosensory** | Subesophageal / Thoracic projection | Ventral/subesophageal regions | `DERIVED` |
| **descending** | Descending Neurons (DNs) | High T-bar posterior projection somas ($z > 38000$) targeting VNC | `DERIVED` |
| **motor** | Efferent motor hubs | Somas associated with motor neuropil | `DERIVED` |
| **interneuron** | Central Brain Local/Projection | Broad central brain somas | `DERIVED` |
| **modulatory** | Aminergic / Modulatory | Major T-bar hubs with broad bilateral arborization | `DERIVED` |
| **memory_association** | Central Complex / Mushroom Body | Protocerebral central complex neuropil somas | `DERIVED` |

---

## 4. Integrity & Canonical Verification

Every loaded graph generates an immutable SHA-256 fingerprint computed from:
1. Canonical sorted body IDs (`neuron_ids`).
2. Soma spatial coordinates (`coordinates`).
3. Compressed Sparse Row structure (`row_offsets` and `col_indices`).
4. Synaptic weight tensor (`weights`).
5. Graph mode identifier.

This hash guarantees deterministic reproducibility across research runs and process restarts.
