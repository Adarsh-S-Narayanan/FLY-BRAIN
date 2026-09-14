# FlyBrain Architectural Specification & Technical Manual

**Authoritative Architecture Document**  
**Version:** 1.0.0-PROVENANCE  
**Target Platform:** Microsoft Windows 11 Pro 64-bit | Vulkan 1.2+ Compute  
**Biological Basis:** Janelia FlyEM *Drosophila* Male Central Nervous System (`male-cns:v1.0`)

---

## 1. System Architecture Overview

FlyBrain is a real, local, Windows 11 compatible, Vulkan-first artificial organism research platform. It grounds neural computation in authentic biological connectomic data from Janelia Research Campus, computing biophysical neural dynamics across high-throughput GPU compute pipelines while maintaining strict provenance contracts.

```mermaid
flowchart TD
    subgraph Biological["1. Biological Grounding (Janelia MaleCNS v1.0)"]
        SOMA["125,506 Somas & T-bars<br/>(2023-27-2 soma_sides.csv)"]
        CONN["99,301 Biological Synapses<br/>(malecns_v1_0_connections.csv)"]
    end

    subgraph Modes["2. Strict Connectome Separation"]
        REAL["GraphMode.REAL<br/>(VERIFIED Biological EM Synapses)"]
        SURR["GraphMode.SPATIAL_SURROGATE<br/>(SURROGATE 3D Proximity Graph)"]
        SYNTH["GraphMode.SYNTHETIC_TEST<br/>(EXPERIMENTAL Regression Graph)"]
    end

    subgraph MemoryCSR["3. Compressed Sparse Row (CSR) Engine"]
        CSR["row_offsets [N+1]<br/>col_indices [M]<br/>weights [M]"]
        POPS["PopulationRegistry<br/>(Visual, Auditory, Olfactory, Motor, etc.)"]
    end

    subgraph Compute["4. Persistent Compute Subsystems"]
        VK["Persistent Vulkan 1.2+ GPU Engine<br/>(shaders/brain_step.spv, 11 Bindings)"]
        CPU["Deterministic CPU LIF Reference Engine<br/>(cpu_lif_step, Exact Numerical Baseline)"]
    end

    subgraph Dynamics["5. Authoritative Simulation Engine (Thread-Locked)"]
        LIF["Leaky Integrate-and-Fire Dynamics<br/>(V_cand, V_reset, Threshold, Refractory Counter)"]
        PLAS["Reward-Modulated Hebbian Plasticity<br/>(Three-factor STDP & Weight Clamping)"]
        DRIVES["Homeostatic Drives<br/>(Energy, Curiosity, Social, Integrity)"]
    end

    subgraph Storage["6. Multi-Store Persistent Memory (SQLite)"]
        MEM["Working (7-slot) | Episodic | Semantic | Dream Studio"]
    end

    subgraph Presentation["7. FlyBrain Lab (Dark Scientific Workstation)"]
        WEB["Three.js 3D Connectome Raycasting Viewer<br/>REST APIs & WebSocket Telemetry Queue"]
    end

    SOMA --> SURR
    CONN --> REAL
    REAL & SURR & SYNTH --> CSR
    CSR & POPS --> VK & CPU
    VK & CPU --> Dynamics
    Dynamics <--> Storage
    Dynamics --> Presentation
```

---

## 2. Strict Connectome Separation: Real vs. Surrogate vs. Synthetic

To guarantee technical rigor and scientific honesty, connectome networks in FlyBrain are strictly separated into three isolated modes governed by `src/connectome/types.py`:

### `GraphMode.REAL` (Status: `VERIFIED`)
- **Biological Source:** Authentic Janelia MaleCNS v1.0 biological synaptic connection table (`malecns/data-raw/malecns_v1_0_connections.csv`).
- **Characteristics:** Contains 99,301 verified synaptic connection pairs across 2,048 hub bodies.
- **Topology:** Sparse, non-symmetric, heavy-tailed degree distribution with natural biological recurrent loops and modular clustering.
- **Invariants:** Every edge represents an empirical electron microscopy reconstructed synapse.

### `GraphMode.SPATIAL_SURROGATE` (Status: `SURROGATE`)
- **Biological Source:** 125,506 empirical somas, hemilateral sides, and presynaptic T-bar capacities (`malecns/data-raw/2023-27-2 soma_sides.csv`).
- **Characteristics:** Connects neurons using 3D Euclidean spatial proximity via a balanced k-d tree. Synaptic capacities are scaled by presynaptic T-bars.
- **Contract:** Labeled strictly as `SURROGATE` in all telemetry, APIs, and manifests. Never presented as authentic biological synaptic pairs.

### `GraphMode.SYNTHETIC_TEST` (Status: `EXPERIMENTAL`)
- **Characteristics:** Deterministic synthetic network generated with explicit recurrent loops and input/output pathways.
- **Purpose:** Used for high-speed automated regression testing, mathematical invariance verification, and headless CI/CD runs.

---

## 3. Biophysical Neural Dynamics: Classical LIF Formulation

FlyBrain implements genuine classical Leaky Integrate-and-Fire (LIF) dynamics across both its GLSL compute shaders and CPU reference engines.

### 3.1 Mathematical Formulation
For each neuron $i \in \{0, \dots, N-1\}$ at simulation step $t$:

1. **Synaptic Current Summation:**
   $$I_{\text{syn}, i}(t) = \sum_{j \in \text{Pre}(i)} W_{ij} \cdot S_j(t-1)$$
   where $W_{ij}$ is the synaptic conductance weight and $S_j(t-1) \in \{0.0, 1.0\}$ is the presynaptic binary spike event.

2. **Absolute Refractory Period Check:**
   If the refractory step counter $R_i(t-1) > 0$:
   $$R_i(t) = R_i(t-1) - 1$$
   $$V_i(t) = V_{\text{reset}}$$
   $$S_i(t) = 0.0$$
   The neuron is inhibited from integrating synaptic currents or emitting action potentials.

3. **Subthreshold Leaky Integration:**
   If $R_i(t-1) == 0$:
   $$V_{\text{cand}, i} = V_{\text{rest}} + \left(V_i(t-1) - V_{\text{rest}}\right) \cdot \lambda + I_{\text{syn}, i}(t) + I_{\text{ext}, i}(t)$$
   where $\lambda = \exp(-\Delta t / \tau_m) \approx 0.85$ is the membrane leak factor.

4. **Action Potential Threshold & Hard Reset:**
   If $V_{\text{cand}, i} \ge V_{\text{thresh}}$ (typically $-50.0 \text{ mV}$ or normalized $1.0$):
   $$S_i(t) = 1.0$$
   $$V_i(t) = V_{\text{reset}}$$
   $$R_i(t) = t_{\text{ref}} \quad (\text{default: } 2 \text{ steps})$$
   Otherwise:
   $$S_i(t) = 0.0$$
   $$V_i(t) = \max\left(V_{\text{cand}, i}, V_{\text{reset}} - 1.0\right)$$
   $$R_i(t) = 0$$

### 3.2 Vulkan Compute Shader Descriptor Layout (`shaders/brain_step.comp`)
The compute pipeline binds 11 persistent storage buffers under `set = 0`:

| Binding | Buffer Name | Type | Access | Description |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `RowOffsets` | `int[]` | Readonly | CSR row pointers $[0 \dots N]$ |
| `1` | `ColIndices` | `int[]` | Readonly | CSR column indices $[0 \dots M-1]$ |
| `2` | `Weights` | `float[]` | Readonly | Synaptic weights $[0 \dots M-1]$ |
| `3` | `PrevSpikes` | `float[]` | Readonly | Previous binary spikes $S(t-1) \in \{0.0, 1.0\}$ |
| `4` | `ExternalInputs` | `float[]` | Readonly | External sensory stimulation currents $I_{\text{ext}}$ |
| `5` | `PotentialsIn` | `float[]` | Readonly | Input membrane potentials $V(t-1)$ |
| `6` | `RefractoryIn` | `int[]` | Readonly | Input refractory step counters $R(t-1)$ |
| `7` | `PotentialsOut` | `float[]` | Writeonly | Updated membrane potentials $V(t)$ |
| `8` | `SpikesOut` | `float[]` | Writeonly | Emitted binary action potentials $S(t)$ |
| `9` | `RefractoryOut` | `int[]` | Writeonly | Updated refractory counters $R(t)$ |
| `10` | `BrainParams` | Struct | Readonly | Scalar parameters: $N, \lambda, V_{\text{thresh}}, V_{\text{reset}}, V_{\text{rest}}, t_{\text{ref}}$ |

---

## 4. Persistent GPU Architecture & Resource Lifecycle

Unlike naive implementations that reallocate buffers and rebuild pipelines on every simulation tick, `VulkanComputeEngine` (`src/compute/vulkan_backend.py`) implements a persistent resource architecture:

1. **Capability-Based Device Selection:**
   - Evaluates physical devices using queue family compute flags and device types:
     $$\text{Score} = \begin{cases} 100 & \text{VK\_PHYSICAL\_DEVICE\_TYPE\_DISCRETE\_GPU} \\ 50 & \text{VK\_PHYSICAL\_DEVICE\_TYPE\_INTEGRATED\_GPU} \\ 10 & \text{VK\_PHYSICAL\_DEVICE\_TYPE\_CPU} \end{cases}$$
   - Automatically binds to discrete GPU if present, otherwise integrated GPU (AMD Radeon 680M verified).

2. **Persistent GPU-Resident State:**
   - Buffers for CSR structures, potentials, spikes, and refractory counters are allocated once during circuit initialization.
   - Per-step execution incurs zero `vkAllocateMemory`, `vkCreateBuffer`, `vkCreateComputePipelines`, or `vkAllocateDescriptorSets` calls.
   - Ping-pong synchronization is achieved via single-submit persistent command buffers and memory pipeline barriers (`VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`).

3. **Synaptic Plasticity Pipeline (`shaders/plasticity.comp`):**
   - Implements three-factor reward-modulated Hebbian learning:
     $$\Delta W_{ij} = \eta \cdot r \cdot \left(A_{\text{pre}, j} \cdot A_{\text{post}, i} - \beta \cdot W_{ij}\right)$$
   - Operates in-place on the GPU-resident `Weights` buffer.

---

## 5. Authoritative Single Simulation Engine & Telemetry Isolation

Concurrency is strictly managed by `SimulationEngine` (`src/brain/simulation_engine.py`):

- **Single Threaded Master Loop:** Background thread (`FlyBrainSimWorker`) is the sole entity modifying the brain runtime state.
- **Reentrant Locking:** Thread-safe `threading.RLock` protects all reads and writes against race conditions.
- **Non-Invasive Telemetry Queue:** Light-weight metric snapshots are dispatched non-blockingly to registered async WebSocket queues (`asyncio.Queue`), ensuring web clients and dashboards never stall simulation steps.

---

## 6. Deterministic Experimentation & Provenance Tracking

Experiments in FlyBrain are completely reproducible and tracked cryptographically by `ExperimentManager` (`src/experiment/manager.py`):

- Every experiment records:
  - Seed, graph mode, neuron count, duration steps
  - SHA-256 hashes of soma and connection raw biological files
  - SHA-256 hashes of compiled SPIR-V compute shaders
  - Exact Git commit SHA
  - Hardware device identifier and driver version
  - Initial 256-bit SHA-256 state hash
  - Final 256-bit SHA-256 state hash
  - Serialized `.npz` brain state snapshot
- Bit-exact replication is verified across independent runs via:
  ```bash
  flybrain experiment verify --result <experiment_id>
  ```

---

## 7. Automated Acceptance Matrix & Verification

Release readiness is enforced by an automated 19-category acceptance matrix (`scripts/run_acceptance_matrix.py`):

```bash
flybrain acceptance-matrix
```

All 19 categories evaluate dynamically to `PASS`, verifying repository cleanliness, biological provenance manifest integrity, contract separation, LIF dynamics, refractory invariance, Vulkan persistence, CPU-Vulkan numerical parity, thread safety, and documentation consistency.
