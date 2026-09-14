# Third-Party Notices & Attribution

FlyBrain incorporates, references, or builds upon data and open-source software packages under various open-source licenses. This document provides formal attribution and licensing notices.

---

## 1. Janelia MaleCNS Connectome Dataset

- **Dataset:** Janelia FlyEM Male Central Nervous System Connectome (`male-cns:v1.0`)
- **Institution:** Janelia Research Campus, Howard Hughes Medical Institute (HHMI) & University of Cambridge
- **Authors & Contributors:** Takemura, S. Y., Aso, Y., Hige, T., Wong, A. M., Lu, Z., Xu, C. S., Hess, H. F., Rubin, G. M., et al.
- **Upstream R Package:** `natverse/malecns` (https://github.com/natverse/malecns)
- **License:** GNU General Public License v3.0 (GPL-3.0)
- **Provenance Files Included in Repository:**
  - `malecns/data-raw/2023-27-2 soma_sides.csv` (125,506 biological neuron somas)
  - `malecns/data-raw/malecns_v1_0_connections.csv` (99,301 verified biological synaptic connections)

---

## 2. Vulkan SDK & SPIR-V Tools

- **Component:** LunarG Vulkan SDK & `glslc` SPIR-V Compiler
- **Organization:** LunarG, Inc. & Khronos Group
- **License:** Apache License 2.0 / MIT License
- **Notice:** Used for compiling GLSL compute shaders (`shaders/brain_step.comp`, `shaders/plasticity.comp`) into SPIR-V binaries (`.spv`).

---

## 3. Python Ecosystem & Core Dependencies

The following software packages are used in FlyBrain under permissive licenses:

- **NumPy & SciPy:** BSD 3-Clause License
- **Vulkan Python Bindings (`vulkan`):** Apache License 2.0
- **FastAPI & Starlette:** MIT License (Copyright (c) Sebastián Ramírez)
- **Uvicorn:** BSD 3-Clause License
- **WebSockets:** BSD 3-Clause License
- **Pillow (PIL):** HPND License
- **SoundFile & SoundDevice:** MIT / BSD Licenses
- **PyTorch & TorchVision:** BSD 3-Clause License (Meta Platforms, Inc.)
- **Hugging Face Transformers & Diffusers:** Apache License 2.0 (Hugging Face, Inc.)
- **Accelerate:** Apache License 2.0
- **Pydantic:** MIT License
- **Requests:** Apache License 2.0
- **psutil:** BSD 3-Clause License

---

## 4. WebGL / Three.js

- **Library:** Three.js (r128)
- **Author:** Ricardo Cabello (Mr.doob) and the Three.js Authors
- **License:** MIT License
- **Notice:** Embedded in `src/ui/static/index.html` for interactive 3D biological connectome rendering.
