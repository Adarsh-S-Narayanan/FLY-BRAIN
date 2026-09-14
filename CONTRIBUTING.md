# Contributing to FlyBrain

Thank you for your interest in contributing to FlyBrain!

FlyBrain is a scientific, research-grade, Vulkan-first artificial organism platform grounded in authentic connectomics. We uphold strict anti-mock and reproducibility engineering standards.

## Code of Conduct
Please review and follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.

## Ground Rules
1. **No Mocks or Placeholders**: Never commit mock production implementations, simulated backends, or synthetic benchmarks.
2. **Deterministic Reproducibility**: All experiments, benchmarks, and tests must use explicit seeds and be fully reproducible.
3. **CPU vs Vulkan Parity**: Any new GPU compute shader must be accompanied by an exact CPU reference implementation and numerical comparison tests under defined tolerances ($\le 10^{-4}$).
4. **Preserve Biological Provenance**: Preserve source IDs and coordinates from the Janelia MaleCNS dataset.

## Development Workflow
1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Set up the development environment:
   ```bash
   uv venv .venv --python 3.12
   uv pip install -r requirements.txt # or install dependencies
   ```
3. Run the automated test suite before opening a PR:
   ```bash
   python tests/test_suite.py
   python src/compute/validator.py
   ```
4. Submit a Pull Request following our PR template.
