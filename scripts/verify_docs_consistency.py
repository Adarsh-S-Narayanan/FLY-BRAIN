#!/usr/bin/env python3
"""
Automated Documentation and Code Consistency Verification Script.
Ensures that claims in docs (synapse counts, neuron counts, Vulkan bindings, endpoints, etc.)
strictly match the codebase and datasets.
"""

import os
import sys
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def verify_docs_consistency() -> bool:
    print("=" * 60)
    print("FLYBRAIN DOCUMENTATION & CODE CONSISTENCY VERIFIER")
    print("=" * 60)
    inconsistencies = []

    # 1. Check MaleCNS Data Counts
    soma_csv = os.path.join(PROJECT_ROOT, "malecns", "data-raw", "2023-27-2 soma_sides.csv")
    conn_csv = os.path.join(PROJECT_ROOT, "malecns", "data-raw", "malecns_v1_0_connections.csv")

    if not os.path.exists(soma_csv) or not os.path.exists(conn_csv):
        inconsistencies.append("Missing MaleCNS biological CSV files in malecns/data-raw/")
    else:
        with open(soma_csv, "r", encoding="utf-8") as f:
            soma_count = sum(1 for _ in f) - 1 # exclude header
        with open(conn_csv, "r", encoding="utf-8") as f:
            conn_count = sum(1 for _ in f) - 1 # exclude header

        print(f"Verified MaleCNS biological somas: {soma_count:,}")
        print(f"Verified MaleCNS biological connections: {conn_count:,}")

        if soma_count != 125506:
            inconsistencies.append(f"MaleCNS soma count mismatch: expected 125,506, got {soma_count}")
        if conn_count != 99301:
            inconsistencies.append(f"MaleCNS connection count mismatch: expected 99,301, got {conn_count}")

    # 2. Check Vulkan Shader Bindings
    brain_comp = os.path.join(PROJECT_ROOT, "shaders", "brain_step.comp")
    if os.path.exists(brain_comp):
        with open(brain_comp, "r", encoding="utf-8") as f:
            content = f.read()
            bindings = re.findall(r"binding\s*=\s*(\d+)", content)
            binding_count = len(bindings)
            print(f"Verified brain_step.comp descriptor bindings: {binding_count} (bindings: {bindings})")
            if binding_count != 11:
                inconsistencies.append(f"brain_step.comp binding count mismatch: expected 11, got {binding_count}")
    else:
        inconsistencies.append("shaders/brain_step.comp not found")

    plasticity_comp = os.path.join(PROJECT_ROOT, "shaders", "plasticity.comp")
    if os.path.exists(plasticity_comp):
        with open(plasticity_comp, "r", encoding="utf-8") as f:
            content = f.read()
            bindings = re.findall(r"binding\s*=\s*(\d+)", content)
            binding_count = len(bindings)
            print(f"Verified plasticity.comp descriptor bindings: {binding_count} (bindings: {bindings})")
            if binding_count != 6:
                inconsistencies.append(f"plasticity.comp binding count mismatch: expected 6, got {binding_count}")
    else:
        inconsistencies.append("shaders/plasticity.comp not found")

    # 3. Check Compiled SPIR-V Shaders
    for spv in ["brain_step.spv", "plasticity.spv"]:
        spv_path = os.path.join(PROJECT_ROOT, "shaders", spv)
        if not os.path.exists(spv_path) or os.path.getsize(spv_path) == 0:
            inconsistencies.append(f"Missing or empty SPIR-V shader binary: shaders/{spv}")
        else:
            print(f"Verified SPIR-V binary: shaders/{spv} ({os.path.getsize(spv_path)} bytes)")

    # 4. Check UI Endpoints
    server_py = os.path.join(PROJECT_ROOT, "src", "ui", "server.py")
    expected_endpoints = [
        "/api/health",
        "/api/state",
        "/api/telemetry",
        "/api/connectome",
        "/api/simulation/start",
        "/api/simulation/pause",
        "/api/simulation/step",
        "/api/simulation/reset",
        "/api/memory",
        "/api/evolution/lineage",
        "/api/evolution/generation",
        "/api/dreams",
        "/api/tools",
        "/api/experiments",
        "/api/diagnostics",
        "/ws/telemetry"
    ]
    if os.path.exists(server_py):
        with open(server_py, "r", encoding="utf-8") as f:
            server_src = f.read()
            for ep in expected_endpoints:
                if ep not in server_src:
                    inconsistencies.append(f"UI Server missing required endpoint: {ep}")
                else:
                    print(f"Verified UI endpoint: {ep}")
    else:
        inconsistencies.append("src/ui/server.py not found")

    # 5. Check Connectome Types & Graph Modes
    from src.connectome.types import GraphMode, ProvenanceStatus
    modes = [m.value for m in GraphMode]
    expected_modes = ["REAL", "SPATIAL_SURROGATE", "SYNTHETIC_TEST"]
    for em in expected_modes:
        if em not in modes:
            inconsistencies.append(f"GraphMode missing mode: {em}")
        else:
            print(f"Verified GraphMode: {em}")

    # 6. Check Provenance Manifest
    prov_manifest = os.path.join(PROJECT_ROOT, "manifests", "malecns_provenance.json")
    if os.path.exists(prov_manifest):
        with open(prov_manifest, "r", encoding="utf-8") as f:
            prov_data = json.load(f)
            if "provenance_metadata" not in prov_data:
                inconsistencies.append("malecns_provenance.json missing 'provenance_metadata'")
            else:
                print("Verified manifests/malecns_provenance.json structure")
    else:
        inconsistencies.append("manifests/malecns_provenance.json not found")

    print("=" * 60)
    if inconsistencies:
        print(f"FAILED: {len(inconsistencies)} documentation/code inconsistencies found:")
        for inc in inconsistencies:
            print(f"  - [FAIL] {inc}")
        return False
    else:
        print("ALL DOCUMENTATION AND CODE CONSISTENCY CHECKS PASSED [100% OK]")
        return True

def main():
    success = verify_docs_consistency()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
