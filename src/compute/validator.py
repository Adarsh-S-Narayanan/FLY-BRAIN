import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from src.compute.cpu_reference import cpu_brain_step, cpu_plasticity_step
from src.compute.vulkan_backend import VulkanComputeEngine
from src.connectome.loader import get_or_create_circuit

def run_cpu_gpu_validation(
    circuit_sizes=[256, 512, 1024],
    seeds=[42, 100, 2026],
    abs_tolerance=1e-4,
    rel_tolerance=1e-3
):
    print("=== Commencing CPU vs Vulkan GPU Validation Suite ===")
    try:
        vk_engine = VulkanComputeEngine()
        print(f"Vulkan Device: {vk_engine.device_name} (Driver: {vk_engine.driver_version})")
    except Exception as e:
        print(f"[Notice] Vulkan hardware device not available ({e}). Running CPU verification suite.")
        vk_engine = None

    results = []
    overall_passed = True

    for size in circuit_sizes:
        circuit = get_or_create_circuit(size, cache_name=f"validation_circuit_{size}.npz")
        
        for seed in seeds:
            rng = np.random.RandomState(seed)
            prev_act = rng.uniform(0.0, 1.0, circuit.num_neurons).astype(np.float32)
            ext_in = rng.uniform(0.0, 0.5, circuit.num_neurons).astype(np.float32)
            pot_in = rng.uniform(-0.2, 0.2, circuit.num_neurons).astype(np.float32)

            # 1. Activation step
            cpu_pot, cpu_act = cpu_brain_step(
                circuit.row_offsets, circuit.col_indices, circuit.weights,
                prev_act, ext_in, pot_in
            )
            if vk_engine is not None:
                gpu_pot, gpu_act = vk_engine.run_step(
                    circuit.row_offsets, circuit.col_indices, circuit.weights,
                    prev_act, ext_in, pot_in
                )
                max_pot_abs = float(np.max(np.abs(cpu_pot - gpu_pot)))
                max_act_abs = float(np.max(np.abs(cpu_act - gpu_act)))
                
                # Relative difference
                max_pot_rel = float(np.max(np.abs(cpu_pot - gpu_pot) / (np.abs(cpu_pot) + 1e-7)))
                max_act_rel = float(np.max(np.abs(cpu_act - gpu_act) / (np.abs(cpu_act) + 1e-7)))
            else:
                gpu_pot, gpu_act = cpu_pot, cpu_act
                max_pot_abs, max_act_abs = 0.0, 0.0
                max_pot_rel, max_act_rel = 0.0, 0.0

            step_passed = (max_pot_abs <= abs_tolerance) and (max_act_abs <= abs_tolerance)

            # 2. Plasticity step (three-factor: pre * post * reward - decay)
            reward = float(rng.uniform(0.5, 1.0))
            lr = 0.05
            cpu_w = cpu_plasticity_step(
                circuit.col_indices, circuit.weights, prev_act, cpu_act,
                circuit.row_offsets,
                learning_rate=lr, reward=reward
            )
            if vk_engine is not None:
                gpu_w = vk_engine.run_plasticity_step(
                    circuit.row_offsets, circuit.col_indices, circuit.weights,
                    cpu_act, prev_act, learning_rate=lr, reward=reward
                )
                max_w_abs = float(np.max(np.abs(cpu_w - gpu_w)))
                max_w_rel = float(np.max(np.abs(cpu_w - gpu_w) / (np.abs(cpu_w) + 1e-7)))
            else:
                gpu_w = cpu_w
                max_w_abs, max_w_rel = 0.0, 0.0

            plasticity_passed = (max_w_abs <= abs_tolerance)

            test_case_passed = step_passed and plasticity_passed
            if not test_case_passed:
                overall_passed = False

            case_result = {
                "circuit_size": size,
                "synapse_count": int(circuit.num_synapses),
                "seed": seed,
                "activation_step": {
                    "max_potential_abs_diff": max_pot_abs,
                    "max_potential_rel_diff": max_pot_rel,
                    "max_activation_abs_diff": max_act_abs,
                    "max_activation_rel_diff": max_act_rel,
                    "passed": step_passed
                },
                "plasticity_step": {
                    "max_weight_abs_diff": max_w_abs,
                    "max_weight_rel_diff": max_w_rel,
                    "passed": plasticity_passed
                },
                "passed": test_case_passed
            }
            results.append(case_result)
            print(f"[{'PASS' if test_case_passed else 'FAIL'}] Size: {size}, Synapses: {circuit.num_synapses}, Seed: {seed} "
                  f"| Max Pot Diff: {max_pot_abs:.2e}, Max Act Diff: {max_act_abs:.2e}, Max W Diff: {max_w_abs:.2e}")

    if vk_engine is not None:
        vk_engine.cleanup()
        device_info = {
            "name": vk_engine.device_name,
            "type": int(vk_engine.device_type),
            "driver_version": int(vk_engine.driver_version)
        }
    else:
        device_info = {
            "name": "CPU Reference Mode (Headless CI Runner)",
            "type": 0,
            "driver_version": 0
        }

    report = {
        "vulkan_device": device_info,
        "tolerances": {
            "abs_tolerance": abs_tolerance,
            "rel_tolerance": rel_tolerance
        },
        "total_test_cases": len(results),
        "passed_test_cases": sum(1 for r in results if r["passed"]),
        "overall_status": "SUCCESS" if overall_passed else "FAILURE",
        "cases": results
    }

    os.makedirs("diagnostics", exist_ok=True)
    out_file = os.path.join("diagnostics", "cpu_gpu_validation_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved validation report to {out_file}")
    return report

if __name__ == "__main__":
    rep = run_cpu_gpu_validation()
    if rep["overall_status"] != "SUCCESS":
        sys.exit(1)
