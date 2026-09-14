import csv
import json
import os
import math
import numpy as np

def analyze_soma_data(csv_path):
    print(f"Reading {csv_path}...")
    neurons = {}
    sides = {}
    tbars_dist = []
    body_sizes = []
    coords = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                body_id = int(row['body'])
                nx = float(row['nx'])
                ny = float(row['ny'])
                nz = float(row['nz'])
                tbars = int(row['tbars'])
                side = row['soma_side']
                body_size = int(row['body_size'])
                
                neurons[body_id] = {
                    "body_id": body_id,
                    "nucleus_id": int(row['nucleus_id']),
                    "pos": (nx, ny, nz),
                    "tbars": tbars,
                    "side": side,
                    "body_size": body_size
                }
                sides[side] = sides.get(side, 0) + 1
                tbars_dist.append(tbars)
                body_sizes.append(body_size)
                coords.append((nx, ny, nz))
            except Exception as e:
                continue

    coords = np.array(coords)
    min_coords = coords.min(axis=0).tolist()
    max_coords = coords.max(axis=0).tolist()
    mean_coords = coords.mean(axis=0).tolist()

    report = {
        "source_file": csv_path,
        "total_neuron_records": len(coords),
        "unique_bodies": len(neurons),
        "soma_side_distribution": sides,
        "coordinate_bounds": {
            "min_nx_ny_nz": min_coords,
            "max_nx_ny_nz": max_coords,
            "mean_nx_ny_nz": mean_coords
        },
        "tbars_summary": {
            "total_tbars": int(sum(tbars_dist)),
            "max_tbars": int(max(tbars_dist)) if tbars_dist else 0,
            "neurons_with_tbars": int(sum(1 for t in tbars_dist if t > 0))
        },
        "body_size_summary": {
            "min": int(min(body_sizes)) if body_sizes else 0,
            "max": int(max(body_sizes)) if body_sizes else 0,
            "mean": float(np.mean(body_sizes)) if body_sizes else 0.0
        }
    }
    return report

def main():
    csv_path = os.path.join("malecns", "data-raw", "2023-27-2 soma_sides.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
    report = analyze_soma_data(csv_path)
    os.makedirs("diagnostics", exist_ok=True)
    out_path = os.path.join("diagnostics", "connectome_validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved report to", out_path)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
