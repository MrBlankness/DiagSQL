"""
Pre-compute the co-occurrence matrix from NL2SQL-Bugs dataset
and save it for use during DiagSQL RL training.

This exports the sub_lift_network_heatmap data mentioned in the paper.
"""
import sys
sys.path.insert(0, 'DiagSQL/codes')

import json
import numpy as np
from pathlib import Path
from cors import CooccurrenceMatrix, ALL_SUB_ERROR_TYPES, SUB_ERROR_TYPE_TO_IDX

DATA_PATH = "DiagSQL/dataset/NL2SQLBugs/NL2SQL-Bugs-with-evidence-enriched.json"
OUTPUT_DIR = Path("DiagSQL/dataset/NL2SQLBugs")


def main():
    print("Computing co-occurrence matrix from NL2SQL-Bugs...")

    # Compute matrix
    matrix = CooccurrenceMatrix()
    matrix.compute_from_dataset(DATA_PATH)

    # Export as JSON (list of edges with log2_lift > threshold)
    edges = []
    for i, type_i in enumerate(ALL_SUB_ERROR_TYPES):
        for j, type_j in enumerate(ALL_SUB_ERROR_TYPES):
            if i == j:
                continue
            log2_lift = float(matrix.M[i, j])
            if abs(log2_lift) > 0.01:  # Keep all non-negligible values
                edges.append({
                    "A": type_i,
                    "B": type_j,
                    "log2_lift": round(log2_lift, 4),
                })

    output_path = OUTPUT_DIR / "cooccurrence_matrix.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(edges)} edges to {output_path}")

    # Also export the full matrix as numpy
    np_path = OUTPUT_DIR / "cooccurrence_matrix.npy"
    np.save(np_path, matrix.M)
    print(f"Exported full matrix ({matrix.M.shape}) to {np_path}")

    # Print some statistics
    print(f"\nTop 10 strongest positive co-occurrences:")
    pairs = []
    for i in range(len(ALL_SUB_ERROR_TYPES)):
        for j in range(i + 1, len(ALL_SUB_ERROR_TYPES)):
            val = matrix.M[i, j]
            if val > 0:
                pairs.append((ALL_SUB_ERROR_TYPES[i], ALL_SUB_ERROR_TYPES[j], val))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, v in pairs[:10]:
        print(f"  {a} <-> {b}: log2(Lift) = {v:.3f}")

    # Print type list for reference
    print(f"\nAll {len(ALL_SUB_ERROR_TYPES)} sub-error types:")
    for i, t in enumerate(ALL_SUB_ERROR_TYPES):
        print(f"  [{i:2d}] {t}")


if __name__ == "__main__":
    main()
