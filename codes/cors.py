"""
DiagSQL - Co-occurrence-aware Reward Shaping (CORS)

Implements Equation (4) and (5) from the paper:
  r_err^(u) = r_base^(u) + Δ_cooccur^(u)
  Δ_cooccur^(u) = λ * Σ_{v ∈ E \ {u}} [ M_{u,v} * b_v * b̂_v - |M_{u,v}| * (b_v - b̂_v)^2 ]

where:
  - M_{u,v} = log2(Lift(u,v)) is from the pre-computed co-occurrence matrix
  - b_v = 1 if error v is in ground truth, 0 otherwise
  - b̂_v = 1 if error v is in prediction, 0 otherwise
  - λ controls the strength of the structural inductive bias
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


# Error type taxonomy (same as in the paper)
SUB_ORDER_BY_MAIN = {
    "Attribute-Related Errors": [
        "Attribute Mismatch",
        "Attribute Redundancy",
        "Attribute Missing",
    ],
    "Table-Related Errors": [
        "Table Mismatch",
        "Table Redundancy",
        "Table Missing",
        "Join Condition Mismatch",
        "Join Type Mismatch",
    ],
    "Value-Related Errors": [
        "Value Mismatch",
        "Data Format Mismatch",
    ],
    "Operator-Related Errors": [
        "Comparison Operator Mismatch",
        "Logical Operator Mismatch",
    ],
    "Condition-Related Errors": [
        "Explicit Condition Missing",
        "Explicit Condition Mismatch",
        "Explicit Condition Redundancy",
        "Implicit Condition Missing",
    ],
    "Function-Related Errors": [
        "Aggregate Functions",
        "Window Functions",
        "Date/Time Functions",
        "Conversion Functions",
        "Math Functions",
        "String Functions",
        "Conditional Functions",
    ],
    "Clause-Related Errors": [
        "Clause Missing",
        "Clause Redundancy",
    ],
    "Subquery-Related Errors": [
        "Subquery Missing",
        "Subquery Mismatch",
        "Partial Query",
    ],
    "Other Errors": [
        "ASC/DESC",
        "DISTINCT",
        "Other",
    ],
}

# Flattened list of all sub-error types
ALL_SUB_ERROR_TYPES = [s for subs in SUB_ORDER_BY_MAIN.values() for s in subs]
# Map sub_error_type -> index
SUB_ERROR_TYPE_TO_IDX = {s: i for i, s in enumerate(ALL_SUB_ERROR_TYPES)}


class CooccurrenceMatrix:
    """
    Pre-computed co-occurrence matrix M where M[u,v] = log2(Lift(u,v)).

    This is constructed from the analysis of the NL2SQL-Bugs dataset.
    """

    def __init__(self, matrix_path: Optional[str] = None):
        """
        Args:
            matrix_path: Path to the pre-computed co-occurrence CSV or JSON.
                         If None, will compute from the NL2SQL-Bugs dataset.
        """
        self.n_types = len(ALL_SUB_ERROR_TYPES)
        self.type_to_idx = SUB_ERROR_TYPE_TO_IDX
        self.idx_to_type = {v: k for k, v in self.type_to_idx.items()}

        # Initialize M as zeros (no co-occurrence signal)
        self.M = np.zeros((self.n_types, self.n_types), dtype=np.float32)

        if matrix_path and Path(matrix_path).exists():
            self._load_from_file(matrix_path)
        else:
            print("[CORS] Warning: No co-occurrence matrix loaded. Using zeros.")

    def _load_from_file(self, path: str):
        """Load co-occurrence matrix from the analysis output."""
        path = Path(path)

        if path.suffix == '.csv':
            self._load_from_csv(path)
        elif path.suffix == '.json':
            self._load_from_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Count non-zero entries
        nonzero = np.count_nonzero(self.M)
        print(f"[CORS] Loaded co-occurrence matrix: {self.n_types}x{self.n_types}, "
              f"{nonzero} non-zero entries")

    def _load_from_csv(self, path: Path):
        """Load from the pairwise stats CSV (sub_pairwise_stats.csv format)."""
        import pandas as pd
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            a = row.get("A", "")
            b = row.get("B", "")
            lift = row.get("lift", 0.0)

            if a in self.type_to_idx and b in self.type_to_idx and lift > 0:
                i = self.type_to_idx[a]
                j = self.type_to_idx[b]
                log2_lift = np.log2(lift) if lift > 0 else 0.0
                self.M[i, j] = log2_lift

    def _load_from_json(self, path: Path):
        """Load from a JSON file with lift values."""
        with open(path, 'r') as f:
            data = json.load(f)

        for entry in data:
            a = entry.get("A", "")
            b = entry.get("B", "")
            log2_lift = entry.get("log2_lift", 0.0)

            if a in self.type_to_idx and b in self.type_to_idx:
                i = self.type_to_idx[a]
                j = self.type_to_idx[b]
                self.M[i, j] = log2_lift

    def compute_from_dataset(self, data_path: str):
        """
        Compute co-occurrence matrix directly from NL2SQL-Bugs dataset.
        This replicates the analysis from analyze_error_causality.py.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Count individual and joint occurrences
        N = 0
        counts = np.zeros(self.n_types, dtype=np.int32)
        joint_counts = np.zeros((self.n_types, self.n_types), dtype=np.int32)

        for row in data:
            if row.get("label", True):  # Skip correct SQL
                continue
            etypes = row.get("error_types") or []
            if not etypes:
                continue

            N += 1
            sub_indices = set()
            for et in etypes:
                sub = (et.get("sub_error_type") or "").strip()
                if sub in self.type_to_idx:
                    sub_indices.add(self.type_to_idx[sub])

            for idx in sub_indices:
                counts[idx] += 1
            for i in sub_indices:
                for j in sub_indices:
                    if i != j:
                        joint_counts[i, j] += 1

        # Compute log2(Lift)
        if N > 0:
            for i in range(self.n_types):
                for j in range(self.n_types):
                    if i == j:
                        continue
                    p_i = counts[i] / N
                    p_j = counts[j] / N
                    p_ij = joint_counts[i, j] / N
                    if p_i > 0 and p_j > 0 and p_ij > 0:
                        lift = p_ij / (p_i * p_j)
                        self.M[i, j] = np.log2(lift)

        nonzero = np.count_nonzero(self.M)
        print(f"[CORS] Computed co-occurrence matrix from {N} error cases, "
              f"{nonzero} non-zero entries")

    def get_lift(self, u: str, v: str) -> float:
        """Get log2(Lift(u, v)) for error pair (u, v)."""
        if u not in self.type_to_idx or v not in self.type_to_idx:
            return 0.0
        i = self.type_to_idx[u]
        j = self.type_to_idx[v]
        return float(self.M[i, j])


def compute_cors_modifier(
    predicted_error: str,
    gt_errors: Set[str],
    pred_errors: Set[str],
    cooccurrence_matrix: CooccurrenceMatrix,
    lambda_: float = 0.3,
    lift_threshold: float = 1.0,
) -> float:
    """
    Compute the CORS modifier Δ_cooccur^(u) for a predicted error u.

    Implements Equation (5):
    Δ_cooccur^(u) = λ * Σ_{v ∈ E \ {u}} [ M_{u,v} * b_v * b̂_v - |M_{u,v}| * (b_v - b̂_v)^2 ]

    Three cases:
    1. Associative Bonus (b_v=1, b̂_v=1): +M_{u,v}  (bonus for correct co-prediction)
    2. Cascading Penalty (b_v=1, b̂_v=0): -|M_{u,v}| (penalty for missing correlated error)
    3. Illogical Combination (b_v=0, b̂_v=1): -|M_{u,v}| (penalty for hallucinating error)

    Args:
        predicted_error: The sub_error_type u being evaluated
        gt_errors: Set of ground truth sub_error_types
        pred_errors: Set of predicted sub_error_types
        cooccurrence_matrix: Pre-computed CooccurrenceMatrix
        lambda_: Strength of the structural inductive bias
        lift_threshold: Minimum |M_{u,v}| to consider (pairs below are ignored)

    Returns:
        The CORS modifier value Δ_cooccur^(u)
    """
    delta = 0.0
    u = predicted_error

    for v in ALL_SUB_ERROR_TYPES:
        if v == u:
            continue

        m_uv = cooccurrence_matrix.get_lift(u, v)

        # Skip pairs with weak co-occurrence signal
        if abs(m_uv) < lift_threshold:
            continue

        b_v = 1.0 if v in gt_errors else 0.0
        b_hat_v = 1.0 if v in pred_errors else 0.0

        # Equation (5): M_{u,v} * b_v * b̂_v - |M_{u,v}| * (b_v - b̂_v)^2
        bonus_term = m_uv * b_v * b_hat_v
        penalty_term = abs(m_uv) * (b_v - b_hat_v) ** 2
        delta += bonus_term - penalty_term

    return lambda_ * delta


def compute_cors_rewards(
    pred_error_types: List[Dict],
    gt_error_types: List[Dict],
    cooccurrence_matrix: CooccurrenceMatrix,
    lambda_: float = 0.3,
    lift_threshold: float = 1.0,
) -> Dict[str, float]:
    """
    Compute CORS-adjusted rewards for all predicted error types.

    For each predicted error u, computes:
        r_err^(u) = r_base^(u) + Δ_cooccur^(u)

    where r_base^(u) depends on whether u is in ground truth:
        - If u ∈ E_gt: r_base = +1 (correct prediction)
        - If u ∉ E_gt: r_base = 0 (hallucination, no positive base reward)

    Args:
        pred_error_types: List of predicted error type dicts
        gt_error_types: List of ground truth error type dicts
        cooccurrence_matrix: Pre-computed CooccurrenceMatrix
        lambda_: CORS lambda parameter
        lift_threshold: Minimum lift to consider

    Returns:
        Dict mapping (error_type, sub_error_type) tuple -> adjusted reward
    """
    # Extract sub_error_types as sets
    gt_subs = set()
    for et in gt_error_types:
        sub = et.get("sub_error_type", "").strip()
        if sub:
            gt_subs.add(sub)

    pred_subs = set()
    for et in pred_error_types:
        sub = et.get("sub_error_type", "").strip()
        if sub:
            pred_subs.add(sub)

    # Compute adjusted reward for each predicted error
    rewards = {}
    for et in pred_error_types:
        error_type = et.get("error_type", "").strip()
        sub_error_type = et.get("sub_error_type", "").strip()
        if not sub_error_type:
            continue

        key = (error_type, sub_error_type)

        # Base reward: +1 if correct (in GT), 0 if hallucinated
        if sub_error_type in gt_subs:
            r_base = 1.0
        else:
            r_base = 0.0

        # CORS modifier
        delta_cooccur = compute_cors_modifier(
            predicted_error=sub_error_type,
            gt_errors=gt_subs,
            pred_errors=pred_subs,
            cooccurrence_matrix=cooccurrence_matrix,
            lambda_=lambda_,
            lift_threshold=lift_threshold,
        )

        rewards[key] = r_base + delta_cooccur

    return rewards
