"""
Analyze the causal / associational structure between SQL error types
in the NL2SQL-Bugs benchmark dataset.

Dataset : "DiagSQL/dataset/NL2SQLBugs/NL2SQL-Bugs-with-evidence-enriched.json"
Output  : "DiagSQL/plot_figure/analysis/"

Methodology
-----------
Each bug case may carry >=1 error tags. Hence we can only estimate
associational quantities and approximate causal direction from asymmetry:

  1. Co-occurrence matrix  C[i,j] = #cases containing both i and j
  2. Conditional probability P(j | i) = C[i,j] / count(i)
  3. Lift                   L(i,j) = P(i,j) / (P(i)*P(j))
  4. PMI                    log2( P(i,j) / (P(i)*P(j)) )
  5. Fisher exact test on the 2x2 contingency table, BH-FDR corrected
  6. A directed "implication-like" edge i -> j if
         P(j|i) - P(i|j) > delta   AND   Lift > 1   AND   q-value < 0.05
     The direction points from the *more frequent / broader* error to
     the one that tends to follow it conditionally.
  7. PC algorithm (constraint-based DAG discovery) on the binary
     occurrence matrix as an extra reference.  Falls back gracefully
     if `causal-learn` is not installed.

Two granularities are analysed:
  * Coarse level  : main error category  (e.g. "Table-Related Errors")
  * Fine level    : sub error type        (e.g. "Table Missing")
"""

from __future__ import annotations

import json
import os
import math
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_PATH = "../dataset/NL2SQLBugs/NL2SQL-Bugs-with-evidence-enriched.json"
OUT_DIR = Path("./analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds for edge selection in the "causal-like" DAG
DELTA_DIR = 0.05      # asymmetry threshold  P(j|i)-P(i|j)
LIFT_MIN = 1.2        # minimum lift
QVAL_MAX = 0.05       # BH-FDR corrected p-value
MIN_SUPPORT = 5       # ignore pairs with too few joint occurrences

# Thresholds for HEATMAP masking (loose on purpose: keep as many visible
# dependencies as possible, but suppress extreme small-sample noise such as
# Window Functions (n=3) × Implicit Condition Missing (n_AB=2), whose lift
# value is mathematically inflated by a tiny P(A)).
HEATMAP_MIN_JOINT     = 3   # need at least 3 joint occurrences
HEATMAP_MIN_CATEGORY  = 10  # both categories must occur in >=10 cases

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 180
mpl.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("white")


# --------------------------------------------------------------------------- #
# Taxonomy ordering (exactly as defined in prompts/prompt.txt)
# --------------------------------------------------------------------------- #
# Main error categories, in the order they appear in prompt.txt
MAIN_ORDER = [
    "Attribute-Related Errors",
    "Table-Related Errors",
    "Value-Related Errors",
    "Operator-Related Errors",
    "Condition-Related Errors",
    "Function-Related Errors",
    "Clause-Related Errors",
    "Subquery-Related Errors",
    "Other Errors",
]

# Sub error types, grouped under each main type, preserving prompt.txt order
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

# Flattened sub-type order
SUB_ORDER = [s for m in MAIN_ORDER for s in SUB_ORDER_BY_MAIN[m]]

# Reverse lookup: sub -> main (derived from taxonomy, used as fallback)
SUB_TO_MAIN_TAXONOMY = {s: m for m, subs in SUB_ORDER_BY_MAIN.items() for s in subs}


def _ordered(seen: set[str], reference: list[str]) -> list[str]:
    """Return items present in *seen* following the order of *reference*.
    Any items in *seen* not present in *reference* are appended at the end,
    sorted alphabetically so the output is still deterministic.
    """
    ordered = [x for x in reference if x in seen]
    extras = sorted(s for s in seen if s not in reference)
    return ordered + extras


# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> list[dict]:
    print(f"[load] reading {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"[load] {len(data)} cases")
    return data


# --------------------------------------------------------------------------- #
# 2. Build binary occurrence matrices
# --------------------------------------------------------------------------- #
def build_matrices(data: list[dict]):
    """Return (df_main, df_sub, sub_to_main) binary matrices over error cases only."""
    main_sets, sub_sets, sub_to_main = [], [], {}
    for row in data:
        if row.get("label", True):            # True -> correct SQL
            continue
        etypes = row.get("error_types") or []
        if not etypes:
            continue
        m_set, s_set = set(), set()
        for et in etypes:
            main = (et.get("error_type") or "").strip()
            sub  = (et.get("sub_error_type") or "").strip()
            if main:
                m_set.add(main)
            if sub:
                s_set.add(sub)
                if main and sub not in sub_to_main:
                    sub_to_main[sub] = main
        main_sets.append(m_set)
        sub_sets.append(s_set)

    # Fill in any missing sub->main mappings from the taxonomy
    for s, m in SUB_TO_MAIN_TAXONOMY.items():
        sub_to_main.setdefault(s, m)

    seen_main = {x for s in main_sets for x in s}
    seen_sub  = {x for s in sub_sets  for x in s}

    # Use the taxonomy order from prompt.txt instead of alphabetical sorting
    all_main = _ordered(seen_main, MAIN_ORDER)
    all_sub  = _ordered(seen_sub,  SUB_ORDER)

    df_main = pd.DataFrame(
        [[1 if c in s else 0 for c in all_main] for s in main_sets],
        columns=all_main,
    )
    df_sub = pd.DataFrame(
        [[1 if c in s else 0 for c in all_sub] for s in sub_sets],
        columns=all_sub,
    )
    print(f"[matrix] error cases: {len(df_main)}  |  main cats: {len(all_main)}  "
          f"|  sub cats: {len(all_sub)}")
    return df_main, df_sub, sub_to_main


# --------------------------------------------------------------------------- #
# 3. Statistics
# --------------------------------------------------------------------------- #
def compute_stats(df: pd.DataFrame, min_support: int = MIN_SUPPORT) -> pd.DataFrame:
    """All pairwise statistics for the given occurrence dataframe."""
    cols = df.columns.tolist()
    N = len(df)
    counts = df.sum(axis=0).to_dict()                 # support per category
    records = []

    X = df.values.astype(np.int8)
    co = X.T @ X                                      # |C x C| integer matrix

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if a == b:
                continue
            n_a   = counts[a]
            n_b   = counts[b]
            n_ab  = int(co[i, j])
            n_naB = n_b - n_ab
            n_Anb = n_a - n_ab
            n_nab = N - n_a - n_b + n_ab
            if n_a == 0 or n_b == 0:
                continue

            p_a   = n_a / N
            p_b   = n_b / N
            p_ab  = n_ab / N
            p_b_given_a = n_ab / n_a if n_a else 0.0
            p_a_given_b = n_ab / n_b if n_b else 0.0
            lift  = (p_ab / (p_a * p_b)) if (p_a * p_b) > 0 else 0.0
            pmi   = math.log2(lift) if lift > 0 else float("-inf")

            # Fisher exact (two-sided). Skip if no joint support.
            try:
                _, p_val = fisher_exact(
                    [[n_ab, n_Anb], [n_naB, n_nab]],
                    alternative="greater",
                )
            except Exception:
                p_val = 1.0

            records.append({
                "A": a, "B": b,
                "n_A": n_a, "n_B": n_b, "n_AB": n_ab, "N": N,
                "P(A)": p_a, "P(B)": p_b,
                "P(B|A)": p_b_given_a, "P(A|B)": p_a_given_b,
                "lift": lift, "pmi": pmi, "p_value": p_val,
                "asymmetry": p_b_given_a - p_a_given_b,
            })

    out = pd.DataFrame(records)
    # BH-FDR correction
    if len(out):
        _, qvals, _, _ = multipletests(out["p_value"].values, method="fdr_bh")
        out["q_value"] = qvals
        out["support_ok"] = out["n_AB"] >= min_support
    return out


# --------------------------------------------------------------------------- #
# 4. Plot heatmaps
# --------------------------------------------------------------------------- #
def _pivot(stats: pd.DataFrame, value: str, labels: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(0.0, index=labels, columns=labels)
    label_set = set(labels)
    for _, r in stats.iterrows():
        a, b = r["A"], r["B"]
        if a in label_set and b in label_set:
            mat.loc[a, b] = r[value]
    return mat


def _build_support_mask(stats: pd.DataFrame,
                        labels: list[str],
                        min_joint: int,
                        min_category: int) -> tuple[pd.DataFrame, dict]:
    """Return (mask, category_freq) where ``mask[A, B] == True`` means the
    cell should be HIDDEN because support is too low to be reliable.

    A cell is masked if **either**:
      * n_AB < min_joint                (too few joint occurrences), **or**
      * min(n_A, n_B) < min_category    (either category itself is too rare).

    The diagonal is always masked.
    """
    mask = pd.DataFrame(False, index=labels, columns=labels)
    freq: dict = {}
    for _, r in stats.iterrows():
        freq[r["A"]] = int(r["n_A"])
        freq[r["B"]] = int(r["n_B"])
    for a in labels:
        for b in labels:
            if a == b:
                mask.loc[a, b] = True
                continue
            na = freq.get(a, 0)
            nb = freq.get(b, 0)
            nab = 0
            row = stats[(stats["A"] == a) & (stats["B"] == b)]
            if not row.empty:
                nab = int(row.iloc[0]["n_AB"])
            if nab < min_joint or min(na, nb) < min_category:
                mask.loc[a, b] = True
    return mask, freq


def _mask_footer(min_joint: int, min_category: int, n_hidden: int,
                 n_total: int) -> str:
    pct = 100.0 * n_hidden / n_total if n_total else 0.0
    return (f"Grey-hatched cells are masked for low support "
            f"(n_AB < {min_joint}  or  min(n_A, n_B) < {min_category}). "
            f"Masked: {n_hidden}/{n_total} cells ({pct:.1f}%).")


def _drop_fully_masked(mask: pd.DataFrame,
                       labels: list[str]) -> list[str]:
    """Return a shortened label list where every row whose *off-diagonal*
    cells are all masked (and the corresponding column) is removed.

    The diagonal is masked by construction, so we look only at off-diagonal
    cells.  A row/column that keeps at least one visible (= not masked)
    off-diagonal cell is retained.
    """
    kept = []
    for lab in labels:
        row_vals = mask.loc[lab, [c for c in labels if c != lab]]
        col_vals = mask.loc[[c for c in labels if c != lab], lab]
        row_all_hidden = bool(row_vals.all())
        col_all_hidden = bool(col_vals.all())
        # keep the label if either its row OR its column has any visible
        # cell -- losing the label means neither as A nor as B does it
        # contribute any reliable pair.
        if not (row_all_hidden and col_all_hidden):
            kept.append(lab)
    return kept


def plot_lift_network_heatmap(stats: pd.DataFrame,
                              labels: list[str],
                              tag: str,
                              freq: dict | None = None,
                              min_joint: int = HEATMAP_MIN_JOINT,
                              min_category: int = HEATMAP_MIN_CATEGORY,
                              log2_lift_edge: float = 1.0):
    """Mantel-style network + lower-triangle lift heatmap.

    The layout mirrors ``mantel_network.png``:

      * Lower-triangle heat-map on the LEFT  -> one square per (A,B) pair
        whose lift cell is *visible* under the current mask.  Cell colour
        encodes log2(lift) (same diverging scale as the original heatmap).
        Squares have a uniform size -- unlike the reference figure we do
        NOT use size to encode a second quantity.
      * Network on the RIGHT / TOP  -> every error category has a node
        anchored on the heat-map diagonal; pairs with log2(lift) > 1 are
        connected by a curved edge whose width scales with log2(lift) and
        whose colour encodes statistical significance (q-value).

    Only categories that remain after the usual support + weak-signal
    mask are drawn, keeping the figure compact.
    """
    # ---------------------------------------------------------------- 0
    # Reuse the same mask pipeline that builds sub_lift_heatmap.png so
    # the two figures stay in sync.
    support_mask_full, freq_auto = _build_support_mask(
        stats, labels, min_joint=min_joint, min_category=min_category)
    freq = freq or freq_auto

    lf_full = _pivot(stats, "lift", labels)
    lf_log_full = np.log2(lf_full.replace(0, np.nan))
    weak_mask = (lf_log_full.abs() <= 1) | lf_log_full.isna()
    combined_mask_full = support_mask_full | weak_mask

    kept = _drop_fully_masked(combined_mask_full, labels)
    if len(kept) < 2:
        print(f"[{tag}] plot_lift_network_heatmap: too few visible "
              f"categories ({len(kept)}) -- skipping.")
        return

    mask = combined_mask_full.loc[kept, kept]
    lf_log = lf_log_full.loc[kept, kept]
    q_mat = _pivot(stats, "q_value", labels).loc[kept, kept]

    n = len(kept)

    # ---------------------------------------------------------------- 1
    # Colour scale for lift cells (symmetric around 0).
    visible = lf_log.mask(mask).values
    visible = visible[np.isfinite(visible)]
    vabs = float(np.nanmax(np.abs(visible))) if visible.size else 1.0
    vabs = max(vabs, 1e-6)
    lift_cmap = mpl.cm.get_cmap("RdBu_r")
    lift_norm = mpl.colors.Normalize(vmin=-vabs, vmax=vabs)

    # ---------------------------------------------------------------- 2
    # Figure layout (minimal, per user request):
    #   * No title, no legends, no "node size" hint.
    #   * Only a bare colour-bar on the far left (no ticks / no labels).
    #   * Colour-bar height equals the heat-map height.
    #   * Nodes sit ON the diagonal, uniform (slightly smaller than a
    #     cell).
    #   * Network edges live entirely in the upper-right triangle of the
    #     heat-map, all black, width driven by log2(lift).
    side = max(10, 0.55 * n + 5)
    fig = plt.figure(figsize=(side * 1.15, side))

    # Leave a narrow strip on the very left for the colour-bar.
    # The heat-map axis keeps its square aspect; we size it so the
    # colour-bar and the heat-map end up the same height in the
    # rendered figure.  We push the colour-bar further to the left
    # and enlarge the gap so its tick labels (placed on the LEFT
    # side of the cbar) never collide with the heat-map y-axis
    # category labels.
    left_cbar   = 0.018
    width_cbar  = 0.012
    gap         = 0.16
    left_heat   = left_cbar + width_cbar + gap
    right_heat  = 0.995
    bottom_heat = 0.05
    top_heat    = 0.97

    ax  = fig.add_axes([left_heat,  bottom_heat,
                        right_heat - left_heat,
                        top_heat   - bottom_heat])
    cax = fig.add_axes([left_cbar,  bottom_heat,
                        width_cbar,
                        top_heat   - bottom_heat])

    ax.set_aspect("equal")
    # Give the axis enough room for left/bottom tick labels and for
    # bezier arcs bulging into the upper-right.
    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(n + 0.2, -0.2)   # flipped y so row 0 is on top
    ax.axis("off")

    # ---------------------------------------------------------------- 3
    # Draw the lower-triangle heat-map (i > j   =>   cell (i, j) shown).
    cell_size = 0.9  # slightly smaller than 1 to leave a thin gutter
    half = cell_size / 2.0
    for i in range(n):
        for j in range(n):
            if j >= i:
                continue  # upper triangle / diagonal -> skip
            val = lf_log.iat[i, j]
            if mask.iat[i, j] or not np.isfinite(val):
                # Masked cell: draw a light grey placeholder so the
                # triangle keeps its shape but the reader still sees
                # "no reliable signal here".
                ax.add_patch(mpl.patches.Rectangle(
                    (j + 0.5 - half, i + 0.5 - half),
                    cell_size, cell_size,
                    facecolor="#eeeeee", edgecolor="#cccccc",
                    linewidth=0.4, zorder=1))
                continue
            colour = lift_cmap(lift_norm(val))
            ax.add_patch(mpl.patches.Rectangle(
                (j + 0.5 - half, i + 0.5 - half),
                cell_size, cell_size,
                facecolor=colour, edgecolor="white",
                linewidth=0.5, zorder=2))

    # ---------------------------------------------------------------- 4
    # Row / column labels (kept for readability).  No leader lines,
    # no diagonal guide.
    for i, lab in enumerate(kept):
        ax.text(-0.15, i + 0.5, lab,
                fontsize=13, ha="right", va="center",
                color="#111", zorder=5)
        ax.text(i + 0.5, n + 0.05, lab,
                fontsize=13, ha="right", va="top",
                rotation=40, rotation_mode="anchor",
                color="#111", zorder=5)

    # Nodes sit on the diagonal at (i+0.5, i+0.5).
    node_xy = {lab: (i + 0.5, i + 0.5) for i, lab in enumerate(kept)}

    # ---------------------------------------------------------------- 5
    # Collect edges: any pair with log2(lift) > log2_lift_edge.
    drawn_edges = []
    lifts_for_scale = []
    for i in range(n):
        for j in range(i + 1, n):
            if bool(mask.iat[i, j]):
                continue
            val = lf_log.iat[i, j]
            if not np.isfinite(val) or val <= log2_lift_edge:
                continue
            drawn_edges.append((kept[i], kept[j], float(val)))
            lifts_for_scale.append(float(val))

    # Continuous line-width mapping from log2(lift).
    # Linear interpolation between a minimum and maximum width so the
    # visual difference is obvious without a legend.
    if lifts_for_scale:
        lw_lo_val = max(min(lifts_for_scale), log2_lift_edge)
        lw_hi_val = max(lifts_for_scale)
    else:
        lw_lo_val, lw_hi_val = log2_lift_edge, log2_lift_edge + 1
    lw_min, lw_max = 0.8, 4.5

    def _edge_width(log2l):
        if lw_hi_val <= lw_lo_val + 1e-9:
            return 0.5 * (lw_min + lw_max)
        t = (log2l - lw_lo_val) / (lw_hi_val - lw_lo_val)
        t = max(0.0, min(1.0, t))
        return lw_min + t * (lw_max - lw_min)

    # Bezier that bulges toward the UPPER-RIGHT so every edge stays in
    # the upper-triangle blank region (i.e. x > y, with our flipped y).
    # Endpoints are (i+0.5, i+0.5) for both nodes, so a straight line
    # between two diagonal points already lies on the diagonal; to push
    # the curve off the diagonal into the upper-right triangle we use
    # control points displaced along the +x / -y direction (i.e. up in
    # screen space since y is flipped).
    def _bezier_upper_right(p0, p3, bulge=0.40):
        p0 = np.asarray(p0, dtype=float)
        p3 = np.asarray(p3, dtype=float)
        # Direction along the diagonal -> (d, d).  Perpendicular toward
        # upper-right region is (+1, -1) in data coords (y flipped =>
        # smaller y is visually higher).
        length = np.linalg.norm(p3 - p0) + 1e-9
        push = length * bulge / math.sqrt(2)
        offset = np.array([push, -push])
        c1 = p0 + (p3 - p0) / 3 + offset
        c2 = p0 + 2 * (p3 - p0) / 3 + offset
        ts = np.linspace(0, 1, 100)
        pts = (
            np.outer((1 - ts) ** 3, p0)
            + np.outer(3 * (1 - ts) ** 2 * ts, c1)
            + np.outer(3 * (1 - ts) * ts ** 2, c2)
            + np.outer(ts ** 3, p3)
        )
        return pts[:, 0], pts[:, 1]

    for (a, b, log2l) in drawn_edges:
        xs, ys = _bezier_upper_right(node_xy[a], node_xy[b], bulge=0.35)
        ax.plot(xs, ys,
                color="black",
                linewidth=_edge_width(log2l),
                alpha=0.85, zorder=3, solid_capstyle="round")

    # ---------------------------------------------------------------- 6
    # Uniform diagonal nodes -- smaller than a heat-map cell.
    # cell_size is 0.9 in data units; we want the node's visible
    # diameter to be a bit smaller -- use a circle patch so sizes stay
    # consistent across figure resolutions.
    node_radius = cell_size * 0.38   # clearly smaller than a cell
    for lab, (x, y) in node_xy.items():
        ax.add_patch(mpl.patches.Circle(
            (x, y), radius=node_radius,
            facecolor="#e63946", edgecolor="#1d3557",
            linewidth=0.8, zorder=6,
        ))

    # ---------------------------------------------------------------- 7
    # Bare colour-bar (keeps numeric ticks, but no textual label /
    # title).  Ticks are drawn on the LEFT of the bar so their
    # numbers do not collide with the heat-map y-axis category
    # labels that sit directly to the right of the cbar.
    sm = mpl.cm.ScalarMappable(cmap=lift_cmap, norm=lift_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#555")
    # Put the ticks / tick labels on the left side of the cbar.
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cax.tick_params(axis="y", which="both",
                    left=True, right=False,
                    labelleft=True, labelright=False,
                    length=4, width=0.8, labelsize=13,
                    colors="#333")
    cax.set_xticks([])
    # Remove the auto-generated textual label if any.
    cbar.set_label("")

    fig.savefig(OUT_DIR / f"{tag}_lift_network_heatmap.png",
                bbox_inches="tight")
    plt.close(fig)

    print(f"[{tag}] lift-network heatmap: {n} categories, "
          f"{len(drawn_edges)} edges  "
          f"-> {tag}_lift_network_heatmap.png")


def plot_heatmaps(stats: pd.DataFrame, labels: list[str], tag: str,
                  min_joint: int = HEATMAP_MIN_JOINT,
                  min_category: int = HEATMAP_MIN_CATEGORY):
    # --- Support mask shared by all heatmaps ---------------------------
    mask_full, _ = _build_support_mask(stats, labels,
                                       min_joint=min_joint,
                                       min_category=min_category)

    # Drop rows/cols that are fully masked -> compact, readable heatmap.
    labels = _drop_fully_masked(mask_full, labels)
    mask = mask_full.loc[labels, labels]

    n_cells = mask.size
    n_hidden = int(mask.values.sum())
    footer = _mask_footer(min_joint, min_category, n_hidden, n_cells)

    # A grey hatched background highlights the masked region so that
    # readers can still see *where* the holes are.
    def _overlay_mask(ax):
        hatched = mask.astype(int).values.astype(float)
        hatched[hatched == 0] = np.nan
        ax.imshow(hatched, cmap=mpl.colors.ListedColormap(["#dddddd"]),
                  aspect="auto", interpolation="nearest", zorder=0,
                  extent=(0, len(labels), len(labels), 0))
        # diagonal hatch pattern on masked cells
        for (i, j), v in np.ndenumerate(mask.values):
            if v:
                ax.add_patch(mpl.patches.Rectangle(
                    (j, i), 1, 1, fill=True, facecolor="#e6e6e6",
                    edgecolor="#bbbbbb", hatch="///", linewidth=0.0,
                    zorder=3))

    # --- Conditional probability  P(B|A)  : rows = A (cause) ------------
    cp = _pivot(stats, "P(B|A)", labels)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels) + 4),
                                    max(6, 0.55 * len(labels) + 3)))
    sns.heatmap(cp, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=.4, linecolor="white", mask=mask,
                cbar_kws={"label": "P(B | A)"}, ax=ax, vmin=0, vmax=1)
    _overlay_mask(ax)
    ax.set_title(f"Conditional probability  P(B | A)   ({tag})")
    ax.set_xlabel("B  (co-occurring error)")
    ax.set_ylabel("A  (conditioning error)")
    ax.text(0.0, 1.02, footer, transform=ax.transAxes,
            fontsize=8, color="#555", ha="left")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}_cond_prob_heatmap.png")
    plt.close(fig)

    # --- Lift -----------------------------------------------------------
    # IMPORTANT: the heatmap colour, the cell annotation, and the colour bar
    # must all live on the SAME scale.  Previously the cells were coloured
    # by log2(lift) while the numbers printed in each cell were the raw
    # lift value, which made the picture confusing (e.g. a cell annotated
    # "1.50" had the colour of "0.58" on the colour bar).  We therefore
    # now print log2(lift) in every cell and add a second colour-bar axis
    # that shows the corresponding linear lift value so users still have a
    # direct intuition of "how many times more frequent than chance".
    lf_full = _pivot(stats, "lift", labels)
    lf_log_full = np.log2(lf_full.replace(0, np.nan))

    # Extra "weak-signal" mask specific to the lift heatmap: hide cells
    # whose |log2(lift)| <= 1 (i.e. lift within [0.5, 2]).  Those are too
    # close to independence to be interesting.  Combine with the shared
    # support mask.
    weak_mask = (lf_log_full.abs() <= 1) | lf_log_full.isna()
    lift_mask_full = mask | weak_mask

    # After the combined mask is built, drop rows/cols that are now
    # fully hidden so the lift heatmap becomes as compact as possible.
    lift_labels = _drop_fully_masked(lift_mask_full, labels)
    lift_mask   = lift_mask_full.loc[lift_labels, lift_labels]
    lf_log      = lf_log_full.loc[lift_labels, lift_labels]

    n_cells_lift  = lift_mask.size
    n_hidden_lift = int(lift_mask.values.sum())
    pct_lift = 100.0 * n_hidden_lift / n_cells_lift if n_cells_lift else 0.0
    footer_lift = (
        f"Grey-hatched cells are masked because "
        f"(a) low support  (n_AB < {min_joint}  or  "
        f"min(n_A, n_B) < {min_category}),  or  "
        f"(b) weak signal  (|log2 lift| <= 1, i.e. lift in [0.5, 2]). "
        f"Masked: {n_hidden_lift}/{n_cells_lift} cells ({pct_lift:.1f}%)."
    )

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(lift_labels) + 4),
                                    max(6, 0.55 * len(lift_labels) + 3)))

    # Symmetric range so 0 (independence, lift=1) maps to white.
    finite = lf_log.values[np.isfinite(lf_log.values)]
    # Only look at *unmasked* finite values for the colour scale --
    # otherwise a masked 3.05 would still inflate vabs.
    visible = lf_log.mask(lift_mask).values
    visible = visible[np.isfinite(visible)]
    if visible.size:
        vabs = float(np.nanmax(np.abs(visible)))
    elif finite.size:
        vabs = float(np.nanmax(np.abs(finite)))
    else:
        vabs = 1.0
    vabs = max(vabs, 1e-6)

    # Local overlay for the (possibly smaller) lift-heatmap labels.
    def _overlay_lift_mask(ax_):
        for (i, j), v in np.ndenumerate(lift_mask.values):
            if v:
                ax_.add_patch(mpl.patches.Rectangle(
                    (j, i), 1, 1, fill=True, facecolor="#e6e6e6",
                    edgecolor="#bbbbbb", hatch="///", linewidth=0.0,
                    zorder=3))

    hm = sns.heatmap(
        lf_log,
        annot=lf_log.round(2), fmt="",
        cmap="RdBu_r", center=0, vmin=-vabs, vmax=vabs,
        linewidths=.4, linecolor="white", mask=lift_mask,
        cbar_kws={"label": "log2 Lift  (cell numbers = same scale)"},
        ax=ax, annot_kws={"fontsize": 8},
    )
    _overlay_lift_mask(ax)

    # Add a secondary colour-bar axis mapping log2(lift) ticks to the raw
    # linear lift value, so readers can still read "x3.0", "x0.5" etc.
    cbar = hm.collections[0].colorbar
    if cbar is not None:
        ticks = cbar.get_ticks()
        # Build a twin axis that shares the colour-bar y-limits.
        cbar_ax = cbar.ax
        twin = cbar_ax.twinx()
        twin.set_ylim(cbar_ax.get_ylim())
        twin.set_yticks(ticks)
        twin.set_yticklabels([f"×{2**t:.2g}" for t in ticks])
        twin.set_ylabel("linear lift  (= 2^log2Lift)", rotation=90,
                        labelpad=8)
        twin.tick_params(axis="y", labelsize=8)

    ax.set_title(
        f"Lift of co-occurrence   log2(P(A,B) / P(A)P(B))   ({tag})\n"
        f"shown only when |log2 lift| > 1  (i.e. lift < 0.5  or  lift > 2)  "
        f"AND support is sufficient",
        fontsize=10,
    )
    ax.set_xlabel("B"); ax.set_ylabel("A")
    ax.text(0.0, 1.10, footer_lift, transform=ax.transAxes,
            fontsize=8, color="#555", ha="left")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}_lift_heatmap.png")
    plt.close(fig)

    # --- Mantel-style network + lower-triangle lift heatmap -------------
    # Reuses the exact same masking rules (support + |log2 lift| > 1)
    # so readers can cross-read it against `{tag}_lift_heatmap.png`.
    plot_lift_network_heatmap(stats, labels, tag,
                              min_joint=min_joint,
                              min_category=min_category,
                              log2_lift_edge=1.0)

    # --- Joint co-occurrence counts -------------------------------------
    # Joint count heatmap keeps ALL cells visible (it is the raw support
    # that readers use to decide whether a lift value is trustworthy).
    co = _pivot(stats, "n_AB", labels).astype(int)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels) + 4),
                                    max(6, 0.55 * len(labels) + 3)))
    sns.heatmap(co, annot=True, fmt="d", cmap="Blues",
                linewidths=.4, linecolor="white",
                cbar_kws={"label": "#cases"}, ax=ax)
    ax.set_title(f"Joint occurrence counts   N(A, B)   ({tag})")
    ax.set_xlabel("B"); ax.set_ylabel("A")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}_cooccurrence_heatmap.png")
    plt.close(fig)


def export_visible_edges(stats: pd.DataFrame,
                         labels: list[str],
                         tag: str,
                         min_joint: int = HEATMAP_MIN_JOINT,
                         min_category: int = HEATMAP_MIN_CATEGORY,
                         log2_lift_min: float = 1.0,
                         filename_suffix: str = ""):
    """Export the list of directed pairs A -> B whose cell is *visible* in
    the lift heatmap under the current masking rules.

    A cell is considered visible when:
      * support is sufficient  (n_AB >= min_joint  and
        min(n_A, n_B) >= min_category), and
      * |log2(lift)| > log2_lift_min  (default 1  =>  lift > 2 or < 0.5).

    Because we only care about *positive* associations, the export also
    filters lift > 1, so the end-to-end rule is:
        lift > 2**log2_lift_min  (default: lift > 2)

    Outputs (the filename_suffix is inserted before the ".csv"):
      * ``{tag}_visible_lift_gt1_edges{suffix}.csv``
      * ``{tag}_visible_lift_gt1_edges_unordered{suffix}.csv``
    """
    # --- 1. Build the same combined mask that the heatmap uses ----------
    support_mask, _ = _build_support_mask(stats, labels,
                                         min_joint=min_joint,
                                         min_category=min_category)
    lf_full = _pivot(stats, "lift", labels)
    lf_log_full = np.log2(lf_full.replace(0, np.nan))
    weak_mask = (lf_log_full.abs() <= log2_lift_min) | lf_log_full.isna()
    combined_mask_full = support_mask | weak_mask

    visible_labels = _drop_fully_masked(combined_mask_full, labels)
    mask = combined_mask_full.loc[visible_labels, visible_labels]

    # --- 2. Keep cells that survive the mask AND have lift > 1 ----------
    lift_threshold = 2.0 ** log2_lift_min  # = 2 when log2_lift_min = 1
    keep_pairs = []
    for a in visible_labels:
        for b in visible_labels:
            if a == b or bool(mask.loc[a, b]):
                continue
            row = stats[(stats["A"] == a) & (stats["B"] == b)]
            if row.empty:
                continue
            r = row.iloc[0]
            if r["lift"] <= lift_threshold:
                continue
            keep_pairs.append({
                "A": a, "B": b,
                "n_A": int(r["n_A"]), "n_B": int(r["n_B"]),
                "n_AB": int(r["n_AB"]),
                "P(B|A)": round(float(r["P(B|A)"]), 4),
                "P(A|B)": round(float(r["P(A|B)"]), 4),
                "lift":   round(float(r["lift"]), 3),
                "log2_lift": round(math.log2(float(r["lift"])), 3),
                "asymmetry": round(float(r["asymmetry"]), 3),
                "q_value": round(float(r["q_value"]), 4),
            })

    edges = pd.DataFrame(keep_pairs)
    if not edges.empty:
        edges = edges.sort_values(
            ["lift", "n_AB"], ascending=[False, False])

    directed_path = OUT_DIR / \
        f"{tag}_visible_lift_gt1_edges{filename_suffix}.csv"
    edges.to_csv(directed_path, index=False)

    # --- 3. Deduplicate to unordered pairs ------------------------------
    seen, unordered = set(), []
    for _, row in edges.iterrows():
        key = frozenset([row["A"], row["B"]])
        if key in seen:
            continue
        seen.add(key)
        if row["asymmetry"] >= 0:
            src, dst = row["A"], row["B"]
            pba, pab = row["P(B|A)"], row["P(A|B)"]
        else:
            src, dst = row["B"], row["A"]
            pba, pab = row["P(A|B)"], row["P(B|A)"]
        unordered.append({
            "stronger_src": src, "stronger_dst": dst,
            "n_AB": row["n_AB"],
            "lift": row["lift"], "log2_lift": row["log2_lift"],
            "P(dst|src)": pba, "P(src|dst)": pab,
            "|asymmetry|": abs(row["asymmetry"]),
            "q_value": row["q_value"],
            "significant(q<0.05)": "yes" if row["q_value"] < 0.05 else "no",
        })
    unordered_df = pd.DataFrame(unordered)
    unordered_path = OUT_DIR / \
        f"{tag}_visible_lift_gt1_edges_unordered{filename_suffix}.csv"
    unordered_df.to_csv(unordered_path, index=False)

    # --- 4. Console summary --------------------------------------------
    kept_cats = len(visible_labels)
    dropped = [l for l in labels if l not in visible_labels]
    print(f"\n[{tag}] visible lift>{lift_threshold:g} edges"
          f"   (|log2 lift| > {log2_lift_min}, support OK)"
          f"   suffix='{filename_suffix or '(none)'}'")
    print(f"  categories kept   : {kept_cats}/{len(labels)}")
    if dropped:
        print(f"  categories dropped: {len(dropped)}")
        for d in dropped:
            print(f"    - {d}")
    print(f"  directed edges     : {len(edges)}")
    print(f"  unordered pairs    : {len(unordered_df)}")
    sig_n = int((unordered_df["significant(q<0.05)"] == "yes").sum()) \
        if len(unordered_df) else 0
    print(f"  significant (q<.05): {sig_n}")
    print(f"  -> {directed_path.name}")
    print(f"  -> {unordered_path.name}")




# --------------------------------------------------------------------------- #
# 5. Directed "causal-like" graph from asymmetry
# --------------------------------------------------------------------------- #
def build_directed_graph(stats: pd.DataFrame,
                         delta: float = DELTA_DIR,
                         lift_min: float = LIFT_MIN,
                         pval_max: float = 0.05,
                         use_qvalue: bool = True,
                         min_support: int = MIN_SUPPORT) -> nx.DiGraph:
    """Directed graph built from pairwise statistics.

    An edge A -> B is added when
      * the joint support is >= min_support
      * Lift(A, B) >= lift_min
      * significance (q-value if use_qvalue else raw p-value) <= pval_max
      * P(B|A) - P(A|B) >= delta   (direction is assigned to the larger)
    """
    G = nx.DiGraph()
    if stats.empty:
        return G
    seen = set()
    for _, r in stats.iterrows():
        a, b = r["A"], r["B"]
        if (a, b) in seen or (b, a) in seen:
            continue
        rev = stats[(stats["A"] == b) & (stats["B"] == a)]
        if rev.empty:
            continue
        rev = rev.iloc[0]
        # direction: use whichever conditional-probability is larger
        if r["asymmetry"] >= rev["asymmetry"]:
            src, dst, fwd = a, b, r
        else:
            src, dst, fwd = b, a, rev
        seen.add((a, b))

        if fwd["n_AB"] < min_support:
            continue
        if fwd["lift"] < lift_min:
            continue
        sig = fwd["q_value"] if use_qvalue else fwd["p_value"]
        if sig > pval_max:
            continue
        if abs(fwd["asymmetry"]) < delta:
            continue

        G.add_edge(
            src, dst,
            weight=float(fwd["P(B|A)"]),
            lift=float(fwd["lift"]),
            asym=float(fwd["asymmetry"]),
            qval=float(fwd["q_value"]),
            pval=float(fwd["p_value"]),
            n_ab=int(fwd["n_AB"]),
        )
    return G


def plot_directed_graph(G: nx.DiGraph, freq: dict, tag: str,
                        color_map: dict | None = None,
                        suffix: str = "causal_graph",
                        sig_label: str = "q",
                        sig_thr: float = QVAL_MAX,
                        lift_min: float = LIFT_MIN,
                        min_support: int = MIN_SUPPORT):
    if G.number_of_edges() == 0:
        print(f"[graph:{tag}:{suffix}] no edges pass thresholds")
        return

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(freq) + 6),
                                    max(8,  0.6 * len(freq) + 4)))
    pos = nx.spring_layout(G, k=2.0 / math.sqrt(max(len(G), 1)),
                           seed=42, iterations=200)

    # Node sizes by frequency
    max_freq = max(freq.values()) if freq else 1
    sizes = [300 + 2500 * (freq.get(n, 1) / max_freq) for n in G.nodes()]

    # Node colours
    if color_map:
        colours = [color_map.get(n, "#cccccc") for n in G.nodes()]
    else:
        colours = ["#4C9CEB"] * len(G.nodes())

    nx.draw_networkx_nodes(G, pos, node_size=sizes,
                           node_color=colours, edgecolors="black",
                           linewidths=1.0, ax=ax)

    # Edge widths / colours proportional to lift
    lifts = np.array([G[u][v]["lift"] for u, v in G.edges()])
    widths = 0.8 + 3.0 * (lifts / (lifts.max() if lifts.max() > 0 else 1))
    edge_colors = lifts
    edges = nx.draw_networkx_edges(
        G, pos,
        width=widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.plasma,
        arrows=True, arrowsize=18,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        node_size=sizes, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    # Edge labels: P(B|A) and lift
    elabels = {(u, v): f"{G[u][v]['weight']:.2f} | L={G[u][v]['lift']:.1f}"
               for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=elabels, font_size=7,
        label_pos=0.5, bbox=dict(boxstyle="round,pad=0.2",
                                 fc="white", ec="none", alpha=0.7),
        ax=ax,
    )
    ax.set_title(
        f"Directed causal-like graph of SQL errors  ({tag})\n"
        f"edge A→B  :  P(B|A) ≫ P(A|B),  lift≥{lift_min}, "
        f"{sig_label}≤{sig_thr}, support≥{min_support}"
    )
    ax.axis("off")

    # colour bar for edges
    if len(lifts):
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=mpl.colors.Normalize(vmin=lifts.min(),
                                                              vmax=lifts.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Lift")

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}_{suffix}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. PC algorithm (optional) for DAG discovery
# --------------------------------------------------------------------------- #
def try_pc_algorithm(df: pd.DataFrame, tag: str, freq: dict,
                     color_map: dict | None = None):
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import CIT
    except Exception as exc:
        print(f"[pc:{tag}] causal-learn not available ({exc}); skip.")
        return

    data = df.values.astype(float)
    # drop all-zero columns to avoid zero-variance issues
    keep = [i for i in range(data.shape[1]) if data[:, i].sum() > 0]
    cols = [df.columns[i] for i in keep]
    data = data[:, keep]
    if data.shape[1] < 2:
        return

    try:
        cg = pc(data, alpha=0.05, indep_test="chisq",
                verbose=False, show_progress=False)
    except Exception as exc:
        print(f"[pc:{tag}] failed: {exc}")
        return

    G = nx.DiGraph()
    G.add_nodes_from(cols)
    A = cg.G.graph  # adjacency: 1 / -1 encoding from causal-learn
    # In causallearn:  A[i,j] = -1 and A[j,i] = 1 -> i -> j
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue
            if A[i, j] == -1 and A[j, i] == 1:
                G.add_edge(cols[i], cols[j])

    if G.number_of_edges() == 0:
        print(f"[pc:{tag}] PC learned no directed edges")
        return

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(cols) + 6),
                                    max(8, 0.6 * len(cols) + 4)))
    pos = nx.spring_layout(G, k=2.0 / math.sqrt(max(len(G), 1)),
                           seed=42, iterations=200)
    max_freq = max(freq.values()) if freq else 1
    sizes = [300 + 2500 * (freq.get(n, 1) / max_freq) for n in G.nodes()]
    colours = [color_map.get(n, "#B7D7F5") for n in G.nodes()] if color_map \
        else ["#B7D7F5"] * len(G.nodes())
    nx.draw_networkx_nodes(G, pos, node_size=sizes,
                           node_color=colours, edgecolors="black",
                           linewidths=1.0, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=18,
                           arrowstyle="-|>", width=1.5,
                           edge_color="#555",
                           connectionstyle="arc3,rad=0.08",
                           node_size=sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    ax.set_title(f"PC-algorithm learned DAG  ({tag},  alpha=0.05, chisq)")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}_pc_dag.png")
    plt.close(fig)
    # persist
    nx.write_gexf(G, OUT_DIR / f"{tag}_pc_dag.gexf")


# --------------------------------------------------------------------------- #
# 7. Per-category bar plot
# --------------------------------------------------------------------------- #
def plot_frequency(freq: dict, tag: str, color_map: dict | None = None,
                   order: list[str] | None = None):
    """Plot category frequency bar chart.

    If *order* is provided, the bars follow that exact order; otherwise the
    bars are sorted by frequency descending (previous behaviour).
    """
    if order is not None:
        names  = [n for n in order if n in freq]
        values = [freq[n] for n in names]
    else:
        items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        names  = [x[0] for x in items]
        values = [x[1] for x in items]
    colours = [color_map.get(n, "#4C9CEB") for n in names] if color_map \
        else ["#4C9CEB"] * len(names)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(names) + 3), 5))
    ax.bar(range(len(names)), values, color=colours, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("#error cases containing this category")
    ax.set_title(f"Frequency of error categories  ({tag})")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center",
                fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}_frequency.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 8. Entry point
# --------------------------------------------------------------------------- #
def main():
    data = load_dataset(DATA_PATH)
    df_main, df_sub, sub_to_main = build_matrices(data)

    # ----- dataset-level summary ------------------------------------------
    total = len(data)
    n_err = int(sum(1 for r in data if r.get("label") is False))
    n_correct = total - n_err
    summary = {
        "total_cases": total,
        "correct_cases": n_correct,
        "error_cases": n_err,
        "error_cases_with_tags": int(len(df_main)),
        "main_categories": list(df_main.columns),
        "sub_categories":  list(df_sub.columns),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # colour main categories; then propagate to sub categories
    palette = sns.color_palette("tab10", n_colors=len(df_main.columns))
    main_colors = {c: mpl.colors.to_hex(palette[i])
                   for i, c in enumerate(df_main.columns)}
    sub_colors  = {s: main_colors.get(sub_to_main.get(s, ""), "#cccccc")
                   for s in df_sub.columns}

    # ----------- MAIN CATEGORY LEVEL --------------------------------------
    print("\n========== MAIN ==========")
    stats_main = compute_stats(df_main)
    stats_main.to_csv(OUT_DIR / "main_pairwise_stats.csv", index=False)

    freq_main = df_main.sum(axis=0).to_dict()
    plot_frequency(freq_main, "main", main_colors,
                   order=list(df_main.columns))
    plot_heatmaps(stats_main, list(df_main.columns), "main")

    G_main = build_directed_graph(stats_main, use_qvalue=True,
                                  pval_max=QVAL_MAX, lift_min=LIFT_MIN)
    plot_directed_graph(G_main, freq_main, "main", main_colors,
                        suffix="causal_graph_strict",
                        sig_label="q", sig_thr=QVAL_MAX,
                        lift_min=LIFT_MIN)
    nx.write_gexf(G_main, OUT_DIR / "main_causal_graph_strict.gexf")

    # Exploratory version: raw p-value, lift >= 1.2
    G_main_exp = build_directed_graph(stats_main, use_qvalue=False,
                                      pval_max=0.05, lift_min=1.2)
    plot_directed_graph(G_main_exp, freq_main, "main", main_colors,
                        suffix="causal_graph_exploratory",
                        sig_label="p", sig_thr=0.05, lift_min=1.2)
    nx.write_gexf(G_main_exp, OUT_DIR / "main_causal_graph_exploratory.gexf")

    try_pc_algorithm(df_main, "main", freq_main, main_colors)

    # ----------- SUB CATEGORY LEVEL ---------------------------------------
    print("\n========== SUB ==========")
    stats_sub = compute_stats(df_sub)
    stats_sub.to_csv(OUT_DIR / "sub_pairwise_stats.csv", index=False)

    freq_sub = df_sub.sum(axis=0).to_dict()
    plot_frequency(freq_sub, "sub", sub_colors,
                   order=list(df_sub.columns))
    plot_heatmaps(stats_sub, list(df_sub.columns), "sub")

    G_sub = build_directed_graph(stats_sub, use_qvalue=True,
                                 pval_max=QVAL_MAX, lift_min=LIFT_MIN)
    plot_directed_graph(G_sub, freq_sub, "sub", sub_colors,
                        suffix="causal_graph_strict",
                        sig_label="q", sig_thr=QVAL_MAX,
                        lift_min=LIFT_MIN)
    nx.write_gexf(G_sub, OUT_DIR / "sub_causal_graph_strict.gexf")

    G_sub_exp = build_directed_graph(stats_sub, use_qvalue=False,
                                     pval_max=0.05, lift_min=1.5)
    plot_directed_graph(G_sub_exp, freq_sub, "sub", sub_colors,
                        suffix="causal_graph_exploratory",
                        sig_label="p", sig_thr=0.05, lift_min=1.5)
    nx.write_gexf(G_sub_exp, OUT_DIR / "sub_causal_graph_exploratory.gexf")

    try_pc_algorithm(df_sub, "sub", freq_sub, sub_colors)

    # ----------- TOP pair report ------------------------------------------
    for name, st in [("main", stats_main), ("sub", stats_sub)]:
        st_ok = st[(st["n_AB"] >= MIN_SUPPORT) & (st["lift"] > 1)]
        top = (st_ok.sort_values("lift", ascending=False)
                    .head(30)[["A", "B", "n_AB", "lift", "P(B|A)",
                               "P(A|B)", "asymmetry", "q_value"]])
        top.to_csv(OUT_DIR / f"{name}_top_pairs_by_lift.csv", index=False)

    # ----------- Visible lift edges (after support mask) ----------------
    # Two outputs per granularity level so users can cross-check:
    #   (a) lift>1  with only the support-mask applied           (baseline)
    #   (b) lift>2  with support-mask + |log2 lift|>1 applied    (strong)
    # (b) matches what is actually drawn on the new lift heatmap.
    for name, st, labels_full in [
        ("main", stats_main, list(df_main.columns)),
        ("sub",  stats_sub,  list(df_sub.columns)),
    ]:
        # (a) baseline: lift > 1, support mask only
        export_visible_edges(st, labels_full, tag=name,
                             log2_lift_min=0.0,
                             filename_suffix="")
        # (b) strong: lift > 2, matches the updated lift heatmap
        export_visible_edges(st, labels_full, tag=name,
                             log2_lift_min=1.0,
                             filename_suffix="_strong")

    print("\nAll outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
