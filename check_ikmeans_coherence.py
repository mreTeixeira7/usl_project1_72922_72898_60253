#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from iKMeans import ikmeans_initialize


EXPECTED_CLUSTERS_1BASED = [
    {
        "name": "Cluster 1",
        "indices": {
            56, 63, 70, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114,
            116, 120, 122, 123, 124, 128, 130, 131, 132, 134, 135, 136, 137, 140, 141, 142,
            143, 144, 145, 146, 147, 149, 150
        },
        "centroid_real": np.array([6.85, 3.08, 5.72, 2.05], dtype=np.float64),
    },
    {
        "name": "Cluster 2",
        "indices": set(range(1, 51)),
        "centroid_real": np.array([5.01, 3.43, 1.46, 0.25], dtype=np.float64),
    },
    {
        "name": "Cluster 3",
        "indices": {
            51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 71, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
            94, 95, 96, 97, 98, 99, 100, 113, 115, 117, 118, 119, 121, 125, 126, 127, 129,
            133, 138, 139, 148
        },
        "centroid_real": np.array([5.88, 2.74, 4.39, 1.43], dtype=np.float64),
    },
]


@dataclass
class MatchedCluster:
    expected_name: str
    predicted_id: int
    expected_size: int
    predicted_size: int
    jaccard: float
    centroid_mae: float
    centroid_max_abs: float


def jaccard(a: set[int], b: set[int]) -> float:
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return inter / union


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check coherence of iKMeans output vs expected iris.report.")
    p.add_argument("--data", default=Path(__file__).with_name("iris.dat"), help="Path to iris.dat")
    p.add_argument("--min-cluster-size", type=int, default=10, help="min_cluster_size passed to iKMeans")
    p.add_argument("--jaccard-threshold", type=float, default=0.85, help="Pass threshold for mean Jaccard")
    p.add_argument(
        "--centroid-max-abs-threshold",
        type=float,
        default=0.25,
        help="Pass threshold for worst matched centroid max-abs error",
    )
    return p.parse_args()


def format_indices(indices: list[int] | set[int], max_per_line: int = 10) -> str:
    """Format indices for display, wrapping to multiple lines if needed."""
    sorted_indices = sorted(indices)
    lines = []
    for i in range(0, len(sorted_indices), max_per_line):
        chunk = sorted_indices[i : i + max_per_line]
        lines.append("   " + "  ".join(f"{idx:3d}" for idx in chunk))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    X = np.loadtxt(data_path, dtype=np.float64)

    # Feature names (for iris dataset)
    feature_names = ["seple", "sepwi", "petle", "petwi"]

    # 1) Run iKMeans initialization stage.
    ap_clusters, _ = ikmeans_initialize(
        X,
        min_cluster_size=args.min_cluster_size,
        tol=1e-12,
        max_iter=10_000,
        use_unit_ranges=False,
    )
    retained = [c for c in ap_clusters if c.size >= args.min_cluster_size]
    if not retained:
        raise RuntimeError("No retained anomalous clusters; cannot perform coherence check.")

    init_raw = np.vstack([c.centroid_raw for c in retained])
    k_pred = init_raw.shape[0]

    # 2) Run downstream KMeans from iKMeans seeds (report-like final clusters).
    km = KMeans(
        n_clusters=k_pred,
        init=init_raw,
        n_init=1,
        max_iter=300,
        random_state=0,
    )
    labels = km.fit_predict(X)

    pred_sets_1based: list[set[int]] = []
    pred_centroids: list[np.ndarray] = []
    pred_indices_0based: list[list[int]] = []
    for cid in range(k_pred):
        idx0 = np.where(labels == cid)[0]
        pred_sets_1based.append(set((idx0 + 1).tolist()))
        pred_centroids.append(X[idx0].mean(axis=0))
        pred_indices_0based.append(idx0.tolist())

    # 3) Match predicted clusters to expected clusters by best mean Jaccard.
    expected = EXPECTED_CLUSTERS_1BASED
    k_exp = len(expected)

    if k_pred != k_exp:
        print("FAIL: number of clusters differs from expected report.")

    m = min(k_exp, k_pred)
    best_perm = None
    best_score = -1.0
    for perm in itertools.permutations(range(k_pred), m):
        score = 0.0
        for i in range(m):
            score += jaccard(expected[i]["indices"], pred_sets_1based[perm[i]])
        score /= m
        if score > best_score:
            best_score = score
            best_perm = perm

    assert best_perm is not None

    matched: list[MatchedCluster] = []
    for i in range(m):
        exp = expected[i]
        pid = best_perm[i]
        jac = jaccard(exp["indices"], pred_sets_1based[pid])
        cdiff = np.abs(pred_centroids[pid] - exp["centroid_real"])
        matched.append(
            MatchedCluster(
                expected_name=exp["name"],
                predicted_id=pid,
                expected_size=len(exp["indices"]),
                predicted_size=len(pred_sets_1based[pid]),
                jaccard=jac,
                centroid_mae=float(cdiff.mean()),
                centroid_max_abs=float(cdiff.max()),
            )
        )

    # ===== Generate report-like output =====
    print(f"Intelligent K-Means resulted in {k_pred} clusters; at data not normalized")
    print(f"Anomalous pattern cardinality to discard = {args.min_cluster_size}")
    print()

    # Compute global mean
    global_mean = X.mean(axis=0)
    global_min = X.min(axis=0)
    global_max = X.max(axis=0)

    print("Features involved:")
    for i, fname in enumerate(feature_names):
        print(f"{fname:<8}Mean = {global_mean[i]:6.2f}")
    print()

    # Print each cluster
    cumulative_contribution = 0.0
    for cluster_num, row in enumerate(matched, 1):
        pid = row.predicted_id
        indices_0based = pred_indices_0based[pid]
        indices_1based = sorted([i + 1 for i in indices_0based])
        centroid_raw = pred_centroids[pid]
        cluster_size = len(indices_0based)

        print(f" Cluster {cluster_num}  ({cluster_size}):")
        print(format_indices(indices_1based))

        # Compute centroid in standardized space
        scales = global_max - global_min
        scales = np.where(scales == 0.0, 1.0, scales)
        centroid_std = (centroid_raw - global_mean) / scales

        # Compute % over/under grand mean
        pct_diff = np.where(
            global_mean != 0,
            100.0 * (centroid_raw - global_mean) / global_mean,
            0.0
        )

        # Print centroid info
        centroid_str = "\t".join(f"{v:.2f}" for v in centroid_raw)
        print(f" Cluster centroid (real) {centroid_str}\t")

        centroid_std_str = "\t".join(f"{v:.3f}" for v in centroid_std)
        print(f" Cluster centroid (stand) {centroid_std_str}\t")

        pct_str = "\t".join(f"{v:5.1f}" for v in pct_diff)
        print(f" Centroid (% over/under grand mean) {pct_str}\t")

        # Compute cluster contribution
        cluster_scatter = cluster_size * np.dot(centroid_std, centroid_std)
        total_scatter_computed = np.sum(((X - global_mean) / scales) ** 2)
        proper_contribution = cluster_scatter / total_scatter_computed if total_scatter_computed > 0 else 0.0
        cumulative_contribution += proper_contribution

        contrib_str = f"{proper_contribution:.3f}\t{cumulative_contribution:.3f}"
        print(f" Cluster contribution (proper and cumulative) {contrib_str}\t")

        # Features significantly larger/smaller
        larger = []
        smaller = []
        threshold_pct = 10.0
        for i, fname in enumerate(feature_names):
            if pct_diff[i] > threshold_pct:
                larger.append(f"{fname} ({pct_diff[i]:.0f}%)")
            elif pct_diff[i] < -threshold_pct:
                smaller.append(f"{fname} ({pct_diff[i]:.0f}%)")

        larger_str = ", ".join(larger) + ("," if larger else "")
        print(f" Features significantly larger than average: {larger_str}")

        smaller_str = ", ".join(smaller) + ("," if smaller else "")
        print(f" Features significantly smaller than average: {smaller_str}")
        print()

    # Print validation summary
    mean_jaccard = float(np.mean([r.jaccard for r in matched]))
    worst_centroid_max_abs = float(np.max([r.centroid_max_abs for r in matched]))
    k_ok = (k_pred == k_exp)
    jac_ok = (mean_jaccard >= args.jaccard_threshold)
    cen_ok = (worst_centroid_max_abs <= args.centroid_max_abs_threshold)

    print("=" * 60)
    print("VALIDATION SUMMARY:")
    print(f"mean_jaccard={mean_jaccard:.4f} (threshold={args.jaccard_threshold})")
    print(f"worst_centroid_max_abs={worst_centroid_max_abs:.4f} (threshold={args.centroid_max_abs_threshold})")
    print(f"k_match={k_ok}")
    print("=" * 60)

    coherent = k_ok and jac_ok and cen_ok
    print("PASS" if coherent else "FAIL")
    if not coherent:
        print("Result is not coherent enough with the expected iris.report under current thresholds.")


if __name__ == "__main__":
    main()
