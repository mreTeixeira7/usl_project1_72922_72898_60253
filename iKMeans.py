from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class APCluster:
    indices: list[int]
    centroid_raw: FloatArray
    centroid_std: FloatArray
    size: int
    scatter_pct: float


def _as_float_matrix(X: FloatArray) -> FloatArray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2D numeric matrix, got ndim={arr.ndim}.")
    if arr.shape[0] == 0:
        raise ValueError("X must contain at least one row.")
    return arr


def compute_feature_statistics(
    X: FloatArray,
    use_unit_ranges: bool = False,
) -> tuple[FloatArray, FloatArray, float]:
    X = _as_float_matrix(X)
    mean = X.mean(axis=0, dtype=np.float64)

    if use_unit_ranges:
        scales = np.ones(X.shape[1], dtype=np.float64)
    else:
        scales = X.max(axis=0) - X.min(axis=0)
        scales = np.where(scales == 0.0, 1.0, scales).astype(np.float64)

    Y = (X - mean) / scales
    total_scatter = float(np.sum(Y * Y, dtype=np.float64))
    return mean, scales, total_scatter


def normalized_squared_distances(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    reference: FloatArray,
) -> FloatArray:
    X = _as_float_matrix(X)
    if len(indices) == 0:
        return np.array([], dtype=np.float64)

    idx = np.asarray(indices, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    diff = (X[idx] - reference) / scales
    return np.einsum("ij,ij->i", diff, diff, dtype=np.float64)


def cluster_centroid(
    X: FloatArray,
    indices: list[int],
) -> FloatArray:
    X = _as_float_matrix(X)
    if len(indices) == 0:
        raise ValueError("cluster_centroid received an empty set of indices.")
    idx = np.asarray(indices, dtype=np.int64)
    return X[idx].mean(axis=0, dtype=np.float64)


def separate_cluster(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    a: FloatArray,
    b: FloatArray,
) -> list[int]:
    X = _as_float_matrix(X)
    if len(indices) == 0:
        return []

    idx = np.asarray(indices, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    da = np.einsum("ij,ij->i", (X[idx] - a) / scales, (X[idx] - a) / scales, dtype=np.float64)
    db = np.einsum("ij,ij->i", (X[idx] - b) / scales, (X[idx] - b) / scales, dtype=np.float64)

    selected = idx[da < db]
    return np.sort(selected).astype(int).tolist()


def extract_anomalous_cluster(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    mean: FloatArray,
    initial_centroid: FloatArray,
    seed_index: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[list[int], FloatArray]:
    X = _as_float_matrix(X)
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if tol <= 0:
        raise ValueError("tol must be strictly positive.")

    # Compatibility behavior (matches instructor iris.report pattern):
    # one centroid update + one reassignment step.
    c = np.asarray(initial_centroid, dtype=np.float64).copy()
    mean = np.asarray(mean, dtype=np.float64)

    members = separate_cluster(X, indices, scales, c, mean)
    if not members:
        members = [int(seed_index)]

    c_new = cluster_centroid(X, members)

    members_2 = separate_cluster(X, indices, scales, c_new, mean)
    if not members_2:
        members_2 = [int(seed_index)]

    c_final = cluster_centroid(X, members_2)
    return members_2, c_final


def ikmeans_initialize(
    X: FloatArray,
    min_cluster_size: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    use_unit_ranges: bool = False,
) -> tuple[list[APCluster], FloatArray]:
    X = _as_float_matrix(X)

    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be a positive integer.")
    if tol <= 0:
        raise ValueError("tol must be strictly positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    mean, scales, total_scatter = compute_feature_statistics(X, use_unit_ranges=use_unit_ranges)
    remains: list[int] = list(range(X.shape[0]))
    ap_clusters: list[APCluster] = []

    while remains:
        # Use residual grand mean at each extraction step (matches reference behavior).
        mean_rem = cluster_centroid(X, remains)
        d_to_mean = normalized_squared_distances(X, remains, scales, mean_rem)
        farthest_pos = int(np.argmax(d_to_mean))
        seed_index = remains[farthest_pos]
        seed = X[seed_index].copy()

        members, centroid_raw = extract_anomalous_cluster(
            X=X,
            indices=remains,
            scales=scales,
            mean=mean_rem,
            initial_centroid=seed,
            seed_index=seed_index,
            tol=tol,
            max_iter=max_iter,
        )
        members = sorted(int(i) for i in members)

        # If anomalous set is too small, discard it and use residual centroid as final seed.
        if len(members) <= min_cluster_size:
            residual_indices = sorted(remains)
            centroid_raw = cluster_centroid(X, residual_indices)
            centroid_std = (centroid_raw - mean) / scales
            norm2 = float(np.dot(centroid_std, centroid_std))
            if total_scatter > 0.0:
                scatter_pct = float(100.0 * len(residual_indices) * norm2 / total_scatter)
            else:
                scatter_pct = 0.0

            ap_clusters.append(
                APCluster(
                    indices=residual_indices,
                    centroid_raw=np.asarray(centroid_raw, dtype=np.float64),
                    centroid_std=np.asarray(centroid_std, dtype=np.float64),
                    size=len(residual_indices),
                    scatter_pct=scatter_pct,
                )
            )
            break

        centroid_std = (centroid_raw - mean) / scales
        norm2 = float(np.dot(centroid_std, centroid_std))
        if total_scatter > 0.0:
            scatter_pct = float(100.0 * len(members) * norm2 / total_scatter)
        else:
            scatter_pct = 0.0

        ap_clusters.append(
            APCluster(
                indices=members,
                centroid_raw=np.asarray(centroid_raw, dtype=np.float64),
                centroid_std=np.asarray(centroid_std, dtype=np.float64),
                size=len(members),
                scatter_pct=scatter_pct,
            )
        )

        member_set = set(members)
        remains = [i for i in remains if i not in member_set]

    retained = [rec for rec in ap_clusters if rec.size > min_cluster_size]
    if not retained:
        raise ValueError("No anomalous cluster satisfies the minimum size.")

    init_centroids = np.vstack([rec.centroid_std for rec in retained]).astype(np.float64)
    return ap_clusters, init_centroids
