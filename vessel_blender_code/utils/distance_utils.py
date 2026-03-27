"""
Distance calculation utilities for point cloud comparison.
"""

from typing import Tuple
import numpy as np
from scipy.spatial.distance import cdist


def compute_chamfer_distance(
    points_a: np.ndarray, points_b: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute chamfer distance between two point clouds.

    Args:
        points_a: First point cloud (N, 3)
        points_b: Second point cloud (M, 3)

    Returns:
        Tuple of (chamfer_distance, a_to_b_mean, b_to_a_mean)
        - chamfer_distance: Symmetric chamfer distance
        - a_to_b_mean: Mean distance from each point in A to nearest in B
        - b_to_a_mean: Mean distance from each point in B to nearest in A
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf"), float("inf"), float("inf")

    # Compute pairwise distances
    distances = cdist(points_a, points_b)

    # For each point in A, find nearest point in B
    a_to_b_distances = np.min(distances, axis=1)
    a_to_b_mean = float(np.mean(a_to_b_distances))

    # For each point in B, find nearest point in A
    b_to_a_distances = np.min(distances, axis=0)
    b_to_a_mean = float(np.mean(b_to_a_distances))

    # Symmetric chamfer distance
    chamfer_distance = (a_to_b_mean + b_to_a_mean) / 2.0

    return chamfer_distance, a_to_b_mean, b_to_a_mean


def compute_hausdorff_95th_percentile(
    points_a: np.ndarray, points_b: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute 95th percentile Hausdorff distance between two point clouds.

    The 95th percentile Hausdorff distance is computed as:
    - For each point in A, find distance to nearest point in B
    - For each point in B, find distance to nearest point in A
    - Take the 95th percentile of each set of distances
    - Return the maximum of the two (symmetric Hausdorff distance)

    Args:
        points_a: First point cloud (N, 3)
        points_b: Second point cloud (M, 3)

    Returns:
        Tuple of (hausdorff_95, a_to_b_95, b_to_a_95)
        - hausdorff_95: 95th percentile Hausdorff distance (max of the two percentiles)
        - a_to_b_95: 95th percentile of distances from A to B
        - b_to_a_95: 95th percentile of distances from B to A
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf"), float("inf"), float("inf")

    # Compute pairwise distances
    distances = cdist(points_a, points_b)

    # For each point in A, find nearest point in B
    a_to_b_distances = np.min(distances, axis=1)
    a_to_b_95 = float(np.percentile(a_to_b_distances, 95))

    # For each point in B, find nearest point in A
    b_to_a_distances = np.min(distances, axis=0)
    b_to_a_95 = float(np.percentile(b_to_a_distances, 95))

    # Symmetric 95th percentile Hausdorff distance (max of the two)
    hausdorff_95 = max(a_to_b_95, b_to_a_95)

    return hausdorff_95, a_to_b_95, b_to_a_95

