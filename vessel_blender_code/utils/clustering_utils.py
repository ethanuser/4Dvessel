"""
Clustering utilities for DBSCAN and cluster trajectory computation.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.cluster import DBSCAN
import vessel_utils as vu


def compute_dbscan_clusters_for_frames(
    frames_data: Dict[int, Any],
    eps: float = 0.045,
    min_samples: int = 2,
) -> Dict[int, np.ndarray]:
    """
    Compute DBSCAN cluster means for each frame, matching the logic from
    numpy_to_displacements_interactive.py.

    Args:
        frames_data: Dictionary mapping frame indices to frame data
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        Dictionary mapping frame indices to cluster means arrays
    """
    frame_indices = sorted(frames_data.keys())
    cluster_means_dict = {}

    # Cluster the first frame to establish base structure
    first_frame_idx = frame_indices[0]
    first_frame_data = frames_data[first_frame_idx]
    first_coords = first_frame_data["coords"]

    if first_coords.shape[0] == 0:
        print("Error: First frame has no coordinates!")
        return cluster_means_dict

    # Apply DBSCAN clustering to first frame
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(first_coords)
    base_labels = clustering.labels_
    base_cluster_means, _ = vu.calculate_cluster_means(first_coords, base_labels)

    if len(base_cluster_means) == 0:
        print("Warning: No clusters found in first frame!")
        base_cluster_means = np.array([]).reshape(0, 3)

    print(f"  Base clusters: {len(base_cluster_means)} cluster means")

    # Process all frames
    for idx, frame_idx in enumerate(frame_indices):
        vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)

        frame_data = frames_data[frame_idx]
        coords = frame_data["coords"]

        if coords.shape[0] == 0 or len(base_cluster_means) == 0:
            cluster_means_dict[frame_idx] = np.array([]).reshape(0, 3)
            continue

        # For first frame, use the base cluster means
        if idx == 0:
            cluster_means = base_cluster_means.copy()
        else:
            # Subsequent frames: use labels from first frame to assign points to clusters
            new_cluster_means = []

            # Get unique labels (excluding noise label -1), sorted to maintain order
            unique_labels = sorted([k for k in np.unique(base_labels) if k != -1])

            # If the point count matches, we can reuse base_labels directly
            if len(coords) == len(base_labels):
                for k in unique_labels:
                    cluster_mask = base_labels == k
                    if np.any(cluster_mask):
                        new_cluster_mean = np.mean(coords[cluster_mask], axis=0)
                        new_cluster_means.append(new_cluster_mean)
                    else:
                        # No points for this label, fall back to previous frame's cluster mean
                        prev_frame_idx = frame_indices[idx - 1]
                        prev_cluster_means = cluster_means_dict.get(
                            prev_frame_idx, base_cluster_means
                        )
                        cluster_idx = len(new_cluster_means)
                        if cluster_idx < len(prev_cluster_means):
                            new_cluster_means.append(prev_cluster_means[cluster_idx])
                        else:
                            new_cluster_means.append(base_cluster_means[cluster_idx])
            else:
                # Different number of points - assign based on proximity to previous cluster positions
                prev_frame_idx = frame_indices[idx - 1]
                prev_cluster_means = cluster_means_dict.get(
                    prev_frame_idx, base_cluster_means
                )

                if len(prev_cluster_means) != len(base_cluster_means):
                    prev_cluster_means = base_cluster_means.copy()

                new_cluster_means = np.zeros_like(base_cluster_means)
                cluster_point_counts = np.zeros(len(base_cluster_means), dtype=int)

                for point in coords:
                    distances = np.linalg.norm(prev_cluster_means - point, axis=1)
                    closest_cluster_idx = np.argmin(distances)
                    new_cluster_means[closest_cluster_idx] += point
                    cluster_point_counts[closest_cluster_idx] += 1

                for k in range(len(base_cluster_means)):
                    if cluster_point_counts[k] > 0:
                        new_cluster_means[k] /= cluster_point_counts[k]
                    else:
                        new_cluster_means[k] = prev_cluster_means[k]

            # Convert to numpy array if it's a list
            if isinstance(new_cluster_means, list):
                cluster_means = np.array(new_cluster_means)
            else:
                cluster_means = new_cluster_means

            # Ensure cluster_means has the same length as base_cluster_means
            if len(cluster_means) != len(base_cluster_means):
                if len(cluster_means) < len(base_cluster_means):
                    padded = np.zeros_like(base_cluster_means)
                    padded[: len(cluster_means)] = cluster_means
                    padded[len(cluster_means) :] = base_cluster_means[
                        len(cluster_means) :
                    ]
                    cluster_means = padded
                else:
                    cluster_means = cluster_means[: len(base_cluster_means)]

        cluster_means_dict[frame_idx] = cluster_means

    print()  # New line after progress bar
    return cluster_means_dict

