"""
Export cluster positions and mesh connections as a function of time.

This module mirrors the clustering / state-loading logic from stress analysis,
but instead of visualizing stresses it exports the cluster centroids and
their mesh connectivity over time into a single NumPy file.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np

from core.clustering import (
    color_then_spatial_clustering,
    calculate_cluster_means,
    create_delaunay_edges,
    remove_outlier_edges,
)
from utils.clustering_state_utils import (
    find_clustering_state_for_experiment,
    load_saved_clustering_state,
    remap_labels_to_saved_means,
)

# Flag to skip export if the file already exists
SKIP_EXISTING_EXPORTS = False


def _compute_cluster_trajectories(
    labels: np.ndarray,
    initial_cluster_means: np.ndarray,
    dataset: np.ndarray,
    use_saved_edges: bool,
    saved_edges: Optional[np.ndarray],
    config_manager,
) -> Dict[str, Any]:
    """
    Compute cluster centroids over time using fixed clustering labels.

    This follows the pattern used in stress_analysis.visualize_deformation:
    - labels are fixed, defining which points belong to each cluster index k
    - for each frame, we recompute the centroid for each k; if a cluster has
      no points in the current frame, we keep the previous position.

    After computing raw centroids, an optional spatial coherence step applies
    Graph Laplacian smoothing on per-frame displacements to reduce tearing
    between adjacent clusters while preserving overall displacement.

    Returns a dictionary with:
        - cluster_positions: (T, K, 3) array of centroids over time
        - edges: (E, 2) array of edge indices
        - initial_lengths: (E,) initial edge lengths (for reference)
    """
    # Extract config for edge construction if we need to build fresh edges
    edge_outlier_threshold = config_manager.get(
        "clustering.edge_outlier_threshold", 0.25
    )

    # ------------------------------------------------------------------
    # Spatial coherence config knobs (safe defaults)
    # ------------------------------------------------------------------
    enable_coherence = config_manager.get(
        "clustering.enable_neighbor_coherence", True
    )
    coherence_alpha = config_manager.get(
        "clustering.coherence_alpha", 0.1
    )
    coherence_iters = config_manager.get(
        "clustering.coherence_iters", 1
    )
    coherence_use_robust = config_manager.get(
        "clustering.coherence_use_robust_weights", True
    )
    coherence_huber_k = config_manager.get(
        "clustering.coherence_huber_k", 2.5
    )
    coherence_ref = config_manager.get(
        "clustering.coherence_ref", "t0"
    )

    cluster_means = initial_cluster_means

    # Decide which edges to use
    if use_saved_edges and saved_edges is not None and len(saved_edges) > 0:
        edges = saved_edges.copy()
        vertices = cluster_means
        initial_lengths = np.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
        )
    else:
        # Build Delaunay edges from the initial cluster means
        vertices, raw_edges, _ = create_delaunay_edges(cluster_means)
        edges, initial_lengths = remove_outlier_edges(
            vertices, raw_edges, config_manager.config
        )

    num_frames = len(dataset)
    num_clusters = len(cluster_means)

    # ------------------------------------------------------------------
    # Build adjacency list from edges (once, outside time loop)
    # ------------------------------------------------------------------
    neighbors: List[List[int]] = [[] for _ in range(num_clusters)]
    if enable_coherence and len(edges) > 0:
        for e in edges:
            a, b = int(e[0]), int(e[1])
            neighbors[a].append(b)
            neighbors[b].append(a)
        # Deduplicate
        for i in range(num_clusters):
            neighbors[i] = list(set(neighbors[i]))

    # Allocate output array: (T, K, 3)
    cluster_positions = np.zeros((num_frames, num_clusters, 3), dtype=np.float64)

    # Store t=0 positions
    cluster_positions[0] = cluster_means.copy()

    # Track previous frame's cluster means for continuity (as in stress_analysis)
    prev_cluster_means = cluster_means.copy()

    # Reference positions for displacement computation (fixed for "t0" mode)
    ref_positions_t0 = cluster_means.copy()

    if enable_coherence:
        print(
            f"\nSpatial coherence enabled: alpha={coherence_alpha}, "
            f"iters={coherence_iters}, ref={coherence_ref}, "
            f"robust={coherence_use_robust}"
        )

    print("\nCalculating cluster trajectories over time...")
    total_frames = num_frames - 1

    for j in range(1, num_frames):
        # Progress bar (same style as other analyses)
        progress = int((j / total_frames) * 50)
        bar = "█" * progress + "░" * (50 - progress)
        print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)

        frame = dataset[j].astype(np.float64)
        new_points = frame[:, :3]  # XYZ only

        # CRITICAL: maintain same cluster index structure across frames
        new_cluster_means = np.zeros_like(cluster_means)

        # Track which clusters were empty (for coherence: leave them unchanged)
        empty_mask = np.zeros(num_clusters, dtype=bool)

        for k in range(num_clusters):
            cluster_mask = labels == k
            if np.any(cluster_mask):
                new_cluster_means[k] = np.mean(new_points[cluster_mask], axis=0)
            else:
                # If a cluster has no points in this frame, keep previous position
                new_cluster_means[k] = prev_cluster_means[k]
                empty_mask[k] = True

        # --------------------------------------------------------------
        # Graph Laplacian smoothing on per-frame displacements to reduce
        # tearing between adjacent clusters while preserving overall
        # displacement.
        # --------------------------------------------------------------
        if enable_coherence and len(edges) > 0:
            # (a) Choose reference positions
            if coherence_ref == "prev":
                ref = prev_cluster_means
            else:  # "t0" (default)
                ref = ref_positions_t0

            # (b) Displacement field relative to reference
            disp = new_cluster_means - ref  # (K, 3)

            # Remember displacement for empty clusters so we can restore it
            disp_empty_backup = disp.copy()

            # (c) Laplacian smoothing iterations with global robust scale
            eps = 1e-12

            # Compute global robust scale from all edge residuals
            if coherence_use_robust:
                edge_res = np.linalg.norm(
                    disp[edges[:, 0]] - disp[edges[:, 1]], axis=1
                )  # (E,)
                med = np.median(edge_res)
                global_scale = np.median(np.abs(edge_res - med)) + eps
            for _it in range(coherence_iters):
                disp_new = disp.copy()
                for i in range(num_clusters):
                    nbrs = neighbors[i]
                    if not nbrs or empty_mask[i]:
                        # Skip empty clusters or nodes with no neighbors
                        continue

                    if coherence_use_robust:
                        # Compute residuals ||disp[i] - disp[j]|| for each neighbor
                        nbr_disps = disp[nbrs]  # (N_nbrs, 3)
                        residuals = np.linalg.norm(
                            disp[i] - nbr_disps, axis=1
                        )  # (N_nbrs,)

                        # Huber-like weights using global MAD scale
                        threshold = coherence_huber_k * global_scale
                        weights = np.where(
                            residuals <= threshold,
                            1.0,
                            threshold / (residuals + eps),
                        )  # (N_nbrs,)

                        w_sum = np.sum(weights) + eps
                        neighbor_avg = np.sum(
                            weights[:, None] * nbr_disps, axis=0
                        ) / w_sum
                    else:
                        # Simple uniform average of neighbor displacements
                        neighbor_avg = np.mean(disp[nbrs], axis=0)

                    disp_new[i] = (
                        (1.0 - coherence_alpha) * disp[i]
                        + coherence_alpha * neighbor_avg
                    )

                disp = disp_new

            # Restore empty-cluster displacements (no smoothing applied)
            disp[empty_mask] = disp_empty_backup[empty_mask]

            # (d) Reconstruct coherent positions
            new_cluster_means = ref + disp

        prev_cluster_means = new_cluster_means.copy()
        cluster_positions[j] = new_cluster_means

    print()  # newline after progress bar

    return {
        "cluster_positions": cluster_positions,
        "edges": edges,
        "initial_lengths": initial_lengths,
    }


def run_cluster_mesh_export(config_manager):
    """
    Export cluster positions and mesh connections as a function of time.

    This:
      - Loads the experiment dataset
      - Applies color-then-spatial clustering on frame 0
      - Loads the saved clustering state (means + edges) from the mesh editor
      - Remaps labels to the saved cluster means
      - Computes cluster centroids for every frame using fixed labels
      - Saves a single NumPy file containing:
          * cluster_positions: (T, K, 3)
          * edges: (E, 2)
          * initial_cluster_means: (K, 3)
          * saved_final_means: (K, 3)
          * initial_lengths: (E,)
          * times: (T,)
          * experiment_name, output_dir
    """
    config = config_manager.config

    # Load basic experiment info
    # Use the name directly from config to avoid truncating names with dots (like rgd0.1)
    experiment_name = config["experiment"]["name"]
    output_dir = config["experiment"]["output_dir"]
    data_path = config["experiment"]["data_path"]

    # Check if export already exists and we should skip it
    if SKIP_EXISTING_EXPORTS:
        export_dir = os.path.join(output_dir, "cluster_mesh_export")
        export_filename = f"cluster_export_{experiment_name}.npy"
        export_path = os.path.join(export_dir, export_filename)
        if os.path.exists(export_path):
            print(f"\n>>> SKIPPING: Export already exists: {export_path}")
            return

    print(f"\n=== CLUSTER MESH EXPORT: {experiment_name} ===")

    # Validate data path
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return

    # Check for existing clustering state (required to keep mesh/edges)
    state_path = find_clustering_state_for_experiment(experiment_name, output_dir)

    if state_path is None:
        print("\n" + "=" * 60)
        print("⚠️  NO CLUSTERING STATE FOUND!")
        print("=" * 60)
        print(
            f"This export requires a clustering state file for experiment: {experiment_name}"
        )
        print(f"Expected location: {os.path.join(output_dir, f'clustering_state_{experiment_name}.json')}")
        print("Please run the mesh editor first to create a clustering state:")
        print()
        print("  python scripts/run_mesh_editor.py")
        print()
        print("Then run this export again.")
        print("=" * 60)
        return

    print(f"\nFound clustering state: {state_path}")

    # Load dataset
    dataset = np.load(data_path)
    print(f"Loaded data shape: {dataset.shape}")

    # Validate and extract frame 0
    if len(dataset.shape) == 3:
        # Multi-time frame data
        frame0 = dataset[0]
    elif len(dataset.shape) == 2:
        # Single frame - still exportable, but trajectories will have T=1
        frame0 = dataset
    else:
        print("Error: Dataset must be 2D (points, 6) or 3D (time_frames, points, 6)")
        return

    print(f"Frame 0 data shape: {frame0.shape}")

    if frame0.shape[1] < 3:
        print(f"Error: Expected at least 3 features (X, Y, Z), got {frame0.shape[1]}")
        return

    # Extract XYZ and RGB (if present)
    xyz0 = frame0[:, :3]
    rgb0 = None
    if frame0.shape[1] >= 6:
        rgb0 = np.clip(frame0[:, 3:6], 0.0, 1.0)

    # Apply color-then-spatial clustering on frame 0
    print("\nApplying color-then-spatial clustering on frame 0...")
    if rgb0 is not None:
        labels, _ = color_then_spatial_clustering(xyz0, rgb0, config_manager.config)
    else:
        # Fallback: spatial-only clustering if no RGB
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(
            eps=config_manager.get("clustering.spatial_eps", 0.045),
            min_samples=config_manager.get("clustering.min_cluster_points", 2),
        ).fit(xyz0)
        labels = clustering.labels_

    # Calculate initial cluster means from labels
    cluster_means0, _ = calculate_cluster_means(xyz0, labels)
    print(f"Calculated {len(cluster_means0)} initial cluster means")

    # Load saved clustering state: final means + edges from mesh editor
    print("\nLoading clustering state...")
    (
        saved_final_means,
        saved_final_edges,
        saved_original_means,
        saved_original_edges,
    ) = load_saved_clustering_state(experiment_name, output_dir)

    if saved_final_means is not None:
        print(
            f"Applied filtering: {len(cluster_means0)} -> {len(saved_final_means)} cluster means"
        )

        # Remap labels to saved cluster means
        labels, _ = remap_labels_to_saved_means(
            xyz0, labels, cluster_means0, saved_final_means
        )

        # Use saved final means as our initial cluster positions
        initial_cluster_means = saved_final_means
        use_saved_edges = saved_final_edges is not None
        edges = saved_final_edges
    else:
        print("No saved state loaded, using original clustering and fresh edges.")
        initial_cluster_means = cluster_means0
        use_saved_edges = False
        edges = None

    if len(initial_cluster_means) == 0:
        print("Warning: No clusters found; nothing to export.")
        return

    # Compute trajectories over time
    traj_data = _compute_cluster_trajectories(
        labels=labels,
        initial_cluster_means=initial_cluster_means,
        dataset=dataset,
        use_saved_edges=use_saved_edges,
        saved_edges=edges,
        config_manager=config_manager,
    )

    cluster_positions = traj_data["cluster_positions"]
    edges_used = traj_data["edges"]
    initial_lengths = traj_data["initial_lengths"]

    # Build time axis
    total_time = config["experiment"].get("total_time")
    num_frames = cluster_positions.shape[0]

    if total_time is not None and total_time > 0:
        times = np.linspace(0.0, float(total_time), num_frames)
    else:
        # Fallback: integer frame indices
        times = np.arange(num_frames, dtype=np.float64)
        print(
            "Note: 'experiment.total_time' not set; using frame indices as time axis."
        )

    # Prepare export directory and file path
    export_dir = os.path.join(output_dir, "cluster_mesh_export")
    os.makedirs(export_dir, exist_ok=True)

    export_filename = f"cluster_export_{experiment_name}.npy"
    export_path = os.path.join(export_dir, export_filename)

    export_data = {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "times": times,  # (T,)
        "cluster_positions": cluster_positions,  # (T, K, 3)
        "edges": edges_used,  # (E, 2)
        "initial_cluster_means": initial_cluster_means,  # (K, 3)
        "saved_final_means": saved_final_means,  # (K, 3) or None
        "initial_lengths": initial_lengths,  # (E,)
    }

    np.save(export_path, export_data, allow_pickle=True)

    print("\nExport complete.")
    print(f"  File: {export_path}")
    print(f"  Frames: {cluster_positions.shape[0]}")
    print(f"  Clusters: {cluster_positions.shape[1]}")
    print(f"  Edges: {len(edges_used)}")


