#!/usr/bin/env python3
"""
Interactive stress visualization from a single exported NumPy file.

This script combines the runner and implementation into a single file.
It reads the NumPy file exported by `cluster_mesh_export.py` and creates
an interactive 3D stress visualization without performing any clustering
or loading clustering state.

Primary usage (recommended):
    - Edit the NPY_FILE constant below to point to the desired export file.

Optional CLI usage:
    python single_file_stress_interactive.py <npy_path> [young_modulus]

Arguments:
    npy_path: Path to the .npy file produced by cluster_mesh_export.py
    young_modulus: Optional Young's modulus in Pa (default: 0.0255e9)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pyvista as pv

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.interactive_utils import (
    create_interactive_visualization as setup_interactive_visualization,
    create_default_text_generator,
    create_default_instructions,
)
import vessel_utils as vu

# ============================================================================
# NPY FILE CONFIGURATION
# ============================================================================
# Specify the exported NumPy file to visualize.
# This should be the file produced by cluster_mesh_export.py.
# Example:
#   NPY_FILE = r"cluster_data\cluster_mesh_timeseries_synthetic_translate_params.npy"
# ============================================================================

NPY_FILE = r"cluster_data/cluster_mesh_timeseries_synthetic_translate_real_params.npy"
# NPY_FILE = r"cluster_data/cluster_mesh_timeseries_lattice_strength_1_25.npy"

# ============================================================================
# PHYSICAL / VISUALIZATION CONSTANTS
# ============================================================================
# Default Young's modulus in Pa (matches stress analysis defaults)
YOUNG_MODULUS_DEFAULT = vu.YOUNG_MODULUS_SILICON

# Fixed stress scale (Pa) for visualization
# You can adjust these to zoom in/out on the stress range of interest.
STRESS_MIN_DEFAULT = -vu.MAX_STRESS_PA
STRESS_MAX_DEFAULT = vu.MAX_STRESS_PA

# Visualization styling (mirrors stress_interactive.py)
CLUSTER_POINT_SIZE = 1
CLUSTER_POINT_COLOR = "red"
EDGE_LINE_WIDTH = 5
EDGE_COLOR = "blue"
EDGE_OPACITY = 0.7


def _resolve_npy_path(npy_path: str, project_root: Path) -> Path:
    """
    Resolve a relative or absolute npy path against the project root and CWD.
    """
    if not os.path.isabs(npy_path):
        # Try relative to project root first
        resolved = project_root / npy_path
        if not resolved.exists():
            # Fallback: relative to current working directory
            resolved = Path(npy_path).resolve()
    else:
        resolved = Path(npy_path)
    return resolved


def _compute_stress_timeseries(
    cluster_positions: np.ndarray,
    edges: np.ndarray,
    initial_lengths: np.ndarray,
    young_modulus: float,
) -> Dict[str, Any]:
    """
    Given cluster positions over time and mesh edges, compute stress on each edge.

    Args:
        cluster_positions: Array of shape (T, K, 3)
        edges: Array of shape (E, 2) with edge indices
        initial_lengths: Array of shape (E,) with edge lengths at t=0
        young_modulus: Young's modulus in Pa

    Returns:
        Dict with:
            - stresses: list of length T, each an array (E,)
            - stress_min: float, global min (after clipping compression to 0)
            - stress_max: float, global max
    """
    num_frames = cluster_positions.shape[0]
    stresses = []

    # We use fixed visualization bounds defined at the top of the file.
    stress_min = STRESS_MIN_DEFAULT
    stress_max = STRESS_MAX_DEFAULT

    for t in range(num_frames):
        positions = cluster_positions[t]  # (K, 3)
        stress = vu.compute_edge_stress(positions, edges, initial_lengths, young_modulus)
        stresses.append(stress)
        
    return {
        "stresses": stresses,
        "stress_min": stress_min,
        "stress_max": stress_max,
    }


# Removed local _generate_color_map, now using vu.get_colors


def run_single_file_stress_interactive(
    npy_path: str,
    young_modulus: float = YOUNG_MODULUS_DEFAULT,
) -> None:
    """
    Load a pre-exported cluster/mesh time-series NumPy file and run an
    interactive stress visualization similar to `stress_interactive.py`.

    Args:
        npy_path: Path to the `.npy` file produced by `cluster_mesh_export.py`
        young_modulus: Young's modulus in Pa (default matches stress analysis)
    """
    if not os.path.exists(npy_path):
        print(f"ERROR: Export file not found: {npy_path}")
        return

    print(f"\n=== SINGLE-FILE STRESS INTERACTIVE ===")
    print(f"Loading export: {npy_path}")

    data = np.load(npy_path, allow_pickle=True).item()

    cluster_positions = data.get("cluster_positions")  # (T, K, 3)
    edges = data.get("edges")  # (E, 2)
    initial_lengths = data.get("initial_lengths")  # (E,)
    times = data.get("times")  # (T,)
    experiment_name = data.get("experiment_name", "UnknownExperiment")

    if cluster_positions is None or edges is None or initial_lengths is None:
        print("ERROR: Export file is missing required keys.")
        print("Expected keys: 'cluster_positions', 'edges', 'initial_lengths'.")
        return

    if cluster_positions.ndim != 3 or cluster_positions.shape[2] != 3:
        print(
            f"ERROR: 'cluster_positions' must have shape (T, K, 3), got {cluster_positions.shape}"
        )
        return

    if edges.ndim != 2 or edges.shape[1] != 2:
        print(f"ERROR: 'edges' must have shape (E, 2), got {edges.shape}")
        return

    num_frames, num_clusters, _ = cluster_positions.shape
    num_edges = edges.shape[0]

    print(f"Frames: {num_frames}, Clusters: {num_clusters}, Edges: {num_edges}")

    # Compute stresses over time
    stress_info = _compute_stress_timeseries(
        cluster_positions=cluster_positions,
        edges=edges,
        initial_lengths=initial_lengths,
        young_modulus=young_modulus,
    )

    stresses = stress_info["stresses"]
    stress_min = stress_info["stress_min"]
    stress_max = stress_info["stress_max"]

    print(
        f"Stress range (after clipping compression): "
        f"{stress_min:.3e} Pa to {stress_max:.3e} Pa"
    )

    # Define the colormap
    cm = vu.get_colormap()

    # Pre-create PolyData objects for each frame
    frame_mesh_data = {}

    # Precompute global bounds across all frames
    all_positions = cluster_positions.reshape(-1, 3)
    bounds = [
        float(all_positions[:, 0].min()),
        float(all_positions[:, 0].max()),
        float(all_positions[:, 1].min()),
        float(all_positions[:, 1].max()),
        float(all_positions[:, 2].min()),
        float(all_positions[:, 2].max()),
    ]

    # Prepare static line connectivity array for PyVista
    if num_edges > 0:
        lines = np.hstack([np.full((num_edges, 1), 2), edges])
    else:
        lines = np.empty((0, 3), dtype=int)

    print("Precomputing mesh data for interactive visualization...")
    for t in range(num_frames):
        positions = cluster_positions[t]
        stress_t = stresses[t]

        # Create PolyData for cluster means
        if num_clusters > 0:
            means_poly = pv.PolyData(positions)
        else:
            means_poly = None

        # Create PolyData for edges
        if num_edges > 0 and num_clusters > 0:
            edges_poly = pv.PolyData(positions, lines=lines)
        else:
            edges_poly = None

        # Precompute stress colors
        if num_edges > 0:
            stress_colors = vu.get_colors(stress_t, cm, vmin=0, vmax=vu.MAX_STRESS_PA, use_abs=True)
        else:
            stress_colors = None

        frame_mesh_data[t] = {
            "means": means_poly,
            "edges": edges_poly,
            "stress_colors": stress_colors,
        }

    # Create frames_data structure for interactive utils
    # Convert frame indices to match expected format
    frame_indices = list(range(num_frames))
    frames_data = {}
    for frame_idx in frame_indices:
        time_val = (
            float(times[frame_idx])
            if times is not None and len(times) == num_frames
            else float(frame_idx)
        )
        frames_data[frame_idx] = {
            "time": time_val,
            "num_clusters": num_clusters,
            "num_edges": num_edges,
        }

    # Create PyVista plotter
    plotter = pv.Plotter(
        title=f"Single-File Stress Visualization - {experiment_name}"
    )

    # Store the current actors
    current_means_actor = None
    current_edges_actor = None

    def update_callback(frame_list_idx, frame_idx, plotter):
        """Update callback for frame changes - handles mesh updates."""
        nonlocal current_means_actor, current_edges_actor

        mesh_data = frame_mesh_data[frame_idx]

        # Store old actors for removal after adding new ones (double-buffering to avoid flash)
        old_means_actor = current_means_actor
        old_edges_actor = current_edges_actor

        # Add new cluster means FIRST (before removing old ones to avoid flash)
        if mesh_data["means"] is not None:
            current_means_actor = plotter.add_mesh(
                mesh_data["means"],
                render_points_as_spheres=True,
                point_size=CLUSTER_POINT_SIZE,
                color=CLUSTER_POINT_COLOR,
                name=f"cluster_means_{frame_idx}",
            )
        else:
            current_means_actor = None

        # Add new edges with stress colors FIRST (before removing old ones to avoid flash)
        if mesh_data["edges"] is not None and mesh_data["stress_colors"] is not None:
            current_edges_actor = plotter.add_mesh(
                mesh_data["edges"],
                scalars=mesh_data["stress_colors"],
                rgb=True,
                opacity=EDGE_OPACITY,
                line_width=EDGE_LINE_WIDTH,
                name=f"delaunay_edges_{frame_idx}",
            )
        elif mesh_data["edges"] is not None:
            current_edges_actor = plotter.add_mesh(
                mesh_data["edges"],
                color=EDGE_COLOR,
                line_width=EDGE_LINE_WIDTH,
                name=f"delaunay_edges_{frame_idx}",
            )
        else:
            current_edges_actor = None

        # Render after adding new actors to ensure they're visible before removing old ones
        plotter.render()

        # NOW remove old actors (after new ones are already added and rendered)
        if old_means_actor is not None:
            try:
                plotter.remove_actor(old_means_actor)
            except Exception:
                pass

        if old_edges_actor is not None:
            try:
                plotter.remove_actor(old_edges_actor)
            except Exception:
                pass

    def text_generator(frame_idx, frame_data):
        """Generate text for frame display."""
        # Calculate padding width based on max frame number
        max_frame = num_frames - 1
        frame_width = len(str(max_frame))
        frame_str = f"{frame_idx:0{frame_width}d}"
        max_frame_str = f"{max_frame:0{frame_width}d}"

        stress_t = stresses[frame_idx]
        avg_stress = float(np.mean(stress_t)) if len(stress_t) > 0 else 0.0
        max_stress = float(np.max(stress_t)) if len(stress_t) > 0 else 0.0

        return (
            f"Frame: {frame_str}/{max_frame_str} | "
            f"Time: {frame_data['time']:.4f} | "
            f"Clusters: {num_clusters} | "
            f"Edges: {num_edges} | "
            f"Avg Stress: {avg_stress/1e6:.2f} MPa | "
            f"Max Stress: {max_stress/1e6:.2f} MPa"
        )

    # Prepare instructions
    instructions = create_default_instructions(
        visualization_description="Red points: Cluster means",
        additional_info=[
            "Colored lines: Stress-coded mesh edges",
        ],
    )

    # Use the standardized interactive visualization
    setup_interactive_visualization(
        plotter=plotter,
        frame_indices=frame_indices,
        frames_data=frames_data,
        update_callback=update_callback,
        text_generator=text_generator,
        instructions=instructions,
        bounds=bounds,
        title=f"Single-File Stress Visualization - {experiment_name}",
        font='courier',
        slider_pointa=(0.1, 0.02),
        slider_pointb=(0.9, 0.02),
    )

    print("Use the slider or arrow keys (Left/Right) to navigate frames\n")
    plotter.show()


def main():
    """
    Main entry point that handles CLI arguments and path resolution.
    """
    # Determine project root (script directory)
    script_dir = Path(__file__).parent
    project_root = script_dir

    # Determine source of npy path:
    # 1) CLI argument (highest priority)
    # 2) NPY_FILE constant (recommended pattern)
    if len(sys.argv) >= 2:
        npy_path_str = sys.argv[1]
    else:
        npy_path_str = NPY_FILE

    if not npy_path_str:
        print("=" * 60)
        print("⚠️  No NPY file specified.")
        print("Edit NPY_FILE in single_file_stress_interactive.py")
        print("or pass the path as a command-line argument:")
        print("  python single_file_stress_interactive.py <npy_path> [young_modulus]")
        print("=" * 60)
        return

    resolved_path = _resolve_npy_path(npy_path_str, project_root)

    # Validate file exists
    if not resolved_path.exists():
        print(f"⚠️  ERROR: File not found: {npy_path_str}")
        print(f"   Resolved path: {resolved_path}")
        print("Please update NPY_FILE in the script or provide a correct path on the CLI.")
        return

    if resolved_path.suffix != ".npy":
        print(f"⚠️  WARNING: File does not have .npy extension: {resolved_path}")

    # Parse optional young_modulus argument
    young_modulus = YOUNG_MODULUS_DEFAULT
    if len(sys.argv) >= 3:
        try:
            young_modulus = float(sys.argv[2])
        except ValueError:
            print(f"⚠️  WARNING: Invalid young_modulus value '{sys.argv[2]}', using default: {YOUNG_MODULUS_DEFAULT}")
            young_modulus = YOUNG_MODULUS_DEFAULT

    try:
        # Run the interactive visualization
        run_single_file_stress_interactive(
            str(resolved_path), young_modulus=young_modulus
        )

        print(f"\n✓ Completed visualization for: {resolved_path.name}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  ERROR during visualization:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()