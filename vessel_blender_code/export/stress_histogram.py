#!/usr/bin/env python3
"""
Create a histogram of stresses measured by each edge based on clustering numpy data.

This script loads the cluster/mesh time-series NumPy file and computes stress
for all edges in the last frame only, then displays a histogram of the stress values.

Usage:
    python stress_histogram.py [npy_path] [young_modulus]

Arguments:
    npy_path: Path to the .npy file produced by cluster_mesh_export.py
    young_modulus: Optional Young's modulus in Pa (default: 0.0255e9)
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import vessel_utils as vu

# Default configuration
NPY_FILE = r"cluster_data/cluster_export_synthetic_translate.npy"
YOUNG_MODULUS_DEFAULT = 1.15e6


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


def compute_last_frame_stresses(
    cluster_positions: np.ndarray,
    edges: np.ndarray,
    initial_lengths: np.ndarray,
    young_modulus: float,
) -> np.ndarray:
    """
    Compute stress for all edges in the last frame only.

    Args:
        cluster_positions: Array of shape (T, K, 3)
        edges: Array of shape (E, 2) with edge indices
        initial_lengths: Array of shape (E,) with edge lengths at t=0
        young_modulus: Young's modulus in Pa

    Returns:
        Array of stress values (E,) for the last frame
    """
    num_frames = cluster_positions.shape[0]
    last_frame_idx = num_frames - 1
    
    # Get positions for the last frame
    positions = cluster_positions[last_frame_idx]  # (K, 3)

    # Use central Neo-Hookean calculation
    stress = vu.compute_edge_stress(positions, edges, initial_lengths, young_modulus)
    
    return stress


def main():
    """
    Main entry point for stress histogram generation.
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
        print("Edit NPY_FILE in stress_histogram.py")
        print("or pass the path as a command-line argument:")
        print("  python stress_histogram.py <npy_path> [young_modulus]")
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
            print(
                f"⚠️  WARNING: Invalid young_modulus value '{sys.argv[2]}', "
                f"using default: {YOUNG_MODULUS_DEFAULT}"
            )
            young_modulus = YOUNG_MODULUS_DEFAULT

    try:
        print(f"\n=== STRESS HISTOGRAM ===")
        print(f"Loading export: {resolved_path}")

        data = np.load(str(resolved_path), allow_pickle=True).item()

        cluster_positions = data.get("cluster_positions")  # (T, K, 3)
        edges = data.get("edges")  # (E, 2)
        initial_lengths = data.get("initial_lengths")  # (E,)
        experiment_name = data.get("experiment_name", "UnknownExperiment")

        if cluster_positions is None or edges is None or initial_lengths is None:
            print("ERROR: Export file is missing required keys.")
            print("Expected keys: 'cluster_positions', 'edges', 'initial_lengths'.")
            return

        if cluster_positions.ndim != 3 or cluster_positions.shape[2] != 3:
            print(
                f"ERROR: 'cluster_positions' must have shape (T, K, 3), "
                f"got {cluster_positions.shape}"
            )
            return

        if edges.ndim != 2 or edges.shape[1] != 2:
            print(f"ERROR: 'edges' must have shape (E, 2), got {edges.shape}")
            return

        num_frames, num_clusters, _ = cluster_positions.shape
        num_edges = edges.shape[0]
        last_frame_idx = num_frames - 1

        print(f"Total frames: {num_frames}, Clusters: {num_clusters}, Edges: {num_edges}")
        print(f"Analyzing last frame (frame {last_frame_idx})")
        print(f"Young's modulus: {young_modulus:.3e} Pa")

        # Get time for last frame if available
        times = data.get("times")
        last_frame_time = None
        if times is not None and len(times) > last_frame_idx:
            last_frame_time = float(times[last_frame_idx])

        # Compute stresses for last frame only
        print(f"Computing stresses for all edges in last frame...")
        all_stresses = compute_last_frame_stresses(
            cluster_positions=cluster_positions,
            edges=edges,
            initial_lengths=initial_lengths,
            young_modulus=young_modulus,
        )

        # Convert to MPa for readability
        all_stresses_mpa = all_stresses / 1e6

        # Statistics
        time_str = f" (Time: {last_frame_time:.4f})" if last_frame_time is not None else ""
        print(f"\nStress Statistics for Last Frame (Frame {last_frame_idx}){time_str}:")
        print(f"  Total edges: {len(all_stresses_mpa):,}")
        print(f"  Min stress: {np.min(all_stresses_mpa):.4f} MPa")
        print(f"  Max stress: {np.max(all_stresses_mpa):.4f} MPa")
        print(f"  Mean stress: {np.mean(all_stresses_mpa):.4f} MPa")
        print(f"  Median stress: {np.median(all_stresses_mpa):.4f} MPa")
        print(f"  Std dev: {np.std(all_stresses_mpa):.4f} MPa")

        # Create histogram
        plt.figure(figsize=(12, 8))

        # Main histogram
        plt.subplot(2, 1, 1)
        n, bins, patches = plt.hist(
            all_stresses_mpa,
            bins=100,
            edgecolor="black",
            alpha=0.7,
            color="steelblue",
        )
        time_str = f" (Time: {last_frame_time:.4f})" if last_frame_time is not None else ""
        plt.xlabel("Stress (MPa)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(
            f"Stress Histogram - {experiment_name}\n"
            f"Last Frame (Frame {last_frame_idx}/{num_frames - 1}){time_str} - {num_edges} edges",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.axvline(
            np.mean(all_stresses_mpa),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(all_stresses_mpa):.4f} MPa",
        )
        plt.axvline(
            np.median(all_stresses_mpa),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(all_stresses_mpa):.4f} MPa",
        )
        plt.legend()

        # Detailed histogram showing all values including zeros
        plt.subplot(2, 1, 2)
        # Show all values including zeros
        plt.hist(
            all_stresses_mpa,
            bins=100,
            edgecolor="black",
            alpha=0.7,
            color="coral",
        )
        plt.xlabel("Stress (MPa)", fontsize=12)
        plt.ylabel("Frequency (Log Scale)", fontsize=12)
        plt.title(
            f"Stress Histogram - Last Frame (All Values Including Zeros) - {len(all_stresses_mpa):,} edges",
            fontsize=12,
        )
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        
        # Add annotation about zero values
        zero_count = np.sum(all_stresses_mpa == 0)
        if zero_count > 0:
            zero_percent = 100 * zero_count / len(all_stresses_mpa)
            plt.text(
                0.02,
                0.98,
                f"Zero stress values: {zero_count:,} ({zero_percent:.1f}%)",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        # Save figure
        output_path = project_root / f"stress_histogram_last_frame_{experiment_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Histogram saved to: {output_path}")

        # Show plot
        plt.show()

        print(f"\n✓ Completed histogram generation for: {resolved_path.name}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  ERROR during histogram generation:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

