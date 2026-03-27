#!/usr/bin/env python3
"""
Script to generate displacement analysis graphs from saved data files.
This script reads the tracked_displacements_mm.npy and tracked_indices.npy files
that were created by the displacement analysis and generates publication-quality plots.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import after path setup
from utils.config_manager import ConfigManager  # type: ignore

def plot_displacement_over_time_publication(displacement_data, custom_title: str, output_dir: str, 
                                         total_experiment_time: float, experiment_name: str):
    """
    Plot displacement over time with publication-quality formatting and no error bars.
    
    Args:
        displacement_data: List of displacement lists for each tracked point
        custom_title: Custom title for the plot
        output_dir: Output directory for the experiment
        total_experiment_time: Total experiment time in seconds
        experiment_name: Name of the experiment for filename
    """
    if not displacement_data or not displacement_data[0]:
        print("No displacement data to plot")
        return
    
    # Convert to numpy array and handle NaN values
    disp_array = np.array(displacement_data)
    num_points, num_frames = disp_array.shape
    
    # Create time array in seconds
    time_points = np.linspace(0, total_experiment_time, num_frames)
    
    # Calculate mean across points for each frame (no std for error bars)
    mean_disp = np.nanmean(disp_array, axis=0)
    
    # Calculate and print the maximum average displacement
    max_avg_displacement = np.nanmax(mean_disp)
    print(f"Maximum average displacement: {max_avg_displacement:.4f} mm")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure with publication-quality dimensions and DPI
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Plot the displacement as a function of time (no error bars)
    ax.plot(time_points, mean_disp, marker='o', linestyle='-', color='#1f77b4', 
            linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    
    # Customize axes
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='normal')
    ax.set_ylabel('Displacement (mm)', fontsize=12, fontweight='normal')
    ax.set_title(custom_title, fontsize=14, fontweight='bold', pad=20)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=1, length=6)
    ax.tick_params(axis='both', which='minor', width=0.5, length=3)
    
    # Add minor grid lines for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    
    # Set axis limits with some padding
    ax.set_xlim(-total_experiment_time * 0.02, total_experiment_time * 1.02)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines slightly thicker
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create a clean filename from the custom title
    # Remove special characters and replace spaces with underscores
    clean_title = custom_title.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    clean_title = ''.join(c for c in clean_title if c.isalnum() or c == '_')
    
    # Save the plot as SVG and PNG
    plot_filename = f'displacement_{experiment_name}_{clean_title}.svg'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight', dpi=300)
    
    # Also save as PNG
    png_filename = f'displacement_{experiment_name}_{clean_title}.png'
    png_path = os.path.join(plots_dir, png_filename)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"Publication-quality displacement plot saved as:")
    print(f"  - SVG: {plot_path}")
    print(f"  - PNG: {png_path}")

def find_displacement_data_files(experiment_name: str, output_dir: str):
    """
    Find the displacement data files for the given experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for the experiment
        
    Returns:
        Tuple of (displacement_file_path, indices_file_path) or (None, None) if not found
    """
    # Look for files in the displacement_analysis subdirectory
    displacement_dir = os.path.join(output_dir, "displacement_analysis")
    
    # Check if the directory exists
    if not os.path.exists(displacement_dir):
        print(f"Displacement analysis directory not found: {displacement_dir}")
        return None, None
    
    # Look for the specific files
    displacement_file = os.path.join(displacement_dir, "tracked_displacements_mm.npy")
    indices_file = os.path.join(displacement_dir, "tracked_indices.npy")
    
    if not os.path.exists(displacement_file):
        print(f"Displacement data file not found: {displacement_file}")
        return None, None
    
    if not os.path.exists(indices_file):
        print(f"Tracked indices file not found: {indices_file}")
        return None, None
    
    return displacement_file, indices_file

def load_displacement_data(displacement_file: str, indices_file: str):
    """
    Load displacement data and indices from the saved files.
    
    Args:
        displacement_file: Path to the displacement data file
        indices_file: Path to the tracked indices file
        
    Returns:
        Tuple of (displacement_data, tracked_indices)
    """
    try:
        # Load displacement data
        displacement_data = np.load(displacement_file)
        print(f"Loaded displacement data shape: {displacement_data.shape}")
        
        # Load tracked indices
        tracked_indices = np.load(indices_file)
        print(f"Loaded tracked indices: {tracked_indices}")
        
        return displacement_data, tracked_indices
        
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None, None

def main():
    """
    Main function to generate displacement analysis graphs from saved data.
    """
    print("="*60)
    print("📊 DISPLACEMENT ANALYSIS GRAPH GENERATOR")
    print("="*60)
    print("This script generates publication-quality displacement graphs from previously saved data.")
    print("It reads the tracked_displacements_mm.npy and tracked_indices.npy files")
    print("that were created by running the displacement analysis.")
    print("="*60)
    
    # Load configuration
    try:
        config_manager = ConfigManager()
        config_manager.print_summary()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    config = config_manager.config
    
    # Validate total experiment time
    total_experiment_time = config['experiment'].get('total_time')
    if total_experiment_time is None or total_experiment_time <= 0:
        print("\n" + "="*60)
        print("⚠️  MISSING EXPERIMENT TIME!")
        print("="*60)
        print("The displacement plot requires the total experiment time to be specified.")
        print("Please add 'total_time' to your experiment configuration:")
        print()
        print("experiment:")
        print("  name: your_experiment_name")
        print("  total_time: 10.0  # Total experiment time in seconds")
        print()
        print("Then run this script again.")
        print("="*60)
        return
    
    # Get experiment info
    experiment_name = config['experiment']['name']
    output_dir = config['experiment']['output_dir']
    
    print(f"\nLooking for displacement data for experiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    
    # Find displacement data files
    displacement_file, indices_file = find_displacement_data_files(experiment_name, output_dir)
    
    if displacement_file is None or indices_file is None:
        print("\n" + "="*60)
        print("⚠️  DISPLACEMENT DATA FILES NOT FOUND!")
        print("="*60)
        print(f"This script requires displacement data files for experiment: {experiment_name}")
        print("Please run the displacement analysis first to create these files:")
        print()
        print("  python scripts/run_displacement_analysis.py")
        print()
        print("Then run this script again.")
        print("="*60)
        return
    
    print(f"\nFound displacement data files:")
    print(f"  - Displacement data: {displacement_file}")
    print(f"  - Tracked indices: {indices_file}")
    
    # Load the data
    displacement_data, tracked_indices = load_displacement_data(displacement_file, indices_file)
    
    if displacement_data is None:
        print("Failed to load displacement data. Exiting.")
        return
    
    # Convert displacement data to the format expected by our plotting function
    # The data should be a list of lists, where each inner list contains displacement values for one point
    displacement_lists = []
    for i in range(displacement_data.shape[0]):
        point_displacements = displacement_data[i].tolist()
        displacement_lists.append(point_displacements)
    
    print(f"\nPrepared displacement data:")
    print(f"  - Number of tracked points: {len(displacement_lists)}")
    print(f"  - Number of time frames: {len(displacement_lists[0]) if displacement_lists else 0}")
    print(f"  - Tracked cluster indices: {tracked_indices}")
    
    # Prompt user for custom title
    print("\n" + "="*50)
    print("📝 CUSTOM TITLE PROMPT")
    print("="*50)
    print("Enter a custom title for your displacement plot.")
    print("This title will appear at the top of the plot and in the filename.")
    print("Examples:")
    print("  - 'Displacement Over Time - Bifurcation Region'")
    print("  - 'Average Displacement vs. Time - Aneurysm Study'")
    print("  - 'Cluster Displacement Analysis - Branch Point'")
    print("="*50)
    
    custom_title = input("Enter custom title: ").strip()
    
    if not custom_title:
        # Provide a default title if user doesn't enter one
        custom_title = f"Displacement Over Time - {experiment_name} ({len(displacement_lists)} points)"
        print(f"Using default title: '{custom_title}'")
    else:
        print(f"Using custom title: '{custom_title}'")
    
    # Generate the displacement plot
    print("\nGenerating publication-quality displacement plot...")
    try:
        plot_displacement_over_time_publication(
            displacement_lists, 
            custom_title,
            output_dir, 
            total_experiment_time,
            experiment_name
        )
        print("\n✅ Publication-quality displacement plot generated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error generating plot: {e}")
        return
    
    print("\n" + "="*60)
    print("🎉 GRAPH GENERATION COMPLETE!")
    print("="*60)
    print(f"Publication-quality displacement plot has been saved to: {output_dir}/plots/")
    print("You can find both SVG and PNG versions of the plot.")
    print("Features:")
    print("  - No error bars for cleaner appearance")
    print("  - Custom title as specified")
    print("  - Publication-quality formatting")
    print("  - High-resolution output (300 DPI)")
    print("  - Professional grid and styling")
    print("="*60)

if __name__ == "__main__":
    main()