#!/usr/bin/env python3
"""
Script to generate displacement analysis graphs from multiple saved data files.
This script reads multiple tracked_displacements_mm.npy and tracked_indices.npy files
that were created by the displacement analysis and generates publication-quality plots
with multiple lines on the same graph.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import after path setup
from utils.config_manager import ConfigManager  # type: ignore

# ── CONFIGURABLE CONSTANTS ──────────────────────────────────────────────────
PLOT_TITLE = "Displacement Analysis of Experiment a14"
LEGEND_LABELS = [
    "Bifurcation Region",
    "Proximal Region",
    # "Distal Region",
]
COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f'   # Gray
]

def extract_region_name_from_filename(filename):
    """
    Extract a meaningful region name from a filename.
    
    Args:
        filename: The filename to extract region name from
        
    Returns:
        A readable region name
    """
    # Remove file extension and common prefixes
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Remove experiment number prefix if present
    if base_name.startswith(('4_', '5_', '1_')):
        base_name = base_name[2:]
    
    # Convert underscores to spaces and capitalize
    region_name = base_name.replace('_', ' ').title()
    
    # Handle common abbreviations
    region_name = region_name.replace('Proximal', 'Proximal Region')
    region_name = region_name.replace('Distal', 'Distal Region')
    region_name = region_name.replace('Bifurcation', 'Bifurcation Region')
    
    return region_name

def plot_multiple_displacements_publication(displacement_data_list, custom_title: str, output_dir: str, 
                                         total_experiment_time: float, experiment_name: str, 
                                         region_names=None):
    """
    Plot multiple displacement datasets over time with publication-quality formatting.
    
    Args:
        displacement_data_list: List of displacement data arrays
        custom_title: Custom title for the plot
        output_dir: Output directory for the experiment
        total_experiment_time: Total experiment time in seconds
        experiment_name: Name of the experiment for filename
        region_names: List of region names for legend labels
    """
    if not displacement_data_list:
        print("No displacement data to plot")
        return
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure with publication-quality dimensions and DPI
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Create time array in seconds (use the first dataset to determine frame count)
    first_dataset = displacement_data_list[0]
    # Handle both numpy arrays and lists
    if hasattr(first_dataset, 'shape'):
        num_frames = first_dataset.shape[1] if len(first_dataset.shape) > 1 else len(first_dataset)
    else:
        num_frames = len(first_dataset[0]) if first_dataset and len(first_dataset) > 0 else 0
    
    time_points = np.linspace(0, total_experiment_time, num_frames)
    
    # Plot each dataset
    max_displacement = 0
    for i, displacement_data in enumerate(displacement_data_list):
        if displacement_data is None or len(displacement_data) == 0:
            continue
            
        # Handle both numpy arrays and lists
        if hasattr(displacement_data, 'shape'):
            # Numpy array
            disp_array = displacement_data
            if len(disp_array.shape) == 1:
                # Single point data
                mean_disp = disp_array
            else:
                # Multiple points data - calculate mean across points
                mean_disp = np.nanmean(disp_array, axis=0)
        else:
            # List format - convert to numpy array first
            disp_array = np.array(displacement_data)
            if len(disp_array.shape) == 1:
                # Single point data
                mean_disp = disp_array
            else:
                # Multiple points data - calculate mean across points
                mean_disp = np.nanmean(disp_array, axis=0)
        
        # Calculate and print the maximum displacement for this dataset
        max_avg_displacement = np.nanmax(mean_disp)
        print(f"Dataset {i+1} - Maximum average displacement: {max_avg_displacement:.4f} mm")
        max_displacement = max(max_displacement, max_avg_displacement)
        
        # Get color and label for this line
        color = COLORS[i % len(COLORS)]
        if region_names and i < len(region_names):
            label = region_names[i]
        else:
            label = LEGEND_LABELS[i % len(LEGEND_LABELS)]
        
        # Plot the displacement as a function of time
        ax.plot(time_points, mean_disp, marker='o', linestyle='-', color=color, 
                linewidth=2, markersize=5, markerfacecolor='white', markeredgewidth=1.5,
                label=label)
    
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
    ax.set_ylim(-max_displacement * 0.05, max_displacement * 1.05)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines slightly thicker
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create a clean filename from the custom title
    # Remove special characters and replace spaces with underscores
    clean_title = custom_title.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    clean_title = ''.join(c for c in clean_title if c.isalnum() or c == '_')
    
    # Save the plot as SVG and PNG
    plot_filename = f'multiple_displacement_{experiment_name}_{clean_title}.svg'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight', dpi=300)
    
    # Also save as PNG
    png_filename = f'multiple_displacement_{experiment_name}_{clean_title}.png'
    png_path = os.path.join(plots_dir, png_filename)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"Publication-quality multiple displacement plot saved as:")
    print(f"  - SVG: {plot_path}")
    print(f"  - PNG: {png_path}")

def find_displacement_data_files(experiment_name: str, output_dir: str):
    """
    Find all displacement data files for the given experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for the experiment
        
    Returns:
        List of tuples (displacement_file_path, indices_file_path) or empty list if none found
    """
    # Look for files in the displacement_analysis subdirectory
    displacement_dir = os.path.join(output_dir, "displacement_analysis")
    
    # Check if the directory exists
    if not os.path.exists(displacement_dir):
        print(f"Displacement analysis directory not found: {displacement_dir}")
        return []
    
    # Find all .npy files that look like displacement data
    displacement_files = []
    indices_files = []
    
    # Look for displacement data files (any .npy file that's not an indices file)
    for file_path in glob.glob(os.path.join(displacement_dir, "*.npy")):
        filename = os.path.basename(file_path)
        if not filename.endswith('_indices.npy') and not filename.endswith('indices.npy'):
            displacement_files.append(file_path)
    
    # Look for corresponding indices files
    for disp_file in displacement_files:
        # Try to find corresponding indices file
        base_name = os.path.splitext(disp_file)[0]
        
        # Check for various naming patterns
        possible_indices_files = [
            f"{base_name}_indices.npy",
            f"{base_name.replace('_displacement', '')}_indices.npy",
            f"{base_name.replace('_displacement', '')}indices.npy"
        ]
        
        indices_file = None
        for possible_file in possible_indices_files:
            if os.path.exists(possible_file):
                indices_file = possible_file
                break
        
        if indices_file:
            indices_files.append(indices_file)
        else:
            print(f"Warning: No indices file found for {disp_file}")
            indices_files.append(None)
    
    # Return pairs of files
    file_pairs = []
    for disp_file, indices_file in zip(displacement_files, indices_files):
        if indices_file:
            file_pairs.append((disp_file, indices_file))
    
    return file_pairs

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
    Main function to generate displacement analysis graphs from multiple saved data files.
    """
    print("="*60)
    print("📊 MULTIPLE DISPLACEMENT ANALYSIS GRAPH GENERATOR")
    print("="*60)
    print("This script generates publication-quality displacement graphs from multiple data files.")
    print("It reads multiple .npy files and plots them on separate lines.")
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
    
    print(f"\nLooking for displacement data files for experiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    
    # Find all displacement data files
    file_pairs = find_displacement_data_files(experiment_name, output_dir)
    
    if not file_pairs:
        print("\n" + "="*60)
        print("⚠️  NO DISPLACEMENT DATA FILES FOUND!")
        print("="*60)
        print(f"This script requires displacement data files for experiment: {experiment_name}")
        print("Please run the displacement analysis first to create these files:")
        print()
        print("  python scripts/run_displacement_analysis.py")
        print()
        print("Then run this script again.")
        print("="*60)
        return
    
    print(f"\nFound {len(file_pairs)} displacement data file pairs:")
    region_names = []
    for i, (disp_file, indices_file) in enumerate(file_pairs):
        print(f"  {i+1}. Displacement: {os.path.basename(disp_file)}")
        print(f"     Indices: {os.path.basename(indices_file)}")
        # Extract region name from filename
        region_name = extract_region_name_from_filename(disp_file)
        region_names.append(region_name)
        print(f"     Region: {region_name}")
    
    # Load all the data
    displacement_data_list = []
    for i, (displacement_file, indices_file) in enumerate(file_pairs):
        print(f"\nLoading dataset {i+1}...")
        displacement_data, tracked_indices = load_displacement_data(displacement_file, indices_file)
        
        if displacement_data is not None:
            displacement_data_list.append(displacement_data)
            print(f"Dataset {i+1} loaded successfully")
        else:
            print(f"Failed to load dataset {i+1}, skipping...")
    
    if not displacement_data_list:
        print("No displacement data could be loaded. Exiting.")
        return
    
    print(f"\nSuccessfully loaded {len(displacement_data_list)} datasets")
    
    # Convert displacement data to the format expected by our plotting function
    # The data should be a list of lists, where each inner list contains displacement values for one point
    processed_data_list = []
    for displacement_data in displacement_data_list:
        if len(displacement_data.shape) == 1:
            # Single point data - convert to list format
            processed_data_list.append([displacement_data.tolist()])
        else:
            # Multiple points data - convert to list of lists
            displacement_lists = []
            for i in range(displacement_data.shape[0]):
                point_displacements = displacement_data[i].tolist()
                displacement_lists.append(point_displacements)
            processed_data_list.append(displacement_lists)
    
    print(f"\nPrepared {len(processed_data_list)} datasets for plotting")
    
    # Use the configured title from constants
    custom_title = PLOT_TITLE
    print(f"Using plot title: '{custom_title}'")
    
    # Generate the multiple displacement plot
    print("\nGenerating publication-quality multiple displacement plot...")
    try:
        plot_multiple_displacements_publication(
            processed_data_list, 
            custom_title,
            output_dir, 
            total_experiment_time,
            experiment_name,
            region_names
        )
        print("\n✅ Publication-quality multiple displacement plot generated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error generating plot: {e}")
        return
    
    print("\n" + "="*60)
    print("🎉 GRAPH GENERATION COMPLETE!")
    print("="*60)
    print(f"Publication-quality multiple displacement plot has been saved to: {output_dir}/plots/")
    print("You can find both SVG and PNG versions of the plot.")
    print("Features:")
    print(f"  - {len(processed_data_list)} datasets plotted on separate lines")
    print("  - No error bars for cleaner appearance")
    print("  - Custom title and legend labels from constants")
    print("  - Publication-quality formatting")
    print("  - High-resolution output (300 DPI)")
    print("  - Professional grid and styling")
    print("  - Automatic color assignment and legend")
    print("="*60)

if __name__ == "__main__":
    main()