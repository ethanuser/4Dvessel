# Vessel Stress Analysis

A Python project for analyzing stress, force, and displacement in vessel structures using point cloud data with color information.

## Project Structure

```
vessel-stress-analysis/
├── config/
│   └── default_config.json          # Default configuration
├── data/
│   ├── raw/                         # Raw data files
│   └── processed/                   # Processed results
├── scripts/
│   ├── run_mesh_editor.py          # Interactive mesh editor
│   ├── run_stress_analysis.py      # Stress analysis
│   ├── run_force_analysis.py       # Force analysis
│   └── run_displacement_analysis.py # Displacement analysis
├── src/
│   ├── analysis/                   # Analysis modules
│   ├── core/                       # Core clustering logic
│   └── utils/                      # Utility functions
└── legacy/                         # Legacy code
```

## Dependencies

- numpy
- matplotlib
- pyvista
- scikit-learn
- scipy
- tkinter (for file dialogs)

## Workflow

### 1. Setup Experiment Configuration
Create or modify experiment configuration files in `config/` directory. Each experiment should have:
- Data file path
- Output directory
- Clustering parameters
- Material properties
- Visualization settings

### 2. Create Clustering State (Required First Step)
**This step is mandatory before running any analysis.**

```bash
python scripts/run_mesh_editor.py
```

The mesh editor will:
- ✅ **Automatically load** your experiment data based on configuration
- ✅ **Find existing clustering states** for your experiment
- ✅ **Load existing state** if found, or start fresh if none exists
- ✅ **Save clustering states** in the experiment's output directory

**Interactive Controls:**
- **Click near cluster means**: Select/deselect cluster means
- **Press 'X'**: Define plane based on camera position
- **Click on cyan edges**: Toggle edge selection
- **Press 'D'**: Delete selected points/edges
- **Press 'C'**: Clear selection
- **Press 'S'**: Save clustering state
- **Press SPACE**: Print current camera position (for config files)

### 3. Run Analysis Scripts
After creating a clustering state, run any analysis:

```bash
python scripts/run_stress_analysis.py
python scripts/run_force_analysis.py
python scripts/run_displacement_analysis.py
```

**Analysis scripts will:**
- ✅ **Automatically find** the clustering state for your experiment
- ✅ **Apply the filtering** from the mesh editor
- ✅ **Save results** in the experiment's output directory
- ✅ **Quit with clear error** if no clustering state exists

## Clustering State System

### Purpose
The clustering state system ensures consistent filtering across all analysis scripts. When you manually edit cluster means and edges in the mesh editor, these changes are saved and automatically applied to all subsequent analyses.

### File Format
Clustering states are saved as JSON files with the naming convention:
```
clustering_state_{experiment_name}.json
```

**Location:** `{experiment_output_dir}/clustering_state_{experiment_name}.json`

**Contents:**
- Original and filtered cluster means
- Original and filtered edges
- Kept point/edge indices
- Experiment metadata

### Automatic State Management
- **Mesh Editor**: Automatically finds and loads existing states
- **Analysis Scripts**: Automatically find and apply states based on experiment configuration
- **No Manual Selection**: No more file dialogs - everything is automatic

## Configuration

### Experiment Configuration
Each experiment has its own configuration file with:

```json
{
  "experiment": {
    "name": "experiment_001",
    "data_path": "data/raw/experiment_001.npy",
    "output_dir": "data/processed/experiment_001",
    "total_time": 10.0
  },
  "clustering": {
    "n_color_clusters": 8,
    "spatial_eps": 0.1,
    "min_cluster_points": 5
  },
  "material_properties": {
    "young_modulus_silicone": 1.15e6
  },
  "visualization": {
    "stress_scale_min": -0.001e9,
    "stress_scale_max": 0.001e9
  },
  "camera": {
    "position": [
      [2.587586159574164, -10.50850103215826, -15.083054681459819],
      [0.0, 0.0, 0.0],
      [-0.9873876090047237, -0.017241296927933046, -0.1573799455590763]
    ]
  }
}
```

**Required Parameters:**
- `total_time`: Total experiment duration in seconds (required for stress analysis plots)
- `name`: Unique experiment identifier
- `data_path`: Path to the NumPy data file
- `output_dir`: Directory for saving results

### Default Configuration
The `config/default_config.json` file contains default values that can be overridden by experiment-specific configurations.

## Troubleshooting

### "NO CLUSTERING STATE FOUND!" Error
If you see this error when running analysis scripts:

```
⚠️  NO CLUSTERING STATE FOUND!
This analysis requires a clustering state file for experiment: experiment_001
Expected location: data/processed/experiment_001/clustering_state_experiment_001.json
Please run the mesh editor first to create a clustering state:
  python scripts/run_mesh_editor.py
```

**Solution:**
1. Run the mesh editor: `python scripts/run_mesh_editor.py`
2. Select your experiment configuration
3. Edit the clustering as needed
4. Press 'S' to save the clustering state
5. Run your analysis script again

### Data Format Requirements
Your data files must be NumPy arrays with shape:
- **Single frame**: `(N, 6)` where N = number of points, 6 = [X, Y, Z, R, G, B]
- **Multi-frame**: `(T, N, 6)` where T = number of time frames

### File Paths
- Ensure all paths in your experiment configuration are correct
- Use absolute paths or paths relative to the project root
- The system will create output directories automatically

## Advanced Usage

### Custom Stress Scale
During stress analysis, you can customize the stress visualization scale:
- The system will prompt for custom min/max values
- Values are in Pascals (Pa)
- Default range is typically ±0.001 GPa

### Multiple Experiments
To work with multiple experiments:
1. Create separate configuration files for each experiment
2. Each experiment will have its own clustering state
3. Analysis scripts will automatically use the correct state for each experiment

### Clustering State Versioning
- Use 'S' key to save the current state (overwrites existing)
- Use 'N' key to save as a new versioned file
- Versioned files include timestamps in the filename
