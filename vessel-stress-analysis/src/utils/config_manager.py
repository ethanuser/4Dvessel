import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import tkinter as tk
from tkinter import filedialog

class ConfigManager:
    """
    Manages experiment configurations and file paths.
    Provides centralized access to all experiment parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, will prompt user to select.
        """
        self.config_path = config_path or self._select_config_file()
        self.config = self._load_config()
        self._setup_paths()
    
    def _select_config_file(self) -> str:
        """Prompt user to select a configuration file."""
        root = tk.Tk()
        root.withdraw()
        config_file = filedialog.askopenfilename(
            title="Select experiment configuration file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="config/experiments"
        )
        root.destroy()
        
        if not config_file:
            # Use default config
            config_file = "config/default_config.json"
            print(f"No config file selected, using default: {config_file}")
        
        return config_file
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config file: {e}")
            raise
    
    def _setup_paths(self):
        """Setup and validate file paths."""
        experiment_name = self.config['experiment']['name']
        output_dir = self.config['experiment']['output_dir']
        
        # Create output directories if they don't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/plots").mkdir(exist_ok=True)
        Path(f"{output_dir}/stress_analysis").mkdir(exist_ok=True)
        Path(f"{output_dir}/force_analysis").mkdir(exist_ok=True)
        Path(f"{output_dir}/displacement_analysis").mkdir(exist_ok=True)
        
        # Update output paths in config
        self.config['experiment']['output_dir'] = output_dir
        self.config['experiment']['plots_dir'] = f"{output_dir}/plots"
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Path to the configuration value (e.g., 'clustering.n_color_clusters')
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Path to the configuration value (e.g., 'camera.position')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_output_path(self, analysis_type: str, filename: str) -> str:
        """
        Get the full output path for a file.
        
        Args:
            analysis_type: Type of analysis (stress, force, displacement)
            filename: Name of the file
            
        Returns:
            Full path to the output file
        """
        output_dir = self.config['experiment']['output_dir']
        return os.path.join(output_dir, f"{analysis_type}_analysis", filename)
    
    def get_plot_path(self, plot_name: str) -> str:
        """
        Get the full path for a plot file.
        
        Args:
            plot_name: Name of the plot file
            
        Returns:
            Full path to the plot file
        """
        plots_dir = self.config['experiment']['plots_dir']
        return os.path.join(plots_dir, plot_name)
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save the current configuration to a file.
        
        Args:
            output_path: Path to save the config. If None, saves to original location.
        """
        save_path = output_path or self.config_path
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {save_path}")
    
    def create_experiment_config(self, experiment_name: str, data_path: str, 
                                output_dir: Optional[str] = None) -> str:
        """
        Create a new experiment configuration file.
        
        Args:
            experiment_name: Name of the experiment
            data_path: Path to the data file
            output_dir: Output directory. If None, uses default pattern.
            
        Returns:
            Path to the created configuration file
        """
        if output_dir is None:
            output_dir = f"data/processed/{experiment_name}"
        
        # Create new config based on default
        new_config = self.config.copy()
        new_config['experiment']['name'] = experiment_name
        new_config['experiment']['data_path'] = data_path
        new_config['experiment']['output_dir'] = output_dir
        
        # Create config file path
        config_dir = Path("config/experiments")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"{experiment_name}.json"
        
        # Save new config
        with open(config_file, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        print(f"Created experiment configuration: {config_file}")
        return str(config_file)
    
    def print_summary(self):
        """Print a summary of the current configuration."""
        print("\n=== EXPERIMENT CONFIGURATION SUMMARY ===")
        print(f"Experiment: {self.config['experiment']['name']}")
        print(f"Data file: {self.config['experiment']['data_path']}")
        print(f"Output directory: {self.config['experiment']['output_dir']}")
        print(f"Clustering parameters:")
        print(f"  - Color clusters: {self.config['clustering']['n_color_clusters']}")
        print(f"  - Spatial epsilon: {self.config['clustering']['spatial_eps']}")
        print(f"  - Min cluster points: {self.config['clustering']['min_cluster_points']}")
        print(f"Analysis enabled:")
        print(f"  - Stress: {self.config['analysis']['stress']['enabled']}")
        print(f"  - Force: {self.config['analysis']['force']['enabled']}")
        print(f"  - Displacement: {self.config['analysis']['displacement']['enabled']}")
        print("==========================================\n") 