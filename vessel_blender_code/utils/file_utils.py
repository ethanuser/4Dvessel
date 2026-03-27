"""
File utility functions for path resolution and file loading.
"""

import os
from pathlib import Path
from typing import Union


def resolve_path(file_path: str, project_root: Path) -> Path:
    """
    Resolve a relative or absolute path against the project root and CWD.
    
    Args:
        file_path: Path string (relative or absolute)
        project_root: Project root directory as Path object
        
    Returns:
        Resolved Path object
    """
    if not os.path.isabs(file_path):
        # Try relative to project root first
        resolved = project_root / file_path
        if not resolved.exists():
            # Fallback: relative to current working directory
            resolved = Path(file_path).resolve()
    else:
        resolved = Path(file_path)
    return resolved


def get_project_root(script_file: str = __file__) -> Path:
    """
    Get the project root directory (parent of utils/ folder).
    
    Args:
        script_file: Path to the current script file
        
    Returns:
        Project root Path object
    """
    script_path = Path(script_file).resolve()
    
    # Traverse up the directory tree to find the project root (where /utils/ resides)
    curr = script_path.parent
    while curr.parent != curr:  # Stop at root
        if (curr / "utils").is_dir():
            return curr
        curr = curr.parent
        
    # Fallback to script parent if not found
    return script_path.parent

