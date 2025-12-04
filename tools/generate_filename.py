"""
Utility functions for file paths and filename generation.
"""

import subprocess
from datetime import datetime
from pathlib import Path

# Project root directory (cached for performance)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """
    Return the root path of the 'building-age-prediction-rome' project.
    
    This method works regardless of the current working directory and
    across different machines, as it determines the path relative to
    this file's location.
    
    Returns:
        Path: The absolute path to the project directory.
    """
    return _PROJECT_ROOT


def get_git_hash() -> str:
    """
    Get the short Git commit hash of the current repository.
    
    Returns:
        str: The short Git hash or 'no-git' if Git is not available.
    """
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def generate_filename(prefix: str, include_timestamp: bool = True, include_git_hash: bool = True) -> str:
    """
    Generate a unique filename with optional timestamp and Git hash.
    
    Args:
        prefix: Prefix for the filename.
        include_timestamp: Whether to include a timestamp (default: True).
        include_git_hash: Whether to include the Git hash (default: True).
    
    Returns:
        str: The generated filename in format '{prefix}_{timestamp}_{git_hash}'.
    """
    parts = [prefix]
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
    
    if include_git_hash:
        parts.append(get_git_hash())
    
    return "_".join(parts)
