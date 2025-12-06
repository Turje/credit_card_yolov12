"""
Configuration management for Roboflow datasets.
"""
import os
from pathlib import Path
from typing import Dict, Optional


class Config:
    """Manages configuration settings."""
    
    # Default paths
    DEFAULT_DATASET_DIR = "datasets"
    DEFAULT_MODEL_DIR = "models"
    DEFAULT_OUTPUT_DIR = "outputs"
    
    def __init__(self):
        """Initialize configuration."""
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        self.dataset_dir = Path(os.getenv("DATASET_DIR", self.DEFAULT_DATASET_DIR))
        self.model_dir = Path(os.getenv("MODEL_DIR", self.DEFAULT_MODEL_DIR))
        self.output_dir = Path(os.getenv("OUTPUT_DIR", self.DEFAULT_OUTPUT_DIR))
        
        # Create directories
        self.dataset_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def get_dataset_path(self, workspace: str, project: str, version: int) -> Path:
        """Get path for a specific dataset."""
        return self.dataset_dir / f"{workspace}_{project}_v{version}"
    
    def validate_api_key(self) -> bool:
        """Check if API key is set."""
        return self.api_key is not None


# Global config instance
config = Config()

