"""
Roboflow Dataset Downloader
A simple utility to download datasets from Roboflow.
"""
import os
import argparse
from pathlib import Path
from roboflow import Roboflow

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class RoboflowDownloader:
    """Handles downloading datasets from Roboflow."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the downloader.
        
        Args:
            api_key: Roboflow API key. If None, reads from ROBOWFLOW_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Roboflow API key required. Set ROBOFLOW_API_KEY env var or pass api_key parameter."
            )
        self.rf = Roboflow(api_key=self.api_key)
    
    def download_dataset(
        self,
        workspace: str,
        project: str,
        version: int,
        format: str = "yolov8",
        location: str = "datasets"
    ):
        """
        Download a dataset from Roboflow.
        
        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version number
            format: Export format (default: yolov8)
            location: Local directory to save dataset (default: datasets)
        
        Returns:
            Path to downloaded dataset
        """
        print(f"Downloading {workspace}/{project} v{version}...")
        
        # Get project
        project_obj = self.rf.workspace(workspace).project(project)
        
        # Get dataset version
        dataset = project_obj.version(version)
        
        # Create output directory
        output_dir = Path(location) / f"{workspace}_{project}_v{version}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        dataset.download(format, location=str(output_dir))
        
        print(f"Dataset downloaded to: {output_dir}")
        return output_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download Roboflow datasets")
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Roboflow workspace name"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project name"
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Dataset version number"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="yolov8",
        help="Export format (default: yolov8)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="datasets",
        help="Output directory (default: datasets)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOWFLOW_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    try:
        downloader = RoboflowDownloader(api_key=args.api_key)
        downloader.download_dataset(
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            format=args.format,
            location=args.location
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

