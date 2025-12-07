"""
Unified configuration loader for training and inference.
Supports YAML config files and CLI argument overrides.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration."""

    framework: str
    name: str
    checkpoint: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    path: str
    format: str
    train_split: str = "train"
    val_split: str = "val"


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int
    batch_size: int
    img_size: int
    output_dir: str
    device: str = "cuda"
    workers: int = 8
    patience: int = 50


@dataclass
class InferenceConfig:
    """Inference configuration."""

    video_path: Optional[str] = None
    image_path: Optional[str] = None
    output_path: str = "outputs"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda"
    save_video: bool = True
    show_video: bool = False


@dataclass
class Config:
    """Unified configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    inference: InferenceConfig


class ConfigLoader:
    """Loads and manages configuration from YAML and CLI arguments."""

    DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to config YAML file (default: configs/config.yaml)
        """
        self.config_path = (
            Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        )
        self._config: Optional[Config] = None

    def load_from_yaml(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dictionary with configuration values

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create it or specify a different path."
            )

        with open(self.config_path, "r") as f:
            config_dict: Dict[str, Any] = yaml.safe_load(f) or {}

        return config_dict

    def parse_cli_args(self) -> argparse.Namespace:
        """
        Parse CLI arguments for training and inference.

        Returns:
            Parsed arguments namespace
        """
        parser = argparse.ArgumentParser(
            description="Object Detection Training and Inference"
        )

        # Model arguments
        parser.add_argument(
            "--framework",
            type=str,
            choices=["ultralytics", "mmdet"],
            help="Model framework",
        )
        parser.add_argument(
            "--model-name", type=str, help="Model name (e.g., yolov8n, rtmdet-s)"
        )
        parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")

        # Dataset arguments
        parser.add_argument(
            "--dataset-path", type=str, help="Path to dataset directory"
        )
        parser.add_argument(
            "--dataset-format",
            type=str,
            choices=["coco", "yolo"],
            help="Dataset format",
        )

        # Training arguments
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, help="Batch size")
        parser.add_argument("--img-size", type=int, help="Image size")
        parser.add_argument(
            "--output-dir", type=str, help="Output directory for models"
        )
        parser.add_argument("--device", type=str, help="Device (cuda, cpu, mps)")

        # Inference arguments
        parser.add_argument("--video-path", type=str, help="Path to input video")
        parser.add_argument("--image-path", type=str, help="Path to input image")
        parser.add_argument("--conf-threshold", type=float, help="Confidence threshold")
        parser.add_argument("--config", type=str, help="Path to config YAML file")

        return parser.parse_args()

    def merge_configs(
        self, yaml_config: Dict[str, Any], cli_args: Optional[argparse.Namespace] = None
    ) -> Config:
        """
        Merge YAML config with CLI arguments (CLI takes precedence).

        Args:
            yaml_config: Configuration from YAML file
            cli_args: Parsed CLI arguments (optional)

        Returns:
            Merged Config object
        """
        # Start with YAML config
        merged = yaml_config.copy()

        # Override with CLI args if provided
        if cli_args:
            if cli_args.framework:
                merged.setdefault("model", {})["framework"] = cli_args.framework
            if cli_args.model_name:
                merged.setdefault("model", {})["name"] = cli_args.model_name
            if cli_args.checkpoint:
                merged.setdefault("model", {})["checkpoint"] = cli_args.checkpoint
            if cli_args.dataset_path:
                merged.setdefault("dataset", {})["path"] = cli_args.dataset_path
            if cli_args.dataset_format:
                merged.setdefault("dataset", {})["format"] = cli_args.dataset_format
            if cli_args.epochs:
                merged.setdefault("training", {})["epochs"] = cli_args.epochs
            if cli_args.batch_size:
                merged.setdefault("training", {})["batch_size"] = cli_args.batch_size
            if cli_args.img_size:
                merged.setdefault("training", {})["img_size"] = cli_args.img_size
            if cli_args.output_dir:
                merged.setdefault("training", {})["output_dir"] = cli_args.output_dir
            if cli_args.device:
                merged.setdefault("training", {})["device"] = cli_args.device
                merged.setdefault("inference", {})["device"] = cli_args.device
            if cli_args.video_path:
                merged.setdefault("inference", {})["video_path"] = cli_args.video_path
            if cli_args.image_path:
                merged.setdefault("inference", {})["image_path"] = cli_args.image_path
            if cli_args.conf_threshold:
                merged.setdefault("inference", {})[
                    "conf_threshold"
                ] = cli_args.conf_threshold

        # Create Config object
        model_config = ModelConfig(**merged.get("model", {}))
        dataset_config = DatasetConfig(**merged.get("dataset", {}))
        training_config = TrainingConfig(**merged.get("training", {}))
        inference_config = InferenceConfig(**merged.get("inference", {}))

        return Config(
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            inference=inference_config,
        )

    def load(self, cli_args: Optional[argparse.Namespace] = None) -> Config:
        """
        Load configuration from YAML and merge with CLI arguments.

        Args:
            cli_args: Parsed CLI arguments (optional)

        Returns:
            Config object
        """
        yaml_config = self.load_from_yaml()
        config = self.merge_configs(yaml_config, cli_args)
        self._config = config
        return config

    def get_config(self) -> Config:
        """
        Get loaded configuration.

        Returns:
            Config object

        Raises:
            ValueError: If config hasn't been loaded yet
        """
        if self._config is None:
            raise ValueError("Config not loaded. Call load() first.")
        return self._config