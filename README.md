# Object Detection Training and Inference Pipeline

A comprehensive, production-ready pipeline for object detection using YOLOv8 (Ultralytics) with support for dataset management, training, inference, and advanced features like per-class thresholds, custom visualization, batch processing, and multi-GPU support.

## 🚀 Features

### Core Capabilities
- **Dataset Management**: Download datasets from Roboflow, convert between formats (COCO/YOLO)
- **Training**: Unified training interface supporting YOLOv8 models
- **Inference**: Advanced inference with video/image processing, batch processing, and export capabilities
- **Visualization**: Custom visualization tools with statistics and overlays
- **Configuration**: YAML-based unified configuration system

### Advanced Inference Features
- ✅ **Per-class confidence thresholds** - Different thresholds for different object classes
- ✅ **Custom visualization** - Custom colors, labels, and confidence display
- ✅ **Detection export** - Export to JSON, CSV, or XML (Pascal VOC) formats
- ✅ **Class filtering** - Include/exclude specific classes
- ✅ **Batch image processing** - Process multiple images efficiently
- ✅ **Multi-GPU support** - Distribute inference across multiple GPUs
- ✅ **Frame skipping** - Faster video processing by skipping frames
- ✅ **Multimodal single GPU initialization** - Efficient model loading for multiple tasks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Management](#dataset-management)
- [Training](#training)
- [Inference](#inference)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Roboflow API key (for dataset downloads)

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd credit_card_yolov12

# Install dependencies
make install
# or
pip install -r requirements.txt

# Setup project directories
make setup
```

### Set Roboflow API Key (Optional)

```bash
export ROBOFLOW_API_KEY=your_api_key_here
```

Or create a `.env` file:
```
ROBOFLOW_API_KEY=your_api_key_here
```

## 🎯 Quick Start

### 1. Download a Dataset

```bash
make download WORKSPACE=your-workspace PROJECT=project-name VERSION=1
```

### 2. Train a Model

```bash
make train-unified DATASET=datasets/your_dataset MODEL_NAME=yolov8n EPOCHS=100
```

### 3. Run Inference

```bash
# Image inference
make inference-image MODEL=models/model_n/weights/best.pt IMAGE=test.jpg OUTPUT=output.jpg

# Video inference
make inference-video MODEL=models/model_n/weights/best.pt VIDEO=test.mp4 OUTPUT=result.mp4
```

## 📦 Dataset Management

### Download from Roboflow

```bash
# Download any dataset
make download WORKSPACE=workspace PROJECT=project VERSION=1

# List downloaded datasets
make list-datasets
```

### Split Dataset

```bash
make split-dataset DATASET=datasets/my_dataset
```

This creates train/val/test splits in `datasets/my_dataset_split/`.

### Convert Formats

The training pipeline automatically converts COCO to YOLO format when needed.

### Using Python API

```python
from src import RoboflowDownloader

downloader = RoboflowDownloader()
downloader.download_dataset(
    workspace="your-workspace",
    project="project-name",
    version=1,
    format="yolov8",
    location="datasets"
)
```

## 🏋️ Training

### Unified Training Interface

The unified training interface supports YOLOv8 models with a flexible configuration system.

#### Using Makefile

```bash
make train-unified \
    DATASET=datasets/my_dataset \
    MODEL_NAME=yolov8n \
    EPOCHS=100 \
    BATCH_SIZE=16 \
    IMG_SIZE=640 \
    DEVICE=cuda
```

#### Using Python CLI

```bash
python src/train_unified.py \
    --dataset-path datasets/my_dataset \
    --model-name yolov8n \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --device cuda \
    --config configs/config.yaml
```

#### Using Python API

```python
from src.models.ultralytics_trainer import UltralyticsTrainer

trainer = UltralyticsTrainer(model_name="yolov8n")
trainer.train(
    dataset_path="datasets/my_dataset",
    epochs=100,
    img_size=640,
    batch_size=16,
    output_dir="models",
    device="cuda"
)
```

### Supported Models

- `yolov8n` - Nano (fastest, smallest)
- `yolov8s` - Small
- `yolov8m` - Medium
- `yolov8l` - Large
- `yolov8x` - XLarge (most accurate)

## 🔍 Inference

### Basic Inference

#### Single Image

```python
from src.models.ultralytics_inference_enhanced import UltralyticsInferenceEnhanced

inference = UltralyticsInferenceEnhanced(
    model_path="models/model_n/weights/best.pt",
    device="cuda"
)

annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    output_path="outputs/result.jpg",
    conf_threshold=0.25
)

print(f"Found {len(detections)} detections")
```

#### Video Processing

```python
total_frames, processed_frames, detections = inference.process_video(
    video_path="test.mp4",
    output_path="outputs/result.mp4",
    conf_threshold=0.25,
    save_video=True
)
```

### Advanced Features

#### Per-Class Confidence Thresholds

```python
# Different thresholds for different classes
class_thresholds = {
    "person": 0.5,   # Higher threshold for person
    "car": 0.3,      # Lower threshold for car
    "bicycle": 0.4
}

annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    class_thresholds=class_thresholds
)
```

#### Class Filtering

```python
# Only show specific classes
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    filter_classes=["person", "car"]  # Only these classes
)

# Exclude specific classes
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    exclude_classes=["background", "noise"]  # Exclude these
)
```

#### Custom Visualization

```python
from src.utils.visualization import get_class_colors

# Get custom colors
custom_colors = get_class_colors(
    ["person", "car", "bicycle"],
    color_scheme="bright"  # or "pastel" or "default"
)

annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    custom_colors=custom_colors,
    show_labels=True,
    show_confidences=True
)
```

#### Detection Export

```python
# Export to JSON
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    export_format="json",
    export_path="outputs/detections.json"
)

# Export to CSV
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    export_format="csv",
    export_path="outputs/detections.csv"
)

# Export to XML (Pascal VOC format)
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    export_format="xml",
    export_path="outputs/detections.xml"
)
```

#### Batch Image Processing

```python
from pathlib import Path

# Process multiple images
image_paths = list(Path("test_images").glob("*.jpg"))

all_detections = inference.process_batch_images(
    image_paths=image_paths,
    output_dir="outputs/batch_results",
    batch_size=8,
    num_workers=4,
    export_format="json"
)
```

#### Video with Frame Skipping

```python
# Process every 2nd frame (2x faster)
total_frames, processed_frames, detections = inference.process_video(
    video_path="long_video.mp4",
    output_path="outputs/fast_result.mp4",
    frame_skip=2,  # Process every 2nd frame
    export_format="json"
)
```

#### Multi-GPU Inference

```python
# Use multiple GPUs
inference = UltralyticsInferenceEnhanced(
    model_path="models/model_n/weights/best.pt",
    device="cuda",
    multi_gpu=True,
    gpu_ids=[0, 1]  # Use GPU 0 and 1
)
```

### Using Makefile Commands

```bash
# Image inference
make inference-image \
    MODEL=models/model_n/weights/best.pt \
    IMAGE=test.jpg \
    OUTPUT=output.jpg \
    CONF_THRESHOLD=0.25

# Video inference
make inference-video \
    MODEL=models/model_n/weights/best.pt \
    VIDEO=test.mp4 \
    OUTPUT=result.mp4 \
    CONF_THRESHOLD=0.25 \
    SHOW=1  # Show video during processing
```

## 📊 Visualization

### Dataset Statistics

```bash
make stats DATASET=datasets/my_dataset
```

Generates statistics plots in `outputs/stats/`:
- Class distribution
- Bounding box statistics
- Bounding box size distribution

### Overlay Images

```bash
make overlay DATASET=datasets/my_dataset
```

Creates images with bounding box overlays in `datasets/my_dataset/overlay/`.

### Visualize Progressive Occlusion

```bash
make visualize-progressive \
    ORIGINAL=datasets/my_dataset_split/test \
    OCC25=datasets/my_dataset_split/test_occlusion_25 \
    OCC50=datasets/my_dataset_split/test_occlusion_50 \
    OCC75=datasets/my_dataset_split/test_occlusion_75
```

### Using Python API

```python
from src.visualize import DatasetVisualizer

visualizer = DatasetVisualizer("datasets/my_dataset")
visualizer.create_overlay_images()
visualizer.generate_statistics()
```

## ⚙️ Configuration

### Config File Structure

The project uses a unified YAML configuration file (`configs/config.yaml`):

```yaml
model:
  framework: "ultralytics"
  name: "yolov8n"
  checkpoint: null

dataset:
  path: "datasets/my_dataset"
  format: "coco"
  train_split: "train"
  val_split: "val"

training:
  epochs: 100
  batch_size: 16
  img_size: 640
  output_dir: "models"
  device: "cuda"
  workers: 8
  patience: 50

inference:
  conf_threshold: 0.25
  iou_threshold: 0.45
  device: "cuda"
  class_thresholds: {}  # Per-class thresholds
  filter_classes: null  # List of classes to include
  exclude_classes: null  # List of classes to exclude
  show_labels: true
  show_confidences: true
  export_format: null  # "json", "csv", "xml"
  frame_skip: 1
  batch_size: 8
  multi_gpu: false
```

### Using Config in Code

```python
from src.utils.config_loader import ConfigLoader

config_loader = ConfigLoader(config_path="configs/config.yaml")
config = config_loader.load()

# Access config values
print(config.model.name)
print(config.training.epochs)
print(config.inference.conf_threshold)
```

## 📁 Project Structure

```
credit_card_yolov12/
├── configs/
│   ├── config.yaml              # Unified configuration
│   └── custom/                   # Custom configs
├── datasets/                     # Downloaded datasets (gitignored)
├── models/                       # Trained models (gitignored)
├── outputs/                      # Output files (gitignored)
│   ├── stats/                    # Statistics plots
│   └── ...
├── src/
│   ├── models/
│   │   ├── ultralytics_trainer.py
│   │   ├── ultralytics_inference.py
│   │   └── ultralytics_inference_enhanced.py  # Advanced features
│   ├── utils/
│   │   ├── config_loader.py      # Config management
│   │   ├── detection_export.py   # Export utilities
│   │   └── visualization.py      # Visualization utilities
│   ├── downloader.py             # Roboflow downloader
│   ├── train.py                  # Legacy training
│   ├── train_unified.py          # Unified training CLI
│   ├── inference.py              # Inference CLI
│   ├── visualize.py              # Visualization tools
│   ├── split_dataset.py          # Dataset splitting
│   └── ...
├── tests/                        # Unit tests
├── requirements.txt              # Dependencies
├── Makefile                      # Convenient commands
├── test_enhanced_inference.py    # Test script
└── README.md                     # This file
```

## 💡 Examples

### Complete Workflow Example

```python
# 1. Download dataset
from src import RoboflowDownloader
downloader = RoboflowDownloader()
downloader.download_dataset(
    workspace="my-workspace",
    project="my-project",
    version=1
)

# 2. Train model
from src.models.ultralytics_trainer import UltralyticsTrainer
trainer = UltralyticsTrainer(model_name="yolov8n")
trainer.train(
    dataset_path="datasets/my_dataset",
    epochs=100,
    batch_size=16
)

# 3. Run inference with advanced features
from src.models.ultralytics_inference_enhanced import UltralyticsInferenceEnhanced

inference = UltralyticsInferenceEnhanced(
    model_path="models/model_n/weights/best.pt"
)

# Process image with per-class thresholds and export
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    output_path="outputs/result.jpg",
    class_thresholds={"person": 0.5, "car": 0.3},
    filter_classes=["person", "car"],
    export_format="json",
    export_path="outputs/detections.json"
)

print(f"Found {len(detections)} detections")
```

### Batch Processing Example

```python
from pathlib import Path
from src.models.ultralytics_inference_enhanced import UltralyticsInferenceEnhanced

inference = UltralyticsInferenceEnhanced(
    model_path="models/model_n/weights/best.pt"
)

# Process all images in a directory
image_dir = Path("test_images")
image_paths = list(image_dir.glob("*.jpg"))

all_detections = inference.process_batch_images(
    image_paths=image_paths,
    output_dir="outputs/batch_results",
    batch_size=8,
    export_format="csv"
)

print(f"Processed {len(image_paths)} images")
print(f"Total detections: {len(all_detections)}")
```

### Video Processing with Export

```python
from src.models.ultralytics_inference_enhanced import UltralyticsInferenceEnhanced

inference = UltralyticsInferenceEnhanced(
    model_path="models/model_n/weights/best.pt"
)

# Process video with frame skipping and export
total_frames, processed_frames, detections = inference.process_video(
    video_path="test_video.mp4",
    output_path="outputs/result.mp4",
    frame_skip=2,  # 2x faster
    class_thresholds={"person": 0.5},
    export_format="json",
    export_path="outputs/video_detections.json"
)

print(f"Processed {processed_frames}/{total_frames} frames")
print(f"Found {len(detections)} total detections")
```

## 🛠️ Available Makefile Commands

```bash
# Setup
make install              # Install dependencies
make setup                # Setup project

# Dataset management
make download             # Download dataset from Roboflow
make split-dataset        # Split dataset into train/val/test
make list-datasets        # List downloaded datasets

# Visualization
make visualize            # Create overlays and statistics
make overlay              # Create overlay images
make stats                # Generate statistics

# Training
make train-model          # Legacy training
make train-unified        # Unified training with config

# Inference
make inference-image      # Image inference
make inference-video      # Video inference

# Code quality
make lint                 # Run linting
make format               # Format code
make type-check           # Type checking
make check                # Run all checks

# Help
make help                 # Show all commands
```

## 🐛 Troubleshooting

### Common Issues

**Model not found:**
- Train a model first or use pretrained YOLOv8: `yolov8n.pt` (downloads automatically)
- Check model path in config or CLI arguments

**CUDA errors:**
- Use `device="cpu"` if no GPU available
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

**Import errors:**
- Make sure you're in the project root directory
- Run: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**No detections:**
- Lower confidence threshold: `conf_threshold=0.1`
- Check if your model supports the classes in your images
- Verify model was trained on similar data

**Memory errors:**
- Reduce batch size: `batch_size=4`
- Use smaller model: `yolov8n` instead of `yolov8x`
- Process images/videos in smaller batches

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [MMDetection Documentation](https://mmdetection.readthedocs.io/) (Future support)

## 🎯 Roadmap

- [ ] MMDetection integration (RTMDet, RT-DETR, YOLOX, DINO)
- [ ] Object tracking support
- [ ] Webcam/streaming inference
- [ ] Model evaluation metrics
- [ ] ONNX/TensorRT export
- [ ] Docker containerization
