# Roboflow Dataset Downloader

A simple, organized tool for downloading datasets from Roboflow for object detection projects.

## Features

- Download any Roboflow dataset with a simple command
- Organized directory structure
- Easy-to-use Makefile commands
- Extensible codebase (< 500 lines)

## Setup

1. **Install dependencies:**
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

2. **Set your Roboflow API key:**
   ```bash
   export ROBOFLOW_API_KEY=your_api_key_here
   ```
   
   Or create a `.env` file:
   ```
   ROBOFLOW_API_KEY=your_api_key_here
   ```

3. **Setup project directories:**
   ```bash
   make setup
   ```

## Usage

### Using Makefile (Recommended)

**Download a dataset:**
```bash
make download WORKSPACE=your-workspace PROJECT=project-name VERSION=1
```

**Example for credit card dataset:**
```bash
make download-credit-card WORKSPACE=your-workspace PROJECT=credit-card VERSION=1
```

**Other useful commands:**
```bash
make help              # Show all available commands
make list-datasets     # List downloaded datasets
make clean             # Clean all downloaded data
make clean-datasets    # Clean only datasets
```

### Using Python directly

```bash
python src/downloader.py \
    --workspace your-workspace \
    --project project-name \
    --version 1 \
    --format yolov8 \
    --location datasets
```

### Using as a Python module

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

## Project Structure

```
credit_card_yolov12/
├── src/
│   ├── __init__.py
│   ├── downloader.py      # Main downloader class
│   └── config.py          # Configuration management
├── datasets/              # Downloaded datasets (gitignored)
├── models/                # Trained models (gitignored)
├── outputs/               # Output files (gitignored)
├── requirements.txt       # Python dependencies
├── Makefile              # Convenient commands
└── README.md             # This file
```

## Getting Your Roboflow API Key

1. Go to [Roboflow](https://roboflow.com)
2. Sign in or create an account
3. Navigate to your account settings
4. Copy your API key
5. Set it as an environment variable or in `.env` file

## Notes

- Datasets are downloaded to `datasets/` directory
- Each dataset is stored in its own folder: `{workspace}_{project}_v{version}`
- Supported formats: yolov8, coco, pascal-voc, etc. (check Roboflow docs)
- All code is kept under 500 lines for simplicity

## Next Steps

This is a basic foundation. You can extend it by:
- Adding dataset validation
- Implementing dataset statistics
- Adding visualization tools
- Integrating with training pipelines
- Adding dataset preprocessing utilities

