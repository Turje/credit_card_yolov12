# Partial Occlusion Dataset Generator

This guide explains how to create partially obscured datasets for training models that need to handle partial object detection.

## Overview

The `obscure.py` script generates modified versions of your dataset where objects are partially occluded. This is useful for:
- Training robust models that can detect partially visible objects
- Data augmentation for challenging scenarios
- Testing model performance under occlusion conditions

## Quick Start

### Basic Usage

```bash
# Using Makefile (simplest)
make obscure DATASET=datasets/credit-cards-coco OUTPUT=datasets/credit-cards-obscured TYPE=patch RATIO=0.3

# Using Python directly
python3 src/obscure.py \
    --dataset datasets/credit-cards-coco \
    --output datasets/credit-cards-obscured \
    --type patch \
    --ratio 0.3
```

## Occlusion Types

### 1. **patch** (Default)
Random colored patches placed over bounding boxes.

```bash
python3 src/obscure.py \
    --dataset datasets/credit-cards-coco \
    --output datasets/credit-cards-patch \
    --type patch \
    --ratio 0.3 \
    --num-patches 3 \
    --patch-size 0.2
```

**Parameters:**
- `--ratio`: Percentage of bbox area to obscure (0.0-1.0)
- `--num-patches`: Number of random patches per bbox
- `--patch-size`: Size of each patch relative to bbox (0.0-1.0)

### 2. **blur**
Blur portions of the bounding box region.

```bash
python3 src/obscure.py \
    --dataset datasets/credit-cards-coco \
    --output datasets/credit-cards-blur \
    --type blur \
    --ratio 0.4 \
    --blur-kernel 51
```

**Parameters:**
- `--ratio`: Percentage of bbox area to blur
- `--blur-kernel`: Blur kernel size (must be odd, e.g., 51, 101)

### 3. **noise**
Add random noise to portions of the bounding box.

```bash
python3 src/obscure.py \
    --dataset datasets/credit-cards-coco \
    --output datasets/credit-cards-noise \
    --type noise \
    --ratio 0.3 \
    --noise-intensity 50.0
```

**Parameters:**
- `--ratio`: Percentage of bbox area to add noise
- `--noise-intensity`: Intensity of noise (0-255)

### 4. **black**
Black patches over bounding boxes.

```bash
python3 src/obscure.py \
    --dataset datasets/credit-cards-coco \
    --output datasets/credit-cards-black \
    --type black \
    --ratio 0.3
```

### 5. **white**
White patches over bounding boxes.

```bash
python3 src/obscure.py \
    --dataset datasets/credit-cards-coco \
    --output datasets/credit-cards-white \
    --type white \
    --ratio 0.3
```

### 6. **random**
Randomly selects one of the above types for each bounding box.

```bash
python3 src/obscure.py \
    --dataset datasets/credit-cards-coco \
    --output datasets/credit-cards-random \
    --type random \
    --ratio 0.3
```

## Examples

### Example 1: Light Occlusion (20% patches)
```bash
make obscure DATASET=datasets/credit-cards-coco \
    OUTPUT=datasets/credit-cards-light \
    TYPE=patch RATIO=0.2 NUM_PATCHES=2
```

### Example 2: Heavy Occlusion (50% blur)
```bash
make obscure DATASET=datasets/credit-cards-coco \
    OUTPUT=datasets/credit-cards-heavy \
    TYPE=blur RATIO=0.5 BLUR_KERNEL=101
```

### Example 3: Reproducible Dataset (with seed)
```bash
python3 src/obscure.py \
    --dataset datasets/credit-cards-coco \
    --output datasets/credit-cards-reproducible \
    --type patch \
    --ratio 0.3 \
    --seed 42
```

## Output Structure

The obscured dataset maintains the same structure as the input:

```
datasets/credit-cards-obscured/
└── train/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Tips

1. **Start with light occlusion** (ratio 0.2-0.3) and gradually increase
2. **Use different types** for variety in training data
3. **Set a seed** for reproducibility when comparing models
4. **Visualize results** using the overlay tool:
   ```bash
   make overlay DATASET=datasets/credit-cards-obscured
   ```

## Combining with Original Dataset

You can combine the original and obscured datasets for training:

```bash
# Create multiple variants
make obscure DATASET=datasets/credit-cards-coco OUTPUT=datasets/credit-cards-patch TYPE=patch RATIO=0.3
make obscure DATASET=datasets/credit-cards-coco OUTPUT=datasets/credit-cards-blur TYPE=blur RATIO=0.3

# Use both original and obscured datasets for training
```

## Parameters Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--type` | `patch` | Occlusion type (patch, blur, noise, black, white, random) |
| `--ratio` | `0.3` | Occlusion ratio (0.0-1.0) |
| `--num-patches` | `3` | Number of patches (for patch type) |
| `--patch-size` | `0.2` | Patch size ratio (for patch type) |
| `--blur-kernel` | `51` | Blur kernel size (for blur type, must be odd) |
| `--noise-intensity` | `50.0` | Noise intensity (for noise type) |
| `--seed` | `None` | Random seed for reproducibility |

