"""
Evaluate model on progressive occlusion test sets.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import tempfile


def evaluate_model_on_test_set(model_path: str, test_dataset_path: str):
    """
    Evaluate model on a test set.
    
    Args:
        model_path: Path to trained model
        test_dataset_path: Path to test dataset
    
    Returns:
        Dictionary with metrics
    """
    import shutil
    
    model = YOLO(model_path)
    test_path = Path(test_dataset_path)
    
    # Find annotation file (could be in test_path/train/ or test_path/)
    ann_file = test_path / "train" / "_annotations.coco.json"
    train_dir = test_path / "train"
    if not ann_file.exists():
        ann_file = test_path / "_annotations.coco.json"
        train_dir = test_path
    
    if not ann_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found. Checked:\n"
            f"  - {test_path / 'train' / '_annotations.coco.json'}\n"
            f"  - {test_path / '_annotations.coco.json'}"
        )
    
    # Load COCO annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    categories = coco_data.get('categories', [])
    class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    images = {img['id']: img for img in coco_data['images']}
    
    # YOLO expects images in train/images/ and labels in train/labels/
    yolo_images_dir = train_dir / "images"
    yolo_labels = train_dir / "labels"
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels.mkdir(parents=True, exist_ok=True)
    
    # Check if conversion already done
    existing_labels = list(yolo_labels.glob("*.txt"))
    existing_images = list(yolo_images_dir.glob("*.jpg")) + list(yolo_images_dir.glob("*.png"))
    
    if len(existing_labels) == 0 or len(existing_images) == 0:
        print(f"  Converting COCO to YOLO format...")
        
        # Copy images and create label files
        for img_id, img_info in images.items():
            img_filename = img_info['file_name']
            img_path = train_dir / img_filename
            
            # Copy image to images directory if not already there
            if img_path.exists():
                dest_img = yolo_images_dir / img_filename
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)
            
            # Get annotations for this image
            img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
            
            # Create YOLO label file
            label_file = yolo_labels / (Path(img_filename).stem + '.txt')
            img_w = img_info['width']
            img_h = img_info['height']
            
            with open(label_file, 'w') as f:
                for ann in img_anns:
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    center_x = (x + w / 2) / img_w
                    center_y = (y + h / 2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    
                    # Get category ID (map to 0-indexed)
                    cat_id = ann['category_id']
                    cat_map = {cat['id']: i for i, cat in enumerate(sorted(categories, key=lambda x: x['id']))}
                    yolo_cat_id = cat_map.get(cat_id, 0)
                    
                    f.write(f"{yolo_cat_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        print(f"  ✅ Created {len(images)} label files and copied images")
    else:
        print(f"  ✅ YOLO format already exists ({len(existing_labels)} labels, {len(existing_images)} images)")
    
    # Create YOLO dataset config
    yolo_config = {
        'path': str(test_path.absolute()),
        'train': 'train',  # Relative to path
        'val': 'train',    # Use train for validation (it's a test set)
        'test': 'train',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
        temp_config = f.name
    
    try:
        # Run validation using the config file
        results = model.val(
            data=temp_config,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            save_json=True,
            project="outputs/evaluation",
            name=test_path.name
        )
    finally:
        # Clean up temp file
        Path(temp_config).unlink(missing_ok=True)
    
    metrics = {
        'mAP50': results.box.map50,
        'mAP50_95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
    }
    
    return metrics


def evaluate_progressive(
    model_path: str,
    test_sets_base: str,
    output_dir: str = "outputs/progressive_evaluation"
):
    """
    Evaluate model on progressive occlusion test sets.
    
    Args:
        model_path: Path to trained model
        test_sets_base: Base directory containing test_occlusion_* folders
        output_dir: Output directory for results
    """
    test_base = Path(test_sets_base)
    
    # If test_base doesn't exist, search for split directory
    if not test_base.exists():
        print(f"⚠️ Test sets base not found: {test_base}")
        print("Searching for split directory...")
        
        # Look for directories ending with _split in multiple locations
        search_dirs = [
            test_base.parent,  # datasets/
            test_base.parent.parent / "datasets",  # MyDrive/credit_card_yolov12/datasets
            Path("/content/drive/MyDrive/credit_card_yolov12/datasets"),
        ]
        
        found_split = None
        for search_dir in search_dirs:
            if search_dir.exists():
                split_dirs = list(search_dir.glob("*_split"))
                if split_dirs:
                    found_split = split_dirs[0]
                    print(f"✅ Found split directory: {found_split}")
                    break
        
        if found_split:
            test_base = found_split
        else:
            raise FileNotFoundError(
                f"Could not find split directory. Searched:\n"
                f"  - {test_base.parent}\n"
                f"  - {test_base.parent.parent / 'datasets'}\n"
                f"  - /content/drive/MyDrive/credit_card_yolov12/datasets"
            )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all test sets
    occlusion_levels = [0, 25, 50, 75]
    results = {}
    
    print("Evaluating model on progressive occlusion test sets...\n")
    
    for level in occlusion_levels:
        test_set_path = test_base / f"test_occlusion_{level}"
        
        if not test_set_path.exists():
            print(f"Warning: Test set not found: {test_set_path}")
            continue
        
        print(f"Evaluating on {level}% occlusion...")
        
        try:
            metrics = evaluate_model_on_test_set(model_path, str(test_set_path))
            results[level] = metrics
            
            print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
            print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}\n")
        
        except Exception as e:
            print(f"  Error evaluating {level}% occlusion: {e}\n")
            continue
    
    # Save results to JSON
    results_file = output_dir / "progressive_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Create visualization
    plot_progressive_results(results, output_dir)
    
    return results


def plot_progressive_results(results: dict, output_dir: Path):
    """Plot progressive occlusion results."""
    if not results:
        print("No results to plot")
        return
    
    occlusion_levels = sorted(results.keys())
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[level][metric] for level in occlusion_levels]
        
        ax.plot(occlusion_levels, values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Occlusion Level (%)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Occlusion Level')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add value labels
        for level, value in zip(occlusion_levels, values):
            ax.text(level, value + 0.02, f'{value:.3f}', ha='center', fontsize=9)
    
    # Combined plot
    ax = axes[5]
    for metric in ['mAP50', 'precision', 'recall']:
        values = [results[level][metric] for level in occlusion_levels]
        ax.plot(occlusion_levels, values, marker='o', label=metric.upper(), linewidth=2)
    
    ax.set_xlabel('Occlusion Level (%)')
    ax.set_ylabel('Score')
    ax.set_title('Key Metrics vs Occlusion Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plot_file = output_dir / "progressive_occlusion_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_file}")
    
    # Create summary table
    df = pd.DataFrame(results).T
    df.index.name = 'Occlusion_Level'
    csv_file = output_dir / "progressive_results.csv"
    df.to_csv(csv_file)
    print(f"CSV saved to: {csv_file}")
    
    print("\n" + "="*60)
    print("PROGRESSIVE OCCLUSION EVALUATION SUMMARY")
    print("="*60)
    print(df.to_string())
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on progressive occlusion test sets")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--test-sets",
        type=str,
        required=True,
        help="Base directory containing test_occlusion_* folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/progressive_evaluation",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_progressive(
            model_path=args.model,
            test_sets_base=args.test_sets,
            output_dir=args.output
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

