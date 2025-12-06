"""
Dataset visualization and overlay generation utilities.
"""
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors


class DatasetVisualizer:
    """Handles dataset visualization and overlay generation."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize visualizer.
        
        Args:
            dataset_path: Path to dataset directory (should contain train/_annotations.coco.json)
        """
        self.dataset_path = Path(dataset_path)
        self.annotation_file = self.dataset_path / "train" / "_annotations.coco.json"
        
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Load COCO annotations
        with open(self.annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build lookup dictionaries
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by image
        self.image_annotations = defaultdict(list)
        for ann in self.annotations:
            self.image_annotations[ann['image_id']].append(ann)
    
    def create_overlay_images(self, output_dir: str = None):
        """
        Create images with bounding box overlays.
        
        Args:
            output_dir: Output directory for overlay images (default: dataset_path/overlay)
        """
        if output_dir is None:
            output_dir = self.dataset_path / "overlay"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        train_dir = self.dataset_path / "train"
        
        # Generate colors for each category
        category_colors = self._generate_colors(len(self.categories))
        cat_id_to_color = {
            cat_id: category_colors[i] 
            for i, cat_id in enumerate(self.categories.keys())
        }
        
        print(f"Creating overlay images in {output_dir}...")
        
        for img_id, img_info in self.images.items():
            img_filename = img_info['file_name']
            img_path = train_dir / img_filename
            
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load image: {img_path}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Draw bounding boxes
            for ann in self.image_annotations[img_id]:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                category_name = self.categories[category_id]['name']
                color = cat_id_to_color[category_id]
                
                x, y, width, height = bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + width), int(y + height)
                
                # Draw rectangle
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{category_name}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    image_rgb,
                    (x1, y1 - text_height - baseline - 2),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                cv2.putText(
                    image_rgb,
                    label,
                    (x1, y1 - baseline - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # Save overlay image
            output_path = output_dir / img_filename
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), image_bgr)
        
        print(f"Created {len(self.images)} overlay images in {output_dir}")
    
    def generate_statistics(self, output_dir: str = None):
        """
        Generate dataset statistics and visualizations.
        
        Args:
            output_dir: Output directory for statistics plots (default: outputs/stats)
        """
        if output_dir is None:
            output_dir = Path("outputs") / "stats"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        # Generate visualizations
        self._plot_class_distribution(stats, output_dir)
        self._plot_bbox_statistics(stats, output_dir)
        self._plot_bbox_size_distribution(stats, output_dir)
        
        # Print summary
        self._print_summary(stats)
        
        print(f"\nStatistics plots saved to {output_dir}")
    
    def _calculate_statistics(self):
        """Calculate dataset statistics."""
        stats = {
            'total_images': len(self.images),
            'total_annotations': len(self.annotations),
            'class_counts': defaultdict(int),
            'bbox_widths': [],
            'bbox_heights': [],
            'bbox_areas': [],
            'annotations_per_image': []
        }
        
        for ann in self.annotations:
            category_id = ann['category_id']
            category_name = self.categories[category_id]['name']
            stats['class_counts'][category_name] += 1
            
            bbox = ann['bbox']
            width, height = bbox[2], bbox[3]
            area = width * height
            
            stats['bbox_widths'].append(width)
            stats['bbox_heights'].append(height)
            stats['bbox_areas'].append(area)
        
        for img_id in self.images.keys():
            stats['annotations_per_image'].append(len(self.image_annotations[img_id]))
        
        return stats
    
    def _plot_class_distribution(self, stats, output_dir):
        """Plot class distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        classes = list(stats['class_counts'].keys())
        counts = list(stats['class_counts'].values())
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        
        ax1.bar(classes, counts, color=colors_list)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution (Count)')
        ax1.grid(axis='y', alpha=0.3)
        
        for i, (cls, count) in enumerate(zip(classes, counts)):
            ax1.text(i, count, str(count), ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors_list, startangle=90)
        ax2.set_title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_bbox_statistics(self, stats, output_dir):
        """Plot bounding box statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Average width and height
        avg_width = np.mean(stats['bbox_widths'])
        avg_height = np.mean(stats['bbox_heights'])
        
        axes[0, 0].bar(['Avg Width', 'Avg Height'], [avg_width, avg_height], color=['skyblue', 'lightcoral'])
        axes[0, 0].set_ylabel('Pixels')
        axes[0, 0].set_title(f'Average Bounding Box Size\nWidth: {avg_width:.1f}px, Height: {avg_height:.1f}px')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Width distribution
        axes[0, 1].hist(stats['bbox_widths'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(avg_width, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_width:.1f}px')
        axes[0, 1].set_xlabel('Width (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Bounding Box Width Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Height distribution
        axes[1, 0].hist(stats['bbox_heights'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(avg_height, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_height:.1f}px')
        axes[1, 0].set_xlabel('Height (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Bounding Box Height Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Annotations per image
        axes[1, 1].hist(stats['annotations_per_image'], bins=range(0, max(stats['annotations_per_image']) + 2), 
                        color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Number of Annotations')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].set_title(f'Annotations per Image\nTotal: {stats["total_annotations"]} annotations')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'bbox_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_bbox_size_distribution(self, stats, output_dir):
        """Plot bounding box size distribution (scatter plot)."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        avg_width = np.mean(stats['bbox_widths'])
        avg_height = np.mean(stats['bbox_heights'])
        
        scatter = ax.scatter(stats['bbox_widths'], stats['bbox_heights'], 
                           alpha=0.5, s=50, c=stats['bbox_areas'], 
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Add average point
        ax.scatter([avg_width], [avg_height], color='red', s=200, 
                  marker='x', linewidth=3, label=f'Average ({avg_width:.1f}, {avg_height:.1f})')
        
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        ax.set_title('Bounding Box Size Distribution\n(Color = Area)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Area (pixels²)')
        plt.tight_layout()
        plt.savefig(output_dir / 'bbox_size_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _print_summary(self, stats):
        """Print summary statistics."""
        print("\n" + "="*50)
        print("DATASET STATISTICS SUMMARY")
        print("="*50)
        print(f"Total Images: {stats['total_images']}")
        print(f"Total Annotations: {stats['total_annotations']}")
        print(f"Average Annotations per Image: {np.mean(stats['annotations_per_image']):.2f}")
        print("\nClass Distribution:")
        for cls, count in sorted(stats['class_counts'].items()):
            percentage = (count / stats['total_annotations']) * 100
            print(f"  {cls}: {count} ({percentage:.1f}%)")
        print(f"\nBounding Box Statistics:")
        print(f"  Average Width: {np.mean(stats['bbox_widths']):.2f} pixels")
        print(f"  Average Height: {np.mean(stats['bbox_heights']):.2f} pixels")
        print(f"  Average Area: {np.mean(stats['bbox_areas']):.2f} pixels²")
        print(f"  Min Width: {np.min(stats['bbox_widths']):.2f} pixels")
        print(f"  Max Width: {np.max(stats['bbox_widths']):.2f} pixels")
        print(f"  Min Height: {np.min(stats['bbox_heights']):.2f} pixels")
        print(f"  Max Height: {np.max(stats['bbox_heights']):.2f} pixels")
        print("="*50)
    
    def _generate_colors(self, n):
        """Generate distinct colors for categories."""
        try:
            cmap = plt.colormaps['tab20']
        except (AttributeError, KeyError):
            cmap = plt.cm.get_cmap('tab20')
        colors_list = [cmap(i / n) for i in range(n)]
        # Convert to BGR for OpenCV (but we'll use RGB)
        return [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_list]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Visualize dataset and create overlays")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Create overlay images with bounding boxes"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Generate statistics and visualizations"
    )
    parser.add_argument(
        "--overlay-dir",
        type=str,
        default=None,
        help="Output directory for overlay images"
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=None,
        help="Output directory for statistics plots"
    )
    
    args = parser.parse_args()
    
    if not args.overlay and not args.stats:
        parser.error("At least one of --overlay or --stats must be specified")
    
    try:
        visualizer = DatasetVisualizer(args.dataset)
        
        if args.overlay:
            visualizer.create_overlay_images(args.overlay_dir)
        
        if args.stats:
            visualizer.generate_statistics(args.stats_dir)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

