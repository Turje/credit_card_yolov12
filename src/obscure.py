"""
Partial occlusion dataset generator.
Creates a modified dataset with partially obscured objects.
"""
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2


class PartialOcclusionGenerator:
    """Generates partially obscured datasets from COCO format."""
    
    OCCLUSION_TYPES = ['patch', 'blur', 'noise', 'black', 'white', 'random']
    
    def __init__(self, dataset_path: str):
        """
        Initialize generator.
        
        Args:
            dataset_path: Path to dataset directory
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
    
    def generate_obscured_dataset(
        self,
        output_path: str,
        occlusion_type: str = 'patch',
        occlusion_ratio: float = 0.3,
        num_patches: int = 3,
        patch_size_ratio: float = 0.2,
        blur_kernel_size: int = 51,
        noise_intensity: float = 50.0,
        random_seed: int = None
    ):
        """
        Generate obscured dataset.
        
        Args:
            output_path: Output directory for obscured dataset
            occlusion_type: Type of occlusion ('patch', 'blur', 'noise', 'black', 'white', 'random')
            occlusion_ratio: Ratio of bounding box area to obscure (0.0-1.0)
            num_patches: Number of patches for 'patch' type occlusion
            patch_size_ratio: Size of each patch relative to bbox (for 'patch' type)
            blur_kernel_size: Kernel size for blur occlusion (must be odd)
            noise_intensity: Intensity of noise (0-255)
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        if occlusion_type not in self.OCCLUSION_TYPES:
            raise ValueError(f"Occlusion type must be one of {self.OCCLUSION_TYPES}")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        output_train_dir = output_path / "train"
        output_train_dir.mkdir(parents=True, exist_ok=True)
        
        train_dir = self.dataset_path / "train"
        
        # Copy categories
        new_coco_data = {
            'info': self.coco_data.get('info', {}),
            'licenses': self.coco_data.get('licenses', []),
            'categories': self.coco_data['categories'],
            'images': [],
            'annotations': []
        }
        
        print(f"Generating obscured dataset with {occlusion_type} occlusion...")
        print(f"Occlusion ratio: {occlusion_ratio*100:.1f}%")
        
        new_image_id = 1
        new_ann_id = 1
        
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
            
            h, w = image.shape[:2]
            obscured_image = image.copy()
            
            # Apply occlusion to each bounding box
            for ann in self.image_annotations[img_id]:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, bbox_w, bbox_h = bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + bbox_w), int(y + bbox_h)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if occlusion_type == 'random':
                    # Randomly select occlusion type for each bbox
                    occ_type = random.choice(['patch', 'blur', 'noise', 'black'])
                else:
                    occ_type = occlusion_type
                
                # Apply occlusion
                self._apply_occlusion(
                    obscured_image,
                    x1, y1, x2, y2,
                    occ_type,
                    occlusion_ratio,
                    num_patches,
                    patch_size_ratio,
                    blur_kernel_size,
                    noise_intensity
                )
            
            # Save obscured image
            output_img_path = output_train_dir / img_filename
            cv2.imwrite(str(output_img_path), obscured_image)
            
            # Add image info
            new_img_info = img_info.copy()
            new_img_info['id'] = new_image_id
            new_img_info['file_name'] = img_filename
            new_coco_data['images'].append(new_img_info)
            
            # Copy annotations
            for ann in self.image_annotations[img_id]:
                new_ann = ann.copy()
                new_ann['id'] = new_ann_id
                new_ann['image_id'] = new_image_id
                new_coco_data['annotations'].append(new_ann)
                new_ann_id += 1
            
            new_image_id += 1
        
        # Save new annotations file
        output_ann_file = output_train_dir / "_annotations.coco.json"
        with open(output_ann_file, 'w') as f:
            json.dump(new_coco_data, f, indent=2)
        
        print(f"\nObscured dataset created at: {output_path}")
        print(f"Total images: {len(new_coco_data['images'])}")
        print(f"Total annotations: {len(new_coco_data['annotations'])}")
    
    def _apply_occlusion(
        self,
        image,
        x1, y1, x2, y2,
        occlusion_type,
        occlusion_ratio,
        num_patches,
        patch_size_ratio,
        blur_kernel_size,
        noise_intensity
    ):
        """Apply occlusion to a region of the image."""
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        if occlusion_type == 'patch':
            # Random patches
            total_area = bbox_w * bbox_h
            target_area = total_area * occlusion_ratio
            
            for _ in range(num_patches):
                patch_w = int(bbox_w * patch_size_ratio * random.uniform(0.5, 1.5))
                patch_h = int(bbox_h * patch_size_ratio * random.uniform(0.5, 1.5))
                
                # Random position within bbox
                patch_x = random.randint(x1, max(x1 + 1, x2 - patch_w))
                patch_y = random.randint(y1, max(y1 + 1, y2 - patch_h))
                
                # Ensure within bounds
                patch_x = min(patch_x, x2 - patch_w)
                patch_y = min(patch_y, y2 - patch_h)
                
                # Random color patch
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                image[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w] = color
        
        elif occlusion_type == 'blur':
            # Blur region
            region = image[y1:y2, x1:x2].copy()
            blurred = cv2.GaussianBlur(region, (blur_kernel_size, blur_kernel_size), 0)
            
            # Apply blur to a portion of the region
            mask = np.random.random((y2-y1, x2-x1)) < occlusion_ratio
            mask = mask[:, :, np.newaxis]
            image[y1:y2, x1:x2] = np.where(mask, blurred, image[y1:y2, x1:x2])
        
        elif occlusion_type == 'noise':
            # Add noise
            noise = np.random.normal(0, noise_intensity, (y2-y1, x2-x1, 3))
            mask = np.random.random((y2-y1, x2-x1, 1)) < occlusion_ratio
            
            noisy_region = np.clip(image[y1:y2, x1:x2].astype(np.float32) + noise, 0, 255).astype(np.uint8)
            image[y1:y2, x1:x2] = np.where(mask, noisy_region, image[y1:y2, x1:x2])
        
        elif occlusion_type == 'black':
            # Black patches
            mask = np.random.random((y2-y1, x2-x1)) < occlusion_ratio
            mask = mask[:, :, np.newaxis]
            image[y1:y2, x1:x2] = np.where(mask, 0, image[y1:y2, x1:x2])
        
        elif occlusion_type == 'white':
            # White patches
            mask = np.random.random((y2-y1, x2-x1)) < occlusion_ratio
            mask = mask[:, :, np.newaxis]
            image[y1:y2, x1:x2] = np.where(mask, 255, image[y1:y2, x1:x2])


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate partially obscured dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to input dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory for obscured dataset"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="patch",
        choices=PartialOcclusionGenerator.OCCLUSION_TYPES,
        help="Type of occlusion (default: patch)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="Occlusion ratio (0.0-1.0, default: 0.3)"
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=3,
        help="Number of patches for 'patch' type (default: 3)"
    )
    parser.add_argument(
        "--patch-size",
        type=float,
        default=0.2,
        help="Patch size ratio relative to bbox for 'patch' type (default: 0.2)"
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=51,
        help="Blur kernel size (must be odd, default: 51)"
    )
    parser.add_argument(
        "--noise-intensity",
        type=float,
        default=50.0,
        help="Noise intensity for 'noise' type (default: 50.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Validate occlusion ratio
    if not 0.0 <= args.ratio <= 1.0:
        parser.error("--ratio must be between 0.0 and 1.0")
    
    # Ensure blur kernel is odd
    if args.blur_kernel % 2 == 0:
        args.blur_kernel += 1
    
    try:
        generator = PartialOcclusionGenerator(args.dataset)
        generator.generate_obscured_dataset(
            output_path=args.output,
            occlusion_type=args.type,
            occlusion_ratio=args.ratio,
            num_patches=args.num_patches,
            patch_size_ratio=args.patch_size,
            blur_kernel_size=args.blur_kernel,
            noise_intensity=args.noise_intensity,
            random_seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

