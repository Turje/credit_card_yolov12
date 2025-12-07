"""
Compare original vs occluded images side by side.
"""
import cv2
import numpy as np
from pathlib import Path
import random
import sys


def compare_images(original_dir: str, occluded_dir: str, num_samples: int = 5):
    """Show side-by-side comparison of original vs occluded images."""
    original_path = Path(original_dir) / "train"
    occluded_path = Path(occluded_dir) / "train"
    
    # Get image files
    original_images = list(original_path.glob("*.jpg"))
    if not original_images:
        print(f"No images found in {original_path}")
        return
    
    # Sample random images
    sample_images = random.sample(original_images, min(num_samples, len(original_images)))
    
    print(f"Showing {len(sample_images)} comparisons...")
    print("Press any key for next image, 'q' to quit\n")
    
    for i, orig_img_path in enumerate(sample_images, 1):
        # Find corresponding occluded image
        occ_img_path = occluded_path / orig_img_path.name
        
        if not occ_img_path.exists():
            print(f"  Occluded image not found: {occ_img_path.name}")
            continue
        
        # Load images
        orig_img = cv2.imread(str(orig_img_path))
        occ_img = cv2.imread(str(occ_img_path))
        
        if orig_img is None or occ_img is None:
            print(f"  Could not load images")
            continue
        
        # Resize to same height
        h1, w1 = orig_img.shape[:2]
        h2, w2 = occ_img.shape[:2]
        
        target_height = min(600, max(h1, h2))
        scale1 = target_height / h1
        scale2 = target_height / h2
        
        orig_resized = cv2.resize(orig_img, (int(w1*scale1), int(h1*scale1)))
        occ_resized = cv2.resize(occ_img, (int(w2*scale2), int(h2*scale2)))
        
        # Make heights match
        h = max(orig_resized.shape[0], occ_resized.shape[0])
        w = orig_resized.shape[1] + occ_resized.shape[1] + 20  # Gap between images
        
        # Create combined image
        combined = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Place images
        combined[:orig_resized.shape[0], :orig_resized.shape[1]] = orig_resized
        combined[:occ_resized.shape[0], orig_resized.shape[1]+20:] = occ_resized
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Occluded (25%)", (orig_resized.shape[1]+30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow(f"Comparison {i}/{len(sample_images)}: {orig_img_path.name}", combined)
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            break
    
    print("\nDone viewing comparisons!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 compare_occlusion.py <original_dataset> <occluded_dataset> [num_samples]")
        print("Example: python3 compare_occlusion.py datasets/credit-cards-coco datasets/sample_occluded 5")
        sys.exit(1)
    
    original_dir = sys.argv[1]
    occluded_dir = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    compare_images(original_dir, occluded_dir, num_samples)

