"""
Quick script to view sample occluded images.
Opens a few random images to see occlusion effects.
"""
import cv2
import random
from pathlib import Path
import sys


def view_samples(dataset_path: str, num_samples: int = 5):
    """View random sample images from dataset."""
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / "train"
    
    # Get all images
    images = list(train_dir.glob("*.jpg"))
    if not images:
        print(f"No images found in {train_dir}")
        return
    
    # Random sample
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    print(f"Showing {len(sample_images)} sample images...")
    print("Press any key to move to next image, 'q' to quit\n")
    
    for i, img_path in enumerate(sample_images, 1):
        print(f"Image {i}/{len(sample_images)}: {img_path.name}")
        
        # Load and display
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not load: {img_path}")
            continue
        
        # Resize if too large
        h, w = img.shape[:2]
        if w > 1200 or h > 800:
            scale = min(1200/w, 800/h)
            new_w, new_h = int(w*scale), int(h*scale)
            img = cv2.resize(img, (new_w, new_h))
        
        cv2.imshow(f"Sample {i}: {img_path.name}", img)
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            break
    
    print("\nDone viewing samples!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 view_samples.py <dataset_path> [num_samples]")
        print("Example: python3 view_samples.py datasets/sample_occluded 5")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    view_samples(dataset_path, num_samples)

