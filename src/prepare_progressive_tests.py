"""
Generate progressive occlusion test sets from test split.
Creates test sets with 0%, 25%, 50%, 75% occlusion.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.obscure import PartialOcclusionGenerator


def generate_progressive_tests(
    test_dataset_path: str,
    output_base: str = None,
    occlusion_type: str = "patch",
    seed: int = 42
):
    """
    Generate progressive occlusion test sets.
    
    Args:
        test_dataset_path: Path to test dataset directory
        output_base: Base output directory (default: same as test_dataset_path parent)
        occlusion_type: Type of occlusion
        seed: Random seed
    """
    import shutil
    import json
    
    test_path = Path(test_dataset_path)
    
    # If test_path doesn't exist, search for it
    if not test_path.exists():
        print(f"⚠️ Test path doesn't exist: {test_path}")
        print("Searching for test dataset...")
        
        # Try common locations
        search_paths = [
            test_path.parent,  # credit-cards-coco_split
            test_path.parent.parent,  # datasets
            test_path.parent.parent.parent / "datasets",  # MyDrive/credit_card_yolov12/datasets
        ]
        
        found_test = None
        for search_base in search_paths:
            if not search_base.exists():
                continue
            
            # Look for test directory
            for item in search_base.iterdir():
                if item.is_dir() and "test" in item.name.lower():
                    # Check if it has annotations
                    test_candidates = [
                        item,
                        item / "train",
                    ]
                    for candidate in test_candidates:
                        if candidate.exists():
                            ann_file = candidate / "_annotations.coco.json"
                            if ann_file.exists():
                                found_test = candidate
                                print(f"✅ Found test dataset at: {found_test}")
                                break
                    if found_test:
                        break
                if found_test:
                    break
            if found_test:
                break
        
        if found_test:
            test_path = found_test.parent if found_test.name == "train" else found_test
            print(f"✅ Using test path: {test_path}")
        else:
            # Last resort: check if original dataset has a test set, or use train set temporarily
            print("⚠️ Test dataset not found in split directories.")
            print("Checking original dataset structure...")
            
            # Look for original dataset
            original_dataset = test_path.parent.parent
            possible_test_locations = [
                original_dataset / "test",
                original_dataset / "test" / "train",
                original_dataset / "val",  # Use val as fallback
                original_dataset / "val" / "train",
                original_dataset / "train",  # Use train as last resort
            ]
            
            for loc in possible_test_locations:
                if loc.exists():
                    ann_file = loc / "_annotations.coco.json"
                    if not ann_file.exists() and loc.parent.exists():
                        ann_file = loc.parent / "_annotations.coco.json"
                    
                    if ann_file.exists():
                        found_test = loc
                        test_path = loc.parent if loc.name == "train" else loc
                        print(f"✅ Found alternative dataset at: {test_path}")
                        print(f"⚠️ Note: Using {loc.name} set. Consider running split_dataset.py first.")
                        break
            
            if not found_test:
                raise FileNotFoundError(
                    f"Test dataset not found. Searched:\n"
                    f"  - {test_path}\n"
                    f"  - {test_path.parent}\n"
                    f"  - {test_path.parent.parent}\n"
                    f"\nPlease run dataset splitting first:\n"
                    f"  python src/split_dataset.py --dataset <dataset_path> --seed 42"
                )
    
    if output_base is None:
        output_base = test_path.parent
    
    output_base = Path(output_base)
    # Ensure output_base directory exists
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find source directory ONCE (could be test/ or test/train/)
    # This will be reused for both baseline and occluded test sets
    src_dir = test_path
    if (test_path / "train").exists():
        src_dir = test_path / "train"
    elif test_path.name == "train":
        # Already pointing to train directory
        src_dir = test_path
    
    # Verify source directory exists and has files
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {src_dir}\n"
            f"Test path: {test_path}\n"
            f"Test path exists: {test_path.exists()}\n"
            f"Please check the test dataset path."
        )
    
    # Verify annotation file exists
    ann_file = src_dir / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found at: {ann_file}\n"
            f"Source directory: {src_dir}\n"
            f"Files in source: {list(src_dir.glob('*'))[:10]}"
        )
    
    print(f"✅ Using source directory: {src_dir}")
    print(f"   Annotation file: {ann_file}")
    
    # Test set with 0% occlusion (original)
    print("\nCreating baseline test set (0% occlusion)...")
    test_0_path = output_base / "test_occlusion_0"
    test_0_path.mkdir(parents=True, exist_ok=True)
    (test_0_path / "train").mkdir(parents=True, exist_ok=True)
    
    # Copy images
    for img_file in src_dir.glob("*.jpg"):
        shutil.copy2(img_file, test_0_path / "train" / img_file.name)
    
    # Copy annotations
    ann_file = src_dir / "_annotations.coco.json"
    if ann_file.exists():
        shutil.copy2(ann_file, test_0_path / "train" / "_annotations.coco.json")
    
    print(f"✅ Baseline test set created: {test_0_path}")
    
    # Generate occluded test sets
    occlusion_levels = [0.25, 0.50, 0.75]
    
    for occlusion_ratio in occlusion_levels:
        print(f"\nGenerating test set with {occlusion_ratio*100:.0f}% occlusion...")
        
        output_path = output_base / f"test_occlusion_{int(occlusion_ratio*100)}"
        
        # Use src_dir that was detected at the beginning (reuse for consistency)
        # obscure.py expects a structure with train/ subdirectory
        # Check if we need to create a temporary structure
        temp_test = None
        if (test_path / "train").exists():
            # test_path already has train/ subdirectory, use it directly
            temp_test = test_path
            print(f"✅ Using existing structure: {temp_test}")
        else:
            # Create a temporary structure for obscure.py
            temp_test = test_path.parent / f"_temp_{test_path.name}"
            temp_test.mkdir(parents=True, exist_ok=True)
            (temp_test / "train").mkdir(parents=True, exist_ok=True)
            print(f"✅ Creating temp structure: {temp_test}")
            
            # Copy files from the detected source directory
            img_count = 0
            for img_file in src_dir.glob("*.jpg"):
                shutil.copy2(img_file, temp_test / "train" / img_file.name)
                img_count += 1
            for img_file in src_dir.glob("*.png"):
                shutil.copy2(img_file, temp_test / "train" / img_file.name)
                img_count += 1
            print(f"   Copied {img_count} images")
            
            # Copy annotations from the detected source directory (already verified to exist)
            ann_file = src_dir / "_annotations.coco.json"
            shutil.copy2(ann_file, temp_test / "train" / "_annotations.coco.json")
            print(f"✅ Copied annotation file to temp directory")
        
        # Verify temp_test has the required structure before proceeding
        if not (temp_test / "train" / "_annotations.coco.json").exists():
            raise FileNotFoundError(
                f"Annotation file missing in temp directory: {temp_test / 'train' / '_annotations.coco.json'}\n"
                f"Temp directory contents: {list((temp_test / 'train').glob('*')) if (temp_test / 'train').exists() else 'train/ does not exist'}"
            )
        
        generator = PartialOcclusionGenerator(str(temp_test))
        generator.generate_obscured_dataset(
            output_path=str(output_path),
            occlusion_type=occlusion_type,
            occlusion_ratio=occlusion_ratio,
            random_seed=seed
        )
        
        # Clean up temp directory if created
        if temp_test != test_path and temp_test.exists():
            shutil.rmtree(temp_test)
        
        print(f"✅ Test set created: {output_path}")
    
    print(f"\n✅ All progressive test sets created!")
    print(f"Test sets available:")
    print(f"  - test_occlusion_0 (baseline)")
    for ratio in occlusion_levels:
        print(f"  - test_occlusion_{int(ratio*100)} ({ratio*100:.0f}% occlusion)")


def main():
    parser = argparse.ArgumentParser(description="Generate progressive occlusion test sets")
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="Path to test dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base directory (default: same as test dataset parent)"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="crop",
        choices=["patch", "blur", "noise", "black", "white", "crop", "random"],
        help="Occlusion type: 'crop' (camera pan/zoom, recommended), 'patch' (random patches), etc. (default: crop)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        generate_progressive_tests(
            test_dataset_path=args.test_dataset,
            output_base=args.output,
            occlusion_type=args.type,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

