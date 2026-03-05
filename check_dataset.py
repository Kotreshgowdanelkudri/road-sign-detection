"""
Helper script to organize dataset and verify structure
"""
import os
import shutil
from pathlib import Path
from config import DATASET_PATH
from utils.label_mapping import discover_dataset_structure, load_label_mapping


def check_dataset_structure():
    """Check and display dataset structure"""
    print("=" * 60)
    print("Dataset Structure Checker")
    print("=" * 60)
    
    info = discover_dataset_structure(DATASET_PATH)
    
    if not info['exists']:
        print(f"\n❌ Dataset not found at: {DATASET_PATH}")
        print("\nPlease create the dataset folder and organize your images.")
        print("\nRecommended structure:")
        print("  dataset/")
        print("    0/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    1/")
        print("      image1.jpg")
        print("    ...")
        return False
    
    print(f"\n✅ Dataset found at: {DATASET_PATH}")
    print(f"Structure: {info['structure']}")
    
    if info['structure'] == 'flat':
        print("\n⚠️  WARNING: Flat structure detected (all images in one folder)")
        print("This structure cannot be used for supervised learning.")
        print("\nPlease organize images into class folders:")
        print("  dataset/0/, dataset/1/, dataset/2/, etc.")
        return False
    
    print(f"\n✅ Folder-based structure detected")
    print(f"Number of classes: {info['num_classes']}")
    
    # Count images per class
    print("\nClass distribution:")
    for class_id in info['classes'][:10]:  # Show first 10
        class_path = os.path.join(DATASET_PATH, str(class_id))
        num_images = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm'))])
        print(f"  Class {class_id}: {num_images} images")
    
    if info['num_classes'] > 10:
        print(f"  ... and {info['num_classes'] - 10} more classes")
    
    # Check label mapping
    print("\n" + "=" * 60)
    print("Label Mapping")
    print("=" * 60)
    
    mapping = load_label_mapping()
    
    if not mapping:
        print("\n⚠️  No label mapping found")
        print("A default mapping will be created during training.")
    elif len(mapping) != info['num_classes']:
        print(f"\n⚠️  Mapping mismatch: {len(mapping)} labels vs {info['num_classes']} classes")
        print("Mapping will be updated during training.")
    else:
        print(f"\n✅ Label mapping valid ({len(mapping)} classes)")
        print("\nSample mappings:")
        for i in list(mapping.keys())[:5]:
            print(f"  {i}: {mapping[i]}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset structure is valid and ready for training!")
    print("=" * 60)
    
    return True


def create_sample_dataset_structure():
    """Create sample dataset structure for demonstration"""
    print("\nCreating sample dataset structure...")
    
    sample_path = os.path.join(os.path.dirname(DATASET_PATH), 'dataset_sample')
    os.makedirs(sample_path, exist_ok=True)
    
    # Create 5 sample class folders
    for i in range(5):
        class_path = os.path.join(sample_path, str(i))
        os.makedirs(class_path, exist_ok=True)
    
    print(f"Sample structure created at: {sample_path}")
    print("\nStructure:")
    print("  dataset_sample/")
    for i in range(5):
        print(f"    {i}/")
    
    print("\nYou can now copy your images into these folders.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--create-sample':
        create_sample_dataset_structure()
    else:
        check_dataset_structure()
