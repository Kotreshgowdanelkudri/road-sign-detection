"""
Quick setup script to verify everything is ready for training
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_PATH, MODEL_PATH, IMAGE_SIZE, BATCH_SIZE, EPOCHS
from utils.label_mapping import discover_dataset_structure, load_label_mapping, validate_mapping

def main():
    print("=" * 70)
    print("🚀 Road Sign Detection - Pre-Training Verification")
    print("=" * 70)
    
    # Check dataset
    print("\n[1/4] Checking dataset...")
    info = discover_dataset_structure(DATASET_PATH)
    
    if not info['exists']:
        print(f"❌ ERROR: Dataset not found at {DATASET_PATH}")
        return False
    
    if info['structure'] != 'folder-based':
        print(f"❌ ERROR: Dataset must be folder-based structure")
        return False
    
    print(f"✅ Dataset found: {DATASET_PATH}")
    print(f"   Structure: {info['structure']}")
    print(f"   Classes: {info['num_classes']}")
    
    # Count total images
    total_images = 0
    for class_id in info['classes']:
        class_path = os.path.join(DATASET_PATH, str(class_id))
        num_images = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm'))])
        total_images += num_images
    
    print(f"   Total images: {total_images:,}")
    
    # Check label mapping
    print("\n[2/4] Checking label mapping...")
    mapping = load_label_mapping()
    
    if not mapping:
        print("⚠️  WARNING: No label mapping found")
        print("   Default mapping will be created during training")
    else:
        is_valid = validate_mapping(mapping, info['num_classes'])
        if is_valid:
            print(f"✅ Label mapping valid ({len(mapping)} classes)")
            print(f"   Sample: {mapping[0]}, {mapping[1]}, {mapping[2]}")
        else:
            print(f"⚠️  WARNING: Mapping has {len(mapping)} entries but dataset has {info['num_classes']} classes")
            print("   Mapping will be updated during training")
    
    # Check dependencies
    print("\n[3/4] Checking dependencies...")
    try:
        import tensorflow
        print(f"✅ TensorFlow: {tensorflow.__version__}")
    except ImportError:
        print("❌ ERROR: TensorFlow not installed")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ ERROR: OpenCV not installed")
        return False
    
    try:
        import flask
        print(f"✅ Flask: {flask.__version__}")
    except ImportError:
        print("❌ ERROR: Flask not installed")
        return False
    
    # Show training configuration
    print("\n[4/4] Training configuration...")
    print(f"✅ Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"✅ Batch size: {BATCH_SIZE}")
    print(f"✅ Max epochs: {EPOCHS}")
    print(f"✅ Model will be saved to: {MODEL_PATH}")
    
    # Estimate training time
    estimated_time_per_epoch = (total_images / BATCH_SIZE) * 0.5  # ~0.5 seconds per batch
    estimated_total_minutes = (estimated_time_per_epoch * EPOCHS) / 60
    
    print(f"\n📊 Estimated training time: {estimated_total_minutes:.0f}-{estimated_total_minutes*1.5:.0f} minutes")
    print(f"   (Actual time depends on your hardware)")
    
    print("\n" + "=" * 70)
    print("✅ All checks passed! Ready to train.")
    print("=" * 70)
    
    print("\n🚀 To start training, run:")
    print("   python train_model.py")
    
    print("\n💡 Tips:")
    print("   - Training will take 30 minutes to 2 hours")
    print("   - You can monitor progress in real-time")
    print("   - Model will be saved automatically")
    print("   - Training history and confusion matrix will be generated")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
