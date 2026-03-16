"""
Create a test video from dataset images for demonstration
This ensures we have a video with actual traffic signs
"""
import cv2
import os
import glob
import numpy as np

print("=" * 70)
print("CREATING TEST VIDEO FROM DATASET IMAGES")
print("=" * 70)

# Path to test dataset
test_dataset_path = r"e:\Rubixe Updated\Kotresh Rubixe\dataset\traffic_sign_classification_dataset\traffic_sign_classification_dataset\test"
output_video_path = r"e:\Rubixe Updated\Kotresh Rubixe\road-sign-app\uploads\test_signs_video.mp4"

# Get all image files from test dataset (limit to first 50 for reasonable video length)
print("\n[1] Collecting images from test dataset...")
image_files = []
for root, dirs, files in os.walk(test_dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_files.append(os.path.join(root, file))
            if len(image_files) >= 50:  # Limit to 50 images
                break
    if len(image_files) >= 50:
        break

print(f"   Found {len(image_files)} images")

if len(image_files) == 0:
    print("\n✗ ERROR: No images found in test dataset")
    exit(1)

# Video settings
fps = 1.0  # 1 frame per second (each sign shows for 1 second - valid for VideoWriter)
frame_size = (640, 640)  # Standard size

print(f"\n[2] Creating video...")
print(f"   FPS: {fps}")
print(f"   Frame size: {frame_size}")
print(f"   Duration: ~{len(image_files)/fps:.1f} seconds")

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Process each image
processed = 0
for img_path in image_files:
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize to standard size
        img_resized = cv2.resize(img, frame_size)
        
        # Write frame
        video_writer.write(img_resized)
        processed += 1
        
        if processed % 10 == 0:
            print(f"   Processed {processed}/{len(image_files)} images...")
    
    except Exception as e:
        print(f"   Warning: Skipped {os.path.basename(img_path)}: {e}")
        continue

video_writer.release()

# Verify video was created
if os.path.exists(output_video_path):
    file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
    print(f"\n[3] ✓ Video created successfully!")
    print(f"   Path: {output_video_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Frames: {processed}")
    print(f"   Duration: ~{processed/fps:.1f} seconds")
    print(f"\n   👉 This video contains REAL traffic signs from your dataset!")
    print(f"   👉 Upload it to test your video processing feature!")
else:
    print(f"\n✗ ERROR: Video creation failed")

print("\n" + "=" * 70)
