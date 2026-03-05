"""
Script to verify video content and check for traffic signs
"""
import cv2
import os

video_path = r"e:\Kotresh Rubixe\road-sign-app\uploads\delhi_highway_drive.mp4"

print("=" * 70)
print("VIDEO VERIFICATION")
print("=" * 70)

# Check file size
file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
print(f"\n[1] File Size: {file_size_mb:.2f} MB")

if file_size_mb > 100:
    print(f"    ⚠️  WARNING: File exceeds 100MB upload limit!")
    print(f"    Need to increase MAX_UPLOAD_SIZE in config.py")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("\n✗ ERROR: Could not open video file")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps if fps > 0 else 0

print(f"\n[2] Video Properties:")
print(f"    Resolution: {width}x{height}")
print(f"    FPS: {fps:.2f}")
print(f"    Total Frames: {frame_count}")
print(f"    Duration: {duration:.2f} seconds")

# Extract sample frames
print(f"\n[3] Extracting sample frames...")
output_dir = r"e:\Kotresh Rubixe\road-sign-app\uploads\frames"
os.makedirs(output_dir, exist_ok=True)

# Extract frames at different timestamps
timestamps = [0, 10, 20, 30, 40, 50]  # seconds
extracted = 0

for ts in timestamps:
    frame_num = int(ts * fps)
    if frame_num >= frame_count:
        continue
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if ret:
        output_path = os.path.join(output_dir, f"frame_{ts:02d}s.jpg")
        cv2.imwrite(output_path, frame)
        extracted += 1
        print(f"    ✓ Saved frame at {ts}s -> frame_{ts:02d}s.jpg")

cap.release()

print(f"\n[4] Summary:")
print(f"    ✓ Video is valid")
print(f"    ✓ Extracted {extracted} sample frames")
print(f"    📁 Frames saved to: {output_dir}")
print(f"\n    👉 Please check the frames manually to see if traffic signs are visible")
print(f"    👉 If file size > 100MB, increase MAX_UPLOAD_SIZE in config.py")

print("\n" + "=" * 70)
