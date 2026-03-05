"""
Configuration settings for Road Sign Detection Application
All paths support environment variable overrides.
"""
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.h5')
IMAGE_SIZE = 64  # Input size for the model (64x64)
CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence for drawing/strong detection
MIN_DISPLAY_CONFIDENCE = 0.10  # Show best guess for images/video even if below threshold

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_SIZE', 200 * 1024 * 1024))  # 200MB default
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Dataset configuration (set DATASET_PATH env var or use default relative path)
DATASET_PATH = os.environ.get('DATASET_PATH', os.path.join(BASE_DIR, 'dataset', 'Train'))
LABEL_MAPPING_PATH = os.path.join(BASE_DIR, 'label_mapping.json')

# Performance settings
FRAME_SKIP = 5  # Process every 5th frame (faster for high-res videos)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Voice settings
VOICE_RATE = 150
VOICE_VOLUME = 1.0

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# ROI Detection settings (expanded ranges for faded/blurry/low-light signs)
HSV_RED_LOWER1 = (0, 60, 60)
HSV_RED_UPPER1 = (15, 255, 255)
HSV_RED_LOWER2 = (165, 60, 60)
HSV_RED_UPPER2 = (180, 255, 255)
HSV_BLUE_LOWER = (95, 60, 60)
HSV_BLUE_UPPER = (135, 255, 255)
HSV_YELLOW_LOWER = (18, 60, 60)
HSV_YELLOW_UPPER = (35, 255, 255)
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 100000
