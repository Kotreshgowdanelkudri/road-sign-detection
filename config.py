"""
Configuration settings for Road Sign Detection Application
All paths support environment variable overrides.
"""
import os

# This gives us the folder where this file is located — the root of our project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# MODEL SETTINGS
# ──────────────────────────────────────────────

# Where the trained model file (.h5) is saved/loaded from
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.h5')

# All images will be resized to 64x64 before going into the model
# Smaller = faster, larger = more detail but needs more memory
IMAGE_SIZE = 64

# Only show a detection if the model is at least 50% confident
# Below this = we still show a "?" guess so the user sees something
CONFIDENCE_THRESHOLD = 0.50

# If confidence is at least 10%, show a "best guess" even if it's uncertain
# This helps when images are blurry or the sign is partially visible
MIN_DISPLAY_CONFIDENCE = 0.10

# ──────────────────────────────────────────────
# UPLOAD SETTINGS
# ──────────────────────────────────────────────

# Folder where uploaded images and videos are stored on disk
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# Maximum file size for uploads — 200 MB by default
# Can be changed with an environment variable MAX_UPLOAD_SIZE
MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_SIZE', 200 * 1024 * 1024))

# Which image file types are accepted for upload
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Which video file types are accepted for upload
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# ──────────────────────────────────────────────
# DATASET SETTINGS
# ──────────────────────────────────────────────

# Where the training images are stored — can be overridden with DATASET_PATH env variable
DATASET_PATH = os.environ.get('DATASET_PATH', os.path.join(BASE_DIR, 'dataset', 'Train'))

# JSON file that maps class numbers (0, 1, 2...) to sign names ("Stop Sign", "Speed Limit"...)
LABEL_MAPPING_PATH = os.path.join(BASE_DIR, 'label_mapping.json')

# ──────────────────────────────────────────────
# PERFORMANCE SETTINGS (webcam / live camera)
# ──────────────────────────────────────────────

# We don't process every frame — only every 5th frame
# This makes live detection much faster on regular computers
FRAME_SKIP = 5

# Resize all frames to 640x480 before processing — balances speed and quality
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ──────────────────────────────────────────────
# VOICE ALERT SETTINGS
# ──────────────────────────────────────────────

# How fast the computer reads the sign name aloud (words per minute)
VOICE_RATE = 150

# Volume of the voice — 1.0 is maximum
VOICE_VOLUME = 1.0

# ──────────────────────────────────────────────
# TRAINING SETTINGS
# ──────────────────────────────────────────────

# How many images the model looks at in one go during training
BATCH_SIZE = 32

# How many times the model goes through the entire dataset during training
# EarlyStopping will stop it before 50 if the model stops improving
EPOCHS = 50

# 20% of data set aside for validation (checking accuracy while training)
VALIDATION_SPLIT = 0.2

# 10% of data set aside for final testing (only used after training is done)
TEST_SPLIT = 0.1

# ──────────────────────────────────────────────
# COLOR DETECTION RANGES (in HSV color space)
# ──────────────────────────────────────────────
# We detect signs by their colors — red, blue, yellow
# HSV = Hue, Saturation, Value — better than RGB for color detection under lighting changes

# Red wraps around in HSV so we need two ranges to catch all shades of red
HSV_RED_LOWER1 = (0, 60, 60)
HSV_RED_UPPER1 = (15, 255, 255)
HSV_RED_LOWER2 = (165, 60, 60)
HSV_RED_UPPER2 = (180, 255, 255)

# Blue range — covers most blue traffic signs
HSV_BLUE_LOWER = (95, 60, 60)
HSV_BLUE_UPPER = (135, 255, 255)

# Yellow range — warning/caution signs
HSV_YELLOW_LOWER = (18, 60, 60)
HSV_YELLOW_UPPER = (35, 255, 255)

# Minimum and maximum size (in pixels) of a colored region to be considered a sign
# Too small = probably noise; too large = probably background
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 100000
