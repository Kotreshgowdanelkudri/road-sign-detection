"""
Preprocessing utilities for traffic sign images
These functions prepare images before passing them to the CNN model
"""
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
from config import IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, TEST_SPLIT


def enhance_image_for_detection(image):
    """
    Improves the quality of blurry or dark images before detection.
    This helps the model recognize signs that are hard to see clearly.
    
    Steps:
      1. CLAHE — improves local contrast (makes details more visible)
      2. Unsharp mask — sharpens the image slightly
    """
    try:
        # Convert from BGR (OpenCV default) to LAB color space
        # LAB separates brightness (L) from color (A, B) — easier to enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE only to the brightness channel
        # CLAHE = Contrast Limited Adaptive Histogram Equalization
        # It brightens dark areas without blowing out bright areas
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Put the channels back together and convert back to BGR
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Sharpen the image using "unsharp masking" technique:
        # Blend the original with a blurred version — the difference adds sharpness
        blurred = cv2.GaussianBlur(image, (3, 3), 1.0)
        image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        # Make sure pixel values stay in valid 0-255 range after blending
        image = np.clip(image, 0, 255).astype(np.uint8)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Image enhancement skipped: {e}")
    return image


def preprocess_image(image, target_size=IMAGE_SIZE):
    """
    Prepares a single image to be fed into the CNN model.
    
    The model was trained on 64x64 RGB images with pixel values between 0 and 1,
    so we need to make sure every image matches that format before predicting.
    """
    # OpenCV reads images as BGR but our model was trained on RGB
    # Swap the channels so colors are correct
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to 64x64 — the size the model expects
    image = cv2.resize(image, (target_size, target_size))

    # Scale pixel values from 0–255 down to 0.0–1.0
    # Neural networks train better with small values
    image = image.astype('float32') / 255.0

    return image


def load_dataset_from_folders(dataset_path, target_size=IMAGE_SIZE):
    """
    Reads all training images from the dataset folder.
    
    The dataset must be organized like:
      dataset/0/image1.png   ← class 0
      dataset/1/image2.png   ← class 1
      ...
    
    Returns:
        X — all images as a numpy array
        y — corresponding class labels (numbers)
        num_classes — how many sign categories were found
    """
    images = []
    labels = []

    # Look inside the dataset folder for subfolders named with numbers (0, 1, 2...)
    class_folders = sorted([d for d in os.listdir(dataset_path)
                            if os.path.isdir(os.path.join(dataset_path, d))])

    # Only keep numeric folder names — ignore any extra folders
    class_folders = [d for d in class_folders if d.isdigit()]
    class_folders = sorted(class_folders, key=lambda x: int(x))

    print(f"Found {len(class_folders)} classes")

    for class_folder in class_folders:
        class_id = int(class_folder)
        class_path = os.path.join(dataset_path, class_folder)

        # Get all image files in this class folder
        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm'))]

        print(f"Loading class {class_id}: {len(image_files)} images")

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip unreadable files

                # Apply the standard preprocessing (resize + normalize)
                img = preprocess_image(img, target_size)

                images.append(img)
                labels.append(class_id)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    # Convert Python lists to numpy arrays (required by Keras)
    X = np.array(images)
    y = np.array(labels)
    num_classes = len(class_folders)

    print(f"Loaded {len(X)} images from {num_classes} classes")

    return X, y, num_classes


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """
    Creates data generators for training and validation.
    
    The training generator applies random augmentations (small rotations, shifts, zoom)
    so the model sees slightly different versions of each image — this prevents overfitting
    and helps the model generalize to new images.
    
    The validation generator does NOT augment — we want clean, unmodified images to
    accurately measure how well the model is doing.
    """
    # Training augmentation settings
    # These simulate real-world variation: camera angle, distance, lighting
    train_datagen = ImageDataGenerator(
        rotation_range=15,         # Randomly rotate up to 15 degrees
        width_shift_range=0.1,     # Shift image left/right by 10%
        height_shift_range=0.1,    # Shift image up/down by 10%
        zoom_range=0.1,            # Zoom in/out by 10%
        horizontal_flip=False,     # Don't flip! "STOP" mirrored is wrong
        brightness_range=[0.8, 1.2],  # Randomly dim or brighten slightly
        fill_mode='nearest'        # Fill any empty space with nearby pixels
    )

    # No augmentation for validation — we measure what the model truly learned
    val_datagen = ImageDataGenerator()

    # Flow generates batches of images during training (doesn't load all at once)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    return train_generator, val_generator


def split_dataset(X, y, val_split=VALIDATION_SPLIT, test_split=TEST_SPLIT):
    """
    Splits the full dataset into three separate sets:
    - Training set (70%) — what the model learns from
    - Validation set (20%) — used to check accuracy during training
    - Test set (10%) — used ONLY at the end to evaluate final model performance
    
    Using stratify=y ensures each split has a similar mix of all sign categories.
    random_state=42 makes the split reproducible (same split every run).
    """
    from sklearn.model_selection import train_test_split

    # First, carve out the test set (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )

    # From the remaining 90%, split into train and validation
    # val_ratio recalculates the fraction needed from the remaining data
    val_ratio = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    print(f"Train set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")

    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_labels(y, num_classes):
    """
    Converts class numbers into one-hot encoded vectors.
    
    Example: class 2 out of 5 classes becomes [0, 0, 1, 0, 0]
    This format is required by the softmax layer in our CNN model.
    """
    return to_categorical(y, num_classes)


def preprocess_frame_for_inference(frame, target_size=IMAGE_SIZE, enhance=False):
    """
    Prepares a single video frame or image for the model to make a prediction.
    
    This is used at detection time (not training time).
    If the image is blurry or dark, we can optionally enhance it first.
    The final output has an extra dimension added (batch dimension)
    because the model always expects a batch of images, even if it's just one.
    """
    if enhance:
        # Apply quality enhancement for difficult images (blurry, dark)
        frame = enhance_image_for_detection(frame.copy())

    # Apply standard preprocessing (BGR→RGB, resize, normalize)
    processed = preprocess_image(frame, target_size)

    # Add batch dimension: shape goes from (64, 64, 3) → (1, 64, 64, 3)
    processed = np.expand_dims(processed, axis=0)
    return processed


if __name__ == "__main__":
    # Quick self-test to make sure the functions work
    print("Testing preprocessing utilities...")

    # Create a random dummy image (simulates a real image)
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Test single image preprocessing
    processed = preprocess_image(dummy_img, 64)
    print(f"Original shape: {dummy_img.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")

    # Test frame preprocessing (adds batch dimension)
    frame_processed = preprocess_frame_for_inference(dummy_img, 64)
    print(f"Frame processed shape: {frame_processed.shape}")
