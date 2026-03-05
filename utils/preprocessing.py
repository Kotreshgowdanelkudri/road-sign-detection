"""
Preprocessing utilities for traffic sign images
"""
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
from config import IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, TEST_SPLIT


def enhance_image_for_detection(image):
    """
    Enhance image to improve detection on blurry or low-contrast images.
    Applies sharpening and contrast enhancement (CLAHE).
    Args:
        image: numpy array (BGR)
    Returns: enhanced image
    """
    try:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Mild sharpen for deblurring (unsharp mask style)
        blurred = cv2.GaussianBlur(image, (3, 3), 1.0)
        image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        image = np.clip(image, 0, 255).astype(np.uint8)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Image enhancement skipped: {e}")
    return image


def preprocess_image(image, target_size=IMAGE_SIZE):
    """
    Preprocess a single image for model input
    Args:
        image: numpy array (BGR format from OpenCV)
        target_size: target size for resizing (default: 64)
    Returns: preprocessed image (RGB, normalized, resized)
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    
    return image


def load_dataset_from_folders(dataset_path, target_size=IMAGE_SIZE):
    """
    Load dataset from folder structure (dataset/0/, dataset/1/, etc.)
    Args:
        dataset_path: path to dataset directory
        target_size: image size
    Returns: X (images), y (labels), num_classes
    """
    images = []
    labels = []
    
    # Get all class folders
    class_folders = sorted([d for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d))])
    
    # Filter numeric folders only
    class_folders = [d for d in class_folders if d.isdigit()]
    class_folders = sorted(class_folders, key=lambda x: int(x))
    
    print(f"Found {len(class_folders)} classes")
    
    for class_folder in class_folders:
        class_id = int(class_folder)
        class_path = os.path.join(dataset_path, class_folder)
        
        # Get all images in this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm'))]
        
        print(f"Loading class {class_id}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Preprocess
                img = preprocess_image(img, target_size)
                
                images.append(img)
                labels.append(class_id)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    X = np.array(images)
    y = np.array(labels)
    num_classes = len(class_folders)
    
    print(f"Loaded {len(X)} images from {num_classes} classes")
    
    return X, y, num_classes


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """
    Create data generators with augmentation for training
    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        batch_size: batch size
    Returns: train_generator, val_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Traffic signs should not be flipped
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Fit generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    return train_generator, val_generator


def split_dataset(X, y, val_split=VALIDATION_SPLIT, test_split=TEST_SPLIT):
    """
    Split dataset into train, validation, and test sets
    Args:
        X: images
        y: labels
        val_split: validation split ratio
        test_split: test split ratio
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    
    # Second split: separate train and validation
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
    Convert labels to one-hot encoding
    Args:
        y: label array
        num_classes: number of classes
    Returns: one-hot encoded labels
    """
    return to_categorical(y, num_classes)


def preprocess_frame_for_inference(frame, target_size=IMAGE_SIZE, enhance=False):
    """
    Preprocess a video frame for model inference
    Same preprocessing as training
    Args:
        frame: video frame (BGR from OpenCV)
        target_size: target size
        enhance: if True, apply sharpening/contrast for blurry images
    Returns: preprocessed frame ready for model
    """
    if enhance:
        frame = enhance_image_for_detection(frame.copy())
    processed = preprocess_image(frame, target_size)
    # Add batch dimension
    processed = np.expand_dims(processed, axis=0)
    return processed


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing utilities...")
    
    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test preprocessing
    processed = preprocess_image(dummy_img, 64)
    print(f"Original shape: {dummy_img.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test frame preprocessing
    frame_processed = preprocess_frame_for_inference(dummy_img, 64)
    print(f"Frame processed shape: {frame_processed.shape}")
