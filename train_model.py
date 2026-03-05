"""
Train CNN model for traffic sign classification
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# TensorFlow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Local imports
from config import (
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH, DATASET_PATH,
    VALIDATION_SPLIT, TEST_SPLIT
)
from utils.preprocessing import (
    load_dataset_from_folders, split_dataset, encode_labels,
    create_data_generators
)
from utils.label_mapping import (
    load_label_mapping, save_label_mapping, discover_dataset_structure,
    create_default_mapping, validate_mapping
)


def build_cnn_model(input_shape, num_classes):
    """
    Build CNN architecture optimized for traffic sign classification
    Args:
        input_shape: input image shape (height, width, channels)
        num_classes: number of output classes
    Returns: compiled Keras model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history
    Args:
        history: Keras training history
        save_path: path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    Args:
        y_true: true labels
        y_pred: predicted labels
        num_classes: number of classes
        save_path: path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(max(10, num_classes // 2), max(8, num_classes // 2)))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    """
    Main training function
    """
    print("=" * 60)
    print("Traffic Sign Classification - Model Training")
    print("=" * 60)
    
    # Step 1: Discover dataset structure
    print("\n[1/7] Discovering dataset structure...")
    dataset_info = discover_dataset_structure(DATASET_PATH)
    
    if not dataset_info['exists']:
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Please ensure your dataset is in the correct location.")
        sys.exit(1)
    
    if dataset_info['structure'] == 'flat':
        print("ERROR: Dataset has flat structure (all images in one folder)")
        print("Please organize images into class folders (0/, 1/, 2/, etc.)")
        sys.exit(1)
    
    num_classes = dataset_info['num_classes']
    print(f"Found {num_classes} classes in dataset")
    print(f"Classes: {dataset_info['classes'][:10]}..." if num_classes > 10 else f"Classes: {dataset_info['classes']}")
    
    # Step 2: Load or create label mapping
    print("\n[2/7] Loading label mapping...")
    label_mapping = load_label_mapping()
    
    if not label_mapping or len(label_mapping) != num_classes:
        print("Creating default label mapping...")
        label_mapping = create_default_mapping(num_classes)
        save_label_mapping(label_mapping)
    
    is_valid = validate_mapping(label_mapping, num_classes)
    if not is_valid:
        print("WARNING: Label mapping validation failed")
    
    # Show sample mappings
    print("\nSample label mappings:")
    for i in list(label_mapping.keys())[:5]:
        print(f"  Class {i}: {label_mapping[i]}")
    
    # Step 3: Load dataset
    print("\n[3/7] Loading dataset...")
    print(f"This may take several minutes for large datasets...")
    X, y, _ = load_dataset_from_folders(DATASET_PATH, IMAGE_SIZE)
    
    if len(X) == 0:
        print("ERROR: No images loaded from dataset")
        sys.exit(1)
    
    print(f"Loaded {len(X)} images")
    print(f"Image shape: {X[0].shape}")
    print(f"Label range: {y.min()} to {y.max()}")
    
    # Step 4: Split dataset
    print("\n[4/7] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    # Encode labels
    y_train_encoded = encode_labels(y_train, num_classes)
    y_val_encoded = encode_labels(y_val, num_classes)
    y_test_encoded = encode_labels(y_test, num_classes)
    
    # Step 5: Build model
    print("\n[5/7] Building CNN model...")
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    model = build_cnn_model(input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Step 6: Train model
    print("\n[6/7] Training model...")
    
    # Create data generators
    train_gen, val_gen = create_data_generators(
        X_train, y_train_encoded,
        X_val, y_val_encoded,
        BATCH_SIZE
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Step 7: Evaluate model
    print("\n[7/7] Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    target_names = [label_mapping.get(i, f"Class {i}") for i in range(num_classes)]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, num_classes)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
