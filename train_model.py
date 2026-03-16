"""
Train CNN model for traffic sign classification

This script:
 1. Reads all training images from the dataset folder
 2. Builds a CNN (Convolutional Neural Network) model
 3. Trains the model to recognize traffic signs
 4. Saves the best model to disk
 5. Shows accuracy results and a confusion matrix
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# TensorFlow/Keras imports for building the neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Our own modules for configuration, data loading, and label handling
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
    Builds the CNN (Convolutional Neural Network) architecture.
    
    A CNN works in two stages:
    
    FEATURE EXTRACTION (Conv + Pool layers):
      - It looks at small patches of the image and learns to detect shapes,
        edges, textures, and eventually complex patterns like sign borders or symbols.
      - MaxPooling reduces the image size after each block, making the network
        faster and focusing on the most important features.
      - BatchNormalization keeps the training stable by normalizing outputs.
      - Dropout randomly turns off neurons during training — this prevents the
        model from memorizing the training data (overfitting).
    
    CLASSIFICATION (Dense layers):
      - After extracting features, the model flattens everything into a 1D list
        and passes it through fully connected layers to make the final decision.
      - The last layer has one output per class and uses softmax to give
        probabilities (they all add up to 1.0).
    
    Architecture summary:
      - Block 1: 2x Conv2D(32 filters) → BatchNorm → MaxPool → Dropout
      - Block 2: 2x Conv2D(64 filters) → BatchNorm → MaxPool → Dropout
      - Block 3: 2x Conv2D(128 filters) → BatchNorm → MaxPool → Dropout
      - Dense 512 → Dense 256 → Dense (num_classes with softmax)
    """
    model = Sequential([
        # ── First Convolutional Block ──────────────────────────────────────────
        # 32 filters of size 3x3 — each filter learns to detect a different pattern
        # padding='same' keeps the output the same size as input
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),  # Normalize to prevent training instability
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),  # Halve the spatial dimensions (e.g. 64→32)
        Dropout(0.25),  # Randomly drop 25% of connections to prevent overfitting

        # ── Second Convolutional Block ─────────────────────────────────────────
        # 64 filters — learning more complex features than block 1
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),  # Further reduce size (32→16)
        Dropout(0.25),

        # ── Third Convolutional Block ──────────────────────────────────────────
        # 128 filters — high-level features like overall sign shape and symbols
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),  # Further reduce size (16→8)
        Dropout(0.25),

        # ── Classification Layers ──────────────────────────────────────────────
        # Flatten converts the 3D feature map to a 1D vector for the Dense layers
        Flatten(),

        # Two fully connected layers to combine all the extracted features
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),  # Higher dropout here — these layers are prone to overfitting
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # Output layer: one neuron per class
        # Softmax converts raw scores into probabilities that sum to 1
        Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history, save_path='training_history.png'):
    """
    Saves two graphs showing how the model improved during training:
    - Left chart: Accuracy on training data vs validation data over epochs
    - Right chart: Loss (error) on training vs validation over epochs
    
    If the validation line diverges far from training, the model is overfitting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy chart
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss chart
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
    Saves a heatmap showing which classes the model confuses with each other.
    
    Rows = actual label, Columns = predicted label
    Bright diagonal = model is correct
    Off-diagonal bright spots = the model is confusing two specific classes
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
    Runs the full training pipeline step by step.
    Each step is numbered so you can track progress in the terminal.
    """
    print("=" * 60)
    print("Traffic Sign Classification - Model Training")
    print("=" * 60)

    # ── Step 1: Check the dataset ──────────────────────────────────────────────
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

    # ── Step 2: Load or create class name mapping ──────────────────────────────
    # label_mapping maps class numbers to human-readable names
    # e.g. {0: "Speed Limit 20", 1: "Speed Limit 30", ...}
    print("\n[2/7] Loading label mapping...")
    label_mapping = load_label_mapping()

    if not label_mapping or len(label_mapping) != num_classes:
        print("Creating default label mapping...")
        label_mapping = create_default_mapping(num_classes)
        save_label_mapping(label_mapping)

    is_valid = validate_mapping(label_mapping, num_classes)
    if not is_valid:
        print("WARNING: Label mapping validation failed")

    # Show a few examples so the user can verify the mapping looks right
    print("\nSample label mappings:")
    for i in list(label_mapping.keys())[:5]:
        print(f"  Class {i}: {label_mapping[i]}")

    # ── Step 3: Load all images from disk ─────────────────────────────────────
    print("\n[3/7] Loading dataset...")
    print(f"This may take several minutes for large datasets...")
    X, y, _ = load_dataset_from_folders(DATASET_PATH, IMAGE_SIZE)

    if len(X) == 0:
        print("ERROR: No images loaded from dataset")
        sys.exit(1)

    print(f"Loaded {len(X)} images")
    print(f"Image shape: {X[0].shape}")
    print(f"Label range: {y.min()} to {y.max()}")

    # ── Step 4: Split data into train / validation / test ─────────────────────
    print("\n[4/7] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # Convert integer class labels to one-hot vectors for the softmax output layer
    y_train_encoded = encode_labels(y_train, num_classes)
    y_val_encoded = encode_labels(y_val, num_classes)
    y_test_encoded = encode_labels(y_test, num_classes)

    # ── Step 5: Build the CNN model ────────────────────────────────────────────
    print("\n[5/7] Building CNN model...")
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)  # 64x64 RGB image
    model = build_cnn_model(input_shape, num_classes)

    # Compile the model — set the optimizer, loss function, and metric to track
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Adam adjusts learning rate automatically
        loss='categorical_crossentropy',       # Standard loss for multi-class problems
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    model.summary()

    # ── Step 6: Train the model ────────────────────────────────────────────────
    print("\n[6/7] Training model...")

    # Data generators apply augmentation and feed images in batches
    train_gen, val_gen = create_data_generators(
        X_train, y_train_encoded,
        X_val, y_val_encoded,
        BATCH_SIZE
    )

    # Callbacks automatically handle common training tasks:
    callbacks = [
        # Save the model file only when validation accuracy improves
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,  # Only overwrite the file if this epoch is the best
            mode='max',
            verbose=1
        ),
        # Stop training automatically if validation loss doesn't improve for 10 epochs
        # This prevents wasting time and prevents overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,  # Go back to the best weights when stopping
            verbose=1
        ),
        # Cut the learning rate in half if validation loss stalls for 5 epochs
        # This helps the model fine-tune more carefully in later training
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Start training! This is where the model actually learns from the data
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Visualize how the model's accuracy and loss changed over time
    plot_training_history(history)

    # ── Step 7: Evaluate on the untouched test set ─────────────────────────────
    # The test set was never seen during training or validation
    # This gives us the most honest measure of real-world performance
    print("\n[7/7] Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Run predictions on the test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Pick the class with highest probability

    # Print per-class accuracy stats (precision, recall, F1)
    print("\nClassification Report:")
    target_names = [label_mapping.get(i, f"Class {i}") for i in range(num_classes)]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Save the confusion matrix image
    plot_confusion_matrix(y_test, y_pred, num_classes)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
