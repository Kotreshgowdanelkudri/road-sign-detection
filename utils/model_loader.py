"""
Custom model loader to handle TensorFlow version incompatibility
Rebuilds the model architecture and loads weights from the trained model
"""
import os
import h5py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from config import IMAGE_SIZE

def build_model_architecture(num_classes=85):
    """
    Build the exact same architecture as the trained model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_model_weights(model, weights_path):
    """
    Load weights from H5 file into model
    """
    try:
        model.load_weights(weights_path)
        return True
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False

def load_compatible_model(model_path, num_classes=85):
    """
    Load model in a TensorFlow version-compatible way
    """
    print(f"Building model architecture for {num_classes} classes...")
    model = build_model_architecture(num_classes)
    
    print(f"Loading weights from {model_path}...")
    if load_model_weights(model, model_path):
        print("✓ Weights loaded successfully")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model compiled successfully")
        
        return model
    else:
        return None

if __name__ == "__main__":
    from config import MODEL_PATH
    from utils.label_mapping import load_label_mapping
    mapping = load_label_mapping()
    num_classes = len(mapping) or 85
    model = load_compatible_model(MODEL_PATH, num_classes=num_classes)
    
    if model:
        print(f"\n✓ Model loaded successfully!")
        print(f"  Output shape: {model.output_shape}")
        
        # Test prediction
        import numpy as np
        dummy_input = np.random.rand(1, 64, 64, 3).astype('float32')
        prediction = model.predict(dummy_input, verbose=0)
        print(f"  Test prediction shape: {prediction.shape}")
        print(f"  Test prediction sum: {np.sum(prediction):.4f} (should be ~1.0)")
    else:
        print("\n✗ Model loading failed")
