"""
Test script to diagnose model loading issues
"""
import os
import sys

print("=" * 70)
print("MODEL LOADING DIAGNOSTIC")
print("=" * 70)

# Test 1: Check if model file exists
model_path = r"e:\Kotresh Rubixe\road-sign-app\model\model.h5"
print(f"\n[1] Checking model file...")
print(f"Path: {model_path}")
print(f"Exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

# Test 2: Try loading with compile=False
print(f"\n[2] Attempting to load with compile=False...")
try:
    from tensorflow.keras.models import load_model
    model = load_model(model_path, compile=False)
    print(f"✓ SUCCESS! Model loaded")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Number of classes: {model.output_shape[1]}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try recompiling
print(f"\n[3] Attempting to recompile model...")
try:
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✓ SUCCESS! Model recompiled")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Try a simple prediction
print(f"\n[4] Testing prediction...")
try:
    import numpy as np
    dummy_input = np.random.rand(1, 64, 64, 3).astype('float32')
    prediction = model.predict(dummy_input, verbose=0)
    print(f"✓ SUCCESS! Prediction works")
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Max probability: {np.max(prediction):.4f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check label mapping
print(f"\n[5] Checking label mapping...")
try:
    import json
    mapping_path = r"e:\Kotresh Rubixe\road-sign-app\label_mapping.json"
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    print(f"✓ SUCCESS! Label mapping loaded")
    print(f"  Number of classes: {len(mapping)}")
    print(f"  First 3 classes: {list(mapping.values())[:3]}")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
