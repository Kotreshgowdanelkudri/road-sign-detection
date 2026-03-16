"""
Simple model loader that uses native Keras loading.
Bypasses the custom architecture rebuilding which was failing on the cloud.
"""
from tensorflow.keras.models import load_model

def load_model_for_inference(model_path):
    """
    Safely load the H5 model file for inference.
    Since we pinned tensorflow-cpu>=2.12,<2.16, native loading is safer
    than manually rebuilding the layers.
    """
    print(f"Loading native Keras model from {model_path}...")
    # Compile=False makes loading much faster and avoids optimizer state issues
    return load_model(model_path, compile=False)
