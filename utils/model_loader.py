"""
Simple model loader that uses native Keras loading.
Bypasses the custom architecture rebuilding which was failing on the cloud.
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten

def _make_keras_layer_compatible(layer_class):
    """
    Creates a wrapper class for keras layers to ignore TF 2.16+ keyword arguments 
    like 'quantization_config' when run on older TF versions (like 2.15 on Render).
    """
    class CompatibleLayer(layer_class):
        def __init__(self, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(**kwargs)
    return CompatibleLayer

def load_model_for_inference(model_path):
    """
    Safely load the H5 model file for inference.
    Since we pinned tensorflow-cpu>=2.12,<2.16, native loading is safer
    than manually rebuilding the layers.
    """
    print(f"Loading native Keras model from {model_path}...")
    
    # Try loading without custom objects first
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        print(f"Native loading failed: {e}")
        print("Trying with compatibility wrappers...")
        
        # Fallback to compatibility wrappers
        custom_objects = {
            'Dense': _make_keras_layer_compatible(Dense),
            'Conv2D': _make_keras_layer_compatible(Conv2D),
            'MaxPooling2D': _make_keras_layer_compatible(MaxPooling2D),
            'BatchNormalization': _make_keras_layer_compatible(BatchNormalization),
            'Dropout': _make_keras_layer_compatible(Dropout),
            'Flatten': _make_keras_layer_compatible(Flatten)
        }
        
        return load_model(model_path, custom_objects=custom_objects, compile=False)
