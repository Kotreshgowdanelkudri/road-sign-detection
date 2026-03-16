"""
Model loader that handles both Keras 2 (TF < 2.16) and Keras 3 (TF >= 2.16).

TF 2.16+ uses Keras 3, which changed the .h5 format and dropped support for
the old 'quantization_config' layer keyword argument workaround used in Keras 2.
This loader auto-detects the Keras version and uses the correct loading strategy.
"""
import os
import sys

def load_model_for_inference(model_path):
    """
    Safely load an H5 model file regardless of Keras/TF version.

    Strategy:
    - Keras 3 (TF >= 2.16): use keras.saving.load_model() which is the new API
    - Keras 2 (TF < 2.16): use the custom_objects workaround for quantization_config
    """
    import tensorflow as tf

    tf_version = tuple(int(x) for x in tf.__version__.split(".")[:2])
    print(f"Loading model with TensorFlow {tf.__version__}...")

    # ── Keras 3 path (TF 2.16+) ──────────────────────────────────────────────
    if tf_version >= (2, 16):
        try:
            # Keras 3 prefers its own load_model entry point
            import keras
            model = keras.saving.load_model(model_path, compile=False)
            print(f"Model loaded successfully (Keras 3 path) from {model_path}")
            return model
        except Exception as e:
            print(f"Keras 3 direct load failed ({e}), trying tf.keras fallback...")
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                print(f"Model loaded successfully (tf.keras fallback) from {model_path}")
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Could not load model from {model_path}.\n"
                    f"Keras 3 error: {e}\n"
                    f"tf.keras error: {e2}"
                ) from e2

    # ── Keras 2 path (TF < 2.16) ─────────────────────────────────────────────
    else:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.layers import (
            Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
        )

        def _compat_layer(layer_class):
            """Strips unknown kwargs like 'quantization_config' on old TF."""
            class CompatLayer(layer_class):
                def __init__(self, **kwargs):
                    kwargs.pop('quantization_config', None)
                    super().__init__(**kwargs)
            return CompatLayer

        custom_objects = {
            'Dense':             _compat_layer(Dense),
            'Conv2D':            _compat_layer(Conv2D),
            'MaxPooling2D':      _compat_layer(MaxPooling2D),
            'BatchNormalization': _compat_layer(BatchNormalization),
            'Dropout':           _compat_layer(Dropout),
            'Flatten':           _compat_layer(Flatten),
        }

        try:
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            print(f"Model loaded successfully (Keras 2 path) from {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Could not load model from {model_path}. Reason: {e}"
            ) from e
