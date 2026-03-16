"""
Model loader compatible with both Keras 2 (TF < 2.16) and Keras 3 (TF >= 2.16).

Root cause of the Render error:
  The model.h5 was saved with quantization_config=None in each layer's config.
  Keras 3 calls Layer.from_config(config) during deserialization — if from_config
  passes unknown kwargs through to __init__, it raises ValueError/TypeError.

Fix:
  Override from_config() on every layer class to strip quantization_config
  before calling the real from_config. This works for both Keras 2 and 3.
"""


def _make_compat_layer(layer_class):
    """
    Wraps a Keras layer class so that both __init__ and from_config
    silently drop the 'quantization_config' key that older TF versions
    baked into saved model configs.
    """
    class CompatLayer(layer_class):

        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            # Keras 3 calls from_config — strip the bad key here
            config = dict(config)
            config.pop('quantization_config', None)
            return super().from_config(config)

    # Keep the original class name so Keras can look it up by name
    CompatLayer.__name__ = layer_class.__name__
    CompatLayer.__qualname__ = layer_class.__qualname__
    return CompatLayer


def load_model_for_inference(model_path):
    """
    Load the .h5 model regardless of TensorFlow / Keras version.
    Uses custom_objects wrappers that strip quantization_config on both
    Keras 2 (TF < 2.16) and Keras 3 (TF >= 2.16).
    """
    import tensorflow as tf
    print(f"Loading model with TensorFlow {tf.__version__}…")

    tf_version = tuple(int(x) for x in tf.__version__.split(".")[:2])

    # ── Keras 3 path (TF >= 2.16) ────────────────────────────────────────────
    if tf_version >= (2, 16):
        import keras
        from keras.layers import (
            Dense, Conv2D, MaxPooling2D, BatchNormalization,
            Dropout, Flatten, InputLayer
        )
        custom_objects = {
            'Dense':             _make_compat_layer(Dense),
            'Conv2D':            _make_compat_layer(Conv2D),
            'MaxPooling2D':      _make_compat_layer(MaxPooling2D),
            'BatchNormalization': _make_compat_layer(BatchNormalization),
            'Dropout':           _make_compat_layer(Dropout),
            'Flatten':           _make_compat_layer(Flatten),
            'InputLayer':        _make_compat_layer(InputLayer),
        }
        try:
            model = keras.saving.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print(f"Model loaded (Keras 3 + compat wrappers) from {model_path}")
            return model
        except Exception as e:
            print(f"Keras 3 load failed: {e}")
            # Final fallback: tf.keras API
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
                print(f"Model loaded (tf.keras fallback) from {model_path}")
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Could not load model from {model_path}.\n"
                    f"Keras 3 error: {e}\ntf.keras error: {e2}"
                ) from e2

    # ── Keras 2 path (TF < 2.16) ─────────────────────────────────────────────
    else:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.layers import (
            Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
        )
        custom_objects = {
            'Dense':             _make_compat_layer(Dense),
            'Conv2D':            _make_compat_layer(Conv2D),
            'MaxPooling2D':      _make_compat_layer(MaxPooling2D),
            'BatchNormalization': _make_compat_layer(BatchNormalization),
            'Dropout':           _make_compat_layer(Dropout),
            'Flatten':           _make_compat_layer(Flatten),
        }
        try:
            model = load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print(f"Model loaded (Keras 2 + compat wrappers) from {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Could not load model from {model_path}. Reason: {e}"
            ) from e
