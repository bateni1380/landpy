"""Optional TensorFlow/Keras model wrapper."""

import numpy as np

from .debugger import Debugger
from .model import Model


def tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("Install landpy[tensorflow] to load TensorFlow models.") from error
    return tf


class TFModel(Model):
    """Run a saved Keras model; input/output are NumPy arrays."""

    def _load_model(self):
        tf = tensorflow()
        if self.model_format == "h5":
            try:
                import tf_keras
            except ImportError as error:
                raise ImportError(
                    "Install landpy[tensorflow] for legacy .h5 model support."
                ) from error
            return tf_keras.models.load_model(self.model_path, compile=False)
        return tf.keras.models.load_model(self.model_path, compile=False)

    @Debugger.method_decorator("Keras inference")
    def inference(self, input):
        """Validate input type and run the model in inference mode."""
        self._validate_input(input)
        output = self._interpreter(input, training=False)
        return (
            [np.asarray(value) for value in output]
            if isinstance(output, (list, tuple))
            else np.asarray(output)
        )
