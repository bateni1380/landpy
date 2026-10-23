"""The original seven-class facial-expression model."""

import numpy as np

from .. import ImageLand
from ..tf_model import TFModel
from .config import MODEL_SHA256, MODEL_URLS


class ExpressionDetector(TFModel):
    """Predict expression scores for an already cropped face.

    Labels describe this model's output classes, not a reliable measurement
    of a person's feelings or intentions.
    """

    def __init__(self, debugger=None, **kwargs):
        if kwargs.get("model_path") is None:
            kwargs.setdefault("sha256", MODEL_SHA256["expression_detector"])
        super().__init__(
            MODEL_URLS["expression_detector"], input_type=np.ndarray, debugger=debugger, **kwargs
        )
        self.output_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def detect_expression(self, input):
        """Return (1, 7) scores; preserve source image and its history."""
        if not isinstance(input, ImageLand):
            raise TypeError("input must be ImageLand.")
        _, h, w, channels = self._interpreter.input_shape
        pixels = input.copy().to_rgb().reshape((h, w)).get_image().astype(np.float32)
        gray = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        data = gray[..., None] if channels == 1 else np.repeat(gray[..., None], channels, axis=2)
        return self.inference(data[None])
