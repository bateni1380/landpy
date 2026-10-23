"""Face segmentation mapped back to the original canvas."""

import numpy as np

from .. import HeatLand, ImageLand
from ..tf_lite_model import TFLiteModel
from .config import MODEL_SHA256, MODEL_URLS


class FaceSegmentationDetector(TFLiteModel):
    """Predict a face mask without modifying the supplied image."""

    def __init__(self, debugger=None, **kwargs):
        if kwargs.get("model_path") is None:
            kwargs.setdefault("sha256", MODEL_SHA256["face_segmentation_detector"])
        super().__init__(
            MODEL_URLS["face_segmentation_detector"],
            input_type=np.ndarray,
            debugger=debugger,
            **kwargs,
        )

    def detect_face_segmentation(self, input):
        """Return a HeatLand resized to the input image's spatial dimensions."""
        if not isinstance(input, ImageLand):
            raise TypeError("input must be ImageLand.")
        detail = self._interpreter.get_input_details()[0]
        h, w = detail["shape"][1:3]
        data = input.copy().to_rgb().reshape((h, w)).get_image()[None].astype(np.float32) / 255
        output = self.inference(data)
        if isinstance(output, list):
            output = output[0]
        heat = np.asarray(output).squeeze()
        if heat.ndim != 2:
            raise ValueError("Expected a single-channel segmentation output.")
        return HeatLand(np.clip(heat, 0, 1), debugger=self._debugger).reshape(input.shape)
