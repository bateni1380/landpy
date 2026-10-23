"""Face mesh landmarks mapped from model space into image pixels."""

import numpy as np

from .. import ImageLand, MarksLand, SquareMarks
from ..tf_lite_model import TFLiteModel
from .config import MODEL_SHA256, MODEL_URLS


class FaceLandmarksDetector(TFLiteModel):
    """Predict a 478-point face mesh (or 468 with legacy weights) for an already cropped face."""

    def __init__(self, debugger=None, **kwargs):
        if kwargs.get("model_path") is None:
            kwargs.setdefault("sha256", MODEL_SHA256["face_landmark_detector"])
        super().__init__(
            MODEL_URLS["face_landmark_detector"], input_type=np.ndarray, debugger=debugger, **kwargs
        )

    def detect_face_landmarks(self, input):
        """Return (MarksLand, presence probability) without changing the input image."""
        if not isinstance(input, ImageLand):
            raise TypeError("input must be ImageLand.")
        _, h, w, _ = self._interpreter.get_input_details()[0]["shape"]
        side = max(input.shape)
        data = (
            input.copy()
            .to_rgb()
            .zoom(SquareMarks(shape=(side, side)))
            .reshape((h, w))
            .get_image()[None]
            .astype(np.float32)
            / 255
        )
        outputs = self.inference(data)
        mesh = next(
            np.asarray(o).reshape(-1, 3)
            for o in outputs
            if np.asarray(o).size in (468 * 3, 478 * 3)
        )
        presence = next(float(np.asarray(o).ravel()[0]) for o in outputs if np.asarray(o).size == 1)
        presence = float(1 / (1 + np.exp(-np.clip(presence, -80, 80))))
        points = np.column_stack((mesh[:, 1] / h * side, mesh[:, 0] / w * side))
        return MarksLand(points, shape=input.shape, debugger=self._debugger), presence


class FaceLandmarksIndices:
    """Original mesh index groups for face-region filters and overlays."""

    LIPS = [
        61,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        291,
        291,
        375,
        321,
        405,
        314,
        17,
        84,
        181,
        91,
        146,
    ]
    UPPER_LIPS = [
        61,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        291,
        308,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        80,
        191,
        78,
    ]
    LIWER_LIPS = [
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        78,
    ]
    LEFT_EYE = [466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362]
    RIGHT_EYE = [246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133]
    GLASSES_INDICES = [64, 294, 347, 118]

    LEFT_BLUSH = [347, 330, 266, 426, 427, 411, 376, 352]
    RIGHT_BLUSH = [118, 101, 36, 206, 207, 187, 147, 123]

    LEFT_BLUSH_RECTANGLE = [116, 100, 92, 213]
    RIGHT_BLUSH_RECTANGLE = [345, 329, 322, 433]

    LEFT_EYE_TRIANGLE = [263, 257, 362]
    RIGHT_EYE_TRIANGLE = [33, 27, 133]

    LEFT_BLUSH_SHADOW = [113, 225, 224, 223, 222, 221, 189, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_BLUSH_SHADOW = [342, 445, 444, 443, 442, 441, 413, 398, 384, 385, 386, 387, 388, 466]
    FACE_OVAL = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]

    LOWER_LIPS = LIWER_LIPS
