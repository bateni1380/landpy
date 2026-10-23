"""Floating-point masks and their geometric transforms."""

import cv2
import numpy as np

from ._geometry import crop_geometry, inverse_warp, rotation, shape2
from .debugger import Debugger
from .land import Land


class HeatLand(Land):
    """A finite 2D heat map with values in [0, 1].

    Addition is pixelwise maximum, subtraction is pixelwise minimum (the
    original intersection convention), negation is complement. Multiplication
    and powers clip their results to [0, 1]. Binary maps require equal shapes.
    """

    def __init__(self, heat_data=None, shape=None, fill_value=0.0, debugger=None):
        super().__init__(debugger)
        if heat_data is not None:
            self.set_heat(heat_data)
        elif shape is not None:
            self.set_heat(np.full(shape2(shape), fill_value, dtype=np.float32))

    @property
    def shape(self):
        """Heat map dimensions, or None before data is set."""
        return None if self._flow_data is None else self._flow_data.shape

    @Debugger.method_decorator("HeatLand.set_heat")
    def set_heat(self, heat):
        """Copy a nonempty finite 2D array with values in [0, 1]."""
        heat = np.asarray(heat, dtype=np.float32)
        if (
            heat.ndim != 2
            or not heat.size
            or not np.isfinite(heat).all()
            or np.any((heat < 0) | (heat > 1))
        ):
            raise ValueError("heat must be a nonempty finite 2D array in [0, 1].")
        return self._set_data(heat.copy())

    @Debugger.method_decorator("HeatLand.get_heat")
    def get_heat(self):
        """Return the mutable heat array."""
        return self._flow_data

    def _combine(self, other, function):
        if not isinstance(other, HeatLand):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Heat maps must have equal shapes; reshape explicitly.")
        return HeatLand(function(self._flow_data, other._flow_data), debugger=self._debugger)

    @Debugger.method_decorator("HeatLand.__add__")
    def __add__(self, other):
        """Return pixelwise union (maximum)."""
        return self._combine(other, np.maximum)

    @Debugger.method_decorator("HeatLand.__sub__")
    def __sub__(self, other):
        """Return pixelwise intersection (minimum), not arithmetic difference."""
        return self._combine(other, np.minimum)

    @Debugger.method_decorator("HeatLand.__neg__")
    def __neg__(self):
        """Return complement (1 - heat)."""
        return HeatLand(1 - self._flow_data, debugger=self._debugger)

    @Debugger.method_decorator("HeatLand.__mul__")
    def __mul__(self, other):
        """Scale heat by a finite scalar and clip to [0, 1]."""
        if not np.isscalar(other) or not np.isfinite(other):
            raise ValueError("Multiplier must be a finite scalar.")
        return HeatLand(np.clip(other * self._flow_data, 0, 1), debugger=self._debugger)

    __rmul__ = __mul__

    @Debugger.method_decorator("HeatLand.__pow__")
    def __pow__(self, other):
        """Raise heat to a finite nonnegative power."""
        if not np.isscalar(other) or not np.isfinite(other) or other < 0:
            raise ValueError("Exponent must be finite and nonnegative.")
        return HeatLand(self._flow_data**other, debugger=self._debugger)

    @Debugger.method_decorator("Rotating heat")
    def rotate(self, angle, center=None, save_trace=False, fill_color=0):
        """Rotate counterclockwise about a pixel (row, column) center."""
        if not 0 <= fill_color <= 1:
            raise ValueError("fill_color must be in [0, 1].")
        matrix = rotation(self.shape, angle, center)
        h, w = self.shape
        result = cv2.warpAffine(self._flow_data, matrix, (w, h), borderValue=float(fill_color))
        if save_trace:
            self._make_trace(
                HeatLand._rotate_back,
                {"matrix": cv2.invertAffineTransform(matrix), "shape": self.shape},
            )
        return self._set_data(result)

    @staticmethod
    def _rotate_back(data, parameters, debugger):
        h, w = parameters["shape"]
        return cv2.warpAffine(data, parameters["matrix"], (w, h))

    @Debugger.method_decorator("HeatLand.reshape")
    def reshape(self, shape, save_trace=False):
        """Resize without quantizing to 8-bit; inverse resizing is approximate."""
        h, w = shape2(shape)
        result = cv2.resize(self._flow_data, (w, h), interpolation=cv2.INTER_LINEAR)
        if save_trace:
            self._make_trace(HeatLand._resize_back, {"shape": self.shape})
        return self._set_data(result)

    @staticmethod
    def _resize_back(data, parameters, debugger):
        h, w = parameters["shape"]
        return cv2.resize(data, (w, h), interpolation=cv2.INTER_LINEAR)

    @Debugger.method_decorator("HeatLand.zoom")
    def zoom(
        self, quadrilateral=None, save_trace=False, fill_heat=0, zoom_factor=None, center=None
    ):
        """Crop/pad a rectangle or rectify a quad; retrace pastes edits back."""
        from .square_marks import SquareMarks

        if quadrilateral is None:
            quadrilateral = SquareMarks(shape=self.shape, zoom_factor=zoom_factor, center=center)
        if not np.isfinite(fill_heat) or not 0 <= fill_heat <= 1:
            raise ValueError("fill_heat must be in [0, 1].")
        matrix, (h, w) = crop_geometry(quadrilateral)
        result = cv2.warpPerspective(self._flow_data, matrix, (w, h), borderValue=float(fill_heat))
        if save_trace:
            self._make_trace(inverse_warp, {"original": self._flow_data, "matrix": matrix})
        return self._set_data(result)

    @Debugger.method_decorator("HeatLand.show")
    def show(self):
        """Plot the heat map with a fixed [0, 1] color scale."""
        Debugger(visual_debug=True).log_image(self._flow_data, type=Debugger.LogTypes.HEAT)
        return self
