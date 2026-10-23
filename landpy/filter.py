"""Color filters weighted by heat maps."""

import numpy as np


class Filter:
    """Blend RGB pixels toward a color with strength ``heat * ratio``.

    Colors are RGB triples in [0, 255] or six-digit hexadecimal strings.
    """

    def __init__(self, color=(255, 255, 255), ratio=1, debugger=None):
        self._debugger = debugger
        self.color = np.asarray(
            self.hex_to_rgb(color) if isinstance(color, str) else color, dtype=float
        )
        if (
            self.color.shape != (3,)
            or not np.isfinite(self.color).all()
            or np.any((self.color < 0) | (self.color > 255))
        ):
            raise ValueError("color must contain three values in [0, 255].")
        if not np.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("ratio must be in [0, 1].")
        self.ratio = ratio
        self.apply_function = self._blend

    def _blend(self, pixels, heat):
        weight = heat * self.ratio
        return pixels * (1 - weight) + self.color * weight

    def set_function(self, function):
        """Use ``function(pixels, heat)`` for blending; return this filter."""
        if not callable(function):
            raise TypeError("function must be callable.")
        self.apply_function = function
        return self

    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert ``#RRGGBB`` or ``RRGGBB`` to an RGB tuple."""
        value = hex_color.removeprefix("#")
        if len(value) != 6:
            raise ValueError("Expected a six-digit hexadecimal color.")
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
