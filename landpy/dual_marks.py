"""Two-point vectors on an image canvas."""

import numpy as np

from .debugger import Debugger
from .marks_land import MarksLand


class DualMarks(MarksLand):
    """A directed segment from start_point to end_point in (row, column)."""

    def __init__(self, start_point=(0.0, 0.0), end_point=(1.0, 1.0), shape=None, debugger=None):
        super().__init__(shape=shape, debugger=debugger)
        self.set_marks(start_point, end_point)

    @Debugger.method_decorator("DualMarks.set_marks")
    def set_marks(self, start_point, end_point, debugger=None):
        """Set endpoints from coordinate pairs or one-point MarksLand objects."""

        def point(value):
            if isinstance(value, MarksLand):
                if len(value.get_marks()) != 1:
                    raise ValueError("Endpoint must contain exactly one point.")
                return value.get_marks()[0]
            return value

        return MarksLand.set_marks(self, [point(start_point), point(end_point)])

    @Debugger.method_decorator("DualMarks.lenght")
    def lenght(self):
        """Return Euclidean length in pixels (historical spelling)."""
        return float(np.linalg.norm(self.get_marks()[1] - self.get_marks()[0]))

    length = lenght

    @Debugger.method_decorator("DualMarks.angle")
    def angle(self):
        """Return degrees counterclockwise from the rightward horizontal axis."""
        dr, dc = self.get_marks()[1] - self.get_marks()[0]
        return float(np.degrees(np.arctan2(-dr, dc)))

    @Debugger.method_decorator("DualMarks.show")
    def show(self):
        """Plot the vector and return this object."""
        Debugger(visual_debug=True).log_image(self._flow_data, type=Debugger.LogTypes.VECTOR)
        return self
