"""Axis-aligned rectangular crop regions (the historical SquareMarks name)."""

import numpy as np

from ._geometry import shape2
from .debugger import Debugger
from .marks_land import MarksLand
from .quad_marks import QuadMarks


class SquareMarks(QuadMarks):
    """Rectangle bounds: minX/maxX are rows; minY/maxY are columns.

    Bounds are pixel coordinates with exclusive upper crop bounds. ``relative``
    converts fractions into pixels. Bounds may extend outside the canvas.
    """

    def __init__(
        self,
        minX=None,
        maxX=None,
        minY=None,
        maxY=None,
        shape=None,
        zoom_factor=None,
        center=None,
        debugger=None,
    ):
        MarksLand.__init__(self, shape=shape, debugger=debugger)
        if all(v is None for v in (minX, maxX, minY, maxY, shape, zoom_factor, center)):
            return
        bounds = (minX, maxX, minY, maxY)
        if zoom_factor is not None:
            if any(v is not None for v in bounds):
                raise ValueError("Use either explicit bounds or zoom_factor.")
            h, w = shape2(shape)
            if not np.isfinite(zoom_factor) or zoom_factor <= 0:
                raise ValueError("zoom_factor must be positive.")
            r, c = center if center is not None else (h / 2, w / 2)
            minX, maxX = r - h / (2 * zoom_factor), r + h / (2 * zoom_factor)
            minY, maxY = c - w / (2 * zoom_factor), c + w / (2 * zoom_factor)
        if maxX is None or maxY is None:
            h, w = shape2(shape)
            maxX = h if maxX is None else maxX
            maxY = w if maxY is None else maxY
        self.set_marks(0 if minX is None else minX, maxX, 0 if minY is None else minY, maxY, shape)

    @staticmethod
    def relative(
        minX=None,
        maxX=None,
        minY=None,
        maxY=None,
        shape=None,
        zoom_factor=None,
        center=None,
        debugger=None,
    ):
        """Build from fractional bounds or a relative zoom center."""
        h, w = shape2(shape)
        if zoom_factor is not None:
            if any(v is not None for v in (minX, maxX, minY, maxY)):
                raise ValueError("Use either explicit bounds or zoom_factor.")
            center = (0.5, 0.5) if center is None else center
            return SquareMarks(
                shape=shape,
                zoom_factor=zoom_factor,
                center=(center[0] * h, center[1] * w),
                debugger=debugger,
            )
        return SquareMarks(
            (0 if minX is None else minX) * h,
            (1 if maxX is None else maxX) * h,
            (0 if minY is None else minY) * w,
            (1 if maxY is None else maxY) * w,
            shape,
            debugger=debugger,
        )

    @Debugger.method_decorator("SquareMarks.set_marks")
    def set_marks(self, minX, maxX, minY, maxY, shape=None):
        """Set rectangle bounds; zero-area rectangles are valid for geometry."""
        if not np.isfinite([minX, maxX, minY, maxY]).all() or minX > maxX or minY > maxY:
            raise ValueError("Bounds must be finite and ordered min <= max.")
        return MarksLand.set_marks(
            self, [[minX, minY], [minX, maxY], [maxX, maxY], [maxX, minY]], shape=shape
        )

    @Debugger.method_decorator("SquareMarks.expand")
    def expand(self, value):
        """Scale rectangle extents around its center by a positive factor."""
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Expansion factor must be positive.")
        dr, dc = (
            (self.maxX() - self.minX()) * (value - 1) / 2,
            (self.maxY() - self.minY()) * (value - 1) / 2,
        )
        return self.set_marks(
            self.minX() - dr, self.maxX() + dr, self.minY() - dc, self.maxY() + dc
        )

    @Debugger.method_decorator("SquareMarks.rotate")
    def rotate(self, angle, center=None, save_trace=False):
        """Reject rotations; convert to QuadMarks with ``to_quad()`` first."""
        raise TypeError("Convert to QuadMarks with to_quad() before rotating.")

    @Debugger.method_decorator("SquareMarks.to_quad")
    def to_quad(self, copy_trace=False):
        """Return an independent general quadrilateral."""
        return self.copy(copy_trace, copy_type=QuadMarks)

    @Debugger.method_decorator("SquareMarks.minX")
    def minX(self):
        """Minimum row coordinate."""
        return float(self.get_marks()[:, 0].min())

    @Debugger.method_decorator("SquareMarks.maxX")
    def maxX(self):
        """Maximum row coordinate."""
        return float(self.get_marks()[:, 0].max())

    @Debugger.method_decorator("SquareMarks.minY")
    def minY(self):
        """Minimum column coordinate."""
        return float(self.get_marks()[:, 1].min())

    @Debugger.method_decorator("SquareMarks.maxY")
    def maxY(self):
        """Maximum column coordinate."""
        return float(self.get_marks()[:, 1].max())

    @Debugger.method_decorator("SquareMarks.area")
    def area(self):
        """Rectangle area in squared pixels."""
        return (self.maxX() - self.minX()) * (self.maxY() - self.minY())

    @Debugger.method_decorator("SquareMarks.subscription")
    def subscription(self, other):
        """Return rectangle intersection, or None if disjoint (legacy name)."""
        if self.shape != other.shape:
            other = other.copy().reshape(self.shape)
        r0, r1 = max(self.minX(), other.minX()), min(self.maxX(), other.maxX())
        c0, c1 = max(self.minY(), other.minY()), min(self.maxY(), other.maxY())
        return (
            SquareMarks(r0, r1, c0, c1, self.shape, debugger=self._debugger)
            if r0 < r1 and c0 < c1
            else None
        )
