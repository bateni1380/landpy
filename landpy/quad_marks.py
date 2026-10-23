"""Four-point regions for perspective transformations."""

import cv2
import numpy as np

from .debugger import Debugger
from .marks_land import MarksLand


class QuadMarks(MarksLand):
    """A convex quadrilateral, ordered top-left then clockwise in image space."""

    def __init__(self, x1=None, x2=None, x3=None, x4=None, shape=None, debugger=None):
        super().__init__(shape=shape, debugger=debugger)
        points = [x1, x2, x3, x4]
        if any(p is not None for p in points):
            if any(p is None for p in points):
                raise ValueError("Provide all four corners.")
            QuadMarks.set_marks(self, *points)

    @Debugger.method_decorator("QuadMarks.set_marks")
    def set_marks(self, x1, x2, x3, x4):
        """Validate and order four distinct convex corners."""
        points = np.asarray([x1, x2, x3, x4], dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Four finite (row, column) points are required.")
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 0] - center[0], points[:, 1] - center[1])
        points = points[np.argsort(angles)]
        points = np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)
        if (
            not cv2.isContourConvex(points[:, ::-1].astype(np.float32))
            or abs(cv2.contourArea(points.astype(np.float32))) < 1e-8
        ):
            raise ValueError("Corners must form a nondegenerate convex quadrilateral.")
        return MarksLand.set_marks(self, points)

    @Debugger.method_decorator("QuadMarks.center")
    def center(self):
        """Return the intersection of the two diagonals in pixel coordinates."""
        p = self.get_marks()
        coefficients = np.column_stack((p[2] - p[0], -(p[3] - p[1])))
        t = np.linalg.solve(coefficients, p[1] - p[0])[0]
        return (p[0] + t * (p[2] - p[0])).tolist()
