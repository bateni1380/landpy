"""Landmarks in pixel (row, column) coordinates."""

import cv2
import numpy as np

from ._geometry import crop_geometry, rotation, shape2, transform_points
from .debugger import Debugger
from .land import Land


class MarksLand(Land):
    """A floating point (N, 2) point set on a (height, width) canvas.

    ``x`` contains points or row coordinates; optional ``y`` contains columns.
    Empty construction is allowed; set data and shape before transforming.
    """

    def __init__(self, x=None, y=None, shape=None, debugger=None):
        super().__init__(debugger)
        self._flow_data = [
            np.empty((0, 2), dtype=float),
            shape2(shape) if shape is not None else None,
        ]
        if x is not None:
            MarksLand.set_marks(self, x, y, shape)

    @property
    def shape(self):
        """Canvas dimensions as (height, width), or None if unset."""
        return self._flow_data[1]

    @Debugger.method_decorator("MarksLand.set_marks")
    def set_marks(self, x, y=None, shape=None):
        """Replace points and optionally canvas shape; copy caller data."""
        if y is not None:
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape:
                raise ValueError("Row and column arrays must have equal shapes.")
            points = np.column_stack((x.ravel(), y.ravel()))
        else:
            points = np.asarray(x, dtype=float)
            if points.size == 0:
                points = np.empty((0, 2))
            elif points.shape == (2,):
                points = points.reshape(1, 2)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("marks must be a finite (N, 2) array.")
        return self._set_data(
            [
                np.array(points, dtype=float, copy=True),
                shape2(shape) if shape is not None else self.shape,
            ]
        )

    @Debugger.method_decorator("MarksLand.get_marks")
    def get_marks(self):
        """Return the mutable (N, 2) point array."""
        return self._flow_data[0]

    @Debugger.method_decorator("MarksLand.reshape")
    def reshape(self, shape, save_trace=False):
        """Scale points to a new canvas, optionally recording its inverse."""
        old, new = shape2(self.shape), shape2(shape)
        matrix = np.diag([new[1] / old[1], new[0] / old[0], 1.0])
        return self._transform(matrix, new, save_trace)

    def _transform(self, matrix, shape, save_trace):
        old_shape = self.shape
        points = self.get_marks()
        result = transform_points(points, matrix) if len(points) else points.copy()
        if save_trace:
            self._make_trace(
                MarksLand._inverse, {"matrix": np.linalg.inv(matrix), "shape": old_shape}
            )
        self._flow_data = [result, shape]
        self._redo.clear()
        return self

    @staticmethod
    def _inverse(data, parameters, debugger):
        points = transform_points(data[0], parameters["matrix"]) if len(data[0]) else data[0].copy()
        return [points, parameters["shape"]]

    @Debugger.method_decorator("Rotating marks")
    def rotate(self, angle, center=None, save_trace=False):
        """Rotate counterclockwise in degrees about a (row, column) center."""
        matrix = np.vstack((rotation(self.shape, angle, center), [0.0, 0.0, 1.0]))
        return self._transform(matrix, self.shape, save_trace)

    @Debugger.method_decorator("MarksLand.move")
    def move(self, vector, save_trace=False):
        """Translate by a (row, column) vector, optionally recording inverse."""
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (2,) or not np.isfinite(vector).all():
            raise ValueError("vector must contain two finite values.")
        matrix = np.array([[1.0, 0.0, vector[1]], [0.0, 1.0, vector[0]], [0.0, 0.0, 1.0]])
        return self._transform(matrix, self.shape, save_trace)

    @Debugger.method_decorator("MarksLand.zoom")
    def zoom(self, quadrilateral=None, zoom_factor=None, center=None, save_trace=False):
        """Map points into a rectangular or perspective crop's coordinates."""
        if quadrilateral is None:
            from .square_marks import SquareMarks

            quadrilateral = SquareMarks(shape=self.shape, zoom_factor=zoom_factor, center=center)
        matrix, shape = crop_geometry(quadrilateral)
        return self._transform(matrix, shape, save_trace)

    def _aligned(self, other):
        if not isinstance(other, MarksLand):
            raise TypeError("Expected another MarksLand.")
        if self.shape != other.shape:
            return other.copy().reshape(self.shape)
        return other

    @Debugger.method_decorator("MarksLand.__add__")
    def __add__(self, other):
        """Return a stable point union; align a copy of the other canvas."""
        if not isinstance(other, MarksLand):
            return NotImplemented
        other = self._aligned(other)
        points = list(dict.fromkeys(map(tuple, np.vstack((self.get_marks(), other.get_marks())))))
        return MarksLand(points, shape=self.shape, debugger=self._debugger)

    @Debugger.method_decorator("MarksLand.__sub__")
    def __sub__(self, other):
        """Return points absent from the other set without mutating operands."""
        other = self._aligned(other)
        excluded = set(map(tuple, other.get_marks()))
        return MarksLand(
            [p for p in self.get_marks() if tuple(p) not in excluded],
            shape=self.shape,
            debugger=self._debugger,
        )

    def __getitem__(self, index):
        """Return selected points as a MarksLand, including scalar selection."""
        return MarksLand(self.get_marks()[index], shape=self.shape, debugger=self._debugger)

    @Debugger.method_decorator("MarksLand.to_list")
    def to_list(self):
        """Return (list of points, canvas shape)."""
        return self.get_marks().tolist(), self.shape

    @Debugger.method_decorator("MarksLand.to_heat")
    def to_heat(self, fill_heat=1):
        """Fill the point polygon with a value in [0, 1] on its canvas."""
        from .heat_land import HeatLand

        if not np.isfinite(fill_heat) or not 0 <= fill_heat <= 1:
            raise ValueError("fill_heat must be in [0, 1].")
        heat = np.zeros(shape2(self.shape), dtype=np.float32)
        if len(self.get_marks()):
            cv2.fillPoly(
                heat, [np.rint(self.get_marks()[:, ::-1]).astype(np.int32)], float(fill_heat)
            )
        return HeatLand(heat, debugger=self._debugger)

    @Debugger.method_decorator("MarksLand.to_square")
    def to_square(self):
        """Return the axis-aligned bounding rectangle; reject an empty set."""
        from .square_marks import SquareMarks

        if len(self.get_marks()) == 0:
            raise ValueError("Cannot bound an empty point set.")
        low, high = self.get_marks().min(axis=0), self.get_marks().max(axis=0)
        return SquareMarks(low[0], high[0], low[1], high[1], self.shape, debugger=self._debugger)

    @Debugger.method_decorator("MarksLand.show")
    def show(self):
        """Plot the point set on its canvas and return this object."""
        Debugger(visual_debug=True).log_image(self._flow_data, type=Debugger.LogTypes.SCATTER)
        return self
