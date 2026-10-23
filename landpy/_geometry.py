"""Shared row/column geometry for images, heat maps and landmarks."""

import cv2
import numpy as np


def shape2(shape):
    if shape is None or len(shape) < 2:
        raise ValueError("shape must contain height and width.")
    values = np.asarray(shape[:2], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values != np.floor(values)):
        raise ValueError("height and width must be positive integers.")
    return tuple(map(int, values))


def rotation(shape, angle, center=None):
    h, w = shape2(shape)
    center = (h / 2, w / 2) if center is None else center
    if len(center) != 2 or not np.isfinite([angle, *center]).all():
        raise ValueError("angle and center must be finite.")
    return cv2.getRotationMatrix2D(tuple(map(float, center[::-1])), float(angle), 1)


def transform_points(points, matrix):
    xy = np.asarray(points, dtype=float)[:, ::-1]
    result = (
        cv2.perspectiveTransform(xy.reshape(1, -1, 2), matrix)
        if matrix.shape == (3, 3)
        else cv2.transform(xy.reshape(1, -1, 2), matrix)
    )
    return result.reshape(-1, 2)[:, ::-1]


def crop_geometry(quadrilateral):
    from .quad_marks import QuadMarks
    from .square_marks import SquareMarks

    if isinstance(quadrilateral, SquareMarks):
        r0, r1 = np.floor(quadrilateral.minX()), np.ceil(quadrilateral.maxX())
        c0, c1 = np.floor(quadrilateral.minY()), np.ceil(quadrilateral.maxY())
        h, w = int(r1 - r0), int(c1 - c0)
        matrix = np.array([[1.0, 0.0, -c0], [0.0, 1.0, -r0], [0.0, 0.0, 1.0]])
    elif isinstance(quadrilateral, QuadMarks):
        pts = quadrilateral.get_marks()[:, ::-1].astype(np.float32)
        w = int(round(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))))
        h = int(round(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))))
        shape2((h, w))
        matrix = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    else:
        raise TypeError("quadrilateral must be SquareMarks or QuadMarks.")
    shape2((h, w))
    return matrix, (h, w)


def inverse_warp(data, parameters, debugger):
    original = parameters["original"]
    if data.ndim != original.ndim or (data.ndim == 3 and data.shape[2] != original.shape[2]):
        raise ValueError("Restore channel mode before retracing a crop.")
    h, w = original.shape[:2]
    inverse = np.linalg.inv(parameters["matrix"])
    warped = cv2.warpPerspective(data.astype(np.float64), inverse, (w, h))
    coverage = cv2.warpPerspective(np.ones(data.shape[:2], dtype=np.float64), inverse, (w, h))
    if original.ndim == 3:
        coverage = coverage[..., None]
    normalized = np.divide(warped, coverage, out=np.zeros_like(warped), where=coverage > 0)
    if original.ndim == 3 and original.shape[2] == 4:
        coverage = coverage * (normalized[..., 3:4] > 0)
    result = original * (1 - coverage) + normalized * coverage
    if np.issubdtype(original.dtype, np.integer):
        result = np.rint(result)
    return result.astype(original.dtype)
