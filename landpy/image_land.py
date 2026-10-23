"""RGB/RGBA image transformations, masked filters and compositing."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ._geometry import crop_geometry, inverse_warp, rotation, shape2
from .debugger import Debugger
from .land import Land


class ImageLand(Land):
    """An owned uint8 RGB/RGBA image loaded from an array, PIL image or path.

    Floating arrays represent [0, 1]; integer arrays represent [0, 255].
    Grayscale input becomes RGB. Operations mutate and return this object;
    addition returns a new image. Coordinates are always (row, column).
    """

    ADDED_POINTS_RADIOUS = 1.5
    ADDED_POINTS_COLOR = (255, 0, 0)
    ADDED_LINES_WIDTH = 2
    ADDED_LINES_COLOR = (255, 255, 0)
    ADDED_VECTORS_WIDTH = 1
    ADDED_VECTORS_COLOR = (255, 255, 255)

    def __init__(self, image=None, debugger=None):
        super().__init__(debugger)
        if image is not None:
            self.set_image(image)

    @property
    def shape(self):
        """Spatial dimensions as (height, width), or None for an empty image."""
        return None if self._flow_data is None else self._flow_data.shape[:2]

    @property
    def mode(self):
        """Current PIL-compatible channel mode, derived from image data."""
        if self._flow_data is None:
            return None
        return "RGBA" if self._flow_data.shape[-1] == 4 else "RGB"

    @Debugger.method_decorator("ImageLand.set_image")
    def set_image(self, image):
        """Copy and validate image data; read paths with Pillow."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = np.array(source.convert("RGBA" if "A" in source.getbands() else "RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGBA" if "A" in image.getbands() else "RGB"))
        data = np.asarray(image)
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=2)
        if data.ndim != 3 or data.shape[-1] not in (3, 4) or not data.size:
            raise ValueError("image must be a nonempty RGB, RGBA or grayscale array.")
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError("image values must be finite numbers.")
        limit = 1 if np.issubdtype(data.dtype, np.floating) else 255
        if np.any((data < 0) | (data > limit)):
            raise ValueError(f"image values must be in [0, {limit}].")
        if limit == 1:
            data = np.rint(data * 255)
        return self._set_data(np.array(data, dtype=np.uint8, copy=True))

    @Debugger.method_decorator("ImageLand.get_image")
    def get_image(self):
        """Return the mutable image array; copy it before editing independently."""
        return self._flow_data

    @Debugger.method_decorator("ImageLand.to_rgba")
    def to_rgba(self):
        """Append opaque alpha when needed, preserving uint8 dtype."""
        if self.mode == "RGB":
            self._set_data(np.dstack((self._flow_data, np.full(self.shape, 255, dtype=np.uint8))))
        return self

    @Debugger.method_decorator("ImageLand.to_rgb")
    def to_rgb(self):
        """Composite RGBA onto black and remove alpha; keep uint8 dtype."""
        if self.mode == "RGBA":
            self._set_data(
                np.rint(
                    self._flow_data[..., :3].astype(float) * (self._flow_data[..., 3:4] / 255.0)
                ).astype(np.uint8)
            )
        return self

    @Debugger.method_decorator("Rotating image")
    def rotate(self, angle, center=None, save_trace=False, fill_color=(0, 0, 0)):
        """Rotate counterclockwise in degrees; inverse resampling is approximate."""
        h, w = self.shape
        matrix = rotation(self.shape, angle, center)
        result = cv2.warpAffine(self._flow_data, matrix, (w, h), borderValue=fill_color)
        if save_trace:
            self._make_trace(
                ImageLand._rotate_back,
                {"matrix": cv2.invertAffineTransform(matrix), "shape": self.shape},
            )
        return self._set_data(result)

    @staticmethod
    def _rotate_back(data, parameters, debugger):
        h, w = parameters["shape"]
        return cv2.warpAffine(data, parameters["matrix"], (w, h))

    @Debugger.method_decorator("ImageLand.reshape")
    def reshape(self, shape, mode=None, save_trace=False):
        """Resize to (height, width); retain mode unless RGB/RGBA is specified."""
        h, w = shape2(shape)
        if mode is not None and mode not in ("RGB", "RGBA"):
            raise ValueError("mode must be RGB or RGBA.")
        image = Image.fromarray(self._flow_data).resize((w, h), Image.Resampling.BILINEAR)
        image = image.convert(mode or self.mode)
        if save_trace:
            self._make_trace(ImageLand._resize_back, {"shape": self.shape, "mode": self.mode})
        return self._set_data(np.array(image))

    @staticmethod
    def _resize_back(data, parameters, debugger):
        h, w = parameters["shape"]
        return np.array(
            Image.fromarray(data)
            .resize((w, h), Image.Resampling.BILINEAR)
            .convert(parameters["mode"])
        )

    @Debugger.method_decorator("ImageLand.zoom")
    def zoom(
        self,
        quadrilateral=None,
        zoom_factor=None,
        center=None,
        save_trace=False,
        fill_color=(0, 0, 0),
    ):
        """Crop/pad a rectangle or rectify a quad; retrace pastes edits back."""
        from .square_marks import SquareMarks

        if quadrilateral is None:
            quadrilateral = SquareMarks(shape=self.shape, zoom_factor=zoom_factor, center=center)
        matrix, (h, w) = crop_geometry(quadrilateral)
        result = cv2.warpPerspective(self._flow_data, matrix, (w, h), borderValue=fill_color)
        if save_trace:
            self._make_trace(inverse_warp, {"original": self._flow_data, "matrix": matrix})
        return self._set_data(result)

    @Debugger.method_decorator("ImageLand.equalize_histogram")
    def equalize_histogram(self, save_trace=False):
        """Equalize Lab luminance, preserving alpha; retrace restores luminance."""
        lab = cv2.cvtColor(self._flow_data[..., :3], cv2.COLOR_RGB2LAB)
        if save_trace:
            self._make_trace(ImageLand._equalize_back, {"luminance": lab[..., 0]})
        lab[..., 0] = cv2.equalizeHist(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return self._set_data(
            np.dstack((rgb, self._flow_data[..., 3])) if self.mode == "RGBA" else rgb
        )

    @staticmethod
    def _equalize_back(data, parameters, debugger):
        if data.shape[:2] != parameters["luminance"].shape:
            raise ValueError("Restore image dimensions before retracing histogram equalization.")
        lab = cv2.cvtColor(data[..., :3], cv2.COLOR_RGB2LAB)
        lab[..., 0] = parameters["luminance"]
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return np.dstack((rgb, data[..., 3])) if data.shape[-1] == 4 else rgb

    @Debugger.method_decorator("ImageLand.implant_filter")
    def implant_filter(self, filter, heat=None):
        """Blend a Filter through a HeatLand mask; preserve image alpha."""
        from .filter import Filter
        from .heat_land import HeatLand

        if not isinstance(filter, Filter):
            raise TypeError("filter must be a Filter.")
        if heat is None:
            heat = HeatLand(shape=self.shape, fill_value=1)
        if not isinstance(heat, HeatLand):
            raise TypeError("heat must be a HeatLand.")
        if heat.shape != self.shape:
            heat = heat.copy().reshape(self.shape)
        result = self._flow_data.copy()
        result[..., :3] = np.clip(
            filter.apply_function(result[..., :3].astype(float), heat.get_heat()[..., None]), 0, 255
        ).astype(np.uint8)
        return self._set_data(result)

    @Debugger.method_decorator("ImageLand.implant_image")
    def implant_image(self, plant_img, src_land=None, plant_land=None, ratio=0.5):
        """Perspective-map an image into a region using straight-alpha blending.

        ``src_land`` is the destination region on this image; ``plant_land``
        selects the source. Regions may be rectangles or convex quads.
        """
        from .quad_marks import QuadMarks
        from .square_marks import SquareMarks

        if not np.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("ratio must be in [0, 1].")
        plant = plant_img.copy() if isinstance(plant_img, ImageLand) else ImageLand(plant_img)
        plant.to_rgba()
        src_land = SquareMarks(shape=self.shape) if src_land is None else src_land
        plant_land = SquareMarks(shape=plant.shape) if plant_land is None else plant_land
        if not isinstance(src_land, QuadMarks) or not isinstance(plant_land, QuadMarks):
            raise TypeError("Regions must be SquareMarks or QuadMarks.")
        matrix = cv2.getPerspectiveTransform(
            plant_land.get_marks()[:, ::-1].astype(np.float32),
            src_land.get_marks()[:, ::-1].astype(np.float32),
        )
        h, w = self.shape
        source = plant.get_image().astype(float) / 255
        alpha = source[..., 3:4]
        source[..., :3] *= alpha
        warped = cv2.warpPerspective(source, matrix, (w, h))
        coverage = np.zeros(plant.shape, dtype=np.float32)
        cv2.fillPoly(coverage, [np.rint(plant_land.get_marks()[:, ::-1]).astype(np.int32)], 1.0)
        mask = cv2.warpPerspective(coverage, matrix, (w, h))[..., None] * ratio
        sa = warped[..., 3:4] * mask
        foreground = warped[..., :3] * mask
        background = self._flow_data[..., :3].astype(float) / 255
        da = self._flow_data[..., 3:4] / 255 if self.mode == "RGBA" else np.ones((*self.shape, 1))
        out_alpha = sa + da * (1 - sa)
        rgb = np.divide(
            foreground + background * da * (1 - sa),
            out_alpha,
            out=np.zeros_like(background),
            where=out_alpha > 0,
        )
        result = np.dstack((rgb, out_alpha)) if self.mode == "RGBA" else rgb
        return self._set_data(np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8))

    @Debugger.method_decorator("ImageLand.__add__")
    def __add__(self, other):
        """Return an image blend or a landmark/heat overlay; keep inputs intact."""
        from .dual_marks import DualMarks
        from .filter import Filter
        from .heat_land import HeatLand
        from .marks_land import MarksLand
        from .quad_marks import QuadMarks

        result = self.copy(copy_trace=True)
        if isinstance(other, ImageLand):
            right = other.copy().reshape(self.shape, mode=self.mode)
            result._set_data(
                ((self._flow_data.astype(float) + right.get_image()) / 2).astype(np.uint8)
            )
        elif isinstance(other, HeatLand):
            result.implant_filter(Filter((255, 165, 0), 0.5), other)
        elif isinstance(other, MarksLand):
            marks = other.copy()
            if marks.shape is not None and marks.shape != self.shape:
                marks.reshape(self.shape)
            points = np.rint(marks.get_marks()[:, ::-1]).astype(np.int32)
            radius = max(1, round(min(self.shape) * self.ADDED_POINTS_RADIOUS / 100))
            rgb = result._flow_data[..., :3].copy()
            for point in points:
                cv2.circle(rgb, tuple(point), radius, self.ADDED_POINTS_COLOR, -1)
            if isinstance(marks, QuadMarks):
                cv2.polylines(
                    rgb,
                    [points],
                    True,
                    self.ADDED_LINES_COLOR,
                    max(1, round(min(self.shape) * self.ADDED_LINES_WIDTH / 100)),
                )
            elif isinstance(marks, DualMarks):
                cv2.arrowedLine(
                    rgb,
                    tuple(points[0]),
                    tuple(points[1]),
                    self.ADDED_VECTORS_COLOR,
                    max(1, round(min(self.shape) * self.ADDED_VECTORS_WIDTH / 100)),
                )
            result._flow_data[..., :3] = rgb
        else:
            return NotImplemented
        return result

    def __radd__(self, other):
        """Support marks + image and heat + image overlays."""
        return self.__add__(other)

    @Debugger.method_decorator("ImageLand.show")
    def show(self, desc=""):
        """Display the image with an optional title and return this object."""
        Debugger(visual_debug=True).log_image(self._flow_data, desc)
        return self
