"""BlazeFace boxes and six keypoints in original image coordinates."""

import math

import numpy as np

from .. import ImageLand, MarksLand, SquareMarks
from ..debugger import Debugger
from ..tf_lite_model import TFLiteModel
from .config import MODEL_SHA256, MODEL_URLS


class FaceTackerIndices:
    """Legacy indices for box corners followed by six facial keypoints."""

    UPPER_LEFT_CORNER, UPPER_RIGHT_CORNER = 0, 1
    LOWER_RIGHT_CORNER, LOWER_LEFT_CORNER = 2, 3
    RIGHT_EYE, LEFT_EYE, NOSE, MOUTH, RIGHT_EAR, LEFT_EAR = range(4, 10)


FaceTrackerIndices = FaceTackerIndices


class FacePoints:
    """One detection: bounding rectangle, six keypoints and confidence p."""

    def __init__(self, face_box, face_points, p, debugger=None):
        self._debugger = debugger or Debugger(debug=False)
        self.face_box, self.face_points, self.p = face_box, face_points, float(p)

    def __getitem__(self, index):
        """Select a box corner (0..3) or facial keypoint (4..9)."""
        if not isinstance(index, int) or not 0 <= index < 10:
            raise IndexError("Face point index must be in 0..9.")
        return self.face_box[index] if index < 4 else self.face_points[index - 4]

    def show(self):
        """Plot this detection's rectangle and landmarks."""
        (self.face_box + self.face_points).show()
        return self


class FaceTracker(TFLiteModel):
    """Detect faces with the original short-range BlazeFace model.

    Confidence logits are converted to probabilities and overlapping detections
    are suppressed. Inference works on a copy of the supplied image.
    """

    def __init__(self, debugger=None, **kwargs):
        if kwargs.get("model_path") is None:
            kwargs.setdefault("sha256", MODEL_SHA256["face_tracker"])
        super().__init__(
            MODEL_URLS["face_tracker"], input_type=np.ndarray, debugger=debugger, **kwargs
        )
        self._anchors = Anchor("front").anchors

    @staticmethod
    def _decode_boxes(raw, anchors, height, width):
        boxes = np.empty_like(raw)
        xc = raw[:, 0] / width * anchors[:, 2] + anchors[:, 0]
        yc = raw[:, 1] / height * anchors[:, 3] + anchors[:, 1]
        w, h = raw[:, 2] / width * anchors[:, 2], raw[:, 3] / height * anchors[:, 3]
        boxes[:, :4] = np.column_stack((yc - h / 2, xc - w / 2, yc + h / 2, xc + w / 2))
        for k in range(6):
            offset = 4 + 2 * k
            boxes[:, offset] = raw[:, offset] / width * anchors[:, 2] + anchors[:, 0]
            boxes[:, offset + 1] = raw[:, offset + 1] / height * anchors[:, 3] + anchors[:, 1]
        return boxes

    def detect_faces(self, input, treshhold=None, top_k=None, *, threshold=None):
        """Return FacePoints sorted by confidence after IoU suppression.

        ``threshold`` is a probability in [0, 1] (default .5); ``treshhold``
        preserves the old spelling. ``top_k`` limits results after suppression.
        The default returns at most one face; threshold alone returns all matches.
        """
        if not isinstance(input, ImageLand):
            raise TypeError("input must be ImageLand.")
        if threshold is not None and treshhold is not None:
            raise ValueError("Use threshold or treshhold, not both.")
        threshold = threshold if threshold is not None else treshhold
        if threshold is None:
            threshold = 0.5
            top_k = 1 if top_k is None else top_k
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k must be a positive integer.")
        _, h, w, _ = self._interpreter.get_input_details()[0]["shape"]
        side = max(input.shape)
        data = (
            input.copy()
            .to_rgb()
            .zoom(SquareMarks(shape=(side, side)))
            .reshape((h, w))
            .get_image()[None]
            .astype(np.float32)
            / 127.5
            - 1
        )
        output = self.inference(data)
        raw = next(o for o in output if o.shape[-1] == 16).reshape(-1, 16)
        logits = next(o for o in output if o.shape[-1] == 1).reshape(-1)
        if len(raw) != len(self._anchors):
            raise ValueError("Model output does not match short-range BlazeFace anchors.")
        boxes = self._decode_boxes(raw, self._anchors, h, w)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        candidates = np.flatnonzero(probabilities >= threshold)
        candidates = candidates[np.argsort(-probabilities[candidates])]
        kept = []
        while len(candidates):
            i = candidates[0]
            kept.append(i)
            if top_k is not None and len(kept) >= top_k:
                break
            rest = candidates[1:]
            low = np.maximum(boxes[i, :2], boxes[rest, :2])
            high = np.minimum(boxes[i, 2:4], boxes[rest, 2:4])
            intersection = np.maximum(0, high - low).prod(axis=1)
            area = np.maximum(0, boxes[i, 2:4] - boxes[i, :2]).prod()
            areas = np.maximum(0, boxes[rest, 2:4] - boxes[rest, :2]).prod(axis=1)
            union = area + areas - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            candidates = rest[iou <= 0.3]
        faces = []
        for i in kept:
            r0, c0, r1, c1 = boxes[i, :4] * side
            box = SquareMarks(r0, r1, c0, c1, input.shape, debugger=self._debugger)
            points = MarksLand(
                boxes[i, 4:].reshape(-1, 2)[:, ::-1] * side,
                shape=input.shape,
                debugger=self._debugger,
            )
            faces.append(FacePoints(box, points, probabilities[i], self._debugger))
        return faces


class SsdAnchorsCalculatorOptions:
    """Original SSD anchor-generation parameters in model input pixels."""

    def __init__(
        self,
        input_size_width,
        input_size_height,
        min_scale,
        max_scale,
        num_layers,
        feature_map_width,
        feature_map_height,
        strides,
        aspect_ratios,
        anchor_offset_x=0.5,
        anchor_offset_y=0.5,
        reduce_boxes_in_lowest_layer=False,
        interpolated_scale_aspect_ratio=1.0,
        fixed_anchor_size=False,
    ):
        self.input_size_width = input_size_width
        self.input_size_height = input_size_height
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.anchor_offset_x = anchor_offset_x
        self.anchor_offset_y = anchor_offset_y
        self.num_layers = num_layers
        self.feature_map_width = feature_map_width
        self.feature_map_height = feature_map_height
        self.feature_map_width_size = len(feature_map_width)
        self.feature_map_height_size = len(feature_map_height)
        self.strides = strides
        self.strides_size = len(strides)
        self.aspect_ratios = aspect_ratios
        self.aspect_ratios_size = len(aspect_ratios)
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer
        self.interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio
        self.fixed_anchor_size = fixed_anchor_size


class Anchor:
    """Generate original BlazeFace front (128px) or back (256px) anchors."""

    def __init__(self, type="front"):
        self.anchors = None
        if type not in ("front", "back"):
            raise ValueError("Anchor type must be front or back.")
        self.type = type
        self._generateAnchors()

    def _generateAnchors(self):
        if self.type == "front":
            # Options to generate anchors for SSD object detection models.
            ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(
                input_size_width=128,
                input_size_height=128,
                min_scale=0.1484375,
                max_scale=0.75,
                anchor_offset_x=0.5,
                anchor_offset_y=0.5,
                num_layers=4,
                feature_map_width=[],
                feature_map_height=[],
                strides=[8, 16, 16, 16],
                aspect_ratios=[1.0],
                reduce_boxes_in_lowest_layer=False,
                interpolated_scale_aspect_ratio=1.0,
                fixed_anchor_size=True,
            )

        elif self.type == "back":
            # Options to generate anchors for SSD object detection models.
            ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(
                input_size_width=256,
                input_size_height=256,
                min_scale=0.15625,
                max_scale=0.75,
                anchor_offset_x=0.5,
                anchor_offset_y=0.5,
                num_layers=4,
                feature_map_width=[],
                feature_map_height=[],
                strides=[16, 32, 32, 32],
                aspect_ratios=[1.0],
                reduce_boxes_in_lowest_layer=False,
                interpolated_scale_aspect_ratio=1.0,
                fixed_anchor_size=True,
            )

        self.anchors = self._gen_anchors(ssd_anchors_calculator_options)

    def _gen_anchors(self, options):
        anchors = []
        # Verify the options.
        if options.strides_size != options.num_layers:
            print("strides_size and num_layers must be equal.")
            return []

        layer_id = 0
        while layer_id < options.strides_size:
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []

            # For same strides, we merge the anchors in the same order.
            last_same_stride_layer = layer_id
            while (
                last_same_stride_layer < options.strides_size
                and options.strides[last_same_stride_layer] == options.strides[layer_id]
            ):
                scale = options.min_scale + (
                    options.max_scale - options.min_scale
                ) * 1.0 * last_same_stride_layer / (options.strides_size - 1.0)
                if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                    # For first layer, it can be specified to use predefined anchors.
                    aspect_ratios.append(1.0)
                    aspect_ratios.append(2.0)
                    aspect_ratios.append(0.5)
                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)
                else:
                    for aspect_ratio_id in range(options.aspect_ratios_size):
                        aspect_ratios.append(options.aspect_ratios[aspect_ratio_id])
                        scales.append(scale)

                    if options.interpolated_scale_aspect_ratio > 0.0:
                        scale_next = (
                            1.0
                            if last_same_stride_layer == options.strides_size - 1
                            else options.min_scale
                            + (options.max_scale - options.min_scale)
                            * 1.0
                            * (last_same_stride_layer + 1)
                            / (options.strides_size - 1.0)
                        )
                        scales.append(math.sqrt(scale * scale_next))
                        aspect_ratios.append(options.interpolated_scale_aspect_ratio)
                last_same_stride_layer += 1
            for i in range(len(aspect_ratios)):
                ratio_sqrts = math.sqrt(aspect_ratios[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)

            feature_map_height = 0
            feature_map_width = 0
            if options.feature_map_height_size > 0:
                feature_map_height = options.feature_map_height[layer_id]
                feature_map_width = options.feature_map_width[layer_id]
            else:
                stride = options.strides[layer_id]
                feature_map_height = math.ceil(1.0 * options.input_size_height / stride)
                feature_map_width = math.ceil(1.0 * options.input_size_width / stride)

            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_id in range(len(anchor_height)):
                        x_center = (x + options.anchor_offset_x) * 1.0 / feature_map_width
                        y_center = (y + options.anchor_offset_y) * 1.0 / feature_map_height
                        w = 0
                        h = 0
                        if options.fixed_anchor_size:
                            w = 1.0
                            h = 1.0
                        else:
                            w = anchor_width[anchor_id]
                            h = anchor_height[anchor_id]

                        new_anchor = np.array([x_center, y_center, h, w])
                        anchors.append(new_anchor)
            layer_id = last_same_stride_layer
        return np.array(anchors)
