from .config import MODEL_URLS
from landpy.tf_lite_model import TFLiteModel
import numpy as np
from landpy import ImageLand, Debugger, SquareMarks, MarksLand
import itertools
import math


class FaceTackerIndices:
    UPPER_LEFT_CORNER = 0
    UPPER_RIGHT_CORNER = 1
    LOWER_LEFT_CORNER = 3
    LOWER_RIGHT_CORNER = 2
    RIGHT_EYE = 4
    LEFT_EYE = 5
    NOSE = 6
    MOUTH = 7
    RIGHT_EAR = 8
    LEFT_EAR = 9


class FacePoints:
    def __init__(self, face_box: SquareMarks, face_points: MarksLand, p: float, debugger: Debugger = Debugger(debug=False)):
        self._debugger = debugger
        self.face_box = face_box
        self.face_points = face_points
        self.p = p
        
    def show(self):
        (self.face_box+self.face_points).show()

    @Debugger.method_decorator('Getting indices of face points', ['FacePoints.__getitem__', 'FacePoints'])
    def __getitem__(self, index):
        if index < 4:
            return self.face_box[index]
        else:
            return self.face_points[index-4]


class FaceTracker(TFLiteModel):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(model_url=MODEL_URLS['face_tracker'],
                         input_type=ImageLand,
                         output_type=list,
                         debugger=debugger)
        self._anchors = Anchor('back').anchors

    @Debugger.static_method_decorator('Converting output data to boxes positions', ['FaceTracker.__decode_boxes', 'FaceTracker'])
    def __decode_face_points(points, org_rect, zoomed_rect,  p, debugger: Debugger = Debugger(debug=False)):
        face_points = np.array(points[0:4])
        face_points = np.array(list(itertools.product(
            face_points[0::2], face_points[1::2])))
        face_points = face_points*np.array(zoomed_rect.shape())
        face_box = SquareMarks(shape=org_rect.shape()).zoom(
            zoomed_rect, save_trace=True)
        face_box.set_marks(
            face_points[0][0], face_points[3][0], face_points[0][1], face_points[3][1])
        face_box.retrace()

        other_points = np.array(
            points[4:]).reshape(-1, 2)[:, ::-1]*np.array(zoomed_rect.shape())
        sm = SquareMarks(shape=org_rect.shape())
        face_points = MarksLand(sm._flow_data[0], shape=sm._flow_data[1]).zoom(
            zoomed_rect, save_trace=True)
        face_points.set_marks(other_points)
        face_points.retrace()

        face_box.set_debugger(debugger)
        face_points.set_debugger(debugger)

        return FacePoints(face_box, face_points, p)

    @Debugger.static_method_decorator('Converting output data to boxes positions', ['FaceTracker.__decode_boxes', 'FaceTracker'])
    def __decode_boxes(raw_boxes, anchors, input_shape=(128, 128), scale_shape=(1, 1), debugger: Debugger = Debugger(debug=False)):
        x_scale, y_scale = input_shape[0], input_shape[1]
        h_scale, w_scale = input_shape[0] / \
            scale_shape[0], input_shape[1]/scale_shape[1]
        min_score_thresh = 0.75
        min_suppression_threshold = 0.3
        boxes = np.zeros_like(raw_boxes)
        x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / h_scale * anchors[:, 3]
        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax
        for k in range(6):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset] / \
                x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / \
                y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y
        return boxes

    @Debugger.method_decorator('Inferencing by model', ['FaceTracker.detect_faces', 'FaceTracker'])
    def detect_faces(self, input: ImageLand, treshhold=None, top_k=None):
        if top_k is not None and treshhold is not None:
            self._debugger.log_exception(
                'You can\'t feed treshhold and top_k at the same time!')
        if top_k is None and treshhold is None:
            top_k = 1
            
        input.make_checkpoint()

        org_rect = SquareMarks(shape=input._flow_data.shape[:2])
        zoomed_rect = SquareMarks(shape=(max(input._flow_data.shape[1], input._flow_data.shape[0]),
                                         max(input._flow_data.shape[0], input._flow_data.shape[1])))
        input.zoom(zoomed_rect, save_trace=False)
        input.reshape((128, 128), save_trace=False)
        input_data = np.expand_dims(input._flow_data, axis=0)
        input_data = np.array(input_data, dtype=np.float32)/255.0

        self.input_type = np.ndarray
        output = super().inference(input_data)
        self.input_type = ImageLand

        faces = []
        points = FaceTracker.__decode_boxes(
            output[0], self._anchors, debugger=Debugger(debug=False))
        probabilities = np.array(output[1][0][:, 0])

        indices = []
        if top_k is not None:
            partitioned_indices = np.argpartition(
                probabilities, -top_k)[-top_k:]
            indices = partitioned_indices[np.argsort(
                -probabilities[partitioned_indices])]
        if treshhold is not None:
            indices = np.where(probabilities > treshhold)[0]

        for i in indices:
            face_points = FaceTracker.__decode_face_points(
                points[0][i], org_rect, zoomed_rect, probabilities[i], self._debugger)
            faces.append(face_points)

        input.undo()
        return faces


class SsdAnchorsCalculatorOptions:
    def __init__(self, input_size_width, input_size_height, min_scale, max_scale, num_layers, feature_map_width, feature_map_height, strides, aspect_ratios, anchor_offset_x=0.5, anchor_offset_y=0.5, reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0, fixed_anchor_size=False):
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
    def __init__(self, type="front"):
        self.anchors = None
        self.type = type
        self._generateAnchors()

    def _generateAnchors(self):
        if self.type == "front":
            # Options to generate anchors for SSD object detection models.
            ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=128, input_size_height=128, min_scale=0.1484375, max_scale=0.75, anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4, feature_map_width=[
            ], feature_map_height=[], strides=[8, 16, 16, 16], aspect_ratios=[1.0], reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0, fixed_anchor_size=True)

        elif self.type == "back":
            # Options to generate anchors for SSD object detection models.
            ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=256, input_size_height=256, min_scale=0.15625, max_scale=0.75, anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4, feature_map_width=[
            ], feature_map_height=[], strides=[16, 32, 32, 32], aspect_ratios=[1.0], reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0, fixed_anchor_size=True)

        self.anchors = self._gen_anchors(ssd_anchors_calculator_options)

    def _gen_anchors(self, options):
        anchors = []
        # Verify the options.
        if (options.strides_size != options.num_layers):
            print("strides_size and num_layers must be equal.")
            return []

        layer_id = 0
        while (layer_id < options.strides_size):
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []

            # For same strides, we merge the anchors in the same order.
            last_same_stride_layer = layer_id
            while (last_same_stride_layer < options.strides_size and options.strides[last_same_stride_layer] == options.strides[layer_id]):
                scale = options.min_scale + \
                    (options.max_scale - options.min_scale) * 1.0 * \
                    last_same_stride_layer / (options.strides_size - 1.0)
                if (last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer):
                    # For first layer, it can be specified to use predefined anchors.
                    aspect_ratios.append(1.0)
                    aspect_ratios.append(2.0)
                    aspect_ratios.append(0.5)
                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)
                else:
                    for aspect_ratio_id in range(options.aspect_ratios_size):
                        aspect_ratios.append(
                            options.aspect_ratios[aspect_ratio_id])
                        scales.append(scale)

                    if (options.interpolated_scale_aspect_ratio > 0.0):
                        scale_next = 1.0 if last_same_stride_layer == options.strides_size - 1 else options.min_scale + \
                            (options.max_scale - options.min_scale) * 1.0 * \
                            (last_same_stride_layer+1) / \
                            (options.strides_size - 1.0)
                        scales.append(math.sqrt(scale * scale_next))
                        aspect_ratios.append(
                            options.interpolated_scale_aspect_ratio)
                last_same_stride_layer += 1
            for i in range(len(aspect_ratios)):
                ratio_sqrts = math.sqrt(aspect_ratios[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)

            feature_map_height = 0
            feature_map_width = 0
            if (options.feature_map_height_size > 0):
                feature_map_height = options.feature_map_height[layer_id]
                feature_map_width = options.feature_map_width[layer_id]
            else:
                stride = options.strides[layer_id]
                feature_map_height = math.ceil(
                    1.0 * options.input_size_height / stride)
                feature_map_width = math.ceil(
                    1.0 * options.input_size_width / stride)

            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_id in range(len(anchor_height)):
                        x_center = (x + options.anchor_offset_x) * \
                            1.0 / feature_map_width
                        y_center = (y + options.anchor_offset_y) * \
                            1.0 / feature_map_height
                        w = 0
                        h = 0
                        if (options.fixed_anchor_size):
                            w = 1.0
                            h = 1.0
                        else:
                            w = anchor_width[anchor_id]
                            h = anchor_height[anchor_id]

                        new_anchor = np.array([x_center, y_center, h, w])
                        anchors.append(new_anchor)
            layer_id = last_same_stride_layer
        return np.array(anchors)
