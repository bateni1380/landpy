import numpy as np
def decode_boxes(raw_boxes, anchors):
            input_shape = (128,128)
            scale_shape = (1,1)
            x_scale, y_scale = input_shape[0], input_shape[1]
            h_scale, w_scale = input_shape[0]/scale_shape[0], input_shape[1]/scale_shape[1]
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
                keypoint_x = raw_boxes[..., offset    ] / x_scale * anchors[:, 2] + anchors[:, 0]
                keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
                boxes[..., offset    ] = keypoint_x
                boxes[..., offset + 1] = keypoint_y
            return boxes
