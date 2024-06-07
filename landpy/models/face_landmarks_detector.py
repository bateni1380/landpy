from landpy.models import TFLiteModel
import numpy as np
from landpy import ImageLand, Debugger, SquareMarks, MarksLand

class FaceLandmarksDetector(TFLiteModel):  
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__('landpy/models/model_files/face_landmark.tflite', ImageLand, list, debugger)
        self.output_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.land_input_shape = ('RGB', (192,192,3), 'float32', True)
        
    @Debugger.method_decorator('Inferencing by model', ['ExpressionDetector.detect_expression', 'ExpressionDetector'])
    def detect_face_landmarks(self, input: ImageLand):
        input.make_checkpoint()

        rect = SquareMarks(0, max(input._flow_data.shape[1]/input._flow_data.shape[0],1), 
                           0, max(input._flow_data.shape[0]/input._flow_data.shape[1],1))
        input.zoom(rect, save_trace=False)
        input.reshape((192,192), save_trace=False)

        input_data = np.expand_dims(input._flow_data, axis=0)
        input_data = np.array(input_data, dtype=np.float32)/255.0

        self.input_type = np.ndarray
        output = super().inference(input_data)
        self.input_type = ImageLand

        land_x = output[0][0,0,0,1::3]/192.0
        land_y = output[0][0,0,0,0::3]/192.0
        
        marks = MarksLand(land_x, land_y, input._debugger)
        p_value = output[1][0,0,0,0]
        
        input.undo()
        
        return marks, p_value

class FaceLandmarksIndices:
    LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    UPPER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
    LIWER_LIPS = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    LEFT_EYE = [466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362]
    RIGHT_EYE = [246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133]
    GLASSES_INDICES = [64,294,347,118]

    LEFT_BLUSH  = [347, 330, 266, 426, 427, 411, 376, 352]
    RIGHT_BLUSH = [118, 101, 36 , 206, 207, 187, 147, 123]

    LEFT_BLUSH_RECTANGLE = [116, 100, 92, 213]
    RIGHT_BLUSH_RECTANGLE = [345, 329, 322, 433]

    LEFT_EYE_TRIANGLE = [263, 257 , 362]
    RIGHT_EYE_TRIANGLE = [33, 27, 133]

    LEFT_BLUSH_SHADOW = [113, 225, 224, 223, 222, 221, 189, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_BLUSH_SHADOW = [342, 445, 444, 443, 442, 441, 413, 398, 384, 385, 386, 387, 388, 466]
    FACE_OVAL = [132, 389, 136, 10, 397, 400, 148, 149, 150, 21, 152, 284, 288, 162, 297, 172, 176, 54, 58, 323, 67, 454, 332, 338, 93, 356, 103, 361, 234, 109, 365, 379, 377, 378, 251, 127]
