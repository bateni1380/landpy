# image.make_checkpoint()

# image = image.reshape([128,128])
# model = load_tflite('files/face_segm.tflite', debugger=debugger)
# input_type = InferenceType([1,128,128,3], np.float32, does_divide=True)
# output = inference_by_tflite(model, image, input_type, debugger=debugger)
# heat = HeatLand(output.reshape([128,128]))

# image.undo()
# image.implant_filter(Filter('#ff0000', ratio=0.5), heat)

from landpy.heat_land import HeatLand
from landpy.tf_lite_model import TFLiteModel
import numpy as np
from landpy import ImageLand, Debugger, SquareMarks, MarksLand
from .config import MODEL_URLS

class FaceSegmentationDetector(TFLiteModel):  
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(MODEL_URLS['face_segmentation_detector'],
                         input_type=ImageLand,
                         output_type=HeatLand,
                         debugger=debugger)
        self.land_input_shape = ('RGB', (192,192,3), 'float32', True)
        
    @Debugger.method_decorator('Inferencing by model')
    def detect_face_segmentation(self, input: ImageLand):
        input.make_checkpoint()
        
        input = input.zoom(-0.1, save_trace=False)
        input.to_rgb()
        input.reshape((128,128), save_trace=False)
                
        input_data = np.expand_dims(input._flow_data, axis=0)
        input_data = np.array(input_data, dtype=np.float32)/255.0

        self.input_type = np.ndarray
        output = super().inference(input_data)
        
        heat = HeatLand(output.reshape([128,128]))
        
        input.undo()
        
        return heat
    