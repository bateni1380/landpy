from landpy.models import TFModel
import numpy as np
from landpy import ImageLand, Debugger, SquareMarks

class ExpressionDetector(TFModel):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__('landpy/models/model_files/facial_expression.h5', ImageLand, list, debugger)
        self.output_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    @Debugger.method_decorator('Inferencing by model', ['ExpressionDetector.detect_expression', 'ExpressionDetector'])
    def detect_expression(self, input: ImageLand):
        input.make_checkpoint()

        rect = SquareMarks(0, max(input._flow_data.shape[1]/input._flow_data.shape[0],1), 
                           0, max(input._flow_data.shape[0]/input._flow_data.shape[1],1))
        input.zoom(rect, save_trace=False)
        input.reshape((96,96), save_trace=False)

        input._flow_data[...,0] = 0.299 * input._flow_data[..., 0] + 0.587 * input._flow_data[..., 1] + 0.114 * input._flow_data[..., 2]
        input._flow_data[...,1] = input._flow_data[...,0]
        input._flow_data[...,2] = input._flow_data[...,0]

        input_data = np.expand_dims(input._flow_data, axis=0)
        input_data = np.array(input_data, dtype=int)

        self.input_type = np.ndarray
        output = super().inference(input_data)
        self.input_type = ImageLand

        input.undo()
        return output

