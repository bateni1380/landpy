from typing import Any
from landpy.debugger import Debugger
from landpy.model import Model
import tensorflow as tf
import numpy as np

class TFLiteModel(Model):
    model_path: str # Inherited from Model
    _interpreter: Any # Inherited from Model
    
    def __init__(self, model_url=None, input_type=None, output_type=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(model_url, input_type, output_type, debugger)
        
    @Debugger.method_decorator('Loading model')
    def _load_model(self):
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        return interpreter

    @Debugger.method_decorator('Inferencing by model')
    def inference(self, input):
        if self.input_type is not None and type(input) is not self.input_type:
            self._debugger.log_exception("inference input type must be " + str(self.input_type))
        
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        
        self._interpreter.set_tensor(input_details[0]['index'], input)
        self._interpreter.invoke()
        output_data = [self._interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if len(output_data)==1:
            output_data = output_data[0]

        return output_data