from typing import Any
from landpy.debugger import Debugger
from landpy.model import Model
import tensorflow as tf
import numpy as np

class TFModel(Model):
    model_path: str # Inherited from Model
    _interpreter: Any # Inherited from Model
    
    def __init__(self, model_url=None, input_type=None, output_type=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(model_url, input_type, output_type, debugger)
        
    @Debugger.method_decorator('Loading model')
    def _load_model(self):
        return tf.keras.models.load_model(self.model_path)

    @Debugger.method_decorator('Inferencing by model')
    def inference(self, input):
        if self.input_type is not None and type(input) is not self.input_type:
            self._debugger.log_exception("inference input type must be " + str(self.input_type))
        
        output_data = self._interpreter.predict(input)
        return output_data