import tensorflow as tf
import numpy as np
from landpy import ImageLand, Debugger

class InferenceType:
    def __init__(self, shape, dtype, does_divide):
        self.shape = shape
        self.dtype = dtype
        self.does_divide = does_divide
    
    def convert(self, input):
        input = np.array(input, dtype=np.float64)
        if self.does_divide:
            input = input/255
        input = input.reshape(self.shape)
        input = input.astype(self.dtype)
        return input

@Debugger.static_method_decorator('Loading a tflite model', ['inference.load_tflite', 'inference'])
def load_tflite(model_path, debugger: Debugger = Debugger(debug=False)):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@Debugger.static_method_decorator('Inferencing by a tflite model', ['inference.inference_by_tflite', 'inference'])
def inference_by_tflite(interpreter, input_image, input_type, debugger: Debugger = Debugger(debug=False)):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input = input_image._flow_data if type(input_image) is ImageLand else input_image
    input = input_type.convert(input)
    
    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

