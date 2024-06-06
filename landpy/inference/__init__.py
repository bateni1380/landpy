import tensorflow as tf
import numpy as np
from landpy import Debugger


@Debugger.static_method_decorator('Loading a tflite model', ['inference.load_tflite', 'inference'])
def load_tflite(model_path, debugger: Debugger = Debugger(debug=False)):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@Debugger.static_method_decorator('Inferencing by a tflite model', ['inference.inference_by_tflite', 'inference'])
def inference_by_tflite(interpreter, input, debugger: Debugger = Debugger(debug=False)):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    if len(output_data)==1:
        output_data = output_data[0]

    return output_data

@Debugger.static_method_decorator('Loading a tf model', ['inference.load_tf', 'inference'])
def load_tf(model_path, debugger: Debugger = Debugger(debug=False)):
    interpreter = tf.keras.models.load_model(model_path)
    return interpreter

@Debugger.static_method_decorator('Inferencing by a tf model', ['inference.inference_by_tf', 'inference'])
def inference_by_tf(interpreter, input, debugger: Debugger = Debugger(debug=False)):
    output_data = interpreter.predict(input)
    return output_data

