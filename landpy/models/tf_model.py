from landpy import ImageLand, Debugger
from landpy.inference import inference_by_tf, load_tf
import tensorflow as tf

class TFModel:
    def __init__(self, model_path=None, input_type=None, output_type=None, debugger: Debugger = Debugger(debug=False)):
        self._debugger = debugger
        self._model_path = model_path
        self.input_type = input_type
        self.output_type = output_type
        self._interpreter = None
        if model_path is not None:
            if model_path.split('.')[-1].lower() == 'h5':
                self._interpreter = load_tf(model_path, self._debugger)
            else:
                self._debugger.log_exception('File type must be tflite!',  ['Model.__init__', 'Model'])

    @Debugger.method_decorator('Inferencing by model', ['Model.inference', 'Model'])
    def inference(self, input):
        if self.input_type is not None and type(input) is not self.input_type:
            self._debugger.log_exception(
                "inference input type must be " + str(self.input_type), self._debugger.get_last_log_categories())
        
        return inference_by_tf(self._interpreter, input, self._debugger)
