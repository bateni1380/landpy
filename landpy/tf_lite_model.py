"""Optional TensorFlow Lite inference, including quantized tensors."""

import numpy as np

from .debugger import Debugger
from .model import Model
from .tf_model import tensorflow


class TFLiteModel(Model):
    """Execute a single-input Lite model; return one array or output list.

    Integer inputs are passed through; floating inputs to quantized models are
    quantized using tensor metadata. Quantized outputs are dequantized.
    """

    def _load_model(self):
        interpreter = tensorflow().lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        return interpreter

    @Debugger.method_decorator("Lite inference")
    def inference(self, input):
        """Run one input tensor using its declared shape and dtype."""
        self._validate_input(input)
        details = self._interpreter.get_input_details()
        if len(details) != 1:
            raise ValueError("TFLiteModel supports exactly one input tensor.")
        detail = details[0]
        data = np.asarray(input)
        if tuple(data.shape) != tuple(detail["shape"]):
            raise ValueError(f"Expected input shape {tuple(detail['shape'])}, got {data.shape}.")
        dtype = detail["dtype"]
        scale, zero = detail["quantization"]
        if np.issubdtype(dtype, np.integer) and np.issubdtype(data.dtype, np.floating):
            if not scale:
                raise ValueError("Cannot quantize input without a positive scale.")
            limits = np.iinfo(dtype)
            data = np.clip(np.rint(data / scale + zero), limits.min, limits.max)
        self._interpreter.set_tensor(detail["index"], data.astype(dtype))
        self._interpreter.invoke()
        outputs = []
        for output in self._interpreter.get_output_details():
            value = self._interpreter.get_tensor(output["index"])
            scale, zero = output["quantization"]
            if scale and np.issubdtype(value.dtype, np.integer):
                value = (value.astype(np.float32) - zero) * scale
            outputs.append(value)
        return outputs[0] if len(outputs) == 1 else outputs
