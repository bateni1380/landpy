from landpy import Debugger, Land
import numpy as np

class HeatLand(Land):
    def __init__(self, heat = None,  debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if heat is not None:
            self.set_heat(heat)


    @Debugger.method_decorator('Sending heat to flow_data', ['HeatLand.set_data', 'HeatLand'])
    def set_heat(self, heat: np.ndarray):
        if type(heat) is not np.ndarray:
            image = np.array(image)

        self._debugger.log_image(
            image, 'Sent Image', categories=self._debugger.get_last_log_categories())

        if image.dtype in ('float32', 'float64'):
            image = (image*255)
        super()._set_data(image.astype('uint8'))

        return self