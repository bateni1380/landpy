from landpy import Debugger, Land
import numpy as np


class HeatLand(Land):
    def __init__(self, heat=None,  debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if heat is not None:
            HeatLand.set_heat(self, heat)

    @Debugger.method_decorator('Sending heat to flow_data', ['HeatLand.set_data', 'HeatLand'])
    def set_heat(self, heat: np.ndarray):
        if type(heat) is not np.ndarray:
            self._debugger.log_exception('heat must be an ndarray!',
                                         categories=['Exception']+self._debugger.get_last_log_categories())
        if len(heat.shape) != 2:
            self._debugger.log_exception('heat must be a 2darray!',
                                         categories=['Exception']+self._debugger.get_last_log_categories())

        self._debugger.log_heat(
            heat, 'Sent Heat', categories=self._debugger.get_last_log_categories())

        super()._set_data(heat)

        return self
    
    @Debugger.method_decorator('Showing the heat', ['HeatLand.show', 'HeatLand'])
    def show(self):
        Debugger(visual_debug=True).log_heat(self._flow_data)

    @Debugger.method_decorator('Sending heat to flow_data', ['HeatLand.set_data', 'HeatLand'])
    def __add__(self, other):
        pass
# TODO: add, sub, multiple, power, minus, zoom, crop, rotate