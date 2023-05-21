from landpy import Debugger, Land
import numpy as np


class HeatLand(Land):
    def __init__(self, heat=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if heat is not None:
            HeatLand.set_heat(self, heat)

    @Debugger.method_decorator('Setting heat', ['HeatLand.set_data', 'HeatLand'])
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

    @Debugger.method_decorator('Adding two heats (maximums in each place)', ['HeatLand.__add__', 'HeatLand'])
    def __add__(self, other):
        if not isinstance(other, HeatLand):
            self._debugger.log_exception("Unsupported operand type for *: {} and MyClass".format(
                type(other)), categories=['Exception']+self._debugger.get_last_log_categories())
        return HeatLand(np.maximum(self._flow_data, other._flow_data), self._debugger)

    @Debugger.method_decorator('Substracting two heats (minimums in each place)', ['HeatLand.__sub__', 'HeatLand'])
    def __sub__(self, other):
        if not isinstance(other, HeatLand):
            self._debugger.log_exception("Unsupported operand type for *: {} and MyClass".format(
                type(other)), categories=['Exception']+self._debugger.get_last_log_categories())
        return HeatLand(np.minimum(self._flow_data, other._flow_data), self._debugger)

    @Debugger.method_decorator('Multiplying a number to a heats', ['HeatLand.__rmul__', 'HeatLand'])
    def __rmul__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            self._debugger.log_exception("Unsupported operand type for *: {} and MyClass".format(
                type(other)), categories=['Exception']+self._debugger.get_last_log_categories())
            
        return HeatLand(np.maximum(np.minimum(other*self._flow_data, 1), 0), self._debugger)

    @Debugger.method_decorator('Powerring a heat to a number', ['HeatLand.__pow__', 'HeatLand'])
    def __pow__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            self._debugger.log_exception("Unsupported operand type for *: {} and MyClass".format(
                type(other)), categories=['Exception']+self._debugger.get_last_log_categories())
            
        return HeatLand(np.maximum(np.minimum(self._flow_data**other, 1), 0), self._debugger)
    
# TODO: zoom, crop, rotate
