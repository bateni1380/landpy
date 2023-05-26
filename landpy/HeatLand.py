from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union
from landpy import Debugger, Land
import math
import numpy as np
from landpy import QuadMarks, SquareMarks, Filter


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
            heat, 'Set Heat', categories=self._debugger.get_last_log_categories())

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

    @Debugger.method_decorator('Zooming in heat', ['HeatLand.zoom', 'HeatLand'])
    def zoom(self, quadrilateral: Union[QuadMarks, SquareMarks], save_trace: bool = True, fill_heat=0):
        if save_trace:
            self._make_trace(HeatLand.__zoom_back, {'last_heat': self._flow_data.copy(),
                                                    'quadrilateral': quadrilateral.copy()})

        self._debugger.save_heat(self._flow_data, 'Heat before zoom',
                                 categories=self._debugger.get_last_log_categories())

        if type(quadrilateral) is SquareMarks:
            minX = math.floor(quadrilateral.minX * self._flow_data.shape[0])
            maxX = math.ceil(quadrilateral.maxX * self._flow_data.shape[0])
            minY = math.floor(quadrilateral.minY * self._flow_data.shape[1])
            maxY = math.ceil(quadrilateral.maxY * self._flow_data.shape[1])
            if minX >= 0 and maxX <= self._flow_data.shape[0] and minY >= 0 and maxY <= self._flow_data.shape[1]:
                self._flow_data = self._flow_data[minX:maxX, minY:maxY]
            else:
                def genX(x):
                    return np.ones([x, self._flow_data.shape[1]], dtype='float32')*fill_heat
                self._flow_data = np.vstack(
                    [genX(max(0, -minX)), self._flow_data, genX(max(0, maxX-self._flow_data.shape[0]))])

                def genY(x):
                    return np.ones([self._flow_data.shape[0], x], dtype='float32')*fill_heat
                self._flow_data = np.hstack(
                    [genY(max(0, -minY)), self._flow_data, genY(max(0, maxY-self._flow_data.shape[1]))])

                x_len = maxX+max(0, -minX)
                y_len = maxY+max(0, -minY)

                self._flow_data = self._flow_data[max(0, minX):max(
                    0, minX)+x_len, max(0, minY):max(0, minY)+y_len].astype('float32')

        elif type(quadrilateral) is QuadMarks:
            self._debugger.log_exception("Zooming in image using QuadMarks is not supported yet!",
                                         "Exception"+self._debugger.get_last_log_categories())

        self._debugger.save_heat(self._flow_data, 'Image after zoom',
                                 categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_heats(
            categories=self._debugger.get_last_log_categories())

    # TODO: zoom back in images using QuadMarks
    @Debugger.static_method_decorator('Zooming back heat', ['HeatLand.__zoom_back', 'HeatLand'])
    def __zoom_back(heat, pars: Dict, debugger: Debugger):
        last_heat = pars['last_heat']
        quadrilateral = pars['quadrilateral']

        debugger.save_heat(heat, 'Heat before zoom back',
                           categories=debugger.get_last_log_categories())
        if type(quadrilateral) is SquareMarks:
            minX = math.floor(quadrilateral.minX * last_heat.shape[0])
            maxX = math.ceil(quadrilateral.maxX * last_heat.shape[0])
            minY = math.floor(quadrilateral.minY * last_heat.shape[1])
            maxY = math.ceil(quadrilateral.maxY * last_heat.shape[1])

            x_len = min(maxX, last_heat.shape[0])-max(0, minX)
            y_len = min(maxY, last_heat.shape[1])-max(0, minY)

            last_heat[max(0, minX):max(0, minX)+x_len, max(0, minY):max(0, minY) +
                      y_len] = heat[max(0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY)+y_len]

        elif type(quadrilateral) is QuadMarks:
            debugger.log_exception("Zooming back in heat using QuadMarks is not supported yet!",
                                   "Exception"+debugger.get_last_log_categories())

        debugger.save_heat(pars['last_heat'], 'Heat after zoom back',
                            categories=debugger.get_last_log_categories())

        debugger.log_saved_heats(
            categories=debugger.get_last_log_categories())

        return last_heat

# TODO: zoom, crop, rotate
