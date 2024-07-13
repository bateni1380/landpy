from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union
from landpy import Debugger, Land
import cv2
import math
import numpy as np
from PIL import Image
from landpy import QuadMarks, SquareMarks, Filter


class HeatLand(Land):
    def __init__(self, heat=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if heat is not None:
            HeatLand.set_heat(self, heat)

    @Debugger.method_decorator('Setting heat', ['HeatLand.set_data', 'HeatLand'])
    def set_heat(self, heat: np.ndarray):
        heat = np.array(heat)
        if len(heat.shape) != 2:
            self._debugger.log_exception('heat must be a 2darray!')

        self._debugger.log_image(heat, 'Set Heat', Debugger.LogTypes.HEAT,
                                 )

        super()._set_data(heat)

        return self

    @Debugger.method_decorator('Showing the heat', ['HeatLand.show', 'HeatLand'])
    def show(self):
        Debugger(visual_debug=True).log_image(
            self._flow_data, '', Debugger.LogTypes.HEAT)
        return self

    @Debugger.method_decorator('Adding two heats (maximums in each place)', ['HeatLand.__add__', 'HeatLand'])
    def __add__(self, other):
        if not isinstance(other, HeatLand):
            self._debugger.log_exception(
                "Unsupported operand type for *: {} and MyClass".format(type(other)))
        self._debugger.save_image(
            self._flow_data, 'Left', Debugger.LogTypes.HEAT)
        self._debugger.save_image(
            other._flow_data, 'Right', Debugger.LogTypes.HEAT)

        result = HeatLand(np.maximum(
            self._flow_data, other._flow_data), self._debugger)

        self._debugger.save_image(
            result._flow_data, 'Result', Debugger.LogTypes.HEAT)
        self._debugger.log_saved_images()
        return result

    @Debugger.method_decorator('Substracting two heats (minimums in each place)', ['HeatLand.__sub__', 'HeatLand'])
    def __sub__(self, other):
        if not isinstance(other, HeatLand):
            self._debugger.log_exception(
                "Unsupported operand type for *: {} and MyClass".format(type(other)))
        self._debugger.save_image(
            self._flow_data, 'Left', Debugger.LogTypes.HEAT)
        self._debugger.save_image(
            other._flow_data, 'Right', Debugger.LogTypes.HEAT)

        result = HeatLand(np.minimum(
            self._flow_data, other._flow_data), self._debugger)

        self._debugger.save_image(
            result._flow_data, 'Result', Debugger.LogTypes.HEAT)
        self._debugger.log_saved_images()
        return result

    @Debugger.method_decorator('Negating a heat (p -> 1-p)', ['HeatLand.__neg__', 'HeatLand'])
    def __neg__(self):
        self._debugger.save_image(
            self._flow_data, 'Before', Debugger.LogTypes.HEAT)

        result = HeatLand(1-self._flow_data, self._debugger)

        self._debugger.save_image(
            result._flow_data, 'After', Debugger.LogTypes.HEAT)
        self._debugger.log_saved_images(
        )
        return result

    @Debugger.method_decorator('Multiplying a number to a heats', ['HeatLand.__rmul__', 'HeatLand'])
    def __rmul__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            self._debugger.log_exception(
                "Unsupported operand type for *: {} and MyClass".format(type(other)))

        return HeatLand(np.maximum(np.minimum(other*self._flow_data, 1), 0), self._debugger)

    @Debugger.method_decorator('Powerring a heat to a number', ['HeatLand.__pow__', 'HeatLand'])
    def __pow__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            self._debugger.log_exception(
                "Unsupported operand type for *: {} and MyClass".format(type(other)))

        return HeatLand(np.maximum(np.minimum(self._flow_data**other, 1), 0), self._debugger)

    @Debugger.method_decorator('Rotating heat', ['HeatLand.rotate', 'HeatLand'])
    def rotate(self, angle: float, center: Tuple = None, save_trace = False, fill_color=0):
        height, width = self._flow_data.shape[:2]
        if center == None:
            center = (int(width/2), int(height/2))

        self._debugger.log('(angle, center) = ({}, {})'.format(
            angle, center), )

        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)

        self._debugger.save_image(
            self._flow_data, 'Heat before rotation', Debugger.LogTypes.HEAT, )
        self._flow_data = cv2.warpAffine(src=self._flow_data, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)
        self._debugger.save_image(
            self._flow_data, 'Heat after rotation', Debugger.LogTypes.HEAT, )

        if save_trace:
            self._make_trace(HeatLand.__rotate_back, {
                             'angle': angle, 'center': center})

        self._debugger.log_saved_images(
        )

        return self

    @Debugger.static_method_decorator('Rotating back heat', ['HeatLand.__rotate_back', 'HeatLand'])
    def __rotate_back(heat, pars: Dict, debugger: Debugger):
        debugger.log('(angle, center) = ({}, {})'
                     .format(-pars['angle'], pars['center']))

        height, width = heat.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(
            center=pars['center'], angle=-pars['angle'], scale=1)

        debugger.save_image(
            image, 'Heat before rotation back', Debugger.LogTypes.HEAT)
        image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        debugger.save_image(
            image, 'Heat after rotation back', Debugger.LogTypes.HEAT)

        debugger.log_saved_images()

        return heat

    @Debugger.method_decorator('Reshaping heat', ['HeatLand.reshape', 'HeatLand'])
    def reshape(self, shape, save_trace = False):
        width, height = shape[1], shape[0]
        self._debugger.log('(w, h) = ({}, {})'.format(
            width, height), )

        heat = Image.fromarray((self._flow_data*255).astype('uint8'))

        if save_trace:
            self._make_trace(HeatLand.__reshape_back, {'width': heat.width,
                                                       'height': heat.height})

        self._debugger.save_image(
            self._flow_data, 'Heat before reshape', Debugger.LogTypes.HEAT, )
        heat = heat.resize((width, height))
        heat = np.array(heat, dtype='float32')/255
        self.set_heat(heat)
        self._debugger.save_image(
            self._flow_data, 'Heat after reshape', Debugger.LogTypes.HEAT, )

        self._debugger.log_saved_images(
        )

        return self

    @Debugger.static_method_decorator('Reshaping back heat', ['HeatLand.__reshape_back', 'HeatLand'])
    def __reshape_back(heat, pars: Dict, debugger: Debugger):
        debugger.log('(w, h) = ({}, {})'.format(pars['width'], pars['height']))

        debugger.save_image(heat, 'Heat before reshape',
                            Debugger.LogTypes.HEAT)
        heat = Image.fromarray((heat*255).astype('uint8'))
        heat = heat.resize((pars['width'], pars['height']))
        heat = np.array(heat, dtype='float32')/255
        debugger.save_image(heat, 'Heat after reshape', Debugger.LogTypes.HEAT)

        debugger.log_saved_images()

        return heat

    @ Debugger.method_decorator('Zooming in heat', ['HeatLand.zoom', 'HeatLand'])
    def zoom(self, quadrilateral: Union[QuadMarks, SquareMarks], save_trace = False, fill_heat=0):
        if save_trace:
            self._make_trace(HeatLand.__zoom_back, {'last_heat': self._flow_data.copy(),
                                                    'quadrilateral': quadrilateral.copy(save_trace=False)})

        self._debugger.save_image(self._flow_data, 'Heat before zoom', Debugger.LogTypes.HEAT,
                                  )

        if type(quadrilateral) is SquareMarks:
            minX = math.floor(quadrilateral.minX() * self._flow_data.shape[0])
            maxX = math.ceil(quadrilateral.maxX() * self._flow_data.shape[0])
            minY = math.floor(quadrilateral.minY() * self._flow_data.shape[1])
            maxY = math.ceil(quadrilateral.maxY() * self._flow_data.shape[1])
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
            self._debugger.log_exception(
                "Zooming in image using QuadMarks is not supported yet!")

        self._debugger.save_image(
            self._flow_data, 'Image after zoom', Debugger.LogTypes.HEAT)

        self._debugger.log_saved_images()

        return self

    @Debugger.static_method_decorator('Zooming back heat', ['HeatLand.__zoom_back', 'HeatLand'])
    def __zoom_back(heat, pars: Dict, debugger: Debugger):
        last_heat = pars['last_heat']
        quadrilateral = pars['quadrilateral']

        debugger.save_image(heat, 'Heat before zoom back',
                            Debugger.LogTypes.HEAT)
        if type(quadrilateral) is SquareMarks:
            minX = math.floor(quadrilateral.minX() * last_heat.shape[0])
            maxX = math.ceil(quadrilateral.maxX() * last_heat.shape[0])
            minY = math.floor(quadrilateral.minY() * last_heat.shape[1])
            maxY = math.ceil(quadrilateral.maxY() * last_heat.shape[1])

            x_len = min(maxX, last_heat.shape[0])-max(0, minX)
            y_len = min(maxY, last_heat.shape[1])-max(0, minY)

            last_heat[max(0, minX):max(0, minX)+x_len, max(0, minY):max(0, minY) +
                      y_len] = heat[max(0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY)+y_len]

        elif type(quadrilateral) is QuadMarks:
            debugger.log_exception(
                "Zooming back in heat using QuadMarks is not supported yet!")

        debugger.save_image(
            pars['last_heat'], 'Heat after zoom back', Debugger.LogTypes.HEAT)

        debugger.log_saved_images()

        return last_heat
