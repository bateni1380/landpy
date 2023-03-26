from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union
import time
from landpy import Debugger, Land
from landpy import MarksLand, QuadMarks, SquareMarks, HeatLand, Filter
from landpy.heat import zero_heat, full_heat
import numpy as np
import cv2
import inspect
import matplotlib.pyplot as plt
import math
from PIL import Image


class ImageLand(Land):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        self._debugger = debugger

    @Debugger.method_decorator('Sending image to flow_data', ['ImageLand.set_data', 'ImageLand'])
    def set_image(self, image: np.ndarray):
        if type(image) is Image:
            image = np.array(image)

        self._debugger.log_image(
            image, 'Sent Image', categories=self._debugger.get_last_log_categories())

        if image.dtype in ('float32', 'float64'):
            image = (image*255)
        super()._set_data(image.astype('uint8'))

        return self

    @Debugger.method_decorator('Returning image!', ['ImageLand.get_image', 'ImageLand'])
    def get_image(self):
        return self._get_data()

    @Debugger.method_decorator('Showing the image', ['ImageLand.show', 'ImageLand'])
    def show(self):
        Debugger(visual_debug=True).log_image(self._flow_data)

    @Debugger.method_decorator('Rotating image', ['ImageLand.rotate', 'ImageLand'])
    def rotate(self, angle: float, center: Tuple = None, save_trace=True):
        height, width = self._flow_data.shape[:2]
        if center == None:
            center = (int(width/2), int(height/2))

        self._debugger.log('(angle, center) = ({}, {})'.format(
            angle, center), categories=self._debugger.get_last_log_categories())

        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)

        self._debugger.save_image(
            self._flow_data, 'Image before rotation', categories=self._debugger.get_last_log_categories())
        self._flow_data = cv2.warpAffine(src=self._flow_data, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        self._debugger.save_image(
            self._flow_data, 'Image after rotation', categories=self._debugger.get_last_log_categories())

        if save_trace:
            self._make_trace(ImageLand.__rotate_back, {
                             'angle': angle, 'center': center})
        else:
            self._make_trace(Land._make_exception, {})

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

    @Debugger.static_method_decorator('Rotating back image', ['ImageLand.__rotate_back', 'ImageLand'])
    def __rotate_back(image, pars: Dict, debugger: Debugger):
        debugger.log('(angle, center) = ({}, {})'
                     .format(-pars['angle'], pars['center']), categories=debugger.get_last_log_categories())

        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(
            center=pars['center'], angle=-pars['angle'], scale=1)

        debugger.save_image(
            image, 'Image before rotation back', categories=debugger.get_last_log_categories())
        image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        debugger.save_image(
            image, 'Image after rotation back', categories=debugger.get_last_log_categories())

        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return image

    @Debugger.method_decorator('Reshaping image', ['ImageLand.reshape', 'ImageLand'])
    def reshape(self, width, height, mode='RGB', save_trace=True):
        self._debugger.log('(w, h, mode) = ({}, {}, {})'.format(
            width, height, mode), categories=self._debugger.get_last_log_categories())

        image = Image.fromarray(self._flow_data)

        if save_trace:
            self._make_trace(ImageLand.__reshape_back, {'width': image.width,
                                                        'height': image.height,
                                                        'mode': image.mode})
        else:
            self._make_trace(Land._make_exception, {})

        self._debugger.save_image(
            self._flow_data, 'Image before reshape', categories=self._debugger.get_last_log_categories())
        image = image.resize((width, height))
        image = image.convert(mode)
        image = np.array(image, dtype='uint8')
        self._flow_data = image
        self._debugger.save_image(
            self._flow_data, 'Image after reshape', categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

    @Debugger.static_method_decorator('Reshaping back image', ['ImageLand.__reshape_back', 'ImageLand'])
    def __reshape_back(image, pars: Dict, debugger: Debugger):
        debugger.log('(w, h, mode) = ({}, {}, {})'.format(
            pars['width'], pars['height'], pars['mode']), categories=debugger.get_last_log_categories())

        debugger.save_image(image, 'Image before reshape',
                            categories=debugger.get_last_log_categories())
        image = Image.fromarray(image)
        image = image.resize((pars['width'], pars['height']))
        image = image.convert(pars['mode'])
        image = np.array(image, dtype='uint8')
        debugger.save_image(image, 'Image after reshape',
                            categories=debugger.get_last_log_categories())

        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return image

    @Debugger.method_decorator('Zooming in image', ['ImageLand.zoom', 'ImageLand'])
    def zoom(self, quadrilateral: Union[QuadMarks, SquareMarks], save_trace=True, fill_color=255):

        if save_trace:
            self._make_trace(ImageLand.__zoom_back, {'last_image': self._flow_data.copy(),
                                                     'quadrilateral': quadrilateral.copy()})

        self._debugger.save_image(self._flow_data, 'Image before zoom',
                                  categories=self._debugger.get_last_log_categories())
        if type(quadrilateral) is SquareMarks:
            minX = math.floor(quadrilateral.minX * self._flow_data.shape[0])
            maxX = math.ceil(quadrilateral.maxX * self._flow_data.shape[0])
            minY = math.floor(quadrilateral.minY * self._flow_data.shape[1])
            maxY = math.ceil(quadrilateral.maxY * self._flow_data.shape[1])

            if minX >= 0 and maxX < self._flow_data[0] and minY >= 0 and maxY < self._flow_data[1]:
                self._flow_data = self._flow_data[minX:maxX, minY:maxY]
            else:
                # pad_shape = (maxX-minX, maxY-minY,self._flow_data.shape[2])
                # img_pad = np.zeros(pad_shape, dtype='uint8')*fill_color

                # x_len = min(maxX, org_shape[0])-max(minX, 0)
                # y_len = min(maxY, org_shape[1])-max(minY, 0)

                # img_pad[max(0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY) +
                #         y_len] = self._flow_data[max(minX, 0):max(minX, 0)+x_len, max(minY, 0):max(minY, 0)+y_len]

                def genX(x):
                    return np.ones([x, self._flow_data.shape[1], self._flow_data.shape[2]], dtype='uint8')*fill_color
                self._flow_data = np.vstack(
                    [genX(max(0, -minX)), self._flow_data, genX(max(0, maxX-self._flow_data.shape[0]))])

                def genY(x):
                    return np.ones([self._flow_data.shape[0], x, self._flow_data.shape[2]], dtype='uint8')*fill_color
                self._flow_data = np.hstack(
                    [genY(max(0, -minY)), self._flow_data, genY(max(0, maxY-self._flow_data.shape[1]))])

                x_len = maxX+max(0,-minX)
                y_len = maxY+max(0,-minY)

                print(max(0, minY),max(0, minY)+y_len)
                self._flow_data = self._flow_data[max(0, minX):max(
                    0, minX)+x_len, max(0, minY):max(0, minY)+y_len]

                # pad_shape =
                # self._flow_data = np.stack(np.zeros(pad_shape, dtype='uint8')*fill_color)
                # self._flow_data = img_pad

        elif type(quadrilateral) is QuadMarks:
            pass
        self._debugger.save_image(self._flow_data, 'Image after zoom',
                                  categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

    def __zoom_back(self, pars, save_trace=True):
        # print(pars['quadrilateral']._flow_data)
        pass  # challenge

    def implant_filter(self, filter: Filter, heat: HeatLand = None):
        if heat is None:
            heat = full_heat(self.flow_data.shape[0], self.flow_data.shape[1])
        pass

    def implant_image(self, src_land: MarksLand, plant_land: MarksLand, plant_img: np.ndarray):
        pass  # challenge
