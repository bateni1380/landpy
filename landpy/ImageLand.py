from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union
import time
from landpy import Debugger, Land
from landpy import MarksLand, QuadMarks, SquareMarks, HeatLand, Filter
from landpy.heat import zero_heat, full_heat
import numpy as np
import cv2
import inspect
import matplotlib.pyplot as plt
from PIL import Image
np.row_stack


class ImageLand(Land):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)

    @Debugger.method_decorator(['ImageLand.set_data', 'ImageLand'], 'Sending image to flow_data')
    def set_image(self, image: np.ndarray):
        if type(image) is Image:
            image = np.array(image)

        self._debugger.log_image(
            image, 'Sent Image', categories=self._debugger.get_last_log_categories())

        if image.dtype in ('float32', 'float64'):
            image = (image*255)
        self._set_data(image.astype('uint8'))

        return self

    @Debugger.method_decorator(['ImageLand.get_image', 'ImageLand'], 'Returning image!')
    def get_image(self):
        return self._get_data()

    @Debugger.method_decorator(['ImageLand.show', 'ImageLand'], 'Showing the image')
    def show(self):
        Debugger(visual_debug=True).log_image(self._flow_data)

    @Debugger.method_decorator(['ImageLand.rotate', 'ImageLand'], 'Rotating image')
    def rotate(self, angle: float, center: Tuple = None, save_trace=True):
        height, width = self._flow_data.shape[:2]
        if center == None:
            center = (int(width/2), int(height/2))

        self._debugger.log('(angle, center) = ({}, {})'.format(
            angle, center), categories=self._debugger.get_last_log_categories())

        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)

        self._debugger.save_image(
            self._flow_data, 'Image before rotation!', categories=self._debugger.get_last_log_categories())
        self._flow_data = cv2.warpAffine(src=self._flow_data, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        self._debugger.save_image(
            self._flow_data, 'Image after rotation!', categories=self._debugger.get_last_log_categories())

        if save_trace:
            self._make_trace(ImageLand.__rotate_back, {
                             'angle': angle, 'center': center})
        else:
            self._make_trace(Land._make_exception, {})

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

    @Debugger.static_method_decorator(['ImageLand.__rotate_back', 'ImageLand'], 'Rotating back image')
    def __rotate_back(image, pars: Dict, debugger: Debugger):
        debugger.log('(angle, center) = ({}, {})'
                     .format(-pars['angle'], pars['center']), categories=debugger.get_last_log_categories())

        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(
            center=pars['center'], angle=-pars['angle'], scale=1)

        debugger.save_image(
            image, 'Image before rotation back!', categories=debugger.get_last_log_categories())
        image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        debugger.save_image(
            image, 'Image after rotation back!', categories=debugger.get_last_log_categories())

        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return image

    @Debugger.method_decorator(['ImageLand.reshape', 'ImageLand'], 'Reshaping image')
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
            self._flow_data, 'Image before reshape!', categories=self._debugger.get_last_log_categories())
        image = image.resize((width, height))
        image = image.convert(mode)
        image = np.array(image, dtype='uint8')
        self._flow_data = image
        self._debugger.save_image(
            self._flow_data, 'Image after reshape!', categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

    @Debugger.static_method_decorator(['ImageLand.reshape', 'ImageLand'], 'Reshaping back image')
    def __reshape_back(image, pars: Dict, debugger: Debugger):
        debugger.log('(w, h, mode) = ({}, {}, {})'.format(
            pars['width'], pars['height'], pars['mode']), categories=debugger.get_last_log_categories())

        debugger.save_image(image, 'Image before reshape!',
                            categories=debugger.get_last_log_categories())
        image = Image.fromarray(image)
        image = image.resize((pars['width'], pars['height']))
        image = image.convert(pars['mode'])
        image = np.array(image, dtype='uint8')
        debugger.save_image(image, 'Image after reshape!',
                            categories=debugger.get_last_log_categories())

        debugger.log_end(categories=debugger.get_last_log_categories())
        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return image

    def zoom(self, quadrilateral: Union[QuadMarks, SquareMarks], save_trace=True):
        '''        log_categories = ['ImageLand.reshape', 'ImageLand']
        self._debugger.log_start('Reshaping image (w, h, mode) = ({}, {}, {})'.format(
            width, height, mode), categories=log_categories)

        if save_trace:
            self._make_trace(ImageLand.__reshape_back, {'width': image.width,
                                                        'height': image.height,
                                                        'mode': image.mode})
        if type(quadrilateral) is SquareMarks:'''
        pass

    def __zoom_back(self, pars, save_trace=True):
        pass  # challenge

    def implant_filter(self, filter: Filter, heat: HeatLand = None):
        if heat is None:
            heat = full_heat(self.flow_data.shape[0], self.flow_data.shape[1])
        pass

    def implant_image(self, src_land: MarksLand, plant_land: MarksLand, plant_img: np.ndarray):
        pass  # challenge
