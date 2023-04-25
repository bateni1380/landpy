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

# TODO: histogram_equalize
class ImageLand(Land):
    def __init__(self, image = None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if image is not None:
            self.set_image(image)


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
    def rotate(self, angle: float, center: Tuple = None, save_trace: bool = True, fill_color = (0, 0, 0)):
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
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)
        self._debugger.save_image(
            self._flow_data, 'Image after rotation', categories=self._debugger.get_last_log_categories())

        if save_trace:
            self._make_trace(ImageLand.__rotate_back, {
                             'angle': angle, 'center': center})

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
    def reshape(self, width, height, mode='RGB', save_trace: bool = True):
        self._debugger.log('(w, h, mode) = ({}, {}, {})'.format(
            width, height, mode), categories=self._debugger.get_last_log_categories())

        image = Image.fromarray(self._flow_data)

        if save_trace:
            self._make_trace(ImageLand.__reshape_back, {'width': image.width,
                                                        'height': image.height,
                                                        'mode': image.mode})

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

    #TODO: zoom in images using QuadMarks
    @Debugger.method_decorator('Zooming in image', ['ImageLand.zoom', 'ImageLand'])
    def zoom(self, quadrilateral: Union[QuadMarks, SquareMarks], save_trace: bool = True, fill_color = (0, 0, 0)):
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
            if minX >= 0 and maxX <= self._flow_data.shape[0] and minY >= 0 and maxY <= self._flow_data.shape[1]:
                self._flow_data = self._flow_data[minX:maxX, minY:maxY]
            else:
                def genX(x):
                    return np.ones([x, self._flow_data.shape[1], self._flow_data.shape[2]], dtype='uint8')*fill_color
                self._flow_data = np.vstack(
                    [genX(max(0, -minX)), self._flow_data, genX(max(0, maxX-self._flow_data.shape[0]))])

                def genY(x):
                    return np.ones([self._flow_data.shape[0], x, self._flow_data.shape[2]], dtype='uint8')*fill_color
                self._flow_data = np.hstack(
                    [genY(max(0, -minY)), self._flow_data, genY(max(0, maxY-self._flow_data.shape[1]))])

                x_len = maxX+max(0, -minX)
                y_len = maxY+max(0, -minY)

                self._flow_data = self._flow_data[max(0, minX):max(
                    0, minX)+x_len, max(0, minY):max(0, minY)+y_len].astype('uint8')

        elif type(quadrilateral) is QuadMarks:
            self._debugger.log_exception("Zooming in image using QuadMarks is not supported yet!",
                               "Exception"+self._debugger.get_last_log_categories())

        self._debugger.save_image(self._flow_data, 'Image after zoom',
                                  categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

    #TODO: zoom back in images using QuadMarks
    @Debugger.static_method_decorator('Zooming back image', ['ImageLand.__zoom_back', 'ImageLand'])
    def __zoom_back(image, pars: Dict, debugger: Debugger):
        last_image = pars['last_image']
        quadrilateral = pars['quadrilateral']

        debugger.save_image(image, 'Image before zoom back',
                            categories=debugger.get_last_log_categories())
        if type(quadrilateral) is SquareMarks:
            minX = math.floor(quadrilateral.minX * last_image.shape[0])
            maxX = math.ceil(quadrilateral.maxX * last_image.shape[0])
            minY = math.floor(quadrilateral.minY * last_image.shape[1])
            maxY = math.ceil(quadrilateral.maxY * last_image.shape[1])

            x_len = min(maxX, last_image.shape[0])-max(0, minX)
            y_len = min(maxY, last_image.shape[1])-max(0, minY)

            if last_image.shape[2] == 4:
                mask = image[max(
                    0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY)+y_len, 3] == 0
                mask = np.stack([mask,mask,mask,mask], axis=2)
                last_image[max(0, minX):max(0, minX)+x_len, max(0, minY):max(0, minY)+y_len] = image[max(
                    0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY)+y_len]*(1-mask) + last_image[max(
                        0, minX):max(0, minX)+x_len, max(0, minY):max(0, minY)+y_len]*(mask)
            else:
                last_image[max(0, minX):max(0, minX)+x_len, max(0, minY):max(0, minY)+y_len] = image[max(
                    0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY)+y_len]

        elif type(quadrilateral) is QuadMarks:
            debugger.log_exception("Zooming back in image using QuadMarks is not supported yet!",
                         "Exception"+debugger.get_last_log_categories())

        debugger.save_image(pars['last_image'], 'Image after zoom back',
                            categories=debugger.get_last_log_categories())

        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return last_image

    # NOT DONE
    @Debugger.method_decorator('implanting filter in image', ['ImageLand.implant_filter', 'ImageLand'])
    def implant_filter(self, filter: Filter, heat: HeatLand = None):
        # NOT DONE
        self._debugger.save_image(self._flow_data, 'Image before implant',
                                  categories=self._debugger.get_last_log_categories())
        self._flow_data = ((self._flow_data+(255, 0, 0, 255))/2).astype(int)
        self._debugger.save_image(self._flow_data, 'Image after implant',
                                  categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

        if heat is None:
            heat = full_heat(
                self._flow_data.shape[0], self._flow_data.shape[1])

    def implant_image(self, src_land: MarksLand, plant_land: MarksLand, plant_img: np.ndarray):
        pass  # challenge
