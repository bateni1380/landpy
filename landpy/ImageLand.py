from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union
import time
from landpy import Debugger, Land
''', MarksLand, QuadMarks, SquadMarks, Filter, HeatLand
from landpy import zero_heat, full_heat'''
import numpy as np
import cv2
import inspect
import matplotlib.pyplot as plt
from PIL import Image

class ImageLand(Land):
    def __init__(self, debugger: Debugger):
        super().__init__(debugger)

    def set_image(self, image: np.ndarray, mode='rgb'):
        self._debugger.log_start('Sending image to flow_data', categories=[
                                 'ImageLand.set_data', 'ImageLand'])
        self._debugger.log_image(image, 'Sent Image', categories=[
                                 'ImageLand.set_data', 'ImageLand'])

        if image.dtype in ('float32', 'float64'):
            image = (image*255)
        self._set_data(image.astype('uint8'))

        self._debugger.log_end(categories=['ImageLand.set_data', 'ImageLand'])

    def __make_exception(image, pars, debugger):
        debugger.log('save_trace parameter was false so you can\'t retrace!', categories=[
            'Exception', 'ImageLand.__make_exception', 'ImageLand'])
        raise Exception(
            'save_trace parameter was false so you can\'t retrace!')

    def rotate(self, angle: float, center: Tuple = None, save_trace=True):
        self._debugger.log_start('Rotating image', categories=[
            'ImageLand.rotate', 'ImageLand'])

        height, width = self._flow_data.shape[:2]
        if center == None:
            center = (int(width/2), int(height/2))

        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)

        self._debugger.save_image(self._flow_data, 'Image before rotation!', categories=[
                                  'ImageLand.rotate', 'ImageLand'])

        self._flow_data = cv2.warpAffine(src=self._flow_data, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        self._debugger.save_image(self._flow_data, 'Image after rotation!', categories=[
                                  'ImageLand.rotate', 'ImageLand'])

        if save_trace:
            self._make_trace(ImageLand.__rotate_back, {'angle': angle, 'center': center})
        else:
            self._make_trace(ImageLand.__make_exception, {})

        self._debugger.log_end(categories=['ImageLand.rotate', 'ImageLand'])
        self._debugger.log_saved_images(
            categories=['ImageLand.rotate', 'ImageLand'])

    def __rotate_back(image, pars: Dict, debugger: Debugger):
        debugger.log_start('Rotate backing image', categories=[
            'ImageLand.__rotate_back', 'ImageLand'])
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(
            center=pars['center'], angle=-pars['angle'], scale=1)
        debugger.save_image(image, 'Image before rotation back!', categories=[
            'ImageLand.__rotate_back', 'ImageLand'])
        image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        debugger.save_image(image, 'Image after rotation back!', categories=[
            'ImageLand.__rotate_back', 'ImageLand'])

        debugger.log_end(
            categories=['ImageLand.__rotate_back', 'ImageLand'])
        debugger.log_saved_images(
            categories=['ImageLand.__rotate_back', 'ImageLand'])
        
        return image



    def reshape(self, height, width, mode = 'RGB', save_trace=True): 
        image = Image.fromarray(self._flow_data)

        if save_trace:
            self._make_trace(ImageLand.__reshape_back, {'height': image.height,
                                                        'width': image.width,
                                                        'mode' : image.mode})
        else:
            self._make_trace(ImageLand.__make_exception, {})

        image = image.resize((height, width))
        image = image.convert(mode)
        image = np.array(image, dtype='uint8')
        self._flow_data = image
        

    def __reshape_back(image, pars: Dict, debugger: Debugger):
        image = Image.fromarray(image)
        image = image.resize((pars['height'], pars['width']))
        image = image.convert(pars['mode'])
        image = np.array(image, dtype='uint8')
        return image
        
    def __rotate_back(self, pars, save_trace=True):
        pass
'''
    def zoom(self, quadrilateral: Union[SquadMarks, QuadMarks], save_trace=True):
        pass

    def __zoom_back(self, pars, save_trace=True):
        pass  # challenge

    def implant_filter(self, filter: Filter, heat: HeatLand = None):
        if heat is None:
            heat = full_heat(self.flow_data.shape[0], self.flow_data.shape[1])
        pass

    def implant_image(self, src_land: MarksLand, plant_land: MarksLand, plant_img: np.ndarray):
        pass  # challenge
    '''
