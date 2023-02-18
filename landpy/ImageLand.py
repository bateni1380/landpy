from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union

from landpy import Debugger, Land, MarksLand, QuadMarks, SquadMarks, Filter, HeatLand
from landpy import zero_heat, full_heat
import numpy as np
import inspect

class ImageLand(Land):
    def __init__(self):
        super().__init__()
        self.flow_img = None

    def set_image(self, image: np.ndarray, debugger: Debugger):
        debugger.log_start('Sending image to flow_img...', categories=['ImageLand.set_data', 'ImageLand'])
        debugger.log_image('Sent Image', image, categories=['ImageLand.set_data', 'ImageLand'])
        self.flow_img = image
        self._set_data(self.flow_img, debugger)
        debugger.log_start('Data sent to flow_img.', categories=['ImageLand.set_data', 'ImageLand'])

    def rotate(self, angle: float, debugger:Debugger, save_trace=True):
        pass
    def __rotate_back(self, pars, debugger:Debugger, save_trace=True):
        pass

    def resize(self, dst_size: Tuple[int,int], debugger:Debugger, save_trace=True):
        pass
    def __rotate_back(self, pars, debugger:Debugger, save_trace=True):
        pass

    def zoom(self, quadrilateral: Union[SquadMarks, QuadMarks], debugger:Debugger, save_trace=True):
        pass
    def __zoom_back(self, pars, debugger:Debugger, save_trace=True): 
        pass #challenge

    def implant_filter(self, filter: Filter, debugger:Debugger, heat: HeatLand=None):
        if heat is None:
            heat = full_heat(self.flow_img.shape[0], self.flow_img.shape[1])
        pass

    def implant_image(self, src_land: MarksLand, plant_land: MarksLand, plant_img: np.ndarray, debugger: Debugger):
        pass #challenge


        