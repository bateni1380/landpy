from landpy import Debugger, Land
from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union

import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageDraw, ImageOps
import cv2


class MarksLand(Land):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)

    def set_marks(self, x, y):
        log_categories = ['MarksLand.set_flow_data', 'MarksLand']
        self._debugger.log_start('Setting marks!', categories=log_categories)
        marks = self.__xy_to_flow_data(x, y)
        self._set_data(marks)
        self._debugger.log_end(categories=log_categories)
        return self

    def get_marks(self):
        pass

    def show(self):
        self._debugger.log('Showing marks.', ['MarksLand.show', 'MarksLand'])
        Debugger(visual_debug=True).log_scatter(self._flow_data)

    def __xy_to_flow_data(self, x, y):
        log_categories = ['MarksLand.__check_flow_data', 'MarksLand']
        if type(x) not in [list, tuple, np.ndarray] or type(y) not in [list, tuple, np.ndarray]:
            self._debugger.log('x and y must be list\\tuple\\ndarray!',
                               categories=['Exception']+log_categories)
            raise Exception('x and y must be list\\tuple\\ndarray!')
        
        x, y = np.array(x), np.array(y)
        if x.shape != y.shape:
            self._debugger.log('x and y must have equal size!',
                               categories=['Exception']+log_categories)
            raise Exception('x and y must have equal size!')
        
        x = np.array(x).reshape(-1,1)
        y = np.array(y).reshape(-1,1)
        marks = np.hstack([x,y])
        return marks
    
    def rotate(self, angle: float, center: Tuple = None, save_trace=True):
        log_categories = ['MarksLand.rotate', 'MarksLand']
        self._debugger.log_start('Rotating marks!', categories=log_categories)

        #height, width = np.max(self._flow_data[:, 0])-np.min(
        #    self._flow_data[:, 0]), np.max(self._flow_data[:, 1])-np.min(self._flow_data[:, 1])
        if center == None:
            center = (0.5, 0.5)
        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle, scale=1)
        
        self._debugger.save_scatter(self._flow_data, 'Marks before rotation!', categories=log_categories)
        
        n = self._flow_data.shape[0]
        homog_points = np.hstack((self._flow_data, np.ones((n, 1))))
        rotated_points = np.dot(rotation_matrix, homog_points.T).T[:, :2]

        self._set_data(rotated_points)
        self._debugger.save_scatter(self._flow_data, 'Marks after rotation!', categories=log_categories)

        if save_trace:
            self._make_trace(MarksLand.__rotate_back, {
                             'angle': angle, 'center': center})
        else:
            self._make_trace(Land._make_exception, {})

        self._debugger.log_end(categories=log_categories)
        self._debugger.log_saved_scatters(categories=log_categories)  

    def __rotate_back(marks, pars: Dict, debugger: Debugger):
        log_categories = ['MarksLand.__rotate_back', 'MarksLand']
        debugger.log_start('Rotate backing marks!', categories=log_categories)

        angle, center = pars['angle'], pars['center']
        height, width = np.max(marks[:, 0])-np.min(
            marks[:, 0]), np.max(marks[:, 1])-np.min(marks[:, 1])
        if center == None:
            center = (int(width/2), int(height/2))
        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)
        
        debugger.save_scatter(marks, 'Marks before rotation back!', categories=log_categories)

        n = marks.shape[0]
        homog_points = np.hstack((marks, np.ones((n, 1))))
        rotated_points = np.dot(rotation_matrix, homog_points.T).T[:, :2]

        debugger.save_scatter(rotated_points, 'Marks after rotation back!', categories=log_categories)

        debugger.log_end(categories=log_categories)
        debugger.log_saved_scatters(categories=log_categories)  
        
        return rotated_points

    def to_square(self):
        from landpy import SquareMarks
        sm = SquareMarks(self._debugger).set_marks(minX=np.min(self._flow_data[:,0]),
                                                   maxX=np.max(self._flow_data[:,0]),
                                                   minY=np.min(self._flow_data[:,1]),
                                                   maxY=np.max(self._flow_data[:,1]))
        return sm

    ################################# TODO #################################
    def zoom():
        pass

    def __zoom_back():
        pass

    def spread():
        pass

    def spread_back():
        pass

    def __add__():
        pass

    def __sub__():
        pass

    def __getitem__():
        pass 

    def to_list():
        pass



