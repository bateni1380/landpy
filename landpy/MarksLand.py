from landpy import Debugger, Land
from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2
np.zeros


class MarksLand(Land):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        self._debugger = debugger

    def set_marks(self, x, y):
        log_categories = ['MarksLand.set_flow_data', 'MarksLand']
        self._debugger.log_start('Setting marks!', categories=log_categories)
        marks = self.__xy_to_flow_data(x, y)
        self._set_data(marks)
        self._debugger.log_end(categories=log_categories)

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
    
    def marks_to_image(marks):
        width, height = 500, 500
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw the scatter plot
        for i in range(marks.shape[0]):
            x_i = int(marks[i,0] * (width - 1))
            y_i = int(marks[i,1] * (height - 1))
            draw.ellipse([(x_i - 3, y_i - 3), (x_i + 3, y_i + 3)], fill='blue')

        img = ImageOps.expand(img, border=(2, 2), fill=(0,0,0))
        return img
    
    def rotate(self, angle: float, center: Tuple = None, save_trace=True):
        log_categories = ['MarksLand.rotate', 'MarksLand']
        self._debugger.log_start('Rotating marks!', categories=log_categories)

        height, width = np.max(self._flow_data[:, 0])-np.min(
            self._flow_data[:, 0]), np.max(self._flow_data[:, 1])-np.min(self._flow_data[:, 1])
        if center == None:
            center = (int(width/2), int(height/2))
        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)
        
        self._debugger.save_sctter(self._flow_data, 'Marks before rotation!', categories=log_categories)
        
        n = self._flow_data.shape[0]
        transformed_points = rotation_matrix.dot(
            np.hstack([self._flow_data[::-1,:], np.ones(shape=(n, 1))]).T).T
        transformed_points[::1,:] = transformed_points[::-1,:]

        self._set_data(transformed_points)
        self._debugger.save_sctter(self._flow_data, 'Marks after rotation!', categories=log_categories)

        if save_trace:
            self._make_trace(MarksLand.__rotate_back, {
                             'angle': angle, 'center': center})
        else:
            self._make_trace(Land._make_exception, {})

        self._debugger.log_end(categories=log_categories)
        self._debugger.log_saved_sctters(categories=log_categories)  

    def __rotate_back(marks, pars: Dict, debugger: Debugger):
        log_categories = ['MarksLand.__rotate_back', 'MarksLand']
        debugger.log_start('Rotate backing marks!', categories=log_categories)

        angle, center = pars['angle'], pars['center']
        height, width = np.max(marks[:, 0])-np.min(
            marks[:, 0]), np.max(marks[:, 1])-np.min(marks[:, 1])
        if center == None:
            center = (int(width/2), int(height/2))
        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle, scale=1)
        
        debugger.save_sctter(marks, 'Marks before rotation back!', categories=log_categories)

        n = marks.shape[0]
        transformed_points = rotation_matrix.dot(
            np.hstack([marks[::-1,:], np.ones(shape=(n, 1))]).T).T
        transformed_points[::1,:] = transformed_points[::-1,:]

        marks = transformed_points
        debugger.save_sctter(marks, 'Marks after rotation back!', categories=log_categories)

        debugger.log_end(categories=log_categories)
        debugger.log_saved_sctters(categories=log_categories)  
        
        return marks
