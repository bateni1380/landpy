from landpy import Debugger, Land
from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union

import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageDraw, ImageOps
import cv2

class MarksSet:
    def __init(self, marks: np.ndarray, shape: Tuple):
        self.marks = marks
        self.shape = shape
    
    def __getitem__(self, index):
        return self.marks[index]
    
class MarksLand(Land):
    _flow_data: MarksSet
    def __init__(self, x=None, y=None, shape=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger=debugger)
        
        if x is None and y is None and shape is None:
            return
        
        if shape is None:
            debugger.log_warning('You need to specify  shape!')

        if x is not None:
            MarksLand.set_marks(self, x, y, shape=shape)

        
    @property
    def shape(self):
        if self._flow_data is None:
            return None
        return self._flow_data.shape
    
    @Debugger.method_decorator('Setting marks')
    def set_marks(self, x, y=None, shape=None):
        if shape is None:
            shape = self.shape()
            
        if y is None:
            marks = MarksLand.__x_to_flow_data(x, self._debugger)
        else:
            marks = MarksLand.__xy_to_flow_data(x, y, self._debugger)
        super()._set_data([marks, shape])
        self._debugger.log_image(
            self._flow_data, "Set marks", Debugger.LogTypes.SCATTER)
        return self

    @Debugger.method_decorator('Reshaping marks')
    def reshape(self, shape):
        self._flow_data[0][:, 0] = np.array(
            (self._flow_data[0][:, 0]/self._flow_data[1][0])*shape[0])
        self._flow_data[0][:, 1] = np.array(
            (self._flow_data[0][:, 1]/self._flow_data[1][1])*shape[1])
        self._flow_data[1] = shape
        return self

    @Debugger.static_method_decorator('Converting to (n,2) matrix')
    def __x_to_flow_data(x, debugger):
        marks = np.array(x)
        if len(marks.shape) == 1:
            if marks.shape[0] != 2:
                debugger.log_exception(
                    'x must be a (n,2) matrice or vector of 2!')
            marks = marks.reshape((1, -1))
        return marks

    @Debugger.static_method_decorator('Converting to (n,2) matrix')
    def __xy_to_flow_data(x, y, debugger):
        if type(x) not in [list, tuple, np.ndarray] or type(y) not in [list, tuple, np.ndarray]:
            debugger.log_exception(
                'x and y must be list\\tuple\\ndarray!')

        x, y = np.array(x), np.array(y)
        if x.shape != y.shape:
            debugger.log_exception(
                'x and y must have equal size!')

        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        marks = np.hstack([x, y])
        return marks

    @Debugger.method_decorator('Getting marks')
    def get_marks(self):
        super()._get_data()

    @Debugger.method_decorator('Showing marks', ['MarksLand.show', 'MarksLand'])
    def show(self):
        Debugger(visual_debug=True).log_image(self._flow_data, '', Debugger.LogTypes.SCATTER)
        return self

    @Debugger.method_decorator('Rotating marks')
    def rotate(self, angle: float, center: Tuple = None, save_trace=False):
        if center is None:
            center = (0.5, 0.5)
        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle, scale=1)

        self._debugger.log('(angle, center) = ({}, {})'.format(
            angle, center))

        self._debugger.save_image(self._flow_data, 'Marks before rotation',
                                  Debugger.LogTypes.SCATTER)

        n = self._flow_data[0].shape[0]
        homog_points = np.hstack((self._flow_data[0], np.ones((n, 1))))
        rotated_points = np.dot(rotation_matrix, homog_points.T).T[:, :2]
        self._flow_data[0] = rotated_points

        self._debugger.save_image(
            self._flow_data, 'Marks after rotation', Debugger.LogTypes.SCATTER)

        if save_trace:
            self._make_trace(MarksLand.__rotate_back, {
                             'angle': angle, 'center': center})

        self._debugger.log_saved_images()
        return self

    @Debugger.static_method_decorator('Rotating back marks')
    def __rotate_back(marks, pars: Dict, debugger: Debugger):
        angle, center = pars['angle'], pars['center']
        height, width = np.max(marks[:, 0])-np.min(
            marks[:, 0]), np.max(marks[:, 1])-np.min(marks[:, 1])
        if center is None:
            center = (int(width/2), int(height/2))

        debugger.log('(angle, center) = ({}, {})'.format(
            angle, center))

        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)

        debugger.save_image(
            marks, 'Marks before rotation back', Debugger.LogTypes.SCATTER)

        n = marks.shape[0]
        homog_points = np.hstack((marks, np.ones((n, 1))))
        rotated_points = np.dot(rotation_matrix, homog_points.T).T[:, :2]

        debugger.save_image(
            rotated_points, 'Marks after rotation back', Debugger.LogTypes.SCATTER)
        debugger.log_saved_images()
        return rotated_points

    @Debugger.method_decorator('Zooming marks')
    def zoom(self, quadrilateral=None, zoom_factor=None, center=None, save_trace=False):
        from landpy import QuadMarks, SquareMarks

        if quadrilateral is None:
            quadrilateral = SquareMarks(
                shape=self.shape(), zoom_factor=zoom_factor, center=center)

        if save_trace:
            self._make_trace(MarksLand.__zoom_back, {
                             'quadrilateral': quadrilateral.copy(save_trace), 'shape':self.shape()})

        self._debugger.save_image(
            self._flow_data, 'Marks before zoom', Debugger.LogTypes.SCATTER)
        if type(quadrilateral) is SquareMarks:
            self._flow_data[0][:, 0] = (self._flow_data[0][:, 0]-quadrilateral.minX())
            self._flow_data[0][:, 1] = (self._flow_data[0][:, 1]-quadrilateral.minY())
            self._flow_data[1] = quadrilateral._flow_data[1]
        elif type(quadrilateral) is QuadMarks:
            self._debugger.log_exception(
                "Zooming in MarksLand using QuadMarks is not supported yet!")

        self._debugger.save_image(
            self._flow_data, 'Marks after zoom', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return self

    @Debugger.static_method_decorator('Zooming back marks')
    def __zoom_back(marksland, pars: Dict, debugger: Debugger):
        from landpy import QuadMarks, SquareMarks
        quadrilateral = pars['quadrilateral']

        debugger.save_image(
            marksland, 'Marks before zoom back', Debugger.LogTypes.SCATTER)
        if type(quadrilateral) is SquareMarks:
            marksland[0][:, 0] = marksland[0][:, 0]+quadrilateral.minX()
            marksland[0][:, 1] = marksland[0][:, 1]+quadrilateral.minY()
            marksland[1] = pars['shape']
        elif type(quadrilateral) is QuadMarks:
            debugger.log_exception(
                "Zooming back in MarksLand using QuadMarks is not supported yet!")

        debugger.save_image(
            marksland, 'Marks after zoom back', Debugger.LogTypes.SCATTER)
        debugger.log_saved_images()
        return marksland

    @Debugger.method_decorator('Moving marks')
    def move(self, vector, save_trace=False):
        if save_trace:
            self._make_trace(MarksLand.__move_back, {'vector': vector})

        self._debugger.save_image(
            self._flow_data, 'Marks before moving', Debugger.LogTypes.SCATTER)

        self._flow_data[0][:, 0] += vector[0]
        self._flow_data[0][:, 1] += vector[1]

        self._debugger.save_image(
            self._flow_data, 'Marks after moving', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return self

    @Debugger.static_method_decorator('Moving back marks')
    def __move_back(marksland, pars: Dict, debugger: Debugger):
        debugger.save_image(marksland, 'Marks before moving back',
                            Debugger.LogTypes.SCATTER)

        marksland[0][:, 0] -= pars['vector'][0]
        marksland[0][:, 1] -= pars['vector'][1]

        debugger.save_image(marksland, 'Marks after moving back',
                            Debugger.LogTypes.SCATTER)
        debugger.log_saved_images()

        return marksland

    @Debugger.method_decorator('Adding two marks (append marks togetter)')
    def __add__(self, other: 'MarksLand'):
        if self.shape() != other.shape():
                self._debugger.log_warning(
                    "The markslands weren\'t in the same size so the second marksland is going to reshape!")
                other.reshape(self.shape())
                
        self._debugger.save_image(
            self._flow_data, 'Left', Debugger.LogTypes.SCATTER)
        self._debugger.save_image(
            other._flow_data, 'Right', Debugger.LogTypes.SCATTER)
        
        marks_set = set({})
        for i in range(self._flow_data[0].shape[0]):
            marks_set.add((self._flow_data[0][i, 0], self._flow_data[0][i, 1]))
        for i in range(other._flow_data[0].shape[0]):
            marks_set.add((other._flow_data[0][i, 0], other._flow_data[0][i, 1]))

        result = MarksLand(self._debugger)
        result.set_marks(list(marks_set), shape=max(self.shape(), other.shape()))

        self._debugger.save_image(
            result._flow_data, 'Result', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return result

    @Debugger.method_decorator('Substracting two marks (remove marks of second object from the first one)')
    def __sub__(self, other):
        if self.shape() != other.shape():
                self._debugger.log_warning(
                    "The markslands weren\'t in the same size so the second marksland is going to reshape!")
                other.reshape(self.shape())
                
        self._debugger.save_image(
            self._flow_data, 'Left', Debugger.LogTypes.SCATTER)
        self._debugger.save_image(
            other._flow_data, 'Right', Debugger.LogTypes.SCATTER)
        marks_set = set({})
        for i in range(self._flow_data[0].shape[0]):
            marks_set.add((self._flow_data[0][i, 0], self._flow_data[0][i, 1]))
        for i in range(other._flow_data[0].shape[0]):
            if (other._flow_data[0][i, 0], other._flow_data[0][i, 1]) in marks_set:
                marks_set.remove(
                    (other._flow_data[0][i, 0], other._flow_data[0][i, 1]))

        result = MarksLand(self._debugger)
        result.set_marks(list(marks_set))
        self._debugger.save_image(
            result._flow_data, 'Result', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return result

    @Debugger.method_decorator('Getting indices of the object')
    def __getitem__(self, index):
        return MarksLand(self._flow_data[0][index], shape=self.shape(), debugger=self._debugger)

    @Debugger.method_decorator('Converting marks object to a python list', ['MarksLand.to_list', 'MarksLand'])
    def to_list(self):
        return ((self._flow_data[0]).tolist(), self._flow_data[1])

    @Debugger.method_decorator('Converting marks object to a landpy heat', ['MarksLand.to_heat', 'MarksLand'])
    def to_heat(self, fill_heat=1):
        self._debugger.save_image(
            self._flow_data, 'Before', Debugger.LogTypes.SCATTER)

        from landpy import HeatLand
        heat_p = np.zeros(self.shape())
        pts = self._flow_data[0].copy()
        pts = pts.astype(int).reshape((-1, 1, 2))
        cv2.fillPoly(heat_p, pts=[pts], color=(fill_heat))
        heat = HeatLand(heat_p, self._debugger)

        self._debugger.save_image(
            heat._flow_data, 'After', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return heat

    @Debugger.method_decorator('Converting MarksLand to SquareMarks')
    def to_square(self):
        self._debugger.save_image(
            self._flow_data, 'Before', Debugger.LogTypes.SCATTER)

        from landpy import SquareMarks
        sm = SquareMarks(self.shape(), self._debugger).set_marks(minX=np.min(self._flow_data[0][:, 0]),
                                                                 maxX=np.max(self._flow_data[0][:, 0]),
                                                                 minY=np.min(self._flow_data[0][:, 1]),
                                                                 maxY=np.max(self._flow_data[0][:, 1]))

        self._debugger.save_image(
            sm._flow_data, 'After', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return sm
