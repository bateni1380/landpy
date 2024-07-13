from landpy import Debugger, Land
from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union

import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageDraw, ImageOps
import cv2


class MarksLand(Land): #TODO ratio must be added and changed in all of project ( for rotating and stuf)
    def __init__(self, x=None, y=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger=debugger)
        if x is not None:
            MarksLand.set_marks(self, x, y)

    @Debugger.method_decorator('Setting marks', ['MarksLand.set_marks', 'MarksLand'])
    def set_marks(self, x, y=None):
        if y is None:
            marks = MarksLand.__x_to_flow_data(x, self._debugger)
        else:
            marks = MarksLand.__xy_to_flow_data(x, y, self._debugger)
        super()._set_data(marks)
        self._debugger.log_image(
            self._flow_data, "Set marks", Debugger.LogTypes.SCATTER)
        return self

    @Debugger.static_method_decorator('Converting to (n,2) matrix', ['MarksLand.__x_to_flow_data', 'MarksLand'])
    def __x_to_flow_data(x, debugger):
        marks = np.array(x)
        if len(marks.shape) == 1:
            if marks.shape[0] != 2:
                debugger.log_exception(
                    'x must be a (n,2) matrice or vector of 2!')
            marks = marks.reshape((1, -1))
        return marks

    @Debugger.static_method_decorator('Converting to (n,2) matrix', ['MarksLand.__xy_to_flow_data', 'MarksLand'])
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

    @Debugger.method_decorator('Getting marks', ['MarksLand.get_marks', 'MarksLand'])
    def get_marks(self):
        super()._get_data()

    @Debugger.method_decorator('Showing marks', ['MarksLand.show', 'MarksLand'])
    def show(self):
        Debugger(visual_debug=True).log_image(
            self._flow_data, '', Debugger.LogTypes.SCATTER)
        return self

    @Debugger.method_decorator('Rotating marks', ['MarksLand.rotate', 'MarksLand'])
    def rotate(self, angle: float, center: Tuple = None, save_trace = False):
        if center is None:
            center = (0.5, 0.5)
        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle, scale=1)

        self._debugger.log('(angle, center) = ({}, {})'.format(
            angle, center))

        self._debugger.save_image(self._flow_data, 'Marks before rotation',
                                  Debugger.LogTypes.SCATTER)

        n = self._flow_data.shape[0]
        homog_points = np.hstack((self._flow_data, np.ones((n, 1))))
        rotated_points = np.dot(rotation_matrix, homog_points.T).T[:, :2]
        self._flow_data = rotated_points

        self._debugger.save_image(
            self._flow_data, 'Marks after rotation', Debugger.LogTypes.SCATTER)

        if save_trace:
            self._make_trace(MarksLand.__rotate_back, {
                             'angle': angle, 'center': center})

        self._debugger.log_saved_images()
        return self

    @Debugger.static_method_decorator('Rotating back marks', ['MarksLand.__rotate_back', 'MarksLand'])
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

    @Debugger.method_decorator('Zooming marks', ['MarksLand.zoom', 'MarksLand'])
    def zoom(self, quadrilateral=None, zoom_factor=None, center=None, save_trace = False):
        from landpy import QuadMarks, SquareMarks

        if quadrilateral is None:
            quadrilateral = SquareMarks(zoom_factor=zoom_factor, center=center)

        if save_trace:
            self._make_trace(MarksLand.__zoom_back, {
                             'quadrilateral': quadrilateral.copy(save_trace)})

        self._debugger.save_image(
            self._flow_data, 'Marks before zoom', Debugger.LogTypes.SCATTER)
        if type(quadrilateral) is SquareMarks:
            self._flow_data[:, 0] = (
                self._flow_data[:, 0]-quadrilateral.minX())/(quadrilateral.maxX()-quadrilateral.minX())
            self._flow_data[:, 1] = (
                self._flow_data[:, 1]-quadrilateral.minY())/(quadrilateral.maxY()-quadrilateral.minY())
        elif type(quadrilateral) is QuadMarks:
            self._debugger.log_exception(
                "Zooming in MarksLand using QuadMarks is not supported yet!")

        self._debugger.save_image(
            self._flow_data, 'Marks after zoom', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return self

    @Debugger.static_method_decorator('Zooming back marks', ['MarksLand.__zoom_back', 'MarksLand'])
    def __zoom_back(marksland, pars: Dict, debugger: Debugger):
        from landpy import QuadMarks, SquareMarks
        quadrilateral = pars['quadrilateral']

        debugger.save_image(
            marksland, 'Marks before zoom back', Debugger.LogTypes.SCATTER)
        if type(quadrilateral) is SquareMarks:
            marksland[:, 0] = (
                marksland[:, 0]*(quadrilateral.maxX()-quadrilateral.minX()))+quadrilateral.minX()
            marksland[:, 1] = (
                marksland[:, 1]*(quadrilateral.maxY()-quadrilateral.minY()))+quadrilateral.minY()
        elif type(quadrilateral) is QuadMarks:
            debugger.log_exception(
                "Zooming back in MarksLand using QuadMarks is not supported yet!")

        debugger.save_image(
            marksland, 'Marks after zoom back', Debugger.LogTypes.SCATTER)
        debugger.log_saved_images()
        return marksland

    @Debugger.method_decorator('Moving marks', ['MarksLand.move', 'MarksLand'])
    def move(self, vector, save_trace = False):
        if save_trace:
            self._make_trace(MarksLand.__move_back, {'vector': vector})

        self._debugger.save_image(
            self._flow_data, 'Marks before moving', Debugger.LogTypes.SCATTER)

        self._flow_data[:, 0] += vector[0]
        self._flow_data[:, 1] += vector[1]

        self._debugger.save_image(
            self._flow_data, 'Marks after moving', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return self

    @Debugger.static_method_decorator('Moving back marks', ['MarksLand.__move_back', 'MarksLand'])
    def __move_back(marksland, pars: Dict, debugger: Debugger):
        debugger.save_image(marksland, 'Marks before moving back',
                            Debugger.LogTypes.SCATTER)

        marksland[:, 0] -= pars['vector'][0]
        marksland[:, 1] -= pars['vector'][1]

        debugger.save_image(marksland, 'Marks after moving back',
                            Debugger.LogTypes.SCATTER)
        debugger.log_saved_images()
        
        return marksland

    @Debugger.method_decorator('Adding two marks (append marks togetter)', ['MarksLand.__add__', 'MarksLand'])
    def __add__(self, other):
        self._debugger.save_image(
            self._flow_data, 'Left', Debugger.LogTypes.SCATTER)
        self._debugger.save_image(
            other._flow_data, 'Right', Debugger.LogTypes.SCATTER)
        marks_set = set({})
        for i in range(self._flow_data.shape[0]):
            marks_set.add((self._flow_data[i, 0], self._flow_data[i, 1]))
        for i in range(other._flow_data.shape[0]):
            marks_set.add((other._flow_data[i, 0], other._flow_data[i, 1]))

        result = MarksLand(self._debugger)
        result.set_marks(list(marks_set))

        self._debugger.save_image(
            result._flow_data, 'Result', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return result

    @Debugger.method_decorator('Substracting two marks (remove marks of second object from the first one)', ['MarksLand.__sub__', 'MarksLand'])
    def __sub__(self, other):
        self._debugger.save_image(
            self._flow_data, 'Left', Debugger.LogTypes.SCATTER)
        self._debugger.save_image(
            other._flow_data, 'Right', Debugger.LogTypes.SCATTER)
        marks_set = set({})
        for i in range(self._flow_data.shape[0]):
            marks_set.add((self._flow_data[i, 0], self._flow_data[i, 1]))
        for i in range(other._flow_data.shape[0]):
            if (other._flow_data[i, 0], other._flow_data[i, 1]) in marks_set:
                marks_set.remove(
                    (other._flow_data[i, 0], other._flow_data[i, 1]))

        result = MarksLand(self._debugger)
        result.set_marks(list(marks_set))
        self._debugger.save_image(
            result._flow_data, 'Result', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return result

    @Debugger.method_decorator('Getting indices of the object', ['MarksLand.__getitem__', 'MarksLand'])
    def __getitem__(self, index):
        return MarksLand(self._flow_data[index], debugger=self._debugger)

    @Debugger.method_decorator('Converting marks object to a python list', ['MarksLand.to_list', 'MarksLand'])
    def to_list(self):
        return (self._flow_data).tolist()

    @Debugger.method_decorator('Converting marks object to a landpy heat', ['MarksLand.to_heat', 'MarksLand'])
    def to_heat(self, shape=(128, 128), fill_heat=1):
        self._debugger.save_image(
            self._flow_data, 'Before', Debugger.LogTypes.SCATTER)

        from landpy import HeatLand
        heat_p = np.zeros(shape)
        pts = self._flow_data.copy()
        pts[:, 0] = pts[:, 0]*shape[0]
        pts[:, 1] = pts[:, 1]*shape[1]
        pts = pts.astype(int).reshape((-1, 1, 2))
        cv2.fillPoly(heat_p, pts=[pts], color=(fill_heat))
        heat = HeatLand(heat_p, self._debugger)

        self._debugger.save_image(
            heat._flow_data, 'After', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return heat

    @Debugger.method_decorator('Converting MarksLand to SquareMarks', ['MarksLand.to_square', 'MarksLand'])
    def to_square(self):
        self._debugger.save_image(
            self._flow_data, 'Before', Debugger.LogTypes.SCATTER)

        from landpy import SquareMarks
        sm = SquareMarks(self._debugger).set_marks(minX=np.min(self._flow_data[:, 0]),
                                                   maxX=np.max(
                                                       self._flow_data[:, 0]),
                                                   minY=np.min(
                                                       self._flow_data[:, 1]),
                                                   maxY=np.max(self._flow_data[:, 1]))

        self._debugger.save_image(
            sm._flow_data, 'After', Debugger.LogTypes.SCATTER)
        self._debugger.log_saved_images()
        return sm
