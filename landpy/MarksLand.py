from landpy import Debugger, Land
from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union

import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageDraw, ImageOps
import cv2


class MarksLand(Land):
    def __init__(self, x=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if x is not None:
            MarksLand.set_marks(self, x)

    @Debugger.method_decorator('Setting marks!', ['MarksLand.set_marks', 'MarksLand'])
    def set_marks(self, x, y=None):
        if y is None:
            marks = MarksLand.__x_to_flow_data(x, self._debugger)
        else:
            marks = MarksLand.__xy_to_flow_data(x, y, self._debugger)
        super()._set_data(marks)
        return self

    @Debugger.static_method_decorator('Converting to (n,2) matrix', ['MarksLand.__x_to_flow_data', 'MarksLand'])
    def __x_to_flow_data(x, debugger):
        marks = np.array(x)
        if len(marks.shape)==1:
            marks = marks.reshape((1, -1))
        return marks

    @Debugger.static_method_decorator('Converting to (n,2) matrix', ['MarksLand.__xy_to_flow_data', 'MarksLand'])
    def __xy_to_flow_data(x, y, debugger):
        if type(x) not in [list, tuple, np.ndarray] or type(y) not in [list, tuple, np.ndarray]:
            debugger.log_exception('x and y must be list\\tuple\\ndarray!',
                         categories=['Exception']+debugger.get_last_log_categories())

        x, y = np.array(x), np.array(y)
        if x.shape != y.shape:
            debugger.log_exception('x and y must have equal size!',
                         categories=['Exception']+debugger.get_last_log_categories())

        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        marks = np.hstack([x, y])
        return marks

    @Debugger.method_decorator('Getting marks!', ['MarksLand.get_marks', 'MarksLand'])
    def get_marks(self):
        super()._get_data()

    @Debugger.method_decorator('Converting MarksLand to SquareMarks', ['MarksLand.to_square', 'MarksLand'])
    def to_square(self):
        from landpy import SquareMarks
        sm = SquareMarks(self._debugger).set_marks(minX=np.min(self._flow_data[:, 0]),
                                                   maxX=np.max(
                                                       self._flow_data[:, 0]),
                                                   minY=np.min(
                                                       self._flow_data[:, 1]),
                                                   maxY=np.max(self._flow_data[:, 1]))
        return sm

    @Debugger.method_decorator('Showing marks', ['MarksLand.show', 'MarksLand'])
    def show(self):
        Debugger(visual_debug=True).log_scatter(self._flow_data)

    @Debugger.method_decorator('Rotating marks', ['MarksLand.rotate', 'MarksLand'])
    def rotate(self, angle: float, center: Tuple = None, save_trace: bool = True):
        if center == None:
            center = (0.5, 0.5)
        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=-angle, scale=1)

        self._debugger.log('(angle, center) = ({}, {})'.format(
            angle, center), categories=self._debugger.get_last_log_categories())

        self._debugger.save_scatter(
            self._flow_data, 'Marks before rotation!', categories=self._debugger.get_last_log_categories())

        n = self._flow_data.shape[0]
        homog_points = np.hstack((self._flow_data, np.ones((n, 1))))
        rotated_points = np.dot(rotation_matrix, homog_points.T).T[:, :2]
        self._flow_data = rotated_points

        self._debugger.save_scatter(
            self._flow_data, 'Marks after rotation!', categories=self._debugger.get_last_log_categories())

        if save_trace:
            self._make_trace(MarksLand.__rotate_back, {
                             'angle': angle, 'center': center})

        self._debugger.log_saved_scatters(
            categories=self._debugger.get_last_log_categories())

    @Debugger.static_method_decorator('Rotating back marks', ['MarksLand.__rotate_back', 'MarksLand'])
    def __rotate_back(marks, pars: Dict, debugger: Debugger):
        angle, center = pars['angle'], pars['center']
        height, width = np.max(marks[:, 0])-np.min(
            marks[:, 0]), np.max(marks[:, 1])-np.min(marks[:, 1])
        if center == None:
            center = (int(width/2), int(height/2))

        debugger.log('(angle, center) = ({}, {})'.format(
            angle, center), categories=debugger.get_last_log_categories())

        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)

        debugger.save_scatter(
            marks, 'Marks before rotation back!', categories=debugger.get_last_log_categories())

        n = marks.shape[0]
        homog_points = np.hstack((marks, np.ones((n, 1))))
        rotated_points = np.dot(rotation_matrix, homog_points.T).T[:, :2]

        debugger.save_scatter(
            rotated_points, 'Marks after rotation back!', categories=debugger.get_last_log_categories())

        debugger.log_saved_scatters(
            categories=debugger.get_last_log_categories())

        return rotated_points

    # TODO: zoom back in marksland using QuadMarks
    @Debugger.method_decorator('Zooming marks', ['MarksLand.zoom', 'MarksLand'])
    def zoom(self, quadrilateral, save_trace: bool = True):
        from landpy import QuadMarks, SquareMarks
        if save_trace:
            self._make_trace(MarksLand.__zoom_back, {
                             'quadrilateral': quadrilateral.copy()})

        self._debugger.save_scatter(
            self._flow_data, 'Marks before zoom!', categories=self._debugger.get_last_log_categories())
        if type(quadrilateral) is SquareMarks:
            self._flow_data[:, 0] = (
                self._flow_data[:, 0]-quadrilateral.minX)/(quadrilateral.maxX-quadrilateral.minX)
            self._flow_data[:, 1] = (
                self._flow_data[:, 1]-quadrilateral.minY)/(quadrilateral.maxX-quadrilateral.minY)
        elif type(quadrilateral) is QuadMarks:
            self._debugger.log_exception("Zooming in MarksLand using QuadMarks is not supported yet!",
                               "Exception"+self._debugger.get_last_log_categories())

        self._debugger.save_scatter(
            self._flow_data, 'Marks after zoom!', categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_scatters(
            categories=self._debugger.get_last_log_categories())

    # TODO: zoom back in marksland using QuadMarks
    @Debugger.static_method_decorator('Zooming back marks', ['MarksLand.__zoom_back', 'MarksLand'])
    def __zoom_back(marksland, pars: Dict, debugger: Debugger):
        from landpy import QuadMarks, SquareMarks
        quadrilateral = pars['quadrilateral']

        debugger.save_scatter(
            marksland, 'Marks before zoom back!', categories=debugger.get_last_log_categories())
        if type(quadrilateral) is SquareMarks:
            marksland[:, 0] = (
                marksland[:, 0]*(quadrilateral.maxX-quadrilateral.minX))+quadrilateral.minX
            marksland[:, 1] = (
                marksland[:, 1]*(quadrilateral.maxY-quadrilateral.minY))+quadrilateral.minY
        elif type(quadrilateral) is QuadMarks:
            debugger.log_exception("Zooming back in MarksLand using QuadMarks is not supported yet!",
                         "Exception"+debugger.get_last_log_categories())

        debugger.save_scatter(
            marksland, 'Marks after zoom back!', categories=debugger.get_last_log_categories())

        debugger.log_saved_scatters(
            categories=debugger.get_last_log_categories())


    def __add__(self, other):
        marks_set = set({})
        for i in range(self._flow_data.shape[0]):
            marks_set.add((self._flow_data[i, 0], self._flow_data[i, 1]))
        for i in range(other._flow_data.shape[0]):
            marks_set.add((other._flow_data[i, 0], other._flow_data[i, 1]))

        new_marks = MarksLand(self._debugger)
        new_marks.set_marks(list(marks_set))
        return new_marks

    def __sub__(self, other):
        marks_set = set({})
        for i in range(self._flow_data.shape[0]):
            marks_set.add((self._flow_data[i, 0], self._flow_data[i, 1]))
        for i in range(other._flow_data.shape[0]):
            if (other._flow_data[i, 0], other._flow_data[i, 1]) in marks_set:
                marks_set.remove((other._flow_data[i, 0], other._flow_data[i, 1]))

        new_marks = MarksLand(self._debugger)
        new_marks.set_marks(list(marks_set))
        return new_marks
    

    def __getitem__(self, ind):
        return (self._flow_data[ind]).tolist()

    def to_list(self):
        return (self._flow_data).tolist()

    def to_heat(self, shape):
        heat_p = np.ones(shape)
        pts = self._flow_data.copy()
        pts[:,0] = (pts[:,0]*shape[0]).astype(int)
        pts[:,1] = (pts[:,1]*shape[1]).astype(int)
        print(pts.tolist())
        cv2.fillPoly(heat_p, pts=[pts.tolist()], color=(1))
        return heat_p

    ################################# TODO #################################