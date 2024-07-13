from landpy import Debugger, MarksLand
import numpy as np

class QuadMarks(MarksLand):
    def __init__(self, x1=None, x2=None, x3=None, x4=None, shape=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(shape=shape, debugger=debugger)
        if x1 is not None and x2 is not None and x3 is not None and x4 is not None:
            QuadMarks.set_marks(self, x1, x2, x3, x4)

    @Debugger.static_method_decorator('Sorting points clockwise', ['QuadMarks.sort_points_clockwise', 'QuadMarks'])
    def __sort_points_clockwise(points, debugger = Debugger(debug=False)):
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_points = points[np.argsort(angles)]
        return sorted_points

    @Debugger.method_decorator('Setting QuadMarks', ['QuadMarks.__init__', 'QuadMarks'])
    def set_marks(self, x1, x2, x3, x4):
        if (len(x1), len(x2), len(x3), len(x4)) != (2, 2, 2, 2):
            self._debugger.log_exception('xi must have two elements!')

        x = np.array([x1[0], x2[0], x3[0], x4[0]]).reshape(-1,1)
        y = np.array([x1[1], x2[1], x3[1], x4[1]]).reshape(-1,1)
        marks = np.hstack([x, y])
        
        marks = QuadMarks.__sort_points_clockwise(marks, debugger=Debugger(debug=False))
        
        super().set_marks(marks)
        return self
    
    @Debugger.method_decorator('Getting center of QuadMarks (intersection of diagonals)', ['QuadMarks.center', 'QuadMarks'])
    def center(self):
        p1, p2, p3, p4 = self._flow_data[0][0], self._flow_data[0][1], self._flow_data[0][2], self._flow_data[0][3]
        a1, b1 = p2 - p1
        a2, b2 = p4 - p3
        
        c1 = a1 * p1[0] + b1 * p1[1]
        c2 = a2 * p3[0] + b2 * p3[1]
        
        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([c1, c2])
        
        try:
            intersection_point = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return self._flow_data[0].mean(axis=0)
        
        return intersection_point.tolist()
