from landpy import Debugger, QuadMarks, MarksLand
import numpy as np


class SquareMarks(QuadMarks):
    def __init__(self, minX=None, maxX=None, minY=None, maxY=None, zoom_factor=None, center=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger=debugger)

        if zoom_factor is not None and (minX is not None or maxX is not None or minY is not None or maxY is not None):
            debugger.log_exception(
                'z_value and (minX, maxX, minY, max_Y) can\'t have value at the same time!')

        if zoom_factor is not None:
            if center is None:
                center = (0.5, 0.5)
            t = float(zoom_factor)
            minX, maxX, minY, maxY = center[0]-0.5*(1/t), center[0]+0.5*(
                1/t), center[1]-0.5*(1/t), center[1]+0.5*(1/t)

        if minX is None:
            minX = 0.0
        if minY is None:
            minY = 0.0
        if maxX is None:
            maxX = 1.0
        if maxY is None:
            maxY = 1.0
        SquareMarks.set_marks(self, minX, maxX, minY, maxY)

    @Debugger.method_decorator('Setting SquareMarks', ['SquareMarks.set_marks', 'SquareMarks'])
    def set_marks(self, minX, maxX, minY, maxY):
        if minX > maxX or minY > maxY:
            self._debugger.log_exception(
                'minX and minY must be less than maxX and maxY!')

        x = [minX, minX, maxX, maxX]
        y = [minY, maxY, maxY, minY]
        MarksLand.set_marks(self, x, y)
        return self

    def minX(self):
        return np.min(self._flow_data[:, 0])

    def maxX(self):
        return np.max(self._flow_data[:, 0])

    def minY(self):
        return np.min(self._flow_data[:, 1])

    def maxY(self):
        return np.max(self._flow_data[:, 1])

    def area(self):
        return (self.maxX()-self.minX())*(self.maxY()-self.minY())

    def subscription(self, other):
        inter_minX = max(self.minX(), other.minX())
        inter_maxX = min(self.maxX(), other.maxX())
        inter_minY = max(self.minY(), other.minY())
        inter_maxY = min(self.maxY(), other.maxY())

        if inter_minX < inter_maxX and inter_minY < inter_maxY:
            return SquareMarks(inter_minX, inter_maxX, inter_minY, inter_maxY)
        else:
            return None

    @Debugger.method_decorator('Copying SquareMarks', ['SquareMarks.copy', 'SquareMarks'])
    def copy(self, copy_trace=True):
        copied_land = super().copy(copy_trace=copy_trace)
        copied_land.minX = self.minX
        copied_land.maxX = self.maxX
        copied_land.minY = self.minY
        copied_land.maxY = self.maxY
        return copied_land
