from landpy import Debugger, QuadMarks, MarksLand
import numpy as np

class SquareMarks(QuadMarks):
    def __init__(self, minX=None, maxX=None, minY=None, maxY=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger=debugger)
        if minX is None:
            minX = 0
        if minY is None:
            minY = 0
        if maxX is None:
            maxX = 1
        if maxY is None:
            maxY = 1
        SquareMarks.set_marks(self, minX, maxX, minY, maxY)

    @Debugger.method_decorator('Setting SquareMarks', ['SquareMarks.set_marks', 'SquareMarks'])
    def set_marks(self, minX, maxX, minY, maxY):
        log_categories = ['SquareMarks.set_marks', 'SquareMarks']
        if minX > maxX or minY > maxY:
            self._debugger.log_exception(
                'minX and minY must be less than maxX and maxY!', log_categories)

        x = [minX, minX, maxX, maxX]
        y = [minY, maxY, maxY, minY]
        MarksLand.set_marks(self, x, y)
        return self
    
    def minX(self):
        return np.min(self._flow_data[:,0])
    def maxX(self):
        return np.max(self._flow_data[:,0])
    def minY(self):
        return np.min(self._flow_data[:,1])
    def maxY(self):
        return np.max(self._flow_data[:,1])
    
    @Debugger.method_decorator('Copying SquareMarks', ['SquareMarks.copy', 'SquareMarks'])
    def copy(self, save_trace=True):
        copied_land = super().copy(save_trace=save_trace)
        copied_land.minX = self.minX
        copied_land.maxX = self.maxX
        copied_land.minY = self.minY
        copied_land.maxY = self.maxY
        return copied_land
