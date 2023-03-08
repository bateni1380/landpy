from landpy import Debugger, Land, MarksLand, QuadMarks

class SquareMarks(QuadMarks):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
    
    def set_marks(self, minX, maxX, minY, maxY):
        x = [minX, minX, maxX, maxX]
        y = [minY, maxY, minY, maxY]
        super().set_marks(x, y)
        self.minX, self.maxX = self._flow_data[0,0], self._flow_data[2,0]
        self.minY, self.maxY = self._flow_data[0,1], self._flow_data[1,1]
        return self
    
