from landpy import Debugger, QuadMarks, MarksLand


class SquareMarks(QuadMarks):
    def __init__(self, minX=None, maxX=None, minY=None, maxY=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if minX is not None and maxX is not None and minY is not None and maxY is not None:
            SquareMarks.set_marks(self, minX, maxX, minY, maxY)
        
    @Debugger.method_decorator('Setting SquareMarks', ['SquareMarks.set_marks', 'SquareMarks'])
    def set_marks(self, minX, maxX, minY, maxY):
        log_categories = ['SquareMarks.set_marks', 'SquareMarks']
        if minX > maxX or minY > maxY :
            self._debugger.log_exception('minX and minY must be less than maxX and maxY!',
                               categories=['Exception']+log_categories)
        
        x = [minX, minX, maxX, maxX]
        y = [minY, maxY, minY, maxY]
        MarksLand.set_marks(self, x, y)
        self.minX, self.maxX = self._flow_data[0,0], self._flow_data[2,0]
        self.minY, self.maxY = self._flow_data[0,1], self._flow_data[1,1]
        return self
    
    @Debugger.method_decorator('Copying SquareMarks', ['SquareMarks.copy', 'SquareMarks'])    
    def copy(self):
        copied_land = super().copy()
        copied_land.minX = self.minX
        copied_land.maxX = self.maxX
        copied_land.minY = self.minY
        copied_land.maxY = self.maxY
        return copied_land
    

    
