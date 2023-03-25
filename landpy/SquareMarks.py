from landpy import Debugger, QuadMarks


class SquareMarks(QuadMarks):
    def __init__(self, minX=None, maxX=None, minY=None, maxY=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if minX is not None and maxX is not None and minY is not None and maxY is not None:
            self.set_marks(minX, maxX, minY, maxY)
        
    @Debugger.method_decorator('Setting SquareMarks', ['SquareMarks.set_marks', 'SquareMarks'])
    def set_marks(self, minX, maxX, minY, maxY):
        log_categories = ['SquareMarks.set_marks', 'SquareMarks']
        if minX > maxX or minY > maxY :
            self._debugger.log('minX and minY must be less than maxX and maxY!',
                               categories=['Exception']+log_categories)
            raise Exception('minX and minY must be less than maxX and maxY!')
        
        x = [minX, minX, maxX, maxX]
        y = [minY, maxY, minY, maxY]
        super().set_marks(x, y)
        self.minX, self.maxX = self._flow_data[0,0], self._flow_data[2,0]
        self.minY, self.maxY = self._flow_data[0,1], self._flow_data[1,1]
        return self
    
