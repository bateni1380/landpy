from landpy import Debugger, MarksLand


class QuadMarks(MarksLand):
    def __init__(self, x1=None, x2=None, x3=None, x4=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if x1 is not None and x2 is not None and x3 is not None and x4 is not None:
            self.set_marks(x1, x2, x3, x4)
            
    @Debugger.method_decorator('Setting QuadMarks', ['QuadMarks.__init__', 'QuadMarks'])
    def set_marks(self, x1, x2, x3, x4):
        log_categories = ['QuadMarks.set_marks', 'QuadMarks']
        if (len(x1), len(x2), len(x3), len(x4)) != (2, 2, 2, 2) :
            self._debugger.log('xi must have two elements!',
                               categories=['Exception']+log_categories)
            raise Exception('xi must have two elements!')
        
        x = [x1[0], x2[0], x3[0], x4[0]]
        y = [x1[1], x2[1], x3[1], x4[1]]
        super().set_marks(x, y)
        return self