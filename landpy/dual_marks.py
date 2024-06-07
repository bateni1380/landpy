from landpy import Debugger, QuadMarks, MarksLand
import numpy as np


class DualMarks(MarksLand):
    def __init__(self, start_point=(0.0,0.0), end_point=(1.0,1.0), debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger=debugger)

        DualMarks.set_marks(self, start_point, end_point, debugger)

    @Debugger.method_decorator('Setting DualMarks', ['DualMarks.set_marks', 'DualMarks'])
    def set_marks(self, start_point, end_point, debugger: Debugger = Debugger(debug=False)):
        if type(start_point) is MarksLand:
            if start_point._flow_data.shape != (1, 2):
                debugger.log_exception('Start point must be a single point!')
            start_point = list(start_point._flow_data[0])
        if type(end_point) is MarksLand:
            if end_point._flow_data.shape != (1, 2):
                debugger.log_exception('End point must be a single point!')
            end_point = list(end_point._flow_data[0])
            
        MarksLand.set_marks(self, [start_point, end_point])
        return self

    @Debugger.method_decorator('Showing marks', ['DualMarks.show', 'DualMarks'])
    def show(self):
        Debugger(visual_debug=True).log_image(
            self._flow_data, '', Debugger.LogTypes.VECTOR)
        
    def lenght(self):
        return (self.maxX()-self.minX())*(self.maxY()-self.minY())

    def angle(self):
        pass

