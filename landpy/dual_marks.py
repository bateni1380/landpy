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
        return self
        
    def lenght(self):
        start_point = (self._flow_data[0][0], self._flow_data[0][1])
        end_point = (self._flow_data[1][0], self._flow_data[1][1])
        distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        return distance

    def angle(self):
        start_point = (self._flow_data[0][1], 1-self._flow_data[0][0])
        end_point = (self._flow_data[1][1], 1-self._flow_data[1][0])
        angle_rad = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        return np.degrees(angle_rad)

