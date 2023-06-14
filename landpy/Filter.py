from landpy import Debugger
import numpy as np

class Filter:
    def __init__(self, color=(255,255,255), debugger=Debugger(debug=False)):
        self._debugger = debugger
        self.color = np.array(color)

    #@Debugger.method_decorator('Setting heat', ['HeatLand.set_data', 'HeatLand'])
    #def set_heat(self, heat: np.ndarray):