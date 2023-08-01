from landpy import Debugger
import numpy as np

class Filter:
    def __init__(self, color=(255,255,255), contrast=1, debugger=Debugger(debug=False)):
        self._debugger = debugger

        if type(color) is str:
            self.color = np.array(Filter.hex_to_rgb(color), dtype=float)
        else:
            self.color = np.array(color, dtype=float)

        self.set_function(lambda pixel_color: (pixel_color+color)/2)

    def set_function(self, f):
        self.apply_function = f 

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")  
        r = int(hex_color[0:2], 16)  
        g = int(hex_color[2:4], 16)  
        b = int(hex_color[4:6], 16)  
        return (r, g, b)
