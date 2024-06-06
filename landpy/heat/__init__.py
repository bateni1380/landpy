from landpy import HeatLand
from landpy.debugger import Debugger
import numpy as np


def zero_heat(width, height, debugger: Debugger = Debugger(debug=False)) -> HeatLand:
    return HeatLand(np.zeros([width, height]), debugger)


def full_heat(width, height, debugger: Debugger = Debugger(debug=False)) -> HeatLand:
    return HeatLand(np.ones([width, height]), debugger)
