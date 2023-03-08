from landpy import Debugger, Land, MarksLand

class QuadMarks(MarksLand):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
