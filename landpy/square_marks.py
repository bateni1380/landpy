from landpy import Debugger, QuadMarks, MarksLand
import numpy as np


class SquareMarks(QuadMarks):
    def __init__(self, minX=None, maxX=None, minY=None, maxY=None, shape=None, zoom_factor=None, center=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(shape=shape, debugger=debugger)

        if minX is None and maxX is None and minY is None and maxY is None and shape is None and zoom_factor is None and center is None:
            return
        
        if shape is None:
            debugger.log_warning('You need to specify  shape!')
            
        if zoom_factor is not None and (minX is not None or maxX is not None or minY is not None or maxY is not None):
            debugger.log_exception(
                'zoom_factor and (minX, maxX, minY, max_Y) can\'t have value at the same time!')

        if zoom_factor is not None:
            if center is None:
                center = (shape[0]/2, shape[1]/2)
            t = float(zoom_factor)
            minX, maxX, minY, maxY = (int(center[0]-0.5*(1/t)*shape[0]), 
                                      int(center[0]+0.5*(1/t)*shape[0]), 
                                      int(center[1]-0.5*(1/t)*shape[1]), 
                                      int(center[1]+0.5*(1/t)*shape[1]))

        minX = 0 if minX is None else minX
        minY = 0 if minY is None else minY
        maxX = shape[0] if maxX is None else maxX
        maxY = shape[1] if maxY is None else maxY

        SquareMarks.set_marks(self, minX, maxX, minY, maxY, shape=shape)

    @Debugger.static_method_decorator('Finding relative positions')
    def relative(minX=None, maxX=None, minY=None, maxY=None, shape=None, zoom_factor=None, center=None, debugger: Debugger = Debugger(debug=False)):
        if shape is None:
            debugger.log_exception('You need to specify  shape!')
            
        if zoom_factor is not None and (minX is not None or maxX is not None or minY is not None or maxY is not None):
            debugger.log_exception(
                'zoom_factor and (minX, maxX, minY, max_Y) can\'t have value at the same time!')

        if zoom_factor is not None:
            if center is None:
                center = (shape[0]/2, shape[1]/2)
            t = float(zoom_factor)
            minX, maxX, minY, maxY = (center[0]-0.5*(1/t), 
                                      center[0]+0.5*(1/t), 
                                      center[1]-0.5*(1/t), 
                                      center[1]+0.5*(1/t))

        minX = 0.0 if minX is None else minX
        minY = 0.0 if minY is None else minY
        maxX = 1.0 if maxX is None else maxX
        maxY = 1.0 if maxY is None else maxY
        
        sm = SquareMarks(shape=shape).set_marks(int(minX*shape[0]), int(maxX*shape[0]), int(minY*shape[1]), int(maxY*shape[1]), shape=shape)
        return sm
        
    @Debugger.method_decorator('Setting SquareMarks')
    def set_marks(self, minX, maxX, minY, maxY, shape=None):
        if shape is None:
            shape = self.shape()
            
        if minX > maxX or minY > maxY:
            self._debugger.log_exception(
                'minX and minY must be less than maxX and maxY!')

        x = [minX, maxX, maxX, minX]
        y = [minY, minY, maxY, maxY]
        MarksLand.set_marks(self, x, y, shape=shape)
        return self

    @Debugger.method_decorator('Enlarging or shrinking the square')
    def expand(self, value):
        minX, maxX, minY, maxY = self.minX(), self.maxX(), self.minY(), self.maxY()
        lenX = (maxX-minX)*(value-1)/2
        lenY = (maxY-minY)*(value-1)/2
        return self.set_marks(minX-lenX, maxX+lenX, minY-lenY, maxY+lenY, shape=self.shape())
        
    @Debugger.method_decorator('Rotating marks')
    def rotate(self, angle: float, center = None, save_trace=False):
        self._debugger.log_exception('You can\'t rotate SquareMarks (use object.to_quad())')
    
    @Debugger.method_decorator('Converting to QuadMarks')
    def to_quad(self, copy_trace=False):
        return self.copy(copy_trace, copy_type=QuadMarks)
        
    def minX(self):
        return np.min(self._flow_data[0][:, 0])

    def maxX(self):
        return np.max(self._flow_data[0][:, 0])

    def minY(self):
        return np.min(self._flow_data[0][:, 1])

    def maxY(self):
        return np.max(self._flow_data[0][:, 1])

    def area(self):
        return (self.maxX()-self.minX())*(self.maxY()-self.minY())

    def subscription(self, other):
        inter_minX = max(self.minX(), other.minX())
        inter_maxX = min(self.maxX(), other.maxX())
        inter_minY = max(self.minY(), other.minY())
        inter_maxY = min(self.maxY(), other.maxY())

        if inter_minX < inter_maxX and inter_minY < inter_maxY:
            return SquareMarks(inter_minX, inter_maxX, inter_minY, inter_maxY)
        else:
            return None

