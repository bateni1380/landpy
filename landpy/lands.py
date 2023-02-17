from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable
from landpy.debugger import Debugger
import inspect

class Land:
    def __init__(self):
        self.__trace: List[Dict[Literal['inv_fun', 'pars'], Tuple[Callable, Dict]]] = []
        self.__flow_data: Any = None
        self.__checkpoint_history: List[Dict[Literal['trace', 'flow_data'], Tuple[List, Any]]] = []
        self.__checkpoint_version: int = -1
    
    def _set_data(self, data: Any, debugger: Debugger):
        debugger.log('Sending data to flow_data...', categories=['Land._set_data', 'Land'], start_process=True)
        self.__flow_data = data
        debugger.log('Data sent to flow_data.', categories=['Land._set_data', 'Land'])
    
    def _make_trace(self, description: str, inv_fun: Callable, pars: Dict, debugger: Debugger):
        debugger.log('Making trace...', categories=['Land._make_trace', 'Land'], start_process=True)
        self.__trace.append({'inv_fun': inv_fun, 'pars': pars})
        debugger.log('Trace made.', categories=['Land._make_trace', 'Land'])
     
    def retrace(self, debugger: Debugger):
        debugger.log('Retracing...', categories=['Land.retrace', 'Land'], start_process=True)
        
        if len(self.__trace)==0:
            debugger.log('Trace list is empty!', categories=['Land.retrace', 'Land'])
            raise Exception('Trace list is empty!')
            
        inv_fun, pars = self.__trace[-1]['inv_fun'], self.__trace[-1]['pars'] 
        self.__flow_data = inv_fun(self.__flow_data, pars)
        
        debugger.log('Retraced.', categories=['Land.retrace', 'Land'])
    
    def __copy_trace(self):
        # TODO........................
        return self.__trace
    
    def make_checkpoint(self, debugger: Debugger):
        debugger.log('Making checkpoint (copying trace)...', categories=['Land.make_checkpoint', 'Land'], start_process=True)
        
        while self.__checkpoint_version+1 < len(self.__checkpoint_history):
            self.__checkpoint_history.pop()
            debugger.log('Version {} deleted!'.format(len(self.__checkpoint_history)), categories=['Land.make_checkpoint', 'Land'])

        self.__checkpoint_history.append(self.__copy_trace())
        self.__checkpoint_version += 1
        # TODO........................
        debugger.log('Checkpoint version {}, {} made.'.format(self.__checkpoint_version, len(self.__checkpoint_history)), categories=['Land.make_checkpoint', 'Land'])

    def undo(self, debugger: Debugger):
        self.__checkpoint_version -= 1
        debugger.log('Land undoed to version {}!'.format(self.__checkpoint_version), categories=['Land.undo', 'Land'])
        
    def redo(self, debugger: Debugger):
        self.__checkpoint_version += 1
        debugger.log('Land redoed to version {}!'.format(self.__checkpoint_version), categories=['Land.undo', 'redo'])
    
    def clear_trace(self, debugger: Debugger):
        debugger.log('Clearing trace...', categories=['Land.clear_trace', 'Land'], start_process=True)
        self.make_checkpoint(debugger)
        self.__trace = []
        debugger.log('Trace cleared!...', categories=['Land.clear_trace', 'Land'])
        
    def clear_checkpoint_history(self, debugger: Debugger):
        debugger.log('Clearing checkpoint history...', categories=['Land.clear_checkpoint_history', 'Land'], start_process=True)
        self.__checkpoint_history = []
        debugger.log('Checkpoint history cleared!', categories=['Land.clear_checkpoint_history', 'Land'])\
        
    def _get_data(self, debugger: Debugger):
        debugger.log('Gending data from land...', categories=['Land._get_data', 'Land'], start_process=True)
        data = self.__flow_data
        debugger.log('Data got from land.', categories=['Land._get_data', 'Land'])
        return data
    
class ImageLand(Land):
    def __init__(self):
        super().__init__()
        