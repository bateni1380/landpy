from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable
from landpy import Debugger
import inspect

class Land:
    def __init__(self, debugger=Debugger(debug=False)):
        self.__trace: List[Dict[Literal['inv_fun', 'pars'], Tuple[Callable, Dict]]] = []
        self._flow_data: Any = None
        self.__checkpoint_history: List[Dict[Literal['trace', 'flow_data'], Tuple[List, Any]]] = []
        self.__checkpoint_version: int = -1
        self._debugger = debugger
    
    def __del__(self):
        del self._debugger
        del self._flow_data
        del self.__trace
        del self.__checkpoint_history
        
    def _set_data(self, data: Any):
        self._debugger.log_start('Sending data to flow_data', categories=['Land._set_data', 'Land'])
        self._flow_data = data
        self._debugger.log_end(categories=['Land._set_data', 'Land'])
    
    def _make_trace(self,  inv_fun: Callable, pars: Dict):
        self._debugger.log_start('Making trace', categories=['Land._make_trace', 'Land'])
        self.__trace.append({'inv_fun': inv_fun, 'pars': pars})
        self._debugger.log_end(categories=['Land._make_trace', 'Land'])
     
    def retrace(self):
        self._debugger.log_start('Retracing', categories=['Land.retrace', 'Land'])
        
        if len(self.__trace)==0:
            self._debugger.log('Trace list is empty!', categories=['Exception', 'Land.retrace', 'Land'])
            raise Exception('Trace list is empty!')
            
        inv_fun, pars = self.__trace[-1]['inv_fun'], self.__trace[-1]['pars'] 
        self._flow_data = inv_fun(self._flow_data, pars, self._debugger)
        
        self._debugger.log_end(categories=['Land.retrace', 'Land'])
    
    def __copy_trace(self):
        # TODO........................
        return self.__trace
    
    def make_checkpoint(self):
        self._debugger.log_start('Making checkpoint version {} (copying trace)'.format(self.__checkpoint_version+1), categories=['Land.make_checkpoint', 'Land'])
        
        while self.__checkpoint_version+1 < len(self.__checkpoint_history):
            self.__checkpoint_history.pop()
            self._debugger.log('Version {} deleted!'.format(len(self.__checkpoint_history)), categories=['Land.make_checkpoint', 'Land'])

        self.__checkpoint_history.append(self.__copy_trace())
        self.__checkpoint_version += 1
        # TODO........................
        self._debugger.log_end(categories=['Land.make_checkpoint', 'Land'])

    def undo(self):
        self.__checkpoint_version -= 1
        self._debugger.log('Land undoed to version {}!'.format(self.__checkpoint_version), categories=['Land.undo', 'Land'])
        
    def redo(self):
        self.__checkpoint_version += 1
        self._debugger.log('Land redoed to version {}!'.format(self.__checkpoint_version), categories=['Land.undo', 'redo'])
    
    def clear_trace(self):
        self._debugger.log_start('Clearing trace', categories=['Land.clear_trace', 'Land'])
        self.make_checkpoint(self._debugger)
        self.__trace = []
        self._debugger.log_end(categories=['Land.clear_trace', 'Land'])
        
    def clear_checkpoint_history(self):
        self._debugger.log_start('Clearing checkpoint history', categories=['Land.clear_checkpoint_history', 'Land'])
        self.__checkpoint_history = []
        self._debugger.log_end(categories=['Land.clear_checkpoint_history', 'Land'])\
        
    def _get_data(self):
        self._debugger.log_start('Gending data from land', categories=['Land._get_data', 'Land'], start_process=True)
        data = self._flow_data
        self._debugger.log_end(categories=['Land._get_data', 'Land'])
        return data
    