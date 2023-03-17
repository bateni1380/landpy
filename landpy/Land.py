from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable
from landpy import Debugger
import inspect

class Land:
    def __init__(self, debugger=Debugger(debug=False)):
        self._debugger = debugger
        self.__trace: List[Dict[Literal['inv_fun', 'pars'], Tuple[Callable, Dict]]] = []
        self._flow_data: Any = None
        self.__checkpoint_history: List[Dict[Literal['trace', 'flow_data'], Tuple[List, Any]]] = []
        self.__checkpoint_version: int = -1
    
    def __del__(self):
        del self._debugger
        del self._flow_data
        del self.__trace
        del self.__checkpoint_history

    @Debugger.method_decorator(['Land._set_data', 'Land'], 'Sending data to flow_data')    
    def _set_data(self, data: Any):
        self._flow_data = data
        return self
    
    @Debugger.method_decorator(['Land._set_data', 'Land'], 'Returning flow_data')    
    def _get_data(self):
        return self._flow_data
    
    @Debugger.method_decorator(['Land.copy', 'Land'], 'Copying Land')    
    def copy(self):
        dir = self.__copy_trace()
        copied_land = type(self)(self._debugger)
        copied_land.__trace = dir['trace']
        copied_land._flow_data = dir['flow_data']
        return copied_land


    @Debugger.method_decorator(['Land._make_trace', 'Land'], 'Making trace')    
    def _make_trace(self,  inv_fun: Callable, pars: Dict):
        self.__trace.append({'inv_fun': inv_fun, 'pars': pars})
    
    @Debugger.static_method_decorator(['Exception', 'Land._make_exception', 'Land'], 'save_trace parameter was false so you can\'t retrace!')
    def _make_exception(data, pars, debugger):
        raise Exception('save_trace parameter was false so you can\'t retrace!')
    
    @Debugger.method_decorator(['Land.retrace', 'Land'], 'Retracing land')   
    def retrace(self):
        if len(self.__trace)==0:
            self._debugger.log('Trace list is empty!', categories=['Exception']+self._debugger.get_last_log_categories())
            raise Exception('Trace list is empty!')
        last_call = self.__trace.pop()
        inv_fun, pars = last_call['inv_fun'], last_call['pars'] 
        self._flow_data = inv_fun(self._flow_data, pars, self._debugger)

    @Debugger.method_decorator(['Land.__copy_trace', 'Land'], 'Copying the whole trace')   
    def __copy_trace(self):
        copied_trace = []
        for item in self.__trace:
            copied_pars = {}
            for key in item['pars']:
                if hasattr(item['pars'][key], 'copy'):
                    copied_pars.update({key:item['pars'][key].copy()})
                else:
                    copied_pars.update({key:item['pars'][key]})
            copied_trace.append({'inv_fun':item['inv_fun'], 'pars':copied_pars})

        flow_data = self._flow_data.copy() if hasattr(self._flow_data, 'copy') else self._flow_data
        return {'trace':copied_trace, 'flow_data':flow_data}
    
    @Debugger.method_decorator(['Land.make_checkpoint', 'Land'], 'Making checkpoint')  
    def make_checkpoint(self):
        self._debugger.log('version {}'.format(self.__checkpoint_version+1), categories=self._debugger.get_last_log_categories())
        
        while self.__checkpoint_version+1 < len(self.__checkpoint_history):
            self.__checkpoint_history.pop()
            self._debugger.log('Version {} deleted!'.format(len(self.__checkpoint_history)), categories=self._debugger.get_last_log_categories())

        self.__checkpoint_history.append(self.__copy_trace())
        self.__checkpoint_version += 1

    @Debugger.method_decorator(['Land.undo', 'Land'], 'Undoing Land')  
    def undo(self):
        self.__trace = self.__checkpoint_history[self.__checkpoint_version]['trace']
        self._flow_data = self.__checkpoint_history[self.__checkpoint_version]['flow_data']
        self.__checkpoint_version -= 1
        self._debugger.log('undoed to version {}!'.format(self.__checkpoint_version), categories=self._debugger.get_last_log_categories())
    
    @Debugger.method_decorator(['Land.redo', 'Land'], 'Redoing Land')  
    def redo(self):
        self.__checkpoint_version += 1
        self.__trace = self.__checkpoint_history[self.__checkpoint_version]['trace']
        self._flow_data = self.__checkpoint_history[self.__checkpoint_version]['flow_data']
        self._debugger.log('redoed to version {}!'.format(self.__checkpoint_version), categories=self._debugger.get_last_log_categories())
    
    @Debugger.method_decorator(['Land.clear_trace', 'Land'], 'Clearing trace')  
    def clear_trace(self):
        self.make_checkpoint()
        self.__trace = []
        
    @Debugger.method_decorator(['Land.clear_checkpoint_history', 'Land'], 'Clearing checkpoint history')  
    def clear_checkpoint_history(self):
        del self.__checkpoint_history
        self.__checkpoint_history = []        