from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable
from landpy import Debugger
import inspect


class Land:
    def __init__(self, debugger=Debugger(debug=False)):
        self._debugger = debugger
        self.__trace: List[Dict[Literal['inv_fun', 'pars'],
                                Tuple[Callable, Dict]]] = []
        self._flow_data: Any = None
        self.__checkpoint_history: List[Dict[Literal['trace',
                                                     'flow_data'], Tuple[List, Any]]] = []
        self.__checkpoint_version: int = -1

    def __del__(self):
        del self._debugger
        del self._flow_data
        del self.__trace
        del self.__checkpoint_history

    def set_debugger(self, debugger):
        self._debugger = debugger
        
    @Debugger.method_decorator('Setting flow_data', ['Land._set_data', 'Land'])
    def _set_data(self, data: Any):
        self._flow_data = data
        return self

    @Debugger.method_decorator('Returning flow_data', ['Land._get_data', 'Land'])
    def _get_data(self):
        return self._flow_data

    @Debugger.method_decorator('Copying Land', ['Land.copy', 'Land'])
    def copy(self, save_trace=True):
        dir = self.__copy_trace()
        copied_land = type(self)()
        if save_trace:
            copied_land.__trace = dir['trace']
            copied_land._debugger = self._debugger
        copied_land._flow_data = dir['flow_data']
        return copied_land

    @Debugger.method_decorator('Making trace', ['Land._make_trace', 'Land'])
    def _make_trace(self,  inv_fun: Callable, pars: Dict):
        self.__trace.append({'inv_fun': inv_fun, 'pars': pars})

    @Debugger.method_decorator('Retracing land', ['Land.retrace', 'Land'])
    def retrace(self):
        if len(self.__trace) == 0:
            self._debugger.log_exception(
                'Trace list is empty!', self._debugger.get_last_log_categories())

        last_call = self.__trace.pop()
        inv_fun, pars = last_call['inv_fun'], last_call['pars']
        self._flow_data = inv_fun(self._flow_data, pars, self._debugger)

    @Debugger.method_decorator('Copying the whole trace', ['Land.__copy_trace', 'Land'])
    def __copy_trace(self, copy_trace=True):
        copied_trace = []
        if copy_trace:
            for item in self.__trace:
                copied_pars = {}
                for key in item['pars']:
                    if hasattr(item['pars'][key], 'copy'):
                        copied_pars.update({key: item['pars'][key].copy()})
                    else:
                        copied_pars.update({key: item['pars'][key]})
                copied_trace.append(
                    {'inv_fun': item['inv_fun'], 'pars': copied_pars})

        flow_data = self._flow_data.copy() if hasattr(
            self._flow_data, 'copy') else self._flow_data
        return {'trace': copied_trace, 'flow_data': flow_data}

    @Debugger.method_decorator('Making checkpoint', ['Land.make_checkpoint', 'Land'])
    def make_checkpoint(self):
        self._debugger.log('version {}'.format(
            self.__checkpoint_version+1), categories=self._debugger.get_last_log_categories())

        while self.__checkpoint_version+1 < len(self.__checkpoint_history):
            self.__checkpoint_history.pop()
            self._debugger.log('Version {} deleted!'.format(len(
                self.__checkpoint_history)), categories=self._debugger.get_last_log_categories())

        self.__checkpoint_history.append(self.__copy_trace())
        self.__checkpoint_version += 1

    @Debugger.method_decorator('Undoing Land', ['Land.undo', 'Land'])
    def undo(self):
        self.__trace = self.__checkpoint_history[self.__checkpoint_version]['trace']
        self._flow_data = self.__checkpoint_history[self.__checkpoint_version]['flow_data']
        self.__checkpoint_version -= 1
        self._debugger.log('undoed to version {}!'.format(
            self.__checkpoint_version), categories=self._debugger.get_last_log_categories())

    @Debugger.method_decorator('Redoing Land', ['Land.redo', 'Land'])
    def redo(self):
        self.__checkpoint_version += 1
        self.__trace = self.__checkpoint_history[self.__checkpoint_version]['trace']
        self._flow_data = self.__checkpoint_history[self.__checkpoint_version]['flow_data']
        self._debugger.log('redoed to version {}!'.format(
            self.__checkpoint_version), categories=self._debugger.get_last_log_categories())

    @Debugger.method_decorator('Clearing trace', ['Land.clear_trace', 'Land'])
    def clear_trace(self):
        self.make_checkpoint()
        self.__trace = []

    @Debugger.method_decorator('Clearing checkpoint history', ['Land.clear_checkpoint_history', 'Land'])
    def clear_checkpoint_history(self):
        del self.__checkpoint_history
        self.__checkpoint_history = []
        self.__checkpoint_version = -1

    @Debugger.method_decorator('Getting i\'th item of land', ['Land.__getitem__', 'Land'])
    def __getitem__(self, ind):
        return self._flow_data[ind]
