"""Mutable containers with inverse operations and independent checkpoints."""

from copy import deepcopy

from .debugger import Debugger


class Land:
    """Base transformation flow.

    ``retrace()`` applies the last recorded inverse to current data.
    Checkpoints restore exact snapshots, including the inverse stack.
    Call ``make_checkpoint()`` before edits, then ``undo()`` / ``redo()``.
    """

    def __init__(self, debugger=None):
        self._debugger = debugger if debugger is not None else Debugger(debug=False)
        self._flow_data = None
        self._trace, self._undo, self._redo = [], [], []

    @Debugger.method_decorator("Land.set_debugger")
    def set_debugger(self, debugger):
        """Use a Debugger for subsequent operations; return this container."""
        self._debugger = debugger
        return self

    def _set_data(self, data):
        self._flow_data = data
        self._redo.clear()
        return self

    def _get_data(self):
        return self._flow_data

    @Debugger.method_decorator("Land.copy")
    def copy(self, copy_trace=False, copy_type=None):
        """Copy data and metadata, optionally keeping independent history."""
        result = object.__new__(copy_type or type(self))
        for key, value in self.__dict__.items():
            if key not in {"_debugger", "_trace", "_undo", "_redo"}:
                setattr(result, key, deepcopy(value))
        result._debugger = self._debugger
        for key in ("_trace", "_undo", "_redo"):
            setattr(result, key, deepcopy(getattr(self, key)) if copy_trace else [])
        return result

    def _make_trace(self, inv_fun, pars):
        self._trace.append((inv_fun, deepcopy(pars)))
        self._redo.clear()

    @Debugger.method_decorator("Land.retrace")
    def retrace(self):
        """Apply the latest inverse; raise IndexError if empty.

        Resampling inverses are approximate. Crop inverses paste edited pixels
        into the original canvas. Use checkpoints for exact restoration.
        """
        if not self._trace:
            raise IndexError("No transformation to retrace.")
        function, parameters = self._trace[-1]
        self._flow_data = function(self._flow_data, deepcopy(parameters), self._debugger)
        self._trace.pop()
        self._redo.clear()
        return self

    def _snapshot(self):
        return deepcopy((self._flow_data, self._trace))

    def _restore(self, state):
        self._flow_data, self._trace = deepcopy(state)

    @Debugger.method_decorator("Land.make_checkpoint")
    def make_checkpoint(self):
        """Save current data and traces as an undo point; discard redo."""
        self._undo.append(self._snapshot())
        self._redo.clear()
        return self

    @Debugger.method_decorator("Land.undo")
    def undo(self):
        """Restore the latest checkpoint; retain current state for redo."""
        if not self._undo:
            raise IndexError("No checkpoint to undo.")
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return self

    @Debugger.method_decorator("Land.redo")
    def redo(self):
        """Restore the last undone state; raise IndexError if empty."""
        if not self._redo:
            raise IndexError("No checkpoint to redo.")
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return self

    @Debugger.method_decorator("Land.clear_trace")
    def clear_trace(self):
        """Checkpoint the current state and forget recorded inverses."""
        self.make_checkpoint()
        self._trace.clear()
        return self

    @Debugger.method_decorator("Land.clear_checkpoint_history")
    def clear_checkpoint_history(self):
        """Forget undo and redo points without changing data or traces."""
        self._undo.clear()
        self._redo.clear()
        return self

    def __getitem__(self, index):
        """Return indexed data."""
        return self._flow_data[index]
