"""Optional categorized timing logs and visual inspection."""

import inspect
import time
import warnings
from copy import deepcopy
from functools import wraps
from pathlib import Path


class Debugger:
    """Control textual and visual logs with optional category filtering.

    ``debug=False`` disables logs. Empty categories accept all events;
    otherwise any matching category admits an event. Plotting is imported only
    when needed. Decorators preserve signatures and unwind after exceptions.
    """

    class LogTypes:
        """Kinds accepted by the visual logging methods."""

        IMAGE, HEAT, SCATTER, VECTOR = range(4)

    def __init__(
        self,
        categories=None,
        textual_debug=False,
        visual_debug=False,
        debug=True,
        show_file=False,
        show_line=False,
    ):
        self.debug = debug
        self.textual_debug = textual_debug
        self.visual_debug = visual_debug
        self.categories = set(categories or [])
        self.show_file, self.show_line = show_file, show_line
        self._times, self._categories, self._images = [], [[]], []

    def append_last_log_categories(self, categories):
        """Push categories inherited by nested logs."""
        self._categories.append(list(categories))

    def get_last_log_categories(self):
        """Return active inherited categories."""
        return list(self._categories[-1])

    def pop_last_log_categories(self):
        """Leave the most recent category scope."""
        if len(self._categories) > 1:
            self._categories.pop()

    def _accepts(self, categories):
        categories = self.get_last_log_categories() if categories is None else categories
        return (
            not self.categories or not categories or bool(self.categories.intersection(categories))
        )

    def log(self, desc, categories=None, end="\n"):
        """Print an event when textual logging and its categories are enabled."""
        if not (self.debug and self.textual_debug and self._accepts(categories)):
            return
        location = ""
        if self.show_file or self.show_line:
            frame = inspect.currentframe().f_back
            parts = []
            if self.show_file:
                parts.append(Path(frame.f_code.co_filename).name)
            if self.show_line:
                parts.append(str(frame.f_lineno))
            location = "[" + ":".join(parts) + "] "
        timing = f" ({time.perf_counter() - self._times[-1][1]:.4f}s)" if self._times else ""
        print("  " * len(self._times) + location + str(desc) + timing, end=end)

    def log_start(self, desc, categories=None):
        """Start a timed process, including processes filtered out of output."""
        self.log(str(desc) + " started", categories)
        self._times.append((desc, time.perf_counter()))

    def log_end(self, categories=None):
        """End the most recently started process; raise RuntimeError if empty."""
        if not self._times:
            raise RuntimeError("No process to end.")
        self.log(str(self._times[-1][0]) + " ended", categories)
        self._times.pop()

    def log_exception(self, desc, categories=None):
        """Log a failure and raise ValueError even when logging is disabled."""
        self.log(desc, categories)
        raise ValueError(desc)

    def log_warning(self, desc, categories=None):
        """Emit a warning when debugging and its categories are enabled."""
        if self.debug and self._accepts(categories):
            warnings.warn(desc, stacklevel=2)

    def log_images(self, images, descs, types, dim=None, categories=None):
        """Show RGB images, heat maps, point sets or vectors in a subplot grid."""
        if not (self.debug and self.visual_debug and self._accepts(categories)) or len(images) == 0:
            return
        import matplotlib.pyplot as plt

        if len(images) != len(descs) or len(images) != len(types):
            raise ValueError("images, descs and types must have equal lengths.")
        rows, columns = dim or (1, len(images))
        if rows * columns < len(images):
            raise ValueError("Grid is too small for the images.")
        fig, axes = plt.subplots(
            rows, columns, squeeze=False, figsize=(4 * columns, 4 * rows), layout="constrained"
        )
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, data, title, kind in zip(axes.flat, images, descs, types):
            ax.set_visible(True)
            if kind in (self.LogTypes.SCATTER, self.LogTypes.VECTOR):
                points, shape = data
                ax.scatter(points[:, 1], points[:, 0])
                ax.set_xlim(0, shape[1])
                ax.set_ylim(shape[0], 0)
                ax.set_aspect("equal")
                if kind == self.LogTypes.VECTOR:
                    ax.annotate(
                        "",
                        xy=points[1, ::-1],
                        xytext=points[0, ::-1],
                        arrowprops={"arrowstyle": "->"},
                    )
            elif kind == self.LogTypes.HEAT:
                artist = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1)
                fig.colorbar(artist, ax=ax, shrink=0.75)
            else:
                ax.imshow(data)
            ax.set_title(title or "")
        plt.show()
        return fig

    def log_image(self, image, desc="", type=0, categories=None):
        """Show one image; ``type`` is a Debugger.LogTypes constant."""
        return self.log_images([image], [desc], [type], categories=categories)

    def save_image(self, image, desc="", type=0, categories=None):
        """Queue an independent visual snapshot for ``log_saved_images``."""
        if self.debug and self.visual_debug and self._accepts(categories):
            self._images.append((deepcopy(image), desc, type))

    def log_saved_images(self, dim=None, categories=None):
        """Display queued snapshots and clear them, including after errors."""
        try:
            if self._images:
                images, titles, types = zip(*self._images)
                return self.log_images(images, titles, types, dim, categories)
        finally:
            self._images.clear()

    @staticmethod
    def method_decorator(desc, log_categories=None):
        """Wrap an instance method using its ``_debugger``."""

        def decorate(function):
            @wraps(function)
            def wrapped(self, *args, **kwargs):
                return Debugger._call(
                    self._debugger, function, desc, log_categories, (self, *args), kwargs
                )

            return wrapped

        return decorate

    @staticmethod
    def static_method_decorator(desc, log_categories=None):
        """Wrap a function whose signature contains a ``debugger`` parameter."""

        def decorate(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                bound = inspect.signature(function).bind(*args, **kwargs)
                bound.apply_defaults()
                debugger = bound.arguments.get("debugger") or Debugger(debug=False)
                return Debugger._call(debugger, function, desc, log_categories, args, kwargs)

            return wrapped

        return decorate

    @staticmethod
    def _call(debugger, function, desc, categories, args, kwargs):
        categories = (
            categories
            if categories is not None
            else [function.__qualname__, *function.__qualname__.split(".")]
        )
        debugger.append_last_log_categories(categories)
        debugger.log_start(desc)
        try:
            return function(*args, **kwargs)
        finally:
            debugger.log_end()
            debugger.pop_last_log_categories()
