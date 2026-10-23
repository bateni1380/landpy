# API guide

## Coordinates and ownership

Every shape is `(height, width)`; channel counts are ignored in shape arguments.
All point coordinates and rotation centers are `(row, column)` pixels. Angles
are degrees counterclockwise. Rectangle upper crop bounds are exclusive. Crop
minima are floored and maxima are ceiled; fractional landmark coordinates remain
floating point. Crop regions may extend outside the canvas. `zoom_factor=2`
selects half the canvas extent around the chosen center; it must be positive.

Each setter copies caller data. Array getters expose the current mutable array;
use `copy()` for independent edits. Histories and traces in `copy(True)` are also
independent. Debuggers are intentionally shared between copies. An empty
constructor allows deferred data assignment; transformations require data.

## Containers and operations

| API | Behavior |
| --- | --- |
| `ImageLand(image=None, debugger=None)` | Accept NumPy RGB/RGBA/grayscale, Pillow images, str/Path paths |
| `set_image`, `get_image`, `shape`, `mode` | Set copied data; get live array and spatial/channel metadata |
| `to_rgba()`, `to_rgb()` | Opaque alpha; alpha composite onto black |
| `rotate(angle, center=None, save_trace=False, fill_color=...)` | Fixed-canvas rotation; pixels outside canvas are lost |
| `reshape(shape, mode=None, save_trace=False)` | Bilinear resize; preserve mode unless specified |
| `zoom(region=None, zoom_factor=None, center=None, save_trace=False, fill_color=...)` | Crop/pad or perspective rectify |
| `equalize_histogram(save_trace=False)` | Equalize Lab luminance, preserve alpha; inverse restores saved luminance with possible color-rounding loss |
| `implant_filter(filter, heat=None)` | Default full mask; mismatched heat resized on a copy |
| `implant_image(plant_img, src_land=None, plant_land=None, ratio=.5)` | Straight-alpha overlay with perspective positioning; ratio in [0, 1] |
| `image + image` | Average images; align size/mode using a copy of the right image |
| `image + marks`, `image + heat` | Draw points/regions/vectors, or apply an orange mask overlay |
| `show(desc='')` | Render with Matplotlib, return self |

`MarksLand(x=None, y=None, shape=None)` takes an N×2 array, a single pair, or
separate equally shaped row/column arrays. Empty sets are supported. `get_marks()`
returns the point array; `to_list()` returns `(points, shape)`; indexing returns
a MarksLand. `reshape`, `rotate`, `move(vector)` and `zoom` accept `save_trace`.
Point union (`+`) deduplicates in stable order; subtraction removes exact point
matches. A different-size right operand is aligned on a copy. `to_heat(fill_heat)`
fills a polygon; `to_square()` computes bounds and rejects empty input.

`SquareMarks(minX, maxX, minY, maxY, shape=...)` is an axis-aligned rectangle.
Use `SquareMarks.relative(..., shape=...)` for fractions. `area()`, `center()`,
`expand(factor)`, bound getters, and `subscription(other)` (intersection) retain
historical names. Zero-area rectangles are allowed for geometry but rejected as
crops. Rotate `box.to_quad()`, since a rotated box is no longer axis-aligned.

`QuadMarks(p1, p2, p3, p4, shape=...)` validates convex corners, sorts clockwise
from top-left, and finds the diagonal intersection with `center()`. Degenerate,
concave and repeated corners are rejected. `DualMarks(start, end, shape=...)`
accepts coordinate pairs or one-point MarksLand objects. `length()`/`lenght()`
and `angle()` measure the vector. Convert specialized regions to general marks
before arbitrary point edits that would violate their geometry.

`HeatLand(heat_data=None, shape=None, fill_value=0)` owns a finite 2D float32
array in [0, 1]. `set_heat`, `get_heat`, `shape`, `show`, `rotate`, `reshape`, and
`zoom` mirror image operations. `+` means maximum, `-` means minimum, unary `-`
means complement. Scalar multiplication clips to [0, 1]; nonnegative powers
adjust strength. Binary heat maps require equal shapes. Resizing does not
quantize heat into 8-bit pixels.

`Filter(color=(255,255,255), ratio=1)` accepts RGB or #RRGGBB. Its default
formula is `pixels*(1-heat*ratio) + color*heat*ratio`.
`set_function(callable)` replaces it with `callable(pixels, broadcast_heat)`.

## History

`Land` provides `copy(copy_trace=False, copy_type=None)`, `make_checkpoint()`,
`undo()`, `redo()`, `retrace()`, `clear_trace()`, `clear_checkpoint_history()`, and
`set_debugger()`. `clear_trace()` first creates an undo point. A failed inverse
leaves its trace in place. There is no automatic disk persistence of history.

Crop retracing restores the old canvas shape and pastes current crop pixels;
fully transparent RGBA pixels leave old pixels intact. Restore channel mode and
retrace any subsequent resize before retracing a crop. Interpolation and
out-of-frame information loss make rotation/perspective inverses approximate.
Checkpoints provide exact restoration and consume memory proportional to the
stored arrays; clear history when it is no longer needed.

## Debugger

`Debugger(categories=None, textual_debug=False, visual_debug=False, debug=True,
show_file=False, show_line=False)` supports `log`, `log_start`, `log_end`,
`log_warning`, and `log_exception`. `debug=False` suppresses output; exceptions
still raise ValueError. Empty categories allow everything; otherwise any class,
method or qualified-name match admits a message. Containers use a disabled
Debugger by default.

`log_image` / `log_images` support `LogTypes.IMAGE`, `HEAT`, `SCATTER`, `VECTOR`.
Point/vector payloads are `[point_array, canvas_shape]`. `save_image` queues a
copy and `log_saved_images` displays and clears the queue. Grid dimensions must
fit all images. `method_decorator` uses `self._debugger`;
`static_method_decorator` binds a `debugger` argument (including defaults). Both
preserve `help()` information and signatures, and unwind scopes on exceptions.

## Models

See [models.md](models.md). `TFModel` and `TFLiteModel` live in `landpy.tf_model`
and `landpy.tf_lite_model`; import the four face classes from
`landpy.pretrained_models`. Model constructors accept `model_path`, `cache_dir`
and `sha256` keyword arguments. Model is abstract; subclasses implement
`_load_model()` and `inference(input)`.
