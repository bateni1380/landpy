import inspect

import numpy as np
import pytest
from PIL import Image

from landpy import (
    Debugger,
    DualMarks,
    Filter,
    HeatLand,
    ImageLand,
    Land,
    MarksLand,
    QuadMarks,
    SquareMarks,
)


@pytest.fixture
def pixels():
    rows, cols = np.indices((12, 20))
    return np.stack((rows * 10, cols * 10, rows + cols), axis=-1).astype(np.uint8)


def test_checkpoint_branching_and_deep_copy():
    marks = MarksLand([[1, 2], [3, 4]], shape=(10, 20))
    marks.make_checkpoint().move((5, 6), save_trace=True)
    clone = marks.copy(copy_trace=True)
    marks.undo()
    np.testing.assert_equal(marks.get_marks(), [[1, 2], [3, 4]])
    marks.redo()
    np.testing.assert_equal(marks.get_marks(), [[6, 8], [8, 10]])
    marks.undo().move((1, 0))
    with pytest.raises(IndexError):
        marks.redo()
    clone.undo()
    np.testing.assert_equal(clone.get_marks(), [[1, 2], [3, 4]])
    clone.make_checkpoint()
    clone.get_marks()[0, 0] = 999
    clone.undo()
    assert clone.get_marks()[0, 0] == 1


@pytest.mark.parametrize("method", ["undo", "redo", "retrace"])
def test_empty_history(method):
    with pytest.raises(IndexError):
        getattr(Land(), method)()


def test_clear_history_and_trace():
    marks = MarksLand([[1, 2]], shape=(10, 10)).move((2, 2), True)
    marks.clear_trace()
    with pytest.raises(IndexError):
        marks.retrace()
    marks.undo().retrace()
    np.testing.assert_allclose(marks.get_marks(), [[1, 2]])
    marks.clear_checkpoint_history()
    with pytest.raises(IndexError):
        marks.undo()


@pytest.mark.parametrize(
    "operation",
    [
        lambda m: m.move((2, -7), True),
        lambda m: m.rotate(37, save_trace=True),
        lambda m: m.reshape((50, 120), True),
        lambda m: m.zoom(SquareMarks(2, 8, -3, 14, shape=(10, 20)), save_trace=True),
    ],
)
def test_landmark_inverse(operation):
    marks = MarksLand([[1, 2], [7, 12]], shape=(10, 20))
    original = marks.get_marks().copy()
    operation(marks).retrace()
    np.testing.assert_allclose(marks.get_marks(), original, atol=1e-12)
    assert marks.shape == (10, 20)


def test_rotation_matches_image_geometry():
    mark = MarksLand([[5, 8]], shape=(10, 20)).rotate(90, center=(5, 5))
    np.testing.assert_allclose(mark.get_marks(), [[2, 5]], atol=1e-12)


def test_set_operations_do_not_modify_right_operand():
    a = MarksLand([[1, 2], [3, 4]], shape=(10, 20))
    b = MarksLand([[2, 4]], shape=(20, 40))
    assert (a + b).to_list() == ([[1.0, 2.0], [3.0, 4.0]], (10, 20))
    assert (a - b).to_list() == ([[3.0, 4.0]], (10, 20))
    assert b.to_list() == ([[2.0, 4.0]], (20, 40))
    assert (a - a).get_marks().shape == (0, 2)
    assert a[0].get_marks().shape == (1, 2)


def test_polygon_uses_row_column_order():
    marks = MarksLand([[1, 8], [1, 12], [3, 12], [3, 8]], shape=(8, 16))
    mask = marks.to_heat(0.7).get_heat()
    assert mask[2, 10] == pytest.approx(0.7)
    assert mask[6, 2] == 0
    assert marks.to_square().area() == 8


def test_rectangle_and_vector_geometry():
    box = SquareMarks.relative(0.2, 0.8, 0.25, 0.75, shape=(10, 20))
    assert box.area() == 60
    assert box.center() == [5, 10]
    assert box.copy().expand(2).area() == 240
    assert box.subscription(SquareMarks(0, 4, 0, 8, shape=(10, 20))).area() == 6
    assert box.subscription(SquareMarks(0, 1, 0, 1, shape=(10, 20))) is None
    vector = DualMarks((5, 5), (2, 9), shape=(10, 20))
    assert vector.length() == vector.lenght() == 5
    assert vector.angle() == pytest.approx(36.8698976)
    with pytest.raises(TypeError):
        box.rotate(30)
    assert isinstance(box.to_quad().rotate(30), QuadMarks)


def test_perspective_points_roundtrip():
    quad = QuadMarks((2, 3), (1, 17), (9, 18), (10, 2), shape=(12, 20))
    marks = MarksLand([[3, 4], [6, 10]], shape=(12, 20))
    marks.zoom(quad, save_trace=True).retrace()
    np.testing.assert_allclose(marks.get_marks(), [[3, 4], [6, 10]], atol=1e-10)
    with pytest.raises(ValueError):
        QuadMarks((0, 0), (0, 1), (0, 2), (0, 3))


@pytest.mark.parametrize("kind", [ImageLand, HeatLand])
def test_crop_edit_retrace_and_outside_canvas(kind, pixels):
    data = pixels if kind is ImageLand else pixels[..., 0] / 255
    obj = kind(data)
    original = obj._flow_data.copy()
    crop = SquareMarks(2, 7, 3, 9, shape=(12, 20))
    obj.zoom(crop, save_trace=True)
    assert obj.shape == (5, 6)
    obj._flow_data[:] = 0
    obj.retrace()
    expected = original.copy()
    expected[2:7, 3:9] = 0
    np.testing.assert_allclose(obj._flow_data, expected)
    obj.zoom(SquareMarks(-2, 15, -3, 22, shape=(12, 20)), save_trace=True)
    assert obj.shape == (17, 25)
    obj.retrace()
    np.testing.assert_allclose(obj._flow_data, expected)
    obj.zoom(SquareMarks(40, 45, 40, 45, shape=(12, 20)), save_trace=True).retrace()
    np.testing.assert_allclose(obj._flow_data, expected)


@pytest.mark.parametrize("kind", [ImageLand, HeatLand])
def test_rotation_resize_history(kind, pixels):
    obj = kind(pixels if kind is ImageLand else pixels[..., 0] / 255)
    original = obj._flow_data.copy()
    obj.make_checkpoint().rotate(20, save_trace=True).reshape((9, 14), save_trace=True)
    obj.retrace().retrace()
    assert obj.shape == (12, 20)
    obj.undo()
    np.testing.assert_array_equal(obj._flow_data, original)


def test_image_input_and_alpha(pixels, tmp_path):
    img = ImageLand(pixels)
    pixels[:] = 0
    assert img.get_image().sum() > 0
    img.to_rgba()
    assert img.get_image().dtype == np.uint8
    np.testing.assert_equal(img.get_image()[..., 3], 255)
    img.make_checkpoint().reshape((6, 10), mode="RGB").undo()
    assert img.mode == "RGBA"
    a = np.full((2, 3, 4), 128, dtype=np.uint8)
    np.testing.assert_equal(ImageLand(a).to_rgb().get_image(), 64)
    assert ImageLand(np.full((2, 3), 0.5)).mode == "RGB"
    path = tmp_path / "image.png"
    Image.fromarray(a).save(path)
    assert ImageLand(path).mode == "RGBA"


def test_masked_filter_preserves_alpha():
    image = ImageLand(np.full((4, 6, 4), 128, dtype=np.uint8))
    mask = HeatLand(np.array([[0, 1], [0, 1]], dtype=float))
    before = mask.get_heat().copy()
    image.implant_filter(Filter("#ff0000", 0.5), mask)
    assert image.get_image()[0, 0, 0] == 128
    assert image.get_image()[0, -1, 0] == 191
    np.testing.assert_equal(image.get_image()[..., 3], 128)
    np.testing.assert_equal(mask.get_heat(), before)
    ImageLand(np.zeros((3, 3, 3), dtype=np.uint8)).implant_filter(Filter())


def test_implant_rgb_and_alpha():
    black = np.zeros((10, 20, 3), dtype=np.uint8)
    red = np.full((4, 4, 4), [255, 0, 0, 128], dtype=np.uint8)
    image = ImageLand(black).implant_image(red, SquareMarks(2, 6, 5, 9, shape=(10, 20)), ratio=1)
    np.testing.assert_equal(image.get_image()[3, 6], [128, 0, 0])
    np.testing.assert_equal(image.get_image()[0, 0], [0, 0, 0])
    transparent = ImageLand(np.zeros((4, 4, 4), dtype=np.uint8)).implant_image(red, ratio=1)
    np.testing.assert_equal(transparent.get_image()[1, 1], [255, 0, 0, 128])
    blue = ImageLand(np.full((4, 4, 3), [0, 0, 255], dtype=np.uint8))
    blue.implant_image(red, ratio=0)
    np.testing.assert_equal(blue.get_image()[1, 1], [0, 0, 255])


def test_addition_overlays_are_independent(pixels):
    img = ImageLand(pixels)
    marks = SquareMarks(2, 6, 3, 9, shape=img.shape)
    original = marks.get_marks().copy()
    for other in [
        marks,
        DualMarks((0, 0), (10, 15), shape=img.shape),
        HeatLand(shape=img.shape),
        ImageLand(pixels),
    ]:
        assert (img + other).shape == img.shape
    assert (marks + img).shape == img.shape
    np.testing.assert_equal(marks.get_marks(), original)
    np.testing.assert_equal(img.get_image(), pixels)


def test_heat_algebra_and_precision():
    a = HeatLand([[0.2, 0.7], [0.4, 0.9]])
    b = HeatLand([[0.5, 0.3], [0.1, 0.8]])
    np.testing.assert_allclose((a + b).get_heat(), [[0.5, 0.7], [0.4, 0.9]])
    np.testing.assert_allclose((a - b).get_heat(), [[0.2, 0.3], [0.1, 0.8]])
    np.testing.assert_allclose((-a).get_heat(), 1 - a.get_heat())
    np.testing.assert_allclose((a * 0.5).get_heat(), 0.5 * a.get_heat())
    assert (10 * a).get_heat().max() == 1
    assert (a**2).get_heat()[0, 0] == pytest.approx(0.04)
    precise = HeatLand(np.full((2, 2), 0.123456)).reshape((8, 12))
    assert precise.get_heat()[3, 3] == pytest.approx(0.123456)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HeatLand([[1.1]]),
        lambda: HeatLand([[np.nan]]),
        lambda: ImageLand(np.full((2, 3, 3), 2.0)),
        lambda: MarksLand([[1, 2, 3]]),
        lambda: SquareMarks(shape=(0, 10)),
        lambda: SquareMarks(shape=(10, 10), zoom_factor=0),
        lambda: Filter("#bad"),
        lambda: Filter(ratio=2),
    ],
)
def test_invalid_data_rejected(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_decorator_exception_cleanup_and_signature(capsys):
    debugger = Debugger(textual_debug=True, categories=["Example"])

    class Example:
        _debugger = debugger

        @Debugger.method_decorator("work")
        def method(self, number=2):
            """A documented method."""
            if number == 0:
                raise RuntimeError("failure")
            self._debugger.log("inside")
            return number

    assert str(inspect.signature(Example.method)) == "(self, number=2)"
    assert Example.method.__doc__ == "A documented method."
    with pytest.raises(RuntimeError):
        Example().method(0)
    assert debugger._times == []
    assert debugger.get_last_log_categories() == []
    assert Example().method() == 2
    assert "inside" in capsys.readouterr().out

    @Debugger.static_method_decorator("static")
    def function(number=3, debugger=None):
        return number

    assert function() == 3


def test_histogram_trace_and_rgba(pixels):
    image = ImageLand(pixels).to_rgba()
    image.equalize_histogram(save_trace=True).retrace()
    assert image.mode == "RGBA"
    np.testing.assert_equal(image.get_image()[..., 3], 255)


def test_rgba_crop_inverse_preserves_existing_alpha():
    pixels = np.full((8, 12, 4), [90, 60, 30, 128], dtype=np.uint8)
    image = ImageLand(pixels).zoom(SquareMarks(2, 6, 3, 8, shape=(8, 12)), save_trace=True)
    image.retrace()
    np.testing.assert_equal(image.get_image(), pixels)


def test_failed_inverse_keeps_trace(pixels):
    image = ImageLand(pixels).zoom(SquareMarks(2, 6, 3, 8, shape=(12, 20)), save_trace=True)
    image.to_rgba()
    with pytest.raises(ValueError, match="channel mode"):
        image.retrace()
    image.to_rgb().retrace()
    np.testing.assert_equal(image.get_image(), pixels)


def test_debugger_visual_types_and_disabled_warning(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    debugger = Debugger(visual_debug=True)
    points = [np.array([[1.0, 2.0], [3.0, 4.0]]), (8, 10)]
    fig = debugger.log_images(
        [np.zeros((8, 10, 3)), np.zeros((8, 10)), points, points],
        ["RGB", "Heat", "Marks", "Vector"],
        [0, 1, 2, 3],
        dim=(2, 2),
    )
    assert fig is not None
    plt.close(fig)
    debugger.save_image(points, "Snapshot", 2)
    debugger.log_saved_images()
    assert not debugger._images
    with pytest.raises(ValueError):
        debugger.log_images([np.zeros((2, 2))], [], [])
    with pytest.raises(RuntimeError):
        debugger.log_end()
    with pytest.raises(ValueError):
        Debugger(debug=False).log_exception("still raises")
    with pytest.warns(UserWarning):
        debugger.log_warning("visible warning")
    plt.close("all")
