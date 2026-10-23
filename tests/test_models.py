import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from landpy import ImageLand
from landpy.model import Model
from landpy.pretrained_models.face_tracker import Anchor
from landpy.tf_lite_model import TFLiteModel


class LocalModel(Model):
    def _load_model(self):
        return Path(self.model_path).read_bytes()

    def inference(self, input):
        return input


def test_import_does_not_require_tensorflow():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; import landpy; import landpy.pretrained_models; assert "tensorflow" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_local_model_checksum_and_validation(tmp_path):
    path = tmp_path / "model.tflite"
    path.write_bytes(b"test model")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    model = LocalModel(model_path=path, sha256=checksum)
    assert model._interpreter == b"test model"
    with pytest.raises(ValueError, match="checksum"):
        LocalModel(model_path=path, sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        LocalModel(model_path=tmp_path / "missing.h5")
    with pytest.raises(ValueError):
        LocalModel("http://example.org/weights.tflite")


def test_atomic_download_and_cache_identity(tmp_path, monkeypatch):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = [b"model data"]
    get = Mock(return_value=response)
    monkeypatch.setattr("landpy.model.requests.get", get)
    first = LocalModel("https://example.org/a/model.tflite", cache_dir=tmp_path)
    same = LocalModel("https://example.org/a/model.tflite", cache_dir=tmp_path)
    assert first.model_path == same.model_path
    assert get.call_count == 1
    other = LocalModel("https://example.org/b/model.tflite", cache_dir=tmp_path)
    assert first.model_path != other.model_path

    def broken(_):
        yield b"partial data"
        raise OSError("network interrupted")

    response.iter_content.side_effect = broken
    with pytest.raises(OSError):
        LocalModel("https://example.org/broken/model.tflite", cache_dir=tmp_path)
    assert not any(
        p.is_file()
        for p in tmp_path.rglob("*")
        if p.parent not in [Path(first.model_path).parent, Path(other.model_path).parent]
    )


def test_quantized_lite_io():
    model = object.__new__(TFLiteModel)
    from landpy import Debugger

    model._debugger = Debugger(debug=False)
    model.input_type = np.ndarray
    interpreter = Mock()
    interpreter.get_input_details.return_value = [
        {"shape": [1, 2], "dtype": np.uint8, "quantization": (0.5, 10), "index": 0}
    ]
    interpreter.get_output_details.return_value = [{"quantization": (0.25, 4), "index": 1}]
    interpreter.get_tensor.return_value = np.array([[4, 8]], dtype=np.uint8)
    model._interpreter = interpreter
    output = model.inference(np.array([[0.0, 1.0]]))
    np.testing.assert_equal(interpreter.set_tensor.call_args.args[1], [[10, 12]])
    np.testing.assert_equal(output, [[0, 1]])
    with pytest.raises(ValueError, match="shape"):
        model.inference(np.zeros((3, 4)))
    with pytest.raises(TypeError):
        model.inference([[0, 1]])


def test_original_anchor_geometry():
    anchors = Anchor("front").anchors
    assert anchors.shape == (896, 4)
    np.testing.assert_allclose(anchors[0], [0.03125, 0.03125, 1, 1])
    assert Anchor("back").anchors.shape == (896, 4)
    with pytest.raises(ValueError):
        Anchor("invalid")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("LANDPY_RUN_MODELS") != "1",
    reason="Set LANDPY_RUN_MODELS=1 to download and execute original models.",
)
def test_all_original_models_preserve_image_and_history():
    from landpy.pretrained_models import (
        ExpressionDetector,
        FaceLandmarksDetector,
        FaceSegmentationDetector,
        FaceTracker,
    )

    image = ImageLand(Path(__file__).parents[1] / "examples/assets/face.jpg")
    original = image.get_image().copy()
    image.make_checkpoint()
    tracker = FaceTracker()
    faces = tracker.detect_faces(image, threshold=0.5)
    assert faces and faces[0].p > 0.5
    assert faces[0].face_points.get_marks().shape == (6, 2)
    crop = image.copy().zoom(faces[0].face_box)
    landmarks, presence = FaceLandmarksDetector().detect_face_landmarks(crop)
    assert landmarks.get_marks().shape == (478, 2)
    assert landmarks.shape == crop.shape
    assert np.isfinite(presence)
    heat = FaceSegmentationDetector().detect_face_segmentation(crop)
    assert heat.shape == crop.shape and heat.get_heat().max() > 0.5
    scores = ExpressionDetector().detect_expression(crop)
    assert scores.shape == (1, 7) and np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(), 1, atol=1e-4)
    np.testing.assert_equal(image.get_image(), original)
    assert len(image._undo) == 1 and not image._redo


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("LANDPY_RUN_MODELS") != "1", reason="Real model downloads require opt-in."
)
def test_legacy_mesh_is_still_supported():
    from landpy.pretrained_models import FaceLandmarksDetector
    from landpy.pretrained_models.config import MODEL_SHA256, MODEL_URLS

    raw = TFLiteModel(
        MODEL_URLS["face_landmark_legacy"], sha256=MODEL_SHA256["face_landmark_legacy"]
    )
    detector = FaceLandmarksDetector(model_path=raw.model_path)
    image = ImageLand(Path(__file__).parents[1] / "examples/assets/face.jpg")
    marks, probability = detector.detect_face_landmarks(image)
    assert marks.get_marks().shape == (468, 2)
    assert 0 <= probability <= 1
