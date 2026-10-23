# Landpy

Build small augmented-reality effects in Python: **crop → edit → retrace**.

![Landpy AR illustration: crop a face region, edit it, and return it](https://raw.githubusercontent.com/bateni1380/landpy/master/docs/assets/landpy-ar-cover.png)

Landpy helps you place face filters and transparent overlays on images. Work on
a crop, use landmarks or masks to control the effect, then call `retrace()` to
bring the edited pixels back onto the original canvas. The crop remembers where
it came from, so you do not have to calculate the paste coordinates yourself.
Use checkpoints and `undo()` when you want to restore the exact earlier state.

**[Start the illustrated tutorial journey](https://github.com/bateni1380/landpy/blob/master/tutorials/1-images-and-coordinates.ipynb)**
for a preview of the face effect, a project map, and your first runnable edit.
The cover above is an AI-generated illustration; the notebooks show executed
results. These examples operate on individual images; live tracking and temporal
smoothing are outside the package's scope.

```bash
pip install landpy
# Optional face-model support (Python 3.10–3.13):
pip install 'landpy[tensorflow]'
```

Python 3.10+ is required for the core package. No weights are downloaded and
TensorFlow is not imported when you import `landpy`. Use a CPU runtime; a GPU
is not required. OpenCV is installed as the headless build for notebooks and
servers; don't install both OpenCV wheel variants in the same environment.

## A first edit

```python
import numpy as np
from landpy import ImageLand, SquareMarks, Filter

image = ImageLand(np.full((120, 180, 3), 180, dtype=np.uint8))
region = SquareMarks.relative(.2, .8, .25, .75, shape=image.shape)
image.make_checkpoint()
image.zoom(region, save_trace=True)
image.implant_filter(Filter('#d52e65', ratio=.5))
image.retrace()  # put the edited crop back into the original image
image.show()
image.undo()     # restore the exact original snapshot
```

Shapes are `(height, width)` and points are pixel `(row, column)` coordinates.
RGB arrays use [0, 255] for integers or [0, 1] for floats. Methods usually mutate
and return the same container; `copy()` and `+` produce independent objects.
`get_image()`, `get_marks()` and `get_heat()` expose mutable arrays.

## Learn step by step

The [eight numbered tutorials](https://github.com/bateni1380/landpy/blob/master/tutorials/README.md) contain saved outputs and
checks. Each has an **Open in Colab** link; no separate notebook hosting account
is needed to open them. Start with
[1. Images and coordinates](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/1-images-and-coordinates.ipynb).

| Class | Purpose |
| --- | --- |
| `Land` | Inverse stack and independent checkpoint history |
| `ImageLand` | RGB/RGBA images, crops, rotations, resizing and compositing |
| `MarksLand` | Pixel landmarks, set operations, polygon masks |
| `SquareMarks` | Axis-aligned rectangles, despite the historical name |
| `QuadMarks` | Convex quadrilaterals and perspective regions |
| `DualMarks` | Directed two-point vectors |
| `HeatLand` | Floating-point masks and heat algebra |
| `Filter` | Mask-weighted color blending or a custom function |
| `Debugger` | Categorized timing logs and visual snapshots |

Use Python's `help(ImageLand.zoom)` or the [API guide](https://github.com/bateni1380/landpy/blob/master/docs/api.md) for details.

## History has two meanings

`save_trace=True` records a transform's inverse. `retrace()` applies that
inverse to the **current** data, which lets an edited crop return to its old
position. Image rotation/resizing inverses are approximate: resampling cannot
recover discarded pixels. Crop inverses preserve the surrounding saved canvas.

`make_checkpoint()` saves an exact copy **before** an edit. `undo()` restores
that snapshot; `redo()` restores the edited state. New operations invalidate
redo. Empty histories raise `IndexError`. Array writes through mutable getters
bypass operation tracking; checkpoint before them and don't rely on automatic
redo invalidation for direct writes.

## Face models

```python
from landpy import ImageLand
from landpy.pretrained_models import FaceTracker, FaceLandmarksDetector

photo = ImageLand('portrait.jpg')
faces = FaceTracker().detect_faces(photo, threshold=.5, top_k=1)
if faces:
    crop = photo.copy().zoom(faces[0].face_box.copy().expand(1.25))
    marks, presence = FaceLandmarksDetector().detect_face_landmarks(crop)
    (crop + marks).show()
```

The default mesh is the compatible 478-point MediaPipe Face Mesh V2 asset;
legacy 468-point weights remain supported with `model_path=...`.
Model files are pinned to a [verified companion repository revision](https://github.com/bateni1380/pretrained_models/tree/83c1ad107a7ab3913f98ed116729f3ace4061f07).
Downloads go to `~/.cache/landpy/models`; set `LANDPY_MODEL_DIR` to change it.
Each detector accepts `model_path`, `cache_dir` and `sha256` keyword arguments.
See [model contracts and provenance](https://github.com/bateni1380/landpy/blob/master/docs/models.md), especially the incomplete
provenance of the original segmentation and expression weights.

`FaceTracker` detects still images independently; it does not perform temporal
tracking. This is an experimental face-effects toolkit, not an accuracy or
latency claim. The core package contains no bundled weights or photos.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,tensorflow]'
python -m pytest
LANDPY_RUN_MODELS=1 python -m pytest -m integration
python scripts/execute_notebooks.py
python -m ruff check landpy tests scripts
python -m build
python -m twine check dist/*
```

[Contributing](https://github.com/bateni1380/landpy/blob/master/CONTRIBUTING.md) · [Migration and release notes](https://github.com/bateni1380/landpy/blob/master/CHANGELOG.md) ·
[Release procedure](https://github.com/bateni1380/landpy/blob/master/docs/releasing.md) · [Original design audit](https://github.com/bateni1380/landpy/blob/master/docs/design.md)

Code is MIT licensed. Third-party weights and the original example photo have
separate provenance; see the model and asset documentation.
