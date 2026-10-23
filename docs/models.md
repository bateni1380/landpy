# Model contracts and update notes

Landpy pins model revision
[`83c1ad107a7ab3913f98ed116729f3ace4061f07`](https://github.com/bateni1380/pretrained_models/tree/83c1ad107a7ab3913f98ed116729f3ace4061f07).
The companion `models.json` contains sizes, SHA-256 hashes, tensor shapes,
preprocessing ranges and provenance. Built-in constructors check those hashes.
TensorFlow is optional; install `landpy[tensorflow]` on Python 3.10–3.13.

| API | Input preparation | Output |
| --- | --- | --- |
| `FaceTracker.detect_faces(image, threshold=.5, top_k=None)` | Aspect-preserving square pad, RGB 128², float32 [-1,1] | Confidence-sorted FacePoints after IoU suppression |
| `FaceLandmarksDetector.detect_face_landmarks(face_crop)` | Square pad, RGB 256², float32 [0,1] | 478 pixel XY points as MarksLand and sigmoid presence probability |
| Legacy mesh via `model_path` | Reads 192² input from tensor metadata | 468 points and sigmoid presence probability |
| `FaceSegmentationDetector.detect_face_segmentation(face_crop)` | RGB 128², float32 [0,1] | HeatLand resized to the supplied crop |
| `ExpressionDetector.detect_expression(face_crop)` | 96² luminance repeated into RGB, float32 [0,255] | (1,7) class scores in output_labels order |

Every method leaves the input image, trace and checkpoints unchanged. Face
landmarks/segmentation/expression expect an already cropped face. `FacePoints`
contains `face_box`, `face_points` and `p`; indices 0–3 address box corners and
4–9 address its six keypoints. Bounds and points are not clamped to the image;
crops support padding outside it. FaceTracker defaults to one result if neither
threshold nor top_k is provided. Detection is independent per frame; it does not
implement temporal tracking or the full MediaPipe task graph.

## Compatible update

The [official Face Landmarker bundle](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
contains a newer 478-point mesh. The companion repository extracts the unmodified
mesh asset from Google's versioned `float16/1` bundle, rather than linking to
`latest`. The loader reads tensor dimensions and accepts both 468 and 478 points.
The original face-region indices remain usable; the ten extra points are irises.
No 52-class blendshape API has been introduced: those coefficients are a different
contract from the original seven expression classes.

The [upstream detector graph](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection.pbtxt)
specifies [-1,1] normalization. The
[landmark graph](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc)
applies sigmoid to presence logits. Landpy now follows those conventions.
Validation covers execution, shapes, finite outputs, source-image preservation,
and the crop/filter/retrace flow. It is not an accuracy benchmark or a claim
that a newer asset performs better for every image.

## Downloads, offline use and backend limits

Defaults download only on construction. The cache is URL-specific to avoid
same-filename collisions. Interrupted downloads are removed; complete files are
verified before backend loading. `LANDPY_MODEL_DIR` overrides the cache root.
Pass `model_path='/path/to/model.tflite'` to avoid networking, and optionally a
`sha256` when supplying a custom local file. Only load trusted model files.

`TFLiteModel.inference` handles exactly one fixed-shape input. Quantized input
metadata scales floating input; integer arrays are passed through as quantized
values. Outputs with quantization metadata are dequantized. A single tensor
returns an ndarray; multiple tensors return a list. Model instances are mutable
backend objects and are not intended for simultaneous threaded inference.
`TFModel` loads `.keras` with Keras and `.h5` with the legacy `tf-keras` loader.

The supported TensorFlow range is 2.16 through 2.21. The existing Lite interpreter
emits an upstream deprecation warning on newer versions in that range. Migrating
to a different runtime should be treated as a separately tested dependency
update; this release keeps a backend verified on both the old and refreshed files.

## Provenance limitations

The original segmentation and expression files lacked upstream sources, dataset
information, evaluation results and licenses. Their defaults are retained,
checksum-pinned, and documented as legacy experimental assets. They are not
included in the Python wheel or covered by the code's MIT license. Do not infer
emotions or intentions from the expression class names. The companion catalog
records the newer mesh's exact upstream download and bundle hash.
