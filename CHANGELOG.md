# Changelog

## 0.1.0 — 2026-10-23

Complete the original image/landmark/heat-flow design and restore an installable
package after the 0.0.1.1 prototype.

- Consistent `(row, column)` pixel points and `(height, width)` spatial shapes.
- Independent copies, exact checkpoints, bounded undo/redo, reliable retracing.
- Convex perspective crops across images, heat maps and landmarks.
- Correct dtype/alpha handling, aligned overlays, functioning image implants.
- Float-preserving heat transformations and documented original mask algebra.
- Exception-safe categorized logging with preserved signatures and docstrings.
- Optional TensorFlow imports, legacy H5 support, quantized Lite input/output,
  atomic model downloads and pinned checksums.
- Compatible MediaPipe 478-point mesh update; original 468-point support retained.
- Eight executed tutorials, regression tests, CI and release automation.

### Migration from the unfinished source

Use imports such as `from landpy import ImageLand`; module paths are lowercase.
`shape` is now consistently a property, not a method. Old fractional-coordinate
examples must use `SquareMarks.relative` or explicitly scale points to pixels.
`zoom_factor` must be positive; the undocumented negative-zoom shorthand is not
supported. Binary heat subtraction retains the original minimum/intersection
meaning. Image reshape preserves its mode by default. Image floating-point
input is validated as [0, 1], not silently wrapped into bytes.

Call `make_checkpoint()` before changes. Undo/redo now restore independent exact
states and reject empty histories instead of indexing unintended entries.
`FaceLandmarksDetector` returns 478 points by default and a presence probability,
not a raw logit. The legacy mesh still returns 468 points when supplied locally.
Face detection uses the upstream [-1, 1] normalization, sigmoid scores and overlap
suppression. Model inference never mutates the input image or its history.
Old typo names `lenght`, `subscription`, `treshhold`, `FaceTackerIndices` and
`LIWER_LIPS` remain available; corrected aliases exist where appropriate.

## 0.0.1.1 — 2023-02-17

Initial PyPI prototype. The published artifact primarily exposed debugging code;
the larger image-processing design lived in the Git repository.
