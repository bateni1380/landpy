# Contributing

Landpy completes a small existing design. Prefer focused bug fixes, clearer
examples and compatible dependency updates over adding unrelated features.

Install `.[dev,tensorflow]` in a virtual environment. Run `python -m pytest`
for the offline suite and `LANDPY_RUN_MODELS=1 python -m pytest -m integration`
for real model inference. The latter downloads pinned weights on first use.
Run the eight notebooks from fresh kernels with `python scripts/execute_notebooks.py`.
Keep executed outputs useful and portable; don't save secrets or local paths.

Check `python -m ruff check landpy tests scripts` and
`python -m ruff format --check landpy tests scripts`. Public operations need
clear docstrings and meaningful regression tests for their behavior. Keep
comments for decisions that the code itself cannot explain.

Model changes must update the companion catalog, pin a revision and hashes,
verify tensor semantics, and rerun the face-effects tutorial. Document source
and license uncertainty instead of assigning an invented license to old weights.
