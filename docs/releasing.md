# Release procedure

1. Keep `landpy.__version__`, `pyproject.toml`, notebook setup pins and changelog
   synchronized. Edit notebooks with nbformat or Jupyter, preserving their
   teaching flow and rerunning affected notebooks after changes.
2. Run tests, the real-model integration and all tutorials. Inspect saved figures.
3. Run `python -m build` and `python -m twine check dist/*`. Install the wheel in
   a fresh environment from outside the source tree and test its public imports.
4. Confirm that the model revision and checksum catalog are available publicly.
5. Publish the source and create a GitHub release tagged `vX.Y.Z` on that commit.
   The `publish.yml` workflow builds and uploads through PyPI Trusted Publishing.
6. Verify PyPI's version metadata, wheel installation, and public Colab links.

The PyPI project must authorize this trusted publisher:

- Owner: `bateni1380`
- Repository: `landpy`
- Workflow: `publish.yml`
- Environment: `pypi`

PyPI authorization is a security-sensitive account operation performed in the
owner's signed-in browser. It replaces the need to copy a long-lived upload token
into this repository. Until that relationship is configured, the publish job
cannot upload. The workflow can be dispatched after fixing account setup.

If an artifact has already been published, do not overwrite it or silently retry
the same version with different contents. Bump the version and repeat validation.
