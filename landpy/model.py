"""Lazy model loading with local paths and atomic cached downloads."""

import hashlib
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import requests

from .debugger import Debugger


class Model(ABC):
    """Load a local or HTTPS .h5/.keras/.tflite model on explicit construction.

    Downloads are cached beneath ``LANDPY_MODEL_DIR`` or ~/.cache/landpy/models.
    Pass ``model_path`` to work offline; ``sha256`` verifies model provenance.
    TensorFlow is optional and imported by concrete loaders only.
    """

    def __init__(
        self,
        model_url=None,
        input_type=None,
        output_type=None,
        debugger=None,
        *,
        model_path=None,
        cache_dir=None,
        sha256=None,
    ):
        self._debugger = debugger if debugger is not None else Debugger(debug=False)
        self.input_type, self.output_type = input_type, output_type
        self.model_url = model_url
        if model_path is not None:
            path = Path(model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(path)
        else:
            if not model_url or urlparse(model_url).scheme != "https":
                raise ValueError("Provide an HTTPS model_url or an existing model_path.")
            name = Path(urlparse(model_url).path).name
            identity = hashlib.sha256(model_url.encode()).hexdigest()[:16]
            directory = Path(
                cache_dir or os.environ.get("LANDPY_MODEL_DIR", "~/.cache/landpy/models")
            ).expanduser()
            path = directory / identity / name
        self.model_path = str(path)
        self.model_name, self.model_format = path.name, path.suffix.lower().lstrip(".")
        if self.model_format not in ("h5", "keras", "tflite"):
            raise ValueError("Model format must be h5, keras or tflite.")
        if not path.exists():
            self._download_model()
        if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256.lower():
            raise ValueError(f"Model checksum mismatch: {path}")
        self._interpreter = self._load_model()

    def _download_model(self):
        path = Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with requests.get(self.model_url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                    temporary = Path(output.name)
                    for chunk in response.iter_content(8192):
                        output.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError("Downloaded model is empty.")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _validate_input(self, data):
        if self.input_type is not None and not isinstance(data, self.input_type):
            raise TypeError(f"Input must be {self.input_type}.")

    @abstractmethod
    def _load_model(self):
        """Return the backend model or interpreter."""

    @abstractmethod
    def inference(self, input):
        """Run inference on one model input."""
