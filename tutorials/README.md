# Landpy tutorials

Run these notebooks in order. Each stands alone and includes setup, explanations,
rendered results and assertions. Tutorials 1–6 use generated images and need no
external files; tutorials 7–8 need the TensorFlow extra and Python 3.10–3.13.

| Notebook | Run in Colab |
| --- | --- |
| [1 images and coordinates](1-images-and-coordinates.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/1-images-and-coordinates.ipynb) |
| [2 landmarks and regions](2-landmarks-and-regions.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/2-landmarks-and-regions.ipynb) |
| [3 heat maps and filters](3-heat-maps-and-filters.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/3-heat-maps-and-filters.ipynb) |
| [4 history and debugging](4-history-and-debugging.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/4-history-and-debugging.ipynb) |
| [5 crops and perspective](5-crops-and-perspective.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/5-crops-and-perspective.ipynb) |
| [6 image compositing](6-image-compositing.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/6-image-compositing.ipynb) |
| [7 model wrappers](7-model-wrappers.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/7-model-wrappers.ipynb) |
| [8 face models and effects](8-face-models-and-effects.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/8-face-models-and-effects.ipynb) |

In Colab, select a CPU Python runtime and run all cells. The setup installs the
pinned `landpy==0.1.0` release if needed. A first model run downloads about 37 MB
of weights; later runs reuse the cache. Colab can require Google sign-in to run.

Locally, install the package and notebook tools from the repository root:

```bash
python -m pip install -e '.[dev,tensorflow]'
python scripts/execute_notebooks.py
```

Or open individual files in Jupyter. The execution script starts each notebook
with a fresh kernel, stops on errors and saves outputs only for successful runs.
To use a custom environment, pass `--kernel your-kernel-name`.
