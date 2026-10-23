# Landpy tutorials

## Build an AR effect, then retrace it home 🪄

Start with **[notebook 1](1-images-and-coordinates.ipynb)**: see the face effect
we are working toward, understand the project map, and make your first edit.
The core idea is simple: crop a region, edit it, then `retrace()` to put the
edited pixels back in the original image. Each lesson adds one useful piece.

| Lesson | The question it answers |
| --- | --- |
| 1 · First edit | What can I build, and how does retrace help? |
| 2 · Placement | Where should an effect go? |
| 3 · Masks | Which pixels should change, and by how much? |
| 4 · History | How is retrace different from undo, and what happened inside? |
| 5 · Perspective | How does an edited crop find its way back to a tilted surface? |
| 6 · Overlays | How do I place a transparent AR sticker? |
| 7 · Models | How do predictions enter the workflow? (Optional technical detour.) |
| 8 · Face effect | How do all the pieces produce the previewed effect? |

Each notebook stands alone with setup, saved visuals, short checks and a small
experiment to try. Debugger snapshots expose useful intermediate steps without
burying the lesson in plotting code. Tutorials 1–6 use generated canvases and
need no external inputs; notebook 1 includes a saved face-effect preview.
Tutorials 7–8 require the TensorFlow extra and Python 3.10–3.13.

| Notebook | Run in Colab |
| --- | --- |
| [1 images and coordinates](1-images-and-coordinates.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/1-images-and-coordinates.ipynb) |
| [2 landmarks and regions](2-landmarks-and-regions.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/2-landmarks-and-regions.ipynb) |
| [3 heat maps and filters](3-heat-maps-and-filters.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/3-heat-maps-and-filters.ipynb) |
| [4 history and debugging](4-history-and-debugging.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/4-history-and-debugging.ipynb) |
| [5 crops and perspective](5-crops-and-perspective.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/5-crops-and-perspective.ipynb) |
| [6 image compositing](6-image-compositing.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/6-image-compositing.ipynb) |
| [7 model wrappers](7-model-wrappers.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/7-model-wrappers.ipynb) |
| [8 face models and effects](8_face_models_and_effects.ipynb) | [Open in Colab](https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/8_face_models_and_effects.ipynb) |

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
