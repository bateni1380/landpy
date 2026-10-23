"""Execute and save every numbered tutorial using the selected Jupyter kernel."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kernel", default="python3")
parser.add_argument("--start", type=int, default=1)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for path in sorted((root / "tutorials").glob("[1-9]-*.ipynb")):
    if int(path.name.split("-")[0]) < args.start:
        continue
    print("Executing", path.name, flush=True)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(
        notebook, timeout=300, kernel_name=args.kernel, resources={"metadata": {"path": str(root)}}
    ).execute()
    nbformat.write(notebook, path)
    print("Saved", path.name, flush=True)
