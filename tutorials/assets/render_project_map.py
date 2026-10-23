import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
fig.patch.set_facecolor("#fffdf6")
ax.set_facecolor("#fffdf6")
ax.set(xlim=(0, 13), ylim=(0, 7))
ax.axis("off")
ax.text(
    0.3,
    6.65,
    "Landpy: build an AR effect, then retrace it home",
    fontsize=21,
    weight="bold",
    color="#102c43",
)


def box(x, y, w, h, text, color="#ddf1ed", size=12):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.12", facecolor=color, edgecolor="#70958f")
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="#102c43",
        linespacing=1.6,
    )


def arrow(a, b, label=None):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 2, "color": "#236d74"})
    if label:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + 0.15,
            label,
            ha="center",
            fontsize=10,
            color="#236d74",
        )


box(0.3, 3.4, 2.2, 1.25, "ImageLand\nphoto / frame")
box(3.5, 3.4, 2.3, 1.25, "MarksLand + regions\nchoose / crop")
box(6.7, 3.4, 2.3, 1.25, "HeatLand + Filter\nmask / blend")
box(10, 3.4, 2.5, 1.25, "Edited ImageLand\nAR effect on crop")
arrow((2.65, 4), (3.3, 4))
arrow((5.95, 4), (6.5, 4))
arrow((9.15, 4), (9.8, 4))
box(3.5, 5.25, 5.5, 0.65, "Optional models → face boxes, landmarks, masks", size=11)
arrow((4.6, 5.1), (4.6, 4.8))
arrow((7.8, 5.1), (7.8, 4.8))
ax.plot([11.25, 11.25, 1.4], [3.2, 2.55, 2.55], color="#d45b66", lw=2)
ax.annotate(
    "",
    xy=(1.4, 3.2),
    xytext=(1.4, 2.55),
    arrowprops={"arrowstyle": "->", "lw": 2, "color": "#d45b66"},
)
ax.text(
    6.35,
    2.15,
    "retrace(): return the edited crop to its saved canvas",
    ha="center",
    fontsize=14,
    color="#b13a50",
    weight="bold",
)
box(
    0.3,
    0.45,
    5.9,
    1.05,
    "Land: shared transformations and history\nrecord inverses • checkpoint → undo / redo",
    color="#e9edf6",
    size=12,
)
box(
    6.6,
    0.45,
    5.9,
    1.05,
    "Debugger: inspect selected steps\ncategorized timing logs • visual snapshots",
    color="#fce6d9",
    size=12,
)
ax.text(
    0.3,
    1.8,
    "Regions: SquareMarks (rectangle) · QuadMarks (perspective) · DualMarks (vector)",
    fontsize=11,
    color="#34495e",
)
fig.savefig("tutorials/assets/project-map.png", dpi=160, facecolor=fig.get_facecolor())
