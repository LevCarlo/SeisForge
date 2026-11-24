"""Shared figure style for docs/ant/scripts/*.py (not a figure itself).

Every figure script imports this module, puts its plotting in a function
``draw()`` that returns a figure, and calls :func:`save_themed`. That draws
the figure twice, once per palette in :data:`THEMES`, and writes

- ``docs/ant/img/<script name>.svg``       (light pages), and
- ``docs/ant/img/<script name>-dark.svg``  (dark pages),

both with a transparent background. The pages include them with
``![...](img/<name>.svg#only-light)`` and ``![...](img/<name>-dark.svg#only-dark)``.

Inside ``draw()`` colours are read as ``_style.MEASURED`` etc. at call time,
so they follow the palette being drawn. Colours on top of data (contours on
a colour map) keep fixed values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMG_DIR = Path(__file__).resolve().parents[1] / "img"

# One palette per page theme: truth in the ink colour, first measurement in
# blue, second (e.g. PMF) in orange, alternatives in grey, highlights in red.
# The dark values are lighter versions of the same hues (assumed values,
# chosen by eye against the site's dark background #14171d).
THEMES = {
    "light": {
        "INK": "#1c1f24", "MUTED": "#5b6470", "ALTERNATIVE": "#9aa3ad",
        "MEASURED": "#1f77b4", "SECOND": "#ff7f0e", "HIGHLIGHT": "#d62728",
        "EXTRA": "#2ca02c", "EXTRA2": "#9467bd", "BACKGROUND": "#ffffff", "GREY_CMAP": "Greys",
    },
    "dark": {
        "INK": "#e6e9ef", "MUTED": "#a7b0bb", "ALTERNATIVE": "#6b7480",
        "MEASURED": "#6fb1e6", "SECOND": "#f0a25e", "HIGHLIGHT": "#ef7a6c",
        "EXTRA": "#7ccf7a", "EXTRA2": "#c3a3e8", "BACKGROUND": "#14171d", "GREY_CMAP": "gray",
    },
}

# Current palette (set by save_themed while a figure is drawn).
INK = MUTED = ALTERNATIVE = MEASURED = SECOND = HIGHLIGHT = EXTRA = EXTRA2 = BACKGROUND = GREY_CMAP = ""
TRUTH = ""  # alias of INK


def use(theme: str) -> None:
    """Make ``THEMES[theme]`` the current palette."""
    globals().update(THEMES[theme])
    globals()["TRUTH"] = THEMES[theme]["INK"]


use("light")

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "image.cmap": "viridis",
        "legend.framealpha": 0.85,
        # Text as paths: the SVG does not depend on the reader's fonts.
        "svg.fonttype": "path",
    }
)


def save_themed(draw, script_file: str, out_dir: Path = IMG_DIR) -> list[Path]:
    """Draw ``draw()`` once per palette and save ``<stem>.svg`` / ``<stem>-dark.svg``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(script_file).stem
    paths = []
    for theme in THEMES:
        use(theme)
        rc = {
            "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
            "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": ALTERNATIVE,
            "legend.facecolor": BACKGROUND, "legend.edgecolor": ALTERNATIVE,
            "svg.hashsalt": stem,  # stable element ids: reruns give identical files
        }
        with plt.rc_context(rc):
            fig = draw()
            path = out_dir / f"{stem}{'' if theme == 'light' else '-' + theme}.svg"
            fig.savefig(path, bbox_inches="tight", transparent=True, metadata={"Date": None})
            plt.close(fig)
        print(f"wrote {path}")
        paths.append(path)
    use("light")
    return paths


def panel_label(ax, text: str) -> None:
    """Put a bold panel letter in the upper-left corner of ``ax``."""
    ax.text(
        0.01, 0.98, text, transform=ax.transAxes, ha="left", va="top",
        fontweight="bold", fontsize=10,
        bbox={"facecolor": BACKGROUND, "edgecolor": "none", "alpha": 0.7, "pad": 1},
    )
