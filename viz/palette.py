"""Shared colors + matplotlib styling for all figures (light surface only; these are static PNG/GIF outputs).

Categorical slots are assigned in fixed order and never cycled: an arm keeps its color whatever else is on the chart.
Sequential = one hue (blue), light -> dark. Text always wears ink tokens, never a series color.
"""
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW = CATEGORICAL[:4]

# blue ramp, steps 100 -> 700
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
PROB_CMAP = LinearSegmentedColormap.from_list("fsf_prob", [SURFACE] + SEQ_BLUE)

# actual-burn panel: 0 = unburned (surface), 1 = burning yesterday and today (persisted), 2 = newly burning today
BURN_COLORS = [SURFACE, AXIS, ORANGE]
BURN_CMAP = ListedColormap(BURN_COLORS, name="fsf_burn")
BURN_LABELS = ["unburned", "burning on both days", "new fire today"]


def style():
    """Quiet chrome: hairline solid grid, no top/right spines, system sans, ink-token text."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
        "lines.linewidth": 2, "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round",
    })
