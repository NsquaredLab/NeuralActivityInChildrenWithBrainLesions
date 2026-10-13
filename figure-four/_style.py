"""
_style.py
=========

Figure style of Figure 4 and its supplementary figures: Nature sizes (183 mm
double column), fonts and palette.

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4
FULL_W = 183 * MM      # Nature double column

# Font sizes (pt) of the published figure
FS = dict(panel=10, title=8, label=8, tick=6, legend=6, annot=6)

# Palette
DARK_BLUE = "#001C55"
BLUE = "#0043CE"
LIGHT_BLUE = "#78A9FF"
PURPLE = "#A46DFD"
PINK = "#FF7EB6"
ORANGE = "#F9873C"
GREY = "#8C8C8C"
LIGHT_GREY = "#D9D9D9"
INK = "#1A1A1A"

GESTURES = ["Rest", "Power Grasp", "Pinch", "Tripod Pinch"]
SHORT = ["Rest", "Power", "Pinch", "Tripod"]
GESTURE_COLORS = {"Rest": LIGHT_BLUE, "Power Grasp": BLUE, "Pinch": PURPLE, "Tripod Pinch": ORANGE}
METRIC_COLORS = {"accuracy": DARK_BLUE, "f1_macro": PINK}

# Sequential map from the palette (monotonic lightness) for matrices
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#FFFFFF", LIGHT_BLUE, BLUE, DARK_BLUE], N=256)

# Wording of the label segments in the figure
WORD = dict(seg="detected attempt", Seg="Detected attempt")


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS["annot"], "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "lines.linewidth": 0.75, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "legend.frameon": False, "axes.labelpad": 2, "svg.hashsalt": "playagain",
    })
