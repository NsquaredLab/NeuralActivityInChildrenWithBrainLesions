"""
_style.py
=========

Figure style of Figure 2: colours, colormaps, one colour per motor unit, the
journal typography and the SVG export.

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# ---- Colours -----------------------------------------------------------------
DARK_BLUE = "#001C55"   # RMS envelope, mean MUAP
BLUE = "#4371CB"        # channel colormap start
ROSA = "#FF8088"        # channel colormap middle
HELLBEIGE = "#FCF9BB"   # channel colormap end

# Colormap of columns without a colour of their own (blue -> rosa -> light beige).
CHANNEL_CMAP = LinearSegmentedColormap.from_list(
    "blau_rosa_hellbeige", [BLUE, ROSA, HELLBEIGE])

# Heatmap colormap when no unit colour is given (white -> blue -> dark blue).
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "weiss_blau_dunkelblau", ["#FFFFFF", BLUE, DARK_BLUE])

# ---- One colour per motor unit -------------------------------------------------
# One sweep from blue over purple to pink. Units 2, 4, 5 and 8 are fixed; the
# others are filled in between their neighbours, unit 1 is a cyan blue ahead of
# 2. DARK_BLUE is left out: it is the RMS envelope's colour.
MU_COLORS = (
    "#1192E8",   # MU 1  cyan blue
    "#0043CE",   # MU 2  blue          (fixed)
    "#3C76E6",   # MU 3  between 2 and 4
    "#78A9FF",   # MU 4  light blue    (fixed)
    "#A46DFD",   # MU 5  purple        (fixed)
    "#C273E5",   # MU 6  1/3 from 5 to 8
    "#E079CE",   # MU 7  2/3 from 5 to 8
    "#FF7EB6",   # MU 8  pink          (fixed)
)


def mu_colors(n_mu):
    """
    One colour per motor unit of a tracking file, in ``MUPulses`` order; more
    units than colours wrap around. The tracking panels and the P2P heatmaps
    both take their unit colours from here, so a unit has one colour in every
    figure.
    """
    return [MU_COLORS[k % len(MU_COLORS)] for k in range(n_mu)]


def mu_heatmap_cmap(color):
    """
    Heatmap colormap of one motor unit: white -> ``color`` -> ``color`` at
    40 % brightness.
    """
    rgb = np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list(
        "mu_heatmap", ["#FFFFFF", tuple(rgb), tuple(rgb * 0.4)])


# ---- Typography and export ---------------------------------------------------
def use_journal_style(base_size=20):
    """Apply the journal typography (Arial + font sizes); pair with :func:`reset_style`."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = base_size
    plt.rcParams["axes.titlesize"] = base_size
    plt.rcParams["axes.labelsize"] = base_size
    plt.rcParams["xtick.labelsize"] = base_size + 2
    plt.rcParams["ytick.labelsize"] = base_size
    plt.rcParams["legend.fontsize"] = base_size + 2


def reset_style():
    """Restore matplotlib's default rcParams (undo :func:`use_journal_style`)."""
    plt.rcParams.update(plt.rcParamsDefault)


def save_svg(fig, savepath, dpi=300):
    """Save ``fig`` as an SVG if ``savepath`` is given (no-op otherwise)."""
    if savepath is None:
        return
    fig.savefig(savepath, format="svg", dpi=dpi,
                metadata={"Date": None})   # no creation date in the file
