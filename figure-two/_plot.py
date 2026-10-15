"""
_plot.py
========

Drawing of the Figure 2 panels. Every function receives ready-made data and
only draws; the data are prepared in ``_tracking.py``.

    - ``plot_mu_tracking_rates``     one column per gesture: RMS envelope on top,
                                     per motor unit its discharge rate with its
                                     pulses underneath, cue periods shaded gray
    - ``plot_mu_heatmap_tracking``   P2P map of one motor unit, one panel per
                                     gesture, one colour scale
    - ``plot_mean_muaps``            mean MUAP of every unit, overall and per
                                     gesture, on one y-axis

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator

import _common as C
from _style import (CHANNEL_CMAP, DARK_BLUE, SCORE_CMAP, mu_colors,
                    mu_heatmap_cmap, reset_style, save_svg, use_journal_style)

log = C.get_logger(__name__)

# DIN A4 width in inches (210 mm) minus a margin.
A4_WIDTH_IN = 8.27 * 0.9

# Width of the gap between two gestures, as a fraction of a column's width.
COLUMN_GAP = 0.35

# Heights in inches of the RMS row, of one unit's rate strip, of its pulse
# strip below it, and of the gap between two units. One unit (rate + pulses)
# is RATES_UNIT_FRACTION of the RMS row, split between rate and pulses
# 0.55 : 0.2.
RATES_RMS_HEIGHT = 1.6
RATES_UNIT_FRACTION = 0.25
RATES_RATE_HEIGHT = RATES_UNIT_FRACTION * RATES_RMS_HEIGHT * 0.55 / 0.75
RATES_PULSE_HEIGHT = RATES_UNIT_FRACTION * RATES_RMS_HEIGHT * 0.2 / 0.75
RATES_UNIT_GAP = 0.12

# Cue shading.
CUE_SHADE = (0.8, 0.8, 0.8)
CUE_ALPHA = 0.45

# Longest rate trace drawn, in points (keeps the SVG small).
RATES_MAX_POINTS = 4000

# Heatmaps: colour of the gap columns between the grids and of the channels
# that could not be filled.
HEATMAP_GAP_COLOR = "white"


def _shade_cues(ax, holds):
    """The gray cue spans, behind everything else in the axes."""
    for start, end in holds or ():
        ax.axvspan(start, end, facecolor=CUE_SHADE, alpha=CUE_ALPHA,
                   edgecolor="none", zorder=0)


def plot_mu_tracking_rates(columns, mu_labels=None, savepath=None,
                           rate_unit="pps", bare=False):
    """
    Tracking panel with discharge rates: one column per gesture, the RMS
    envelope on top and, per motor unit, its discharge rate with its pulses
    directly underneath.

    The unit strips are in file order and the same in every column, so reading
    along a strip is reading one motor unit through the recording day. All
    rate strips share one y-scale.

    Parameters
    ----------
    columns : sequence of dict
        One entry per gesture, drawn left to right. Keys: ``label``, ``rms``,
        ``fsamp``, ``emg_mean`` (optional), ``spikes`` (per unit, discharge
        times in s from the start of the gesture), ``color``, ``rates`` (per
        unit, discharge rate in pps on the EMG's time base) and ``holds``
        (cue periods in s, shaded gray in every row).
    mu_labels : sequence of str or None, optional
        One label per unit, e.g. ``["MU 1 (28.4dB)", ...]``.
    savepath : str or None, optional
        If given, saved there as SVG.
    rate_unit : str, optional
        Unit on the rate axis (default ``"pps"``).
    bare : bool, optional
        Drop every text but keep the tick marks and the layout (default False).

    Returns
    -------
    matplotlib.figure.Figure
    """
    columns = list(columns)
    if not columns:
        raise ValueError("columns is empty -- nothing to draw.")

    use_journal_style(base_size=9)

    envelopes = [np.asarray(column["rms"], dtype=float).ravel()
                 for column in columns]
    durations = [rms.size / float(column["fsamp"])
                 for rms, column in zip(envelopes, columns)]

    y_max = max((float(np.max(rms)) for rms in envelopes if rms.size),
                default=1.0)
    has_emg = any(column.get("emg_mean") is not None for column in columns)
    y_min = -0.4 * y_max if has_emg else 0.0

    # The mean EMG behind the envelope has its own axis on the right, in µV
    # and one scale for every column. Its limits are the RMS limits stretched
    # by emg_peak / (0.5 * y_max): the two zeros coincide and the largest
    # deflection reaches half the RMS peak.
    emg_peak = 0.0
    for column in columns:
        emg = column.get("emg_mean")
        if emg is not None and np.size(emg):
            emg_peak = max(emg_peak, float(np.max(np.abs(emg))))
    has_emg = has_emg and emg_peak > 0
    emg_stretch = emg_peak / (0.5 * y_max) if has_emg else 1.0
    emg_ticks = (MaxNLocator(nbins=4, symmetric=True).tick_values(
        -emg_peak, emg_peak) if has_emg else [])
    emg_ticks = [t for t in emg_ticks
                 if abs(t) <= emg_peak and t >= y_min * emg_stretch]
    # Only the part of the left axis that the RMS can take.
    rms_ticks = [t for t in MaxNLocator(nbins=5).tick_values(0, y_max * 1.1)
                 if 0 <= t <= y_max * 1.1]

    n_mu = max((len(column.get("rates") or ()) for column in columns),
               default=0)
    if mu_labels is None:
        mu_labels = [f"MU {k + 1}" for k in range(n_mu)]

    # One rate scale for every strip of every column.
    rate_max = 0.0
    for column in columns:
        for rate in column.get("rates") or ():
            rate = np.asarray(rate, dtype=float)
            if rate.size and np.isfinite(rate).any():
                rate_max = max(rate_max, float(np.nanmax(rate)))
    rate_top = rate_max * 1.1 if rate_max > 0 else 1.0

    auto = [i for i, column in enumerate(columns) if column.get("color") is None]
    palette = (CHANNEL_CMAP(np.linspace(0, 0.55, len(auto))) if len(auto) > 1
               else [DARK_BLUE] * len(auto))
    colors = [column.get("color") for column in columns]
    for position, i in enumerate(auto):
        colors[i] = palette[position]
    unit_colors = mu_colors(n_mu)

    # Grid rows: RMS, gap, then per unit rate / pulses / gap.
    heights = [RATES_RMS_HEIGHT, 2 * RATES_UNIT_GAP]
    for _ in range(n_mu):
        heights += [RATES_RATE_HEIGHT, RATES_PULSE_HEIGHT, RATES_UNIT_GAP]
    top_in, bottom_in = 0.35, 0.5
    fig_h = top_in + bottom_in + sum(heights)
    fig = plt.figure(figsize=(A4_WIDTH_IN, fig_h))
    grid = fig.add_gridspec(len(heights), len(columns), height_ratios=heights,
                            hspace=0, wspace=COLUMN_GAP, left=0.17,
                            right=0.92 if has_emg else 0.98,
                            top=1 - top_in / fig_h, bottom=bottom_in / fig_h)

    tick_step = 10 if rate_top > 25 else 5
    rate_ticks = np.arange(0, rate_top, tick_step)

    for i, (column, rms, color) in enumerate(zip(columns, envelopes, colors)):
        fsamp = float(column["fsamp"])
        x_right = durations[i]
        holds = column.get("holds")

        # ---- The RMS row ---------------------------------------------------
        ax_rms = fig.add_subplot(grid[0, i])
        ax_emg = ax_rms.twinx() if has_emg else None
        if ax_emg is not None:
            ax_rms.set_zorder(ax_emg.get_zorder() + 1)
            ax_rms.patch.set_visible(False)
        _shade_cues(ax_emg if ax_emg is not None else ax_rms, holds)
        emg_mean = column.get("emg_mean")
        if ax_emg is not None and emg_mean is not None:
            emg_mean = np.asarray(emg_mean, dtype=float).ravel()
            lighter = tuple(min(1.0, c + 0.3) for c in to_rgba(color)[:3])
            ax_emg.plot(np.arange(emg_mean.size) / fsamp, emg_mean,
                        color=lighter, alpha=0.3, linewidth=0.5)
        if ax_emg is not None:
            ax_emg.set_ylim(y_min * emg_stretch, y_max * 1.1 * emg_stretch)
            ax_emg.set_yticks(emg_ticks)
            ax_emg.spines["right"].set_bounds(
                max(-emg_peak, y_min * emg_stretch), emg_peak)
            for spine in ("top", "left", "bottom"):
                ax_emg.spines[spine].set_visible(False)
            if i < len(columns) - 1:
                ax_emg.spines["right"].set_visible(False)
                ax_emg.tick_params(axis="y", labelright=False, length=0)
            else:
                ax_emg.set_ylabel("Mean EMG (µV)")
        ax_rms.plot(np.arange(rms.size) / fsamp, rms, color=color,
                    linewidth=1.5)
        ax_rms.set_title(column["label"], color=color)
        ax_rms.set_ylim(y_min, y_max * 1.1)
        ax_rms.set_yticks(rms_ticks)
        if has_emg:
            ax_rms.spines["left"].set_bounds(0, y_max * 1.1)
        ax_rms.set_xlim(0, x_right)
        ax_rms.tick_params(axis="x", labelbottom=False, length=0)
        for spine in ("top", "right", "bottom"):
            ax_rms.spines[spine].set_visible(False)
        if i:
            ax_rms.tick_params(axis="y", labelleft=False, length=0)
        else:
            ax_rms.set_ylabel("RMS (µV)")

        # ---- Per unit: rate, pulses underneath -----------------------------
        rates = column.get("rates") or ()
        spikes = column.get("spikes") or ()
        last_ax = ax_rms
        for k in range(n_mu):
            row = 2 + 3 * k
            ax_rate = fig.add_subplot(grid[row, i], sharex=ax_rms)
            ax_pulse = fig.add_subplot(grid[row + 1, i], sharex=ax_rms)
            train = (np.asarray(spikes[k], dtype=float).ravel()
                     if k < len(spikes) else np.empty(0))

            _shade_cues(ax_rate, holds)
            if k < len(rates) and train.size:
                rate = np.asarray(rates[k], dtype=float).ravel()
                step = max(1, int(np.ceil(rate.size / RATES_MAX_POINTS)))
                ax_rate.plot(np.arange(rate.size)[::step] / fsamp,
                             rate[::step], color=unit_colors[k], linewidth=1.3)
            else:
                ax_rate.text(0.5, 0.45, "silent", transform=ax_rate.transAxes,
                             ha="center", va="center", fontsize="small",
                             color=(0.6, 0.6, 0.6))
            ax_rate.set_ylim(0, rate_top)
            ax_rate.set_yticks(rate_ticks)
            ax_rate.tick_params(axis="x", labelbottom=False, length=0)
            for spine in ("top", "right", "bottom"):
                ax_rate.spines[spine].set_visible(False)
            if i:
                ax_rate.tick_params(axis="y", labelleft=False, length=0)
            else:
                ax_rate.tick_params(axis="y", labelsize=6.5, length=2)
                ax_rate.set_ylabel(mu_labels[k] if k < len(mu_labels)
                                   else f"MU {k + 1}",
                                   rotation=0, ha="right", va="center",
                                   fontsize=min(8.0, 110.0 / max(n_mu, 1)),
                                   labelpad=4)

            _shade_cues(ax_pulse, holds)
            if train.size:
                ax_pulse.eventplot([train], lineoffsets=0.5, linelengths=0.8,
                                   linewidths=0.8, colors=[unit_colors[k]])
            ax_pulse.set_ylim(0, 1)
            ax_pulse.set_yticks([])
            ax_pulse.tick_params(axis="x", labelbottom=False, length=0)
            for spine in ("top", "right", "bottom"):
                ax_pulse.spines[spine].set_visible(False)
            last_ax = ax_pulse

        last_ax.tick_params(axis="x", labelbottom=True, length=3)
        last_ax.spines["bottom"].set_visible(True)
        last_ax.set_xlabel("Time (s)")
        ax_rms.set_xlim(0, x_right)

    if bare:
        # Stripped after drawing, so the bare figure cannot drift away from
        # the labelled one.
        for ax in fig.axes:
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelbottom=False, labelleft=False,
                           labelright=False)
            for text in list(ax.texts):
                text.remove()
    elif n_mu:
        fig.text(0.005, (bottom_in + sum(heights[2:]) / 2) / fig_h,
                 f"Discharge rate ({rate_unit})", rotation=90,
                 ha="left", va="center")

    save_svg(fig, savepath)
    plt.show()
    reset_style()
    return fig


def plot_mu_heatmap_tracking(maps, names, valid, n_spikes, title,
                             min_spikes=None, vmin=None, vmax=None,
                             color=None, savepath=None):
    """
    P2P heatmap of one motor unit, one panel per gesture, on one colour scale.

    Parameters
    ----------
    maps : list of ndarray, shape (rows, cols)
        P2P map per gesture in µV (holes inside the grids already filled); NaN
        is drawn white. The entries of gestures not in ``valid`` are ignored.
    names : sequence of str
        Gesture names, one per entry of ``maps``.
    valid : sequence of int
        The gestures with enough discharges for a map; the others stay empty.
    n_spikes : sequence of int
        Discharges per gesture, for the panel titles.
    title : str
        Figure title.
    min_spikes : int or None, optional
        Only for the title of an empty panel.
    vmin, vmax : float or None, optional
        Colour scale in µV; ``None`` takes the smallest / largest value of the
        maps in ``valid``.
    color : colour spec or None, optional
        The unit's colour; the scale runs white -> this colour -> a darker
        shade of it. ``None`` keeps white -> blue -> dark blue.
    savepath : str or None, optional
        Saved there as SVG; the figure is closed afterwards either way.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(maps)
    shown = [maps[g] for g in valid]
    if vmin is None:
        vmin = min(float(np.nanmin(m)) for m in shown)
    if vmax is None:
        vmax = max(float(np.nanmax(m)) for m in shown)

    cmap = SCORE_CMAP.copy() if color is None else mu_heatmap_cmap(color)
    cmap.set_bad(color=HEATMAP_GAP_COLOR)

    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1.0, 4.8), squeeze=False)
    axes = axes[0]
    im = None
    for g, ax in enumerate(axes):
        rows, cols = maps[g].shape
        if g in valid:
            im = ax.imshow(np.ma.masked_invalid(maps[g]), cmap=cmap,
                           vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(f"{names[g]}\n(n={n_spikes[g]})", fontsize=10)
        else:
            ax.imshow(np.full((rows, cols), np.nan), cmap=cmap, aspect="equal")
            ax.set_title(f"{names[g]}\n(n={n_spikes[g]} < {min_spikes})",
                         fontsize=10)
        ax.set_xticks(np.arange(cols))
        ax.set_xticklabels(np.arange(1, cols + 1), fontsize=6)
        ax.set_yticks(np.arange(rows))
        ax.set_yticklabels(np.arange(1, rows + 1) if g == 0 else [],
                           fontsize=6)
        ax.set_xlabel("Grid column", fontsize=8)
        if g == 0:
            ax.set_ylabel("Grid row", fontsize=8)

    # One colour bar for the whole row: the scale is shared.
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.03)
    cbar.set_label("Peak-to-peak (µV)", fontsize=9)
    fig.suptitle(title, fontsize=11)
    save_svg(fig, savepath)
    plt.close(fig)
    return fig


def plot_mean_muaps(waveforms, titles, fsamp, title=None, savepath=None,
                    task_waveforms=None, task_titles=None, task_names=None):
    """
    Mean MUAP of every motor unit side by side, on one shared y-axis.

    With ``task_waveforms`` the figure becomes a grid: the first row is the
    overall mean MUAP, below it one row per task, one column per unit.

    Parameters
    ----------
    waveforms : sequence of 1-D array_like
        One mean MUAP per unit in µV, centred on the discharge.
    titles : sequence of str
        Panel titles, one per waveform.
    fsamp : float
        Sampling frequency in Hz (time axis in ms, 0 at the discharge).
    title : str or None, optional
        Figure title.
    savepath : str or None, optional
        Saved there as SVG; the figure is closed afterwards either way.
    task_waveforms : sequence of sequence of (1-D array_like or None), optional
        ``task_waveforms[u][g]`` is the mean MUAP of unit ``u`` in task ``g``,
        ``None`` for a task without one (the panel stays empty).
    task_titles : sequence of sequence of str, optional
        Panel titles, shaped like ``task_waveforms``.
    task_names : sequence of str, optional
        Row labels, one per task.

    Returns
    -------
    matplotlib.figure.Figure
    """
    waveforms = [np.asarray(w, dtype=float).ravel() for w in waveforms]
    if not waveforms:
        raise ValueError("waveforms is empty -- nothing to draw.")

    n = len(waveforms)
    # Row 0 is the overall mean; one row per task below it.
    rows = [list(zip(waveforms, titles))]
    row_names = ["Overall"]
    if task_waveforms is not None:
        n_task = len(task_names)
        for g in range(n_task):
            rows.append([(None if task_waveforms[u][g] is None else
                          np.asarray(task_waveforms[u][g], dtype=float).ravel(),
                          task_titles[u][g]) for u in range(n)])
        row_names += list(task_names)

    drawn = [wf for row in rows for wf, _ in row if wf is not None]
    lo = min(float(w.min()) for w in drawn)
    hi = max(float(w.max()) for w in drawn)
    pad = 0.05 * (hi - lo or 1.0)

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(3.2 * n, 3.0 * n_rows + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    for r, row in enumerate(rows):
        for u, (wf, head) in enumerate(row):
            ax = axes[r, u]
            ax.axhline(0, color=(0.85, 0.85, 0.85), linewidth=0.6, zorder=0)
            if wf is not None:
                half = wf.size // 2
                t_ms = np.arange(-half, wf.size - half) / fsamp * 1000
                ax.plot(t_ms, wf, color=DARK_BLUE, linewidth=1.4)
                ax.set_xlim(t_ms[0], t_ms[-1])
            ax.set_title(head, fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Time (ms)")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        label = "Amplitude (µV)"
        if n_rows > 1:
            label = f"{row_names[r]}\n{label}"
        axes[r, 0].set_ylabel(label)
    axes[0, 0].set_ylim(lo - pad, hi + pad)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_svg(fig, savepath)
    plt.close(fig)
    return fig
