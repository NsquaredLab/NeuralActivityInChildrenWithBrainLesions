"""
_plot.py
========

Figures of the Figure 5 pipeline and the figure style they use.

    - ``make_exergame_figure`` / ``plot_exergame``
                    RMS envelope + shaded cue windows + binary prediction +
                    activation threshold of one run
    - ``make_confusion_figure`` / ``plot_confusion_matrix_event``
                    event-based 2x2 confusion matrix (TN undefined)

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as C
from _metrics import compute_confusion_matrix_event, compute_latency

log = C.get_logger(__name__)

# ---- Colours -----------------------------------------------------------------
DARK_BLUE = "#001C55"   # RMS envelope
CUE_RED = "#FC7C86"     # cue shading
CYAN = "#11B1D1"        # activation threshold

# DIN A4 width in inches (210 mm) minus a margin.
A4_WIDTH_IN = 8.27 * 0.9


# ---------------------------------------------------------------- style helpers
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


def upsample(reference, signal):
    """
    Linearly resample ``signal`` onto the length of ``reference`` (both mapped
    on a normalised [0, 1] axis).
    """
    signal = np.atleast_1d(np.asarray(signal, dtype=float))
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, len(reference))
    return np.interp(x_new, x_old, signal)


# ---------------------------------------------------------------- run plot
def plot_exergame(rms, ground_truth, prediction, fs, latencies=None,
                  threshold=None, cutX=None, savepath=None):
    """
    Draw the exergame figure: RMS envelope (left axis), shaded cue windows,
    binary prediction (right axis) and activation threshold.

    Parameters
    ----------
    rms : ndarray, shape (n_frames,)
        RMS envelope of the EMG (converted to microvolts for display).
    ground_truth : list of (float, float)
        Cue windows ``(start, end)`` in seconds (shaded).
    prediction : array_like
        Binary prediction (0/1) at control cadence; resampled onto the RMS length.
    fs : float
        EMG sampling frequency in Hz.
    latencies : list of float or None, optional
        Per-cue reaction latencies in seconds (not drawn).
    threshold : float or ndarray or None, optional
        Activation threshold in RMS units (drawn dashed); omitted if ``None``.
    cutX : float or None, optional
        If given, crop the plot to [0, cutX] seconds.
    savepath : str or None, optional
        If given, saved to this path as SVG (300 dpi).

    Returns
    -------
    matplotlib.figure.Figure
    """
    use_journal_style(base_size=10)

    # Time axis (RMS defines the plot's time base) and prediction alignment.
    time = np.arange(len(rms)) / fs
    prediction = upsample(rms, prediction)

    # The threshold is stretched onto the same time base *before* any cropping.
    if threshold is None:
        threshold_line = None
    elif np.ndim(threshold) > 0:
        threshold_line = upsample(rms, np.atleast_1d(threshold))
    else:
        threshold_line = np.full_like(time, float(threshold))

    # Optional crop to [0, cutX].
    if cutX is not None:
        mask = time <= cutX
        time, rms, prediction = time[mask], rms[mask], prediction[mask]
        if threshold_line is not None:
            threshold_line = threshold_line[mask]
        ground_truth = [(start, min(end, cutX))
                        for (start, end) in ground_truth if start <= cutX]

    fig, ax1 = plt.subplots(figsize=(A4_WIDTH_IN, A4_WIDTH_IN * 0.28))
    ax2 = ax1.twinx()

    # ---- RMS envelope (left axis, mV -> µV) --------------------------------------
    ax1.plot(time, rms * 1000, label="RMS (EMG)",
             color=DARK_BLUE, alpha=0.7, linewidth=1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMS Amplitude (µV)", color=DARK_BLUE)
    ax1.tick_params(axis="y", labelcolor=DARK_BLUE, width=1, length=6)
    ax1.tick_params(axis="x", width=1, length=6, labelsize=10)
    if cutX is not None:
        ax1.set_xticks(np.arange(0, cutX + 4, 4))
    else:
        ax1.set_xticks(np.arange(0, len(time) / fs, 4))

    # Shade the cue windows.
    for (start, end) in ground_truth:
        ax1.axvspan(start, end, color=CUE_RED, alpha=0.3,
                    label="Cue" if (start, end) == ground_truth[0] else "")

    # ---- Binary prediction (right axis) -----------------------------------------
    ax2.plot(time, prediction, label="Prediction",
             color="black", alpha=0.8, linewidth=1)
    ax2.set_ylabel("Prediction", color="black")
    ax2.tick_params(axis="y", labelcolor="black", width=1, length=6)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([0, 1])

    # ---- Activation threshold ----------------------------------------------------
    if threshold_line is not None:
        ax1.plot(time, threshold_line * 1000, label="Threshold",
                 color=CYAN, linestyle="--", linewidth=1)  # threshold in mV -> µV

    # ---- Combined legend ---------------------------------------------------------
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Attached to ax2 (drawn on top of ax1) so the opaque box covers the
    # prediction trace.
    leg = ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8,
                     framealpha=1.0, facecolor="white", edgecolor="black")
    leg.get_frame().set_alpha(1.0)
    leg.set_zorder(10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    save_svg(fig, savepath)
    plt.show()
    reset_style()
    return fig


def make_exergame_figure(pkl_path, ground_truth=None, cutX=None, savepath=None):
    """
    The run plot of one ``myocontrol_data*.pkl`` recording (threshold =
    ``rms_reference * prediction_threshold``). ``ground_truth=None`` reads the
    cue windows from the recording's ``metronome`` track.
    """
    with C.step(log, "load exergame recording"):
        rec = C.load_exergame_recording(pkl_path)
    ground_truth = C.resolve_ground_truth(rec, ground_truth)

    # Reaction latency per cue.
    latencies = compute_latency(ground_truth, rec.predictions, C.FS, C.FRAME)

    with C.step(log, "draw exergame figure"):
        return plot_exergame(rec.rms, ground_truth, rec.predictions, C.FS,
                             latencies=latencies, threshold=rec.threshold,
                             cutX=cutX, savepath=savepath)


# ---------------------------------------------------------------- confusion matrix
def plot_confusion_matrix_event(counts, n_gt, title=None, savepath=None):
    """
    Draw the event-based confusion matrix as a 2x2 heatmap::

                    Pred Active     Pred Rest
        GT Active       TP              FN
        GT Rest         FP             N/A

    The "true negative event" cell is undefined for event-based metrics and is
    labelled "N/A". ``n_gt`` (number of cue windows) scales the colour axis.
    """
    tp, fn, fp = counts["tp"], counts["fn"], counts["fp"]
    cm = np.array([[tp, fn],
                   [fp, np.nan]], dtype=float)  # TN cell masked

    scale = 1.2
    size = 32 * scale
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(figsize=(6 * scale, 5 * scale))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(cm)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=n_gt)

    # Annotate each defined cell; label the N/A cell.
    for (r, c), value in np.ndenumerate(cm):
        if np.isnan(value):
            ax.text(c, r, "N/A", ha="center", va="center",
                    fontsize=size, color="gray")
        else:
            txt_color = "white" if value > 0.6 * n_gt else "black"
            ax.text(c, r, f"{int(value)}", ha="center", va="center",
                    fontsize=size, color=txt_color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Active", "Rest"], fontsize=size - 4)
    ax.set_yticklabels(["Active", "Rest"], fontsize=size - 4,
                       rotation=90, va="center")
    ax.set_xlabel("Prediction", fontsize=size)
    ax.set_ylabel("Ground Truth", fontsize=size)
    full_title = "Event-based" if title is None else f"{title} (Event-based)"
    ax.set_title(full_title, fontsize=size)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(np.arange(0, n_gt + 1))
    cbar.ax.tick_params(labelsize=size - 6)

    plt.tight_layout()
    save_svg(fig, savepath)
    plt.show()
    reset_style()
    return fig


def make_confusion_figure(pkl_path, ground_truth=None, title=None, savepath=None,
                          min_burst_seconds=0.5):
    """
    The event-based confusion matrix of one ``myocontrol_data*.pkl`` recording.
    ``ground_truth=None`` reads the cue windows from the recording's
    ``metronome`` track.
    """
    with C.step(log, "load exergame recording"):
        rec = C.load_exergame_recording(pkl_path)
    ground_truth = C.resolve_ground_truth(rec, ground_truth)

    counts, metrics, detected = compute_confusion_matrix_event(
        ground_truth, rec.predictions, C.FS, C.FRAME,
        min_burst_seconds=min_burst_seconds)
    log.info("Event-based -> Precision: %.3f, Recall: %.3f, F1: %.3f",
             metrics["precision"], metrics["recall"], metrics["f1"])
    log.debug("Per-cue detected: %s", detected)

    with C.step(log, "draw confusion matrix"):
        return plot_confusion_matrix_event(counts, n_gt=len(ground_truth),
                                           title=title, savepath=savepath)
