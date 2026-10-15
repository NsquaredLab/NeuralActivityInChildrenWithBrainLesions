"""
_tracking.py
============

Figure builders of the Figure 2 pipeline: they take the tracking file of a
recording day (all gestures decomposed with one set of unit filters, so unit
``k`` is the same motor unit in every gesture), compute what a panel shows and
hand it to ``_plot.py``.

    - ``make_tracking_rates_figure``   RMS envelope + discharge rate and pulses
                                       of every tracked unit, cue shaded
    - ``prepare_tracking_grid``        tracking file in grid layout, checked
                                       against the gesture segments and filtered
    - ``make_heatmap_figure``          P2P heatmap per tracked unit and gesture
    - ``make_mean_muap_figure``        mean MUAP per tracked unit, overall and
                                       per gesture

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os

import numpy as np

import _common as C
import _emg as E
from _plot import plot_mean_muaps, plot_mu_heatmap_tracking, plot_mu_tracking_rates
from _style import DARK_BLUE, mu_colors

log = C.get_logger(__name__)

# Default RMS window of the tracking panel, in seconds.
TRACKING_RMS_WINDOW = 0.2

# Default Hann window of the discharge rates, in seconds.
TRACKING_RATE_WINDOW = 1.0

# Default second smoothing of the RMS envelope, in seconds: a Hann window over
# the finished envelope (the moving average of compute_rms is a rectangle and
# leaves the envelope jagged). 0 turns it off.
TRACKING_RATES_RMS_SMOOTH = 0.5

# Heatmaps / mean MUAPs: default half window around a discharge in ms and
# fewest discharges a gesture needs for a MUAP.
SHIMMER_WIN_MS = 25.0
SHIMMER_MIN_SPIKES = 10


def _smooth_envelope(envelope, fsamp, seconds):
    """
    ``envelope`` convolved with a unit-area Hann window of ``seconds``.

    Edge-padded before the convolution, so the envelope keeps its level at
    both ends. ``seconds <= 0``, or a window of fewer than three samples,
    returns it unchanged.
    """
    envelope = np.asarray(envelope, dtype=float)
    n = int(round(seconds * fsamp))
    if seconds <= 0 or n < 3 or envelope.size == 0:
        return envelope
    kernel = np.hanning(n)
    kernel /= kernel.sum()
    half = n // 2
    padded = np.pad(envelope, (half, n - 1 - half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def block_spike_samples(mu_pulses, start, stop):
    """
    The discharges of every motor unit inside one gesture, in samples from its
    start. One entry per motor unit, in file order -- a unit that is silent in
    this gesture keeps its place as an empty array.
    """
    trains = []
    for pulses in mu_pulses:
        pulses = np.asarray(pulses, dtype=float).ravel()
        inside = pulses[(pulses >= start) & (pulses < stop)]
        trains.append(inside - start)
    return trains


def _mu_labels(mu_pulses, pnr):
    """``MU k (PNR dB)`` per unit, or ``MU k`` when the file has no PNR."""
    if pnr is not None and np.size(pnr) == mu_pulses.size:
        return [f"MU {k + 1} ({value:.1f}dB)"
                for k, value in enumerate(np.ravel(pnr))]
    return [f"MU {k + 1}" for k in range(mu_pulses.size)]


def make_tracking_rates_figure(tracking, save_prefix=None,
                               rms_window=TRACKING_RMS_WINDOW, show_emg=True,
                               rate_window=TRACKING_RATE_WINDOW,
                               cue_blocks=None, bare_copy=False,
                               rms_smooth=TRACKING_RATES_RMS_SMOOTH):
    """
    Tracking panel with discharge rates: one column per gesture, the RMS
    envelope on top and below it every motor unit's discharge rate (spike
    train convolved with a ``rate_window``-second Hann window, in pps) with
    its pulses directly underneath. The cue periods are shaded gray: from the
    ``labels`` of ``cue_blocks`` when given, otherwise from the tracking
    file's trigger.

    Parameters
    ----------
    tracking : dict
        As returned by ``_common.load_tracking``.
    save_prefix : str or None, optional
        SVG prefix; saved as ``<prefix>_tracking_rates.svg``.
    rms_window : float, optional
        RMS window in seconds.
    show_emg : bool, optional
        Draw the mean over all channels behind each envelope (default True).
    rate_window : float, optional
        Length of the Hann window in seconds.
    cue_blocks : list of dict or None, optional
        The gesture segments of ``_common.recording_day_blocks``, in the same
        order as the tracking file's blocks; their ``labels`` give the cue
        periods.
    bare_copy : bool, optional
        Also save the same figure without any text as
        ``<prefix>_tracking_rates_bare.svg`` (default False).
    rms_smooth : float, optional
        Hann window in seconds laid over the finished RMS envelope.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when not one block of the file could be drawn.
    """
    emg = np.asarray(tracking["emg"], dtype=float)
    fsamp = float(tracking["fsamp"])
    mu_pulses = np.asarray(tracking["MUPulses"], dtype=object).ravel()
    trigger = tracking.get("trigger")

    if emg.size == 0:
        log.warning("The tracking file holds no channel -- no figure.")
        return None
    if cue_blocks is not None and len(cue_blocks) != len(tracking["blocks"]):
        log.warning("%d cue segment(s) for %d tracking block(s) -- the cues "
                    "cannot be matched, falling back to the trigger.",
                    len(cue_blocks), len(tracking["blocks"]))
        cue_blocks = None
    if cue_blocks is None and trigger is None:
        log.warning("Neither cue segments nor a trigger -- the rate panel is "
                    "drawn without cue shading.")

    n_samples = emg.shape[1]
    columns = []
    for b, block in enumerate(tracking["blocks"]):
        start = max(0, int(block["start"]))
        stop = min(n_samples, int(block["stop"]))
        if stop <= start:
            log.warning("Tracking: skipping block %s, its samples (%d..%d) are "
                        "not inside the signal (%d samples).",
                        block["label"], block["start"], block["stop"], n_samples)
            continue

        segment = emg[:, start:stop]
        spikes = block_spike_samples(mu_pulses, start, stop)
        # "same": the rate on the gesture's own time base.
        rates = [E.hann_discharge_rate(train.astype(int), stop - start, fsamp,
                                       window=rate_window, mode="same")
                 for train in spikes]

        holds = []
        labels = cue_blocks[b].get("labels") if cue_blocks is not None else None
        if labels is not None:
            holds = E.hold_windows(labels, fsamp)
        elif trigger is not None and trigger.size >= stop:
            on, off = E.extract_trigger(trigger[start:stop])
            holds = [(s / fsamp, (e + 1) / fsamp) for s, e in zip(on, off)]
        log.debug("%s: %d cue period(s), mean rate per MU %s pps.",
                  block["label"], len(holds),
                  [round(float(r[r > 0].mean()), 1) if np.any(r > 0) else 0
                   for r in rates])

        columns.append({
            "label": block["label"],
            "rms": _smooth_envelope(E.compute_rms(segment, fsamp, rms_window),
                                    fsamp, rms_smooth),
            "fsamp": fsamp,
            "emg_mean": np.mean(segment, axis=0) if show_emg else None,
            "spikes": [train / fsamp for train in spikes],
            "rates": rates,
            "holds": holds,
            # The envelope in DARK_BLUE: the colours below belong to the
            # motor units. The column title names the gesture.
            "color": DARK_BLUE,
        })

    if not columns:
        log.warning("Not one block of the tracking file could be drawn.")
        return None

    mu_labels = _mu_labels(mu_pulses, tracking.get("PNR"))
    savepath = (None if save_prefix is None
                else f"{save_prefix}_tracking_rates.svg")
    with C.step(log, f"tracking rate panel ({len(columns)} gesture(s), "
                     f"{mu_pulses.size} MU(s))"):
        fig = plot_mu_tracking_rates(columns, mu_labels=mu_labels,
                                     savepath=savepath)
    if bare_copy:
        bare_path = (None if save_prefix is None
                     else f"{save_prefix}_tracking_rates_bare.svg")
        plot_mu_tracking_rates(columns, mu_labels=mu_labels,
                               savepath=bare_path, bare=True)
    return fig


def prepare_tracking_grid(tracking_mat, blocks, attribute=None):
    """
    Load a tracking ``.mat`` in grid layout, check it against the gesture
    segments and filter it -- once, for the heatmaps and the mean MUAPs.

    Parameters
    ----------
    tracking_mat : str
        The tracking ``.mat`` (``_common.load_tracking_grid``).
    blocks : list of dict
        The gesture segments, as ``_common.recording_day_blocks`` returns them.
    attribute : str or None, optional
        Recording-day key; its channels in ``_common.EXTRA_DISCARDED_CHANNELS``
        are discarded on top of the file's own ``discardChannelsVec``.

    Returns
    -------
    dict
        As ``load_tracking_grid``, with ``sig`` filtered (50 Hz notch +
        20-500 Hz band-pass) and ``discard`` extended.
    """
    grid = C.load_tracking_grid(tracking_mat)
    fs = grid["fsamp"]
    n_samples = grid["sig"].shape[-1]

    rates = {b["fsamp"] for b in blocks if "fsamp" in b}
    if rates and rates != {fs}:
        raise ValueError(f"The segments were recorded at {sorted(rates)} Hz, "
                         f"the tracking file at {fs:g} Hz.")
    if blocks[-1]["stop"] != n_samples:
        raise ValueError(f"The segments cover {blocks[-1]['stop']} samples, "
                         f"the tracking file holds {n_samples} -- is the pkl "
                         f"the one this tracking file was built from?")

    grid["discard"] = E.add_extra_discards(grid["discard"], attribute)
    with C.step(log, "filtering the tracking grid"):
        grid["sig"] = E.bandpass_notch_grid(grid["sig"], fs)
    return grid


def _cut_per_gesture(sig, pulses, blocks, half):
    """Every discharge of one unit cut out of ``sig``, one array per gesture."""
    empty = np.empty((0,) + sig.shape[:2] + (2 * half + 1,))
    per = []
    for b in blocks:
        pg = pulses[(pulses >= b["start"]) & (pulses < b["stop"])]
        per.append(E.cut_muap_waveforms(sig, pg, half)[0] if len(pg) else empty)
    return per


def make_heatmap_figure(tracking_mat, blocks, out_dir, save_prefix=None,
                        win_ms=SHIMMER_WIN_MS, min_spikes=SHIMMER_MIN_SPIKES,
                        grid=None, attribute=None):
    """
    P2P heatmap of every tracked motor unit, one panel per gesture.

    For each unit and gesture the discharges are cut out of the filtered grid
    (``±win_ms``) and averaged; the peak-to-peak amplitude of that mean MUAP on
    every electrode is the map. Discarded electrodes inside a grid are filled
    from a neighbour (``fill_heatmap_nans``), the gap columns stay white. Every
    panel of every unit shares one colour scale, 0 to the largest P2P value of
    the whole file. One SVG per unit (``..._MU_<id>_heatmap.svg``); the title
    carries the spatial correlation of every pair of gestures, computed on the
    unfilled maps.

    Parameters
    ----------
    tracking_mat : str
        The tracking ``.mat``.
    blocks : list of dict
        The gesture segments, as ``_common.recording_day_blocks`` returns them.
    out_dir : str
        Directory for the SVGs; created if missing.
    save_prefix : str or None, optional
        Put in front of every file name.
    win_ms : float, optional
        Half window around a discharge in ms (default 25).
    min_spikes : int, optional
        A gesture with fewer discharges gets no map (default 10).
    grid : dict or None, optional
        The output of :func:`prepare_tracking_grid`; ``None`` prepares it here.
    attribute : str or None, optional
        Only used when ``grid`` is ``None``.

    Returns
    -------
    list of dict
        Per drawn unit: ``MU``, ``maps``, ``spatial``, ``valid``, ``n_spikes``,
        ``pnr`` and ``color``.
    """
    if grid is None:
        grid = prepare_tracking_grid(tracking_mat, blocks, attribute=attribute)
    fs = grid["fsamp"]
    sig = grid["sig"]
    discard = grid["discard"]
    half = int(round(win_ms / 1000 * fs))
    names = [b["label"] for b in blocks]

    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{save_prefix}_" if save_prefix else ""

    # Each unit's heatmap in its colour of the tracking rate panel.
    unit_colors = mu_colors(len(grid["pulses"]))

    # Two passes: every map first, then the figures -- the colour scale ends
    # at the largest value of the whole file, known only after the last unit.
    results = []
    for u, (pulses, mid) in enumerate(zip(grid["pulses"], grid["ids"])):
        per = _cut_per_gesture(sig, pulses, blocks, half)
        valid = [g for g in range(len(blocks)) if len(per[g]) >= min_spikes]
        if not valid:
            log.info("MU %s: fewer than %d discharges in every gesture -- no "
                     "heatmap.", mid, min_spikes)
            continue

        raw = {g: E.p2p_map(per[g].mean(0), discard) for g in valid}
        spatial = {(g1, g2): E.spatial_correlation(raw[g1], raw[g2])
                   for i, g1 in enumerate(valid) for g2 in valid[i + 1:]}
        nan_map = np.full(discard.shape, np.nan)
        maps = [E.fill_heatmap_nans(raw[g], discard.astype(int), C.GRIDS)
                if g in raw else nan_map for g in range(len(blocks))]

        results.append({"MU": mid, "maps": maps, "spatial": spatial,
                        "valid": valid, "n_spikes": [len(w) for w in per],
                        "pnr": grid["pnr"][u], "color": unit_colors[u]})

    if not results:
        log.warning("No motor unit has %d discharges in any gesture -- no "
                    "heatmap.", min_spikes)
        return results

    vmax = max(float(np.nanmax(r["maps"][g]))
               for r in results for g in r["valid"])
    log.info("Heatmap colour scale: 0 to %.1f µV (largest P2P of the file).",
             vmax)

    for r in results:
        simtxt = "   ".join(f"{names[g1]}↔{names[g2]}: {c:.2f}"
                            for (g1, g2), c in r["spatial"].items())
        title = f"MU {r['MU']}  |  PNR {r['pnr']:.1f} dB"
        if simtxt:
            title += f"\nspatial r  {simtxt}"
        plot_mu_heatmap_tracking(
            r["maps"], names, r["valid"], r["n_spikes"], title,
            min_spikes=min_spikes, vmin=0.0, vmax=vmax, color=r["color"],
            savepath=os.path.join(out_dir,
                                  f"{prefix}MU_{r['MU']}_heatmap.svg"))
        log.info("MU %s: heatmap, spatial r=%s", r["MU"],
                 {f"{names[g1]}-{names[g2]}": round(c, 2)
                  for (g1, g2), c in r["spatial"].items()})
    return results


def make_mean_muap_figure(tracking_mat, blocks, out_dir, save_prefix=None,
                          win_ms=SHIMMER_WIN_MS, min_spikes=SHIMMER_MIN_SPIKES,
                          grid=None, attribute=None):
    """
    Mean MUAP of every tracked motor unit, overall and per gesture.

    Every discharge of the unit inside a gesture segment is cut out of the
    filtered grid (``±win_ms``) and averaged. The top row shows that mean on
    the unit's strongest electrode (largest peak-to-peak among the usable
    ones), one column per unit. Below it one row per gesture: the mean of that
    gesture's discharges only, on the *same* electrode; a gesture with fewer
    than ``min_spikes`` discharges stays empty. All panels share one y-axis.
    Saved as ``<prefix>_mean_muap.svg``.

    Parameters
    ----------
    tracking_mat : str
        The tracking ``.mat``.
    blocks : list of dict
        The gesture segments, as ``_common.recording_day_blocks`` returns them.
    out_dir : str
        Directory for the SVG; created if missing.
    save_prefix : str or None, optional
        Put in front of the file name.
    win_ms : float, optional
        Half window around a discharge in ms (default 25).
    min_spikes : int, optional
        A unit with fewer discharges in total gets no column, a gesture with
        fewer an empty panel (default 10).
    grid : dict or None, optional
        The output of :func:`prepare_tracking_grid`; ``None`` prepares it here.
    attribute : str or None, optional
        Only used when ``grid`` is ``None``.

    Returns
    -------
    list of dict
        Per drawn unit: ``MU``, ``n``, ``channel`` (0-based ``(row, col)``),
        ``p2p`` in µV, ``waveform`` and ``per_task``.
    """
    if grid is None:
        grid = prepare_tracking_grid(tracking_mat, blocks, attribute=attribute)
    fs = grid["fsamp"]
    discard = grid["discard"]
    half = int(round(win_ms / 1000 * fs))
    names = [b["label"] for b in blocks]

    results = []
    for pulses, mid in zip(grid["pulses"], grid["ids"]):
        # All gestures together, but only the discharges inside a segment --
        # the same ones the heatmaps average.
        per = _cut_per_gesture(grid["sig"], pulses, blocks, half)
        waveforms = np.concatenate(per)
        if len(waveforms) < min_spikes:
            log.info("MU %s: %d discharge(s) in total, fewer than %d -- no "
                     "mean MUAP.", mid, len(waveforms), min_spikes)
            continue
        mean = waveforms.mean(0)
        amp = E.p2p_map(mean, discard)
        i, j = np.unravel_index(np.nanargmax(amp), amp.shape)
        # Per gesture on the overall strongest channel, not its own -- else
        # the rows would show different electrodes.
        per_task = []
        for w in per:
            wf = w[:, i, j].mean(0) if len(w) >= min_spikes else None
            per_task.append({"n": len(w), "waveform": wf,
                             "p2p": None if wf is None else float(np.ptp(wf))})
        results.append({"MU": mid, "n": len(waveforms), "channel": (i, j),
                        "p2p": float(amp[i, j]), "waveform": mean[i, j],
                        "per_task": per_task})
        log.info("MU %s: mean MUAP of %d discharges, strongest channel r%d c%d "
                 "(%.1f µV p-p).", mid, len(waveforms), i + 1, j + 1,
                 amp[i, j])

    if not results:
        log.warning("No motor unit has %d discharges -- no mean MUAP figure.",
                    min_spikes)
        return results

    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{save_prefix}_" if save_prefix else ""
    titles = [f"MU {r['MU']}  (n={r['n']})\nch r{r['channel'][0] + 1} "
              f"c{r['channel'][1] + 1}, {r['p2p']:.0f} µV p-p"
              for r in results]
    task_titles = [[f"{names[g]}  (n={t['n']})\n{t['p2p']:.0f} µV p-p"
                    if t["waveform"] is not None
                    else f"{names[g]}  (n={t['n']} < {min_spikes})"
                    for g, t in enumerate(r["per_task"])] for r in results]
    with C.step(log, f"mean MUAP figure ({len(results)} MU(s))"):
        plot_mean_muaps([r["waveform"] for r in results], titles, fs,
                        title=f"Mean MUAP, overall and per gesture  |  "
                              f"±{win_ms:.0f} ms",
                        savepath=os.path.join(out_dir,
                                              f"{prefix}mean_muap.svg"),
                        task_waveforms=[[t["waveform"] for t in r["per_task"]]
                                        for r in results],
                        task_titles=task_titles, task_names=names)
    return results
