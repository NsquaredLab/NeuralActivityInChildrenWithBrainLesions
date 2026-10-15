"""
_xcorr.py
=========

Cross-correlation building blocks of the Figure 3 pipeline:

    - ``hann_discharge_rate``      discharge rate = spike train convolved with a
                                   Hann window (pps)
    - ``xcorr_coeff``              MATLAB's ``xcorr(x - mean(x), y - mean(y),
                                   maxlag, 'normalized')``; the value at lag 0 is
                                   Pearson's r
    - ``hold_windows``             0/1 labels -> hold periods
    - ``compute_task_hold_xcorr``  per unit: xcorr of its rate with the cue, per
                                   hold and over the whole gesture
    - ``build_overview`` / ``COLUMNS_*``
                                   the per-unit overview table and the layout of
                                   the tables written by steps 01 and 02

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import numpy as np
from scipy.signal import correlate as _correlate

import _common as C

log = C.get_logger(__name__)

# Default largest shift of the rate against the cue, in seconds (both
# directions). The steps pass their own value.
MAX_LAG_SECONDS = 1.5

# Default length of the Hann window of the discharge rate, in seconds. The
# steps pass their own window.
HANN_RATE_WINDOW = 2.0

# Layout of the tables written by steps 01 and 02 (read by steps 03 and 04).
COLUMNS_XCORR_HOLD = [
    "patient_id", "visit", "task", "mu", "pnr", "hold", "n_discharges",
    "active", "xcorr_peak", "xcorr_lag_s", "xcorr_lag0",
    "max_lag_s", "min_discharges_per_ramp", "rate_window_s", "source",
]
COLUMNS_XCORR_MU = [
    "patient_id", "visit", "task", "mu", "pnr", "n_holds", "n_holds_active",
    "xcorr_peak_mean", "xcorr_lag_mean_s", "xcorr_lag0_mean",
    "xcorr_whole_peak", "xcorr_whole_lag_s", "xcorr_whole_lag0",
    "max_lag_s", "min_discharges_per_ramp", "rate_window_s", "source",
]
COLUMNS_OVERVIEW = [
    "patient_id", "visit", "task", "mu", "n_holds_used",
    "xcorr_lag0_hold_mean", "xcorr_lag0_hold_std", "xcorr_global_lag0",
]


# ---------------------------------------------------------------- discharge rate
def hann_discharge_rate(spike_indices, n_samples, fsamp,
                        window=HANN_RATE_WINDOW, mode="full"):
    """
    Discharge rate as the spike train convolved with a **Hann window**.

    The MATLAB definition, reproduced::

        hannwindow = hann(fsamp * win_length);
        st_smooth  = conv(spike_train, hannwindow);

    ``numpy.hanning(N)`` is MATLAB's symmetric ``hann(N)``, and ``mode="full"``
    is what ``conv`` does by default.

    Parameters
    ----------
    spike_indices : array_like of int
        Discharge sample indices of one motor unit (``MUPulses[i]``).
    n_samples : int
        Length of the recording in samples. Discharges outside
        ``[0, n_samples)`` are dropped.
    fsamp : float
        Sampling frequency in Hz.
    window : float, optional
        Length of the Hann window in seconds (default :data:`HANN_RATE_WINDOW`).
    mode : {"full", "same"}, optional
        ``"full"`` (default) is MATLAB's ``conv``: ``n_samples + N - 1`` long.
        ``"same"`` cuts the central ``n_samples`` out of it, the trace that
        lines up with the recording's time base.

    Returns
    -------
    ndarray
        Discharge rate in pulses per second. The convolution is divided by
        ``sum(window) / fsamp`` so the trace is a rate rather than a sum of
        window weights.
    """
    if mode not in ("full", "same"):
        raise ValueError(f"mode must be 'full' or 'same', not {mode!r}.")

    n_samples = int(n_samples)
    width = max(int(round(window * fsamp)), 1)

    spike_train = np.zeros(n_samples, dtype=float)
    idx = np.asarray(spike_indices, dtype=int).ravel()
    inside = idx[(idx >= 0) & (idx < n_samples)]
    if inside.size < idx.size:
        log.debug("Dropped %d discharge(s) outside the recording.",
                  idx.size - inside.size)
    spike_train[inside] = 1.0

    # numpy.hanning is MATLAB's hann: symmetric, zero at both ends, N points.
    kernel = np.hanning(width)
    total = float(np.sum(kernel))
    smoothed = np.convolve(spike_train, kernel, "full")
    if total > 0:
        smoothed = smoothed * float(fsamp) / total

    if mode == "same":
        # The central n_samples, the same window numpy's "same" would return.
        start = (smoothed.size - n_samples) // 2
        return smoothed[start:start + n_samples]
    return smoothed


# ---------------------------------------------------------------- cross-correlation
def xcorr_coeff(x, y, fsamp, max_lag=MAX_LAG_SECONDS):
    """
    MATLAB's ``xcorr(x - mean(x), y - mean(y), maxlag, 'normalized')``.

    Both signals are mean-subtracted first -- without that, two non-negative
    signals such as a firing rate and a trigger correlate highly whatever their
    timing. The normalisation is MATLAB's ``'normalized'`` (``'coeff'``):
    divided by ``sqrt(Rxx(0) * Ryy(0))``, so the value at lag 0 is Pearson's r.
    As in MATLAB, a shift pads with zeros: the further the lag, the shorter the
    overlap.

    Parameters
    ----------
    x, y : 1-D array_like
        The two signals, of the same length -- here the discharge rate and the
        cue.
    fsamp : float
        Sampling frequency in Hz.
    max_lag : float, optional
        Largest shift in seconds, in both directions (MATLAB's ``maxlag``).

    Returns
    -------
    lags_s : ndarray
        The lags in seconds, ``-max_lag .. +max_lag``. **Positive means** ``x``
        **follows** ``y`` -- the rate trails the cue.
    c : ndarray
        The coefficient at every lag, or all ``nan`` if either signal is flat.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must have the same length, got {x.size} "
                         f"and {y.size}.")

    lag = min(int(round(max_lag * fsamp)), max(x.size - 1, 0))
    lags = np.arange(-lag, lag + 1)
    x = x - x.mean()
    y = y - y.mean()
    norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if norm == 0:
        return lags / fsamp, np.full(lags.size, np.nan)

    # "full" puts lag m at index m + (N - 1); keep only |m| <= lag.
    full = _correlate(x, y, mode="full", method="auto")
    centre = x.size - 1
    return lags / fsamp, full[centre - lag:centre + lag + 1] / norm


# ---------------------------------------------------------------- hold periods
def extract_trigger(trigger, threshold, normalize=True):
    """
    Start/stop sample indices of every contiguous run where ``trigger`` is
    above ``threshold`` (min-max scaled to 0..1 first when ``normalize``).
    """
    trigger = np.asarray(trigger, dtype=float).ravel()
    if normalize:
        t_min, t_max = np.min(trigger), np.max(trigger)
        trigger = (np.zeros_like(trigger) if t_max - t_min <= 0
                   else (trigger - t_min) / (t_max - t_min))
    above = trigger > threshold
    # Pad with a zero on each side so runs at the very start/end are detected.
    d = np.diff(np.concatenate(([0], above.astype(int), [0])))
    start = np.where(d == 1)[0]
    stop = np.where(d == -1)[0] - 1
    return start, stop


def hold_windows(labels, fsamp):
    """
    The gesture hold periods as ``(start, end)`` pairs in seconds, from the
    per-sample 0/1 ``labels``.
    """
    labels = np.asarray(labels).ravel().astype(float)
    # The labels are already binary, so threshold at 0.5 on the raw values.
    start, stop = extract_trigger(labels, threshold=0.5, normalize=False)
    # +1 so the window ends just past the last active sample.
    return [(float(s) / fsamp, float(e + 1) / fsamp) for s, e in zip(start, stop)]


# ---------------------------------------------------------------- per task
# Keys the per-hold cross-correlation needs.
XCORR_KEYS = ("labels", "fsamp", "trigger", "MUPulses")


def compute_task_hold_xcorr(task, task_name=None, rate_window=1.5,
                            max_lag=MAX_LAG_SECONDS, min_discharges_per_hold=5):
    """
    Cross-correlation of every motor unit's discharge rate with the trigger,
    one hold at a time.

    Per unit, the discharges are smoothed into a rate over the whole task
    (:func:`hann_discharge_rate`, so the smoothing does not start at the hold's
    edge), then rate and trigger are cut to each hold and correlated with
    :func:`xcorr_coeff`.

    A hold counts as **active** for a unit when the unit discharged at least
    ``min_discharges_per_hold`` times inside it. Every hold is reported; the
    per-unit means are taken over the active ones only. No PNR threshold and
    no gate on the detected holds: every unit, every hold.

    Next to that, the same cross-correlation once over the **whole task** --
    rate and trigger of the entire gesture, rest included. A unit that fires in
    the pauses rather than during the cue shows up there with a negative
    ``xcorr_whole_lag0``.

    Parameters
    ----------
    task : dict
        Task dict; must contain ``labels``, ``fsamp``, ``trigger`` and
        ``MUPulses``. ``PNR`` is copied into the rows when present.
    task_name : str or None, optional
        Task key, copied into every row.
    rate_window : float, optional
        Length of the Hann window of the discharge rate, in seconds.
    max_lag : float, optional
        Largest shift in seconds, in both directions.
    min_discharges_per_hold : int, optional
        Fewest discharges inside a hold for it to count as active.

    Returns
    -------
    hold_rows : list of dict
        One row per unit and hold: ``mu``, ``hold``, ``n_discharges``,
        ``active``, ``xcorr_peak`` (largest coefficient), ``xcorr_lag_s`` (its
        lag, positive = the rate trails the trigger), ``xcorr_lag0``.
    mu_rows : list of dict
        One row per unit: ``n_holds``, ``n_holds_active``, the means of the
        three values over the active holds (``nan`` if none was active), and
        ``xcorr_whole_peak`` / ``xcorr_whole_lag_s`` / ``xcorr_whole_lag0``
        over the whole task (``nan`` for a unit that never fires).
    """
    missing = [key for key in XCORR_KEYS if key not in task]
    if missing:
        raise KeyError(f"Task is missing {missing} -- the cross-correlation "
                       f"needs the decomposition. Present keys: {sorted(task)}")

    labels = np.asarray(task["labels"]).ravel()
    fsamp = float(task["fsamp"])
    trigger = np.asarray(task["trigger"], dtype=float).ravel()
    if trigger.size != labels.size:
        raise ValueError(
            f"trigger has {trigger.size} samples but labels have {labels.size}. "
            f"Both should be on the EMG's time base.")

    n_samples = labels.size
    mu_pulses = list(np.asarray(task["MUPulses"], dtype=object).ravel())
    pnr = task.get("PNR")
    pnr = (np.full(len(mu_pulses), np.nan) if pnr is None
           else np.atleast_1d(np.asarray(pnr, dtype=float)).ravel())
    windows = [(int(round(start * fsamp)), int(round(end * fsamp)))
               for start, end in hold_windows(labels, fsamp)]

    hold_rows = []
    mu_rows = []
    with C.step(log, f"per-hold xcorr ({len(mu_pulses)} MUs, "
                     f"{len(windows)} holds)"):
        for i, pulses in enumerate(mu_pulses):
            pulses = np.asarray(pulses, dtype=int).ravel()
            rate = hann_discharge_rate(pulses, n_samples, fsamp,
                                       window=rate_window, mode="same")
            unit_pnr = float(pnr[0] if pnr.size == 1 else pnr[i])

            active = []
            for h, (start, end) in enumerate(windows):
                n_discharges = int(np.count_nonzero(
                    (pulses >= start) & (pulses < end)))
                is_active = n_discharges >= min_discharges_per_hold
                peak = lag_s = lag0 = float("nan")
                if is_active:
                    lags_s, c = xcorr_coeff(rate[start:end],
                                            trigger[start:end], fsamp,
                                            max_lag=max_lag)
                    if np.any(np.isfinite(c)):
                        # max, not max(abs): a unit firing against the
                        # movement does not follow it.
                        best = int(np.nanargmax(c))
                        peak = float(c[best])
                        lag_s = float(lags_s[best])
                        lag0 = float(c[lags_s.size // 2])
                        active.append((peak, lag_s, lag0))
                hold_rows.append({
                    "task": task_name, "mu": i + 1, "pnr": unit_pnr,
                    "hold": h + 1, "n_discharges": n_discharges,
                    "active": is_active, "xcorr_peak": peak,
                    "xcorr_lag_s": lag_s, "xcorr_lag0": lag0,
                })

            means = (np.mean(active, axis=0) if active
                     else [float("nan")] * 3)

            # The whole gesture, rest included -- no ramps selected.
            whole_peak = whole_lag = whole_lag0 = float("nan")
            lags_s, c = xcorr_coeff(rate, trigger, fsamp, max_lag=max_lag)
            if np.any(np.isfinite(c)):
                best = int(np.nanargmax(c))
                whole_peak = float(c[best])
                whole_lag = float(lags_s[best])
                whole_lag0 = float(c[lags_s.size // 2])
            mu_rows.append({
                "task": task_name, "mu": i + 1, "pnr": unit_pnr,
                "n_holds": len(windows), "n_holds_active": len(active),
                "xcorr_peak_mean": float(means[0]),
                "xcorr_lag_mean_s": float(means[1]),
                "xcorr_lag0_mean": float(means[2]),
                "xcorr_whole_peak": whole_peak,
                "xcorr_whole_lag_s": whole_lag,
                "xcorr_whole_lag0": whole_lag0,
            })

    log.info("%s: xcorr -- active holds per MU %s", task_name,
             [row["n_holds_active"] for row in mu_rows])
    return hold_rows, mu_rows


# ---------------------------------------------------------------- overview table
def _mean_std(values):
    """Mean and sample std (ddof=1) of the finite values; nan where undefined."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    return float(np.mean(values)), std


def build_overview(hold_rows, mu_rows):
    """
    The overview table: one row per unit and task.

    A hold is valid for a unit when it was active and its xcorr came out
    finite -- the same holds that make up ``n_holds_active`` in ``mu_rows``.
    A unit without a valid hold keeps its row, with ``n_holds_used = 0``.
    """
    lag0_by_unit = {}
    for row in hold_rows:
        if row["active"] and np.isfinite(row["xcorr_lag0"]):
            lag0_by_unit.setdefault((row["task"], row["mu"]), []).append(
                row["xcorr_lag0"])

    overview = []
    for row in mu_rows:
        lag0 = lag0_by_unit.get((row["task"], row["mu"]), [])
        hold_mean, hold_std = _mean_std(lag0)
        overview.append({"patient_id": row["patient_id"], "visit": row["visit"],
                         "task": row["task"], "mu": row["mu"],
                         "n_holds_used": len(lag0),
                         "xcorr_lag0_hold_mean": hold_mean,
                         "xcorr_lag0_hold_std": hold_std,
                         "xcorr_global_lag0": row["xcorr_whole_lag0"]})
    return overview
