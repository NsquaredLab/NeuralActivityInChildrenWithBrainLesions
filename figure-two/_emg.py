"""
_emg.py
=======

Signal building blocks of the Figure 2 pipeline:

    - ``compute_rms``            mean RMS envelope over all channels (moving
                                 average of the squared signal)
    - ``bandpass_notch_grid``    50 Hz notch + 20-500 Hz 4th-order Butterworth
                                 band-pass (zero phase) of a grid
    - ``hann_discharge_rate``    discharge rate = spike train convolved with a
                                 Hann window (pps)
    - ``extract_trigger`` / ``hold_windows``
                                 active runs of a trigger / of the 0/1 labels
                                 -> hold periods
    - ``cut_muap_waveforms`` / ``p2p_map`` / ``spatial_correlation`` /
      ``fill_heatmap_nans`` / ``add_extra_discards``
                                 MUAPs on the electrode grid and their
                                 peak-to-peak maps

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

import _common as C

log = C.get_logger(__name__)

# Default length of the Hann window of the discharge rate, in seconds. The
# scripts pass their own window.
HANN_RATE_WINDOW = 2.0


# ---------------------------------------------------------------- envelope / filters
def compute_rms(signal, fs, window):
    """
    Mean RMS envelope across all channels of a ``(channels, samples)`` EMG array.

    A moving average of the squared signal (length ``window`` seconds) is taken
    per channel -- mirroring MATLAB's ``movmean`` -- and the per-channel RMS
    traces are averaged into one 1-D reference signal.

    Parameters
    ----------
    signal : ndarray
        EMG, ``(channels, samples)`` (a 1-D signal is treated as one channel;
        an accidentally transposed array is auto-corrected).
    fs : float
        Sampling frequency in Hz.
    window : float
        Moving-average window length in **seconds**.

    Returns
    -------
    ndarray, shape (n_samples,)
        Mean RMS envelope.
    """
    window_size = int(window * fs)
    signal = np.atleast_2d(signal)
    # Auto-orient so rows are channels (channels are expected to be the smaller dim).
    if signal.shape[0] > signal.shape[1]:
        signal = signal.T

    num_channels, num_samples = signal.shape
    rms_signal = np.zeros((num_channels, num_samples))
    for ch in range(num_channels):
        squared = signal[ch, :] ** 2
        # Fast moving average via cumulative sum.
        cumsum = np.cumsum(np.insert(squared, 0, 0))
        mov_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        # Pad back to the original length (centered), repeating the edge values.
        pad_left = window_size // 2
        pad_right = num_samples - len(mov_avg) - pad_left
        mov_avg = np.pad(mov_avg, (pad_left, pad_right), mode="edge")
        rms_signal[ch, :] = np.sqrt(mov_avg)

    return np.mean(rms_signal, axis=0)


def bandpass_notch_grid(sig, fs, low=20.0, high=500.0, notch_freq=50.0,
                        notch_q=30.0):
    """
    The study's EMG cleaning for a whole grid: a 50 Hz notch followed by a
    20-500 Hz 4th-order Butterworth band-pass, both zero-phase.

    Parameters
    ----------
    sig : ndarray, shape (rows, cols, samples)
        EMG in grid layout; electrodes that are NaN (missing / gap columns)
        stay NaN.
    fs : float
        Sampling frequency in Hz.
    low, high : float, optional
        Band-pass corner frequencies in Hz (default 20 / 500).
    notch_freq : float, optional
        Notch centre frequency in Hz (default 50, mains hum).
    notch_q : float, optional
        Notch quality factor (default 30).

    Returns
    -------
    ndarray, same shape as ``sig``
    """
    b_notch, a_notch = iirnotch(notch_freq, notch_q, fs)
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    out = np.full_like(sig, np.nan)
    ok = ~np.isnan(sig[..., 0])
    out[ok] = sosfiltfilt(sos, filtfilt(b_notch, a_notch, sig[ok], axis=-1),
                          axis=-1)
    return out


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


# ---------------------------------------------------------------- hold periods
def normalize_trigger(trigger):
    """Min-max scale a trigger trace to 0..1 (all zeros if it is flat)."""
    trigger = np.asarray(trigger, dtype=float).ravel()
    t_min, t_max = np.min(trigger), np.max(trigger)
    if t_max - t_min <= 0:
        return np.zeros_like(trigger)
    return (trigger - t_min) / (t_max - t_min)


def extract_trigger(trigger, threshold=C.REST_THRESHOLD, normalize=True):
    """
    Find the start/stop sample indices of every contiguous run where ``trigger``
    is above ``threshold``.

    Parameters
    ----------
    trigger : 1-D ndarray
        Ground-truth trace (high during holds).
    threshold : float, optional
        Activation threshold as a fraction of the trace's own range (default
        ``_common.REST_THRESHOLD``).
    normalize : bool, optional
        Min-max scale the trace to 0..1 before thresholding (default ``True``).
        Set to ``False`` to treat ``threshold`` as an absolute level.

    Returns
    -------
    start, stop : ndarray of int
        Indices of the first and last active sample of each run.
    """
    trigger = normalize_trigger(trigger) if normalize \
        else np.asarray(trigger, dtype=float).ravel()
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


# ---------------------------------------------------------------- motor-unit maps
def cut_muap_waveforms(sig, pulses, half):
    """
    Cut the window around every discharge out of every electrode of a grid.

    Parameters
    ----------
    sig : ndarray, shape (rows, cols, samples)
        EMG in grid layout (NaN for missing electrodes).
    pulses : array of int
        Discharge sample indices, 0-based. Discharges without a full window
        inside the signal are dropped.
    half : int
        Half window in samples; the window is ``2 * half + 1`` long, centred.

    Returns
    -------
    waveforms : ndarray, shape (n_spikes, rows, cols, 2 * half + 1)
    pulses : ndarray
        The discharges that were kept.
    """
    n = sig.shape[-1]
    pulses = np.asarray(pulses, dtype=int).ravel()
    p = pulses[(pulses - half >= 0) & (pulses + half < n)]
    idx = p[:, None] + np.arange(-half, half + 1)[None, :]
    w = sig[:, :, idx]                 # rows, cols, spikes, time
    return np.moveaxis(w, 2, 0), p


def p2p_map(mean_muap, discard):
    """
    Peak-to-peak amplitude of a mean MUAP on every electrode; discarded or
    empty electrodes (``discard`` True) become NaN.
    """
    amp = np.ptp(mean_muap, axis=-1)
    return np.where(discard, np.nan, amp)


def spatial_correlation(map_a, map_b):
    """
    Pearson r of two P2P maps over the electrodes valid in both.

    Pass the maps *before* :func:`fill_heatmap_nans`: a filled electrode
    repeats its neighbour and would count twice. NaN with fewer than three
    common electrodes or a flat map.
    """
    ok = ~(np.isnan(map_a) | np.isnan(map_b))
    if ok.sum() < 3:
        return np.nan
    a = map_a[ok] - map_a[ok].mean()
    b = map_b[ok] - map_b[ok].mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else np.nan


def fill_heatmap_nans(amp_map, discardChannelsVec, grids, return_source=False):
    """
    Fill the NaNs of discarded channels *inside* the physical grids with a
    neighbouring channel's value, so the heatmap has no holes within a grid.

    The gap columns *between* grids are intentionally left as NaN.

    Parameters
    ----------
    amp_map : ndarray, shape (rows, cols)
        Amplitude map with NaNs at discarded/empty channels.
    discardChannelsVec : ndarray, shape (rows, cols)
        1 = discarded channel.
    grids : list of dict
        Electrode-grid layout (``_common.GRIDS``); 1-indexed ``rows``/``cols``.
    return_source : bool, optional
        If True, also return, for every filled channel, the ``(row, col)`` of
        the neighbour whose value was used.

    Returns
    -------
    ndarray, shape (rows, cols)
        Copy of ``amp_map`` with in-grid holes filled.
    """
    m, n = amp_map.shape

    # Mask of positions that belong to a physical grid (1) vs. gap columns (0).
    is_grid = np.zeros(amp_map.shape)
    for g in grids:
        rows = np.array(g["rows"]) - 1
        cols = np.array(g["cols"]) - 1
        is_grid[np.ix_(rows, cols)] = 1

    new_map = amp_map.copy()
    source = [[None for _ in range(n)] for _ in range(m)]
    n_filled, unfillable = 0, []
    for i in range(m):
        for j in range(n):
            if discardChannelsVec[i][j] == 1 and is_grid[i][j] == 1:
                # Prefer the left neighbour, then right, then two-to-the-left.
                if j > 0 and not np.isnan(amp_map[i][j - 1]):
                    new_map[i, j] = amp_map[i, j - 1]
                    source[i][j] = (i, j - 1)
                elif j == 0 and not np.isnan(amp_map[i][j + 1]):
                    new_map[i, j] = amp_map[i, j + 1]
                    source[i][j] = (i, j + 1)
                elif j > 0 and j + 1 < n and not np.isnan(amp_map[i][j + 1]):
                    new_map[i, j] = amp_map[i, j + 1]
                    source[i][j] = (i, j + 1)
                elif j > 1 and not np.isnan(amp_map[i][j - 2]):
                    new_map[i, j] = amp_map[i, j - 2]
                    source[i][j] = (i, j - 2)
                else:
                    unfillable.append((i, j))
                    continue
                n_filled += 1

    log.debug("Heatmap NaN-fill: %d discarded channels interpolated from a "
              "neighbour", n_filled)
    if unfillable:
        log.warning("Could not fill %d discarded channel(s) -- no valid "
                    "neighbour: %s", len(unfillable), unfillable)

    if return_source:
        return new_map, source
    return new_map


def add_extra_discards(discard, attribute):
    """
    Mark the channels of ``_common.EXTRA_DISCARDED_CHANNELS[attribute]`` as
    discarded, on top of the file's own ``discardChannelsVec``. Returns a
    copy with the dtype of ``discard``.
    """
    out = np.array(discard, copy=True)
    extra = C.EXTRA_DISCARDED_CHANNELS.get(attribute, ())
    for i, j in extra:
        out[i, j] = 1
    if extra:
        log.info("%s: %d additional channel(s) discarded: %s", attribute,
                 len(extra), [(i + 1, j + 1) for i, j in extra])
    return out
