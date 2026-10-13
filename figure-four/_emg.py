"""
_emg.py
=======

EMG building blocks of the Figure 4 pipeline:

    - ``flatten_biosignal``      recording frames -> (channels x samples), EMG channels only
    - ``parse_channel_spec`` / ``channels_to_spec`` / ``keep_mask_from_bad_spec``
                                 channel lists as "1-24; 51" strings (1-indexed)
    - ``apply_filters`` / ``rms_moving_window``
                                 2nd-order Butterworth band-pass 20-500 Hz and
                                 band-stop 45-55 Hz (zero phase), RMS over
                                 WINDOW_SAMPLES samples with a STRIDE_SAMPLES stride
    - ``write_rms_features`` / ``load_rms_dataset``
                                 the per-session RMS feature file (step 02) and its
                                 loader (steps 03 and 07)
    - ``CATBOOST_PARAMS``        classifier settings (CPU and a fixed seed, so every
                                 fit is bit-for-bit reproducible)

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

GESTURES = ["Rest", "Power Grasp", "Pinch", "Tripod Pinch"]   # canonical class order (label index)
N_BIOSIGNAL_CHANNELS = 384      # EMG channels of the amplifier; the auxiliary channels follow them

WINDOW_SAMPLES = 360            # 176 ms at 2048 Hz
STRIDE_SAMPLES = 64             # 31 ms at 2048 Hz
BANDPASS_HZ = (20.0, 500.0)
BANDSTOP_HZ = (45.0, 55.0)
BUTTER_ORDER = 2

# CPU + a fixed random_seed makes CatBoost bit-for-bit reproducible. GPU training
# is non-deterministic (floating-point reduction ordering) even with a seed set.
CATBOOST_PARAMS = {
    "iterations": 1000,
    "l2_leaf_reg": 5,
    "border_count": 254,
    "task_type": "CPU",
    "random_seed": 42,
    "train_dir": None,
    "verbose": False,
    "allow_writing_files": False,
}


# ---------------------------------------------------------------- recordings and channels
def flatten_biosignal(biosignal_3d: np.ndarray) -> np.ndarray:
    """(channels, samples_per_frame, frames) -> (channels, samples), without the
    auxiliary channels at the tail (float32)."""
    channels = biosignal_3d.shape[0]
    flat = biosignal_3d.transpose(0, 2, 1).reshape(channels, -1)
    return flat[:N_BIOSIGNAL_CHANNELS].astype(np.float32)


def parse_channel_spec(spec: str, n_channels: int) -> set[int]:
    """0-indexed channels of a spec such as "1-24; 51" (1-indexed, ranges
    inclusive, separators ; or ,). An empty spec means all channels."""
    if not spec.strip():
        return set(range(n_channels))
    kept: set[int] = set()
    for part in spec.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip()) - 1
            end = int(b.strip())
            kept.update(range(max(0, start), min(n_channels, end)))
        else:
            idx = int(part) - 1
            if 0 <= idx < n_channels:
                kept.add(idx)
    return kept


def channels_to_spec(channels: set[int], n_channels: int) -> str:
    """Inverse of ``parse_channel_spec`` (0-indexed channels -> 1-indexed spec)."""
    if not channels or len(channels) == n_channels:
        return ""
    sorted_ch = sorted(channels)
    parts: list[str] = []
    run_start = prev = sorted_ch[0]
    for c in sorted_ch[1:]:
        if c == prev + 1:
            prev = c
            continue
        parts.append(f"{run_start + 1}" if run_start == prev else f"{run_start + 1}-{prev + 1}")
        run_start = prev = c
    parts.append(f"{run_start + 1}" if run_start == prev else f"{run_start + 1}-{prev + 1}")
    return "; ".join(parts)


def keep_mask_from_bad_spec(channels_1idx: np.ndarray, bad_spec: str) -> np.ndarray:
    """Boolean mask over ``channels_1idx`` (1-indexed amplifier channels): False
    for the channels in ``bad_spec``."""
    if not bad_spec.strip():
        return np.ones(len(channels_1idx), dtype=bool)
    bad_1idx = {g + 1 for g in parse_channel_spec(bad_spec, N_BIOSIGNAL_CHANNELS)}
    return np.array([int(c) not in bad_1idx for c in channels_1idx], dtype=bool)


# ---------------------------------------------------------------- filters and RMS
def apply_filters(signal: np.ndarray, fs: int, bandpass_hz: tuple[float, float],
                  bandstop_hz: tuple[float, float], order: int) -> np.ndarray:
    """Zero-phase Butterworth band-pass, then band-stop, along the last axis (float32)."""
    sos_bp = butter(order, bandpass_hz, btype="bandpass", fs=fs, output="sos")
    out = sosfiltfilt(sos_bp, signal, axis=-1)
    sos_bs = butter(order, bandstop_hz, btype="bandstop", fs=fs, output="sos")
    out = sosfiltfilt(sos_bs, out, axis=-1)
    return out.astype(np.float32)


def rms_moving_window(signal: np.ndarray, window: int, stride: int) -> np.ndarray:
    """signal: (n_channels, n_samples). Returns (n_channels, n_features) with
    n_features = max(0, (n_samples - window) // stride + 1)."""
    if signal.shape[1] < window:
        return np.zeros((signal.shape[0], 0), dtype=np.float32)
    windows = sliding_window_view(signal, window_shape=window, axis=-1)
    windows = windows[:, ::stride]  # (n_channels, n_features, window)
    return np.sqrt((windows ** 2).mean(axis=-1)).astype(np.float32)


# ---------------------------------------------------------------- RMS feature file
def write_rms_features(path: str, signal: np.ndarray, fs: int, region_starts: np.ndarray,
                       region_ends: np.ndarray, region_labels: list[str], source_files: list[str],
                       source_span_idx: list[int], bad_spec: str) -> None:
    """RMS features of a labelled signal (all amplifier channels x samples, the
    labelled regions joined): the channels in ``bad_spec`` are dropped, the rest
    is filtered as a whole, and the RMS is computed within each region (no window
    crosses a region boundary). Regions are written grouped by class in GESTURES
    order (stable, so chronological within a class); regions shorter than one
    window are dropped."""
    n_ch = signal.shape[0]
    channels_1idx = np.arange(1, n_ch + 1).astype(np.int32)
    bad_mask = keep_mask_from_bad_spec(channels_1idx, bad_spec)
    kept_channels_1idx = channels_1idx[bad_mask]
    print(f"  channels: {bad_mask.sum()}/{len(bad_mask)} kept")
    signal_f = apply_filters(signal[bad_mask].astype(np.float32), fs, BANDPASS_HZ, BANDSTOP_HZ, BUTTER_ORDER)

    label_to_idx = {g: i for i, g in enumerate(GESTURES)}
    region_order = sorted((i for i in range(len(region_labels)) if region_labels[i] in label_to_idx),
                          key=lambda i: label_to_idx[region_labels[i]])
    feature_blocks, label_blocks = [], []
    new_starts, new_ends, new_labels, new_files, new_span_idx = [], [], [], [], []
    cursor = 0
    for i in region_order:
        feats = rms_moving_window(signal_f[:, int(region_starts[i]):int(region_ends[i])],
                                  WINDOW_SAMPLES, STRIDE_SAMPLES)
        n = feats.shape[1]
        if n == 0:
            continue
        feature_blocks.append(feats)
        label_blocks.append(np.full(n, label_to_idx[region_labels[i]], dtype=np.int8))
        new_starts.append(cursor)
        new_ends.append(cursor + n)
        new_labels.append(region_labels[i])
        new_files.append(source_files[i])
        new_span_idx.append(source_span_idx[i])
        cursor += n

    features = np.concatenate(feature_blocks, axis=1)
    feature_rate = fs / STRIDE_SAMPLES
    print(f"  -> features {features.shape}, {features.shape[1] / feature_rate:.1f}s @ {feature_rate:.1f} Hz, "
          f"{len(new_starts)} regions")
    np.savez_compressed(
        path,
        features=features,
        labels=np.concatenate(label_blocks),
        fs=np.int32(fs),
        samples_per_frame=np.int32(STRIDE_SAMPLES),
        feature_rate=np.float64(feature_rate),
        feature_window_samples=np.int32(WINDOW_SAMPLES),
        kept_channels_1idx=kept_channels_1idx,
        channel_spec=f"1-{n_ch}",
        bad_channels=bad_spec,
        region_starts=np.array(new_starts, dtype=np.int64),
        region_ends=np.array(new_ends, dtype=np.int64),
        region_labels=np.array(new_labels),
        source_files=np.array(new_files),
        source_span_idx=np.array(new_span_idx, dtype=np.int32),
        bandpass_hz=np.array(BANDPASS_HZ, dtype=np.float64),
        bandstop_hz=np.array(BANDSTOP_HZ, dtype=np.float64),
        butter_order=np.int32(BUTTER_ORDER),
    )
    print(f"  saved {path}")


def load_rms_dataset(path: str) -> dict:
    """Load an RMS feature file. The per-frame class index and region id are
    rebuilt from the region spans and labels; frames outside any region get -1."""
    npz = np.load(path, allow_pickle=False)
    n_frames = int(npz["features"].shape[1])
    region_starts = npz["region_starts"].astype(np.int64)
    region_ends = npz["region_ends"].astype(np.int64)
    region_labels = np.array([str(lbl) for lbl in npz["region_labels"]])
    label_to_idx = {g: i for i, g in enumerate(GESTURES)}
    region_ids_per_frame = np.full(n_frames, -1, dtype=np.int64)
    labels = np.full(n_frames, -1, dtype=np.int64)
    for region_idx, (s, e, lbl) in enumerate(zip(region_starts, region_ends, region_labels)):
        if lbl not in label_to_idx:
            continue
        region_ids_per_frame[int(s):int(e)] = region_idx
        labels[int(s):int(e)] = label_to_idx[lbl]
    return {
        "features": npz["features"],  # (n_channels, n_frames)
        "labels": labels,
        "fs": int(npz["fs"]),
        "feature_rate_hz": float(npz["feature_rate"]),
        "kept_channels_1idx": npz["kept_channels_1idx"].astype(np.int64),
        "channel_spec": str(npz["channel_spec"]),
        "bad_channels": str(npz["bad_channels"]),
        "region_starts": region_starts,
        "region_ends": region_ends,
        "region_labels": region_labels,
        "region_ids_per_frame": region_ids_per_frame,
        "n_frames": n_frames,
    }
