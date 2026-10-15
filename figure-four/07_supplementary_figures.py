"""
07_supplementary_figures.py
===========================

Step 7 of the Figure 4 pipeline: one supplementary figure per session (offline
decoding of the cued hand tasks, Supplementary Figs. 2b-7b).

    A  all recordings of the session, concatenated (dashed lines: recording
       boundaries): the separate rest recording and the task recordings. Cue,
       periods in which the cue exceeded 5 % of its maximum (closing, holding
       and opening; for the P02 pinch recording the corrected cue of step 01).
       Label, the step-01 attempt segments (filled: analysed, labelled with the
       instructed gesture; hatched: detected but excluded) and the window of
       the rest recording used as rest class (outline, step 02). HD-sEMG,
       N_TRACES consecutive analysed channels of one electrode grid after the
       band-pass and band-stop filters; shading, analysed attempts. Mean RMS,
       RMS envelope (window and stride of the features) averaged over all
       analysed channels
    B  confusion matrix of the pooled out-of-sample predictions (% of true class)
    C  accuracy of every held-out repetition per class; line, mean; dashed,
       chance level 1/K
    D  UMAP embedding (UMAP_PARAMS) of the z-scored RMS feature vectors of all
       frames, coloured by class; unsupervised, for illustration only
    E  out-of-sample prediction of every frame, frames joined by class as used
       for decoding: true class, predicted class and the predicted probability
       of the true class, p(true)

All numbers come from the outputs of steps 01-03; the raw recordings are read
again only for the traces of panel A. The UMAP embedding is deterministic
(fixed random_state, single-threaded) and is cached in data/<S>_umap.npy.

Input / Output
--------------
    in : data/labels/<S>.labels_auto_v2.json (step 01), data/<S>_dataset.json and
         data/<S>_rms_features.npz (step 02), data/<S>_loro.npz and
         results/loro_results.json (step 03), and the recording-day pickle and
         rest recording of every session for the traces of panel A (paths in
         sessions.local.json)
    out: results/supplementary/SuppFig_<S>.pdf / .png / .svg   (183 x 170 mm)
         data/<S>_umap.npy                                    (UMAP embedding)

The SVGs keep text as text, contain no clipping masks or <use> references and
name their groups (gid), so they can be edited in Illustrator or PowerPoint.
Panel A needs the raw recordings, which are not distributed with this
repository (identifiable human EMG of a vulnerable participant group); the data
are available from the authors on reasonable request.

Dependencies
------------
    numpy, scipy, matplotlib, umap-learn

Usage
-----
    uv run python figure-four/07_supplementary_figures.py [SESSION ...]   (default: all sessions)

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import gc
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import _common as C
import _emg as E
import _style as ns
from _svg import save_svg

ns.apply()

OUT_DIR = os.path.join(C.RESULTS_DIR, "supplementary")
G = ns.GESTURES
GCOL = [ns.GESTURE_COLORS[g] for g in G]
TITLE = {"P01-S1": "P01 | Session 1", "P01-S2": "P01 | Session 2"}
HEADLINE = {"A": "Recordings, cues and detected attempts", "B": "Confusion matrix",
            "C": "Accuracy per held-out repetition", "D": "RMS feature space (UMAP)",
            "E": "Frame-wise predictions"}
N_TRACES = 24           # consecutive analysed channels shown in panel A
GRID = 64               # channels per electrode grid
DECIM = 128             # samples per min/max bin of the vector EMG traces (62.5 ms at 2048 Hz)
CUE_THR = 0.05          # cue level that marks a cue as on (as in step 01)
UMAP_PARAMS = dict(n_neighbors=30, min_dist=0.3, random_state=0)
W_MM, H_MM = 183, 170   # Nature maximum figure area


with open(os.path.join(C.RESULTS_DIR, "loro_results.json"), encoding="utf-8") as _f:
    LRES = json.load(_f)


def runs(lbl):
    """(start index, length, value) of every run of identical values."""
    d = np.flatnonzero(np.diff(lbl)) + 1
    starts = np.r_[0, d]
    ends = np.r_[d, len(lbl)]
    return [(int(s), int(e - s), int(lbl[s])) for s, e in zip(starts, ends)]


def minmax(x, b):
    """Min/max decimation per bin: keeps the visual envelope of the EMG with few vector points."""
    nb = x.shape[1] // b
    seg = x[:, :nb * b].reshape(x.shape[0], nb, b)
    out = np.empty((x.shape[0], 2 * nb), np.float32)
    out[:, 0::2] = seg.min(2)
    out[:, 1::2] = seg.max(2)
    return out, np.repeat(np.arange(nb) * b + b / 2, 2)


def contiguous_channels(kept, n):
    """n consecutive analysed channels (0-indexed): the first such window inside
    one electrode grid, else the first window anywhere."""
    ks = set(int(c) for c in kept)
    starts = [a for a in range(int(kept.max()) - n + 2) if all(c in ks for c in range(a, a + n))]
    assert starts, f"no {n} consecutive analysed channels"
    inside = [a for a in starts if a // GRID == (a + n - 1) // GRID]
    a = (inside or starts)[0]
    return np.arange(a, a + n)


# ------------------------------------------------------------------ data
def umap_embedding(session, X):
    path = os.path.join(C.DATA_DIR, f"{session}_umap.npy")
    if os.path.isfile(path):
        emb = np.load(path)
        if emb.shape == (len(X), 2):
            return emb
    import umap
    Xz = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    emb = umap.UMAP(**UMAP_PARAMS).fit_transform(Xz)
    np.save(path, emb)
    return emb


def prepare(cfg, session):
    """Everything the five panels of one session need."""
    labels = C.load_labels(session)
    with open(os.path.join(C.DATA_DIR, f"{session}_dataset.json"), encoding="utf-8") as f:
        info = json.load(f)
    kept = np.array(info["kept_channels_1idx"]) - 1
    assert list(kept + 1) == labels["channels_kept_1idx"], f"{session}: channels of steps 01 and 02 differ"
    show = contiguous_channels(kept, N_TRACES)
    rows = np.searchsorted(kept, show)

    day = C.load_pickle(C.session_path(cfg, session, "recording"))
    rest_raw = C.load_pickle(C.session_path(cfg, session, "rest_recording"))
    tasks = C.analysed_tasks(day, session)
    fs = int(day["tasks"][tasks[0]]["fsamp"])
    assert int(rest_raw["device_information"]["sampling_frequency"]) == fs, f"{session}: rest sampling rate"
    items = [("Rest recording", C.rest_signal(rest_raw, session), None)]
    del rest_raw
    items += [(t.split("_")[0], day["tasks"][t]["emg"], t) for t in tasks]

    D = dict(session=session, kept=kept, show=show, traces=[], trace_t=[], env=[], env_t=[], cues=[], segs=[],
             bounds=[0.0], names=[], rest_win=info["rest_window_s"])
    off = 0.0
    for name, emg, task in items:
        xf = E.apply_filters(emg[kept].astype(np.float32), fs, E.BANDPASS_HZ, E.BANDSTOP_HZ, E.BUTTER_ORDER)
        env = E.rms_moving_window(xf, E.WINDOW_SAMPLES, E.STRIDE_SAMPLES).mean(0)
        D["env"].append(env)
        D["env_t"].append((np.arange(len(env)) * E.STRIDE_SAMPLES + E.WINDOW_SAMPLES / 2) / fs + off)
        tr, tt = minmax(xf[rows], DECIM)
        D["traces"].append(tr)
        D["trace_t"].append(tt / fs + off)
        dur = emg.shape[1] / fs
        del xf
        if task is not None:
            lt = labels["tasks"][task]
            assert abs(lt["duration_s"] - dur) < 1e-9, f"{session} {task}: labels belong to another recording"
            # cue-on periods as found by step 01 (corrected cue where needed); a cue that started before
            # the recording or was still on at its end is drawn up to the recording edge
            D["cues"] += [(max(c["onset_s"], 0.0) + off, min(c["opening_end_s"], dur) + off) for c in lt["cues"]]
            D["segs"] += [(sg["start_s"] + off, sg["end_s"] + off,
                           C.JSON_TO_GESTURE[sg["gesture"]] if sg["quality"] == "good" else "excluded")
                          for sg in lt["segments"]]
        off += dur
        D["bounds"].append(off)
        D["names"].append(name)
    del day, items
    gc.collect()
    for k in ("traces", "trace_t", "env", "env_t"):
        D[k] = np.concatenate(D[k], axis=-1)

    # decoding results (steps 02 and 03)
    d = E.load_rms_dataset(os.path.join(C.DATA_DIR, f"{session}_rms_features.npz"))
    keep = E.keep_mask_from_bad_spec(d["kept_channels_1idx"], d["bad_channels"])
    fm = d["region_ids_per_frame"] >= 0
    # the feature file holds only analysed channels and labelled frames, so its matrix is the classifier
    # input of step 03. It is used as stored (frames x channels view): the float32 z-scoring rounds
    # differently on a reordered copy, and UMAP would amplify that into a different embedding.
    assert keep.all() and fm.all(), f"{session}: feature file contains unused channels or frames"
    X = d["features"].T
    z = np.load(os.path.join(C.DATA_DIR, f"{session}_loro.npz"))
    D["loro"] = {k: z[k] for k in ("y", "oof", "proba")}
    D["rate"] = float(z["feature_rate"])
    assert np.array_equal(D["loro"]["y"], d["labels"]), f"{session}: LORO frames do not match the features"
    D["emb"] = umap_embedding(session, X)
    D["res"] = LRES[session]
    cm = np.array(D["res"]["confusion_matrix"], float)
    D["cm"], D["present"] = cm, [k for k in range(4) if cm[k].sum() > 0]
    return D


# ------------------------------------------------------------------ panels
def panel_a(fig, axes, D):
    axCue, axLab, axEMG, axEnv = axes
    for ax, gid in zip(axes, ("panel_a_cue", "panel_a_labels", "panel_a_emg", "panel_a_mean_rms")):
        ax.set_gid(gid)
    for ax in (axCue, axLab, axEMG):
        ax.sharex(axEnv)
    for a, b in D["cues"]:
        axCue.axvspan(a, b, color=ns.GREY, lw=0, gid="a_cue_window")
    for a, b, g in D["segs"]:
        if g == "excluded":
            axLab.axvspan(a, b, facecolor="white", edgecolor=ns.GREY, hatch="//////", lw=0.3, gid="a_label_excluded")
        else:
            axLab.axvspan(a, b, color=ns.GESTURE_COLORS[g], lw=0, gid=f"a_label_{g.replace(' ', '_')}")
    axLab.axvspan(*D["rest_win"], ymin=0.1, ymax=0.9, facecolor="none", edgecolor=ns.GESTURE_COLORS["Rest"],
                  lw=0.9, gid="a_rest_window")
    for ax, name in ((axCue, "Cue"), (axLab, "Label")):
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([name])
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    tr, show = D["traces"], D["show"]
    spread = np.percentile(np.abs(tr), 99.5) * 2.0
    for a, b, g in D["segs"]:
        if g != "excluded":
            axEMG.axvspan(a, b, color=ns.GESTURE_COLORS[g], alpha=0.12, lw=0, gid=f"a_emg_shade_{g.replace(' ', '_')}")
    for i in range(len(show)):
        axEMG.plot(D["trace_t"], tr[i] + i * spread, color=ns.DARK_BLUE, lw=0.2, gid=f"a_emg_ch{show[i] + 1}")
    axEMG.set_ylim(-spread, len(show) * spread)
    axEMG.set_yticks([0, (len(show) - 1) * spread])           # consecutive channels: first and last suffice
    axEMG.set_yticklabels([f"ch {show[0] + 1}", f"ch {show[-1] + 1}"])
    axEMG.set_ylabel("HD-sEMG (#)")
    axEMG.tick_params(axis="x", labelbottom=False, length=0)
    axEMG.spines["bottom"].set_visible(False)
    for b0, b1, nm in zip(D["bounds"][:-1], D["bounds"][1:], D["names"]):
        axCue.text((b0 + b1) / 2, 1.5, nm, transform=axCue.get_xaxis_transform(), ha="center", va="bottom",
                   fontsize=ns.FS["title"], fontweight="bold", gid="a_task_names")
    for b in D["bounds"][1:-1]:
        for ax in axes:
            ax.axvline(b, color=ns.INK, lw=0.5, ls=(0, (3, 2)), gid="a_task_boundary")
    axEnv.plot(D["env_t"], D["env"], color=ns.DARK_BLUE, lw=0.5, gid="a_mean_rms_line")
    axEnv.set_ylim(0, np.percentile(D["env"], 99.8) * 1.15)
    axEnv.set_ylabel("Mean RMS\n(a.u.)")
    axEnv.set_xlim(0, D["bounds"][-1])
    axEnv.set_xlabel("Time (s)")


def legend_a(fig):
    leg = [Patch(color=ns.GESTURE_COLORS[g], label=g) for g in G[1:]] + \
          [Patch(facecolor="none", edgecolor=ns.GESTURE_COLORS["Rest"], lw=0.9, label="Rest window"),
           Patch(facecolor="white", edgecolor=ns.GREY, hatch="//////", lw=0.3, label="Excluded")]
    fig.legend(handles=leg, loc="upper right", bbox_to_anchor=(181 / W_MM, 1 - 5.6 / H_MM), ncol=len(leg),
               handlelength=1.0, handletextpad=0.3, columnspacing=0.8, borderaxespad=0).set_gid("a_legend")


def panel_b(fig, ax, cax, D):
    ax.set_gid("panel_b")
    cm, present = D["cm"], D["present"]
    rs = cm.sum(1, keepdims=True)
    pct = (100 * cm / np.where(rs == 0, 1, rs))[np.ix_(present, present)]
    n = len(present)
    qm = ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, pct, cmap=ns.SEQ, vmin=0, vmax=100,
                       edgecolors="none", linewidth=0)
    qm.set_gid("b_cells")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=ns.FS["annot"],
                    fontweight="bold" if i == j else "normal", color="white" if pct[i, j] > 55 else ns.INK,
                    gid="b_cell_values")
    ax.set_xticks(range(n))
    ax.set_xticklabels([ns.SHORT[k] for k in present], rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([ns.SHORT[k] for k in present])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(qm, cax=cax, ticks=[0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("% of true class")
    cax.set_gid("b_colorbar")


def panel_c(fig, ax, D):
    ax.set_gid("panel_c")
    present, r = D["present"], D["res"]
    rng = np.random.default_rng(0)                            # jitter of the dots only
    for xi, k in enumerate(present):
        v = np.array([a for u, a in r["unit_accuracy"].items() if r["unit_class"][u] == k])
        assert len(v) == r["n_units_per_class"][G[k]]
        ax.scatter(xi + rng.uniform(-0.12, 0.12, len(v)), v, s=5, color=GCOL[k], lw=0, alpha=0.85,
                   gid=f"c_repetitions_{ns.SHORT[k]}")
        ax.plot([xi - 0.25, xi + 0.25], [v.mean()] * 2, color=ns.INK, lw=0.8, gid=f"c_mean_{ns.SHORT[k]}")
    ax.axhline(1 / len(present), color=ns.GREY, lw=0.8, ls=(0, (2, 1)), gid="c_chance")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([ns.SHORT[k] for k in present], rotation=30, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylabel("Accuracy")


def panel_d(fig, ax, D):
    ax.set_gid("panel_d")
    emb, y = D["emb"], D["loro"]["y"]
    for k in range(4):                                        # one editable group per class
        m = y == k
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=0.6, color=GCOL[k], lw=0, gid=f"d_points_{ns.SHORT[k]}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def panel_e(fig, axes, D):
    axGT, axPR, axP = axes
    for ax, gid in zip(axes, ("panel_e_true", "panel_e_predicted", "panel_e_p_true")):
        ax.set_gid(gid)
    z, rate = D["loro"], D["rate"]
    for ax, lbl, nm in ((axGT, z["y"], "true"), (axPR, z["oof"], "pred")):
        for k in range(4):                                    # one rectangle per run, grouped by class
            bars = [(s / rate, n / rate) for s, n, c in runs(np.asarray(lbl)) if c == k]
            if bars:
                ax.broken_barh(bars, (0, 1), facecolors=GCOL[k], edgecolor="none", lw=0, gid=f"e_{nm}_{ns.SHORT[k]}")
        ax.set_ylim(0, 1)
    p_true = z["proba"][np.arange(len(z["y"])), z["y"]]
    axP.plot(np.arange(len(z["y"])) / rate, p_true, color=ns.DARK_BLUE, lw=0.4, gid="e_p_true_line")
    axP.set_ylim(0, 1.02)
    axP.set_yticks([0, 1])
    for ax in axes:
        ax.set_xlim(0, len(z["y"]) / rate)
    axP.set_ylabel("p(true)")
    axP.set_xlabel("Time (s)")
    for ax, nm in ((axGT, "True"), (axPR, "Predicted")):
        ax.set_yticks([0.5])
        ax.set_yticklabels([nm])
        ax.tick_params(length=0, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)


# ------------------------------------------------------------------ layout (mm from the top left)
def axes_mm(fig, x0, y0, w, h):
    return fig.add_axes([x0 / W_MM, 1 - (y0 + h) / H_MM, w / W_MM, h / H_MM])


def letter(fig, x, y, s):
    """Panel letter and its bold headline, top-aligned at y mm."""
    fig.text(x / W_MM, 1 - y / H_MM, s, fontsize=ns.FS["panel"], fontweight="bold", va="top", gid=f"label_{s}")
    fig.text((x + 4) / W_MM, 1 - (y + 0.6) / H_MM, HEADLINE[s], fontsize=ns.FS["title"], fontweight="bold",
             va="top", gid=f"headline_{s}")


def make(cfg, session):
    D = prepare(cfg, session)
    fig = plt.figure(figsize=(W_MM * ns.MM, H_MM * ns.MM))
    fig.text(1 / W_MM, 1 - 0.3 / H_MM, TITLE.get(session, session), fontsize=ns.FS["title"], fontweight="bold",
             va="top", gid="title")
    L, R = 14, 181                                            # left / right edge of the plots
    letter(fig, 1, 5.5, "A")
    A = [axes_mm(fig, L, 14.0, R - L, 2.6), axes_mm(fig, L, 18.0, R - L, 2.6),
         axes_mm(fig, L, 22.0, R - L, 39.5), axes_mm(fig, L, 63.0, R - L, 13.5)]
    panel_a(fig, A, D)
    legend_a(fig)
    for x, s in ((1, "B"), (65, "C"), (130.5, "D")):
        letter(fig, x, 86.8, s)
    n = len(D["present"])
    side = min(34.0, 8.8 * n)                                 # square matrix, at most 34 mm
    panel_b(fig, axes_mm(fig, L, 92.0, side, side), axes_mm(fig, L + side + 2.0, 92.0, 1.5, side), D)
    panel_c(fig, axes_mm(fig, 79, 92.0, 44, 34), D)
    panel_d(fig, axes_mm(fig, 140, 92.0, R - 140, 34), D)
    letter(fig, 1, 138.5, "E")
    panel_e(fig, [axes_mm(fig, L, 144.5, R - L, 3.2), axes_mm(fig, L, 148.7, R - L, 3.2),
                  axes_mm(fig, L, 153.2, R - L, 9.0)], D)
    out = os.path.join(OUT_DIR, f"SuppFig_{session}")
    fig.savefig(f"{out}.pdf", metadata={"CreationDate": None})
    fig.savefig(f"{out}.png", dpi=600)
    save_svg(fig, f"{out}.svg")
    plt.close(fig)
    r = D["res"]
    print(f"{session}: ch {D['show'][0] + 1}-{D['show'][-1] + 1} of {len(D['kept'])} analysed, "
          f"rest window {D['rest_win'][1] - D['rest_win'][0]:.1f} s, {len(D['loro']['y'])} frames, "
          f"acc {r['accuracy']:.2f}, F1 {r['f1_macro']:.2f} -> {out}", flush=True)


def main():
    sessions = sys.argv[1:] or C.SESSIONS
    unknown = [s for s in sessions if s not in C.SESSIONS]
    if unknown:
        raise SystemExit(f"unknown sessions {unknown}; known: {C.SESSIONS}")
    cfg = C.load_config()
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in sessions:
        make(cfg, s)
        gc.collect()


if __name__ == "__main__":
    main()
