"""
05_figure.py
============

Step 5 of the Figure 4 pipeline: draw Figure 4 (offline decoding of cued hand
tasks from forearm HD-sEMG, leave-one-repetition-out).

    A  Analysis workflow, shown for P03: one power-grasp recording with its cue,
       the automatically labelled attempts and 8 example HD-sEMG channels
       (left); the labelled dataset (attempts of all tasks and the rest window,
       concatenated by class) as mean RMS feature (right, top); the
       leave-one-repetition-out scheme with one example run per class (right,
       bottom)
    B  accuracy and macro F1 per session with 95 % intervals, single held-out
       repetitions and chance levels; mean ± s.d. (sample s.d.) across sessions
    C  confusion matrix (% of true class), averaged across sessions
    D  one-vs-rest AUROC per task: sessions with 95 % intervals, filled when the
       Holm-corrected permutation test is significant (P < 0.05)

Input / Output
--------------
    in : data/labels/P03.labels_auto_v2.json (step 01), data/P03_rms_features.npz
         (step 02), data/P03_loro.npz + results/loro_results.json (step 03),
         results/loro_stats.json (step 04), and the P03 recording-day pickle
         for the raw traces of panel A (path in sessions.local.json)
    out: results/figure4/Figure4.pdf / .png / .svg
         results/figure4/panels/panel_a.svg ... panel_d.svg

The SVGs keep text as text, contain no clipping masks or <use> references and
name their groups (gid), so they can be laid out and edited in Illustrator or
PowerPoint. Panel A needs the raw recording, which is not distributed with this
repository (identifiable human EMG of a vulnerable participant group); the data
are available from the authors on reasonable request.

Dependencies
------------
    numpy, scipy, matplotlib

Usage
-----
    uv run python figure-four/05_figure.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

import _common as C
import _emg as E
import _style as ns
from _svg import save_svg

ns.apply()

OUT_DIR = os.path.join(C.RESULTS_DIR, "figure4")
G, S = ns.GESTURES, C.SESSIONS
EX_SESSION, EX_TASK = "P03", "Power Grasp_trial1_left"


def light(c, f=0.28):
    """Opaque tint of a colour (no transparency, so it prints and edits cleanly)."""
    r, g, b = to_rgb(c)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def dark(c, f=0.3):
    """Darker shade for text in a palette colour (keeps light colours readable)."""
    r, g, b = to_rgb(c)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def runs_idx(mask):
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# ------------------------------------------------------------------ performance metrics
with open(os.path.join(C.RESULTS_DIR, "loro_results.json"), encoding="utf-8") as _f:
    lres = json.load(_f)
M = {}
for s in S:
    r = lres[s]
    ua = r["unit_accuracy"]
    M[s] = dict(acc=r["accuracy"], f1=r["f1_macro"], acc_ci=r["accuracy_ci95"], f1_ci=r["f1_macro_ci95"],
                unit_acc=np.array([ua[u] for u in ua]), K=r["K"], present=[g for g in G if g in r["auroc"]],
                auc=r["auroc"], cm=np.array(r["confusion_matrix"]), chance_acc=r["chance_accuracy"],
                chance_f1=r["chance_f1"], n_ch=r["n_channels"], n_units=len(ua))


# ------------------------------------------------------------------ example data for the method panel
def load_example():
    cfg = C.load_config()
    day = C.load_pickle(C.session_path(cfg, EX_SESSION, "recording"))
    lab = C.load_labels(EX_SESSION)
    td = day["tasks"][EX_TASK]
    fs = int(td["fsamp"])
    f = np.load(os.path.join(C.DATA_DIR, f"{EX_SESSION}_rms_features.npz"))
    kept = f["kept_channels_1idx"] - 1
    show = kept[np.linspace(0, len(kept) - 1, 8).round().astype(int)]
    x = E.apply_filters(td["emg"][show].astype(np.float32), fs, E.BANDPASS_HZ, E.BANDSTOP_HZ, E.BUTTER_ORDER)
    b = 64                                                    # min/max decimation -> light vector traces
    nb = x.shape[1] // b
    seg = x[:, :nb * b].reshape(len(show), nb, b)
    tr = np.empty((len(show), 2 * nb), np.float32)
    tr[:, 0::2], tr[:, 1::2] = seg.min(2), seg.max(2)
    t_tr = np.repeat(np.arange(nb) * b + b / 2, 2) / fs
    cue = [(a / fs, e / fs) for a, e in runs_idx(td["trigger"] > 0.05)]
    segs = [(sg["start_s"], sg["end_s"], C.JSON_TO_GESTURE[sg["gesture"]] if sg["quality"] == "good" else "excluded")
            for sg in lab["tasks"][EX_TASK]["segments"]]
    T = td["emg"].shape[1] / fs
    del day

    rate = float(f["feature_rate"])
    feats = f["features"]
    y = np.full(feats.shape[1], -1)
    blocks = []
    for s0, e0, g in zip(f["region_starts"], f["region_ends"], f["region_labels"]):
        y[int(s0):int(e0)] = G.index(str(g))
    for k in range(4):
        idx = np.where(y == k)[0]
        if len(idx):
            blocks.append((k, idx[0], idx[-1] + 1))
    block_starts = {s0 for _, s0, _ in blocks}
    seg_bounds = [int(s0) for s0 in f["region_starts"] if int(s0) not in block_starts]   # joins between segments
    n_seg = {k: int(sum(str(g) == G[k] for g in f["region_labels"])) for k in range(4)}
    zl = np.load(os.path.join(C.DATA_DIR, f"{EX_SESSION}_loro.npz"))
    unit = zl["unit"]
    assert len(unit) == len(y), "LORO units do not match the feature frames"
    units = []
    for u in np.unique(unit):
        idx = np.where(unit == u)[0]
        units.append((int(u), int(y[idx[0]]), int(idx[0]), int(idx[-1]) + 1))
    units.sort(key=lambda t: t[2])
    return dict(tr=tr, t_tr=t_tr, show=show, n_kept=len(kept), cue=cue, segs=segs, T=T,
                env=feats.mean(0), rate=rate, y=y, blocks=blocks,
                seg_bounds=seg_bounds, n_seg=n_seg, units=units,
                buffer=int(round(lres[EX_SESSION]["buffer_s"] * rate)))


EX = load_example()


# ------------------------------------------------------------------ panels
def panel_a(fig, spec):
    """Method: example recording with labels (left), dataset + leave-one-repetition-out scheme (right)."""
    outer = GridSpecFromSubplotSpec(1, 2, spec, width_ratios=[1, 1.45], wspace=0.16)
    L = GridSpecFromSubplotSpec(3, 1, outer[0], height_ratios=[0.13, 0.17, 2.0], hspace=0.12)
    R = GridSpecFromSubplotSpec(2, 1, outer[1], height_ratios=[1.0, 1.15], hspace=0.62)
    axCue, axLab, axEMG = (fig.add_subplot(L[i]) for i in range(3))
    axFeat, axFold = fig.add_subplot(R[0]), fig.add_subplot(R[1])
    for ax, gid in ((axCue, "a1_cue"), (axLab, "a1_labels"), (axEMG, "a1_emg"),
                    (axFeat, "a2_dataset_rms"), (axFold, "a3_folds")):
        ax.set_gid(gid)

    # -- left: one recording with cue, labels, HD-sEMG
    for ax in (axCue, axLab):
        ax.sharex(axEMG)
    for a, b in EX["cue"]:
        axCue.axvspan(a, b, color=ns.GREY, lw=0, gid="a1_cue_window")
    for a, b, g in EX["segs"]:
        if g == "excluded":
            axLab.axvspan(a, b, facecolor="white", edgecolor=ns.GREY, hatch="//////", lw=0.3, gid="a1_label_excluded")
        else:
            axLab.axvspan(a, b, color=ns.GESTURE_COLORS[g], lw=0, gid=f"a1_label_{g.replace(' ', '_')}")
            axEMG.axvspan(a, b, color=light(ns.GESTURE_COLORS[g], 0.18), lw=0, zorder=0, gid="a1_label_shading")
    for ax, nm in ((axCue, "Cue"), (axLab, "Label")):
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([nm])
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    tr = EX["tr"]
    spread = np.percentile(np.abs(tr), 99.5) * 2.0
    for i in range(len(tr)):
        axEMG.plot(EX["t_tr"], tr[i] + i * spread, color=ns.DARK_BLUE, lw=0.25, zorder=2,
                   gid=f"a1_emg_ch{EX['show'][i] + 1}")
    axEMG.set_ylim(-spread, len(tr) * spread)
    axEMG.set_yticks([i * spread for i in range(len(tr))])           # exact channel numbers of the shown traces
    axEMG.set_yticklabels([str(int(c) + 1) for c in EX["show"]])
    axEMG.tick_params(axis="y", length=1.5, pad=1)
    axEMG.set_ylabel(f"HD-sEMG\n({len(tr)} of {EX['n_kept']} ch)")
    axEMG.set_xlim(0, EX["T"])
    axEMG.set_xlabel("Time (s)")
    axCue.set_title("Recording and labelling (P03, power grasp task)", pad=3, loc="left")
    leg = [Patch(color=ns.GESTURE_COLORS["Power Grasp"], label=ns.WORD["Seg"]),
           Patch(facecolor="white", edgecolor=ns.GREY, hatch="//////", lw=0.3, label="Excluded")]
    axEMG.legend(handles=leg, loc="upper right", bbox_to_anchor=(1.0, -0.2), ncol=2, handlelength=1.0,
                 handletextpad=0.3, columnspacing=0.8, borderaxespad=0).set_gid("a1_legend")

    # -- right top: the labelled dataset (all classes concatenated) as RMS features
    rate = EX["rate"]
    t = np.arange(len(EX["env"])) / rate
    heads = []
    for k, s0, e0 in EX["blocks"]:
        axFeat.axvspan(s0 / rate, e0 / rate, color=light(ns.GESTURE_COLORS[G[k]], 0.22), lw=0, zorder=0,
                       gid=f"a2_block_{ns.SHORT[k]}")
        dur = (e0 - s0) / rate
        sub = f"rest recording, {dur:.0f} s" if G[k] == "Rest" else \
            f"{EX['n_seg'][k]} segment{'s' if EX['n_seg'][k] != 1 else ''}, {dur:.0f} s"
        heads.append(axFeat.text((s0 + e0) / 2 / rate, 1.02, f"{G[k]}\n{sub}", transform=axFeat.get_xaxis_transform(),
                                 ha="center", va="bottom", fontsize=ns.FS["annot"],
                                 color=dark(ns.GESTURE_COLORS[G[k]]), fontweight="bold", linespacing=1.1,
                                 gid="a2_class_names"))
    axFeat.set_xlim(0, t[-1])
    # narrow blocks: push neighbouring class headers apart so they do not overlap (right to left, then clamp)
    r = fig.canvas.get_renderer()
    inv = axFeat.transData.inverted()
    for _ in range(2):
        for a_, b_ in zip(heads[-2::-1], heads[:0:-1]):
            ba, bb = a_.get_window_extent(r), b_.get_window_extent(r)
            over = ba.x1 + 12 - bb.x0                      # at least ~12 px (at figure dpi) between headers
            if over > 0:
                dx = inv.transform((ba.x0 - over, 0))[0] - inv.transform((ba.x0, 0))[0]
                a_.set_x(a_.get_position()[0] + dx)
        last = heads[-1].get_window_extent(r)
        ax_right = axFeat.get_window_extent(r).x1 + 12
        if last.x1 > ax_right:
            heads[-1].set_x(heads[-1].get_position()[0] - (inv.transform((last.x1, 0))[0] - inv.transform((ax_right, 0))[0]))
    for b in EX["seg_bounds"]:                                # where cut-out segments were joined
        axFeat.axvline(b / rate, color="white", lw=0.9, zorder=1, gid="a2_segment_joins")
    axFeat.plot(t, EX["env"], color=ns.DARK_BLUE, lw=0.4, zorder=2, gid="a2_mean_rms")
    axFeat.set_xlim(0, t[-1])
    axFeat.set_ylim(0, np.percentile(EX["env"], 99.5) * 1.1)
    axFeat.set_ylabel("Mean RMS\n(a.u.)")
    axFeat.tick_params(axis="x", labelbottom=False)
    axFeat.set_title(f"Dataset: {ns.WORD['seg']}s cut out and joined by class (white lines: joins)",
                     pad=22, loc="left")

    # -- right bottom: leave-one-repetition-out, one example run per class
    n = len(EX["y"])
    units = EX["units"]
    by_class = {}
    for i, (u, k, s0, e0) in enumerate(units):
        by_class.setdefault(k, []).append(i)
    examples = [v[min(1, len(v) - 1)] if k in (1, 3) else v[0] for k, v in sorted(by_class.items())]
    for row_i, ex in enumerate(examples):
        row = len(examples) - 1 - row_i
        _, k_test, ts0, te0 = units[ex]
        for i, (u, k, s0, e0) in enumerate(units):
            col = ns.GESTURE_COLORS[G[k]]
            if i == ex:
                axFold.broken_barh([(s0 / rate, (e0 - s0) / rate)], (row + 0.12, 0.76), facecolors=col,
                                   edgecolor="none", gid=f"a3_run{ex + 1}_test")
                continue
            a, b = s0, e0
            if k_test == 0 and k == 0:                           # 1 s buffer next to a held-out rest part
                if e0 == ts0:
                    b = max(s0, e0 - EX["buffer"])
                if s0 == te0:
                    a = min(e0, s0 + EX["buffer"])
            if b > a:
                axFold.broken_barh([(a / rate, (b - a) / rate)], (row + 0.12, 0.76), facecolors=light(col, 0.3),
                                   edgecolor="none", gid=f"a3_run{ex + 1}_train")
    for i, (u, k, s0, e0) in enumerate(units):                     # repetition numbers + borders
        axFold.text((s0 + e0) / 2 / rate, len(examples) + 0.22, str(i + 1), ha="center", va="center",
                    fontsize=ns.FS["annot"], color=dark(ns.GESTURE_COLORS[G[k]]), fontweight="bold", gid="a3_repetition_numbers")
        if i:
            axFold.plot([s0 / rate] * 2, [0.05, len(examples)], color="white", lw=0.6, zorder=3,
                        gid="a3_repetition_borders")
    axFold.text(-0.005, len(examples) + 0.22, "Repetition", ha="right", va="center", fontsize=ns.FS["annot"], color=ns.GREY,
                transform=axFold.get_yaxis_transform(), gid="a3_repetition_label")
    axFold.set_ylim(0, len(examples) + 0.5)
    axFold.set_xlim(0, n / rate)
    axFold.set_yticks([len(examples) - 0.5 - i for i in range(len(examples))])
    axFold.set_yticklabels([f"Run {e + 1}" for e in examples])
    axFold.tick_params(axis="y", length=0)
    axFold.spines["left"].set_visible(False)
    axFold.set_xlabel("Joined labelled data (s)")
    axFold.set_title("Leave-one-repetition-out: each repetition is held out once (dark) and predicted by a model\n"
                     "trained on all other repetitions (light). Rest is split into equal parts; 1 s next to a held-out\n"
                     f"rest part is not used for training. P03: {len(units)} runs, {len(examples)} shown.",
                     pad=3, loc="left", fontsize=ns.FS["annot"], linespacing=1.15)
    axFold.text(0.0, -0.4, "Band-pass 20–500 Hz, notch 45–55 Hz  →  RMS (176 ms window, 31 ms step)\n"
                "→  z-score on training data  →  CatBoost", transform=axFold.transAxes, ha="left", va="top",
                fontsize=ns.FS["annot"], color=ns.GREY, linespacing=1.2, gid="a3_pipeline_text")

    return axCue


def panel_b(fig, ax):
    ax.set_gid("panel_b")
    rng = np.random.default_rng(1)
    off = {"accuracy": -0.17, "f1_macro": 0.17}
    for key, mk, ck in (("accuracy", "acc", "chance_acc"), ("f1_macro", "f1", "chance_f1")):
        col = ns.METRIC_COLORS[key]
        for i, s in enumerate(S):
            xi = i + off[key]
            if key == "accuracy":
                v = M[s]["unit_acc"]
                ax.scatter(xi + rng.uniform(-0.06, 0.06, len(v)), v, s=2.5, color=col, alpha=0.35, lw=0, zorder=2,
                           gid=f"b_repetitions_{s}")
            lo, hi = M[s][mk + "_ci"]
            eb = ax.errorbar(xi, M[s][mk], yerr=[[M[s][mk] - lo], [hi - M[s][mk]]], fmt="o", ms=3.2, color=col,
                             mec="white", mew=0.4, elinewidth=0.6, capsize=0, zorder=3)
            eb.lines[0].set_gid(f"b_{key}_{s}")
            ax.plot([xi - 0.12, xi + 0.12], [M[s][ck]] * 2, color=ns.GREY, lw=0.8, ls=(0, (2, 1)), zorder=1,
                    gid=f"b_{key}_chance_{s}")
        vals = np.array([M[s][mk] for s in S])
        xi = len(S) + off[key]
        ax.scatter(xi + rng.uniform(-0.05, 0.05, len(S)), vals, s=4, color=col, alpha=0.6, lw=0, zorder=2,
                   gid=f"b_{key}_sessions")
        eb = ax.errorbar(xi, vals.mean(), yerr=vals.std(ddof=1), fmt="D", ms=3.4, color=col, mec="white", mew=0.4,
                         elinewidth=0.6, capsize=0, zorder=3)
        eb.lines[0].set_gid(f"b_{key}_group_mean")
        ax.plot([xi - 0.12, xi + 0.12], [np.mean([M[s][ck] for s in S])] * 2, color=ns.GREY, lw=0.8,
                ls=(0, (2, 1)), gid=f"b_{key}_chance_mean")
    ax.axvline(len(S) - 0.5, color=ns.LIGHT_GREY, lw=0.5, gid="b_separator")
    ax.set_xticks(np.arange(len(S) + 1))
    ax.set_xticklabels(S + ["All"])
    ax.set_xlim(-0.6, len(S) + 0.6)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Accuracy / macro F1")
    xt = ax.get_xaxis_transform()
    for i, s in enumerate(S):
        ax.text(i, -0.13, f"K = {M[s]['K']}\n{M[s]['n_units']} rep.", transform=xt, ha="center", va="top",
                fontsize=ns.FS["tick"], color=ns.GREY, gid="b_session_info")
    acc_h = (plt.Line2D([], [], marker="o", ls="-", lw=0.6, ms=3.2, mec="white", mew=0.4,
                        color=ns.METRIC_COLORS["accuracy"]),)
    f1_h = (plt.Line2D([], [], marker="o", ls="-", lw=0.6, ms=3.2, mec="white", mew=0.4,
                       color=ns.METRIC_COLORS["f1_macro"]),)
    rep_h = (plt.Line2D([], [], marker="o", ls="", ms=2, alpha=0.35, color=ns.METRIC_COLORS["accuracy"]),)
    chance_h = (plt.Line2D([], [], color=ns.GREY, lw=0.8, ls=(0, (2, 1))),)
    ax.legend([acc_h, f1_h, rep_h, chance_h],
              ["Accuracy (95% CI)", "Macro F1 (95% CI)", "Single repetitions", "Chance"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)}, loc="lower left",
              bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.6, handletextpad=0.3, columnspacing=0.9,
              borderaxespad=0.2).set_gid("b_legend")


def panel_c(fig, ax):
    ax.set_gid("panel_c")
    stack = []
    for s in S:
        cm = M[s]["cm"].astype(float)
        rs = cm.sum(1, keepdims=True)
        stack.append(np.where(rs > 0, cm / np.where(rs == 0, 1, rs), np.nan))
    stack = np.array(stack)
    rows = np.nanmean(stack, 0) * 100
    nsess = (~np.isnan(stack[:, :, 0])).sum(0)
    qm = ax.pcolormesh(np.arange(5) - 0.5, np.arange(5) - 0.5, rows, cmap=ns.SEQ, vmin=0, vmax=100,
                       edgecolors="none", linewidth=0)
    qm.set_gid("c_cells")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_aspect("equal")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{rows[i, j]:.0f}", ha="center", va="center", fontsize=ns.FS["annot"],
                    fontweight="bold" if i == j else "normal", color="white" if rows[i, j] > 55 else ns.INK,
                    gid="c_cell_values")
    ax.set_xticks(range(4))
    ax.set_xticklabels(ns.SHORT, rotation=30, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{g} ({n})" for g, n in zip(ns.SHORT, nsess)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (n sessions)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([1.06, 0.0, 0.05, 1.0])
    cax.set_gid("c_colorbar")
    cb = fig.colorbar(qm, cax=cax)
    cb.solids.set_rasterized(False)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([0, 50, 100])
    cb.ax.tick_params(labelsize=ns.FS["tick"], width=0.4, length=1.5)
    cb.set_label("% of true class", fontsize=ns.FS["label"])


def panel_d(fig, ax):
    """One-vs-rest AUROC per task: one point per session (fixed order, left to right P01-S1 ... P05) with its 95% CI
    (class-stratified bootstrap over repetitions); filled = permutation test vs chance significant after Holm
    correction (P < 0.05), open = not significant. Statistics from results/loro_stats.json (step 04)."""
    ax.set_gid("panel_d")
    with open(os.path.join(C.RESULTS_DIR, "loro_stats.json"), encoding="utf-8") as f:
        st = json.load(f)["sessions"]
    step = 0.11
    for k, g in enumerate(G):
        ss = [s for s in S if g in M[s]["auc"]]
        if not ss:
            continue
        xs = k + (np.arange(len(ss)) - (len(ss) - 1) / 2) * step
        col = ns.GESTURE_COLORS[g]
        for x, s in zip(xs, ss):
            a = st[s]["auroc"][g]
            assert abs(a["value"] - M[s]["auc"][g]) < 1e-9, f"stats do not match LORO results: {s} {g}"
            ax.plot([x, x], a["ci95"], color=col, lw=0.5, solid_capstyle="butt", zorder=2,
                    gid=f"d_ci_{ns.SHORT[k]}")
            sig = a["p_holm"] < 0.05
            ax.scatter([x], [a["value"]], s=6, color=col if sig else "white", edgecolor=col, lw=0.5, zorder=3,
                       gid=f"d_sessions_{ns.SHORT[k]}" if sig else f"d_sessions_ns_{ns.SHORT[k]}")
        ax.plot([k - 0.33, k + 0.33], [np.median([M[s]["auc"][g] for s in ss])] * 2, color=ns.INK, lw=0.8,
                zorder=4, gid=f"d_median_{ns.SHORT[k]}")
    ax.axhline(0.5, color=ns.GREY, lw=0.8, ls=(0, (2, 1)), gid="d_chance")
    ax.set_xticks(range(4))
    ax.set_xticklabels(ns.SHORT, rotation=30, ha="right")
    ax.set_xlim(-0.6, 3.6)
    lo = min(min(a["ci95"][0] for a in st[s]["auroc"].values()) for s in S)
    ymin = min(0.4, np.floor(lo * 10) / 10)                  # every interval visible
    ax.set_ylim(ymin, 1.01)
    ax.set_yticks(np.round(np.arange(ymin, 1.001, 0.1), 1))
    ax.set_ylabel("AUROC (one-vs-rest)")


# ------------------------------------------------------------------ composite
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(ns.FULL_W, 150 * ns.MM))
    gs = GridSpec(2, 1, figure=fig, left=0.075, right=0.975, top=0.925, bottom=0.13, hspace=0.4,
                  height_ratios=[1.25, 1])
    panel_a(fig, gs[0])
    fig.text(0.008, 0.955, "A", fontsize=ns.FS["panel"], fontweight="bold", va="bottom", gid="label_a")
    row = GridSpecFromSubplotSpec(1, 3, gs[1], width_ratios=[2.0, 1.05, 0.85], wspace=0.62)
    axB = fig.add_subplot(row[0])
    panel_b(fig, axB)
    axC = fig.add_subplot(row[1])
    panel_c(fig, axC)
    axD = fig.add_subplot(row[2])
    panel_d(fig, axD)
    ytop = axB.get_position().y1 + 0.01                       # b, c, d labels on one line
    for lbl, x in (("B", 0.008), ("C", row[1].get_position(fig).x0 - 0.06), ("D", axD.get_position().x0 - 0.06)):
        fig.text(x, ytop, lbl, fontsize=ns.FS["panel"], fontweight="bold", va="bottom", gid=f"label_{lbl.lower()}")
    out = os.path.join(OUT_DIR, "Figure4")
    fig.savefig(f"{out}.pdf", metadata={"CreationDate": None})
    fig.savefig(f"{out}.png", dpi=600)
    save_svg(fig, f"{out}.svg")
    plt.close(fig)

    # ------------------------------------------------------------------ single-panel SVGs
    pdir = os.path.join(OUT_DIR, "panels")
    os.makedirs(pdir, exist_ok=True)
    f = plt.figure(figsize=(ns.FULL_W, 62 * ns.MM))
    panel_a(f, GridSpec(1, 1, figure=f, left=0.075, right=0.975, top=0.84, bottom=0.2)[0])
    save_svg(f, os.path.join(pdir, "panel_a.svg"))
    plt.close(f)
    for k, fn, (w, h), rect in (("b", panel_b, (96, 50), [0.12, 0.33, 0.86, 0.62]),
                                ("c", panel_c, (48, 46), [0.34, 0.24, 0.5, 0.7]),
                                ("d", panel_d, (40, 46), [0.25, 0.24, 0.72, 0.72])):
        f = plt.figure(figsize=(w * ns.MM, h * ns.MM))
        fn(f, f.add_axes(rect))
        save_svg(f, os.path.join(pdir, f"panel_{k}.svg"))
        plt.close(f)
    print("saved", out, "+ panels", pdir)


if __name__ == "__main__":
    main()
