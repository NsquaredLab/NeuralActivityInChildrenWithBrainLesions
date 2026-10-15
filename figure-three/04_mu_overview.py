"""
04_mu_overview.py
=================

Step 4 of the Figure 3 pipeline: motor-unit overview with the hold labels as
the cue (Fig3_MU_overview, DIN A4 portrait).

The cue is the box cue built from the hold labels (``*.labels_auto_v2.json``,
quality "good", matching gesture) by step 02. All statistics (panel a means,
panel c median/IQR, Mann-Whitney U, P, AUC) are recomputed from these tables.

Gestures without any used label segment (``n_segments_used = 0``): no hold
exists, so no MU can have a valid hold (valid = 0). A MU there counts as
"active, no valid hold" if it has >= MIN_DISCHARGES_PER_RAMP discharges in the
gesture, else "not active". Marked "no labels" in panel a.

Panels
------
    a   hold-wise r (lag-0 xcorr of the 1.5 s Hann discharge rate and the cue,
        per valid hold, ``xcorr_lag0_hold_mean``) per gesture and recording
        day: bar = mean across MUs with >= 1 valid hold, error bar = s.d.
        (ddof = 1, only n >= 2), dots = single MUs. Below: MU status per
        gesture and day (valid hold / active without a valid hold / not
        active), valid/total. Empty slot = gesture not recorded.
    b   whole-gesture r (``xcorr_whole_lag0``, rest included), best gesture
        per MU, by recording day. Filled = cue-modulated by eye, open = not,
        open square = 'maybe' (counted as not). Colour = gesture of the best r.
        Label = cue-modulated / MUs shown. Panels b, c show every MU with a
        computable whole-gesture r -- also MUs that discharge only at rest
        (negative r); the valid-hold criterion applies to panel a only.
    c   the same values pooled: by eye yes vs. no, median and IQR, two-sided
        Mann-Whitney U, AUC = U / (n_yes * n_no).

The by-eye verdict (``_common.BY_EYE`` / ``BY_EYE_MAYBE``) was made on the
same rates (with the trigger as cue) -- agreement, not independent validation.

Input / Output
--------------
    in : data/labels_overview.csv, data/labels_xcorr_per_mu.csv,
         data/labels_xcorr_per_hold.csv, data/mu_corr_labels_values.csv (step 02)
    out: results/Fig3_MU_overview.svg
         results/Fig3_MU_overview_values_a.csv    (per gesture and day, panel a)
         results/Fig3_MU_overview_values_bc.csv   (per motor unit, panels b, c)

Needs no raw data.

Dependencies
------------
    numpy, scipy, pandas, matplotlib

Usage
-----
    uv run python figure-three/04_mu_overview.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")           # save only, no window
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

import _common as C               # noqa: E402
from _style import GCOL, GESTURES  # noqa: E402


# ===========================================================================
# 1. Import of all data
# ===========================================================================
OVERVIEW_GLOB = "labels_overview.csv"       # per MU and task (panel a)
XCORR_MU_GLOB = "labels_xcorr_per_mu.csv"   # per MU and task (b, c)
XCORR_HOLD_GLOB = "labels_xcorr_per_hold.csv"   # per MU, task, hold (b, c)
# per MU and task: n_segments_used (labels present?) and n_discharges
LABEL_MU_GLOB = "mu_corr_labels_values.csv"
MIN_DISCHARGES_PER_RAMP = 5   # as in 02_xcorr_labels.py

# By-eye verdict: cue-modulated MUs per recording day, 1-based numbering of
# the tracking file. 'maybe' is counted as not.
BY_EYE, BY_EYE_MAYBE = C.BY_EYE, C.BY_EYE_MAYBE


def read_csvs(pattern):
    """The CSV(s) matching ``pattern`` in DATA_DIR, concatenated, plus
    ``session`` (P01-S1) and ``gesture`` (task name before '_')."""
    files = sorted(glob.glob(os.path.join(C.DATA_DIR, pattern)))
    if not files:
        raise SystemExit(f"No CSV for {pattern} (DATA_DIR = {C.DATA_DIR}) -- run "
                         f"02_xcorr_labels.py first")
    df = pd.concat([pd.read_csv(f, sep=";", decimal=",", encoding="utf-8-sig")
                    for f in files], ignore_index=True)
    # always rebuilt from patient_id + visit
    df["session"] = df["patient_id"] + "-" + df["visit"]
    df["gesture"] = df["task"].str.split("_").str[0]
    print(f"read {len(files)} x {pattern}: {len(df)} rows")
    return df


# ===========================================================================
# 2. Settings
# ===========================================================================
OUT_NAME = "Fig3_MU_overview"
PANEL_LOWER = True       # Nature: a, b, c
FIG_W_MM, FIG_H_MM = 210, 195   # A4 portrait width, height as small as readable
FS_SMALL, FS_NORMAL, FS_LARGE, FS_PANEL = 8, 10, 11, 12   # font sizes (pt)
SEED = 7                 # jitter of the dots only

INK, MUTED, GRID, INACTIVE = "#1a1a1a", "#6b6b6b", "#e3e3e3", "#e4e4e4"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS_NORMAL, "axes.labelsize": FS_NORMAL,
    "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "hatch.linewidth": 0.6,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Math text (italic P, superscript) in the same font as the rest.
_SANS = next((f for f in ("Arial", "Liberation Sans")
              if any(f == e.name for e in mpl.font_manager.fontManager.ttflist)),
             "DejaVu Sans")
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": _SANS,
                     "mathtext.it": f"{_SANS}:italic",
                     "mathtext.bf": f"{_SANS}:bold", "mathtext.cal": _SANS,
                     "mathtext.default": "regular"})


# ===========================================================================
# 3. Preparation
# ===========================================================================
def prepare_overview(df, label_mu):
    """
    Panel a: MU status per task from the overview table.

    Task without labels (n_segments_used = 0): no hold, so valid = False;
    active without a valid hold if the MU has >= MIN_DISCHARGES_PER_RAMP
    discharges in the task (the flat cue leaves xcorr_global_lag0 empty).
    """
    key = ["session", "task", "mu"]
    lm = label_mu[key + ["n_segments_used", "n_discharges"]]
    if lm.duplicated(key).any():
        raise SystemExit(f"{LABEL_MU_GLOB}: duplicate (session, task, mu)")
    df = df.merge(lm, on=key, how="left", validate="one_to_one")
    if df["n_segments_used"].isna().any():
        miss = df.loc[df["n_segments_used"].isna(), key]
        raise SystemExit(f"No label info for:\n{miss}")
    df["labels_present"] = df["n_segments_used"] > 0
    nolab = ~df["labels_present"]
    df["valid"] = df["xcorr_lag0_hold_mean"].notna() & ~nolab
    active = np.where(nolab,
                      df["n_discharges"] >= MIN_DISCHARGES_PER_RAMP,
                      df["xcorr_global_lag0"].notna())
    df["active_nohold"] = ~df["valid"] & active
    df["inactive"] = ~df["valid"] & ~active
    return df


def by_eye(session, mu):
    if mu in BY_EYE_MAYBE.get(session, []):
        return "maybe"
    return "yes" if mu in BY_EYE.get(session, []) else "no"


def prepare_units(xmu, xhold):
    """
    Panels b, c: one row per MU and recording day.

    r_best = max over gestures of xcorr_whole_lag0; best_gesture = where.
    included = r_best computable (the MU discharges in >= 1 gesture).
    n_valid_holds (>= MIN_DISCHARGES_PER_RAMP discharges in a hold) is
    kept for information only -- a MU firing only at rest is a finding.
    """
    active = xhold["active"].astype(str).str.lower() == "true"
    n_valid = (xhold[active].groupby(["session", "mu"]).size()
               .rename("n_valid_holds"))
    best = xmu.loc[xmu.groupby(["session", "mu"])["xcorr_whole_lag0"]
                   .idxmax().dropna()]
    best = (best.set_index(["session", "mu"])[["task", "xcorr_whole_lag0"]]
            .rename(columns={"task": "best_task",
                             "xcorr_whole_lag0": "r_best"}))
    U = (xmu[["session", "patient_id", "mu"]].drop_duplicates()
         .set_index(["session", "mu"]).join(n_valid).join(best).reset_index())
    U["n_valid_holds"] = U["n_valid_holds"].fillna(0).astype(int)
    U["by_eye_raw"] = [by_eye(s, m) for s, m in zip(U["session"], U["mu"])]
    U["by_eye"] = U["by_eye_raw"].replace({"maybe": "no"})
    U["included"] = U["r_best"].notna()
    U["best_gesture"] = U["best_task"].str.split("_").str[0]
    return U.sort_values(["session", "mu"]).reset_index(drop=True)


def session_labels(sessions, sep="\n"):
    """P01 has two recording days -> S1, S2 (visit order)."""
    out = {}
    for s in sessions:
        pid = s[:3]
        same = sorted(x for x in sessions if x[:3] == pid)
        out[s] = pid if len(same) == 1 else f"{pid}{sep}S{same.index(s) + 1}"
    return out


# ===========================================================================
# 4. Panels
# ===========================================================================
def _clean(ax, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)


def panel_a(ax, axn, df, sessions, rng):
    bar_w, gap = 0.72, 1.1
    lab = session_labels(sessions)
    n_s = len(sessions)
    xt, xl, spans, rows = [], [], [], []
    for gi, g in enumerate(GESTURES):
        x0 = gi * (n_s + gap)
        spans.append((x0 - 0.5, x0 + n_s - 0.5))
        for si, s in enumerate(sessions):
            x = x0 + si
            xt.append(x)
            xl.append(lab[s])
            sub = df[(df.session == s) & (df.gesture == g)]
            if sub.empty:
                continue                      # gesture not recorded
            v = sub.loc[sub.valid, "xcorr_lag0_hold_mean"].to_numpy()
            n_v, n_a, n_i = len(v), int(sub.active_nohold.sum()), \
                int(sub.inactive.sum())
            mean = v.mean() if n_v else np.nan
            sd = v.std(ddof=1) if n_v > 1 else np.nan
            has_labels = bool(sub.labels_present.any())
            rows.append(dict(gesture=g, session=s, labels_present=has_labels,
                             n_valid=n_v,
                             n_active_no_valid_hold=n_a, n_not_active=n_i,
                             mean=mean, sd=sd))
            if not has_labels:
                ax.text(x, 0.04, "no labels", rotation=90, ha="center",
                        va="bottom", fontsize=FS_SMALL, color=MUTED, zorder=5)
            if n_v:
                ax.bar(x, mean, bar_w, color=GCOL[g], lw=0, zorder=2,
                       alpha=0.9)
                if n_v > 1:
                    ax.errorbar(x, mean, yerr=sd, color=INK, lw=0.6,
                                capsize=1.6, capthick=0.6, zorder=4)
                ax.scatter(x + rng.uniform(-0.2, 0.2, n_v), v, s=10,
                           facecolor="white", edgecolor=INK, lw=0.4, zorder=5)
            axn.bar(x, n_v, bar_w, color=GCOL[g], lw=0, zorder=2, alpha=0.9)
            if n_a:
                axn.bar(x, n_a, bar_w, bottom=n_v, facecolor="white",
                        edgecolor=GCOL[g], hatch="//////", lw=0.5, zorder=2)
            if n_i:
                axn.bar(x, n_i, bar_w, bottom=n_v + n_a, color=INACTIVE,
                        lw=0, zorder=2)
            axn.text(x, n_v + n_a + n_i + 0.6, f"{n_v}/{n_v + n_a + n_i}",
                     ha="center", va="bottom", fontsize=FS_SMALL, color=INK)

    ax.axhline(0, color=INK, lw=0.6, zorder=3)
    ax.set(ylim=(-1, 1), yticks=[-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Hold-wise r\n(mean ± s.d. across MUs)")
    axn.set(ylim=(0, 15.5), yticks=[0, 6, 12])
    axn.set_ylabel("MUs (n)")
    for a in (ax, axn):
        _clean(a)
        a.tick_params(axis="x", length=0)
    axn.set_xticks(xt)
    axn.set_xticklabels(xl, linespacing=0.95, fontsize=FS_SMALL)
    axn.set_xlim(spans[0][0] - 0.3, spans[-1][1] + 0.3)
    plt.setp(ax.get_xticklabels(), visible=False)

    # gesture headers: name + colour rule (identifies the colour of all panels)
    tr = ax.get_xaxis_transform()
    for g, (a0, a1) in zip(GESTURES, spans):
        ax.plot([a0 + 0.1, a1 - 0.1], [1.035, 1.035], color=GCOL[g], lw=2.0,
                solid_capstyle="butt", transform=tr, clip_on=False)
        ax.text((a0 + a1) / 2, 1.07, g, ha="center", va="bottom",
                fontsize=FS_LARGE, fontweight="bold", transform=tr)

    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=0.4,
               ms=3.6, label="Single MU"),
        Patch(facecolor=MUTED, lw=0, label="MU with valid hold"),
        Patch(facecolor="white", edgecolor=MUTED, hatch="//////", lw=0.5,
              label="MU active, no valid hold"),
        Patch(color=INACTIVE, label="MU not active"),
    ]
    axn.legend(handles=handles, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.36), frameon=False, fontsize=FS_SMALL,
               handlelength=1.3, handleheight=0.8, columnspacing=1.6)
    return pd.DataFrame(rows)


def _unit_markers(ax, x, U, size=28):
    """Filled = by eye yes, open = no, open square = maybe; colour = gesture."""
    for xi, (_, r) in zip(x, U.iterrows()):
        c = GCOL.get(r["best_gesture"], MUTED)
        if r["by_eye_raw"] == "yes":
            kw = dict(marker="o", facecolor=c, edgecolor="white", lw=0.3)
        elif r["by_eye_raw"] == "maybe":
            kw = dict(marker="s", facecolor="white", edgecolor=c, lw=0.8)
        else:
            kw = dict(marker="o", facecolor="white", edgecolor=c, lw=0.8)
        ax.scatter(xi, r["r_best"], s=size, zorder=3, **kw)


def panel_b(ax, U, sessions, rng):
    lab = session_labels(sessions, sep=" ")
    A = U[U["included"]]
    for i, s in enumerate(sessions):
        G = A[A["session"] == s]
        _unit_markers(ax, i + rng.uniform(-0.16, 0.16, len(G)), G)
        n_excl = int(((U["session"] == s) & ~U["included"]).sum())
        txt = f"{(G['by_eye'] == 'yes').sum()}/{len(G)}" + ("†" if n_excl else "")
        ax.text(i, 1.0, txt, ha="center", va="bottom", fontsize=FS_SMALL)
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.set(xticks=range(len(sessions)),
           xticklabels=[lab[s] for s in sessions],
           xlim=(-0.6, len(sessions) - 0.4), ylim=(-0.4, 1.0),
           yticks=[-0.4, 0, 0.4, 0.8])
    ax.set_ylabel("Whole-gesture r\n(best gesture per MU)")
    ax.tick_params(axis="x", length=0)
    _clean(ax)


def fmt_p(p):
    """P = 2.3 × 10^-5 (Nature style); plain for P >= 0.001."""
    if p >= 1e-3:
        return f"$\\mathit{{P}}$ = {p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"$\\mathit{{P}}$ = {m} × 10$^{{{int(e)}}}$"


def panel_c(ax, U, rng):
    A = U[U["included"]]
    groups = [("yes", "Cue-modulated"), ("no", "Not modulated")]
    stats = {}
    for i, (g, _) in enumerate(groups):
        G = A[A["by_eye"] == g]
        _unit_markers(ax, i + rng.uniform(-0.17, 0.17, len(G)), G)
        v = G["r_best"].dropna().to_numpy()
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        stats[g] = dict(n=v.size, median=med, q25=q25, q75=q75,
                        mean=v.mean(), sd=v.std(ddof=1))
        ax.plot([i + 0.28, i + 0.28], [q25, q75], color=INK, lw=0.8,
                zorder=4, solid_capstyle="butt")
        ax.plot([i + 0.22, i + 0.34], [med, med], color=INK, lw=1.2, zorder=4)
    y = A.loc[A["by_eye"] == "yes", "r_best"].dropna()
    n = A.loc[A["by_eye"] == "no", "r_best"].dropna()
    u, p = mannwhitneyu(y, n, alternative="two-sided")
    auc = u / (len(y) * len(n))
    stats.update(u=u, p=p, auc=auc)
    # significance bracket
    top = 0.93
    ax.plot([0, 0, 1, 1], [top - 0.03, top, top, top - 0.03], color=INK,
            lw=0.6)
    ax.text(0.5, top + 0.01, fmt_p(p), ha="center", va="bottom",
            fontsize=FS_SMALL)
    ax.text(0.5, -0.36, f"AUC = {auc:.2f}", ha="center", va="bottom",
            fontsize=FS_SMALL)
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.set(xticks=[0, 1],
           xticklabels=[f"{lab}\n(n = {stats[g]['n']})" for g, lab in groups],
           xlim=(-0.55, 1.6), ylim=(-0.4, 1.0), yticks=[-0.4, 0, 0.4, 0.8])
    ax.set_xlabel("By eye", labelpad=2)
    ax.tick_params(axis="x", length=0)
    plt.setp(ax.get_yticklabels(), visible=False)
    _clean(ax)
    return stats


def legend_bc(fig, ax_b):
    h = [Line2D([], [], ls="", marker="o", mfc=MUTED, mec="white", ms=5.5,
                label="Cue-modulated (by eye)"),
         Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, mew=0.8,
                ms=5.5, label="Not modulated"),
         Line2D([], [], ls="", marker="s", mfc="white", mec=MUTED, mew=0.8,
                ms=5.2, label="Uncertain (counted as not)"),
         Line2D([], [], ls="", marker="", label="Colour: gesture with highest r"),
         ]
    ax_b.legend(handles=h, ncol=4, loc="lower left", frameon=False,
                fontsize=FS_SMALL, bbox_to_anchor=(0.0, 1.11), handletextpad=0.3,
                columnspacing=1.4, borderaxespad=0)


# ===========================================================================
# 5. Figure and export
# ===========================================================================
def main():
    out_stem = os.path.join(C.results_dir(), OUT_NAME)
    overview = read_csvs(OVERVIEW_GLOB)
    xcorr_mu = read_csvs(XCORR_MU_GLOB)
    xcorr_hold = read_csvs(XCORR_HOLD_GLOB)
    label_mu = read_csvs(LABEL_MU_GLOB)

    rng = np.random.default_rng(SEED)
    df = prepare_overview(overview, label_mu)
    U = prepare_units(xcorr_mu, xcorr_hold)
    sessions = sorted(df["session"].unique())
    differ = set(U["session"]) ^ set(sessions)
    if differ:
        print(f"Warning: sessions differ between the inputs: {sorted(differ)}")

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.48,
                             left=0.085, right=0.99, top=0.935, bottom=0.085)
    ga = outer[0].subgridspec(2, 1, height_ratios=[3, 1.05], hspace=0.08)
    ax_a = fig.add_subplot(ga[0])
    ax_n = fig.add_subplot(ga[1], sharex=ax_a)
    gb = outer[1].subgridspec(1, 2, width_ratios=[2.35, 1], wspace=0.06)
    ax_b = fig.add_subplot(gb[0])
    ax_c = fig.add_subplot(gb[1], sharey=ax_b)

    vals_a = panel_a(ax_a, ax_n, df, sessions, rng)
    panel_b(ax_b, U, sessions, rng)
    stats = panel_c(ax_c, U, rng)
    legend_bc(fig, ax_b)

    n_excl = int((~U["included"]).sum())
    if n_excl:
        ax_b.text(0.0, -0.22, f"† {n_excl} MU without a computable r "
                  f"(no discharge) not shown", transform=ax_b.transAxes,
                  fontsize=FS_SMALL, color=MUTED, va="top")

    letters = "abc" if PANEL_LOWER else "ABC"
    pa, pb, pc = (ax.get_position() for ax in (ax_a, ax_b, ax_c))
    for L, x, y in ((letters[0], 0.004, pa.y0 + 1.07 * pa.height),
                    (letters[1], 0.004, pb.y1 + 0.065),
                    (letters[2], pc.x0 - 0.012, pb.y1 + 0.065)):
        fig.text(x, y, L, fontsize=FS_PANEL, fontweight="bold", va="bottom",
                 ha="left")

    fig.align_ylabels([ax_a, ax_n, ax_b])
    fig.savefig(f"{out_stem}.svg", metadata={"Date": None})   # no creation date
    plt.close(fig)

    vals_a.to_csv(f"{out_stem}_values_a.csv", sep=";", decimal=",",
                  index=False, float_format="%.4f")
    U.to_csv(f"{out_stem}_values_bc.csv", sep=";", decimal=",", index=False,
             float_format="%.5f")
    print(f"Wrote {OUT_NAME}.svg and the value tables")
    for _, r in vals_a[~vals_a["labels_present"]].iterrows():
        print(f"  no labels: {r['session']} {r['gesture']} -> valid 0, "
              f"active {r['n_active_no_valid_hold']}, "
              f"not active {r['n_not_active']}")
    print(f"MUs: {len(U)}, with a computable r: {int(U['included'].sum())}")
    for _, r in U[~U["included"]].iterrows():
        print(f"  not shown: {r['session']} MU {r['mu']} (no computable r)")
    for _, r in U[U["included"] & (U["n_valid_holds"] == 0)].iterrows():
        print(f"  shown, but no valid hold: {r['session']} MU {r['mu']} "
              f"(r_best = {r['r_best']:.2f})")
    for g in ("yes", "no"):
        s = stats[g]
        print(f"by eye {g}: n = {s['n']}, median {s['median']:.2f} "
              f"[IQR {s['q25']:.2f} to {s['q75']:.2f}], mean {s['mean']:.2f} "
              f"± {s['sd']:.2f} (s.d., ddof = 1)")
    print(f"Mann-Whitney U = {stats['u']:.1f}, P = {stats['p']:.2g} "
          f"(two-sided), AUC = {stats['auc']:.2f}")
    print("Caveat: the by-eye verdict was made on the same rates (with the "
          "trigger as cue) -- agreement, not independent validation.")


if __name__ == "__main__":
    main()
