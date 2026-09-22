# -*- coding: utf-8 -*-
"""Supplementary Figure S2 -- is q95 a 95th percentile when the response is exactly reducible?

Every run is exactly reducible (sigma_true = 0), so a run whose sigma_true_hat exceeds a
given percentile of its own null distribution is a false alarm.  Two panels, two files.

a  figS2a_null: observed against nominal false-alarm rate.
   One field, 600 runs: data/s2_null_calib_600.npz (data/s2_null_calibration.py), the
   low-Re pipe geometry with y = exp(0.7 (0.8 u - 0.6 v)).
   800 fields: data/s2_null_fields_800.npz (data/s2_null_fields.py), new exactly reducible
   single-ridge fields (five ridge profiles, random direction), each at eps x 1 and x 4.
   All runs use the deployed N_MC 1200, N_FIELDSIM 6400 and n = 6.  For each run p is the
   share of its own null draws of sigma_true that reach sigma_true_hat, and a curve is the
   share of runs with p at or below the nominal rate; at 5% this is the deployed test
   sigma_true_hat > q95.  No interval bars here: at 5% the three would overlap; panel b
   carries the 5% intervals.  Grey band: the central 95% range for 600 runs of an exactly
   calibrated test (normal approximation, clipped at zero); 600 is the smaller sample, so
   this is the wider of the two bands.  The y axis runs to 24% so that the eps x 1 curve
   (21% at a nominal 20%) stays inside the frame.

b  figS2b_floor: false-alarm rate at q95 for the 800 fields, all together and grouped by
   q95 / read-out floor.  The read-out floor is the field's noise-free sigma_alpha, which on
   an exactly reducible field is read-out error only.  Clopper-Pearson 95% intervals.  At
   eps x 4 no field has a ratio below 3.

"Nominal" (the diagonal in a, the 5% line in b) is named in the key rather than on the
plot: on the plot the label crowded the eps x 1 curve.  The earlier points for 100 fields
(lam = 0 runs of the S1 data, N_FIELDSIM 1600) are gone: the 800-field runs replace them
with the deployed settings.  Axes are in words, not alpha and p, which mean other things
elsewhere in the paper.

Arial 7 pt, black text, lines of at least 1 pt, drawn at final size, white background; both
panels use the frame of figS1_power.
Outputs: figS2a_null.png / .svg and figS2b_floor.png / .svg, 88 mm wide each
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, colours

MM = 1 / 25.4
FS = 7
DATA = ROOT.parent.parent / "si_data" / "data"
INK = "#1f2937"
#          eps multiplier, key label,                              colour (as S1)
LEVELS = ((1.0, r"800 fields, noise $\varepsilon\times1$", "#3f6fa0"),
          (4.0, r"800 fields, noise $\varepsilon\times4$", "#c0603a"))
BAND = "#e3e6ea"
LW, MS = 1.2, 3.2
AMAX, YMAX_A = 20.0, 24.0                                   # panel a, percent
ALPHA0 = 5.0                                                # the level of q95
NOMINAL_LS = (0, (4, 3))
#          tick label,   ratio from, ratio to
GROUPS = (("All fields", None, None),
          ("< 3", 0.0, 3.0),
          ("3–10", 3.0, 10.0),
          ("≥ 10", 10.0, np.inf))
W, H = 88 * MM, 69 * MM
L, B, R, T = 0.46, 0.40, 0.10, 0.36                         # the frame of figS1_power


def frame():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    return fig, ax


def key(ax, handles, labels):
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=FS, handlelength=2.6, handletextpad=0.5,
              labelspacing=0.35, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)


def nominal_handle():
    return Line2D([], [], color=C.C_MUTED, lw=1.0, ls=NOMINAL_LS)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("%s.%s" % (name, ext)), dpi=400, facecolor="white")
    plt.close(fig)
    print("%s  %.0f x %.0f mm" % (name, W * 25.4, H * 25.4))


def cp_interval(k, n, level=0.95):
    """Clopper-Pearson interval for k successes in n trials, as fractions."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def panel_a(one_p, A, c):
    fig, ax = frame()
    ax.set_xlim(0, AMAX)
    ax.set_ylim(0, YMAX_A)
    alpha = np.linspace(0.0, AMAX / 100.0, 2001)

    half = 1.96 * np.sqrt(alpha * (1 - alpha) / one_p.size)
    ax.fill_between(alpha * 100, 100 * np.maximum(alpha - half, 0.0), 100 * (alpha + half),
                    color=BAND, lw=0, zorder=0)
    ax.plot([0, AMAX], [0, AMAX], color=C.C_MUTED, lw=1.0, ls=NOMINAL_LS, zorder=1)
    ax.axvline(ALPHA0, color=C.C_MUTED, lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.text(ALPHA0 + 0.25, YMAX_A - 0.5, r"$q_{95}$", ha="left", va="top", fontsize=FS,
            color="black")

    curves = [("One field, %d runs" % one_p.size, one_p, INK)]
    curves += [(lab, A[A[:, c["eps_mult"]] == m, c["p"]], col) for m, lab, col in LEVELS]
    handles = []
    print("panel a: runs with p at or below the nominal rate")
    for lab, p, col in curves:
        share = np.array([np.mean(p <= a) for a in alpha])
        ax.plot(alpha * 100, share * 100, color=col, lw=LW, drawstyle="steps-post", zorder=3)
        handles.append(Line2D([], [], color=col, lw=LW))
        cells = []
        for a in (1, 5, 10, 20):
            k = int(np.sum(p <= a / 100.0))
            lo, hi = cp_interval(k, p.size)
            cells.append("%d%%: %.2f%% (%.2f-%.2f)" % (a, 100.0 * k / p.size, 100 * lo, 100 * hi))
        print("   %-34s n %d | %s" % (lab.replace("$", ""), p.size, " | ".join(cells)))

    # first column: the one field and the nominal line; second column: the 800 fields
    key(ax, [handles[0], nominal_handle()] + handles[1:],
        [curves[0][0], "Nominal"] + [cv[0] for cv in curves[1:]])
    ticks = [0, 5, 10, 15, 20]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(direction="in", length=2)
    ax.set_xlabel("Nominal false-alarm rate (%)", labelpad=2)
    ax.set_ylabel("Observed false-alarm rate (%)", labelpad=2)
    save(fig, "figS2a_null")


def panel_b(A, c):
    fig, ax = frame()
    xs = np.array([0.0, 1.3, 2.3, 3.3])                     # a wider gap after "All fields"
    ax.set_xlim(-0.6, 3.9)
    ax.set_ylim(0, 16)
    ax.axhline(ALPHA0, color=C.C_MUTED, lw=1.0, ls=NOMINAL_LS, zorder=1)

    handles = []
    print("panel b: false-alarm rate at q95")
    for (m, lab, col), dx in zip(LEVELS, (-0.13, 0.13)):
        D = A[A[:, c["eps_mult"]] == m]
        ratio = D[:, c["q95"]] / np.maximum(D[:, c["floor"]], 1e-12)
        h = None
        for x0, (gname, lo, hi) in zip(xs, GROUPS):
            sel = np.ones(len(D), bool) if lo is None else (ratio >= lo) & (ratio < hi)
            n = int(sel.sum())
            if n == 0:
                ax.text(x0 + dx, 0.4, "no fields", ha="center", va="bottom", fontsize=FS,
                        color="black")
                print("   %-34s %-10s none" % (lab.replace("$", ""), gname))
                continue
            k = int(D[sel, c["exceed"]].sum())
            r = 100.0 * k / n
            l_, h_ = cp_interval(k, n)
            h = ax.errorbar([x0 + dx], [r], yerr=[[r - 100 * l_], [100 * h_ - r]], fmt="o",
                            color=col, ms=MS, mec="none", elinewidth=1.0, capsize=0, zorder=4)
            print("   %-34s %-10s %3d of %3d = %5.2f%% (%.2f-%.2f)"
                  % (lab.replace("$", ""), gname, k, n, r, 100 * l_, 100 * h_))
        handles.append(h)

    # first column: the two noise levels; second column: the nominal line
    key(ax, handles + [nominal_handle()], [lab for _, lab, _ in LEVELS] + ["Nominal"])
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in GROUPS])
    ax.tick_params(axis="x", length=0)                      # categories: no tick marks
    ax.tick_params(axis="y", direction="in", length=2)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel(r"Fields grouped by $q_{95}$ / read-out floor", labelpad=3)
    ax.set_ylabel(r"False-alarm rate at $q_{95}$ (%)", labelpad=2)
    save(fig, "figS2b_floor")


def main():
    one_p = np.load(DATA / "s2_null_calib_600.npz")["p"]
    d = np.load(DATA / "s2_null_fields_800.npz", allow_pickle=True)
    c = {str(name): i for i, name in enumerate(d["cols"])}
    panel_a(one_p, d["rows"], c)
    panel_b(d["rows"], c)


if __name__ == "__main__":
    main()
