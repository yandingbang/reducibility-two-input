# -*- coding: utf-8 -*-
"""Supplementary Figure S5 -- what the declared tolerance tau_1 trades.

Data: data/s5_tau1_1200.npz, written by data/s5_tau1_sweep.py; the same numbers are printed
by data/s5_tau1_summary.py.  1200 synthetic single-ridge fields with a linear transverse term
(lam = 0 in 298 of them, otherwise log-uniform on [0.001, 0.3]), each measured once at eps x 1
and once at x 4, one merged campaign with the deployed settings (N_MERGE 6, N_MC 1200,
N_FIELDSIM 6400).  Nothing in a run depends on tau_1, so every tau_1 re-scores the same runs
with the stage-one rule: not reducible if sigma_true_hat > max(tau_1, q95); reducible if
sigma_true_hat < tau_1 and q95 < tau_1; undetermined otherwise.  The truth is the analytic
sigma_true: reducible at tau_1 when sigma_true < tau_1.

a  figS5a_undetermined: the share of runs left undetermined, over all 1200 runs per level.
b  figS5b_errors: the two wrong verdicts, each over its own population,
     wrongly not reducible   among runs with sigma_true < tau_1   (solid, circles)
     wrongly reducible       among runs with sigma_true > tau_1   (dashed, squares),
   with Clopper-Pearson 95% intervals (one run per field, so the runs are independent).
   At each tau_1 the four series are set slightly apart along x so that the bars stay apart.

Points at tau_1 = 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8 deg on a logarithmic axis, like the field
design.  Dotted line: the deployed tau_1 = 4 deg, named in the key.  Colour is the noise level
as in S1 and S2 (blue x 1, terracotta x 4); in b the line style and marker are the error, as
the family in S1.  The key sits above the plot.

Axes of the figS1_power frame; the top margin is taller for the three-row key of b, and a
uses the same size so the two files line up.
Arial 7 pt, black text, lines of at least 1 pt, drawn at final size, white background.
Outputs: figS5a_undetermined.png / .svg and figS5b_errors.png / .svg, 88 mm wide each
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import common as C                                   # noqa: E402  fonts, colours

MM = 1 / 25.4
FS = 7
DATA = ROOT.parent.parent / "si_data" / "data" / "s5_tau1_1200.npz"
#          eps multiplier, key label,                  colour (as S1)
LEVELS = ((1.0, r"Noise $\varepsilon\times1$", "#3f6fa0"),
          (4.0, r"Noise $\varepsilon\times4$", "#c0603a"))
#          key label,               line style,       marker (as the families of S1)
ERRORS = (("False NOT-RED",    "-",              "o"),      # the paper's terms (Supplementary Table 3)
          ("Missed departure", (0, (3.2, 1.6)), "s"))
TAUS = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
DEPLOYED = 4.0
DEPLOYED_LS = (0, (1, 2))
LW, MS = 1.2, 3.0
# panel b, x offset in decades per (error, eps multiplier): not reducible left, reducible right
OFFSET = {(0, 1.0): -0.03, (0, 4.0): -0.01, (1, 1.0): 0.01, (1, 4.0): 0.03}
W = 88 * MM
L, B, R = 0.46, 0.40, 0.10                                  # the frame of figS1_power ...
AXH = 69 * MM - 0.40 - 0.36                                 # ... and its axes height
T = 0.50                                                    # room for a three-row key
H = B + AXH + T


def frame():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, AXH / H])
    ax.set_xscale("log")
    ax.set_xlim(0.42, 9.5)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(direction="in", length=2)
    ax.axvline(DEPLOYED, color=C.C_MUTED, lw=1.0, ls=DEPLOYED_LS, zorder=1)
    ax.set_xlabel(r"Declared tolerance, $\tau_1$ (°)", labelpad=2)
    return fig, ax


def key(ax, handles, labels):
    return ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
                     frameon=False, fontsize=FS, handlelength=2.6, handletextpad=0.5,
                     labelspacing=0.35, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)


def deployed_handle():
    return Line2D([], [], color=C.C_MUTED, lw=1.0, ls=DEPLOYED_LS)


def save(fig, leg, name):
    fig.canvas.draw()
    bb, fb = leg.get_window_extent(), fig.bbox
    print("%s  %.0f x %.1f mm; key clears the top by %.1f mm and the right edge by %.1f mm"
          % (name, W * 25.4, H * 25.4, (fb.y1 - bb.y1) / fig.dpi * 25.4, (fb.x1 - bb.x1) / fig.dpi * 25.4))
    for ext in ("png", "svg"):
        fig.savefig(ROOT / ("%s.%s" % (name, ext)), dpi=400, facecolor="white")
    plt.close(fig)


def cp_interval(k, n, level=0.95):
    """Clopper-Pearson interval for k successes in n trials, as fractions."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def runs(A, c, m):
    D = A[A[:, c["eps_mult"]] == m]
    return D[:, c["sigma_true"]], D[:, c["q95"]], D[:, c["sigma_true_hat"]]


def verdicts(st, q95, tau):
    nr = st > np.maximum(tau, q95)
    rd = (st <= tau) & (q95 < tau)          # <= on the statistic since 2026-09-17, as v1
    return nr, rd, ~(nr | rd)


def panel_a(A, c):
    fig, ax = frame()
    ax.set_ylim(-3, 104)
    print("panel a: undetermined, over all runs")
    for m, lab, col in LEVELS:
        s, q, st = runs(A, c, m)
        un = np.array([100.0 * np.mean(verdicts(st, q, t)[2]) for t in TAUS])
        ax.plot(TAUS, un, color=col, lw=LW, marker="o", ms=MS, mec="none", zorder=3)
        print("   eps x%g  n %d | %s" % (m, len(s), "  ".join("%g: %.2f%%" % (t, u) for t, u in zip(TAUS, un))))
    handles = [Line2D([], [], color=col, lw=LW) for _, _, col in LEVELS] + [deployed_handle()]
    labels = [lab for _, lab, _ in LEVELS] + [r"Deployed $\tau_1$"]
    leg = key(ax, handles, labels)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("UNRESOLVED (%)", labelpad=2)
    save(fig, leg, "figS5a_undetermined")


def panel_b(A, c):
    fig, ax = frame()
    ax.set_ylim(-0.7, 24)          # 2026-09-16: the x4 interval at tau_1 = 8 now reaches 22.7%
    print("panel b: wrong verdicts, each over its own population")
    for m, lab, col in LEVELS:
        s, q, st = runs(A, c, m)
        for ei, (ename, ls, mk) in enumerate(ERRORS):
            ys, lo, hi, cells = [], [], [], []
            for t in TAUS:
                nr, rd, _ = verdicts(st, q, t)
                pop = (s < t) if ei == 0 else (s > t)
                k, n = int(((nr if ei == 0 else rd) & pop).sum()), int(pop.sum())
                l_, h_ = cp_interval(k, n)
                ys.append(100.0 * k / n)
                lo.append(100.0 * l_)
                hi.append(100.0 * h_)
                cells.append("%g: %d/%d %.2f%% (%.1f-%.1f)" % (t, k, n, 100.0 * k / n, 100 * l_, 100 * h_))
            ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
            ax.errorbar(TAUS * 10 ** OFFSET[(ei, m)], ys, yerr=[ys - lo, hi - ys], color=col, lw=LW,
                        ls=ls, marker=mk, ms=MS, mec="none", elinewidth=1.0, capsize=0, zorder=3)
            print("   eps x%g  %-22s %s" % (m, ename, " | ".join(cells)))
    # first column: the two noise levels and the deployed line; second column: the two errors
    handles = [Line2D([], [], color=col, lw=LW) for _, _, col in LEVELS] + [deployed_handle()] + \
              [Line2D([], [], color=C.C_MUTED, lw=LW, ls=ls, marker=mk, ms=MS, mec="none")
               for _, ls, mk in ERRORS]
    labels = [lab for _, lab, _ in LEVELS] + [r"Deployed $\tau_1$"] + [e[0] for e in ERRORS]
    leg = key(ax, handles, labels)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("Error rate (%)", labelpad=2)
    save(fig, leg, "figS5b_errors")


def main():
    d = np.load(DATA, allow_pickle=True)
    c = {str(name): i for i, name in enumerate(d["cols"])}
    panel_a(d["rows"], c)
    panel_b(d["rows"], c)


if __name__ == "__main__":
    main()
